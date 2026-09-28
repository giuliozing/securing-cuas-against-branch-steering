import json
import tempfile
import unittest
from pathlib import Path

from cobra.brh.annotator import AnnotationError, annotate
from cobra.brh.skeleton import extract_skeleton
from cobra.brh.writer import BRHConfig, _apply_tool_servers, generate_plan_constraints, reset_branch_state
from cobra.brh.schema import BranchConstraints, McpConstraints, PlanConstraints

PLAN = '''\
Here is the plan:

```python
page = get_page_text("https://shop.example.com/usb-c-hub")
perceived_price = query_price(page)
if perceived_price <= 50:
    place_order("payments.example.com", "SKU-7741", perceived_price)
else:
    print("too expensive")
```
'''

TASK = "Buy a USB-C hub from shop.example.com if the price is at most 50 GBP"

GOOD_ANNOTATION = """\
```json
{
  "branches": {
    "root": {
      "description": "observe product page",
      "trigger_var": null,
      "http_constraints": {"allowed_domains": ["shop.example.com"], "fields": []},
      "mcp_constraints": null
    },
    "if_L3_true": {
      "description": "purchase branch",
      "trigger_var": "perceived_price",
      "http_constraints": {
        "allowed_domains": ["shop.example.com", "payments.example.com"],
        "fields": [
          {"path": "amount", "op": "<=", "value": "trigger_value"},
          {"path": "currency", "op": "in", "value": ["GBP", "EUR"]},
          {"path": "product_id", "op": "==", "value": "SKU-7741"}
        ]
      },
      "mcp_constraints": null
    },
    "if_L3_false": {
      "description": "abort branch",
      "trigger_var": "perceived_price",
      "http_constraints": {"allowed_domains": [], "fields": []},
      "mcp_constraints": null
    }
  }
}
```
"""

BAD_ANNOTATION = """\
```json
{"branches": {"if_L99_true": {"description": "hallucinated",
  "http_constraints": {"allowed_domains": ["evil.example.com"], "fields": []},
  "mcp_constraints": null}}}
```
"""


class ScriptedLLM:
    """Fake llm_call returning scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        return self.responses.pop(0)


class AnnotateTest(unittest.TestCase):
    def setUp(self):
        self.skeleton = extract_skeleton(PLAN)

    def test_good_annotation_accepted_first_try(self):
        llm = ScriptedLLM([GOOD_ANNOTATION])
        constraints = annotate(llm, TASK, self.skeleton, "plan_001")
        self.assertEqual(constraints.plan_id, "plan_001")
        self.assertEqual(
            constraints.branches["if_L3_true"].http_constraints.fields[0].value,
            "trigger_value",
        )
        self.assertEqual(len(llm.prompts), 1)

    def test_retry_with_feedback_then_success(self):
        llm = ScriptedLLM([BAD_ANNOTATION, GOOD_ANNOTATION])
        constraints = annotate(llm, TASK, self.skeleton, "plan_001")
        self.assertEqual(len(llm.prompts), 2)
        retry_prompt = llm.prompts[1][1]
        self.assertIn("rejected", retry_prompt)
        self.assertIn("if_L99_true", retry_prompt)
        self.assertIn("if_L3_true", constraints.branches)

    def test_exhausted_retries_raise(self):
        llm = ScriptedLLM([BAD_ANNOTATION] * 3)
        with self.assertRaises(AnnotationError):
            annotate(llm, TASK, self.skeleton, "plan_001", max_retries=3)


# place_order is called at L4 inside the L3 purchase branch.
MCP_MANIFEST = {"place_order": ["amount", "product_id"]}

GOOD_MCP_ANNOTATION = """\
```json
{
  "branches": {
    "root": {
      "description": "observe product page",
      "trigger_var": null,
      "http_constraints": {"allowed_domains": ["shop.example.com"], "fields": []},
      "mcp_constraints": null
    },
    "if_L3_true": {
      "description": "purchase branch",
      "trigger_var": "perceived_price",
      "http_constraints": {
        "allowed_domains": ["shop.example.com", "payments.example.com"],
        "fields": [{"path": "amount", "op": "<=", "value": "trigger_value"}]
      },
      "mcp_constraints": {
        "allowed_tools": ["place_order"],
        "param_rules": [
          {"tool": "place_order", "param": "amount", "op": "<=", "value": "trigger_value"},
          {"tool": "place_order", "param": "product_id", "source": "from_plan", "value": "SKU-7741"}
        ]
      }
    },
    "if_L3_false": {
      "description": "abort branch",
      "trigger_var": "perceived_price",
      "http_constraints": {"allowed_domains": [], "fields": []},
      "mcp_constraints": null
    }
  }
}
```
"""


class AnnotateMcpTest(unittest.TestCase):
    def setUp(self):
        self.skeleton = extract_skeleton(PLAN)

    def test_manifest_appears_in_prompt(self):
        llm = ScriptedLLM([GOOD_MCP_ANNOTATION])
        annotate(llm, TASK, self.skeleton, "plan_001", mcp_tools=MCP_MANIFEST)
        prompt = llm.prompts[0][1]
        self.assertIn("MCP tools available", prompt)
        self.assertIn("place_order", prompt)

    def test_mcp_annotation_accepted_and_parsed(self):
        llm = ScriptedLLM([GOOD_MCP_ANNOTATION])
        constraints = annotate(llm, TASK, self.skeleton, "plan_001", mcp_tools=MCP_MANIFEST)
        mcp = constraints.branches["if_L3_true"].mcp_constraints
        self.assertIsNotNone(mcp)
        self.assertEqual(mcp.allowed_tools, ["place_order"])
        self.assertEqual(mcp.param_rules[0].param, "amount")
        self.assertEqual(mcp.param_rules[1].source, "from_plan")

    def test_http_only_annotation_rejected_when_tool_unauthorised(self):
        # GOOD_ANNOTATION leaves mcp null, but with the manifest the L3 branch
        # calls place_order and must authorise it → rejected, retries exhaust
        llm = ScriptedLLM([GOOD_ANNOTATION] * 3)
        with self.assertRaises(AnnotationError):
            annotate(llm, TASK, self.skeleton, "plan_001", max_retries=3, mcp_tools=MCP_MANIFEST)


class GeneratePlanConstraintsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = BRHConfig(out_dir=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_writes_both_files(self):
        llm = ScriptedLLM([GOOD_ANNOTATION])
        log_dir = Path(self.tmp.name) / "logs"
        log_dir.mkdir()
        constraints = generate_plan_constraints(
            llm, TASK, PLAN, "plan_001", self.config, log_dir=log_dir
        )

        written = json.loads(self.config.constraints_path.read_text())
        self.assertEqual(written["plan_id"], "plan_001")
        self.assertEqual(written["generated_by"], "p-llm")
        self.assertEqual(set(written["branches"]), {"root", "if_L3_true", "if_L3_false"})
        self.assertEqual(
            written["branches"]["if_L3_true"]["http_constraints"]["fields"][0],
            {"path": "amount", "op": "<=", "value": "trigger_value"},
        )
        self.assertEqual(
            written["branches"]["if_L3_true"]["http_constraints"]["fields"][1],
            {"path": "currency", "op": "in", "value": ["GBP", "EUR"]},
        )

        state = json.loads(self.config.state_path.read_text())
        self.assertIsNone(state["active_branch"])
        self.assertEqual(state["plan_id"], "plan_001")
        self.assertTrue(state["ts"].endswith("Z"))

        self.assertTrue((log_dir / "plan_constraints_plan_001.json").exists())
        self.assertEqual(constraints.task, TASK)

    def test_annotation_failure_falls_back_to_static(self):
        llm = ScriptedLLM(["not json at all"] * 3)
        constraints = generate_plan_constraints(
            llm, TASK, PLAN, "plan_002", self.config, max_retries=3
        )
        self.assertEqual(constraints.generated_by, "static-fallback")
        written = json.loads(self.config.constraints_path.read_text())
        self.assertEqual(
            written["branches"]["if_L3_true"]["http_constraints"]["allowed_domains"],
            ["payments.example.com"],
        )
        # state file still reset, fail-closed
        state = json.loads(self.config.state_path.read_text())
        self.assertIsNone(state["active_branch"])

    def test_unexpected_annotator_crash_falls_back_to_static(self):
        # A non-AnnotationError crash inside annotate() (e.g. a malformed model
        # payload that trips `'NoneType' object is not subscriptable`) must NOT
        # propagate: it would reset branch_state to fail-closed null, which
        # over-blocks the plan's own benign traffic and mis-attributes the block
        # as brh_inactive. Degrade to the same static fallback as AnnotationError.
        class CrashingLLM:
            def __call__(self, system_prompt: str, user_prompt: str) -> str:
                raise TypeError("'NoneType' object is not subscriptable")

        constraints = generate_plan_constraints(
            CrashingLLM(), TASK, PLAN, "plan_crash", self.config, max_retries=3
        )
        self.assertEqual(constraints.generated_by, "static-fallback")
        written = json.loads(self.config.constraints_path.read_text())
        self.assertEqual(
            written["branches"]["if_L3_true"]["http_constraints"]["allowed_domains"],
            ["payments.example.com"],
        )
        # The constraints file exists and is consistent with the plan (the
        # enforcers never see a missing/stale file for this plan).
        self.assertEqual(set(written["branches"]), {"root", "if_L3_true", "if_L3_false"})

    def test_field_policy_seed_injects_subset_pin_into_every_branch(self):
        import os
        from unittest import mock

        seed = json.dumps([
            {"path": "sans", "op": "subset", "value": ["a.acme.local", "b.acme.local"]}
        ])
        llm = ScriptedLLM([GOOD_ANNOTATION])
        with mock.patch.dict(os.environ, {"BRH_SEED_FIELD_POLICY": seed}):
            constraints = generate_plan_constraints(
                llm, TASK, PLAN, "plan_seed", self.config
            )
        written = json.loads(self.config.constraints_path.read_text())
        for key, branch in written["branches"].items():
            pins = [
                f for f in branch["http_constraints"]["fields"]
                if f["path"] == "sans" and f["op"] == "subset"
            ]
            self.assertEqual(
                len(pins), 1, f"branch {key} must carry exactly one seeded subset pin"
            )
            self.assertEqual(pins[0]["value"], ["a.acme.local", "b.acme.local"])

    def test_field_policy_seed_absent_is_noop(self):
        import os
        from unittest import mock

        llm = ScriptedLLM([GOOD_ANNOTATION])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRH_SEED_FIELD_POLICY", None)
            constraints = generate_plan_constraints(
                llm, TASK, PLAN, "plan_noseed", self.config
            )
        written = json.loads(self.config.constraints_path.read_text())
        # root has no field pins in the good annotation; seed must not add any.
        self.assertEqual(written["branches"]["root"]["http_constraints"]["fields"], [])

    def test_no_partial_files_on_atomic_write(self):
        reset_branch_state("plan_x", self.config)
        leftovers = [
            p for p in Path(self.tmp.name).iterdir() if p.name.startswith(".branch_state")
        ]
        self.assertEqual(leftovers, [])


class ApplyToolServersTest(unittest.TestCase):
    def _make_constraints(self, allowed_tools):
        mcp = McpConstraints(allowed_tools=allowed_tools)
        branch = BranchConstraints(mcp_constraints=mcp)
        return PlanConstraints(
            plan_id="p", task="t",
            branches={"root": BranchConstraints(), "if_L1_true": branch, "if_L1_false": BranchConstraints()},
        )

    def test_injects_server_id_for_authorised_tool(self):
        constraints = self._make_constraints(["place_order"])
        _apply_tool_servers(constraints, {"place_order": "shop-server"})
        mcp = constraints.branches["if_L1_true"].mcp_constraints
        self.assertEqual(mcp.allowed_tool_servers, {"place_order": "shop-server"})

    def test_no_entry_for_tool_not_in_server_map(self):
        constraints = self._make_constraints(["place_order", "get_product"])
        _apply_tool_servers(constraints, {"place_order": "shop-server"})
        mcp = constraints.branches["if_L1_true"].mcp_constraints
        self.assertIn("place_order", mcp.allowed_tool_servers)
        self.assertNotIn("get_product", mcp.allowed_tool_servers)

    def test_does_not_overwrite_existing_entry(self):
        constraints = self._make_constraints(["place_order"])
        constraints.branches["if_L1_true"].mcp_constraints.allowed_tool_servers["place_order"] = "original"
        _apply_tool_servers(constraints, {"place_order": "other-server"})
        mcp = constraints.branches["if_L1_true"].mcp_constraints
        self.assertEqual(mcp.allowed_tool_servers["place_order"], "original")

    def test_noop_when_server_map_is_none(self):
        constraints = self._make_constraints(["place_order"])
        _apply_tool_servers(constraints, None)
        mcp = constraints.branches["if_L1_true"].mcp_constraints
        self.assertEqual(mcp.allowed_tool_servers, {})

    def test_noop_on_branch_without_mcp(self):
        constraints = self._make_constraints([])
        constraints.branches["if_L1_true"].mcp_constraints = None
        _apply_tool_servers(constraints, {"place_order": "shop-server"})
        self.assertIsNone(constraints.branches["if_L1_true"].mcp_constraints)


if __name__ == "__main__":
    unittest.main()
