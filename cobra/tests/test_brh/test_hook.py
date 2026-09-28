"""Tests for the runtime BRH hook (`cobra.brh.hook`).

The hook is deliberately interpreter-independent: `namespace` and
`eval_args` are duck-typed, so these tests run with the system python
(no pydantic_ai / agentdojo needed) using small fakes that mirror the
interpreter's `Namespace.get(...).raw` and frozen-dataclass `EvalArgs`.
"""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from cobra.brh.hook import (
    BRHBranchStep,
    BRHRuntime,
    activate_root,
    attach,
    on_branch_entry,
    on_branch_exit,
)
from cobra.brh.schema import (
    BranchConstraints,
    FieldConstraint,
    HttpConstraints,
    McpConstraints,
    McpParamRule,
    PlanConstraints,
)


class FakeCaMeLValue:
    def __init__(self, raw):
        self.raw = raw


class FakeNamespace:
    def __init__(self, variables=None):
        self.variables = variables or {}

    def get(self, name):
        return self.variables.get(name)


@dataclasses.dataclass(frozen=True)
class FakeEvalArgs:
    brh_runtime: object = None
    brh_branch_path: tuple = ()


def make_constraints(plan_id="plan_test", branches=None):
    return PlanConstraints(
        plan_id=plan_id,
        task="buy a USB-C hub if price is at most 50",
        branches=branches
        or {
            "root": BranchConstraints(
                description="observe product page",
                http_constraints=HttpConstraints(allowed_domains=["shop.example.com"]),
            ),
            "if_L3_true": BranchConstraints(
                description="purchase branch",
                trigger_var="perceived_price",
                http_constraints=HttpConstraints(
                    allowed_domains=["shop.example.com", "payments.example.com"],
                    fields=[
                        FieldConstraint(path="amount", op="<=", value="trigger_value"),
                        FieldConstraint(path="currency", op="in", value=["GBP", "EUR"]),
                        FieldConstraint(path="product_id", op="==", value="SKU-7741"),
                    ],
                ),
            ),
            "if_L3_false": BranchConstraints(
                description="abort branch",
                trigger_var="perceived_price",
            ),
        },
    )


class HookTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = Path(self._tmp.name) / "branch_state.json"

    def runtime(self, constraints=None):
        return BRHRuntime(constraints=constraints or make_constraints(), state_path=self.state_path)

    def read_state(self):
        with open(self.state_path, encoding="utf-8") as f:
            return json.load(f)


class TestNoOp(HookTestCase):
    def test_no_runtime_returns_eval_args_unchanged_and_writes_nothing(self):
        eval_args = FakeEvalArgs()
        out = on_branch_entry(3, True, FakeNamespace(), eval_args)
        self.assertIs(out, eval_args)
        self.assertFalse(self.state_path.exists())

    def test_activate_root_with_none_runtime_is_noop(self):
        activate_root(None)  # must not raise


class TestAttach(HookTestCase):
    def test_attach_sets_runtime_and_resets_path(self):
        rt = self.runtime()
        eval_args = FakeEvalArgs(brh_branch_path=(BRHBranchStep("if_L9_true", 1),))
        attached = attach(eval_args, rt)
        self.assertIs(attached.brh_runtime, rt)
        self.assertEqual(attached.brh_branch_path, ())


class TestRootActivation(HookTestCase):
    def test_root_state_written_with_root_constraints(self):
        activate_root(self.runtime())
        state = self.read_state()
        self.assertEqual(state["plan_id"], "plan_test")
        self.assertEqual(state["active_branch"], "root")
        self.assertEqual(state["branch_path"], ["root"])
        self.assertIsNone(state["trigger_var"])
        self.assertIsNone(state["trigger_value"])
        self.assertEqual(state["http_constraints"]["allowed_domains"], ["shop.example.com"])
        self.assertEqual(state["http_constraints"]["fields"], [])
        self.assertIsNone(state["mcp_constraints"])
        self.assertIn("ts", state)


class TestBranchEntry(HookTestCase):
    def test_single_branch_resolves_trigger_value(self):
        rt = self.runtime()
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue(42.99)})
        out = on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))

        self.assertEqual(out.brh_branch_path, (BRHBranchStep("if_L3_true", 42.99),))
        state = self.read_state()
        self.assertEqual(state["active_branch"], "if_L3_true")
        self.assertEqual(state["description"], "purchase branch")
        self.assertEqual(state["trigger_var"], "perceived_price")
        self.assertEqual(state["trigger_value"], 42.99)
        self.assertEqual(state["branch_path"], ["root", "if_L3_true"])
        # Root and branch domains merged, deduplicated, order-preserving.
        self.assertEqual(
            state["http_constraints"]["allowed_domains"],
            ["shop.example.com", "payments.example.com"],
        )
        fields = {f["path"]: f for f in state["http_constraints"]["fields"]}
        self.assertEqual(fields["amount"], {"path": "amount", "op": "<=", "value": 42.99})
        self.assertEqual(fields["product_id"]["value"], "SKU-7741")

    def test_false_arm_inherits_only_root(self):
        rt = self.runtime()
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue(99)})
        out = on_branch_entry(3, False, ns, attach(FakeEvalArgs(), rt))

        self.assertEqual(out.brh_branch_path, (BRHBranchStep("if_L3_false", 99),))
        state = self.read_state()
        self.assertEqual(state["active_branch"], "if_L3_false")
        self.assertEqual(state["trigger_value"], 99)
        self.assertEqual(state["http_constraints"]["allowed_domains"], ["shop.example.com"])
        self.assertEqual(state["http_constraints"]["fields"], [])

    def test_in_set_is_never_resolved(self):
        rt = self.runtime()
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue(10)})
        on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))
        fields = {f["path"]: f for f in self.read_state()["http_constraints"]["fields"]}
        self.assertEqual(fields["currency"]["value"], ["GBP", "EUR"])

    def test_missing_trigger_var_leaves_placeholder_unresolved(self):
        rt = self.runtime()
        out = on_branch_entry(3, True, FakeNamespace(), attach(FakeEvalArgs(), rt))
        self.assertEqual(out.brh_branch_path, (BRHBranchStep("if_L3_true", None),))
        state = self.read_state()
        self.assertIsNone(state["trigger_value"])
        fields = {f["path"]: f for f in state["http_constraints"]["fields"]}
        # Unresolved placeholder passes through; the enforcer fails closed on it.
        self.assertEqual(fields["amount"]["value"], "trigger_value")

    def test_non_scalar_trigger_value_becomes_null(self):
        rt = self.runtime()
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue([1, 2, 3])})
        on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))
        state = self.read_state()
        self.assertIsNone(state["trigger_value"])
        fields = {f["path"]: f for f in state["http_constraints"]["fields"]}
        self.assertEqual(fields["amount"]["value"], "trigger_value")

    def test_bool_trigger_value_is_kept_as_bool(self):
        branches = {
            "root": BranchConstraints(),
            "if_L2_true": BranchConstraints(
                trigger_var="confirmed",
                http_constraints=HttpConstraints(
                    allowed_domains=["shop.example.com"],
                    fields=[FieldConstraint(path="confirmed", op="==", value="trigger_value")],
                ),
            ),
            "if_L2_false": BranchConstraints(),
        }
        rt = self.runtime(make_constraints(branches=branches))
        ns = FakeNamespace({"confirmed": FakeCaMeLValue(True)})
        on_branch_entry(2, True, ns, attach(FakeEvalArgs(), rt))
        state = self.read_state()
        self.assertIs(state["trigger_value"], True)
        self.assertIs(state["http_constraints"]["fields"][0]["value"], True)


class TestNestedBranches(HookTestCase):
    def make_nested(self):
        return make_constraints(
            branches={
                "root": BranchConstraints(
                    http_constraints=HttpConstraints(allowed_domains=["shop.example.com"]),
                ),
                "if_L4_true": BranchConstraints(
                    description="price ok",
                    trigger_var="price",
                    http_constraints=HttpConstraints(
                        allowed_domains=["payments.example.com"],
                        fields=[FieldConstraint(path="amount", op="<=", value="trigger_value")],
                    ),
                ),
                "if_L4_false": BranchConstraints(trigger_var="price"),
                "if_L4_true.if_L7_true": BranchConstraints(
                    description="in stock",
                    trigger_var="stock",
                    http_constraints=HttpConstraints(
                        # Duplicate of root's domain: must be deduplicated.
                        allowed_domains=["shop.example.com"],
                        fields=[FieldConstraint(path="quantity", op="<=", value="trigger_value")],
                    ),
                ),
                "if_L4_true.if_L7_false": BranchConstraints(),
            }
        )

    def test_nested_entry_merges_ancestors_with_per_arm_resolution(self):
        rt = self.runtime(self.make_nested())
        ns = FakeNamespace({"price": FakeCaMeLValue(42.99), "stock": FakeCaMeLValue(3)})

        eval_args = attach(FakeEvalArgs(), rt)
        after_outer = on_branch_entry(4, True, ns, eval_args)
        # The outer trigger variable gets reassigned before the inner if:
        # the ancestor's placeholder must keep the value captured at entry.
        ns.variables["price"] = FakeCaMeLValue(999)
        after_inner = on_branch_entry(7, True, ns, after_outer)

        self.assertEqual(
            after_inner.brh_branch_path,
            (BRHBranchStep("if_L4_true", 42.99), BRHBranchStep("if_L7_true", 3)),
        )
        state = self.read_state()
        self.assertEqual(state["active_branch"], "if_L4_true.if_L7_true")
        self.assertEqual(state["branch_path"], ["root", "if_L4_true", "if_L7_true"])
        self.assertEqual(state["trigger_var"], "stock")
        self.assertEqual(state["trigger_value"], 3)
        self.assertEqual(
            state["http_constraints"]["allowed_domains"],
            ["shop.example.com", "payments.example.com"],
        )
        fields = {f["path"]: f["value"] for f in state["http_constraints"]["fields"]}
        self.assertEqual(fields["amount"], 42.99)  # ancestor's own captured value
        self.assertEqual(fields["quantity"], 3)

    def test_sibling_after_nested_uses_parent_path(self):
        # The eval_args returned for a body must not leak to siblings: a
        # second top-level if uses the original (root) path again.
        rt = self.runtime(self.make_nested())
        ns = FakeNamespace({"price": FakeCaMeLValue(10), "stock": FakeCaMeLValue(1)})
        eval_args = attach(FakeEvalArgs(), rt)
        on_branch_entry(4, True, ns, eval_args)
        # Simulating the interpreter: sibling `if` receives `eval_args`,
        # not the body's extended copy.
        out = on_branch_entry(4, False, ns, eval_args)
        self.assertEqual(out.brh_branch_path, (BRHBranchStep("if_L4_false", 10),))
        self.assertEqual(self.read_state()["active_branch"], "if_L4_false")


class TestBranchExit(HookTestCase):
    def test_exit_top_level_restores_root(self):
        # The disk state must not stay frozen on the entered branch after
        # the if block ends: a post-branch request in the parent (root)
        # scope must see root constraints, not the branch's super-set.
        rt = self.runtime()
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue(42.99)})
        eval_args = attach(FakeEvalArgs(), rt)
        on_branch_entry(3, True, ns, eval_args)
        self.assertEqual(
            self.read_state()["http_constraints"]["allowed_domains"],
            ["shop.example.com", "payments.example.com"],
        )
        # _eval_if passes the *parent* eval_args (root path) on exit.
        on_branch_exit(eval_args)
        state = self.read_state()
        self.assertEqual(state["active_branch"], "root")
        self.assertEqual(state["branch_path"], ["root"])
        self.assertIsNone(state["trigger_var"])
        self.assertIsNone(state["trigger_value"])
        self.assertEqual(state["http_constraints"]["allowed_domains"], ["shop.example.com"])

    def test_exit_nested_pops_one_level(self):
        # Leaving the inner if restores the outer arm's scope (not root),
        # with the outer arm's own captured trigger value preserved.
        rt = self.runtime(TestNestedBranches().make_nested())
        ns = FakeNamespace({"price": FakeCaMeLValue(42.99), "stock": FakeCaMeLValue(3)})
        eval_args = attach(FakeEvalArgs(), rt)
        after_outer = on_branch_entry(4, True, ns, eval_args)
        on_branch_entry(7, True, ns, after_outer)
        # Inner if exits → parent is the outer arm (after_outer's path).
        on_branch_exit(after_outer)
        state = self.read_state()
        self.assertEqual(state["active_branch"], "if_L4_true")
        self.assertEqual(state["branch_path"], ["root", "if_L4_true"])
        self.assertEqual(state["trigger_var"], "price")
        self.assertEqual(state["trigger_value"], 42.99)
        self.assertEqual(
            state["http_constraints"]["allowed_domains"],
            ["shop.example.com", "payments.example.com"],
        )
        fields = {f["path"]: f["value"] for f in state["http_constraints"]["fields"]}
        self.assertEqual(fields["amount"], 42.99)

    def test_exit_with_none_runtime_is_noop(self):
        on_branch_exit(FakeEvalArgs())  # no runtime → must not raise or write
        self.assertFalse(self.state_path.exists())

    def test_exit_never_raises_on_write_failure(self):
        blocker = Path(self._tmp.name) / "blocker"
        blocker.write_text("not a directory")
        rt = BRHRuntime(constraints=make_constraints(), state_path=blocker / "branch_state.json")
        on_branch_exit(attach(FakeEvalArgs(), rt))  # must not raise


class TestVarPlaceholder(HookTestCase):
    """`"var:<name>"` field bounds resolve from a named plan variable that is
    not the branch trigger."""

    def make_var(self, nested=False):
        branches = {
            "root": BranchConstraints(
                http_constraints=HttpConstraints(allowed_domains=["shop.example.com"]),
            ),
            # The buying arm branches on `tier`, but the amount is capped by the
            # `price` read earlier (not a trigger here) — expressed as var:price.
            "if_L3_true": BranchConstraints(
                description="gold member buys",
                trigger_var="tier",
                http_constraints=HttpConstraints(
                    allowed_domains=["shop.example.com"],
                    fields=[FieldConstraint(path="amount", op="<=", value="var:price")],
                ),
            ),
            "if_L3_false": BranchConstraints(trigger_var="tier"),
        }
        if nested:
            branches["if_L3_true.if_L5_true"] = BranchConstraints(trigger_var="stock")
            branches["if_L3_true.if_L5_false"] = BranchConstraints(trigger_var="stock")
        return make_constraints(branches=branches)

    def test_var_bound_resolves_to_named_variable_not_trigger(self):
        rt = self.runtime(self.make_var())
        ns = FakeNamespace({"tier": FakeCaMeLValue("gold"), "price": FakeCaMeLValue(40.0)})
        on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))
        fields = {f["path"]: f["value"] for f in self.read_state()["http_constraints"]["fields"]}
        self.assertEqual(fields["amount"], 40.0)  # the price, not the trigger "gold"

    def test_var_bound_unresolved_when_variable_missing(self):
        rt = self.runtime(self.make_var())
        ns = FakeNamespace({"tier": FakeCaMeLValue("gold")})  # no `price` in scope
        on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))
        fields = {f["path"]: f["value"] for f in self.read_state()["http_constraints"]["fields"]}
        self.assertEqual(fields["amount"], "var:price")  # stays placeholder => fail-closed

    def make_var_mcp(self):
        # The gold-member buy arm branches on `tier`, not price, and acts via an
        # MCP tool whose amount is capped by var:price — and, unlike make_var,
        # carries NO HTTP field referencing var:price. Reproduces mc11: the
        # var: bound exists only in mcp_constraints.param_rules.
        branches = {
            "root": BranchConstraints(
                http_constraints=HttpConstraints(allowed_domains=["shop.example.com"]),
            ),
            "if_L3_true": BranchConstraints(
                description="gold member buys via MCP",
                trigger_var="tier",
                mcp_constraints=McpConstraints(
                    allowed_tools=["place_order"],
                    param_rules=[
                        McpParamRule(tool="place_order", param="amount", op="<=", value="var:price"),
                    ],
                ),
            ),
            "if_L3_false": BranchConstraints(trigger_var="tier"),
        }
        return make_constraints(branches=branches)

    def test_mcp_var_bound_resolves_from_param_rule_only(self):
        # Regression: the hook used to read var: names only from HTTP fields, so
        # an MCP-only var:price (no twin HTTP field) was never resolved -> stayed
        # a placeholder -> MCP proxy fail-closed -> benign order blocked (mc11).
        rt = self.runtime(self.make_var_mcp())
        ns = FakeNamespace({"tier": FakeCaMeLValue("gold"), "price": FakeCaMeLValue(42.99)})
        on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))
        rules = {r["param"]: r["value"] for r in self.read_state()["mcp_constraints"]["param_rules"]}
        self.assertEqual(rules["amount"], 42.99)  # resolved from `price`, not left as "var:price"

    def test_mcp_var_bound_unresolved_when_variable_missing(self):
        rt = self.runtime(self.make_var_mcp())
        ns = FakeNamespace({"tier": FakeCaMeLValue("gold")})  # no `price` in scope
        on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))
        rules = {r["param"]: r["value"] for r in self.read_state()["mcp_constraints"]["param_rules"]}
        self.assertEqual(rules["amount"], "var:price")  # stays placeholder => fail-closed

    def test_root_mcp_var_bound_is_dropped_not_fail_closed(self):
        # The MCPTox Track 2 Financial::12::28 false positive. The plan loops over
        # discovered tickers calling get_current_crypto_price(ticker=t); the
        # annotator pins ticker == var:t at ROOT. `t` is a loop variable, never
        # captured (root state is written once at plan start). Fail-closed would
        # block every benign ticker; instead the rule is dropped and the tool
        # stays gated by allowed_tools (the cross-tool defense).
        branches = {
            "root": BranchConstraints(
                description="iterate fetching BTC price",
                mcp_constraints=McpConstraints(
                    allowed_tools=["get_available_crypto_tickers", "get_current_crypto_price"],
                    param_rules=[
                        McpParamRule(tool="get_current_crypto_price", param="ticker",
                                     op="==", value="var:t"),
                    ],
                ),
            ),
        }
        rt = self.runtime(make_constraints(branches=branches))
        activate_root(rt)
        state = self.read_state()
        self.assertEqual(state["active_branch"], "root")
        # the allowlist (the actual cross-tool defense) survives intact
        self.assertEqual(state["mcp_constraints"]["allowed_tools"],
                         ["get_available_crypto_tickers", "get_current_crypto_price"])
        # the unresolvable var:t rule is gone (no fail-closed placeholder left)
        self.assertEqual(state["mcp_constraints"]["param_rules"], [])

    def test_var_bound_survives_pop(self):
        # The captured value travels in the step, so exiting a child (no
        # namespace) restores the parent's resolved bound — and a later
        # reassignment of `price` must not change it.
        rt = self.runtime(self.make_var(nested=True))
        ns = FakeNamespace(
            {"tier": FakeCaMeLValue("gold"), "price": FakeCaMeLValue(40.0), "stock": FakeCaMeLValue(2)}
        )
        eval_args = attach(FakeEvalArgs(), rt)
        after_outer = on_branch_entry(3, True, ns, eval_args)   # captures price=40
        on_branch_entry(5, True, ns, after_outer)               # child inherits 40
        child = {f["path"]: f["value"] for f in self.read_state()["http_constraints"]["fields"]}
        self.assertEqual(child["amount"], 40.0)
        ns.variables["price"] = FakeCaMeLValue(999.0)
        on_branch_exit(after_outer)                             # pop, no namespace re-read
        parent = {f["path"]: f["value"] for f in self.read_state()["http_constraints"]["fields"]}
        self.assertEqual(parent["amount"], 40.0)


class TestFailClosed(HookTestCase):
    def test_unknown_branch_key_writes_empty_allowlist(self):
        rt = self.runtime()
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue(5)})
        out = on_branch_entry(99, True, ns, attach(FakeEvalArgs(), rt))
        # Path is still extended (deeper writes stay fail-closed too).
        self.assertEqual(out.brh_branch_path, (BRHBranchStep("if_L99_true", None),))
        state = self.read_state()
        self.assertEqual(state["active_branch"], "if_L99_true")
        self.assertEqual(state["http_constraints"], {"allowed_domains": [], "fields": [], "allowed_endpoints": []})
        self.assertIsNone(state["mcp_constraints"])

    def test_write_failure_never_raises_and_keeps_eval_args(self):
        blocker = Path(self._tmp.name) / "blocker"
        blocker.write_text("not a directory")
        rt = BRHRuntime(
            constraints=make_constraints(),
            state_path=blocker / "sub" / "branch_state.json",
        )
        eval_args = attach(FakeEvalArgs(), rt)
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue(5)})
        out = on_branch_entry(3, True, ns, eval_args)  # must not raise
        self.assertIs(out, eval_args)
        activate_root(rt)  # must not raise

    def test_broken_namespace_never_raises(self):
        class ExplodingNamespace:
            def get(self, name):
                raise RuntimeError("boom")

        rt = self.runtime()
        eval_args = attach(FakeEvalArgs(), rt)
        out = on_branch_entry(3, True, ExplodingNamespace(), eval_args)
        self.assertIs(out, eval_args)


class TestMcpConstraints(HookTestCase):
    def test_mcp_merge_and_param_rule_resolution(self):
        branches = {
            "root": BranchConstraints(),
            "if_L2_true": BranchConstraints(
                trigger_var="price",
                http_constraints=HttpConstraints(allowed_domains=["shop.example.com"]),
                mcp_constraints=McpConstraints(
                    allowed_tools=["get_product", "place_order"],
                    param_rules=[
                        McpParamRule(tool="place_order", param="amount", op="<=", value="trigger_value"),
                        McpParamRule(tool="place_order", param="product_id", source="from_plan"),
                    ],
                ),
            ),
            "if_L2_false": BranchConstraints(),
        }
        rt = self.runtime(make_constraints(branches=branches))
        ns = FakeNamespace({"price": FakeCaMeLValue(42.99)})
        on_branch_entry(2, True, ns, attach(FakeEvalArgs(), rt))
        mcp = self.read_state()["mcp_constraints"]
        self.assertEqual(mcp["allowed_tools"], ["get_product", "place_order"])
        rules = {r["param"]: r for r in mcp["param_rules"]}
        self.assertEqual(rules["amount"]["value"], 42.99)
        # "from_plan" is not resolvable at branch entry: passes through.
        self.assertEqual(rules["product_id"]["source"], "from_plan")
        self.assertIsNone(rules["product_id"]["value"])

    def test_all_branches_without_mcp_yield_null(self):
        rt = self.runtime()
        ns = FakeNamespace({"perceived_price": FakeCaMeLValue(5)})
        on_branch_entry(3, True, ns, attach(FakeEvalArgs(), rt))
        self.assertIsNone(self.read_state()["mcp_constraints"])


class TestToolServersReachTheState(unittest.TestCase):
    """`allowed_tool_servers` must survive the plan -> branch_state hop.

    MCP proxy reads the pin from the STATE (`mcp_proxy/check.py`), so one that exists only in
    plan_constraints.json enforces nothing: if the merge builds the MCP block without
    the key, the same-named-squatter defence is unreachable through the BRH path
    however correctly the map was built upstream."""

    def _merged(self, servers):
        from cobra.brh.hook import _merge_constraints
        from cobra.brh.schema import PlanConstraints
        plan = PlanConstraints.model_validate({
            "plan_id": "p", "task": "t",
            "branches": {"root": {
                "description": "", "trigger_var": None,
                "http_constraints": {"allowed_domains": [], "fields": [],
                                     "allowed_endpoints": []},
                "mcp_constraints": {"allowed_tools": ["pay.issue"], "param_rules": [],
                                    "allowed_params": {},
                                    "allowed_tool_servers": servers}}}})
        return _merge_constraints(plan, ())[1]

    def test_pin_is_written_and_normalised_to_a_list(self):
        self.assertEqual(self._merged({"pay.issue": "steerweb"})["allowed_tool_servers"],
                         {"pay.issue": ["steerweb"]})
        self.assertEqual(self._merged({"pay.issue": ["a", "b"]})["allowed_tool_servers"],
                         {"pay.issue": ["a", "b"]})

    def test_absent_pin_stays_absent_not_empty_fail_closed(self):
        # No pin means name-only matching, the documented default. An empty dict must
        # not be read as "no server may serve this tool".
        self.assertEqual(self._merged({})["allowed_tool_servers"], {})

    def test_enforcer_accepts_the_approved_server_and_blocks_a_squatter(self):
        from cobra.mcp_proxy.check import check_tools_call
        state = {"active_branch": "root",
                 "mcp_constraints": self._merged({"pay.issue": "steerweb"})}
        self.assertTrue(check_tools_call(state, "pay.issue", {}, server_id="steerweb").allow)
        bad = check_tools_call(state, "pay.issue", {}, server_id="squatter")
        self.assertFalse(bad.allow)
        self.assertEqual(bad.reason, "mpt_tool")


if __name__ == "__main__":
    unittest.main()
