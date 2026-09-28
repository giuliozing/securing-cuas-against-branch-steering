"""Unit tests for the two-phase tool approval module (cobra.mcp_proxy.approval).

All tests run without real LLMs or human interaction:
- P-LLM and Q-LLM are replaced by deterministic mock callables.
- human_select_tool is exercised via patched builtins.input.
- approval_loop AUTO mode requires no mocks at all.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from cobra.mcp_proxy.approval import (
    ApprovalMode,
    SufficiencyResult,
    ToolCandidate,
    approval_loop,
    human_select_tool,
    phase2_confirm_plan,
    select_tool_candidates,
    sufficiency_check,
)
from cobra.mcp_proxy.registry import load_registry, registry_key, tool_hash
from cobra.brh.schema import BranchConstraints, McpConstraints, McpParamRule, PlanConstraints

# -- shared fixtures ------------------------------------------------------------

_SEND_EMAIL = {
    "name": "send_email",
    "description": "Send an email to a recipient.",
    "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}},
}
_SEARCH_WEB = {
    "name": "search_web",
    "description": "Search the internet for a query.",
    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
_READ_FILE = {
    "name": "read_file",
    "description": "Read the contents of a local file.",
    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
}

_ALL_TOOLS = [_SEND_EMAIL, _SEARCH_WEB, _READ_FILE]
_SERVER_ID = "test-server"


def _mock_sufficient_llm(system: str, user: str) -> str:
    return '```json\n{"sufficient": true}\n```'


def _mock_insufficient_llm(capability: str):
    def _call(system: str, user: str) -> str:
        return f'```json\n{{"sufficient": false, "missing_capability": "{capability}"}}\n```'
    return _call


def _mock_qllm_picks(name: str):
    """Q-LLM that always recommends a single tool by name."""
    def _call(system: str, user: str) -> str:
        tool = next((t for t in _ALL_TOOLS if t["name"] == name), _SEND_EMAIL)
        return (
            '```json\n'
            f'[{{"name": "{tool["name"]}", "description": "{tool["description"]}", "reason": "best match"}}]'
            '\n```'
        )
    return _call


def _make_constraints(tool: str = "send_email") -> PlanConstraints:
    branch = BranchConstraints(
        description="send branch",
        mcp_constraints=McpConstraints(
            allowed_tools=[tool],
            param_rules=[McpParamRule(tool=tool, param="to", op="==", value="alice@example.com")],
            allowed_params={tool: ["to", "body"]},
            allowed_tool_servers={tool: _SERVER_ID},
        ),
    )
    return PlanConstraints(plan_id="p1", task="send email", branches={"root": branch})


# -- SufficiencyCheckTest -------------------------------------------------------

class SufficiencyCheckTest(unittest.TestCase):
    def test_sufficient_response(self):
        r = sufficiency_check(_mock_sufficient_llm, "send an email", {"send_email": ["to", "body"]})
        self.assertTrue(r.sufficient)
        self.assertIsNone(r.missing_capability)

    def test_insufficient_response(self):
        llm = _mock_insufficient_llm("send email")
        r = sufficiency_check(llm, "send an email", {})
        self.assertFalse(r.sufficient)
        self.assertEqual(r.missing_capability, "send email")

    def test_empty_manifest_string_shows_none(self):
        # should not crash; passes manifest={} gracefully
        r = sufficiency_check(_mock_sufficient_llm, "task", {})
        self.assertTrue(r.sufficient)

    def test_parse_failure_defaults_to_sufficient(self):
        r = sufficiency_check(lambda s, u: "not json at all", "task", {})
        self.assertTrue(r.sufficient)

    def test_missing_capability_field_fallback(self):
        llm = lambda s, u: '```json\n{"sufficient": false}\n```'
        r = sufficiency_check(llm, "task", {})
        self.assertFalse(r.sufficient)
        self.assertEqual(r.missing_capability, "unspecified capability")


# -- SelectToolCandidatesTest ---------------------------------------------------

class SelectToolCandidatesTest(unittest.TestCase):
    def test_returns_named_candidates(self):
        candidates = select_tool_candidates(
            _mock_qllm_picks("send_email"), "send email", _ALL_TOOLS, n=3
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "send_email")
        self.assertEqual(candidates[0].reason, "best match")

    def test_unknown_name_skipped(self):
        def _bad_llm(s, u):
            return '```json\n[{"name": "nonexistent_tool", "description": "x", "reason": "y"}]\n```'
        candidates = select_tool_candidates(_bad_llm, "something", _ALL_TOOLS)
        self.assertEqual(candidates, [])

    def test_empty_tool_list_returns_empty(self):
        candidates = select_tool_candidates(_mock_qllm_picks("send_email"), "req", [])
        self.assertEqual(candidates, [])

    def test_parse_failure_returns_empty(self):
        candidates = select_tool_candidates(lambda s, u: "not json", "req", _ALL_TOOLS)
        self.assertEqual(candidates, [])

    def test_n_cap_respected(self):
        def _three_llm(s, u):
            items = [
                {"name": t["name"], "description": t["description"], "reason": "ok"}
                for t in _ALL_TOOLS
            ]
            return '```json\n' + __import__('json').dumps(items) + '\n```'
        candidates = select_tool_candidates(_three_llm, "req", _ALL_TOOLS, n=2)
        self.assertLessEqual(len(candidates), 2)


# -- HumanSelectToolTest --------------------------------------------------------

class HumanSelectToolTest(unittest.TestCase):
    def _make_candidates(self) -> list[ToolCandidate]:
        return [
            ToolCandidate("send_email", "Send email", "best match", raw_tool=_SEND_EMAIL),
            ToolCandidate("search_web", "Search web", "second best", raw_tool=_SEARCH_WEB),
        ]

    def test_valid_selection(self):
        with patch("builtins.input", return_value="1"):
            choice = human_select_tool(self._make_candidates())
        self.assertIsNotNone(choice)
        self.assertEqual(choice.name, "send_email")

    def test_zero_returns_none(self):
        with patch("builtins.input", return_value="0"):
            choice = human_select_tool(self._make_candidates())
        self.assertIsNone(choice)

    def test_invalid_then_valid(self):
        responses = iter(["99", "abc", "2"])
        with patch("builtins.input", side_effect=lambda _: next(responses)):
            choice = human_select_tool(self._make_candidates())
        self.assertEqual(choice.name, "search_web")

    def test_empty_candidates_returns_none(self):
        choice = human_select_tool([])
        self.assertIsNone(choice)


# -- ApprovalLoopAutoTest -------------------------------------------------------

class ApprovalLoopAutoTest(unittest.TestCase):
    def test_auto_approves_all_tools_no_llm(self):
        with tempfile.TemporaryDirectory() as d:
            reg_path = os.path.join(d, "reg.json")
            manifest, server_map = approval_loop(
                "send an email",
                None,  # p_llm_call unused in AUTO
                None,  # q_llm_call unused in AUTO
                _ALL_TOOLS,
                _SERVER_ID,
                registry_path=reg_path,
                mode=ApprovalMode.AUTO,
            )
        self.assertEqual(set(manifest), {"send_email", "search_web", "read_file"})
        self.assertTrue(all(v == _SERVER_ID for v in server_map.values()))

    def test_auto_seeds_registry(self):
        with tempfile.TemporaryDirectory() as d:
            reg_path = os.path.join(d, "reg.json")
            approval_loop(
                "task", None, None, _ALL_TOOLS, _SERVER_ID,
                registry_path=reg_path, mode=ApprovalMode.AUTO,
            )
            reg = load_registry(reg_path)
        for tool in _ALL_TOOLS:
            key = registry_key(_SERVER_ID, tool["name"])
            self.assertIn(key, reg)
            self.assertEqual(reg[key]["hash"], tool_hash(tool))
            self.assertTrue(reg[key]["approved"])

    def test_auto_empty_tool_list(self):
        manifest, server_map = approval_loop(
            "task", None, None, [], _SERVER_ID, mode=ApprovalMode.AUTO,
        )
        self.assertEqual(manifest, {})
        self.assertEqual(server_map, {})


# -- ApprovalLoopInteractiveTest ------------------------------------------------

class ApprovalLoopInteractiveTest(unittest.TestCase):
    def test_single_round_approval(self):
        """P-LLM says insufficient once (email), Q-LLM picks send_email, human picks [1].
        Second round: P-LLM says sufficient."""
        insufficient = _mock_insufficient_llm("send email")
        call_count = [0]

        def p_llm(s, u):
            call_count[0] += 1
            if call_count[0] == 1:
                return insufficient(s, u)
            return _mock_sufficient_llm(s, u)

        with tempfile.TemporaryDirectory() as d:
            reg_path = os.path.join(d, "reg.json")
            with patch("builtins.input", return_value="1"):
                manifest, server_map = approval_loop(
                    "send an email",
                    p_llm,
                    _mock_qllm_picks("send_email"),
                    _ALL_TOOLS,
                    _SERVER_ID,
                    registry_path=reg_path,
                    mode=ApprovalMode.INTERACTIVE,
                )
            reg = load_registry(reg_path)

        self.assertIn("send_email", manifest)
        self.assertEqual(server_map["send_email"], _SERVER_ID)
        self.assertIn(registry_key(_SERVER_ID, "send_email"), reg)

    def test_prior_approved_tools_loaded_from_registry(self):
        """If send_email is already in the registry, the loop starts with it in the manifest."""
        with tempfile.TemporaryDirectory() as d:
            reg_path = os.path.join(d, "reg.json")
            # pre-seed via AUTO mode
            approval_loop(
                "task", None, None, [_SEND_EMAIL], _SERVER_ID,
                registry_path=reg_path, mode=ApprovalMode.AUTO,
            )
            # now run INTERACTIVE — P-LLM says sufficient immediately
            manifest, _ = approval_loop(
                "send email",
                _mock_sufficient_llm,
                None,
                _ALL_TOOLS,
                _SERVER_ID,
                registry_path=reg_path,
                mode=ApprovalMode.INTERACTIVE,
            )
        self.assertIn("send_email", manifest)

    def test_human_picks_none_breaks_loop(self):
        """Human picks 0 (none) — loop stops without adding any tool."""
        with tempfile.TemporaryDirectory() as d:
            reg_path = os.path.join(d, "reg.json")
            with patch("builtins.input", return_value="0"):
                manifest, _ = approval_loop(
                    "task",
                    _mock_insufficient_llm("email"),
                    _mock_qllm_picks("send_email"),
                    _ALL_TOOLS,
                    _SERVER_ID,
                    registry_path=reg_path,
                    mode=ApprovalMode.INTERACTIVE,
                )
        self.assertEqual(manifest, {})

    def test_max_rounds_guard(self):
        """Loop exits after max_rounds even if P-LLM always says insufficient."""
        with tempfile.TemporaryDirectory() as d:
            reg_path = os.path.join(d, "reg.json")
            responses = iter(["1", "1", "1"])
            with patch("builtins.input", side_effect=lambda _: next(responses)):
                manifest, _ = approval_loop(
                    "task",
                    _mock_insufficient_llm("something"),
                    _mock_qllm_picks("send_email"),
                    _ALL_TOOLS,
                    _SERVER_ID,
                    registry_path=reg_path,
                    mode=ApprovalMode.INTERACTIVE,
                    max_rounds=3,
                )
        # after 3 rounds approving send_email (idempotent), it is in the manifest
        self.assertIn("send_email", manifest)


# -- Phase2ConfirmPlanTest ------------------------------------------------------

class Phase2ConfirmPlanTest(unittest.TestCase):
    def test_auto_mode_is_passthrough(self):
        c = _make_constraints("send_email")
        result = phase2_confirm_plan(c, mode=ApprovalMode.AUTO)
        self.assertIs(result, c)
        self.assertIn("send_email", result.branches["root"].mcp_constraints.allowed_tools)

    def test_interactive_approve_all(self):
        c = _make_constraints("send_email")
        with patch("builtins.input", return_value=""):
            result = phase2_confirm_plan(c, mode=ApprovalMode.INTERACTIVE)
        self.assertIn("send_email", result.branches["root"].mcp_constraints.allowed_tools)

    def test_interactive_veto_removes_tool(self):
        c = _make_constraints("send_email")
        with patch("builtins.input", return_value="1"):  # veto item [1] = send_email
            result = phase2_confirm_plan(c, mode=ApprovalMode.INTERACTIVE)
        mcp = result.branches["root"].mcp_constraints
        self.assertNotIn("send_email", mcp.allowed_tools)
        self.assertEqual(mcp.param_rules, [])
        self.assertNotIn("send_email", mcp.allowed_params)
        self.assertNotIn("send_email", mcp.allowed_tool_servers)

    def test_interactive_invalid_input_keeps_all(self):
        c = _make_constraints("send_email")
        with patch("builtins.input", return_value="abc"):
            result = phase2_confirm_plan(c, mode=ApprovalMode.INTERACTIVE)
        self.assertIn("send_email", result.branches["root"].mcp_constraints.allowed_tools)

    def test_interactive_branch_without_mcp_skipped(self):
        branch = BranchConstraints(description="http only", mcp_constraints=None)
        c = PlanConstraints(plan_id="p2", task="t", branches={"root": branch})
        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            result = phase2_confirm_plan(c, mode=ApprovalMode.INTERACTIVE)
        self.assertIsNone(result.branches["root"].mcp_constraints)


if __name__ == "__main__":
    unittest.main()
