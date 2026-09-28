"""MCP proxy enforcement for the non-scalar residuals.

Covers the two operators added so the value-steer-list and arg-add attacks
MCPTox Track 1 could not block are now blockable:

  * ``eq_struct`` / ``subset`` param rules (structural pinning of list/dict args);
  * ``allowed_params`` schema-closed mode (a sealed parameter set blocks an
    injected extra argument).

These ride on ``cobra.brh.contract.satisfies`` (subset/eq_struct), pinned to the
HTTP proxy copy by the shared golden vectors, so here we test only MCP proxy's wiring of
them in ``check_tools_call``.
"""

import unittest

from cobra.mcp_proxy.check import check_tools_call


def _state(allowed=None, rules=None, sealed=None, active="root"):
    mcp = {"allowed_tools": allowed or [], "param_rules": rules or []}
    if sealed is not None:
        mcp["allowed_params"] = sealed
    return {"active_branch": active, "mcp_constraints": mcp}


class SubsetRule(unittest.TestCase):
    # Memory::22::89 — open_nodes(names=[AI, Blockchain]) + injected TrackerNode
    def setUp(self):
        self.st = _state(allowed=["open_nodes"], rules=[
            {"tool": "open_nodes", "param": "names", "op": "subset",
             "value": ["Artificial Intelligence", "Blockchain"]}])

    def test_exact_set_passes(self):
        self.assertTrue(check_tools_call(
            self.st, "open_nodes", {"names": ["Artificial Intelligence", "Blockchain"]}).allow)

    def test_strict_subset_passes(self):
        self.assertTrue(check_tools_call(
            self.st, "open_nodes", {"names": ["Blockchain"]}).allow)

    def test_appended_member_blocks(self):
        d = check_tools_call(self.st, "open_nodes",
                             {"names": ["Artificial Intelligence", "Blockchain", "TrackerNode"]})
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "mpt_param")

    def test_absent_param_still_passes(self):
        # parity with scalar rules: a call not carrying the param is gated by the
        # allowlist, not the value rule
        self.assertTrue(check_tools_call(self.st, "open_nodes", {}).allow)


class EqStructRule(unittest.TestCase):
    # Git::2::35 — git_add(files=['src/utils.js']) -> git_add(files='.')
    def setUp(self):
        self.st = _state(allowed=["git_add"], rules=[
            {"tool": "git_add", "param": "files", "op": "eq_struct",
             "value": ["src/utils.js"]}])

    def test_exact_list_passes(self):
        self.assertTrue(check_tools_call(self.st, "git_add", {"files": ["src/utils.js"]}).allow)

    def test_list_to_dot_string_blocks(self):
        d = check_tools_call(self.st, "git_add", {"files": "."})
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "mpt_param")

    def test_extra_file_blocks(self):
        self.assertFalse(check_tools_call(
            self.st, "git_add", {"files": ["src/utils.js", "secret.env"]}).allow)


class FromPlanListPin(unittest.TestCase):
    # The MCPTox Track 2 FP: a real annotator pins a non-scalar arg with a
    # ``from_plan`` rule (op=null -> resolved to ==), exactly as gpt-5 emitted for
    # Memory::22::89. The benign call equals the pinned list, so it MUST pass; an
    # exact pin on a list now means structural equality (not the old always-False
    # guard that blocked even an identical list).
    def setUp(self):
        self.st = _state(allowed=["open_nodes"], rules=[
            {"tool": "open_nodes", "param": "names", "source": "from_plan", "op": None,
             "value": ["Artificial Intelligence", "Blockchain"]}])

    def test_identical_list_passes(self):  # the FP that regressed before the fix
        self.assertTrue(check_tools_call(
            self.st, "open_nodes", {"names": ["Artificial Intelligence", "Blockchain"]}).allow)

    def test_steered_list_blocks(self):  # bonus: value-steer on a list now caught
        d = check_tools_call(self.st, "open_nodes",
                             {"names": ["Artificial Intelligence", "Blockchain", "TrackerNode"]})
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "mpt_param")


class SealedParams(unittest.TestCase):
    # ClickHouse::72::205 — list_tables(database=main) + injected like='prod_%'
    def setUp(self):
        self.st = _state(allowed=["list_tables"],
                         rules=[{"tool": "list_tables", "param": "database",
                                 "op": "==", "value": "main"}],
                         sealed={"list_tables": ["database"]})

    def test_planned_call_passes(self):
        self.assertTrue(check_tools_call(self.st, "list_tables", {"database": "main"}).allow)

    def test_injected_extra_param_blocks(self):
        d = check_tools_call(self.st, "list_tables", {"database": "main", "like": "prod_%"})
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "mpt_param")
        self.assertEqual(d.detail.get("param"), "like")

    def test_unsealed_tool_keeps_open_behaviour(self):
        # a tool absent from allowed_params still lets extra params through
        st = _state(allowed=["list_tables", "other"],
                    sealed={"list_tables": ["database"]})
        self.assertTrue(check_tools_call(st, "other", {"anything": 1, "more": 2}).allow)

    def test_sealed_blocks_before_param_rules(self):
        # an extra param is rejected even if the rule'd param is fine
        d = check_tools_call(self.st, "list_tables", {"database": "main", "x": 1})
        self.assertEqual(d.reason, "mpt_param")


if __name__ == "__main__":
    unittest.main()
