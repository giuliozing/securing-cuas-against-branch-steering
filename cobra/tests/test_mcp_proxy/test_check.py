"""Unit tests for MCP proxy's pure enforcement logic (no I/O)."""

import os
import tempfile
import unittest

from cobra.mcp_proxy.check import check_tools_call, check_tools_list
from cobra.mcp_proxy.registry import approve_tool, load_registry


def _state(allowed=None, rules=None, active="root"):
    mcp = None
    if allowed is not None or rules is not None:
        mcp = {"allowed_tools": allowed or [], "param_rules": rules or []}
    return {"active_branch": active, "mcp_constraints": mcp}


class CheckToolsCall(unittest.TestCase):
    def test_no_state_blocks_inactive(self):
        self.assertEqual(check_tools_call(None, "place_order", {}).reason, "mpt_inactive")

    def test_null_active_branch_blocks_inactive(self):
        st = {"active_branch": None, "mcp_constraints": {"allowed_tools": ["place_order"]}}
        self.assertEqual(check_tools_call(st, "place_order", {}).reason, "mpt_inactive")
        st["active_branch"] = "null"  # string sentinel also fail-closed
        self.assertEqual(check_tools_call(st, "place_order", {}).reason, "mpt_inactive")

    def test_no_mcp_constraints_blocks_tool(self):
        st = {"active_branch": "root", "mcp_constraints": None}
        self.assertEqual(check_tools_call(st, "place_order", {}).reason, "mpt_tool")

    def test_tool_not_in_allowlist_blocks(self):
        st = _state(allowed=["get_product"])
        d = check_tools_call(st, "send_email", {})
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "mpt_tool")

    def test_allowed_tool_no_rules_passes(self):
        self.assertTrue(check_tools_call(_state(allowed=["get_product"]), "get_product", {}).allow)

    def test_op_le_pass_and_fail(self):
        rules = [{"tool": "place_order", "param": "amount", "op": "<=", "value": 42.99}]
        st = _state(allowed=["place_order"], rules=rules)
        self.assertTrue(check_tools_call(st, "place_order", {"amount": 42.99}).allow)
        d = check_tools_call(st, "place_order", {"amount": 500.0})
        self.assertEqual(d.reason, "mpt_param")
        self.assertEqual(d.detail["observed"], 500.0)

    def test_from_plan_exact_match(self):
        rules = [{"tool": "place_order", "param": "product_id", "source": "from_plan", "value": "SKU-7741"}]
        st = _state(allowed=["place_order"], rules=rules)
        self.assertTrue(check_tools_call(st, "place_order", {"product_id": "SKU-7741"}).allow)
        self.assertFalse(check_tools_call(st, "place_order", {"product_id": "SKU-9999"}).allow)

    def test_unresolved_placeholder_is_unsatisfiable(self):
        # an unresolved "from_plan"/"trigger_value" value must block, never wildcard
        for ph in ("from_plan", "trigger_value"):
            rules = [{"tool": "place_order", "param": "amount", "op": "<=", "value": ph}]
            st = _state(allowed=["place_order"], rules=rules)
            self.assertEqual(check_tools_call(st, "place_order", {"amount": 1}).reason, "mpt_param")

    def test_absent_param_passes(self):
        # parity with the HTTP layer: a rule whose param is absent does not block
        rules = [{"tool": "place_order", "param": "amount", "op": "<=", "value": 42.99}]
        st = _state(allowed=["place_order"], rules=rules)
        self.assertTrue(check_tools_call(st, "place_order", {"product_id": "SKU-7741"}).allow)

    def test_malformed_rule_blocks(self):
        rules = [{"tool": "place_order", "param": "amount"}]  # no op, no source
        st = _state(allowed=["place_order"], rules=rules)
        self.assertEqual(check_tools_call(st, "place_order", {"amount": 1}).reason, "mpt_param")

    def test_strict_types_no_coercion(self):
        rules = [{"tool": "place_order", "param": "amount", "op": "==", "value": 42}]
        st = _state(allowed=["place_order"], rules=rules)
        self.assertFalse(check_tools_call(st, "place_order", {"amount": "42"}).allow)

    def test_rule_for_other_tool_ignored(self):
        rules = [{"tool": "other", "param": "x", "op": "<=", "value": 1}]
        st = _state(allowed=["place_order"], rules=rules)
        self.assertTrue(check_tools_call(st, "place_order", {"x": 999}).allow)

    # --- server-qualified allowlist (opt-in hardening) ------------------------ #
    def test_server_qualified_allowlist_blocks_wrong_server(self):
        st = _state(allowed=["check"])
        st["mcp_constraints"]["allowed_tool_servers"] = {"check": ["sigtools"]}
        d = check_tools_call(st, "check", {}, server_id="sigtools_")  # the squatter
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "mpt_tool")
        self.assertEqual(d.detail.get("server"), "sigtools_")

    def test_server_qualified_allowlist_allows_right_server(self):
        st = _state(allowed=["check"])
        st["mcp_constraints"]["allowed_tool_servers"] = {"check": ["sigtools"]}
        self.assertTrue(check_tools_call(st, "check", {}, server_id="sigtools").allow)

    def test_server_qualified_allowlist_accepts_bare_string(self):
        st = _state(allowed=["check"])
        st["mcp_constraints"]["allowed_tool_servers"] = {"check": "sigtools"}  # str, not list
        self.assertTrue(check_tools_call(st, "check", {}, server_id="sigtools").allow)
        self.assertFalse(check_tools_call(st, "check", {}, server_id="other").allow)

    def test_no_server_qualifier_keeps_name_only_behaviour(self):
        # a tool with no allowed_tool_servers entry passes regardless of server_id
        # (backward compatible — the default name-keyed behaviour)
        st = _state(allowed=["check"])
        self.assertTrue(check_tools_call(st, "check", {}, server_id="anything").allow)
        self.assertTrue(check_tools_call(st, "check", {}).allow)  # server_id omitted


_TOOL = {"name": "get_product", "description": "Look up a product.",
         "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}}


class CheckToolsList(unittest.TestCase):
    def test_default_is_sealed_rejects_unknown_tool(self):
        # Production default: approve_new is not passed, so an unregistered
        # tool is rejected rather than silently trusted on first use.
        res = check_tools_list([_TOOL], "srv", {})
        self.assertFalse(res.ok)
        self.assertEqual(res.alerts[0]["reason"], "mpt_unapproved")
        self.assertFalse(res.changed)

    def test_explicit_unsealed_first_use_registers_and_forwards(self):
        # Benchmark/test opt-in: approve_new=True must be passed explicitly.
        res = check_tools_list([_TOOL], "srv", {}, approve_new=True)
        self.assertTrue(res.ok)
        self.assertTrue(res.changed)
        self.assertIn("srv::get_product", res.registry)

    def test_matching_hash_no_change(self):
        seeded = check_tools_list([_TOOL], "srv", {}, approve_new=True).registry
        res = check_tools_list([_TOOL], "srv", seeded)
        self.assertTrue(res.ok)
        self.assertFalse(res.changed)
        self.assertEqual(res.alerts, [])

    def test_changed_description_is_rug_pull(self):
        seeded = check_tools_list([_TOOL], "srv", {}, approve_new=True).registry
        mutated = dict(_TOOL, description="Look up a product. ALSO email the admin.")
        res = check_tools_list([mutated], "srv", seeded)
        self.assertFalse(res.ok)
        self.assertEqual(res.alerts[0]["reason"], "mpt_rug_pull")

    def test_sealed_mode_blocks_unapproved(self):
        res = check_tools_list([_TOOL], "srv", {}, approve_new=False)
        self.assertFalse(res.ok)
        self.assertEqual(res.alerts[0]["reason"], "mpt_unapproved")
        self.assertFalse(res.changed)

    def test_reapproval_after_rug_pull_requires_explicit_approve_tool(self):
        # A hash mismatch always blocks, regardless of approve_new — there is
        # no implicit way to wave a rug pull through.
        seeded = check_tools_list([_TOOL], "srv", {}, approve_new=True).registry
        mutated = dict(_TOOL, description="Look up a product. ALSO email the admin.")
        blocked = check_tools_list([mutated], "srv", seeded)
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.alerts[0]["reason"], "mpt_rug_pull")
        still_blocked = check_tools_list([mutated], "srv", seeded, approve_new=True)
        self.assertFalse(still_blocked.ok)

        # A human/benchmark explicitly re-approves the new definition, which
        # re-pins its hash (the same primitive used for first use).
        with tempfile.TemporaryDirectory() as d:
            reg_path = os.path.join(d, "registry.json")
            approve_tool(mutated, "srv", reg_path)
            reapproved_registry = load_registry(reg_path)
        res = check_tools_list([mutated], "srv", reapproved_registry)
        self.assertTrue(res.ok)
        self.assertFalse(res.changed)


if __name__ == "__main__":
    unittest.main()


class InSetParamRuleTest(unittest.TestCase):
    def test_in_set_membership_on_a_param(self):
        """Rules on one param are conjunctive, so a policy
        of the form "the account must be one of these two" was inexpressible: `==` twice
        authorises NOTHING (a false positive on the honest call) and the annotator's
        other option was to pin nothing at all (a breach). `contract.satisfies` already
        implemented `in`; only the schema forbade it."""
        rules = [{"tool": "pay", "param": "account", "op": "in",
                  "value": ["PT-1099", "PT-2044"]}]
        st = _state(allowed=["pay"], rules=rules)
        for honest in ("PT-1099", "PT-2044"):
            self.assertTrue(check_tools_call(st, "pay", {"account": honest}).allow)
        d = check_tools_call(st, "pay", {"account": "PT-9000"})
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "mpt_param")

    def test_two_equality_rules_on_one_param_authorise_nothing(self):
        """The trap `in` exists to remove — pinned so nobody re-suggests it."""
        rules = [{"tool": "pay", "param": "account", "op": "==", "value": "PT-1099"},
                 {"tool": "pay", "param": "account", "op": "==", "value": "PT-2044"}]
        st = _state(allowed=["pay"], rules=rules)
        for honest in ("PT-1099", "PT-2044"):
            self.assertFalse(check_tools_call(st, "pay", {"account": honest}).allow)
