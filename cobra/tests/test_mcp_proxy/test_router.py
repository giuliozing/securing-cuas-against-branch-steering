"""Unit tests for MCP proxy frame routing (pure, no pipes)."""

import unittest

from cobra.mcp_proxy.check import check_tools_list
from cobra.mcp_proxy.router import route_client_frame, route_server_frame

_TOOL = {"name": "get_product", "description": "Look up.",
         "inputSchema": {"type": "object"}}


def _state(allowed=None, rules=None):
    return {"active_branch": "root",
            "mcp_constraints": {"allowed_tools": allowed or [], "param_rules": rules or []}}


class RouteClientFrame(unittest.TestCase):
    def test_allowed_call_forwarded_to_server(self):
        frame = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "get_product", "arguments": {}}}
        r = route_client_frame(frame, _state(allowed=["get_product"]), {})
        self.assertIs(r.to_server, frame)
        self.assertIsNone(r.to_client)
        self.assertEqual(r.alerts, [])

    def test_blocked_call_returns_error_and_alert(self):
        frame = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                 "params": {"name": "send_email", "arguments": {}}}
        r = route_client_frame(frame, _state(allowed=["get_product"]), {})
        self.assertIsNone(r.to_server)
        self.assertEqual(r.to_client["id"], 7)
        self.assertIn("error", r.to_client)
        self.assertEqual(r.alerts[0]["reason"], "mpt_tool")

    def test_non_call_request_recorded_in_pending(self):
        pending = {}
        frame = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        r = route_client_frame(frame, None, pending)
        self.assertIs(r.to_server, frame)
        self.assertEqual(pending, {3: "tools/list"})

    def test_server_id_threaded_to_server_qualified_check(self):
        # the call's server_id reaches check_tools_call: a same-named tool from an
        # unapproved server is blocked under the opt-in server-qualified allowlist.
        st = _state(allowed=["check"])
        st["mcp_constraints"]["allowed_tool_servers"] = {"check": ["sigtools"]}
        frame = {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                 "params": {"name": "check", "arguments": {}}}
        blocked = route_client_frame(frame, st, {}, server_id="sigtools_")
        self.assertIsNone(blocked.to_server)
        self.assertEqual(blocked.alerts[0]["reason"], "mpt_tool")
        allowed = route_client_frame(frame, st, {}, server_id="sigtools")
        self.assertIs(allowed.to_server, frame)


class RouteServerFrame(unittest.TestCase):
    def test_default_is_sealed_unknown_tool_blocked(self):
        # Production default: approve_new is not passed, so an unregistered
        # tool from a fresh tools/list is rejected, not silently trusted.
        pending = {3: "tools/list"}
        frame = {"jsonrpc": "2.0", "id": 3, "result": {"tools": [_TOOL]}}
        r, reg, changed = route_server_frame(frame, "srv", {}, pending)
        self.assertIn("error", r.to_client)
        self.assertEqual(r.alerts[0]["reason"], "mpt_unapproved")
        self.assertFalse(changed)
        self.assertNotIn("srv::get_product", reg)

    def test_explicit_unsealed_first_use_forwarded_registry_changed(self):
        # Benchmark/test opt-in: approve_new=True must be passed explicitly.
        pending = {3: "tools/list"}
        frame = {"jsonrpc": "2.0", "id": 3, "result": {"tools": [_TOOL]}}
        r, reg, changed = route_server_frame(frame, "srv", {}, pending, approve_new=True)
        self.assertIs(r.to_client, frame)
        self.assertTrue(changed)
        self.assertIn("srv::get_product", reg)
        self.assertEqual(pending, {})  # consumed

    def test_tools_list_rug_pull_replaced_with_error(self):
        seeded = check_tools_list([_TOOL], "srv", {}, approve_new=True).registry
        pending = {3: "tools/list"}
        mutated = dict(_TOOL, description="Look up. EVIL")
        frame = {"jsonrpc": "2.0", "id": 3, "result": {"tools": [mutated]}}
        r, _reg, _changed = route_server_frame(frame, "srv", seeded, pending)
        self.assertIn("error", r.to_client)
        self.assertEqual(r.alerts[0]["reason"], "mpt_rug_pull")
        # the surfaced client message reason mirrors the alert (not hardcoded)
        self.assertEqual(r.to_client["error"]["message"], "BRH MCP proxy blocked: mpt_rug_pull")

    def test_tools_list_sealed_unapproved_surfaces_its_own_reason(self):
        # a sealed-mode block is mpt_unapproved, NOT mpt_rug_pull — the client
        # message must reflect that (regression: the reason was hardcoded).
        pending = {3: "tools/list"}
        frame = {"jsonrpc": "2.0", "id": 3, "result": {"tools": [_TOOL]}}
        r, _reg, _changed = route_server_frame(frame, "srv", {}, pending, approve_new=False)
        self.assertIn("error", r.to_client)
        self.assertEqual(r.alerts[0]["reason"], "mpt_unapproved")
        self.assertEqual(r.to_client["error"]["message"], "BRH MCP proxy blocked: mpt_unapproved")

    def test_unrelated_response_forwarded_untouched(self):
        frame = {"jsonrpc": "2.0", "id": 9, "result": {"ok": True}}
        r, reg, changed = route_server_frame(frame, "srv", {"k": 1}, {})
        self.assertIs(r.to_client, frame)
        self.assertFalse(changed)
        self.assertEqual(reg, {"k": 1})


if __name__ == "__main__":
    unittest.main()
