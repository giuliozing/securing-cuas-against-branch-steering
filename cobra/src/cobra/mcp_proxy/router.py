"""Frame routing for MCP proxy — pure decisions over already-parsed JSON-RPC frames.

Separated from the asyncio I/O (``proxy.py``) so the routing logic — which
direction a frame goes, whether it is replaced by a synthetic error, what alert
it raises — is unit-testable without real pipes. ``proxy.py`` only does the
stdio reading/writing, the per-frame ``branch_state.json`` read, and registry
persistence; everything decision-shaped lives here.

Correlation: a ``tools/list`` arrives as a request (client->server) and its tool
definitions arrive later in the matching response (server->client). We remember
the request id -> method in ``pending`` on the way out, and look it up on the
response to know it is a ``tools/list`` result worth hashing.
"""

from __future__ import annotations

import dataclasses

from cobra.mcp_proxy.check import check_tools_call, check_tools_list


@dataclasses.dataclass
class Routing:
    """Where a frame goes. ``to_server`` forwards downstream; ``to_client`` is
    either a forwarded response or a synthetic error sent back upstream."""

    to_server: dict | None = None
    to_client: dict | None = None
    alerts: list = dataclasses.field(default_factory=list)


def jsonrpc_error(req_id, reason: str, detail: dict) -> dict:
    """A JSON-RPC error the client receives in place of the blocked call/result.

    The message tells the agent it was blocked by policy (so it terminates the
    step rather than blindly retrying) and carries the reason/detail for logs."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32000, "message": f"BRH MCP proxy blocked: {reason}", "data": detail},
    }


def route_client_frame(frame: dict, state: dict | None, pending: dict,
                       *, server_id: str | None = None) -> Routing:
    """Routes a client->server frame. ``tools/call`` is checked against `state`;
    other requests are recorded in `pending` for response correlation. ``server_id``
    is threaded to the check for the opt-in server-qualified allowlist."""
    method = frame.get("method")
    if method == "tools/call" and "id" in frame:
        params = frame.get("params") or {}
        decision = check_tools_call(state, params.get("name"), params.get("arguments") or {},
                                    server_id=server_id)
        if decision.allow:
            return Routing(to_server=frame)
        return Routing(
            to_client=jsonrpc_error(frame.get("id"), decision.reason, decision.detail),
            alerts=[{"reason": decision.reason, "detail": decision.detail}],
        )
    if method and "id" in frame:
        pending[frame["id"]] = method
    return Routing(to_server=frame)


def route_server_frame(frame: dict, server_id: str, registry: dict, pending: dict,
                       *, approve_new: bool = False):
    """Routes a server->client frame. A response to a recorded ``tools/list`` is
    hashed; anything else is forwarded untouched.

    Returns ``(Routing, registry, changed)`` — the registry may grow on
    first-use; the proxy persists it when ``changed``.
    """
    rid = frame.get("id")
    method = pending.pop(rid, None) if rid is not None else None
    result = frame.get("result")
    if method == "tools/list" and isinstance(result, dict):
        res = check_tools_list(result.get("tools") or [], server_id, registry, approve_new=approve_new)
        if res.ok:
            return Routing(to_client=frame), res.registry, res.changed
        # The surfaced reason is the first anomaly's: a hash mismatch is
        # ``mpt_rug_pull``, an unregistered tool in sealed mode ``mpt_unapproved``.
        # (Both block; the full per-tool list rides in ``detail.alerts``.)
        reason = res.alerts[0].get("reason", "mpt_tools_list") if res.alerts else "mpt_tools_list"
        return (
            Routing(to_client=jsonrpc_error(rid, reason, {"alerts": res.alerts}), alerts=res.alerts),
            res.registry,
            res.changed,
        )
    return Routing(to_client=frame), registry, False
