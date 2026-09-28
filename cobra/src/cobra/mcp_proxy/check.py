"""Pure enforcement logic for MCP proxy: the tool-hash rug-pull check and the
per-call `tools/call` argument check.

Kept free of any I/O (no stdio, no disk) so it is unit-testable with the system
Python (it imports ``cobra.brh.contract``, which needs only pydantic, like the
``test_brh`` suite). The asyncio proxy (``proxy.py``) is the thin I/O shell that
feeds parsed frames into these functions.

``tools/call`` param-rule semantics import ``cobra.brh.contract.satisfies``
directly — MCP proxy lives in this repo, so unlike the HTTP proxy HTTP enforcer (which
keeps a stdlib copy and is pinned to the reference via the golden vectors) it
shares the *exact* reference implementation: strict types, no coercion, bool is
not a number, unresolved placeholders unsatisfiable, ordering ops numeric-only.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from cobra.mcp_proxy.registry import registry_key, tool_hash
from cobra.brh.contract import satisfies

_MISSING = object()


@dataclasses.dataclass(frozen=True)
class Decision:
    """Result of a ``tools/call`` check. ``allow`` gates forwarding; on a block
    ``reason``/``detail`` go into the alert and the synthetic JSON-RPC error."""

    allow: bool
    reason: str | None = None
    detail: dict = dataclasses.field(default_factory=dict)


def check_tools_call(state: dict | None, tool: str, args: dict | None,
                     *, server_id: str | None = None) -> Decision:
    """Checks a ``tools/call`` against the active branch's MCP constraints.

    Fail-closed throughout: no active branch / no MCP constraints / tool not in
    the allowlist all block. Param rules are conjunctive; an absent param passes
    (the allowlist is the gate — parity with the HTTP layer's absent-field rule,
    same declared residual: param renaming). An unresolved placeholder ``value``
    is unsatisfiable via ``satisfies`` (never a wildcard), so it blocks.

    ``server_id`` is the calling proxy's registry namespace; it is only consulted
    by the opt-in server-qualified allowlist (``allowed_tool_servers``).
    """
    args = args or {}
    if not state or state.get("active_branch") in (None, "null"):
        return Decision(False, "mpt_inactive", {"tool": tool})
    mcp = state.get("mcp_constraints")
    if not mcp:
        return Decision(False, "mpt_tool", {"tool": tool, "why": "no MCP constraints on the active branch"})
    allowed = mcp.get("allowed_tools") or []
    if tool not in allowed:
        return Decision(False, "mpt_tool", {"tool": tool, "allowed": list(allowed)})
    # Server-qualified allowlist: if the plan
    # pinned which server(s) may serve this tool, a call arriving from a different
    # server_id is blocked even under trust-on-first-use — closing the call-time gap
    # where a same-named squatter passes the name-keyed allowlist. A tool with no
    # entry keeps the default (name-only) behaviour, so existing plans are untouched.
    tool_servers = (mcp.get("allowed_tool_servers") or {}).get(tool)
    if tool_servers is not None:
        allowed_servers = [tool_servers] if isinstance(tool_servers, str) else list(tool_servers)
        if server_id not in allowed_servers:
            return Decision(False, "mpt_tool", {
                "tool": tool, "server": server_id, "allowed_servers": allowed_servers,
                "why": "tool from unapproved server (server-qualified allowlist)",
            })
    # Schema-closed param mode: if the plan
    # sealed this tool's parameter set, an argument whose name is not in that set
    # is blocked. Opt-in per tool — a tool absent from allowed_params keeps the
    # default open behaviour (extra params pass), so existing plans are untouched.
    sealed = (mcp.get("allowed_params") or {}).get(tool)
    if sealed is not None:
        sealed_set = set(sealed)
        for name in args:
            if name not in sealed_set:
                return Decision(False, "mpt_param", {
                    "tool": tool, "param": name, "why": "param not in sealed allowed_params",
                    "allowed_params": list(sealed),
                })
    for rule in mcp.get("param_rules") or []:
        if rule.get("tool") != tool:
            continue
        param = rule.get("param")
        observed = args.get(param, _MISSING)
        if observed is _MISSING:
            continue
        op = rule.get("op")
        if op is None:
            # A "from_plan" rule with no op is an exact-value pin (==); a rule
            # with neither op nor source is malformed -> fail-closed.
            if rule.get("source") == "from_plan":
                op = "=="
            else:
                return Decision(False, "mpt_param", {"tool": tool, "param": param, "why": "malformed rule (no op/source)"})
        if not satisfies(op, rule.get("value"), observed):
            return Decision(False, "mpt_param", {
                "tool": tool, "param": param, "op": op,
                "value": rule.get("value"), "observed": observed,
            })
    return Decision(True)


@dataclasses.dataclass(frozen=True)
class ToolListResult:
    """Result of a ``tools/list`` check. ``ok`` gates forwarding the response;
    ``registry``/``changed`` carry the (possibly updated) hash registry."""

    ok: bool
    alerts: list
    registry: dict
    changed: bool


def check_tools_list(tools, server_id: str, registry: dict, *, approve_new: bool = False) -> ToolListResult:
    """Compares each tool definition's hash to the registry.

    Three cases per tool: unknown -> in sealed mode (``approve_new=False``,
    the default) an ``mpt_unapproved`` alert; explicitly unsealed
    (``approve_new=True``, opt-in for benchmarks/tests) -> trust-on-first-use
    register; hash matches -> no change; hash differs -> ``mpt_rug_pull``. The
    whole response is blocked (``ok=False``) if *any* tool is anomalous.
    """
    reg = dict(registry)
    changed = False
    alerts: list = []
    ok = True
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for tool in tools or []:
        name = tool.get("name")
        key = registry_key(server_id, name)
        h = tool_hash(tool)
        rec = reg.get(key)
        if rec is None:
            if approve_new:
                reg[key] = {"hash": h, "ts": ts, "approved": True}
                changed = True
            else:
                ok = False
                alerts.append({"reason": "mpt_unapproved",
                               "detail": {"tool": name, "server": server_id, "observed": h}})
        elif rec.get("hash") == h:
            continue
        else:
            ok = False
            alerts.append({"reason": "mpt_rug_pull",
                           "detail": {"tool": name, "server": server_id,
                                      "expected": rec.get("hash"), "observed": h}})
    return ToolListResult(ok, alerts, reg, changed)
