"""MCP proxy — MCP Provenance Tracker.

A transparent stdio JSON-RPC proxy between an MCP client and server that
(1) pins tool-definition hashes on ``tools/list`` to catch rug pulls and
(2) checks every ``tools/call`` against the active plan branch in
``branch_state.json`` (allowed tools + param rules, via the shared BRH contract).

Pure logic (``check``/``router``/``registry``/``state``/``alerts``) is importable
without the asyncio I/O shell (``proxy``), so it unit-tests on the system Python.
"""

from cobra.mcp_proxy.check import Decision, ToolListResult, check_tools_call, check_tools_list
from cobra.mcp_proxy.manifest import manifest_from_tools
from cobra.mcp_proxy.registry import canonical_tool_bytes, registry_key, tool_hash
from cobra.mcp_proxy.router import Routing, jsonrpc_error, route_client_frame, route_server_frame

__all__ = [
    "Decision",
    "ToolListResult",
    "check_tools_call",
    "check_tools_list",
    "manifest_from_tools",
    "canonical_tool_bytes",
    "registry_key",
    "tool_hash",
    "Routing",
    "jsonrpc_error",
    "route_client_frame",
    "route_server_frame",
]
