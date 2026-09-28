"""Build the BRH annotation manifest from MCP tool definitions.

The BRH annotator takes an approved-tool manifest — ``{tool name: [param
names]}`` — so it can constrain only real MCP tools and only their real params
(see `cobra.brh.validator.McpManifest`). That manifest is *derived* from the
tool definitions a `tools/list` returns (the same defs MCP proxy hash-pins): the tool
name plus the parameter names declared in its ``inputSchema.properties``. This
is the bridge between the MCP side (tool defs) and the annotation side, so the
P-LLM never needs the tool *descriptions* — only names and param names.
"""

from __future__ import annotations

import json
import subprocess
from typing import Iterable


def server_map_for_tools(tools: Iterable[dict], server_id: str) -> dict[str, str]:
    """``[{name, ...}]`` + server_id -> ``{name: server_id}`` for all named tools.

    Convenience helper: build the server_map arg for ``generate_plan_constraints``
    when all tools in a tools/list response come from the same approved server."""
    return {tool["name"]: server_id for tool in (tools or []) if tool.get("name")}


def direct_tools_list(server_argv: list[str]) -> list[dict]:
    """One-shot subprocess: spawn server, send tools/list, return tools, kill.

    Used by the approval-loop orchestrator to enumerate available tools before
    starting the full MCP proxy session. The server is killed after the first
    response regardless of how it exits."""
    proc = subprocess.Popen(
        server_argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n"
    try:
        stdout, _ = proc.communicate(input=request, timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return []
    finally:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        first_line = stdout.strip().split("\n")[0]
        response = json.loads(first_line)
        return response.get("result", {}).get("tools") or []
    except (ValueError, IndexError):
        return []


def manifest_from_tools(tools: Iterable[dict]) -> dict[str, list[str]]:
    """``[{name, inputSchema, ...}]`` -> ``{name: [param names]}``.

    Tools without a name are skipped; a tool with no ``inputSchema.properties``
    maps to an empty param list (it can be authorised but pins no params)."""
    manifest: dict[str, list[str]] = {}
    for tool in tools or []:
        name = tool.get("name")
        if not name:
            continue
        schema = tool.get("inputSchema") or {}
        properties = schema.get("properties") or {}
        manifest[name] = list(properties.keys())
    return manifest
