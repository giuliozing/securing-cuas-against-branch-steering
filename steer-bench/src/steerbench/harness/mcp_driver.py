"""The guarded MCP action, issued through the MCP proxy — suite S3's analogue of
`driver.submit_action`.

Topology:

    caller -> MCP proxy (:9394) -> FastMCP server (:9393) -> POST direct to the site

MCP proxy is therefore the **only** enforcer on this path: the tool records its effect by
POSTing direct to the site, unproxied, so HTTP proxy never sees it. A blocked call
never reaches the server, so nothing is recorded — which is what makes the verdict
executor-independent and identical in shape to the HTTP suites (`/__state` says
whether the action happened; `brh_alerts.jsonl` says why not).

Used LLM-free by `oracles/s3.py`, and by the paid structural run as the tool the
plan calls.
"""

from __future__ import annotations

import asyncio


#: How much of a tool result travels back to the caller. Was a flat `[:200]` on the
#: `repr` of the whole `CallToolResult`, which is fine for suite S3 — its tools answer
#: with a one-line "OK … recorded" — and silently wrong for a tool that RETURNS A
#: DOCUMENT (suite S7's `read_tool`), where the document is the payload the cell is
#: about. A truncation that changes what the agent reads is not a display setting.
RESULT_LIMIT = 8000


def _text(res) -> str:
    """The tool's own text, when it has one; otherwise its repr.

    Unwrapped rather than repr'd because the repr is `CallToolResult(content=[TextContent(
    type='text', text='…')])` — the payload wrapped in three layers of machinery that a
    plan then has to hand to the quarantined model verbatim. Nothing downstream reads
    this string (the verdict comes from `blocked`, the alerts and `/__state`), so the
    change cannot move an S3/S4 number; what it changes is what a PLAN can do with the
    answer, which is the whole point for S7's document-reading tool."""
    parts = []
    for item in (getattr(res, "content", None) or []):
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
    out = "\n".join(parts) if parts else str(res)
    return out[:RESULT_LIMIT]


def call_tool(tool: str, arguments: dict, mpt_url: str, timeout: float = 30.0) -> dict:
    """Invoke one MCP tool through MCP proxy. Returns the enforcer-visible outcome.

    A block surfaces as an exception from the client (MCP proxy answers with a JSON-RPC
    error carrying the `mpt_*` reason), so it is reported rather than raised: the
    caller's job is to observe what the enforcer did, not to handle a failure."""
    async def _run() -> dict:
        from fastmcp import Client
        async with Client(mpt_url) as c:
            res = await c.call_tool(tool, arguments or {})
            return {"tool": tool, "blocked": False, "result": _text(res)}

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout))
    except Exception as e:  # noqa: BLE001 - a block is an expected outcome here
        return {"tool": tool, "blocked": True, "error": str(e)[:200]}


def list_tools(mpt_url: str, timeout: float = 30.0) -> dict:
    """Drive a `tools/list` through MCP proxy — the S3/M5 (rug pull) and M6 (squatting)
    surface. The hash check happens on the RESPONSE, so a rug pull blocks the whole
    listing before any description reaches a consumer."""
    async def _run() -> dict:
        from fastmcp import Client
        async with Client(mpt_url) as c:
            tools = await c.list_tools()
            return {"blocked": False, "tools": [t.name for t in tools]}

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout))
    except Exception as e:  # noqa: BLE001
        return {"blocked": True, "error": str(e)[:200]}
