

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.tools.tool import Tool

HERE = Path(__file__).resolve().parent

from steerbench.tasks import registry as reg
from steerbench.mcp import toolkit
from steerbench.mcp import distractors
from steerbench.mcp import s3_tools
from steerbench.mcp import s4_tools
from steerbench.mcp import s7_tools

MCP_PORT = 9393


def init_server(name: str = "STEER-Bench-Web", *, rugged: bool = False,
                only_themes: set[str] | None = None) -> FastMCP:
    """`rugged` serves suite S3's post-approval descriptions (the M5 rug pull);
    `only_themes` restricts the surface, used by S3/M6 to stand up a *second* server
    that squats a same-named tool in its own registry namespace."""
    mcp = FastMCP(name)
    seen: set[str] = set()
    for mcp_name, fn, desc in s3_tools.build_all(reg.TASKS, rugged=rugged):
        theme = mcp_name.split(".", 1)[0]
        if only_themes is not None and theme not in only_themes:
            continue
        if mcp_name not in seen:
            seen.add(mcp_name)
            mcp.add_tool(Tool.from_function(fn, name=mcp_name, description=desc))
    # Suite S4's seam channel: server-side action tools that record DIRECT, so the
    # MCP path is enforced by MCP proxy rather than by HTTP proxy (tasks/s4_seam.py). Added
    # before the toolkit's proxy-routed verbs; the names never collide.
    for mcp_name, fn, desc in s4_tools.build_all(reg.TASKS):
        theme = mcp_name.split(".", 1)[0]
        if only_themes is not None and theme not in only_themes:
            continue
        if mcp_name not in seen:
            seen.add(mcp_name)
            mcp.add_tool(Tool.from_function(fn, name=mcp_name, description=desc))
    # Suite S7's MCP insertion task: one tool per step of the procedure, plus the
    # off-plan one. The last is registered exactly like the others on purpose — a tool
    # the server does not publish would be refused by FastMCP before MCP proxy saw it, and the
    # cell would credit schema validation with a block the enforcer never made.
    for mcp_name, fn, desc in s7_tools.build_all(reg.TASKS):
        theme = mcp_name.split(".", 1)[0]
        if only_themes is not None and theme not in only_themes:
            continue
        if mcp_name not in seen:
            seen.add(mcp_name)
            mcp.add_tool(Tool.from_function(fn, name=mcp_name, description=desc))
    if only_themes is not None:
        return mcp
    for task in reg.TASKS:
        if task.on_mcp_axis:
            continue  # S3 themes are served by s3_tools above
        for mcp_name, fn, desc in toolkit.build_for_task(task):
            if mcp_name in seen:
                continue
            seen.add(mcp_name)
            mcp.add_tool(Tool.from_function(fn, name=mcp_name, description=desc))
    for name_, fn, desc in distractors.build():
        if name_ not in seen:
            seen.add(name_)
            mcp.add_tool(Tool.from_function(fn, name=name_, description=desc))
    return mcp


def main() -> None:
    ap = argparse.ArgumentParser(description="STEER-Bench MCP server")
    ap.add_argument("--debug", action="store_true", help="stdio transport (MCP inspector)")
    ap.add_argument("--port", type=int, default=MCP_PORT)
    ap.add_argument("--rugged", action="store_true",
                    help="S3/M5: serve the post-approval tool descriptions (rug pull)")
    ap.add_argument("--only-themes", default="",
                    help="S3/M6: comma themes — stand up a squatter server exposing "
                         "only these (same tool names, different registry namespace)")
    args = ap.parse_args()

    themes = {t.strip() for t in args.only_themes.split(",") if t.strip()} or None
    mcp = init_server(rugged=args.rugged, only_themes=themes)
    if args.debug:
        mcp.run()
    else:
        mcp.run(transport="http", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
