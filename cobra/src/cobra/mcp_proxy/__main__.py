"""CLI entry point for MCP proxy — two modes:

Stdio proxy (original)::

    python -m cobra.mcp_proxy --server <exe> --args "<args>"

HTTP reverse proxy (for MCP servers using streamable-http transport, e.g. OSWorld-MCP)::

    python -m cobra.mcp_proxy --http-upstream http://localhost:9292/mcp --http-port 9191

The two modes are mutually exclusive; ``--server`` activates stdio, ``--http-upstream``
activates HTTP.  All other flags (``--server-id``, ``--brh-dir``, etc.) apply to both.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from cobra.mcp_proxy.alerts import DEFAULT_ALERTS_PATH
from cobra.mcp_proxy.proxy import run_proxy
from cobra.mcp_proxy.registry import DEFAULT_REGISTRY_PATH
from cobra.mcp_proxy.state import DEFAULT_BRH_DIR


def main() -> None:
    p = argparse.ArgumentParser(
        prog="cobra.mcp_proxy",
        description="BRH MCP Provenance Tracker — stdio or HTTP transparent proxy",
    )
    # --- stdio mode ---
    p.add_argument("--server", default=None, help="downstream MCP server executable (stdio mode)")
    p.add_argument("--args", default="", help="space-separated args for the server (stdio mode)")
    # --- HTTP mode ---
    p.add_argument("--http-upstream", default=None,
                   metavar="URL",
                   help="upstream MCP HTTP endpoint, e.g. http://localhost:9292/mcp")
    p.add_argument("--http-port", type=int, default=9191,
                   metavar="PORT",
                   help="local port for the HTTP proxy to listen on (default: 9191)")
    # --- shared ---
    p.add_argument("--server-id", default=None,
                   help="registry namespace; defaults to --server or --http-upstream")
    p.add_argument("--brh-dir", default=DEFAULT_BRH_DIR, help="dir holding branch_state.json")
    p.add_argument("--registry", default=DEFAULT_REGISTRY_PATH, help="tool-hash registry path")
    p.add_argument("--alerts", default=DEFAULT_ALERTS_PATH, help="alert JSONL path")
    p.add_argument("--benchmark-allow-new-tools", action="store_true",
                   help="INSECURE, opt-in only: trust-on-first-use any tool not yet "
                        "in the registry, with no human review. For unattended "
                        "benchmarks/tests only — never use this for a production "
                        "deployment. Without this flag the proxy is sealed by "
                        "default: an unregistered tool is rejected.")
    a = p.parse_args()

    if a.http_upstream and a.server:
        p.error("--server and --http-upstream are mutually exclusive")
    if not a.http_upstream and not a.server:
        p.error("one of --server (stdio mode) or --http-upstream (HTTP mode) is required")

    if a.benchmark_allow_new_tools:
        print(
            "[mcp_proxy] WARNING: --benchmark-allow-new-tools — unregistered tools "
            "are trusted on first use with no human review. Unattended "
            "benchmarks/tests only; never use this in production.",
            file=sys.stderr,
        )

    shared = dict(
        server_id=a.server_id,
        brh_dir=a.brh_dir,
        registry_path=a.registry,
        alerts_path=a.alerts,
        approve_new=a.benchmark_allow_new_tools,
    )

    if a.http_upstream:
        from cobra.mcp_proxy.proxy_http import run_proxy_http
        shared["server_id"] = shared["server_id"] or a.http_upstream
        asyncio.run(run_proxy_http(a.http_upstream, a.http_port, **shared))
    else:
        server_argv = [a.server, *(a.args.split() if a.args else [])]
        shared["server_id"] = shared["server_id"] or a.server
        asyncio.run(run_proxy(server_argv, **shared))


if __name__ == "__main__":
    main()
