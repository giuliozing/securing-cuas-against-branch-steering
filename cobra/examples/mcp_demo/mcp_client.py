"""A tiny synchronous MCP stdio client that talks *through* MCP proxy.

This is the "MCP-client wrapper" the CaMeL runtime would use to reach an MCP
server: it spawns ``python -m cobra.mcp_proxy --server python --args "<server>"`` and
exchanges newline-delimited JSON-RPC frames with it. Because MCP proxy is in the
middle, every ``tools/list``/``tools/call`` is checked before it reaches the
mock server — exactly MCP proxy's real deployment topology.

Synchronous request/response (write a frame, read one line) is enough here: the
protocol is strictly 1-response-per-request (a forwarded result, or MCP proxy's
synthetic error). The CaMeL tool functions in ``live_mcp_tools`` call
``tools_call`` so an interpreted plan's tool call becomes a real JSON-RPC frame.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


class McpProxyClient:
    def __init__(self, server_script: str, *, brh_dir: str, registry_path: str,
                 alerts_path: str, server_id: str = "mockmcp", sealed: bool = True):
        argv = [
            sys.executable, "-m", "cobra.mcp_proxy",
            "--server", sys.executable,
            "--args", server_script,
            "--server-id", server_id,
            "--brh-dir", brh_dir,
            "--registry", registry_path,
            "--alerts", alerts_path,
        ]
        if not sealed:
            # INSECURE, opt-in only: see --benchmark-allow-new-tools in
            # cobra.mcp_proxy.__main__. Never pass sealed=False in production.
            argv.append("--benchmark-allow-new-tools")
        env = dict(os.environ)
        # MCP proxy must import cobra.mcp_proxy; the demo runs with PYTHONPATH=src, inherited here.
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env
        )
        self._id = 0

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        req: dict = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            req["params"] = params
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP proxy/stdio closed unexpectedly")
        return json.loads(line)

    def tools_list(self) -> dict:
        return self._rpc("tools/list")

    def tools_call(self, name: str, arguments: dict) -> dict:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def mutate(self) -> dict:
        return self._rpc("__mutate")

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def block_reason(response: dict) -> str | None:
    """Returns the MCP proxy block reason from a synthetic error, else None (a result).

    MCP proxy errors carry ``message == "BRH MCP proxy blocked: <reason>"``."""
    err = response.get("error")
    if not err:
        return None
    message = err.get("message", "")
    prefix = "BRH MCP proxy blocked: "
    return message[len(prefix):] if message.startswith(prefix) else message
