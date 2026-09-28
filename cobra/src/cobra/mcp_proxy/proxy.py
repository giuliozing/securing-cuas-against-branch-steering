"""MCP proxy — transparent stdio JSON-RPC proxy between an MCP client and server.

Run as::

    python -m cobra.mcp_proxy --server <server-exe> --args "<server args>"

It spawns the downstream MCP server as a subprocess and pipes JSON-RPC frames
both ways: newline-delimited JSON, one object per line. Two
concurrent tasks — client->server and server->client — each parse a frame,
apply the relevant check (``router``), and forward, replace with a synthetic
error, or alert. All decision logic lives in ``router``/``check``; this module
is only the I/O shell.

Enforcer contract (shared with HTTP proxy): on a blocked ``tools/call`` we send a
JSON-RPC error to the client instead of forwarding; on a rug pull we send an
error in place of the ``tools/list`` result. A frame we cannot parse is passed
through unchanged (it is not a checked method); the per-frame ``branch_state``
read is fail-closed inside ``check``.
"""

from __future__ import annotations

import asyncio
import json
import sys

from cobra.mcp_proxy.alerts import append_alert
from cobra.mcp_proxy.registry import load_registry, save_registry
from cobra.mcp_proxy.router import route_client_frame, route_server_frame
from cobra.mcp_proxy.state import read_state


def _encode(frame: dict) -> bytes:
    return (json.dumps(frame) + "\n").encode("utf-8")


async def _stdio_streams() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    w_transport, w_proto = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
    writer = asyncio.StreamWriter(w_transport, w_proto, None, loop)
    return reader, writer


async def _client_to_server(client_r, server_w, client_w, pending, cfg) -> None:
    while True:
        line = await client_r.readline()
        if not line:
            break
        try:
            frame = json.loads(line)
        except ValueError:
            server_w.write(line)
            await server_w.drain()
            continue
        routing = route_client_frame(frame, read_state(cfg["brh_dir"]), pending,
                                     server_id=cfg["server_id"])
        for alert in routing.alerts:
            append_alert(alert["reason"], alert["detail"], cfg["alerts_path"])
        if routing.to_server is not None:
            server_w.write(_encode(routing.to_server))
            await server_w.drain()
        if routing.to_client is not None:
            client_w.write(_encode(routing.to_client))
            await client_w.drain()
    server_w.close()


async def _server_to_client(server_r, client_w, pending, registry_box, cfg) -> None:
    while True:
        line = await server_r.readline()
        if not line:
            break
        try:
            frame = json.loads(line)
        except ValueError:
            client_w.write(line)
            await client_w.drain()
            continue
        routing, new_reg, changed = route_server_frame(
            frame, cfg["server_id"], registry_box["reg"], pending, approve_new=cfg["approve_new"]
        )
        for alert in routing.alerts:
            append_alert(alert["reason"], alert["detail"], cfg["alerts_path"])
        if changed:
            registry_box["reg"] = new_reg
            save_registry(new_reg, cfg["registry_path"])
        if routing.to_client is not None:
            client_w.write(_encode(routing.to_client))
            await client_w.drain()


async def run_proxy(server_argv, *, server_id, brh_dir, registry_path, alerts_path, approve_new) -> None:
    proc = await asyncio.create_subprocess_exec(
        *server_argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=None
    )
    client_r, client_w = await _stdio_streams()
    pending: dict = {}
    registry_box = {"reg": load_registry(registry_path)}
    cfg = {
        "brh_dir": brh_dir,
        "registry_path": registry_path,
        "alerts_path": alerts_path,
        "server_id": server_id,
        "approve_new": approve_new,
    }
    c2s = asyncio.create_task(_client_to_server(client_r, proc.stdin, client_w, pending, cfg))
    s2c = asyncio.create_task(_server_to_client(proc.stdout, client_w, pending, registry_box, cfg))
    await asyncio.wait({c2s, s2c}, return_when=asyncio.FIRST_COMPLETED)
    if proc.returncode is None:
        proc.terminate()
