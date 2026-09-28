"""MCP proxy — transparent HTTP reverse proxy for MCP servers using streamable-http transport.

Intercepts every POST to ``/mcp`` between an MCP client and an HTTP-transport MCP
server. The same enforcement logic
as the stdio proxy (``proxy.py``) applies — only the I/O shell differs:

* ``tools/call`` request  → ``route_client_frame`` → block or forward upstream.
* ``tools/list`` response → ``route_server_frame`` → hash-check, block rug pull.
* Unparseable body / batch / SSE response → transparent pass-through.
* All decision logic lives in ``router``/``check``; this module is only the HTTP shell.

Usage::

    python -m cobra.mcp_proxy --http-upstream http://localhost:9292/mcp --http-port 9191

Then point the MCP client at ``http://localhost:9191/mcp`` instead of ``:9292``.
"""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
from aiohttp import web

from cobra.mcp_proxy.alerts import append_alert
from cobra.mcp_proxy.registry import load_registry, save_registry
from cobra.mcp_proxy.router import route_client_frame, route_server_frame
from cobra.mcp_proxy.state import read_state

log = logging.getLogger(__name__)

_SKIP_HEADERS = frozenset({"host", "content-length", "transfer-encoding", "connection"})


def _jsonrpc_from_sse(raw: bytes) -> dict | None:
    """Extract the first JSON-RPC response/error from an SSE body.

    FastMCP sends ``data: <json>\\n\\n`` events; we extract the first
    ``{"result": ...}`` or ``{"error": ...}`` frame. Returns None if no
    JSON-RPC frame is found (e.g. pure notification stream).
    """
    for line in raw.decode(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and ("result" in obj or "error" in obj):
            return obj
    return None

# MCP streamable-HTTP transport requires the client to accept BOTH media types.
# Real MCP clients (fastmcp.Client, OsworldMcpClient) send this automatically.
# We inject it when forwarding so raw callers (e.g. plain aiohttp) also work.
_MCP_ACCEPT = "application/json, text/event-stream"

# Typed AppKey slots — avoids aiohttp NotAppKeyWarning and gives type safety.
_K_UPSTREAM = web.AppKey("upstream_url", str)
_K_SERVER_ID = web.AppKey("server_id", str)
_K_BRH_DIR = web.AppKey("brh_dir")
_K_ALERTS = web.AppKey("alerts_path", str)
_K_APPROVE = web.AppKey("approve_new", bool)
_K_REG_BOX = web.AppKey("registry_box", dict)   # {"reg": dict, "path": str}
# Expose for tests that need direct registry access
REG_BOX_KEY = _K_REG_BOX


def _forward_headers(request_headers: "aiohttp.CIMultiDictProxy") -> dict:
    hdrs = {k: v for k, v in request_headers.items() if k.lower() not in _SKIP_HEADERS}
    # Ensure MCP Accept requirement is met regardless of what the client sent.
    hdrs["Accept"] = _MCP_ACCEPT
    return hdrs


async def _proxy_raw(request: web.Request, upstream_url: str) -> web.Response:
    """Pass request through to upstream unchanged and relay response."""
    body = await request.read()
    async with aiohttp.ClientSession() as session:
        async with session.request(
            request.method,
            upstream_url,
            data=body,
            headers=_forward_headers(request.headers),
            allow_redirects=False,
        ) as resp:
            resp_body = await resp.read()
            return web.Response(
                status=resp.status,
                headers={k: v for k, v in resp.headers.items()
                         if k.lower() not in _SKIP_HEADERS},
                body=resp_body,
            )


async def _handle_mcp(request: web.Request) -> web.Response:
    """Intercept POST /mcp: enforce tools/call and tools/list, forward the rest."""
    app = request.app
    upstream_url: str = app[_K_UPSTREAM]
    server_id: str = app[_K_SERVER_ID]
    brh_dir = app[_K_BRH_DIR]
    alerts_path: str = app[_K_ALERTS]
    approve_new: bool = app[_K_APPROVE]
    reg_box: dict = app[_K_REG_BOX]

    body = await request.read()

    try:
        frame = json.loads(body)
    except (ValueError, TypeError):
        return await _proxy_raw(request, upstream_url)

    # Batch frames: pass through uninspected (MCP clients don't batch tools/call in practice)
    if isinstance(frame, list):
        return await _proxy_raw(request, upstream_url)

    # --- client → server check ---
    state = read_state(brh_dir)
    routing = route_client_frame(frame, state, {}, server_id=server_id)
    for alert in routing.alerts:
        append_alert(alert["reason"], alert["detail"], alerts_path)

    if routing.to_client is not None:
        # Blocked: synthetic JSON-RPC error, no upstream contact.
        return web.Response(
            content_type="application/json",
            body=json.dumps(routing.to_client).encode(),
        )

    # --- forward to upstream ---
    async with aiohttp.ClientSession() as session:
        async with session.post(
            upstream_url,
            data=body,
            headers=_forward_headers(request.headers),
        ) as resp:
            ct = resp.content_type or ""
            resp_body = await resp.read()
            # Capture passthrough headers from upstream (e.g. Mcp-Session-Id).
            # Exclude hop-by-hop headers; keep everything else so the MCP session
            # handshake works end-to-end through the proxy.
            upstream_hdrs = {k: v for k, v in resp.headers.items()
                             if k.lower() not in _SKIP_HEADERS
                             and k.lower() != "content-type"}
            if "text/event-stream" in ct:
                # SSE response: try to extract a JSON-RPC frame for tools/list
                # hash-pinning.  Non-tools/list SSE (notifications, etc.) pass
                # through unchanged.  A rug-pull returns a JSON error; an OK
                # tools/list or an unrecognised SSE returns the original SSE.
                sse_frame = _jsonrpc_from_sse(resp_body)
                if sse_frame is not None and frame.get("method") == "tools/list":
                    fid2 = frame.get("id")
                    method2 = frame.get("method")
                    pending2: dict = {fid2: method2} if fid2 is not None else {}
                    r2, new_reg2, changed2 = route_server_frame(
                        sse_frame, server_id, reg_box["reg"], pending2,
                        approve_new=approve_new,
                    )
                    for alert in r2.alerts:
                        append_alert(alert["reason"], alert["detail"], alerts_path)
                    if changed2:
                        reg_box["reg"] = new_reg2
                        save_registry(new_reg2, reg_box["path"])
                    if r2.to_client is not sse_frame:
                        # Rug pull blocked: return JSON-RPC error (not SSE)
                        return web.Response(
                            content_type="application/json",
                            body=json.dumps(r2.to_client).encode(),
                            headers=upstream_hdrs,
                        )
                # Pass SSE through (tools/list OK or non-tools/list stream)
                return web.Response(status=resp.status, content_type=ct,
                                    body=resp_body, headers=upstream_hdrs)

    # --- server → client check (tools/list hash pinning) ---
    try:
        resp_frame = json.loads(resp_body)
    except (ValueError, TypeError):
        return web.Response(content_type="application/json", body=resp_body,
                            headers=upstream_hdrs)

    # Pre-populate pending with this request's id→method so route_server_frame
    # can correlate: in HTTP the request and response are part of the same call,
    # unlike stdio where they are separate frames on a shared stream.
    fid = frame.get("id")
    method = frame.get("method")
    pending: dict = {fid: method} if fid is not None and method else {}

    routing2, new_reg, changed = route_server_frame(
        resp_frame, server_id, reg_box["reg"], pending, approve_new=approve_new,
    )
    for alert in routing2.alerts:
        append_alert(alert["reason"], alert["detail"], alerts_path)
    if changed:
        reg_box["reg"] = new_reg
        save_registry(new_reg, reg_box["path"])

    return web.Response(
        content_type="application/json",
        body=json.dumps(routing2.to_client).encode(),
        headers=upstream_hdrs,
    )


async def _handle_other(request: web.Request) -> web.Response:
    """Pass non-/mcp paths through transparently (health checks, etc.)."""
    upstream_base: str = request.app[_K_UPSTREAM].rsplit("/mcp", 1)[0]
    return await _proxy_raw(request, upstream_base + request.path)


def _make_app(
    upstream_url: str,
    *,
    server_id: str,
    brh_dir: str | None,
    registry_path: str,
    alerts_path: str,
    approve_new: bool,
) -> web.Application:
    app = web.Application()
    app[_K_UPSTREAM] = upstream_url
    app[_K_SERVER_ID] = server_id
    app[_K_BRH_DIR] = brh_dir
    app[_K_ALERTS] = alerts_path
    app[_K_APPROVE] = approve_new
    app[_K_REG_BOX] = {"reg": load_registry(registry_path), "path": registry_path}
    app.router.add_post("/mcp", _handle_mcp)
    app.router.add_route("*", "/{path_info:.*}", _handle_other)
    return app


async def run_proxy_http(
    upstream_url: str,
    port: int,
    *,
    server_id: str,
    brh_dir: str | None = None,
    registry_path: str,
    alerts_path: str,
    approve_new: bool = False,
) -> None:
    """Start the HTTP proxy and block until cancelled.

    ``approve_new`` defaults to sealed (``False``): unregistered tools are
    rejected. Pass ``True`` only for an explicit, unattended benchmark/test
    run — never as an implicit production default."""
    app = _make_app(
        upstream_url,
        server_id=server_id,
        brh_dir=brh_dir,
        registry_path=registry_path,
        alerts_path=alerts_path,
        approve_new=approve_new,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    log.info("MCP proxy HTTP proxy  http://127.0.0.1:%d  →  %s  (server_id=%s)",
             port, upstream_url, server_id)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
