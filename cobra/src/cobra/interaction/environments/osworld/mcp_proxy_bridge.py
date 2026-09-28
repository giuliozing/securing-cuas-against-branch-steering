"""MCP proxy ↔ OSWorld-MCP bridge (BRH wiring).

The OSWorld-MCP benchmark exposes 157 MCP tools via a FastMCP HTTP server running
inside the guest VM on port 9292. The MCP client (OsworldMcpClient) runs on the
host. Without BRH, client → server is a direct connection through QEMU's hostfwd:

    host:mcp_port --Docker--▶ container:9292 --QEMU hostfwd--▶ guest:9292

With BRH_MCP=1 the MCP proxy HTTP proxy intercepts every tools/call and tools/list:

    host:9191 (MCP proxy) --▶ host:mcp_port --▶ guest:9292

This module provides:

  * ``ensure_mcp_server_in_guest(env)``   — idempotent deploy + start of the
    OSWorld-MCP FastMCP HTTP server in the QEMU guest (port 9292).
  * ``build_mcp_manifest(app_hint, upstream_url)``  — query the live MCP server
    and return a ``McpManifest`` (tool name → param names) for the P-LLM annotator.
  * ``MptHttpBridge``  — start-stop lifecycle for the MCP proxy HTTP proxy running in a
    daemon thread so the main benchmark loop is unblocked.
  * ``patch_mcp_client_url``  — redirect ``OsworldMcpClient`` to the proxy URL.

Lifecycle in ``user_tasks.py`` ``init_environment`` (both blocks are best-effort)::

    # BEFORE reset() — deploy the MCP server so it is ready when tasks run
    if os.environ.get("BRH_MCP") == "1":
        ensure_mcp_server_in_guest(ui.env)

    ui.env.reset(task_config=self.example)

    # AFTER reset() — start the host-side proxy and build the manifest
    if os.environ.get("BRH_MCP") == "1":
        upstream = f"http://localhost:{ui.env.provider.mcp_port}/mcp"
        bridge = MptHttpBridge(upstream, ...)
        bridge.start()
        ui._pab_mpt_bridge = bridge
        patch_mcp_client_url(bridge.proxy_url)
        ui._pab_mcp_manifest = build_mcp_manifest(app_hint, upstream)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tarfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_PROXY_PORT = int(__import__("os").environ.get("BRH_MCP_PROXY_PORT", "9191"))


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_mcp_manifest(
    app_hint: str | None,
    upstream_url: str = "http://localhost:9292/mcp",
) -> dict[str, list[str]]:
    """Query the OSWorld-MCP server and return a ``McpManifest``.

    ``app_hint`` is the app-domain substring used by ``OsworldMcpClient``'s RAG
    filter (e.g. ``"google_chrome"``, ``"libreoffice_calc"``).  Pass ``None`` to
    skip filtering and return every tool.

    Returns ``{tool_name: [param_names]}`` — the ``McpManifest`` type expected by
    ``generate_plan_constraints(mcp_tools=...)``.
    """
    from cobra.mcp_proxy.manifest import manifest_from_tools

    try:
        tools = _list_tools_http(app_hint, upstream_url)
    except Exception as exc:
        log.warning("mpt_osworld_bridge: could not fetch tool list from %s: %s", upstream_url, exc)
        return {}
    manifest = manifest_from_tools(tools)
    log.info("mpt_osworld_bridge: manifest built — %d tools (app_hint=%r)", len(manifest), app_hint)
    return manifest


def build_mcp_descriptions(
    app_hint: str | None,
    upstream_url: str = "http://localhost:9292/mcp",
) -> dict[str, str]:
    """Return ``{tool_name: description}`` for the app's MCP tools.

    Companion to :func:`build_mcp_manifest`, which intentionally drops descriptions
    (the BRH annotator only needs names + param names). Descriptions are surfaced to
    the P-LLM *only* for pre-approved/trusted tools so it can judge, from trusted
    text, whether a tool actually completes a task step. OSWorld-MCP treats all
    built-in tools as trusted; in a real deployment this must be gated on explicit
    human approval + hash-pin.
    """
    try:
        tools = _list_tools_http(app_hint, upstream_url)
    except Exception as exc:
        log.warning("mpt_osworld_bridge: could not fetch tool descriptions from %s: %s", upstream_url, exc)
        return {}
    return {
        t["name"]: (t.get("description") or "").strip()
        for t in tools
        if t.get("name")
    }


def _list_tools_http(app_hint: str | None, upstream_url: str) -> list[dict]:
    """One-shot async call to the upstream MCP HTTP server to enumerate tools.

    Performs the full MCP session handshake (initialize → notifications/initialized
    → tools/list) because FastMCP requires a session before responding to requests.
    Handles both JSON and SSE (text/event-stream) response bodies.
    """
    import aiohttp

    async def _fetch() -> list[dict]:
        ACCEPT = "application/json, text/event-stream"
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession() as http:
            # 1. initialize → obtain session ID
            init = {
                "jsonrpc": "2.0", "id": 0, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mpt-manifest", "version": "0"},
                },
            }
            async with http.post(upstream_url, json=init,
                                 headers={"Accept": ACCEPT},
                                 timeout=timeout) as r:
                sid = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
                await r.read()  # drain body

            if not sid:
                log.warning("_list_tools_http: no session ID from %s — tools/list may fail", upstream_url)

            # 2. notifications/initialized (fire-and-forget)
            notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            hdrs = {"Accept": ACCEPT}
            if sid:
                hdrs["Mcp-Session-Id"] = sid
            async with http.post(upstream_url, json=notif, headers=hdrs,
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                await r.read()

            # 3. tools/list
            payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            if sid:
                hdrs["Mcp-Session-Id"] = sid
            async with http.post(upstream_url, json=payload, headers=hdrs,
                                 timeout=timeout) as r:
                body = await _parse_mcp_response(r)

        tools: list[dict] = (body.get("result") or {}).get("tools") or []
        if app_hint:
            tools = _rag_filter(tools, app_hint)
        return tools

    return asyncio.run(_fetch())


async def _parse_mcp_response(r) -> dict:
    """Read an MCP HTTP response regardless of JSON vs SSE body."""
    ct = r.headers.get("Content-Type", "")
    raw = await r.read()
    if "text/event-stream" in ct:
        for line in raw.decode(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("data:"):
                try:
                    obj = json.loads(line[len("data:"):].strip())
                    if "result" in obj or "error" in obj:
                        return obj
                except Exception:
                    pass
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


_EXCLUDE_APPS = {
    "libreoffice_calc", "libreoffice_impress", "libreoffice_writer",
    "code", "vlc", "google_chrome", "thunderbird",
}


def _rag_filter(tools: list[dict], app_hint: str) -> list[dict]:
    """Mirror OsworldMcpClient's RAG filtering: prefer exact app match, fall back
    to all non-excluded tools if no match found."""
    matched = [t for t in tools if app_hint in (t.get("name") or "")]
    if matched:
        return matched
    return [t for t in tools if not any(excl in (t.get("name") or "") for excl in _EXCLUDE_APPS)]


# ---------------------------------------------------------------------------
# MCP proxy HTTP proxy lifecycle
# ---------------------------------------------------------------------------

class MptHttpBridge:
    """Runs the MCP proxy HTTP proxy in a daemon thread.

    The proxy listens on ``proxy_port`` (default 9191) and forwards to
    ``upstream_url``.  ``start()`` blocks until the proxy is ready (bound).
    ``stop()`` signals shutdown and waits for the thread to exit.

    Attach to ``ui._pab_mpt_bridge``; ``_teardown_base_ui`` calls ``stop()``.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        proxy_port: int = _PROXY_PORT,
        brh_dir: str | Path | None = None,
        registry_path: str | None = None,
        alerts_path: str | None = None,
        server_id: str | None = None,
        approve_new: bool = False,
    ) -> None:
        self.upstream_url = upstream_url
        self.proxy_port = proxy_port
        self.proxy_url = f"http://localhost:{proxy_port}/mcp"
        brh_dir = Path(brh_dir) if brh_dir else Path("/tmp/brh")
        self._pab_dir = brh_dir
        self._registry_path = registry_path or str(brh_dir / "mcp_registry.json")
        self._alerts_path = alerts_path or str(brh_dir / "mcp_alerts.jsonl")
        self._server_id = server_id or upstream_url
        self._approve_new = approve_new
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._stop_event: asyncio.Event | None = None

    def start(self) -> None:
        """Start the proxy thread and block until the proxy is listening."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="mpt-http-proxy")
        self._thread.start()
        if not self._ready.wait(timeout=10):
            log.warning("mpt_osworld_bridge: proxy did not signal ready within 10 s")
        else:
            log.info("mpt_osworld_bridge: MCP proxy ready at %s → %s",
                     self.proxy_url, self.upstream_url)

    def stop(self) -> None:
        """Signal shutdown and wait for the thread to exit."""
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        from cobra.mcp_proxy.proxy_http import _make_app
        from aiohttp import web

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _serve() -> None:
            self._stop_event = asyncio.Event()
            app = _make_app(
                self.upstream_url,
                server_id=self._server_id,
                brh_dir=str(self._pab_dir),
                registry_path=self._registry_path,
                alerts_path=self._alerts_path,
                approve_new=self._approve_new,
            )
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", self.proxy_port)
            await site.start()
            self._ready.set()
            await self._stop_event.wait()
            await runner.cleanup()

        try:
            loop.run_until_complete(_serve())
        except Exception as exc:
            log.error("mpt_osworld_bridge: proxy thread crashed: %s", exc)
            self._ready.set()  # unblock start() even on failure
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# OsworldMcpClient URL redirect helper
# ---------------------------------------------------------------------------

def patch_mcp_client_url(proxy_url: str = "http://localhost:9191/mcp") -> None:
    """Redirect OsworldMcpClient to the MCP proxy URL for this process.

    Idempotent: calling again with the same URL is a no-op.  Call this after
    ``MptHttpBridge.start()`` so all subsequent MCP tool calls go through MCP proxy.
    """
    try:
        from osworld_mcp.mcp.osworld_mcp_client import OsworldMcpClient  # type: ignore[import]
        OsworldMcpClient.config["mcpServers"]["osworld_mcp"]["url"] = proxy_url
        log.info("mpt_osworld_bridge: OsworldMcpClient redirected → %s", proxy_url)
    except ImportError:
        log.debug("mpt_osworld_bridge: OsworldMcpClient not importable (osworld_mcp package absent)")


# ---------------------------------------------------------------------------
# In-guest MCP server deploy + lifecycle
# ---------------------------------------------------------------------------

_GUEST_HOME = "/home/user"
_GUEST_MCP_DIR = f"{_GUEST_HOME}/mcp_server"
_GUEST_MCP_LOG = f"{_GUEST_HOME}/mcp_server.log"
_MCP_GUEST_PORT = 9292

# server.py with graceful per-package imports so missing deps (playwright,
# pyautogui display issues) skip that tool class instead of crashing the server.
_SERVER_PY = '''\
"""OSWorld-MCP FastMCP server — graceful import variant for BRH wiring."""
import json
from fastmcp import FastMCP
from fastmcp.tools.tool import Tool
try:
    import mcp.types as types
except ImportError:
    class _FakeMcpTypes:
        class Tool:
            def __init__(self, name, description, inputSchema): pass
    types = _FakeMcpTypes()


def _try_cls(module_path, *names):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return [getattr(mod, n) for n in names]
    except Exception as exc:
        print(f"[mcp-server] skip {module_path}: {exc}")
        return []


_TOOL_CLASSES = (
    _try_cls("tools.package.code",                 "CodeTools", "VSCodeTools")
    + _try_cls("tools.package.google_chrome",      "BrowserTools")
    + _try_cls("tools.package.libreoffice_calc",   "CalcTools", "CalcToolsPlus")
    + _try_cls("tools.package.libreoffice_impress","ImpressTools", "PresentationToolsUNO")
    + _try_cls("tools.package.libreoffice_writer", "WriterTools")
    + _try_cls("tools.package.vlc",               "VLCTools")
    + _try_cls("tools.package.os",                "UnifiedTools")
)

_META = {
    "CodeTools": "code", "VSCodeTools": "code2",
    "BrowserTools": "google_chrome",
    "CalcTools": "libreoffice_calc", "CalcToolsPlus": "libreoffice_calc2",
    "ImpressTools": "libreoffice_impress", "PresentationToolsUNO": "libreoffice_impress2",
    "WriterTools": "libreoffice_writer",
    "UnifiedTools": "os", "VLCTools": "vlc",
}


def init_server(name="OSWorld"):
    mcp = FastMCP(name)
    all_tools = []
    for cls in _TOOL_CLASSES:
        prefix = _META.get(cls.__name__, cls.__name__.lower())
        apis_path = f"tools/apis/{prefix}.json"
        try:
            with open(apis_path, encoding="utf-8") as f:
                apis = json.load(f)
        except FileNotFoundError:
            continue
        for entry in apis:
            fn_info = entry.get("function", {})
            method = fn_info["name"].split(".")[-1]
            func = getattr(cls, method, None)
            if not func:
                continue
            tool_name = f"{prefix}.{method}"
            mcp.add_tool(Tool.from_function(func, name=tool_name))
            all_tools.append(types.Tool(
                name=tool_name,
                description=fn_info["description"],
                inputSchema=fn_info["parameters"],
            ))
    all_tools.sort(key=lambda t: t.name)

    async def list_tools():
        return all_tools

    mcp._mcp_server.list_tools()(list_tools)
    return mcp


if __name__ == "__main__":
    mcp = init_server("OSWorld")
    mcp.run(transport="http", host="0.0.0.0", port=9292)
'''


def _mcp_server_root() -> Path:
    """Locate the osworld_mcp/mcp/mcp_server directory on the host."""
    override = os.environ.get("OSWORLD_MCP_SERVER_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "osworld_mcp" / "mcp" / "mcp_server"
        if (cand / "tools" / "apis").is_dir():
            return cand
    raise FileNotFoundError(
        "could not locate osworld_mcp/mcp/mcp_server on host; "
        "set OSWORLD_MCP_SERVER_ROOT to override"
    )


def _make_mcp_server_tar() -> bytes:
    """Build a tarball of the MCP server files to push to the guest.

    Includes tools/apis/*.json and tools/package/*.py (the tool implementations).
    Replaces server.py with the graceful-import variant so missing deps (playwright,
    uno) cause individual tool classes to be skipped instead of crashing the server.
    """
    root = _mcp_server_root()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Graceful server.py (replaces original)
        server_bytes = _SERVER_PY.encode()
        info = tarfile.TarInfo(name="server.py")
        info.size = len(server_bytes)
        tf.addfile(info, io.BytesIO(server_bytes))

        # tools/apis/*.json — needed at runtime for tool metadata
        apis_dir = root / "tools" / "apis"
        for json_file in sorted(apis_dir.glob("*.json")):
            tf.add(json_file, arcname=f"tools/apis/{json_file.name}")

        # tools/package/*.py — tool implementations
        pkg_dir = root / "tools" / "package"
        for py_file in sorted(pkg_dir.glob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            tf.add(py_file, arcname=f"tools/package/{py_file.name}")

    return buf.getvalue()


def _sh_guest(env, cmd: str, timeout: int = 120) -> str:
    """Run a bash command in the QEMU guest via the OSWorld controller."""
    pysrc = (
        "import subprocess,sys\n"
        f"r=subprocess.run(['bash','-lc', {cmd!r}], capture_output=True, text=True, timeout={timeout})\n"
        "print('RC=%d' % r.returncode)\n"
        "sys.stdout.write(r.stdout)\n"
        "sys.stderr.write(r.stderr)\n"
    )
    try:
        res = env.controller.execute_python_command(pysrc)
    except Exception as exc:
        return f"<exec error: {exc}>"
    if isinstance(res, dict):
        return (res.get("output") or "") + (res.get("error") or "")
    return str(res)


def _push_to_guest(env, guest_path: str, data: bytes) -> bool:
    """Push binary data to a path in the QEMU guest."""
    import base64
    b64 = base64.b64encode(data).decode()
    pysrc = (
        "import base64,pathlib\n"
        f"p=pathlib.Path({guest_path!r}); p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_bytes(base64.b64decode({b64!r}))\n"
        "print('OK', p.stat().st_size)\n"
    )
    try:
        res = env.controller.execute_python_command(pysrc)
        return bool(res)
    except Exception as exc:
        log.warning("mpt_osworld_bridge: push to %s failed: %s", guest_path, exc)
        return False


def _port_up_in_guest(env, port: int) -> bool:
    out = _sh_guest(env, f"(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':{port} ' && echo UP || echo DOWN")
    return "UP" in out


# LibreOffice profile dir in the OSWorld Ubuntu guest (user "user").
_LO_PROFILE = "/home/user/.config/libreoffice/4"


def preclean_libreoffice_guest(env, *, timeout: int = 60) -> str:
    """Reset LibreOffice to a clean, recovery-free state in the QEMU guest.

    Motivation: under the
    BRH_MCP lightweight reset the container is *not* recreated between attempts
    (qcow2 is read-only, is_environment_used=False), so a soffice crash during
    attempt N leaves a "Document Recovery" modal + stale profile lock that
    blocks _open_setup and confuses verify on attempts N+1..5. This helper is
    called before each LibreOffice task's setup so every attempt starts from a
    clean, recovery-free profile (graceful soffice shutdown + pending-recovery
    strip), regardless of whether the prior attempt crashed soffice.

    Idempotent and best-effort. Gated by the caller to libreoffice tasks only,
    so blast radius on non-libreoffice tasks is zero.
    """
    # Two failure modes produce the blocking "Document Recovery" dialog that hangs
    # the next _open_setup (open_file waits up to 1800s for a window titled after
    # the doc, but the recovery dialog owns the window instead):
    #   (a) a genuine soffice crash during the attempt, and
    #   (b) our own SIGKILL of a *live* soffice — an unclean shutdown that makes
    #       LibreOffice write a RecoveryList entry and offer recovery on relaunch.
    # So we (1) shut soffice down GRACEFULLY (SIGTERM, wait; SIGKILL only if it
    # refuses) to avoid manufacturing recovery state, then (2) STRIP any pending
    # recovery entries that a real crash already wrote — remove every
    # /org.openoffice.Office.Recovery item from registrymodifications.xcu (the
    # RecoveryList is what triggers the dialog; merely appending "disabled" flags
    # does NOT clear an already-pending recovery), and (3) drop stale lock/backup.
    reg = f"{_LO_PROFILE}/user/registrymodifications.xcu"
    cmd = (
        # (1) graceful shutdown first, escalate to -9 only if still alive
        "pkill -TERM -f soffice.bin 2>/dev/null; pkill -TERM -f oosplash 2>/dev/null; "
        "for i in $(seq 1 8); do pgrep -f soffice.bin >/dev/null || break; sleep 1; done; "
        "pkill -9 -f soffice.bin 2>/dev/null; pkill -9 -f oosplash 2>/dev/null; sleep 1; "
        # (3) stale locks + recovery backup store
        f"rm -f {_LO_PROFILE}/.lock {_LO_PROFILE}/user/.lock 2>/dev/null; "
        "rm -f /home/user/Desktop/.~lock.*# 2>/dev/null; "
        f"rm -rf {_LO_PROFILE}/user/backup 2>/dev/null; mkdir -p {_LO_PROFILE}/user/backup 2>/dev/null; "
        "echo cleaned"
    )
    # (2) strip the pending recovery from the registry (regex over the single XML).
    strip_recovery_py = (
        "import pathlib,re\n"
        f"p=pathlib.Path({reg!r})\n"
        "if p.exists():\n"
        "    t=p.read_text()\n"
        "    n=len(t)\n"
        "    t=re.sub(r'<item oor:path=\"/org.openoffice.Office.Recovery[^\"]*\">.*?</item>', '', t, flags=re.S)\n"
        "    p.write_text(t)\n"
        "    print('recovery-stripped', n-len(t))\n"
        "else:\n"
        "    print('no-registry')\n"
    )
    out = _sh_guest(env, cmd, timeout=timeout)
    try:
        res = env.controller.execute_python_command(strip_recovery_py)
        out += " | " + (res.get("output", "") if isinstance(res, dict) else str(res))
    except Exception as exc:
        log.warning("preclean_libreoffice_guest: recovery-strip failed: %s", exc)
    log.info("preclean_libreoffice_guest: %s", out.strip())
    return out


def ensure_mcp_server_in_guest(
    env,
    *,
    guest_port: int = _MCP_GUEST_PORT,
    pip_timeout: int = 400,
    start_timeout: int = 60,
) -> bool:
    """Idempotently deploy and start the OSWorld-MCP HTTP server in the QEMU guest.

    Mirrors ``ensure_http_proxy()`` in ``http_proxy_bridge.py``:  pushes a tarball of
    the MCP server files to ``~/mcp_server/``, pip-installs ``fastmcp`` +
    ``uvicorn`` (pyautogui is typically pre-installed in the guest), and starts
    the server in background with ``DISPLAY=:0``.  Returns True if the server is
    listening on ``guest_port`` after startup.

    Idempotent: if the port is already bound, returns True immediately without
    re-deploying.  Call this BEFORE ``reset()`` so the server is ready when task
    config steps run.
    """
    if _port_up_in_guest(env, guest_port):
        log.info("mpt_osworld_bridge: MCP server already listening in guest on :%d — reusing", guest_port)
        return True

    log.info("mpt_osworld_bridge: deploying MCP server to guest (port %d)", guest_port)

    # 1. Push server tarball
    try:
        tar_data = _make_mcp_server_tar()
    except FileNotFoundError as exc:
        log.error("mpt_osworld_bridge: cannot build server tarball: %s", exc)
        return False

    _push_to_guest(env, f"{_GUEST_HOME}/mcp_server.tgz", tar_data)
    _sh_guest(env, f"rm -rf {_GUEST_MCP_DIR} && mkdir -p {_GUEST_MCP_DIR} && "
                   f"tar xzf {_GUEST_HOME}/mcp_server.tgz -C {_GUEST_MCP_DIR}")

    # 2. Install Python deps (fastmcp + uvicorn; pyautogui usually pre-installed)
    _sh_guest(env,
        "pip3 install --user -q fastmcp uvicorn 2>&1 | tail -3",
        timeout=pip_timeout,
    )
    # playwright is optional — BrowserTools skipped gracefully if missing
    _sh_guest(env,
        "pip3 install --user -q playwright 2>&1 | tail -3 ; true",
        timeout=pip_timeout,
    )

    # 3. Start server in background
    _sh_guest(env, (
        f"cd {_GUEST_MCP_DIR} && "
        "DISPLAY=:0 PYTHONPATH=. "
        "setsid nohup python3 server.py "
        f"> {_GUEST_MCP_LOG} 2>&1 < /dev/null & "
        "echo started_pid=$!"
    ))

    # 4. Wait for port to be listening
    deadline = time.time() + start_timeout
    while time.time() < deadline:
        if _port_up_in_guest(env, guest_port):
            log.info("mpt_osworld_bridge: MCP server up in guest on :%d", guest_port)
            return True
        time.sleep(2)

    # Log tail for diagnosis
    log_tail = _sh_guest(env, f"tail -20 {_GUEST_MCP_LOG} 2>/dev/null || echo '(no log)'")
    log.error(
        "mpt_osworld_bridge: MCP server did not start within %ds. Log tail:\n%s",
        start_timeout, log_tail,
    )
    return False


_UNO_PORT = 2002


def ensure_soffice_uno_listener_in_guest(env, *, uno_port: int = _UNO_PORT, start_timeout: int = 20) -> bool:
    """Idempotently expose the UNO API socket the LibreOffice MCP tools need.

    osworld-mcp's libreoffice_calc/impress/writer tools connect via
    ``uno:socket,host=localhost,port=2002;urp;...`` (see
    ``mcp/mcp_server/tools/package/libreoffice_calc.py``), matching the
    upstream ``launch_soffice.sh`` helper it ships. OSWorld's own task-config
    "open" step just double-clicks the file — it never starts soffice with
    ``--accept=socket``, so that port is never listening and every LibreOffice
    MCP tool call fails with "Connector: couldn't connect to socket
    (Connection refused)" regardless of the tool or its arguments.

    Call this AFTER ``env.reset()`` (once the target document is already
    open). LibreOffice enforces a single-instance policy: invoking ``soffice
    --accept=...`` while an instance is already running forwards the option
    to that running instance via its own IPC instead of spawning a second
    window, so this does not disturb the open document.

    Idempotent: if the port is already bound, returns True immediately.
    """
    if _port_up_in_guest(env, uno_port):
        log.info("mpt_osworld_bridge: UNO socket already listening in guest on :%d — reusing", uno_port)
        return True

    log.info("mpt_osworld_bridge: exposing UNO socket in guest (port %d) for LibreOffice MCP tools", uno_port)
    _sh_guest(env, (
        f'DISPLAY=:0 setsid nohup soffice --accept="socket,host=localhost,port={uno_port};urp;" '
        '--norestore --nologo --nodefault > /tmp/soffice_uno_listener.log 2>&1 < /dev/null & '
        "echo started_pid=$!"
    ))

    deadline = time.time() + start_timeout
    while time.time() < deadline:
        if _port_up_in_guest(env, uno_port):
            log.info("mpt_osworld_bridge: UNO socket up in guest on :%d", uno_port)
            return True
        time.sleep(2)

    log_tail = _sh_guest(env, "tail -20 /tmp/soffice_uno_listener.log 2>/dev/null || echo '(no log)'")
    log.error(
        "mpt_osworld_bridge: UNO socket did not come up within %ds. Log tail:\n%s",
        start_timeout, log_tail,
    )
    return False
