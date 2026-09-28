"""HTTP proxy <-> OSWorld-guest bridge.

The BRH hook writes ``branch_state.json`` on the HOST; the HTTP proxy enforcer runs
IN the guest (the rootless double-NAT makes a host-side proxy unreachable from the
guest). This module connects the two:

  * ``ensure_http_proxy(env, ...)``  — idempotent in-guest bootstrap: pip-install
    mitmproxy, push the ``cobra.http_proxy`` package, install the mitmproxy CA into
    the system store *and* Chrome's NSS db, force Chrome through the proxy via a
    managed policy, and start ``mitmdump`` with the real addon (monitor|enforce).
  * ``StateSyncer``               — host thread that pushes the host
    ``branch_state.json`` into the guest whenever it changes (host->guest sync,
    the counterpart of HTTP proxy's per-request polling of that file).
  * ``pull_alerts(env)``          — fetch the guest alert log back for metrics.

Only the in-guest sudo is used (user ``user`` is in the sudo group; password is the
docker-provider ``client_password``). No host sudo anywhere.

All functions are best-effort and log instead of raising, so a wiring hiccup never
crashes the benchmark pipeline (the enforcer itself stays fail-closed in-guest).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import tarfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

GUEST_PW = os.environ.get("BRH_GUEST_PASSWORD", "password")
GUEST_HOME = "/home/user"
GUEST_PKG_DIR = f"{GUEST_HOME}/cm"
GUEST_STATE = f"{GUEST_HOME}/branch_state.json"
GUEST_ALERTS = f"{GUEST_HOME}/brh_alerts.jsonl"
GUEST_PASSLIST = f"{GUEST_HOME}/passlist.txt"
GUEST_FRONT = f"{GUEST_HOME}/http_proxy_front.py"
LISTEN_PORT = int(os.environ.get("BRH_HTTP_PROXY_PORT", "8080"))
# Concurrent enforcer workers: N mitmdump backends behind a round-robin TCP front,
# so a heavy page's connection fan-out is spread across N event loops/vCPUs instead
# of saturating one. N=1 keeps the original single-proxy path.
WORKERS = max(1, int(os.environ.get("BRH_HTTP_PROXY_WORKERS", "4")))

# In-guest round-robin TCP front (stdlib asyncio only; pure splice, no TLS/HTTP parse
# — the backends do the proxy work). Pushed to the guest and run with system python3.
_FRONT_SRC = '''\
"""HTTP proxy round-robin TCP front: distributes Chrome\'s proxy connections across N
mitmdump backends so a heavy page hits N event loops/cores, not one. Pure TCP relay.
Failover: if the chosen backend is dead (OSError/ECONNREFUSED), tries the next one
in round-robin order before closing the client connection."""
import argparse, asyncio, itertools


async def _pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        try:
            writer.write_eof()
        except (OSError, RuntimeError):
            pass


async def _handle(cr, cw, backends, rr):
    # Try each backend in round-robin order; skip dead ones (ECONNREFUSED / OSError).
    br = bw = None
    for _ in range(len(backends)):
        try:
            br, bw = await asyncio.open_connection(*next(rr))
            break
        except OSError:
            continue
    if br is None:
        try:
            cw.close()
        except OSError:
            pass
        return
    t1 = asyncio.ensure_future(_pipe(cr, bw))
    t2 = asyncio.ensure_future(_pipe(br, cw))
    await asyncio.wait([t1, t2])
    for w in (cw, bw):
        try:
            w.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--backends", required=True)
    args = ap.parse_args()
    backends = [("127.0.0.1", int(p)) for p in args.backends.split(",") if p.strip()]
    rr = itertools.cycle(backends)

    async def run():
        async def cb(r, w):
            await _handle(r, w, backends, rr)
        server = await asyncio.start_server(cb, args.listen_host, args.listen_port)
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
'''


# --------------------------------------------------------------------------- #
# guest command helpers
# --------------------------------------------------------------------------- #
def _sh(env, cmd: str, timeout: int = 120) -> str:
    """Run a bash login-shell command in the guest, return combined stdout/stderr."""
    pysrc = (
        "import subprocess,sys\n"
        f"r=subprocess.run(['bash','-lc', {cmd!r}], capture_output=True, text=True, timeout={timeout})\n"
        "print('RC=%d' % r.returncode)\n"
        "sys.stdout.write(r.stdout)\n"
        "sys.stderr.write(r.stderr)\n"
    )
    try:
        res = env.controller.execute_python_command(pysrc)
    except Exception as e:  # controller hiccup (e.g. slow /execute) — non-fatal
        return f"<exec error: {e}>"
    if isinstance(res, dict):
        return (res.get("output") or "") + (res.get("error") or "")
    return str(res)


def _push_bytes(env, guest_path: str, data: bytes) -> bool:
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
    except Exception as e:
        logger.warning("http_proxy_bridge: push to %s failed: %s", guest_path, e)
        return False


def _http_proxy_root() -> Path:
    """Locate the host `http_proxy` package dir (contains __init__.py + proxy/)."""
    override = os.environ.get("HTTP_PROXY_PROXY_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "http_proxy" / "legacy" / "http_proxy"
        if (cand / "proxy" / "http_proxy_addon.py").exists():
            return cand
    raise FileNotFoundError("could not locate http_proxy/legacy/http_proxy on host")


def _make_http_proxy_tar() -> bytes:
    root = _http_proxy_root()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(root / "__init__.py", arcname="http_proxy/__init__.py")
        proxy = root / "proxy"
        for fn in os.listdir(proxy):
            if fn.endswith(".py") and not fn.endswith("_test.py"):
                tf.add(proxy / fn, arcname=f"http_proxy/proxy/{fn}")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def _port_up(env, port: int) -> bool:
    out = _sh(env, f"(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':{port} ' && echo UP || echo DOWN")
    return "UP" in out


def _listener_up(env) -> bool:
    # The thing on LISTEN_PORT is the front (WORKERS>1) or the single mitmdump (WORKERS==1).
    return _port_up(env, LISTEN_PORT)


def _start_backend(env, port: int, mode: str, guest_state: str, alerts_path: str,
                   passlist_opt: str, idx) -> None:
    """Launch one mitmdump enforcer backend on its own port (own log/alert file)."""
    tag = "" if idx is None else f".{idx}"
    start = (
        f"cd {GUEST_HOME} && PYTHONPATH={GUEST_PKG_DIR} setsid nohup ~/.local/bin/mitmdump "
        f"--listen-host 127.0.0.1 --listen-port {port} -q "
        f"-s {GUEST_PKG_DIR}/http_proxy/proxy/http_proxy_addon.py "
        f"--set brh_mode={mode} --set brh_state={guest_state} --set brh_alerts={alerts_path} "
        f"{passlist_opt}"
        f"> {GUEST_HOME}/mitm{tag}.log 2>&1 < /dev/null & echo pid=$!"
    )
    _sh(env, start)


def wait_guest_ready(env, timeout: int = 120, interval: float = 3.0) -> bool:
    """Block until the in-guest controller answers a trivial command.

    ``ensure_http_proxy`` now runs *before* ``DesktopEnv.reset()`` so that Chrome is
    launched already proxied (see ``user_tasks.init_environment``). At that point the container is booted (``_start_emulator``
    ran in ``DesktopEnv.__init__``) but the guest HTTP server may need a few seconds
    before ``execute_python_command`` succeeds. OSWorld's own ``setup()`` has an
    equivalent ``/terminal`` retry loop; we mirror it for the pre-reset window.
    Best-effort: returns ``False`` on timeout (the caller still proceeds; the
    enforcer stays fail-closed in-guest).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "__guest_ready__" in _sh(env, "echo __guest_ready__", timeout=20):
            return True
        time.sleep(interval)
    logger.warning("http_proxy_bridge: guest controller not ready after %ss", timeout)
    return False


def inject_chrome_proxy(task_config: dict, port: int = LISTEN_PORT) -> bool:
    """Append ``--proxy-server`` to the task's ``google-chrome`` launch step.

    Why: OSWorld launches Chrome from a ``launch``
    setup step inside ``reset()`` (e.g. ``["google-chrome", "--remote-debugging-
    port=1337"]``), and Chrome only honours a proxy from its launch flags / managed
    policy *at launch time*. ``ensure_http_proxy``'s managed policy was written after
    reset, so the agent's Chrome never routed through mitmproxy. We make the launch
    flag the *primary* mechanism: mutate the task config in place so Chrome is born
    proxied. Idempotent (skips if a ``--proxy-server`` arg is already present);
    leaves the managed policy in place as a secondary safeguard. Returns ``True``
    if a chrome launch step was found (and is now proxied).

    Note: Chrome does not proxy loopback by default, so the in-guest CDP/remote-
    debugging socket (port 1337/9222, used by OSWorld setup + evaluators) is
    unaffected.
    """
    proxy_arg = f"--proxy-server=http://127.0.0.1:{port}"
    patched = False
    for step in (task_config or {}).get("config", []) or []:
        if not isinstance(step, dict) or step.get("type") != "launch":
            continue
        cmd = (step.get("parameters") or {}).get("command")
        if not isinstance(cmd, list) or not cmd or cmd[0] != "google-chrome":
            continue
        if any(isinstance(a, str) and a.startswith("--proxy-server") for a in cmd):
            patched = True  # already proxied (idempotent re-entry)
            continue
        cmd.append(proxy_arg)
        patched = True
        logger.info("http_proxy_bridge: injected %s into chrome launch step", proxy_arg)
    if not patched:
        logger.warning("http_proxy_bridge: no google-chrome launch step found to proxy")
    return patched


def ensure_http_proxy(env, *, mode: str = "monitor", passlist_text: str | None = None,
                    guest_state: str = GUEST_STATE, guest_alerts: str = GUEST_ALERTS) -> dict:
    """Idempotently bring up the in-guest HTTP proxy. Returns guest paths."""
    info = {"guest_state": guest_state, "guest_alerts": guest_alerts, "port": LISTEN_PORT, "mode": mode}

    # When called pre-reset, the guest server may not answer immediately.
    wait_guest_ready(env)

    if _listener_up(env):
        logger.info("http_proxy_bridge: proxy already listening on :%d — reusing", LISTEN_PORT)
        # still refresh passlist/state files below
    else:
        logger.info("http_proxy_bridge: installing + starting in-guest HTTP proxy (mode=%s)", mode)
        # 1. mitmproxy (user-level) if missing
        _sh(env, "command -v ~/.local/bin/mitmdump >/dev/null 2>&1 || pip3 install --user -q mitmproxy", timeout=400)
        # 2. push the cobra.http_proxy package
        _push_bytes(env, f"{GUEST_HOME}/cm_pkg.tgz", _make_http_proxy_tar())
        _sh(env, f"cd {GUEST_HOME} && rm -rf cm && mkdir cm && tar xzf cm_pkg.tgz -C cm")

    # 3. passlist (always refresh)
    if passlist_text:
        _push_bytes(env, GUEST_PASSLIST, passlist_text.encode())

    # 4. (re)start the enforcer if needed. WORKERS>1 -> N mitmdump backends (N event
    #    loops across N vCPUs) behind a round-robin TCP front on LISTEN_PORT, so a
    #    heavy page's connection fan-out is spread instead of saturating one loop.
    #    WORKERS==1 -> original single proxy.
    if not _listener_up(env):
        passlist_opt = f"--set brh_passlist={GUEST_PASSLIST} " if passlist_text else ""
        if WORKERS <= 1:
            _start_backend(env, LISTEN_PORT, mode, guest_state, guest_alerts, passlist_opt, None)
            time.sleep(6)
        else:
            back_ports = [LISTEN_PORT + 1 + i for i in range(WORKERS)]
            for i, port in enumerate(back_ports):
                # Per-worker alert file ({guest_alerts}.{i}); merged on pull (the
                # per-process de-dup means each worker emits its own first-per-host,
                # but metrics count distinct hosts, so FP-A/FP-B are unaffected).
                _start_backend(env, port, mode, guest_state, f"{guest_alerts}.{i}",
                               passlist_opt, i)
            # Front the backends only once they are bound (else it relays to nothing).
            deadline = time.time() + 45
            while time.time() < deadline and not all(_port_up(env, p) for p in back_ports):
                time.sleep(1.0)
            _push_bytes(env, GUEST_FRONT, _FRONT_SRC.encode())
            csv = ",".join(str(p) for p in back_ports)
            front = (
                f"cd {GUEST_HOME} && setsid nohup python3 {GUEST_FRONT} "
                f"--listen-host 127.0.0.1 --listen-port {LISTEN_PORT} --backends {csv} "
                f"> {GUEST_HOME}/front.log 2>&1 < /dev/null & echo pid=$!"
            )
            _sh(env, front)
            time.sleep(2)
            info["backends"] = back_ports
        info["workers"] = WORKERS

    # 5. install the mitmproxy CA (system store + Chrome NSS db), idempotent
    _sh(env, (
        f"test -f /usr/local/share/ca-certificates/mitmproxy.crt || "
        f"(echo {GUEST_PW} | sudo -S cp ~/.mitmproxy/mitmproxy-ca-cert.pem "
        f"/usr/local/share/ca-certificates/mitmproxy.crt && "
        f"echo {GUEST_PW} | sudo -S update-ca-certificates >/dev/null 2>&1); "
        f"command -v certutil >/dev/null 2>&1 || (echo {GUEST_PW} | sudo -S apt-get install -y -q libnss3-tools >/dev/null 2>&1)"
    ), timeout=200)
    _sh(env, (
        "mkdir -p ~/.pki/nssdb && "
        "(certutil -d sql:$HOME/.pki/nssdb -L >/dev/null 2>&1 || certutil -N --empty-password -d sql:$HOME/.pki/nssdb); "
        "certutil -d sql:$HOME/.pki/nssdb -L | grep -qi mitmproxy || "
        "certutil -d sql:$HOME/.pki/nssdb -A -t 'C,,' -n mitmproxy -i ~/.mitmproxy/mitmproxy-ca-cert.pem"
    ))

    # 6. force Chrome through the proxy for *every* launch via a managed policy
    policy = json.dumps({"ProxyMode": "fixed_servers", "ProxyServer": f"127.0.0.1:{LISTEN_PORT}"})
    _sh(env, (
        f"echo {GUEST_PW} | sudo -S mkdir -p /etc/opt/chrome/policies/managed && "
        f"echo {GUEST_PW} | sudo -S bash -c \"cat > /etc/opt/chrome/policies/managed/http_proxy.json <<'EOF'\n{policy}\nEOF\""
    ))

    info["listening"] = _listener_up(env)
    logger.info("http_proxy_bridge: ensure_http_proxy done: %s", info)
    return info


def restart_http_proxy_backends(env, *, mode: str | None = None,
                               passlist_text: str | None = None,
                               guest_state: str = GUEST_STATE,
                               guest_alerts: str = GUEST_ALERTS) -> None:
    """Kill existing mitmdump backends + front; start fresh ones.

    Call between task retries so each retry gets unstressed backends.
    Fast: mitmproxy is already installed — only kills + restarts processes (~8-15s).
    Must be called BEFORE env.reset() so Chrome finds a live proxy at launch.
    """
    if mode is None:
        mode = os.environ.get("BRH_HTTP_PROXY_MODE", "monitor")
    if passlist_text is None:
        pl_path = os.environ.get("BRH_HTTP_PROXY_PASSLIST")
        passlist_text = (Path(pl_path).read_text(encoding="utf-8")
                         if pl_path and Path(pl_path).exists() else None)

    logger.info("http_proxy_bridge: restarting backends for new retry (mode=%s)", mode)
    # SIGKILL (not SIGTERM) so stuck mitmdump processes can't ignore the signal.
    _sh(env, "pkill -9 -f mitmdump 2>/dev/null || true; pkill -9 -f http_proxy_front 2>/dev/null || true")
    # Wait until all backend ports are confirmed free — a new mitmdump that tries
    # to bind a port still in use exits immediately, producing a PID but never listening.
    all_ports = ([LISTEN_PORT] if WORKERS <= 1
                 else [LISTEN_PORT] + [LISTEN_PORT + 1 + i for i in range(WORKERS)])
    free_deadline = time.time() + 20
    while time.time() < free_deadline and any(_port_up(env, p) for p in all_ports):
        time.sleep(1.0)
    still_up = [p for p in all_ports if _port_up(env, p)]
    if still_up:
        logger.warning("http_proxy_bridge: ports still up after kill: %s", still_up)

    if passlist_text:
        _push_bytes(env, GUEST_PASSLIST, passlist_text.encode())
    passlist_opt = f"--set brh_passlist={GUEST_PASSLIST} " if passlist_text else ""

    if WORKERS <= 1:
        _start_backend(env, LISTEN_PORT, mode, guest_state, guest_alerts, passlist_opt, None)
        time.sleep(6)
    else:
        back_ports = [LISTEN_PORT + 1 + i for i in range(WORKERS)]
        for i, port in enumerate(back_ports):
            _start_backend(env, port, mode, guest_state, f"{guest_alerts}.{i}",
                           passlist_opt, i)
        deadline = time.time() + 45
        while time.time() < deadline and not all(_port_up(env, p) for p in back_ports):
            time.sleep(1.0)
        csv = ",".join(str(p) for p in back_ports)
        _sh(env,
            f"cd {GUEST_HOME} && setsid nohup python3 {GUEST_FRONT} "
            f"--listen-host 127.0.0.1 --listen-port {LISTEN_PORT} --backends {csv} "
            f"> {GUEST_HOME}/front.log 2>&1 < /dev/null & echo pid=$!")
        # Wait up to 10s for the front to start listening (was 2s, too short under load).
        front_deadline = time.time() + 10
        while time.time() < front_deadline and not _listener_up(env):
            time.sleep(1.0)

    logger.info("http_proxy_bridge: backends restarted, listening=%s", _listener_up(env))


def push_state(env, host_state_path: str | os.PathLike, guest_state: str = GUEST_STATE) -> bool:
    try:
        data = Path(host_state_path).read_bytes()
    except FileNotFoundError:
        return False
    return _push_bytes(env, guest_state, data)


def pull_alerts(env, guest_alerts: str = GUEST_ALERTS) -> list[dict]:
    # Glob covers both the single-proxy file and the per-worker shards ({guest_alerts}.{i}).
    out = _sh(env, f"cat {guest_alerts}* 2>/dev/null || true")
    alerts = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                alerts.append(json.loads(line))
            except Exception:
                pass
    return alerts


# --------------------------------------------------------------------------- #
# host -> guest state syncer
# --------------------------------------------------------------------------- #
class StateSyncer(threading.Thread):
    """Polls the host branch_state.json and pushes it into the guest on change.

    Polling (not a hook callback) keeps the BRH interpreter hook decoupled from
    DesktopEnv. Branch transitions are infrequent (per `if`, not per request), so
    a sub-second poll is cheap. Always pushes once at start so the guest has a
    state before the agent's first request.
    """

    def __init__(self, env, host_state_path: str | os.PathLike,
                 guest_state: str = GUEST_STATE, interval: float = 0.25):
        super().__init__(daemon=True, name="brh-state-syncer")
        self.env = env
        self.host_state_path = str(host_state_path)
        self.guest_state = guest_state
        self.interval = interval
        self._stop = threading.Event()
        self._last_sig = None

    def _sig(self):
        try:
            st = os.stat(self.host_state_path)
            return (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            return None

    def run(self) -> None:
        logger.info("StateSyncer: watching %s -> guest %s", self.host_state_path, self.guest_state)
        while not self._stop.is_set():
            sig = self._sig()
            if sig is not None and sig != self._last_sig:
                if push_state(self.env, self.host_state_path, self.guest_state):
                    self._last_sig = sig
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


class AlertMirror(threading.Thread):
    """Mirrors the guest HTTP proxy alert log back to a host file for metrics.

    The guest is ephemeral per task, so alerts must be pulled host-side while it
    is alive. Alerts only grow, so we just re-fetch and overwrite the host copy.
    """

    def __init__(self, env, host_alerts_path: str | os.PathLike,
                 guest_alerts: str = GUEST_ALERTS, interval: float = 2.0):
        super().__init__(daemon=True, name="brh-alert-mirror")
        self.env = env
        self.host_alerts_path = str(host_alerts_path)
        self.guest_alerts = guest_alerts
        self.interval = interval
        self._stop = threading.Event()

    def _mirror_once(self) -> None:
        # Glob merges the per-worker alert shards ({guest_alerts}.{i}) when WORKERS>1.
        out = _sh(self.env, f"cat {self.guest_alerts}* 2>/dev/null || true")
        lines = [l for l in out.splitlines() if l.strip().startswith("{")]
        if lines:
            Path(self.host_alerts_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.host_alerts_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._mirror_once()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._mirror_once()  # final flush
        except Exception:
            pass
