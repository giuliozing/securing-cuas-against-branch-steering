"""Every path, port and interpreter the harness needs — resolved, never hard-coded.

Three rules this module exists to enforce:

1. **Nothing outside the package is a constant.** The defence stack (the HTTP proxy mitm
   addon, the COBRA source tree, `mitmdump`) is *discovered* — environment variable
   first, then a small list of conventional locations — and its absence is a clear
   error at the point of use, not an ImportError at module load. The benchmark itself
   (site, tasks, judge, adapters) never touches any of it.

2. **Ports are allocated, not assumed.** A collision is silent-and-wrong rather than
   loud. `alloc_port()` asks the OS for a free one; the defaults survive only as
   *preferences*, so an existing invocation with `--site-port 8700` behaves exactly as
   it did.

3. **Module-global mutability is preserved on purpose.** Sharded runners reassign
   `SITE_PORT` after import, and `control()` resolves it at call time — see its
   docstring. Freezing it into a default argument would break that.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import time
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------
# Package layout (always known)
# --------------------------------------------------------------------------

PKG = Path(__file__).resolve().parent               # …/src/steerbench
SRC = PKG.parent                                    # …/src
PROJECT = SRC.parent                                # the repo/package root
SITE_APP = PKG / "site" / "app.py"
FORWARD_ADDON = PKG / "harness" / "forward.py"
RESULTS = Path(os.environ.get("STEERBENCH_RESULTS", PROJECT / "results"))

# --------------------------------------------------------------------------
# Ports — preferences, not constants
# --------------------------------------------------------------------------

SITE_PORT = int(os.environ.get("STEERBENCH_SITE_PORT", 8700))
PROXY_PORT = int(os.environ.get("STEERBENCH_PROXY_PORT", 8781))
MCP_PORT = int(os.environ.get("STEERBENCH_MCP_PORT", 9393))


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def alloc_port(prefer: int | None = None) -> int:
    """A free loopback port: `prefer` if it is actually free, else one the OS picks.

    The preference keeps every documented invocation reproducible while removing the
    class of failure where a second run silently talks to the first run's site."""
    if prefer is not None and port_free(prefer):
        return prefer
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------
# The defence stack — discovered, and only needed by the `brh` adapter
# --------------------------------------------------------------------------

_CANDIDATE_ROOTS = (
    # The submission layout this benchmark ships in: steer-bench/ and cobra/ as
    # sibling directories under one checkout.
    PROJECT.parent,
    PROJECT.parent.parent,
)


class MissingDefenceStack(RuntimeError):
    """Raised at point of use, never at import.

    The benchmark, the open track and the substrate gate all run without any of this;
    only the defended track needs it. Failing here with a named environment variable is
    the difference between "install the extra" and "the package is broken"."""


def _first_existing(*paths: Path | None) -> Path | None:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def cobra_src() -> Path:
    """The COBRA source tree (`cobra.brh`, `cobra.mcp_proxy`, `cobra.interpreter`)."""
    found = _first_existing(
        Path(os.environ["STEERBENCH_COBRA_SRC"]) if os.environ.get("STEERBENCH_COBRA_SRC") else None,
        *[r / "cobra" / "src" for r in _CANDIDATE_ROOTS],
    )
    if found is None:
        raise MissingDefenceStack(
            "COBRA source not found. The defended track needs it; set STEERBENCH_COBRA_SRC "
            "to the directory containing `cobra/`. The substrate gate and the open track "
            "do not require it.")
    return found


def enforcer_addon() -> Path:
    """The mitmproxy addon that enforces `branch_state.json` on the wire."""
    found = _first_existing(
        Path(os.environ["STEERBENCH_ENFORCER_ADDON"]) if os.environ.get("STEERBENCH_ENFORCER_ADDON") else None,
        *[r / "cobra" / "src" / "cobra" / "http_proxy" / "mitm_addon.py"
          for r in _CANDIDATE_ROOTS],
    )
    if found is None:
        raise MissingDefenceStack(
            "HTTP proxy enforcer addon not found. Set STEERBENCH_ENFORCER_ADDON to "
            "mitm_addon.py, or run the open track, which needs no enforcer.")
    return found


def mitmdump() -> Path:
    """`mitmdump` as an EXTERNAL BINARY, deliberately.

    mitmdump is never imported — only spawned — so it does not belong in the
    package's dependency graph at all. PATH first, then a conventional venv
    location next to the COBRA checkout."""
    env = os.environ.get("STEERBENCH_MITMDUMP")
    if env:
        return Path(env)
    on_path = shutil.which("mitmdump")
    if on_path:
        return Path(on_path)
    found = _first_existing(*[r / "cobra" / ".venv" / "bin" / "mitmdump"
                              for r in _CANDIDATE_ROOTS])
    if found is None:
        raise MissingDefenceStack(
            "mitmdump not found on PATH. The defended track proxies the guarded wire "
            "through it; install mitmproxy or set STEERBENCH_MITMDUMP.")
    return found


def python_for_subprocess() -> str:
    """The interpreter used to spawn the site and the MCP servers.

    One interpreter for all of the benchmark's own Python (the site needs only flask +
    jinja2 + the stdlib-only task registry). `sys.executable` is right in every normal case; the override exists for
    the case where the caller's interpreter cannot import the package."""
    return os.environ.get("STEERBENCH_PYTHON") or __import__("sys").executable


def mcp_proxy_cmd(upstream_port: int, port: int, *, brh_dir, registry, alerts,
                   server_id: str = "steerweb", sealed: bool = True) -> list[str]:
    """The `python -m cobra.mcp_proxy` argv every caller that fronts an MCP
    deployment with the enforcer should build from — one place to translate
    `sealed` into the CLI's actual flag, so a future CLI rename cannot drift
    out of step with only some of its callers the way `--sealed` once did.

    `sealed=True` (the default, and MCP proxy's own production-safe default)
    passes no flag: an unregistered tool is rejected (`mpt_unapproved`).
    `sealed=False` passes `--benchmark-allow-new-tools` — trust-on-first-use,
    for a proxy that must complete an unattended first `tools/list` against
    an empty registry (an approval-pinning step, or an explicit A4/TOFU arm).
    Never make `sealed=False` a caller's own default."""
    cmd = [python_for_subprocess(), "-m", "cobra.mcp_proxy",
           "--http-upstream", f"http://127.0.0.1:{upstream_port}/mcp",
           "--http-port", str(port), "--brh-dir", str(brh_dir),
           "--registry", str(registry), "--alerts", str(alerts),
           "--server-id", server_id]
    if not sealed:
        cmd.append("--benchmark-allow-new-tools")
    return cmd


# --------------------------------------------------------------------------
# The harness token
# --------------------------------------------------------------------------
#
# `GET /<theme>/quote` and `/<theme>/__item/<i>/quote` return the REAL wire body —
# the answer. They exist because the SITE uses them: the guarded action is issued
# server-to-server, which is what makes "perceived ≠ real" a fact no agent can see. But
# they sit on the same base URL an adapter is handed as `TaskView.page_url`, so without
# a gate an adapter (or a browser navigating one URL further) could read the answer and
# score ~0% ASR with full utility and no defence at all.
#
# The token is what separates "the harness asking" from "the agent asking". It is
# generated per stack, travels to every child through `subprocess_env`, and is sent by
# `harness/driver.py` as a header. A browser has no header to send, and an adapter that
# was handed only a URL has nothing to send either.
#
# **What it is not.** It is not a sandbox against a hostile in-process adapter: an
# adapter is Python running in the runner's own process and can read anything, including
# this. Nothing short of process isolation changes that, and claiming otherwise would be
# worse than the hole. What it removes is the shortcut that could be taken WITHOUT
# realising it is cheating — a plain GET on the page's own origin. Taking it now requires
# deliberately extracting a secret, which is a different act and a reportable one.

HARNESS_TOKEN_ENV = "STEERBENCH_HARNESS_TOKEN"
HARNESS_HEADER = "X-Steer-Harness"


def new_harness_token() -> str:
    return secrets.token_urlsafe(24)


def harness_token() -> str:
    """The active token, resolved at CALL time.

    Never cached: `stack.build()` sets it after this module is imported, and a value
    frozen at import would be an empty string forever — i.e. a driver that cannot reach
    the endpoint it is the only legitimate caller of."""
    return os.environ.get(HARNESS_TOKEN_ENV, "")


def harness_headers() -> dict:
    tok = harness_token()
    return {HARNESS_HEADER: tok} if tok else {}


def subprocess_env(extra: dict | None = None) -> dict:
    """Environment for a spawned child that must import `steerbench`.

    Prepends the source root to PYTHONPATH so a plain checkout works without an
    install; an installed package simply finds itself first and this is a no-op."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else str(SRC)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def wait_port(port: int, timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"port {port} did not come up")


def control(method: str, path: str, payload: dict | None = None,
            port: int | None = None) -> dict:
    """Hit the site's control plane directly (never through the proxy).

    `port` defaults to the CURRENT module-global SITE_PORT resolved at call time,
    not a value bound when this function was defined. Sharded runners reassign
    `config.SITE_PORT` after import (e.g. --site-port 8710); a `port=SITE_PORT`
    default would freeze the import-time value and silently send every control
    call to the wrong site."""
    if port is None:
        port = SITE_PORT
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def read_alerts(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
