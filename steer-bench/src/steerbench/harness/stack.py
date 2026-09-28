"""The one place a STEER-Bench process stack is brought up.

"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import subprocess
import tempfile
from pathlib import Path

from steerbench import config


@dataclasses.dataclass
class Stack:
    """A running benchmark stack and the artefact paths its enforcers write."""

    brh_dir: Path
    site_port: int
    proxy_port: int | None
    procs: list[subprocess.Popen] = dataclasses.field(default_factory=list)

    @property
    def state_path(self) -> Path:
        return self.brh_dir / "branch_state.json"

    @property
    def alerts_path(self) -> Path:
        return self.brh_dir / "brh_alerts.jsonl"

    @property
    def constraints_path(self) -> Path:
        return self.brh_dir / "plan_constraints.json"

    @property
    def direct(self) -> str:
        """Base URL of the site, unproxied — the perception channel."""
        return f"http://127.0.0.1:{self.site_port}"

    @property
    def proxy(self) -> str:
        """Base URL of the enforcing proxy — the guarded action's route."""
        if self.proxy_port is None:
            raise RuntimeError("this stack was built without an enforcing proxy")
        return f"http://127.0.0.1:{self.proxy_port}"

    def env(self, extra: dict | None = None) -> dict:
        """Child environment: package importable + the two channel addresses every
        tool surface reads (`steerbench.surfaces.wire`).

        A proxy-less stack sets `STEERWEB_PROXY` to the **empty string** rather than
        leaving it out, and the difference is the whole open track. `site/app.py` reads
        that variable with a documented default (`:8781`, so the site is usable
        standalone), so *absent* means "use the default proxy" — which on this track is a
        `mitmdump` that was never started. Every guarded action would then die with
        `Connection refused` inside the site and the cell would score NOTHING: an agent
        that acted correctly would be indistinguishable from one that refused to act.
        Empty is the one value that says "explicitly none", and it is set HERE
        rather than in the runner so it holds for any caller of `build(proxy=False)`.
        """
        base = {"STEERWEB_DIRECT": self.direct,
                "STEERWEB_PROXY": self.proxy if self.proxy_port is not None else ""}
        base.update(extra or {})
        return config.subprocess_env(base)

    def spawn(self, cmd: list[str], env: dict | None = None, *,
              quiet: bool = True) -> subprocess.Popen:
        kw = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
        p = subprocess.Popen([str(c) for c in cmd], env=env or self.env(), **kw)
        self.procs.append(p)
        return p

    def close(self) -> None:
        for p in reversed(self.procs):
            if p.poll() is not None:
                continue
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                with contextlib.suppress(Exception):
                    p.wait(timeout=5)
        self.procs.clear()


# ---------------------------------------------------------------------------
# Bring-up
# ---------------------------------------------------------------------------


def start_site(stack: Stack) -> subprocess.Popen:
    p = stack.spawn([config.python_for_subprocess(), config.SITE_APP, stack.site_port])
    config.wait_port(stack.site_port)
    return p


def start_proxy(stack: Stack, *, mode: str = "enforce") -> subprocess.Popen:
    """`mitmdump` running the UNCHANGED HTTP proxy enforcer plus the host remap.

    Order matters: enforcer first (see the module docstring)."""
    p = stack.spawn([
        config.mitmdump(), "-q", "--listen-port", stack.proxy_port,
        "-s", config.enforcer_addon(), "-s", config.FORWARD_ADDON,
        "--set", f"brh_state={stack.state_path}",
        "--set", f"brh_alerts={stack.alerts_path}",
        "--set", f"brh_mode={mode}",
        "--set", f"site_port={stack.site_port}",
    ], quiet=False)
    config.wait_port(stack.proxy_port)
    return p


def start_mcp_server(stack: Stack, port: int, *, rugged: bool = False,
                     themes: str = "") -> subprocess.Popen:
    """One FastMCP deployment. `rugged` serves S3/M5's post-approval descriptions;
    `themes` narrows the surface, which is how S3/M6 stands up a squatter."""
    cmd = [config.python_for_subprocess(), config.PKG / "mcp" / "server.py",
           "--port", port]
    if rugged:
        cmd.append("--rugged")
    if themes:
        cmd += ["--only-themes", themes]
    p = stack.spawn(cmd)
    config.wait_port(port)
    return p


def start_mpt(stack: Stack, upstream_port: int, port: int, *, registry: Path,
              server_id: str = "steerweb") -> subprocess.Popen:
    """The MCP enforcer (`cobra.mcp_proxy`) in front of one deployment.

    Always sealed (no `--benchmark-allow-new-tools`): every caller drives
    `tools/call` only, never `tools/list`, so the tool-hash registry this flag
    gates is never consulted here. Part of the defended track only:
    `cobra_src()` raises `MissingDefenceStack` with a named environment
    variable if the COBRA tree is absent."""
    env = stack.env({"PYTHONPATH": f"{config.cobra_src()}:{config.SRC}"})
    cmd = config.mcp_proxy_cmd(upstream_port, port, brh_dir=stack.brh_dir,
                               registry=registry, alerts=stack.alerts_path,
                               server_id=server_id)
    p = stack.spawn(cmd, env)
    config.wait_port(port)
    return p


def build(*, site_port: int | None = None, proxy_port: int | None = None,
          brh_dir: Path | None = None, mode: str = "enforce",
          proxy: bool = True, prefix: str = "steerbench_") -> Stack:
    """Bring up the site (+ the enforcing proxy) and return the running stack.

    `proxy=False` builds a site-only stack — that is the OPEN track, where no enforcer
    exists and the verdict comes from `/__state` alone. Keeping it a flag rather than a
    second function is deliberate: the two tracks must share one bring-up, or an
    open-track result would be measuring a different environment.

    The caller owns teardown (`stack.close()`); `launch()` is the context-managed form.
    Both exist because the oracles are structured as one long `try:` with a report
    printed in `finally:`, and wrapping that in a `with` would reindent the part of
    each file that actually matters.
    """
    # One token per stack, before anything is spawned: `subprocess_env` copies the
    # parent environment, so setting it here is what carries it to the site, the MCP
    # server and MCP proxy. `setdefault`, so a caller that pinned one (a sharded run sharing
    # a site) keeps it. Without this the `/quote` endpoints stay open and an adapter
    # could read the answer off the base URL it was handed (`config.HARNESS_TOKEN_ENV`).
    os.environ.setdefault(config.HARNESS_TOKEN_ENV, config.new_harness_token())

    brh_dir = Path(brh_dir or tempfile.mkdtemp(prefix=prefix))
    stack = Stack(
        brh_dir=brh_dir,
        site_port=config.alloc_port(site_port if site_port is not None else config.SITE_PORT),
        proxy_port=(config.alloc_port(proxy_port if proxy_port is not None else config.PROXY_PORT)
                    if proxy else None),
    )
    # The module globals are what `config.control()` and the tool surfaces resolve at
    # call time, so a stack that allocated a different port must publish it.
    config.SITE_PORT = stack.site_port
    if stack.proxy_port is not None:
        config.PROXY_PORT = stack.proxy_port
    try:
        start_site(stack)
        if proxy:
            start_proxy(stack, mode=mode)
    except Exception:
        stack.close()
        raise
    return stack


@contextlib.contextmanager
def launch(**kw):
    """`build()` with guaranteed teardown on any exit path, including SystemExit."""
    stack = build(**kw)
    try:
        yield stack
    finally:
        stack.close()
