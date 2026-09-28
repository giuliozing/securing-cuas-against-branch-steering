"""Behaviour tests for `harness/stack.py` — the one launcher every check uses.

One bring-up module removes drift and concentrates risk: every oracle and the paid
runner fail together if this is wrong. So it gets tests for the three properties every
bring-up needs.

Most of this file is **substrate**: a site-only stack needs no enforcer, which is the
same bring-up the open track will use. That it can run at all, on a machine with no
defence stack installed, is the point of the `proxy=False` flag.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import urllib.request

import pytest

from steerbench import config
from steerbench.harness import stack


# --- port allocation: the silent-and-wrong failure mode ---------------------


def test_alloc_port_prefers_the_documented_default():
    """Existing invocations must behave exactly as they did: `--site-port 8700` still
    means 8700 whenever 8700 is actually free."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert config.alloc_port(free) == free


def test_alloc_port_falls_back_when_the_preference_is_taken():
    """A port collision does not raise — the runner POSTs /__config to somebody else's
    site while its own perceptions 404 — so ports are allocated, not assumed."""
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        got = config.alloc_port(taken)
        assert got != taken
        assert config.port_free(got)


# --- children must be able to import the package ---------------------------


def test_subprocess_env_makes_the_package_importable():
    """The site and the MCP server are executed as FILES (mitmproxy-style addons and
    FastMCP's `__main__` want it that way), so they cannot inherit the parent's
    imports."""
    env = config.subprocess_env()
    assert str(config.SRC) in env["PYTHONPATH"].split(os.pathsep)
    r = subprocess.run([sys.executable, "-c", "import steerbench; print(steerbench.__version__)"],
                       env=env, capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "3.0.0.dev0"


def test_subprocess_env_preserves_an_existing_pythonpath():
    env = config.subprocess_env()
    saved = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = "/some/other/root"
    try:
        env = config.subprocess_env()
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert parts[0] == str(config.SRC) and "/some/other/root" in parts
    finally:
        if saved is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved


# --- the site-only stack: what the OPEN track will use ----------------------


@pytest.mark.substrate
@pytest.mark.slow
def test_site_only_stack_serves_a_task_and_tears_down():
    """No enforcer, no COBRA, no mitmdump — exactly what a third party has. This also
    pins the judge's evidence surface: `/__state` must exist and start empty."""
    from steerbench.tasks import registry as reg
    task = next(t for t in reg.TASKS if t.on_http_axis)

    st = stack.build(proxy=False, prefix="steerbench_test_")
    try:
        assert st.proxy_port is None
        config.control("POST", "/__config", task.benign_config(), port=st.site_port)
        page = urllib.request.urlopen(f"{st.direct}/{task.theme}/", timeout=10)
        assert page.status == 200
        assert config.control("GET", "/__state", port=st.site_port) == {"actions": []}
        procs = list(st.procs)
        assert procs and all(p.poll() is None for p in procs)
    finally:
        st.close()

    assert all(p.poll() is not None for p in procs), "close() left a process running"
    assert st.procs == []


@pytest.mark.substrate
def test_a_stack_without_a_proxy_refuses_to_name_one():
    """`proxy` is the guarded action's route. Returning a plausible-looking URL for a
    stack that has no enforcer would let an open-track run silently believe it was
    enforced."""
    st = stack.Stack(brh_dir=config.PROJECT, site_port=1, proxy_port=None)
    with pytest.raises(RuntimeError):
        _ = st.proxy


@pytest.mark.substrate
def test_stack_env_carries_both_channel_addresses():
    """`surfaces/wire.py` reads STEERWEB_DIRECT for perception and STEERWEB_PROXY for
    the guarded action; a stack that published only one would send the guarded action
    to the perception channel, i.e. round the enforcer."""
    st = stack.Stack(brh_dir=config.PROJECT, site_port=8700, proxy_port=8781)
    env = st.env()
    assert env["STEERWEB_DIRECT"] == "http://127.0.0.1:8700"
    assert env["STEERWEB_PROXY"] == "http://127.0.0.1:8781"


@pytest.mark.substrate
def test_a_proxyless_stack_says_so_explicitly_rather_than_by_omission():
    """The open track's defining property, as an environment variable.

    `site/app.py` reads STEERWEB_PROXY with a documented default (`:8781`, so the site is
    usable standalone), so *omitting* it does not mean "no proxy" — it means "the default
    proxy", which on a `proxy=False` stack is never started. Every guarded action would
    then die with `Connection refused` INSIDE the site process and the cell would score
    NOTHING, which is exactly what an agent that declined to act scores. Hence the empty
    string, and hence this test."""
    off = stack.Stack(brh_dir=config.PROJECT, site_port=8700, proxy_port=None)
    assert off.env()["STEERWEB_PROXY"] == ""


@pytest.mark.substrate
@pytest.mark.slow
def test_a_grounded_click_records_an_action_with_no_enforcer_present():
    """The open track, end to end, with no model: click the bridge, read the evidence.

    This is the test that would have caught the defect above, and it is written at the
    level the defect lived at — the guarded action is issued by the SITE process, so
    nothing the runner or the adapter believes about its own environment is evidence
    that the action can actually leave."""
    from steerbench.tasks import registry as reg
    task = next(t for t in reg.TASKS if t.on_http_axis)

    port = config.alloc_port(None)
    os.environ["STEERBENCH_REMAP"] = f"http://127.0.0.1:{port}"
    st = stack.build(proxy=False, site_port=port, prefix="steerbench_test_open_")
    try:
        config.control("POST", "/__config", task.benign_config(), port=st.site_port)
        req = urllib.request.Request(f"{st.direct}/{task.theme}/confirm",
                                     data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            assert r.status == 200, "the GUI bridge itself failed"
        actions = config.control("GET", "/__state", port=st.site_port)["actions"]
        assert len(actions) == 1, f"no action recorded: {actions}"
        assert actions[0]["host"] == task.domain.lower()
    finally:
        st.close()
        os.environ.pop("STEERBENCH_REMAP", None)


# --- the defence stack is optional, and says so ----------------------------


@pytest.mark.substrate
def test_missing_defence_stack_names_the_variable_to_set(monkeypatch):
    """The whole release rests on this: a missing defence must be a clear message at
    point of use, never an ImportError at module load, or `pip install steer-bench`
    looks broken to everyone who is not us."""
    monkeypatch.setenv("STEERBENCH_COBRA_SRC", "/nonexistent/cobra/src")
    monkeypatch.setattr(config, "_CANDIDATE_ROOTS", ())
    with pytest.raises(config.MissingDefenceStack) as e:
        config.cobra_src()
    assert "STEERBENCH_COBRA_SRC" in str(e.value)


@pytest.mark.substrate
def test_importing_the_benchmark_never_needs_a_defence():
    """The import graph is the guarantee, so it is asserted rather than described: the
    task registry, the site and the judge must load in a subprocess where every
    defence-stack discovery variable points at nothing."""
    env = config.subprocess_env({
        "STEERBENCH_COBRA_SRC": "/nonexistent",
        "STEERBENCH_ENFORCER_ADDON": "/nonexistent",
        "STEERBENCH_MITMDUMP": "/nonexistent",
    })
    code = ("import steerbench.tasks.registry as r, steerbench.site.app, "
            "steerbench.harness.evaluator as e; "
            "print(len(r.TASKS), e.outcome({'variant':'benign'}))")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip() == "101 OK"
