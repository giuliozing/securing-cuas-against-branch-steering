"""Command-construction tests for every caller that fronts an MCP deployment with
`cobra.mcp_proxy` — no site, no proxy, no real subprocess.

MCP proxy's CLI moved from `--sealed` (opt-in rejection) to sealed-by-default with
an explicit `--benchmark-allow-new-tools` opt-in for trust-on-first-use. That
migration was missed in three places at once — `harness/run.py`,
`oracles/s3.py`, `oracles/s6.py` — each of which used to build the argv by hand:
`--sealed` stopped being a flag the CLI accepts (`unrecognized arguments`), and a
caller that used to mean "unsealed" by simply omitting `--sealed` silently became
sealed instead. Every caller now builds its argv through the single
`config.mcp_proxy_cmd`, so these tests both pin its own translation and check
every caller passes it the sealed/unsealed intent it actually needs.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from steerbench import config


# --- config.mcp_proxy_cmd: the one place `sealed` becomes a flag ------------


def test_sealed_default_emits_no_flag():
    cmd = config.mcp_proxy_cmd(1, 2, brh_dir="/tmp/b", registry="/tmp/r.json",
                               alerts="/tmp/a.jsonl")
    assert "--benchmark-allow-new-tools" not in cmd
    assert "--sealed" not in cmd


def test_unsealed_emits_the_explicit_benchmark_flag():
    cmd = config.mcp_proxy_cmd(1, 2, brh_dir="/tmp/b", registry="/tmp/r.json",
                               alerts="/tmp/a.jsonl", sealed=False)
    assert "--benchmark-allow-new-tools" in cmd
    assert "--sealed" not in cmd


@pytest.mark.parametrize("sealed", [True, False])
def test_no_command_ever_contains_the_old_flag(sealed):
    """`--sealed` is not accepted by the CLI any more; if it ever leaks back in,
    argparse fails every caller with `unrecognized arguments`."""
    cmd = config.mcp_proxy_cmd(1, 2, brh_dir="/tmp/b", registry="/tmp/r.json",
                               alerts="/tmp/a.jsonl", sealed=sealed)
    assert "--sealed" not in cmd


def test_module_source_never_hardcodes_the_old_flag():
    """Regression guard for the exact bug class: a caller building its own argv by
    hand, bypassing `mcp_proxy_cmd`, and drifting the next time the CLI changes."""
    import steerbench.harness.run as run_mod
    import steerbench.harness.stack as stack_mod
    import steerbench.oracles.s3 as s3_mod
    import steerbench.oracles.s4 as s4_mod
    import steerbench.oracles.s6 as s6_mod
    for mod in (run_mod, stack_mod, s3_mod, s4_mod, s6_mod):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "--sealed" not in src, f"{mod.__name__} still hardcodes --sealed"


# --- harness/run.py: launch_mcp_stack ---------------------------------------


class _FakePopen:
    """Records the argv it was built with; never actually spawns anything."""

    def __init__(self, cmd, **kw):
        self.args = list(cmd)

    def poll(self):
        return None


def _mcp_task(sub, theme="t"):
    mcp = types.SimpleNamespace(sub=sub)
    return types.SimpleNamespace(on_mcp_axis=True, mcp=mcp, theme=theme)


def _patch_launch_environment(monkeypatch):
    from steerbench.harness import run as run_mod
    monkeypatch.setattr(run_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(config, "wait_port", lambda *a, **kw: None)
    return run_mod


def test_launch_mcp_stack_main_proxy_is_unsealed(monkeypatch, tmp_path):
    """The main proxy performs the very first tools/list against an empty
    registry — it must stay unsealed or nothing downstream ever gets pinned."""
    run_mod = _patch_launch_environment(monkeypatch)
    ports = run_mod._mcp_ports(20000)
    procs = run_mod.launch_mcp_stack(tmp_path, tmp_path / "alerts.jsonl", [], ports,
                                     env={})
    main_cmd = next(p.args for p in procs if str(ports["mpt"]) in p.args)
    assert "--benchmark-allow-new-tools" in main_cmd
    assert "--sealed" not in main_cmd


def test_launch_mcp_stack_m5_pinned_is_sealed_tofu_is_not(monkeypatch, tmp_path):
    run_mod = _patch_launch_environment(monkeypatch)
    monkeypatch.setattr("steerbench.harness.mcp_driver.list_tools", lambda url: {})
    ports = run_mod._mcp_ports(20100)
    tasks = [_mcp_task("M5")]
    procs = run_mod.launch_mcp_stack(tmp_path, tmp_path / "alerts.jsonl", tasks, ports,
                                     env={})
    pinned_cmd = next(p.args for p in procs if str(ports["mpt_pinned"]) in p.args)
    tofu_cmd = next(p.args for p in procs if str(ports["mpt_tofu"]) in p.args)
    assert "--benchmark-allow-new-tools" not in pinned_cmd  # A5: sealed
    assert "--benchmark-allow-new-tools" in tofu_cmd        # A4: explicit TOFU
    for cmd in (pinned_cmd, tofu_cmd):
        assert "--sealed" not in cmd


def test_launch_mcp_stack_m6_squatter_command_is_well_formed(monkeypatch, tmp_path):
    """M6's defence (`allowed_tool_servers`) is enforced on `tools/call`, never on
    the tool-hash registry, so the squatter proxy's sealed state is not what the
    suite measures — this only pins that the command still builds and never
    regresses to the removed flag."""
    run_mod = _patch_launch_environment(monkeypatch)
    ports = run_mod._mcp_ports(20200)
    tasks = [_mcp_task("M6", theme="squat_theme")]
    procs = run_mod.launch_mcp_stack(tmp_path, tmp_path / "alerts.jsonl", tasks, ports,
                                     env={})
    squat_cmd = next(p.args for p in procs if str(ports["mpt_squat"]) in p.args)
    assert "--sealed" not in squat_cmd


# --- oracles/s3.py: _mpt -----------------------------------------------------


def test_s3_mpt_defaults_sealed_and_opts_out_explicitly(monkeypatch, tmp_path):
    import steerbench.oracles.s3 as s3_mod
    captured = {}

    def fake_spawn(cmd, env):
        captured["cmd"] = cmd
        return _FakePopen(cmd)

    monkeypatch.setattr(s3_mod, "_spawn", fake_spawn)

    s3_mod._mpt(1, 2, tmp_path, tmp_path / "r.json", tmp_path / "a.jsonl", env={})
    assert "--benchmark-allow-new-tools" not in captured["cmd"]

    s3_mod._mpt(1, 3, tmp_path, tmp_path / "r2.json", tmp_path / "a.jsonl", env={},
               sealed=False)
    assert "--benchmark-allow-new-tools" in captured["cmd"]
