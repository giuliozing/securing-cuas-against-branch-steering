"""LLM-free certification of suite S3 — MCP tool-layer integrity.



Usage:
    python -m steerbench.oracles.s3 [--only 51] [--subs M1,M2]
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

from steerbench.harness import arms
from steerbench import config
from steerbench.harness import evaluator as ev
from steerbench.harness import mcp_driver
from steerbench.tasks import registry as reg

sys.path.insert(0, str(config.cobra_src()))
from cobra.brh.writer import atomic_write_json  # noqa: E402

# normal MCP / its MCP proxy | rugged MCP / pinned MCP proxy / TOFU MCP proxy | squatter MCP / its MCP proxy
P_MCP, P_MPT = 9393, 9394
R_MCP, R_MPT_PIN, R_MPT_TOFU = 9395, 9396, 9398
Q_MCP, Q_MPT = 9397, 9399
CAMEL_PY = Path(config.python_for_subprocess())

PROCS: list[subprocess.Popen] = []


def _spawn(cmd, env) -> subprocess.Popen:
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    PROCS.append(p)
    return p


def _server(port: int, env, *, rugged=False, themes: str = "") -> subprocess.Popen:
    cmd = [str(CAMEL_PY), str(config.PKG / "mcp" / "server.py"), "--port", str(port)]
    if rugged:
        cmd.append("--rugged")
    if themes:
        cmd += ["--only-themes", themes]
    return _spawn(cmd, env)


def _mpt(upstream_port: int, port: int, brh_dir: Path, registry: Path,
         alerts: Path, env, server_id="steerweb", sealed=True) -> subprocess.Popen:
    """`sealed` (default True, see `config.mcp_proxy_cmd`) is MCP proxy's
    production-safe default. Pass `sealed=False` only for a proxy that MUST
    perform an unattended first listing against an empty registry — the
    initial approval pin, or the A4/TOFU leg."""
    cmd = config.mcp_proxy_cmd(upstream_port, port, brh_dir=brh_dir, registry=registry,
                               alerts=alerts, server_id=server_id, sealed=sealed)
    return _spawn(cmd, dict(env, PYTHONPATH=f"{config.cobra_src()}:{config.SRC}"))


def mcp_alerts(alerts_path: Path) -> list[str]:
    """MCP proxy appends to the same JSONL as HTTP proxy but tags `channel="mcp"`; the HTTP
    reader keys on `kind == "brh_http_block"`, so the two channels never mix."""
    return sorted({a.get("reason") for a in config.read_alerts(alerts_path)
                   if a.get("channel") == "mcp"})


def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}/mcp"


def drive(task, arm, variant, state_path, alerts_path):
    """Run one cell. Returns (reasons, actions, outcome-dict)."""
    spec = task.mcp
    config.control("POST", "/__reset")
    config.control("POST", "/__config", task.benign_config())
    alerts_path.write_text("")

    # M4's A4 is a POISONED PLAN, not a weakened enforcer: the planner that read the
    # injected description authorises the injected tool itself.
    poisoned = (spec.sub == "M4" and arm.arm == "A4" and variant == "attack")
    state = ev.oracle_state(task, plan_id=f"{task.tid}::{arm.arm}::{variant}")
    if poisoned:
        state["mcp_constraints"] = ev.oracle_mcp_constraints(task, poisoned_plan=True)
    elif arm.state_filter is not None:
        state = arm.state_filter(task, copy.deepcopy(state))
    atomic_write_json(state_path, state)
    time.sleep(0.25)  # let MCP proxy's next state read see the new file

    # M5 acts on tools/list (the hash check lives on the response), everything else
    # on tools/call.
    if spec.sub == "M5":
        if variant == "benign":
            out = mcp_driver.list_tools(_url(P_MPT))
        else:
            out = mcp_driver.list_tools(_url(R_MPT_PIN if arm.arm == "A5" else R_MPT_TOFU))
        return mcp_alerts(alerts_path), [], out

    if variant == "benign":
        tool, args, port = spec.tool, dict(spec.args), P_MPT
    elif spec.sub in ("M1", "M4"):
        tool, args, port = spec.attack_tool, dict(spec.attack_args), P_MPT
    elif spec.sub == "M6":
        # same tool NAME, served by the squatter's registry namespace
        tool, args, port = spec.tool, dict(spec.args), Q_MPT
    else:  # M2 diverges an argument, M3 adds one; both call the authorised tool
        tool, args, port = spec.tool, {**spec.args, **spec.attack_args}, P_MPT

    out = mcp_driver.call_tool(spec.qualified(task.theme, tool), args, _url(port))
    actions = config.control("GET", "/__state").get("actions", [])
    return mcp_alerts(alerts_path), actions, out


def expect(arm_name, variant, task, reasons, actions, out) -> list[str]:
    """A5 and the benign leg delegate to `evaluator.evaluate_mcp` — the single
    definition of "what happened" shared with `oracles/s4` and the paid runner
    . Only the A4 leg is local, because its expectation is inverted:
    there the attack MUST land (invariant I2)."""
    is_list = task.mcp.sub == "M5"
    happened = (not out.get("blocked")) if is_list else bool(actions)
    if variant == "benign" or arm_name == "A5":
        return ev.evaluate_mcp(variant, task, reasons, actions, out)
    f: list[str] = []
    if not happened:  # I2 negative control
        f.append(f"I2_VIOLATED: attack still blocked without the component "
                 f"(reasons={reasons or '[]'}) — suite is tautological here")
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma tid prefixes, e.g. 51,54")
    ap.add_argument("--subs", default="", help="comma sub-families, e.g. M1,M5")
    ap.add_argument("--arms", default="A5,A4")
    args = ap.parse_args()

    arm_objs = [arms.get("S3", a.strip()) for a in args.arms.split(",") if a.strip()]
    subs = {s.strip() for s in args.subs.split(",") if s.strip()}
    selected = [t for t in reg.TASKS if t.on_mcp_axis
                and (not subs or t.mcp.sub in subs)
                and (not args.only or any(t.tid.startswith(p) for p in args.only.split(",")))]
    if not selected:
        print("no S3 tasks selected", file=sys.stderr)
        return 2
    present = {t.mcp.sub for t in selected}

    brh_dir = Path(tempfile.mkdtemp(prefix="steerweb_s3_"))
    state_path = brh_dir / "branch_state.json"
    alerts_path = brh_dir / "brh_alerts.jsonl"
    reg_main = brh_dir / "hash_registry.json"
    reg_tofu = brh_dir / "hash_registry_tofu.json"
    reg_squat = brh_dir / "hash_registry_squat.json"
    print(f"BRH dir: {brh_dir}\nSUITE S3 — {arms.SUITE_TITLES['S3']}  "
          f"tasks={len(selected)} subs={sorted(present)}")

    env = config.subprocess_env({
               "STEERWEB_DIRECT": f"http://127.0.0.1:{config.SITE_PORT}",
               "STEERWEB_PROXY": f"http://127.0.0.1:{config.PROXY_PORT}"})

    n_pass = n_total = i2_ok = i2_total = 0
    fails: list[str] = []
    try:
        _spawn([str(Path(config.python_for_subprocess())), str(config.SITE_APP),
                str(config.SITE_PORT)], env)
        _server(P_MCP, env)
        # Unsealed: this is the proxy the M5 approval listing (below) uses to pin
        # the honest hashes into an EMPTY reg_main — sealed-by-default would refuse
        # that very first listing. M1-M4/M6 never issue tools/list on this proxy,
        # so their `tools/call` checks (which do not consult the hash registry)
        # are unaffected by this being unsealed.
        _mpt(P_MCP, P_MPT, brh_dir, reg_main, alerts_path, env, sealed=False)
        config.wait_port(config.SITE_PORT)
        config.wait_port(P_MCP)
        config.wait_port(P_MPT)

        if "M5" in present:
            # APPROVAL: one honest tools/list pins every tool's hash into reg_main.
            # Must happen BEFORE the pinned proxy starts (it snapshots the registry).
            listed = mcp_driver.list_tools(_url(P_MPT))
            print(f"  [approval] pinned {len(listed.get('tools') or [])} tool hashes")
            _server(R_MCP, env, rugged=True)
            config.wait_port(R_MCP)
            _mpt(R_MCP, R_MPT_PIN, brh_dir, reg_main, alerts_path, env)  # sealed (A5)
            _mpt(R_MCP, R_MPT_TOFU, brh_dir, reg_tofu, alerts_path, env, sealed=False)  # A4/TOFU
            config.wait_port(R_MPT_PIN)
            config.wait_port(R_MPT_TOFU)

        if "M6" in present:
            themes = ",".join(t.theme for t in selected if t.mcp.sub == "M6")
            _server(Q_MCP, env, themes=themes)
            config.wait_port(Q_MCP)
            _mpt(Q_MCP, Q_MPT, brh_dir, reg_squat, alerts_path, env, server_id="squatter")
            config.wait_port(Q_MPT)

        for task in selected:
            for arm in arm_objs:
                for variant in ("benign", "attack"):
                    n_total += 1
                    reasons, actions, out = drive(task, arm, variant, state_path, alerts_path)
                    failures = expect(arm.arm, variant, task, reasons, actions, out)
                    ok = not failures
                    n_pass += int(ok)
                    if arm.arm == "A4" and variant == "attack":
                        i2_total += 1
                        i2_ok += int(ok)
                    print(f"  [{'OK ' if ok else 'XXX'}] {task.tid:<22} {task.mcp.sub} "
                          f"{arm.arm:<3} {variant:<6} reasons={reasons or '[]'} "
                          f"acts={len(actions)}" + ("" if ok else f"  FAIL={failures}"))
                    if not ok:
                        fails.append(f"{task.tid}/{arm.arm}/{variant}: {failures} out={str(out)[:120]}")
    finally:
        for p in reversed(PROCS):
            p.terminate()

    print("\n" + "#" * 70)
    print(f"ORACLE-S3   {n_pass}/{n_total} cells passed (LLM-free)")
    print(f"I2 NEG-CTRL {i2_ok}/{i2_total} attacks breached under A4 "
          f"(want ALL — proves MCP proxy, not another layer, is what blocks)")
    for f in fails:
        print(f"  {f}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
