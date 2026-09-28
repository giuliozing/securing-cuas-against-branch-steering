"""LLM-free certification of suite S6 — manifest trust.

Usage:
    python -m steerbench.oracles.s6 [--subs T1,T2a] [--only 66]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

from steerbench.harness import arms
from steerbench import config
from steerbench.harness import driver
from steerbench.harness import evaluator as ev
from steerbench.harness import mcp_driver
from steerbench.tasks import registry as reg
from steerbench.harness import trust_model

sys.path.insert(0, str(config.cobra_src()))
from cobra.brh.sitemap_trust import (ApprovalMode, approve_sitemap,  # noqa: E402
                                     gate_sitemaps, manifest_from_gated)
from cobra.brh.validator import sanitize_sitemap  # noqa: E402
from cobra.brh.writer import atomic_write_json  # noqa: E402

MCP_PORT, MPT_SEALED, MPT_TOFU = 9393, 9396, 9398
CAMEL_PY = Path(config.python_for_subprocess())
PROCS: list[subprocess.Popen] = []


def _spawn(cmd, env):
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    PROCS.append(p)
    return p


def fetch_sitemap(theme: str) -> list[dict]:
    """Untrusted input, over the wire, exactly as `writer._discover_raw_sitemaps` would."""
    return driver._get(f"http://127.0.0.1:{config.SITE_PORT}/{theme}/sitemap.json", proxy=None)


def manifest_for(task, raw: list[dict], arm_name: str, registry_path: Path):
    """Produce the planner-visible manifest under the arm's trust posture.

    A5 runs the production gate. A4 skips it entirely — that is the ablation: not a
    weaker gate, but *no gate*, with the free text exposed as a trusted manifest's
    descriptions are meant to be."""
    if arm_name == "A5":
        admitted, _pending = gate_sitemaps({task.domain: raw}, mode=ApprovalMode.AUTO,
                                           registry_path=str(registry_path))
        return manifest_from_gated(admitted)
    return sanitize_sitemap(raw, include_descriptions=True)


def drive(task, arm, variant, state_path, alerts_path, brh_dir):
    spec = task.trust
    cfg = task.benign_config() if variant == "benign" else task.attack_config()
    config.control("POST", "/__reset")
    config.control("POST", "/__config", cfg)
    alerts_path.write_text("")

    # Registry posture, rebuilt per cell so cells stay independent.
    registry_path = brh_dir / f"sitemap_registry_{task.tid}_{arm.arm}_{variant}.json"
    registry_path.write_text("{}")
    if spec.preapproved:
        # The realistic deployment posture: a human vetted and pinned this site's
        # sitemap before the run. Pinning the CLEAN one is what makes a later change
        # detectable at all.
        approve_sitemap(task.domain, list(spec.clean), str(registry_path), by="human")

    raw = fetch_sitemap(task.theme)
    manifest = manifest_for(task, raw, arm.arm, registry_path)
    domains, endpoints = trust_model.authorised_surface(task, manifest)

    state = ev.oracle_state(task, plan_id=f"{task.tid}::{arm.arm}::{variant}")
    state["http_constraints"] = {"allowed_domains": domains, "allowed_endpoints": endpoints}
    atomic_write_json(state_path, state)

    # T4 is a false-positive guard: both variants drive the BENIGN action, because the
    # question is whether a changed/unknown sitemap stops honest work.
    target = (f"http://{task.domain}{task.action_path}"
              if (variant == "benign" or spec.sub == "T4") else spec.attack_url)
    status, _ = driver._post(target, {"ref": "s6"},
                             proxy=f"http://127.0.0.1:{config.PROXY_PORT}")

    reasons = sorted({a.get("reason") for a in config.read_alerts(alerts_path)
                      if a.get("kind") == "brh_http_block"})
    actions = config.control("GET", "/__state").get("actions", [])
    return reasons, actions, {"status": status, "target": target,
                              "manifest": len(manifest), "domains": domains,
                              # the planner's menu under this arm's gate — what makes a
                              # `latent` cell non-vacuous (evaluator.evaluate_trust)
                              "menu": trust_model.menu_paths(manifest)}


def drive_mcp(task, arm, variant, alerts_path):
    """T3 — the MCP manifest analogue. A narrow surface was approved at seal time; the
    server then exposes the full one. A5 runs sealed (unregistered tool =>
    `mpt_unapproved`, listing blocked); A4 opts into trust-on-first-use and accepts it."""
    alerts_path.write_text("")
    if variant == "benign":
        # The honest case: the server still serves the surface that was approved, so
        # the hashes match and the listing goes through. Listing the WIDENED server
        # here would make the benign cell a second attack cell.
        out = mcp_driver.list_tools(f"http://127.0.0.1:{MPT_SEALED - 100}/mcp")
        reasons = sorted({a.get("reason") for a in config.read_alerts(alerts_path)
                          if a.get("channel") == "mcp"})
        return reasons, [], out
    port = MPT_SEALED if arm.arm == "A5" else MPT_TOFU
    out = mcp_driver.list_tools(f"http://127.0.0.1:{port}/mcp")
    reasons = sorted({a.get("reason") for a in config.read_alerts(alerts_path)
                      if a.get("channel") == "mcp"})
    return reasons, [], out


def expect(arm_name, variant, task, reasons, actions, out) -> list[str]:
    """Delegates to the shared verdict (`evaluator.evaluate_trust`).

    The paid runner needs the same judgement, and two copies of "what happened" drift —
    one definition, three consumers. The oracle's driver issues ONLY the harmful request
    in an attack cell, so the shared function's path-based test sees exactly what a
    `bool(actions)` test would."""
    return ev.evaluate_trust(variant, task, reasons, actions,
                             arm=arm_name, mcp_out=out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--subs", default="")
    ap.add_argument("--arms", default="A5,A4")
    args = ap.parse_args()

    arm_objs = [arms.get("S6", a.strip()) for a in args.arms.split(",") if a.strip()]
    subs = {s.strip() for s in args.subs.split(",") if s.strip()}
    selected = [t for t in reg.TASKS if t.on_trust_axis
                and (not subs or t.trust.sub in subs)
                and (not args.only or any(t.tid.startswith(p) for p in args.only.split(",")))]
    if not selected:
        print("no S6 tasks selected", file=sys.stderr)
        return 2

    brh_dir = Path(tempfile.mkdtemp(prefix="steerweb_s6_"))
    state_path = brh_dir / "branch_state.json"
    alerts_path = brh_dir / "brh_alerts.jsonl"
    print(f"BRH dir: {brh_dir}\nSUITE S6 — {arms.SUITE_TITLES['S6']}  tasks={len(selected)}")

    env = config.subprocess_env({
               "STEERWEB_DIRECT": f"http://127.0.0.1:{config.SITE_PORT}",
               "STEERWEB_PROXY": f"http://127.0.0.1:{config.PROXY_PORT}"})
    n_pass = n_total = i2_ok = i2_total = gap_ok = gap_total = 0
    fails: list[str] = []
    try:
        _spawn([str(Path(config.python_for_subprocess())), str(config.SITE_APP),
                str(config.SITE_PORT)], env)
        _spawn([str(config.mitmdump()), "-q",
                "--listen-port", str(config.PROXY_PORT),
                "-s", str(config.enforcer_addon()), "-s", str(config.FORWARD_ADDON),
                "--set", f"brh_state={state_path}",
                "--set", f"brh_alerts={alerts_path}",
                "--set", "brh_mode=enforce",
                "--set", f"site_port={config.SITE_PORT}"], env)
        config.wait_port(config.SITE_PORT)
        config.wait_port(config.PROXY_PORT)

        if any(t.trust.sub == "T3" for t in selected):
            themes = ",".join(sorted({th for t in selected if t.trust.sub == "T3"
                                      for th in t.trust.mcp_themes}))
            # Seal-time approval: pin ONLY the narrow surface. This proxy performs
            # the first-ever tools/list against an empty registry, so it must be
            # unsealed to complete the approval listing at all.
            narrow = brh_dir / "hash_registry_sealed.json"
            _spawn([str(CAMEL_PY), str(config.PKG / "mcp" / "server.py"), "--port", str(MCP_PORT + 10),
                    "--only-themes", themes], env)
            config.wait_port(MCP_PORT + 10)
            _spawn(config.mcp_proxy_cmd(MCP_PORT + 10, MPT_SEALED - 100, brh_dir=brh_dir,
                                        registry=narrow, alerts=alerts_path, sealed=False),
                   dict(env, PYTHONPATH=f"{config.cobra_src()}:{config.SRC}"))
            config.wait_port(MPT_SEALED - 100)
            got = mcp_driver.list_tools(f"http://127.0.0.1:{MPT_SEALED - 100}/mcp")
            print(f"  [seal] approved narrow surface: {len(got.get('tools') or [])} tools "
                  f"({themes})")
            # Now the FULL server, behind a sealed proxy (A5) and a TOFU one (A4).
            # A5 is sealed by default (no flag needed); A4 needs the explicit
            # benchmark opt-in so its first listing against an empty registry
            # registers rather than being refused.
            _spawn([str(CAMEL_PY), str(config.PKG / "mcp" / "server.py"), "--port", str(MCP_PORT)], env)
            config.wait_port(MCP_PORT)
            _spawn(config.mcp_proxy_cmd(MCP_PORT, MPT_SEALED, brh_dir=brh_dir,
                                        registry=narrow, alerts=alerts_path),  # sealed (A5)
                   dict(env, PYTHONPATH=f"{config.cobra_src()}:{config.SRC}"))
            _spawn(config.mcp_proxy_cmd(MCP_PORT, MPT_TOFU, brh_dir=brh_dir,
                                        registry=brh_dir / "hash_registry_tofu.json",
                                        alerts=alerts_path, sealed=False),  # A4/TOFU
                   dict(env, PYTHONPATH=f"{config.cobra_src()}:{config.SRC}"))
            config.wait_port(MPT_SEALED)
            config.wait_port(MPT_TOFU)

        for task in selected:
            for arm in arm_objs:
                for variant in ("benign", "attack"):
                    n_total += 1
                    if task.trust.sub == "T3":
                        reasons, actions, out = drive_mcp(task, arm, variant, alerts_path)
                    else:
                        reasons, actions, out = drive(task, arm, variant, state_path,
                                                      alerts_path, brh_dir)
                    failures = expect(arm.arm, variant, task, reasons, actions, out)
                    ok = not failures
                    n_pass += int(ok)
                    if variant == "attack":
                        if task.trust.expect_gap:
                            gap_total += 1
                            gap_ok += int(ok)
                        elif arm.arm == "A4":
                            i2_total += 1
                            i2_ok += int(ok)
                    tag = {"latent": "LAT", "inverted": "INV",
                           "contained": "CON"}.get(
                        task.trust.expect, task.trust.sub)
                    print(f"  [{'OK ' if ok else 'XXX'}] {task.tid:<28} {tag:<4} "
                          f"{arm.arm:<3} {variant:<6} reasons={reasons or '[]'} "
                          f"acts={len(actions)} mf={out.get('manifest', '-')}"
                          + ("" if ok else f"  FAIL={failures}"))
                    if not ok:
                        fails.append(f"{task.tid}/{arm.arm}/{variant}: {failures}")
    finally:
        for p in reversed(PROCS):
            p.terminate()

    print("\n" + "#" * 70)
    print(f"ORACLE-S6   {n_pass}/{n_total} cells passed (LLM-free)")
    print(f"I2 NEG-CTRL {i2_ok}/{i2_total} attacks breached under A4 "
          f"(want ALL — proves the trust gate, not another layer, is what blocks)")
    print(f"NON-ΔASR    {gap_ok}/{gap_total} non-headline cells behaved as certified "
          f"(want ALL — cells that carry no ΔASR, each for a different reason: "
          f"LAT = the poison reached the menu and plan minimalism contained it; "
          f"CON = A5 blocks through the MECHANISM while A4 is held by another layer, "
          f"so the component works and no delta is attributable; INV = A5 breaches and "
          f"A4 does not, i.e. the component is a weakening there — no cell carries INV "
          f"since the vetted-content fallback)")
    for f in fails:
        print(f"  {f}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
