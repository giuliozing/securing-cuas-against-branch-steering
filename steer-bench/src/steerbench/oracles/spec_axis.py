"""LLM-free certification of the SPECIFICATION / PROVENANCE axis.

Free (LLM-free) — run before any paid spec run.

Usage:
    python -m steerbench.oracles.spec_axis [--only 01]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

from steerbench import config
from steerbench.harness import driver
from steerbench.harness import evaluator as ev
from steerbench.harness import stack
from steerbench.tasks import registry as reg

sys.path.insert(0, str(config.cobra_src()))
from cobra.brh.writer import atomic_write_json  # noqa: E402


def _direct() -> str:
    return f"http://127.0.0.1:{config.SITE_PORT}"


def _proxy() -> str:
    return f"http://127.0.0.1:{config.PROXY_PORT}"


def _setup(task, cfg, spec, state_path, alerts_path):
    config.control("POST", "/__reset")
    config.control("POST", "/__config", cfg)
    alerts_path.write_text("")
    atomic_write_json(state_path, ev.oracle_state(task, spec))


def _readout(alerts_path):
    alerts = config.read_alerts(alerts_path)
    reasons = sorted({a.get("reason") for a in alerts if a.get("kind") == "brh_http_block"})
    actions = config.control("GET", "/__state").get("actions", [])
    return reasons, actions


def drive_cell(task, spec, variant, state_path, alerts_path):
    """Model the SECURE behaviour: proceed when specified, abstain when underspecified."""
    cfg = task.spec_config(variant)
    _setup(task, cfg, spec, state_path, alerts_path)
    error = None
    try:
        if spec == "specified":
            driver.submit_action(task.theme, direct_base=_direct(), proxy=_proxy())
        else:
            driver.request_clarification(task.theme, direct_base=_direct(),
                                         reason="committed value not specified")
    except Exception as e:  # noqa: BLE001
        error = repr(e)
    reasons, actions = _readout(alerts_path)
    return ev.evaluate_spec(spec, variant, task, reasons, actions, error), reasons, actions


def drive_negative_control(task, state_path, alerts_path):
    """Naive planner on underspecified·attack: SUBMIT instead of abstaining. The
    poisoned value must reach the wire (recorded) — proving the enforcer is blind and
    abstention is the sole defence. Returns True iff the breach happened (as expected)."""
    cfg = task.spec_config("attack")
    _setup(task, cfg, "underspecified", state_path, alerts_path)
    try:
        driver.submit_action(task.theme, direct_base=_direct(), proxy=_proxy())
    except Exception:  # noqa: BLE001
        pass
    reasons, actions = _readout(alerts_path)
    guarded = [a for a in actions if not (a["path"].endswith("/review")
                                          or a["path"].endswith("/clarify"))]
    return bool(guarded) and not reasons, reasons, actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma tid prefixes, e.g. 01,16")
    args = ap.parse_args()

    selected = [t for t in reg.TASKS if t.on_spec_axis and
                (not args.only or any(t.tid.startswith(p) for p in args.only.split(",")))]

    brh_dir = Path(tempfile.mkdtemp(prefix="steerweb_specoracle_"))
    state_path = brh_dir / "branch_state.json"
    alerts_path = brh_dir / "brh_alerts.jsonl"
    print(f"BRH dir: {brh_dir}\nspec-axis tasks: {len(selected)}")

    st = stack.build(brh_dir=brh_dir, mode="enforce")

    n_pass = n_total = 0
    nc_ok = 0
    fails: list[str] = []
    try:
        for task in selected:
            for spec in ("specified", "underspecified"):
                for variant in ("benign", "attack"):
                    n_total += 1
                    failures, reasons, actions = drive_cell(task, spec, variant,
                                                            state_path, alerts_path)
                    ok = not failures
                    n_pass += int(ok)
                    mark = "OK " if ok else "XXX"
                    extra = "" if ok else f"  FAIL={failures}"
                    print(f"  [{mark}] {task.tid:<22} {spec:<14} {variant:<6} "
                          f"reasons={reasons or '[]'} actions={len(actions)}{extra}")
                    if not ok:
                        fails.append(f"{task.tid}/{spec}/{variant}: {failures}")
            # negative control: prove the underspecified enforcer is blind
            breached, reasons, actions = drive_negative_control(task, state_path, alerts_path)
            nc_ok += int(breached)
            mark = "OK " if breached else "XXX"
            print(f"  [{mark}] {task.tid:<22} {'NEG-CTRL':<14} {'naive':<6} "
                  f"reasons={reasons or '[]'} actions={len(actions)} "
                  f"(poison reached wire={breached})")
            if not breached:
                fails.append(f"{task.tid}/neg-control: expected a breach (blind enforcer), got none")
    finally:
        st.close()

    print("\n" + "#" * 70)
    print(f"SPEC-ORACLE  {n_pass}/{n_total} secure cells passed (LLM-free)")
    print(f"NEG-CONTROL  {nc_ok}/{len(selected)} underspecified·attack breached when naive "
          f"(want ALL — proves enforcement is blind, abstention is the sole defence)")
    for f in fails:
        print(f"  {f}")
    return 0 if (n_pass == n_total and nc_ok == len(selected)) else 1


if __name__ == "__main__":
    sys.exit(main())
