"""LLM-free harness certification for STEER-Bench.

Usage:
    python -m steerbench.oracles.http_axis [--only 01]
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


def drive(task, variant, state_path, alerts_path):
    cfg = task.benign_config() if variant == "benign" else task.attack_config()
    config.control("POST", "/__reset")
    config.control("POST", "/__config", cfg)
    alerts_path.write_text("")

    # The annotator's output, written directly (no LLM). Atomic replace so the
    # enforcer's stat-gated StateReader picks up a fresh inode.
    atomic_write_json(state_path, ev.oracle_state(task))

    error = None
    try:
        driver.submit_action(
            task.theme,
            direct_base=f"http://127.0.0.1:{config.SITE_PORT}",
            proxy=f"http://127.0.0.1:{config.PROXY_PORT}",
        )
    except Exception as e:  # noqa: BLE001
        error = repr(e)

    alerts = config.read_alerts(alerts_path)
    reasons = sorted({a.get("reason") for a in alerts if a.get("kind") == "brh_http_block"})
    actions = config.control("GET", "/__state").get("actions", [])
    return ev.evaluate(variant, task, reasons, actions, error), reasons, actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma tid prefixes, e.g. 01,13")
    args = ap.parse_args()

    # Only the HTTP axis (the 50 wire-divergence tasks). Suite S3 drives an MCP
    # `tools/call` through MCP proxy and suite S6 drives the sitemap trust gate; neither has
    # a path through this harness, and both have their own oracle (`oracles/s3.py`,
    # `oracles/s6.py`). Filtering on the named predicate rather than on "not S3" is what
    # keeps the next attack_kind from silently broadening this loop.
    selected = [t for t in reg.TASKS
                if t.on_http_axis
                and (not args.only or any(t.tid.startswith(p) for p in args.only.split(",")))]

    brh_dir = Path(tempfile.mkdtemp(prefix="steerweb_oracle_"))
    state_path = brh_dir / "branch_state.json"
    alerts_path = brh_dir / "brh_alerts.jsonl"
    print(f"BRH dir: {brh_dir}")

    st = stack.build(brh_dir=brh_dir, mode="enforce")

    n_pass = n_total = 0
    fails: list[str] = []
    try:
        for task in selected:
            for variant in ("benign", "attack"):
                n_total += 1
                failures, reasons, actions = drive(task, variant, state_path, alerts_path)
                ok = not failures
                n_pass += int(ok)
                mark = "OK " if ok else "XXX"
                extra = "" if ok else f"  FAIL={failures}"
                print(f"  [{mark}] {task.tid:<20} {variant:<6} "
                      f"reasons={reasons or '[]'} actions={len(actions)}{extra}")
                if not ok:
                    fails.append(f"{task.tid}/{variant}: {failures}")
    finally:
        st.close()

    print("\n" + "#" * 70)
    print(f"ORACLE {n_pass}/{n_total} configurations passed (LLM-free)")
    for f in fails:
        print(f"  {f}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
