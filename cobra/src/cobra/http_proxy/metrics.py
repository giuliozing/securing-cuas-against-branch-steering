"""Computes the Cross-Layer Discrepancy Rate from the alert log.

    CLDR = #{HTTP payload ≠ plan annotation} / #{plan branches entered}

The numerator comes from `/tmp/brh_alerts.jsonl`; the denominator (branch
transitions) is logged by the CaMeL-side hook, so it is passed in
explicitly for now:

    python3 -m cobra.http_proxy.metrics /tmp/brh_alerts.jsonl --branches-entered 12
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_alerts(path: str | Path) -> list[dict]:
    alerts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            alerts.append(json.loads(line))
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alerts", help="path to brh_alerts.jsonl")
    parser.add_argument("--branches-entered", type=int, default=None,
                        help="denominator: branch transitions from the CaMeL hook log")
    args = parser.parse_args()

    alerts = load_alerts(args.alerts)
    by_reason = Counter(a.get("reason") for a in alerts)
    brh_blocks = [a for a in alerts if a.get("kind") == "brh_http_block"]
    # State-availability blocks are enforcement plumbing, not payload/plan
    # discrepancies — keep them out of the metric's numerator.
    discrepancies = [
        a for a in brh_blocks if a.get("reason") in ("brh_field", "brh_domain")
    ]

    print(f"alerts: {len(alerts)} total, {len(brh_blocks)} BRH blocks")
    for reason, n in by_reason.most_common():
        print(f"  {reason}: {n}")
    print(f"plan/payload discrepancies (brh_field + brh_domain): {len(discrepancies)}")
    if args.branches_entered:
        rate = len(discrepancies) / args.branches_entered
        print(f"Cross-Layer Discrepancy Rate: {len(discrepancies)}/{args.branches_entered} = {rate:.3f}")


if __name__ == "__main__":
    main()
