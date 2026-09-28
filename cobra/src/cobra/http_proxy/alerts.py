"""Append-only JSONL alert log (`/tmp/brh_alerts.jsonl` by default).

The alert schema is designed for the benchmark metrics:
the Cross-Layer Discrepancy Rate numerator is a line count over this
file, and every record carries enough request/state context (active
branch, state timestamp, violated constraint, observed value) to
attribute or discard a block during analysis — e.g. false positives
from the inherent state-transition race are identifiable by comparing
``ts`` with ``state_ts``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from cobra.http_proxy.brh_check import Decision, BRHState, RequestView


def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def build_alert(
    kind: str,
    mode: str,
    state: BRHState,
    view: RequestView,
    decision: Decision,
) -> dict[str, Any]:
    return {
        "ts": _utc_now_iso(),
        "kind": kind,
        "mode": mode,
        "reason": decision.reason,
        "detail": decision.detail,
        "plan_id": state.plan_id,
        "active_branch": state.active_branch,
        "state_ts": state.state_ts,
        "method": view.method,
        "url": view.url,
        "host": view.host,
    }


def append_alert(path: str | Path, alert: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(alert, default=str) + "\n")
