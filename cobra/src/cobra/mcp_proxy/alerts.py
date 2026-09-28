"""Alert sink for MCP proxy — the same JSONL file HTTP proxy appends to.

Using one ``/tmp/brh_alerts.jsonl`` for both channels lets ``metrics.py`` compute
the Cross-Channel Discrepancy Rate over HTTP and MCP together.
Each MCP proxy record carries ``channel="mcp"`` and one of the reasons:
``mpt_rug_pull`` / ``mpt_unapproved`` (tools/list) and ``mpt_tool`` / ``mpt_param``
/ ``mpt_inactive`` (tools/call).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

DEFAULT_ALERTS_PATH = os.environ.get("BRH_ALERTS", "/tmp/brh_alerts.jsonl")


def append_alert(reason: str, detail: dict, path: str | None = None) -> None:
    """Appends one alert line; never raises (a logging failure must not break
    the proxy)."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "channel": "mcp",
        "reason": reason,
        "detail": detail,
    }
    try:
        with open(path or DEFAULT_ALERTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
