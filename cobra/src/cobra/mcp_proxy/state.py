"""Reads ``branch_state.json`` — MCP proxy's view of the active plan branch.

``branch_state.json`` is the single IPC channel between the CaMeL process and
the enforcers; MCP proxy is now its third reader (after HTTP proxy's
HTTP addon). MCP proxy polls it on every ``tools/call``, exactly as HTTP proxy polls it
on every HTTP request — the file is tiny, the read is microseconds, and there is
no push channel to keep in sync.

Fail-closed: a missing/unparsable file, or one that is not a JSON object, yields
``None`` and the caller blocks. This is the enforcer contract (opposite of the
hook's never-raise), shared with HTTP proxy.
"""

from __future__ import annotations

import json
import os

DEFAULT_BRH_DIR = os.environ.get("BRH_DIR", "/tmp/brh")


def state_path(brh_dir: str | None = None) -> str:
    return os.path.join(brh_dir or DEFAULT_BRH_DIR, "branch_state.json")


def read_state(brh_dir: str | None = None) -> dict | None:
    try:
        with open(state_path(brh_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None
