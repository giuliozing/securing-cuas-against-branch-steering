"""Tool-definition hash registry for MCP proxy.

Each MCP tool definition (``name`` + ``description`` + ``inputSchema``) is
serialised canonically and SHA-256 hashed. The registry pins the hash that was
trusted at approval time; a later ``tools/list`` whose hash differs is a rug
pull (the tool changed after approval).

Canonicalisation must be stable across servers/runs: we hash a JSON object with
sorted keys and no incidental whitespace, over exactly the three fields that
make up the tool's trusted surface. Anything else (ordering of keys in the wire
JSON, added transport metadata) must not change the hash.

The registry is keyed by ``"<server_id>::<tool_name>"`` — per-server, since two
servers may legitimately expose a tool of the same name (cross-server shadowing
is out of scope, but per-server keying is the correct baseline for it).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

DEFAULT_REGISTRY_PATH = os.path.expanduser("~/.mpt/hash_registry.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_tool_bytes(tool: dict) -> bytes:
    """Canonical byte serialisation of a tool's trusted surface.

    Only ``name``/``description``/``inputSchema`` are hashed; sorted keys +
    compact separators make the result independent of incidental JSON ordering.
    """
    payload = {
        "name": tool.get("name"),
        "description": tool.get("description"),
        "inputSchema": tool.get("inputSchema"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def tool_hash(tool: dict) -> str:
    return hashlib.sha256(canonical_tool_bytes(tool)).hexdigest()


def registry_key(server_id: str, tool_name: str | None) -> str:
    return f"{server_id}::{tool_name}"


def load_registry(path: str | None = None) -> dict:
    """Loads the registry, returning ``{}`` if missing/unparsable.

    A missing registry is the legitimate first-run state (trust-on-first-use),
    not an error; an unparsable one is treated the same here — the caller's
    enforcement (sealed mode) is what decides whether an unknown tool is blocked.
    """
    path = path or DEFAULT_REGISTRY_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_registry(registry: dict, path: str | None = None) -> None:
    """Atomically writes the registry (temp file + ``os.replace``)."""
    path = path or DEFAULT_REGISTRY_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def approve_tool(tool: dict, server_id: str, path: str | None = None) -> None:
    """Add (or refresh) a single tool in the persistent registry.

    Idempotent: re-approving the same tool updates its timestamp but keeps
    the same hash. Called by the approval loop after the human picks a tool."""
    reg = load_registry(path)
    key = registry_key(server_id, tool.get("name"))
    reg[key] = {"hash": tool_hash(tool), "ts": _utc_now_iso(), "approved": True}
    save_registry(reg, path)


def approved_tools_from_registry(
    all_tools: list[dict], server_id: str, registry: dict
) -> list[dict]:
    """Filter all_tools to those whose name+hash are present in registry as approved.

    Used by the approval loop to seed the initial manifest from prior sessions.
    A tool is included only if its current hash matches the stored hash — a
    server-side change (rug pull) is therefore automatically excluded."""
    result = []
    for tool in all_tools or []:
        name = tool.get("name")
        if not name:
            continue
        rec = registry.get(registry_key(server_id, name))
        if rec and rec.get("approved") and rec.get("hash") == tool_hash(tool):
            result.append(tool)
    return result
