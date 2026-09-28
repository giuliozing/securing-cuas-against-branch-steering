"""Emit the curated calls to the repo's on-disk formats.

- HTTP calls  -> ``sitemap.json`` in the exact agent-sitemap shape consumed by
  ``cobra.brh.validator.sanitize_sitemap`` and hash-pinned by
  ``cobra.brh.sitemap_trust``. Declared ``body_fields`` are written as the keys
  of the entry's ``body`` object, which is where the sanitizer reads them.
- MCP calls   -> ``mcp_manifest.json`` as a list of ``{name, description,
  inputSchema}`` tools (the shape MCP proxy hash-pins on ``tools/list``).
"""

from __future__ import annotations

import json
import os
from typing import Any

from .models import HttpCall, McpCall


def http_to_sitemap_entry(call: HttpCall) -> dict[str, Any]:
    """Render one ``HttpCall`` as a repo agent-sitemap entry."""
    body: dict[str, Any] = {f: "" for f in call.body_fields}
    return {
        "category": call.category or "uncategorized",
        "semantic_action": call.description,
        "url": call.url,
        "method": call.method.upper(),
        "body": body,
        "regex": "",
        "resource_types": [],
        "tags": list(call.tags),
        "children": [],
        "example_urls": list(call.example_urls),
        "priority": call.priority,
    }


def mcp_to_manifest_entry(call: McpCall) -> dict[str, Any]:
    """Render one ``McpCall`` as an MCP tool manifest entry."""
    return {
        "name": call.name,
        "description": call.description,
        "inputSchema": call.input_schema,
    }


def write_outputs(
    http_calls: list[HttpCall],
    mcp_calls: list[McpCall],
    out_dir: str,
) -> tuple[str, str | None]:
    """Write sitemap.json (+ mcp_manifest.json if any). Returns their paths."""
    os.makedirs(out_dir, exist_ok=True)

    sitemap = [http_to_sitemap_entry(c) for c in http_calls]
    sitemap_path = os.path.join(out_dir, "sitemap.json")
    with open(sitemap_path, "w", encoding="utf-8") as f:
        json.dump(sitemap, f, indent=4, ensure_ascii=False)

    manifest_path: str | None = None
    if mcp_calls:
        manifest = [mcp_to_manifest_entry(c) for c in mcp_calls]
        manifest_path = os.path.join(out_dir, "mcp_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

    return sitemap_path, manifest_path
