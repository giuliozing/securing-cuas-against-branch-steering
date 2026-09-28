"""Data models for proposed HTTP and MCP calls.

An ``HttpCall`` is the tool's internal representation of one agent-sitemap
endpoint; it is emitted to the repo's ``sitemap.json`` format by
``emit.py``. An ``McpCall`` is one MCP tool (name + description +
inputSchema), emitted to an MCP manifest.

Both carry a ``selected`` flag driven by the terminal UI and a ``kind``
discriminator used for display and (de)serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HttpCall:
    """One HTTP endpoint the site owner may want an agent to call.

    ``body_fields`` are declared wire-body field names; on emit they become
    the *keys* of the sitemap entry's ``body`` object, which is exactly what
    ``cobra.brh.validator.sanitize_sitemap`` reads to derive constraints.
    ``url`` is a template that may contain ``{placeholders}``.
    """

    name: str
    description: str
    method: str = "GET"
    url: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    body_fields: list[str] = field(default_factory=list)
    example_urls: list[str] = field(default_factory=list)
    priority: int = 2
    selected: bool = True

    kind: str = field(default="http", init=False)


@dataclass
class McpCall:
    """One MCP tool exposed for agents, with a JSON-Schema ``input_schema``."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    selected: bool = True

    kind: str = field(default="mcp", init=False)


def call_to_dict(call: HttpCall | McpCall) -> dict[str, Any]:
    """Serialise a call to a plain dict for the JSON editor / add form."""
    if isinstance(call, HttpCall):
        return {
            "type": "http",
            "name": call.name,
            "description": call.description,
            "method": call.method,
            "url": call.url,
            "category": call.category,
            "tags": list(call.tags),
            "body_fields": list(call.body_fields),
            "example_urls": list(call.example_urls),
            "priority": call.priority,
        }
    return {
        "type": "mcp",
        "name": call.name,
        "description": call.description,
        "input_schema": call.input_schema,
    }


def call_from_dict(data: dict[str, Any], *, selected: bool = True) -> HttpCall | McpCall:
    """Rebuild a call from the JSON editor. Raises ValueError on bad input."""
    kind = str(data.get("type", "http")).lower()
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("field 'name' is required")
    description = str(data.get("description", "")).strip()

    if kind == "mcp":
        schema = data.get("input_schema") or {"type": "object", "properties": {}, "required": []}
        if not isinstance(schema, dict):
            raise ValueError("'input_schema' must be a JSON object")
        return McpCall(name=name, description=description, input_schema=schema, selected=selected)

    method = str(data.get("method", "GET")).upper().strip() or "GET"
    tags = data.get("tags") or []
    body_fields = data.get("body_fields") or []
    example_urls = data.get("example_urls") or []
    if not isinstance(tags, list) or not isinstance(body_fields, list):
        raise ValueError("'tags' and 'body_fields' must be JSON arrays")
    try:
        priority = int(data.get("priority", 2))
    except (TypeError, ValueError):
        priority = 2
    return HttpCall(
        name=name,
        description=description,
        method=method,
        url=str(data.get("url", "")).strip(),
        category=str(data.get("category", "")).strip(),
        tags=[str(t) for t in tags],
        body_fields=[str(b) for b in body_fields],
        example_urls=[str(u) for u in example_urls],
        priority=priority,
        selected=selected,
    )
