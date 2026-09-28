"""mitmproxy flow → `RequestView` (BRH layer) / legacy `Action` (Policy
layer).

No mitmproxy import: the flow is duck-typed (`flow.request.method`,
`.pretty_url`, `.headers.get(...)`, `.raw_content`), so this module is
unit-testable with fakes and the parsing logic stays identical whatever
drives it. The body is parsed **once** per request and feeds both
enforcement layers — the BRH check and `Policy.evaluate` see the same
representation of the payload.

Body parsing mirrors the semantics of the legacy
`Action.from_request` (http_proxy/policy/policy.py): JSON, form-urlencoded
and multipart bodies become dicts, anything else lands under ``"_raw"``.
One deliberate fix vs. the legacy code: multipart parsing feeds the
Content-Type header (with its boundary) to the email parser — parsing
the bare body bytes, as policy.py does, never yields the parts.
"""

from __future__ import annotations

import email
import json
import urllib.parse
from email.utils import collapse_rfc2231_value
from typing import Any

from cobra.http_proxy.brh_check import RequestView


def parse_body(content_type: str, raw: bytes | str | None) -> dict[str, Any]:
    """Parses a request body into a dict according to its content type."""
    if not raw:
        return {}
    ct = (content_type or "").lower()

    if "application/json" in ct:
        text = raw.decode(errors="ignore") if isinstance(raw, bytes) else raw
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}
        # Non-object JSON (list/scalar) has no addressable fields; keep it
        # visible without inventing paths into it (no list traversal).
        return parsed if isinstance(parsed, dict) else {"_json": parsed}

    if "application/x-www-form-urlencoded" in ct:
        text = raw.decode(errors="ignore") if isinstance(raw, bytes) else raw
        return dict(urllib.parse.parse_qsl(text, keep_blank_values=True))

    if "multipart/form-data" in ct:
        raw_bytes = raw.encode() if isinstance(raw, str) else raw
        # The boundary lives in the Content-Type header, so the parser
        # needs the header line, not just the body.
        msg = email.message_from_bytes(
            b"Content-Type: " + content_type.encode(errors="ignore") + b"\r\n\r\n" + raw_bytes
        )
        body: dict[str, Any] = {}
        for part in msg.walk():
            if part.get_content_disposition() != "form-data":
                continue
            name_param = part.get_param("name", header="content-disposition")
            name = collapse_rfc2231_value(name_param) if name_param else None
            if not name:
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True)
            if filename:
                body[name] = {
                    "filename": filename,
                    "content_type": part.get_content_type(),
                    "content": payload if isinstance(payload, bytes) else str(payload),
                }
            else:
                body[name] = payload.decode(errors="ignore") if isinstance(payload, bytes) else str(payload)
        return body

    return {"_raw": raw.decode(errors="ignore") if isinstance(raw, bytes) else raw}


def build_request_view(
    method: str, url: str, content_type: str, raw_body: bytes | str | None
) -> RequestView:
    parts = urllib.parse.urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = None
    return RequestView(
        host=(parts.hostname or "").lower(),
        port=port,
        method=method.upper(),
        url=url,
        body=parse_body(content_type, raw_body),
        query=urllib.parse.parse_qs(parts.query, keep_blank_values=True),
    )


def request_view_from_flow(flow: Any) -> RequestView:
    req = flow.request
    return build_request_view(
        method=req.method,
        url=req.pretty_url,
        content_type=req.headers.get("content-type", ""),
        raw_body=req.raw_content,
    )


def action_from_view(view: RequestView, sitemap: Any | None):
    """Builds a legacy `Action` for the static Policy layer from the
    already-parsed view. Imported lazily: the legacy policy module pulls
    in playwright, which the BRH-only deployment does not need."""
    from cobra.http_proxy.policy.policy import Action, Endpoint

    return Action(
        endpoint=Endpoint(url=view.url, method=view.method),
        domain=urllib.parse.urlsplit(view.url).netloc,
        tags=sitemap.get_tags(view.method, view.url, view.body) if sitemap else [],
        body=view.body,
    )
