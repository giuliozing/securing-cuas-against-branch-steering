"""BRH enforcement layer for HTTP proxy: reads
`branch_state.json` and decides whether an HTTP request is authorised by
the currently active plan branch.

This module is **stdlib-only and self-contained by design** (option B of
the parity decision): the only runtime contract with the CaMeL process
is the state file itself, so the enforcer stays deployable without the
cobra-cua repository or its dependency chain. The constraint-evaluation
semantics implemented here is a deliberate duplicate of
`cobra/brh/contract.py`; the two are kept in lockstep by the shared
golden vectors (`brh_contract_vectors.json` at the repository root),
which both test suites consume — a divergence turns into a named red
test, not a silent enforcement gap.

Fail-closed table (every degenerate input blocks):

    state file missing            → block  (CaMeL not running / wrong --set brh_state)
    state file unparsable         → block  (should not happen: writes are atomic)
    active_branch: null           → block  (legitimate between plans; alert de-spammed)
    http_constraints null/missing → block  (empty domain allowlist authorises nothing)
    unresolved placeholder        → that constraint is unsatisfiable when the field is present
    malformed constraint entry    → block  (a constraints file we cannot read is not a
                                            constraints file we can trust)

Field-constraint scope: a constraint applies to every occurrence of its
``path`` found in the request — dot-separated traversal of the parsed
body (``order.amount`` → ``body["order"]["amount"]``) plus query
parameters under the literal path string. A request that does not carry
the field at all passes the constraint: the domain allowlist is the
gate, field constraints narrow the values *when the field travels*.
(Documented residual risk: renaming the field sidesteps the value check;
the request still has to survive the domain gate and the server's own
schema.) Note that query parameter values are always strings, so a
numeric constraint on a query-only field can never be satisfied —
strict typing is kept even there, deliberately.

Domain matching: exact, case-insensitive comparison on the bare
hostname. No wildcards, no ports — the cobra-side validator guarantees
allowlist entries are bare lowercase hostnames; an entry in any other
form simply never matches (narrowing, hence safe).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reserved placeholder strings (mirrors cobra/brh/schema.py PLACEHOLDERS).
PLACEHOLDERS = frozenset({"trigger_value", "from_plan"})


# ---------------------------------------------------------------------------
# Constraint evaluation — parity with cobra/brh/contract.py via the shared
# vector suite. Change brh_contract_vectors.json first, then both sides.
# ---------------------------------------------------------------------------


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value in PLACEHOLDERS


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def typed_equal(expected: Any, observed: Any) -> bool:
    """Strict-typed equality: numbers with numbers, strings with strings,
    bool only with bool (bool ⊂ int in Python must not leak through).
    ``None`` (a JSON ``null``) only equals ``None`` — never a wildcard, and
    never equal to a present value."""
    if expected is None or observed is None:
        return expected is None and observed is None
    if isinstance(expected, bool) or isinstance(observed, bool):
        return isinstance(expected, bool) and isinstance(observed, bool) and expected == observed
    if _is_number(expected) and _is_number(observed):
        return expected == observed
    if isinstance(expected, str) and isinstance(observed, str):
        return expected == observed
    return False


def _has_placeholder(value: Any) -> bool:
    """True if a placeholder string appears anywhere in `value` (recursively)."""
    if is_placeholder(value):
        return True
    if isinstance(value, list):
        return any(_has_placeholder(v) for v in value)
    if isinstance(value, dict):
        return any(_has_placeholder(v) for v in value.values())
    return False


def struct_equal(expected: Any, observed: Any) -> bool:
    """Recursive value equality (scalars use the strict typing of ``==``; lists
    element-wise in order; dicts by identical key set + per-key equality;
    ``None`` leaves compared the same way — only ``None`` equals ``None``)."""
    if expected is None or observed is None:
        return expected is None and observed is None
    if isinstance(expected, bool) or isinstance(observed, bool):
        return isinstance(expected, bool) and isinstance(observed, bool) and expected == observed
    if _is_number(expected) and _is_number(observed):
        return expected == observed
    if isinstance(expected, str) and isinstance(observed, str):
        return expected == observed
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(
            struct_equal(e, o) for e, o in zip(expected, observed)
        )
    if isinstance(expected, dict) and isinstance(observed, dict):
        return set(expected) == set(observed) and all(
            struct_equal(expected[k], observed[k]) for k in expected
        )
    return False


def _subset(value: Any, observed: Any) -> bool:
    """Every element of `observed` (list) value-matches a member of `value`
    (list); an empty observed satisfies, a non-list observed never does."""
    if not isinstance(value, list) or not isinstance(observed, list):
        return False
    return all(any(struct_equal(member, o) for member in value) for o in observed)


def satisfies(op: Any, value: Any, observed: Any) -> bool:
    """True iff `observed` satisfies ``(op, value)``. Total: malformed
    input (unknown op, wrong shape, unresolved placeholder) is
    unsatisfiable, never an exception."""
    if op == "in":
        if not isinstance(value, list) or not value:
            return False
        if any(is_placeholder(member) for member in value):
            return False
        return any(typed_equal(member, observed) for member in value)
    if op == "subset":
        return False if _has_placeholder(value) else _subset(value, observed)
    if op == "eq_struct":
        return False if _has_placeholder(value) else struct_equal(value, observed)
    if op == "==":
        # An exact pin on a non-scalar IS structural equality: a "from_plan"
        # list/dict pin must mean "equal to this value", never "always false".
        # (Handled before the list-guard below, which exists only to fail-close
        # the ordering ops <=/>=, where a list/dict value is meaningless.)
        if isinstance(value, (list, dict)):
            return False if _has_placeholder(value) else struct_equal(value, observed)
        if is_placeholder(value):
            return False
        return typed_equal(value, observed)
    if isinstance(value, list):
        return False
    if is_placeholder(value):
        return False
    if op == "<=" or op == ">=":
        if not (_is_number(value) and _is_number(observed)):
            return False
        return observed <= value if op == "<=" else observed >= value
    return False


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


def _compile_endpoint_pattern(path_pattern: str) -> re.Pattern:
    """`{param}` → `[^/]+`, `*` → `.*`, then anchor-match the whole path.

    Examples: ``/checkout/{id}`` matches ``/checkout/42``;
    ``/api/*`` matches ``/api/v1/users``.
    """
    # Escape everything except the two placeholder forms, then convert them.
    # We split on {…} and * to avoid re-escaping our own substitutions.
    parts = re.split(r"(\{[^}]*\}|\*)", path_pattern)
    regex_parts = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            regex_parts.append("[^/]+")
        elif part == "*":
            regex_parts.append(".*")
        else:
            regex_parts.append(re.escape(part))
    return re.compile("^" + "".join(regex_parts) + "$")


@dataclass(frozen=True)
class BRHState:
    """Parsed `branch_state.json`, normalised for the checker.

    `status` is one of ``"active"``, ``"inactive"`` (active_branch null),
    ``"missing"``, ``"malformed"``.
    """

    status: str
    plan_id: str | None = None
    active_branch: str | None = None
    state_ts: str | None = None
    allowed_domains: tuple[str, ...] = ()
    fields: tuple[Any, ...] = ()
    # Endpoint allowlist: plan-time P-LLM annotations — [{method, domain, path_pattern}].
    # Non-empty only when an HTTP manifest was present at annotation time.
    allowed_endpoints: tuple[Any, ...] = ()
    # Runtime sitemap: fetched by hook at branch-entry for domains not in allowed_endpoints.
    # {domain: [{method, path}]} — used for weaker "is this endpoint in the sitemap?" check.
    sitemap_schema: dict[str, Any] = field(default_factory=dict)


def read_state(path: str | Path) -> BRHState:
    """Reads and classifies the state file. Polled on every request
    (architecture decision: no caching — the file is tiny, writes are
    atomic via os.replace, and HTTP latency dwarfs the read)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return BRHState(status="missing")
    except OSError:
        return BRHState(status="malformed")
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("state root is not an object")
    except ValueError:
        return BRHState(status="malformed")

    plan_id = data.get("plan_id")
    state_ts = data.get("ts")
    active_branch = data.get("active_branch")
    if active_branch is None:
        return BRHState(status="inactive", plan_id=plan_id, state_ts=state_ts)

    http = data.get("http_constraints")
    if not isinstance(http, dict):
        # Active branch with no readable HTTP block: authorises nothing.
        http = {}
    raw_domains = http.get("allowed_domains")
    domains = tuple(
        d.lower() for d in raw_domains if isinstance(d, str)
    ) if isinstance(raw_domains, list) else ()
    raw_fields = http.get("fields")
    fields = tuple(raw_fields) if isinstance(raw_fields, list) else ()
    raw_ep = http.get("allowed_endpoints")
    allowed_endpoints = tuple(
        ep for ep in raw_ep
        if isinstance(ep, dict)
        and isinstance(ep.get("method"), str)
        and isinstance(ep.get("domain"), str)
        and isinstance(ep.get("path_pattern"), str)
    ) if isinstance(raw_ep, list) else ()

    raw_sitemap = data.get("sitemap_schema")
    sitemap_schema: dict = {}
    if isinstance(raw_sitemap, dict):
        for dom, entries in raw_sitemap.items():
            if isinstance(dom, str) and isinstance(entries, list):
                sitemap_schema[dom.lower()] = [
                    e for e in entries
                    if isinstance(e, dict)
                    and isinstance(e.get("method"), str)
                    and isinstance(e.get("path"), str)
                ]

    return BRHState(
        status="active",
        plan_id=plan_id,
        active_branch=active_branch,
        state_ts=state_ts,
        allowed_domains=domains,
        fields=fields,
        allowed_endpoints=allowed_endpoints,
        sitemap_schema=sitemap_schema,
    )


class StateReader:
    """Per-request state access with a ``stat``-gated parse cache.

    The addon calls this on **every** HTTP request. The bare ``read_state``
    re-opens, decodes and ``json.loads`` the file each time; under a heavy
    page's fan-out (united.com: ~1600 requests over ~40 hosts) those parses
    serialise inside mitmproxy's single event loop and stall it — Chrome then
    reports ``ERR_PROXY_CONNECTION_FAILED`` on connections the proxy can no
    longer accept in time. This reader removes the redundant parse from that
    hot path while keeping the **exact same freshness contract**:

      * The file's identity ``(st_ino, st_mtime_ns, st_size)`` is checked with
        a single ``os.stat`` on every call — the same per-request freshness
        check the uncached path does implicitly.
      * The writer (``cobra.brh.writer.atomic_write_json``) updates the file by
        writing a fresh temp file and ``os.replace``-ing it in, so **every**
        update swaps in a new inode and bumps mtime/size. An unchanged identity
        tuple therefore guarantees unchanged content, and the cached ``BRHState``
        is returned without re-reading.
      * The ``stat`` is taken *before* the read: if a write lands between the
        two, we cache the freshly-read (newer) content under the older key, and
        the next call's ``stat`` sees the new key and re-reads — so we never
        serve state across a write. This is the identical one-request race the
        uncached path already had; caching adds no staleness.

    Degenerate states (missing/malformed) are cached too: fail-closed stays
    fast, and any later valid write presents a new identity and is re-read.
    Not thread-safe — one instance per addon, used only from the event loop.
    """

    def __init__(self) -> None:
        self._key: tuple[int, int, int] | None = None
        self._state: BRHState | None = None

    def read(self, path: str | Path) -> BRHState:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            self._key, self._state = None, None
            return BRHState(status="missing")
        except OSError:
            self._key, self._state = None, None
            return BRHState(status="malformed")
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        if key == self._key and self._state is not None:
            return self._state
        state = read_state(path)
        self._key, self._state = key, state
        return state


# ---------------------------------------------------------------------------
# Request view and decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestView:
    """The slice of an HTTP request the BRH layer evaluates. Built by
    `adapter.request_view_from_flow`; plain data so the checker is
    testable without mitmproxy."""

    host: str
    """Bare hostname, lowercase, no port."""
    port: int | None
    method: str
    url: str
    body: dict[str, Any] = field(default_factory=dict)
    query: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


ALLOW = Decision(allowed=True)


def _host_allowed(host: str, allowed_domains: tuple[str, ...]) -> bool:
    """Match a host against the allow-list. Two entry forms, so sub-domain
    widening stays *explicit and auditable* rather than implicit:

      * bare ``shop.example.com``  — EXACT host match only. No widening: an
        attacker-controlled ``user.shop.example.com`` (sub-domain takeover,
        user-content sub-domains) is NOT authorised by listing the apex.
      * wildcard ``*.example.com``  — the domain AND any sub-domain
        (``example.com``, ``media.example.com``). Used only for a task's OWN
        trusted starting site, whose first-party sub-domains (CDN/telemetry:
        ``media.united.com``, ``unagi.amazon.com``) would otherwise be flagged
        as an enforcement-granularity false positive.

    The wildcard form must be emitted deliberately (by the starting-domain
    bootstrap / annotator for the own site); third-party entries stay exact."""
    host = (host or "").lower()
    for d in allowed_domains:
        if d.startswith("*."):
            base = d[2:]
            if host == base or host.endswith("." + base):
                return True
        elif host == d:
            return True
    return False


def check(state: BRHState, view: RequestView) -> Decision:
    """Evaluates a request against the current BRH state. Returns the
    first violation found; reasons are stable machine-readable codes
    (consumed by the alert log and the metrics script)."""
    if state.status == "missing":
        return Decision(False, "brh_state_missing", {})
    if state.status == "malformed":
        return Decision(False, "brh_state_malformed", {})
    if state.status == "inactive":
        return Decision(False, "brh_inactive", {"plan_id": state.plan_id})

    if not _host_allowed(view.host, state.allowed_domains):
        return Decision(
            False,
            "brh_domain",
            {"host": view.host, "allowed_domains": list(state.allowed_domains)},
        )

    endpoint_decision = _check_endpoint(state, view)
    if endpoint_decision is not None:
        return endpoint_decision

    for entry in state.fields:
        try:
            path, op, value = entry["path"], entry["op"], entry["value"]
            if not isinstance(path, str) or not isinstance(op, str):
                raise TypeError("path/op must be strings")
        except (KeyError, TypeError):
            return Decision(False, "brh_constraint_malformed", {"constraint": repr(entry)})
        for where, observed in _occurrences(view, path):
            if not satisfies(op, value, observed):
                return Decision(
                    False,
                    "brh_field",
                    {"path": path, "op": op, "value": value, "observed": observed, "where": where},
                )

    return ALLOW


def _url_path(url: str) -> str:
    """Extract the path component from a URL (e.g. '/checkout' from 'https://shop.com/checkout?x=1')."""
    try:
        # Fast parse: find path between the third '/' and any '?' or '#'.
        after_scheme = url.find("://")
        if after_scheme >= 0:
            start = url.find("/", after_scheme + 3)
            if start < 0:
                return "/"
        else:
            start = 0 if url.startswith("/") else -1
            if start < 0:
                return "/"
        end = len(url)
        for ch in ("?", "#"):
            idx = url.find(ch, start)
            if 0 <= idx < end:
                end = idx
        return url[start:end] or "/"
    except Exception:
        return "/"


def _check_endpoint(state: BRHState, view: RequestView) -> Decision | None:
    """Endpoint-level check: HTTP analog of MCP's allowed_tools.

    Two-tier logic:
    1. If allowed_endpoints has entries for view.host → (method, path) must match one.
    2. Else if sitemap_schema has entries for view.host → (method, path) must appear there.
    3. Neither present → no endpoint check (domain check already passed; unknown schema).

    Returns a blocking Decision or None (pass through to field check).
    """
    host = (view.host or "").lower()
    method = (view.method or "").upper()
    path = _url_path(view.url)

    # --- Tier 1: plan-annotated allowed_endpoints (static domains) ---
    host_eps = [
        ep for ep in state.allowed_endpoints
        if ep.get("domain", "").lower() == host
    ]
    if host_eps:
        for ep in host_eps:
            if ep.get("method", "").upper() == method:
                try:
                    pattern = _compile_endpoint_pattern(ep["path_pattern"])
                    if pattern.match(path):
                        return None  # matched → pass
                except Exception:
                    continue
        return Decision(
            False,
            "brh_endpoint",
            {"host": host, "method": method, "path": path,
             "allowed_endpoints": [{"method": e.get("method"), "path_pattern": e.get("path_pattern")} for e in host_eps]},
        )

    # --- Tier 2: runtime sitemap_schema (runtime-discovered domains) ---
    sitemap_entries = state.sitemap_schema.get(host)
    if sitemap_entries:
        for entry in sitemap_entries:
            if entry.get("method", "").upper() == method:
                try:
                    pattern = _compile_endpoint_pattern(entry["path"])
                    if pattern.match(path):
                        return None  # matched → pass
                except Exception:
                    continue
        return Decision(
            False,
            "brh_endpoint_sitemap",
            {"host": host, "method": method, "path": path},
        )

    return None  # no endpoint schema available → pass (domain check is the gate)


def _occurrences(view: RequestView, path: str) -> list[tuple[str, Any]]:
    """All occurrences of `path` in the request: dot-path traversal of
    the body (dicts only — lists are not descended into) plus
    query parameters under the literal path string."""
    found: list[tuple[str, Any]] = []
    node: Any = view.body
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            node = _MISSING
            break
    if node is not _MISSING:
        found.append(("body", node))
    for value in view.query.get(path, []):
        found.append(("query", value))
    return found


_MISSING = object()
