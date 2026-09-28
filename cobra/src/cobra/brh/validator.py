"""Validation of LLM-produced annotations against the AST skeleton.

The validator is what bounds the annotator's blast radius: the LLM can
only fill semantic slots for branches that exist in the skeleton, cannot
drop branches, cannot contradict the statically extracted facts, and
cannot authorise anything on arms that have no executable body. Every
rejection is returned as a human-readable error string that is fed back
to the annotator on retry.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Mapping, Sequence
from urllib.parse import urlparse

from cobra.brh.schema import (
    FROM_PLAN_PLACEHOLDER,
    PLACEHOLDERS,
    TRIGGER_VALUE_PLACEHOLDER,
    BranchConstraints,
    FieldConstraint,
    PlanConstraints,
    var_placeholder_name,
)
from cobra.brh.contract import is_placeholder
from cobra.brh.skeleton import ROOT_KEY, PlanSkeleton

# The MCP tool manifest: the set of *approved* MCP tools the P-LLM is
# allowed to constrain, each mapped to its declared parameter names (from the
# server's inputSchema). `None`/empty = HTTP-only annotation. It is passed in by
# the caller — after the two-phase approval
# the approved tool names/params are known — so the annotator never has to
# discover MCP tools from descriptions.
McpManifest = Mapping[str, Sequence[str]]


# ---------------------------------------------------------------------------
# HTTP endpoint manifest (from agent sitemap)
# ---------------------------------------------------------------------------

# Whitelisted HTTP methods — anything else is rejected by sanitize_sitemap.
_ALLOWED_METHODS: frozenset[str] = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})

# Permitted characters in a path template: URL-safe chars, path params {name},
# and wildcards (*). No whitespace, no quotes, no semicolons, no newlines —
# the path ends up verbatim in the annotator prompt so we keep it injection-free.
_SAFE_PATH_RE = re.compile(r"^[a-zA-Z0-9/_{}.*\-]*$")

# Body field names must be valid (dot-path) identifiers.
_FIELD_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


@dataclasses.dataclass(frozen=True)
class HttpEndpoint:
    """Structural description of one HTTP endpoint from the agent sitemap.

    Only the fields safe to expose to the planner are kept here.  All
    free-text fields from the raw sitemap entry (semantic_action, tags,
    category, example_urls, priority, children) are stripped by
    ``sanitize_sitemap`` before this object is created — except
    ``description``, which is populated only when the caller opted in
    (``include_descriptions=True``, reserved for human-approved hash-pinned
    sitemaps — the two-state exposure model in ``cobra.brh.sitemap_trust``).
    """

    method: str             # whitelisted HTTP verb, uppercased
    domain: str             # bare hostname, e.g. "gitlab.com"
    path_template: str      # URL path pattern, e.g. "/{group}/{project}/issues"
    body_fields: frozenset  # frozenset[str] — declared wire-body field names
    description: str = ""   # semantic_action free text; non-empty ONLY for human-approved sitemaps


HttpManifest = Sequence[HttpEndpoint]

# Description free text kept for human-approved sitemaps is still normalised:
# whitespace collapsed, backticks removed (they could escape the ```json fence
# in the annotator prompt), and capped so a description cannot dominate it.
_MAX_DESCRIPTION_LEN = 200


def _clean_description(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.replace("`", "'").split())[:_MAX_DESCRIPTION_LEN]


def sanitize_sitemap(raw: list[dict], *, include_descriptions: bool = False) -> list[HttpEndpoint]:
    """Convert a raw sitemap JSON list into a safe ``HttpManifest``.

    Security guarantee: only structural, whitelist-validated fields survive —
    method (enum), domain (hostname-regex), path (URL-safe chars), body field
    names (identifier regex).  Every free-text field (semantic_action, tags,
    category, example_urls, priority, children) is discarded, so a compromised
    sitemap entry cannot inject text into the planner prompt.

    ``include_descriptions=True`` keeps a normalised ``semantic_action`` as
    ``description``. Only pass it for sitemaps that are human-approved and
    hash-pinned in the sitemap trust registry (cobra.brh.sitemap_trust): the
    free text was vetted at approval time, so it is trusted content
    (two-state model). TOFU-approved or unapproved sitemaps must stay blind.

    Invalid or unparseable entries are silently skipped (fail-safe: the
    annotator gets less guidance but no injection path exists).
    """
    result: list[HttpEndpoint] = []
    for entry in raw:
        # --- method (whitelist) ---
        raw_method = entry.get("method", "")
        if not isinstance(raw_method, str):
            continue
        method = raw_method.upper().strip()
        if method not in _ALLOWED_METHODS:
            continue

        # --- URL → domain + path (structural parse only) ---
        raw_url = entry.get("url", "")
        if not isinstance(raw_url, str):
            continue
        try:
            parsed = urlparse(raw_url)
            netloc = parsed.netloc or ""
            # Strip port if present (allowed_domains never include ports).
            if ":" in netloc and not netloc.startswith("["):  # skip IPv6 brackets
                netloc = netloc.rsplit(":", 1)[0]
            domain = netloc.lower()
            path = parsed.path or "/"
        except Exception:
            continue

        # Validate domain format (may be empty for relative-URL entries → skip).
        if not domain:
            continue
        if not _HOSTNAME_RE.match(domain):
            continue

        # Validate path: only safe characters allowed.
        if not _SAFE_PATH_RE.match(path):
            continue

        # --- body field names (identifier whitelist) ---
        raw_body = entry.get("body") or {}
        if not isinstance(raw_body, dict):
            raw_body = {}
        body_fields: frozenset[str] = frozenset(
            k for k in raw_body
            if isinstance(k, str) and _FIELD_NAME_RE.match(k)
        )

        result.append(HttpEndpoint(
            method=method,
            domain=domain,
            path_template=path,
            body_fields=body_fields,
            description=_clean_description(entry.get("semantic_action")) if include_descriptions else "",
        ))

    return result

# Bare lowercase hosts only — no scheme, path, port or uppercase: junk an
# LLM annotator might emit would otherwise never match in the enforcer or
# match the wrong thing. Loopback/IPv4/single-label forms are accepted
# because benchmark deployments (WebArena, OSWorld containers) live on
# localhost, IP literals and docker service names — and the skeleton
# extracts those from literal URLs, so rejecting them here would make
# such plans impossible to annotate validly (every annotation would fail
# either the format check or the static-domain cross-check).
_HOSTNAME_RE = re.compile(
    r"^(?:"
    r"\*\.(?:[a-z0-9-]+\.)+[a-z]{2,}"  # wildcard: *.example.com (own-site sub-domains; not *.tld)
    r"|(?:[a-z0-9-]+\.)+[a-z]{2,}"  # dotted public hostname
    r"|(?:\d{1,3}\.){3}\d{1,3}"  # IPv4 literal
    r"|[a-z0-9-]+"  # single-label host (localhost, docker service names)
    r")$"
)

# An "in" set larger than this is almost certainly the annotator dumping
# data instead of expressing a constraint.
MAX_IN_SET_SIZE = 50


def validate(
    constraints: PlanConstraints,
    skeleton: PlanSkeleton,
    mcp_tools: McpManifest | None = None,
    http_manifest: HttpManifest | None = None,
) -> list[str]:
    """Returns a list of validation errors (empty list = valid).

    ``mcp_tools`` is the approved MCP tool manifest (tool -> param names). When
    given, ``mcp_constraints`` are validated against it; when omitted,
    ``mcp_constraints`` must be null (HTTP-only).

    ``http_manifest`` is the sanitized agent sitemap (from ``sanitize_sitemap``).
    When given and an endpoint declares ``body_fields``, every ``fields[]`` path
    in branches covering that domain must use a declared field name — no guessing.
    Endpoints with no ``body_fields`` do not trigger this check (schema unknown).
    """
    errors: list[str] = []
    skeleton_keys = skeleton.all_keys()
    annotated_keys = set(constraints.branches.keys())
    mcp_tools = mcp_tools or {}
    mcp_mode = bool(mcp_tools)

    for key in sorted(annotated_keys - skeleton_keys):
        errors.append(
            f"Branch key '{key}' does not exist in the plan skeleton. "
            f"Valid keys: {sorted(skeleton_keys)}."
        )
    for key in sorted(skeleton_keys - annotated_keys):
        errors.append(f"Branch key '{key}' is missing: every skeleton key must be annotated.")

    for key, branch in constraints.branches.items():
        _check_in_sets(key, branch, errors)
        _check_from_plan_fields(key, branch, errors)
        _check_var_placeholders(key, branch, skeleton, errors)
        if http_manifest:
            _check_http_fields_against_manifest(key, branch, http_manifest, errors)
            _check_allowed_endpoints_against_manifest(key, branch, http_manifest, errors)
        if mcp_mode:
            arm = skeleton.arms.get(key)
            called = skeleton.root_called_functions if key == ROOT_KEY else (arm.called_functions if arm else [])
            has_body = True if key == ROOT_KEY else (arm.has_body if arm else True)
            _check_mcp(key, branch, called, has_body, mcp_tools, skeleton, errors)
        else:
            _check_no_mcp(key, branch, errors)
        if key == ROOT_KEY:
            _check_domains(key, branch, errors)
            _check_root_static_domains(branch, skeleton, errors)
            continue
        arm = skeleton.arms.get(key)
        if arm is None:
            continue  # already reported above

        if (
            branch.trigger_var is not None
            and arm.trigger_var is not None
            and branch.trigger_var != arm.trigger_var
        ):
            errors.append(
                f"Branch '{key}': trigger_var '{branch.trigger_var}' does not match "
                f"the condition variable '{arm.trigger_var}' extracted from the AST."
            )

        if not arm.has_body and _authorises_something(branch):
            errors.append(
                f"Branch '{key}' has no executable body (implicit empty arm) and must "
                "be fail-closed: empty allowed_domains, empty fields, null mcp_constraints."
            )

        if http_manifest:
            _check_trigger_var_pinned(key, branch, arm, http_manifest, errors, skeleton)

        missing = [d for d in arm.static_domains if d not in branch.http_constraints.allowed_domains]
        if arm.has_body and missing:
            errors.append(
                f"Branch '{key}': domains {missing} appear literally in the branch body "
                "but are not in allowed_domains. Add them or the branch will block its own traffic."
            )

        _check_domains(key, branch, errors)

    return errors


def _check_http_fields_against_manifest(
    key: str,
    branch: BranchConstraints,
    manifest: HttpManifest,
    errors: list[str],
) -> None:
    """If manifest endpoints covering the branch's domains declare body_fields,
    every ``fields[]`` path must use a declared field name.

    The check is intentionally lazy: if no endpoint in the manifest both (a)
    covers a domain authorised by this branch and (b) declares body_fields, the
    check is skipped entirely — we do not know the wire schema, so we cannot
    validate it.  This means the current GitLab sitemap (body: {}) adds domain
    guidance without triggering false positives.  A sitemap with explicit body
    schemas (e.g. mock_shop) gets full field-name enforcement.
    """
    allowed = set(branch.http_constraints.allowed_domains)
    if not branch.http_constraints.fields:
        return

    # Collect body_fields from all manifest endpoints whose domain is allowed.
    known_fields: set[str] = set()
    for ep in manifest:
        if ep.domain in allowed and ep.body_fields:
            known_fields.update(ep.body_fields)

    # No schema info available for the authorised domains → nothing to enforce.
    if not known_fields:
        return

    for fc in branch.http_constraints.fields:
        root_name = fc.path.split(".")[0]
        if root_name not in known_fields:
            errors.append(
                f"Branch '{key}': field path '{fc.path}' is not declared in the "
                f"HTTP manifest for this branch's domains "
                f"(known fields: {sorted(known_fields)}). "
                "Use only field names from the manifest — do not guess wire schema."
            )


# The scalar comparison ops a FieldConstraint can express. `in`/`subset`/
# `eq_struct` never arise from an `if x <op> const` test, so they are irrelevant
# here. Anything outside this set — notably the STRICT ops — is deliberately
# inexpressible in the constraint language, which is exactly
# why the under-pin guard below must not demand it.
_ENFORCEABLE_ASSERTION_OPS: frozenset[str] = frozenset({"<=", ">=", "=="})

# What an arm asserts is the condition on the true arm and its NEGATION on the
# false one. Kept explicit (rather than "true arm only") because plans routinely
# put the guarded action in the else: `if amount > 1000: review() else: pay()`
# asserts `amount <= 1000` on the arm that pays, and that bound must still be
# pinned.
_NEGATED_OP: dict[str, str] = {
    "<=": ">", ">": "<=", ">=": "<", "<": ">=", "==": "!=", "!=": "==",
}


def _arm_assertion(arm) -> tuple[str, object] | None:
    """The bound this arm establishes about its trigger variable, or None.

    None means "this arm asserts nothing a FieldConstraint could carry", and the
    two ways that happens are the whole point of this function:

      * the test is not a simple ``<var> <op> <literal>`` (so we cannot name the
        bound at all), or the tested operand is not the bare trigger variable;
      * the arm's assertion needs a STRICT op — e.g. the false arm of
        ``amount <= 1000`` asserts ``amount > 1000``. The constraint language has
        no ``>``; demanding a pin there asks the annotator for something the
        language cannot represent, while rule 6 forbids inventing one. That
        contradiction is unsatisfiable, and an unsatisfiable check does not
        fail closed here — it burns the retries and drops the WHOLE annotation
        to the domain-only fallback, which is fail-OPEN on precisely the field
        layer this guard exists to protect.
    """
    cmp_ = getattr(arm, "condition_comparison", None)
    if not cmp_ or cmp_.get("operand") != arm.trigger_var:
        return None
    op = cmp_["op"] if arm.is_true_arm else _NEGATED_OP.get(cmp_["op"])
    if op not in _ENFORCEABLE_ASSERTION_OPS:
        return None
    return op, cmp_["literal"]


def _gated_wire_field(arm, known_fields: set[str], skeleton) -> str | None:
    """The wire field this arm's condition actually gates, or None.

    Two ways a plan variable can be that field:

      * it is NAMED like the field (`if amount <= 1000`) — the original rule, and the
        only one that works when the plan handles a single row;
      * it was READ from the field (`amount_0 = read_item_number(0, "amount")`), which
        is what a plan does the moment it handles more than one row and has to index
        its variables. `skeleton.var_field_reads` carries that binding.

    Conservative in the same way as before: if neither says which field is gated, the
    guard stays silent rather than guessing at a wire schema."""
    if arm.trigger_var in known_fields:
        return arm.trigger_var
    reads = getattr(skeleton, "var_field_reads", {}) or {}
    hit = [lit for lit in reads.get(arm.trigger_var, ()) if lit in known_fields]
    return hit[0] if len(hit) == 1 else None


def missing_trigger_pin(branch: BranchConstraints, arm, manifest: HttpManifest,
                        skeleton=None):
    """The pin a branch owes for its own gating condition, or None if it owes none.

    Single source of truth for the under-pin guard: :func:`_check_trigger_var_pinned`
    turns it into a retry error, and :func:`repair_trigger_pins` turns it into the
    deterministic last-resort fix. They can therefore never disagree about what is
    missing — the failure mode that made the guard destructive in the first place.

    Fires ONLY when all of the following hold, each one a deliberate narrowing:
      * the arm has a body and gates on a variable;
      * the arm's assertion is expressible (:func:`_arm_assertion`);
      * the trigger variable NAME literally matches a ``body_field`` declared by a
        manifest endpoint on one of the branch's allowed domains — a strong signal
        that the tested value IS the wire field being gated. When the plan's
        variable name differs from the wire field name (the common case in other
        benchmarks) the check is skipped entirely;
      * nothing already pins that path.
    """
    if not arm.has_body or arm.trigger_var is None:
        return None
    assertion = _arm_assertion(arm)
    if assertion is None:
        return None
    allowed = set(branch.http_constraints.allowed_domains)
    known_fields: set[str] = set()
    for ep in manifest:
        if ep.domain in allowed and ep.body_fields:
            known_fields.update(ep.body_fields)
    field = _gated_wire_field(arm, known_fields, skeleton)
    if field is None:
        return None  # nothing ties the tested value to a declared wire field
    op, literal = assertion
    pins = [fc for fc in branch.http_constraints.fields
            if fc.path.split(".")[0] == field]
    # What the branch owes is the PERCEIVED value, not the policy constant. Entering
    # `if amount <= 250000:` already established `amount <= 250000`; re-asserting that on
    # the wire authorises every other value that also satisfies it — which is exactly the
    # branch-steering substitution this system exists to catch: the wire payload is
    # compared against the perceived value. Only `trigger_value` binds the
    # wire to what the plan actually branched on.
    #
    # Measured on `101-settlement-desk` (STEER-Bench S9 ULTRA): the row displayed
    # 120000 and the wire carried 240000 — inside the policy ceiling, outside the
    # perceived value — and gpt-5 had pinned the ceiling at all four levels of the tree.
    # The same annotator on S1's single-gate plans emits `<= trigger_value` and blocks,
    # so what failed was not the model but this guard, which used to accept either and
    # ask for the weaker one when it repaired.
    #
    # A pin is accepted iff it binds a RUNTIME value (`trigger_value` / `var:`) or is a
    # literal STRICTER than the arm's own bound — a real constraint the annotator chose
    # to add, which must be left alone.
    if any(is_placeholder(fc.value) or var_placeholder_name(fc.value) is not None
           or (fc.op == op and not _no_stronger_than(fc.value, op, literal))
           for fc in pins):
        return None
    demand = {"path": field, "op": op, "value": TRIGGER_VALUE_PLACEHOLDER}
    if pins:
        # Diagnostic only — never part of the constraint. Its presence tells the guard's
        # message that a bound IS there and it is the wrong one, which is a different
        # thing to say than "you dropped the bound".
        demand["vacuous_literal"] = pins[0].value
    return demand


def _no_stronger_than(value, op: str, literal) -> bool:
    """Is `value` no stronger a bound than the one the arm itself asserts?

    True means the pin adds nothing to entering the branch, i.e. it is vacuous."""
    try:
        v, lit = float(value), float(literal)
    except (TypeError, ValueError):
        return value == literal
    if op == "<=":
        return v >= lit
    if op == ">=":
        return v <= lit
    return v == lit


def repair_trigger_pins(
    constraints: PlanConstraints,
    skeleton: PlanSkeleton,
    manifest: HttpManifest,
) -> list[str]:
    """Add every missing gating pin in place; return one description per repair.

    The escape hatch for the guard's one genuinely bad outcome. A bound the branch
    *itself* establishes is not a judgement call — it is derivable from the AST —
    so when the annotator will not restate it, deriving it is strictly better than
    discarding a whole annotation and running the wire on domain-only constraints.
    Used only as a last resort, after the retries (`annotator.annotate`).

    Provenance note, because it is what separates this from the harness field
    seed: the injected value is the literal the PLAN tested, not a value supplied
    by the benchmark. The chain stays plan-derived.
    """
    repairs: list[str] = []
    for key, branch in constraints.branches.items():
        arm = skeleton.arms.get(key)
        if arm is None:
            continue
        pin = missing_trigger_pin(branch, arm, manifest, skeleton)
        if pin is None:
            continue
        # `vacuous_literal` is diagnostic, not part of the constraint. Its presence means
        # a policy bound is already on this path and only the perceived-value binding is
        # missing: APPEND rather than replace, because constraints on one path are
        # conjunctive and the policy ceiling is still worth holding.
        vacuous = pin.pop("vacuous_literal", None)
        branch.http_constraints.fields.append(FieldConstraint(**pin))
        why = (f" (the literal {vacuous!r} already there bounds nothing the branch did "
               f"not already establish)" if vacuous is not None else "")
        repairs.append(f"{key}: pinned {pin['path']} {pin['op']} {pin['value']!r}{why}")
    return repairs


def _check_trigger_var_pinned(
    key: str,
    branch: BranchConstraints,
    arm,
    manifest: HttpManifest,
    errors: list[str],
    skeleton=None,
) -> None:
    """Fail-closed guard against *under-pinning* a gating condition.

    Counterpart to rule 6's minimalism: rule 6 keeps the annotator from inventing
    constraints, but nothing forced it to KEEP a bound that the branch's own
    condition establishes — so a branch like ``if dosage <= 50: submit_action()``
    could be annotated with ``fields: []`` and leave ``dosage`` ungated on the
    wire.

    A surviving error feeds the annotator retry loop; if the pin is still missing
    after the retries, :func:`repair_trigger_pins` supplies it deterministically
    rather than letting the annotation collapse.
    """
    pin = missing_trigger_pin(branch, arm, manifest, skeleton)
    if pin is None:
        return
    if "vacuous_literal" in pin:
        errors.append(
            f"Branch '{key}': field '{pin['path']}' is pinned to "
            f"{pin['vacuous_literal']!r} — the same bound entering this branch already "
            f"established, so it authorises every other value that also satisfies it. "
            f"Pin the value the branch actually tested: add "
            f'{{"path": "{arm.trigger_var}", "op": "{pin["op"]}", "value": '
            f'"{TRIGGER_VALUE_PLACEHOLDER}"}} (rule 3a). Keep the literal beside it if '
            f"the policy bound also matters — constraints on one path are combined with "
            f"AND.")
        return
    errors.append(
        f"Branch '{key}': the gating condition tests '{arm.trigger_var}', which is the "
        f"wire field '{pin['path']}' (declared as a body_field on this branch's "
        f"domains), but no fields[] entry bounds it. A branch that gates on a value the "
        f"wire sends MUST pin that value — add a fields[] entry for '{pin['path']}' "
        f"(op with \"trigger_value\"/\"var:<name>\", or a literal per rules 3/3a/3b). "
        f"Rule 6's minimalism does not license dropping a bound the branch itself tests."
    )


def _check_allowed_endpoints_against_manifest(
    key: str,
    branch: BranchConstraints,
    manifest: HttpManifest,
    errors: list[str],
) -> None:
    """Each allowed_endpoints entry must reference a manifest endpoint for an
    allowed domain. Prevents the P-LLM from inventing endpoint patterns."""
    allowed_ep = branch.http_constraints.allowed_endpoints
    if not allowed_ep:
        return
    allowed_domains = set(branch.http_constraints.allowed_domains)
    valid: set[tuple[str, str, str]] = {
        (ep.method, ep.domain, ep.path_template)
        for ep in manifest
        if ep.domain in allowed_domains
    }
    for ep in allowed_ep:
        if (ep.method, ep.domain, ep.path_pattern) not in valid:
            errors.append(
                f"Branch '{key}': allowed_endpoint {ep.method} {ep.domain}{ep.path_pattern!r} "
                "is not in the HTTP manifest for this branch's domains. "
                f"Valid entries: {sorted(valid)}."
            )


def _authorises_something(branch: BranchConstraints) -> bool:
    return bool(
        branch.http_constraints.allowed_domains
        or branch.http_constraints.fields
        or branch.http_constraints.allowed_endpoints
        or (branch.mcp_constraints and branch.mcp_constraints.allowed_tools)
    )


def _check_in_sets(key: str, branch: BranchConstraints, errors: list[str]) -> None:
    """Semantic checks on `op: "in"` sets (structure is enforced by the schema)."""
    for fc in branch.http_constraints.fields:
        if fc.op != "in":
            continue
        members = fc.value
        placeholders = [m for m in members if isinstance(m, str) and m in PLACEHOLDERS]
        if placeholders:
            errors.append(
                f"Branch '{key}', field '{fc.path}': placeholders {placeholders} are not "
                "allowed inside an 'in' set. Use a comparison op with the placeholder instead."
            )
        var_members = [m for m in members if var_placeholder_name(m) is not None]
        if var_members:
            errors.append(
                f"Branch '{key}', field '{fc.path}': variable placeholders {var_members} "
                "are not allowed inside an 'in' set. Use a comparison op with the "
                "placeholder instead."
            )
        member_kinds = {
            "number" if isinstance(m, (int, float)) and not isinstance(m, bool) else "string"
            for m in members
        }
        if len(member_kinds) > 1:
            errors.append(
                f"Branch '{key}', field '{fc.path}': 'in' set members must all be "
                "of the same type (all strings or all numbers)."
            )
        if len(members) > MAX_IN_SET_SIZE:
            errors.append(
                f"Branch '{key}', field '{fc.path}': 'in' set has {len(members)} members "
                f"(max {MAX_IN_SET_SIZE}). A set this large is not a meaningful constraint."
            )


def _check_from_plan_fields(key: str, branch: BranchConstraints, errors: list[str]) -> None:
    """An HTTP field pinned to the bare ``"from_plan"`` marker authorises everything.

    The marker resolves from nothing: ``hook._merge_constraints`` cannot bind it at
    branch entry, so it either compares the real value against the literal string
    ``"from_plan"`` (refusing the honest request) or — as it now does, to avoid that
    false block — is DROPPED, leaving the path unpinned. Both readings
    are wrong and the second is worse: a field the annotator believed it had
    constrained reaches the wire with no bound at all, fail-OPEN, and nothing in the
    artefacts says so.

    Rule 9a already forbids the same form for MCP params, for the same reason and in
    the same words; this is the missing HTTP half of that rule. The check lives in the
    validator rather than in the system prompt deliberately: the annotation loop
    re-asks with this message attached, so an annotation that never emits the marker
    sees a byte-identical prompt and no other suite's number can move."""
    for fc in branch.http_constraints.fields:
        if fc.value == FROM_PLAN_PLACEHOLDER:
            errors.append(
                f"Branch '{key}', field '{fc.path}': \"from_plan\" is not a usable value "
                "for an HTTP field. Unlike \"trigger_value\" and \"var:<name>\" it resolves "
                "from nothing at runtime, so the field ends up with NO bound at all. "
                "Write the LITERAL the task fixes (e.g. the value the stated policy "
                "names) — you wrote the plan, so you know it. If the bound is a runtime "
                "value, use \"trigger_value\" when it IS the branch's trigger, or "
                "\"var:<name>\" for another plan variable; but note that at \"root\" "
                "neither resolves either, so a plan that never branches must pin a literal."
            )


def _check_var_placeholders(
    key: str, branch: BranchConstraints, skeleton: PlanSkeleton, errors: list[str]
) -> None:
    """A ``"var:<name>"`` field bound must name a variable the plan assigns.

    The hook resolves it from the namespace; a name the plan never defines
    would resolve to nothing and silently fail-closed, so reject it here and
    let the annotator correct it (or use ``trigger_value``)."""
    for fc in branch.http_constraints.fields:
        if fc.op == "in":
            continue  # 'in' members handled by _check_in_sets
        name = var_placeholder_name(fc.value)
        if name is not None and name not in skeleton.variable_names:
            errors.append(
                f"Branch '{key}', field '{fc.path}': value '{fc.value}' references "
                f"variable '{name}', which the plan never assigns. Known variables: "
                f"{sorted(skeleton.variable_names)}."
            )


def _check_mcp(
    key: str,
    branch: BranchConstraints,
    called_functions: list[str],
    has_body: bool,
    mcp_tools: McpManifest,
    skeleton: PlanSkeleton,
    errors: list[str],
) -> None:
    """Validates ``mcp_constraints`` against the approved tool manifest.

    Mirrors the HTTP checks: a tool may only be authorised if it is in the
    manifest (no invented tools), a branch that *calls* a manifest tool must
    authorise it (or it blocks its own calls — the MCP analogue of the
    static-domain cross-check), and each ``param_rule`` must target an allowed
    tool and a parameter that tool actually declares. ``var:`` placeholders are
    checked against the plan's variables, as for fields."""
    manifest_tools = set(mcp_tools.keys())
    mcp = branch.mcp_constraints
    authorised = set(mcp.allowed_tools) if mcp else set()

    for tool in sorted(authorised - manifest_tools):
        errors.append(
            f"Branch '{key}': allowed_tool '{tool}' is not in the MCP tool manifest "
            f"{sorted(manifest_tools)}."
        )

    if has_body:
        missing = list(dict.fromkeys(
            t for t in called_functions if t in manifest_tools and t not in authorised
        ))
        if missing:
            errors.append(
                f"Branch '{key}': MCP tools {missing} are called in this branch but not "
                "listed in allowed_tools (the branch would block its own tool calls)."
            )

    if mcp is None:
        return
    for rule in mcp.param_rules:
        if rule.tool not in authorised:
            errors.append(
                f"Branch '{key}': param_rule targets tool '{rule.tool}', which is not in "
                f"allowed_tools {sorted(authorised)}."
            )
        elif rule.param not in (mcp_tools.get(rule.tool) or []):
            errors.append(
                f"Branch '{key}': param_rule references parameter '{rule.param}' which tool "
                f"'{rule.tool}' does not declare (params: {list(mcp_tools.get(rule.tool) or [])})."
            )
        if rule.op is None and rule.source is None:
            errors.append(
                f"Branch '{key}': param_rule for '{rule.tool}.{rule.param}' must specify an "
                "op (<=|>=|==) or source='from_plan'."
            )
        if rule.op is not None and rule.value is None:
            errors.append(
                f"Branch '{key}': param_rule for '{rule.tool}.{rule.param}' has op "
                f"'{rule.op}' but no value."
            )
        name = var_placeholder_name(rule.value)
        if name is not None and name not in skeleton.variable_names:
            errors.append(
                f"Branch '{key}': param_rule value '{rule.value}' references variable "
                f"'{name}', which the plan never assigns. Known variables: "
                f"{sorted(skeleton.variable_names)}."
            )

    for tool, params in (mcp.allowed_params or {}).items():
        if tool not in authorised:
            errors.append(
                f"Branch '{key}': allowed_params seals tool '{tool}', which is not in "
                f"allowed_tools {sorted(authorised)}."
            )
            continue
        declared = mcp_tools.get(tool) or []
        unknown = [p for p in params if p not in declared]
        if unknown:
            errors.append(
                f"Branch '{key}': allowed_params for '{tool}' lists parameter(s) {unknown} "
                f"that the tool does not declare (params: {list(declared)})."
            )

    for tool in (mcp.allowed_tool_servers or {}):
        if tool not in authorised:
            errors.append(
                f"Branch '{key}': allowed_tool_servers pins tool '{tool}', which is not in "
                f"allowed_tools {sorted(authorised)}."
            )


def _check_no_mcp(key: str, branch: BranchConstraints, errors: list[str]) -> None:
    """Without a manifest, no MCP tool is approved, so mcp_constraints must be
    null (HTTP-only annotation)."""
    mcp = branch.mcp_constraints
    if mcp is not None and (mcp.allowed_tools or mcp.param_rules):
        errors.append(
            f"Branch '{key}': no MCP tool manifest was provided, so mcp_constraints "
            "must be null."
        )


def _check_domains(key: str, branch: BranchConstraints, errors: list[str]) -> None:
    for domain in branch.http_constraints.allowed_domains:
        if not _HOSTNAME_RE.match(domain):
            errors.append(
                f"Branch '{key}': allowed domain '{domain}' is not a bare lowercase "
                "hostname (no scheme, no path, no port)."
            )


def _check_root_static_domains(
    branch: BranchConstraints, skeleton: PlanSkeleton, errors: list[str]
) -> None:
    missing = [
        d for d in skeleton.root_static_domains
        if d not in branch.http_constraints.allowed_domains
    ]
    if missing:
        errors.append(
            f"Branch 'root': domains {missing} appear literally outside any branch "
            "but are not in root allowed_domains."
        )


def build_fallback(skeleton: PlanSkeleton, plan_id: str, task: str) -> PlanConstraints:
    """Constraints derived from static facts only (annotation failed).

    Fail-closed in spirit: each arm only allows the domains that appear
    literally in its own body, with no field constraints. Traffic to any
    other domain is blocked by the enforcers.
    """
    branches: dict[str, BranchConstraints] = {
        ROOT_KEY: BranchConstraints(
            description="static fallback — root",
            http_constraints={"allowed_domains": list(skeleton.root_static_domains)},
        )
    }
    for key, arm in skeleton.arms.items():
        branches[key] = BranchConstraints(
            description=f"static fallback — condition `{arm.condition}` is {arm.is_true_arm}",
            trigger_var=arm.trigger_var,
            http_constraints={
                "allowed_domains": list(arm.static_domains) if arm.has_body else []
            },
        )
    return PlanConstraints(
        plan_id=plan_id, task=task, generated_by="static-fallback", branches=branches
    )
