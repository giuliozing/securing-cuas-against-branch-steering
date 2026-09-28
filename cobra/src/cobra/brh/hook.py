"""Runtime BRH hook: branch transitions → atomic `branch_state.json` writes.

This is the runtime half of the BRH. The interpreter
calls `on_branch_entry` from `_eval_if` after the condition value is
known and before either arm executes; this module resolves the static
annotations of `plan_constraints.json` against the runtime state and
writes the merged, placeholder-resolved constraints for the enforcers
(HTTP proxy, MCP proxy) to poll.

Design constraints honoured here:

- **Never raise into the interpreter.** Every public entry point
  swallows all exceptions, logs to stderr and degrades to a no-op. A
  broken hook must never corrupt plan execution (fail-safe contract).
- **No interpreter imports.** `namespace` and `eval_args` are used
  duck-typed (`namespace.get(name).raw`, `dataclasses.replace(eval_args,
  ...)`) so this module — like the rest of `cobra.brh` — stays
  importable and testable without the interpreter's dependency chain
  (pydantic_ai, agentdojo). The dependency arrow goes interpreter →
  hook only.
- **Per-arm placeholder resolution.** Each `"trigger_value"`
  placeholder is resolved with the trigger value of the *arm that
  declared it*, captured at that arm's entry time and carried in
  `BRHBranchStep` (not with the leaf's value, and not re-read from the
  namespace, where the variable may have been reassigned since).
- **Fail-closed on desync.** If a branch key on the active path is
  missing from the constraints (which the validator should make
  impossible), the state is written with an empty domain allowlist —
  the enforcers then block everything rather than trusting a
  constraints file that does not describe the running plan.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Union

from cobra.brh.schema import (
    FROM_PLAN_PLACEHOLDER,
    TRIGGER_VALUE_PLACEHOLDER,
    PlanConstraints,
    var_placeholder_name,
)
from cobra.brh.skeleton import ROOT_KEY
from cobra.brh.writer import _utc_now_iso, atomic_write_json

# Scalars accepted as a runtime trigger value. Anything else (lists,
# tool result objects, ...) becomes null: the placeholder stays
# unresolved and the enforcer treats the constraint as unsatisfiable.
TriggerValue = Union[bool, int, float, str, None]
_SCALAR_TYPES = (bool, int, float, str)


@dataclasses.dataclass(frozen=True)
class BRHBranchStep:
    """One entered arm on the current branch path (root excluded)."""

    branch_id: str
    """AST-derived arm id, e.g. ``if_L10_true``."""
    trigger_value: TriggerValue
    """The arm's trigger variable value, captured at arm entry time."""
    var_values: tuple[tuple[str, TriggerValue], ...] = ()
    """Resolved ``"var:<name>"`` field bounds of *this arm*, captured at entry.

    Carried in the step (not re-read from the namespace) so the value survives
    a pop: `on_branch_exit` rewrites a parent scope's state with no namespace in
    hand, exactly as for `trigger_value`. Stored as (name, scalar) pairs;
    name-not-found / non-scalar is omitted so the placeholder stays unresolved
    (fail-closed)."""


@dataclasses.dataclass(frozen=True)
class BRHRuntime:
    """Everything the hook needs at runtime, attached to `EvalArgs`.

    `EvalArgs.brh_constraints: PlanConstraints` is the field the interpreter
    consults; in practice the hook also needs to know *where* to write
    `branch_state.json`, so both travel together in this composite.
    """

    constraints: PlanConstraints
    state_path: Path
    # Domains whose sitemap has already been fetched (shared across branch transitions).
    # Key = domain, value = list of {method, path} dicts (or empty list = no sitemap).
    # Must be a mutable dict to accumulate across calls; frozen dataclass is OK because
    # we only mutate the dict's contents, not the attribute reference.
    _sitemap_cache: dict[str, list[dict[str, str]]] = dataclasses.field(default_factory=dict)
    # Field pins derived from values the plan DECLARED it commits at a guarded tool
    # call (see `on_tool_call`). Same mutable-content rationale as the cache above.
    # They are re-applied on every subsequent state write so a later branch
    # transition cannot silently drop them.
    _commit_pins: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def _fetch_sitemap_entries(domain: str, timeout: int = 5) -> list[dict[str, str]]:
    """Fetch ``/sitemap.json`` for ``domain`` (HTTPS-first, HTTP fallback).

    Returns a list of ``{method, path}`` dicts for the enforcer's runtime check.
    Returns empty list if the sitemap is absent or unparseable.  Never raises.
    """
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}/sitemap.json"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                data = json.loads(resp.read())
                if not isinstance(data, list):
                    return []
                entries = []
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    method = entry.get("method", "")
                    raw_url = entry.get("url", "")
                    if not isinstance(method, str) or not isinstance(raw_url, str):
                        continue
                    method = method.upper().strip()
                    # Extract path from URL (structural parse only).
                    try:
                        from urllib.parse import urlparse as _up
                        path = _up(raw_url).path or "/"
                    except Exception:
                        continue
                    if method:
                        entries.append({"method": method, "path": path})
                return entries
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            continue
    return []


def _log(message: str) -> None:
    print(f"⚠️ BRH hook: {message}", file=sys.stderr)


def attach(eval_args: Any, runtime: BRHRuntime) -> Any:
    """Returns a copy of `eval_args` with the BRH runtime attached.

    Duck-typed on purpose: works on any dataclass exposing the
    ``brh_runtime`` / ``brh_branch_path`` fields (the interpreter's
    `EvalArgs` after the BRH patch).
    """
    return dataclasses.replace(eval_args, brh_runtime=runtime, brh_branch_path=())


def activate_root(runtime: BRHRuntime | None) -> None:
    """Writes the root-active state at plan execution start.

    Between plans `branch_state.json` is reset to ``active_branch:
    null`` (block everything). Plans legitimately make requests *before*
    reaching any `if` (e.g. fetching the product page that produces the
    trigger value), and the `root` annotation exists precisely to
    authorise that traffic — so execution start is itself a branch
    transition: ``null → root``. Never raises.
    """
    if runtime is None:
        return
    try:
        _write_scope_state(runtime, steps=())
    except Exception as e:  # noqa: BLE001 — fail-safe contract
        _log(f"root activation failed ({e!r}); state not updated, enforcement degraded")


def on_branch_entry(
    lineno: int, condition_is_true: bool, namespace: Any, eval_args: Any
) -> Any:
    """Records a branch transition; returns the eval_args for the chosen body.

    Called by `_eval_if` with the `ast.If` line number and the evaluated
    condition. Writes `branch_state.json` (merged ancestor constraints,
    placeholders resolved) and returns a *new* eval_args whose
    ``brh_branch_path`` includes the entered arm — the caller passes it
    to the body's `_eval_stmt_list` so nested ``if``s see the extended
    path while siblings keep the parent's. No-op (returns `eval_args`
    unchanged) when no runtime is attached. Never raises.
    """
    runtime: BRHRuntime | None = getattr(eval_args, "brh_runtime", None)
    if runtime is None:
        return eval_args
    try:
        return _on_branch_entry(lineno, condition_is_true, namespace, eval_args, runtime)
    except Exception as e:  # noqa: BLE001 — fail-safe contract
        _log(
            f"branch transition at line {lineno} failed ({e!r}); "
            "state not updated, enforcement degraded"
        )
        return eval_args


def on_branch_exit(eval_args: Any) -> None:
    """Restores the parent scope's state when execution leaves a branch.

    `on_branch_entry` only ever *pushes* state: it writes the entered
    arm's constraints and never reverts them. Without a matching pop,
    `branch_state.json` would stay frozen on the last-entered arm after
    the `if` block ends — so any request issued in the parent scope
    afterwards (a sibling `if`, a statement after the block, the rest of
    an enclosing arm) would be enforced against the child's accumulated
    (super-set) constraints: silent over-authorisation.

    `_eval_if` calls this on every exit path of an `if` it entered,
    passing the *parent* `eval_args` (whose `brh_branch_path` is the
    scope being returned to — `()` at top level, i.e. root). The matching
    scope state is recomputed and rewritten, so the disk state always
    reflects the current lexical scope. No-op without a runtime; never
    raises (fail-safe contract).
    """
    runtime: BRHRuntime | None = getattr(eval_args, "brh_runtime", None)
    if runtime is None:
        return
    try:
        parent_steps: tuple[BRHBranchStep, ...] = tuple(getattr(eval_args, "brh_branch_path", ()))
        _write_scope_state(runtime, steps=parent_steps)
    except Exception as e:  # noqa: BLE001 — fail-safe contract
        _log(f"branch exit failed ({e!r}); state not restored, enforcement degraded")


# ---------------------------------------------------------------------------
# Commit gate (provenance at the tool boundary)
# ---------------------------------------------------------------------------
#
# The branch hook above answers "which constraints are active?". This answers a
# different question the branch hook structurally cannot: "may this value be
# COMMITTED to an irreversible action at all?".
#
# Motivation. A guarded tool that takes no arguments
# keeps the branch-steering premise honest — the wire body is resolved
# server-side, so perceived≠real stays invisible to the agent — but it also means
# the committed value never crosses the interpreter, so provenance can only be
# enforced by the planner obeying prose. Measured consequence: the moment another
# prompt fragment contradicts the rule, the plan commits a value read off the
# (untrusted) record and nothing underneath stops it.
#
# The discriminator needed already exists in the dual-LLM design and is only
# routed here: a literal written by the P-LLM must come from the prompt (the
# P-LLM is blind to the screen by invariant) so it carries `User`/`CaMeL`
# sources; a value read through a tool carries `Tool(<name>)` with no inner
# sources, which `capabilities.utils.is_trusted` already rejects.
#
# Two effects, both fail-closed:
#   1. a commitment that is missing or untrusted DENIES the call (`brh_provenance`);
#   2. a commitment that passes is PINNED onto the wire (`==`), so a value that is
#      trusted but wrong (a hallucinated literal) is still caught by the HTTP
#      enforcer. This is what lets the harness's oracle field seed be dropped: the
#      pin now derives from the plan's own trusted literal.
#
# Opt-in per deployment via `BRH_COMMIT_GATE` (JSON, see `_commit_gate_config`).
# Unset — the default everywhere except the STEER-Bench specification axis — makes
# `on_tool_call` return None before doing any work, so no other benchmark changes
# behaviour or pays measurable cost.

_GATE_CACHE: tuple[str, dict[str, Any]] | None = None


def _commit_gate_config() -> dict[str, Any]:
    """Parse ``BRH_COMMIT_GATE``; ``{}`` (gate off) on anything malformed.

    Shape — one entry per guarded tool::

        {"submit_action": {"container": "committed",
                           "require": ["account"],
                           "pins": {"account": "account"}}}

    ``container``  argument holding a mapping of commitments (omit: the call's own
                   keyword arguments are the commitments);
    ``require``    commitment keys that MUST be present (a plan that declares
                   nothing must not be able to slip past the gate by declaring
                   nothing — the empty declaration is the obvious bypass);
    ``pins``       commitment key -> wire body dot-path pinned ``==`` on success.
    """
    global _GATE_CACHE
    import os

    raw = os.environ.get("BRH_COMMIT_GATE", "")
    if not raw.strip():
        return {}
    if _GATE_CACHE is not None and _GATE_CACHE[0] == raw:
        return _GATE_CACHE[1]
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            parsed = {}
    except json.JSONDecodeError:
        _log("BRH_COMMIT_GATE is not valid JSON; commit gate disabled")
        parsed = {}
    _GATE_CACHE = (raw, parsed)
    return parsed


def _alerts_path(runtime: BRHRuntime) -> Path:
    """Alerts live next to the state file — the same directory the enforcers are
    pointed at by the harness, so a tool-channel block lands in the same JSONL as
    the HTTP and MCP ones and is read with the same code."""
    return Path(runtime.state_path).parent / "brh_alerts.jsonl"


def _alert(runtime: BRHRuntime, reason: str, detail: dict[str, Any],
           *, kind: str = "brh_tool_block", channel: str = "tool") -> None:
    """Append one tool-channel record to the shared alerts JSONL.

    Two kinds, on two channels, and the separation is load-bearing. `brh_tool_block`
    on channel `tool` is a DENIAL and is what every consumer counts as a gate firing.
    `brh_tool_audit` on channel `tool_audit` is an ACCEPTANCE — the gate was consulted
    and let the call through. It exists because "no denial" had two meanings that no
    artefact could tell apart: the gate accepted, or the gate was never reached because
    the plan stopped earlier. The audit record is what makes gate REACHABILITY
    measurable instead of inferred. It must never land on channel `tool`, or an accepted call would be
    read as a block and score GATED."""
    try:
        payload = {
            "kind": kind,
            "channel": channel,
            "reason": reason,
            "plan_id": runtime.constraints.plan_id,
            "detail": detail,
            "ts": _utc_now_iso(),
        }
        path = _alerts_path(runtime)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:  # noqa: BLE001 — alerting must never break the run
        _log(f"could not write tool alert ({e!r})")


def _commitments(arg_values: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The (key -> CaMeLValue) map this call declares it commits.

    Duck-typed on the container: a CaMeL dict exposes its element values under
    ``_python_value`` keyed by CaMeL strings. Reading ``.raw`` instead would throw
    the capability metadata away — which is the entire signal — so the raw path is
    used only for the key names."""
    container = spec.get("container")
    if not container:
        return dict(arg_values)
    # `make_args_by_keyword_preserve_values` keys POSITIONAL arguments "0", "1", …
    # so `submit_action({...})` and `submit_action(committed={...})` arrive
    # differently. Both are legitimate plans, so both must be read — otherwise the
    # gate would deny a perfectly obedient plan for its calling convention.
    holder = arg_values.get(container, arg_values.get("0"))
    if holder is None:
        return {}
    inner = getattr(holder, "_python_value", None)
    if isinstance(inner, dict):
        return {(k.raw if hasattr(k, "raw") else k): v for k, v in inner.items()}
    return {}


def on_tool_call(tool_name: str, arg_values: dict[str, Any], eval_args: Any) -> str | None:
    """Provenance check at the tool boundary. Returns a denial reason or None.

    Called by the interpreter's `_eval_call` after arguments are evaluated and
    before the callable runs, with the CaMeL *values* (not raws) so capability
    metadata is still attached. Returning a string aborts the call.

    Fail-safe like the rest of this module — it never raises into the interpreter
    — but NOT fail-open: an internal error while checking a tool the operator
    explicitly put under the gate denies the call (`brh_gate_error`). A tool that
    is not in the config is allowed without inspection, which is what keeps this
    inert for every deployment that does not set `BRH_COMMIT_GATE`.
    """
    config = _commit_gate_config()
    if not config or tool_name not in config:
        return None
    runtime: BRHRuntime | None = getattr(eval_args, "brh_runtime", None)
    spec = config[tool_name] or {}
    try:
        from cobra.capabilities.utils import is_trusted  # local: no interpreter import at module load

        declared = _commitments(arg_values or {}, spec)
        missing = [k for k in (spec.get("require") or []) if k not in declared]
        if missing:
            detail = {"tool": tool_name, "missing": missing, "declared": sorted(declared)}
            if runtime is not None:
                _alert(runtime, "brh_provenance", detail)
            return (f"provenance: {tool_name} must declare {missing} — a value you commit "
                    f"must be given in your instructions; ask for it instead of acting")
        untrusted = []
        for key, val in declared.items():
            try:
                ok = is_trusted(val)
            except Exception:  # noqa: BLE001 — an uninspectable value is not trusted
                ok = False
            if not ok:
                untrusted.append(key)
        if untrusted:
            detail = {"tool": tool_name, "untrusted": untrusted}
            if runtime is not None:
                _alert(runtime, "brh_provenance", detail)
            return (f"provenance: {tool_name} was given {untrusted} from an untrusted "
                    f"source (the on-file record / the screen). A committed value must "
                    f"come from your instructions; ask for it instead of acting")
        if runtime is not None:
            # Accepted: audit BEFORE pinning, so a cell that dies inside `_apply_
            # commit_pins` still records that the gate was reached and said yes.
            _alert(runtime, "brh_commit_ok",
                   {"tool": tool_name, "declared": sorted(declared)},
                   kind="brh_tool_audit", channel="tool_audit")
            _apply_commit_pins(runtime, declared, spec, eval_args)
        return None
    except Exception as e:  # noqa: BLE001 — fail-CLOSED: the operator asked for this gate
        _log(f"commit gate failed on '{tool_name}' ({e!r}); denying fail-closed")
        return f"provenance: commit gate error on {tool_name} ({e!r})"


def _apply_commit_pins(runtime: BRHRuntime, declared: dict[str, Any],
                       spec: dict[str, Any], eval_args: Any) -> None:
    """Turn accepted commitments into ``==`` wire pins and republish the state.

    The pin is what makes the guarantee end-to-end rather than merely procedural:
    provenance decides whether the value may be committed, the pin decides that the
    request actually carries *that* value. A trusted-but-wrong literal (the planner
    inventing a plausible IBAN) is therefore caught by the HTTP enforcer, not
    waved through by the gate.

    Written both into the runtime (so later branch transitions re-emit them) and
    into the current state file (so the pin is live for the call happening *now* —
    and so a harness that wrote the state by hand, without a PlanConstraints, is
    still covered)."""
    pins = spec.get("pins") or {}
    if not pins:
        return
    added: list[dict[str, Any]] = []
    for key, path in pins.items():
        if key not in declared:
            continue
        raw = getattr(declared[key], "raw", declared[key])
        # Op by value shape, because the enforcers' constraint language says so:
        # `satisfies` refuses `==` outright when the pinned value is a container
        # (`brh_check.satisfies`: `if isinstance(value, list): return False`) and
        # routes structural equality through `eq_struct`. Emitting `==` for a list
        # would therefore not be "a strict pin" but an UNSATISFIABLE one — it would
        # block the honest wire too, which is the false-positive shape this gate
        # exists to avoid.
        op = "eq_struct" if isinstance(raw, (list, tuple, dict)) else "=="
        if isinstance(raw, tuple):
            raw = list(raw)
        entry = {"path": path, "op": op, "value": raw}
        if entry not in runtime._commit_pins:
            runtime._commit_pins.append(entry)
        added.append(entry)
    if not added:
        return
    steps: tuple[BRHBranchStep, ...] = tuple(getattr(eval_args, "brh_branch_path", ()))
    leaf_key = ".".join(s.branch_id for s in steps) if steps else ROOT_KEY
    # Only recompute the scope state when the constraints actually describe this
    # scope. `_write_scope_state` does not raise on desync — it writes the
    # fail-closed state — so calling it blindly would turn "the plan committed a
    # trusted value" into "every request is now blocked", i.e. a self-inflicted
    # false positive on the benign leg. Otherwise patch the file in place.
    if leaf_key in runtime.constraints.branches:
        _write_scope_state(runtime, steps=steps)
    else:
        _patch_state_fields(runtime, added)


def _patch_state_fields(runtime: BRHRuntime, entries: list[dict[str, Any]]) -> None:
    """Append pins to the state file as it stands (no PlanConstraints needed)."""
    try:
        state = json.loads(Path(runtime.state_path).read_text())
    except Exception:  # noqa: BLE001
        return
    http = state.get("http_constraints")
    if not isinstance(http, dict):
        return
    fields = list(http.get("fields") or [])
    for e in entries:
        if e not in fields:
            fields.append(e)
    http["fields"] = fields
    atomic_write_json(Path(runtime.state_path), state)


def _on_branch_entry(
    lineno: int,
    condition_is_true: bool,
    namespace: Any,
    eval_args: Any,
    runtime: BRHRuntime,
) -> Any:
    branch_id = f"if_L{lineno}_{'true' if condition_is_true else 'false'}"
    parent_steps: tuple[BRHBranchStep, ...] = tuple(getattr(eval_args, "brh_branch_path", ()))
    leaf_key = ".".join([*(s.branch_id for s in parent_steps), branch_id])

    leaf = runtime.constraints.branches.get(leaf_key)
    trigger_var = leaf.trigger_var if leaf is not None else None
    trigger_value = _extract_trigger_value(namespace, trigger_var)
    var_values = _extract_var_values(namespace, leaf)

    steps = (*parent_steps, BRHBranchStep(branch_id, trigger_value, var_values))
    _write_scope_state(runtime, steps=steps)
    return dataclasses.replace(eval_args, brh_branch_path=steps)


def _write_scope_state(runtime: BRHRuntime, *, steps: tuple[BRHBranchStep, ...]) -> None:
    """Writes `branch_state.json` for the scope identified by `steps`.

    The single write path shared by root activation, branch entry and
    branch exit: it derives the leaf key (dot-joined arm ids, or
    `ROOT_KEY` when empty), the leaf's `trigger_var` (from the
    constraints, `None` on desync) and the scope's `trigger_value` (the
    last step's captured value), then delegates to `_write_state`. The
    per-arm captured `trigger_value`s already live in the steps, so a
    scope's state is fully reproducible from its path alone — exit needs
    no re-extraction from the namespace.
    """
    if steps:
        leaf_key = ".".join(s.branch_id for s in steps)
        trigger_value = steps[-1].trigger_value
    else:
        leaf_key = ROOT_KEY
        trigger_value = None
    leaf = runtime.constraints.branches.get(leaf_key)
    trigger_var = leaf.trigger_var if leaf is not None else None
    _write_state(
        runtime,
        steps=steps,
        leaf_key=leaf_key,
        trigger_var=trigger_var,
        trigger_value=trigger_value,
    )


def _update_sitemap_cache(
    runtime: BRHRuntime, merged_http: dict[str, Any]
) -> None:
    """Fetch sitemaps for domains in allowed_domains that have no allowed_endpoints entry.

    These are "runtime domains" — their endpoint semantics weren't known at plan
    annotation time, so we check them against the live sitemap instead.
    Cached per domain across branch transitions (in runtime._sitemap_cache);
    an empty list means "fetched, no sitemap found" — avoids re-fetching.
    """
    allowed_domains: list[str] = merged_http.get("allowed_domains") or []
    allowed_endpoints: list[dict] = merged_http.get("allowed_endpoints") or []
    # Domains already covered by plan-annotated allowed_endpoints.
    covered_domains: set[str] = {
        ep.get("domain", "").lower()
        for ep in allowed_endpoints
        if isinstance(ep, dict)
    }
    for domain in allowed_domains:
        domain = domain.lower()
        if domain in covered_domains or domain.startswith("*"):
            continue  # plan-annotated or wildcard — no runtime fetch needed
        if domain not in runtime._sitemap_cache:
            runtime._sitemap_cache[domain] = _fetch_sitemap_entries(domain)


def _extract_trigger_value(namespace: Any, trigger_var: str | None) -> TriggerValue:
    """Reads the trigger variable's raw scalar from the CaMeL namespace.

    The condition result in `_eval_if` is only a CaMeLTrue/CaMeLFalse;
    the value of interest lives in the namespace under `trigger_var`.
    Returns None (→ placeholders stay unresolved, enforcer fails closed
    on them) when the variable is absent or not a scalar.
    """
    if trigger_var is None:
        return None
    camel_value = namespace.get(trigger_var)
    if camel_value is None:
        _log(f"trigger variable '{trigger_var}' not in namespace; trigger_value=null")
        return None
    raw = camel_value.raw
    if isinstance(raw, _SCALAR_TYPES):
        return raw
    _log(
        f"trigger variable '{trigger_var}' is non-scalar "
        f"({type(raw).__name__}); trigger_value=null"
    )
    return None


def _extract_var_values(
    namespace: Any, branch: Any
) -> tuple[tuple[str, TriggerValue], ...]:
    """Resolves a branch's ``"var:<name>"`` bounds from the namespace.

    Returns (name, scalar) pairs for the variables this branch references by
    name (`"var:price"` → reads `price`) in *either* its HTTP field constraints
    *or* its MCP param rules — the merge resolves both channels from this same
    map, so an MCP-only plan whose `var:` bound lives only in `param_rules` must
    contribute its name too (otherwise it would never resolve → fail-closed
    false positive). A variable that is absent or non-scalar is omitted, so its
    placeholder stays unresolved (fail-closed) — exactly like an unresolved
    trigger value. Reads the *bare* `.raw` value; the data-flow provenance CaMeL
    also carries is not used here (that would be a heavier design)."""
    if branch is None:
        return ()
    names: list[str] = []
    for fc in branch.http_constraints.fields:
        if fc.op == "in":
            continue
        name = var_placeholder_name(fc.value)
        if name is not None and name not in names:
            names.append(name)
    if branch.mcp_constraints is not None:
        for rule in branch.mcp_constraints.param_rules:
            name = var_placeholder_name(rule.value)
            if name is not None and name not in names:
                names.append(name)
    out: list[tuple[str, TriggerValue]] = []
    for name in names:
        camel_value = namespace.get(name)
        raw = camel_value.raw if camel_value is not None else None
        if isinstance(raw, _SCALAR_TYPES):
            out.append((name, raw))
        else:
            _log(f"var bound '{name}' is missing or non-scalar; left unresolved")
    return tuple(out)


def _resolve(
    value: Any, trigger_value: TriggerValue, var_values: dict[str, TriggerValue]
) -> Any:
    if value == TRIGGER_VALUE_PLACEHOLDER and trigger_value is not None:
        return trigger_value
    name = var_placeholder_name(value)
    if name is not None and var_values.get(name) is not None:
        return var_values[name]
    return value


def _merge_constraints(
    constraints: PlanConstraints, steps: tuple[BRHBranchStep, ...]
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    """Merges constraints from root to leaf into one flat block.

    The enforcers read a single authoritative block, so the union is
    computed here at write time: ``allowed_domains``/``allowed_tools``
    are deduplicated unions, ``fields``/``param_rules`` concatenations
    (same-path constraints are conjunctive by contract, so concatenation
    is the correct semantics). Each arm's placeholders are resolved with
    *that arm's* captured trigger value; ``"from_plan"`` passes through
    untouched (not resolvable at branch entry). Returns None if any key
    on the path is missing from the constraints (desync → fail-closed).
    """
    lookup: list[tuple[str, TriggerValue, dict[str, TriggerValue]]] = [(ROOT_KEY, None, {})]
    key_parts: list[str] = []
    for step in steps:
        key_parts.append(step.branch_id)
        lookup.append((".".join(key_parts), step.trigger_value, dict(step.var_values)))

    missing = [key for key, _, _ in lookup if key not in constraints.branches]
    if missing:
        _log(
            f"branch keys {missing} missing from plan_constraints "
            "(plan/constraints desync); writing fail-closed state"
        )
        return None

    domains: list[str] = []
    fields: list[dict[str, Any]] = []
    allowed_endpoints: list[dict[str, str]] = []
    tools: list[str] = []
    param_rules: list[dict[str, Any]] = []
    allowed_tool_servers: dict[str, list[str]] = {}
    allowed_params: dict[str, list[str]] = {}
    has_mcp = False
    for key, trigger_value, var_values in lookup:
        branch = constraints.branches[key]
        for domain in branch.http_constraints.allowed_domains:
            if domain not in domains:
                domains.append(domain)
        for fc in branch.http_constraints.fields:
            entry = fc.model_dump(mode="json")
            # "in" sets cannot contain placeholders (validator-enforced).
            if fc.op != "in":
                entry["value"] = _resolve(entry["value"], trigger_value, var_values)
            # `from_plan` in a VALUE is unresolvable by construction — see the identical
            # rule for MCP param rules below. The
            # asymmetry was the bug: rule 9a stopped offering the form to MCP params and
            # rule 3 still offers it to HTTP fields, so the same annotation that is
            # degraded to allowlist-only on one channel false-blocks the honest request
            # on the other (e.g. STEER-Bench `77-refund-ceiling`: the policy pin AND
            # `amount == from_plan` together refused the honest refund).
            #
            # Dropping, not failing closed, for the reason stated below: the value is
            # unknowable at every branch, so fail-closed is a guaranteed false positive
            # rather than a defence, and any real pin the annotator emitted alongside it
            # still applies (constraints on one path are conjunctive).
            if entry.get("value") == FROM_PLAN_PLACEHOLDER and not entry.get("source"):
                _log(
                    f"field {entry.get('path')} pins the unresolvable marker "
                    f"{FROM_PLAN_PLACEHOLDER!r}; dropping (a literal is what pins)"
                )
                continue
            # And the ROOT `var:` case, on the same terms as the MCP param rule below:
            # root state is written once at plan start, with an empty namespace, so a
            # `var:` bound that survives resolution there is unresolvable BY
            # CONSTRUCTION. Fail-closed would compare the real value against the literal
            # string and refuse every request, honest ones included — a guaranteed false
            # positive rather than a defence. At a BRANCH a missing var is a real desync
            # and stays fail-closed; only root is dropped.
            #
            # E.g. the policy pin AND `total == "var:value_to_submit"` on the same path:
            # constraints on one path are conjunctive, so the pair refused the honest
            # claim while the policy pin alone would have admitted it.
            if key == ROOT_KEY and var_placeholder_name(entry.get("value")) is not None:
                _log(
                    f"root field {entry.get('path')} pins unresolvable "
                    f"{entry['value']!r}; dropping (root-scope var bound — a literal, or "
                    "a pin on the branch that reads it, is what pins)"
                )
                continue
            fields.append(entry)
        for ep in branch.http_constraints.allowed_endpoints:
            ep_entry = ep.model_dump(mode="json")
            if ep_entry not in allowed_endpoints:
                allowed_endpoints.append(ep_entry)
        if branch.mcp_constraints is not None:
            has_mcp = True
            for tool in branch.mcp_constraints.allowed_tools:
                if tool not in tools:
                    tools.append(tool)
            for rule in branch.mcp_constraints.param_rules:
                entry = rule.model_dump(mode="json")
                if entry.get("value") is not None:
                    entry["value"] = _resolve(entry["value"], trigger_value, var_values)
                # A `var:<name>` left unresolved at ROOT is unresolvable *by
                # construction*: root state is written once at plan start (empty
                # namespace) and loops are not branch transitions, so a value
                # bound at root (e.g. a `for` loop variable) is never captured.
                # Fail-closed there would block the benign call on every observed
                # value — a guaranteed false positive — while the tool is already
                # gated by allowed_tools / allowed_params (the cross-tool and
                # arg-add defense). So drop the rule: the param degrades to
                # allowlist-only, never to a false block. At a *branch* a missing
                # var is instead a real desync, so it stays fail-closed (below).
                # Scoped to `var:` — `from_plan`/`trigger_value` always fail-closed.
                # `from_plan` in a VALUE is unresolvable by construction, and unlike
                # `var:` it is unresolvable at every branch, not only at root: there is
                # no namespace to read it from — it is a marker meaning "the plan fixed
                # this", and the plan's own value is nowhere in the record. Left alone,
                # the enforcer compares the real argument against the literal string
                # "from_plan" and refuses the HONEST call. Same remedy as
                # the root `var:` case below — degrade the param to allowlist-only
                # rather than false-block it — and the annotator prompt (rule 9a) no
                # longer offers the form, so this is the backstop, not the fix.
                if entry.get("value") == FROM_PLAN_PLACEHOLDER and not entry.get("source"):
                    _log(
                        f"param rule {entry.get('tool')}.{entry.get('param')} pins the "
                        f"unresolvable marker {FROM_PLAN_PLACEHOLDER!r}; dropping "
                        "(allowlist-only for this param — a literal is what pins)"
                    )
                    continue
                if key == ROOT_KEY and var_placeholder_name(entry.get("value")) is not None:
                    _log(
                        f"root param rule {entry.get('tool')}.{entry.get('param')} pins "
                        f"unresolvable {entry['value']!r}; dropping (allowlist-only for "
                        "this param — root-scope var bound, e.g. a loop variable)"
                    )
                    continue
                param_rules.append(entry)
            # Sealed param sets accumulate per tool root->leaf (order-preserving
            # union), so a nested branch can only widen, never silently reopen.
            for tool, params in branch.mcp_constraints.allowed_params.items():
                acc = allowed_params.setdefault(tool, [])
                for p in params:
                    if p not in acc:
                        acc.append(p)
            # Server-qualified allowlist, same accumulation. It was MISSING from the
            # state block entirely: `writer._apply_tool_servers` injected it into
            # plan_constraints.json, the validator accepted it — and then this merge
            # dropped it, so `check_tools_call`, which reads
            # `mcp["allowed_tool_servers"]` from branch_state, never saw a pin. The
            # consequence is not cosmetic: without it the defence against a same-named
            # squatter is unreachable through the BRH path.
            for tool, srv in branch.mcp_constraints.allowed_tool_servers.items():
                acc = allowed_tool_servers.setdefault(tool, [])
                for sid in ([srv] if isinstance(srv, str) else list(srv)):
                    if sid not in acc:
                        acc.append(sid)

    http = {"allowed_domains": domains, "fields": fields, "allowed_endpoints": allowed_endpoints}
    mcp = ({"allowed_tools": tools, "param_rules": param_rules,
            "allowed_params": allowed_params,
            "allowed_tool_servers": allowed_tool_servers} if has_mcp else None)
    return http, mcp


# Authorises nothing: an empty domain allowlist blocks every request.
_FAIL_CLOSED_HTTP: dict[str, Any] = {"allowed_domains": [], "fields": [], "allowed_endpoints": []}


def _write_state(
    runtime: BRHRuntime,
    *,
    steps: tuple[BRHBranchStep, ...],
    leaf_key: str,
    trigger_var: str | None,
    trigger_value: TriggerValue,
) -> None:
    merged = _merge_constraints(runtime.constraints, steps)
    if merged is None:
        http, mcp = dict(_FAIL_CLOSED_HTTP), None
        description = ""
    else:
        http, mcp = merged
        description = runtime.constraints.branches[leaf_key].description
        # Commit pins (see `on_tool_call`) are plan-derived like everything else in
        # this block, but they are learned at call time rather than at annotation
        # time, so they are re-applied on every subsequent write — a later branch
        # transition must not silently drop a constraint the plan already earned.
        # Never applied to the fail-closed state: that one authorises nothing and
        # adding a field pin to it would only confuse the alert attribution.
        for pin in runtime._commit_pins:
            if pin not in http["fields"]:
                http["fields"].append(pin)

    # Fetch sitemaps for runtime domains (not blocked by plan or cache).
    _update_sitemap_cache(runtime, http)

    # Build sitemap_schema: only domains with a non-empty sitemap.
    sitemap_schema = {
        domain: entries
        for domain, entries in runtime._sitemap_cache.items()
        if entries
    }

    atomic_write_json(
        runtime.state_path,
        {
            "plan_id": runtime.constraints.plan_id,
            "active_branch": leaf_key,
            "description": description,
            "trigger_var": trigger_var,
            "trigger_value": trigger_value,
            "branch_path": [ROOT_KEY, *(s.branch_id for s in steps)],
            "http_constraints": http,
            "mcp_constraints": mcp,
            "sitemap_schema": sitemap_schema,
            "ts": _utc_now_iso(),
        },
    )
