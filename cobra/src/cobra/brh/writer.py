"""Orchestration and atomic on-disk I/O for the BRH files.

`generate_plan_constraints` is the single entry point called by the
PrivilegedLLM after a plan has been generated and before the interpreter
runs it:

    plan markdown --extract_skeleton--▶ skeleton (deterministic)
                  --annotate----------▶ constraints (LLM, validated)
                  │   └- on failure --▶ static fallback (fail-closed)
                  --write-------------▶ plan_constraints.json (atomic)
                  --reset-------------▶ branch_state.json (active_branch null)

Both files are written with the tmp-file + ``os.replace`` pattern so the
enforcers (HTTP proxy, MCP proxy) can poll them without ever observing a partial
write. `branch_state.json` is reset to ``active_branch: null`` for every
new plan: the enforcers treat that as fail-closed until the BRH
interpreter hook activates the first branch.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from cobra.mcp_proxy.approval import ApprovalMode, phase2_confirm_plan
from cobra.brh.annotator import AnnotationError, LLMCall, annotate
from cobra.brh.sitemap_trust import sitemap_approval_loop
from cobra.brh.validator import HttpManifest, McpManifest
from cobra.brh.schema import FieldConstraint, PlanConstraints
from cobra.brh.skeleton import PlanSkeleton, extract_skeleton
from cobra.brh.validator import build_fallback

DEFAULT_BRH_DIR = Path(os.environ.get("BRH_DIR", "/tmp/brh"))


@dataclasses.dataclass(frozen=True)
class BRHConfig:
    out_dir: Path = DEFAULT_BRH_DIR

    @property
    def constraints_path(self) -> Path:
        return self.out_dir / "plan_constraints.json"

    @property
    def state_path(self) -> Path:
        return self.out_dir / "branch_state.json"


def atomic_write_json(path: Path, payload: dict[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _seed_domains() -> list[str]:
    """Own-site domains to union into EVERY branch of a generated plan,
    deterministically, regardless of what the annotator emitted. Set per task by
    the OSWorld harness (``BRH_SEED_DOMAINS`` = comma-separated, e.g.
    ``united.com,www.united.com,*.united.com``). These are the task's trusted
    starting/asset domains (the same set fed to the planner prompt): seeding them
    into every branch guarantees (a) the agent's own-site traffic is never 403'd
    on a branch where the LLM under-authorised it, and (b) the post-task evaluator
    — which navigates the task's own URL through the proxy to score it — is not
    blocked on a terminal `false` branch. Empty/unset = no seed (no-op)."""
    raw = os.environ.get("BRH_SEED_DOMAINS", "")
    return [d.strip().lower() for d in raw.split(",") if d.strip()]


def _discover_raw_sitemaps(
    domains: set[str],
    sitemap_path: str = "/sitemap.json",
    timeout: int = 5,
) -> dict[str, list[dict]]:
    """Fetch the RAW sitemap for each domain (HTTPS-first, HTTP fallback).

    Returns ``{domain: raw_entry_list}``. The raw content is NOT trusted here:
    it must pass the sitemap trust gate (hash-pinned registry + approval,
    cobra.brh.sitemap_trust) before any of it reaches the planner.
    Silent on network/parse errors (fail-safe: annotator gets less guidance).
    """
    raw_by_domain: dict[str, list[dict]] = {}
    for domain in sorted(domains):
        for scheme in ("https", "http"):
            url = f"{scheme}://{domain}{sitemap_path}"
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                    data = json.loads(resp.read())
                    if isinstance(data, list):
                        raw_by_domain[domain] = data
                    break  # stop trying http if https succeeded (even if empty list)
            except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
                continue
    return raw_by_domain


def _skeleton_domains(skeleton: PlanSkeleton) -> set[str]:
    """All static domains extracted from the skeleton (root + every arm)."""
    domains: set[str] = set(skeleton.root_static_domains)
    for arm in skeleton.arms.values():
        domains.update(arm.static_domains)
    return domains


def _apply_tool_servers(constraints: PlanConstraints, server_map: dict[str, str] | None) -> None:
    """Deterministically inject allowed_tool_servers from the approved-server manifest.

    For every branch whose mcp_constraints authorises a tool that appears in
    server_map, pins that tool to its approved server_id. Analogous to
    _apply_domain_seed: post-annotation so the LLM annotator never needs to
    know about server identities. Only adds entries; never overwrites one the
    annotator already emitted (allowing test scenarios to set it explicitly)."""
    if not server_map:
        return
    for branch in constraints.branches.values():
        mcp = branch.mcp_constraints
        if mcp is None:
            continue
        for tool in mcp.allowed_tools:
            if tool in server_map and tool not in mcp.allowed_tool_servers:
                mcp.allowed_tool_servers[tool] = server_map[tool]


def _apply_domain_seed(constraints: PlanConstraints) -> None:
    seed = _seed_domains()
    if not seed:
        return
    for branch in constraints.branches.values():
        allowed = branch.http_constraints.allowed_domains
        for d in seed:
            if d not in allowed:
                allowed.append(d)


def _seed_field_policies() -> list[dict]:
    """Field-level policy pins to union into EVERY branch, deterministically,
    from TRUSTED task policy — the field analog of :func:`_seed_domains`. Set by
    the harness via ``BRH_SEED_FIELD_POLICY`` = a JSON list of ``{path, op, value}``
    entries (e.g. a ``subset`` pin of a wire list against the org's owned-domain
    allowlist, or an ``eq_struct`` pin of a structured body against the planned
    object). These are policy facts the harness vouches for (like the domain seed),
    so pinning them deterministically removes the dependence on the LLM annotator —
    which for the structural ops was unreliable (it under-pinned or emitted an
    unresolvable placeholder). Safe to apply to every branch: the enforcer PASSES a
    constraint whose field is absent from the request, so a branch that does not
    carry the field is unaffected; a branch that does carry it is correctly gated.
    Empty/unset = no-op."""
    raw = os.environ.get("BRH_SEED_FIELD_POLICY", "")
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict] = []
    for e in parsed:
        if (
            isinstance(e, dict)
            and isinstance(e.get("path"), str)
            and isinstance(e.get("op"), str)
            and "value" in e
        ):
            out.append({"path": e["path"], "op": e["op"], "value": e["value"]})
    return out


def _apply_field_policy_seed(constraints: PlanConstraints) -> None:
    seed = _seed_field_policies()
    if not seed:
        return
    for branch in constraints.branches.values():
        existing = branch.http_constraints.fields
        for pin in seed:
            try:
                fc = FieldConstraint(path=pin["path"], op=pin["op"], value=pin["value"])
            except Exception:  # noqa: BLE001 — a malformed seed must not abort the plan
                continue
            if not any(
                f.path == fc.path and f.op == fc.op and f.value == fc.value for f in existing
            ):
                existing.append(fc)


def reset_branch_state(plan_id: str, config: BRHConfig) -> None:
    """Writes the fail-closed initial state for a new plan."""
    atomic_write_json(
        config.state_path,
        {
            "plan_id": plan_id,
            "active_branch": None,
            "description": None,
            "trigger_var": None,
            "trigger_value": None,
            "branch_path": [],
            "http_constraints": None,
            "mcp_constraints": None,
            "ts": _utc_now_iso(),
        },
    )


def generate_plan_constraints(
    llm_call: LLMCall,
    task: str,
    plan_markdown: str,
    plan_id: str,
    config: BRHConfig = BRHConfig(),
    log_dir: Path | None = None,
    max_retries: int = 3,
    mcp_tools: McpManifest | None = None,
    server_map: dict[str, str] | None = None,
    http_manifest: HttpManifest | None = None,
    approval_mode: ApprovalMode = ApprovalMode.INTERACTIVE,
    raw_sitemaps: dict[str, list[dict]] | None = None,
    q_llm_call: LLMCall | None = None,
    sitemap_registry_path: str | None = None,
) -> PlanConstraints:
    """Builds and writes plan_constraints.json for a freshly generated plan.

    Returns the written PlanConstraints. If the annotator fails after all
    retries, the static fallback (literal domains only, fail-closed
    everywhere else) is written instead — never nothing: the enforcers
    must always find a constraints file consistent with the current plan.

    ``mcp_tools`` is the approved MCP tool manifest (tool -> param names); pass
    it when the plan may call MCP tools so the annotator can fill
    ``mcp_constraints``. Omitted (None) = HTTP-only.

    ``server_map`` is ``{tool_name: server_id}`` for the approved MCP server(s).
    When given, ``allowed_tool_servers`` is injected deterministically into every
    branch that authorises a matching tool — the LLM annotator is not asked to
    fill it. Build it with ``cobra.mcp_proxy.manifest.server_map_for_tools``.

    ``http_manifest`` is an ALREADY-APPROVED sanitized agent sitemap — either
    the output of ``cobra.brh.sitemap_trust.sitemap_approval_loop`` or a
    harness fixture the caller vouches for; when given, the annotator uses
    endpoint schemas to produce correct field paths and the validator enforces
    that field paths come from declared body_fields. When omitted, raw
    sitemaps (``raw_sitemaps`` if given, else discovered from each skeleton
    domain's ``/sitemap.json``) are NOT trusted: they pass through the sitemap
    trust gate (SHA-256 hash-pinned registry, cobra.brh.sitemap_trust) exactly
    like MCP tool definitions — a changed sitemap is excluded fail-closed (or
    prompts re-approval in INTERACTIVE mode) and unknown sites go through the
    blind P-LLM → Q-LLM → human approval loop in INTERACTIVE mode.

    ``q_llm_call`` (optional) is the quarantined Q-LLM used by the INTERACTIVE
    sitemap loop to read unapproved sitemap descriptions and propose sites;
    ``sitemap_registry_path`` overrides the registry location (default
    ``$BRH_SITEMAP_REGISTRY`` or ``~/.brh/sitemap_registry.json``).

    ``approval_mode`` controls both the sitemap trust gate above and phase-2
    plan confirmation: INTERACTIVE (default, production-safe) prompts the
    human for sitemap (re-)approval and to review planned tool usage per
    branch before the file is written. AUTO needs no human interaction
    (sitemap TOFU pinning + pass-through confirmation) — an unattended,
    fail-open mode for benchmarks/tests only; callers must opt in
    explicitly and it must never be a production default.

    Raises:
        InvalidPlanError: the plan markdown/source itself is unusable; the
            interpreter would reject it for the same reason.
    """
    skeleton = extract_skeleton(plan_markdown)

    if http_manifest is None:
        raw_by_domain = (
            raw_sitemaps if raw_sitemaps is not None
            else _discover_raw_sitemaps(_skeleton_domains(skeleton))
        )
        if raw_by_domain:
            http_manifest = sitemap_approval_loop(
                task, llm_call, q_llm_call, raw_by_domain,
                registry_path=sitemap_registry_path, mode=approval_mode,
            ) or None

    try:
        constraints = annotate(
            llm_call, task, skeleton, plan_id,
            max_retries=max_retries, mcp_tools=mcp_tools, http_manifest=http_manifest,
        )
    except AnnotationError as e:
        print(
            f"⚠️ BRH: annotation failed ({e}); writing static fallback constraints "
            f"(errors: {e.last_errors})"
        )
        constraints = build_fallback(skeleton, plan_id, task)
    except Exception as e:  # noqa: BLE001
        # An UNEXPECTED annotator crash (e.g. a malformed model payload that
        # trips a subscript) must NOT propagate: the enforcers must always find
        # a constraints file consistent with the current plan (docstring above).
        # If it propagates, the caller resets branch_state to fail-closed *null*,
        # which over-blocks even the plan's own benign own-site traffic AND
        # mis-attributes the block as `brh_inactive` instead of the real
        # `brh_domain`/`brh_field`. Degrade to the same static fallback used for
        # AnnotationError (own-site domains only, fail-closed elsewhere): the
        # deterministic domain seed below still authorises the task's trusted
        # host, so an off-policy destination is blocked with the correct reason.
        import traceback
        print(
            f"⚠️ BRH: annotator crashed ({e!r}); writing static fallback "
            f"constraints. Traceback:\n{traceback.format_exc()}"
        )
        constraints = build_fallback(skeleton, plan_id, task)

    # Deterministic own-site seed: union the task's trusted domains into every
    # branch so own-site authorisation never depends on annotator compliance.
    _apply_domain_seed(constraints)
    # Deterministic field-policy seed: pin trusted policy field values (subset /
    # eq_struct against org allowlists / planned objects) into every branch, so
    # structural-op enforcement never depends on the (unreliable) LLM annotator.
    _apply_field_policy_seed(constraints)
    # Deterministic server-qualified allowlist: pin each approved tool to its
    # server_id so a same-named squatter is blocked at call time.
    _apply_tool_servers(constraints, server_map)
    # Phase-2 plan confirmation: human may veto tools before the file is written.
    # AUTO mode (default) is a pure pass-through — no interaction, no change.
    constraints = phase2_confirm_plan(constraints, mode=approval_mode)

    payload = constraints.to_json()
    atomic_write_json(config.constraints_path, payload)
    reset_branch_state(plan_id, config)

    if log_dir is not None:
        atomic_write_json(Path(log_dir) / f"plan_constraints_{plan_id}.json", payload)

    return constraints
