"""Sitemap trust registry + two-phase site approval (HTTP analogue of MCP proxy).

Until now the agent sitemap fed to the annotator was implicitly trusted. This
module gives it the same trust model as MCP tool definitions: hash pinning
plus two-phase approval.

- Every site's sitemap is SHA-256 hashed canonically and pinned in a
  persistent registry (``~/.brh/sitemap_registry.json``), keyed by the site's
  domain, with an approved/rejected verdict and approval provenance
  (``human`` vs ``tofu``).
- A sitemap whose hash differs from the pinned one is a *changed* sitemap
  (HTTP rug pull): it never reaches the planner until a human re-approves it.
- New (unregistered) sites follow the blind protocol: the P-LLM never reads
  unapproved sitemap free text. It sees only the approved sites' structural
  surface (domain + method + path templates) and declares whether they
  suffice for the task. If not, the quarantined Q-LLM reads the unapproved
  candidates' sitemaps (including free-text ``semantic_action`` descriptions)
  and proposes the list of needed-but-unapproved sites; the human approves;
  the approved sitemap is hash-pinned. Only then does the P-LLM see its
  endpoints.
- Two-state description exposure: endpoint descriptions are
  exposed to the P-LLM only for *human-approved* pinned sitemaps. TOFU
  (AUTO-mode) registrations stay description-blind — feeding a TOFU-approved
  description to the planner would reintroduce day-one poisoning.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import re
import sys
from typing import Mapping

from cobra.mcp_proxy.approval import ApprovalMode, LLMCall
from cobra.mcp_proxy.registry import _utc_now_iso, load_registry, save_registry
from cobra.brh.validator import HttpEndpoint, sanitize_sitemap

DEFAULT_SITEMAP_REGISTRY_PATH = os.path.expanduser("~/.brh/sitemap_registry.json")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _registry_path(path: str | None) -> str:
    return path or os.environ.get("BRH_SITEMAP_REGISTRY") or DEFAULT_SITEMAP_REGISTRY_PATH


def _parse_json(text: str) -> dict | list:
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text.strip())


# -- registry primitives (mirror of cobra.mcp_proxy.registry, per-site) -------------

def canonical_sitemap_bytes(raw: list[dict]) -> bytes:
    """Canonical byte serialisation of a raw sitemap (the site's trusted surface).

    The *entire* raw entry list is hashed — including free-text fields such as
    ``semantic_action`` — so a description-only change is still a hash
    mismatch, exactly like MCP tool descriptions."""
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sitemap_hash(raw: list[dict]) -> str:
    return hashlib.sha256(canonical_sitemap_bytes(raw)).hexdigest()


def load_sitemap_registry(path: str | None = None) -> dict:
    return load_registry(_registry_path(path))


def save_sitemap_registry(registry: dict, path: str | None = None) -> None:
    save_registry(registry, _registry_path(path))


def approve_sitemap(
    domain: str, raw: list[dict], path: str | None = None, *, by: str = "human"
) -> None:
    """Pin (or re-pin) a site's sitemap hash as approved, **with its content**.

    ``by`` records approval provenance: ``"human"`` (explicit decision — the
    only provenance that unlocks description exposure to the P-LLM) or
    ``"tofu"`` (AUTO-mode first-sight registration).

    ``content`` stores the exact entries that were vetted. With only a hash, on a
    rug pull the gate could *detect* the change but had nothing to fall back to, so
    the site's whole surface vanished — the annotator emitted no ``allowed_endpoints``
    (empty is fail-OPEN in ``brh_check._check_endpoint``, so a rug pull adding a path
    on the same host won), and the planner lost its endpoint menu (so honest work
    stopped too).

    Storing content makes the registry hold *trusted data*, not just a fingerprint.
    That is the point — a pin the user believes means "this surface is approved"
    should be able to produce that surface — but it also means the registry file is
    now integrity-relevant state, not a cache."""
    reg = load_sitemap_registry(path)
    reg[domain] = {"hash": sitemap_hash(raw), "ts": _utc_now_iso(), "approved": True,
                   "approved_by": by, "content": list(raw)}
    save_sitemap_registry(reg, path)


def approved_content(domain: str, registry: Mapping) -> list[dict] | None:
    """The last vetted entries for a domain, or None if this record predates content
    pinning (or the site was rejected).

    Tolerant by design: a hash-only registry record has no ``content`` key, and
    such a record must keep behaving exactly as it did — exclusion with no fallback —
    rather than crashing or silently inventing a surface. That keeps every existing
    ``~/.brh/sitemap_registry.json`` valid and makes the change additive."""
    rec = registry.get(domain)
    if not isinstance(rec, dict) or not rec.get("approved"):
        return None
    content = rec.get("content")
    return list(content) if isinstance(content, list) and content else None


def reject_sitemap(domain: str, raw: list[dict], path: str | None = None) -> None:
    """Record a human rejection: the site stays blocked until a human flips it."""
    reg = load_sitemap_registry(path)
    reg[domain] = {"hash": sitemap_hash(raw), "ts": _utc_now_iso(), "approved": False, "approved_by": "human"}
    save_sitemap_registry(reg, path)


class SitemapStatus(enum.Enum):
    APPROVED = "approved"  # pinned, approved, hash matches
    CHANGED = "changed"    # pinned but hash differs → re-approval required
    UNKNOWN = "unknown"    # never seen
    REJECTED = "rejected"  # human said no — stays blocked regardless of hash


def sitemap_status(domain: str, raw: list[dict], registry: Mapping) -> SitemapStatus:
    rec = registry.get(domain)
    if not isinstance(rec, dict):
        return SitemapStatus.UNKNOWN
    if not rec.get("approved"):
        return SitemapStatus.REJECTED
    if rec.get("hash") == sitemap_hash(raw):
        return SitemapStatus.APPROVED
    return SitemapStatus.CHANGED


# -- trust gate ----------------------------------------------------------------

@dataclasses.dataclass
class GatedSitemap:
    raw: list[dict]
    human_approved: bool  # True only for human-provenance approvals (two-state model)


def gate_sitemaps(
    raw_by_domain: Mapping[str, list[dict]],
    *,
    mode: ApprovalMode = ApprovalMode.INTERACTIVE,
    registry_path: str | None = None,
) -> tuple[dict[str, GatedSitemap], dict[str, list[dict]]]:
    """Split candidate sitemaps into (admitted, pending_unknown) via the registry.

    - APPROVED (hash match): admitted; descriptions exposed iff human-approved.
    - CHANGED (hash mismatch): never admitted here — AUTO excludes fail-closed
      (no human to ask); INTERACTIVE handles re-approval in the loop.
    - REJECTED: excluded in both modes.
    - UNKNOWN: INTERACTIVE (the default) returns them as pending for the
      demand-driven approval loop — nothing is trusted without a human
      decision. AUTO pins trust-on-first-use and admits (description-blind);
      this is an unattended, fail-open mode for benchmarks/tests only and
      must be selected explicitly by the caller, never relied on as a
      default.
    """
    registry = load_sitemap_registry(registry_path)
    admitted: dict[str, GatedSitemap] = {}
    pending: dict[str, list[dict]] = {}
    if mode == ApprovalMode.AUTO:
        print(
            "[sitemap-trust] WARNING: ApprovalMode.AUTO — unknown sitemaps are "
            "trusted on first use with no human review. Unattended "
            "benchmarks/tests only; never use this mode in production.",
            file=sys.stderr,
        )
    for domain, raw in raw_by_domain.items():
        status = sitemap_status(domain, raw, registry)
        if status is SitemapStatus.APPROVED:
            by = registry[domain].get("approved_by")
            admitted[domain] = GatedSitemap(raw=raw, human_approved=by == "human")
        elif status is SitemapStatus.REJECTED:
            print(f"[sitemap-trust] '{domain}': rejected in registry — excluded.", file=sys.stderr)
        elif status is SitemapStatus.CHANGED:
            if mode == ApprovalMode.AUTO:
                # Fail-closed against the CHANGE, not against the site. The new
                # sitemap is never admitted — that part is unchanged and is what
                # stops the rug pull. What is new is the fallback: if the record
                # carries the vetted content, the site keeps the surface a human
                # (or TOFU) once approved, so exclusion means "revert to the last
                # vetted state" rather than "this site no longer exists".
                #
                # It is an availability decision taken deliberately: should the
                # site legitimately have MOVED an endpoint, we now act on a stale
                # but vetted map instead of refusing. INTERACTIVE re-approval is
                # the real answer in production; this is the unattended fallback,
                # and it is strictly better than the previous behaviour, where the
                # endpoint layer went ABSENT — which `brh_check` treats as
                # fail-OPEN, i.e. the rug pull won by deleting the defence.
                vetted = approved_content(domain, registry)
                if vetted is None:
                    print(
                        f"[sitemap-trust] '{domain}': sitemap changed since approval "
                        "(hash mismatch) — excluded fail-closed, no vetted content "
                        "pinned; re-approve interactively.",
                        file=sys.stderr,
                    )
                else:
                    by = registry[domain].get("approved_by")
                    admitted[domain] = GatedSitemap(raw=vetted, human_approved=by == "human")
                    print(
                        f"[sitemap-trust] '{domain}': sitemap changed since approval "
                        "(hash mismatch) — the new one is excluded; falling back to "
                        "the last approved surface.",
                        file=sys.stderr,
                    )
            else:
                pending[domain] = raw  # loop will run the re-approval prompt
        else:  # UNKNOWN
            if mode == ApprovalMode.AUTO:
                approve_sitemap(domain, raw, registry_path, by="tofu")
                admitted[domain] = GatedSitemap(raw=raw, human_approved=False)
            else:
                pending[domain] = raw
    return admitted, pending


def manifest_from_gated(admitted: Mapping[str, GatedSitemap]) -> list[HttpEndpoint]:
    """Sanitized manifest for the planner; descriptions only for human-approved."""
    endpoints: list[HttpEndpoint] = []
    for domain in sorted(admitted):
        gated = admitted[domain]
        endpoints.extend(sanitize_sitemap(gated.raw, include_descriptions=gated.human_approved))
    return endpoints


# -- P-LLM sufficiency check (blind: structural surface only) -----------------

_SITE_SUFFICIENCY_SYSTEM = """\
You are a planning assistant. You receive an agent task and the list of web \
sites currently approved for HTTP use (domains and structural endpoint \
templates only — you never see endpoint descriptions). Decide whether the \
approved sites are sufficient to complete the task.

Output exactly one fenced ```json block:
{"sufficient": true}
or
{"sufficient": false, "missing_capability": "<one short English sentence describing the single most important missing site capability>"}

Rules:
- Report only ONE missing capability per response (the most critical one).
- Do not mention specific domain names or guess at what other sites might exist.
- If you are unsure but can attempt the task with the approved sites, say sufficient=true.
"""


@dataclasses.dataclass
class SiteSufficiencyResult:
    sufficient: bool
    missing_capability: str | None = None


def _structural_site_lines(admitted: Mapping[str, GatedSitemap]) -> str:
    if not admitted:
        return "  (none)"
    lines = []
    for domain in sorted(admitted):
        eps = sanitize_sitemap(admitted[domain].raw)  # structural only, never free text
        templates = [f"{ep.method} {ep.path_template}" for ep in eps]
        lines.append(f"  {domain}: endpoints={templates}")
    return "\n".join(lines)


def site_sufficiency_check(
    p_llm_call: LLMCall, task: str, admitted: Mapping[str, GatedSitemap]
) -> SiteSufficiencyResult:
    """Ask the P-LLM whether the approved sites cover ``task``.

    The P-LLM receives only domains + structural endpoint templates — never
    sitemap free text. On parse failure, defaults to sufficient=True."""
    user_prompt = f"Task: {task}\n\nApproved sites:\n{_structural_site_lines(admitted)}"
    raw = p_llm_call(_SITE_SUFFICIENCY_SYSTEM, user_prompt)
    try:
        data = _parse_json(raw)
        if not isinstance(data, dict) or data.get("sufficient"):
            return SiteSufficiencyResult(sufficient=True)
        cap = data.get("missing_capability") or "unspecified capability"
        return SiteSufficiencyResult(sufficient=False, missing_capability=cap)
    except (ValueError, AttributeError, KeyError):
        return SiteSufficiencyResult(sufficient=True)


# -- Q-LLM site selection (quarantined: reads descriptions) -------------------

_SITE_SELECTION_SYSTEM = """\
You are a site selection assistant. A task planner needs a web site matching a \
described capability. You receive the planner's requirement and the sitemaps \
of candidate sites that are NOT yet approved (with endpoint descriptions).

Select the sites the planner needs, at most {n}. Output exactly one fenced \
```json block containing a list of objects:

[
  {{"domain": "...", "summary": "...", "reason": "..."}},
  ...
]

Rules:
- Return at most {n} objects, ordered best-first; only sites truly needed.
- "summary" is one sentence describing what the site's endpoints offer.
- "reason" is one sentence explaining why the site matches the requirement.
- Only include domains from the provided candidates; copy each domain verbatim.
- If no candidate matches, return an empty list.
"""


@dataclasses.dataclass
class SiteCandidate:
    domain: str
    summary: str
    reason: str
    raw_sitemap: list[dict] = dataclasses.field(default_factory=list, repr=False, compare=False)


def _describe_site(domain: str, raw: list[dict]) -> str:
    lines = [f"Site: {domain}"]
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        method = entry.get("method", "?")
        url = entry.get("url", "?")
        action = entry.get("semantic_action", "(no description)")
        lines.append(f"  {method} {url} — {action}")
    return "\n".join(lines)


def select_site_candidates(
    q_llm_call: LLMCall,
    requirement: str,
    pending: Mapping[str, list[dict]],
    n: int = 3,
) -> list[SiteCandidate]:
    """Q-LLM reads unapproved sitemaps (with descriptions) and proposes sites.

    The Q-LLM is quarantined (no tool-calling): a poisoned sitemap description
    yields at worst a bad recommendation, caught by the human review below.
    On parse failure returns an empty list."""
    if not pending:
        return []
    site_blocks = "\n\n".join(_describe_site(d, pending[d]) for d in sorted(pending))
    raw = q_llm_call(
        _SITE_SELECTION_SYSTEM.format(n=n),
        f"Missing capability: {requirement}\n\nCandidate sites:\n{site_blocks}",
    )
    try:
        items = _parse_json(raw)
        if not isinstance(items, list):
            return []
        candidates: list[SiteCandidate] = []
        for item in items[:n]:
            domain = item.get("domain") if isinstance(item, dict) else None
            if not domain or domain not in pending:
                continue
            candidates.append(SiteCandidate(
                domain=domain,
                summary=item.get("summary", ""),
                reason=item.get("reason", ""),
                raw_sitemap=pending[domain],
            ))
        return candidates
    except (ValueError, AttributeError, KeyError):
        return []


# -- human gates (CLI) ---------------------------------------------------------

def human_select_sites(candidates: list[SiteCandidate]) -> list[SiteCandidate]:
    """CLI: present Q-LLM candidates with the RAW sitemap descriptions (not
    only the Q-LLM paraphrase — same rule as MCP approval); return the human's
    approved subset (possibly empty)."""
    if not candidates:
        print("  [sitemap-approval] No candidates available from Q-LLM.")
        return []
    print("\n  Candidate sites (Q-LLM ranked, best first):")
    for i, c in enumerate(candidates, 1):
        print(f"  [{i}] {c.domain}")
        print(f"      Summary: {c.summary}")
        print(f"      Why: {c.reason}")
        print("      Raw sitemap:")
        for line in _describe_site(c.domain, c.raw_sitemap).splitlines()[1:]:
            print(f"    {line}")
    print("  [0] None of the above — skip this capability")
    while True:
        raw = input(f"  Approve sites (comma-separated numbers, 0 to skip) [0-{len(candidates)}]: ").strip()
        if raw == "0":
            return []
        try:
            indices = {int(x.strip()) - 1 for x in raw.split(",")}
        except ValueError:
            print(f"  Please enter comma-separated numbers between 0 and {len(candidates)}.")
            continue
        if all(0 <= i < len(candidates) for i in indices):
            return [candidates[i] for i in sorted(indices)]
        print(f"  Please enter comma-separated numbers between 0 and {len(candidates)}.")


def human_reapprove_sitemap(domain: str, raw: list[dict]) -> bool:
    """CLI: a pinned sitemap changed (hash mismatch) — show the new content and
    ask the human to re-approve. Returning False keeps the old pin and excludes
    the changed sitemap from this run."""
    print(f"\n  [sitemap-approval] Sitemap for '{domain}' CHANGED since approval (hash mismatch).")
    print("  New sitemap content:")
    for line in _describe_site(domain, raw).splitlines()[1:]:
        print(f"  {line}")
    while True:
        answer = input("  Re-approve the changed sitemap? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False


# -- main approval loop --------------------------------------------------------

def sitemap_approval_loop(
    task: str,
    p_llm_call: LLMCall | None,
    q_llm_call: LLMCall | None,
    raw_by_domain: Mapping[str, list[dict]],
    *,
    registry_path: str | None = None,
    mode: ApprovalMode = ApprovalMode.INTERACTIVE,
    max_rounds: int = 5,
) -> list[HttpEndpoint]:
    """Run the sitemap trust gate + demand-driven site approval; return the manifest.

    INTERACTIVE mode (the default, production-safe): changed sitemaps prompt
    for human re-approval; unknown sites go through the blind demand-driven
    loop — P-LLM (structural surface only) declares a missing capability,
    Q-LLM (reads descriptions) proposes candidate sites, the human approves;
    approved sitemaps are hash-pinned and only then exposed to the P-LLM
    (with descriptions, the two-state model).

    AUTO mode: registry gate only — approved sitemaps pass, unknown ones are
    pinned trust-on-first-use (description-blind), changed/rejected ones are
    excluded fail-closed. No LLM calls, no human interaction. This is an
    unattended, fail-open mode for benchmarks/tests only — callers must opt
    in explicitly by passing ``mode=ApprovalMode.AUTO``; it must never be the
    implicit behavior of a production deployment.

    The returned manifest is ready to pass to
    ``generate_plan_constraints(http_manifest=...)``.
    """
    admitted, pending = gate_sitemaps(raw_by_domain, mode=mode, registry_path=registry_path)

    if mode == ApprovalMode.AUTO:
        return manifest_from_gated(admitted)

    # Re-approval pass for changed sitemaps (pinned, hash mismatch).
    registry = load_sitemap_registry(registry_path)
    for domain in sorted(list(pending)):
        if sitemap_status(domain, pending[domain], registry) is not SitemapStatus.CHANGED:
            continue
        raw = pending.pop(domain)
        if human_reapprove_sitemap(domain, raw):
            approve_sitemap(domain, raw, registry_path, by="human")
            admitted[domain] = GatedSitemap(raw=raw, human_approved=True)
            print(f"  Re-approved '{domain}' (new hash pinned).")
        else:
            print(f"  '{domain}' left excluded (old pin kept).")

    # Demand-driven approval of unknown sites (P-LLM blind → Q-LLM → human).
    if isinstance(pending, Mapping):
        pending = dict(pending)
    for round_num in range(1, max_rounds + 1):
        if not pending or p_llm_call is None:
            break
        print(f"\n[sitemap-approval {round_num}/{max_rounds}] Checking site sufficiency…")
        result = site_sufficiency_check(p_llm_call, task, admitted)
        if result.sufficient:
            print(f"  P-LLM: sufficient — {len(admitted)} site(s) approved.")
            break
        cap = result.missing_capability
        print(f"  P-LLM: missing site capability — \"{cap}\"")
        if q_llm_call is None:
            print("  [sitemap-approval] No Q-LLM available to propose sites; stopping.")
            break
        candidates = select_site_candidates(q_llm_call, cap or "", pending, n=3)
        chosen = human_select_sites(candidates)
        if not chosen:
            print("  Skipping capability. Proceeding with current site set.")
            break
        for c in chosen:
            approve_sitemap(c.domain, c.raw_sitemap, registry_path, by="human")
            admitted[c.domain] = GatedSitemap(raw=c.raw_sitemap, human_approved=True)
            pending.pop(c.domain, None)
            print(f"  Approved '{c.domain}' (hash pinned).")
    else:
        print(f"[sitemap-approval] max_rounds={max_rounds} reached; proceeding.")

    return manifest_from_gated(admitted)
