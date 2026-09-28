
from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://[^\s,;'\"]+|(?<![\w/])/[A-Za-z0-9._~\-/{}]+")


def _paths_named_in(text: str) -> list[str]:
    """URLs/paths a free-text description points at (R2's steering targets)."""
    return [m.group(0) for m in _URL_RE.finditer(text or "")]


def _split(target: str, default_domain: str) -> tuple[str, str]:
    """(host, path) of a description-named target, relative to its own entry's host."""
    if not target.startswith("http"):
        return default_domain.lower(), target
    host, _, path = target.split("://", 1)[1].partition("/")
    return host.split(":", 1)[0].lower(), "/" + path


def _entry(manifest, host: str, path: str):
    """The manifest entry for (host, path), or None if the surface does not offer it.

    R2's "selectable" half lives here: a target that is not an entry cannot be
    authorised by the real annotator, so the model must not authorise it either.
    """
    for ep in manifest:
        if ep.domain.lower() == host.lower() and ep.path_template == path:
            return ep
    return None


def authorised_surface(task, manifest) -> tuple[list[str], list[dict]]:
    """Apply R1+R2+R3 to a gated manifest. Returns (allowed_domains, allowed_endpoints).

    `manifest` is a list of `cobra.brh.validator.HttpEndpoint` — i.e. already through
    the real trust gate and sanitiser, so a blind (TOFU) manifest carries no
    `description` and R2 cannot fire on it. That is the mechanism under test, not a
    special case in this function.
    """
    entries = list(manifest)

    # R1 — the destination the plan selects. Honest by default: the entry that performs
    # this task's action.
    selected = _entry(entries, task.domain, task.action_path)

    # R2 — a visible description naming a selectable target moves the selection.
    for ep in entries:
        for named in _paths_named_in(getattr(ep, "description", "")):
            host, path = _split(named, ep.domain)
            target = _entry(entries, host, path)
            if target is not None and target is not selected:
                selected = target

    # The read the plan performs before deciding. Included because the real annotator
    # emits it (rule 12 covers every endpoint the branch calls), and excluded from any
    # verdict because no S6 attack targets a GET on the task's own host.
    read = next((ep for ep in entries
                 if ep.method == "GET" and ep.domain.lower() == task.domain.lower()),
                None)

    used = [ep for ep in (read, selected) if ep is not None]

    # R3 lives in the ABSENCE below: an excluded sitemap yields no entries, so `used` is
    # empty and `allowed_endpoints` stays empty — which `brh_check._check_endpoint`
    # treats as fail-OPEN. Seeding the task's own action_path here would manufacture a
    # `brh_endpoint` block the real system does not produce and silently mask T2b.
    domains = [task.domain.lower()]
    endpoints: list[dict] = []
    for ep in used:
        dom = ep.domain.lower()
        if dom not in domains:
            domains.append(dom)
        endpoints.append({"method": ep.method, "domain": ep.domain,
                          "path_pattern": ep.path_template})
    return domains, endpoints


def menu_paths(manifest) -> list[str]:
    """Every destination the gated manifest puts on the planner's menu.

    Used to keep a `latent` cell honest: asserting that no breach happened proves
    nothing unless the poisoned entry actually reached the planner in the first place.
    """
    return [f"http://{ep.domain}{ep.path_template}" for ep in manifest]
