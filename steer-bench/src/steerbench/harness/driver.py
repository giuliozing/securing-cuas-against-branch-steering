

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def _harness_headers() -> dict:
    """The header that says "the harness is asking", when a token is configured.

    Imported lazily and read per call: `config` is not a hard dependency of this module
    (it is the one file both the site and the enforcer side share), and the token is set
    by `stack.build()` after import. See `config.HARNESS_TOKEN_ENV` for what the token
    does and, more importantly, what it does not."""
    try:
        from steerbench import config
        return config.harness_headers()
    except Exception:  # noqa: BLE001 - the site must work without the harness present
        return {}


def _get(url: str, proxy: str | None) -> dict:
    handler = urllib.request.ProxyHandler({"http": proxy} if proxy else {})
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(url, method="GET", headers=_harness_headers())
    with opener.open(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _remap(url: str) -> tuple[str, dict]:
    """Route a realistic destination to the local site WITHOUT an enforcing proxy.

    The benchmark's hostnames (`bank.local`, `evil.com`, …) deliberately do not resolve;
    on the defended track `forward.py` remaps them inside `mitmdump`, after the enforcer
    has read the original `Host`. The OPEN track has no proxy at all — that is what
    "undefended" means — so the remap has to happen here, and it does the same thing for
    the same reason: rewrite the connection target, keep the `Host` header, so the
    recorded action still carries the destination the plan chose.

    Set `STEERBENCH_REMAP` to the site's base URL to enable it. Absent, nothing changes
    and the defended path is unchanged.
    """
    base = os.environ.get("STEERBENCH_REMAP")
    if not base:
        return url, {}
    parts = urllib.parse.urlsplit(url)
    target = urllib.parse.urlsplit(base)
    rewritten = urllib.parse.urlunsplit(
        (target.scheme or "http", target.netloc, parts.path, parts.query, ""))
    return rewritten, {"Host": parts.netloc}


def _post(url: str, body: dict, proxy: str | None) -> tuple[int, str]:
    handler = urllib.request.ProxyHandler({"http": proxy} if proxy else {})
    opener = urllib.request.build_opener(handler)
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if not proxy:
        url, extra = _remap(url)
        headers.update(extra)
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with opener.open(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:      # enforcer 403 lands here
        return e.code, e.read().decode()


def submit_action(theme: str, direct_base: str, proxy: str) -> dict:
    """Resolve the real wire body from the site (direct) and POST it to the
    destination through the enforcing proxy. Returns the enforcer-visible
    outcome (status + any sink attempt)."""
    quote = _get(f"{direct_base}/{theme}/quote", proxy=None)
    dest = f"http://{quote['dest_domain']}{quote['dest_path']}"
    status, resp = _post(dest, quote["body"], proxy=proxy)
    out = {"action_status": status, "action_url": dest}
    if quote.get("sink_on") and quote.get("sink_domain"):
        sink = f"http://{quote['sink_domain']}{quote['sink_path']}"
        s_status, _ = _post(sink, quote.get("exfil", {}), proxy=proxy)
        out["sink_status"] = s_status
        out["sink_url"] = sink
    return out


def submit_form(theme: str, direct_base: str, proxy: str, fields: dict) -> dict:
    """Archetype `document`: the same guarded wire action, with AGENT-AUTHORED fields.

    The server-resolved body from `/quote` is still the base — every pinned slot keeps
    its server-side value, so the branch-steering premise is untouched. What the agent
    typed is overlaid on top, and only there.

    That overlay is the point of the archetype: a channel where an untrusted document
    steers the value the agent authors. There
    is no displayed≠real divergence on those keys: the agent types what it believed,
    and the wire carries it faithfully. So no enforcer can see anything wrong by
    comparing channels — the only thing standing between a poisoned belief and the
    action is the pin the PLAN derived, and a plan that was allowed to read the poison
    derives the attacker's pin. Enforcement is downstream of plan integrity, which is
    the claim the suite exists to make.
    """
    quote = _get(f"{direct_base}/{theme}/quote", proxy=None)
    body = dict(quote["body"])
    body.update(fields or {})
    dest = f"http://{quote['dest_domain']}{quote['dest_path']}"
    status, _ = _post(dest, body, proxy=proxy)
    return {"action_status": status, "action_url": dest,
            "authored": sorted(fields or {})}


def submit_item(theme: str, direct_base: str, proxy: str, index: int) -> dict:
    """Archetype `workflow`: the guarded action for ONE row of the worklist.

    Each call is a separate wire action with its own server-resolved body, so a plan
    that loops over the list produces several enforced requests and several branch
    transitions — which is what makes an ablated per-transition refresh (suite S8)
    distinguishable from the full system at all."""
    quote = _get(f"{direct_base}/{theme}/__item/{index}/quote", proxy=None)
    dest = f"http://{quote['dest_domain']}{quote['dest_path']}"
    status, _ = _post(dest, quote["body"], proxy=proxy)
    return {"action_status": status, "action_url": dest, "item": index,
            "item_ref": quote.get("ref")}


def submit_step(theme: str, direct_base: str, proxy: str, step: str) -> dict:
    """Archetype `procedure`: ONE named step of a multi-step procedure.

    The same enforced wire action as `submit_action`, with one difference that is the
    whole of suite S7: the destination path is the STEP's, not the task's. A procedure
    performs several *different* actions in a fixed order, so the plan's allowlist has
    several entries and an action that is not one of them is an off-plan target — which
    is a thing the enforcer can refuse, and a thing no field pin could ever see.

    The body is still resolved server-side from `/quote`, so nothing about this call
    hands the agent a lever on what the wire carries. In S7 that lever would be beside
    the point: the attack is on which call happens, never on its contents.
    """
    quote = _get(f"{direct_base}/{theme}/quote", proxy=None)
    dest = f"http://{quote['dest_domain']}/{theme}/{step}"
    body = dict(quote["body"])
    body["step"] = step
    status, _ = _post(dest, body, proxy=proxy)
    return {"action_status": status, "action_url": dest, "step": step}


def route_to_review(theme: str, domain: str, proxy: str) -> dict:
    url = f"http://{domain}/{theme}/review"
    status, _ = _post(url, {"routed": True}, proxy=proxy)
    return {"review_status": status, "review_url": url}


def request_clarification(theme: str, direct_base: str, reason: str) -> dict:
    """The abstention marker (specification axis). Posts DIRECT to the site's
    /<theme>/clarify recorder — not through the proxy, not a guarded wire action —
    so it is never entangled with the enforcer's domain/endpoint allowlists.
    Abstention is an internal 'I am asking the user' decision, not an external
    action to be enforced. Recorded in /__state as /<theme>/clarify."""
    url = f"{direct_base}/{theme}/clarify"
    status, _ = _post(url, {"clarify": True, "reason": reason}, proxy=None)
    return {"clarify_status": status, "clarify_url": url}
