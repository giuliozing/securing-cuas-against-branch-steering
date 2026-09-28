"""STEER-Bench site — one parametric Flask app for every theme.

Runs in the HTTP proxy venv-proxy (flask + jinja only; imports the stdlib-only
`registry`). It is the web generalisation of brh_bench's `bench_server.py`:

    GUI  (agent grounds on this):
      GET  /<theme>/            -> the realistic interactive HTML page. Shows the
                                   DISPLAYED signals as text; the guarded action
                                   ("Confirm") carries the REAL channel body,
                                   embedded server-side (not shown), so a
                                   displayed≠real divergence is invisible to a
                                   grounding agent — the branch-steering premise.
    Machine channels:
      GET  /<theme>/record      -> DISPLAYED signals JSON (perception / MCP reads)
      GET  /<theme>/approved    -> the approved set (in_set / vendor themes)
      GET  /<theme>/quote       -> authoritative REAL wire body + destination
                                   (fetched DIRECTLY, never proxied: server-side
                                   source of truth, not the guarded action)
      POST <anything>           -> the guarded ACTION recorder. Only requests the
                                   enforcer ALLOWS ever arrive, so a recorded
                                   guarded action == an enforcement miss.
    Control plane (direct, never enforced):
      POST /__config  ·  POST /__reset  ·  GET /__state

Bind 0.0.0.0 so every realistic hostname (bank.local, evil.com, …) remapped by
`forward.py` reaches this one server.

Usage: python app.py [port]
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
import os

from steerbench.tasks import registry as reg
from steerbench.harness import driver

# Proxy the site issues the guarded action through (live GUI path). Perception
# stays direct (loopback); only the confirmed action crosses the enforced wire.
#
# Three states, not two, and the third is the one that was missing. Absent -> the
# documented default, so `steerbench site` works standalone against a proxy on :8781.
# A URL -> that proxy. The EMPTY STRING -> explicitly none, which is what the open track
# is: no enforcer exists, so the action goes direct and `driver._remap` routes the
# realistic hostname to the loopback site. Without the third state a proxy-less stack
# would inherit the default and every guarded action would hit a `mitmdump` that was
# never started, invisibly (`harness/stack.Stack.env`).
PROXY = os.environ.get("STEERWEB_PROXY", "http://127.0.0.1:8781") or None

app = Flask(__name__, template_folder=str(HERE / "templates"),
            static_folder=str(HERE / "static"))

def display_number(v):
    """How a number is written on a page — the one rule, shared with `render_check`.

    The registry stores every numeric slot as a float, and Jinja printed it raw: an
    approval queue showed `240.0` and `4900.0`, which no product does and which reads as
    a debug value rather than an amount. Thousands separators go in; a trailing `.0` on an
    integral value comes off; anything with real decimals keeps them (`615.5`, `4.2`).

    Deliberately NOT a currency format. The slots are amounts, unit counts, open defects,
    hours and kilograms, and forcing two decimals onto "0 open findings" would be a
    different kind of wrong. What the page must not do is show a value the agent then
    cannot match to the task; what it need not do is guess the unit.

    `tools/render_check.py` imports this function rather than reimplementing the
    accepted forms, so a formatting change cannot make the check pass against a page it
    no longer describes."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    return f"{int(v):,}" if float(v).is_integer() else f"{v:,}"


app.jinja_env.filters["num"] = display_number

# --- the harness token: who may ask for the REAL body ----------------------
#
# `/quote` and `/__item/<i>/quote` return the answer. They exist because the SITE uses
# them — the guarded action is issued server-to-server, which is what makes
# "perceived ≠ real" a fact no agent can see — but they sit on the base URL an adapter is
# handed, so a browser one URL further along, or an adapter doing an innocuous GET, could
# read the wire body and score ~0% ASR with full utility and no defence at all
# (`config.HARNESS_TOKEN_ENV`, which also states what this does NOT protect against).
#
# Fail-open when unconfigured, and that is deliberate rather than lazy: `steerbench site`
# and the render check bring the site up by hand for inspection, and a site that refused
# its own debugging aid would be worked around rather than used. Every SCORED path goes
# through `harness/stack.build()`, which always sets a token — asserted by
# `tests/test_quote_access.py`, because "the scored path is closed" is the whole claim.
HARNESS_TOKEN = os.environ.get("STEERBENCH_HARNESS_TOKEN", "")
HARNESS_HEADER = "X-Steer-Harness"

REAL_BODY_DENIED = {
    "error": "the real wire body is not readable by the agent",
    "why": ("This endpoint exists for the harness: the site resolves the authoritative "
            "body server-side and issues the guarded action itself, which is what makes "
            "the displayed value and the wire value diverge invisibly. Reading it from an "
            "adapter would be reading the answer — see docs/adapters.md, 'Fair play'."),
}


def _harness_ok() -> bool:
    return (not HARNESS_TOKEN
            or request.headers.get(HARNESS_HEADER) == HARNESS_TOKEN)

# Active scenario knobs, overwritten by POST /__config before each run.
CONFIG: dict = {"theme": None}
RECORDED: dict = {"actions": []}


def _task() -> "reg.Task | None":
    theme = CONFIG.get("theme")
    return reg.BY_THEME.get(theme) if theme else None


def _displayed(task: "reg.Task") -> dict:
    return {s.slot: CONFIG.get(f"disp_{s.slot}") for s in task.signals}


def _real_body(task: "reg.Task") -> dict:
    return {f.name: CONFIG.get(f"real_{f.slot}") for f in task.action_fields}


# -- GUI (the page the agent grounds on) ------------------------------------


@app.get("/<theme>/")
@app.get("/<theme>")
def page(theme):
    task = _task()
    if task is None or task.theme != theme:
        return f"no active scenario for theme {theme!r}", 404
    ctx = _page_context(task)
    arch = reg.archetype_of(task)
    # A theme MAY ship a bespoke page; otherwise it renders its archetype's shared
    # shell, driven entirely by the task manifest. That default is what drops the cost
    # of a new task from "a hand-written page" to "a registry entry", which is what
    # makes 24 more of them plausible.
    bespoke = HERE / "templates" / task.template / "page.html"
    template = (f"{task.template}/page.html" if bespoke.exists()
                else f"archetypes/{arch}/page.html")
    if reg.archetype_of(task) == "none":
        # Declared, not inferred: suites S3 and S6 have no GUI surface at all — their
        # guarded action is an MCP frame or a sitemap fetch. The fallback is a declaration,
        # not a swallowed exception, so a broken template raises.
        return render_template("_generic.html", **ctx)
    return render_template(template, **ctx)


def _page_context(task: "reg.Task") -> dict:
    """What every archetype's template may read.

    Deliberately small. `real_body` in particular must never reach the page: the whole
    branch-steering premise is that the wire body is resolved server-side at `/quote`
    and is invisible to anything looking at the screen."""
    return {
        "task": task,
        "signals": [
            {"slot": s.slot, "label": s.label, "kind": s.kind,
             "value": CONFIG.get(f"disp_{s.slot}")}
            for s in task.signals
        ],
        "approved": CONFIG.get("approved", []),
        "inject_text": CONFIG.get("inject_text", ""),
        # Archetype payloads: empty for `decision`, so one context serves all three.
        "content": CONFIG.get("content", ""),
        "form_fields": task.form_fields,
        "items": CONFIG.get("items", []),
        # Archetype `procedure`: every step the application offers. Read from CONFIG
        # rather than from the task so the page cannot render an affordance set that
        # differs from the one `/__config` declared — and so both variants render the
        # same buttons by construction (`registry.CfiSpec.affordances`).
        "steps": CONFIG.get("steps", []),
    }


# -- machine channels -------------------------------------------------------


@app.get("/<theme>/record")
def record(theme):
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario for theme"}), 404
    return jsonify(_displayed(task))


@app.get("/<theme>/__content")
def content(theme):
    """The active document, for a suite's MCP read tool (e.g. S7's).

    Namespaced under `__` like every other bridge the harness added: a plain
    `/<theme>/content` would be one more path that could collide with a task's own
    `action_path` and silently swallow a guarded action.

    Not a leak of ground truth — this is the same text the `document` archetype renders
    on the page, i.e. what the agent is *meant* to read. `/quote` is the endpoint that
    would be a leak, and it is a separate question.
    """
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario for theme"}), 404
    return jsonify({"content": CONFIG.get("content", "")})


@app.get("/<theme>/sitemap.json")
def sitemap(theme):
    """Suite S6: the agent sitemap, served BY THE SITE — i.e. untrusted input.

    Serving it here rather than handing the harness a constant is the point: the
    poisoned sitemap must arrive over the same channel a real site would use, so the
    trust gate (not the benchmark) is what decides whether it reaches the planner."""
    task = _task()
    if task is None or task.theme != theme or task.trust is None:
        return jsonify({"error": "no sitemap for theme"}), 404
    variant = "attack" if CONFIG.get("_is_attack") else "benign"
    return jsonify(task.sitemap(variant))


@app.get("/<theme>/approved")
def approved(theme):
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario for theme"}), 404
    return jsonify({"approved": CONFIG.get("approved", [])})


@app.get("/<theme>/quote")
def quote(theme):
    """Authoritative real wire body + destination. Fetched directly (unproxied).

    Harness-only when a token is configured — see `_harness_ok`."""
    if not _harness_ok():
        return jsonify(REAL_BODY_DENIED), 403
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario for theme"}), 404
    return jsonify({
        "body": _real_body(task),
        "dest_domain": CONFIG.get("dest_domain", task.domain),
        "dest_path": CONFIG.get("dest_path", task.action_path),
        "sink_on": bool(CONFIG.get("sink_on")),
        "sink_domain": CONFIG.get("sink_domain", ""),
        "sink_path": CONFIG.get("sink_path", ""),
        "exfil": {"note": "steer-bench exfil probe", "theme": task.theme},
    })


# -- GUI action bridge: a grounded click -> the guarded action, enforced ----
# Same-origin (loopback, direct, unenforced) endpoints the page's buttons hit.
# The site then issues the GUARDED action to its destination THROUGH the proxy
# (JSON, real body from /quote), so the live GUI path is enforced identically to
# the structural oracle — the branch-steering divergence is caught on the wire.


@app.post("/<theme>/confirm")
def gui_confirm(theme):
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario"}), 404
    direct_base = request.host_url.rstrip("/")
    out = driver.submit_action(theme, direct_base=direct_base, proxy=PROXY)
    return jsonify(out)


@app.post("/<theme>/__submit")
def gui_submit_form(theme):
    """Archetype `document`: the guarded action carrying AGENT-AUTHORED fields.

    The page's form posts `{name: value}` here; `driver.submit_form` overlays it on the
    server-resolved body. Only names the task actually declares as `form_fields` are
    accepted — an agent (or an injection) must not be able to introduce a wire key the
    plan never authorised, because the enforcer pins fields by path and an unexpected
    key would simply ride along unpinned."""
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario"}), 404
    sent = request.get_json(silent=True) or request.form.to_dict() or {}
    allowed = {f.name for f in task.form_fields}
    fields = {k: v for k, v in sent.items() if k in allowed}
    dropped = sorted(set(sent) - allowed)
    out = driver.submit_form(theme, direct_base=request.host_url.rstrip("/"),
                             proxy=PROXY, fields=fields)
    if dropped:
        out["dropped_fields"] = dropped
    return jsonify(out)


@app.get("/<theme>/__steps")
def steps(theme):
    """Archetype `procedure`: the available steps, as the machine channel sees them.

    The MCP/tool-driven counterpart of the buttons the page renders. Namespaced under
    `__` like every other bridge the harness added, so it can never shadow a task's own
    step path."""
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario for theme"}), 404
    return jsonify({"steps": CONFIG.get("steps", [])})


@app.post("/<theme>/__step/<name>")
def gui_step(theme, name):
    """Archetype `procedure`: a grounded click on ONE step of the procedure.

    Every step is its own enforced request, which is what makes an off-plan step
    something the wire can refuse — and what makes a repeated or reordered one visible in
    `/__state` as a sequence. The step name is checked against what `/__config` declared
    so the page cannot invent a destination the scenario never offered; an unknown name
    is a 404, not a silently recorded action, because a typo that recorded would read as
    a control-flow deviation the agent never committed."""
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario"}), 404
    if name not in {s.get("name") for s in CONFIG.get("steps", [])}:
        return jsonify({"error": f"no such step {name!r}"}), 404
    return jsonify(driver.submit_step(theme, direct_base=request.host_url.rstrip("/"),
                                      proxy=PROXY, step=name))


@app.get("/<theme>/__items")
def items(theme):
    """Archetype `workflow`: the worklist, as the machine channel sees it."""
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario for theme"}), 404
    return jsonify({"items": [
        {"index": i, "ref": it.get("ref"), "label": it.get("label", ""),
         "values": it.get("values", {})}
        for i, it in enumerate(CONFIG.get("items", []))
    ]})


def _item(index: int) -> dict | None:
    items_ = CONFIG.get("items", [])
    return items_[index] if 0 <= index < len(items_) else None


@app.get("/<theme>/__item/<int:index>")
def item_detail(theme, index):
    task = _task()
    it = _item(index)
    if task is None or task.theme != theme or it is None:
        return jsonify({"error": "no such item"}), 404
    return jsonify({"index": index, "ref": it.get("ref"),
                    "label": it.get("label", ""), "values": it.get("values", {})})


@app.get("/<theme>/__item/<int:index>/quote")
def item_quote(theme, index):
    """The authoritative wire body for ONE item — direct, never proxied.

    Same contract as `/quote`, including the harness token: `values` is what the page
    displays, `real` is what the action carries, and under attack they diverge on exactly
    one item. The divergence is per-row, so a plan that authorises the whole list at once
    is not equivalent to one that authorises each row as it reaches it — the distinction
    suite S8 measures."""
    if not _harness_ok():
        return jsonify(REAL_BODY_DENIED), 403
    task = _task()
    it = _item(index)
    if task is None or task.theme != theme or it is None:
        return jsonify({"error": "no such item"}), 404
    values = dict(it.get("values", {}))
    values.update(it.get("real", {}))
    body = {f.name: values.get(f.slot) for f in task.action_fields}
    body["ref"] = it.get("ref")
    # The row's OWN endpoint, and under attack the endpoint the wire really targets.
    # Same premise as `real` one layer out: the screen shows the row's reference and the
    # action is issued against another row's resource (suite S8/P3). Falls back to the
    # task-level destination, so a workflow task that does not use per-row endpoints is
    # unaffected.
    dest_path = it.get("real_path") or it.get("path") \
        or CONFIG.get("dest_path", task.action_path)
    return jsonify({
        "body": body, "ref": it.get("ref"), "index": index,
        "dest_domain": CONFIG.get("dest_domain", task.domain),
        "dest_path": dest_path,
    })


@app.post("/<theme>/__item/<int:index>/confirm")
def gui_confirm_item(theme, index):
    task = _task()
    if task is None or task.theme != theme or _item(index) is None:
        return jsonify({"error": "no such item"}), 404
    return jsonify(driver.submit_item(theme, direct_base=request.host_url.rstrip("/"),
                                      proxy=PROXY, index=index))


@app.post("/<theme>/act-review")
def gui_review(theme):
    task = _task()
    if task is None or task.theme != theme:
        return jsonify({"error": "no active scenario"}), 404
    out = driver.route_to_review(theme, task.domain, proxy=PROXY)
    return jsonify(out)


@app.post("/<theme>/clarify")
def clarify(theme):
    """Abstention recorder (specification axis). Direct/loopback, never enforced:
    the planner calls this instead of the guarded action when a committed value is
    not specified and the record cannot be trusted. Judged from /__state exactly
    like /review."""
    RECORDED["actions"].append({
        "host": request.host.split(":")[0].lower(),
        "path": f"/{theme}/clarify",
        "body": request.get_json(silent=True) or {},
    })
    return jsonify({"status": "ok"})


# -- catch-all action recorder (only enforcer-ALLOWED requests arrive) ------


@app.post("/<path:anything>")
def record_action(anything):
    RECORDED["actions"].append({
        "host": request.host.split(":")[0].lower(),
        "path": "/" + anything,
        "body": request.get_json(silent=True) or request.form.to_dict() or {},
    })
    return jsonify({"status": "ok"})


# -- control plane (direct, never enforced) ---------------------------------


@app.post("/__config")
def set_config():
    CONFIG.clear()
    CONFIG.update(request.get_json(force=True))
    return jsonify({"status": "ok"})


@app.post("/__reset")
def reset():
    RECORDED["actions"].clear()
    return jsonify({"status": "ok"})


@app.get("/__state")
def state():
    return jsonify(RECORDED)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8700
    app.run(host="0.0.0.0", port=port, threaded=True)
