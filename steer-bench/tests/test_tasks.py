"""Schema validation for the task registry — the SUBSTRATE half of the gate.

Needs nothing but this package: no defence stack, no processes, no network. This is the
check a contributor adding a task runs, and the one that must stay green for anyone who
clones the repository.

It is a test rather than a convention because *growing* `TASKS` is invisible from the
file being edited: a new `attack_kind` silently broadens every consumer that iterates
the shared list. The inventory pin at the bottom is the cheap guard against the
next one — if a task appears or moves suite, exactly one assertion fails and says so.
"""

from __future__ import annotations

import pytest

from steerbench import config
from steerbench.harness import arms
from steerbench.tasks import registry as reg


# --- identity ---------------------------------------------------------------


def test_ids_and_themes_are_unique(tasks):
    """`BY_THEME` is a dict, so a duplicate theme would silently drop a task from every
    theme-keyed consumer (the site, the MCP namespace) while `TASKS` still counted it."""
    assert len({t.tid for t in tasks}) == len(tasks)
    assert len({t.theme for t in tasks}) == len(tasks)


# --- the attack is fully specified ------------------------------------------


def test_attack_kind_carries_its_target(tasks):
    for t in tasks:
        if t.attack_kind == "domain":
            assert t.attacker_domain, f"{t.tid}: domain attack with no attacker_domain"
        if t.attack_kind == "endpoint":
            assert t.attack_path, f"{t.tid}: endpoint attack with no attack_path"


def test_expected_reason_matches_the_layer_attacked(tasks):
    """The verdict asserts a STRUCTURAL reason, so a task expecting the wrong one would
    score BLOCKED-but-failed forever and look like a defect in the enforcer."""
    want = {"field": "brh_field", "domain": "brh_domain", "endpoint": "brh_endpoint"}
    for t in tasks:
        if t.attack_kind in want:
            assert t.expect_attack_reason == want[t.attack_kind], t.tid
        # S9 attacks an authority granted ABOVE the leaf, and which layer that is is a
        # per-level property, not a per-task one: the reason must match the grant of the
        # level the attack violates, or the cell would assert the wrong structural reason
        # and read as an enforcer defect forever.
        if t.on_ultra_axis:
            grant = t.ultra.levels[t.ultra.level].grant
            assert t.expect_attack_reason == {"field": "brh_field",
                                              "endpoint": "brh_endpoint"}[grant], t.tid


def test_every_http_attack_carries_bait(tasks):
    """`inject_text` is what a naive classifier is meant to believe. Without it the
    policy-prediction contrast measures nothing."""
    for t in tasks:
        if t.on_http_axis:
            assert t.inject_text, f"{t.tid}: no injected text"


# --- slots are consistent across the three channels -------------------------


def test_action_fields_reference_real_signals(tasks):
    """A field pinned to a slot no signal displays cannot be perceived, so the plan
    could never have derived the pin — the cell would measure the harness, not the
    component."""
    for t in tasks:
        slots = {s.slot for s in t.signals}
        for f in t.action_fields:
            assert f.slot in slots, f"{t.tid}: field {f.name} pins unknown slot {f.slot}"


def test_attack_overlay_only_touches_displayed_slots(tasks):
    for t in tasks:
        assert not (set(t.real_attack) - set(t.disp)), \
            f"{t.tid}: real_attack names slots that are never displayed"


def test_signal_count_stays_under_the_perception_ceiling(tasks):
    """<=3 by design (`registry.Signal`): the grounding executor has to read them off a
    screenshot, and the benchmark must not become an OCR test.

    **S9 is exempt, and the exemption is the finding rather than a waiver.** A plan
    three or four levels deep needs one perceived value per level, so depth and this
    ceiling cannot both hold — the suite exists to measure what happens at depth, so it
    is the ceiling that gives way. What that costs is stated where it is paid: those
    five tasks are certified structurally (`oracles/s9.py`) and are declared in
    `docs/DATASHEET.md` as outside the live-GUI grounding leg. The bound is asserted
    anyway, at the depth the suite actually authors, so "exempt" does not become
    "unbounded"."""
    for t in tasks:
        if t.on_ultra_axis:
            assert len(t.signals) == len(t.ultra.levels) <= 4, (
                f"{t.tid}: {len(t.signals)} signals for {len(t.ultra.levels)} levels — "
                f"an ultra task perceives exactly one value per level of its tree")
            continue
        assert len(t.signals) <= 3, f"{t.tid}: {len(t.signals)} signals"


# --- the benign/attack contract ---------------------------------------------


def test_benign_config_has_no_divergence(tasks):
    """The false-positive probe is only meaningful if the benign wire genuinely mirrors
    what is displayed. A stray divergence here would make FP=0 a tautology.

    Scoped to the HTTP axis, because only those tasks have a displayed channel — see
    `test_trust_tasks_declare_signals_they_never_display` for the S6 case, which is
    recorded rather than asserted away."""
    for t in tasks:
        if not t.on_http_axis:
            continue
        cfg = t.benign_config()
        for s in t.signals:
            assert cfg[f"disp_{s.slot}"] == cfg[f"real_{s.slot}"], f"{t.tid}/{s.slot}"
        assert cfg["_is_attack"] is False


def test_trust_tasks_declare_signals_they_never_display():
    """A registry inconsistency, pinned as a fact rather than fixed here.

    All 11 S6 tasks carry three `Signal`s with **no matching entry in `disp`**, so
    `_base_cfg` emits neither `disp_<slot>` nor `real_<slot>` for them. It is harmless
    — S6 has no GUI surface at all (`template="generic"`), its attack is the
    sitemap the site serves, and nothing on its path reads a signal — but the
    declaration reads as if a page showed those values, and it is exactly the shape
    that lets a future consumer iterate `TASKS` and get `None` unexpectedly.

    This test turns 'someone will remember' into 'the suite says so', and it must be
    UPDATED, not deleted, if these tasks change.
    """
    from steerbench.tasks import registry as reg
    undisplayed = {t.tid for t in reg.TASKS
                   if any(s.slot not in t.disp for s in t.signals)}
    # S8 and S9 join the set for a DIFFERENT and legitimate reason, which is why the pin
    # names the populations rather than one number: a workflow task's values are per row
    # (`WorkItem.values`), so `disp` is empty by design and the signals are the queue's
    # column headers. The S6 case is still the inconsistency this test was written for.
    assert undisplayed == {t.tid for t in reg.TASKS
                           if t.on_trust_axis or t.on_step_axis or t.on_ultra_axis}
    assert len(undisplayed) == 26
    assert sum(1 for t in reg.TASKS if t.on_trust_axis) == 11


def test_attack_config_diverges_somewhere(tasks):
    """Every attack cell must differ from its benign twin on the layer it claims to
    attack — otherwise the enforcer has nothing to catch and A5 'passes' vacuously."""
    for t in tasks:
        if not t.on_http_axis:
            continue
        b, a = t.benign_config(), t.attack_config()
        diverged = (
            any(b.get(f"real_{s}") != a.get(f"real_{s}") for s in t.disp)
            or b["dest_domain"] != a["dest_domain"]
            or b["dest_path"] != a["dest_path"]
        )
        assert diverged, f"{t.tid}: attack config is identical to benign"


# --- the sandbox ------------------------------------------------------------


def test_every_referenced_host_is_remapped(tasks):
    """`forward.py` remaps exactly `all_hosts()` to loopback. A host reachable from a
    task but absent from that inventory would leave the sandbox and hit the real
    internet — which is how S6's poisoned-sitemap destinations were nearly missed."""
    known = set(reg.all_hosts())
    for t in tasks:
        for host in (t.domain, t.attacker_domain, t.sink_domain):
            if host:
                assert host.lower() in known, f"{t.tid}: {host} not in all_hosts()"


def test_sink_is_declared_in_full(tasks):
    for t in tasks:
        if t.sink_on_attack:
            assert t.sink_domain and t.sink_path, f"{t.tid}: partial sink declaration"


# --- the specification axis -------------------------------------------------


def test_spec_axis_tasks_are_complete(tasks):
    for t in tasks:
        if not t.on_spec_axis:
            continue
        assert t.instruction_core, f"{t.tid}: on the spec axis with no instruction_core"
        for slot in t.committed_slots:
            assert slot in t.disp, f"{t.tid}: commits {slot}, which has no displayed value"


def test_spec_postures_produce_three_distinct_prompts(tasks):
    """`specified` supplies the value, `underspecified` withholds it, `delegated`
    authorises the record. If two collapsed, one arm would measure the other."""
    for t in tasks:
        if not t.on_spec_axis:
            continue
        prompts = {t.instruction_specified(), t.instruction_underspecified(),
                   t.instruction_delegated()}
        assert len(prompts) == 3, f"{t.tid}: spec postures are not distinct"


def test_poison_config_leaves_the_wire_consistent(tasks):
    """Perception poisoning has NO wire divergence — that is what makes the enforcer
    structurally blind and abstention the sole defence. A divergence here would let the
    field layer catch it and silently destroy the suite's attribution."""
    for t in tasks:
        if not t.on_spec_axis:
            continue
        cfg = t.poison_config()
        for slot in t.committed_slots:
            assert cfg[f"disp_{slot}"] == cfg[f"real_{slot}"], f"{t.tid}/{slot}"


# --- the GUI substrate ------------------------------------------------------


def test_http_tasks_have_a_page_and_others_declare_none(tasks):
    """A task has a page iff its ARCHETYPE says so — not iff it is on the HTTP axis.

    The two do not coincide once a suite can carry a GUI page under an `attack_kind` other than field/domain/endpoint, so keying on
    `on_http_axis` would have demanded such tasks declare `template="generic"` and call
    themselves page-less. Keying on the archetype asks the question the assertion is
    actually about, and `archetype_of` derives it for every task that does not
    declare one."""
    for t in tasks:
        arch = reg.archetype_of(t)
        if arch == "none":
            assert t.template == "generic", \
                f"{t.tid}: has no GUI surface but names template {t.template!r}"
        else:
            page = config.PKG / "site" / "templates" / t.template / "page.html"
            shell = config.PKG / "site" / "templates" / "archetypes" / arch / "page.html"
            assert page.exists() or shell.exists(), (
                f"{t.tid}: archetype {arch!r} has neither a bespoke page nor a shell")


# --- the inventory pin ------------------------------------------------------


def test_inventory_matches_the_published_counts(tasks):
    """Task-sets overlap by design, so these are membership counts and
    are never summed. Update this test, the README, and `dataset/tasks.json`
    together — that coupling is the point: a task added without a documentation
    change fails here."""
    from collections import Counter
    counts = Counter(s for t in tasks for s in arms.suites_of(t))
    assert len(tasks) == 101
    assert counts == {"S1": 26, "S2": 24, "S3": 15, "S4": 18, "S5": 11, "S6": 11,
                      "S8": 10, "S9": 5, "S7": 10}


@pytest.mark.parametrize("predicate,want", [
    ("on_http_axis", 50), ("on_mcp_axis", 15), ("on_trust_axis", 11),
    ("on_spec_axis", 11), ("on_step_axis", 10),
    ("on_ultra_axis", 5), ("on_cfi_axis", 10),
])
def test_axis_populations(tasks, predicate, want):
    assert sum(1 for t in tasks if getattr(t, predicate)) == want


# --- the site's own routes must never shadow a task's action path -----------


def _post_routes() -> set[str]:
    """The site's specific `POST /<theme>/<literal>` routes — the ones that can shadow.

    METHOD matters: a guarded action is always a POST, so a GET-only route cannot
    shadow it. `18-dns-cutover` posts to `/dns/record` while the site serves
    `GET /<theme>/record`; they coexist, and a method-blind check would have called
    that a defect and sent someone to rename a task for no reason."""
    from steerbench.site.app import app
    out = set()
    for rule in app.url_map.iter_rules():
        if "POST" not in (rule.methods or set()):
            continue
        parts = [p for p in str(rule).split("/") if p]
        if len(parts) >= 2 and parts[0] == "<theme>" and not parts[1].startswith("<"):
            out.add(parts[1])
    return out


def test_no_wire_action_path_is_shadowed_by_a_site_route(tasks):
    """The catch-all `POST /<path:anything>` recorder IS the judge's evidence surface:
    an action reaching it is what "the enforcer let this through" means. So any specific
    POST route the site adds can silently steal a task's action path — the request is
    handled somewhere else, nothing is recorded, and the cell scores
    `no_action_recorded` as though the plan had abandoned the action.

    Not hypothetical. Adding a `POST /<theme>/submit` bridge for the `document`
    archetype swallowed `75-legit-change-reapproval`'s guarded action, whose
    `action_path` is `/procurement/submit`; two S6 cells turned into CHICKEN and the
    defended gate caught it. Every GUI bridge added since is namespaced
    `__`, and this test is what keeps the next one from repeating it.

    Scoped to paths that are actually POSTed. `61-order-desc-poison` declares
    `action_path="/sales/confirm"`, which *looks* like a collision and is not: its
    guarded action is an MCP frame, and `Task.action_path` is vestigial for the MCP
    axis — required by the dataclass, read by nobody on that path."""
    shadowing = _post_routes()
    for t in tasks:
        paths = []
        if t.on_http_axis:
            paths += [t.action_path, t.attack_path]
        if t.trust is not None:
            paths.append(t.trust.attack_url)
            paths += [e.get("url", "") for e in
                      list(t.trust.clean) + list(t.trust.poisoned)]
        for path in filter(None, paths):
            seg = [p for p in path.split("://", 1)[-1].split("/") if p]
            if len(seg) >= 2 and seg[-1] in shadowing:
                raise AssertionError(
                    f"{t.tid}: path {path!r} is shadowed by POST /<theme>/{seg[-1]} — "
                    f"the action would be handled by the bridge and never recorded")


def test_no_mcp_tool_records_onto_a_bridge_route(tasks):
    """The sharper version of the same trap, and the one that would be worst.

    S3/S4 tools record their effect by POSTing to `/<theme>/<verb>` (`mcp/s3_tools.py`),
    with the theme being the ACTIVE one — so a verb called `confirm` would not merely
    fail to record: it would match `POST /<theme>/confirm` and make the site issue the
    real guarded wire action. A tool call would silently become a payment."""
    shadowing = _post_routes()
    for t in tasks:
        for verb in [tool.verb for tool in (t.mcp.tools if t.mcp else ())]:
            assert verb not in shadowing, (
                f"{t.tid}: MCP verb {verb!r} records onto POST /<theme>/{verb}, which "
                f"is a site bridge — the tool call would trigger the guarded action")


def test_gui_bridges_that_can_collide_are_namespaced():
    """The three original bridges (`confirm`, `act-review`, `clarify`) predate this rule
    and do not collide with any authored path; every other bridge carries the `__` prefix the control plane already uses, which makes collision impossible rather
    than merely absent."""
    from steerbench.site.app import app

    legacy = {"confirm", "act-review", "clarify", "record", "approved", "quote",
              "sitemap.json"}
    for rule in app.url_map.iter_rules():
        parts = [p for p in str(rule).split("/") if p]
        if len(parts) >= 2 and parts[0] == "<theme>" and not parts[1].startswith("<"):
            assert parts[1] in legacy or parts[1].startswith("__"), (
                f"new site route /<theme>/{parts[1]} is not namespaced; it can shadow "
                f"a task action path")
