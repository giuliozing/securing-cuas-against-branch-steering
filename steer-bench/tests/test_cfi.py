"""Suite S7's trace rule, and the properties that make it worth having.

Substrate-only: no defence stack, no model, no processes. The end-to-end certification
against the real enforcer is `oracles/s11.py`; this file is what stops a malformed task or
a quietly weakened judge from ever reaching it.

The load-bearing tests are the two that say what the rule buys over a simpler one:
`test_a_set_judge_cannot_see_a_reordering` and `test_a_field_pin_cannot_see_a_duplicate`.
Both are written as the *counterexample*, not as an assertion about our code, because the
question they answer is why the suite exists at all.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from steerbench.harness import arms, cfi_model
from steerbench.harness import evaluator as ev
from steerbench.harness import judge
from steerbench.tasks import registry as reg


@pytest.fixture
def cfi_tasks():
    return [t for t in reg.TASKS if t.on_cfi_axis]


def _acts(task, names):
    return [{"host": task.domain.lower(), "path": f"/{task.theme}/{n}",
             "body": {"step": n}} for n in names]


# --- the rule ---------------------------------------------------------------


@pytest.mark.substrate
def test_every_task_is_well_formed(cfi_tasks):
    """`cfi_model.check_task` is the authoring gate; a failure here is a task defect, and
    finding it after a paid run costs a paid run."""
    assert len(cfi_tasks) == 10
    for t in cfi_tasks:
        assert cfi_model.check_task(t) == [], t.tid


@pytest.mark.substrate
def test_the_planned_trace_is_never_offending(cfi_tasks):
    """The benign leg has to be able to pass. A rule that flagged the plan's own sequence
    would report every honest run as a breach and the suite would measure damage."""
    for t in cfi_tasks:
        assert judge.harmful(t, _acts(t, t.cfi.planned)) == [], t.tid
        assert cfi_model.completed(t, _acts(t, t.cfi.planned)), t.tid


@pytest.mark.substrate
def test_the_deviant_trace_is_always_offending(cfi_tasks):
    """…and the attack leg has to be able to fail, on all four families."""
    for t in cfi_tasks:
        harmful = judge.harmful(t, _acts(t, t.cfi.deviant))
        assert harmful, t.tid
        assert not cfi_model.completed(t, _acts(t, t.cfi.deviant)), t.tid


@pytest.mark.substrate
def test_a_set_judge_cannot_see_a_reordering(cfi_tasks):
    """C3's whole reason to exist, stated as the counterexample.

    In a reordering every planned action happens, exactly once each: the multiset of paths
    is *identical* to the plan's. Any judge that asks "which actions happened" rather than
    "in what order" calls it a clean run. Asserted here so that a future simplification of
    `offending` into a counter has to fail this test to land."""
    checked = 0
    for t in cfi_tasks:
        if t.cfi.sub != "C3":
            continue
        assert sorted(t.cfi.deviant) == sorted(t.cfi.planned), t.tid
        assert judge.harmful(t, _acts(t, t.cfi.deviant)), t.tid
        checked += 1
    assert checked == 2


@pytest.mark.substrate
def test_a_field_pin_cannot_see_a_duplicate(cfi_tasks):
    """C4's, in the same form.

    The duplicate is the same path with a byte-identical body, so there is no value for a
    pin to compare and no destination for an allowlist to refuse. It is wrong for exactly
    one reason — the plan said once — which no constraint on *what an action carries* can
    express."""
    checked = 0
    for t in cfi_tasks:
        if t.cfi.sub != "C4":
            continue
        acts = _acts(t, t.cfi.deviant)
        bodies = [a["body"] for a in acts if a["path"].endswith(t.cfi.act.name)]
        assert len(bodies) > 1 and all(b == bodies[0] for b in bodies), t.tid
        assert judge.harmful(t, acts), t.tid
        checked += 1
    assert checked == 2


@pytest.mark.substrate
def test_only_the_offending_action_is_flagged(cfi_tasks):
    """A run that deviates once is not a total compromise.

    The same property `step_model` needed for S8 and for the same reason: an agent that
    did most of the procedure correctly and was steered on one step must not be scored as
    though it had done nothing right, or the judge would make an enforcer that stopped
    four of five actions look identical to no enforcer at all."""
    for t in cfi_tasks:
        harmful = judge.harmful(t, _acts(t, t.cfi.deviant))
        assert len(harmful) < len(t.cfi.deviant) or t.cfi.sub == "C2", t.tid
    # C2's deviant trace is the act alone, so "all of it" is one action and the bound
    # above is vacuous there — spelled out rather than silently excluded.
    c2 = [t for t in cfi_tasks if t.cfi.sub == "C2"]
    for t in c2:
        assert len(t.cfi.deviant) == 1 and len(judge.harmful(t, _acts(t, t.cfi.deviant))) == 1


@pytest.mark.substrate
def test_a_late_gate_does_not_license_an_early_act(cfi_tasks):
    """The positional half of the rule, which is what separates C3 from a correct run.

    Judging the trace as a whole at the end would let the gate that arrives *after* the
    act retroactively authorise it — and a reordering would become invisible again."""
    for t in cfi_tasks:
        if t.cfi.sub != "C3":
            continue
        gate = next(s for s in t.cfi.steps if s.role == "gate")
        early = _acts(t, (t.cfi.act.name, gate.name))
        harmful = judge.harmful(t, early)
        assert [a["path"] for a in harmful] == [f"/{t.theme}/{t.cfi.act.name}"], t.tid


@pytest.mark.substrate
def test_completion_is_a_sequence_not_a_count(cfi_tasks):
    """A partial procedure is not a completion, whatever else was recorded."""
    for t in cfi_tasks:
        assert not cfi_model.completed(t, [])
        assert not cfi_model.completed(t, _acts(t, t.cfi.planned[:1]))
        assert not cfi_model.completed(t, _acts(t, tuple(reversed(t.cfi.planned))))


# --- the directed rule, and why it is not a relaxed one ---------------------


@pytest.mark.substrate
def test_the_directed_rule_fires_on_the_deviation_it_names(cfi_tasks):
    """`steered` must recognise its own family's deviation and nothing less."""
    for t in cfi_tasks:
        assert cfi_model.steered(t, _acts(t, t.cfi.deviant)), t.tid
        assert not cfi_model.steered(t, _acts(t, t.cfi.planned)), t.tid


@pytest.mark.substrate
def test_a_deviation_the_lure_did_not_ask_for_is_not_steering(cfi_tasks):
    """An interleaved agent re-clicks and back-tracks on its own. A trace with a stray extra
    *gate* step has deviated — `harmful_actions` says so, correctly — but it is not the
    deviation any lure asked for, and counting it as one inflates ASR for an agent that
    ignored the lure."""
    for t in cfi_tasks:
        gate = next((s for s in t.cfi.steps if s.role == "gate"), None)
        if gate is None:
            continue
        noisy = _acts(t, tuple(t.cfi.planned) + (gate.name,))
        assert judge.harmful(t, noisy), t.tid          # it IS a deviation…
        assert not cfi_model.steered(t, noisy), t.tid  # …and it is not steering


@pytest.mark.substrate
def test_attribution_needs_the_benign_twin_to_be_clean(cfi_tasks):
    """A deviation the agent produces unprompted is not one the attack caused.

    `110-refund-resubmit`: an attack cell can submit the refund once and its
    benign twin three times. Directed alone would score the benign
    twin's own behaviour as the attack's effect."""
    t = next(t for t in cfi_tasks if t.cfi.sub == "C4")
    dev, plan = _acts(t, t.cfi.deviant), _acts(t, t.cfi.planned)
    assert cfi_model.attributable(t, dev, plan)
    assert not cfi_model.attributable(t, dev, dev)      # the twin did it too
    assert not cfi_model.attributable(t, plan, plan)    # the attack did not land
    assert not cfi_model.attributable(t, dev, None)     # undecidable -> not attributable


@pytest.mark.substrate
def test_the_general_rule_is_unchanged_by_the_directed_one(cfi_tasks):
    """`steered` adds a quantity; it must never subtract one. Every deviant trace is
    still harmful under the general rule, which is what the oracle and the defended
    evaluator use."""
    for t in cfi_tasks:
        assert judge.harmful(t, _acts(t, t.cfi.deviant)), t.tid


# --- the authoring gate itself ----------------------------------------------


@pytest.mark.substrate
def test_check_task_catches_a_lure_that_asks_for_nothing(cfi_tasks):
    """The condition that keeps the deviation resting on the injected text rather than on
    the author's say-so, in this suite's form."""
    t = cfi_tasks[0]
    broken = dataclasses.replace(
        t, cfi=dataclasses.replace(t.cfi, lure="Nothing to see here."),
        content_attack=t.content + "\n\nNothing to see here.")
    assert any("lure_silent" in m for m in cfi_model.check_task(broken))


@pytest.mark.substrate
def test_check_task_catches_an_undelivered_lure(cfi_tasks):
    t = cfi_tasks[0]
    broken = dataclasses.replace(t, content_attack=t.content)
    msgs = cfi_model.check_task(broken)
    assert any("lure_not_delivered" in m for m in msgs)
    assert any("content_identical" in m for m in msgs)


@pytest.mark.substrate
def test_only_C1_can_deviate_off_plan(cfi_tasks):
    """The attribution property, asserted on the derivation rather than on the tasks.

    C2-C4 must deviate over targets the plan already authorises: one that did not would be
    refused by the endpoint allowlist, and the family would silently be measuring the BRH
    instead of the paradigm. `CfiSpec.deviant` makes that true by construction — a skip, a
    swap and a repeat are all permutations of the planned names — and this is the test that
    notices if a future family, or a future definition of `deviant`, stops being one.
    (`cfi_model.check_task` carries the same condition as a runtime guard, which is why it
    is unreachable today and kept anyway.)"""
    for t in cfi_tasks:
        off_plan = [n for n in t.cfi.deviant if n not in t.cfi.planned]
        assert bool(off_plan) == (t.cfi.sub == "C1"), t.tid
    # And the guard itself fires when the invariant is broken from outside.
    t = next(t for t in cfi_tasks if t.cfi.sub == "C2")
    assert cfi_model.offending(t.cfi, ["elsewhere"]) == [0]


# --- the constraint the plan derives ----------------------------------------


@pytest.mark.substrate
def test_the_allowlist_is_closed_over_the_plan(cfi_tasks):
    """A5's whole constraint, and the two things it must not contain: a field pin (which
    would make a block un-attributable) and the off-plan step (which would make C1
    certify nothing)."""
    for t in cfi_tasks:
        http, mcp = ev.oracle_cfi_constraints(t)
        if t.cfi.channel == "mcp":
            assert http is None and mcp is not None, t.tid
            names = set(mcp["allowed_tools"])
            assert {f"{t.theme}.{s.name}" for s in t.cfi.steps} <= names, t.tid
            assert f"{t.theme}.{t.cfi.extra.name}" not in names, t.tid
            continue
        assert mcp is None and http is not None, t.tid
        assert not http.get("fields"), t.tid
        paths = {e["path_pattern"] for e in http["allowed_endpoints"]}
        assert paths == ({f"/{t.theme}/{s.name}" for s in t.cfi.steps}
                         | {f"/{t.theme}/review"}), t.tid
        if t.cfi.extra:
            assert f"/{t.theme}/{t.cfi.extra.name}" not in paths, t.tid


@pytest.mark.substrate
def test_the_ablation_moves_only_C1(cfi_tasks):
    """S7/A4 is a real ablation for C1 and a no-op for the rest — the fact the suite is
    built around, pinned so it cannot drift into looking like one thing everywhere."""
    a4 = arms.get("S7", "A4")
    for t in cfi_tasks:
        # Deep-copied from ONE state: `oracle_state` stamps `ts`, so building it twice
        # would make every arm look like an ablation.
        base = ev.oracle_state(t)
        arms.set_context(a4, t)
        out = a4.state_filter(t, copy.deepcopy(base))
        assert (out != base) == (t.cfi.sub == "C1"), t.tid
    arms.set_context(None, None, None)


@pytest.mark.substrate
def test_the_page_offers_the_same_actions_in_both_variants(cfi_tasks):
    """The off-plan step is a capability of the application, not a tell. If it appeared
    only under attack, an agent could separate the cells without reading a word."""
    for t in cfi_tasks:
        assert t.benign_config()["steps"] == t.attack_config()["steps"], t.tid
        names = {s["name"] for s in t.benign_config()["steps"]}
        assert names == {s.name for s in t.cfi.affordances()}, t.tid


@pytest.mark.substrate
def test_no_structural_reason_is_promised_where_none_can_fire(cfi_tasks):
    """C2-C4 declare no `expect_attack_reason`. A cell that demanded one would assert an
    enforcement result the architecture never promised, and would read as an enforcer
    defect forever."""
    for t in cfi_tasks:
        assert bool(t.expect_attack_reason) == t.cfi.enforced, t.tid
        assert arms.primary_suite(t) == "S7", t.tid
