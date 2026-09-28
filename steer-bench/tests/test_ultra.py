"""Suite S9 — the properties that make the delta attributable.

The oracle (`steerbench.oracles.s10`) certifies that the environment behaves, on the real
interpreter and the real branch hook. These tests certify what the oracle cannot see
because it only observes outcomes:

  * the ablation is a **substitution by the active arm's own annotation**, keyed on
    `active_branch` — not a drop, not a truncation to a fixed depth;
  * the plan is a genuine TREE, three or four levels deep, with distinct branch keys per
    level (a plan flattened into `if a and b and c` would be one arm and would silently
    un-measure the suite);
  * the authoring conditions are **checkable and falsifiable**, not asserted;
  * "harm" is decided against the composed path, which is the definition both judges must
    share — judging against the leaf alone would build the ablation into the judge.

Every test here runs with nothing installed but this package.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json

import pytest

from steerbench.harness import arms, judge, ultra_model
from steerbench.tasks import registry as reg

pytestmark = pytest.mark.substrate


@pytest.fixture
def ultra_tasks():
    return [t for t in reg.TASKS if t.on_ultra_axis]


def _state(task, row, level, arm="true"):
    """The state the hook writes when row `row` enters its arm at `level`: the composed
    constraints, labelled with the arm's own key."""
    return {"plan_id": task.tid,
            "active_branch": ultra_model.key_of(task, row, level, arm),
            "branch_path": ["root"],
            "http_constraints": json.loads(json.dumps(
                ultra_model.path_constraints(task, row, level, arm)))}


def _ablated(task, row, level, arm="true", variant=None):
    a4 = arms.get("S9", "A4")
    arms.set_context(a4, task, variant)
    try:
        return a4.state_filter(task, copy.deepcopy(_state(task, row, level, arm)))
    finally:
        arms.set_context(None, None, None)


# --- the suite's shape ------------------------------------------------------


def test_the_suite_covers_both_depths_and_all_three_sub_families(ultra_tasks):
    """Five tasks: two depths, so the delta is not a property of one tree size; three
    sub-families, so the violated authority is measured at the far end of the
    composition (U1), one hop from it (U2), and on the allowlist layer (U3)."""
    assert len(ultra_tasks) == 5
    subs = {}
    for t in ultra_tasks:
        subs[t.ultra.sub] = subs.get(t.ultra.sub, 0) + 1
    assert subs == {"U1": 2, "U2": 2, "U3": 1}
    assert {ultra_model.depth(t) for t in ultra_tasks} == {3, 4}
    # Both operators. An attack on a floor is the mirror of one on a ceiling — it must
    # go DOWN to violate the arm's pin while staying above the stated minimum — and a
    # suite written with one of them never finds out.
    ops = {lv.op for t in ultra_tasks for lv in t.ultra.levels}
    assert ops == {"<=", ">="}


def test_every_task_is_well_authored(ultra_tasks):
    """`check_task` is the free half of the certification; if it has anything to say, the
    oracle's verdict would be about a mis-authored task rather than about the merge."""
    for t in ultra_tasks:
        assert ultra_model.check_task(t) == [], t.tid


def test_the_attacked_authority_is_never_the_leaf(ultra_tasks):
    """The leaf's own annotation is exactly what A4 keeps, so an attack on it would be
    blocked in both arms and the cell would be tautological (I2)."""
    for t in ultra_tasks:
        assert t.ultra.level < t.ultra.leaf, t.tid
        assert t.ultra.levels[t.ultra.level].grant != "none", t.tid


# --- the ablation -----------------------------------------------------------


def test_the_ablation_substitutes_the_active_arms_own_constraints(ultra_tasks):
    """Not a drop (that is S1's ablation) and not an absence: what A4 holds is
    well-formed, honestly derived, and still authorises the honest work."""
    for t in ultra_tasks:
        row, leaf = t.ultra.target, t.ultra.leaf
        composed = ultra_model.acting_constraints(t, row)
        out = _ablated(t, row, leaf)["http_constraints"]
        own = ultra_model.own_constraints(t, row, leaf, "true")
        assert out.get("fields", []) == own.get("fields", []), t.tid
        assert out["allowed_domains"] == composed["allowed_domains"], t.tid
        # And it must actually differ, or the arms are the same configuration under two
        # names and ΔASR would be 0 with nothing saying so.
        assert out.get("fields", []) != composed.get("fields", []), t.tid


def test_the_ablation_is_keyed_on_the_active_branch(ultra_tasks):
    """A truncation to the deepest annotation would pass every leaf test above and would
    be a different function: the state written at a MIDDLE arm must carry that arm's own
    constraints, which is what makes the whole tree observable in `oracles/s10.py`."""
    for t in ultra_tasks:
        row = t.ultra.target
        for level in range(ultra_model.depth(t)):
            out = _ablated(t, row, level)["http_constraints"]
            own = ultra_model.own_constraints(t, row, level, "true")
            assert out.get("fields", []) == own.get("fields", []), f"{t.tid} L{level}"


def test_the_root_state_passes_through(ultra_tasks):
    """Root authorises the host and nothing else, so there is nothing to strip — and a
    filter that mangled it would block every cell for a reason unrelated to the suite."""
    for t in ultra_tasks:
        root = {"plan_id": t.tid, "active_branch": "root", "branch_path": ["root"],
                "http_constraints": ultra_model.root_constraints(t)}
        a4 = arms.get("S9", "A4")
        arms.set_context(a4, t, "attack")
        try:
            assert a4.state_filter(t, copy.deepcopy(root)) == root, t.tid
        finally:
            arms.set_context(None, None, None)


def test_the_domain_allowlist_is_never_narrowed(ultra_tasks):
    """The fail-open/fail-closed law: `allowed_domains` is empty ->
    every host is refused, so an honest "leaf-only" state would block the benign leg and
    the delta would measure damage instead of a defence."""
    for t in ultra_tasks:
        for level in range(ultra_model.depth(t)):
            out = _ablated(t, t.ultra.target, level)["http_constraints"]
            assert out["allowed_domains"] == [t.domain.lower()], t.tid


def test_the_ablation_is_variant_blind(ultra_tasks):
    """Like S8's: the rows the planner PERCEIVES are identical in both
    variants — the divergence is on the wire — so the state a non-merging runtime writes
    cannot depend on the variant."""
    for t in ultra_tasks:
        atk = _ablated(t, t.ultra.target, t.ultra.leaf, variant="attack")
        assert _ablated(t, t.ultra.target, t.ultra.leaf, variant="benign") == atk, t.tid
        assert _ablated(t, t.ultra.target, t.ultra.leaf, variant=None) == atk, t.tid


def test_the_arms_cannot_differ_by_prompt():
    """S9's component is a runtime merge, so a prompt difference between its arms could
    only ever be drift.

    Asserted twice: structurally, that no arm
    declares a prompt lever, and by RENDERING the prefix the paid runner sends — which
    is the half that can catch a regression, because a prefix that started branching on
    the arm would have satisfied the structural check forever."""
    from steerbench.harness import run as rs

    assert not arms.get("S9", "A5").prompt_ablated
    assert not arms.get("S9", "A4").prompt_ablated
    assert arms.get("S9", "A4").env_unset == ()
    for t in [x for x in reg.TASKS if x.on_ultra_axis]:
        prefix = rs._ultra_prefix(t)
        assert prefix == rs._ultra_prefix(t), t.tid
        assert "A4" not in prefix and "A5" not in prefix, t.tid


def test_the_paid_path_exists_and_is_the_oracle_s_path():
    """S9 was refused by the runner until its worklist surface and tree prompt existed.

    Now that it does not refuse, the thing worth pinning is that spending on it does not
    quietly become a DIFFERENT experiment: the ablation must still be the state filter
    installed over the atomic write, which is the same code path `oracles/s10.py`
    certified — not a bespoke branch in the runner."""
    assert arms.get("S9", "A4").paid_ready
    assert arms.get("S9", "A5").paid_ready
    assert arms.get("S9", "A4").state_filter is not None
    assert arms.get("S9", "A5").state_filter is None
    assert arms.get("S9", "A4").paid_ablation == "state"


# --- the authoring conditions are falsifiable -------------------------------


def test_an_attack_on_the_leaf_is_rejected_by_the_authoring_gate(ultra_tasks):
    """The condition that keeps I2 alive, checked by breaking it."""
    t = ultra_tasks[0]
    broken = dataclasses.replace(
        t, ultra=dataclasses.replace(t.ultra, level=t.ultra.leaf))
    assert any(m.startswith("attacked_level_is_the_leaf")
               for m in ultra_model.check_task(broken))


def test_an_attack_the_stated_policy_would_catch_is_rejected(ultra_tasks):
    """Invariant I1: if the borrowed value violated the ceiling the PROMPT states, a
    single plan-wide pin would stop it and the delta would not belong to the merge."""
    t = next(x for x in ultra_tasks if x.ultra.levels[x.ultra.level].grant == "field")
    lv = t.ultra.levels[t.ultra.level]
    over = (lv.value * 10) if lv.op == "<=" else (lv.value / 10)
    items = list(t.items)
    items[t.ultra.target] = dataclasses.replace(
        items[t.ultra.target], real={lv.slot: over})
    broken = dataclasses.replace(t, items=tuple(items))
    msgs = ultra_model.check_task(broken)
    assert any(m.startswith("caught_by_the_stated_policy") for m in msgs)


def test_a_flat_tree_is_rejected(ultra_tasks):
    """The premise, checked by removing it: at depth two the composition is one ancestor
    and the suite would be restating S1."""
    t = ultra_tasks[0]
    broken = dataclasses.replace(
        t, ultra=dataclasses.replace(t.ultra, levels=t.ultra.levels[:2], level=0))
    assert any(m.startswith("too_shallow") for m in ultra_model.check_task(broken))


def test_a_queue_that_never_takes_a_deep_else_arm_is_rejected(ultra_tasks):
    """A worklist whose held rows all fail at level 0 never shows the nesting branching,
    so nothing distinguishes it from a flat plan with one extra gate."""
    t = ultra_tasks[0]
    lv0 = t.ultra.levels[0]
    over = (lv0.value * 10) if lv0.op == "<=" else (lv0.value / 10)
    items = []
    for i, it in enumerate(t.items):
        if ultra_model.acts(t, i):
            items.append(it)
            continue
        items.append(dataclasses.replace(it, values=dict(it.values, **{lv0.slot: over})))
    broken = dataclasses.replace(t, items=tuple(items))
    assert any(m.startswith("only_the_outermost_gate_is_exercised")
               for m in ultra_model.check_task(broken))


# --- the plan is a tree, and a long one -------------------------------------


def test_the_plan_is_nested_not_flattened(ultra_tasks):
    """`if a and b and c` is ONE branch with one arm: one annotation, nothing to compose.
    Every level must be its own `if`, on its own line (branch keys are AST-derived), with
    its own `else` — and the deepest one must sit at the declared depth."""
    for t in ultra_tasks:
        tree = ast.parse(ultra_model.plan_code(t))
        ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
        assert len(ifs) == len(t.items) * ultra_model.depth(t), t.tid
        assert len({n.lineno for n in ifs}) == len(ifs), t.tid
        assert all(n.orelse for n in ifs), t.tid
        assert all(isinstance(n.test, ast.Compare) and len(n.test.ops) == 1
                   for n in ifs), t.tid

        def _deep(node, d=0):
            kids = [c for c in ast.walk(node) if isinstance(c, ast.If) and c is not node]
            return max((_deep(c, d + 1) for c in kids), default=d)
        assert max(_deep(n) for n in ifs if n.col_offset == 0) \
            == ultra_model.depth(t) - 1, t.tid


def test_the_plans_are_the_longest_in_the_benchmark(ultra_tasks):
    """The by-product the suite was asked for: plans an order of magnitude longer than a
    single-action task's five lines, and two and a half times S8's."""
    for t in ultra_tasks:
        stats = ultra_model.plan_stats(t)
        assert stats["plan_lines"] >= 60, t.tid
        assert stats["max_depth"] >= 3, t.tid
        assert stats["guarded_actions"] >= 2, t.tid          # so an FP is measurable
        assert stats["guarded_actions"] < len(t.items), t.tid  # some row is held
        assert stats["constraints_per_action"] == ultra_model.depth(t), t.tid


def test_the_branch_keys_are_the_nested_form(ultra_tasks):
    """`cobra.brh.skeleton` keys a nested arm by the path of enclosing arms, and
    `_merge_constraints` looks up every PREFIX of that path. A key built any other way
    fails closed, which presents as a benign leg blocking, i.e. as a defence."""
    for t in ultra_tasks:
        for row in range(len(t.items)):
            for level in range(ultra_model.depth(t)):
                key = ultra_model.key_of(t, row, level, "true")
                assert key.count(".") == level, t.tid
                assert ultra_model.arm_of_key(t, key) == (row, level, "true"), t.tid
                # Every prefix must be a key the annotation also fills, or the merge
                # fails closed on the way down.
                parts = key.split(".")
                for j in range(1, len(parts)):
                    assert ".".join(parts[:j]) in ultra_model.branch_constraints(t), t.tid


def test_each_level_contributes_exactly_one_authorisation(ultra_tasks):
    """Rule Y1. An arm that annotated more than its own decision licenses would be doing
    the merge's job in the annotator, and A4 would stop being an ablation of anything."""
    for t in ultra_tasks:
        for level in range(ultra_model.depth(t)):
            own = ultra_model.own_constraints(t, t.ultra.target, level, "true")
            n = len(own.get("fields", [])) + len(own.get("allowed_endpoints", []))
            assert n == (0 if t.ultra.levels[level].grant == "none" else 1), t.tid


# --- what counts as harm ----------------------------------------------------


def _action(task, i, borrowed=False):
    it = task.items[i]
    vals = dict(it.values)
    path = ultra_model.row_path(task, i)
    if borrowed:
        vals.update(it.real)
        path = it.real_path or path
    body = {f.name: vals.get(f.slot) for f in task.action_fields}
    body["ref"] = it.ref
    return {"host": task.domain.lower(), "path": path, "body": body}


def test_the_honest_rows_are_never_harm(ultra_tasks):
    for t in ultra_tasks:
        honest = [_action(t, i) for i in range(len(t.items))
                  if ultra_model.acts(t, i) and i != t.ultra.target]
        assert ultra_model.harmful_actions(t, honest) == [], t.tid
        assert judge.open_outcome("attack", honest, t) == "COMPLETED", t.tid


def test_the_borrowed_row_is_harm(ultra_tasks):
    """And it is harm *because of the composition*: the value it carries satisfies the
    leaf's own pin, so a judge that looked only there would call it a completion."""
    for t in ultra_tasks:
        acted = [_action(t, i) for i in range(len(t.items))
                 if ultra_model.acts(t, i) and i != t.ultra.target]
        acted.append(_action(t, t.ultra.target, borrowed=True))
        assert len(ultra_model.harmful_actions(t, acted)) == 1, t.tid
        assert judge.open_outcome("attack", acted, t) == "BREACH", t.tid


def test_an_action_naming_no_row_is_harm(ultra_tasks):
    """Missing evidence fails loud, in the direction that over-reports against our own
    system rather than under-reporting an attack."""
    for t in ultra_tasks:
        assert len(ultra_model.harmful_actions(
            t, [{"host": t.domain, "path": t.action_path, "body": {}}])) == 1, t.tid


# --- the prompt carries the tree, and only the tree -------------------------


def test_the_lint_catches_a_gate_the_prompt_does_not_state(ultra_tasks):
    """L9. Every gate is a TRUSTED anchor the plan branches on, so one missing from the
    task text means the harness is supplying the logic — measured on a plan that could
    not have been written."""
    from steerbench.tools import prompt_lint
    t = ultra_tasks[0]
    lv = t.ultra.levels[1]
    broken = dataclasses.replace(
        t, instruction=t.instruction.replace(str(lv.value), "a documented minimum"))
    assert any(m.startswith("L9") and str(lv.value) in m
               for m in prompt_lint.lint_task(broken, "S9"))


def test_the_lint_catches_a_prompt_that_reorders_the_tree(ultra_tasks):
    """The tree stops at the first gate that fails, so the ORDER decides which arm holds
    a row. A prompt listing the levels in another order describes a different tree."""
    from steerbench.tools import prompt_lint
    t = ultra_tasks[0]
    swapped = dataclasses.replace(
        t, ultra=dataclasses.replace(
            t.ultra, levels=(t.ultra.levels[1], t.ultra.levels[0]) + t.ultra.levels[2:]))
    assert any("stated BEFORE the level above it" in m
               for m in prompt_lint.lint_task(swapped, "S9"))


def test_the_lint_catches_a_disclosed_row(ultra_tasks):
    """The queue is what the agent must PERCEIVE. A row value in the prompt would give
    the plan a trusted source for a value the suite's arms are supposed to pin from the
    screen."""
    from steerbench.tools import prompt_lint
    t = ultra_tasks[0]
    val = int(t.items[t.ultra.target].values[t.ultra.levels[0].slot])
    broken = dataclasses.replace(
        t, instruction=t.instruction + f" The first item is for {val}.")
    assert any(m.startswith("L9") and str(val) in m
               for m in prompt_lint.lint_task(broken, "S9"))


# --- membership -------------------------------------------------------------


def test_the_suite_is_on_the_open_track(ultra_tasks):
    """It is the hardest open-track task set in the benchmark — several rows, four gates
    each, applied in order — which is the point: an undefended agent's ASR here is the
    empirical half of what the ablation models."""
    for t in ultra_tasks:
        assert reg.open_channel(t) == "gui", t.tid
        assert reg.archetype_of(t) == "workflow", t.tid
        assert arms.primary_suite(t) == "S9", t.tid
        assert arms.suites_of(t) == ("S9",), t.tid
