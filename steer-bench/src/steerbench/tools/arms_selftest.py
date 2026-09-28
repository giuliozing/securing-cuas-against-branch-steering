"""LLM-free self-test for the arm machinery (`arms.py`).

The arm system is the one piece whose failure would be **silent and directional**: if
a filter does not actually reach the state the enforcer reads, `A4` quietly behaves
like `A5`, ΔASR collapses to 0, and the honest conclusion ("the component does
nothing") is indistinguishable from the bug. So it gets its own test.

Covers, in order of how easy each is to get wrong:

  1. **Both bindings of `atomic_write_json`.** `hook.py:49` does `from cobra.brh.writer
     import atomic_write_json`, so it holds its OWN reference. Patching only the writer
     module would miss every per-branch transition — i.e. almost every write — while
     still passing a naive test that exercises the writer path.
  2. **Filter semantics per suite**, including the asymmetry that S2 *widens* the domain
     allowlist rather than dropping it (dropping is fail-closed in `brh_check`).
  3. **A5 is inert.** An arm-less run must be byte-identical with the machinery loaded.
  4. **Non-branch_state writes untouched** (`plan_constraints.json` must pass through).

Usage:
    python -m steerbench.tools.arms_selftest
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

from steerbench.harness import arms
from steerbench import config
from steerbench.harness import evaluator as ev
from steerbench.tasks import registry as reg

sys.path.insert(0, str(config.cobra_src()))

RESULTS: list[tuple[bool, str]] = []


def check(cond: bool, label: str) -> None:
    RESULTS.append((bool(cond), label))
    print(f"  [{'OK ' if cond else 'XXX'}] {label}")


def main() -> int:
    from cobra.brh import hook as _hook
    from cobra.brh import writer as _writer

    arms.install()
    tmp = Path(tempfile.mkdtemp(prefix="arms_selftest_"))
    state_p = tmp / "branch_state.json"
    other_p = tmp / "plan_constraints.json"

    field_task = next(t for t in reg.TASKS if t.attack_kind == "field")
    dom_task = next(t for t in reg.TASKS if t.attack_kind == "domain")
    ep_task = next(t for t in reg.TASKS if t.attack_kind == "endpoint")

    def write_via(mod, task, arm, path=state_p):
        arms.set_context(arm, task)
        mod.atomic_write_json(path, ev.oracle_state(task))
        return json.loads(path.read_text())

    print("1. both bindings are patched")
    s = write_via(_writer, field_task, arms.get("S1", "A4"))
    check("fields" not in s["http_constraints"], "writer binding: S1/A4 drops fields")
    s = write_via(_hook, field_task, arms.get("S1", "A4"))
    check("fields" not in s["http_constraints"],
          "hook binding: S1/A4 drops fields (the from-import trap)")

    print("2. filter semantics")
    s = write_via(_writer, dom_task, arms.get("S2", "A4"))
    doms = s["http_constraints"]["allowed_domains"]
    check(dom_task.attacker_domain.lower() in doms,
          f"S2/A4 domain: attacker host admitted ({dom_task.attacker_domain})")
    check(dom_task.domain.lower() in doms,
          "S2/A4 domain: WIDENED not dropped (empty allowlist is fail-closed)")
    s = write_via(_writer, ep_task, arms.get("S2", "A4"))
    check("allowed_endpoints" not in s["http_constraints"],
          "S2/A4 endpoint: allowed_endpoints dropped")

    print("3. A5 is inert")
    ref = ev.oracle_state(field_task)
    s = write_via(_writer, field_task, arms.get("S1", "A5"))
    check(s["http_constraints"] == ref["http_constraints"],
          "S1/A5 leaves the state untouched")
    arms.set_context(None, None)
    _writer.atomic_write_json(state_p, ref)
    check(json.loads(state_p.read_text())["http_constraints"] == ref["http_constraints"],
          "no context (arm-less run) leaves the state untouched")

    print("4. only branch_state is filtered")
    arms.set_context(arms.get("S1", "A4"), field_task)
    _writer.atomic_write_json(other_p, ev.oracle_state(field_task))
    check("fields" in json.loads(other_p.read_text())["http_constraints"],
          "plan_constraints.json passes through unfiltered")

    print("5. the source state is not mutated in place")
    src = ev.oracle_state(field_task)
    before = json.dumps(src, sort_keys=True)
    arms.set_context(arms.get("S1", "A4"), field_task)
    _writer.atomic_write_json(state_p, src)
    check(json.dumps(src, sort_keys=True) == before,
          "caller's dict survives the filter (deepcopy, no aliasing)")

    print("6. posture arms carry NO state filter (S4, S6) and lever arms carry levers")
    # S4 and S6 ablate upstream of the state — the second enforcer is physically off
    # the path (S4), the trust gate never runs (S6). A state filter here would model
    # the symptom and certify a defence that never happened, and for
    # S4 it would invert the result outright: with `mcp_constraints` merely removed,
    # `check_tools_call` blocks EVERYTHING, so the ablated system would look stronger.
    for suite in ("S4", "S6"):
        check(arms.get(suite, "A4").state_filter is None,
              f"{suite}/A4 is a posture, not a state filter")
    check("BRH_SEED_FIELD_POLICY" in arms.get("S1", "A4").env_unset,
          "S1/A4 unsets the field seed (else the seed re-injects the ablated pins)")
    check("BRH_COMMIT_GATE" in arms.get("S5", "A4").env_unset,
          "S5/A4 unsets the commit gate (provenance has TWO layers; A4 removes both)")

    print("7. S3/A4 ablates one MCP proxy mechanism per sub-family")
    mcp_task = next((t for t in reg.TASKS if t.on_mcp_axis and t.mcp.sub == "M2"), None)
    if mcp_task is not None:
        st = ev.oracle_state(mcp_task)
        st["mcp_constraints"] = ev.oracle_mcp_constraints(mcp_task)
        check("param_rules" in (st["mcp_constraints"] or {}), "M2 pins param_rules under A5")
        out = arms.get("S3", "A4").state_filter(mcp_task, json.loads(json.dumps(st)))
        check("param_rules" not in out["mcp_constraints"],
              "S3/A4 drops param_rules but keeps the tool allowlist (fail-closed layer)")
        check("allowed_tools" in out["mcp_constraints"],
              "S3/A4 keeps allowed_tools (dropping it would block everything)")

    print("8. S5/A4 drops pins on the COMMITTED slots only, and not on the falsifier leg")
    # 01-bank-wire commits `account` and gates on `amount`: the ablation is provenance,
    # so the threshold pin must survive untouched while the commitment pin goes. The
    # pin removed here is an annotator-emitted `var:` placeholder that never resolves and
    # would otherwise defend A4.
    spec_task = next((t for t in reg.TASKS
                      if t.on_spec_axis and t.tid.startswith("01")), None)
    if spec_task is not None:
        f5 = arms.get("S5", "A4").state_filter
        st = {"http_constraints": {"allowed_domains": ["bank.local"], "fields": [
            {"path": "account", "op": "==", "value": "var:account"},
            {"path": "amount", "op": "<=", "value": 2000}]}}
        os.environ.pop("BRH_COMMIT_GATE", None)
        out = f5(spec_task, json.loads(json.dumps(st)))
        paths = [f["path"] for f in out["http_constraints"]["fields"]]
        check("account" not in paths,
              "S5/A4 drops the committed-slot pin (else the annotator defends the arm)")
        check("amount" in paths,
              "S5/A4 keeps the non-committed pin (the ablation is provenance, not fields)")
        os.environ["BRH_COMMIT_GATE"] = "{}"
        out = f5(spec_task, json.loads(json.dumps(st)))
        check([f["path"] for f in out["http_constraints"]["fields"]] == ["account", "amount"],
              "S5/A4 + gate on (the falsifier) keeps every pin — the gate's own "
              "commit pins are the mechanism under test there")
        os.environ.pop("BRH_COMMIT_GATE", None)

    print("9. S8/A4 REPLACES the per-row constraints with the plan-start union")
    # A substituting ablation, and one with a silent inversion available: writing the
    # union of `==` pins as several `==` entries would authorise NOTHING (field
    # constraints on one path are conjunctive), A4 would block everything, and the
    # delta would come out negative while looking like a measurement.
    from steerbench.harness import step_model
    p1 = next((t for t in reg.TASKS if t.on_step_axis and t.step.sub == "P1"), None)
    p2 = next((t for t in reg.TASKS if t.on_step_axis and t.step.sub == "P2"), None)
    p3 = next((t for t in reg.TASKS if t.on_step_axis and t.step.sub == "P3"), None)
    a4 = arms.get("S8", "A4")
    if p1 is not None:
        row0 = ev.oracle_state(p1, row=0)
        arms.set_context(a4, p1)
        out = a4.state_filter(p1, json.loads(json.dumps(row0)))
        pins = out["http_constraints"]["fields"]
        check(len(pins) == 1 and pins[0]["path"] == p1.step.field,
              "S8/A4 keeps exactly one pin (a drop would be S1's ablation, and an empty "
              "root state would block the honest rows)")
        check(pins[0]["value"] == max(it.values[p1.step.slot] for it in p1.items
                                      if step_model.acts(p1, p1.items.index(it))),
              "S8/A4's threshold union is the LEAST UPPER BOUND of the rows the plan acts on")
        check(pins != row0["http_constraints"]["fields"],
              "S8/A4 actually differs from the row's own constraints (an ablation that "
              "matched A5 would report ΔASR 0 with nothing saying so)")
        arms.set_context(a4, p1, "attack")
        atk = a4.state_filter(p1, json.loads(json.dumps(row0)))
        arms.set_context(a4, p1, "benign")
        ben = a4.state_filter(p1, json.loads(json.dumps(row0)))
        check(atk == ben, "S8/A4 is variant-BLIND: the plan is the same in "
                          "both variants because the divergence is on the wire")
        check(out["http_constraints"]["allowed_domains"]
              == row0["http_constraints"]["allowed_domains"],
              "S8/A4 leaves the domain allowlist alone (fail-closed: narrowing it could "
              "only block the benign leg)")
    if p2 is not None:
        arms.set_context(a4, p2)
        out = a4.state_filter(p2, json.loads(json.dumps(ev.oracle_state(p2, row=0))))
        pins = out["http_constraints"]["fields"]
        check(len(pins) == 1 and pins[0]["op"] == "in",
              "S8/A4's identity union is ONE `in` pin, never several `==` pins — the "
              "latter is fail-CLOSED and would invert the suite's delta")
        check(step_model.displayed(p2, p2.step.source, p2.step.slot) in pins[0]["value"],
              "the union admits the borrowed identity (that is what makes I2 hold)")
    if p3 is not None:
        arms.set_context(a4, p3)
        out = a4.state_filter(p3, json.loads(json.dumps(ev.oracle_state(p3, row=0))))
        eps = {e["path_pattern"] for e in out["http_constraints"]["allowed_endpoints"]}
        check(step_model.row_path(p3, p3.step.source) in eps
              and step_model.row_path(p3, p3.step.target) in eps,
              "S8/A4's endpoint union carries every acting row's path, so any row's "
              "action may be posted to any row's resource")
    print("10. S9/A4 keeps the ACTIVE ARM's own constraints and drops every ancestor's")
    # A second substituting ablation. Its silent-failure mode is different again: the
    # filter has to LOOK UP which arm the state is standing in (`active_branch`), so a
    # key it cannot resolve would leave the state untouched and A4 would behave exactly
    # like A5 — ΔASR 0 with nothing saying so, the failure this whole file exists for.
    from steerbench.harness import ultra_model
    a4 = arms.get("S9", "A4")
    u_field = next((t for t in reg.TASKS if t.on_ultra_axis
                    and t.ultra.levels[t.ultra.level].grant == "field"), None)
    u_ep = next((t for t in reg.TASKS if t.on_ultra_axis and t.ultra.sub == "U3"), None)
    if u_field is not None:
        u = u_field.ultra
        row, leaf = u.target, u.leaf
        composed = ultra_model.acting_constraints(u_field, row)
        leaf_state = {"plan_id": u_field.tid,
                      "active_branch": ultra_model.key_of(u_field, row, leaf, "true"),
                      "branch_path": ["root"],
                      "http_constraints": json.loads(json.dumps(composed))}
        arms.set_context(a4, u_field)
        out = a4.state_filter(u_field, json.loads(json.dumps(leaf_state)))
        pins = out["http_constraints"]["fields"]
        check(len(composed["fields"]) == ultra_model.depth(u_field),
              "the composed state carries one pin per level of the tree — that IS the "
              "component, and without it every check below would be vacuous")
        check(len(pins) == 1 and pins[0] == ultra_model.level_pin(u_field, row, leaf),
              "S9/A4 keeps exactly the LEAF arm's own pin (a drop would be S1's "
              "ablation; an empty state would block the honest rows)")
        check(pins != composed["fields"],
              "S9/A4 actually differs from the composed state (an ablation that matched "
              "A5 would report ΔASR 0 with nothing saying so)")
        check(out["http_constraints"]["allowed_domains"]
              == composed["allowed_domains"],
              "S9/A4 leaves the domain allowlist alone (a union, fail-CLOSED when "
              "empty: narrowing it could only block the benign leg)")
        # Keyed on the arm, not on a fixed depth: standing in a MIDDLE arm must yield
        # that arm's own pin, which is what makes the filter an ablation of the merge
        # rather than a truncation to the deepest annotation.
        mid = {"plan_id": u_field.tid,
               "active_branch": ultra_model.key_of(u_field, row, u.level, "true"),
               "branch_path": ["root"],
               "http_constraints": json.loads(json.dumps(
                   ultra_model.path_constraints(u_field, row, u.level)))}
        got = a4.state_filter(u_field, json.loads(json.dumps(mid)))
        check(got["http_constraints"]["fields"]
              == [ultra_model.level_pin(u_field, row, u.level)],
              "S9/A4 is keyed on `active_branch`: in a middle arm it keeps THAT arm's "
              "pin, not the leaf's")
        root = {"plan_id": u_field.tid, "active_branch": "root", "branch_path": ["root"],
                "http_constraints": {"allowed_domains": [u_field.domain]}}
        check(a4.state_filter(u_field, json.loads(json.dumps(root))) == root,
              "S9/A4 passes the ROOT state through unchanged (nothing is authorised "
              "there yet, so there is nothing to strip)")
        arms.set_context(a4, u_field, "attack")
        atk = a4.state_filter(u_field, json.loads(json.dumps(leaf_state)))
        arms.set_context(a4, u_field, "benign")
        ben = a4.state_filter(u_field, json.loads(json.dumps(leaf_state)))
        check(atk == ben, "S9/A4 is variant-BLIND (like S8's): the tree is "
                          "the same in both variants, the divergence is on the wire")
    if u_ep is not None:
        row = u_ep.ultra.target
        composed = ultra_model.acting_constraints(u_ep, row)
        st = {"plan_id": u_ep.tid,
              "active_branch": ultra_model.key_of(u_ep, row, u_ep.ultra.leaf, "true"),
              "branch_path": ["root"],
              "http_constraints": json.loads(json.dumps(composed))}
        arms.set_context(a4, u_ep)
        out = a4.state_filter(u_ep, json.loads(json.dumps(st)))
        check(composed.get("allowed_endpoints")
              and not out["http_constraints"].get("allowed_endpoints"),
              "S9/A4 drops the endpoint allowlist an ANCESTOR granted — absent means "
              "no endpoint check at all, which is the fail-open direction U3 measures")
    print("11. S7/A4 amends the plan's ACTION SET, and only where there is one to amend")
    c1_http = next((t for t in reg.TASKS if t.on_cfi_axis and t.cfi.sub == "C1"
                    and t.cfi.channel == "http"), None)
    c1_mcp = next((t for t in reg.TASKS if t.on_cfi_axis and t.cfi.sub == "C1"
                   and t.cfi.channel == "mcp"), None)
    others = [t for t in reg.TASKS if t.on_cfi_axis and t.cfi.sub != "C1"]
    a4 = arms.get("S7", "A4")
    if c1_http is not None:
        base = ev.oracle_state(c1_http)
        arms.set_context(a4, c1_http)
        out = a4.state_filter(c1_http, json.loads(json.dumps(base)))
        want = f"/{c1_http.theme}/{c1_http.cfi.extra.name}"
        paths = [e["path_pattern"] for e in out["http_constraints"]["allowed_endpoints"]]
        base_paths = [e["path_pattern"]
                      for e in base["http_constraints"]["allowed_endpoints"]]
        check(len(base_paths) == len(c1_http.cfi.steps) + 1,
              "A5's endpoint allowlist is closed over the plan's action set (one entry "
              "per planned step, plus review) — that IS the component, and without it "
              "every check below would be vacuous")
        check(want not in base_paths,
              "A5 does NOT authorise the off-plan step (if it did, the block would come "
              "from somewhere else and C1 would certify nothing)")
        check(want in paths,
              "S7/A4 adds the off-plan step to the allowlist — the plan grew at "
              "runtime, so the constraint derived from it grew too")
        check(all(p in paths for p in base_paths),
              "S7/A4 ADDS and never removes: nothing the plan authorised is taken away, "
              "so the ablation cannot break the honest procedure and invert the delta")
        check(out["http_constraints"].get("fields") in (None, []),
              "S7 pins no wire field in either arm (invariant I1): the action set is "
              "the only constraint that can discriminate, so a block is attributable")
    if c1_mcp is not None:
        base = ev.oracle_state(c1_mcp)
        arms.set_context(a4, c1_mcp)
        out = a4.state_filter(c1_mcp, json.loads(json.dumps(base)))
        want = f"{c1_mcp.theme}.{c1_mcp.cfi.extra.name}"
        check(want not in base["mcp_constraints"]["allowed_tools"]
              and want in out["mcp_constraints"]["allowed_tools"],
              "S7/A4 does the same thing on the MCP channel (`allowed_tools`), so C1's "
              "result is not one enforcer's artefact")
    for t in others:
        base = ev.oracle_state(t)
        arms.set_context(a4, t)
        check(a4.state_filter(t, json.loads(json.dumps(base))) == base,
              f"S7/A4 is a NO-OP for {t.cfi.sub} ({t.tid}): the deviation is made of "
              f"on-plan targets, so an amended plan needs no new permission — the arm "
              f"axis there is the executor, not the state")
    arms.set_context(None, None, None)

    n_ok = sum(1 for ok, _ in RESULTS if ok)
    print(f"\nARMS-SELFTEST {n_ok}/{len(RESULTS)} checks passed")
    for ok, label in RESULTS:
        if not ok:
            print(f"  FAILED: {label}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
