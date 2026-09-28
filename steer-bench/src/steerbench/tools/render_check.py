"""Render every task's GUI with Flask's test client and check its archetype contract.

LLM-free, CPU-only, no processes: the SUBSTRATE half of the gate, and the one check a
third party with no defence stack can run in full.

An archetype is a contract about affordances, so this is where the contract is enforced:

  decision   3 actions (confirm · review · clarify), a #status readout, a bespoke shell
  document   the above plus an untrusted content pane and a form to fill from it
  workflow   the above plus a worklist and per-item links
  none       no page at all — asserted as such, not skipped, because "no page" and
             "template silently failed to render" must not look identical

Affordances are found by `data-steer-action`, not by grepping for JS function names. The declarative attribute is what an agent adapter reads too, so the check and
the agent agree on what an affordance is by construction.
"""

from __future__ import annotations

import sys

from markupsafe import escape

from steerbench import config
from steerbench.site.app import app, display_number
from steerbench.tasks import registry as reg

GENERIC_SENTINEL = "synthetic environment"   # only in _base.html's default shell

# archetype -> the actions its pages must expose
REQUIRED_ACTIONS = {
    "decision": ("confirm", "review", "clarify"),
    "document": ("confirm", "review", "clarify"),
    "workflow": ("confirm", "review", "clarify"),
    # `procedure` has NO confirm affordance, and its absence is the contract rather than
    # an omission: the steps ARE the actions, so a page with one "confirm" would collapse
    # a procedure back into the single guarded action every other suite has, and the
    # control-flow question could not be asked of it.
    "procedure": ("review", "clarify"),
}


def _row_segment(html: str, index: int) -> str:
    """The markup of one worklist row: from its `data-steer-item` marker to the next
    one (or to the end). Deliberately crude — a real parser would be a dependency the
    substrate gate must not have, and the markers are a contract the templates already
    have to honour for an agent to find the rows at all."""
    marker = f'data-steer-item="{index}"'
    start = html.find(marker)
    if start < 0:
        return ""
    nxt = html.find('data-steer-item="', start + len(marker))
    return html[start:nxt if nxt > 0 else len(html)]


def _check_page(html: str, task, arch: str, variant: str, fails: list[str]) -> None:
    tag = f"{task.tid}/{variant}"

    for action in REQUIRED_ACTIONS[arch]:
        if f'data-steer-action="{action}"' not in html:
            fails.append(f"{tag}: no affordance for {action!r}")
    if 'id="status"' not in html:
        fails.append(f"{tag}: no #status readout")

    if GENERIC_SENTINEL in html:
        fails.append(f"{tag}: fell back to the GENERIC shell")

    # Every displayed value must actually be ON the page. A signal the agent cannot
    # read is a branch it cannot take, so the cell would measure the template rather
    # than the component — the shape of the `46-token-allowance` defect,
    # there caused by a bound with no Signal at all.
    for s in task.signals:
        val = task.disp.get(s.slot)
        if val is None or isinstance(val, bool) or not isinstance(val, (str, int, float)):
            continue          # lists/dicts are rendered as chips; checked structurally
        # The site's own formatter is one of the accepted forms, imported rather than
        # restated: `display_number` is what the shared shells actually print, so a
        # change to it must not be able to leave this check passing against a page it no
        # longer describes. The looser forms stay for the 50 bespoke templates, which
        # format their own values.
        forms = {str(val), str(display_number(val))}
        if isinstance(val, (int, float)):
            forms.add(f"{val:,}")          # thousands separators
            forms.add(f"{val:,.2f}")
        rendered = any(f in html for f in forms)
        if not rendered and isinstance(val, str):
            # A multi-word value may be split across markup (a name over two spans),
            # so require its longest token rather than the whole string.
            longest = max(str(val).split(), key=len, default="")
            rendered = len(longest) >= 4 and longest in html
        if not rendered:
            fails.append(f"{tag}: displayed value for {s.slot!r} "
                         f"({val!r}) is not on the page")

    if variant == "attack" and task.inject_text:
        if str(escape(task.inject_text[:24])) not in html:
            fails.append(f"{tag}: injection banner not rendered")

    if arch == "document":
        if 'data-steer-pane="content"' not in html:
            fails.append(f"{tag}: document archetype with no content pane")
        if "<form" not in html and 'data-steer-field=' not in html:
            fails.append(f"{tag}: document archetype with nothing to fill in")
        # The document's TEXT must be on the page, not merely a pane to hold it: for the
        # `document` archetype this is the entire substrate — a lure that did not render
        # is an attack the agent never saw, and the cell would report a clean pass for a
        # task that never ran, and here it would look like a defence succeeding.
        body = task.content_attack if variant == "attack" else task.content
        probe = " ".join(body.split())[:48]
        if probe and str(escape(probe)) not in " ".join(html.split()):
            fails.append(f"{tag}: the document body is not rendered ({probe!r})")
        for f in task.form_fields:
            if f'data-steer-field="{f.name}"' not in html:
                fails.append(f"{tag}: no input for declared form field {f.name!r}")

    if arch == "procedure":
        spec = getattr(task, "cfi", None)
        if 'data-steer-pane="content"' not in html:
            fails.append(f"{tag}: procedure archetype with no content pane")
        flat = " ".join(html.split())
        body = task.content_attack if variant == "attack" else task.content
        probe = " ".join(body.split())[:48]
        if probe and str(escape(probe)) not in flat:
            fails.append(f"{tag}: the case notes are not rendered ({probe!r})")
        if variant == "attack" and spec is not None:
            lure = " ".join(spec.lure.split())[:48]
            if str(escape(lure)) not in flat:
                fails.append(f"{tag}: the injected sentence is not on the page ({lure!r})")
        # Every affordance the scenario declares must be individually clickable — the
        # OFF-PLAN one included, and in BOTH variants. If the extra step appeared only
        # under attack, an agent could separate the cells without reading a word of the
        # document and the suite would measure a diff instead of an injection; if it were
        # missing entirely, C1's deviation would be unreachable and its cells would report
        # a clean pass for an attack that was never offered.
        for st in (spec.affordances() if spec else ()):
            if f'data-steer-step="{st.name}"' not in html:
                fails.append(f"{tag}: no affordance for step {st.name!r}")
            if str(escape(st.label)) not in flat:
                fails.append(f"{tag}: step {st.name!r} has no label on the page")

    if arch == "workflow":
        if 'data-steer-item=' not in html:
            fails.append(f"{tag}: workflow archetype with no worklist items")
        # Each row must be individually visible and individually actionable. A worklist
        # rendered as prose, or with one button for the whole queue, would make the plan
        # a single action again — which is the exact shape suite S8 exists to leave
        # behind, so a template regression there would quietly un-measure the suite.
        flat = " ".join(html.split())
        for i, it in enumerate(task.items):
            if str(escape(it.ref)) not in flat:
                fails.append(f"{tag}: row {i} ({it.ref!r}) is not on the page")
            if f'data-steer-item-action="{i}"' not in html:
                fails.append(f"{tag}: row {i} has no action affordance")
            for slot, val in it.values.items():
                if not isinstance(val, (str, int, float)) or isinstance(val, bool):
                    continue
                forms = ({str(val), f"{val:,}", str(display_number(val))}
                         if isinstance(val, (int, float)) else {str(escape(val))})
                if not any(f in flat for f in forms):
                    fails.append(f"{tag}: row {i}'s displayed {slot!r} ({val!r}) is not "
                                 f"on the page")
            # And the premise itself: what the row's WIRE will carry must never be
            # visible ON THAT ROW. The divergence is server-side (resolved at
            # `/__item/<i>/quote`), so a template that leaked `real` would hand the agent
            # the answer and the cell would measure the page instead of the enforcer.
            #
            # Scoped to the row, and that is not a detail: in S8 the attacked value is
            # *another row's* displayed value, so it is legitimately on the page. A
            # page-wide search would flag every P2 task and would be asking the wrong
            # question — the leak is the wrong value appearing where THIS row is shown.
            if variant == "attack" and it.real:
                seg = _row_segment(html, i)
                for slot, val in it.real.items():
                    if val == it.values.get(slot):
                        continue
                    probe = str(escape(val)) if isinstance(val, str) else str(val)
                    if probe and probe in seg:
                        fails.append(f"{tag}: row {i}'s REAL {slot!r} is rendered on that "
                                     f"row — the divergence must stay server-side")


def main() -> int:
    client = app.test_client()
    fails: list[str] = []
    seen = {a: 0 for a in reg.ARCHETYPES}

    for t in reg.TASKS:
        arch = reg.archetype_of(t)
        seen[arch] += 1

        if arch == "none":
            # Asserted, not skipped: the claim is "this task has no GUI surface", and
            # the way that fails silently is a template appearing where none should be.
            page = config.PKG / "site" / "templates" / t.template / "page.html"
            if t.template != "generic" or page.exists():
                fails.append(f"{t.tid}: archetype 'none' but a bespoke template exists")
            continue

        # The same rule the renderer uses (`site/app.page`): a bespoke page if the
        # theme ships one, otherwise the archetype's shared shell. Demanding a bespoke
        # page here would contradict the thing being checked — a task that renders
        # correctly from the shell would be reported as broken, which is worse than a
        # missing file because it argues against the working code.
        page = config.PKG / "site" / "templates" / t.template / "page.html"
        shell = config.PKG / "site" / "templates" / "archetypes" / arch / "page.html"
        if not page.exists() and not shell.exists():
            fails.append(f"{t.tid}: archetype {arch!r} has neither {t.template}/page.html "
                         f"nor a shell at archetypes/{arch}/page.html")
            continue

        for variant in ("benign", "attack"):
            cfg = t.benign_config() if variant == "benign" else t.attack_config()
            client.post("/__config", json=cfg)
            r = client.get(f"/{t.theme}/")
            if r.status_code != 200:
                fails.append(f"{t.tid}/{variant}: HTTP {r.status_code}")
                continue
            _check_page(r.get_data(as_text=True), t, arch, variant, fails)

    inventory = " · ".join(f"{a} {n}" for a, n in seen.items() if n)
    if fails:
        print(f"RENDER-CHECK FAIL — {len(fails)} problem(s) across {len(reg.TASKS)} "
              f"tasks ({inventory}):")
        for f in fails:
            print(f"  x {f}")
        return 1
    print(f"RENDER-CHECK OK — {len(reg.TASKS)}/{len(reg.TASKS)} tasks satisfy their "
          f"archetype contract ({inventory}), benign+attack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
