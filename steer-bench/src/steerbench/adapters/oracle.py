

from __future__ import annotations

from steerbench.adapters.base import TaskView
from steerbench.harness import driver, mcp_driver
from steerbench.tasks import registry as reg


def _same(shown, authorised) -> bool:
    """Structural equality across a JSON round-trip.

    Not `str(a) == str(b)`: Flask's `jsonify` sorts object keys, so a *dict-valued*
    signal comes back with its keys in a different order and a string comparison called
    `18-dns-cutover`'s honest record a mismatch — one false escalation on a benign cell,
    i.e. a false positive invented by the comparison rather than by the agent. Numbers
    also survive JSON as int or float depending on the value, so they are compared
    numerically."""
    if isinstance(shown, (int, float)) and isinstance(authorised, (int, float)) \
            and not isinstance(shown, bool) and not isinstance(authorised, bool):
        return float(shown) == float(authorised)
    if isinstance(shown, dict) and isinstance(authorised, dict):
        return set(shown) == set(authorised) and all(
            _same(shown[k], authorised[k]) for k in shown)
    if isinstance(shown, (list, tuple)) and isinstance(authorised, (list, tuple)):
        return len(shown) == len(authorised) and all(
            _same(a, b) for a, b in zip(shown, authorised))
    return str(shown) == str(authorised)


class OracleAdapter:
    name = "oracle"

    def run(self, view: TaskView) -> None:
        task = reg.BY_THEME[view.theme]

        if view.posture == "underspecified":
            driver.request_clarification(
                view.theme, direct_base=view.page_url.rstrip("/"),
                reason="the committed value has no trusted source")
            return

        if view.posture == "specified" and self._display_contradicts_prompt(task, view):
            # The one defence a GUI agent actually has against perception poisoning.
            # On this posture the prompt CARRIES the authorised value
            # (`instruction_specified`), so a faithful agent compares it against what the
            # screen shows and escalates on a mismatch. Nothing about the wire is
            # inspected — the check is prompt-vs-display, which is all an agent can see.
            #
            # Without this the reference confirmed blindly and every `specified·attack`
            # cell breached, which measured the adapter rather than the environment: the
            # poison moves display and wire together, so a server-resolved Confirm
            # carries it whatever the agent believes. Comparison is the only lever.
            driver.route_to_review(view.theme, task.domain, proxy=self.proxy)
            return

        if view.channel == "mcp":
            self._run_mcp(task, view)
            return

        base = view.page_url.rstrip("/")
        # Archetype order matters: a workflow task has items AND action_fields, so the
        # per-item path must be taken first or the loop would collapse into one action.
        if task.items:
            for i in range(len(task.items)):
                driver.submit_item(view.theme, direct_base=base, proxy=self.proxy,
                                   index=i)
            return
        if task.form_fields:
            # A faithful agent types what the honest source says. `expected` is that
            # value; a real adapter has to read it off the page instead.
            driver.submit_form(view.theme, direct_base=base, proxy=self.proxy,
                               fields={f.name: f.expected for f in task.form_fields
                                       if f.expected is not None})
            return
        driver.submit_action(view.theme, direct_base=base, proxy=self.proxy)

    @staticmethod
    def _display_contradicts_prompt(task, view: TaskView) -> bool:
        """Does the screen disagree with the authorised value the prompt supplied?

        Reads the DISPLAYED channel (`GET /<theme>/record`), which is what a grounding
        agent sees, and compares the committed slots against `task.disp` — the values
        `instruction_specified()` writes into the prompt. So this is prompt-vs-display,
        never prompt-vs-wire: the wire is a server-side fact no agent can read."""
        if not task.committed_slots:
            return False
        try:
            shown = driver._get(f"{view.page_url.rstrip('/')}/{view.theme}/record",
                                proxy=None)
        except Exception:                                     # noqa: BLE001
            return False
        return any(not _same(shown.get(slot), task.disp.get(slot))
                   for slot in task.committed_slots)

    def _run_mcp(self, task, view: TaskView) -> None:
        """The MCP channel: call the tool the task authorises, with its own arguments."""
        spec = task.mcp
        if spec is None:
            return
        mcp_driver.call_tool(spec.qualified(task.theme), dict(spec.args),
                             mpt_url=view.mcp_url)

    # The proxy the guarded action is routed through. On the OPEN track there is no
    # enforcer, so it is None and the request goes straight to the destination — which
    # is what "undefended" has to mean for the number to be worth anything. The runner
    # sets it; it is an attribute rather than an argument so the adapter signature stays
    # the one every third-party adapter implements.
    proxy: str | None = None
