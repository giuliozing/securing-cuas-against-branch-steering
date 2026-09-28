"""Tests for the BRH commit gate (`cobra.brh.hook.on_tool_call`).

The gate answers "may this value be COMMITTED?", which the branch hook
structurally cannot: the branch hook sees conditions, not the arguments of the
guarded call. See the block comment above `on_tool_call`.

Like `test_hook.py` these run interpreter-free: `is_trusted` is imported lazily
inside the gate, so a fake value exposing `metadata` + `get_dependencies()` is
enough to exercise the real predicate from `cobra.capabilities.utils` when it is
importable, and a monkeypatched one otherwise. The fakes mirror exactly what the
interpreter hands over — CaMeL *values*, not raws, because the raw path throws
away the capability metadata that is the entire signal.
"""

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

from cobra.capabilities import Capabilities, sources
from cobra.brh import hook as brh_hook
from cobra.brh.hook import BRHBranchStep, BRHRuntime, on_tool_call
from cobra.brh.schema import BranchConstraints, HttpConstraints, PlanConstraints


class FakeValue:
    """Minimal `CaMeLValue` surface used by `capabilities.utils.is_trusted`."""

    def __init__(self, raw, metadata: Capabilities, deps=()):
        self.raw = raw
        self._metadata = metadata
        self._deps = tuple(deps)

    @property
    def metadata(self):
        return self._metadata

    def get_dependencies(self, visited_objects=frozenset()):
        return self._deps, visited_objects


class FakeDict(FakeValue):
    """A CaMeL dict: `_python_value` maps CaMeL keys to CaMeL values."""

    def __init__(self, mapping: dict, metadata: Capabilities | None = None):
        self._python_value = {FakeValue(k, _trusted()): v for k, v in mapping.items()}
        super().__init__({k: v.raw for k, v in mapping.items()},
                         metadata or _trusted(), tuple(mapping.values()))


def _trusted() -> Capabilities:
    return Capabilities(frozenset({sources.SourceEnum.User}), None)


def _tool_sourced() -> Capabilities:
    """What `value.wrap_output` attaches to any tool result: `Tool(name)` with no
    inner sources, which `is_trusted` rejects."""
    return Capabilities(frozenset({sources.Tool("read_record_text")}), None)


@dataclasses.dataclass(frozen=True)
class FakeEvalArgs:
    brh_runtime: object = None
    brh_branch_path: tuple = ()


GATE = json.dumps({
    "submit_action": {"container": "committed", "require": ["account"],
                      "pins": {"account": "account"}}
})


class CommitGateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.state_path = self.dir / "branch_state.json"
        self._env = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self._env)))
        os.environ["BRH_COMMIT_GATE"] = GATE
        brh_hook._GATE_CACHE = None

    def runtime(self, branches=None):
        constraints = PlanConstraints(
            plan_id="plan_gate", task="pay the invoice",
            branches=branches or {"root": BranchConstraints(
                description="root",
                http_constraints=HttpConstraints(allowed_domains=["bank.local"]))},
        )
        return BRHRuntime(constraints=constraints, state_path=self.state_path)

    def alerts(self):
        path = self.dir / "brh_alerts.jsonl"
        if not path.exists():
            return []
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    def state(self):
        return json.loads(self.state_path.read_text())


class TestGateOff(CommitGateTestCase):
    def test_unset_env_is_a_noop(self):
        os.environ.pop("BRH_COMMIT_GATE")
        brh_hook._GATE_CACHE = None
        args = {"committed": FakeDict({"account": FakeValue("X", _tool_sourced())})}
        self.assertIsNone(on_tool_call("submit_action", args, FakeEvalArgs(self.runtime())))
        self.assertEqual(self.alerts(), [])

    def test_tool_not_under_the_gate_is_untouched(self):
        args = {"reason": FakeValue("why", _tool_sourced())}
        self.assertIsNone(on_tool_call("route_to_review", args, FakeEvalArgs(self.runtime())))

    def test_malformed_config_disables_the_gate(self):
        os.environ["BRH_COMMIT_GATE"] = "{not json"
        brh_hook._GATE_CACHE = None
        args = {"committed": FakeDict({"account": FakeValue("X", _tool_sourced())})}
        self.assertIsNone(on_tool_call("submit_action", args, FakeEvalArgs(self.runtime())))


class TestDenial(CommitGateTestCase):
    def test_untrusted_commitment_is_denied_and_alerted(self):
        args = {"committed": FakeDict({"account": FakeValue("GB00 EVIL", _tool_sourced())})}
        rt = self.runtime()
        denial = on_tool_call("submit_action", args, FakeEvalArgs(rt))
        self.assertIsNotNone(denial)
        self.assertIn("untrusted", denial)
        alerts = self.alerts()
        self.assertEqual([a["reason"] for a in alerts], ["brh_provenance"])
        self.assertEqual(alerts[0]["channel"], "tool")
        self.assertEqual(alerts[0]["detail"]["untrusted"], ["account"])

    def test_missing_commitment_is_denied(self):
        """The empty declaration is the obvious bypass: a plan that declares
        nothing would otherwise submit with a clean conscience while the wire body
        (resolved server-side) still carries the poisoned value."""
        args = {"committed": FakeDict({})}
        denial = on_tool_call("submit_action", args, FakeEvalArgs(self.runtime()))
        self.assertIn("must declare ['account']", denial)

    def test_untrusted_dependency_taints_a_literal_key(self):
        """Provenance is transitive: a value derived from a tool read is untrusted
        even if its own metadata says otherwise (`get_all_sources` unions deps)."""
        tainted = FakeValue("GB00", _trusted(),
                            deps=(FakeValue("raw", _tool_sourced()),))
        args = {"committed": FakeDict({"account": tainted})}
        self.assertIsNotNone(on_tool_call("submit_action", args, FakeEvalArgs(self.runtime())))


class TestAllowAndPin(CommitGateTestCase):
    def test_trusted_commitment_passes_and_pins_the_wire_field(self):
        args = {"committed": FakeDict({"account": FakeValue("DE89 3704", _trusted())})}
        rt = self.runtime()
        self.assertIsNone(on_tool_call("submit_action", args, FakeEvalArgs(rt)))
        # No DENIAL — and the acceptance is audited on its own channel. The two must
        # stay apart: every consumer reads channel `tool` as "the gate fired", so an
        # accepted call landing there would be scored GATED. The audit record is what
        # makes "the gate was never consulted" distinguishable from "the gate said
        # yes" in a result file.
        self.assertEqual([a for a in self.alerts() if a["channel"] == "tool"], [])
        self.assertEqual([(a["kind"], a["reason"]) for a in self.alerts()],
                         [("brh_tool_audit", "brh_commit_ok")])
        fields = self.state()["http_constraints"]["fields"]
        self.assertIn({"path": "account", "op": "==", "value": "DE89 3704"}, fields)

    def test_container_values_are_pinned_with_eq_struct(self):
        """`satisfies` refuses `==` when the pinned value is a container and routes
        structural equality through `eq_struct`, so emitting `==` for a list would
        be unsatisfiable — it would block the HONEST wire. Caught live on the two
        structural-op spec tasks (16 SANs list, 18 DNS record object)."""
        os.environ["BRH_COMMIT_GATE"] = json.dumps({
            "submit_action": {"container": "committed", "require": ["sans"],
                              "pins": {"sans": "sans"}}})
        brh_hook._GATE_CACHE = None
        args = {"committed": FakeDict(
            {"sans": FakeValue(["a.acme.local", "b.acme.local"], _trusted())})}
        self.assertIsNone(on_tool_call("submit_action", args, FakeEvalArgs(self.runtime())))
        self.assertIn({"path": "sans", "op": "eq_struct",
                       "value": ["a.acme.local", "b.acme.local"]},
                      self.state()["http_constraints"]["fields"])

    def test_positional_container_is_read_too(self):
        args = {"0": FakeDict({"account": FakeValue("DE89 3704", _trusted())})}
        self.assertIsNone(on_tool_call("submit_action", args, FakeEvalArgs(self.runtime())))

    def test_pin_survives_a_later_branch_transition(self):
        rt = self.runtime(branches={
            "root": BranchConstraints(description="root",
                                      http_constraints=HttpConstraints(
                                          allowed_domains=["bank.local"])),
            "if_L3_true": BranchConstraints(description="act",
                                            http_constraints=HttpConstraints(
                                                allowed_domains=["bank.local"])),
        })
        args = {"committed": FakeDict({"account": FakeValue("DE89", _trusted())})}
        self.assertIsNone(on_tool_call("submit_action", args, FakeEvalArgs(rt)))
        # A subsequent transition rewrites the state from the constraints; the pin
        # the plan already earned must not disappear with it.
        brh_hook._write_scope_state(rt, steps=(BRHBranchStep("if_L3_true", None),))
        fields = self.state()["http_constraints"]["fields"]
        self.assertIn({"path": "account", "op": "==", "value": "DE89"}, fields)

    def test_unknown_scope_patches_the_state_in_place(self):
        """A harness that wrote `branch_state.json` by hand (every STEER oracle
        does) has no matching branch key. Recomputing the scope would write the
        FAIL-CLOSED state and block the benign leg, so the file is patched
        instead."""
        rt = self.runtime()
        self.state_path.write_text(json.dumps({
            "plan_id": "hand", "active_branch": "act",
            "http_constraints": {"allowed_domains": ["bank.local"], "fields": []},
        }))
        args = {"committed": FakeDict({"account": FakeValue("DE89", _trusted())})}
        self.assertIsNone(on_tool_call(
            "submit_action", args, FakeEvalArgs(rt, (BRHBranchStep("if_L9_true", None),))))
        state = self.state()
        self.assertEqual(state["active_branch"], "act")
        self.assertEqual(state["http_constraints"]["allowed_domains"], ["bank.local"])
        self.assertIn({"path": "account", "op": "==", "value": "DE89"},
                      state["http_constraints"]["fields"])


class TestFailClosed(CommitGateTestCase):
    def test_internal_error_denies_for_a_gated_tool(self):
        """Fail-safe (never raises into the interpreter) but NOT fail-open: the
        operator explicitly asked for this tool to be gated."""
        class Exploding:
            @property
            def _python_value(self):
                raise RuntimeError("boom")

        denial = on_tool_call("submit_action", {"committed": Exploding()},
                              FakeEvalArgs(self.runtime()))
        self.assertIsNotNone(denial)

    def test_no_runtime_still_denies_an_untrusted_commitment(self):
        """The gate is a decision, not a side-effect: with no BRHRuntime attached
        there is nowhere to write the alert, but the call must still be refused."""
        args = {"committed": FakeDict({"account": FakeValue("X", _tool_sourced())})}
        self.assertIsNotNone(on_tool_call("submit_action", args, FakeEvalArgs(None)))


if __name__ == "__main__":
    unittest.main()
