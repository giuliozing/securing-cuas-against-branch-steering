"""The free gate, as a test suite — every check asserted against the frozen contract.

`tests/expected_gate.json` is the contract: it was produced by RE-RUNNING every check
against the current tree. Each check here runs the real thing and asserts its headline
string appears verbatim. So "nothing changed" stops being a claim someone
has to re-verify by hand and becomes `pytest`.

Each check runs in its **own process**, deliberately. They bring up sites, proxies, MCP
servers and MCP proxy on real ports and they mutate `os.environ`; sharing an interpreter would
make one check's leftovers another check's input, which is precisely the cell-independence
property this isolation is about. It also means these are minutes, not seconds — hence the
`slow` marker.

    pytest -m "not slow"      # unit tests + schema: the fast loop
    pytest -m substrate       # what a colleague with no defence stack can run
    pytest                    # everything (skips `defended` if the stack is absent)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

CONTRACT = json.loads((Path(__file__).parent / "expected_gate.json").read_text())
RUNNABLE = [c for c in CONTRACT["checks"] if c.get("module")]


def _ids(checks):
    return [c["id"] for c in checks]


def _run(check) -> str:
    r = subprocess.run([sys.executable, "-m", check["module"], *check.get("args", [])],
                       capture_output=True, text=True, timeout=1800)
    want_rc = check.get("expect_exit_code", 0)
    out = r.stdout + r.stderr
    assert r.returncode == want_rc, (
        f"{check['id']}: exit {r.returncode}, expected {want_rc}\n{out[-2000:]}")
    return out


def _assert_contract(check, out: str) -> None:
    for key in ("assert", "assert_2", "assert_3"):
        want = check.get(key)
        if want:
            assert want in out, (
                f"{check['id']}: expected {want!r} in output.\n"
                f"This is the regression contract from tag {CONTRACT['baseline_tag']} — "
                f"a mismatch means a MEASURED number moved.\n{out[-2000:]}")


@pytest.mark.substrate
@pytest.mark.slow
@pytest.mark.parametrize(
    "check", [c for c in RUNNABLE if c["track"] == "substrate"],
    ids=_ids([c for c in RUNNABLE if c["track"] == "substrate"]))
def test_substrate_check(check):
    """Needs nothing but this package — the half a third party can run."""
    _assert_contract(check, _run(check))


@pytest.mark.defended
@pytest.mark.slow
@pytest.mark.parametrize(
    "check", [c for c in RUNNABLE if c["track"] == "defended"],
    ids=_ids([c for c in RUNNABLE if c["track"] == "defended"]))
def test_defended_check(check):
    """Needs the [brh] extra: COBRA + the HTTP proxy enforcer + mitmdump."""
    _assert_contract(check, _run(check))


# --- properties of the contract itself --------------------------------------


def test_contract_covers_every_gate_check():
    """A check that loses its `module` entry would silently stop running while the file
    still looked complete."""
    ids = {c["id"] for c in CONTRACT["checks"]}
    engine = {c["id"] for c in CONTRACT["checks"] if c["track"] == "engine"}
    assert len(RUNNABLE) + len(engine) == len(ids)
    assert engine == {"engine_test_brh", "engine_test_mpt", "http_proxy_enforcer"}


def test_contract_totals_are_self_consistent():
    """The totals block exists so a partially-updated contract is visible as arithmetic
    rather than as a plausible-looking table."""
    t = CONTRACT["totals"]
    oracle = sum(c["counts"]["of"] for c in CONTRACT["checks"]
                 if c["id"].startswith("oracle_"))
    harness = sum(c["counts"]["of"] for c in CONTRACT["checks"]
                  if c["id"] in ("prompt_lint", "arms_selftest", "validate_paid_paths"))
    engine = sum(c["counts"]["tests"] for c in CONTRACT["checks"]
                 if "tests" in c.get("counts", {}))
    i2 = sum(c["counts"].get("i2_of", 0) for c in CONTRACT["checks"] if "counts" in c)
    assert (oracle, harness, engine, i2) == (
        t["oracle_cells"], t["harness_checks"], t["engine_tests"],
        t["i2_negative_controls"])


def test_the_spend_guard_is_asserted_by_exit_code():
    """The one check whose contract is a refusal: S4 must decline to spend, and it must
    do so by exiting non-zero, not by printing a warning and running anyway."""
    guard = next(c for c in CONTRACT["checks"] if c["id"] == "spend_guard_S4")
    assert guard["expect_exit_code"] == 2
