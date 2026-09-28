"""Shared fixtures and the substrate/defended split.

The split is not cosmetic: it is what makes the benchmark publishable. Anything marked
`defended` needs the system under test (COBRA + the HTTP proxy enforcer + `mitmdump`),
which is the separate `cobra/` package plus the optional `[brh]` extra. Everything else must pass on a bare `pip install -e .[dev]`, because that
is what a colleague cloning the repo will have.

So a missing defence stack **skips** rather than fails. A test that reported red on a
machine that was never meant to run it would train everyone to ignore red.
"""

from __future__ import annotations

import shutil

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "defended: needs the [brh] extra (COBRA + HTTP proxy + mitmdump)")
    config.addinivalue_line(
        "markers", "slow: brings up real processes; minutes, not seconds")
    config.addinivalue_line(
        "markers", "substrate: needs nothing but this package")


def _defence_available() -> tuple[bool, str]:
    from steerbench import config as cfg
    try:
        cfg.cobra_src()
    except cfg.MissingDefenceStack as e:
        return False, str(e).splitlines()[0]
    try:
        cfg.enforcer_addon()
    except cfg.MissingDefenceStack as e:
        return False, str(e).splitlines()[0]
    try:
        cfg.mitmdump()
    except cfg.MissingDefenceStack as e:
        return False, str(e).splitlines()[0]
    try:
        import cobra.brh.writer  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"cobra not importable: {type(e).__name__}"
    return True, ""


def pytest_collection_modifyitems(config, items):
    ok, why = _defence_available()
    if ok:
        return
    skip = pytest.mark.skip(reason=f"defended track unavailable — {why}")
    for item in items:
        if "defended" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def expected_gate() -> dict:
    """The regression contract (`expected_gate.json`), captured by re-running every
    check."""
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent / "expected_gate.json").read_text())


@pytest.fixture(scope="session")
def tasks():
    from steerbench.tasks import registry as reg
    return reg.TASKS


@pytest.fixture(scope="session")
def has_playwright() -> bool:
    return shutil.which("playwright") is not None
