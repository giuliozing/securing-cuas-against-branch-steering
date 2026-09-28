"""Shared `brh_check` operator semantics, for authoring-time checks only.

Several suite models (`step_model.py`, `ultra_model.py`) need to ask, at authoring
time, "would `brh_check` admit this value under this pin?" without driving the real
enforcer. `satisfies` below is a local reimplementation of the operator semantics for
exactly that purpose — it is deliberately NOT the certification, which always runs the
real interpreter and the real hook (see each suite's `oracles/*.py`).
"""

from __future__ import annotations

import re

_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _as_float(v) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("£", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def satisfies(pin: dict, value) -> bool:
    """Would `brh_check` admit `value` under `pin`?

    Unknown operators return False rather than passing: an operator this file does not
    model must not be able to certify a task by accident."""
    op, want = pin["op"], pin["value"]
    if op == "in":
        return value in (want or [])
    if op == "subset":
        got = value if isinstance(value, (list, tuple)) else [value]
        return set(got) <= set(want or [])
    if op in ("==", "eq_struct"):
        a, b = _as_float(value), _as_float(want)
        if a is not None and b is not None:
            return a == b
        return value == want
    a, b = _as_float(value), _as_float(want)
    if a is None or b is None:
        return False
    return {"<=": a <= b, "<": a < b, ">=": a >= b, ">": a > b}.get(op, False)
