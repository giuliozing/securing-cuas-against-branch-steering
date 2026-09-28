"""Reference evaluator for the BRH constraint semantics.

This is the CaMeL-side source of truth for what an ``op``/``value`` pair
means against an observed runtime value. Consumers:

- the contract test suite (`tests/test_brh/test_contract.py`), which
  runs the shared golden vectors (`brh_contract_vectors.json` at the
  repository root) against this implementation;
- the MCP proxy, which can import `satisfies` directly for `tools/call`
  param-rule checks.

The HTTP enforcer (`cobra/http_proxy/brh_check.py`)
deliberately does **not** import this module: the only runtime contract
between the CaMeL process and the enforcer process is `branch_state.json`,
and the enforcer must stay deployable without this repository's code.
Parity between the two implementations is enforced empirically by the
shared vector file — both suites consume it, so a semantic divergence
turns into a named red test on the diverging side, not a silent
enforcement gap. When the contract evolves, change the vectors first,
then both implementations.

Semantics (all fail-closed):

- **Strict types, no coercion** — mirrors the pydantic Strict* types in
  `schema.py`: ``"42" != 42``, the state file is a security contract.
- **bool is not a number** — Python's ``bool ⊂ int`` would otherwise let
  ``True`` satisfy ``== 1``.
- int/float cross-comparison is allowed (``42 <= 42.99``).
- String equality is exact and case-sensitive.
- **Unresolved placeholders are unsatisfiable**, never wildcards — even
  on a literal string match.
- Ordering ops are numeric-only (no lexicographic comparison).
- ``in`` uses the same typed equality as ``==``; empty sets and sets
  containing a placeholder member are unsatisfiable.
- **Structural ops for non-scalar args** (the MCP value-steer / arg-add
  residuals). ``eq_struct`` pins a whole
  list/dict by recursive value equality (scalars compared with the same
  strict typing as ``==``); ``subset`` requires every element of an
  observed list to value-match a member of the planned list, so an
  attack that *appends* to a list (e.g. ``open_nodes`` + ``TrackerNode``)
  fails. Both reject any unresolved placeholder anywhere in ``value``.
- Unknown ops are unsatisfiable, not errors.
"""

from __future__ import annotations

from typing import Any

from cobra.brh.schema import PLACEHOLDERS

_PLACEHOLDER_SET = frozenset(PLACEHOLDERS)


def is_placeholder(value: Any) -> bool:
    """True if `value` is one of the reserved placeholder strings."""
    return isinstance(value, str) and value in _PLACEHOLDER_SET


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def typed_equal(expected: Any, observed: Any) -> bool:
    """Strict-typed equality: numbers with numbers, strings with strings.

    bool only equals bool (constraint values are never bool per schema,
    but observed payload values can be). ``None`` (a JSON ``null``) only
    equals ``None`` — never a wildcard, and never equal to a present value.
    """
    if expected is None or observed is None:
        return expected is None and observed is None
    if isinstance(expected, bool) or isinstance(observed, bool):
        return isinstance(expected, bool) and isinstance(observed, bool) and expected == observed
    if _is_number(expected) and _is_number(observed):
        return expected == observed
    if isinstance(expected, str) and isinstance(observed, str):
        return expected == observed
    return False


def _has_placeholder(value: Any) -> bool:
    """True if a placeholder string appears anywhere in `value` (recursively).

    A structural constraint over a list/dict is unsatisfiable while any leaf is
    still an unresolved placeholder — never a wildcard."""
    if is_placeholder(value):
        return True
    if isinstance(value, list):
        return any(_has_placeholder(v) for v in value)
    if isinstance(value, dict):
        return any(_has_placeholder(v) for v in value.values())
    return False


def struct_equal(expected: Any, observed: Any) -> bool:
    """Recursive value equality. Scalars use the same strict typing as ``==``
    (numbers cross int/float, strings exact, bool only with bool, ``None``
    only with ``None``); lists are equal element-wise in order; dicts are
    equal by identical key set and per-key structural equality — both may
    hold ``None`` (JSON ``null``) leaves, compared the same way. Everything
    else is unequal (fail-closed)."""
    if expected is None or observed is None:
        return expected is None and observed is None
    if isinstance(expected, bool) or isinstance(observed, bool):
        return isinstance(expected, bool) and isinstance(observed, bool) and expected == observed
    if _is_number(expected) and _is_number(observed):
        return expected == observed
    if isinstance(expected, str) and isinstance(observed, str):
        return expected == observed
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(
            struct_equal(e, o) for e, o in zip(expected, observed)
        )
    if isinstance(expected, dict) and isinstance(observed, dict):
        return set(expected) == set(observed) and all(
            struct_equal(expected[k], observed[k]) for k in expected
        )
    return False


def _subset(value: Any, observed: Any) -> bool:
    """Every element of `observed` (a list) value-matches a member of `value`
    (a list). An observed list that is empty trivially satisfies; an observed
    non-list never does."""
    if not isinstance(value, list) or not isinstance(observed, list):
        return False
    return all(any(struct_equal(member, o) for member in value) for o in observed)


def satisfies(op: Any, value: Any, observed: Any) -> bool:
    """True iff `observed` satisfies the constraint ``(op, value)``.

    Total function: malformed input (unknown op, wrong value shape,
    unresolved placeholder) returns False — unsatisfiable, fail-closed.
    """
    if op == "in":
        if not isinstance(value, list) or not value:
            return False
        if any(is_placeholder(member) for member in value):
            return False
        return any(typed_equal(member, observed) for member in value)
    if op == "subset":
        return False if _has_placeholder(value) else _subset(value, observed)
    if op == "eq_struct":
        return False if _has_placeholder(value) else struct_equal(value, observed)
    if op == "==":
        # An exact pin on a non-scalar IS structural equality: a "from_plan"
        # list/dict pin must mean "equal to this value", never "always false".
        # (Handled before the list-guard below, which exists only to fail-close
        # the ordering ops <=/>=, where a list/dict value is meaningless.)
        if isinstance(value, (list, dict)):
            return False if _has_placeholder(value) else struct_equal(value, observed)
        if is_placeholder(value):
            return False
        return typed_equal(value, observed)
    if isinstance(value, list):
        return False
    if is_placeholder(value):
        return False
    if op == "<=" or op == ">=":
        if not (_is_number(value) and _is_number(observed)):
            return False
        return observed <= value if op == "<=" else observed >= value
    return False
