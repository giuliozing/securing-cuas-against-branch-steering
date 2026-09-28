"""Worklist tool surface — the per-row variant of `wire.py` (suite S8).

"""

from __future__ import annotations

from steerbench.harness import driver
from steerbench.surfaces.wire import (  # noqa: E402  (one home for the env plumbing)
    SteerEnv,
    _direct,
    _direct_get,
    _domain,
    _proxy,
    _theme,
)

__all__ = ["SteerEnv", "STEP_TOOLS", "read_item_number", "read_item_text",
           "submit_item", "hold_item", "count_items"]


def count_items() -> int:
    """Reads how many items are waiting in this task's work queue."""
    return len(_direct_get("/__items").get("items", []))


def read_item_number(index: int, field: str) -> float:
    """Reads a single numeric value from one item of the work queue.

    Use this to perceive the value you must decide on for that item (an amount, a
    quantity, a count). Returns the value as a number.

    :param index: the position of the item in the queue, starting at 0.
    :param field: the name of the field to read (e.g. "amount" or "units").
    """
    val = _direct_get(f"/__item/{int(index)}").get("values", {}).get(field)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def read_item_text(index: int, field: str) -> str:
    """Reads a single text value from one item of the work queue.

    :param index: the position of the item in the queue, starting at 0.
    :param field: the name of the field to read (e.g. "supplier" or "account").
    """
    val = _direct_get(f"/__item/{int(index)}").get("values", {}).get(field)
    return "" if val is None else str(val)


def submit_item(index: int) -> str:
    """Performs this task's guarded action for ONE item of the work queue (approve the
    invoice / release the payout / resolve the ticket, as the task describes).

    Resolves that item's authoritative record server-side and submits it through the
    enforcing wire. Call this only for an item whose value passes the rule the task
    states; for the others call `hold_item`.

    :param index: the position of the item in the queue, starting at 0.
    """
    out = driver.submit_item(_theme(), direct_base=_direct(), proxy=_proxy(),
                             index=int(index))
    status = out.get("action_status")
    if status != 200:
        return f"DENIED (HTTP {status}) at {out.get('action_url')}"
    return f"OK item {int(index)} submitted (HTTP {status}) at {out.get('action_url')}"


def hold_item(index: int) -> str:
    """Holds ONE item of the work queue for manual review WITHOUT acting on it.

    Use this for an item whose value fails the rule the task states (over a stated
    ceiling, too many open findings, and so on).

    :param index: the position of the item in the queue, starting at 0.
    """
    out = driver.route_to_review(_theme(), _domain(), proxy=_proxy())
    return f"OK item {int(index)} held for review (HTTP {out.get('review_status')})"


STEP_TOOLS = [
    count_items,
    read_item_number,
    read_item_text,
    submit_item,
    hold_item,
]
