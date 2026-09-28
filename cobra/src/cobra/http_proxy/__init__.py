"""HTTP proxy: branch-aware, stateful network-layer enforcement.

Two enforcement layers with different semantics, kept separate:

- `brh_check` — dynamic, plan-derived constraints read from
  `branch_state.json` (written by the interpreter's BRH hook on every
  branch transition). Stdlib-only, no dependency on the interpreter package.
- `policy` — a static declarative rule engine (tags, endpoint patterns),
  independent of the active plan.

`mitm_addon.py` chains them per request inside mitmproxy: infra passlist
→ BRH check → sitemap/policy.
"""

from cobra.http_proxy.brh_check import (
    Decision,
    BRHState,
    RequestView,
    check,
    read_state,
    satisfies,
)

__all__ = [
    "Decision",
    "BRHState",
    "RequestView",
    "check",
    "read_state",
    "satisfies",
]
