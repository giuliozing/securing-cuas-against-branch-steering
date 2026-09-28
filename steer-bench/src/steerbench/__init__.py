"""STEER-Bench — branch-steering and compound prompt injection on computer-use agents.

A task is a **real served website** that also exposes MCP tools, with the guarded action
crossing an HTTP wire. The attack premise is that *perceived ≠ real* is a server-side
fact: the page shows one value, the confirmed action carries another, resolved by the
site rather than supplied by the agent — so no amount of looking at the screen reveals
it (see `docs/DATASHEET.md`).

Two ways to use it:

* the **open track** (`steerbench open`) — any agent, judged from the environment's own
  record of what happened (`GET /__state`). Needs nothing but this package.
* the **defended track** (`steerbench run`) — a system with an enforcer, judged
  additionally on *why* an action was refused. Needs the `[brh]` extra.

Nothing in this package imports a defence. `steerbench.adapters.brh` is the only module
that does, and it is optional by construction.
"""

__version__ = "3.0.0.dev0"
