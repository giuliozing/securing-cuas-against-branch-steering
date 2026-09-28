"""The adapter API — the benchmark's public extension point.

An **adapter** is the system under test. It receives a `TaskView` and acts on the live
environment; the benchmark then judges what the environment recorded. Nothing else about
the agent is inspected: not its transcript, not its reasoning, not its tool calls.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable


@dataclasses.dataclass(frozen=True)
class TaskView:
    """Everything an adapter is allowed to know about the task it is running."""

    tid: str
    theme: str
    instruction: str
    channel: str                    # "gui" | "mcp" — where the work is done
    page_url: str | None = None     # live page, for a `gui` task
    mcp_url: str | None = None      # MCP endpoint, for an `mcp` task
    posture: str = ""               # spec posture, when the cell has one
    step_budget: int = 20           # a courtesy, not a limit the harness enforces
    #: The page's archetype ("decision" | "document" | "workflow" | "procedure").
    #: NOT ground truth: it says what kind of surface the agent is looking at, which the
    #: agent can see for itself in the screenshot. It is passed because an adapter that
    #: describes the affordances to its model must describe the RIGHT ones — telling a
    #: model operating a multi-step procedure that "the page offers three controls at the
    #: bottom" is a harness defect wearing the costume of a model failure.
    archetype: str = "decision"

    def __post_init__(self) -> None:
        if self.channel == "gui" and not self.page_url:
            raise ValueError(f"{self.tid}: gui task with no page_url")
        if self.channel == "mcp" and not self.mcp_url:
            raise ValueError(f"{self.tid}: mcp task with no mcp_url")


@runtime_checkable
class AgentAdapter(Protocol):
    """The whole contract: a name, and a method that acts.

    `run` returns nothing on purpose. An adapter reporting its own outcome would be
    reporting a claim; the benchmark scores facts the environment recorded. An adapter
    that raises is recorded as an error on that cell and the run continues — a crash on
    one task must not silently truncate a run.
    """

    name: str

    def run(self, view: TaskView) -> None:
        ...
