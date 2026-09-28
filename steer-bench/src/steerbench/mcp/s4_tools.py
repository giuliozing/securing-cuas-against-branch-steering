"""MCP tools for suite S4 — the seam channel of the 18 compound tasks.

Thin wrapper over `s3_tools._make`: same generated-signature machinery (so FastMCP
publishes a real `inputSchema` and the arguments actually reach MCP proxy), same
DIRECT-to-site recording (so HTTP proxy is *not* in this path and a block can only
come from MCP proxy). The specs come from `tasks/s4_seam.py`.

Names are namespaced `<theme>.perform` / `<theme>.<attack_verb>` and therefore never
collide with the proxy-routed verbs `mcp/toolkit.py` already publishes for the same
themes. Both surfaces coexisting is the realistic case, not an accident: a server
that exposes more than the plan authorised is the precondition for the whole suite.
"""

from __future__ import annotations

import sys
from pathlib import Path


from steerbench.mcp import s3_tools
from steerbench.tasks import s4_seam


def build_for_task(task):
    spec = s4_seam.S4_SPECS.get(task.tid)
    if spec is None:
        return
    for t in spec.tools:
        yield (f"{task.theme}.{t.verb}",
               s3_tools._make(task.theme, t.verb, t.params, t.harmful),
               t.description)


def build_all(tasks):
    for task in tasks:
        yield from build_for_task(task)
