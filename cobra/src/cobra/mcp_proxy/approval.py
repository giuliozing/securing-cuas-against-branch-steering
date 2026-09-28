"""Two-phase tool approval: demand-driven P-LLM ↔ Q-LLM ↔ human loop.

Phase 1 — pre-plan approval loop (``approval_loop``):
  The P-LLM checks whether currently approved tools suffice for the task.
  If not, it emits a single missing-capability description; the Q-LLM reads
  the full server tool list (with descriptions) and proposes up to 3 candidates;
  the human picks one; it is persisted to the registry. Repeats until the
  P-LLM is satisfied (or max_rounds exceeded).

  Security properties:
  - P-LLM sees only ``{tool_name: [param_names]}`` — never descriptions.
  - Q-LLM sees descriptions only for candidate selection, never for planning.
  - Human is the final trust gate: inspects name + description + rationale.
  - Registry is pre-seeded before the MCP proxy session starts → no TOFU.

Phase 2 — post-annotation plan confirmation (``phase2_confirm_plan``):
  After BRH annotation, the human reviews planned tool usage per branch and
  may veto tools. AUTO mode is a pass-through (no interaction, for benchmarks).
"""

from __future__ import annotations

import dataclasses
import enum
import json
import re
import sys
from typing import Callable

from cobra.mcp_proxy.manifest import manifest_from_tools
from cobra.mcp_proxy.registry import (
    approve_tool,
    approved_tools_from_registry,
    load_registry,
)
from cobra.brh.schema import PlanConstraints
from cobra.brh.validator import McpManifest

# Same type as cobra.brh.annotator.LLMCall: (system_prompt, user_prompt) -> str
LLMCall = Callable[[str, str], str]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


class ApprovalMode(enum.Enum):
    AUTO = "auto"               # no human interaction, trust-on-first-use — opt-in
                                 # only, for unattended benchmarks/tests; never the
                                 # default for a production deployment
    INTERACTIVE = "interactive" # CLI human-in-the-loop — the production-safe default


@dataclasses.dataclass
class SufficiencyResult:
    sufficient: bool
    missing_capability: str | None = None  # populated when sufficient=False


@dataclasses.dataclass
class ToolCandidate:
    name: str
    description: str
    reason: str
    raw_tool: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)


# -- helpers ------------------------------------------------------------------

def _parse_json(text: str) -> dict | list:
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text.strip())


# -- P-LLM sufficiency check ---------------------------------------------------

_SUFFICIENCY_SYSTEM = """\
You are a planning assistant. You receive an agent task and the list of MCP \
tools currently approved for use (names and parameter schemas only — you never \
see tool descriptions). Decide whether the approved tools are sufficient to \
complete the task.

Output exactly one fenced ```json block:
{"sufficient": true}
or
{"sufficient": false, "missing_capability": "<one short English sentence describing the single most important missing capability>"}

Rules:
- Report only ONE missing capability per response (the most critical one).
- Do not mention specific tool names or guess at what tools might exist on the server.
- If you are unsure but can attempt the task with the available tools, say sufficient=true.
"""


def sufficiency_check(
    p_llm_call: LLMCall, task: str, manifest: McpManifest
) -> SufficiencyResult:
    """Ask the P-LLM whether ``manifest`` covers ``task``.

    P-LLM receives only tool names and param names — never descriptions.
    On parse failure, defaults to sufficient=True to avoid infinite loops."""
    tool_lines = "\n".join(
        f"  {name}: params={params}" for name, params in manifest.items()
    ) if manifest else "  (none)"
    user_prompt = f"Task: {task}\n\nApproved tools:\n{tool_lines}"
    raw = p_llm_call(_SUFFICIENCY_SYSTEM, user_prompt)
    try:
        data = _parse_json(raw)
        if not isinstance(data, dict):
            return SufficiencyResult(sufficient=True)
        if data.get("sufficient"):
            return SufficiencyResult(sufficient=True)
        cap = data.get("missing_capability") or "unspecified capability"
        return SufficiencyResult(sufficient=False, missing_capability=cap)
    except (ValueError, AttributeError, KeyError):
        return SufficiencyResult(sufficient=True)


# -- Q-LLM candidate selection -------------------------------------------------

_SELECTION_SYSTEM = """\
You are a tool selection assistant. A task planner needs a tool matching a \
described capability. You receive the full list of available MCP tools (with \
names, descriptions, and parameter schemas) and the planner's requirement.

Select the best {n} candidate tools. Output exactly one fenced ```json block \
containing a list of objects:

[
  {{"name": "...", "description": "...", "reason": "..."}},
  ...
]

Rules:
- Return at most {n} objects, ordered best-first.
- "reason" is one sentence explaining why this tool matches the requirement.
- Only include tools from the provided list; copy name and description verbatim.
- If fewer than {n} suitable tools exist, return fewer items.
"""


def select_tool_candidates(
    q_llm_call: LLMCall,
    requirement: str,
    candidate_tools: list[dict],
    n: int = 3,
) -> list[ToolCandidate]:
    """Q-LLM reads full tool list (with descriptions) and returns up to n candidates.

    Only tools in ``candidate_tools`` (not yet approved) are presented.
    On parse failure returns an empty list (no crash, human sees no candidates)."""
    if not candidate_tools:
        return []
    tool_descriptions = "\n\n".join(
        f"Tool: {t.get('name')}\n"
        f"Description: {t.get('description', '(no description)')}\n"
        f"Parameters: {list((t.get('inputSchema') or {}).get('properties', {}).keys())}"
        for t in candidate_tools
    )
    raw = q_llm_call(
        _SELECTION_SYSTEM.format(n=n),
        f"Missing capability: {requirement}\n\nAvailable tools:\n{tool_descriptions}",
    )
    try:
        items = _parse_json(raw)
        if not isinstance(items, list):
            return []
        by_name = {t.get("name"): t for t in candidate_tools}
        candidates: list[ToolCandidate] = []
        for item in items[:n]:
            name = item.get("name")
            if not name or name not in by_name:
                continue
            raw_tool = by_name[name]
            candidates.append(ToolCandidate(
                name=name,
                description=item.get("description") or raw_tool.get("description", ""),
                reason=item.get("reason", ""),
                raw_tool=raw_tool,
            ))
        return candidates
    except (ValueError, AttributeError, KeyError):
        return []


# -- human selection (CLI) -----------------------------------------------------

def human_select_tool(candidates: list[ToolCandidate]) -> ToolCandidate | None:
    """CLI: present candidates with descriptions; return human's choice or None."""
    if not candidates:
        print("  [approval] No candidates available from Q-LLM.")
        return None
    print("\n  Candidates (Q-LLM ranked, best first):")
    for i, c in enumerate(candidates, 1):
        print(f"  [{i}] {c.name}")
        print(f"      {c.description}")
        print(f"      Why: {c.reason}")
    print("  [0] None of the above — skip this capability")
    while True:
        raw = input(f"  Select [0-{len(candidates)}]: ").strip()
        if raw == "0":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 0 and {len(candidates)}.")


# -- main approval loop --------------------------------------------------------

def approval_loop(
    task: str,
    p_llm_call: LLMCall,
    q_llm_call: LLMCall,
    all_tools: list[dict],
    server_id: str,
    *,
    registry_path: str | None = None,
    mode: ApprovalMode = ApprovalMode.INTERACTIVE,
    max_rounds: int = 5,
) -> tuple[McpManifest, dict[str, str]]:
    """Run the pre-plan approval loop; return ``(manifest, server_map)``.

    INTERACTIVE mode (the default): demand-driven loop — P-LLM ↔ Q-LLM ↔ human
    until the P-LLM declares sufficiency (or ``max_rounds`` is exhausted). Only
    approved tools are returned in the manifest. This is the production-safe
    mode: no tool is trusted without an explicit human decision.

    AUTO mode: approve all tools in ``all_tools`` immediately, no LLM calls,
    no human interaction (trust-on-first-use). This is an unattended,
    fail-open mode for benchmarks and automated tests only — it must never be
    selected implicitly. Callers opt in by passing ``mode=ApprovalMode.AUTO``
    explicitly; doing so is safe only because the registry is seeded before
    MCP proxy starts, so the proxy itself (``McpProxyClient`` /
    ``python -m cobra.mcp_proxy``) can then run sealed (its default) with no
    trust-on-first-use of its own.

    In both modes the returned ``manifest`` and ``server_map`` are ready to
    pass directly to ``generate_plan_constraints(mcp_tools=manifest,
    server_map=server_map)``. Start the MCP proxy sealed (the default) after
    this call so it runs without TOFU.
    """
    if mode == ApprovalMode.AUTO:
        print(
            "[approval] WARNING: ApprovalMode.AUTO — trust-on-first-use, no human "
            "review. This mode is for unattended benchmarks/tests only and must "
            "never be used for a production deployment.",
            file=sys.stderr,
        )
        for tool in all_tools or []:
            if tool.get("name"):
                approve_tool(tool, server_id, registry_path)
        manifest = manifest_from_tools(all_tools)
        server_map = {name: server_id for name in manifest}
        return manifest, server_map

    # INTERACTIVE: seed from prior approved sessions, then iterate
    registry = load_registry(registry_path)
    approved = list(approved_tools_from_registry(all_tools, server_id, registry))
    approved_names: set[str] = {t["name"] for t in approved}
    manifest: McpManifest = manifest_from_tools(approved)

    for round_num in range(1, max_rounds + 1):
        print(f"\n[approval {round_num}/{max_rounds}] Checking sufficiency…")
        result = sufficiency_check(p_llm_call, task, manifest)
        if result.sufficient:
            print(f"  P-LLM: sufficient — {len(manifest)} tool(s) approved.")
            break

        cap = result.missing_capability
        print(f"  P-LLM: missing capability — \"{cap}\"")

        unapproved = [t for t in all_tools if t.get("name") not in approved_names]
        candidates = select_tool_candidates(q_llm_call, cap, unapproved, n=3)
        choice = human_select_tool(candidates)

        if choice is None:
            print("  Skipping capability. Proceeding with current tool set.")
            break

        approve_tool(choice.raw_tool, server_id, registry_path)
        approved.append(choice.raw_tool)
        approved_names.add(choice.name)
        manifest = manifest_from_tools(approved)
        print(f"  Approved '{choice.name}'. Manifest: {sorted(manifest)}")
    else:
        print(f"[approval] max_rounds={max_rounds} reached; proceeding.")

    server_map = {name: server_id for name in manifest}
    return manifest, server_map


# -- phase 2: post-annotation plan confirmation --------------------------------

def phase2_confirm_plan(
    constraints: PlanConstraints,
    *,
    mode: ApprovalMode = ApprovalMode.INTERACTIVE,
) -> PlanConstraints:
    """Human reviews planned tool usage per branch; may veto tools before write.

    INTERACTIVE mode (the default) shows ``allowed_tools`` per branch and
    prompts for vetoes; vetoed tools are removed from ``allowed_tools``,
    ``param_rules``, ``allowed_params``, and ``allowed_tool_servers`` before
    the file is written.

    AUTO mode is a pure pass-through (no interaction, no change to
    constraints). Unattended benchmarks/tests only — callers must opt in
    explicitly by passing ``mode=ApprovalMode.AUTO``."""
    if mode == ApprovalMode.AUTO:
        return constraints

    print("\n[phase2] Review planned tool usage per branch:")
    for key, branch in constraints.branches.items():
        mcp = branch.mcp_constraints
        if not mcp or not mcp.allowed_tools:
            continue
        sorted_tools = sorted(mcp.allowed_tools)
        print(f"\n  Branch '{key}': {branch.description or '(no description)'}")
        for i, tool in enumerate(sorted_tools, 1):
            print(f"    [{i}] {tool}")
        raw = input(
            "  Veto tools by number (comma-separated), or Enter to approve all: "
        ).strip()
        if not raw:
            continue
        try:
            veto_indices = {int(x.strip()) - 1 for x in raw.split(",")}
        except ValueError:
            print("  Invalid input — keeping all tools.")
            continue
        vetoed = {sorted_tools[i] for i in veto_indices if 0 <= i < len(sorted_tools)}
        if not vetoed:
            continue
        mcp.allowed_tools = [t for t in mcp.allowed_tools if t not in vetoed]
        mcp.param_rules = [r for r in mcp.param_rules if r.tool not in vetoed]
        for t in vetoed:
            mcp.allowed_params.pop(t, None)
            mcp.allowed_tool_servers.pop(t, None)
        print(f"  Vetoed: {sorted(vetoed)}")

    return constraints
