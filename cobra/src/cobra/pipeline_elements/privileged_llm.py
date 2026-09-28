# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing the implementation for the PrivilegedLLM."""

import dataclasses
import os
import time
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar
from pathlib import Path
import re
import openai
import anthropic
from google import genai

import pydantic
import yaml
from agentdojo import agent_pipeline, functions_runtime
#from agentdojo import types as ad_types
from cobra.interaction.vlms import extended_types as ad_types
from agentdojo.agent_pipeline import tool_execution
from agentdojo.default_suites.v1.banking.task_suite import BankingEnvironment
from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment
from agentdojo.default_suites.v1.workspace.task_suite import (
    WorkspaceEnvironment,
)
from agentdojo.default_suites.v1.slack.task_suite import SlackEnvironment
from pydantic_ai.models import KnownModelName

from cobra import quarantined_llm, system_prompt_generator
from cobra.capabilities import is_trusted
from cobra.interpreter import interpreter, result
from cobra.interpreter import namespace as ns
from cobra.interpreter.value import CaMeLValue
from cobra.pipeline_elements.agentdojo_function import (
    make_agentdojo_namespace,
)
from cobra.pipeline_elements.security_policies import (
    AgentDojoSecurityPolicyEngine,
)
from cobra.interaction.environments.base_ui_uitars import (
    ActionCall,
    FindResult,
    DoneResponse,
    Instruction,
    CallModel,
    PromptInjectionCall,
)
from cobra.interaction.environments.osworld.base_ui_task_suite_uitars import (
    UIEnvironment
)
from cobra.interaction.environments.osworld.base_ui_task_suite_opencua import (
    UIEnvironment_OpenCUA
)
from cobra.interaction.environments.osworld.base_ui_task_suite_anthropic import (
    UIEnvironment_Anthropic
)
UIEnv = UIEnvironment|UIEnvironment_OpenCUA|UIEnvironment_Anthropic
#from cobra.interaction.environments.base_ui import check_done

# Now yaml.safe_dump will work with datetime objects
# But you'd need to use yaml.dump instead of yaml.safe_dump
def custom_yaml_dump(obj: dict | list) -> str:
    return yaml.dump(obj, default_flow_style=False)


_T = TypeVar("_T", bound=str | int | float | pydantic.BaseModel)
_E = TypeVar("_E", bound=functions_runtime.TaskEnvironment)

_ANNOTATION_LINE_RE = re.compile(r'^\s*#\s*>>>\s.*$', re.MULTILINE)

def _strip_old_annotations_and_make_mapper(src: str):
    """
    Remove prior '# >>> ...' annotation lines from `src` and return:
      - the cleaned source
      - a function map_lineno(orig_lineno) -> new_lineno after removals
    """
    lines = src.splitlines()
    cleaned = []
    removed_before = [0] * (len(lines) + 2)  # prefix counts of removed lines
    removed = 0
    for i, line in enumerate(lines, start=1):
        if _ANNOTATION_LINE_RE.match(line):
            removed += 1
        else:
            cleaned.append(line)
        removed_before[i + 1] = removed

    def map_lineno(n: int) -> int:
        if n <= 0:
            return n
        n = min(n, len(lines))
        return n - removed_before[n]

    return "\n".join(cleaned), map_lineno
def extract_print_output(tool_calls: Sequence[interpreter.FunctionCall]) -> str:
    """Extracts and concatenates arguments from print calls."""
    parts: list[str] = []
    for tc in tool_calls:
        if tc.function == "print" and not isinstance(tc.output, Exception):
            parts.append(str(tc.output))
    return "".join(parts)


def function_call_from_ad_function_call(function_call: interpreter.FunctionCall) -> functions_runtime.FunctionCall:
    args = {k: (v if not isinstance(v, type) else repr(v)) for k, v in function_call.args.items()}
    return functions_runtime.FunctionCall(function=function_call.function, args=args)


def make_ad_tool_calls(
    tool_calls: Sequence[interpreter.FunctionCall],
) -> list[tuple[functions_runtime.FunctionCall, Any]]:
    return [
        (
            function_call_from_ad_function_call(tool_call),
            tool_call.output,
        )
        for tool_call in tool_calls
        if not isinstance(tool_call.output, Exception) and not tool_call.is_builtin
    ]


def format_camel_exception(camel_exception: interpreter.CaMeLException, code: str) -> str:
    exception = camel_exception.exception
    node = camel_exception.nodes[-1]
    try:
        extracted_code = interpreter.extract_code_block(code)
    except interpreter.InvalidOutputError:
        extracted_code = code
    formatted_code = _highlight_exception_code(
        extracted_code,
        camel_exception.exception,
        node.lineno,
        node.col_offset if hasattr(node, "col_offset") else 0,
        node.end_lineno if hasattr(node, "end_lineno") else None,
        node.end_col_offset if hasattr(node, "end_col_offset") else None,
    )
    if is_trusted(camel_exception):
        exception_text = str(exception)
    else:
        exception_text = "<The exception was redacted because it came from an untrusted source. Try to infer what the problem was from the context provided.>"
    return f"""
Traceback (most recent call last):
File "<stdin>", line {camel_exception.nodes[-1].lineno}, in <module>
{formatted_code}

{type(exception).__name__}: {exception_text}
"""


def make_error_messages(code: str, interpretation_error: interpreter.CaMeLException) -> list[ad_types.ChatMessage]:
    return [
        ad_types.ChatAssistantMessage(
            role="assistant",
            content=[ad_types.text_content_block_from_string(code)],
            tool_calls=None,
        ),
        ad_types.ChatUserMessage(
            role="user",
            content=[
                ad_types.text_content_block_from_string(f"""\
Running the code gave the following error:
{format_camel_exception(interpretation_error, code)}
Provide the new code with the error fixed. Provide *all the code* so that I \
can directly run it. If the error comes from a search query that did not \
return any results, then try the query with different parameters. The code \
up to the line before the one where the exception was thrown has already been \
executed and the variables and defined classes will \
still be accessible to you. It's very important that you do not re-write code to run \
functions that have side-effects (e.g., functions that send an email). The environment is reset, \
so make sure to provide a full revised version of the code from the beginning of the task.
""")
            ],
        ),
    ]

def format_tool_calls_results(tool_calls_results):
        """Format tool calls results for better readability"""
        output = []
        output.append("\n" + "="*80)
        output.append("TOOL CALLS RESULTS")
        output.append("="*80 + "\n")
        
        for idx, (function_call, result) in enumerate(tool_calls_results, 1):
            output.append(f"[{idx}] Function: {function_call.function}")
            output.append(f"    Line: {function_call.args.get('__lineno', 'N/A')}")
            
            # Format arguments (excluding internal ones)
            args = {k: v for k, v in function_call.args.items() 
                    if not k.startswith('__')}
            if args:
                output.append(f"    Args:")
                for key, val in args.items():
                    if isinstance(val, list) and val and isinstance(val[0], tuple):
                        output.append(f"      {key}:")
                        for k, v in val:
                            output.append(f"        {k}: {v}")
                    else:
                        # Truncate long values
                        val_str = str(val)
                        if len(val_str) > 100:
                            val_str = val_str[:100] + "..."
                        output.append(f"      {key}: {val_str}")
            
            # Handle None results
            if result is None:
                output.append(f"    Result: None")
                output.append("")
                continue
                
            # Format result based on type
            output.append(f"    Result Type: {type(result).__name__}")
            output.append(f"    Result:")
            
            # Handle different result types
            if isinstance(result, ActionCall):
                output.append(f"      Status: {result.status}")
                if result.str_messages:
                    output.append(f"      Messages: {result.str_messages}")
                    
            elif isinstance(result, FindResult):
                output.append(f"      Status: {result.result.status}")
                if result.result.str_messages:
                    output.append(f"      Messages: {result.result.str_messages}")
                if result.start:
                    output.append(f"      Start: ({result.start.x1}, {result.start.y1})")
                if result.end:
                    output.append(f"      End: ({result.end.x1}, {result.end.y1})")
                if result.direction:
                    output.append(f"      Direction: {result.direction}")
                    
            elif isinstance(result, DoneResponse):
                output.append(f"      Done: {result.done}")
                
            elif isinstance(result, Instruction):
                text_preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
                output.append(f"      Text: {text_preview}")
                output.append(f"      Length: {result.length}")
                
            elif isinstance(result, CallModel):
                output.append(f"      Name: {result.name}")
                args_str = str(result.args)
                if len(args_str) > 100:
                    args_str = args_str[:100] + "..."
                output.append(f"      Args: {args_str}")
                
            elif isinstance(result, PromptInjectionCall):
                output.append(f"      Detected: {result.detected}")
                if result.str_messages:
                    output.append(f"      Messages: {result.str_messages}")
                    
            else:
                # Fallback for unknown types
                result_str = str(result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                output.append(f"      {result_str}")
            
            output.append("")  # Blank line between calls
        
        return "\n".join(output)

def _highlight_exception_code(
    code: str,
    exception: Exception,
    lineno: int,
    col_offset: int,
    end_lineno: int | None,
    end_col_offset: int | None,
) -> str:
    """Highlights the affected code in a multi-line Python code string.

    Args:
        code: The multi-line Python code.
        exception: The exception that occurred.
        lineno: The line number where the exception occurred (1-based).
        col_offset: The column offset where the exception occurred (0-based).
        end_lineno: The end line number of the affected code (1-based), or None if it's a single line.
        end_col_offset: The end column offset of the affected code (0-based), or None if it's a single line.

    Returns:
        A string with the affected code highlighted, or an error message if there's an issue.
    """

    lines = code.splitlines(keepends=True)

    if not 1 <= lineno <= len(lines):
        return f"Error: lineno {lineno} is out of range (1-{len(lines)})."

    highlighted_code = ""

    if end_lineno is None or end_lineno == lineno:  # Single line error
        line = lines[lineno - 1]
        highlighted_code += line
        highlighted_code += (
            " " * col_offset + "^" * (1 if end_col_offset is None else max(1, end_col_offset - col_offset)) + "\n"
        )
    elif 1 <= end_lineno <= len(lines) and end_lineno > lineno:  # Multiline error
        for i in range(lineno - 1, end_lineno):
            current_line = lines[i]
            highlighted_code += current_line
            if i == lineno - 1:
                highlighted_code += " " * col_offset + "^" * (len(current_line) - 1 - col_offset) + "\n"
            elif i == end_lineno - 1:
                highlighted_code += (
                    "^" * (len(current_line) - 1 if end_col_offset is None else end_col_offset) + "\n"
                )  # Fixed here
            else:
                highlighted_code += "^" * (len(current_line) - 1) + "\n"

    else:
        return f"Error: end_lineno {end_lineno} is out of range or not after lineno."

    return highlighted_code

def collect_print_trace(tool_calls: Sequence[interpreter.FunctionCall]) -> list[tuple[int, str]]:
    """Return [(lineno, text), ...] for each print in execution order."""
    out: list[tuple[int, str]] = []
    for tc in tool_calls:
        if tc.function == "print" and not isinstance(tc.output, Exception):
            ln = tc.args.get("__lineno")
            if isinstance(ln, int):
                out.append((ln, str(tc.output)))
    return out

def inline_annotate(src: str, trace: list[tuple[int, str]]) -> str:
    lines = src.splitlines()
    inserts = 0
    for lineno, text in trace:
        i = max(0, min(len(lines) - 1, lineno - 1)) + inserts
        lines.insert(i + 1, f"# >>> {text}")
        inserts += 1
    return "\n".join(lines)

def _get_quarantined_llm(model: KnownModelName) -> KnownModelName:
    if "openai" in model and "o1" in model:
        return "openai:gpt-4o"
    return model

# TOOL_FUNCS = set(f.__name__ for f in TOOLS)

def _find_user_task_number(query: str, env: functions_runtime.TaskEnvironment) -> str | None:
    """Find the user task number by matching the query against the prompt mapping.
    
    Args:
        query: The user query string
        env: The task environment to determine the suite
        
    Returns:
        The user task number (e.g., "user_task_2") or None if not found
    """
    # Determine suite name from environment type
    if isinstance(env, WorkspaceEnvironment):
        suite_name = "workspace"
    elif isinstance(env, TravelEnvironment):
        suite_name = "travel"
    elif isinstance(env, BankingEnvironment):
        suite_name = "banking"
    elif isinstance(env, SlackEnvironment):
        suite_name = "slack"
    else:
        return None
    
    # Load the mapping file
    mapping_file = Path(__file__).parent / "user_task_prompt_mapping.json"
    if not mapping_file.exists():
        return None
    
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception:
        return None
    
    # Normalize query for comparison (strip whitespace)
    query_normalized = query.strip()
    
    # Search for matching prompt in the suite
    suite_tasks = mapping.get(suite_name, {})
    for task_id, task_data in suite_tasks.items():
        prompt = task_data.get("prompt", "").strip()
        if prompt == query_normalized:
            return task_id
    
    return None


def highlight_executed_plan(plan: str, calls: Sequence[functions_runtime.FunctionCall], path: str) -> None:
    # Collect executed line numbers
    executed_line_numbers = {c.args.get("__lineno") for c in calls if "__lineno" in c.args}

    plan = plan.replace("```python\n", "").replace("\n```", "") # .replace("\n\n", "\n")

    # Save to file (e.g., HTML to view in browser)
    html_lines = []
    for i, line in enumerate(plan.splitlines(), start=0):
        if i+1 in executed_line_numbers:
            html_lines.append(f"<span style='color:green'>{line}</span>")
        else:
            html_lines.append(f"<span style='color:gray'>{line}</span>")
    html_plan = "<br>\n".join(html_lines)

    Path(path).write_text(html_plan, encoding="utf-8")

    print("✅ Saved to highlighted_plan.html")


def _append_sweep_progress(task_id: str, attempt: int, max_attempts: int, outcome: str) -> None:
    """Append one per-attempt outcome line to the shared progress file, if
    SWEEP_PROGRESS_FILE is set. Used to follow progress across parallel shards:
    every worker (its own process) appends here. A single write() of a short line
    to an O_APPEND file is atomic across processes on Linux (line < PIPE_BUF), so
    concurrent shards interleave whole lines without a lock. Best-effort/no-raise."""
    path = os.environ.get("SWEEP_PROGRESS_FILE")
    if not path:
        return
    try:
        import time
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tpid={os.getpid()}\t"
            f"{task_id}\tattempt={attempt}/{max_attempts}\t{outcome}\n"
        )
        with open(path, "a") as f:
            f.write(line)
    except Exception:
        pass


# --- Fused single-attempt planning (BRH_PLAN_FUSION, Appendix C of the paper) ---
# K strategy-diverse candidate plans are generated, then fused by the P-LLM into one
# phased plan (feasibility gate → MCP → neutralize → GUI-primary → GUI-alternate) that
# internalizes what the pass@k retry loop provides externally. Every fusion input is
# trusted (task text, tool names/inputSchema/pre-approved descriptions, and the P-LLM's
# own candidate plans) — the blind-planner invariant is untouched, and the fused plan
# flows through the unchanged BRH annotation/validation/enforcement pipeline.

_FUSION_STRATEGIES: list[tuple[str, str]] = [
    (
        "mcp_first",
        "[STRATEGY DIRECTIVE — MCP-first] For THIS plan, favor MCP tools: for every step, "
        "judge from the trusted tool descriptions whether an available MCP tool can COMPLETE "
        "the step — perform and confirm the change, not merely open or navigate to a page or "
        "dialog. Use call_mcp_tool(name='<tool>', arguments={...}) with EXACTLY the listed "
        "parameter names for every such step; use the GUI only for steps no tool completes. "
        "ALWAYS capture each result in a variable and branch on the 'mcp_tool_error' prefix. "
        "If no MCP tools are listed for this task, produce your best direct plan instead.",
    ),
    (
        "gui_direct",
        "[STRATEGY DIRECTIVE — GUI-direct] Do NOT use call_mcp_tool() in this plan. Prefer "
        "the most direct, robust GUI mechanisms available: address-bar navigation and direct "
        "URLs (e.g. chrome://settings/...), keyboard shortcuts and hotkeys, type-to-navigate, "
        "direct value entry into fields. Avoid long chains of menu clicks whenever a direct "
        "mechanism exists — every extra click is a grounding failure point.",
    ),
    (
        "gui_menu",
        "[STRATEGY DIRECTIVE — GUI-menu] Do NOT use call_mcp_tool() in this plan, and do NOT "
        "rely on keyboard shortcuts or address-bar tricks for navigation. Use conventional "
        "GUI navigation only: menus, toolbars, dialogs and buttons, clicking through the "
        "standard visual path a human would take. This plan is deliberately an ALTERNATIVE "
        "mechanism to a shortcut-based plan.",
    ),
    (
        "feasibility_skeptic",
        "[STRATEGY DIRECTIVE — feasibility skeptic] Treat the task as possibly IMPOSSIBLE: "
        "the control, setting or affordance it names may not exist in this environment. Your "
        "plan must FIRST perform a bounded search for it (at most 3-4 probing steps), verify "
        "its existence on the observed state, and call mark_fail() as the FINAL statement of "
        "that path if the evidence shows it genuinely does not exist. Only after that gate "
        "include the execution path for the case it does exist. Never invent hotkeys, menus "
        "or fields that perception has not confirmed.",
    ),
    (
        "verification_heavy",
        "[STRATEGY DIRECTIVE — verification-heavy] After EVERY sub-goal in your plan, verify "
        "the specific observable effect with check_done()/verify_hypothesis() on the concrete "
        "artifact (the changed value, the visible result — not a general impression), and "
        "include one bounded recovery step for each verification that can fail. Never call "
        "mark_done() unless the final verification passed.",
    ),
]

_FUSION_MERGE_TEMPLATE = """\
You previously produced {n} candidate plans for this task, each under a different strategy \
directive. Now ASSEMBLE them into ONE single robust plan. This is your ONLY execution \
attempt — there are no retries — so the plan itself must contain the fallbacks that retries \
would otherwise provide.

CANDIDATE PLANS:
{candidates}

ASSEMBLY RULES (mandatory):
1. Assemble, do not average. Take the feasibility gate from [feasibility_skeptic] ONLY if \
the task could plausibly be impossible; take the MCP phase from [mcp_first] ONLY for steps \
where a tool COMPLETES the action (per its trusted description); take the primary GUI path \
(G1) from the most robust direct mechanism (prefer [gui_direct]); take the alternate GUI \
path (G2) from a candidate whose mechanism DIFFERS from G1 (prefer [gui_menu]); apply \
[verification_heavy]'s discipline throughout. Never include a step no candidate justifies; \
drop redundant/duplicate steps.
2. Phase structure (omit phases that do not apply, keep the order): feasibility gate → MCP \
phase → neutralize → GUI-primary (G1) → verify → GUI-alternate (G2) → verify → terminal. \
The feasibility gate has EXACTLY TWO outcomes: (a) confirmed impossibility → mark_fail() as \
the last statement of that path; or (b) anything else — uncertainty, an errored probe, a \
failed MCP call — fall through to the execution phases leaving NO state behind. G1 and G2 \
must NEVER be guarded by a variable assigned in the feasibility gate or the MCP phase (no \
`abort` / `skip` / `impossible` flags); the ONLY value that may carry across phases is the \
verification result that says whether the effect is already achieved.
3. Every call_mcp_tool(...) result must be captured in a variable and tested for the \
'mcp_tool_error' prefix. On MCP success, verify the actual EFFECT with check_done() before \
skipping the GUI phase — a tool that only opens a page/dialog reports success without \
completing anything. A failed or unverified MCP phase falls through to the GUI path.
4. Neutralize before each fallback arm: the previous phase may have left a dialog or menu \
open — press('esc') / close it and return to a known state before G1/G2 acts.
5. The terminal verdict must be a FRESH verification. Immediately before mark_done(), run a \
final check_done() describing the task's measurable end-state, and gate mark_done() on THAT \
result only. NEVER gate mark_done() on phase-success flags or on any aggregate of earlier \
results — `if a or b or c: mark_done()` is FORBIDDEN. Phase flags only decide which phase \
runs next; they never decide whether the task is complete. mark_fail() ONLY on evidence of \
genuine absence after bounded search, and as the LAST statement of its path — nothing may \
execute after it.
6. Idempotency: gate G2 and any retry on a verification showing the effect is ABSENT. Never \
blindly re-execute toggle-like actions (a repeat on a mis-read success turns the setting \
back off). Idempotent actions (open file X, set value to V) may be re-executed safely.
7. Interpreter limits: no def / lambda / while / try / classes / imports / generators. Only \
sequential statements and if/elif/else. Express bounded retries as at most 2 unrolled \
repetitions, never loops.
8. Emit print("FUSION_PHASE:<phase>:<start|ok|fail>") at each phase boundary (phases: \
feasibility, mcp, g1, g2, terminal).
9. Budget: at most ~120 statements and at most 4 call_mcp_tool calls. Do not perceptually \
re-verify facts the task states as given (authorized targets, file names, values).
10. Verify with INDEPENDENT evidence. After a non-error call_mcp_tool, establish the effect \
from the CURRENT observed state (check_done / a fresh read of the document, file or screen) \
— never from the tool's own return text, because a tool that merely opens a page or dialog \
reports success without completing anything. For toggle-like steps (enable/disable, set/ \
unset, switch), verify the state BEFORE acting, act only if the target state is absent, then \
re-verify. When a verification is uncertain or ambiguous, report NOT done and continue to the \
next phase — a premature "done" skips every remaining phase and loses the task, whereas an \
extra verified attempt costs nothing.
11. Absence of evidence is NOT evidence of absence. Conclude that something is impossible \
ONLY from a probe that actually RAN and explicitly showed the affordance missing (a visual or \
textual confirmation of absence). A probe that errored — an mcp_tool_error, an empty read, a \
failed find — proves nothing about the task: on an errored probe, proceed to the execution \
phases and let them decide.

Output the complete final plan code only — all the code, directly runnable."""

_FUSION_REGEN_NOTE = (
    "[fusion regeneration] No state-changing action was executed before this error, so the "
    "environment is still in its initial state. This is still your single attempt: provide "
    "the FULL corrected plan (all the code, from the beginning), keeping the same phased "
    "structure and assembly rules as before. Fix the error's root cause; do not simply "
    "delete the failing phase."
)

# Marker the OSWorld suites embed in the error text when the guest VM itself is
# unreachable (see base_ui_uitars.EnvironmentUnavailableError). Matched as a string so
# this shared module keeps zero import coupling to the benchmark suites.
_ENV_UNAVAILABLE_MARKER = "ENV_UNAVAILABLE"

# Tools that observe but never change the environment: an interpreter crash whose trace
# contains ONLY these calls left the VM bit-identical to attempt start, so one plan
# regeneration is equivalent to the plan-compile retries the system already performs
# freely (code-is-None loop, annotator validation retries). Anything else is mutating.
_FUSION_READ_ONLY_TOOLS = frozenset({
    "find",
    "check_done",
    "verify_hypothesis",
    "get_page_elements",
    "get_page_text",
    "summarize_screenshot_content",
    "find_element_by_text",
    "get_google_credentials",
    "no_op",
    "wait",
    "query_ai_assistant",
})


class PrivilegedLLM(agent_pipeline.BasePipelineElement):
    """A pipeline element that generates and interprets code expressing the user query.

    Args:
        llm: the LLM to use to generate the code.
        system_prompt_generator: a function which takes as argument the functions available
          to the LLM and returns the system prompt to use.
        security_policies: a list of security policy tuples, where the first element is the tool name and the second one is the security policy to apply.
        eval_mode: the evaluation mode for the interpreter.
        quarantined_llm_retries: the number of retries for the quarantined LLM.
    """

    def __init__(
        self,
        llm: agent_pipeline.BasePipelineElement,
        security_policy_engine: type[AgentDojoSecurityPolicyEngine],
        quarantined_llm_model: KnownModelName,
        system_prompt_generator: Callable[
            [Iterable[functions_runtime.Function], set[str]], str
        ] = system_prompt_generator.default_system_prompt_generator,
        eval_mode: interpreter.MetadataEvalMode = interpreter.MetadataEvalMode.NORMAL,
        quarantined_llm_retries: int = 10,
        max_attempts: int = 10,
        part_path: Path | None = None,
        pllm_exposure_level: int = 0,
        system_defense_level: int = 2,
        plan_analysis_only: bool = False,
        brh_enabled: bool = False,
        brh_dir: Path | None = None,
        brh_mcp_tools: Mapping[str, Sequence[str]] | None = None,
        brh_server_map: dict[str, str] | None = None,
    ) -> None:
        """Initializes the PrivilegedLLM.

        ``brh_mcp_tools`` is the approved MCP tool manifest (tool name -> param
        names); when given, the BRH annotator is invited to produce
        ``mcp_constraints`` for plans that call those tools. ``None`` =
        HTTP-only annotation.

        ``brh_server_map`` is ``{tool_name: server_id}`` built by the approval
        loop; passed to ``generate_plan_constraints`` to inject
        ``allowed_tool_servers`` deterministically."""
        self.llm = llm
        self.system_prompt_generator = system_prompt_generator
        self.security_policy_engine = security_policy_engine
        self.eval_mode = eval_mode
        self.quarantined_llm_retries = quarantined_llm_retries
        self.dummy_runtime = functions_runtime.FunctionsRuntime()
        self.max_attempts = max_attempts
        self._plan_counter = 0
        self._part_path = part_path
        self._path = None
        self.pllm_exposure_level = pllm_exposure_level
        self.system_defense_level = system_defense_level
        # Gemini thinking and o1-* do not support JSON mode, so we fall back to base base Gemini/4o
        self.quarantined_llm_model: KnownModelName = _get_quarantined_llm(quarantined_llm_model)
        self.last_token_usage = None
        # Per-call accumulator for BRH annotation retries (reset before each generate_plan_constraints).
        self._pab_call_in = 0
        self._pab_call_out = 0
        # Per-task totals (reset at start of __call__, written as a summary at end).
        self._task_planning_in = 0
        self._task_planning_out = 0
        self._task_pab_in = 0
        self._task_pab_out = 0
        self._patch_llm_client()
        self.plan_analysis_only = plan_analysis_only
        self.brh_enabled = brh_enabled
        self.brh_dir = brh_dir
        self.brh_mcp_tools = brh_mcp_tools
        self.brh_server_map = brh_server_map
        # Constraints of the *current* plan, handed to the interpreter hook via
        # EvalArgs. None whenever constraints generation failed (hook no-ops,
        # branch_state.json stays fail-closed) — never reuse a previous plan's.
        self._pab_runtime = None
        # Fused single-attempt planning (opt-in via BRH_PLAN_FUSION=1; see
        # Appendix C of the paper). fusion_k candidate plans are generated and
        # fused into one phased plan on the first attempt. Inert when the flag is off.
        self.fusion_enabled = os.environ.get("BRH_PLAN_FUSION") == "1"
        self.fusion_k = int(os.environ.get("BRH_FUSION_K", "5")) if self.fusion_enabled else 0
        # One no-side-effects plan regeneration per task (auditable; reset in query()).
        self._fusion_regen_used = False
    
    def _patch_llm_client(self):
        """Patch the LLM client's API call methods to capture token usage."""
        
        # Try to patch OpenAI client
        if hasattr(self.llm, 'client') and isinstance(self.llm.client, openai.OpenAI):
            self._patch_openai_client(self.llm.client)
            print("✅ Patched OpenAI client for token tracking")
            return
        
        # Try to patch Anthropic client (AsyncAnthropic)
        if hasattr(self.llm, 'client'):
            # Check for AsyncAnthropic (used in the actual implementation)
            from anthropic import AsyncAnthropic
            if isinstance(self.llm.client, AsyncAnthropic):
                self._patch_anthropic_async_client(self.llm.client)
                print("✅ Patched Anthropic async client for token tracking")
                return
        
        # Try to patch Google Gemini client
        if hasattr(self.llm, 'client') and isinstance(self.llm.client, genai.Client):
            self._patch_gemini_client(self.llm.client)
            print("✅ Patched Gemini client for token tracking")
            return
        
        print(f"⚠️  Could not patch LLM client for token tracking - client type: {type(self.llm.client) if hasattr(self.llm, 'client') else 'no client attribute'}")

    def _patch_openai_client(self, client: openai.OpenAI):
        """Patch the OpenAI client's chat.completions.create to capture usage."""
        original_create = client.chat.completions.create
        
        def wrapped_create(*args, **kwargs):
            completion = original_create(*args, **kwargs)
            # Store the usage information
            if hasattr(completion, 'usage') and completion.usage:
                self.last_token_usage = (
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    completion.usage.total_tokens
                )
            return completion
        
        client.chat.completions.create = wrapped_create

    def _patch_anthropic_async_client(self, client):
        """Patch the Anthropic async client's messages.stream to capture usage."""
        from anthropic import AsyncAnthropic
        import asyncio
        from contextlib import asynccontextmanager
        
        original_stream = client.messages.stream
        
        @asynccontextmanager
        async def wrapped_stream(*args, **kwargs):
            async with original_stream(*args, **kwargs) as stream:
                # Let the stream complete normally
                yield stream
                
                # After the stream completes, capture usage
                try:
                    final_message = await stream.get_final_message()
                    if hasattr(final_message, 'usage') and final_message.usage:
                        self.last_token_usage = (
                            final_message.usage.input_tokens,
                            final_message.usage.output_tokens,
                            final_message.usage.input_tokens + final_message.usage.output_tokens
                        )
                except Exception as e:
                    print(f"⚠️  Failed to capture Anthropic usage: {e}")
        
        client.messages.stream = wrapped_stream

    def _patch_gemini_client(self, client: genai.Client):
        """Patch the Gemini client's models.generate_content to capture usage."""
        # Gemini uses synchronous API
        if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
            original_generate_content = client.models.generate_content
            
            def wrapped_generate_content(*args, **kwargs):
                response = original_generate_content(*args, **kwargs)
                # Store the usage information
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    self.last_token_usage = (
                        response.usage_metadata.prompt_token_count,
                        response.usage_metadata.candidates_token_count,
                        response.usage_metadata.total_token_count
                    )
                return response
            
            client.models.generate_content = wrapped_generate_content
            print("✅ Patched Gemini client (models.generate_content)")
        else:
            print("⚠️  Could not find Gemini generate_content method to patch")

    def _write_token_to_file(
        self,
        env: functions_runtime.TaskEnvironment,
        role: str,
        in_tok: int,
        out_tok: int,
    ) -> None:
        """Append one token-usage line to token_count_file if configured."""
        if not (isinstance(env, UIEnv) and hasattr(env.base_ui, "token_count_file") and env.base_ui.token_count_file):
            return
        model_name = getattr(self.llm, "model", None) or getattr(self.llm, "name", "unknown_model")
        with open(env.base_ui.token_count_file, "a") as f:
            f.write(
                f"Model {model_name} Function {role} - "
                f"Input Tokens: {in_tok}, Output Tokens: {out_tok}, "
                f"Total Tokens: {in_tok + out_tok}\n"
            )

    def _pab_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Adapter exposing the P-LLM as a plain (system, user) -> str call for the BRH annotator."""
        brh_messages = [
            ad_types.ChatSystemMessage(
                role="system", content=[ad_types.text_content_block_from_string(system_prompt)]
            ),
            ad_types.ChatUserMessage(
                role="user", content=[ad_types.text_content_block_from_string(user_prompt)]
            ),
        ]
        self.last_token_usage = None
        _, _, _, [*_, reply], _ = self.llm.query(
            query=user_prompt, runtime=self.dummy_runtime, messages=brh_messages
        )
        if self.last_token_usage:
            self._pab_call_in += self.last_token_usage[0]
            self._pab_call_out += self.last_token_usage[1]
        assert reply["role"] == "assistant"
        if not reply["content"]:
            return ""
        return ad_types.get_text_content_as_str(reply["content"])

    def _write_pab_constraints(self, query: str, code: str, env: functions_runtime.TaskEnvironment) -> None:
        """Writes plan_constraints.json and resets branch_state.json before execution."""
        from cobra.brh import BRHConfig, generate_plan_constraints, reset_branch_state
        from cobra.brh.hook import BRHRuntime
        from cobra.brh.skeleton import InvalidPlanError

        task_id = env.base_ui.ID if isinstance(env, UIEnv) else "task"
        plan_id = f"{task_id}_plan_{self._plan_counter}"
        config = BRHConfig(out_dir=self.brh_dir) if self.brh_dir is not None else BRHConfig()
        # For OSWorld-MCP: the manifest is built at init_environment time and stored
        # on env.base_ui (BaseUI), not on the UIEnvironment wrapper. Walk both.
        mcp_tools = (
            self.brh_mcp_tools
            or getattr(env, "_pab_mcp_manifest", None)
            or getattr(getattr(env, "base_ui", None), "_pab_mcp_manifest", None)
        )
        # Optional agent sitemap for HTTP endpoint pinning, supplied by the
        # harness on the env (same convention as the MCP manifest above).
        # `_pab_raw_sitemaps` ({domain: raw_entries}) is the preferred form: it
        # goes through the sitemap trust gate (hash-pinned registry, changed
        # sitemaps excluded fail-closed — cobra.brh.sitemap_trust), exactly
        # like MCP tool definitions. `_pab_http_manifest` (pre-sanitized) is
        # the legacy form and is treated as already-approved fixture input.
        raw_sitemaps = (
            getattr(env, "_pab_raw_sitemaps", None)
            or getattr(getattr(env, "base_ui", None), "_pab_raw_sitemaps", None)
        )
        http_manifest = (
            getattr(env, "_pab_http_manifest", None)
            or getattr(getattr(env, "base_ui", None), "_pab_http_manifest", None)
        )
        # Default (unset or "interactive"): human-in-the-loop gates (sitemap
        # re-approval + new-site proposal, phase-2 plan confirmation) — the
        # production-safe posture. BRH_APPROVAL=auto is an explicit, opt-in
        # escape hatch for unattended benchmarks/tests: no interaction,
        # sitemap trust-on-first-use. It must be set deliberately by a
        # benchmark harness and is never the default for a live deployment.
        from cobra.mcp_proxy.approval import ApprovalMode
        approval_mode = ApprovalMode(os.environ.get("BRH_APPROVAL", "interactive"))
        self._pab_runtime = None
        self._pab_call_in = 0
        self._pab_call_out = 0
        try:
            constraints = generate_plan_constraints(
                llm_call=self._pab_llm_call,
                task=query,
                plan_markdown=code,
                plan_id=plan_id,
                config=config,
                log_dir=self._path,
                mcp_tools=mcp_tools,
                server_map=self.brh_server_map,
                http_manifest=http_manifest,
                raw_sitemaps=raw_sitemaps,
                approval_mode=approval_mode,
            )
            self._pab_runtime = BRHRuntime(constraints=constraints, state_path=config.state_path)
            print(f"✅ BRH: wrote {config.constraints_path} for {plan_id}")
        except InvalidPlanError as e:
            print(f"⚠️ BRH: unparseable plan {plan_id} ({e}); resetting state fail-closed")
            reset_branch_state(plan_id, config)
        except Exception as e:
            print(f"⚠️ BRH: constraints generation failed for {plan_id} ({e}); resetting state fail-closed")
            reset_branch_state(plan_id, config)
        finally:
            if self._pab_call_in > 0:
                self._write_token_to_file(env, "PrivilegedLLM_pab_annotation", self._pab_call_in, self._pab_call_out)
                print(f"✅ BRH annotation tokens: Input={self._pab_call_in}, Output={self._pab_call_out}, Total={self._pab_call_in + self._pab_call_out}")
                self._task_pab_in += self._pab_call_in
                self._task_pab_out += self._pab_call_out

    def run_code(
        self,
        code: str,
        env: functions_runtime.TaskEnvironment,
        namespace: ns.Namespace,
        dependencies: Iterable[CaMeLValue],
    ) -> tuple[
        str,
        Sequence[tuple[functions_runtime.FunctionCall, functions_runtime.FunctionReturnType]],
        interpreter.CaMeLException | None,  # Add optional error return
        ns.Namespace,
        Iterable[CaMeLValue],
    ]:
        """Interprets the code in `code` by calling the provided functions.

        Args:
            code: a string with the code to interpret
            env: the task environment.
            ad_functions_runtime: the functions runtime.

        Returns:
            A tuple containing the final model output, a list of tuples where the first
            element is the function call that was done and the second one is the result,
            and an optional Exception if one occurred during interpretation.
        """

        eval_args = interpreter.EvalArgs(self.security_policy_engine(env), self.eval_mode)

        # BRH: hand the current plan's constraints to the interpreter hook and
        # activate the root scope (root-level requests are authorised by the
        # `root` annotation; before this write the state is fail-closed).
        if self.brh_enabled and self._pab_runtime is not None:
            from cobra.brh import hook as brh_hook

            eval_args = brh_hook.attach(eval_args, self._pab_runtime)
            brh_hook.activate_root(self._pab_runtime)

        print(code)
        if isinstance(env, UIEnv):
            plan_path = self._path / f"plan_{self._plan_counter}.txt"
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(code)


        interpreter_res, updated_namespace, tool_calls, dependencies = interpreter.parse_and_interpret_code(
            code, namespace, [], dependencies, eval_args
        )

        if isinstance(env, UIEnv):
            highlighted_plan_path = self._path / f"highlighted_plan_{self._plan_counter}.html"
            highlight_executed_plan(code, tool_calls, str(highlighted_plan_path))

        # Build inline-annotated source from the tool call log
        try:
            raw_src = interpreter.extract_code_block(code)
        except interpreter.InvalidOutputError:
            raw_src = code  # fallback if not fenced
        # 1) Strip any prior '# >>> ...' lines the model hallucinated
        clean_src, map_lineno = _strip_old_annotations_and_make_mapper(raw_src)

        # 2) Remap print line numbers to the cleaned source, then annotate
        trace = collect_print_trace(tool_calls)
        remapped_trace = [(map_lineno(ln), text) for (ln, text) in trace]
        annotated_source = f"```python\n{inline_annotate(clean_src, remapped_trace)}\n```"


        printed_output = extract_print_output(tool_calls)
        ad_tool_calls = make_ad_tool_calls(tool_calls)

        match interpreter_res:
            case result.Error(error):
                return (printed_output, ad_tool_calls, error, updated_namespace, dependencies, annotated_source)  # Return the exception
            case result.Ok(v):
                res = v

        return (
            f"{printed_output}\n{res.raw if res.raw is not None else ''}",
            ad_tool_calls,
            None,
            updated_namespace,
            dependencies,
            annotated_source
            )# No error

    # ------------------------------------------------------------------ fusion mode
    def _fusion_llm_call(
        self,
        query: str,
        messages: Sequence[ad_types.ChatMessage],
        env: functions_runtime.TaskEnvironment,
        label: str,
    ) -> str:
        """One P-LLM call on an explicit message thread, with per-label token logging.

        Fusion calls are accounted under ``planning`` in the task summary (they ARE
        planning cost), with a distinct Function label per call so the overhead is
        measurable from token_count_file."""
        self.last_token_usage = None
        _, _, _, [*_, reply], _ = self.llm.query(
            query=query, runtime=self.dummy_runtime, messages=messages
        )
        if self.last_token_usage:
            in_t, out_t, _ = self.last_token_usage
            self._task_planning_in += in_t
            self._task_planning_out += out_t
            self._write_token_to_file(env, label, in_t, out_t)
        assert reply["role"] == "assistant"
        if not reply["content"]:
            return ""
        return ad_types.get_text_content_as_str(reply["content"])

    def _fusion_dump(self, env: functions_runtime.TaskEnvironment, filename: str, text: str) -> None:
        """Persist a fusion artifact next to the plan dumps (best-effort)."""
        try:
            if isinstance(env, UIEnv) and self._path is not None:
                (self._path / filename).write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"[fusion] dump {filename} failed: {e}")

    @staticmethod
    def _fusion_user_msg(text: str) -> ad_types.ChatUserMessage:
        return ad_types.ChatUserMessage(
            role="user", content=[ad_types.text_content_block_from_string(text)]
        )

    def _fusion_generate(
        self,
        query: str,
        base_messages: Sequence[ad_types.ChatMessage],
        env: functions_runtime.TaskEnvironment,
    ) -> str | None:
        """K strategy-diverse candidates → one fusion call → lint (+≤2 repairs).

        Returns the plan code to execute, or None to fall back to the plain single
        planning call (the caller's existing while-loop acts as the safety net)."""
        try:
            from cobra.interaction.environments import plan_lint
        except Exception as e:
            print(f"[fusion] plan_lint unavailable ({e}) — fusion disabled for this task")
            return None

        _task_id = getattr(getattr(env, "base_ui", None), "ID", "task")

        # --- Stage 1: candidates (sequential — token accounting is not thread-safe).
        candidates: list[tuple[str, str]] = []
        for name, directive in _FUSION_STRATEGIES[: self.fusion_k]:
            msgs = [*base_messages, self._fusion_user_msg(directive)]
            code = self._fusion_llm_call(
                query, msgs, env, f"PrivilegedLLM_fusion_candidate_{name}"
            )
            print(f"[fusion] candidate [{name}]: {len(code)} chars")
            if code:
                candidates.append((name, code))
                self._fusion_dump(env, f"fusion_candidate_{name}.txt", code)
        if not candidates:
            print("[fusion] no candidates produced — falling back to single planning call")
            return None

        # --- Stage 2: fusion call.
        cand_block = "\n\n".join(f"--- candidate [{n}] ---\n{c}" for n, c in candidates)
        merge_msg = _FUSION_MERGE_TEMPLATE.format(n=len(candidates), candidates=cand_block)
        merge_thread: list[ad_types.ChatMessage] = [*base_messages, self._fusion_user_msg(merge_msg)]
        fused = self._fusion_llm_call(query, merge_thread, env, "PrivilegedLLM_fusion_merge")

        # --- Stage 3: deterministic lint + ≤2 repair rounds.
        lint_log: list[str] = []
        lint_res = plan_lint.lint_plan(fused) if fused else plan_lint.LintResult(
            hard=["empty fusion output"], repair=[]
        )
        repairs = 0
        while fused and not lint_res.ok and repairs < 2:
            lint_log.append(f"round {repairs}: hard={lint_res.hard} repair={lint_res.repair}")
            merge_thread = [
                *merge_thread,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(fused)],
                    tool_calls=None,
                ),
                self._fusion_user_msg(plan_lint.format_repair_message(lint_res)),
            ]
            repairs += 1
            fused = self._fusion_llm_call(
                query, merge_thread, env, f"PrivilegedLLM_fusion_repair_{repairs}"
            )
            lint_res = plan_lint.lint_plan(fused) if fused else plan_lint.LintResult(
                hard=["empty repair output"], repair=[]
            )
        lint_log.append(f"final: hard={lint_res.hard} repair={lint_res.repair}")
        self._fusion_dump(env, "fusion_lint.log", "\n".join(lint_log))

        if fused and lint_res.executable:
            # Soft findings don't block execution — a fused plan missing a verify is
            # still better than dropping the fusion structure entirely.
            if lint_res.repair:
                print(f"[fusion] accepting fused plan with soft lint findings: {lint_res.repair}")
            self._fusion_dump(env, "fusion_merged.txt", fused)
            return fused

        # --- Fallback: best single candidate, unfused (fusion is monotone — never
        # worse than best-of-K).
        fb = plan_lint.pick_fallback(candidates)
        if fb is not None:
            fb_name, fb_code = fb
            print(f"[fusion] fused plan unusable — falling back to candidate [{fb_name}]")
            self._fusion_dump(env, "fusion_fallback.txt", f"[{fb_name}]\n{fb_code}")
            _append_sweep_progress(_task_id, 1, self.max_attempts, f"FUSION_FALLBACK:{fb_name}")
            return fb_code
        print("[fusion] no usable candidate either — falling back to single planning call")
        return None

    def _fusion_regenerate(
        self,
        query: str,
        code: str,
        interpretation_error: "interpreter.CaMeLException",
        base_messages: Sequence[ad_types.ChatMessage],
        env: functions_runtime.TaskEnvironment,
    ) -> str | None:
        """One plan regeneration after a crash that executed zero mutating calls."""
        try:
            from cobra.interaction.environments import plan_lint
        except Exception:
            plan_lint = None
        msgs = [
            *base_messages,
            *make_error_messages(code, interpretation_error),
            self._fusion_user_msg(_FUSION_REGEN_NOTE),
        ]
        new_code = self._fusion_llm_call(query, msgs, env, "PrivilegedLLM_fusion_regen")
        if not new_code:
            return None
        if plan_lint is not None and not plan_lint.lint_plan(new_code).executable:
            print("[fusion] regenerated plan fails hard lint — keeping the failed outcome")
            return None
        self._fusion_dump(env, "fusion_regen_plan.txt", new_code)
        return new_code

    def _generate_and_interpret_code(
        self,
        query: str,
        runtime: functions_runtime.FunctionsRuntime,
        namespace: ns.Namespace,
        env: functions_runtime.TaskEnvironment,
        messages: Sequence[ad_types.ChatMessage],
        privileged_llm_messages: Sequence[ad_types.ChatMessage],
        system_prompt: str,
        previous_printed_output: str,
        dependencies: Iterable[CaMeLValue],
    ) -> tuple[
        str,  # tried_code (clean code without annotations)
        str,  # annotated_code (code with annotations)
        str,  # previous_printed_output + formatted_model_output
        Sequence[tuple[functions_runtime.FunctionCall, functions_runtime.FunctionReturnType]],
        interpreter.CaMeLException | None,
        list[ad_types.ChatMessage],
        list[ad_types.ChatMessage],
        ns.Namespace,
        Iterable[CaMeLValue],
    ]:
        if len(privileged_llm_messages) == 0:
            privileged_llm_messages = [
                ad_types.ChatSystemMessage(
                    role="system", content=[ad_types.text_content_block_from_string(system_prompt)]
                ),
                ad_types.ChatUserMessage(role="user", content=[ad_types.text_content_block_from_string(query)]),
            ]

        code = None
        attempts = 5
        print(f"PrivilegedLLM messages: {privileged_llm_messages}")
        print(f"Query: {query}")
        model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'name', 'unknown_model')

        # Fused single-attempt planning (opt-in): K candidates + fusion on the first
        # attempt only. If fusion yields nothing, the while-loop below is the safety
        # net (plain single planning call — identical to the non-fusion path).
        _fusion_active = self.fusion_k > 1 and self._plan_counter == 0
        if _fusion_active:
            code = self._fusion_generate(query, list(privileged_llm_messages), env)
            _fusion_active = bool(code)

        while (code is None or code == "") and attempts > 0:
            self.last_token_usage = None

            _, _, _, [*_, code_message], _ = self.llm.query(
                query=query,
                runtime=self.dummy_runtime,
                messages=privileged_llm_messages,
            )

            if self.last_token_usage:
                in_t, out_t, _ = self.last_token_usage
                self._task_planning_in += in_t
                self._task_planning_out += out_t
                self._write_token_to_file(env, "PrivilegedLLM_planning", in_t, out_t)
                print(f"✅ P-LLM planning tokens: Input={in_t}, Output={out_t}, Total={in_t + out_t}")
            
            if self.llm.name is not None and "gemini" in self.llm.name and "exp" in self.llm.name:
                time.sleep(6)
            assert code_message["role"] == "assistant"
            if code_message["content"]:
                code = ad_types.get_text_content_as_str(code_message["content"])
            attempts -= 1

        if code is None or code == "":
            empty_message = ad_types.ChatAssistantMessage(
                role="assistant",
                content=[
                    ad_types.text_content_block_from_string(
                        "The generated code was `None`. This message is added by the system does not come from the assistant."
                    )
                ],
                tool_calls=None,
            )
            return (
                "",  # tried_code
                "",  # annotated_code
                previous_printed_output,
                [],
                None,
                [*messages, empty_message],
                list(privileged_llm_messages),
                namespace,
                dependencies,
            )

        # --- Plan-analysis-only: save the plan, skip execution ---
        if self.plan_analysis_only:
            updated_messages = [
                *messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(code)],
                    tool_calls=None,
                ),
            ]

            # Write plan to disk
            if self._part_path is not None:
                if isinstance(env, UIEnv):
                    dump_dir = self._path if self._path is not None else self._part_path / env.base_ui.ID
                else:
                    base_dir = self._path if self._path is not None else self._part_path
                    dump_dir = base_dir / "none"
                dump_dir.mkdir(parents=True, exist_ok=True)
                out_file = dump_dir / f"model_output_{self._plan_counter}.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(code)

            # Skip execution, return empty results
            return (
                code,       # tried_code
                code,       # annotated_code (no execution → no annotations)
                previous_printed_output,
                [],          # empty tool_calls_results
                None,        # no interpretation_error
                updated_messages,
                list(privileged_llm_messages),
                namespace,
                dependencies,
            )

        # BRH: derive and persist plan constraints before any execution starts.
        if self.brh_enabled:
            self._write_pab_constraints(query, code, env)

        # Keep the pre-execution namespace/dependencies for the fusion no-side-effects
        # regeneration (re-running a corrected plan must start from the same bindings).
        _ns_in, _deps_in = namespace, dependencies
        model_output, tool_calls_results, interpretation_error, namespace, dependencies, annotated= self.run_code(
            code, env, namespace, dependencies
        )

        # Fusion no-side-effects regeneration:
        # at --max-attempts 1 an interpreter crash is fatal even when the environment
        # was never touched. If the crashed plan executed ONLY read-only tools, the VM
        # is bit-identical to attempt start, so one regeneration is equivalent to the
        # plan-compile retries the system already performs freely. One per task,
        # auditable via the FUSION_REGEN progress-file line. A crash after any
        # mutating call is a genuine failure — no second chance.
        if _fusion_active and interpretation_error is not None and not self._fusion_regen_used:
            _mutating = [
                c.function for c, _ in tool_calls_results
                if c.function not in _FUSION_READ_ONLY_TOOLS
            ]
            # An unreachable guest is an infrastructure failure: the plan is fine and a
            # new one would crash on the very next perception call, burning ~15k P-LLM
            # tokens for nothing. Attribute it and let the attempt end.
            _env_dead = _ENV_UNAVAILABLE_MARKER in str(interpretation_error)
            if _env_dead:
                # The attempt loop tags the progress line ENV_UNAVAILABLE; nothing to
                # add here beyond declining the (useless) regeneration.
                print("[fusion] guest unavailable (ENV_UNAVAILABLE) — no regeneration")
            elif not _mutating:
                self._fusion_regen_used = True
                _regen_code = self._fusion_regenerate(
                    query, code, interpretation_error, list(privileged_llm_messages), env
                )
                if _regen_code:
                    _append_sweep_progress(
                        getattr(getattr(env, "base_ui", None), "ID", "task"),
                        self._plan_counter + 1, self.max_attempts, "FUSION_REGEN",
                    )
                    code = _regen_code
                    if self.brh_enabled:
                        self._write_pab_constraints(query, code, env)
                    model_output, tool_calls_results, interpretation_error, namespace, dependencies, annotated = self.run_code(
                        code, env, _ns_in, _deps_in
                    )
            else:
                print(f"[fusion] crash after mutating calls {_mutating[:5]} — no regeneration")
        print(f"Annotations: {annotated}")
        print(f"Model output: {model_output}")
        print(f"Tool calls results: {format_tool_calls_results(tool_calls_results)}")
        print(f"Tool calls results: {tool_calls_results}")
        print(f"Interpretation error: {interpretation_error}")


        tool_calls: list[functions_runtime.FunctionCall] = []
        tool_call_messages: list[ad_types.ChatMessage] = []

        for tool_call, tool_result in tool_calls_results:
            tool_call_messages.append(
                ad_types.ChatToolResultMessage(
                    role="tool",
                    tool_call=tool_call,
                    content=[
                        ad_types.text_content_block_from_string(
                            tool_execution.tool_result_to_str(tool_result, dump_fn=custom_yaml_dump)
                        )
                    ],
                    tool_call_id=None,
                    error=None,
                )
            )
            tool_calls.append(tool_call)

        if interpretation_error:
            error_messages = make_error_messages(code, interpretation_error)
            updated_messages = [*messages, *tool_call_messages, *error_messages]
            privileged_llm_messages = [*privileged_llm_messages, *error_messages]
        else:
            updated_messages = [
                *messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(code)],
                    tool_calls=tool_calls,
                ),
                *tool_call_messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(previous_printed_output + model_output)],
                    tool_calls=None,
                ),
            ]
        # 1) sentence breaks with exceptions (., but not before ., digits, ' or ", and don't double if newline exists)
        formatted_model_output = re.sub(
            r'(?<!\.)\.(?![.\d\'"\u2019\u201D])(?!\s*\n)\s*',
            '.\n',
            model_output,
        )

        # 2) add newline in `length=\d+text=` patterns (allows optional spaces before text=)
        formatted_model_output = re.sub(
            r'length=(\d+)(?=\s*text=)',
            r'length=\1\n',
            formatted_model_output,
        )


        if self._part_path is not None:
            # ensure path exists
            if isinstance(env, UIEnv):
                dump_dir = self._part_path / env.base_ui.ID
            else:
                # For non-UI environments, use self._path if set (includes task number folder)
                # Add 'none' subdirectory to match where none.json is saved
                base_dir = self._path if self._path is not None else self._part_path
                dump_dir = base_dir / "none"
            dump_dir.mkdir(parents=True, exist_ok=True)
            out_file = dump_dir / f"model_output_{self._plan_counter}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(annotated)

        return (
            code,  # return clean code instead of annotated
            annotated,  # return annotated separately
            previous_printed_output + formatted_model_output,
            tool_calls_results,
            interpretation_error,
            updated_messages,
            list(privileged_llm_messages),
            namespace,
            dependencies,
        )

    def query(
        self,
        query: str,
        runtime: functions_runtime.FunctionsRuntime,
        env: _E = functions_runtime.EmptyEnv(),
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[
        str,
        functions_runtime.FunctionsRuntime,
        _E,
        Sequence[ad_types.ChatMessage],
        dict,
    ]:
        """Generates and interprets code which expresses the user query."""
        self._task_planning_in = 0
        self._task_planning_out = 0
        self._task_pab_in = 0
        self._task_pab_out = 0
        self._fusion_regen_used = False
        privileged_llm_messages = []
        if not isinstance(env, UIEnv):
            def query_ai_assistant(query: str, output_schema: type[_T]) -> _T:
                return quarantined_llm.query_quarantined_llm(
                    llm=(self.quarantined_llm_model),
                    query=query,
                    output_schema=output_schema,
                    retries=self.quarantined_llm_retries,
                )

            query_ai_assistant.__doc__ = quarantined_llm.query_quarantined_llm.__doc__
            runtime.register_function(query_ai_assistant)
            
            # Set up path with user task number folder for non-UI environments
            # The benchmark/OutputLogger adds model+cobra and suite_name to the path
            # We need to match that structure: part_path / f"{model_name}+cobra" / suite_name / user_task_number
            if self._part_path is not None:
                user_task_number = _find_user_task_number(query, env)
                
                # Get model name from llm
                model_name = getattr(self.llm, 'name', None) or getattr(self.llm, 'model', None) or 'unknown'
                # Handle openrouter format: "openai/gpt-5" -> "gpt-5"
                if '/' in model_name:
                    if model_name.startswith("x-ai/") or model_name.startswith("moonshotai/") or model_name.startswith("deepseek/") or model_name.startswith("openai/gpt-oss"):
                        model_name = model_name.replace('/','_')
                    else:
                        model_name = model_name.split('/')[1]
                # Get suite name from environment type
                if isinstance(env, WorkspaceEnvironment):
                    suite_name = "workspace"
                elif isinstance(env, TravelEnvironment):
                    suite_name = "travel"
                elif isinstance(env, BankingEnvironment):
                    suite_name = "banking"
                elif isinstance(env, SlackEnvironment):
                    suite_name = "slack"
                else:
                    suite_name = "unknown"
                
                # Construct path: part_path / f"{model_name}+cobra" / suite_name / user_task_number
                model_camel = f"{model_name}+cobra"
                if user_task_number:
                    self._path = self._part_path / model_camel / suite_name / user_task_number
                else:
                    self._path = self._part_path / model_camel / suite_name
                if (self._path / "none" / "model_output_0.txt").exists():
                    self._path = self._part_path / model_camel / suite_name / "unknown_task_number"
                self._path.mkdir(parents=True, exist_ok=True)
                
        else:
            if self._part_path is not None:
                self._path = self._part_path / env.base_ui.ID / "none"
                self._path.mkdir(parents=True, exist_ok=True)

                # NEW: write a one-time suite marker file
                marker = self._path / "suite.txt"
                tid   = str(env.base_ui.OSW_CLI)
                marker.write_text(f"task_id={tid}\n", encoding="utf-8")

                env.base_ui.path = self._path

                # saving the first screenshot (skip in plan-only mode — no VM needed)
                if not self.plan_analysis_only:
                    env.base_ui.take_and_save_screenshot(step=0, retry=0, attempt=0)
                
                

        builtins_namespace = ns.Namespace.with_builtins()
        # Models get confused in the other suites and should not use datetime stuff
        classes_to_exclude = (
            {
                "datetime",
                "timedelta",
                "date",
                "time",
                "NaiveDatetime",
                "timezone",
            }
            if not isinstance(env, WorkspaceEnvironment | TravelEnvironment)
            else set()
        )

        # For UIEnv, try to pass pllm_exposure_level; for non-UIEnv, use default behavior
        if isinstance(env, UIEnv):
            system_prompt = self.system_prompt_generator(
                runtime.functions.values(), 
                classes_to_exclude,
                pllm_exposure_level=self.pllm_exposure_level,
                system_defense_level=self.system_defense_level
            )
        else:
            system_prompt = self.system_prompt_generator(
                runtime.functions.values(), 
                classes_to_exclude
            )

        if isinstance(env, BankingEnvironment):
            system_prompt += "\n\nNote that, in the transaction history, the transactions from the user have 'me' as sender, and still habe positive amounts."

        if not isinstance(env, WorkspaceEnvironment):
            new_variables = {k: v for k, v in builtins_namespace.variables.items() if k not in classes_to_exclude}
            builtins_namespace = dataclasses.replace(builtins_namespace, variables=new_variables)

        namespace = builtins_namespace.add_variables(make_agentdojo_namespace(builtins_namespace, runtime, env))

        model_output = ""
        dependencies = ()
        had_previous_error = False
        for i in range(self.max_attempts):
            self._plan_counter = i
            # Fix#1b: clear the per-attempt terminal-FAIL intent before running the plan.
            if isinstance(env, UIEnv):
                try:
                    setattr(env.base_ui, "_plan_terminal_intent", None)
                except Exception:
                    pass
            print(f"[Attempt {i+1}/{self.max_attempts}] Prompting P-LLM for code…")
            (tried_code, annotated_code, model_output, _, interpretation_error, messages, privileged_llm_messages, namespace, dependencies) = (
                self._generate_and_interpret_code(
                    query,
                    runtime,
                    namespace,
                    env,
                    messages,
                    privileged_llm_messages,
                    system_prompt,
                    model_output,
                    dependencies,
                )
            )
            print(f"[Attempt {i+1}] P-LLM returned {interpretation_error}, {model_output}")

            # If we had a successful run after previous errors, clear the error history
            if not interpretation_error and had_previous_error:
                print(f"[Attempt {i+1}] Clearing accumulated error history after successful run")
                # Keep only non-error messages (filter out both error messages AND the code that caused them)
                cleaned_messages = []
                skip_next = False  # Flag to skip assistant message before error
                
                for idx, msg in enumerate(privileged_llm_messages):
                    if msg["role"] == "user":
                        content_str = ad_types.get_text_content_as_str(msg["content"])
                        # If this is an error message, skip it and mark to skip the previous assistant message
                        if "Traceback" in content_str and "Running the code gave the following error" in content_str:
                            # Remove the last added message if it was the assistant code that caused this error
                            if cleaned_messages and cleaned_messages[-1]["role"] == "assistant":
                                cleaned_messages.pop()
                            continue  # Skip this error message
                        else:
                            cleaned_messages.append(msg)
                    else:
                        cleaned_messages.append(msg)
                
                privileged_llm_messages = cleaned_messages
                had_previous_error = False
            elif interpretation_error:
                had_previous_error = True

            task_done = False
            _sweep_task_id = getattr(getattr(env, "base_ui", None), "ID", "task")
            if not interpretation_error:
                if self.plan_analysis_only:
                    pass  # don't break — keep generating plans
                elif isinstance(env, UIEnv):
                    try:
                        score = env.base_ui.env.evaluate()
                        # Fix#1b: make mark_fail() effectively plan-terminating for the
                        # OSWorld `infeasible` evaluator, which scores only on
                        # action_history[-1] == "FAIL". If the plan expressed FAIL intent
                        # but later actions buried it, re-assert FAIL as the terminal
                        # action and re-evaluate. Guarded by score != 1.0 so a genuinely
                        # completed (feasible) task is NEVER overridden — no regression.
                        if score != 1.0 and getattr(env.base_ui, "_plan_terminal_intent", None) == "FAIL":
                            _hist = getattr(env.base_ui.env, "action_history", None)
                            if not _hist or _hist[-1] != "FAIL":
                                env.base_ui.env.step("FAIL", 0)
                                score = env.base_ui.env.evaluate()
                                _append_sweep_progress(_sweep_task_id, i + 1, self.max_attempts, "FAIL_reasserted")
                        _append_sweep_progress(_sweep_task_id, i + 1, self.max_attempts, f"score={score}")
                        if score==1.0:
                            print(f"Task completed successfully with score: {score}")
                            task_done = True
                            break
                    except Exception as e:
                        print(f"Error evaluating environment: {e}")
                        _append_sweep_progress(_sweep_task_id, i + 1, self.max_attempts, f"EVAL_ERROR:{e}")
                else:
                    task_done = True  # Assume task is done if no error and not UIEnv
                    break
            else:
                # Distinguish "the plan is broken" from "the VM went away": only the
                # first is a benchmark signal. Results marked ENV_UNAVAILABLE should
                # be re-run rather than resumed past as a FAIL.
                _outcome = (
                    "ENV_UNAVAILABLE"
                    if _ENV_UNAVAILABLE_MARKER in str(interpretation_error)
                    else "PLAN_ERROR"
                )
                _append_sweep_progress(_sweep_task_id, i + 1, self.max_attempts, _outcome)
            
            if not task_done and isinstance(env, UIEnv) and not self.plan_analysis_only:
                print("Task not done yet, continuing…")
                # NOTE : try without environment reset for now
                print(f"Resetting environment for attempt {i+1}…")
                env.base_ui.reset()

                # mcphint HARD two-phase policy (BRH_MCP_PREFER): attempts 1..BRH_MCP_ROUNDS
                # see the full MCP manifest and are incentivised to use it; after the last
                # MCP round fails, HARD-switch to GUI-only for all remaining attempts by
                # (a) rebuilding the planner query from the GUI-only prompt variant (no MCP
                # manifest at all) and (b) RESETTING the planner thread so the tool list is
                # no longer anywhere in context. The GUI-only `query` persists for attempts
                # BRH_MCP_ROUNDS+1..max (the switch fires only once, at i+1 == rounds).
                # Applied regardless of pllm_exposure_level.
                # In fusion mode the MCP→GUI schedule is internalized as in-plan
                # phases, so the attempt-level hard switch must never fire (even if
                # someone runs fusion with --max-attempts > 1).
                if os.environ.get("BRH_MCP_PREFER") == "1" and not self.fusion_enabled:
                    _mcp_rounds = int(os.environ.get("BRH_MCP_ROUNDS", "2"))
                    if i + 1 == _mcp_rounds and i + 1 < self.max_attempts:
                        _gui_prompt = getattr(getattr(env, "base_ui", None), "_pab_gui_only_prompt", None)
                        if _gui_prompt:
                            query = _gui_prompt
                            # Empty thread → _generate_and_interpret_code rebuilds a fresh
                            # [system, user(GUI-only query)] with NO MCP manifest in context.
                            privileged_llm_messages = []
                            print(f"[Attempt {i+1}] mcphint: HARD switch to GUI-only "
                                  f"(MCP manifest removed from planner context for attempts {_mcp_rounds+1}+)")
                        else:
                            # Fallback (no stashed GUI prompt, e.g. task without MCP tools):
                            # keep the softer directive-append behaviour.
                            _gui_directive = (
                                f"[MCP→GUI policy] The first {_mcp_rounds} attempt(s) are over and the task is "
                                f"still not complete. For this and all remaining attempts, do NOT use "
                                f"call_mcp_tool(); use GUI navigation only (find / find_element_by_text / click / "
                                f"type_text / hotkey) to complete the task, including selecting any dialog options "
                                f"and clicking the final confirm button."
                            )
                            privileged_llm_messages.append(
                                ad_types.ChatUserMessage(
                                    role="user",
                                    content=[ad_types.text_content_block_from_string(_gui_directive)],
                                )
                            )
                            print(f"[Attempt {i+1}] mcphint: switching planner to GUI-only for remaining attempts")

                if self.pllm_exposure_level == 0:
                    # Level 0: No feedback at all
                    pass
                elif self.pllm_exposure_level == 1:
                    # Level 1: Show clean code without annotations
                    privileged_llm_messages.append(
                        ad_types.ChatAssistantMessage(
                            role="assistant",
                            content=[ad_types.text_content_block_from_string(tried_code)],
                            tool_calls=None,
                        )
                    )
                    failure_note = (
                        f"Your previous code ran without interpreter errors but did not complete the task "
                        f"(score=0). Please revise your plan or try a different approach."
                    )
                    privileged_llm_messages.append(
                        ad_types.ChatUserMessage(
                            role="user",
                            content=[ad_types.text_content_block_from_string(failure_note)]
                        )
                    )
                elif self.pllm_exposure_level == 2:
                    # Level 2: Show annotated code with detailed feedback (current behavior)
                    privileged_llm_messages.append(
                        ad_types.ChatAssistantMessage(
                            role="assistant",
                            content=[ad_types.text_content_block_from_string(annotated_code)],
                            tool_calls=None,
                        )
                    )
                    failure_note = (
                        f"Your previous code ran without interpreter errors but did not complete the task "
                        f"(score=0). Please revise your plan or try a different approach. Adding print statements can help you debug your code and understand the problems better, especially if you print out the thoughts from `find`, i.e., the `thought` of FindResult. Summaries of screenshot will help you identify the problem for the next run if you add print statements. For example you might be mistaken about the initial state of the environment. Longer summaries provide more environment information."
                    )
                    privileged_llm_messages.append(
                        ad_types.ChatUserMessage(
                            role="user",
                            content=[ad_types.text_content_block_from_string(failure_note)]
                        )
                    )

        extra_args["camel_namespace"] = namespace
        if messages[-1]["role"] == "user" and "\n\nTraceback" in ad_types.get_text_content_as_str(
            messages[-1]["content"]
        ):
            messages = [*messages, ad_types.ChatAssistantMessage(role="assistant", content=[ad_types.text_content_block_from_string("Task failed after all retry attempts.")], tool_calls=None)]

        # Write P-LLM task summary (planning + annotation; Q-LLM steps are in the same file per-step).
        if isinstance(env, UIEnv) and hasattr(env.base_ui, "token_count_file") and env.base_ui.token_count_file:
            task_id = getattr(env.base_ui, "ID", "unknown")
            total_in = self._task_planning_in + self._task_pab_in
            total_out = self._task_planning_out + self._task_pab_out
            with open(env.base_ui.token_count_file, "a") as f:
                f.write(
                    f"P_LLM_TASK_SUMMARY task={task_id} "
                    f"planning_in={self._task_planning_in} planning_out={self._task_planning_out} "
                    f"annotation_in={self._task_pab_in} annotation_out={self._task_pab_out} "
                    f"total_plm_in={total_in} total_plm_out={total_out}\n"
                )
            print(
                f"✅ P-LLM task summary [{task_id}]: "
                f"planning={self._task_planning_in + self._task_planning_out} tok, "
                f"annotation={self._task_pab_in + self._task_pab_out} tok, "
                f"total={total_in + total_out} tok"
            )

        return query, runtime, env, messages, extra_args
