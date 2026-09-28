"""Deterministic pre-flight lint for fused single-attempt plans (BRH_PLAN_FUSION).

Stdlib-only (``ast`` + ``re``) on purpose: this module lives next to the OSWorld
suites — NOT in the shared ``cobra/brh`` package — so it has zero blast radius on
other benchmarks and is importable without any heavy dependency.

Checks (fused planning, Appendix C of the paper):

  L1 (hard)   — constructs the CaMeL interpreter cannot run: def / lambda / while /
                try / class / import / yield / generator expressions.
  L2 (repair) — every ``call_mcp_tool`` result must be captured in a variable that is
                later tested (directly or through one derived assignment), and at
                least one GUI action must appear after the last ``call_mcp_tool``
                (the in-plan MCP→GUI fallback arm).
  L3 (repair) — ``mark_done()`` must be preceded (source order) by at least one
                ``check_done``/``verify_hypothesis`` call; ``mark_fail()`` must be
                the last statement of its enclosing block.
  L3b (repair)— ``mark_done()`` must be gated on the LAST verification, never on an
                aggregate of phase-success flags (``a or b or c``). Attacks failure
                class C1 (false completion).
  L4          — the code must parse (hard) and stay within budget: ≤120 statements,
                ≤4 ``call_mcp_tool`` calls (repair).
  L5 (repair) — GUI phases must not be guarded by a feasibility/MCP *gate flag* (a
                boolean set before any GUI action and not derived from a verification
                or an MCP result, e.g. ``abort = True``). Attacks failure class C2
                (feasibility/abort cascade).

The lint never blocks a run: callers repair (≤2 rounds) and, if hard violations
remain, fall back to the best single candidate (``pick_fallback``).
"""

from __future__ import annotations

import ast
import dataclasses
import re

# Constructs that crash the CaMeL interpreter (prompt patch-A class).
_FORBIDDEN_NODES: dict[type, str] = {
    ast.FunctionDef: "def",
    ast.AsyncFunctionDef: "async def",
    ast.Lambda: "lambda",
    ast.While: "while",
    ast.Try: "try/except",
    ast.ClassDef: "class",
    ast.Import: "import",
    ast.ImportFrom: "import",
    ast.GeneratorExp: "generator expression",
    ast.Yield: "yield",
    ast.YieldFrom: "yield from",
    ast.AsyncFor: "async for",
    ast.AsyncWith: "async with",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
}

# GUI action primitives (TOOLS_0 subset) that count as a GUI fallback arm.
GUI_ACTION_TOOLS = frozenset({
    "locate_and_click", "click", "left_single", "left_double", "right_single",
    "type_text", "hotkey", "press", "keydown", "keyup", "drag", "select",
    "scroll", "run_single_uitars", "find", "find_element_by_text", "hover",
})

VERIFY_TOOLS = frozenset({"check_done", "verify_hypothesis"})

MAX_STATEMENTS = 120
MAX_MCP_CALLS = 4

# Fallback priority when the fused plan is unusable.
FALLBACK_PRIORITY = (
    "gui_direct", "mcp_first", "feasibility_skeptic", "gui_menu", "verification_heavy",
)

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclasses.dataclass
class LintResult:
    hard: list[str]
    repair: list[str]

    @property
    def ok(self) -> bool:
        return not self.hard and not self.repair

    @property
    def executable(self) -> bool:
        """Hard-clean: safe to hand to the interpreter even with soft findings."""
        return not self.hard


def extract_code(text: str) -> str:
    """Strip a markdown code fence if present (mirrors interpreter.extract_code_block
    leniently — falls back to the raw text so the lint sees what run_code would)."""
    m = _FENCE_RE.search(text or "")
    return m.group(1) if m else (text or "")


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _calls_in(node: ast.AST) -> set[str]:
    names = (_call_name(c) for c in ast.walk(node) if isinstance(c, ast.Call))
    return {n for n in names if n}


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _assigns_by_name(tree: ast.AST) -> dict[str, list[tuple[int, ast.expr]]]:
    """name -> [(lineno, assigned value expression), ...] for simple Name targets."""
    out: dict[str, list[tuple[int, ast.expr]]] = {}
    for a in ast.walk(tree):
        if not isinstance(a, ast.Assign):
            continue
        for t in a.targets:
            if isinstance(t, ast.Name):
                out.setdefault(t.id, []).append((a.lineno, a.value))
    return out


def _effect_derived(assigns: dict[str, list[tuple[int, ast.expr]]]) -> set[str]:
    """Names whose value (transitively) comes from a verification or an MCP result.

    These carry real evidence about the world, so they are legitimate cross-phase
    carriers and are exempt from the L5 gate-flag rule.
    """
    sources = VERIFY_TOOLS | {"call_mcp_tool"}
    derived: set[str] = set()
    changed = True
    while changed:                     # fixpoint; plans are tiny, this converges fast
        changed = False
        for name, entries in assigns.items():
            if name in derived:
                continue
            for _, value in entries:
                if (_calls_in(value) & sources) or (_names_in(value) & derived):
                    derived.add(name)
                    changed = True
                    break
    return derived


def _guarding_if(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.If | None:
    """Nearest enclosing ``if`` that *guards* ``node`` (node sits in body/orelse)."""
    cur = node
    while cur in parents:
        parent = parents[cur]
        if isinstance(parent, ast.If) and cur is not parent.test:
            return parent
        cur = parent
    return None


def _has_gui_action(node: ast.AST) -> bool:
    return any(
        isinstance(c, ast.Call) and _call_name(c) in GUI_ACTION_TOOLS
        for stmt in getattr(node, "body", []) + getattr(node, "orelse", [])
        for c in ast.walk(stmt)
    )


def lint_plan(code: str) -> LintResult:
    hard: list[str] = []
    repair: list[str] = []
    src = extract_code(code)

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return LintResult(hard=[f"L4-parse: the plan is not valid Python: {e}"], repair=[])

    # ---- L1: forbidden constructs -------------------------------------------------
    for node in ast.walk(tree):
        label = _FORBIDDEN_NODES.get(type(node))
        if label is not None:
            hard.append(
                f"L1: line {getattr(node, 'lineno', '?')}: `{label}` is not supported by the "
                f"plan interpreter — inline the logic using only sequential statements and "
                f"if/elif/else."
            )

    # ---- L4: budget ---------------------------------------------------------------
    n_statements = sum(isinstance(n, ast.stmt) for n in ast.walk(tree))
    if n_statements > MAX_STATEMENTS:
        repair.append(
            f"L4: the plan has {n_statements} statements (budget {MAX_STATEMENTS}) — remove "
            f"redundant steps; keep one primary and one alternate path per phase."
        )

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    mcp_calls = [c for c in calls if _call_name(c) == "call_mcp_tool"]
    if len(mcp_calls) > MAX_MCP_CALLS:
        repair.append(
            f"L4: {len(mcp_calls)} call_mcp_tool calls (budget {MAX_MCP_CALLS}) — keep only "
            f"the calls whose tool COMPLETES a step of the task."
        )

    # ---- L2: MCP fallback contract ------------------------------------------------
    if mcp_calls:
        # (a) every call_mcp_tool result must be captured by an assignment.
        captured: set[str] = set()
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        for a in assigns:
            if any(isinstance(c, ast.Call) and _call_name(c) == "call_mcp_tool"
                   for c in ast.walk(a.value)):
                captured |= {t.id for t in a.targets if isinstance(t, ast.Name)}
        for stmt in ast.walk(tree):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                    and _call_name(stmt.value) == "call_mcp_tool":
                repair.append(
                    f"L2: line {stmt.lineno}: call_mcp_tool result is discarded — assign it "
                    f"to a variable and test it for the 'mcp_tool_error' prefix."
                )
        # (b) each captured name must be tested — directly in an `if` test, or through
        # one derived assignment (e.g. mcp_failed = res.startswith(...)).
        tested: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.If):
                tested |= _names_in(n.test)
        derived_ok: set[str] = set()
        for a in assigns:
            targets = {t.id for t in a.targets if isinstance(t, ast.Name)}
            if targets & tested:
                derived_ok |= _names_in(a.value)
        for name in sorted(captured):
            if name not in tested and name not in derived_ok:
                repair.append(
                    f"L2: MCP result variable `{name}` is never tested — branch on "
                    f"`{name}.startswith('mcp_tool_error')` (or a derived flag) with a GUI "
                    f"fallback arm."
                )
        # (c) a GUI action must exist after the last call_mcp_tool (the fallback arm).
        last_mcp_line = max(c.lineno for c in mcp_calls)
        has_gui_after = any(
            _call_name(c) in GUI_ACTION_TOOLS and c.lineno > last_mcp_line for c in calls
        )
        if not has_gui_after:
            repair.append(
                "L2: no GUI action appears after the last call_mcp_tool — the plan must "
                "contain a GUI fallback path (PHASE G1) reachable when MCP fails or its "
                "effect does not verify."
            )

    # ---- L3: verification discipline ---------------------------------------------
    done_calls = [c for c in calls if _call_name(c) == "mark_done"]
    if done_calls:
        first_done = min(c.lineno for c in done_calls)
        has_verify_before = any(
            _call_name(c) in VERIFY_TOOLS and c.lineno < first_done for c in calls
        )
        if not has_verify_before:
            repair.append(
                f"L3: mark_done() at line {first_done} has no preceding check_done()/"
                f"verify_hypothesis() — verify the task's specific end-state before "
                f"declaring completion."
            )

    for body in _iter_bodies(tree):
        for idx, stmt in enumerate(body):
            if _stmt_calls(stmt, "mark_fail") and idx != len(body) - 1:
                repair.append(
                    f"L3: line {stmt.lineno}: statements follow mark_fail() in the same "
                    f"branch — mark_fail() must be the LAST statement of its path."
                )

    # ---- L3b: the done-verdict must BE the fresh verification (fix F1b) -----------
    assigns = _assigns_by_name(tree)
    parents = _parent_map(tree)

    for done in done_calls:
        # The last verification-derived assignment before this mark_done().
        last_line, last_verify = -1, set()
        for name, entries in assigns.items():
            for lineno, value in entries:
                if lineno < done.lineno and (_calls_in(value) & VERIFY_TOOLS):
                    if lineno > last_line:
                        last_line, last_verify = lineno, {name}
                    elif lineno == last_line:
                        last_verify.add(name)
        if not last_verify:
            continue                    # L3 already reports the missing verification

        guard = _guarding_if(done, parents)
        if guard is None:
            repair.append(
                f"L3b: line {done.lineno}: mark_done() is unconditional — gate it on a FINAL "
                f"check_done() of the task's end-state, evaluated immediately before it."
            )
            continue

        test_names = _names_in(guard.test)

        # (a) a disjunction of phase flags is never a completion verdict.
        or_expr: ast.BoolOp | None = None
        if isinstance(guard.test, ast.BoolOp) and isinstance(guard.test.op, ast.Or):
            or_expr = guard.test
        else:
            for name in sorted(test_names):     # one level of derived assignment
                entries = assigns.get(name, [])
                if len(entries) == 1 and isinstance(entries[0][1], ast.BoolOp) \
                        and isinstance(entries[0][1].op, ast.Or):
                    or_expr = entries[0][1]
                    break
        if or_expr is not None and len(_names_in(or_expr)) >= 2:
            repair.append(
                f"L3b: line {done.lineno}: mark_done() is gated on a disjunction of phase "
                f"flags (`a or b or c`) — stale flags from an earlier phase (e.g. an "
                f"opener-tool 'success') then declare the task complete. Run a FRESH "
                f"check_done() of the end-state immediately before mark_done() and gate on "
                f"that single result."
            )
            continue

        # (b) the guard must reference the last verification (or derive from it).
        if test_names & last_verify:
            continue
        derived_ok = any(
            (_names_in(value) & last_verify)
            for name in test_names
            for lineno, value in assigns.get(name, [])
            if lineno < done.lineno
        )
        if not derived_ok:
            stale = ", ".join(sorted(test_names)) or "<no name>"
            repair.append(
                f"L3b: line {done.lineno}: mark_done() is gated on `{stale}`, not on the "
                f"most recent verification (`{', '.join(sorted(last_verify))}` at line "
                f"{last_line}) — re-verify the end-state immediately before mark_done() and "
                f"gate on that result."
            )

    # ---- L5: GUI phases must not inherit a feasibility/MCP gate flag (fix F2b) ----
    gui_calls = [c for c in calls if _call_name(c) in GUI_ACTION_TOOLS]
    if gui_calls:
        first_gui = min(c.lineno for c in gui_calls)
        evidence = _effect_derived(assigns)
        gate_flags = {
            name
            for name, entries in assigns.items()
            if name not in evidence
            and all(lineno < first_gui for lineno, _ in entries)
            and any(isinstance(v, ast.Constant) and isinstance(v.value, bool)
                    for _, v in entries)
        }
        flagged: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            hit = (_names_in(node.test) & gate_flags) - flagged
            if hit and _has_gui_action(node):
                flagged |= hit
                repair.append(
                    f"L5: line {node.lineno}: a GUI phase is guarded by the gate flag "
                    f"`{', '.join(sorted(hit))}`, set before any GUI action and not derived "
                    f"from a verification or an MCP result. The feasibility gate must either "
                    f"call mark_fail() outright or fall through leaving NO state — remove the "
                    f"flag from this condition so the GUI phases always get their turn."
                )

    return LintResult(hard=hard, repair=repair)


def _iter_bodies(tree: ast.AST):
    """Yield every statement list (module body, if/else bodies, for bodies, …)."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                yield body


def _stmt_calls(stmt: ast.stmt, func_name: str) -> bool:
    return any(
        isinstance(n, ast.Call) and _call_name(n) == func_name for n in ast.walk(stmt)
    )


def format_repair_message(result: LintResult) -> str:
    """One user message listing the violations for the ≤2-round repair loop."""
    lines = result.hard + result.repair
    bullet = "\n".join(f"- {v}" for v in lines)
    return (
        "Your assembled plan violates the mandatory plan contract. Fix ONLY the "
        "violations listed below and change nothing else; then output the complete "
        "corrected plan (all the code).\n" + bullet
    )


def pick_fallback(candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Best single candidate to run unfused when the fused plan stays hard-broken.

    ``candidates`` is ``[(strategy_name, code), ...]``. Returns the first candidate in
    FALLBACK_PRIORITY order whose code is executable (parses, no L1); as a last resort
    the first candidate that merely parses; else None.
    """
    by_name = dict(candidates)
    ordered = [n for n in FALLBACK_PRIORITY if n in by_name]
    ordered += [n for n, _ in candidates if n not in ordered]
    for name in ordered:
        if lint_plan(by_name[name]).executable:
            return name, by_name[name]
    for name in ordered:
        try:
            ast.parse(extract_code(by_name[name]))
            return name, by_name[name]
        except SyntaxError:
            continue
    return None
