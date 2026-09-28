"""Deterministic AST extraction of the plan branch skeleton (the "C" half).

The skeleton is the structural source of truth for `plan_constraints.json`:
branch keys, nesting, trigger variables and statically extractable facts
(literal domains, called functions, comparison literals) are computed here
with `ast` — never by an LLM — so they are guaranteed to match what the
CaMeL interpreter sees at runtime in `_eval_if`.

Branch identifiers follow this convention:

    if_L{lineno}_true / if_L{lineno}_false

where ``lineno`` is the 1-based line number of the ``if`` statement in the
source extracted from the P-LLM markdown answer (the exact same source the
interpreter parses: see `extract_single_code_block`, which mirrors
`cobra.interpreter.interpreter.extract_code_block`). Nested arms are keyed
by dot-joined paths, e.g. ``if_L10_true.if_L18_false``. An ``elif`` is
represented by Python's AST as an ``if`` nested in the parent's ``orelse``,
so it naturally becomes ``if_L10_false.if_L12_true``.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from typing import Any

ROOT_KEY = "root"

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-#]*)\n(.*?)\n```", re.DOTALL)

# Heuristic domain detection for the static cross-check. False negatives
# are acceptable (the annotator adds domains anyway); false positives
# would force junk into allowed_domains, hence the extension blocklist.
_DOMAIN_RE = re.compile(r"^(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}$")
_URL_RE = re.compile(r"https?://([^/\s:?#'\"]+)")
# Characters a hostname can contain. Applied to `_URL_RE` captures only, which are
# otherwise unvalidated: the capture stops at "/:?#" and quotes, so a URL written
# inside prose or built with `.format()` yields a token that is NOT a hostname.
# See `_clean_url_host` for why this matters more than it looks.
_HOSTNAME_CHARS_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
_NON_DOMAIN_SUFFIXES = {
    "json", "txt", "py", "html", "htm", "png", "jpg", "jpeg", "gif", "svg",
    "csv", "pdf", "md", "xml", "yaml", "yml", "js", "css", "sh", "zip",
    "tar", "gz", "log", "ini", "cfg", "conf", "db", "sqlite", "doc",
    "docx", "xls", "xlsx", "ppt", "pptx", "mp3", "mp4", "wav", "exe",
}

_OP_MAP: dict[type, str] = {
    ast.LtE: "<=",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.Lt: "<",
    ast.Gt: ">",
    ast.NotEq: "!=",
}

_FLIP = {"<=": ">=", ">=": "<=", "<": ">", ">": "<", "==": "==", "!=": "!="}


class InvalidPlanError(Exception):
    """The P-LLM output does not contain exactly one parseable code block."""


def extract_single_code_block(markdown_text: str) -> str:
    """Extracts the plan source from the P-LLM markdown answer.

    Mirrors `cobra.interpreter.interpreter.extract_code_block` (same regex,
    same strip, same "exactly one block" rule) so that line numbers in the
    skeleton always match the source the interpreter executes. Kept local
    to avoid importing the interpreter (and its heavy dependency chain)
    from this standalone module.
    """
    blocks = [match[1].strip() for match in _FENCE_RE.findall(markdown_text)]
    if len(blocks) != 1:
        raise InvalidPlanError(
            f"Expected exactly one fenced code block in the plan, found {len(blocks)}."
        )
    return blocks[0]


@dataclasses.dataclass
class BranchArm:
    """One arm (true or false) of an `if` statement in the plan."""

    key: str
    branch_id: str
    branch_path: list[str]
    lineno: int
    is_true_arm: bool
    condition: str
    trigger_var: str | None
    condition_comparison: dict[str, Any] | None
    has_body: bool
    called_functions: list[str] = dataclasses.field(default_factory=list)
    static_domains: list[str] = dataclasses.field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Compact dict used in the annotator prompt."""
        return {
            "key": self.key,
            "branch_path": self.branch_path,
            "condition": self.condition,
            "condition_is_true": self.is_true_arm,
            "trigger_var": self.trigger_var,
            "condition_comparison": self.condition_comparison,
            "has_body": self.has_body,
            "called_functions": self.called_functions,
            "static_domains": self.static_domains,
        }


@dataclasses.dataclass
class PlanSkeleton:
    source: str
    arms: dict[str, BranchArm]
    root_called_functions: list[str] = dataclasses.field(default_factory=list)
    root_static_domains: list[str] = dataclasses.field(default_factory=list)
    # Names the plan assigns to (assignment / for / with targets). Used by the
    # validator to reject a "var:<name>" field placeholder that references a
    # variable the plan never defines.
    variable_names: set[str] = dataclasses.field(default_factory=set)
    # Which wire field each plan variable was READ from, as the string literals passed
    # to the call that assigned it: `amount_0 = read_item_number(0, "amount")` records
    # `amount_0 -> ("amount",)`.
    #
    # This is what makes a variable a PERCEIVED VALUE of a field, and without it the
    # validator's under-pin guard can only recognise the case where the plan happens to
    # name its variable exactly like the wire field. Any plan that indexes per row —
    # `amount_0`, `exposure_1` — was therefore unguarded, which is how a four-level tree
    # breached with all four levels "pinned" to policy constants.
    #
    # The literals are recorded raw and unranked: only the caller holds the wire schema,
    # so only the caller can say which of them names a field. The skeleton stays a pure
    # description of the plan.
    var_field_reads: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)

    def all_keys(self) -> set[str]:
        return {ROOT_KEY, *self.arms.keys()}

    def summary(self) -> dict[str, Any]:
        return {
            "root": {
                "key": ROOT_KEY,
                "called_functions": self.root_called_functions,
                "static_domains": self.root_static_domains,
            },
            "branches": [arm.summary() for arm in self.arms.values()],
            "variable_names": sorted(self.variable_names),
            "var_field_reads": {k: list(v) for k, v in sorted(self.var_field_reads.items())},
        }

    def numbered_source(self) -> str:
        return "\n".join(
            f"{i:>4} | {line}" for i, line in enumerate(self.source.splitlines(), start=1)
        )


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _looks_like_domain(token: str) -> bool:
    token = token.strip().strip(".,;:!?")
    if not _DOMAIN_RE.match(token):
        return False
    suffix = token.rsplit(".", 1)[-1].lower()
    return suffix not in _NON_DOMAIN_SUFFIXES


def _clean_url_host(host: str) -> str | None:
    """Normalise a `_URL_RE` capture, or drop it if it cannot be a hostname.

    The capture is unvalidated: it takes everything up to the first "/:?#" or quote,
    so two shapes that a P-LLM writes routinely produce a token no hostname check can
    ever accept, and both were observed in production —

      * a URL ending a sentence inside a string literal → ``scholar.google.com.``
        (also ``dblp.org.``, ``www.speedtest.net.``, ``paybatch.local.``)
      * a `.format()` template → ``procurement.local{0}``

    Left in, such a token DEADLOCKS annotation: `validator._check_root_static_domains`
    demands it appear in ``allowed_domains`` while `_check_domains` rejects it as not a
    bare hostname, so no annotation can satisfy both. The three retries burn, and
    `build_fallback` writes a domain-only state carrying the unmatchable token —
    blocking the plan's own honest traffic while silently dropping the field and
    endpoint layers.

    The fix belongs here rather than in the validator: relaxing the hostname check
    would admit the junk into the enforced allowlist, where matching is exact-host, so
    it would still never match — the false positive would just move downstream. This
    function only ever normalises a token the validator would reject anyway, so it
    cannot remove a domain that was carrying a working constraint. A trailing dot is
    stripped rather than dropped (it is the FQDN root form of a real domain, and the
    cross-check should still force the annotator to authorise it); a token with
    characters no hostname can hold has nothing to recover and is dropped.

    IP literals are deliberately still accepted (WASP plans are full of
    ``127.0.0.1``) — this is a character check, not `_looks_like_domain`. The character
    set is a strict superset of `validator._HOSTNAME_RE`'s, so nothing droppable here
    was annotatable there. That comment block states this very principle for
    loopback/IPv4/single-label hosts; this closes the extraction end of it.
    """
    host = host.strip().rstrip(".")
    if not host or not _HOSTNAME_CHARS_RE.match(host):
        return None
    return host.lower()


def _domains_in_string(text: str) -> list[str]:
    found = [h for h in (_clean_url_host(m.group(1)) for m in _URL_RE.finditer(text))
             if h]
    for token in re.split(r"[\s'\"<>()\[\]{},]+", text):
        token = token.strip().strip(".,;:!?")
        if token and _looks_like_domain(token):
            found.append(token.lower())
    return _dedup(found)


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    return ast.unparse(func)


def _collect_facts(node: ast.AST, calls: list[str], domains: list[str]) -> None:
    """Collects called functions and literal domains from an AST subtree."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            calls.append(_call_name(sub))
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            domains.extend(_domains_in_string(sub.value))


def _root_variable(node: ast.AST) -> str | None:
    """Root variable name a condition depends on, or None for direct calls."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        elif isinstance(node, ast.UnaryOp):
            node = node.operand
        elif isinstance(node, ast.BoolOp):
            node = node.values[0]
        elif isinstance(node, ast.Compare):
            node = node.left
        else:
            return None


def _comparison_info(test: ast.expr) -> dict[str, Any] | None:
    """Extracts `{op, literal, operand}` from a simple `x <op> const` test."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    op = _OP_MAP.get(type(test.ops[0]))
    if op is None:
        return None
    left, right = test.left, test.comparators[0]
    if isinstance(right, ast.Constant) and not isinstance(left, ast.Constant):
        operand, literal = left, right.value
    elif isinstance(left, ast.Constant) and not isinstance(right, ast.Constant):
        operand, literal = right, left.value
        op = _FLIP[op]
    else:
        return None
    if not isinstance(literal, (int, float, str, bool)):
        return None
    return {"op": op, "literal": literal, "operand": ast.unparse(operand)}


def _walk_stmts(
    stmts: list[ast.stmt],
    branch_path: list[str],
    arms: dict[str, BranchArm],
    sink_calls: list[str],
    sink_domains: list[str],
) -> None:
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            # Calls/literals in the test run *before* the branch is taken,
            # so their facts belong to the enclosing arm.
            _collect_facts(stmt.test, sink_calls, sink_domains)

            key_prefix = [p for p in branch_path if p != ROOT_KEY]
            for suffix, body in (("true", stmt.body), ("false", stmt.orelse)):
                branch_id = f"if_L{stmt.lineno}_{suffix}"
                key = ".".join([*key_prefix, branch_id])
                arm = BranchArm(
                    key=key,
                    branch_id=branch_id,
                    branch_path=[*branch_path, branch_id],
                    lineno=stmt.lineno,
                    is_true_arm=suffix == "true",
                    condition=ast.unparse(stmt.test),
                    trigger_var=_root_variable(stmt.test),
                    condition_comparison=_comparison_info(stmt.test),
                    has_body=bool(body),
                )
                arms[key] = arm
                _walk_stmts(
                    body, arm.branch_path, arms, arm.called_functions, arm.static_domains
                )
        elif isinstance(stmt, (ast.For, ast.While, ast.With, ast.Try)):
            # Loop/with/try bodies execute within the current arm: any `if`
            # inside them still gets its own branch key (unique by lineno),
            # while other facts accumulate on the enclosing arm.
            for header in _stmt_headers(stmt):
                _collect_facts(header, sink_calls, sink_domains)
            for inner in _stmt_bodies(stmt):
                _walk_stmts(inner, branch_path, arms, sink_calls, sink_domains)
        else:
            _collect_facts(stmt, sink_calls, sink_domains)


def _stmt_headers(stmt: ast.stmt) -> list[ast.AST]:
    if isinstance(stmt, ast.For):
        return [stmt.target, stmt.iter]
    if isinstance(stmt, ast.While):
        return [stmt.test]
    if isinstance(stmt, ast.With):
        return list(stmt.items)
    return []


def _stmt_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for attr in ("body", "orelse", "finalbody"):
        block = getattr(stmt, attr, None)
        if block:
            bodies.append(block)
    for handler in getattr(stmt, "handlers", []):
        bodies.append(handler.body)
    return bodies


def _field_reads(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Plan variable -> the string literals of the call that assigned it.

    Deliberately shallow: one Name target, one Call value, its literal string arguments
    (positional and keyword alike). No attempt to say which literal is a field name —
    that needs the wire schema, which lives with the caller.

    A variable assigned more than once keeps the literals of every assignment, because
    the guard only ever asks whether a given field is among them."""
    reads: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        lits = tuple(
            a.value for a in [*call.args, *(k.value for k in call.keywords)]
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        )
        if lits:
            reads[target.id] = tuple(dict.fromkeys(reads.get(target.id, ()) + lits))
    return reads


def extract_skeleton(plan: str, *, is_markdown: bool = True) -> PlanSkeleton:
    """Builds the deterministic branch skeleton of a plan.

    Args:
        plan: the P-LLM answer (markdown with one fenced code block) or,
            with ``is_markdown=False``, the already-extracted source.

    Raises:
        InvalidPlanError: no single code block, or the source does not parse.
    """
    source = extract_single_code_block(plan) if is_markdown else plan
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise InvalidPlanError(f"Plan source does not parse: {e}") from e

    skeleton = PlanSkeleton(source=source, arms={})
    skeleton.variable_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    skeleton.var_field_reads = _field_reads(tree)
    _walk_stmts(
        tree.body,
        [ROOT_KEY],
        skeleton.arms,
        skeleton.root_called_functions,
        skeleton.root_static_domains,
    )
    skeleton.root_called_functions = _dedup(skeleton.root_called_functions)
    skeleton.root_static_domains = _dedup(skeleton.root_static_domains)
    for arm in skeleton.arms.values():
        arm.called_functions = _dedup(arm.called_functions)
        arm.static_domains = _dedup(arm.static_domains)
    return skeleton
