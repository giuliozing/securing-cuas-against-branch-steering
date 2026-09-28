"""Semantic annotation of the plan skeleton via an LLM call (the "B" half).

The annotator receives the user task, the line-numbered plan source and
the deterministic skeleton, and fills in only the semantic slots
(allowed domains, HTTP field constraints, descriptions). Its output is
parsed with pydantic and checked by `validator.validate`; on failure the
errors are fed back and the call is retried with the same fixed plan.

The LLM is injected as a plain ``llm_call(system_prompt, user_prompt) ->
str`` callable so this module stays independent from the agentdojo
pipeline (and trivially fakeable in tests).
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol

import pydantic

from cobra.brh import validator
from cobra.brh.schema import PlanConstraints
from cobra.brh.skeleton import PlanSkeleton
from cobra.brh.validator import HttpManifest, McpManifest

LLMCall = Callable[[str, str], str]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


class AnnotationError(Exception):
    """The annotator could not produce a valid annotation within retries."""

    def __init__(self, message: str, last_errors: list[str] | None = None):
        super().__init__(message)
        self.last_errors = last_errors or []


_SYSTEM_PROMPT = """\
You are a security annotator for an AI agent execution plan. The plan is a \
Python program whose conditional branches have already been enumerated \
deterministically (the "skeleton"). Your only job is to fill in, for every \
branch key in the skeleton, the constraints that the network-layer enforcer \
will apply while that branch is active.

Output format: exactly one fenced ```json code block containing an object:

{
  "branches": {
    "<branch key>": {
      "description": "<short human-readable purpose of the branch>",
      "trigger_var": "<the skeleton's trigger_var for this branch, or null>",
      "http_constraints": {
        "allowed_domains": ["<bare lowercase hostnames>"],
        "fields": [
          {"path": "<field path in the HTTP request body>", "op": "<=|>=|==", "value": <literal or placeholder>},
          {"path": "<field path>", "op": "in", "value": [<two or more literals of the same type>]}
        ],
        "allowed_endpoints": [
          {"method": "<METHOD>", "domain": "<bare hostname>", "path_pattern": "<path from manifest>"}
        ]
      },
      "mcp_constraints": null
      // ...or, when an MCP tool manifest is given (see the user prompt):
      // "mcp_constraints": {
      //   "allowed_tools": ["<MCP tool names this branch may call>"],
      //   "param_rules": [
      //     {"tool": "<tool>", "param": "<param>", "op": "<=|>=|==", "value": <literal, "trigger_value" or "var:<name>">},
      //     {"tool": "<tool>", "param": "<param>", "source": "from_plan", "value": <literal>}
      //   ]
      // }
    },
    ...
  }
}

Strict rules:
1. Use EXACTLY the branch keys listed in the skeleton, plus "root". Do not \
invent keys, do not omit any key.
2. allowed_domains is the complete list of domains the agent's HTTP traffic \
may legitimately reach while that branch is active (page loads, APIs, \
payments...). It MUST include every domain listed in the skeleton's \
static_domains for that branch. Domains are bare lowercase hostnames: no \
scheme, no path, no port.
3. fields express limits on values the agent will send (e.g. a checkout \
amount). Use the placeholder "trigger_value" when the limit must equal the \
runtime value of the branch's trigger variable (e.g. the perceived price \
that made the condition true). Use "from_plan" when the value becomes known \
during execution from data the plan itself retrieved (e.g. a product id). \
Use concrete literals when the plan fixes them (e.g. a currency).
3a. PLACEMENT of a "trigger_value" field is critical: "trigger_value" always \
resolves to the trigger variable of the branch that DECLARES the field. So a \
limit must be declared on the branch whose trigger variable IS the bounding \
value. Example: "the charged amount must not exceed the displayed price" \
belongs on the branch that tested the price (e.g. `if price <= 50`, \
trigger_var "price") — NOT on a deeper branch that tests a different variable \
(stock, shipping fee, membership tier), where "trigger_value" would wrongly \
resolve to that other value and reject legitimate requests. Because \
constraints accumulate from "root" down to the active leaf, a field placed on \
an ancestor branch still applies inside its nested descendants — so place the \
bound once, on the branch whose condition reads the bounding value, even if \
the action that sends the field happens several levels deeper. Worked example: \
for a plan `if price <= 50: if stock >= 1: if tier == "gold": place_order()`, \
the amount bound goes on branch "if_L?_true" with trigger_var "price" as \
{"path": "amount", "op": "<=", "value": "trigger_value"}; the nested "stock" \
and "tier" branches leave "amount" unconstrained — the limit still reaches the \
order through accumulation, resolved to the price.
3b. When the bounding value is a plan variable that is NOT the trigger of any \
branch on the path to the action (e.g. the order happens inside `if tier == \
"gold":` but the cap is the `price` read earlier), you cannot use \
"trigger_value" — it would resolve to the wrong variable. Instead reference the \
variable by name with the prefix "var:", e.g. {"path": "amount", "op": "<=", \
"value": "var:price"}. The name after "var:" must be EXACTLY a variable the \
plan assigns (it is resolved from that variable's runtime value). Prefer \
"trigger_value" when the bound IS the branch's trigger; use "var:<name>" only \
for another in-scope plan variable. "var:" is for single comparison ops, never \
inside an "in" set.
4. Use "in" when a field may take one of a few known values (e.g. \
{"path": "currency", "op": "in", "value": ["GBP", "EUR"]}). The value must \
be a non-empty JSON array of concrete literals, all of the same type. \
Placeholders are NOT allowed inside the array — if the limit depends on \
the trigger variable, use a comparison op with "trigger_value" instead.
5. Constraints on the same path are combined with AND. Express a numeric \
range with two entries on the same path, e.g. {"op": ">=", "value": 10} \
and {"op": "<=", "value": 50}. There is no dedicated range operator.
6. Only add fields you can justify from the task and the plan. An empty \
list is better than an invented constraint.
7. Branches with "has_body": false authorise nothing: empty allowed_domains, \
empty fields, mcp_constraints null.
8. "root" covers traffic outside any branch (initial navigation, \
observation). Its allowed_domains must include the skeleton's root \
static_domains.
9. mcp_constraints: if (and only if) the user prompt includes an "MCP tools \
available" manifest, fill mcp_constraints for branches that call those tools; \
otherwise mcp_constraints is always null. For a branch, set "allowed_tools" to \
the manifest tools the agent legitimately calls while that branch is active — \
it MUST include every manifest tool the branch's body calls (a tool the branch \
calls but does not authorise would be blocked). Tools NOT in the manifest must \
never appear.
9a. "param_rules" constrain the arguments of an allowed tool, using the SAME \
value language as http fields: comparison ops ("<=", ">=", "=="), membership \
("in", with a non-empty list of concrete literals), and the placeholders \
"trigger_value" / "var:<name>" / "from_plan". When the policy allows one of \
SEVERAL values, you MUST use a single "in" rule: rules on the same parameter are \
combined with AND, so two "==" rules on one parameter authorise NOTHING and the \
honest call is refused. Every rule's "tool" must \
be in that branch's allowed_tools and its "param" must be a parameter the tool \
declares in the manifest. "trigger_value" placement follows rules 3a/3b exactly \
(it resolves to the trigger of the branch that declares the rule); use \
"var:<name>" for an in-scope plan variable that is not the branch trigger. For a \
value the PLAN itself fixes (e.g. a product id), write the LITERAL — you wrote the \
plan, so you know it. Do NOT put "from_plan" in "value": unlike the other two it \
resolves from nothing at runtime, so the enforcer would compare the real argument \
against the string "from_plan" and refuse the honest call. The provenance form \
{"source": "from_plan", "value": <literal>} is available when you want to record \
that the pin came from the plan.
9b. Only constrain tools the plan actually uses; a branch with no MCP calls \
keeps mcp_constraints null.
10. Be minimal: a domain, field or tool you do not list is blocked by default \
(fail-closed). List what the plan needs — nothing more.
11. When an HTTP endpoint manifest is given in the user prompt (see below), \
field constraints MUST use ONLY path names listed in the endpoint's \
"body_fields". Do not invent or guess field names not in the manifest. \
If an endpoint has no "body_fields", do not add field constraints for it — \
the wire schema is unknown and any field name you write would be a guess. \
"allowed_domains" must include the domains of all manifest endpoints the plan \
visits; a domain clearly named in the task may also appear even if not in the \
manifest. A field path not in the manifest is always a validation error.
12. When an HTTP endpoint manifest is given, annotate "allowed_endpoints" for \
every branch that makes HTTP requests: list the endpoints (method + domain + \
path_pattern) from the manifest that the branch may legitimately call. Use \
the method and path_template values verbatim from the manifest entry — do not \
invent or paraphrase. Only include endpoints whose domain is in \
allowed_domains and whose structural purpose (inferred from method + path) \
matches what the branch does. An endpoint whose domain is in allowed_domains \
but is absent from allowed_endpoints will be blocked by the enforcer (HTTP \
analog of MCP's allowed_tools). If no manifest is given, omit \
allowed_endpoints (empty list).
"""


def _http_manifest_section(http_manifest: HttpManifest | None) -> str:
    """Render the sanitized sitemap as a compact JSON block for the annotator.

    Only structural fields survive sanitize_sitemap (method, domain,
    path_template, body_fields); free text is stripped before this function is
    called, so the block is injection-free. The one exception is
    ``description``: it is non-empty only for sitemaps that are human-approved
    and hash-pinned in the sitemap trust registry (cobra.brh.sitemap_trust) —
    vetted at approval time, hence trusted content.
    """
    if not http_manifest:
        return ""
    entries = []
    has_descriptions = False
    for ep in http_manifest:
        entry: dict = {"method": ep.method, "domain": ep.domain, "path_template": ep.path_template}
        if ep.body_fields:
            entry["body_fields"] = sorted(ep.body_fields)
        if ep.description:
            entry["description"] = ep.description
            has_descriptions = True
        entries.append(entry)
    provenance = (
        "descriptions present only for human-approved, hash-pinned sitemaps"
        if has_descriptions
        else "structural data only, free text stripped"
    )
    return f"""
# HTTP endpoint manifest (from agent sitemap — {provenance})
```json
{json.dumps(entries, indent=2)}
```
Use this manifest for two purposes (rules 11 and 12):
- `allowed_endpoints`: list every manifest endpoint (method + domain + path_pattern) \
the branch may legitimately call (use path_template as path_pattern verbatim). \
Only entries matching the branch's task purpose should appear. Enforcer blocks \
any call to a domain whose endpoints are not listed here.
- `fields[]`: use ONLY path names from `body_fields` of matching endpoints. \
Endpoints with no `body_fields` have unknown wire schema — do not add field \
constraints for them. A field path not in body_fields is a validation error.
`allowed_domains` must cover every domain in the manifest the plan visits.
"""


def _mcp_manifest_section(mcp_tools: McpManifest | None) -> str:
    if not mcp_tools:
        return ""
    manifest = {tool: list(params) for tool, params in mcp_tools.items()}
    return f"""
# MCP tools available (approved — only these may appear in allowed_tools / param_rules)
```json
{json.dumps(manifest, indent=2)}
```
These are the only MCP tools the agent may call. For each branch that calls one \
of them, list it in mcp_constraints.allowed_tools and add any param_rules per \
rules 9/9a. Tools not in this manifest must never be authorised.
"""


def _seal_params_section(mcp_tools: McpManifest | None) -> str:
    """The `allowed_params` clause — OPT-IN, and deliberately so.

    MCP proxy enforces schema-closed arguments (`allowed_params`: an argument NAME outside the
    set is refused) and `BranchConstraints` carries the field, but nothing ever told the
    annotator it existed: `grep allowed_params annotator.py` returned nothing, every
    branch_state shipped `allowed_params: {}`, and an injected EXTRA argument — the one
    deviation class param_rules cannot catch, since a rule only constrains a parameter
    that is present — rode along.

    Gated on `BRH_SEAL_PARAMS` because the capability is opt-in by design
    and because sealing is fail-CLOSED on argument names: a harness whose plans
    legitimately pass arguments the annotator did not enumerate would start seeing
    refusals. Unset — every other benchmark — the prompt is byte-identical to before,
    so this cannot regress a suite that never asked for it."""
    if not mcp_tools or not os.environ.get("BRH_SEAL_PARAMS"):
        return ""
    return """
11. SEAL THE ARGUMENT NAMES. For every tool you authorise, add an "allowed_params" \
entry: "allowed_params": {"<tool>": ["<param>", ...]}. An argument name outside that \
set is refused, which is what stops an extra argument being appended to an otherwise \
legitimate call. Read the names OFF THE PLAN'S OWN CALL SITE — the keyword arguments \
that appear in the code you were given — and nowhere else. The manifest lists every \
parameter the tool DECLARES, which is a superset: a parameter the tool declares but \
this plan never passes MUST be omitted, and including it defeats the whole point, \
because the injected argument is by definition one the tool accepts. If the plan calls \
`restore(snapshot=x)` on a tool declaring `snapshot` and `notify_webhook`, the correct \
entry is ["snapshot"].
"""


def build_user_prompt(
    task: str,
    skeleton: PlanSkeleton,
    plan_id: str,
    mcp_tools: McpManifest | None = None,
    http_manifest: HttpManifest | None = None,
) -> str:
    return f"""\
# Task given to the agent
{task}

# Plan (line-numbered source — line numbers match the skeleton's if_L<n> keys)
{skeleton.numbered_source()}

# Branch skeleton (deterministic — keys are final)
```json
{json.dumps(skeleton.summary(), indent=2)}
```
{_http_manifest_section(http_manifest)}\
{_mcp_manifest_section(mcp_tools)}{_seal_params_section(mcp_tools)}
Produce the constraints JSON for plan "{plan_id}" covering all keys: \
{sorted(skeleton.all_keys())}.
"""


def _extract_json(response: str) -> dict:
    matches = _JSON_FENCE_RE.findall(response)
    candidates = matches if matches else [response.strip()]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            last_error = e
    raise AnnotationError(f"No parseable JSON object in annotator response: {last_error}")


def annotate(
    llm_call: LLMCall,
    task: str,
    skeleton: PlanSkeleton,
    plan_id: str,
    max_retries: int = 3,
    mcp_tools: McpManifest | None = None,
    http_manifest: HttpManifest | None = None,
) -> PlanConstraints:
    """Runs the annotation loop and returns validated PlanConstraints.

    ``mcp_tools`` is the approved MCP tool manifest (tool -> param names); when
    given, the prompt invites and the validator checks ``mcp_constraints``.

    ``http_manifest`` is the sanitized agent sitemap (from ``sanitize_sitemap``);
    when given, the prompt shows endpoint schemas and the validator enforces that
    field paths come from declared body_fields (for endpoints that declare them).

    Raises:
        AnnotationError: after `max_retries` failed attempts.
    """
    user_prompt = build_user_prompt(task, skeleton, plan_id, mcp_tools, http_manifest)
    errors: list[str] = []
    last_parsed: PlanConstraints | None = None

    for _ in range(max_retries):
        response = llm_call(_SYSTEM_PROMPT, user_prompt)
        try:
            payload = _extract_json(response)
            # plan_id/task/generated_by are authoritative on our side: the
            # model only ever provides the branches.
            constraints = PlanConstraints(
                plan_id=plan_id,
                task=task,
                generated_by="p-llm",
                branches=payload.get("branches", payload),
            )
        except (AnnotationError, pydantic.ValidationError) as e:
            errors = [str(e)]
        else:
            errors = validator.validate(constraints, skeleton, mcp_tools, http_manifest)
            if not errors:
                return constraints
            last_parsed = constraints

        error_block = "\n".join(f"- {e}" for e in errors)
        user_prompt = build_user_prompt(task, skeleton, plan_id, mcp_tools, http_manifest) + f"""

Your previous answer was rejected with these errors:
{error_block}

Return the corrected, complete JSON (all branch keys, one fenced json block).
"""

    # Last resort before giving up: supply the bounds the PLAN ITSELF establishes.
    #
    # Raising here is not the safe default it looks like. The caller
    # (`writer.generate_plan_constraints`) degrades an AnnotationError to
    # `build_fallback`, which carries domains only — so a plan rejected for a
    # MISSING FIELD PIN ends up enforced with NO field pins at all, and a
    # same-domain field attack walks through.
    #
    # A gating bound is derivable from the AST, so when the annotator will not
    # restate it we derive it and keep everything it got right. Deliberately
    # narrow: only pins that `validator.missing_trigger_pin` identifies, and only
    # if the repaired annotation then validates CLEAN — any other outstanding
    # error still raises, exactly as before.
    if last_parsed is not None and http_manifest:
        repairs = validator.repair_trigger_pins(last_parsed, skeleton, http_manifest)
        if repairs:
            residual = validator.validate(last_parsed, skeleton, mcp_tools, http_manifest)
            if not residual:
                print(f"🔧 BRH: annotation repaired deterministically ({'; '.join(repairs)})")
                return last_parsed
            errors = residual

    raise AnnotationError(
        f"Annotation failed after {max_retries} attempts.", last_errors=errors
    )
