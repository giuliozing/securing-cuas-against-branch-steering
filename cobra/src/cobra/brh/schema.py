"""Pydantic models for the BRH `plan_constraints.json` contract.

This module defines the on-disk schema shared between the P-LLM side
(which writes the file before execution) and the enforcers (HTTP proxy,
MCP) which read the resolved `branch_state.json` derived from it.

The schema follows this format:

- branch keys are deterministic, AST-derived (e.g. ``if_L10_true``,
  nested arms joined with dots: ``if_L10_true.if_L18_false``);
- field/param values may be concrete literals or one of two
  placeholders resolved by the BRH hook at branch-entry time.
"""

from __future__ import annotations

from typing import Literal, Union

import pydantic

# Placeholder resolved by the BRH hook with the runtime value of the
# branch trigger variable (e.g. the price perceived by the Q-LLM).
TRIGGER_VALUE_PLACEHOLDER = "trigger_value"
# Placeholder for values that become known during execution from plan
# data (e.g. a product id read from the page earlier in the plan).
FROM_PLAN_PLACEHOLDER = "from_plan"

PLACEHOLDERS = (TRIGGER_VALUE_PLACEHOLDER, FROM_PLAN_PLACEHOLDER)

# Placeholder *prefix* binding a field to a named plan variable that is not
# (necessarily) the branch's trigger variable — e.g. "var:price" caps a
# field at the runtime value of the plan variable `price`. The BRH hook
# resolves it from the namespace at branch-entry time. This lets a field be
# bounded by a value the plan read but never used as a branch condition on
# the active path; like every
# placeholder it lives only in `plan_constraints.json` and is gone (resolved
# to a literal, or left unresolved → fail-closed) by the time the enforcers
# read `branch_state.json`.
VAR_PLACEHOLDER_PREFIX = "var:"


def var_placeholder_name(value: object) -> str | None:
    """Returns the variable name of a ``"var:<name>"`` placeholder, else None."""
    if isinstance(value, str) and value.startswith(VAR_PLACEHOLDER_PREFIX):
        return value[len(VAR_PLACEHOLDER_PREFIX) :] or None
    return None

# Comparison ops, shared with MCP param rules.
Op = Literal["<=", ">=", "=="]
# HTTP field constraints additionally support finite-set membership and the
# structural ops (``subset``/``eq_struct``) that the enforcer (brh_check /
# contract) and the STEER-bench oracle already evaluate for wire fields — the
# schema historically lagged behind them, so the annotator/pipeline could not
# emit a subset/eq_struct FIELD pin (only MCP params could). Aligning FieldOp
# with McpOp closes that gap: ``eq_struct`` pins a whole list/dict body value by
# recursive equality, ``subset`` requires the wire list ⊆ a planned/policy set.
FieldOp = Literal["<=", ">=", "==", "in", "subset", "eq_struct"]
# MCP param rules additionally support structural ops for non-scalar args
#: ``eq_struct`` pins a whole list/dict, ``subset`` blocks
# additions to a planned list (the value-steer / arg-add residuals).
McpOp = Literal["<=", ">=", "==", "in", "subset", "eq_struct"]

# Strict types: no silent coercion (JSON `true` must not become `1`,
# and "42" must stay a string) — the file is a security contract.
ConstraintValue = Union[pydantic.StrictInt, pydantic.StrictFloat, pydantic.StrictStr]


def _is_strict_scalar(value: object) -> bool:
    """A ``ConstraintValue`` at runtime: int/float/str but NOT bool (bool ⊂ int
    in Python; a JSON ``true`` must never masquerade as ``1`` in a constraint)."""
    return isinstance(value, (int, float, str)) and not isinstance(value, bool)


class FieldConstraint(pydantic.BaseModel):
    """Constraint on a single field of an HTTP request body/query.

    Constraints on the same ``path`` are conjunctive (AND): a numeric
    range is expressed with two entries (``>=`` and ``<=``) on the same
    path, not with a dedicated operator.

    ``op: "in"`` requires ``value`` to be a non-empty list of concrete
    literals; ``subset`` a list; ``eq_struct`` any list/dict/scalar body value;
    the scalar comparison ops require a single scalar. Placeholder semantics
    inside "in" sets are rejected by the validator.
    """

    path: str
    op: FieldOp
    value: ConstraintValue | list | dict

    @pydantic.model_validator(mode="after")
    def _check_op_value_pairing(self) -> "FieldConstraint":
        if self.op == "in":
            if not isinstance(self.value, list):
                raise ValueError("op 'in' requires a list of literal values")
            if not self.value:
                raise ValueError("op 'in' requires a non-empty list of literal values")
            # Strict-scalar members (no bool, no nesting): the `list` field type
            # is deliberately loose to also carry subset lists and eq_struct
            # dicts, so the member typing that `list[ConstraintValue]` used to give
            # 'in' is re-enforced here (the file is a security contract).
            if any(not _is_strict_scalar(m) for m in self.value):
                raise ValueError("op 'in' members must be strict scalars (int/float/str, not bool)")
        elif self.op == "subset":
            if not isinstance(self.value, list):
                raise ValueError("op 'subset' requires a list value")
        elif self.op == "eq_struct":
            if self.value is None:
                raise ValueError("op 'eq_struct' requires a value")
        elif isinstance(self.value, (list, dict)):
            raise ValueError(f"op '{self.op}' takes a single scalar value, not a list/dict")
        return self


class EndpointPattern(pydantic.BaseModel):
    """A single (method, domain, path_pattern) tuple the P-LLM has approved for a branch.

    HTTP analog of MCP's ``allowed_tools``: the P-LLM infers which endpoints the
    branch may legitimately call from the structural sitemap (method + path_template
    only — no semantic_action), and writes one entry per approved endpoint.
    The enforcer uses domain to scope the check: only requests to ``domain`` are
    tested against ``path_pattern``; requests to other allowed domains fall through
    to the ``sitemap_schema`` runtime check (or pass unchecked if no sitemap exists).
    """

    method: str
    domain: str
    path_pattern: str


class HttpConstraints(pydantic.BaseModel):
    allowed_domains: list[str] = pydantic.Field(default_factory=list)
    fields: list[FieldConstraint] = pydantic.Field(default_factory=list)
    # HTTP analog of MCP's allowed_tools: the P-LLM annotates which endpoints
    # (method + domain + path_pattern) the branch may legitimately call.
    # Enforcer checks: if a domain has entries here, (method, path) must match one.
    # Omitted/empty = no endpoint check for that domain (backward-compatible).
    allowed_endpoints: list[EndpointPattern] = pydantic.Field(default_factory=list)


class McpParamRule(pydantic.BaseModel):
    """Constraint on one parameter of an MCP ``tools/call``.

    Scalar ops (``<=``/``>=``/``==``) take a single value; ``eq_struct`` pins a
    whole list/dict by recursive value equality; ``subset`` requires the
    observed list to be ⊆ the planned list (so an injected append is blocked).
    ``source='from_plan'`` with no op is an exact scalar pin (``==``)."""

    tool: str
    param: str
    source: Literal["from_plan"] | None = None
    op: McpOp | None = None
    value: ConstraintValue | list | dict | None = None

    @pydantic.model_validator(mode="after")
    def _check_op_value_pairing(self) -> "McpParamRule":
        if self.op == "in":
            # Membership, mirroring `FieldConstraint`. Rules on one param are
            # conjunctive, so without it a policy of the form "the account must be one
            # of these two" has no expressible form — `==` twice authorises NOTHING.
            if not isinstance(self.value, list) or not self.value:
                raise ValueError("op 'in' requires a non-empty list of literal values")
            if any(not _is_strict_scalar(m) for m in self.value):
                raise ValueError("op 'in' members must be strict scalars (int/float/str, not bool)")
        elif self.op == "subset":
            if not isinstance(self.value, list):
                raise ValueError("op 'subset' requires a list value")
        elif self.op == "eq_struct":
            if self.value is None:
                raise ValueError("op 'eq_struct' requires a value")
        elif self.op in ("<=", ">=", "=="):
            if isinstance(self.value, (list, dict)):
                raise ValueError(f"op '{self.op}' takes a single scalar value, not a list/dict")
        return self


class McpConstraints(pydantic.BaseModel):
    allowed_tools: list[str] = pydantic.Field(default_factory=list)
    param_rules: list[McpParamRule] = pydantic.Field(default_factory=list)
    # Schema-closed param mode: a tool
    # listed here has a *sealed* parameter set — any argument on the wire whose
    # name is not in the list is blocked by MCP proxy. A tool absent from this map
    # keeps the default open behaviour (absent-param-passes / extra-param-passes).
    allowed_params: dict[str, list[str]] = pydantic.Field(default_factory=dict)
    # Server-qualified allowlist: pins each listed
    # tool to the server namespace(s) that may serve it at call time. MCP proxy checks
    # the calling proxy's server_id against this map; a tool with no entry keeps
    # name-only matching (backward-compatible). Emitted deterministically by
    # writer._apply_tool_servers from the approved-server manifest — the LLM
    # annotator is not asked to fill this field.
    allowed_tool_servers: dict[str, str | list[str]] = pydantic.Field(default_factory=dict)


class BranchConstraints(pydantic.BaseModel):
    description: str = ""
    trigger_var: str | None = None
    http_constraints: HttpConstraints = pydantic.Field(default_factory=HttpConstraints)
    # Null for HTTP-only annotation (no MCP tool manifest); filled when the
    # plan calls MCP tools.
    mcp_constraints: McpConstraints | None = None

    @classmethod
    def fail_closed(
        cls, description: str = "", trigger_var: str | None = None
    ) -> "BranchConstraints":
        """Constraints that authorise nothing (empty domain allowlist)."""
        return cls(
            description=description,
            trigger_var=trigger_var,
            http_constraints=HttpConstraints(),
            mcp_constraints=None,
        )


class PlanConstraints(pydantic.BaseModel):
    """Root object of `plan_constraints.json`."""

    plan_id: str
    task: str
    generated_by: str = "p-llm"
    branches: dict[str, BranchConstraints]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
