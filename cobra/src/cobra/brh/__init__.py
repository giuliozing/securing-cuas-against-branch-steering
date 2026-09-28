"""Branch Resolution Hub (BRH) — constraint initialization and resolution.

Generates `plan_constraints.json` from a freshly generated plan, before any
execution begins. Structure is derived deterministically from the AST
(`skeleton`), semantics are filled by a validated LLM call (`annotator`),
and everything is written atomically (`writer`). At runtime, the shared
predicate logic (`contract`) is what the interpreter and the HTTP/MCP
proxies call to evaluate branch conditions and action parameters against
the active constraints.
"""

from cobra.brh.annotator import AnnotationError, annotate
from cobra.brh.contract import is_placeholder, satisfies, typed_equal
from cobra.brh.schema import (
    BranchConstraints,
    FieldConstraint,
    HttpConstraints,
    McpConstraints,
    PlanConstraints,
)
from cobra.brh.sitemap_trust import (
    SitemapStatus,
    approve_sitemap,
    gate_sitemaps,
    reject_sitemap,
    sitemap_approval_loop,
    sitemap_hash,
    sitemap_status,
)
from cobra.brh.skeleton import InvalidPlanError, PlanSkeleton, extract_skeleton
from cobra.brh.validator import build_fallback, sanitize_sitemap, validate
from cobra.brh.writer import (
    BRHConfig,
    atomic_write_json,
    generate_plan_constraints,
    reset_branch_state,
)

__all__ = [
    "AnnotationError",
    "annotate",
    "is_placeholder",
    "satisfies",
    "typed_equal",
    "BranchConstraints",
    "FieldConstraint",
    "HttpConstraints",
    "McpConstraints",
    "PlanConstraints",
    "InvalidPlanError",
    "PlanSkeleton",
    "extract_skeleton",
    "build_fallback",
    "sanitize_sitemap",
    "validate",
    "SitemapStatus",
    "approve_sitemap",
    "gate_sitemaps",
    "reject_sitemap",
    "sitemap_approval_loop",
    "sitemap_hash",
    "sitemap_status",
    "BRHConfig",
    "atomic_write_json",
    "generate_plan_constraints",
    "reset_branch_state",
]
