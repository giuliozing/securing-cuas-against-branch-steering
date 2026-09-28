"""CLI entrypoint: ``python -m mesa <url> [options]``.

MESA — MCP and Endpoint Sitemap generator for Agents.

Pipeline (three curated stages):
  crawl
    → propose HTTP        → curate & confirm HTTP        (Step 1/3)
    → propose MCP per HTTP → curate & confirm MCP tools   (Step 2/3)
    → polish everything    → review & approve all         (Step 3/3)
    → emit
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlparse

import httpx

from .crawler import crawl, cookies_from_storage_state
from .emit import write_outputs
from .llm import default_model, make_llm
from .proposer import propose_http, propose_mcp_from_http
from .refiner import refine
from .tui import curate_calls

# ANSI accent matching the TUI (green).
_ACCENT = "\033[38;5;35m"
_DIM = "\033[2m"
_RST = "\033[0m"

_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _banner(msg: str) -> None:
    print(f"{_ACCENT}▌{_RST} {msg}")


def _with_scheme(url: str) -> str:
    """Mirror the crawler's default-scheme rule, for host matching."""
    return url if urlparse(url).scheme else "https://" + url


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="mesa",
        description="MESA — MCP and Endpoint Sitemap generator for Agents.",
    )
    p.add_argument("url", help="site URL, e.g. https://example.com")
    p.add_argument("-o", "--out", default="./mesa_out", help="output directory")
    p.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "openai", "anthropic"],
        help="LLM provider (default: openrouter)",
    )
    p.add_argument("--model", default=None, help="model id (defaults per provider)")
    p.add_argument("--max-pages", type=int, default=6, help="max pages to crawl")
    p.add_argument(
        "--storage-state",
        default=None,
        metavar="PATH",
        help="Playwright storage_state JSON — crawl as the logged-in owner "
        "(most sites expose their agentic surface only to a session)",
    )
    p.add_argument(
        "--text-budget",
        type=int,
        default=None,
        metavar="CHARS",
        help="characters of crawl evidence passed to the LLM (default 16000). "
        "Truncation cuts from the tail, so raise this alongside --max-pages",
    )
    p.add_argument("--no-llm", action="store_true", help="skip all LLM calls (heuristics only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    llm = None
    if not args.no_llm:
        llm = make_llm(args.provider, args.model)
        if llm is None:
            key = _KEY_ENV.get(args.provider, "API key")
            print(
                f"{_DIM}No {key} found — running without LLM "
                f"(heuristic proposals, no schema refinement).{_RST}"
            )

    model_label = args.model or default_model(args.provider)

    cookies = None
    if args.storage_state:
        try:
            cookies = cookies_from_storage_state(
                args.storage_state, urlparse(_with_scheme(args.url)).netloc
            )
        except (OSError, ValueError, json.JSONDecodeError) as e:
            _banner(f"Could not read storage state {args.storage_state}: {e}")
            return 2
        if not cookies:
            _banner(f"No cookies in {args.storage_state} match {args.url} — aborting.")
            return 2

    _banner(f"Crawling {args.url}{' as logged-in user' if cookies else ''} …")
    try:
        evidence = crawl(
            args.url,
            max_pages=args.max_pages,
            cookies=cookies,
            text_budget=args.text_budget,
        )
    except httpx.HTTPError as e:
        _banner(f"Crawl failed: {e}")
        return 2
    if not evidence.pages:
        _banner("Could not fetch any page from the site — aborting.")
        return 2

    _banner(
        f"Crawled {len(evidence.pages)} page(s); "
        f"{sum(len(p.forms) for p in evidence.pages)} form(s) found."
    )
    if evidence.existing_agent_sitemap:
        _banner(
            f"Note: site already serves /sitemap.json "
            f"({len(evidence.existing_agent_sitemap)} entries)."
        )

    # ---- Step 1/3 — propose + curate HTTP calls -------------------------
    _banner(f"Step 1/3 · proposing HTTP calls{' with ' + model_label if llm else ''} …")
    http_calls = propose_http(evidence, llm)
    _banner(f"Proposed {len(http_calls)} HTTP call(s).")

    curated = curate_calls(
        http_calls,
        [],
        title="Step 1/3 — HTTP calls to expose",
        subtitle=f"{evidence.domain} · toggle/edit/add/remove the HTTP calls, then Enter",
    )
    if curated is None:
        _banner("Cancelled.")
        return 1
    http_calls, _ = curated
    if not http_calls:
        _banner("No HTTP calls selected — nothing to build.")
        return 1

    # ---- Step 2/3 — propose + curate MCP tools (one per HTTP call) ------
    _banner(
        f"Step 2/3 · proposing MCP tools (~one per HTTP call)"
        f"{' with ' + model_label if llm else ''} …"
    )
    mcp_calls = propose_mcp_from_http(http_calls, evidence, llm)
    _banner(f"Proposed {len(mcp_calls)} MCP tool(s).")

    curated = curate_calls(
        [],
        mcp_calls,
        title="Step 2/3 — MCP tools to expose",
        subtitle="one MCP tool per HTTP call · toggle/edit/add/remove, then Enter",
    )
    if curated is None:
        _banner("Cancelled.")
        return 1
    _, mcp_calls = curated

    # ---- Step 3/3 — polish everything, then review + approve -----------
    if llm is not None:
        _banner(f"Step 3/3 · polishing descriptions and schemas with {model_label} …")
        http_calls, mcp_calls = refine(http_calls, mcp_calls, evidence, llm)

    confirmed = curate_calls(
        http_calls,
        mcp_calls,
        title="Step 3/3 — review & approve everything",
        subtitle="review the polished HTTP + MCP fields/schemas · edit anything · Enter to write files",
    )
    if confirmed is None:
        _banner("Cancelled.")
        return 1
    http_calls, mcp_calls = confirmed
    if not http_calls and not mcp_calls:
        _banner("Nothing approved — nothing to emit.")
        return 1

    # Duplicate (method, url) sitemap entries are indistinguishable to BRH — keep the first.
    seen_endpoints = set()
    deduped = []
    for c in http_calls:
        key = (c.method.upper(), c.url)
        if key in seen_endpoints:
            continue
        seen_endpoints.add(key)
        deduped.append(c)
    if len(deduped) < len(http_calls):
        _banner(f"Dropped {len(http_calls) - len(deduped)} duplicate endpoint(s).")
    http_calls = deduped

    sitemap_path, manifest_path = write_outputs(http_calls, mcp_calls, args.out)
    _banner(f"Wrote {len(http_calls)} endpoint(s) → {sitemap_path}")
    if manifest_path:
        _banner(f"Wrote {len(mcp_calls)} MCP tool(s) → {manifest_path}")
    _banner("Done. sitemap.json is ready for the BRH sitemap-trust pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
