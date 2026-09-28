# MESA: agent sitemap and MCP manifest generator

A terminal tool for website owners to review the HTTP calls and MCP tools
they want to expose to agents. It writes an agent sitemap and MCP manifest.

Given a site URL, MESA does the following:

1. **Crawls** the site (homepage + a few same-origin pages, following HTTP and
   HTML meta-refresh redirects) and gathers the observable structure — page
   titles, forms (action/method/inputs), links, `robots.txt`, `sitemap.xml`,
   and any existing `/sitemap.json`.
2. **HTTP review.** Proposes the essential HTTP calls (short *name* +
   one-line *description*), via an LLM (default) or a deterministic
   forms/links heuristic. The owner curates them in a full-screen terminal UI
   (toggle / edit-as-JSON / add / remove) and confirms with **Enter**.
3. **MCP review.** From the confirmed HTTP calls, proposes MCP tools, usually
   one per HTTP call, each with a name, description,
   and JSON-Schema `input_schema` derived from the HTTP call's URL placeholders
   (`{like_this}`) and body fields. Works with or without an LLM (the
   deterministic 1:1 mapping guarantees at least one tool per HTTP call). The
   owner curates and confirms.
4. **Final review.** An LLM refines the descriptions and schemas
   of everything (HTTP: `semantic_action`, `method`, `url` template, `tags`,
   `body_fields`, `category`, `priority`; MCP: description + `inputSchema`). The
   owner reviews, may still edit, and approves.
5. **Output.** Writes files for the BRH and MCP proxy pipeline.

## Output

- `sitemap.json` — the agent sitemap in the exact shape consumed by
  `cobra.brh.validator.sanitize_sitemap` and hash-pinned by
  `cobra.brh.sitemap_trust`. Declared `body_fields` are written as the keys of
  each entry's `body` object (which is where the sanitizer reads them).
- `mcp_manifest.json` — a list of `{name, description, inputSchema}` MCP tools
  (the shape MCP proxy hash-pins on `tools/list`), written only if MCP calls were kept.

## Usage

Install the package with `pip install -e .` from `mesa/`. This installs the
crawler and terminal UI dependencies. To use a native model provider, install
the corresponding optional extra (`.[openai]` or `.[anthropic]`). Then run:

> The Python module name is `mesa`. (The "agent sitemap" wording above refers
> to the *output format* — the endpoint sitemap the BRH pipeline consumes.)

```bash
# default: OpenRouter (OpenAI-compatible); model ids are vendor-qualified
OPENROUTER_API_KEY=sk-or-... python -m mesa https://example.com -o ./out
OPENROUTER_API_KEY=sk-or-... python -m mesa https://example.com --model anthropic/claude-sonnet-5

# native providers
OPENAI_API_KEY=sk-...    python -m mesa https://example.com --provider openai
ANTHROPIC_API_KEY=...    python -m mesa https://example.com --provider anthropic

# no LLM at all (heuristic HTTP proposals + deterministic MCP mapping, no polish)
python -m mesa https://example.com --no-llm
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `url` | — | site URL (scheme optional; `https://` assumed) |
| `-o, --out` | `./mesa_out` | output directory |
| `--provider` | `openrouter` | `openrouter`, `openai`, or `anthropic` |
| `--model` | provider default | model id (OpenRouter default `openai/gpt-5`) |
| `--max-pages` | `6` | max pages to crawl |
| `--storage-state` | — | Playwright `storage_state` JSON: crawl as the logged-in owner |
| `--text-budget` | `16000` | characters of crawl evidence passed to the LLM |
| `--no-llm` | off | skip all LLM calls (heuristics only) |

### Crawling a site that requires login

Many sites expose relevant pages only after sign-in. An anonymous crawl may
find only the login page. `--storage-state`
accepts a Playwright `storage_state` JSON (what `context.storage_state()`
writes) and carries that session through the crawl; links that would end the
session (`/logout`, `/users/sign_out`, …) are skipped.

```bash
python -m mesa http://127.0.0.1:8023 --storage-state ./.auth/gitlab_state.json
```

### Evidence budget

`--max-pages` limits the pages fetched; `--text-budget` limits the crawl text
sent to the LLM. Text beyond the budget is cut from the end. At the default
budget, about 7–12 pages fit. Increase both settings for a larger site.

If no API key is present the tool prints a notice and continues in no-LLM mode.
If stdin/stdout is not a TTY the interactive stages pass through unchanged
(useful for scripted/CI runs).

MESA is a generator, not a precomputed website dataset. This package contains
the generator and output format, but no crawl snapshots or evaluated site
manifests. The site owner supplies a URL and reviews the generated files.

## Terminal UI keys

```
↑/↓ (or k/j)  move            space  toggle include (◉/○)
e             edit call JSON   a      add a call
d / x         delete call      enter  confirm this stage
C-c / q       quit             C-s    save (in the JSON editor)
```

Editing a call opens its fields as JSON; fix them and press `C-s`. Invalid JSON
or a missing `name` is reported in the status bar and the editor stays open.

## Module layout

| File | Role |
|------|------|
| `crawler.py` | read-only crawl (HTTP + meta-refresh) → `CrawlEvidence` |
| `proposer.py` | Step 1 HTTP proposals (`propose_http`) + Step 2 MCP-per-HTTP (`propose_mcp_from_http`) |
| `tui.py` | full-screen `prompt_toolkit` curation/confirmation UI |
| `refiner.py` | Step 3 polish — LLM refinement of descriptions + schemas |
| `emit.py` | write `sitemap.json` + `mcp_manifest.json` |
| `llm.py` | `llm_call(system, user) -> str` abstraction (OpenRouter/OpenAI/Anthropic) |
| `models.py` | `HttpCall` / `McpCall` + (de)serialisation |
| `__main__.py` | pipeline orchestration + CLI |
