# MCP proxy demo

This demo sends JSON-RPC traffic through a running `python -m cobra.mcp_proxy`
process. It uses a hand-written active branch and needs no LLM. See
`examples/http_proxy_demo` for the equivalent HTTP path.

## Topology

```
mcp_client.py  --stdio JSON-RPC--▶  python -m cobra.mcp_proxy  --stdio--▶  mock_mcp_server.py
 (the cobra-side                     reads branch_state.json,           (untrusted
  MCP client wrapper)                hashes tools/list, checks           downstream server,
                                     tools/call                          stdlib only)
```

The demo supplies `branch_state.json` directly. In a normal run, the BRH hook
writes it. The MCP proxy reads it for each `tools/call`, as the HTTP proxy does
for each HTTP request.

## Files

| File | Role |
|------|------|
| `mock_mcp_server.py` | Controllable stdio MCP server: `get_product` + `place_order`, and a `__mutate` control method that changes `place_order`'s description at runtime (the rug pull). Stdlib only, no cobra imports |
| `mcp_client.py` | Minimal synchronous MCP stdio client that spawns MCP proxy and talks through it; `block_reason()` extracts MCP proxy's block reason from a synthetic error |
| `run_demo.py` | LLM-free end-to-end demo: one client session, five checks, prints PASS/FAIL + the alert log |

## Run

```bash
cd cobra
PYTHONPATH=src python3 examples/mcp_demo/run_demo.py
```

## Checks

| # | Action | Expected |
|---|--------|----------|
| 1 | `tools/list` (first use) | forwarded; both tool hashes registered (trust-on-first-use, explicitly opted into with `sealed=False` — the production default is sealed) |
| 2 | `place_order` benign (amount 42.99, SKU-7741) | forwarded; order placed |
| 3 | `place_order` overcharge (amount 500) | blocked `mpt_param` (500 > the 42.99 cap in `branch_state.json`) |
| 4 | `send_email` (not in `allowed_tools`) | blocked `mpt_tool` |
| 5 | `__mutate` then `tools/list` again | blocked `mpt_rug_pull` (description changed after approval) |

Check 3 exercises branch steering over MCP. The plan caps `place_order` at the
perceived price; `contract.satisfies`, also used by the HTTP proxy, rejects the
higher amount sent on the wire.

## Coverage

- **Covers:** the real MCP proxy process (its asyncio stdio shell, not just the pure
  logic the unit tests hit), the MCP-client wrapper, and all four block reasons
  (`mpt_param` / `mpt_tool` / `mpt_rug_pull`, plus `mpt_inactive` reachable by
  pointing `--brh-dir` at a null/empty state).
- **Does not cover:** a P-LLM plan that generates the
  `mcp_constraints` (already wired via `PrivilegedLLM(brh_mcp_tools=...)`) and
  whose interpreted tool calls flow through `mcp_client`. That needs the
  MCP-backed tools registered in the CaMeL pipeline and a paid LLM run. The
  manifest those tools imply is built with
  `cobra.mcp_proxy.manifest.manifest_from_tools` from this server's
  `tools/list`.
