# MCP proxy unit tests

These tests cover tool-definition hash checks, `tools/call` argument checks,
and frame routing in `src/cobra/mcp_proxy/`. The asyncio stdio shell in
`proxy.py` is exercised by the end-to-end demo instead.

## Running

```bash
cd cobra
PYTHONPATH=src python3 -m unittest discover -s tests/test_mcp_proxy -t .
```

The suite needs `python3` and `pydantic`, but no API key or virtual
environment. Its imports run through `cobra.mcp_proxy`, `cobra.brh.contract`,
and `cobra.brh.schema`, as in `tests/test_brh/`.

## Design under test

Decision functions accept parsed dictionaries and return dataclasses, so the
tests do not need live pipes:

- `check.py` — the policy: `check_tools_call(state, tool, args) -> Decision`,
  `check_tools_list(tools, server_id, registry) -> ToolListResult`.
- `router.py` — per-frame routing: `route_client_frame` / `route_server_frame`
  return a `Routing(to_server, to_client, alerts)` over already-parsed frames.
- `registry.py` — tool-definition hashing + on-disk registry.

`proxy.py` handles line I/O and calls these functions. It contains no policy.

---

## `test_check.py` — the policy (16 tests)

Helper `_state(allowed, rules, active="root")` builds the `branch_state.json`
shape the BRH hook would write (`mcp_constraints=None` when neither `allowed`
nor `rules` is given, to exercise the "active branch, no MCP" case).

### `CheckToolsCall` — one test per fail-closed exit, in check order

| Test | Asserts |
|------|---------|
| `test_no_state_blocks_inactive` | `state=None` → `mpt_inactive` |
| `test_null_active_branch_blocks_inactive` | `active_branch` of `None` **and** the `"null"` string sentinel → `mpt_inactive` |
| `test_no_mcp_constraints_blocks_tool` | active branch but `mcp_constraints=None` → `mpt_tool` |
| `test_tool_not_in_allowlist_blocks` | tool ∉ `allowed_tools` → `mpt_tool` (tool-mismatch attack) |
| `test_allowed_tool_no_rules_passes` | allowed tool, no param rules → allow (minimal happy path) |
| `test_op_le_pass_and_fail` | `amount <= 42.99`: 42.99 passes, 500 → `mpt_param`, and `detail["observed"] == 500.0` (the alert payload is correct, not just the verdict) |
| `test_from_plan_exact_match` | `source=from_plan`, no `op` → treated as `==`: SKU-7741 passes, SKU-9999 blocks |
| `test_unresolved_placeholder_is_unsatisfiable` | a `value` of `"from_plan"` or `"trigger_value"` **blocks** — an unresolved placeholder is never a wildcard (the key safety property) |
| `test_absent_param_passes` | a rule whose `param` is absent from the call passes (parity with the HTTP layer; allowlist is the gate) |
| `test_malformed_rule_blocks` | a rule with neither `op` nor `source` → `mpt_param` (fail-closed, not ignored) |
| `test_strict_types_no_coercion` | `== 42` (int) vs `"42"` (str) blocks — verifies MCP proxy **delegates to `contract.satisfies`** rather than reimplementing type semantics |
| `test_rule_for_other_tool_ignored` | a rule with `tool:"other"` does not affect a `place_order` call |

### `CheckToolsList` — hash pinning and the sealed default

Pattern: **seed then mutate** — use `check_tools_list(...).registry` as the
approved baseline, then re-list with/without a change. `approve_new` defaults
to `False` (sealed): trust-on-first-use is opt-in, never implicit.

| Test | Asserts |
|------|---------|
| `test_default_is_sealed_rejects_unknown_tool` | no `approve_new` passed, empty registry → `ok=False`, `mpt_unapproved` (the production default) |
| `test_explicit_unsealed_first_use_registers_and_forwards` | `approve_new=True`, empty registry → `ok`, `changed`, key present (the explicit benchmark/test opt-in) |
| `test_matching_hash_no_change` | re-list of the identical tool → `ok`, **not** `changed`, no alerts |
| `test_changed_description_is_rug_pull` | mutated description → `ok=False`, `alerts[0].reason == "mpt_rug_pull"` |
| `test_sealed_mode_blocks_unapproved` | `approve_new=False` explicitly, empty registry → `ok=False`, `mpt_unapproved`, not `changed` |
| `test_reapproval_after_rug_pull_requires_explicit_approve_tool` | a hash mismatch blocks regardless of `approve_new`; only an explicit `registry.approve_tool(...)` re-pins it |

---

## `test_registry.py` — hashing & persistence (7 tests)

### `Canonicalisation` — the hash's robustness

| Test | Asserts |
|------|---------|
| `test_hash_is_key_order_independent` | two dicts with the same content but different key order (incl. nested `inputSchema`) → **same hash** (`sort_keys=True`). Without this you get spurious rug pulls |
| `test_description_change_changes_hash` | description change → different hash |
| `test_input_schema_change_changes_hash` | schema change → different hash |
| `test_ignores_extraneous_fields` | adding a `_meta` field leaves `canonical_tool_bytes` byte-identical — only `name+description+inputSchema` are hashed; transport metadata must not trigger rug pulls |

### `Persistence`

| Test | Asserts |
|------|---------|
| `test_missing_file_is_empty` | missing path → `{}` (legitimate first run, not an error) |
| `test_round_trip` | save→load through a **nested** temp path → equal, and `save_registry` creates missing dirs |
| `test_unparsable_file_is_empty` | a corrupt `"{not json"` file → `{}`, no exception |

---

## `test_router.py` — frame routing (6 tests)

Verifies **where a frame goes**, not the policy (already covered by
`test_check`). Uses `assertIs` for forwarding to assert the frame travels
*unchanged* (same object), not copied/rebuilt.

### `RouteClientFrame`

| Test | Asserts |
|------|---------|
| `test_allowed_call_forwarded_to_server` | allowed `tools/call` → `to_server is frame`, `to_client is None`, no alerts |
| `test_blocked_call_returns_error_and_alert` | blocked call → `to_server is None`, `to_client["id"] == 7` (the JSON-RPC id is preserved so the client correlates), `"error"` present, alert `mpt_tool` |
| `test_non_call_request_recorded_in_pending` | a `tools/list` request is forwarded **and** `pending` becomes `{3: "tools/list"}` (the correlation state mutation) |

### `RouteServerFrame`

| Test | Asserts |
|------|---------|
| `test_default_is_sealed_unknown_tool_blocked` | no `approve_new` passed → `to_client` carries `"error"`, alert `mpt_unapproved`, registry unchanged (the production default) |
| `test_explicit_unsealed_first_use_forwarded_registry_changed` | `approve_new=True` → `to_client is frame`, `changed=True`, key in registry, **and `pending == {}`** (the id was consumed with `pop`) |
| `test_tools_list_rug_pull_replaced_with_error` | mutated description → `to_client` carries `"error"` (the poisoned list is **not** forwarded), alert `mpt_rug_pull` |
| `test_tools_list_sealed_unapproved_surfaces_its_own_reason` | a sealed-mode block surfaces `mpt_unapproved`, not a hardcoded `mpt_rug_pull` |
| `test_unrelated_response_forwarded_untouched` | a response whose `id` is not a pending `tools/list` → forwarded identical, `changed=False`, registry unchanged |

---

## `test_manifest.py` — tool defs → BRH manifest (5 tests)

`manifest_from_tools` turns a server's `tools/list` defs into the
`{tool: [param names]}` manifest the BRH annotator consumes. Tests: names map to
their `inputSchema.properties` keys; missing/empty schema → empty param list;
unnamed tools skipped; empty/`None` input → `{}`.

## Asyncio shell coverage

These unit tests do not drive `proxy.py`'s line reading, subprocess launch, or
JSON-RPC I/O. `examples/mcp_demo/run_demo.py` covers those paths with a real
`python -m cobra.mcp_proxy` process, an MCP client wrapper, and a mock stdio
server. It runs five checks without an LLM.

Note: the type/placeholder tests in `test_check` do **not** re-test the
`satisfies` semantics themselves — those are pinned by the shared golden vectors
(`brh_contract_vectors.json`, consumed by `tests/test_brh/` and the HTTP proxy
suite). MCP proxy imports `contract.satisfies` directly, so these tests verify MCP proxy
*calls the contract correctly* and surfaces its `False` as an `mpt_param` block.
