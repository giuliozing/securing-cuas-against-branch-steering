# COBRA

COBRA provides control-flow integrity (CFI) and data-flow integrity (DFI) for
computer-use agents that use both web interfaces and Model Context Protocol
(MCP) tools. It derives constraints from a trusted plan and applies them to
HTTP and MCP actions. This prevents untrusted content from steering execution
to a branch chosen by an attacker.

This package contains the interpreter, Branch Resolution Hub (BRH), HTTP and
MCP proxies, and the Q-VLM interaction layer used in the utility evaluation.

## Layout

```
src/cobra/
  interpreter/       the deterministic plan interpreter (CFI: executes only
                      actions authorized by the committed plan)
  brh/                Branch Resolution Hub: constraint initialization
                      (skeleton, annotator, writer) from the trusted plan,
                      plus the shared predicate logic (contract.py) the
                      interpreter and both proxies use to validate branch
                      conditions and action parameters at runtime (DFI)
  mcp_proxy/          deterministic MCP proxy: tool-definition hash pinning,
                      server binding, per-call argument checks against the
                      active branch's constraints
  http_proxy/         deterministic HTTP proxy (mitmproxy addon): branch-aware,
                      stateful enforcement of hosts, endpoints and fields for
                      GUI-mediated web traffic; policy/ + approval/ are the
                      owner-facing sitemap/tool review and approval boundary
  pipeline_elements/  the privileged planner (P-LLM) and supporting pipeline
                      elements (function calling, tool filtering)
  interaction/        the quarantined interaction model (Q-VLM): VLM wrappers
                      (UI-TARS, OpenCUA, Anthropic, Gemini, Kimi) and the
                      environment adapters used to observe/act in a live
                      OSWorld VM during the utility evaluation
  quarantined_llm.py, qllm_vision.py, models.py, ...  shared runtime plumbing

tests/                unit tests, mirroring the layout above
examples/
  mcp_demo/           minimal end-to-end demo of the MCP proxy against a
                      mock MCP server (no API key required)
  http_proxy_demo/    minimal end-to-end demo of the HTTP proxy: a fresh
                      plan is fail-closed, branch activation opens exactly
                      the authorized requests, and branch steering /
                      cross-domain exfiltration are blocked
```

## Install

```bash
pip install -e ".[mcp,http_proxy]"
```

The `osworld` extra (and a running OSWorld VM) is only needed to exercise
the Q-VLM interaction layer against a live environment; the interpreter,
BRH, and both proxies do not depend on it.

## Run the demos

```bash
python examples/mcp_demo/run_demo.py
python examples/http_proxy_demo/run_demo.py   # needs the http_proxy extra
```

## Run the tests

```bash
pytest tests/test_interpreter tests/test_brh tests/test_mcp_proxy tests/test_http_proxy tests/test_pipeline_elements
```

## Provenance

COBRA extends two Apache-2.0 research artifacts: an interpreter and plan
language, and an HTTP sandboxing proxy. The BRH, MCP proxy, and enforcement
across HTTP and MCP are COBRA additions. See [NOTICE.md](NOTICE.md) for attribution.

## Constraint test vectors

[`brh_contract_vectors.json`](brh_contract_vectors.json) contains 62 synthetic
test cases for the BRH constraint contract. Its `vectors` array gives each
case an `id`, `constraint` (`op` and `value`), `observed` value, expected
outcome, and explanatory `note`. The interpreter and HTTP proxy tests read
the same vectors to check that they agree. The file is a test fixture under
this package's Apache-2.0 license.
