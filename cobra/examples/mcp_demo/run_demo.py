"""End-to-end MCP proxy demo — real MCP proxy process, no LLM (analogue of examples/http_proxy_demo).

Drives a single MCP client session through a real `python -m cobra.mcp_proxy` proxy in
front of the mock server, and checks each of MCP proxy's behaviours against a
hand-written `branch_state.json` (the file the BRH hook would write from a real
plan — hand-built here so the demo needs no P-LLM, exactly as `examples/http_proxy_demo`
hand-builds constraints for HTTP proxy):

  1. tools/list (first use)      -> forwarded, hashes registered
  2. place_order benign          -> forwarded, order placed
  3. place_order overcharge 500  -> blocked mpt_param (amount > 42.99 cap)
  4. send_email (not allowed)    -> blocked mpt_tool
  5. __mutate + tools/list again -> blocked mpt_rug_pull (description changed)

Run (free, no API key):
    cd cobra
    PYTHONPATH=src python3 examples/mcp_demo/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mcp_client import McpProxyClient, block_reason  # noqa: E402

SERVER = os.path.join(_HERE, "mock_mcp_server.py")

# The active branch the BRH hook would have written: place_order is authorised,
# capped at the perceived price 42.99 and pinned to SKU-7741; get_product too.
BRANCH_STATE = {
    "plan_id": "demo",
    "active_branch": "if_L1_true",
    "trigger_var": "price",
    "trigger_value": 42.99,
    "branch_path": ["root", "if_L1_true"],
    "http_constraints": None,
    "mcp_constraints": {
        "allowed_tools": ["get_product", "place_order"],
        "param_rules": [
            {"tool": "place_order", "param": "amount", "op": "<=", "value": 42.99},
            {"tool": "place_order", "param": "product_id", "source": "from_plan", "value": "SKU-7741"},
        ],
    },
}


def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"  {'✅' if ok else '❌'} {label}: {detail}")
    return ok


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="brh_mcp_")
    brh_dir = os.path.join(tmp, "brh")
    os.makedirs(brh_dir, exist_ok=True)
    registry = os.path.join(tmp, "hash_registry.json")
    alerts = os.path.join(tmp, "brh_alerts.jsonl")
    with open(os.path.join(brh_dir, "branch_state.json"), "w") as f:
        json.dump(BRANCH_STATE, f)

    print(f"artifacts: {tmp}")
    # Demo-only, explicit opt-in: step 1 below demonstrates trust-on-first-use
    # against a fresh, empty registry. A production deployment starts sealed
    # (the default) with the registry pre-seeded by the approval loop instead.
    client = McpProxyClient(SERVER, brh_dir=brh_dir, registry_path=registry, alerts_path=alerts,
                             sealed=False)
    results: list[bool] = []
    try:
        r = client.tools_list()
        tools = r.get("result", {}).get("tools", [])
        results.append(_check("1 tools/list first use", len(tools) == 2 and block_reason(r) is None,
                              f"{len(tools)} tools registered, forwarded"))

        r = client.tools_call("place_order", {"product_id": "SKU-7741", "amount": 42.99, "currency": "GBP"})
        results.append(_check("2 benign place_order", r.get("result", {}).get("status") == "ok",
                              f"order_id={r.get('result', {}).get('order_id')}"))

        r = client.tools_call("place_order", {"product_id": "SKU-7741", "amount": 500.0, "currency": "GBP"})
        results.append(_check("3 overcharge 500", block_reason(r) == "mpt_param",
                              f"reason={block_reason(r)}, data={r.get('error', {}).get('data')}"))

        r = client.tools_call("send_email", {"to": "x@y.z"})
        results.append(_check("4 disallowed tool", block_reason(r) == "mpt_tool",
                              f"reason={block_reason(r)}"))

        client.mutate()
        r = client.tools_list()
        results.append(_check("5 rug pull after mutate", block_reason(r) == "mpt_rug_pull",
                              f"reason={block_reason(r)}"))
    finally:
        client.close()

    if os.path.exists(alerts):
        print("\nalerts (/tmp .. /brh_alerts.jsonl):")
        with open(alerts) as f:
            for line in f:
                rec = json.loads(line)
                print(f"  - {rec['reason']}: {rec['detail']}")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
