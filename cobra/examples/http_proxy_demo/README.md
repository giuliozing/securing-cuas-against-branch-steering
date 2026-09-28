# HTTP proxy demo

This demo runs a synthetic shop through the BRH hook and HTTP proxy. The hook
(`cobra.brh.hook`) writes constraints to `branch_state.json`; the mitmproxy
addon checks requests against that state and records blocked requests in
`brh_alerts.jsonl`.

## Install

```bash
cd cobra
pip install -e ".[http_proxy]"   # pydantic, mitmproxy, flask
```

## Run

```bash
cd examples/http_proxy_demo
python3 run_demo.py
```

The script checks five cases. All should pass:

1. With `active_branch: null`, the GET is blocked before execution.
2. After the root branch is activated, the planned GET passes.
3. In the purchase branch, checkout at `perceived_price=42.99` passes.
4. Checkout with `amount=500` returns 403 with reason `brh_field`.
5. A request to an unplanned domain (the same server by IP) returns 403 with reason `brh_domain`.

The script then prints the final `branch_state.json` and alert log. The hook
resolves `amount <= trigger_value` to `42.99`.

Metrics over the produced alerts:

```bash
python3 -m cobra.http_proxy.metrics /tmp/brh_demo_*/brh_alerts.jsonl --branches-entered 2
```

## Notes

- The annotator/validator step is bypassed here (constraints are built
  directly with the schema models): the validator requires public dotted
  hostnames, which a loopback demo cannot satisfy. That half of the
  chain is demonstrated by `python3 -m cobra.brh.demo`.
- The shop is reachable as both `localhost` and `127.0.0.1`; only
  `localhost` is in the plan's `allowed_domains`, so the IP address
  doubles as the attacker's unplanned domain.
