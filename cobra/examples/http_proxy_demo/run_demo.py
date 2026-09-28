"""End-to-end demo: BRH constraints → real hook → branch_state.json →
mitmproxy intercept → BLOCK with reason.

The branch states are produced by the **real** CaMeL-side hook
(`cobra.brh.hook` — the same calls `_eval_if` makes, with a fake
namespace standing in for the interpreter's), and enforcement is the
**real** HTTP proxy addon inside mitmdump. Only the annotator/validator
step is bypassed (constraints are built directly with the schema
models): the validator requires public dotted hostnames, which a
loopback demo cannot satisfy — that half of the chain is demonstrated
by `python3 -m cobra.brh.demo` on the CaMeL side.

Scenario (shop is reachable as both `localhost` and `127.0.0.1`; only
`localhost` is in the plan's allowed domains, so the same server doubles
as the "unplanned exfil domain" when addressed by IP):

    1. fresh plan, active_branch null      GET  /product       → 403 fail-closed
    2. root activated                      GET  /product       → 200
    3. purchase branch, price 42.99        POST /checkout 42.99 → 200
    4. branch steering: payload 500        POST /checkout 500   → 403 brh_field
    5. exfil to unplanned domain           POST 127.0.0.1/exfil → 403 brh_domain

Requirements: this package installed with the `http_proxy` extra
(`pip install -e ".[http_proxy]"`) — pydantic, mitmproxy, flask.

Usage: python3 run_demo.py
"""

import dataclasses
import json
import shutil
import subprocess
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
COBRA_SRC = DEMO_DIR.parents[1] / "src"
ADDON = COBRA_SRC / "cobra" / "http_proxy" / "mitm_addon.py"
PYTHON_BIN = sys.executable
MITMDUMP_BIN = shutil.which("mitmdump") or "mitmdump"

SHOP_PORT, PROXY_PORT = 8500, 8081
PROXY = {"http": f"http://127.0.0.1:{PROXY_PORT}"}

sys.path.insert(0, str(COBRA_SRC))

from cobra.brh import hook  # noqa: E402
from cobra.brh.schema import (  # noqa: E402
    BranchConstraints,
    FieldConstraint,
    HttpConstraints,
    PlanConstraints,
)
from cobra.brh.writer import BRHConfig, atomic_write_json, reset_branch_state  # noqa: E402


# --- fakes mirroring the interpreter's namespace/EvalArgs (as in cobra.brh.demo)

class FakeValue:
    def __init__(self, raw):
        self.raw = raw


class FakeNamespace:
    def __init__(self, variables):
        self._variables = variables

    def get(self, name):
        return self._variables.get(name)


@dataclasses.dataclass(frozen=True)
class FakeEvalArgs:
    brh_runtime: object = None
    brh_branch_path: tuple = ()


def build_constraints() -> PlanConstraints:
    return PlanConstraints(
        plan_id="demo_e2e",
        task="Buy the USB-C hub if the price is at most 50 GBP",
        generated_by="demo-script",
        branches={
            "root": BranchConstraints(
                description="initial product page fetch",
                http_constraints=HttpConstraints(allowed_domains=["localhost"]),
            ),
            "if_L4_true": BranchConstraints(
                description="purchase branch — price within budget",
                trigger_var="perceived_price",
                http_constraints=HttpConstraints(
                    allowed_domains=["localhost"],
                    fields=[
                        FieldConstraint(path="amount", op="<=", value="trigger_value"),
                        FieldConstraint(path="currency", op="==", value="GBP"),
                        FieldConstraint(path="product_id", op="==", value="SKU-7741"),
                    ],
                ),
            ),
            "if_L4_false": BranchConstraints(description="too expensive — no requests"),
        },
    )


def wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"port {port} did not come up")


def via_proxy(method: str, url: str, payload: dict | None = None) -> tuple[int, str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(PROXY))
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with opener.open(req, timeout=10) as resp:
            return resp.status, resp.read().decode(errors="ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="ignore")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="brh_demo_"))
    config = BRHConfig(out_dir=tmp)
    alerts_path = tmp / "brh_alerts.jsonl"

    constraints = build_constraints()
    atomic_write_json(config.constraints_path, constraints.to_json())
    reset_branch_state(constraints.plan_id, config)
    runtime = hook.BRHRuntime(constraints=constraints, state_path=config.state_path)
    print(f"BRH dir: {tmp}")

    shop = subprocess.Popen([PYTHON_BIN, str(DEMO_DIR / "shop_server.py"),
                             str(SHOP_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mitm = subprocess.Popen([MITMDUMP_BIN, "-q",
                             "--listen-port", str(PROXY_PORT), "-s", str(ADDON),
                             "--set", f"brh_state={config.state_path}",
                             "--set", f"brh_alerts={alerts_path}"])
    failures = 0
    try:
        wait_port(SHOP_PORT)
        wait_port(PROXY_PORT)

        def step(name: str, expected: int, got: tuple[int, str]) -> None:
            nonlocal failures
            status, body = got
            ok = status == expected
            failures += 0 if ok else 1
            mark = "✅" if ok else "❌"
            print(f"{mark} {name}: HTTP {status} (expected {expected})")
            if status == 403:
                reason = json.loads(body).get("reason") if body.startswith("{") else "?"
                print(f"     blocked by HTTP proxy, reason: {reason}")

        product_url = f"http://localhost:{SHOP_PORT}/product/SKU-7741"
        checkout_url = f"http://localhost:{SHOP_PORT}/checkout"

        # 1. Fresh plan: active_branch null → everything blocked.
        step("1. fail-closed before execution (active_branch null)",
             403, via_proxy("GET", product_url))

        # 2. Execution starts: run_code activates root.
        hook.activate_root(runtime)
        step("2. root active — planned product fetch", 200, via_proxy("GET", product_url))

        # 3. Q-LLM perceives price 42.99; plan enters the purchase branch.
        #    Same call _eval_if makes; placeholder amount<=trigger_value
        #    resolves to 42.99 in branch_state.json.
        eval_args = hook.attach(FakeEvalArgs(), runtime)
        namespace = FakeNamespace({"perceived_price": FakeValue(42.99)})
        hook.on_branch_entry(4, True, namespace, eval_args)
        step("3. purchase branch — benign checkout (amount 42.99)", 200,
             via_proxy("POST", checkout_url,
                       {"amount": 42.99, "currency": "GBP", "product_id": "SKU-7741"}))

        # 4. Branch steering: the visual channel was manipulated; the HTTP
        #    payload reveals it (500 > 42.99).
        step("4. branch steering — payload exceeds plan constraint", 403,
             via_proxy("POST", checkout_url,
                       {"amount": 500, "currency": "GBP", "product_id": "SKU-7741"}))

        # 5. Exfiltration to a domain no branch authorises (same server,
        #    addressed by IP — not in allowed_domains).
        step("5. cross-site exfil — unplanned domain", 403,
             via_proxy("POST", f"http://127.0.0.1:{SHOP_PORT}/exfil",
                       {"secret": "session-token"}))

        print("\n--- branch_state.json (final) ---")
        print(config.state_path.read_text())
        print("--- brh_alerts.jsonl ---")
        if alerts_path.exists():
            for line in alerts_path.read_text().splitlines():
                a = json.loads(line)
                print(f"  [{a['ts']}] {a['reason']} {a['method']} {a['url']} {a['detail']}")
        print(f"\n{'ALL STEPS PASSED' if failures == 0 else f'{failures} STEP(S) FAILED'}")
        return 0 if failures == 0 else 1
    finally:
        mitm.terminate()
        shop.terminate()


if __name__ == "__main__":
    sys.exit(main())
