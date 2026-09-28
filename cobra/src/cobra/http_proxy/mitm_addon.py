"""HTTP proxy mitmproxy addon: BRH dynamic layer + optional static Policy
layer.

Run (no install needed — the script puts ``src/`` on sys.path itself):

    mitmdump -s src/cobra/http_proxy/mitm_addon.py \\
        --set brh_state=/tmp/brh/branch_state.json \\
        --set brh_alerts=/tmp/brh_alerts.jsonl \\
        [--set brh_mode=enforce|monitor] \\
        [--set brh_block_style=respond|kill] \\
        [--set brh_passlist=/path/to/passlist.txt] \\
        [--set http_proxy_policy=/path/to/policy.json] \\
        [--set http_proxy_sitemap=/path/to/sitemap.json]

Per-request chain:

    infra passlist  — environment noise (browser telemetry, OCSP, CDNs);
                      deployment config, *not* plan-derived policy. Out of
                      the declared threat model, documented as such.
    BRH check       — dynamic plan constraints from branch_state.json,
                      polled on every request (no caching by design).
    Policy layer    — legacy static rules (tags/endpoints via sitemap),
                      unchanged semantics; only loaded if configured.

Behaviour choices:

- **Blocking style**: default is a synthetic 403 JSON response telling
  the agent to terminate and not retry (mirrors the legacy PolicyDenied
  message) — ``kill`` is available for comparison but produces an opaque
  connection reset the agent tends to retry, polluting both the run and
  the alert-based metrics.
- **monitor mode** logs and alerts but never blocks: used to tune the
  passlist on real benchmarks and to measure false positives separately
  from utility impact.
- **Fail-closed on internal errors**: if this addon itself raises, the
  request is blocked (in enforce mode), not forwarded — the opposite of
  the CaMeL-side hook's fail-safe contract, because the two components
  have opposite duties: the hook must never corrupt plan execution, the
  enforcer must never let traffic slip on a malfunction.
- Consecutive ``brh_inactive`` blocks emit a single alert (the null
  state between plans is legitimate; one line is signal, a thousand are
  noise).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parents[2]
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

from mitmproxy import ctx, http  # noqa: E402

from cobra.http_proxy import adapter, alerts, brh_check  # noqa: E402

logger = logging.getLogger("cobra.http_proxy")


class HTTPProxyAddon:
    def __init__(self) -> None:
        self._passlist: list[str] = []
        self._policy = None
        self._sitemap = None
        self._inactive_alerted = False
        # Alert de-duplication keys (reason, host, plan_id) already recorded. A
        # heavy page fans out the SAME violation hundreds of times (united.com:
        # ~1600 flagged requests over ~40 hosts); appending+logging each one
        # synchronously inside mitmproxy's request hook stalls the event loop and
        # Chrome sees ERR_PROXY_CONNECTION_FAILED. De-duping collapses that to one
        # alert per (host, plan_id) — the metric counts distinct hosts per plan
        # anyway — while enforce-mode blocking still fires on EVERY request.
        self._seen: set = set()
        # stat-gated parse cache for branch_state.json: the per-request read is
        # the other synchronous I/O on the hot path the de-dup did not remove.
        # Keeps the same freshness contract (stat every request, re-read on any
        # change) while skipping the json parse when the state is unchanged.
        self._state_reader = brh_check.StateReader()

    # -- options ------------------------------------------------------

    def load(self, loader) -> None:
        loader.add_option("brh_state", str, "/tmp/brh/branch_state.json",
                          "Path to the BRH branch_state.json written by the CaMeL hook.")
        loader.add_option("brh_alerts", str, "/tmp/brh_alerts.jsonl",
                          "Append-only JSONL alert log.")
        loader.add_option("brh_mode", str, "enforce",
                          "enforce: block violations; monitor: log/alert only.")
        loader.add_option("brh_block_style", str, "respond",
                          "respond: synthetic 403 JSON; kill: connection reset.")
        loader.add_option("brh_passlist", str, "",
                          "Optional file of infrastructure domains exempt from enforcement "
                          "(one per line, '#' comments, '*.suffix' wildcards).")
        loader.add_option("http_proxy_policy", str, "",
                          "Optional legacy Policy JSON; enables the static layer.")
        loader.add_option("http_proxy_sitemap", str, "",
                          "Optional sitemap JSON for tag resolution in the static layer.")

    def configure(self, updated) -> None:
        if "brh_passlist" in updated:
            self._passlist = self._load_passlist(ctx.options.brh_passlist)
        if "http_proxy_policy" in updated or "http_proxy_sitemap" in updated:
            self._load_policy(ctx.options.http_proxy_policy, ctx.options.http_proxy_sitemap)

    @staticmethod
    def _load_passlist(path: str) -> list[str]:
        if not path:
            return []
        entries: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                entries.append(line)
        logger.info("http_proxy: infra passlist loaded (%d entries)", len(entries))
        return entries

    def _load_policy(self, policy_path: str, sitemap_path: str) -> None:
        self._policy = None
        self._sitemap = None
        if not policy_path:
            return
        # The legacy policy module imports playwright; a BRH-only
        # deployment must keep working without it.
        try:
            from cobra.http_proxy.policy.policy import Policy
            from cobra.http_proxy.policy.sitemap import Sitemap
        except ImportError as e:
            logger.error("http_proxy: static Policy layer disabled (import failed: %r)", e)
            return
        self._policy = Policy.from_json(Path(policy_path).read_text(encoding="utf-8"))
        if sitemap_path:
            self._sitemap = Sitemap(Path(sitemap_path).read_text(encoding="utf-8"))
        logger.info("http_proxy: static Policy layer enabled (%s)", self._policy.name)

    # -- per-request chain ---------------------------------------------

    def request(self, flow: http.HTTPFlow) -> None:
        try:
            self._request(flow)
        except Exception:
            logger.exception("http_proxy: internal error — failing closed")
            decision = brh_check.Decision(False, "brh_internal_error", {})
            state = brh_check.BRHState(status="malformed")
            view = brh_check.RequestView(
                host=flow.request.pretty_host.lower(), port=flow.request.port,
                method=flow.request.method, url=flow.request.pretty_url,
            )
            self._violation(flow, "brh_http_block", state, view, decision)

    def _request(self, flow: http.HTTPFlow) -> None:
        view = adapter.request_view_from_flow(flow)
        if self._passlisted(view.host):
            return

        # BRH dynamic layer.
        state = self._state_reader.read(ctx.options.brh_state)
        decision = brh_check.check(state, view)
        if not decision.allowed:
            self._violation(flow, "brh_http_block", state, view, decision)
            return
        self._inactive_alerted = False

        # Legacy static layer.
        if self._policy is not None:
            action = adapter.action_from_view(view, self._sitemap)
            effect = self._policy.evaluate(action)
            if effect == "deny":
                self._violation(
                    flow, "policy_http_block", state, view,
                    brh_check.Decision(False, "policy_deny", {"policy": self._policy.name}),
                )

    def _passlisted(self, host: str) -> bool:
        for entry in self._passlist:
            if entry.startswith("*."):
                base = entry[2:]
                if host == base or host.endswith("." + base):
                    return True
            elif host == entry:
                return True
        return False

    # -- block + alert --------------------------------------------------

    def _violation(self, flow, kind, state, view, decision) -> None:
        mode = ctx.options.brh_mode
        # Record (alert + log) only the first time we see a (reason, host, plan_id)
        # so a heavy page's repeated identical violations don't stall the event
        # loop. Enforce-mode blocking is OUTSIDE this guard — every request is
        # still blocked, only the I/O is de-duplicated.
        key = (decision.reason, view.host, state.plan_id)
        if key not in self._seen:
            self._seen.add(key)
            logger.warning(
                "http_proxy %s [%s] %s %s — %s %s",
                "BLOCK" if mode == "enforce" else "MONITOR",
                decision.reason, view.method, view.url, kind, decision.detail,
            )
            alerts.append_alert(
                ctx.options.brh_alerts, alerts.build_alert(kind, mode, state, view, decision)
            )
        if mode == "enforce":
            self._block(flow, decision)

    @staticmethod
    def _block(flow: http.HTTPFlow, decision: brh_check.Decision) -> None:
        if ctx.options.brh_block_style == "kill":
            flow.kill()
            return
        body = json.dumps(
            {
                "error": "HTTP proxy: request blocked",
                "reason": decision.reason,
                "detail": decision.detail,
                "instruction": (
                    "Policy violation. The agent MUST terminate the task "
                    "immediately, not retry it."
                ),
            },
            default=str,
        ).encode()
        flow.response = http.Response.make(
            403, body, {"content-type": "application/json"}
        )


addons = [HTTPProxyAddon()]
