"""Tests for the BRH enforcement layer (`cobra.http_proxy.brh_check`).

Includes the shared contract-vector suite: the same
`brh_contract_vectors.json` consumed by the interpreter-side tests
(`tests/test_brh/test_contract.py`) is run against this module's
`satisfies` — that pairing is what keeps the two independent
implementations in parity.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from cobra.http_proxy.brh_check import (
    Decision,
    BRHState,
    RequestView,
    StateReader,
    check,
    read_state,
    satisfies,
)

# tests/test_http_proxy/test_brh_check.py -> test_http_proxy -> tests -> cobra (package root)
VECTORS_PATH = Path(__file__).resolve().parents[2] / "brh_contract_vectors.json"


def load_vectors() -> list[dict]:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]


def make_view(**kwargs) -> RequestView:
    defaults = dict(
        host="shop.example.com",
        port=443,
        method="POST",
        url="https://shop.example.com/checkout",
        body={},
        query={},
    )
    defaults.update(kwargs)
    return RequestView(**defaults)


def make_state(**kwargs) -> BRHState:
    defaults = dict(
        status="active",
        plan_id="plan_001",
        active_branch="if_L10_true",
        state_ts="2026-06-12T00:00:00.000Z",
        allowed_domains=("shop.example.com",),
        fields=(),
    )
    defaults.update(kwargs)
    return BRHState(**defaults)


class TestContractVectors(unittest.TestCase):
    def test_enforcer_matches_shared_vectors(self):
        vectors = load_vectors()
        self.assertGreater(len(vectors), 20)
        for v in vectors:
            with self.subTest(id=v["id"], note=v.get("note", "")):
                got = satisfies(v["constraint"]["op"], v["constraint"]["value"], v["observed"])
                self.assertEqual(
                    got,
                    v["expect"] == "satisfied",
                    f"vector '{v['id']}': expected {v['expect']}, enforcer said {got}",
                )


class TestReadState(unittest.TestCase):
    def _write(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_missing_file(self):
        self.assertEqual(read_state("/nonexistent/branch_state.json").status, "missing")

    def test_malformed_json(self):
        self.assertEqual(read_state(self._write("{not json")).status, "malformed")

    def test_non_object_root(self):
        self.assertEqual(read_state(self._write("[1, 2]")).status, "malformed")

    def test_inactive_state(self):
        path = self._write(json.dumps({"plan_id": "p1", "active_branch": None, "ts": "T"}))
        state = read_state(path)
        self.assertEqual(state.status, "inactive")
        self.assertEqual(state.plan_id, "p1")

    def test_active_state(self):
        path = self._write(json.dumps({
            "plan_id": "p1",
            "active_branch": "if_L10_true",
            "ts": "T",
            "http_constraints": {
                "allowed_domains": ["Shop.Example.COM"],
                "fields": [{"path": "amount", "op": "<=", "value": 42.99}],
            },
        }))
        state = read_state(path)
        self.assertEqual(state.status, "active")
        self.assertEqual(state.allowed_domains, ("shop.example.com",))  # lowercased
        self.assertEqual(len(state.fields), 1)

    def test_active_with_null_http_constraints_authorises_nothing(self):
        path = self._write(json.dumps({
            "plan_id": "p1", "active_branch": "root", "http_constraints": None, "ts": "T",
        }))
        state = read_state(path)
        self.assertEqual(state.status, "active")
        self.assertEqual(state.allowed_domains, ())


class TestStateReader(unittest.TestCase):
    """The stat-gated parse cache must preserve read_state's freshness while
    skipping the parse when the file is unchanged."""

    ACTIVE = json.dumps({
        "plan_id": "p1", "active_branch": "root", "ts": "T",
        "http_constraints": {"allowed_domains": ["shop.example.com"], "fields": []},
    })

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        self.path = Path(self.dir) / "branch_state.json"

    def _atomic_write(self, content: str) -> None:
        # Mirror cobra.brh.writer.atomic_write_json: temp + os.replace, so each
        # write swaps in a NEW inode (the property the cache keys on).
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".branch_state.")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, self.path)

    def test_unchanged_file_is_not_reparsed(self):
        self._atomic_write(self.ACTIVE)
        r = StateReader()
        first = r.read(self.path)
        second = r.read(self.path)
        # Same identity object on the second call ⇒ the parse was skipped.
        self.assertIs(first, second)
        self.assertEqual(first.status, "active")
        self.assertEqual(first.allowed_domains, ("shop.example.com",))

    def test_atomic_rewrite_is_picked_up(self):
        self._atomic_write(self.ACTIVE)
        r = StateReader()
        first = r.read(self.path)
        self._atomic_write(json.dumps({"plan_id": "p2", "active_branch": None, "ts": "T"}))
        second = r.read(self.path)
        self.assertIsNot(first, second)
        self.assertEqual(second.status, "inactive")
        self.assertEqual(second.plan_id, "p2")

    def test_missing_then_appearing(self):
        r = StateReader()
        self.assertEqual(r.read(self.path).status, "missing")
        self._atomic_write(self.ACTIVE)
        self.assertEqual(r.read(self.path).status, "active")

    def test_malformed_is_cached_and_recovers(self):
        self._atomic_write("{not json")
        r = StateReader()
        self.assertEqual(r.read(self.path).status, "malformed")
        self.assertEqual(r.read(self.path).status, "malformed")
        self._atomic_write(self.ACTIVE)
        self.assertEqual(r.read(self.path).status, "active")

    def test_freshness_matches_uncached_read_state(self):
        self._atomic_write(self.ACTIVE)
        r = StateReader()
        self.assertEqual(r.read(self.path), read_state(self.path))


class TestCheckDegenerateStates(unittest.TestCase):
    def test_missing_blocks(self):
        d = check(BRHState(status="missing"), make_view())
        self.assertEqual((d.allowed, d.reason), (False, "brh_state_missing"))

    def test_malformed_blocks(self):
        d = check(BRHState(status="malformed"), make_view())
        self.assertEqual((d.allowed, d.reason), (False, "brh_state_malformed"))

    def test_inactive_blocks(self):
        d = check(BRHState(status="inactive", plan_id="p1"), make_view())
        self.assertEqual((d.allowed, d.reason), (False, "brh_inactive"))


class TestCheckDomains(unittest.TestCase):
    def test_allowed_domain_passes(self):
        self.assertTrue(check(make_state(), make_view()).allowed)

    def test_unlisted_domain_blocks(self):
        d = check(make_state(), make_view(host="evil.example.net"))
        self.assertEqual((d.allowed, d.reason), (False, "brh_domain"))
        self.assertEqual(d.detail["host"], "evil.example.net")

    def test_no_subdomain_widening(self):
        # A bare allow-list entry authorises ONLY the exact host: listing the
        # apex must not implicitly widen to sub-domains.
        d = check(make_state(), make_view(host="sub.shop.example.com"))
        self.assertEqual((d.allowed, d.reason), (False, "brh_domain"))

    def test_wildcard_entry_covers_subdomains_and_apex(self):
        # An explicit `*.shop.example.com` entry authorises the apex and any
        # sub-domain (own-site first-party CDN/telemetry), but not a different
        # registrable domain.
        st = make_state(allowed_domains=("*.shop.example.com",))
        self.assertTrue(check(st, make_view(host="shop.example.com")).allowed)
        self.assertTrue(check(st, make_view(host="media.shop.example.com")).allowed)
        d = check(st, make_view(host="evil.example.net"))
        self.assertEqual((d.allowed, d.reason), (False, "brh_domain"))

    def test_empty_allowlist_blocks_everything(self):
        d = check(make_state(allowed_domains=()), make_view())
        self.assertEqual((d.allowed, d.reason), (False, "brh_domain"))


class TestCheckFields(unittest.TestCase):
    AMOUNT_LE = {"path": "amount", "op": "<=", "value": 42.99}

    def test_violating_body_field_blocks(self):
        d = check(
            make_state(fields=(self.AMOUNT_LE,)),
            make_view(body={"amount": 500}),
        )
        self.assertEqual((d.allowed, d.reason), (False, "brh_field"))
        self.assertEqual(d.detail["observed"], 500)
        self.assertEqual(d.detail["where"], "body")

    def test_satisfying_body_field_passes(self):
        d = check(make_state(fields=(self.AMOUNT_LE,)), make_view(body={"amount": 42.99}))
        self.assertTrue(d.allowed)

    def test_absent_field_passes(self):
        # The domain allowlist is the gate; field constraints narrow
        # values when the field travels.
        d = check(make_state(fields=(self.AMOUNT_LE,)), make_view(body={"note": "hi"}))
        self.assertTrue(d.allowed)

    def test_dot_path_traversal(self):
        fc = {"path": "order.amount", "op": "<=", "value": 42.99}
        d = check(make_state(fields=(fc,)), make_view(body={"order": {"amount": 500}}))
        self.assertEqual((d.allowed, d.reason), (False, "brh_field"))

    def test_query_param_occurrence_is_checked(self):
        # Query values are strings: strict typing makes a numeric
        # constraint unsatisfiable there — deliberate.
        d = check(
            make_state(fields=(self.AMOUNT_LE,)),
            make_view(method="GET", body={}, query={"amount": ["42.99"]}),
        )
        self.assertEqual((d.allowed, d.reason), (False, "brh_field"))
        self.assertEqual(d.detail["where"], "query")

    def test_string_constraint_on_query_param(self):
        fc = {"path": "currency", "op": "==", "value": "GBP"}
        ok = check(make_state(fields=(fc,)), make_view(query={"currency": ["GBP"]}))
        self.assertTrue(ok.allowed)
        bad = check(make_state(fields=(fc,)), make_view(query={"currency": ["USD"]}))
        self.assertEqual((bad.allowed, bad.reason), (False, "brh_field"))

    def test_every_query_occurrence_must_satisfy(self):
        fc = {"path": "currency", "op": "==", "value": "GBP"}
        d = check(make_state(fields=(fc,)), make_view(query={"currency": ["GBP", "USD"]}))
        self.assertEqual((d.allowed, d.reason), (False, "brh_field"))

    def test_unresolved_placeholder_blocks_when_field_present(self):
        fc = {"path": "amount", "op": "<=", "value": "trigger_value"}
        d = check(make_state(fields=(fc,)), make_view(body={"amount": 1}))
        self.assertEqual((d.allowed, d.reason), (False, "brh_field"))

    def test_malformed_constraint_entry_blocks(self):
        d = check(make_state(fields=({"op": "<="},)), make_view())
        self.assertEqual((d.allowed, d.reason), (False, "brh_constraint_malformed"))

    def test_conjunctive_same_path_constraints(self):
        fields = (
            {"path": "amount", "op": ">=", "value": 10},
            {"path": "amount", "op": "<=", "value": 50},
        )
        self.assertTrue(check(make_state(fields=fields), make_view(body={"amount": 30})).allowed)
        self.assertFalse(check(make_state(fields=fields), make_view(body={"amount": 5})).allowed)
        self.assertFalse(check(make_state(fields=fields), make_view(body={"amount": 99})).allowed)


class TestDecision(unittest.TestCase):
    def test_decision_defaults(self):
        self.assertTrue(Decision(True).allowed)
        self.assertIsNone(Decision(True).reason)


class TestCompileEndpointPattern(unittest.TestCase):
    def _m(self, pattern, path):
        from cobra.http_proxy.brh_check import _compile_endpoint_pattern
        return bool(_compile_endpoint_pattern(pattern).match(path))

    def test_exact_path_matches(self):
        self.assertTrue(self._m("/checkout", "/checkout"))

    def test_exact_path_no_partial(self):
        self.assertFalse(self._m("/checkout", "/checkout/extra"))

    def test_path_param_matches_segment(self):
        self.assertTrue(self._m("/users/{id}", "/users/42"))

    def test_path_param_does_not_span_slash(self):
        self.assertFalse(self._m("/users/{id}", "/users/42/extra"))

    def test_wildcard_spans_slashes(self):
        self.assertTrue(self._m("/api/*", "/api/v1/users"))

    def test_multiple_params(self):
        self.assertTrue(self._m("/groups/{group}/projects/{project}", "/groups/foo/projects/bar"))


class TestCheckAllowedEndpoints(unittest.TestCase):
    """Tier-1: plan-annotated allowed_endpoints blocks unlisted (method, path)."""

    def _state(self, endpoints):
        return make_state(allowed_endpoints=tuple(endpoints))

    def test_matching_endpoint_passes(self):
        st = self._state([{"method": "POST", "domain": "shop.example.com", "path_pattern": "/checkout"}])
        d = check(st, make_view(method="POST", url="https://shop.example.com/checkout"))
        self.assertTrue(d.allowed)

    def test_wrong_method_blocked(self):
        st = self._state([{"method": "GET", "domain": "shop.example.com", "path_pattern": "/checkout"}])
        d = check(st, make_view(method="POST", url="https://shop.example.com/checkout"))
        self.assertEqual(d.reason, "brh_endpoint")

    def test_unlisted_path_blocked(self):
        st = self._state([{"method": "POST", "domain": "shop.example.com", "path_pattern": "/checkout"}])
        d = check(st, make_view(method="POST", url="https://shop.example.com/delete-account"))
        self.assertEqual(d.reason, "brh_endpoint")

    def test_path_with_param_matches(self):
        st = self._state([{"method": "GET", "domain": "shop.example.com", "path_pattern": "/products/{id}"}])
        d = check(st, make_view(method="GET", url="https://shop.example.com/products/42"))
        self.assertTrue(d.allowed)

    def test_no_endpoints_for_host_passes_through(self):
        # allowed_endpoints has entries only for OTHER domain → no check for this host
        st = self._state([{"method": "POST", "domain": "other.example.com", "path_pattern": "/api"}])
        d = check(st, make_view(method="POST", url="https://shop.example.com/checkout"))
        self.assertTrue(d.allowed)

    def test_empty_allowed_endpoints_passes_through(self):
        d = check(make_state(allowed_endpoints=()), make_view())
        self.assertTrue(d.allowed)


class TestCheckSitemapSchema(unittest.TestCase):
    """Tier-2: runtime sitemap_schema blocks unlisted (method, path) for runtime domains."""

    def _state(self, sitemap):
        return make_state(sitemap_schema=sitemap)

    def test_matching_sitemap_entry_passes(self):
        st = self._state({"shop.example.com": [{"method": "POST", "path": "/checkout"}]})
        d = check(st, make_view(method="POST", url="https://shop.example.com/checkout"))
        self.assertTrue(d.allowed)

    def test_unlisted_path_in_sitemap_blocked(self):
        st = self._state({"shop.example.com": [{"method": "GET", "path": "/products"}]})
        d = check(st, make_view(method="POST", url="https://shop.example.com/delete-account"))
        self.assertEqual(d.reason, "brh_endpoint_sitemap")

    def test_no_sitemap_for_host_passes_through(self):
        st = self._state({})
        d = check(st, make_view(method="POST", url="https://shop.example.com/checkout"))
        self.assertTrue(d.allowed)

    def test_tier1_takes_precedence_over_tier2(self):
        # If host has allowed_endpoints (tier-1), sitemap_schema is ignored for it.
        st = make_state(
            allowed_endpoints=({"method": "POST", "domain": "shop.example.com", "path_pattern": "/checkout"},),
            sitemap_schema={"shop.example.com": [{"method": "GET", "path": "/anything"}]},
        )
        # Matches tier-1 → pass (sitemap says only GET /anything but that's irrelevant)
        self.assertTrue(check(st, make_view(method="POST", url="https://shop.example.com/checkout")).allowed)
        # Doesn't match tier-1 → blocked by tier-1 (not tier-2)
        d = check(st, make_view(method="GET", url="https://shop.example.com/anything"))
        self.assertEqual(d.reason, "brh_endpoint")


class TestReadStateWithEndpoints(unittest.TestCase):
    """read_state correctly parses allowed_endpoints and sitemap_schema."""

    def _write(self, payload: dict) -> Path:
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_parses_allowed_endpoints(self):
        path = self._write({
            "plan_id": "p1", "active_branch": "root", "ts": "T",
            "http_constraints": {
                "allowed_domains": ["shop.example.com"],
                "fields": [],
                "allowed_endpoints": [
                    {"method": "POST", "domain": "shop.example.com", "path_pattern": "/checkout"},
                ],
            },
        })
        state = read_state(path)
        self.assertEqual(len(state.allowed_endpoints), 1)
        self.assertEqual(state.allowed_endpoints[0]["path_pattern"], "/checkout")

    def test_malformed_endpoint_entry_skipped(self):
        path = self._write({
            "plan_id": "p1", "active_branch": "root", "ts": "T",
            "http_constraints": {
                "allowed_domains": ["shop.example.com"],
                "fields": [],
                "allowed_endpoints": [
                    {"method": 123, "domain": "shop.example.com", "path_pattern": "/checkout"},
                    {"method": "POST", "domain": "shop.example.com", "path_pattern": "/ok"},
                ],
            },
        })
        state = read_state(path)
        self.assertEqual(len(state.allowed_endpoints), 1)
        self.assertEqual(state.allowed_endpoints[0]["path_pattern"], "/ok")

    def test_parses_sitemap_schema(self):
        path = self._write({
            "plan_id": "p1", "active_branch": "root", "ts": "T",
            "http_constraints": {"allowed_domains": ["shop.example.com"], "fields": []},
            "sitemap_schema": {
                "shop.example.com": [{"method": "POST", "path": "/checkout"}]
            },
        })
        state = read_state(path)
        self.assertIn("shop.example.com", state.sitemap_schema)
        self.assertEqual(state.sitemap_schema["shop.example.com"][0]["path"], "/checkout")

    def test_missing_sitemap_schema_empty(self):
        path = self._write({
            "plan_id": "p1", "active_branch": "root", "ts": "T",
            "http_constraints": {"allowed_domains": ["shop.example.com"], "fields": []},
        })
        state = read_state(path)
        self.assertEqual(state.sitemap_schema, {})


if __name__ == "__main__":
    unittest.main()
