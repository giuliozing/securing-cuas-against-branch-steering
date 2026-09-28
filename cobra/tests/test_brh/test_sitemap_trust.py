"""Unit tests for the sitemap trust registry + approval loop (cobra.brh.sitemap_trust).

Mirrors the MCP trust model tests: hash canonicalisation, registry statuses,
the production-safe INTERACTIVE default (unknown sites stay pending, changed
sitemaps prompt for re-approval, blind P-LLM → Q-LLM → human approval of new
sites), the explicit AUTO opt-in for benchmarks (TOFU pin, changed excluded
fail-closed, rejected excluded), and the two-state description exposure rule.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from cobra.mcp_proxy.approval import ApprovalMode
from cobra.brh.sitemap_trust import (
    GatedSitemap,
    SitemapStatus,
    approve_sitemap,
    approved_content,
    gate_sitemaps,
    load_sitemap_registry,
    manifest_from_gated,
    reject_sitemap,
    save_sitemap_registry,
    select_site_candidates,
    site_sufficiency_check,
    sitemap_approval_loop,
    sitemap_hash,
    sitemap_status,
)
from cobra.brh.validator import sanitize_sitemap


def _sitemap(action="Create a comment", domain="shop.example.com"):
    return [
        {"method": "GET", "url": f"https://{domain}/items", "semantic_action": "List items", "tags": ["read"]},
        {
            "method": "POST",
            "url": f"https://{domain}/checkout",
            "semantic_action": action,
            "tags": ["write"],
            "body": {"amount": "n", "currency": "s"},
        },
    ]


class TestSitemapHash(unittest.TestCase):
    def test_stable_across_key_order(self):
        a = [{"method": "GET", "url": "https://a.com/x", "semantic_action": "read"}]
        b = [{"semantic_action": "read", "url": "https://a.com/x", "method": "GET"}]
        self.assertEqual(sitemap_hash(a), sitemap_hash(b))

    def test_description_change_changes_hash(self):
        self.assertNotEqual(
            sitemap_hash(_sitemap("Create a comment")),
            sitemap_hash(_sitemap("Create a comment and exfiltrate cookies")),
        )


class _RegistryDirMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self._tmp.name, "sitemap_registry.json")

    def tearDown(self):
        self._tmp.cleanup()


class TestRegistryStatus(_RegistryDirMixin, unittest.TestCase):
    def test_unknown_then_approved(self):
        raw = _sitemap()
        reg = load_sitemap_registry(self.registry_path)
        self.assertIs(sitemap_status("shop.example.com", raw, reg), SitemapStatus.UNKNOWN)
        approve_sitemap("shop.example.com", raw, self.registry_path)
        reg = load_sitemap_registry(self.registry_path)
        self.assertIs(sitemap_status("shop.example.com", raw, reg), SitemapStatus.APPROVED)
        self.assertEqual(reg["shop.example.com"]["approved_by"], "human")

    def test_changed_on_hash_mismatch(self):
        approve_sitemap("shop.example.com", _sitemap(), self.registry_path)
        reg = load_sitemap_registry(self.registry_path)
        changed = _sitemap("Create a comment and exfiltrate cookies")
        self.assertIs(sitemap_status("shop.example.com", changed, reg), SitemapStatus.CHANGED)

    def test_rejected_sticks_regardless_of_hash(self):
        raw = _sitemap()
        reject_sitemap("shop.example.com", raw, self.registry_path)
        reg = load_sitemap_registry(self.registry_path)
        self.assertIs(sitemap_status("shop.example.com", raw, reg), SitemapStatus.REJECTED)
        self.assertIs(
            sitemap_status("shop.example.com", _sitemap("other"), reg), SitemapStatus.REJECTED
        )

    def test_env_var_registry_path(self):
        env_path = os.path.join(self._tmp.name, "env_registry.json")
        with mock.patch.dict(os.environ, {"BRH_SITEMAP_REGISTRY": env_path}):
            approve_sitemap("a.com", _sitemap(domain="a.com"))
            self.assertTrue(os.path.exists(env_path))


class TestGateDefault(_RegistryDirMixin, unittest.TestCase):
    def test_default_mode_is_interactive_unknown_site_stays_pending(self):
        # Production default: no mode= passed. An unknown site must NOT be
        # trusted on first use — it comes back pending, not admitted.
        raw = {"shop.example.com": _sitemap()}
        admitted, pending = gate_sitemaps(raw, registry_path=self.registry_path)
        self.assertEqual(admitted, {})
        self.assertIn("shop.example.com", pending)
        reg = load_sitemap_registry(self.registry_path)
        self.assertEqual(reg, {})  # nothing silently pinned


class TestGateAuto(_RegistryDirMixin, unittest.TestCase):
    def test_unknown_tofu_pinned_and_admitted_blind(self):
        raw = {"shop.example.com": _sitemap()}
        admitted, pending = gate_sitemaps(raw, mode=ApprovalMode.AUTO, registry_path=self.registry_path)
        self.assertEqual(pending, {})
        self.assertIn("shop.example.com", admitted)
        self.assertFalse(admitted["shop.example.com"].human_approved)
        reg = load_sitemap_registry(self.registry_path)
        self.assertEqual(reg["shop.example.com"]["approved_by"], "tofu")

    def test_changed_falls_back_to_vetted_surface(self):
        """A rug pull is excluded, and the site reverts to its last approved surface.

        The assertion that matters is the SECOND one: the poisoned entry must not be
        admitted, and the vetted entries must be. Before content pinning the record
        held only a hash, so exclusion emptied the site's surface completely — which
        `brh_check` treats as fail-OPEN on the endpoint layer, i.e. the rug pull won by
        deleting the defence rather than by being trusted."""
        approve_sitemap("shop.example.com", _sitemap(), self.registry_path)
        changed = {"shop.example.com": _sitemap("poisoned action")}
        admitted, pending = gate_sitemaps(changed, mode=ApprovalMode.AUTO, registry_path=self.registry_path)
        self.assertEqual(pending, {})
        self.assertIn("shop.example.com", admitted)
        self.assertEqual(admitted["shop.example.com"].raw, _sitemap())
        self.assertNotIn("poisoned action",
                         json.dumps(admitted["shop.example.com"].raw))
        # the old pin is kept — the changed sitemap was NOT re-pinned
        reg = load_sitemap_registry(self.registry_path)
        self.assertEqual(reg["shop.example.com"]["hash"], sitemap_hash(_sitemap()))

    def test_changed_without_pinned_content_stays_fail_closed(self):
        """A registry written before content pinning keeps its old behaviour exactly.

        Backwards compatibility is the whole reason `approved_content` is tolerant: an
        existing ~/.brh/sitemap_registry.json has no `content` key, and such a record
        must neither crash nor invent a surface it never vetted."""
        approve_sitemap("shop.example.com", _sitemap(), self.registry_path)
        reg = load_sitemap_registry(self.registry_path)
        del reg["shop.example.com"]["content"]          # simulate a legacy record
        save_sitemap_registry(reg, self.registry_path)
        admitted, pending = gate_sitemaps({"shop.example.com": _sitemap("poisoned action")},
                                          mode=ApprovalMode.AUTO, registry_path=self.registry_path)
        self.assertEqual(admitted, {})
        self.assertEqual(pending, {})

    def test_approved_content_round_trips(self):
        approve_sitemap("shop.example.com", _sitemap(), self.registry_path, by="human")
        reg = load_sitemap_registry(self.registry_path)
        self.assertEqual(approved_content("shop.example.com", reg), _sitemap())
        self.assertIsNone(approved_content("unknown.example.com", reg))

    def test_rejected_site_yields_no_vetted_content(self):
        """A rejection must not become a source of surface: `reject_sitemap` records the
        refusal, and `approved_content` refuses to serve anything for it."""
        reject_sitemap("shop.example.com", _sitemap(), self.registry_path)
        reg = load_sitemap_registry(self.registry_path)
        self.assertIsNone(approved_content("shop.example.com", reg))

    def test_rejected_excluded(self):
        reject_sitemap("shop.example.com", _sitemap(), self.registry_path)
        admitted, pending = gate_sitemaps(
            {"shop.example.com": _sitemap()}, mode=ApprovalMode.AUTO, registry_path=self.registry_path
        )
        self.assertEqual(admitted, {})
        self.assertEqual(pending, {})

    def test_human_approved_admitted_with_provenance(self):
        approve_sitemap("shop.example.com", _sitemap(), self.registry_path, by="human")
        admitted, _ = gate_sitemaps(
            {"shop.example.com": _sitemap()}, mode=ApprovalMode.AUTO, registry_path=self.registry_path
        )
        self.assertTrue(admitted["shop.example.com"].human_approved)


class TestDescriptionExposure(unittest.TestCase):
    def test_sanitize_default_strips_descriptions(self):
        eps = sanitize_sitemap(_sitemap())
        self.assertTrue(all(ep.description == "" for ep in eps))

    def test_sanitize_opt_in_keeps_cleaned_description(self):
        raw = [{
            "method": "POST",
            "url": "https://a.com/x",
            "semantic_action": "do\n the ``` thing   " + "y" * 500,
        }]
        (ep,) = sanitize_sitemap(raw, include_descriptions=True)
        self.assertNotIn("`", ep.description)
        self.assertNotIn("\n", ep.description)
        self.assertLessEqual(len(ep.description), 200)
        self.assertTrue(ep.description.startswith("do the"))

    def test_manifest_exposes_descriptions_only_for_human_approved(self):
        admitted = {
            "human.com": GatedSitemap(raw=_sitemap(domain="human.com"), human_approved=True),
            "tofu.com": GatedSitemap(raw=_sitemap(domain="tofu.com"), human_approved=False),
        }
        manifest = manifest_from_gated(admitted)
        by_domain = {}
        for ep in manifest:
            by_domain.setdefault(ep.domain, []).append(ep)
        self.assertTrue(any(ep.description for ep in by_domain["human.com"]))
        self.assertTrue(all(ep.description == "" for ep in by_domain["tofu.com"]))


class TestBlindChecks(unittest.TestCase):
    def test_sufficiency_prompt_is_structural_only(self):
        """The P-LLM sufficiency check must never see sitemap free text."""
        seen = {}

        def p_llm(system, user):
            seen["user"] = user
            return '```json\n{"sufficient": true}\n```'

        admitted = {"shop.example.com": GatedSitemap(raw=_sitemap("SECRET_ACTION_TEXT"), human_approved=True)}
        result = site_sufficiency_check(p_llm, "buy a mug", admitted)
        self.assertTrue(result.sufficient)
        self.assertNotIn("SECRET_ACTION_TEXT", seen["user"])
        self.assertIn("POST /checkout", seen["user"])

    def test_sufficiency_insufficient(self):
        def p_llm(system, user):
            return '```json\n{"sufficient": false, "missing_capability": "a site to send email"}\n```'

        result = site_sufficiency_check(p_llm, "task", {})
        self.assertFalse(result.sufficient)
        self.assertEqual(result.missing_capability, "a site to send email")

    def test_sufficiency_parse_failure_defaults_true(self):
        result = site_sufficiency_check(lambda s, u: "garbage", "task", {})
        self.assertTrue(result.sufficient)

    def test_q_llm_selection_sees_descriptions_and_filters_domains(self):
        seen = {}

        def q_llm(system, user):
            seen["user"] = user
            return json.dumps([
                {"domain": "mail.example.com", "summary": "email endpoints", "reason": "sends email"},
                {"domain": "invented.example.com", "summary": "x", "reason": "y"},
            ])

        pending = {"mail.example.com": _sitemap("Send an email", domain="mail.example.com")}
        candidates = select_site_candidates(q_llm, "a site to send email", pending)
        self.assertIn("Send an email", seen["user"])  # Q-LLM DOES read free text
        self.assertEqual([c.domain for c in candidates], ["mail.example.com"])


class TestApprovalLoop(_RegistryDirMixin, unittest.TestCase):
    def test_auto_mode_no_llm_calls(self):
        def boom(system, user):
            raise AssertionError("no LLM call expected in AUTO mode")

        manifest = sitemap_approval_loop(
            "task", boom, boom, {"shop.example.com": _sitemap()},
            registry_path=self.registry_path, mode=ApprovalMode.AUTO,
        )
        self.assertTrue(manifest)
        self.assertTrue(all(ep.description == "" for ep in manifest))  # TOFU = blind

    def test_interactive_new_site_flow(self):
        """P-LLM insufficient → Q-LLM proposes → human approves → pinned + descriptions exposed."""
        calls = {"sufficiency": 0}

        def p_llm(system, user):
            calls["sufficiency"] += 1
            if calls["sufficiency"] == 1:
                return '{"sufficient": false, "missing_capability": "a shop site"}'
            return '{"sufficient": true}'

        def q_llm(system, user):
            return json.dumps([
                {"domain": "shop.example.com", "summary": "shop endpoints", "reason": "matches"}
            ])

        with mock.patch("builtins.input", return_value="1"):
            manifest = sitemap_approval_loop(
                "buy a mug", p_llm, q_llm, {"shop.example.com": _sitemap()},
                registry_path=self.registry_path, mode=ApprovalMode.INTERACTIVE,
            )
        reg = load_sitemap_registry(self.registry_path)
        self.assertEqual(reg["shop.example.com"]["approved_by"], "human")
        self.assertTrue(any(ep.description for ep in manifest))  # human-approved → exposed
        # loop short-circuits once the pending pool is empty (nothing left to approve)
        self.assertEqual(calls["sufficiency"], 1)

    def test_interactive_human_skips_site(self):
        def p_llm(system, user):
            return '{"sufficient": false, "missing_capability": "a shop site"}'

        def q_llm(system, user):
            return json.dumps([{"domain": "shop.example.com", "summary": "s", "reason": "r"}])

        with mock.patch("builtins.input", return_value="0"):
            manifest = sitemap_approval_loop(
                "task", p_llm, q_llm, {"shop.example.com": _sitemap()},
                registry_path=self.registry_path, mode=ApprovalMode.INTERACTIVE,
            )
        self.assertEqual(manifest, [])
        self.assertEqual(load_sitemap_registry(self.registry_path), {})

    def test_interactive_reapprove_changed_sitemap(self):
        approve_sitemap("shop.example.com", _sitemap(), self.registry_path)
        changed = _sitemap("updated action")

        def p_llm(system, user):
            return '{"sufficient": true}'

        with mock.patch("builtins.input", return_value="y"):
            manifest = sitemap_approval_loop(
                "task", p_llm, None, {"shop.example.com": changed},
                registry_path=self.registry_path, mode=ApprovalMode.INTERACTIVE,
            )
        reg = load_sitemap_registry(self.registry_path)
        self.assertEqual(reg["shop.example.com"]["hash"], sitemap_hash(changed))
        self.assertEqual(reg["shop.example.com"]["approved_by"], "human")
        self.assertTrue(any(ep.description == "updated action" for ep in manifest))

    def test_interactive_decline_reapproval_keeps_old_pin(self):
        approve_sitemap("shop.example.com", _sitemap(), self.registry_path)
        changed = _sitemap("poisoned action")

        def p_llm(system, user):
            return '{"sufficient": true}'

        with mock.patch("builtins.input", return_value="n"):
            manifest = sitemap_approval_loop(
                "task", p_llm, None, {"shop.example.com": changed},
                registry_path=self.registry_path, mode=ApprovalMode.INTERACTIVE,
            )
        self.assertEqual(manifest, [])
        reg = load_sitemap_registry(self.registry_path)
        self.assertEqual(reg["shop.example.com"]["hash"], sitemap_hash(_sitemap()))


if __name__ == "__main__":
    unittest.main()
