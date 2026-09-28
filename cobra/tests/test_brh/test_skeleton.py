import unittest

from cobra.brh.skeleton import (
    InvalidPlanError,
    extract_single_code_block,
    extract_skeleton,
)

PLAN = '''\
```python
page = get_page_text("https://shop.example.com/usb-c-hub")
perceived_price = query_price(page)
print(perceived_price)
if perceived_price <= 50:
    product_id = find_product_id(page)
    order = place_order("payments.example.com", product_id, perceived_price)
    if order.confirmed:
        print("order ok")
    else:
        report_error(order)
else:
    print("too expensive")
```
'''


class ExtractCodeBlockTest(unittest.TestCase):
    def test_single_block(self):
        src = extract_single_code_block(PLAN)
        self.assertTrue(src.startswith("page ="))
        self.assertTrue(src.endswith('print("too expensive")'))

    def test_zero_blocks_raises(self):
        with self.assertRaises(InvalidPlanError):
            extract_single_code_block("no code here")

    def test_two_blocks_raises(self):
        with self.assertRaises(InvalidPlanError):
            extract_single_code_block("```python\na = 1\n```\n```python\nb = 2\n```")


class SkeletonTest(unittest.TestCase):
    def setUp(self):
        self.skeleton = extract_skeleton(PLAN)

    def test_branch_keys(self):
        self.assertEqual(
            set(self.skeleton.arms.keys()),
            {
                "if_L4_true",
                "if_L4_false",
                "if_L4_true.if_L7_true",
                "if_L4_true.if_L7_false",
            },
        )

    def test_branch_path_accumulates(self):
        nested = self.skeleton.arms["if_L4_true.if_L7_true"]
        self.assertEqual(nested.branch_path, ["root", "if_L4_true", "if_L7_true"])
        self.assertEqual(nested.lineno, 7)

    def test_trigger_var_from_condition(self):
        self.assertEqual(self.skeleton.arms["if_L4_true"].trigger_var, "perceived_price")
        self.assertEqual(self.skeleton.arms["if_L4_true.if_L7_true"].trigger_var, "order")

    def test_condition_comparison(self):
        comp = self.skeleton.arms["if_L4_true"].condition_comparison
        self.assertEqual(comp, {"op": "<=", "literal": 50, "operand": "perceived_price"})
        # attribute test has no constant comparison
        self.assertIsNone(self.skeleton.arms["if_L4_true.if_L7_true"].condition_comparison)

    def test_flipped_constant_comparison(self):
        sk = extract_skeleton("if 50 >= price:\n    pass", is_markdown=False)
        comp = sk.arms["if_L1_true"].condition_comparison
        self.assertEqual(comp, {"op": "<=", "literal": 50, "operand": "price"})

    def test_static_domains_per_arm(self):
        self.assertEqual(self.skeleton.root_static_domains, ["shop.example.com"])
        self.assertEqual(
            self.skeleton.arms["if_L4_true"].static_domains, ["payments.example.com"]
        )
        self.assertEqual(self.skeleton.arms["if_L4_false"].static_domains, [])

    def test_called_functions_per_arm(self):
        self.assertIn("get_page_text", self.skeleton.root_called_functions)
        self.assertIn("place_order", self.skeleton.arms["if_L4_true"].called_functions)
        self.assertNotIn("place_order", self.skeleton.root_called_functions)

    def test_all_arms_have_body_here(self):
        self.assertTrue(all(a.has_body for a in self.skeleton.arms.values()))

    def test_variable_names_collected(self):
        # Assignment targets only (used by the validator to vet "var:<name>");
        # function names and read-only uses are not variables.
        self.assertEqual(
            self.skeleton.variable_names,
            {"page", "perceived_price", "product_id", "order"},
        )

    def test_if_without_else_has_empty_false_arm(self):
        sk = extract_skeleton("if x:\n    do_thing()", is_markdown=False)
        self.assertFalse(sk.arms["if_L1_false"].has_body)
        self.assertTrue(sk.arms["if_L1_true"].has_body)

    def test_elif_nests_under_false_arm(self):
        src = "if a:\n    f()\nelif b:\n    g()"
        sk = extract_skeleton(src, is_markdown=False)
        self.assertIn("if_L1_false.if_L3_true", sk.arms)

    def test_if_inside_for_gets_own_key(self):
        src = 'for item in items:\n    fetch("https://api.example.com/x")\n    if item.ok:\n        save(item)'
        sk = extract_skeleton(src, is_markdown=False)
        self.assertIn("if_L3_true", sk.arms)
        self.assertEqual(sk.root_static_domains, ["api.example.com"])

    def test_filename_not_mistaken_for_domain(self):
        sk = extract_skeleton('save("results.json")\nopen_site("shop.example.com")', is_markdown=False)
        self.assertEqual(sk.root_static_domains, ["shop.example.com"])

    def test_domain_inside_sentence(self):
        sk = extract_skeleton('go("navigate to shop.example.com, please")', is_markdown=False)
        self.assertEqual(sk.root_static_domains, ["shop.example.com"])

    def test_syntax_error_raises(self):
        with self.assertRaises(InvalidPlanError):
            extract_skeleton("if while:", is_markdown=False)


class UrlHostCleaningTest(unittest.TestCase):
    """A `_URL_RE` capture is not yet a hostname (`skeleton._clean_url_host`).

    Both malformed shapes below were extracted from real plans and each one
    DEADLOCKED annotation: the validator demands every static domain appear in
    `allowed_domains` and simultaneously rejects it as not a bare hostname, so all
    three retries fail and the run degrades to `build_fallback` — domain-only state
    carrying an unmatchable host, with the field and endpoint layers silently gone.
    """

    def test_sentence_final_period_is_stripped_not_dropped(self):
        # Also seen: 'scholar.google.com.', 'dblp.org.', 'www.speedtest.net.'
        sk = extract_skeleton(
            'ask("Return the full URL to call on the base http://paybatch.local. ")',
            is_markdown=False)
        # Recovered, not discarded: the cross-check must still force the annotator to
        # authorise the domain the plan really talks to.
        self.assertEqual(sk.root_static_domains, ["paybatch.local"])

    def test_format_placeholder_host_is_dropped(self):
        # "http://procurement.local{0}".format(...)
        sk = extract_skeleton(
            'submit_to("http://procurement.local{0}".format("/procurement/submit"))',
            is_markdown=False)
        self.assertEqual(sk.root_static_domains, [])

    def test_ip_literal_still_extracted(self):
        # WASP runs entirely on loopback — this must NOT be filtered as "not a domain".
        sk = extract_skeleton('post("http://127.0.0.1:8023/g/p/-/notes")',
                              is_markdown=False)
        self.assertEqual(sk.root_static_domains, ["127.0.0.1"])

    def test_clean_url_unaffected(self):
        sk = extract_skeleton('post("http://paybatch.local/pay/settle")',
                              is_markdown=False)
        self.assertEqual(sk.root_static_domains, ["paybatch.local"])

    def test_deadlock_is_gone_end_to_end(self):
        """The property that matters: the most permissive annotation an LLM could
        write — authorise exactly the static domains — now validates."""
        from cobra.brh.schema import PlanConstraints
        from cobra.brh.validator import validate

        for src in ('ask("see http://paybatch.local. ")\npost("http://paybatch.local/x")',
                    'submit_to("http://procurement.local{0}".format("/s"))'):
            sk = extract_skeleton(src, is_markdown=False)
            constraints = PlanConstraints(
                plan_id="t", task="t",
                branches={k: {"description": "d",
                              "http_constraints": {
                                  "allowed_domains": list(sk.root_static_domains)}}
                          for k in sk.all_keys()},
            )
            self.assertEqual(validate(constraints, sk), [], msg=src)


if __name__ == "__main__":
    unittest.main()
