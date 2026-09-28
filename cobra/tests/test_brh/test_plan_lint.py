"""Unit tests for the fused single-attempt plan lint (Appendix C of the paper).

Stdlib-only — plan_lint has no heavy dependencies, so these always run under the
system-Python test_brh suite.
"""

import unittest

from cobra.interaction.environments import plan_lint


GOOD_FUSED_PLAN = '''\
print("FUSION_PHASE:mcp:start")
res = call_mcp_tool(name="os.set_wallpaper", arguments={"path": "/tmp/a.png"})
mcp_failed = res.startswith("mcp_tool_error")
ok = False
if not mcp_failed:
    ok = check_done("the wallpaper is now /tmp/a.png")
if mcp_failed or not ok:
    print("FUSION_PHASE:g1:start")
    press("esc")
    locate_and_click("Settings")
    type_text("/tmp/a.png")
    ok = check_done("the wallpaper is now /tmp/a.png")
    if not ok:
        print("FUSION_PHASE:g2:start")
        press("esc")
        find_element_by_text("Appearance")
        click(0.5, 0.5)
        ok = check_done("the wallpaper is now /tmp/a.png")
if ok:
    print("FUSION_PHASE:terminal:ok")
    mark_done()
'''


class ExtractCodeTest(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(plan_lint.extract_code("```python\nx = 1\n```"), "x = 1\n")

    def test_fenced_no_lang(self):
        self.assertEqual(plan_lint.extract_code("```\nx = 1\n```"), "x = 1\n")

    def test_raw_passthrough(self):
        self.assertEqual(plan_lint.extract_code("x = 1"), "x = 1")

    def test_empty(self):
        self.assertEqual(plan_lint.extract_code(""), "")


class L1ForbiddenTest(unittest.TestCase):
    def test_good_plan_is_clean(self):
        res = plan_lint.lint_plan(GOOD_FUSED_PLAN)
        self.assertEqual(res.hard, [])
        self.assertEqual(res.repair, [])
        self.assertTrue(res.ok)
        self.assertTrue(res.executable)

    def test_def_is_hard(self):
        res = plan_lint.lint_plan("def f():\n    pass\nf()")
        self.assertTrue(any("`def`" in v for v in res.hard))
        self.assertFalse(res.executable)

    def test_while_is_hard(self):
        res = plan_lint.lint_plan("while True:\n    wait(1)")
        self.assertTrue(any("`while`" in v for v in res.hard))

    def test_try_is_hard(self):
        res = plan_lint.lint_plan("try:\n    click(1, 1)\nexcept Exception:\n    pass")
        self.assertTrue(any("try/except" in v for v in res.hard))

    def test_lambda_import_class_are_hard(self):
        res = plan_lint.lint_plan("import os\nf = lambda x: x\nclass A:\n    pass")
        joined = "\n".join(res.hard)
        self.assertIn("import", joined)
        self.assertIn("lambda", joined)
        self.assertIn("class", joined)

    def test_syntax_error_is_hard(self):
        res = plan_lint.lint_plan("if x\n    pass")
        self.assertTrue(any("L4-parse" in v for v in res.hard))
        self.assertFalse(res.executable)


class L2McpContractTest(unittest.TestCase):
    def test_discarded_result(self):
        code = 'call_mcp_tool(name="t", arguments={})\nlocate_and_click("x")\nmark_done()'
        res = plan_lint.lint_plan(code)
        self.assertTrue(any("discarded" in v for v in res.repair))

    def test_untested_result(self):
        code = 'r = call_mcp_tool(name="t", arguments={})\nlocate_and_click("x")\nmark_done()'
        res = plan_lint.lint_plan(code)
        self.assertTrue(any("never tested" in v for v in res.repair))

    def test_derived_flag_counts_as_tested(self):
        code = (
            'r = call_mcp_tool(name="t", arguments={})\n'
            'bad = r.startswith("mcp_tool_error")\n'
            'if bad:\n    locate_and_click("x")\n'
        )
        res = plan_lint.lint_plan(code)
        self.assertFalse(any("never tested" in v for v in res.repair))

    def test_no_gui_after_mcp(self):
        code = (
            'r = call_mcp_tool(name="t", arguments={})\n'
            'if r.startswith("mcp_tool_error"):\n    wait(1)\n'
            'mark_done()'
        )
        res = plan_lint.lint_plan(code)
        self.assertTrue(any("no GUI action" in v for v in res.repair))

    def test_no_mcp_no_l2(self):
        res = plan_lint.lint_plan('locate_and_click("x")\nok = check_done("d")\nif ok:\n    mark_done()')
        self.assertFalse(any(v.startswith("L2") for v in res.repair))

    def test_mcp_budget(self):
        code = "\n".join(
            f'r{i} = call_mcp_tool(name="t", arguments={{}})' for i in range(5)
        ) + '\nif r0.startswith("e"):\n    locate_and_click("x")\n'
        res = plan_lint.lint_plan(code)
        self.assertTrue(any("budget 4" in v for v in res.repair))


class L3VerifyTest(unittest.TestCase):
    def test_mark_done_without_verify(self):
        res = plan_lint.lint_plan('locate_and_click("x")\nmark_done()')
        self.assertTrue(any("mark_done" in v for v in res.repair))

    def test_mark_done_with_verify_ok(self):
        res = plan_lint.lint_plan('ok = check_done("d")\nif ok:\n    mark_done()')
        self.assertEqual(res.repair, [])

    def test_mark_fail_not_last(self):
        code = 'found = check_done("exists")\nif not found:\n    mark_fail()\n    wait(1)'
        res = plan_lint.lint_plan(code)
        self.assertTrue(any("LAST statement" in v for v in res.repair))

    def test_mark_fail_last_ok(self):
        code = 'found = check_done("exists")\nif not found:\n    print("absent")\n    mark_fail()'
        res = plan_lint.lint_plan(code)
        self.assertFalse(any("mark_fail" in v for v in res.repair))


class L4BudgetTest(unittest.TestCase):
    def test_statement_budget(self):
        code = "\n".join(f"x{i} = {i}" for i in range(130))
        res = plan_lint.lint_plan(code)
        self.assertTrue(any("budget 120" in v for v in res.repair))


class RepairMessageTest(unittest.TestCase):
    def test_lists_all_violations(self):
        res = plan_lint.lint_plan('call_mcp_tool(name="t", arguments={})\nmark_done()')
        msg = plan_lint.format_repair_message(res)
        self.assertIn("Fix ONLY the", msg)
        for v in res.hard + res.repair:
            self.assertIn(v, msg)


class FallbackTest(unittest.TestCase):
    GOOD = 'ok = check_done("d")\nif ok:\n    mark_done()'
    BROKEN = "while True:\n    pass"
    UNPARSABLE = "if x\n  pass"

    def test_priority_order(self):
        cands = [("mcp_first", self.GOOD), ("gui_direct", self.GOOD)]
        name, _ = plan_lint.pick_fallback(cands)
        self.assertEqual(name, "gui_direct")

    def test_skips_hard_broken(self):
        cands = [("gui_direct", self.BROKEN), ("mcp_first", self.GOOD)]
        name, _ = plan_lint.pick_fallback(cands)
        self.assertEqual(name, "mcp_first")

    def test_last_resort_parses(self):
        cands = [("gui_direct", self.UNPARSABLE), ("mcp_first", self.BROKEN)]
        name, _ = plan_lint.pick_fallback(cands)
        self.assertEqual(name, "mcp_first")  # BROKEN parses; UNPARSABLE does not

    def test_none_when_nothing_parses(self):
        self.assertIsNone(plan_lint.pick_fallback([("gui_direct", self.UNPARSABLE)]))

    def test_unknown_strategy_names_still_considered(self):
        cands = [("custom", self.GOOD)]
        name, _ = plan_lint.pick_fallback(cands)
        self.assertEqual(name, "custom")


# --------------------------------------------------------------------------------
# F1b / F2b — the two mechanical gap-FAIL classes of fused plans.
# Fixtures are real shapes observed in traces.
# --------------------------------------------------------------------------------

# Task 24 (chrome): terminal verdict = OR over stale phase flags, incl. an
# opener-tool "success", while g1 and g2 both failed.
C1_AGGREGATE_PLAN = '''\
res = call_mcp_tool(name="google_chrome.open_privacy_settings", arguments={})
mcp_failed = res.startswith("mcp_tool_error")
mcp_success = False
if not mcp_failed:
    mcp_success = check_done("the privacy settings page is open")
press("esc")
locate_and_click("Clear data")
g1_success = check_done("the browsing data was cleared")
press("esc")
find_element_by_text("Settings")
g2_success = check_done("the browsing data was cleared")
any_success = mcp_success or g1_success or g2_success
if any_success:
    mark_done()
'''

# Task 165 (calc): an errored probe sets abort=True and the flag disables G1/G2,
# so the plan dies without ever touching the GUI (1 screenshot).
C2_ABORT_PLAN = '''\
info = call_mcp_tool(name="libreoffice_calc.env_info", arguments={})
column_present = verify_hypothesis("the 'spent' column exists", info)
abort = False
if not column_present:
    abort = True
if not abort:
    locate_and_click("Column C")
    type_text("=SUM(B2:B10)")
ok = check_done("the spent column is filled")
if ok:
    mark_done()
'''


class L3bAggregateVerdictTest(unittest.TestCase):
    def test_task24_or_aggregate_is_flagged(self):
        res = plan_lint.lint_plan(C1_AGGREGATE_PLAN)
        self.assertTrue(any("L3b" in v and "disjunction" in v for v in res.repair), res.repair)
        self.assertTrue(res.executable)  # repair severity: never blocks the run

    def test_inline_or_in_guard_is_flagged(self):
        res = plan_lint.lint_plan(
            'a = check_done("x")\nb = check_done("y")\nif a or b:\n    mark_done()'
        )
        self.assertTrue(any("L3b" in v for v in res.repair), res.repair)

    def test_fresh_verify_guard_is_clean(self):
        res = plan_lint.lint_plan('ok = check_done("the file was saved")\nif ok:\n    mark_done()')
        self.assertEqual([v for v in res.repair if "L3b" in v], [])

    def test_conjunction_with_last_verify_is_clean(self):
        res = plan_lint.lint_plan(
            'err = False\nok = check_done("saved")\nfinal_ok = ok and not err\n'
            'if final_ok:\n    mark_done()'
        )
        self.assertEqual([v for v in res.repair if "L3b" in v], [])

    def test_stale_flag_guard_is_flagged(self):
        res = plan_lint.lint_plan(
            'mcp_ok = check_done("dialog open")\nlocate_and_click("Clear")\n'
            'fresh = check_done("data cleared")\nif mcp_ok:\n    mark_done()'
        )
        self.assertTrue(any("L3b" in v and "most recent verification" in v
                            for v in res.repair), res.repair)

    def test_unconditional_mark_done_is_flagged(self):
        res = plan_lint.lint_plan('ok = check_done("saved")\nmark_done()')
        self.assertTrue(any("L3b" in v and "unconditional" in v for v in res.repair), res.repair)

    def test_good_fused_plan_has_no_l3b(self):
        res = plan_lint.lint_plan(GOOD_FUSED_PLAN)
        self.assertEqual([v for v in res.repair if "L3b" in v], [])


class L5GateFlagTest(unittest.TestCase):
    def test_task165_abort_flag_is_flagged(self):
        res = plan_lint.lint_plan(C2_ABORT_PLAN)
        self.assertTrue(any("L5" in v and "abort" in v for v in res.repair), res.repair)
        self.assertTrue(res.executable)

    def test_verify_derived_guard_is_clean(self):
        # `already_done` carries real evidence — a legitimate cross-phase carrier.
        res = plan_lint.lint_plan(
            'already_done = check_done("the setting is enabled")\n'
            'if not already_done:\n    locate_and_click("Enable")'
        )
        self.assertEqual([v for v in res.repair if "L5" in v], [])

    def test_mcp_derived_guard_is_clean(self):
        res = plan_lint.lint_plan(
            'res = call_mcp_tool(name="t", arguments={})\n'
            'mcp_failed = res.startswith("mcp_tool_error")\n'
            'if mcp_failed:\n    locate_and_click("Menu")\n'
            'ok = check_done("done")\nif ok:\n    mark_done()'
        )
        self.assertEqual([v for v in res.repair if "L5" in v], [])

    def test_good_fused_plan_has_no_l5(self):
        res = plan_lint.lint_plan(GOOD_FUSED_PLAN)
        self.assertEqual([v for v in res.repair if "L5" in v], [])

    def test_flag_reassigned_after_gui_is_not_a_gate(self):
        # `ok` is re-assigned during the GUI phase → not a pre-GUI gate flag.
        res = plan_lint.lint_plan(
            'ok = False\nif not ok:\n    locate_and_click("Go")\n    ok = True'
        )
        self.assertEqual([v for v in res.repair if "L5" in v], [])

    def test_each_gate_flag_reported_once(self):
        res = plan_lint.lint_plan(
            'abort = False\nif True:\n    abort = True\n'
            'if not abort:\n    locate_and_click("A")\n'
            'if not abort:\n    type_text("b")'
        )
        self.assertEqual(len([v for v in res.repair if "L5" in v]), 1, res.repair)


if __name__ == "__main__":
    unittest.main()
