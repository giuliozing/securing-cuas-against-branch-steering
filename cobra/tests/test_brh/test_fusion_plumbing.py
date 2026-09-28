"""Plumbing tests for fused single-attempt planning (BRH_PLAN_FUSION).

Drives PrivilegedLLM._fusion_generate with a stubbed LLM (canned responses) and
asserts the candidate → merge → lint → repair/fallback flow, message-thread shapes,
and that the mode is inert when the flag is off.

Skips cleanly if the heavy privileged_llm import chain (openai/anthropic/agentdojo)
is unavailable — e.g. under the system-Python test run; execute it from the project
venv to run these.
"""

import unittest

try:
    import agentdojo.task_suite  # noqa: F401 — breaks agentdojo's circular init when
    # privileged_llm would otherwise be the first agentdojo importer.
    from cobra.pipeline_elements import privileged_llm as pllm

    _ad_types = pllm.ad_types
    _EmptyEnv = pllm.functions_runtime.EmptyEnv
    HAVE_DEPS = True
except Exception:  # pragma: no cover - dependency-gated
    HAVE_DEPS = False

GOOD_PLAN = 'ok = check_done("the artifact")\nif ok:\n    mark_done()'
GOOD_FUSED = (
    'res = call_mcp_tool(name="t", arguments={})\n'
    'mcp_failed = res.startswith("mcp_tool_error")\n'
    "ok = False\n"
    "if not mcp_failed:\n"
    '    ok = check_done("the artifact")\n'
    "if mcp_failed or not ok:\n"
    '    press("esc")\n'
    '    locate_and_click("Settings")\n'
    '    ok = check_done("the artifact")\n'
    "if ok:\n"
    "    mark_done()\n"
)
HARD_BROKEN = "while True:\n    pass"


if HAVE_DEPS:

    class _FakeLLM:
        """Returns canned assistant texts in order; records every message thread."""

        name = "fake"
        model = "fake"

        def __init__(self, responses):
            self.responses = list(responses)
            self.threads = []

        def query(self, query, runtime, messages=(), extra_args=None):
            self.threads.append(list(messages))
            text = self.responses.pop(0) if self.responses else ""
            reply = _ad_types.ChatAssistantMessage(
                role="assistant",
                content=[_ad_types.text_content_block_from_string(text)],
                tool_calls=None,
            )
            return query, runtime, None, [reply], {}

    def _make_pllm(responses, k=5):
        """Bare PrivilegedLLM via __new__ — only the attributes fusion touches."""
        p = object.__new__(pllm.PrivilegedLLM)
        p.llm = _FakeLLM(responses)
        p.dummy_runtime = None
        p.last_token_usage = None
        p._task_planning_in = 0
        p._task_planning_out = 0
        p._path = None
        p.max_attempts = 1
        p.fusion_enabled = True
        p.fusion_k = k
        p._fusion_regen_used = False
        return p

    def _base_messages():
        return [
            _ad_types.ChatSystemMessage(
                role="system",
                content=[_ad_types.text_content_block_from_string("SYSTEM")],
            ),
            _ad_types.ChatUserMessage(
                role="user", content=[_ad_types.text_content_block_from_string("TASK")]
            ),
        ]


@unittest.skipUnless(HAVE_DEPS, "privileged_llm import chain unavailable")
class FusionGenerateTest(unittest.TestCase):
    def test_happy_path_five_candidates_one_merge(self):
        p = _make_pllm([GOOD_PLAN] * 5 + [GOOD_FUSED])
        out = p._fusion_generate("TASK", _base_messages(), _EmptyEnv())
        self.assertEqual(out, GOOD_FUSED)
        # 5 candidate calls + 1 merge call, no repairs.
        self.assertEqual(len(p.llm.threads), 6)
        # Each candidate thread = base(2) + strategy directive(1).
        for t in p.llm.threads[:5]:
            self.assertEqual(len(t), 3)
            self.assertEqual(t[-1]["role"], "user")
        # Merge thread = base(2) + merge message with all candidate blocks.
        merge_text = _ad_types.get_text_content_as_str(p.llm.threads[5][-1]["content"])
        for name, _ in pllm._FUSION_STRATEGIES:
            self.assertIn(f"[{name}]", merge_text)

    def test_directives_follow_strategy_order(self):
        p = _make_pllm([GOOD_PLAN] * 5 + [GOOD_FUSED])
        p._fusion_generate("TASK", _base_messages(), _EmptyEnv())
        for i, (_, directive) in enumerate(pllm._FUSION_STRATEGIES):
            sent = _ad_types.get_text_content_as_str(p.llm.threads[i][-1]["content"])
            self.assertEqual(sent, directive)

    def test_fusion_k_limits_candidates(self):
        p = _make_pllm([GOOD_PLAN] * 3 + [GOOD_FUSED], k=3)
        out = p._fusion_generate("TASK", _base_messages(), _EmptyEnv())
        self.assertEqual(out, GOOD_FUSED)
        self.assertEqual(len(p.llm.threads), 4)  # 3 candidates + 1 merge

    def test_repair_round_fixes_bad_merge(self):
        # Merge emits a hard-broken plan, first repair emits a good one.
        p = _make_pllm([GOOD_PLAN] * 5 + [HARD_BROKEN, GOOD_FUSED])
        out = p._fusion_generate("TASK", _base_messages(), _EmptyEnv())
        self.assertEqual(out, GOOD_FUSED)
        self.assertEqual(len(p.llm.threads), 7)  # 5 + merge + 1 repair
        repair_text = _ad_types.get_text_content_as_str(p.llm.threads[6][-1]["content"])
        self.assertIn("Fix ONLY the", repair_text)
        self.assertIn("`while`", repair_text)

    def test_fallback_to_best_candidate_when_repairs_exhausted(self):
        # Merge + both repairs stay hard-broken → fallback = gui_direct candidate.
        gui_direct_code = 'hotkey("ctrl", "l")\nok = check_done("d")\nif ok:\n    mark_done()'
        cands = [GOOD_PLAN, gui_direct_code, GOOD_PLAN, GOOD_PLAN, GOOD_PLAN]
        p = _make_pllm(cands + [HARD_BROKEN, HARD_BROKEN, HARD_BROKEN])
        out = p._fusion_generate("TASK", _base_messages(), _EmptyEnv())
        self.assertEqual(out, gui_direct_code)
        self.assertEqual(len(p.llm.threads), 8)  # 5 + merge + 2 repairs

    def test_soft_findings_do_not_block(self):
        # Fused plan misses the verify-before-done (L3 soft) but is executable.
        soft = 'locate_and_click("x")\nmark_done()'
        p = _make_pllm([GOOD_PLAN] * 5 + [soft, soft, soft])
        out = p._fusion_generate("TASK", _base_messages(), _EmptyEnv())
        self.assertEqual(out, soft)  # accepted after repair rounds, still executable

    def test_no_candidates_returns_none(self):
        p = _make_pllm([""] * 5)
        self.assertIsNone(p._fusion_generate("TASK", _base_messages(), _EmptyEnv()))


@unittest.skipUnless(HAVE_DEPS, "privileged_llm import chain unavailable")
class FusionRegenerateTest(unittest.TestCase):
    def _error(self):
        exc = pllm.interpreter.CaMeLException(
            exception=ValueError("boom"),
            nodes=[__import__("ast").parse("x = 1").body[0]],
            dependencies=(),
        )
        return exc

    def test_regenerate_returns_linted_plan(self):
        p = _make_pllm([GOOD_FUSED])
        out = p._fusion_regenerate("TASK", "x = 1", self._error(), _base_messages(), _EmptyEnv())
        self.assertEqual(out, GOOD_FUSED)
        # Thread = base(2) + make_error_messages(2) + regen note(1).
        self.assertEqual(len(p.llm.threads[0]), 5)
        note = _ad_types.get_text_content_as_str(p.llm.threads[0][-1]["content"])
        self.assertIn("fusion regeneration", note)

    def test_regenerate_rejects_hard_broken(self):
        p = _make_pllm([HARD_BROKEN])
        out = p._fusion_regenerate("TASK", "x = 1", self._error(), _base_messages(), _EmptyEnv())
        self.assertIsNone(out)

    def test_read_only_tool_set_matches_suite_expectations(self):
        for name in ("find", "check_done", "verify_hypothesis", "wait", "no_op"):
            self.assertIn(name, pllm._FUSION_READ_ONLY_TOOLS)
        for name in ("click", "type_text", "hotkey", "call_mcp_tool", "mark_fail",
                     "mark_done", "run_single_uitars", "locate_and_click"):
            self.assertNotIn(name, pllm._FUSION_READ_ONLY_TOOLS)


if __name__ == "__main__":
    unittest.main()
