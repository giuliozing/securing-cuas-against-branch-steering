"""Unit tests for the Q-LLM vision primitives (`cobra.qllm_vision`).

These need agentdojo (the `Depends` annotation), so they run under the project
venv (`uv run`), unlike the stdlib `tests/test_brh` suite. The VLM call is
patched out (`_query_vlm`), so no API key / provider chain is touched: we check
the primitives embed the image correctly and pass the right schema/model.
"""

import unittest
from unittest import mock

from cobra import qllm_vision
from cobra.qllm_vision import (
    ScreenProvider,
    _embed_image,
    read_number_from_screen,
    read_text_from_screen,
)


class _Boom(Exception):
    pass


class QllmVisionTest(unittest.TestCase):
    def _screen(self):
        return ScreenProvider(image_b64="BASE64IMG", vlm_model="openrouter:openai/gpt-4o-mini")

    def test_read_number_embeds_image_and_returns_float(self):
        captured = {}

        def fake(vlm, query, schema):
            captured.update(vlm=vlm, query=query, schema=schema)
            return 42.99

        with mock.patch.object(qllm_vision, "_query_vlm", fake):
            out = read_number_from_screen(self._screen(), "the product price")

        self.assertEqual(out, 42.99)
        self.assertEqual(captured["vlm"], "openrouter:openai/gpt-4o-mini")
        self.assertIs(captured["schema"], float)
        self.assertIn("BASE64IMG", captured["query"])
        self.assertIn("<<<'QVLMB64:png'>>>", captured["query"])
        self.assertIn("the product price", captured["query"])

    def test_read_text_uses_str_schema(self):
        captured = {}

        def fake(vlm, query, schema):
            captured["schema"] = schema
            return "GBP"

        with mock.patch.object(qllm_vision, "_query_vlm", fake):
            out = read_text_from_screen(self._screen(), "the currency")
        self.assertEqual(out, "GBP")
        self.assertIs(captured["schema"], str)

    def test_embed_image_uses_the_vlm_sentinel(self):
        wrapped = _embed_image("ABC123")
        self.assertTrue(wrapped.startswith("<<<'QVLMB64:png'>>>"))
        self.assertIn("ABC123", wrapped)
        self.assertTrue(wrapped.strip().endswith("<<<ENDQVLMB64>>>"))

    def test_perceive_number_pure_is_image_source_agnostic(self):
        captured = {}

        def fake(vlm, query, schema):
            captured.update(vlm=vlm, query=query, schema=schema)
            return 42.99

        with mock.patch.object(qllm_vision, "_query_vlm", fake):
            out = qllm_vision.perceive_number("IMGB64", "the price", "vlm-x")
        self.assertEqual(out, 42.99)
        self.assertEqual(captured["vlm"], "vlm-x")
        self.assertIs(captured["schema"], float)
        self.assertIn("IMGB64", captured["query"])
        self.assertIn("the price", captured["query"])

    def test_live_screenshot_source_reuses_the_same_core(self):
        # The OSWorld primitive (tomorrow) reads base_ui.screenshot() then calls
        # the same pure `perceive_*` — emulate that a live image source flows
        # through the one shared core, no static ScreenProvider involved.
        class FakeLiveUI:
            def screenshot_b64(self):
                return "LIVEPIXELS"

        ui = FakeLiveUI()
        with mock.patch.object(qllm_vision, "_query_vlm", lambda v, q, s: 7.0):
            out = qllm_vision.perceive_number(ui.screenshot_b64(), "the price", "vlm")
        self.assertEqual(out, 7.0)

    def test_perception_errors_propagate(self):
        # The primitive must not swallow a failed read (e.g. blank screen ->
        # NotEnoughInformationError): it propagates so the plan fails closed.
        def boom(vlm, query, schema):
            raise _Boom()

        with mock.patch.object(qllm_vision, "_query_vlm", boom):
            with self.assertRaises(_Boom):
                read_number_from_screen(self._screen(), "the price")


if __name__ == "__main__":
    unittest.main()
