"""Unit tests for BaseUI.screenshot's guest-outage handling.

Context: every perception primitive raised an opaque ``UnidentifiedImageError``
once the guest server went away. The capture
path now retries transient misses and, on a genuinely dead guest, raises a typed
error carrying the ENV_UNAVAILABLE marker so the attempt loop can attribute the
failure to infrastructure instead of to the plan.

LLM-free and VM-free: the controller is a stub.
"""

import io
import types
import unittest

import PIL.Image

# Import order matters here: agentdojo's suite registry
# first, then cobra.interpreter — importing base_ui_uitars or privileged_llm ahead of
# them hits pre-existing circular imports (cobra.interpreter.value ↔ cobra.capabilities,
# and agentdojo's banking suite ↔ its v1_1_1 tasks).
import agentdojo.benchmark  # noqa: F401 — import-order prerequisite
import cobra.interpreter.interpreter  # noqa: F401 — import-order prerequisite
from cobra.interaction.environments import base_ui_uitars as bui


def _png_bytes(color="red"):
    buf = io.BytesIO()
    PIL.Image.new("RGB", (8, 6), color).save(buf, "PNG")
    return buf.getvalue()


class _Controller:
    """Returns the queued payloads in order; the last one repeats forever."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def get_screenshot(self):
        self.calls += 1
        idx = min(self.calls - 1, len(self.payloads) - 1)
        return self.payloads[idx]


def _fake_ui(controller):
    """A minimal stand-in for BaseUI carrying only what screenshot() touches."""
    return types.SimpleNamespace(
        env=types.SimpleNamespace(controller=controller),
        history_images=[],
        max_history=3,
        image_height=0,
        image_width=0,
    )


class ScreenshotResilienceTest(unittest.TestCase):
    def setUp(self):
        # Keep the backoff out of the test's wall clock.
        self._delay = bui.SCREENSHOT_RETRY_DELAY
        bui.SCREENSHOT_RETRY_DELAY = 0.0

    def tearDown(self):
        bui.SCREENSHOT_RETRY_DELAY = self._delay

    def test_first_attempt_success_does_not_retry(self):
        ctrl = _Controller([_png_bytes()])
        ui = _fake_ui(ctrl)
        shot = bui.BaseUI.screenshot(ui)
        self.assertEqual(ctrl.calls, 1)
        self.assertTrue(shot.png_b64)
        self.assertEqual((ui.image_width, ui.image_height), (8, 6))

    def test_transient_none_is_retried(self):
        ctrl = _Controller([None, _png_bytes()])
        shot = bui.BaseUI.screenshot(_fake_ui(ctrl))
        self.assertEqual(ctrl.calls, 2)
        self.assertTrue(shot.png_b64)

    def test_undecodable_payload_is_retried(self):
        ctrl = _Controller([b"not-an-image", _png_bytes()])
        shot = bui.BaseUI.screenshot(_fake_ui(ctrl))
        self.assertEqual(ctrl.calls, 2)
        self.assertTrue(shot.png_b64)

    def test_dead_guest_raises_typed_marked_error(self):
        ctrl = _Controller([None])
        with self.assertRaises(bui.EnvironmentUnavailableError) as cm:
            bui.BaseUI.screenshot(_fake_ui(ctrl))
        self.assertEqual(ctrl.calls, bui.SCREENSHOT_RETRIES)
        # The marker must survive into the message: privileged_llm matches on the
        # string (no import coupling), so a rename there would silently break the
        # fusion regeneration guard.
        self.assertIn(bui.ENV_UNAVAILABLE_MARKER, str(cm.exception))

    def test_marker_matches_the_planner_side_constant(self):
        from cobra.pipeline_elements import privileged_llm

        self.assertEqual(bui.ENV_UNAVAILABLE_MARKER,
                         privileged_llm._ENV_UNAVAILABLE_MARKER)

    def test_add_appends_to_history_within_bound(self):
        ctrl = _Controller([_png_bytes()])
        ui = _fake_ui(ctrl)
        ui.max_history = 2
        for _ in range(3):
            bui.BaseUI.screenshot(ui, add=True)
        self.assertEqual(len(ui.history_images), 2)


if __name__ == "__main__":
    unittest.main()
