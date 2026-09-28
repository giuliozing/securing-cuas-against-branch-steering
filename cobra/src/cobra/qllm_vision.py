"""Q-LLM vision primitives: VLM-backed perception the P-LLM plan can call.

Faithful to CaMeL's dual-LLM split. These are **quarantined-LLM** primitives:
they have no tool-calling power, they only read structured data out of an image
via `quarantined_llm.query_quarantined_vlm`, and the P-LLM uses them to perceive
a value it then branches on. Whatever a (possibly adversarial) image makes the
VLM read is untrusted data — the BRH/enforcer is what cross-checks the resulting
plan branch against the real HTTP payload.

Unlike the OSWorld UI primitives (`find`, `get_page_text` in
`cobra.interaction...base_ui_uitars`), which read a *live VM screenshot* and are
welded to the GUI executor, these read a **static image** supplied by a
`ScreenProvider` env dependency. So they run with no GUI/VM: vision drives the decision, HTTP carries the action, the BRH joins them.

Usage (P-LLM plan): call `read_number_from_screen("the product price")`; the
`screen` argument is injected from the environment (`Depends("screen")`) and is
not visible to the planner. The harness builds the environment with the current
`ScreenProvider` (the rendered page + the VLM model id).
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

import pydantic
from agentdojo import functions_runtime

_T = TypeVar("_T")


class ScreenProvider(pydantic.BaseModel):
    """The current static screen the vision primitives read, plus the VLM.

    Holds the image as base64 PNG and the quarantined VLM model id (e.g.
    ``"openrouter:openai/gpt-4o-mini"``). The harness sets a fresh provider per
    run/scenario; in a live OSWorld deployment this would instead wrap a live
    screenshot source. Kept a plain pydantic model so it can be an agentdojo
    environment field resolved by ``Depends("screen")``."""

    image_b64: str = ""
    vlm_model: str = ""

    def current_image_b64(self) -> str:
        return self.image_b64


def _embed_image(b64: str, mime: str = "png") -> str:
    """Wraps a base64 image in the sentinel `query_quarantined_vlm` understands.

    Same format as `base_ui_uitars.wrap_qvlm_b64`, re-defined here so this core
    module does not import the heavy OSWorld UI chain (pyautogui/gymnasium)."""
    return f"<<<'QVLMB64:{mime}'>>>\n{b64}\n<<<ENDQVLMB64>>>"


def _query_vlm(vlm_model: str, query: str, output_schema: type[_T]) -> _T:
    """Indirection over `query_quarantined_vlm` (lazy import keeps this module
    light and lets tests patch the VLM call without the provider chain)."""
    from cobra.quarantined_llm import query_quarantined_vlm

    return query_quarantined_vlm(vlm_model, query, output_schema)


# --- pure perception (image-source-agnostic) --------------------------------
# The reusable core: a base64 image + a question -> a parsed value. No env, no
# tool-calling — quarantined data extraction. Shared by the static screen
# primitives below (static `ScreenProvider`) and by the OSWorld suite's
# live-screenshot primitive (`base_ui.screenshot()`), so the perception logic
# lives in one place regardless of where the pixels come from.


def perceive(image_b64: str, instructions: str, output_schema: type[_T], vlm_model: str) -> _T:
    """Reads structured data out of `image_b64` via the quarantined VLM.

    Returns the parsed value, or raises `NotEnoughInformationError` if the image
    does not clearly contain it (the perception fails — the caller must not guess)."""
    query = f"{instructions}\n\n{_embed_image(image_b64)}"
    return _query_vlm(vlm_model, query, output_schema)


def perceive_number(image_b64: str, description: str, vlm_model: str) -> float:
    """`perceive` specialised to a single number (e.g. a displayed price)."""
    return perceive(
        image_b64, f"Read this single number from the screenshot: {description}.", float, vlm_model
    )


def perceive_text(image_b64: str, description: str, vlm_model: str) -> str:
    """`perceive` specialised to a short text (e.g. a currency code or a tier).

    The instruction pins the model to the *bare* value: without it some VLMs
    (e.g. UITARS) answer in a full sentence ("The invoice currency code is EUR."),
    which then fails a caller's exact-match branch (`currency == "EUR"`)."""
    return perceive(
        image_b64,
        f"Read this short text from the screenshot: {description}. "
        f"Respond with ONLY the exact value shown, verbatim — no sentence, no label, "
        f"no explanation, no quotes.",
        str, vlm_model,
    )


def read_number_from_screen(
    screen: Annotated[ScreenProvider, functions_runtime.Depends("screen")],
    description: str,
) -> float:
    """Reads a single numeric value shown on the current screen and returns it.

    Use this to perceive an on-screen number you need to decide on — e.g. a
    price displayed on a product page. The value is read from the image by a
    vision model; if the screen does not clearly show the requested number the
    call raises `NotEnoughInformationError` (do not guess a value yourself).

    :param description: what number to read, in plain words (e.g. "the product
        price in GBP, as a number without the currency symbol").
    """
    return perceive_number(screen.current_image_b64(), description, screen.vlm_model)


def read_text_from_screen(
    screen: Annotated[ScreenProvider, functions_runtime.Depends("screen")],
    description: str,
) -> str:
    """Reads a short piece of text shown on the current screen and returns it.

    Use this to perceive an on-screen string (a label, a currency code, a
    product name). Raises `NotEnoughInformationError` if the screen does not
    clearly show it.

    :param description: what text to read, in plain words.
    """
    return perceive_text(screen.current_image_b64(), description, screen.vlm_model)


ALL_PRIMITIVES: list[Any] = [read_number_from_screen, read_text_from_screen]
