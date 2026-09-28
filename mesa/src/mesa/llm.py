"""Minimal ``llm_call(system_prompt, user_prompt) -> str`` abstraction.

Mirrors the injectable ``LLMCall`` signature used across ``cobra.brh`` so the
builder shares the project's convention. Default provider is **OpenRouter**
(OpenAI-compatible API); native OpenAI and Anthropic are also supported. If the
relevant API key is missing the factory returns ``None`` and callers fall back
to deterministic behaviour (see ``proposer.py``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

LLMCall = Callable[[str, str], str]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model ids are provider-qualified on OpenRouter ("<vendor>/<model>").
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5"
DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

DEFAULT_MODELS = {
    "openrouter": DEFAULT_OPENROUTER_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
}


def default_model(provider: str) -> str:
    return DEFAULT_MODELS.get((provider or "openrouter").lower(), DEFAULT_OPENROUTER_MODEL)


def make_llm(
    provider: str = "openrouter", model: Optional[str] = None
) -> Optional[LLMCall]:
    """Build an ``llm_call`` for the given provider, or ``None`` if unavailable.

    Returns ``None`` (rather than raising) when the relevant API key is missing,
    so the tool degrades gracefully to a no-LLM path.
    """
    provider = (provider or "openrouter").lower()

    if provider == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            return None
        return _make_openai_compatible(
            model or DEFAULT_OPENROUTER_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            extra_headers={
                "HTTP-Referer": "https://github.com/mesa",
                "X-Title": "mesa",
            },
        )
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        return _make_openai_compatible(model or DEFAULT_OPENAI_MODEL)
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        return _make_anthropic(model or DEFAULT_ANTHROPIC_MODEL)
    raise ValueError(f"unknown provider: {provider!r}")


def _make_openai_compatible(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> LLMCall:
    """OpenAI SDK client, optionally pointed at an OpenAI-compatible endpoint."""
    from openai import OpenAI

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    client = OpenAI(**kwargs)

    def call(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_headers=extra_headers or None,
        )
        if not resp.choices:
            raise ValueError(f"LLM returned no choices (model={model!r})")
        return resp.choices[0].message.content or ""

    return call


def _make_anthropic(model: str) -> LLMCall:
    import anthropic

    client = anthropic.Anthropic()

    def call(system_prompt: str, user_prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

    return call


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """Best-effort parse of a JSON value from an LLM response.

    Handles bare JSON, fenced ```json blocks, and leading/trailing prose by
    falling back to the first balanced ``{...}`` / ``[...]`` span. Raises
    ``ValueError`` if nothing parses.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response")

    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object/array found in LLM response")
