"""Hosted Kimi-K2.5 as the OSWorld Q-LLM (executor VLM).

Drop-in replacement for :class:`UITarsVLM` when the executor is a hosted
OpenAI-compatible chat model instead of the local UI-TARS vLLM.  The message
format, the UITARS action grammar and `parse_action_to_structure_output` are
reused verbatim, so `BaseUI` is unchanged: only the transport, the sampling
knobs and the coordinate convention differ.

Two behavioural differences vs. the local UITARS path:

1. **Reasoning off by default.**  K2.5 ships with reasoning ON, which measured
   38 s/step against 6.9 s with it off on an identical 1920x1080 OSWorld
   screenshot, for no demonstrated grounding benefit.  `OPENROUTER_REASONING`
   (off | low | medium | high | <int> | default) controls it.
2. **Normalised grounding.**  K2.5 answers with fractions of the screen
   (`0.018, 0.184`) where UI-TARS answers in smart-resized pixels.  The
   conversion lives in `BaseUI._coords_to_pixels` under `coord_mode="auto"`;
   this module only pins the coordinate space in the prompt (COORD_ADDENDUM)
   so the two defences are independent, exactly as in the ReAct agent.

Transport is `openai.OpenAI` against an arbitrary `base_url` (OpenRouter by
default), so the same class serves an Azure AI Foundry Kimi deployment by
setting KIMI_BASE_URL/KIMI_API_KEY.
"""

import logging
import os
import time
from collections.abc import Sequence

import openai

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from cobra.interaction.vlms.extended_types import (
    ChatAssistantMessage,
    ChatMessage,
    text_content_block_from_string,
)
from cobra.interaction.environments.base_ui_uitars import (
    parse_action_to_structure_output,
)

logger = logging.getLogger(__name__)


# Appended to the *system* message of every executor call.  UI-TARS was trained
# on this grammar and needs no such hint; a general-purpose model does.
COORD_ADDENDUM = (
    "\nIMPORTANT — coordinate space: the screenshot is exactly {w}x{h} pixels. "
    "Every coordinate inside start_box/end_box MUST be an absolute integer "
    "pixel value, with 0 <= x < {w} and 0 <= y < {h} "
    "(e.g. click(start_box='(1520,340)')). Never use normalised/relative "
    "coordinates in [0,1] and never use percentages.\n"
    "Answer with the exact 'Thought: ...\\nAction: ...' format requested, one "
    "Action per reply, and nothing else — no markdown fences, no commentary."
)


def parse_reasoning_env(value: str | None):
    """`OPENROUTER_REASONING` -> the OpenRouter `reasoning` payload field.

    off                    -> {"enabled": False}
    low | medium | high    -> {"effort": <value>}
    <int>                  -> {"max_tokens": <int>}
    default | None | ""    -> None (send nothing, provider default)
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("", "default"):
        return None
    if v == "off":
        return {"enabled": False}
    if v in ("low", "medium", "high"):
        return {"effort": v}
    if v.isdigit():
        return {"max_tokens": int(v)}
    raise ValueError(f"Unrecognised OPENROUTER_REASONING={value!r}")


class InvalidModelOutputError(Exception): ...


class KimiVLM(BasePipelineElement):
    """UITARS-grammar executor backed by a hosted chat model."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float | None = 0.0,
        top_p: float | None = 0.9,
        max_tokens: int = 4096,
        max_retries: int = 4,
        screen_size: tuple[int, int] = (1920, 1080),
        reasoning=None,
        usage_log_path: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.screen_width, self.screen_height = screen_size
        self.reasoning = reasoning
        self.usage_log_path = usage_log_path or os.getenv("KIMI_USAGE_LOG")

    # ------------------------------------------------------------------ #
    # prompt
    # ------------------------------------------------------------------ #
    def _with_coord_addendum(self, messages: list) -> list:
        """Append COORD_ADDENDUM to the first system message (or prepend one).

        Non-destructive: operates on a shallow copy so the caller's message
        list — reused across retries in BaseUI — is never mutated twice.
        """
        addendum = COORD_ADDENDUM.format(w=self.screen_width, h=self.screen_height)
        out = list(messages)
        for i, msg in enumerate(out):
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list):
                    new_content = list(content) + [{"type": "text", "text": addendum}]
                else:
                    new_content = f"{content}{addendum}"
                out[i] = {**msg, "content": new_content}
                return out
        return [{"role": "system", "content": [{"type": "text", "text": addendum}]}, *out]

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _log_usage(self, rec: dict) -> None:
        if not self.usage_log_path:
            return
        try:
            import fcntl
            import json

            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as exc:  # accounting must never kill a run
            logger.warning("usage log write failed: %s", exc)

    def _chat(
        self,
        messages: list,
        token_count_file: str | None,
        function_name: str,
    ) -> str:
        extra_body: dict = {"usage": {"include": True}}
        if self.reasoning is not None:
            extra_body["reasoning"] = self.reasoning

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body=extra_body,
                )
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "[KimiVLM] %s attempt %d/%d transport error: %s",
                    function_name, attempt, self.max_retries, exc,
                )
                time.sleep(min(2 ** attempt, 20))
                continue

            response = (completion.choices[0].message.content or "") if completion.choices else ""
            usage = getattr(completion, "usage", None)
            rec = {
                "ts": time.time(),
                "fn": function_name,
                "model": self.model,
                "latency_s": round(time.time() - t0, 3),
                "in": getattr(usage, "prompt_tokens", None),
                "out": getattr(usage, "completion_tokens", None),
                "cost": getattr(usage, "cost", None),
                "empty": not response.strip(),
            }
            self._log_usage(rec)
            if usage is not None and token_count_file is not None:
                with open(token_count_file, "a") as f:
                    f.write(
                        f"Model {self.model} Function {function_name or 'unknown'} - "
                        f"Input Tokens: {usage.prompt_tokens}, Output Tokens: "
                        f"{usage.completion_tokens}, Total Tokens: {usage.total_tokens}\n"
                    )

            if response.strip():
                logger.info("[KimiVLM] %s replied %d chars in %.1fs",
                            function_name, len(response), rec["latency_s"])
                return response

            # An empty completion is a provider hiccup, not a decision: retry it
            # rather than handing BaseUI an unparseable step.
            last_err = InvalidModelOutputError("empty completion")
            logger.warning("[KimiVLM] %s attempt %d/%d returned empty content",
                           function_name, attempt, self.max_retries)
            time.sleep(min(2 ** attempt, 20))

        raise InvalidModelOutputError(
            f"{self.model} produced no usable response after {self.max_retries} attempts: {last_err}"
        )

    # ------------------------------------------------------------------ #
    # pipeline element
    # ------------------------------------------------------------------ #
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
        token_count_file: str | None = None,
        function_name: str = "",
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        response = self._chat(
            self._with_coord_addendum(list(messages)),
            token_count_file=token_count_file,
            function_name=function_name,
        )
        logger.info("Model response: %s", response)
        try:
            parsed_responses = parse_action_to_structure_output(response)
        except Exception:
            logger.error("Error parsing response (no actionable plan), with response: %s", response)
            # Same contract as UITarsVLM: an unparseable Action block degrades to
            # an empty plan (the raw text is still returned as content) so the
            # caller treats the step as a no-op instead of crashing the plan.
            parsed_responses = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(response)],
                tool_calls=[],
            )
        return query, runtime, env, [*messages, parsed_responses], extra_args
