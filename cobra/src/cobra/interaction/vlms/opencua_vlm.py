"""
Use a locally hosted model.

Expects an OpenAI-compatible API server (your OpenCUA backend or vLLM) to be running,
e.g. launched with:

    uvicorn opencua_backend:app --port 15002
    # or
    vllm serve /path/to/huggingface/model --port 8000

Then create the OpenAI client with:
    openai.OpenAI(base_url="http://localhost:<port>/v1", api_key="...")

Set model="opencua" when targeting the OpenCUA backend (it enforces that).
"""
import logging
import random
from typing import Sequence, List

import openai
from openai.types.chat import ChatCompletionMessageParam

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from cobra.interaction.vlms.extended_types import ChatMessage, ChatAssistantMessage, text_content_block_from_string
from cobra.interaction.environments.base_ui_opencua import parse_action_to_structure_output_opencua

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

class InvalidModelOutputError(Exception):
    ...

# ---------------------------------------------------------------------------
# Helpers: coerce cobra.interaction ChatMessage -> OpenAI-compatible messages
# ---------------------------------------------------------------------------

def _coerce_messages_to_openai(messages: Sequence[ChatMessage]) -> list[ChatCompletionMessageParam]:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            parts = []
            for part in content:
                t = part.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif t == "image":
                    # vLLM's OpenAI server rejects an unknown content-part type, so the
                    # suite's {"type": "image", "image": "data:..."} block is normalized
                    # to the standard image_url form here.
                    img = part.get("image")
                    if isinstance(img, str) and img.startswith("data:image"):
                        parts.append({"type": "image_url", "image_url": {"url": img}})
                elif t == "image_url":
                    iu = part.get("image_url") or {}
                    url = iu.get("url")
                    if isinstance(url, str):
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    parts.append({"type": "text", "text": str(part)})
            out.append({"role": role, "content": parts})
            continue

        out.append({"role": role, "content": str(content)})
    return out


def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: List[ChatCompletionMessageParam] | Sequence[ChatMessage],
    temperature: float | None = 1.0,
    top_p: float | None = 0.9,
    max_tokens: int = 8192,
) -> str:
    """
    Posts to /v1/chat/completions using the OpenAI client.

    NOTE: Do NOT send unsupported fields (e.g., frequency_penalty, seed) to the OpenCUA backend,
    since its Pydantic schema would 422 on unknown keys.
    """
    oa_messages = _coerce_messages_to_openai(messages)

    logger.info("▶️  POST /chat/completions  model=%s  msgs=%d", model, len(oa_messages))
    try:
        resp = client.chat.completions.create(
            model=model,                      # For your OpenCUA server, this must be "opencua"
            messages=oa_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            #seed=random.randint(0, 1000000),
        )
        response = resp.choices[0].message.content
        logger.info("✅  backend replied %d chars", len(response) if response else -1)
    except Exception as e:
        logger.exception("OpenAI-compatible backend error: %s", e)
        response = ""

    if not response:
        raise InvalidModelOutputError("No response from model")
    return response


class OpenCUAVLM(BasePipelineElement):
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float | None = 0.0,
        top_p: float | None = 0.9,
    ) -> None:
        """
        Keep the same signature. Provide an OpenAI client configured with:
            openai.OpenAI(base_url="http://localhost:<port>/v1", api_key="...")

        For your OpenCUA backend, set model="opencua".
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:

        # Call the OpenAI-compatible server with properly coerced messages
        response = chat_completion_request(
            self.client,
            model=self.model,                 # <-- "opencua" for your backend
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=8192,
        )

        logger.info("Model response: %s", response)

        try:
            parsed_responses = parse_action_to_structure_output_opencua(response)
        except Exception:
            logger.error(
                "Error when parsing response from client, with response: %s",
                response,
            )
            # Same fallback as UITarsVLM: an empty-plan assistant message carrying the
            # raw text, so an unparseable step is a no-op instead of a NameError crash.
            parsed_responses = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(response)],
                tool_calls=[],
            )

        return query, runtime, env, [*messages, parsed_responses], extra_args