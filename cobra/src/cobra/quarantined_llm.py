# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os

from typing import TypeVar, Union, Sequence, cast, Literal
from pathlib import Path
import base64
import mimetypes
import httpx
import openai
from pydantic import BaseModel, Field, create_model
import pydantic_ai
from pydantic_ai import models
from pydantic_ai import BinaryContent
from google import genai
from google.genai import types as genai_types
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.openai import OpenAIProvider
from agentdojo.functions_runtime import FunctionsRuntime, EmptyEnv, Env
from agentdojo import agent_pipeline
from cobra.interaction.vlms.extended_types import ChatMessage
from cobra.interaction.vlms.gemini_vlm import GoogleVLM
import re
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
_T = TypeVar("_T")


class NotEnoughInformationError(Exception): ...


_SYSTEM_PROMPT = """\
You are a helpful assistant that assists a user to parse unstructured data into structured data. \
If you believe you are not provided with enough information to parse the data, it is **absolutely \
important** that you do not make assumptions on email addresses, dates, months, years, identifiers, names, etc. \
If you believe that you do not have enough information, set `have_enough_information` to false and the rest to dummy values. \
This is **extremely important** as wrong data cannot be detected! When asked for time data, do not specify the timezone.
"""


def query_quarantined_llm(
    llm: models.KnownModelName | models.Model,
    query: str,
    output_schema: type[_T],
    retries: int = 10,
) -> _T:
    """Queries a Large Language Model with `query` and returns the language model's output. It must be used to process unstructured data into structured one.

    It is absolutely imperative that you use this function to parse data whose structure you don't know insted of parsing using regular expressions and/or
    string manipulation.

    There is no need to specify the expected output format in the query itself as the format will be specified on the side with `output_schema`
    with the build-in API of the assistant.

    If not provided with enough information, the model will raise a `NotEnoughInformationError`. This means that you might
    need to provide the assistant with more information such as email addresses, dates, identifiers, names, etc. **DO NOT HESITATE TO PASS THE ENTIRE
    OBJECTS RETURN FROM THE TOOL FUNCTIONS IF THE MODEL DOES NOT HAVE ENOUGH INFORMATION**.

    :param query: a string with the query. Make sure to provide sufficient instructions to the AI assistant so that it can understand what it needs to do.
    Avoid just passing it tool outputs without additional instructions. Provide as much details as possible. Keep in mind that the assitant does not have
    direct access to the variables, so you need to insert **all the relevant information in the prompt**. When in doubt, do not hesitate to just provide full tool outputs, as long
    as they are provided with instructions on what to do. It is highly preferable to provide more information than necessary, rather than less information.
    :param output_schema: a Pydantic BaseModel class that specifies the expected output format from the model.
      The fields should have types as specific as possible to make sure the parsing is correct and accurate.
      allowed types are:
      - `int`
      - `str`
      - `float`
      - `bool`
      - `datetime.datetime` (assume `datetime` is imported from `datetime`)
      - `enum.Enum` classes
      - `pydantic.BaseModel` classes that you can define (assume that `BaseModel` is imported from `pydantic`) or are already defined in these instructions.
      - `pydantic.EmailStr` (assume that `EmailStr` is imported from `pydantic`)
    """

    enough_information = (
        bool,
        Field(
            description="Whether the prompt provided **you** enough information to perform the task. It is **absolutely** imperative to set this to false if you don't have enough information to solve this task. Do not make assumptions of any sorts."
        ),
    )

    if issubclass(output_schema, BaseModel):
        output_model = create_model(
            output_schema.__name__,
            __base__=output_schema,
            have_enough_information=enough_information,
        )
    else:
        output_model = create_model(
            "Result",
            output=(output_schema, Field(description="The requested value")),
            have_enough_information=enough_information,
        )
    #model = pydantic_ai.Agent(llm, result_type=output_model, retries=retries, system_prompt=_SYSTEM_PROMPT) # old for pydantic <v0.6.0
    model = pydantic_ai.Agent(llm, output_type=output_model, retries=retries, system_prompt=_SYSTEM_PROMPT) # new for pydantic >=v0.6.0

    #res = model.run_sync(query).data # old for pydantic <v0.6.0
    res = model.run_sync(query).output

    if isinstance(llm, str) and "gemini" in llm and "exp" in llm:
        time.sleep(6)

    if not res.have_enough_information:  # type: ignore
        raise NotEnoughInformationError()

    if issubclass(output_schema, BaseModel):
        return res  # type: ignore
    return res.output  # type: ignore

_T = TypeVar("_T")

# --------------------------------------------------------------------------- #
# Constants & exceptions                                                      #
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT_VLM = """\
You are a helpful assistant that extracts structured data from *{data_sources}*. \
If critical details are **not visible or readable in the supplied {data_sources}**, you **must** set \
`have_enough_information` to `false`. Never hallucinate email addresses, dates, IDs, or text that cannot \
be verified in the user-supplied material.
"""

_DATA_URL_RX = re.compile(
    r"data:image/(?P<mime>png|jpeg|jpg);base64,(?P<data>[A-Za-z0-9+/=\s_-]+?)"
    r"(?=\s*(?:<<<\s*QVLMB64:|data:image/|Screenshot\s+PNG\s*\(base64\):|$))",
    re.IGNORECASE | re.DOTALL,
)

_SCREENSHOT_RX = re.compile(
    r"Screenshot\s+PNG\s*\(base64\):\s*(?P<data>[A-Za-z0-9+/=\s_-]+?)"
    r"(?=\s*(?:<<<\s*QVLMB64:|data:image/|Screenshot\s+PNG\s*\(base64\):|$))",
    re.IGNORECASE | re.DOTALL,
)

_SENTINEL_RX = re.compile(
    r"""
    <<<\s*          # opening chevrons
    ['"]?           # optional quote
    \s*QVLMB64\s*:\s*(?P<mime>png|jpeg|jpg)\s*  # tag + mime
    ['"]?           # optional quote
    \s*>>>\s*       # close
    (?P<data>[A-Za-z0-9+/=\s_-]+?)              # base64 (lazy)
    \s*<<<\s*ENDQVLMB64\s*>>>                   # explicit end
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

_DOMTREE_RX = re.compile(
    r"""
    <<<DOMTREE>>>      # opening sentinel
    \n?                # optional newline after sentinel
    (?P<data>.*?)      # capture DOM content lazily
    \n?<<<ENDDOMTREE>>> # closing sentinel
    """,
    re.DOTALL | re.VERBOSE,
)


_INLINE_LIMIT = 19 * 1024 * 1024   # ~20 MB – Gemini’s inline bytes limit

# --------------------------------------------------------------------------- #
# image conversion                                                            #
# --------------------------------------------------------------------------- #
def _b64_to_part(b64: str, mime: str = "image/png") -> genai_types.Part:
    raw = base64.b64decode(re.sub(r"\s+", "", b64))
    logger.debug("Decoded image – %d bytes (mime %s)", len(raw), mime)
    if len(raw) > _INLINE_LIMIT:
        raise ValueError("Image exceeds inline-upload limit – use Files API.")
    return genai_types.Part.from_bytes(data=raw, mime_type=mime)


import binascii

def _b64_diag(label: str, s: str) -> None:
    compact = re.sub(r"\s+", "", s)
    urlsafe = ("-" in compact) or ("_" in compact)
    pad = len(compact) - len(compact.rstrip("="))
    mod4 = len(compact) % 4
    expected_pad = (4 - mod4) % 4
    bad = [(i, c) for i, c in enumerate(compact)
           if c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-"]
    logger.info("[b64 %s] len=%d mod4=%d pad=%d expected_pad=%d urlsafe=%s",
                label, len(compact), mod4, pad, expected_pad, urlsafe)
    if mod4 == 1:
        logger.warning("[b64 %s] length %% 4 == 1 → corrupted block", label)
    if (mod4 in (2, 3)) and pad == 0:
        logger.warning("[b64 %s] missing '=' padding (needs %d)", label, expected_pad)
    if bad:
        preview = "".join(c for _, c in bad[:20])
        logger.warning("[b64 %s] invalid chars (first 20): %r (count=%d)", label, preview, len(bad))

def _b64_bytes_safe(label: str, b64s: str) -> bytes:
    _b64_diag(label, b64s)
    s = re.sub(r"\s+", "", b64s)
    try:
        return base64.b64decode(s, validate=True)
    except binascii.Error as e:
        logger.warning("[b64 %s] strict decode failed: %r; retrying with normalization", label, e)
        s2 = s.replace("-", "+").replace("_", "/")
        s2 += "=" * (-len(s2) % 4)
        _b64_diag(label + " (normalized)", s2)
        try:
            return base64.b64decode(s2, validate=True)
        except binascii.Error as e2:
            logger.error("[b64 %s] normalized decode still failing: %r", label, e2)
            raise

# ---- replacement -----------------------------------------------------------
def _query_to_parts(prompt: str) -> list[genai_types.Part]:
    parts: list[genai_types.Part] = []
    idx = 0

    # Prefer sentinel matches; fall back to data-URL and legacy inline formats.
    patterns = (_SENTINEL_RX, _DATA_URL_RX, _SCREENSHOT_RX) # , _DOMTREE_RX)
    matches = []
    for rx in patterns:
        matches.extend(rx.finditer(prompt))
    matches.sort(key=lambda m: m.start())

    logger.info("Found %d image matches", len(matches))

    for i, m in enumerate(matches, start=1):
        # text before the image
        if m.start() > idx:
            snippet = prompt[idx:m.start()].strip()
            if snippet:
                parts.append(genai_types.Part.from_text(text=snippet))

        gd = m.groupdict()
        raw_b64 = gd.get("data") or ""
        # choose label based on which regex matched
        pat = m.re.pattern
        if pat == _SENTINEL_RX.pattern:
            label = f"img#{i} (sentinel)"
        elif pat == _DATA_URL_RX.pattern:
            label = f"img#{i} (data-url)"
        else:
            label = f"img#{i} (inline)"

        # robust decode
        try:
            raw = _b64_bytes_safe(label, raw_b64)
        except Exception:
            raise ValueError(f"Invalid base64 payload for {label}")

        # mime
        mime_sub = gd.get("mime")
        mime_type = f"image/{mime_sub}" if mime_sub else "image/png"

        if len(raw) > _INLINE_LIMIT:
            raise ValueError("Image exceeds inline-upload limit – use Files API.")
        parts.append(genai_types.Part.from_bytes(data=raw, mime_type=mime_type))

        idx = m.end()

    # trailing text
    tail = prompt[idx:].strip()
    if tail:
        parts.append(genai_types.Part.from_text(text=tail))

    logger.info("Processed %d parts", len(parts))
    return parts or [genai_types.Part.from_text(text=prompt)]

def query_quarantined_vlm(
    vlm: object | str,
    query: str,
    output_schema: type[_T],
    model_level_defense: int = 0,
    token_count_file: Union[Path, str, None] = None,
    function_name: str = "",
    *,
    retries: int = 2,
) -> _T:
    logger.info("↪️  query_quarantined_vlm called – schema=%s", output_schema.__name__)

    if model_level_defense <= 1:
        description = "Set false if text does NOT contain the requested info."
    else:
        description = "Set false if screenshots and/or DOM + text do NOT contain the requested info."

    # 1) build augmented result model
    enough_field = (
        bool,
        Field(
            description=description
        ),
    )
    if issubclass(output_schema, BaseModel):
        result_model = create_model(
            output_schema.__name__,
            __base__=output_schema,
            have_enough_information=enough_field,
        )
    else:
        result_model = create_model(
            "Result",
            output=(output_schema, Field(description="Parsed value")),
            have_enough_information=enough_field,
        )
    logger.info("Result model assembled: %s", result_model.schema())

    # 2) normalize `vlm`
    agent_model = None

    if isinstance(vlm, str):
        if vlm.startswith("openrouter:"):
            # OpenRouter path
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set in environment")
            model_id = vlm.split(":")[-1]

            http_client = httpx.AsyncClient(  # Changed from Client to AsyncClient
                timeout=httpx.Timeout(
                    100.0,      # total timeout
                    connect=30.0,  # connection timeout
                    read=100.0,    # read timeout
                )
            )
            
            agent_model = OpenAIChatModel(
                model_id,
                provider=OpenRouterProvider(
                    api_key=api_key,
                    http_client=http_client
                ),
            )
        elif vlm.startswith("google:"):
            # Google/Gemini is DISABLED on the quarantine path by policy. The
            # quarantine side must only ever call the primary q_llm (the local
            # UITARS / Q-LLM) — never a third, external, paid model ("only 2
            # LLMs" invariant). A stray "google:" spec (e.g. an un-overridden
            # field default) must fail loudly here rather than silently billing a
            # Gemini call.
            raise ValueError(
                f"Google/Gemini models are disabled on the quarantine path: {vlm!r}. "
                "Only the primary local q_llm (an 'openai:'/'openrouter:' spec pointing "
                "at the local Q-LLM endpoint) may be used for quarantined helper calls."
            )
        elif vlm.startswith("openai:"):
            
            model_id = vlm.split(":")[-1]
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    100.0,      
                    connect=30.0,
                    read=100.0,
                )
            )
            # base_url from OPENAI_BASE_URL lets `openai:<model>` target a local
            # OpenAI-compatible endpoint (e.g. the UITARS vLLM on :8000), so the
            # Q-LLM helper tools run on the SAME local model as the executor
            # (no third, paid model). Falls back to the real OpenAI API if unset.
            _base_url = os.getenv("OPENAI_BASE_URL")
            provider = OpenAIProvider(
                api_key=os.getenv("OPENAI_API_KEY", "dummy"),
                base_url=_base_url,
                http_client=http_client,
            )
            agent_model = OpenAIChatModel(
                model_id,
                provider=provider,
            )
        else:
            raise ValueError(f"Unsupported string model identifier (only openrouter and openai strings are supported): {vlm}. google/gemini is disabled; anthropic models are WIP.")
    else:
        # The only object type this path ever received was a GoogleVLM instance.
        # Google/Gemini is disabled (see the "google:" branch above): reject any
        # non-string quarantine model so no third model can be invoked.
        raise ValueError(
            f"Non-string quarantine model {type(vlm).__name__} is disabled: only the "
            "primary local q_llm string spec may be used for quarantined helper calls."
        )
        
    # 3) build multimodal content
    parts = _query_to_parts(query)
    num_parts = len(parts)
    num_images = sum(1 for p in parts if p.inline_data is not None)
    num_text = sum(1 for p in parts if p.text is not None)

    logger.info(
        "Prompt split into %d parts (%d text, %d images)",
        num_parts, num_text, num_images
    )

    if model_level_defense <= 1:
        data_sources = "text"
    else:
        data_sources = "images and/or DOM trees and text"
    system_prompt = _SYSTEM_PROMPT_VLM.format(
        data_sources=data_sources
    )

    # 4) model run via Pydantic-AI
    logger.info("Sending request to model (%s)…", model_id)

    # now one Agent call.
    # temperature=0: structured extraction must be deterministic — the same
    # screenshot should always yield the same parsed value. Without this the
    # provider default (~1.0) makes perception stochastic, so a multi-signal
    # plan occasionally misreads a value and takes the wrong branch (mirrors the
    # temperature-0 pin already applied to the P-LLM).
    agent = pydantic_ai.Agent(
        agent_model,
        output_type=result_model,
        retries=retries,
        system_prompt=system_prompt,
        model_settings={"temperature": 0.0},
    )

    messages: list[object] = []
    for p in parts:
        if p.inline_data is not None:
            messages.append(
                BinaryContent(
                    data=p.inline_data.data,
                    media_type=p.inline_data.mime_type,
                )
            )
        else:
            messages.append(p.text)

    try:
        result = agent.run_sync(messages)
        res = result.output
    except Exception as e:
        logger.exception("‼️  Model call failed")
        raise
    
    usage = result.usage()
    if usage:
        # print(
        #     "Token usage - Input: %d, Output: %d, Total: %d",
        #     usage.input_tokens,
        #     usage.output_tokens,
        #     usage.total_tokens
        # )
        if token_count_file is not None:
            with open(token_count_file, "a") as f:
                f.write(f"Model {model_id} Function {function_name if function_name else 'unknown'} - Input Tokens: {usage.input_tokens}, Output Tokens: {usage.output_tokens}, Total Tokens: {usage.total_tokens}\n")
    else:
        logger.warning("No token usage information available from provider")

    logger.info("Raw model response: %s", res)

    # experimental endpoint back-off
    if not agent_model and "gemini" in vlm.model and "exp" in vlm.model:  # type: ignore[attr-defined]
        logger.debug("Sleeping 10 s for Gemini exp model")
        time.sleep(10)

    if not getattr(res, "have_enough_information", True):
        logger.info("Model reports insufficient info → raising")
        raise NotEnoughInformationError()

    logger.info("✅  Parsed OK")
    return res if issubclass(output_schema, BaseModel) else res.output