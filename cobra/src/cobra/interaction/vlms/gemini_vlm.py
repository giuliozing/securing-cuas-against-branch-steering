"""
Subclass of Agent-Dojo's GoogleLLM that supports Gemini Pro Vision
(image-and-text prompts) while preserving all existing behaviour.

Import and use `GoogleVLM` instead of `GoogleLLM`.
"""


import base64
import mimetypes
from pathlib import Path
from typing import List, Sequence, cast

from agentdojo.agent_pipeline.llms.google_llm import (
    GoogleLLM,
    chat_completion_request,
    _message_to_google as _base_message_to_google,
    _merge_tool_result_messages,
    _function_to_google as _base_function_to_google,
    _google_to_assistant_message as _base_google_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
#from agentdojo.types import ChatAssistantMessage, ChatMessage
from cobra.interaction.vlms.extended_types import ChatAssistantMessage, ChatMessage
from google import genai
from google.genai import types as genai_types

# --------------------------------------------------------------------------- #
# Helper functions (typed)                                                    #
# --------------------------------------------------------------------------- #
def _b64_part(img_bytes: bytes, mime: str = "image/png") -> genai_types.Part:
    return genai_types.Part.from_bytes(data=img_bytes, mime_type=mime)

def _path_part(path: Path) -> genai_types.Part:
    mime, _ = mimetypes.guess_type(path.name)
    return _b64_part(path.read_bytes(), mime or "application/octet-stream")


# --------------------------------------------------------------------------- #
# Sub-class with vision support                                               #
# --------------------------------------------------------------------------- #
class GoogleVLM(GoogleLLM):
    """GoogleLLM + image handling for Gemini Pro Vision."""

    # ---------- constructors ------------------------------------------------ #
    def __init__(
        self,
        model: str,
        client: genai.Client | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 65535,
    ) -> None:
        super().__init__(model=model, client=client, temperature=temperature, max_tokens=max_tokens)

    # ---------- private helpers -------------------------------------------- #
    def _message_to_google(self, msg: ChatMessage) -> genai_types.Content:  # type: ignore[override]
        content: genai_types.Content = _base_message_to_google(msg)      # text + tool-calls

        # --- Only user messages may contain screenshots ----------------------------------
        if not msg.get("content"):
            return content

        for blk in msg.get("content", []):
            t = blk.get("type")
            if t not in ("image", "image_url"):
                continue

            # ensure parts list exists
            if content.parts is None:
                content.parts = []

            # raw bytes?
            if t == "image" and "data" in blk:
                content.parts.append(_b64_part(
                    cast(bytes, blk["data"]),
                    cast(str, blk.get("mime", "image/png"))
                ))
            # local file?
            elif t == "image" and "path" in blk:
                content.parts.append(_path_part(Path(cast(str, blk["path"]))))
            # otherwise treat as a URI (either blk["url"] or blk["image_url"]["url"])
            else:
                # pick whichever field exists
                uri = blk.get("url") \
                   or blk.get("image") \
                   or (blk.get("image_url", {}) or {}).get("url")
                mime = blk.get("mime", "image/png")
                content.parts.append(
                    genai_types.Part.from_uri(file_uri=uri, mime_type=mime)
                )

        return content
    # public query identical except for one extra line that marks the last user msg
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        first_message, *other_messages = messages
        if first_message["role"] == "system":
            system_instruction = first_message["content"][0]["content"]
        else:
            system_instruction = None
            other_messages = messages
        google_messages = [self._message_to_google(message) for message in other_messages]
        google_messages = _merge_tool_result_messages(google_messages)

        google_functions = [_base_function_to_google(tool) for tool in runtime.functions.values()]
        google_tools: genai_types.ToolListUnion | None = (
            [genai_types.Tool(function_declarations=google_functions)] if google_functions else None
        )
        generation_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            tools=google_tools,
            system_instruction=system_instruction
        )
        completion = chat_completion_request(
            self.model,
            self.client,
            google_messages,  # type: ignore
            generation_config=generation_config,
        )
        output = _base_google_to_assistant_message(completion)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
    

