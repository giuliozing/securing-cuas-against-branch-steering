"""
Anthropic Claude-backed VLM adapter with computer-use tools (claude-3-7+).

Usage:
    from anthropic import Anthropic
    client = Anthropic(api_key=...)

    vlm = ClaudeVLM(
        client=client,
        model="claude-3-7-sonnet-20250219",   # or "claude-4-sonnet-20250514", "claude-4-5-sonnet-20250929"
        system="You are a helpful agent.",    # optional
        enable_tools=True,                    # ← computer-use tools + thinking for new models
    )

    # Build all messages (including any prior assistant thinking/tool blocks) in base_ui.
    query, runtime, env, msgs, extra = vlm.query(
        query="...",
        runtime=runtime,
        env=env,
        messages=camel_messages,              # Sequence[ChatMessage] (caller-owned state)
    )

    # If the model emitted a tool call, *caller* should append a tool_result message and re-call:
    # tool_result_turn = {
    #   "role": "user",
    #   "content": [{
    #       "type": "tool_result",
    #       "tool_use_id": "<ID from extra['tool_uses']>",
    #       "content": [{"type": "text", "text": "Success"}]   # or add an {"type":"image",...}
    #   }]
    # }
    # msgs2 = [*msgs, tool_result_turn]
    # query, runtime, env, msgs3, extra = vlm.query(query="", runtime=runtime, env=env, messages=msgs2)
"""

import logging
from typing import Sequence, List, Optional, Dict, Any, Tuple

from anthropic import Anthropic
from anthropic.types.beta import BetaMessageParam, BetaTextBlockParam

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from cobra.interaction.vlms.extended_types import ChatMessage

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Models that support the 2025-01-24 computer-use tool and extended thinking
NEW_CLAUDE_MODELS = {
    "claude-3-7-sonnet-20250219",
    "claude-4-opus-20250514",
    "claude-4-sonnet-20250514",
    "claude-sonnet-4-5-20250929",
}

def _is_new_claude(model: str) -> bool:
    return model in NEW_CLAUDE_MODELS

COMPUTER_USE_BETA_20250124 = "computer-use-2025-01-24"

def claude_messages_request(
    client: Anthropic,
    model: str,
    messages: Sequence[ChatMessage] | List[BetaMessageParam],
    max_tokens: int = 8192,
    system: Optional[str] = None,
    betas: Optional[List[str]] = None,
    *,
    enable_tools: bool = False,
    enable_thinking: bool = False,
    display_width_px: int = 1280,
    display_height_px: int = 720,
    thinking_budget_tokens: int = 1024,
):
    """
    Calls Anthropic:
      - Always uses the **beta** endpoint when either tools **or** thinking are enabled.
      - Uses the stable v1 endpoint only when both are disabled.
    Returns the raw SDK response object.
    """

    # Build betas header safely (NEVER send empty strings)
    use_betas = [b for b in (betas or []) if b]  # remove empties
    tools = None
    extra_body = None

    # If thinking is requested, set extra_body
    if enable_thinking:
        extra_body = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget_tokens}}

    # If tool use is requested, attach the computer tool and (for new schema) its beta flag
    if enable_tools:
        if _is_new_claude(model):
            tools = [{
                "name": "computer",
                "type": "computer_20250124",
                "display_width_px": display_width_px,
                "display_height_px": display_height_px,
                "display_number": 1,
            }]
            if COMPUTER_USE_BETA_20250124 not in use_betas:
                use_betas.append(COMPUTER_USE_BETA_20250124)
        else:
            tools = [{
                "name": "computer",
                "type": "computer_20241022",
                "display_width_px": display_width_px,
                "display_height_px": display_height_px,
                "display_number": 1,
            }]

    # Build args WITHOUT Nones
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:      # system_blocks is a list; only add if non-empty
        kwargs["system"] = [BetaTextBlockParam(type="text", text=system)]
    if tools:       # only if tool use enabled
        kwargs["tools"] = tools
    if enable_thinking and extra_body:
        kwargs["extra_body"] = extra_body
    if use_betas:   # <-- critical: only include when non-empty
        kwargs["betas"] = use_betas

    response = client.beta.messages.create(**kwargs)

    logger.info("✅  Anthropic replied; blocks=%d", len(getattr(response, "content", []) or []))
    return response


# ---------------------------------------------------------------------------
# Public adapter (stateless; base_ui owns history)
# ---------------------------------------------------------------------------

class ClaudeVLM(BasePipelineElement):
    def __init__(
        self,
        client: Anthropic,
        model: str = "claude-3-7-sonnet-20250219",
        system: Optional[str] = None,
        betas: Optional[List[str]] = None,
        *,
        enable_tools: bool = True,
        enable_thinking: bool = True,
        display_width_px: int = 1280,
        display_height_px: int = 720,
        thinking_budget_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.betas = betas or []
        self.enable_tools = enable_tools
        self.enable_thinking = enable_thinking
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px
        self.thinking_budget_tokens = thinking_budget_tokens

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
        token_count_file: str | None = None,
        function_name: str = "",
    ) -> Tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:


        response = claude_messages_request(
            self.client,
            model=self.model,
            messages=messages,
            max_tokens=8192,
            system=self.system,
            betas=self.betas,
            enable_tools=self.enable_tools,
            enable_thinking=self.enable_thinking,
            display_width_px=self.display_width_px,
            display_height_px=self.display_height_px,
            thinking_budget_tokens=self.thinking_budget_tokens,
        )

        usage = getattr(response, "usage", None)
        if usage and token_count_file:
            with open(token_count_file, "a") as f:
                f.write(f"Model {self.model} Function {function_name if function_name else 'unknown'} - Input Tokens: {usage.input_tokens}, Output Tokens: {usage.output_tokens}\n")
                

        return query, runtime, env, [*messages, response], extra_args
