"""Use a locally hosted model.

Expects an OpenAI-compatible API server to be running on port 8000, e.g. launched with:

```
vllm serve /path/to/huggingface/model
```
"""

import random
from collections.abc import Sequence
import openai
from openai.types.chat import ChatCompletionMessageParam

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env,  FunctionsRuntime
from cobra.interaction.vlms.extended_types import  ChatMessage, ChatAssistantMessage, text_content_block_from_string
from cobra.interaction.environments.base_ui_uitars import parse_action_to_structure_output

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
class InvalidModelOutputError(Exception): ...


def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float | None = 1.0,
    top_p: float | None = 0.9,
    token_count_file: str | None = None,
    function_name: str = "",
) -> str:
    logger.info("▶️  POST /chat/completions  model=%s  msgs=%d", model, len(messages))
    try:
        # Get the full response object
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            frequency_penalty=1,
            max_tokens=8192,
            temperature=temperature,
            top_p=top_p,
            seed=random.randint(0, 1000000),
        )
        
        # Extract content
        response = completion.choices[0].message.content
        
        # Log token usage
        if completion.usage:
            logger.info(
                "Token usage - Input: %d, Output: %d, Total: %d",
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
                completion.usage.total_tokens
            )
        if token_count_file is not None:
            with open(token_count_file, "a") as f:
                f.write(f"Model {model} Function {function_name if function_name else 'unknown'} - Input Tokens: {completion.usage.prompt_tokens}, Output Tokens: {completion.usage.completion_tokens}, Total Tokens: {completion.usage.total_tokens}\n")
        else:
            logger.warning("No token usage information available")
        
        logger.info("✅  backend replied %d chars", len(response))
        
    except Exception as e:
        print(f"[debug] error: {e}")
        response = ""
    
    if response is None:
        raise InvalidModelOutputError("No response from model")
    
    return response


class UITarsVLM(BasePipelineElement):
    def __init__(
        self, client: openai.OpenAI, model: str, temperature: float | None = 0.0, top_p: float | None = 0.9
    ) -> None:
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
        token_count_file: str | None = None,
        function_name: str = "",
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:

        response = chat_completion_request(
            self.client,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            token_count_file=token_count_file,
            function_name=function_name,
        )
        logger.info("Model response: %s", response)
        try:
            parsed_responses = parse_action_to_structure_output(response)
        except Exception as e:
            logger.error(f"Error parsing response (no actionable plan), with response: {response}")
            # Unparseable Action block → fall back to an empty-plan assistant message
            # (same shape as the parser's no-"Action:" path) so the caller still gets the
            # raw text as content and treats the step as a no-op instead of crashing.
            parsed_responses = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(response)],
                tool_calls=[],
            )

        return query, runtime, env, [*messages, parsed_responses], extra_args