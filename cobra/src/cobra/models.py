import os
import time
from pathlib import Path

import anthropic
import openai
from agentdojo import agent_pipeline, functions_runtime
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.models import MODEL_NAMES
from google import genai
from openai.types.chat import ChatCompletionReasoningEffort
from pydantic_ai.models import KnownModelName

from cobra.interaction.system_user_prompts import (
    default_system_prompt_generator_osworld_pllm,
    baseline_system_prompt_generator_osworld_pllm,
)
from cobra.interpreter.interpreter import MetadataEvalMode
from cobra.pipeline_elements.anthropic_tool_filter import AnthropicLLMToolFilter
from cobra.pipeline_elements.privileged_llm import PrivilegedLLM
from cobra.pipeline_elements.replay_privileged_llm import PrivilegedLLMReplayer, UserInjectionTasksGetter
from cobra.pipeline_elements.security_policies import (
    ADNoSecurityPolicyEngine,
    AgentDojoSecurityPolicyEngine,
    BankingSecurityPolicyEngine,
    SlackSecurityPolicyEngine,
    TravelSecurityPolicyEngine,
    WorkspaceSecurityPolicyEngine,
)
from cobra.interaction.security_policies_osworld import OSWorldSecurityPolicyEngine


def _llm_http_kwargs() -> dict:
    """Bounded HTTP behaviour for the planner/annotator clients.

    These clients used to be built with no `timeout`, i.e. the SDK default of
    600 s, and every call sits inside agentdojo's tenacity wrapper
    (`stop_after_attempt(3)`) on top of the SDK's own retries. A connection the
    peer had already closed therefore stalled for ~2 h (CLOSE-WAIT sockets) before
    anything gave up. Bounding the single request caps the worst case at
    timeout x (max_retries + 1) x 3.

    Defaults are deliberately generous — the planner emits ~22k annotation
    tokens, so this must not cut a legitimately slow call short. Override with
    CAMEL_LLM_TIMEOUT_S / CAMEL_LLM_MAX_RETRIES.
    """
    return {
        "timeout": float(os.getenv("CAMEL_LLM_TIMEOUT_S", "420")),
        "max_retries": int(os.getenv("CAMEL_LLM_MAX_RETRIES", "1")),
    }


def _pipeline_tag(model_spec: str) -> str:
    """Model tag for pipeline/log-dir naming, with '/' sanitized to '_'.

    OpenRouter model specs are "provider:vendor/model" (e.g.
    "openrouter:openai/gpt-5"), so `model_spec.split(':')[1]` contains a '/'.
    `TraceLogger.save()` (agentdojo) sanitizes pipeline_name with the same
    replace before building the save path, but `load_task_results()` does
    NOT apply it when reading `agent_pipeline.name` back — the raw '/' was
    read as a path separator, so every prior result silently 404'd and
    force_rerun=False never actually skipped a completed task. Sanitizing at
    the source (here) keeps both sides consistent.
    """
    return model_spec.split(":")[1].replace("/", "_")

_thinking_efforts = ["low", "medium", "high"]
_oai_thinking_models = {
    "o4-mini-2025-04-16": "ChatGPT",
    "o4-mini": "ChatGPT",
    "o3-2025-04-16": "ChatGPT",
    "o3-mini-2025-01-31": "ChatGPT",
    "o1-2024-12-17": "ChatGPT",
    "codex-mini-latest": "ChatGPT",
}
_oai_thinking_models_with_effort = {
    f"{model}-{effort}": name for model, name in _oai_thinking_models.items() for effort in _thinking_efforts
}
_supported_model_names = {
    "gpt-oss-20b": "OpenAI GPT-OSS",
    "gpt-oss-120b": "OpenAI GPT-OSS",
    "gemini-2.5-flash": "AI model developed by Google",
    "gemini-2.5-flash-preview-05-20": "AI model developed by Google",
    "gemini-2.5-pro-preview-06-05": "AI model developed by Google",
    "gemini-2.5-pro": "AI model developed by Google",
    "gemini-2.0-flash-lite-001": "AI model developed by Google",
    "gemini-3-pro-preview": "AI model developed by Google",
    "claude-3-5-haiku-20241022": "Claude",
    "claude-4-5-haiku-20250805": "Claude",
    "claude-3-5-sonnet-20241022": "Claude",
    "claude-3-7-sonnet-20250219": "Claude",
    "claude-sonnet-4-20250514": "Claude",
    "claude-sonnet-4-5-20250929": "Claude",
    "claude-opus-4-20250514": "Claude",
    "claude-opus-4-1-20250805": "Claude",
    "gpt-4o-2024-08-06": "GPT-4",
    "gpt-4o-mini-2024-07-18": "GPT-4",
    "gpt-4.1-2025-04-14": "ChatGPT",
    "gpt-4.1-nano-2025-04-14": "ChatGPT",
    "gpt-5": "ChatGPT",
    "gpt-5.1": "ChatGPT",
    "kimi-k2-thinking": "Kimi",
    "grok-4": "Grok",
    "deepseek-r1-0528": "DeepSeek",
    "deepseek-r1": "DeepSeek",
} | _oai_thinking_models_with_effort
suffixes = ["", "+cobra", "+cobra+secpol", "+cobra+secpol+strict"]

CAMEL_MODEL_NAMES = {f"{model}{suffix}": name for model, name in _supported_model_names.items() for suffix in suffixes}

_SECURITY_POLICY_ENGINES: dict[str, type[AgentDojoSecurityPolicyEngine]] = {
    "workspace": WorkspaceSecurityPolicyEngine,
    "travel": TravelSecurityPolicyEngine,
    "banking": BankingSecurityPolicyEngine,
    "slack": SlackSecurityPolicyEngine,
    "osworld": OSWorldSecurityPolicyEngine
}


def _is_oai_reasoning_model(model: str) -> bool:
    return "o4" in model or "o3" in model or "o1" in model or "codex" in model


MODEL_NAMES.update(CAMEL_MODEL_NAMES)


class Sleep(agent_pipeline.BasePipelineElement):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self._amount = amount

    def query(
        self,
        query: str,
        runtime,
        env=functions_runtime.EmptyEnv(),
        messages=[],
        extra_args={},
    ) -> tuple:
        if self._amount > 0:
            time.sleep(self._amount)
        return query, runtime, env, messages, extra_args


def make_tools_pipeline(
    model: KnownModelName,
    use_original: bool,
    replay_with_policies: bool,
    attack_name: str,
    reasoning_effort: ChatCompletionReasoningEffort,
    thinking_budget_tokens: int | None,
    suite: str,
    ad_defense: str | None,
    eval_mode: MetadataEvalMode,
    q_llm: KnownModelName | None,
    part_path: Path = Path(),
    max_attempts: int = 10,
    pllm_exposure_level: int = 0,
    system_defense_level: int = 2,
    unoptimized_cua: bool = False,
    plan_analysis_only: bool = False,
    brh_enabled: bool = False,
    brh_dir: Path | None = None,
) -> agent_pipeline.AgentPipeline:
    if "openrouter:" in model:
        # model string should be like "openrouter:openai/gpt-5"
        router_model = model.split(":")[1]

        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            **_llm_http_kwargs(),
        )
        # OSS models use the standard Chat Completions interface (no reasoning_effort)
        llm = agent_pipeline.OpenAILLM(client, router_model, None)
    elif "azure:" in model:
        # model string like "azure:openai/gpt-5"; the Azure deployment name is
        # the last path segment ("gpt-5").
        deployment = model.split(":", 1)[1].split("/")[-1]
        client = openai.OpenAI(
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            **_llm_http_kwargs(),
        )
        # reasoning_effort=None -> NOT_GIVEN, and temperature keeps its 0.0 default
        # which chat_completion_request sends as NOT_GIVEN (0.0 is falsy) — so Azure
        # uses gpt-5's default temperature (1.0), identical to the openrouter runs.
        llm = agent_pipeline.OpenAILLM(client, deployment, None)
    elif "google:" in model:
        # vertexai.init(project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION"))
        # llm = GoogleLLM(model.split(":")[1])
        # client = genai.Client(vertexai=True, project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION"))
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if model == "google:gemini-2.0-flash-lite-001":
            max_tokens = 8192
        else:
            max_tokens = 65535
        llm = agent_pipeline.GoogleLLM(model.split(":")[1], client, max_tokens=max_tokens)
    elif "openai:" in model:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), **_llm_http_kwargs())
        # reasoning models do not support temperature and their "system" message is called "developer" message
        if _is_oai_reasoning_model(model):
            llm = agent_pipeline.OpenAILLM(client, model.split(":")[1], reasoning_effort, None)
        else:
            llm = agent_pipeline.OpenAILLM(client, model.split(":")[1], None)
    elif "anthropic:" in model:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if thinking_budget_tokens:
            max_tokens = 8192 + thinking_budget_tokens
        else:
            max_tokens = 8192
        llm = agent_pipeline.AnthropicLLM(
            client, model.split(":")[1], thinking_budget_tokens=thinking_budget_tokens, max_tokens=max_tokens
        )
    else:
        raise ValueError("Invalid model")

    llm.name = model.split(":")[1]

    engine = _SECURITY_POLICY_ENGINES[suite]

    if suite=="osworld":
        osworld_prompt_gen = (
            baseline_system_prompt_generator_osworld_pllm
            if unoptimized_cua
            else default_system_prompt_generator_osworld_pllm
        )
        # Need to add add env and stuff on init vlm somewhere but maybe has to do with tasksuite? uitars will not be llm or q_llm so different init needed
        tools_pipeline = agent_pipeline.AgentPipeline(
            [
                # Adds the user query to the history
                agent_pipeline.InitQuery(),
                # Generates the code and writes it to `extra_args`
                PrivilegedLLM(
                    llm,
                    OSWorldSecurityPolicyEngine,
                    q_llm or model,
                    osworld_prompt_gen,
                    part_path=part_path,
                    max_attempts = max_attempts,
                    pllm_exposure_level = pllm_exposure_level,
                    system_defense_level=system_defense_level,
                    plan_analysis_only=plan_analysis_only,
                    brh_enabled=brh_enabled,
                    brh_dir=brh_dir,
                ),

            ]
        )
        # Used for logging
        if "openai" in model and _is_oai_reasoning_model(model):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{reasoning_effort}+cobra"
        elif "anthropic" in model and (("3-7-sonnet" in model or "sonnet-4" in model) and thinking_budget_tokens):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{thinking_budget_tokens}+cobra"
        else:
            tools_pipeline.name = f"{_pipeline_tag(model)}+cobra"

        if unoptimized_cua:
            tools_pipeline.name += "+unopt"

        if plan_analysis_only:
            tools_pipeline.name += "+plan_only"

        if q_llm:
            tools_pipeline.name += f"-q:{_pipeline_tag(q_llm)}"
    elif use_original:
        if "exp" in model:
            print("Adding 'Sleep' pipeline element.")
            tools_loop = agent_pipeline.ToolsExecutionLoop([agent_pipeline.ToolsExecutor(), Sleep(6), llm])
        else:
            tools_loop = agent_pipeline.ToolsExecutionLoop([agent_pipeline.ToolsExecutor(), llm])

        tools_pipeline = agent_pipeline.AgentPipeline(
            [agent_pipeline.SystemMessage(load_system_message(None)), agent_pipeline.InitQuery(), llm, tools_loop]
        )

        if ad_defense == "tool_filter" and isinstance(llm, agent_pipeline.AnthropicLLM):
            tools_pipeline = agent_pipeline.AgentPipeline(
                [
                    agent_pipeline.SystemMessage(load_system_message(None)),
                    agent_pipeline.InitQuery(),
                    AnthropicLLMToolFilter(llm.client, llm.name),
                    llm,
                    tools_loop,
                ]
            )
        elif ad_defense is not None:
            tools_pipeline = agent_pipeline.AgentPipeline.from_config(
                agent_pipeline.PipelineConfig(
                    llm=llm, defense=ad_defense, system_message_name=None, system_message=None
                )
            )

        if _is_oai_reasoning_model(model):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{reasoning_effort}"
        elif "anthropic" in model and (("3-7-sonnet" in model or "sonnet-4" in model) and thinking_budget_tokens):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{thinking_budget_tokens}"
        else:
            tools_pipeline.name = _pipeline_tag(model)
        if ad_defense is not None:
            tools_pipeline.name += f"+{ad_defense}"

    elif replay_with_policies:
        if _is_oai_reasoning_model(model):
            pipeline_name = f"{_pipeline_tag(model)}-{reasoning_effort}+cobra"
        else:
            pipeline_name = f"{_pipeline_tag(model)}+cobra"
        tools_pipeline = agent_pipeline.AgentPipeline(
            [
                agent_pipeline.InitQuery(),
                UserInjectionTasksGetter(),
                PrivilegedLLMReplayer(pipeline_name, attack_name, engine, eval_mode),
            ]
        )
        # Used for logging
        if "openai" in model and _is_oai_reasoning_model(model):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{reasoning_effort}+cobra+secpol"
        elif "anthropic" in model and (("3-7-sonnet" in model or "sonnet-4" in model) and thinking_budget_tokens):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{thinking_budget_tokens}+cobra+secpol"
        else:
            tools_pipeline.name = f"{_pipeline_tag(model)}+cobra+secpol"
        if eval_mode == MetadataEvalMode.STRICT:
            tools_pipeline.name += "+strict"
    else:
        tools_pipeline = agent_pipeline.AgentPipeline(
            [
                # Adds the user query to the history
                agent_pipeline.InitQuery(),
                # Generates the code and writes it to `extra_args`
                PrivilegedLLM(
                    llm,
                    ADNoSecurityPolicyEngine,
                    q_llm or model,
                    part_path=part_path,
                    max_attempts=max_attempts,
                    plan_analysis_only=plan_analysis_only,
                    brh_enabled=brh_enabled,
                    brh_dir=brh_dir,
                ),
            ]
        )
        # Used for logging
        if "openai" in model and _is_oai_reasoning_model(model):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{reasoning_effort}+cobra"
        elif "anthropic" in model and (("3-7-sonnet" in model or "sonnet-4" in model) and thinking_budget_tokens):
            tools_pipeline.name = f"{_pipeline_tag(model)}-{thinking_budget_tokens}+cobra"
        else:
            tools_pipeline.name = f"{_pipeline_tag(model)}+cobra"

        if plan_analysis_only:
            tools_pipeline.name += "+plan_only"

        if q_llm:
            tools_pipeline.name += f"-q:{_pipeline_tag(q_llm)}"

    return tools_pipeline
