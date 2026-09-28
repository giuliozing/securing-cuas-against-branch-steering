from pathlib import Path
import os
import openai
from typing import Annotated

from pydantic import Field

from agentdojo.functions_runtime import TaskEnvironment, make_function, Depends
from agentdojo.task_suite.task_suite import TaskSuite

from osworld.desktop_env.desktop_env import DesktopEnv

from cobra.interaction.environments.base_ui_uitars import BaseUI, find, locate_and_click, run_single_uitars, check_done, drag, select, click, left_single, left_double, right_single, hover, scroll, type_text, hotkey, press, keydown, keyup, wait, mark_done, mark_fail, reset_memory, summarize_screenshot_content, no_op, get_google_credentials,get_page_elements,get_page_text,find_element_by_text, verify_hypothesis
from cobra.interaction.vlms.uitars_vlm import UITarsVLM


def build_ui(
        defense_level: int = 0,
        q_llm: str | None = None,
        q_llm_second_check: str | None = None,
        path_to_vm: str | None = None,
        path_to_vlm: str | None = None,
        port: int = 15001,
        token_count_file: str | None = None,
        provider_name: str = "docker",
        temperature: float = 0.0,
    ) -> BaseUI:  # <--- add argument
    if path_to_vm is None:
        raise ValueError("path_to_vm must be provided as an absolute path.")
    if not Path(path_to_vm).is_absolute():
        raise ValueError(f"path_to_vm must be an absolute path, got relative path: {path_to_vm}")
    # provider_name defaults to "docker" (rootless podman on this host, via DOCKER_HOST);
    # temperature defaults to 0.0 for parity with the naive-ReAct baseline (was hardcoded 0.6).
    env = DesktopEnv(
        provider_name=provider_name,
        path_to_vm=path_to_vm,
        action_space="pyautogui",
        screen_size=(1920,1080),
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=True,
    )

    # Hosted-executor path: `path_to_vlm` naming a Kimi model routes the executor
    # to a remote OpenAI-compatible endpoint (OpenRouter by default, or an Azure
    # AI Foundry deployment via KIMI_BASE_URL/KIMI_API_KEY) instead of the local
    # UI-TARS vLLM. Everything downstream — messages, UITARS action grammar,
    # parser, tools — is shared; only transport and coordinate space differ.
    if "kimi" in (path_to_vlm or "").lower():
        from cobra.interaction.vlms.kimi_vlm import KimiVLM, parse_reasoning_env

        client = openai.OpenAI(
            base_url=os.getenv("KIMI_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("KIMI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            timeout=300.0,
        )
        vlm = KimiVLM(
            client=client,
            model=path_to_vlm,
            temperature=temperature,
            screen_size=(1920, 1080),
            reasoning=parse_reasoning_env(os.getenv("OPENROUTER_REASONING", "off")),
        )
        coord_mode = "auto"
    elif os.environ.get("CLAUDE_HOSTED_EXECUTOR") == "1":
        # Sonnet-as-Q-LLM: same hosted-executor mechanism as the Kimi branch
        # above (KimiVLM is a generic wrapper for any hosted OpenAI-compatible
        # chat model against the shared UITARS action grammar) — NOT Anthropic's
        # native computer-use tool (base_ui_task_suite_anthropic.py), which
        # needs an ANTHROPIC_API_KEY this box does not have. See
        # _select_builder_from_path in user_tasks.py for the routing gate.
        from cobra.interaction.vlms.kimi_vlm import KimiVLM, parse_reasoning_env

        client = openai.OpenAI(
            base_url=os.getenv("SONNET_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("SONNET_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            timeout=300.0,
        )
        vlm = KimiVLM(
            client=client,
            model=path_to_vlm,
            temperature=temperature,
            screen_size=(1920, 1080),
            reasoning=parse_reasoning_env(os.getenv("OPENROUTER_REASONING", "off")),
        )
        coord_mode = "auto"
    else:
        client = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key=os.getenv("UITARS_API_KEY"),
            timeout=300.0,
        )

        vlm = UITarsVLM(
            client=client,
            model=path_to_vlm,
            temperature=temperature,
        )
        coord_mode = "uitars"

    ui = BaseUI(env=env, vlm=vlm, coord_mode=coord_mode)
    ui.model_defense_level = defense_level
    ui.q_llm = q_llm
    ui.q_llm_second_check = q_llm_second_check
    ui.defense_map = {
        0: ui.model_defense_0,
        1: ui.model_defense_1,
        2: ui.model_defense_2,
        3: ui.model_defense_3,
        4: ui.model_defense_4,
    }
    ui.token_count_file = token_count_file
    return ui


class UIEnvironment(TaskEnvironment):
    base_ui: Annotated[
        BaseUI,
        Depends(build_ui),
    ] = Field(default=None, exclude=True)
    
    def model_copy(self, *, deep: bool = False) -> "UIEnvironment":
        # always do a shallow copy, even if deep=True
        return super().model_copy(deep=False)



TOOLS_0 = [
    find,
    locate_and_click,
    run_single_uitars,
    check_done,
    drag,
    select,
    click,
    left_single,
    left_double,
    right_single,
    hover, 
    scroll, 
    type_text, 
    hotkey, 
    press, 
    keydown, 
    keyup, 
    wait, 
    mark_done, 
    mark_fail,
    reset_memory,
    summarize_screenshot_content,
    get_google_credentials,
    no_op,
    get_page_elements,
    get_page_text,
    find_element_by_text,
    verify_hypothesis
]
DISALLOWED = {
    summarize_screenshot_content,
    get_page_elements,
    get_page_text,
    verify_hypothesis
}
TOOLS_1 = [tool for tool in TOOLS_0 if tool != run_single_uitars]
TOOLS_2 = [tool for tool in TOOLS_1 if tool not in DISALLOWED]
TOOLS_BASELINE = [tool for tool in TOOLS_1 if tool != verify_hypothesis]

TOOLS = {j: v for j, v in enumerate([TOOLS_0, TOOLS_1, TOOLS_2])}

HERE = Path(__file__).resolve().parent
SUITE_DATA = (
    HERE
    / ".."  # up from cobra.interaction/
    / ".."  # up from src/
    / "data"
    / "suites"
    / "osworld"
)
SUITE_DATA = SUITE_DATA.resolve()


def call_mcp_tool(
    env: Annotated[BaseUI, Depends("base_ui")],
    name: str,
    arguments: dict | None = None,
) -> str:
    """Call a named MCP tool through the BRH proxy (BRH_MCP=1 runs only).

    Use this when an MCP tool provides a shortcut over GUI navigation.
    Example: call_mcp_tool(name='libreoffice_calc.transpose_range', arguments={'source_range': 'B2:F5', 'target_cell': 'B8'})

    :param name: MCP tool name as registered on the server (e.g. 'google_chrome.open_appearance_settings').
    :param arguments: Keyword arguments for the tool, as a dict (e.g. {'source_range': 'B2:F5'}). Use the exact parameter names listed for the tool. Omit or pass {} for tools that take no arguments.
    :returns: Tool output as a string, or an error message prefixed with 'mcp_tool_error: '.
    """
    import asyncio as _asyncio
    import logging as _log

    # Use the MCP proxy when BRH_MCP=1, otherwise call the upstream directly.
    # We intentionally bypass OsworldMcpClient.call_tool because that class uses a
    # three-server MCPConfig (osworld_mcp + filesystem + git).  fastmcp's
    # MCPConfigTransport creates a composite server for multi-server configs and
    # mounts each upstream with the server name as a prefix separator ('_'), so
    # 'google_chrome.open_appearance_settings' becomes unreachable — the client
    # would need to call 'osworld_mcp_google_chrome.open_appearance_settings'.
    # A single-server config connects directly, no prefix is applied.
    if os.environ.get("BRH_MCP") == "1":
        proxy_port = os.environ.get("BRH_MCP_PROXY_PORT", "9191")
        server_url = f"http://localhost:{proxy_port}/mcp"
    else:
        server_url = "http://localhost:9292/mcp"

    call_args = arguments or {}
    _log.getLogger(__name__).info("[call_mcp_tool] url=%s name=%s args=%s", server_url, name, call_args)

    try:
        from fastmcp import Client as _FastMCPClient

        # Circuit-breaker: a wedged in-guest MCP server makes the BRH proxy return
        # 504 and drives the streamable-http transport into an infinite GET-stream
        # reconnect loop, so call_tool() never returns and the whole run hangs.
        # Bound it so a stuck server raises TimeoutError -> caught below -> the
        # 'mcp_tool_error:' contract triggers the GUI fallback instead of hanging.
        mcp_timeout = float(os.environ.get("BRH_MCP_CALL_TIMEOUT", "45"))

        async def _call():
            cfg = {
                "mcpServers": {
                    "osworld_mcp": {"url": server_url, "transport": "streamable-http"}
                }
            }
            async with _FastMCPClient(cfg) as _client:
                return await _asyncio.wait_for(
                    _client.call_tool(name, call_args), timeout=mcp_timeout
                )

        result = _asyncio.run(_call())

        # Extract the textual payload from the CallToolResult. Using str(result)
        # would forward the whole 'CallToolResult(content=[TextContent(...)])'
        # wrapper (and any raw object repr inside) to the planner, which then
        # cannot tell success from a non-actionable result and never triggers the
        # GUI fallback.
        def _result_to_text(res) -> str:
            parts = []
            for block in getattr(res, "content", None) or []:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
            data = getattr(res, "data", None)
            if data is not None:
                return str(data)
            return str(res)

        text = _result_to_text(result)

        # A tool that reports an error must map to the 'mcp_tool_error:' contract
        # so the caller falls back to GUI navigation.
        if getattr(result, "is_error", False):
            return f"mcp_tool_error: {text}"

        _log.getLogger(__name__).info("[call_mcp_tool] result=%s", text[:200])
        return text
    except Exception as exc:
        _log.getLogger(__name__).error("[call_mcp_tool] error: %s", exc)
        return f"mcp_tool_error: {exc}"


def make_osworld_task_suite(
        system_defense_level: int = 0,
        unoptimized_cua: bool = False) -> TaskSuite[UIEnvironment]:

    tools = TOOLS_BASELINE if unoptimized_cua else TOOLS[system_defense_level]
    tool_fns = list(tools)
    if os.environ.get("BRH_MCP") == "1":
        tool_fns = [*tool_fns, call_mcp_tool]
    return TaskSuite[UIEnvironment](
        "osworld",
        UIEnvironment,
        [make_function(tool) for tool in tool_fns],
        SUITE_DATA,
    )
