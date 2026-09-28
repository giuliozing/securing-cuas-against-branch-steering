from pathlib import Path
import os
from typing import Annotated

from pydantic import Field

from anthropic import Anthropic

from agentdojo.functions_runtime import TaskEnvironment, make_function, Depends
from agentdojo.task_suite.task_suite import TaskSuite

from osworld.desktop_env.desktop_env import DesktopEnv
from osworld.mm_agents.anthropic.utils import (
    SYSTEM_PROMPT
)
from cobra.interaction.environments.base_ui_anthropic import (
    BaseUI_Anthropic,
    # Mouse actions
    left_single,
    left_double, 
    right_single,
    left_triple,
    middle_click,
    left_press,
    left_mouse_down,
    left_mouse_up,
    hover,
    drag,
    scroll,
    type_text,
    key,  
    hold_key,
    find,
    run_single_claude,
    check_done,
    summarize_screenshot_content,
    wait,
    mark_done,
    mark_fail,
    reset_memory,
    get_google_credentials,
    no_op,
    get_page_elements,
    get_page_text,
    find_element_by_text, 
    verify_hypothesis
)
from cobra.interaction.vlms.anthropic_vlm import ClaudeVLM


def build_ui_anthropic(
        defense_level: int = 0,
        q_llm: str | None = None,
        q_llm_second_check: str | None = None,
        path_to_vm: str | None = None,
        path_to_vlm: str | None = "claude-4-5-sonnet-20250929",
        token_count_file: str | None = None,
    ) -> BaseUI_Anthropic:  # <--- add argument
    if path_to_vm is None:
        raise ValueError("path_to_vm must be provided as an absolute path.")
    if not Path(path_to_vm).is_absolute():
        raise ValueError(f"path_to_vm must be an absolute path, got relative path: {path_to_vm}")
    env = DesktopEnv(
        path_to_vm=path_to_vm,
        action_space="pyautogui",
        screen_size=(1920,1080),
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=True,
    )
    
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"),timeout=300.0)

    vlm = ClaudeVLM(
        client=client,
        model=path_to_vlm, 
        enable_tools=True,
        system=SYSTEM_PROMPT
    )

    ui = BaseUI_Anthropic(env=env, vlm=vlm)
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


class UIEnvironment_Anthropic(TaskEnvironment):
    base_ui: Annotated[
        BaseUI_Anthropic,
        Depends(build_ui_anthropic),
    ] = Field(default=None, exclude=True)
    
    def model_copy(self, *, deep: bool = False) -> "UIEnvironment_Anthropic":
        # always do a shallow copy, even if deep=True
        return super().model_copy(deep=False)



TOOLS_0 = [
    # Mouse actions
    left_single,
    left_double,
    right_single,
    left_triple,
    middle_click,
    left_press,
    left_mouse_down,
    left_mouse_up,  
    hover,
    drag,
    scroll,
    type_text,
    key,     
    hold_key,            
    find,
    run_single_claude,
    check_done,
    summarize_screenshot_content,
    wait,
    mark_done,
    mark_fail,
    reset_memory,
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
TOOLS_1 = [tool for tool in TOOLS_0 if tool != run_single_claude]
TOOLS_2 = [tool for tool in TOOLS_1 if tool not in DISALLOWED]

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


def make_osworld_task_suite_Anthropic(
        system_defense_level: int = 0) -> TaskSuite[UIEnvironment_Anthropic]:
        
    return TaskSuite[UIEnvironment_Anthropic](
        "osworld",
        UIEnvironment_Anthropic,
        [make_function(tool) for tool in TOOLS[system_defense_level]],
        SUITE_DATA,
    )
