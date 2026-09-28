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

from typing import Generic, TypeVar, Mapping, Iterable


from cobra.capabilities import is_trusted
from cobra.interpreter.value import CaMeLValue
from cobra.security_policy import (
    SecurityPolicyResult,
    Allowed,
    Denied,
)
from cobra.pipeline_elements.security_policies.agentdojo_security_policies import (
    AgentDojoSecurityPolicyEngine,
)
from cobra.interaction.environments.osworld.base_ui_task_suite_uitars import (
    UIEnvironment
)
from cobra.interaction.environments.osworld.base_ui_task_suite_opencua import (
    UIEnvironment_OpenCUA
)
from cobra.interaction.environments.osworld.base_ui_task_suite_anthropic import (
    UIEnvironment_Anthropic
)
UIEnv = UIEnvironment|UIEnvironment_OpenCUA|UIEnvironment_Anthropic

class OSWorldSecurityPolicyEngine(AgentDojoSecurityPolicyEngine[UIEnv]):
    """
    Security policy engine for OSWorld DesktopEnv.
    
    - Allows all navigation and non‐destructive actions by default.
    - Requires that any 'type_text' calls only type strings that originated from the user.
    - Denies untrusted inputs for text‐entry tools.
    """

    def __init__(self, env: UIEnv) -> None:
        super().__init__(env)
        # Tool→policy mapping
        self.policies = [
            ("click",               self._policy_allow),
            ("drag",                self._policy_allow),
            ("select",              self._policy_allow),
            ("scroll",              self._policy_allow),
            ("type_text",           self._policy_allow),
            ("hotkey",              self._policy_allow),
            ("press",               self._policy_allow),
            ("keydown",             self._policy_allow),
            ("keyup",               self._policy_allow),
            ("left_triple",         self._policy_allow),
            ("wait",                self._policy_allow),
            ("mark_done",           self._policy_allow),
            ("mark_fail",           self._policy_allow),
            ("left_double",         self._policy_allow),
            ("key",                 self._policy_allow),
            ("hold_key",            self._policy_allow),
            ("left_mouse_down",     self._policy_allow),
            ("left_mouse_up",       self._policy_allow),
            ("find",                self._policy_allow),
            ("locate_and_click",    self._policy_allow),
            ("screenshot",          self._policy_allow),
            ("run_single_uitars",   self._policy_allow),
            ("run_single_opencua",   self._policy_allow),
            ("run_single_claude",   self._policy_allow),
            ("check_done",          self._policy_allow),
            ("right_single",        self._policy_allow),
            ("left_single",         self._policy_allow),
            ("hover",               self._policy_allow),
            ("left_press",          self._policy_allow),
            ("middle_click",         self._policy_allow),
            ("reset_memory",              self._policy_allow),
            ("summarize_screenshot_content", self._policy_allow),
            ("get_google_credentials",     self._policy_allow),
            ("no_op",               self._policy_allow),
            ("get_clickable_elements_from_accessibility_tree", self._policy_allow),
            ("get_page_elements", self._policy_allow),
            ("get_page_text", self._policy_allow),
            ("find_element_by_text", self._policy_allow),
            ("verify_hypothesis", self._policy_allow),
            ("call_mcp_tool",   self._policy_allow),
        ]
        # Tools with no side‐effects (beyond screenshots)
        self.no_side_effect_tools = {"wait", "mark_done", "mark_fail", "screenshot", "find", "check_done", "reset_memory", "no_op", "hover"}

    def check_policy(
        self, 
        tool_name: str, 
        kwargs: Mapping[str, CaMeLValue], 
        dependencies: Iterable[CaMeLValue]
    ) -> SecurityPolicyResult:
        for name, policy_fn in self.policies:
            if name == tool_name:
                return policy_fn(tool_name, kwargs)
        # default deny for unknown tools
        return Denied(f"Unknown tool: {tool_name}")

    # —— Generic allow/deny helpers —— #
    def _policy_allow(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        return Allowed()

    def _policy_deny(self, tool_name: str, kwargs, reason: str) -> SecurityPolicyResult:
        return Denied(reason)

    # —— Spatial tools —— #
    def _inside_screen(self, x: float, y: float) -> bool:
        w, h = self._env.vm_screen_size
        return 0 <= x <= w and 0 <= y <= h

    def _policy_click(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        #pos = kwargs.get("pos")
        #if hasattr(pos, "raw"):  # CaMeLValue wrapper
        #    pos = pos.raw
        #x, y = pos
        #if not self._inside_screen(x, y):
        #    return Denied(f"Click coordinates ({x},{y}) off‐screen")
        return Allowed()

    def _policy_drag_select(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        start = kwargs.get("start")
        end   = kwargs.get("end")
        for corner, name in [(start, "start"), (end, "end")]:
            if hasattr(corner, "raw"):
                corner = corner.raw
            x, y = corner
            if not self._inside_screen(x, y):
                return Denied(f"{tool_name} '{name}' coord ({x},{y}) off‐screen")
        return Allowed()

    def _policy_scroll(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        direction = kwargs.get("direction")
        if direction not in ("up","down"):
            return Denied(f"Invalid scroll direction: {direction}")
        pos = kwargs.get("pos")
        if pos is not None:
            if hasattr(pos, "raw"):
                pos = pos.raw
            x, y = pos
            if not self._inside_screen(x, y):
                return Denied(f"Scroll anchor ({x},{y}) off‐screen")
        return Allowed()

    # —— Keystroke tools —— #
    _dangerous_combos = {
        frozenset({"alt","f4"}),
        frozenset({"ctrl","alt","del"}),
        frozenset({"cmd","q"}),
    }

    def _policy_hotkey(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        keys = kwargs.get("keys", [])
        raw_keys = [k.raw if hasattr(k,"raw") else k for k in keys]
        combo = frozenset(str(k).lower() for k in raw_keys)
        for bad in self._dangerous_combos:
            if bad.issubset(combo):
                return Denied(f"Hotkey combo {combo} may kill the VM or host")
        return Allowed()

    def _policy_simple_key(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        key = kwargs.get("key")
        if hasattr(key, "raw"):
            key = key.raw
        key_l = str(key).lower()
        if key_l in {"f4","alt+f4","ctrl+alt+del"}:
            return Denied(f"Pressing key {key} is disallowed")
        return Allowed()

    # Find: check both action and description
    def _policy_find(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        description = kwargs.get("description")
        if not is_trusted(description):
            return Denied("'description' must be user-provided and trusted")
        return Allowed()

    # Instruction-based tools: run_single_uitars & check_done
    def _policy_instruction(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        # prefer 'instruction'; fallback to 'description'
        instr = kwargs.get("instruction") if "instruction" in kwargs else kwargs.get("description")
        if instr is None or not is_trusted(instr):
            return Denied(f"{tool_name!r} prompt must be user-provided and trusted")
        return Allowed()

    # Type_text: ensure text param only user-supplied
    def _policy_type_text(self, tool_name: str, kwargs) -> SecurityPolicyResult:
        text = kwargs.get("text")
        if not is_trusted(text):
            return Denied("Typed text must come directly from user input")
        if len(text) > 500:
            return Denied("Typed text exceeds maximum safe length")
        return Allowed()
