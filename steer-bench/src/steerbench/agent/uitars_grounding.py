"""UI-TARS grounding for the STEER-Bench browser loop."""

from __future__ import annotations

import ast
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# UITARS action-space + prompt template (pure strings; imports cleanly).
from osworld.mm_agents.prompts import (  # noqa: E402
    UITARS_NORMAL_ACTION_SPACE,
    UITARS_USR_PROMPT_THOUGHT,
)

# --- Qwen2-VL smart-resize (replicated from old_uitars_agent.py) -------------

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def _round_by(n: float, f: int) -> int:
    return round(n / f) * f


def _ceil_by(n: float, f: int) -> int:
    return math.ceil(n / f) * f


def _floor_by(n: float, f: int) -> int:
    return math.floor(n / f) * f


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError("aspect ratio too extreme for smart_resize")
    h_bar = max(factor, _round_by(height, factor))
    w_bar = max(factor, _round_by(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by(height / beta, factor)
        w_bar = _floor_by(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by(height * beta, factor)
        w_bar = _ceil_by(width * beta, factor)
    return h_bar, w_bar


# --- action parsing ----------------------------------------------------------


@dataclass
class Action:
    type: str                       # click | left_double | right_single | type | hotkey | scroll | wait | finished
    x: int | None = None            # pixel (viewport) for pointer actions
    y: int | None = None
    text: str = ""                  # type content / hotkey / finished content
    direction: str = ""             # scroll direction
    thought: str = ""
    raw: str = ""


_ACTION_RE = re.compile(r"Action:\s*(.+)", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\s*Action:|$)", re.DOTALL)


def _center_pixels(box: str, view_w: int, view_h: int) -> tuple[int, int]:
    """`box` is '(x,y)' or '(x1,y1,x2,y2)' in smart-resized pixel space; return the
    box centre in real viewport pixels."""
    nums = [float(n) for n in box.replace("(", "").replace(")", "").split(",") if n.strip()]
    sr_h, sr_w = smart_resize(view_h, view_w)
    fx, fy = [], []
    for i, n in enumerate(nums):
        (fy if (i + 1) % 2 == 0 else fx).append(n / (sr_h if (i + 1) % 2 == 0 else sr_w))
    cx = sum(fx) / len(fx)
    cy = sum(fy) / len(fy)
    return int(round(cx * view_w)), int(round(cy * view_h))


def parse(text: str, view_w: int, view_h: int) -> Action:
    """Parse a UI-TARS 'Thought:/Action:' response into a grounded Action."""
    thought = ""
    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group(1).strip()
    m = _ACTION_RE.search(text)
    if not m:
        return Action(type="wait", thought=thought, raw=text)
    action_str = m.group(1).split("\n\n")[0].strip()
    # strip UI-TARS box markers so ast can parse the call
    clean = action_str.replace("<|box_start|>", "").replace("<|box_end|>", "")
    try:
        node = ast.parse(clean, mode="eval").body
        fn = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        kwargs = {kw.arg: (kw.value.value if isinstance(kw.value, ast.Constant) else None)
                  for kw in node.keywords}
    except Exception:
        return Action(type="wait", thought=thought, raw=text)

    act = Action(type=fn, thought=thought, raw=action_str)
    if fn in ("click", "left_double", "right_single", "scroll"):
        box = kwargs.get("start_box") or ""
        if box:
            act.x, act.y = _center_pixels(box, view_w, view_h)
        act.direction = kwargs.get("direction", "") or ""
    elif fn == "type":
        act.text = kwargs.get("content", "") or ""
    elif fn == "hotkey":
        act.text = kwargs.get("key", "") or ""
    elif fn == "finished":
        act.text = kwargs.get("content", "") or ""
    return act


# --- UITARS client -----------------------------------------------------------


class UITARSGrounder:
    """One vLLM UI-TARS client. `step()` returns the parsed grounded Action for
    the current screenshot; a short textual history curbs loops."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 language: str = "English") -> None:
        import openai
        self.model = model or os.environ.get("UITARS_MODEL", "UI-TARS-1.5-7B")
        base = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        self.client = openai.OpenAI(base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
        self.language = language
        self.history: list[str] = []

    def _prompt(self, instruction: str) -> str:
        base = UITARS_USR_PROMPT_THOUGHT.format(
            action_space=UITARS_NORMAL_ACTION_SPACE,
            language=self.language,
            instruction=instruction,
        )
        if self.history:
            base += "\n\n## Previous actions\n" + "\n".join(self.history[-6:])
        return base

    def step(self, instruction: str, image_b64: str, view_w: int, view_h: int) -> Action:
        prompt = self._prompt(instruction)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]}],
            temperature=0.0,
            max_tokens=512,
        )
        text = resp.choices[0].message.content or ""
        act = parse(text, view_w, view_h)
        self.history.append(f"Action: {act.raw}")
        return act
