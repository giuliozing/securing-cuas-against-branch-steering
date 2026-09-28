"""OpenCUA grounding for the STEER-Bench browser loop."""

from __future__ import annotations

import ast
import os
import re

from steerbench.agent.uitars_grounding import Action, smart_resize

_SECTION = {
    "observation": re.compile(r"^##\s*Observation\s*:?[\n\r]+(.*?)(?=^##|\Z)",
                              re.DOTALL | re.MULTILINE),
    "thought": re.compile(r"^##\s*Thought\s*:?[\n\r]+(.*?)(?=^##|\Z)",
                          re.DOTALL | re.MULTILINE),
    "action": re.compile(r"^##\s*Action\s*:?[\n\r]+(.*?)(?=^##|\Z)",
                         re.DOTALL | re.MULTILINE),
}
_CODE = re.compile(r"```(?:code|python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_CALL = re.compile(r"(?:pyautogui|computer)\.\w+\([^)]*\)")

#: pyautogui verb -> the `Action.type` `browser_env.apply` dispatches on. `moveTo` and
#: `dragTo` are absent deliberately: these pages have no drag affordance, and mapping them
#: onto a click would turn a model that did something meaningless into a model that did
#: something wrong.
_VERBS = {
    "click": "click", "leftClick": "click", "doubleClick": "left_double",
    "tripleClick": "left_double", "rightClick": "right_single",
    "write": "type", "typewrite": "type", "press": "hotkey", "hotkey": "hotkey",
    "scroll": "scroll",
}
#: Positional parameter names, so `pyautogui.click(500, 300)` reads the same as
#: `pyautogui.click(x=500, y=300)`. Taken from `opencua_agent.function_parameters`.
_POSITIONAL = {
    "click": ("x", "y", "clicks", "interval", "button", "duration"),
    "leftClick": ("x", "y", "clicks", "interval", "button", "duration"),
    "doubleClick": ("x", "y", "interval", "button", "duration"),
    "tripleClick": ("x", "y", "interval", "button", "duration"),
    "rightClick": ("x", "y", "duration"),
    "write": ("message", "interval"),
    "typewrite": ("message", "interval"),
    "press": ("keys", "presses", "interval"),
    "hotkey": (),
    "scroll": ("clicks", "x", "y"),
}


def to_pixels(x: float, y: float, view_w: int, view_h: int) -> tuple[int, int]:
    """A model coordinate, in viewport pixels.

    Two conventions occur and both are handled, because getting this wrong does not look
    like a coordinate bug: every click lands in the top-left corner and the run reads as a
    grounding failure. `relative` (OpenCUA's default) is a fraction of the screen;
    `qwen25` is a pixel in the smart-resized image the vision tower actually saw."""
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(round(x * view_w)), int(round(y * view_h))
    sr_h, sr_w = smart_resize(view_h, view_w)
    return int(round(x / sr_w * view_w)), int(round(y / sr_h * view_h))


def parse(text: str, view_w: int, view_h: int) -> Action:
    """One OpenCUA response -> one grounded `Action`.

    An unparseable response becomes `wait` rather than an exception, which is the same
    choice `uitars_grounding.parse` makes and for the same reason: a model that emitted
    prose is a *result* about that model, and a harness that crashed on it would report an
    infrastructure failure where there is a behavioural one. The raw text is carried on
    the action so the transcript still shows what happened.
    """
    parts = {k: (m.group(1).strip() if (m := rx.search(text)) else "")
             for k, rx in _SECTION.items()}
    thought = parts["thought"] or parts["observation"]
    blocks = _CODE.findall(text)
    if not blocks:
        return Action(type="wait", thought=thought, raw=text)
    code = blocks[-1].strip()
    low = code.lower()
    if "computer.terminate" in low or "terminate(" in low:
        return Action(type="finished", thought=thought, raw=code,
                      text="failure" if "fail" in low else "success")
    if "computer.wait" in low:
        return Action(type="wait", thought=thought, raw=code)

    call = None
    for m in _CALL.finditer(code):
        call = m.group(0)                       # the LAST call is the effective one
    if call is None:
        return Action(type="wait", thought=thought, raw=code)
    try:
        node = ast.parse(call, mode="eval").body
        verb = node.func.attr
        names = _POSITIONAL.get(verb, ())
        args = {names[i]: ast.literal_eval(a)
                for i, a in enumerate(node.args) if i < len(names)}
        # `hotkey('ctrl', 'a')` is variadic: its positional args ARE the keys.
        if verb == "hotkey":
            args["keys"] = [ast.literal_eval(a) for a in node.args]
        args.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
    except Exception:                            # noqa: BLE001 - see the docstring
        return Action(type="wait", thought=thought, raw=code)

    act = Action(type=_VERBS.get(verb, "wait"), thought=thought, raw=call)
    if act.type in ("click", "left_double", "right_single") and "x" in args:
        act.x, act.y = to_pixels(float(args["x"]), float(args["y"]), view_w, view_h)
    elif act.type == "type":
        act.text = str(args.get("message", ""))
    elif act.type == "hotkey":
        keys = args.get("keys", "")
        act.text = " ".join(keys) if isinstance(keys, (list, tuple)) else str(keys)
    elif act.type == "scroll":
        clicks = float(args.get("clicks", 0) or 0)
        act.direction = "up" if clicks > 0 else "down"
        if "x" in args:
            act.x, act.y = to_pixels(float(args["x"]), float(args["y"]), view_w, view_h)
        else:
            act.x, act.y = view_w // 2, view_h // 2
    return act


class OpenCUAGrounder:
    """One vLLM OpenCUA client, with the same `step()` contract as `UITARSGrounder`.

    The history is the model's own `## Action` lines, which is what OpenCUA's
    `ACTION_HISTORY_TEMPLATE` feeds back in the OSWorld agent. Screenshots are NOT
    accumulated: the loop sends the current one, so a long run cannot silently exceed the
    server's context and start failing in a way that looks like the model giving up.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 cot_level: str = "l2") -> None:
        import openai
        from osworld.mm_agents.opencua.prompts import build_sys_prompt

        self.model = model or os.environ.get("OPENCUA_MODEL", "opencua")
        base = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8001/v1")
        self.client = openai.OpenAI(
            base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
        self.system_prompt = build_sys_prompt(level=cot_level, use_random=False)
        self.history: list[str] = []

    def _user_text(self, instruction: str) -> str:
        text = f"# Task Instruction:\n{instruction}\n"
        if self.history:
            text += ("\nYou have already taken the following actions:\n"
                     + "\n".join(f"Step {i + 1}: {a}"
                                 for i, a in enumerate(self.history[-6:]))
                     + "\n")
        text += "\nPlease generate the next move according to the screenshot, the task "\
                "instruction and the previous steps.\n"
        return text

    def step(self, instruction: str, image_b64: str, view_w: int, view_h: int) -> Action:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": self._user_text(instruction)},
                ]},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
        act = parse(text, view_w, view_h)
        self.history.append(act.raw or act.type)
        return act
