"""Full-screen terminal UI for curating and confirming sitemap/MCP calls.

Built on ``prompt_toolkit``. Presents the proposed calls as a checkbox list the
owner navigates with the arrow keys: toggle inclusion, edit a call's fields as
JSON, add a new call, or delete one. The same widget backs both stages —
stage 1 (select from proposals) and stage 2 (confirm/tweak the LLM's refined
output) — differing only in title/subtitle.

Design goal: a compact, keyboard-driven panel in the spirit of the Claude Code
terminal UI. Degrades to a no-op passthrough when stdin/stdout is not a TTY.

All dynamic text is rendered via ``FormattedText`` tuples (never ``HTML``) so
call names, descriptions, and error messages containing ``< > & '`` cannot
break rendering.
"""

from __future__ import annotations

import json
import sys

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import DynamicContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from .models import HttpCall, McpCall, call_from_dict, call_to_dict

_STYLE = Style.from_dict(
    {
        "header": "bold #3fb950",
        "subtitle": "#9a9a9a",
        "frame.border": "#3fb950",
        "sel": "reverse bold",
        "http": "#56d364",
        "mcp": "#2ea043",
        "dim": "#8a8a8a",
        "ok": "#7fbf7f",
        "err": "bold #e05561",
        "help": "#7a7a7a",
        "key": "bold #3fb950",
    }
)

_HINT_LIST = "space toggle · e edit · a add · d delete · enter confirm"

_NEW_TEMPLATE = {
    "type": "http",
    "name": "new_call",
    "description": "",
    "method": "GET",
    "url": "https://",
    "category": "",
    "tags": [],
    "body_fields": [],
    "priority": 2,
}


def _trunc(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


class CallEditor:
    def __init__(self, items, title: str, subtitle: str):
        self.items = list(items)
        self.title = title
        self.subtitle = subtitle
        self.index = 0
        self.mode = "list"  # "list" | "edit"
        self.status = ("ok", _HINT_LIST)
        self.result = None  # set on confirm
        self.text_area: TextArea | None = None
        self._editing_new = False

        self.list_window = Window(
            content=FormattedTextControl(
                self._list_fragments, focusable=True, get_cursor_position=self._cursor
            ),
            wrap_lines=False,
        )
        self.kb = self._build_keys()
        self.app: Application | None = None

    # ---- list rendering -------------------------------------------------
    def _cursor(self) -> Point:
        return Point(0, self.index)

    def _list_fragments(self) -> FormattedText:
        frags: list[tuple[str, str]] = []
        if not self.items:
            return FormattedText([("class:dim", "  (no calls — press 'a' to add one)\n")])
        for i, c in enumerate(self.items):
            cur = i == self.index
            base = "class:sel" if cur else ""
            pointer = "❯" if cur else " "
            box = "◉" if getattr(c, "selected", True) else "○"
            kind_style = "class:sel" if cur else ("class:http" if c.kind == "http" else "class:mcp")
            frags.append((base, f" {pointer} {box} "))
            frags.append((kind_style, f"{c.kind.upper():4} "))
            frags.append((base, c.name))
            frags.append(("class:sel" if cur else "class:dim", f"   {_trunc(c.description, 58)}\n"))
        return FormattedText(frags)

    def _header(self) -> FormattedText:
        return FormattedText(
            [("class:header", self.title), ("", "\n"), ("class:subtitle", self.subtitle)]
        )

    def _status_bar(self) -> FormattedText:
        style, msg = self.status
        return FormattedText([(f"class:{style}", msg)])

    def _help_bar(self) -> FormattedText:
        if self.mode == "edit":
            return FormattedText(
                [("class:key", "C-s"), ("class:help", " save   "),
                 ("class:key", "C-c"), ("class:help", " cancel edit")]
            )
        pairs = [
            ("↑/↓", "move"), ("space", "toggle"), ("e", "edit"),
            ("a", "add"), ("d", "delete"), ("enter", "confirm"), ("C-c", "quit"),
        ]
        frags: list[tuple[str, str]] = []
        for key, label in pairs:
            frags.append(("class:key", key))
            frags.append(("class:help", f" {label}   "))
        return FormattedText(frags)

    # ---- container ------------------------------------------------------
    def _container(self):
        if self.mode == "edit" and self.text_area is not None:
            body = Frame(self.text_area, title="edit call (JSON)")
        else:
            body = Frame(self.list_window, title="calls")
        return HSplit(
            [
                Window(FormattedTextControl(self._header), height=2),
                body,
                Window(FormattedTextControl(self._status_bar), height=1),
                Window(FormattedTextControl(self._help_bar), height=1),
            ]
        )

    # ---- key bindings ---------------------------------------------------
    def _build_keys(self) -> KeyBindings:
        kb = KeyBindings()
        list_mode = Condition(lambda: self.mode == "list")
        edit_mode = Condition(lambda: self.mode == "edit")

        @kb.add("up", filter=list_mode)
        @kb.add("k", filter=list_mode)
        def _(event):
            if self.items:
                self.index = (self.index - 1) % len(self.items)

        @kb.add("down", filter=list_mode)
        @kb.add("j", filter=list_mode)
        def _(event):
            if self.items:
                self.index = (self.index + 1) % len(self.items)

        @kb.add("space", filter=list_mode)
        def _(event):
            if self.items:
                c = self.items[self.index]
                c.selected = not getattr(c, "selected", True)

        @kb.add("e", filter=list_mode)
        def _(event):
            if self.items:
                self._open_editor(call_to_dict(self.items[self.index]), new=False)

        @kb.add("a", filter=list_mode)
        def _(event):
            self._open_editor(dict(_NEW_TEMPLATE), new=True)

        @kb.add("d", filter=list_mode)
        @kb.add("x", filter=list_mode)
        def _(event):
            if self.items:
                del self.items[self.index]
                self.index = max(0, min(self.index, len(self.items) - 1))

        @kb.add("enter", filter=list_mode)
        def _(event):
            self.result = [c for c in self.items if getattr(c, "selected", True)]
            event.app.exit()

        @kb.add("c-c", filter=list_mode)
        @kb.add("q", filter=list_mode)
        def _(event):
            self.result = None
            event.app.exit()

        @kb.add("c-s", filter=edit_mode)
        def _(event):
            self._save_editor()

        @kb.add("c-c", filter=edit_mode)
        def _(event):
            self._close_editor()

        return kb

    # ---- editor ---------------------------------------------------------
    def _open_editor(self, data: dict, *, new: bool):
        self._editing_new = new
        self.text_area = TextArea(
            text=json.dumps(data, indent=2, ensure_ascii=False),
            multiline=True,
            scrollbar=True,
            focusable=True,
            wrap_lines=False,
        )
        self.mode = "edit"
        self.status = ("ok", "edit fields, then C-s to save")
        if self.app is not None:
            self.app.layout.focus(self.text_area)

    def _close_editor(self):
        self.mode = "list"
        self.text_area = None
        self.status = ("ok", _HINT_LIST)
        if self.app is not None:
            self.app.layout.focus(self.list_window)

    def _save_editor(self):
        assert self.text_area is not None
        try:
            data = json.loads(self.text_area.text)
            call = call_from_dict(data, selected=True)
        except (json.JSONDecodeError, ValueError) as e:
            self.status = ("err", f"invalid: {e}")
            return
        if self._editing_new:
            self.items.append(call)
            self.index = len(self.items) - 1
        else:
            call.selected = getattr(self.items[self.index], "selected", True)
            self.items[self.index] = call
        self._close_editor()

    # ---- run ------------------------------------------------------------
    def run(self, *, _input=None, _output=None):
        # ``_input``/``_output`` are test-only hooks (piped input + dummy
        # output) so the event loop can be driven headlessly; production
        # callers pass neither and get the real terminal.
        self.app = Application(
            layout=Layout(DynamicContainer(self._container), focused_element=self.list_window),
            key_bindings=self.kb,
            style=_STYLE,
            full_screen=True,
            mouse_support=False,
            input=_input,
            output=_output,
        )
        self.app.run()
        return self.result


def curate_calls(http_calls, mcp_calls, *, title: str, subtitle: str):
    """Run the editor. Returns (http, mcp) filtered to selected, or None if cancelled.

    Falls back to returning the inputs unchanged when stdin/stdout is not a TTY.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return http_calls, mcp_calls

    items = [*http_calls, *mcp_calls]
    result = CallEditor(items, title, subtitle).run()
    if result is None:
        return None
    http = [c for c in result if isinstance(c, HttpCall)]
    mcp = [c for c in result if isinstance(c, McpCall)]
    return http, mcp
