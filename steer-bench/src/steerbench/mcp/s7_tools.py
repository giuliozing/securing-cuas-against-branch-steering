

from __future__ import annotations

import json
import os
import urllib.request

from steerbench.mcp import s3_tools

#: The off-plan verb takes the argument its lure supplies; the planned verbs take the
#: record reference. Neither is pinned — S7 constrains which tools may be called, never
#: what they carry (invariant I1), so a parameter rule here would let the cell pass for
#: a reason that has nothing to do with control flow.
_PARAMS = {"set_payout_account": ("iban", "ref")}


def _direct() -> str:
    return os.environ.get("STEERWEB_DIRECT", "http://127.0.0.1:8700")


def _make_reader(theme: str, verb: str):
    """A read tool that fetches the ACTIVE document from the site.

    Fetched rather than closed over: the benign and attack documents are the same task
    with a different `/__config`, so a tool holding the text at build time would serve
    the benign document for every cell and the attack would silently never happen."""

    def _read() -> str:
        url = f"{_direct()}/{theme}/__content"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as r:
            return json.loads(r.read().decode()).get("content", "")

    _read.__name__ = verb
    return _read


def build_for_task(task):
    spec = getattr(task, "cfi", None)
    if spec is None or spec.channel != "mcp":
        return
    if spec.read_tool:
        yield (f"{task.theme}.{spec.read_tool}",
               _make_reader(task.theme, spec.read_tool),
               f"Read the record pack for {task.title}.")
    for st in spec.affordances():
        yield (f"{task.theme}.{st.name}",
               s3_tools._make(task.theme, st.name, _PARAMS.get(st.name, ("ref",)),
                              harmful=st.role == "act"),
               st.describe or st.label)


def build_all(tasks):
    for task in tasks:
        yield from build_for_task(task)
