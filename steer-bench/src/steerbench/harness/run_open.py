
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

from steerbench import config
from steerbench.adapters.base import TaskView
from steerbench.harness import judge, stack
from steerbench.tasks import registry as reg

BUILTIN_ADAPTERS = {
    "oracle": "steerbench.adapters.oracle:OracleAdapter",
    "react": "steerbench.adapters.react:ReactAdapter",
    # A ReAct loop over the page's declared affordances instead of pixels, for models
    # that decide well and ground badly. A DIFFERENT agent, not a configuration of the
    # one above — see `adapters/dom_react.py` for why the two are never ranked against
    # each other and what is comparable anyway.
    "react-dom": "steerbench.adapters.dom_react:DomReactAdapter",
}


def load_adapter(spec: str):
    """`oracle` / `react`, or any `module:Class` — third-party adapters are first-class."""
    target = BUILTIN_ADAPTERS.get(spec, spec)
    if ":" not in target:
        raise SystemExit(f"unknown adapter {spec!r}; use one of "
                         f"{sorted(BUILTIN_ADAPTERS)} or module:Class")
    mod_name, cls_name = target.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise SystemExit(f"adapter {spec!r} could not be imported: {e}") from None
    return getattr(mod, cls_name)()


def select(only: str, channels: set[str], suites: str = "") -> list:
    """The tasks this run covers, by channel, tid prefix and suite.

    `suites` is the selector an outside user reaches for first — "run S1" is how the
    benchmark is described, while a tid prefix presumes you already know the registry.
    Membership comes from `arms.suites_of`, the same function the defended track uses, so
    the two tracks cannot disagree about what "suite S1" contains."""
    from steerbench.harness import arms

    want = {s.strip().upper() for s in suites.split(",") if s.strip()}
    # S4 needs its own refusal, and silence here would be worse than an error. Its 18
    # tasks are BORROWED from S1/S2 and are on the open track for their own primary leg;
    # selecting "S4" would quietly return them and label an S1 measurement as a seam
    # result. S6 needs no special case — its tasks have no open channel at all, so the
    # empty selection already says so.
    if "S4" in want:
        raise SystemExit(
            "suite S4 is not on the open track: its premise is a COMPROMISED executor "
            "issuing an off-plan call, which an honest agent cannot produce — the cell "
            "would measure an agent declining to attack itself. Its 18 tasks are already "
            "here under S1/S2, which is the leg an open-track run actually exercises.")
    out = []
    for t in reg.TASKS:
        ch = reg.open_channel(t)
        if ch is None or ch not in channels:
            continue
        if only and not any(t.tid.startswith(p) for p in only.split(",")):
            continue
        if want and not (want & set(arms.suites_of(t))):
            continue
        out.append(t)
    return out


def instruction_for(task, posture: str) -> str:
    if not posture:
        return task.instruction
    return {
        "specified": task.instruction_specified,
        "underspecified": task.instruction_underspecified,
        "delegated": task.instruction_delegated,
    }[posture]()


def run_cell(adapter, task, variant: str, posture: str, st: stack.Stack) -> dict:
    cfg = (task.spec_config(variant) if posture
           else (task.benign_config() if variant == "benign" else task.attack_config()))
    config.control("POST", "/__reset", port=st.site_port)
    config.control("POST", "/__config", cfg, port=st.site_port)

    channel = reg.open_channel(task)
    view = TaskView(
        tid=task.tid, theme=task.theme, channel=channel,
        instruction=instruction_for(task, posture),
        page_url=st.direct if channel == "gui" else None,
        mcp_url=os.environ.get("STEERBENCH_OPEN_MCP") if channel == "mcp" else None,
        posture=posture, archetype=reg.archetype_of(task),
    )

    error = None
    t0 = time.time()
    try:
        adapter.run(view)
    except Exception as e:                                   # noqa: BLE001
        # One crashing task must not truncate a run; it is recorded and scored on the
        # evidence, which for a crash is usually NOTHING.
        error = repr(e)
    took = round(time.time() - t0, 2)

    actions = config.control("GET", "/__state", port=st.site_port).get("actions", [])
    return {
        "adapter": adapter.name, "tid": task.tid, "theme": task.theme,
        "channel": channel, "variant": variant, "posture": posture,
        "outcome": judge.open_outcome(variant, actions, task),
        "n_actions": len(actions),
        "paths": [a.get("path") for a in actions],
        "seconds": took, "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="oracle",
                    help="oracle | react | module:Class")
    ap.add_argument("--only", default="", help="comma tid prefixes")
    ap.add_argument("--suite", default="",
                    help="comma suites, e.g. S1,S8 (membership per arms.suites_of)")
    ap.add_argument("--variants", default="benign,attack")
    ap.add_argument("--spec", default="",
                    help="comma spec postures (specified,underspecified); "
                         "`delegated` is not on the open track")
    ap.add_argument("--channels", default="gui",
                    help="comma channels to run: gui,mcp")
    ap.add_argument("--out", default="")
    ap.add_argument("--site-port", type=int, default=None)
    args = ap.parse_args(argv)

    if "delegated" in args.spec.split(","):
        raise SystemExit(
            "the `delegated` posture is not on the open track: it needs the explicit "
            "commit surface (surfaces/spec.submit_action(committed=…)) so a capability "
            "gate can inspect the declaration, and that has no GUI equivalent. Run it "
            "on the defended track.")

    channels = {c for c in args.channels.split(",") if c}
    tasks = select(args.only, channels, args.suite)
    if not tasks:
        raise SystemExit(
            f"no tasks selected (channels={sorted(channels)} suite={args.suite or '*'} "
            f"only={args.only or '*'}) — note S3's tasks are on the `mcp` channel and "
            f"S4/S6 are not on the open track at all (`tasks/registry.open_channel`)")

    postures = [p for p in args.spec.split(",") if p] or [""]
    variants = [v for v in args.variants.split(",") if v]

    adapter = load_adapter(args.adapter)
    out_path = Path(args.out) if args.out else (
        config.RESULTS / f"open_{adapter.name}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The open track's environment must be set BEFORE the site is spawned, not after.
    #
    # On this track the guarded action is issued by the SITE process — a grounded
    # click hits `/<theme>/confirm`, and `driver.submit_action` runs inside Flask — so the
    # remap that turns `http://bank.local/...` into the loopback site has to be in *that*
    # process's environment. `config.subprocess_env` copies `os.environ` at spawn time, so
    # setting it afterwards would reach the runner and nothing else: every guarded action
    # would leave for a hostname that does not resolve, and the cell would score NOTHING,
    # which is indistinguishable from an agent that looked at the page and declined to act.
    #
    # `STEERWEB_PROXY` is cleared here for the mirror-image reason: a leaked value from a
    # defended run in the same shell would put an enforcer in front of a track whose whole
    # definition is that there is none.
    site_port = config.alloc_port(args.site_port if args.site_port is not None
                                  else config.SITE_PORT)
    os.environ["STEERBENCH_REMAP"] = f"http://127.0.0.1:{site_port}"
    os.environ["STEERWEB_DIRECT"] = os.environ["STEERBENCH_REMAP"]
    os.environ["STEERWEB_PROXY"] = ""      # explicitly none — see `stack.Stack.env`
    st = stack.build(proxy=False, site_port=site_port, prefix="steerbench_open_")
    if "mcp" in channels:
        # The MCP channel with NO MCP proxy in front of it: the agent talks to the FastMCP
        # server directly. That is what undefended means here — the same asymmetry as
        # the HTTP channel running without the HTTP proxy.
        mcp_port = config.alloc_port(config.MCP_PORT)
        stack.start_mcp_server(st, mcp_port)
        os.environ["STEERBENCH_OPEN_MCP"] = f"http://127.0.0.1:{mcp_port}/mcp"
        print(f"  MCP (unenforced): {os.environ['STEERBENCH_OPEN_MCP']}")
    if hasattr(adapter, "proxy"):
        adapter.proxy = None

    print(f"OPEN track · adapter={adapter.name} · site={st.direct} · "
          f"tasks={len(tasks)} · channels={sorted(channels)}")

    rows: list[dict] = []
    try:
        with out_path.open("w") as f:
            for task in tasks:
                for posture in postures:
                    if posture and not task.on_spec_axis:
                        continue
                    for variant in variants:
                        row = run_cell(adapter, task, variant, posture, st)
                        rows.append(row)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(f"  {row['outcome']:<10} {task.tid:<24} "
                              f"{(posture or '-'):<14} {variant:<7} "
                              f"paths={row['paths']}"
                              + (f"  ERROR={row['error']}" if row["error"] else ""))
    finally:
        st.close()
        os.environ.pop("STEERBENCH_REMAP", None)

    s = judge.summarise(rows)
    print("\n" + "#" * 70)
    print(f"OPEN TRACK · adapter={adapter.name} · {s['cells']} cells")
    if s["asr"] is not None:
        print(f"  ASR            {s['breaches']}/{s['attack_cells']} "
              f"({s['asr']*100:.0f}%)   — a harmful action reached the wire")
    if s["utility"] is not None:
        print(f"  utility        {s['completed']}/{s['benign_cells']} "
              f"({s['utility']*100:.0f}%)   — honest work completed")
    print(f"  abstention     {s['abstained_correctly']}/{s['abstention_expected']} "
          f"correct · {s['false_abstentions']} false")
    print(f"  reviewed {s['reviewed']} · nothing {s['nothing']} · errors {s['errors']}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
