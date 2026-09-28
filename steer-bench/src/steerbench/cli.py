"""`steerbench` — the one documented entry point.

One verb list for every benchmark operation.

The **gate** split is the load-bearing part, and it is what makes the release possible:

    steerbench gate --substrate   # needs nothing but this package
    steerbench gate --defended    # needs the [brh] extra: our system under test

The substrate half certifies that the *environment* is sound — the pages render, the
tasks are well-formed, the attack is reachable. Anyone can run it. The defended half
certifies our components, and needs the COBRA/HTTP proxy stack that is deliberately not
part of this distribution.

**Honest boundary, stated rather than fudged.** `prompt_lint` belongs conceptually to
the substrate — it reads task text — but today it imports the defended runner (`harness/run.py`) to render a
prompt exactly as the runner would, and that pulls in the planner pipeline. Until the
prompt assembly is separated from the pipeline import it runs in the defended half. A
gate that claimed to be dependency-free and then failed on an import would be worse than
one that says where the line currently is.
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys

# (module, argv, human label) — argv is appended after the module's own name.
SUBSTRATE: list[tuple[str, list[str], str]] = [
    ("steerbench.tools.render_check", [], "render check (every task page, both variants)"),
]

DEFENDED: list[tuple[str, list[str], str]] = [
    ("steerbench.oracles.http_axis", [], "HTTP axis — 50 tasks"),
    ("steerbench.oracles.spec_axis", [], "specification axis — abstention"),
    ("steerbench.oracles.s1_s2", ["--suite", "S1"], "S1 WIRE"),
    ("steerbench.oracles.s1_s2", ["--suite", "S2"], "S2 DEST"),
    ("steerbench.oracles.s3", [], "S3 MCP"),
    ("steerbench.oracles.s4", [], "S4 SEAM"),
    ("steerbench.oracles.s5", [], "S5 PROV"),
    ("steerbench.oracles.s6", [], "S6 TRUST"),
    ("steerbench.oracles.s7", [], "S7 CFI"),
    ("steerbench.oracles.s8", [], "S8 STEP"),
    ("steerbench.oracles.s9", [], "S9 ULTRA"),
    ("steerbench.tools.prompt_lint", [], "prompt lint (arm equality, defence leakage)"),
    ("steerbench.tools.arms_selftest", [], "arm plumbing self-test"),
    ("steerbench.tools.validate_paid_paths", [], "paid path, validated unpaid"),
]


def _run_module(mod: str, argv: list[str]) -> int:
    """Each check in its own process: they bring up sites and proxies on real ports and
    mutate `os.environ`, so sharing an interpreter would make one check's leftovers
    another check's input — this isolation is what keeps cells independent."""
    return subprocess.run([sys.executable, "-m", mod, *argv]).returncode


def _substrate_plan() -> list[tuple[str, list[str], str]]:
    """The public half of the gate.

    `render_check` alone would understate it: the properties that make a third party's number comparable to ours — the
    task schema, the arm matrix, the two judges agreeing on what a breach is, the
    authoring conditions of S8, S9 and S7 — are all asserted in `tests/`, and a colleague
    running `steerbench gate --substrate` had no way to reach them. So the substrate
    tests run here too, in their own process, whenever the checkout has them (an
    installed wheel does not ship `tests/`, and a gate that failed on a missing directory
    would be worse than one that says what it could run)."""
    plan = list(SUBSTRATE)
    from steerbench import config
    tests = config.PROJECT / "tests"
    if tests.is_dir():
        plan.append(("pytest", ["-q", "-m", "substrate and not slow", str(tests)],
                     "substrate tests (schema, judge agreement, arms, archetypes)"))
    return plan


def cmd_gate(args: argparse.Namespace) -> int:
    both = not (args.substrate or args.defended)
    plan: list[tuple[str, list[str], str]] = []
    if args.substrate or both:
        plan += _substrate_plan()
    if args.defended or both:
        plan += DEFENDED

    failed: list[str] = []
    for mod, argv, label in plan:
        print(f"\n=== {label} ===", flush=True)
        if _run_module(mod, argv) != 0:
            failed.append(label)

    print("\n" + "#" * 70)
    if failed:
        print(f"GATE FAILED — {len(failed)}/{len(plan)} checks did not pass:")
        for f in failed:
            print(f"  x {f}")
        return 1
    print(f"GATE OK — {len(plan)}/{len(plan)} checks passed (LLM-free, no spend)")
    return 0


def cmd_run(args: argparse.Namespace, rest: list[str]) -> int:
    """The DEFENDED track. Spends: it asks a real planner for a plan."""
    sys.argv = ["steerbench-run", *rest]
    runpy.run_module("steerbench.harness.run", run_name="__main__")
    return 0


def cmd_open(args: argparse.Namespace, rest: list[str]) -> int:
    """The OPEN track: any agent, no enforcer, verdict from the environment alone."""
    from steerbench.harness import run_open
    return run_open.main(rest)


def cmd_site(args: argparse.Namespace) -> int:
    from steerbench.site.app import app
    app.run(host="0.0.0.0", port=args.port, threaded=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="steerbench", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="run the LLM-free certification (free, no spend)")
    g.add_argument("--substrate", action="store_true",
                   help="only the checks that need nothing but this package")
    g.add_argument("--defended", action="store_true",
                   help="only the checks that need the [brh] extra")

    sub.add_parser("run", help="the defended track (PAID: calls a planner)",
                   add_help=False)
    sub.add_parser("open", help="the open track: any agent, judged from the environment",
                   add_help=False)
    s = sub.add_parser("site", help="serve the benchmark site alone")
    s.add_argument("--port", type=int, default=8700)

    args, rest = ap.parse_known_args(argv)
    if args.cmd == "gate":
        return cmd_gate(args)
    if args.cmd == "run":
        return cmd_run(args, rest)
    if args.cmd == "open":
        return cmd_open(args, rest)
    if args.cmd == "site":
        return cmd_site(args)
    ap.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
