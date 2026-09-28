"""Screenshot every task's page to a PNG — the visual catalogue of the benchmark.

Spins the site, POSTs each task's config, navigates a headless Chromium to the page and
saves a full-page PNG. No proxy: rendering is the perception channel, never the guarded
action, so nothing here is enforced.

**Which tasks it covers, and the rule it must not get wrong.** Every task whose archetype
is not `none` — and a page is found the way `site/app.page` finds it: the theme's bespoke
template if it ships one, otherwise its archetype's shared shell. Requiring a
bespoke template, or iterating `on_http_axis`, would silently skip every task that
renders from a shell (e.g. S8's worklists): a tool demanding a file the renderer does
not need reports working code as missing.

Needs the `[live]` extra and a browser (`python -m playwright install chromium`).

    python -m steerbench.tools.shoot                      # benign, every page
    python -m steerbench.tools.shoot --variant attack     # the attack variant
    python -m steerbench.tools.shoot --archetype document,workflow
    python -m steerbench.tools.shoot --only 91,77
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

from steerbench import config
from steerbench.tasks import registry as reg
from steerbench.agent.browser_env import BrowserEnv  # noqa: E402


def has_page(task) -> bool:
    """The renderer's own rule, restated once (`site/app.page`)."""
    arch = reg.archetype_of(task)
    if arch == "none":
        return False
    bespoke = config.PKG / "site" / "templates" / task.template / "page.html"
    shell = config.PKG / "site" / "templates" / "archetypes" / arch / "page.html"
    return bespoke.exists() or shell.exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="benign", choices=["benign", "attack"])
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="comma tid prefixes")
    ap.add_argument("--archetype", default="", help="comma archetypes to include")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        config.RESULTS / ("screenshots" if args.variant == "benign"
                          else "screenshots_attack"))
    out.mkdir(parents=True, exist_ok=True)

    want = {a.strip() for a in args.archetype.split(",") if a.strip()}
    prefixes = [p for p in args.only.split(",") if p]
    tasks = [t for t in reg.TASKS
             if has_page(t)
             and (not want or reg.archetype_of(t) in want)
             and (not prefixes or any(t.tid.startswith(p) for p in prefixes))]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    server = subprocess.Popen(
        [str(Path(config.python_for_subprocess())), str(config.SITE_APP),
         str(config.SITE_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    env = None
    seen: Counter = Counter()
    try:
        config.wait_port(config.SITE_PORT)
        env = BrowserEnv(headless=True).start()
        for t in tasks:
            arch = reg.archetype_of(t)
            cfg = t.benign_config() if args.variant == "benign" else t.attack_config()
            config.control("POST", "/__config", cfg)
            env.goto(f"http://127.0.0.1:{config.SITE_PORT}/{t.theme}/")
            png = out / f"{t.tid}.png"
            env.page.screenshot(path=str(png), full_page=True)
            seen[arch] += 1
            print(f"  {t.tid:<22} {arch:<9} {t.family:<15} -> {png.name}")
    finally:
        if env:
            env.close()
        server.terminate()
    print(f"\n{sum(seen.values())} screenshots ({args.variant}) in {out}")
    print("  " + " · ".join(f"{k} {v}" for k, v in sorted(seen.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
