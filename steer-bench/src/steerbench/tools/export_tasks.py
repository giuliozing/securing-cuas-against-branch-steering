"""Export the task declarations without benchmark runs or scores."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from steerbench.tasks import registry


OUTPUT = Path(__file__).resolve().parents[3] / "dataset" / "tasks.json"


def export() -> dict:
    return {
        "name": "STEER-Bench",
        "source": "src/steerbench/tasks/registry.py",
        "tasks": [{**dataclasses.asdict(task), "archetype": registry.archetype_of(task)}
                  for task in registry.TASKS],
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(export(), indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(registry.TASKS)} tasks to {OUTPUT}")


if __name__ == "__main__":
    main()
