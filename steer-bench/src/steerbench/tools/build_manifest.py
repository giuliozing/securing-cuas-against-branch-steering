"""Compatibility entry point for exporting STEER-Bench task declarations.

The task export lives in ``dataset/tasks.json``. This command does not produce
experimental results or coverage reports.
"""

from __future__ import annotations

from steerbench.tools.export_tasks import main


if __name__ == "__main__":
    main()
