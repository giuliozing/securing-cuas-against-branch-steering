# Keep injection tasks if you use them
from .injection_tasks import *  # noqa: F401

# 1) Public suite = the exact same instance user_tasks will register into
# from .base_ui_task_suite import task_suite as osworld_task_suite
from .base_ui_task_suite_uitars import make_osworld_task_suite
from .base_ui_task_suite_opencua import make_osworld_task_suite_OpenCUA
from .base_ui_task_suite_anthropic import make_osworld_task_suite_Anthropic
# 2) Import the user_tasks module for its side-effects (auto-scan + register)
from . import user_tasks as _user_tasks  # DO NOT star-import; just load module

# 3) Re-export the CLI → suite-id map so main can import from the package
CLI_TO_SUITE_ID = getattr(_user_tasks, "CLI_TO_SUITE_ID", {})

# 4) Optional sanity check: make sure both modules see the SAME suite object
# assert make_osworld_task_suite() is _user_tasks.task_suite, (
#     f"Suite identity mismatch: {id(make_osworld_task_suite())} vs {id(_user_tasks.task_suite)}"
# )

__all__ = ["osworld_task_suite", "CLI_TO_SUITE_ID", "make_osworld_task_suite", "make_osworld_task_suite_OpenCUA", "make_osworld_task_suite_Anthropic"]
