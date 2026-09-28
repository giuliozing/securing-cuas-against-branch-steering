# Code and research artifacts

This directory contains the code and benchmark artifacts described in the paper.
It has three independently installable packages:

- **`cobra/`** implements the COBRA defense: a deterministic plan interpreter,
  the Branch Resolution Hub, HTTP and MCP proxies, a privileged planner
  (P-LLM), a quarantined interaction model (Q-VLM), and the OSWorld adapters
  used in the utility evaluation. See [COBRA](cobra/README.md).

- **`steer-bench/`** contains STEER-Bench, the component attribution benchmark
  from the paper's Evaluation section. It includes 101 tasks in nine suites,
  the benchmark site, MCP tools, open and defended harnesses, and LLM-free
  oracles. Eight suites measure data-flow integrity (DFI); S7 checks the
  control-flow integrity (CFI) guarantee. See [STEER-Bench](steer-bench/README.md).

- **`mesa/`** contains MESA, which crawls a site and proposes an agent sitemap
  and MCP manifest for the site owner to review. See [MESA](mesa/README.md).

Install a package with `pip install -e .` from its directory. Each has its own
license; see [COBRA's notices](cobra/NOTICE.md) for third-party attribution.

## Citation

See [`CITATION.cff`](CITATION.cff).


