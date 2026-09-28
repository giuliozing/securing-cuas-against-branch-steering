# Third-party notices

This package is licensed under Apache License 2.0 (see `LICENSE`). It adapts
the following published artifacts under their respective Apache-2.0 licenses:

- **`interpreter/`, `pipeline_elements/`, `capabilities/`, `security_policy.py`,
  `system_prompt_generator.py`, `models.py`, `chat_turn.py`,
  `conditional_cache.py`, `count_tokens.py`, `custom_yaml.py`,
  `qllm_vision.py`, `quarantined_llm.py`** — adapted from the `camel`
  interpreter released alongside E. Debenedetti, I. Shumailov, T. Fan,
  J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, F. Tramèr, *Defeating
  Prompt Injections by Design*, arXiv:2503.18813, 2025 (Google, Google
  DeepMind, ETH Zürich). This underlying interpreter and type system is
  referred to as "CaMeL" in code retained from that release; COBRA's own
  contributions (`brh/`, `mcp_proxy/`, `http_proxy/`) are new code built on
  top of it.

- **`http_proxy/`** — adapted from `cellmate`, released alongside L. Meng,
  H. Feng, I. Shumailov, E. Fernandes, *Cellmate: Sandboxing Browser AI
  Agents*, arXiv:2512.12594, 2025, and extended here into a branch-aware,
  stateful HTTP enforcement proxy driven by COBRA's Branch Resolution Hub.

- **`interaction/`** — the Q-VLM interaction-model and OSWorld
  environment-adapter layer (VLM wrappers and `UIEnvironment*` classes)
  extends the CaMeL-CUA research artifact accompanying H. Foerster,
  T. Blanchard, K. Nikolić, I. Shumailov, C. Zhang, R. Mullins,
  N. Papernot, F. Tramèr, Y. Zhao, *Camels Can Use Computers Too:
  System-Level Security for Computer Use Agents*, arXiv:2601.09923, 2026,
  which is itself an extension of the interpreter above.

The upstream code carries no guarantee of correctness, security, or fitness
for a particular purpose. Consult each project's repository and license for
its terms. Upstream copyright headers remain unchanged.
