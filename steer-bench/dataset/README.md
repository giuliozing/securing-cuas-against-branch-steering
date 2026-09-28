# STEER-Bench task data

[`tasks.json`](tasks.json) contains the 101 task declarations from
`src/steerbench/tasks/registry.py`. Generate it with
`python -m steerbench.tools.export_tasks` after installing the package. The
export contains task definitions and ground truth, with no run logs, model
outputs, or measured scores. The Python registry remains the source of truth.

The file is a JSON object with `name`, `source`, and `tasks`. Each entry in
`tasks` is a direct serialization of a `Task` dataclass. Common fields include:

| Field | Meaning |
|---|---|
| `tid`, `theme`, `title`, `family` | Stable identifier and scenario labels |
| `instruction` | Instruction presented to the agent |
| `archetype`, `template`, `skin` | Page interaction type and presentation |
| `domain`, `action_path`, `action_fields` | Guarded request destination and body fields |
| `signals`, `disp` | Values displayed to the agent |
| `real_attack`, `attack_kind`, `attack_channels` | Attack variant and delivery channels |
| `expect_attack_reason` | Expected enforcement reason |
| `mcp`, `trust`, `step`, `ultra`, `cfi` | Suite-specific declarations; `null` when absent |

Nested objects retain the field names of the dataclasses in `registry.py`.
Python tuples become JSON arrays. `disp` describes the displayed values;
`real_attack` describes values resolved by the site on the attack variant.
These fields include answers and must not be passed to an evaluated agent.
The adapter API in [`docs/adapters.md`](../docs/adapters.md) exposes only its
restricted `TaskView`.

Task scenarios were authored for this benchmark. The source and this export
are distributed under the package's [MIT license](../LICENSE). See the
[datasheet](../docs/DATASHEET.md) for composition and limitations. See the
[package README](../README.md) for installation and evaluation commands.
