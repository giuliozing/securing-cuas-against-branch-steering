# Datasheet — STEER-Bench

This datasheet follows *Datasheets for Datasets* (Gebru et al.). The task
declarations are in [`../dataset/tasks.json`](../dataset/tasks.json) and are
generated from `src/steerbench/tasks/registry.py` with
`python -m steerbench.tools.export_tasks`. Measured results are in the paper,
not this package.

## Motivation

STEER-Bench measures whether an agent defense resists *branch steering*: a
guarded action uses a value, destination, or timing that differs from what the
agent perceived. It also measures the effect of individual defense components.

It was built for a research project on branch-aware sandboxing for
computer-use agents. See `CITATION.cff`.

## Composition

101 tasks. Each is a scenario: a served website, an optional MCP tool surface, a guarded
action on an enforced HTTP wire, and two variants (**benign** and **attack**) that differ
only on the layer the task claims to attack.

| dimension | breakdown |
|---|---|
| suites (membership; task-sets overlap) | S1 26 · S2 24 · S3 15 · S4 18 · S5 11 · S6 11 · S7 10 · S8 10 · S9 5 |
| archetypes | decision 50 · workflow 15 · procedure 9 · none 27 |
| open track | gui 74 · mcp 16 · excluded 11 |
| block reasons | `brh_field` · `brh_domain` · `brh_endpoint` · `mpt_*` |
| plan shape (minimum sufficient plan) | 5 lines and one branch for 90 tasks · 25 lines and five branches for S8 · **60–90 lines and a tree three or four levels deep for S9** |
| bespoke templates | 50/101 (the rest render their archetype's shared shell with a per-theme skin) |

**Instances are synthetic.** Values, names, accounts, addresses, and documents
were created for this benchmark. They contain no real personal data or scraped
content. Hostnames such as `finance.local` and `evil.com` are mapped to the
local site.

**Labels.** Each task declares its expected enforcement reason and the values
for its benign and attack variants. The JSON export is generated directly from
the registry. Evaluated adapters receive only `adapters/base.TaskView`, which
does not expose these answers.

## Collection process

Tasks were authored, not collected. Each one is checked before it counts:

* `render_check` — the page honours its archetype's contract in both variants, every
  displayed value is actually on the page, and no wire value leaks onto it;
* `prompt_lint` — the arms' prompts are byte-identical wherever the ablation is
  enforcement-side, and the task text never teaches the defence;
* the suite's **oracle** — the cell is driven end-to-end through the real enforcer with no
  model, asserting both that the defence holds and that the attack **lands** with the
  component off.

## Uses

The benchmark supports evaluation of branch steering defenses, comparison of
an agent's open-track behavior with a defended configuration, and attribution
to individual defense components. It is not a general agent capability test.
The attack's server-side divergence also makes it unsuitable for measuring
prompt-injection detection from the screen alone.

Open-track ASR is measured without an enforcer. Defended-track ASR includes
the enforcer. These scores answer different questions.



## Distribution

The task declarations and benchmark code are released under the package's
[MIT license](../LICENSE). Measured run records and result tables are excluded
from this package. Task IDs identify scenarios and should remain stable when
the dataset is extended.
