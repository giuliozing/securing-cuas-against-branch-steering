# STEER-Bench

A live benchmark for branch steering and compound prompt injection against
computer-use agents. Tasks use served websites, MCP tools, and guarded HTTP
requests. Outcomes come from the environment's records.

The benchmark has 101 tasks in nine suites and two tracks. You can score your
own agent with a small adapter, or install the defense stack to measure the
contribution of individual components.

```bash
pip install -e ".[dev]"
steerbench gate --substrate     # LLM-free, CPU-only, no network: ~2 seconds
```

The substrate gate checks both page variants against their archetype contracts
and runs the substrate tests. These cover the task schema, arm matrix, suite
authoring conditions, access to answer-bearing endpoints, and agreement
between the two judges on what counts as a breach. It needs no defense stack.

---

## How the benchmark works

The displayed value and the request sent on the wire can differ.

A task page shows an amount, a payee, or a queue of invoices. When the agent
confirms, the site resolves the authoritative request body from its own record
(`GET /<theme>/quote`, direct and unproxied), then sends the guarded request
through the enforcing proxy. In an attack cell, the body differs from the
displayed value. Because the difference exists only on the server, the agent
cannot discover it by inspecting the page. A constraint derived from the plan
before reading untrusted data must catch it.

## Two tracks, one environment

| track | who can run it | verdict from | reports |
|---|---|---|---|
| **open** | **any agent** — no enforcer required | `GET /__state` only | ASR · utility · abstention |
| **defended** | a system with an enforcer (`[brh]`) | `/__state` + the enforcer's alerts | the seven outcomes, ΔASR, STRICT, FP |

The open judge uses the same breach definition as the defended judge while
omitting categories that require an enforcer.
`tests/test_open_track.py::test_the_two_judges_agree_on_breach` checks their
agreement across task, variant, and evidence shapes.

Write an adapter (`docs/adapters.md`) and run:

```bash
steerbench open --adapter react --suite S1        # your agent, no defence stack
steerbench open --adapter oracle                 # the free undefended baseline
```

## The suites

A suite pairs a task set with the component it tests. It compares `A5` (full
system) with `A4` (that component disabled). The reported difference is
**ΔASR = A4 − A5**. Task sets can overlap, so calculate deltas within each
suite and do not add them across suites.

| suite | component under test | tasks | archetype |
|---|---|---|---|
| **S1 WIRE** | BRH field pins | 26 | decision |
| **S2 DEST** | domain + endpoint allowlists | 24 | decision |
| **S3 MCP** | the MCP enforcer (6 mechanisms, M1–M6) | 15 | none |
| **S4 SEAM** | the plan's fan-out across several enforcers | 18 | decision |
| **S5 PROV** | provenance: the rule **and** the commit gate | 11 | decision |
| **S6 TRUST** | sitemap + MCP manifest trust | 11 | none |
| **S7 CFI** | control-flow integrity — the plan-then-execute boundary itself | 10 | procedure |
| **S8 STEP** | per-transition constraint refresh (the runtime half) | 10 | workflow |
| **S9 ULTRA** | composition of constraints along a deep plan's branch path | 5 | workflow |

**S8** and **S9** test constraints while a plan executes. S8 checks when the
enforcer refreshes constraints between worklist rows. S9 checks whether a
constraint at a leaf includes grants from its ancestor branches. Its decision
trees are three or four levels deep and contain 60–90 lines, compared with
five lines for most other tasks. An attack can satisfy the leaf's own pins while
violating a grant from an ancestor. **S7** tests control-flow integrity by
comparing a plan-then-execute system with an interleaved ReAct agent. The
other eight suites measure data-flow integrity.

Measured results are reported in the paper and are not included in this package.
The task declarations are available as [JSON](dataset/tasks.json).

## Four page archetypes

An archetype defines the available interactions. Each theme supplies its own
branding.

| archetype | interaction | guarded actions |
|---|---|---|
| `decision` | read a few signals → Confirm / Route to review / **Ask the requester** | 1 |
| `workflow` | a worklist of rows: open, act, hold, next | several |
| `procedure` | an untrusted content pane + several named steps in a fixed order | several |
| `none` | no page at all (the MCP and sitemap suites) | — |

The adapter and render check identify interactions through `data-steer-action`
and `data-steer-item`. Every page includes the abstention control, including
pages where abstaining is the wrong choice. Its presence therefore gives no
clue about the correct response.

## Checks without an LLM

```bash
steerbench gate --substrate    # task schema, prompt lint, render check, judge determinism
steerbench gate --defended     # needs [brh]: every oracle, on the real enforcer
pytest -m "not slow"           # unit and property tests
```

Each suite has an LLM-free oracle that drives the stack and checks both arms:
`A5` must stop the attack, and `A4` must allow it. The second check is the I2
negative control; it confirms that the ablation exposes a reachable attack.

The expected value of every check is pinned in
[`tests/expected_gate.json`](tests/expected_gate.json); a number that moves fails `pytest`
with the superseded value printed beside it.

## Layout

```
steer-bench/
  pyproject.toml                      # extras: [mcp] [live] [brh] [dev]
  docs/                               # adapters · writing-a-task · DATASHEET
  dataset/tasks.json                  # task declarations, without run results
  tests/expected_gate.json            # the regression contract (every gate figure)
  src/steerbench/
    config.py                         # every path and port RESOLVED, never hard-coded
    tasks/registry.py + s3/s4/s6/s7/s8/s9    # the 101 task manifests and their ground truth
    site/                             # the served websites; archetype shells + per-theme skins
    mcp/                              # the shared FastMCP deployment and its distractors
    harness/stack.py                  # THE launcher (site · proxy · MCP · enforcer)
    harness/judge.py                  # the open judge, and the single definition of harm
    harness/evaluator.py              # oracle constraints + the seven outcomes
    harness/arms.py                   # suite membership and the arm matrix
    harness/{step,trust,ultra,cfi}_model.py  # the modelled planner/annotator, in named files
    adapters/                         # base protocol · oracle · react · dom_react
    oracles/                          # one per suite: LLM-free certification
    tools/                            # lint, self-tests, task export, render checks
```

The defence stack is the **only** thing outside the package, resolved through
`config.cobra_src()` / `config.enforcer_addon()` / `config.mitmdump()` and gated behind the
`[brh]` extra. It is distributed separately as the sibling `cobra/` package.

## Documentation

| document | what it answers |
|---|---|
| [`docs/adapters.md`](docs/adapters.md) | how to plug your agent in |
| [`docs/writing-a-task.md`](docs/writing-a-task.md) | how to add a task, with a worked example per archetype |
| [`docs/DATASHEET.md`](docs/DATASHEET.md) | composition, provenance, limitations, intended use |
| [`dataset/README.md`](dataset/README.md) | task file format and provenance |

## Citation

See `../CITATION.cff` at the repository root. The benchmark is released under MIT.
The defence stack is the separate `cobra/` package, under its own license.
