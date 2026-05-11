# Longitudinal benchmark v4 — 30 sessions, mechanical criteria, graded outcomes

The v2 + v3 benches saturated `useful_axis = 1.000` from session 1 onward because every
session ran a single `intent add --level task` followed by a single accept — there was no
way for the axis to land between 0 and 1.

v4 fixes that by structuring each session around one task with **3 step intents**, each
step verdict-checked **mechanically** (no operator-confirmation step). Steps within a task
may reject independently, so `useful_axis = satisfied_step_weight / total_step_weight`
actually moves.

## Structure

- 30 sessions across 5 arcs of 6. Sessions within an arc share semantic vocabulary so
  arrow1 has a real cluster to promote on. The last arc (E, 25-30) is cross-arc and tests
  whether earlier arcs' memory helps.

| Arc | Sessions | Cluster              |
|-----|----------|----------------------|
| A   | 1–6      | Recall path          |
| B   | 7–12     | Consolidation        |
| C   | 13–18    | State / eval         |
| D   | 19–24    | CLI surface          |
| E   | 25–30    | Cross-arc (recall)   |

- Each arc opens with a `goal`-level umbrella intent. Goal weight is 10x task weight, so a
  completed arc lands a heavy chunk of `useful_axis`.

- Each task adds 3 `step` intents under the task. Each step has one **mechanical** verdict
  criterion that the evaluator checks post-claude-p.

## Criterion forms

The evaluator (`_probe/longitudinal_v4/evaluate_steps.py`) supports four:

| Form                              | Accept condition                                          |
|-----------------------------------|-----------------------------------------------------------|
| `regex:<pattern>:stdout`          | Pattern matches the claude -p response body               |
| `regex:<pattern>:<repo-file>`     | Pattern matches the file's current contents              |
| `not_regex:<pattern>:stdout`      | Pattern does NOT match the response (anti-hallucination) |
| `file_exists:<repo-relative-path>`| File exists in the repo after the session                |
| `exit_code:0:<shell-command>`     | Command exits 0 (run from repo root)                      |

Patterns are Python `re` regex; everything is case-insensitive (`re.IGNORECASE`).

## Files

```
benchmarks/user_bench/harness_longitudinal_v4/
  goals.jsonl          # 5 arc-level umbrella intents
  tasks.jsonl          # 30 tasks; each carries arc + 3 steps + criteria
  README.md            # this file
_probe/longitudinal_v4/
  run_session.py       # one session: session-mark, intents, claude -p, capture
  evaluate_steps.py    # mechanical criterion → intent accept|reject per step
  post_session.py      # consolidate + expertise + scorecard snapshot
  run_all.py           # drives sessions 1..30 sequentially, resumable
  FINDINGS.md          # written after the bench completes
```

## Running the bench

```
# from repo root
python _probe/longitudinal_v4/run_all.py            # full bench
python _probe/longitudinal_v4/run_all.py --from 7   # resume from session 7
python _probe/longitudinal_v4/run_all.py --only 1   # single session
```

The runner is idempotent: if `s{N}_response.json` already exists for session N, that
session's claude -p call is skipped and only the post-claude steps re-run. Delete the
file (or `--force`) to re-run end-to-end.

## What the bench is — and isn't — testing

Testing:
- Whether `useful_axis` / `effortless_axis` can actually move on a closed-loop substrate
  with graded outcomes (the v3 axes were structurally saturated).
- Whether arrow1 cluster promotion fires on semantically-dense task arcs.
- Whether recall_hit_rate trends upward as memory accumulates.
- Whether `secondary_recall_dropped_at` shifts from `daemon_unreachable` / cold-start
  reasons early to `ok` / `below_recurrence` late.

Not testing rigorously:
- Whether memory **causally** improved agent answers. That requires a paired
  memory-on-vs-memory-off bench, which is a separate experiment.
