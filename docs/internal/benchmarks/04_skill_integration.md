# Benchmark 4 — Skill / agent task completion

> **Status: deferred to v2.0.1.** Methodology locked here; harness +
> committed numbers ship in v2.0.1. v1's user-doc surfaces this bench
> as a "coming soon" link that points here.

## What it measures

When a real Anthropic skill or Claude Code workflow uses
`rlat skill-context`, do tasks complete more reliably and at lower token
cost than the same skill running with grep/glob primitives only?

Headline metrics:

- **Task success rate** = `n_completed_correctly` / `n_tasks`
  (per skill, per approach)
- **Tool-use turns to completion** — proxy for agent-loop divergence
- **Total tokens per completed task**
- **$ per completed task**

## Approaches per skill

| Key | Approach |
|---|---|
| `rlat_inject` | skill receives `rlat skill-context` markdown blocks at the start |
| `grep_glob_only` | skill has only grep + Read; no rlat |
| `read_only` | skill has only whole-file Read; no search |
| `no_corpus` | skill has no corpus access (control) |

## Three skills under test

1. **deep-research** — multi-source synthesis. Tasks: "summarise the
   evolution of feature X across the codebase". Measures whether
   `rlat skill-context` replaces the multi-source dance grep+Read
   normally requires.
2. **find-and-fix-bug** — single-file mutation. Tasks: "the test
   `tests/foo.py:test_bar` is failing — locate and fix the cause".
   Measures whether rlat helps locate the relevant code faster.
3. **explain-this-codebase** — architectural Q&A. Tasks: "what does
   X module do, who calls it, why was Y the design choice".
   Measures whether the primer + skill-context combination beats
   grep-from-scratch.

30 hand-designed tasks per skill (90 total) with ground-truth completion
criteria. LLM-as-judge grades against the criterion.

## Why deferred to v2.0.1

The 90 hand-designed tasks are real product work — each task is a 5-15
minute design + validate + rubric exercise. 90 × 10 min = 15 hours.
Plus three skills need to be either real (deep-research is referenced in
CLAUDE.md as a real skill) or simulable.

Estimated total: 4 days dev + ~$120-200 API. Slips Stage 1.

## Locked references

- The `deep-research` skill is referenced in CLAUDE.md as already
  implemented. Reuse its real surface, don't re-implement.
- `feedback_skill_cartridge_wiring.md`: skills reference existing
  knowledge models via SKILL.md frontmatter — don't build per-skill
  cartridges. Bench 4 honours this: every skill points at
  `resonance-lattice.rlat`.

## Reproducibility (when it ships)

```bash
pip install rlat[bench]
rlat install-encoder
rlat build ./docs ./src -o resonance-lattice.rlat
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.skill_integration.run \
  --skill deep-research --skill find-and-fix-bug --skill explain-codebase \
  --output benchmarks/results/user_bench/skill_integration.json \
  --budget-usd 200
```

## Honest framing

- Skill-success rate is bounded by the skill's own quality. A poorly-
  designed skill won't complete tasks even with perfect retrieval. The
  bench measures the *delta* attributable to rlat, not the absolute
  skill score.
- Tool-use turns is a noisy metric — skills sometimes succeed on turn 2,
  sometimes turn 5, with no obvious cause. We report median + interquartile
  range, not just mean.
- We don't claim rlat helps every skill. If `find-and-fix-bug` doesn't
  benefit (because the failure is a deterministic stack trace), we
  report that result and note the conditions under which rlat helps vs
  doesn't.
