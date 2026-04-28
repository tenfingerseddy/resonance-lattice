# Skill Integration — Spec

Internal design document for `rlat skill-context`. The user-facing guide is at [docs/user/SKILLS.md](../user/SKILLS.md).

## Design thesis

Skills are execution instructions; rlat is grounded context. The integration is the existing Anthropic `!command` primitive — a skill body calls `rlat skill-context` and the shell command's stdout replaces the placeholder before the model sees the skill.

This inverts the v0.11 design. v0.11 added a `cartridges:` array to skill frontmatter and assumed a downstream consumer would resolve it. With MCP deferred there was no consumer; the field was decorative. v2.0 makes rlat a great citizen of the skill system that already exists, instead of asking the skill system to know about rlat.

## Anthropic skill primitives we depend on

Spec-verified against [code.claude.com/docs/en/skills.md](https://code.claude.com/docs/en/skills.md) and [code.claude.com/docs/en/hooks.md](https://code.claude.com/docs/en/hooks.md):

| Primitive | What it does | Documented at |
|-----------|--------------|----------------|
| `` !`<command>` `` block | Runs the shell command before the skill is sent to the model; stdout replaces the placeholder | skills.md#inject-dynamic-context |
| Argument substitution | `$arg_name` in skill body interpolates from `arguments:` frontmatter at invocation time | skills.md (frontmatter reference) |
| Unknown frontmatter keys | Silently ignored by the YAML parser | skills.md (no explicit validation rejection documented) |
| Relative-link references | Skills can link to `reference.md` etc. via markdown; loaded by Claude tool calls when read | skills.md#add-supporting-files |

## Anthropic primitives we deliberately do *not* depend on

- **Hooks for context injection.** Hooks observe and block; they cannot modify what Claude sees in the prompt. SessionStart can write to `$CLAUDE_ENV_FILE` but not to the conversation. PreToolUse can deny but not augment.
- **Custom frontmatter keys with consumers.** No documented extension/passthrough mechanism (`x-*`, `custom_metadata`, `ext`). Adding `knowledge: foo.rlat` would be safe (silently ignored) but pointless without a consumer.
- **Skill-scoped hooks** (`hooks:` frontmatter field). Documented for automation/blocking, not context augmentation.

## CLI surface

```
rlat skill-context <km.rlat> --query Q [--query Q ...]
                  [--top-k N] [--token-budget T]
                  [--source-root DIR] [--strict]
                  [--mode augment|knowledge|constrain]
```

Single entry point, no subcommands. Designed to compose: a skill body emits one or more `!command` blocks, each calling `rlat skill-context` with a different query set or different `.rlat`.

The same `--mode` flag exists on `rlat search --format context` and resolves through the same shared module ([cli/_grounding.py](../../src/resonance_lattice/cli/_grounding.py)) so the directive header and gate semantics stay identical across both LLM-facing surfaces.

## Output format contract

The harness suite [tests/harness/skill_context.py](../../tests/harness/skill_context.py) enforces eight contracts. Breaking any of them breaks downstream skill parsing.

### 1. Per-query header line

Every query block opens with an HTML comment carrying `ConfidenceMetrics` and the active `--mode`:

```
<!-- rlat skill-context query='<query>' band=<name> mode=<mode> top1_score=N.NNN top1_top2_gap=N.NNN source_diversity=N.NN drift_fraction=N.NN -->
```

Format-checked via the regex in `_check_format` in the harness suite. Skill instructions can reference these metrics directly to tell the model when to hedge.

### 2. Section heading

Each query gets a markdown H2 heading:

```
## Context for: "<query>"
```

### 3. Per-passage anchor + drift tag + score

Each passage block:

```
> **source: `<source_file>:<char_offset>+<char_length>`** — score N.NNN `[<drift_status>]`
>
> <passage text, line-prefixed with "> ">
```

The anchor format `<source_file>:<char_offset>+<char_length>` is the same as `cli/search.py` text format — consumers that already parse `rlat search` output can parse skill-context output identically.

### 4. Drift banner (warn mode)

When any returned passage has `drift_status != "verified"` and `--strict` is not set, output begins with:

```
> ⚠ **DRIFT WARNING**: at least one passage below has stale or missing source.
> Treat content as advisory; refresh with `rlat refresh <km>` for canonical results.
```

### 5. Drift abort (strict mode)

`--strict` + any drift → exit code 2 + stderr error message. No banner; no partial output (stdout-buffer is *not* flushed before the error).

Verified 2026-04-28 against Claude Code: a non-zero exit from any inline `` !`<command>` `` block aborts the skill render entirely, surfaces a visible `Shell command failed for pattern "..."` error to the user (with stdout echoed into the error context, not the prompt), and the skill content never enters the conversation. Skill authors can therefore rely on rc=2 (drift) and rc=3 (`--strict-names`) as a hard gate — Claude Code itself enforces "do not load this skill" on non-zero exit; downstream tooling does not need to inspect the exit code separately. Anything written to stdout on the failure path appears in the user-visible error, so the rc=non-zero path must not emit secrets or noisy diagnostics.

### 6. Grounding-mode header (always ships)

Every invocation stamps a directive at the top of stdout *before* any drift banner or query block:

```
<!-- rlat-mode: <augment|knowledge|constrain> -->
> **Grounding mode: <mode>.** <imperative directive to the consumer LLM>
```

The directive ships unconditionally — even when the per-query body is suppressed by the gate, even when `--strict` aborts non-zero (the directive doesn't ship in that case because nothing ships, but on success or warn-mode it always does). This is what tells the consumer LLM how to *read* the output; without it the LLM has no defined relationship to the corpus.

### 7. Constrain mode never suppresses

Under `--mode constrain`, the gate is disabled (`min_top1_score=0.0`, `max_drift_fraction=1.0`). The full passage body always renders regardless of confidence. Refusal on thin evidence is the consumer LLM's job — the directive instructs it explicitly.

### 8. Augment / knowledge modes suppress on weak retrieval

Under `--mode augment`, the gate fires when `top1_score < 0.30` (cosine floor below which retrieval has genuinely failed for this query) OR `drift_fraction > 0.30`. Under `--mode knowledge`, the lighter floor `top1_score < 0.15` triggers (drift is allowed). When the gate fires, the dynamic body is replaced with a marker that surfaces *which* metric tripped:

```
*(no confident evidence under mode=`augment`; top1_score=0.215, drift_fraction=0.00)*
```

The directive header still ships, so the LLM sees the mode instruction and the suppression marker — it can either fall back to training (augment / knowledge) or refuse (constrain).

The augment threshold was moved from `top1_top2_gap < 0.05` to `top1_score < 0.30` after bench 2 (2026-04-27) showed the gap-based gate over-suppresses on paraphrase-rich corpora — top-1 ≈ top-2 is the *signal* of strong retrieval (same fact stated multiple ways), not weak retrieval. Bench 2 v4 (2026-04-28, Microsoft Fabric, 63 questions, single-shot lane) drove the default to `augment` — on broad documentation corpora the LLM partially knows, augment yields 76.5% answerable accuracy at 3.9% hallucination, strictly better than the LLM alone (56.9% / 19.6%). `constrain` remains the safety floor (66.7% / 2.0% / 91.7% distractor refusal) for compliance / audit workloads. The `knowledge` mode under `rlat deep-search` ties augment at 92.2% accuracy / **0% hallucination** — the launch recommendation when the deep-search loop runs.

Thresholds live in [cli/_grounding.py](../../src/resonance_lattice/cli/_grounding.py:`_THRESHOLDS`); revising them is a backward-incompatible change to the gate semantics and should be locked into BENCHMARK_GATE.md.

### 9. Distinctive-name verification (warn + strict)

A second gate independent of the score-based grounding-mode gate: per query, the rendered passage text is checked against the question's distinctive proper nouns / acronyms / alphanumeric IDs. Tokens not present in any retrieved passage trigger a refusal directive:

```
<!-- rlat-namecheck: missing `MVE` -->
> ⚠ **Name verification failed.** The question references `MVE`, but no
> retrieved passage contains this exact name. The corpus may describe an
> adjacent or differently-named entity. Do NOT answer as if the corpus's
> content is about `MVE`. …
```

Default behaviour: prepend the directive to the body. `--strict-names` aborts non-zero (rc=3) so a skill loader can gate. The check skips when the body is already mode-suppressed (would be redundant noise).

This addresses the name-aliasing distractor failure mode that score-based gating cannot — bench 2 v3 showed `top1_score` distributions overlap completely between answerable and distractor questions on paraphrase-rich documentation corpora. Score-floor falsified, name-check the right mitigation. Canonical case fb37: question said `MVE`, corpus said `MLV`, all rlat lanes hallucinated.

Implementation: [cli/_namecheck.py](../../src/resonance_lattice/cli/_namecheck.py). Distinctive-token heuristic = ALL-CAPS acronyms ≥2 chars, alphanumeric IDs (digit + alpha), capitalised words ≥3 chars not in the product-domain stopword list. Stopwords skip common English function words plus `microsoft`, `azure`, `power`, `bi`, `data`, `table`, etc. — words common enough in question phrasing that requiring verbatim presence would over-refuse.

Cross-reference: [docs/internal/benchmarks/02_distractor_floor_analysis.md](benchmarks/02_distractor_floor_analysis.md).

### 10. `rlat deep-search` — multi-hop research loop

A separate CLI verb, not a `skill-context` flag — but the contract is the same shape: in-process retrieval, namecheck across the union of all hops, mode-aware grounding directives.

`rlat deep-search km.rlat "<q>"` runs a 4-hop loop (planner → search → refine → answer) and returns a synthesised answer + the evidence union. On the Fabric 5-lane bench (relaxed rubric):

| Approach | Accuracy | Hallucination | $/q |
|---|---:|---:|---:|
| `rlat search --format context --mode augment` (single-shot) | 76.5% | 3.9% | $0.004 |
| `rlat deep-search` | **92.2%** | **2.0%** | $0.010 |

Anthropic-only for v2.0 (Sonnet 4.6 hardcoded). Use this when correctness matters more than latency / spend. Implementation: [src/resonance_lattice/deep_search/](../../src/resonance_lattice/deep_search/), CLI in [cli/deep_search.py](../../src/resonance_lattice/cli/deep_search.py).

## Token budget

`--token-budget` × `MaterialiserConfig.chars_per_token` = char budget. The truncation rule:

- Iterate query blocks in submission order
- Add the next block if `(used + len(block)) <= budget`
- Stop when the next add would overflow
- **Carve-out**: the first block is *always* kept, even if it alone overflows. The first block is the skill-author's first preset; truncating it silently is worse than overflowing the budget. An oversized single passage is the corpus author's problem to fix.

This means: skill authors should put their highest-priority preset first, then lower-priority presets, then `$user_query` last. Then the user-query is the first thing dropped under budget pressure — matching the intuition that preset context is more important than dynamic context.

## Reuse

`cli/skill_context.py` reuses existing primitives end-to-end:

| Primitive | Source | What for |
|-----------|--------|----------|
| `load_or_exit` | `cli/_load.py` | Friendly archive load + error on bad path |
| `open_store_or_exit` | `cli/_load.py` | Friendly store open + error on unsupported mode |
| `field.retrieve` | `field/__init__.py` | ANN/dense dispatch (built on top of `dense.search` + `ann.search`) |
| `field.encoder.Encoder` | `field/encoder.py` | Encoder with auto-runtime ONNX/OpenVINO selection |
| `store.verified.verify_hits` | `store/verified.py` | Per-hit drift status + materialised text |
| `MaterialiserConfig.chars_per_token` | `config.py` | Conservative chars/token heuristic for budget |

Confidence metric formulae (`top1_score`, `top1_top2_gap`, `source_diversity`, `drift_fraction`) are inlined in `_confidence_metrics` rather than reusing `rql.evidence` because `rql.CitationHit` drops per-passage drift status (it's only summarised in the aggregate metric). The CLI needs per-passage drift to render the `[verified]` / `[drifted]` / `[missing]` tag, so it computes the same formulae from `VerifiedHit` directly.

## What this kills from v0.11

- `cartridges:` plural frontmatter array → not needed; `!command` does the work
- Preset registries in skill frontmatter (encoder choice, retrieval mode, rerank settings) → single recipe, nothing to express
- Hook-based skill-knowledge loader → impossible per Anthropic docs (hooks can't inject)
- "skill-aware" rlat that reads SKILL.md → unnecessary inversion; skill calls rlat, not the reverse

## Future extensions (not v2.0)

- **`rlat skill-init <name>` scaffolder** — creates `.claude/skills/<name>/SKILL.md` with a starter template containing a placeholder `!rlat skill-context` block. One-day scope; ship as a v2.0 polish item if time permits.
- **Frontmatter `knowledge:` field as advisory metadata** — purely for tooling that wants to enumerate "which skills use which corpora" (e.g. `rlat refresh --used-by-skills`). Does not drive runtime; deferred until a consumer exists.
- **MCP server (`rlat-mcp`)** — exposes `rlat_skill_context(km, queries)` as a tool. Same trust contracts, same output format, different invocation mechanism. Designed-for in v2.0; will land in v2.1 per the MCP-deferral deviation in REBUILD_PLAN.
- **Hard relevance threshold** — `--min-score N.NN` drops hits below the threshold instead of returning them with weak scores. Hold for now; the per-query confidence header already lets the model self-judge.
- **Per-query token caps** — `--per-query-tokens N` instead of one global cap. Useful for skills with many queries; defer until a real skill needs it.

## Performance budget

Locked floor (BENCHMARK_GATE.md):

| Phase | Time | Notes |
|-------|------|-------|
| Encoder cold | 942 ms | First query in a session |
| Encoder warm | 12 ms | Subsequent queries |
| Search (HNSW) | <50 ms | For corpora ≥ N>5000 passages |
| Search (exact) | <30 ms | For N≤5000 |
| verify_hits | <20 ms | Per-source-file cache hit |
| Format + I/O | ~10 ms | Markdown rendering + stdout |

Total for 2 queries against 50K-passage corpus: ~120 ms warm. Skill `!command` block budget is interactive: <500 ms is target, <100 ms is ideal.

The `harness/skill_context.py` suite does not currently lock these numbers (no perf assertion) — adding `harness/skill_context_perf.py` is on the post-launch follow-up list.

## Cross-references

- User guide: [docs/user/SKILLS.md](../user/SKILLS.md)
- CLI reference: [docs/user/CLI.md](../user/CLI.md) (when written)
- Anthropic skills docs: https://code.claude.com/docs/en/skills.md
- Anthropic hooks docs: https://code.claude.com/docs/en/hooks.md
