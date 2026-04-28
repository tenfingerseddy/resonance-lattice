# Using rlat with Anthropic Skills

Skills are *execution instructions*; rlat is *grounded context*. This guide shows how to wire a knowledge model (`.rlat`) into an Anthropic skill so the skill always sees fresh, query-relevant passages with citations — instead of a frozen reference dump.

## The integration in one line

A skill body uses Anthropic's [dynamic-context primitive](https://code.claude.com/docs/en/skills.md#inject-dynamic-context):

```markdown
!`rlat skill-context fabric-docs.rlat --query "$user_question" --top-k 5`
```

The shell command runs **before the model sees the skill**, and its stdout *replaces the placeholder* in context. The output is markdown with citation anchors, drift status, and confidence metrics — ready for the model to ground on.

That's the entire integration. No new YAML, no hooks, no config. The skill body calls rlat as a normal `!command` block.

## A complete example

`.claude/skills/fabric-helper/SKILL.md`:

````markdown
---
name: fabric-helper
description: Answer Microsoft Fabric questions with grounded citations
arguments:
  - user_query: "what the user is asking about Fabric"
---

# Fabric Helper

You answer Microsoft Fabric questions. The context blocks below are grounded
passages from the official docs. **Cite every claim using the
`[file:offset]` anchors. If no passage covers the question, say so — do not
invent details.**

## Foundational context (preset queries — always loaded)
!`rlat skill-context fabric-docs.rlat \
   --query "Microsoft Fabric workspace fundamentals" \
   --query "lakehouse vs warehouse decision criteria" \
   --top-k 3`

## Query-specific context (the user's actual question)
!`rlat skill-context fabric-docs.rlat --query "$user_query" --top-k 5`
````

When the user invokes this skill, three things happen:

1. **Preset queries always run.** "Workspace fundamentals" and "lakehouse vs warehouse" passages get pulled — the foundational concepts the skill needs to reason well.
2. **The user's query gets dynamically retrieved.** `$user_query` interpolates via Anthropic's skill argument substitution; rlat searches the corpus for that specific question.
3. **Both blocks land in the model's context with citations and drift status.** The model sees passages tagged `[verified]` (current) or `[drifted]` (source has changed since build) and can self-judge whether to make strong claims.

If you add a new feature to the Fabric corpus and run `rlat refresh fabric-docs.rlat`, the very next skill invocation includes that feature — no skill edit required.

## What gets injected

`rlat skill-context` outputs markdown that looks like this:

```markdown
<!-- rlat-mode: augment -->
> **Grounding mode: augment.** Use the passages below as primary context
> for this corpus's domain. Cite them when answering; prefer them over
> your training knowledge when the two conflict.

<!-- rlat skill-context query='lakehouse vs warehouse' band=base mode=augment
     top1_score=0.812 top1_top2_gap=0.124 source_diversity=0.83 drift_fraction=0.00 -->
## Context for: "lakehouse vs warehouse"

> **source: `docs/fabric/lakehouse-overview.md:1247+512`** — score 0.812 `[verified]`
>
> A lakehouse is a unified data store that combines the flexibility of a
> data lake with the performance characteristics of a data warehouse...

> **source: `docs/fabric/warehouse-decision.md:0+847`** — score 0.778 `[verified]`
>
> Choose a warehouse when you need...
```

Four things to notice:

- **The grounding-mode header** is the consumer LLM's directive. `augment` (default) tells the model to treat the passages as primary context for this corpus's domain and prefer them over training when they conflict. See [Grounding modes](#grounding-modes) below.
- **The per-query comment** carries the `ConfidenceMetrics` from the retrieval. The model can read these. Low `source_diversity` → answer cautiously (only one source backs the claim); high `drift_fraction` → flag the answer as potentially stale.
- **Each passage** carries a `[source_file:char_offset+length]` anchor. The skill can instruct the model to cite using exactly this format, and any consumer can quote-verify by reading the file at that offset.
- **`[verified]`** means the source file's content_hash still matches what was hashed at build time. Other values are `[drifted]` (source moved) and `[missing]` (source file gone).

## Trust features

The integration enforces three trust contracts as the *output format itself*, not as opt-in flags:

### Citations are always on

Every passage carries its source anchor. There is no `--no-citations` flag — a skill that uses `rlat skill-context` is structurally incapable of grounding on uncited content.

### Drift detection is always on

If any passage in the result has stale or missing source, the output begins with a warning banner:

```markdown
> ⚠ **DRIFT WARNING**: at least one passage below has stale or missing source.
> Treat content as advisory; refresh with `rlat refresh fabric-docs.rlat`
> for canonical results.
```

The model sees this banner. Skill instructions can tell it how to react (e.g. "if you see a DRIFT WARNING, prepend your answer with 'Note: this answer may not reflect recent corpus changes.'").

### Strict mode for hard requirements

Pass `--strict` and any drifted/missing source aborts non-zero:

```bash
!`rlat skill-context fabric-docs.rlat --strict --query "$user_query"`
```

The skill load fails. Use this when stale grounding is unacceptable — e.g., the skill answers questions where wrong-but-confident is worse than no-answer.

## Grounding modes

`--mode` controls *how the consumer LLM should treat the retrieved passages*. It's a directive stamped at the top of the markdown output, paired with a confidence-gate that suppresses the dynamic body on weak retrieval. Three modes:

### `augment` (default) — passages are primary context

```bash
!`rlat skill-context fabric-docs.rlat --query "$user_query"`     # mode=augment is the default
```

The header tells the LLM: *"Use these passages as primary context. Cite them when answering. Prefer them over your training knowledge when the two conflict."* The gate fires when `top1_score < 0.30` (retrieval has genuinely failed for this query) or `drift_fraction > 0.30` — on weak retrieval the body is replaced with a *no confident evidence* marker so the LLM falls back to training instead of grounding on noise.

**Default mode.** Bench 2 v4 (Microsoft Fabric, single-shot, Sonnet 4.6 partially-known): **76.5% answerable accuracy / 3.9% hallucination, vs 56.9% / 19.6% for the LLM alone** — augment adds 19.6 pp of correct answers while cutting hallucination by 5×. The right shape for broad domain corpora where the LLM already has solid prior knowledge.

### `constrain` — passages are the only source of truth

```bash
!`rlat skill-context compliance-rules.rlat --mode constrain --query "$user_query"`
```

The header tells the LLM: *"Answer ONLY from these passages. If they do not cover the question, refuse explicitly — do not draw on training knowledge."* No gate — the body always ships, and refusal on thin evidence is the LLM's job.

Bench 2 v4 (single-shot): 66.7% accuracy / **2.0% hallucination** / **91.7% distractor refusal** — trades 10 pp answerable accuracy for halving the hallucination rate vs default and the highest distractor-refusal in the suite. Pair with `--strict` so drifted source aborts the skill load entirely.

```bash
!`rlat skill-context compliance-rules.rlat --strict --mode constrain --query "$user_query"`
```

The right mode for compliance, regulatory, audit, or any context where *wrong-but-confident is worse than no answer*.

### `knowledge` — passages supplement training

```bash
!`rlat skill-context fabric-docs.rlat --mode knowledge --query "$user_query"`
```

The header tells the LLM: *"These passages supplement your existing knowledge. Ground claims about this corpus's domain in them; you may draw on general knowledge for surrounding context."* Lighter gate — only suppresses on very weak retrieval (`top1_score < 0.15`); drift is allowed.

Use when the corpus is *partial coverage* of a domain the LLM already knows reasonably well — e.g., a project's bespoke conventions layered on top of general programming knowledge, where you want the LLM to use both.

### Picking a mode

| Mode | When to use | Gate fires on |
|------|-------------|---------------|
| `augment` | Broad domain corpus the LLM partially knows (default) | top1_score < 0.30 OR drift_fraction > 0.30 |
| `constrain` | Fact extraction, compliance, regulatory, audit | never (LLM does the refusal) |
| `knowledge` | Partial-coverage corpus on top of LLM-known domain | top1_score < 0.15 only |

## Token budget

`rlat skill-context` caps the per-query block portion of the output via `--token-budget` (default `4000`). When multiple `--query` flags add up to more than the budget, **later query blocks are dropped first** — so the skill-author's preset queries always survive, and the user's dynamic query gets truncated last. The grounding-mode directive header and the drift banner ship outside the budget; the directive is a small, fixed-size instruction the consumer LLM must always see.

```bash
!`rlat skill-context fabric-docs.rlat \
   --query "preset 1" --query "preset 2" --query "$user_query" \
   --top-k 5 --token-budget 2000`
```

If the budget overflows, the user-query block goes first, then preset 2, then preset 1. The first preset is always preserved (even if it alone overflows the budget — that's the skill author's bug to fix, not ours to silently drop).

## Patterns

### Preset + user (most common)

Two `!command` blocks, or one with multiple `--query` flags:

```markdown
## Foundations
!`rlat skill-context fabric-docs.rlat --query "core concepts" --top-k 3`

## Query
!`rlat skill-context fabric-docs.rlat --query "$user_question"`
```

### Decomposed user query

If the skill body extracts sub-questions from the user's input, each can hit the corpus separately:

```markdown
!`rlat skill-context fabric-docs.rlat \
   --query "$entity" --query "$action" --query "$context" --top-k 3`
```

### Strict + cross-corpus

A skill that pulls from two `.rlat` files runs two `!command` blocks, both `--strict`:

```markdown
## Fabric context
!`rlat skill-context fabric-docs.rlat --strict --query "$user_query"`

## Power BI context
!`rlat skill-context powerbi-developer.rlat --strict --query "$user_query"`
```

If either is drifted, the skill load fails — refresh both before grounding.

### Name-aliasing protection

`--strict-names` adds a second safety check: distinctive proper nouns / acronyms / alphanumeric IDs from the query are presence-checked against the rendered passages. Default behaviour prepends a refusal directive when a token is missing; `--strict-names` aborts non-zero (rc=3). Pair with `--strict` for compliance work where wrong-named entities must never silently slide through:

```markdown
!`rlat skill-context fabric-docs.rlat --strict --strict-names --query "$user_query"`
```

Catches the failure mode where the user's question mentions a fake or adjacent product (e.g. `MaterializedViewExpress (MVE)`) but the corpus only describes a similarly-named real entity (`MaterializedLakeView (MLV)`). Score-based gating cannot tell these apart on a paraphrase-rich corpus — the name-check can.

### Multi-hop research

For hard questions that need cross-file synthesis ("why did we pick X over Y?", "summarise the trade-offs", "what's the full history of …"), `rlat` ships **two surfaces** that run the same plan → retrieve → refine → synthesize loop:

- **`deep-research` skill** (`.claude/skills/deep-research/SKILL.md`) — drives the loop natively in your Claude Code session. **No API key needed**; your Claude subscription covers the LLM hops. This is the right pick for nearly every Claude Code user.
- **`rlat deep-search` CLI verb** — same loop, exposed as a CLI command for non-Claude-Code agents, CI pipelines, and batch jobs. **Requires an Anthropic API key** ($0.009-0.025/q). Bench-validated headline: 92.2% accuracy / 0% hallucination on the Microsoft Fabric corpus; same numbers apply to the skill version because it's the same loop.

If you're in a Claude Code session: just ask the research question; the `deep-research` skill triggers automatically on cross-file synthesis questions. If you're authoring a skill that runs in a non-Claude-Code agent or you need a programmatic surface, use the CLI:

```markdown
## Hard question (CLI surface — API key required)
!`rlat deep-search fabric-docs.rlat "$user_question" --format markdown`
```

See [docs/user/API_KEYS.md](API_KEYS.md) for when each surface is the right pick.

### The two-skill structure

`rlat` ships **two skills** in `.claude/skills/`, each handling a discrete capability:

| Skill | What it does | Triggers on |
|---|---|---|
| **`rlat`** | Executes any rlat CLI workflow on the user's behalf — setup, build, refresh, search, memory recall, compare, convert, optimise, skill-context generation, programmatic deep-search | Project-setup signals, build/refresh signals, single-hop fact questions, memory recall, cross-corpus comparison, storage-mode conversion, opt-in optimise |
| **`deep-research`** | Drives the multi-hop research loop (plan → retrieve → refine → synthesize) interactively in this conversation | Cross-file synthesis questions, rationale ("why X over Y"), trade-offs, contradictions across sources, "summarise the X across the codebase", historical "why" questions, follow-ups after a thin single-shot search |

The two skills don't overlap: `rlat` orchestrates **CLI workflows**; `deep-research` orchestrates **the multi-hop research loop in conversation**. A single-hop fact question goes to `rlat search`; a multi-file synthesis question goes to `deep-research`. The CLI verb `rlat deep-search` (programmatic surface for the same loop, requires API key) is documented inside the `rlat` skill — not a separate skill — because it's just one more workflow in the rlat surface.

This 2-skill design follows Anthropic's published guidance on composite workflow skills: one skill per discrete capability, positive trigger specificity, decision-gate-driven bodies that orchestrate sub-workflows internally rather than fragmenting into many narrow-trigger skills. Sources: [code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills), [platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).

## Performance

`!command` blocks run synchronously every time the skill loads. The locked numbers from `BENCHMARK_GATE.md`:

- Warm encoder: **12 ms** per query
- Search (FAISS HNSW M=32 efC=200 efS=128): **<50 ms** for corpora up to ~500K passages
- Format + I/O: **~10 ms**

Two queries against a 50K-passage corpus = ~120 ms total. Well inside the budget for an interactive skill load.

If you see slow loads, the most likely culprit is encoder cold-start (~942 ms vs 12 ms warm). The encoder warms on first use within a session and stays warm; the second-and-onwards skill invocations are fast.

## Frequently asked

**Do I need to add `knowledge:` frontmatter?** No. The integration uses Anthropic's existing `!command` primitive — frontmatter additions are silently ignored by Claude Code (verified safe), but they don't drive any rlat behavior. The `!command` blocks are the source of truth.

**What if the corpus doesn't cover the question?** rlat returns the closest passages anyway. The model sees them with their (low) scores in the header line, and skill instructions should tell it to say "I don't know" when scores are weak. Future versions may add a hard relevance threshold.

**Does this work outside Claude Code?** Yes. The Claude Agent SDK supports skills the same way. An MCP wrapper would expose `rlat skill-context` as a tool, and the skill body would call it via tool-use — same retrieval, same trust contracts, different invocation mechanism.

**What about static reference files?** A skill can mix `!command` blocks for dynamic context with relative-link references for static guides. Use rlat for anything that benefits from query-relevant retrieval; use static files for stable how-to material.

## See also

- [STORAGE_MODES.md](STORAGE_MODES.md) — choosing bundled / local / remote for your `.rlat`
- [CLI.md](CLI.md) — full `rlat skill-context` reference
- [docs/internal/SKILL_INTEGRATION.md](../internal/SKILL_INTEGRATION.md) — spec, format contracts, design rationale
