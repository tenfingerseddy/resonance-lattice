---
name: deep-research
description: >-
  Iterative multi-hop research pattern for hard questions that span many
  files or require reasoning across evidence (rationale, trade-offs,
  contradictions, cross-workstream synthesis, historical "why" questions,
  long-horizon investigation). Runs a plan → search → refine → search →
  synthesize loop against the `rlat` skill's tools (semantic search +
  knowledge-model composition) and grep, with an explicit hop budget so it
  terminates. Use when a single `rlat search` / `grep` call clearly won't
  surface the answer — typical signals are "summarize across", "what have
  we decided about", "where have we contradicted ourselves on", "explain
  the full rationale", "compare approaches over time", or any question
  whose answer requires synthesising evidence from ≥3 sources. Skip for
  exact-symbol lookups, simple file reads, or one-shot factual queries —
  those are cheaper with a single `rlat search` or grep + Read.
---

# Deep Research

Iterative, budgeted research loop for questions that don't yield to one-shot retrieval. Trades LLM turns for answer quality — **opt-in when the question is hard, not by default**.

Inspired by the recursive language models pattern (MIT CSAIL, 2025): the reader programmatically refines its own queries until it has enough evidence to answer, rather than synthesising from a single fixed retrieval.

---

## When to trigger

Trigger signals (any of):

- Question asks for synthesis across ≥3 sources: *"summarize the tradeoffs across…"*, *"where have we contradicted ourselves on…"*, *"what's the full history of…"*
- Question asks for rationale, not fact: *"why did we pick X over Y?"*, *"explain the design constraints behind Z"*
- First `rlat search` returned coverage `partial` / `edge` / `gap`, or the user rejected the one-shot answer as incomplete
- Question explicitly spans workstreams, domains, or time ranges
- User says "dig into", "really understand", "go deep on", "investigate"

**Don't trigger** for:

- Exact-symbol lookups (`EncoderConfig`, `DEFAULT_MAX_TOKENS`) → grep + Read
- "How do I run X?" / "What's the command for Y?" → one-shot `rlat search` or CLI help
- User already gave a file path and line number
- Trivially short answers ("what version is this?")

---

## Workflow

Copy this checklist and tick items as you go. The budget is non-negotiable — stop when you hit it even if the answer feels incomplete, and tell the user what's missing.

```
Deep Research Progress:
- [ ] Step 1: Plan — draft 3-5 sub-queries covering the question's facets
- [ ] Step 2: First probe — run the broadest query, read coverage assessment
- [ ] Step 3: Refine — for each gap/thin area, run a targeted follow-up
- [ ] Step 4: Cross-check — where sources disagree, run a contradictions pass
- [ ] Step 5: Drill down — grep for exact symbols surfaced by search
- [ ] Step 6: Synthesize — answer with citations, flag what's still uncertain

Budget: max 8 `rlat search` calls + 4 grep calls + 4 Read calls. Stop at limit.
```

### Step 1 — Plan

Before running any tool, write out (to the user, one line each) the 3-5 sub-queries you intend to run. This makes the plan visible and lets the user redirect before you burn turns. Each sub-query should target *one facet* — decomposition is the point, not a bag-of-words reformulation of the original question.

Example. User asks: *"Why did we flip the default store mode from embedded to local, and what did we learn from the pivot?"* Sub-queries:

1. "store mode default history embedded local"
2. "three-layer semantic router pivot rationale"
3. "lossless store architecture decision"
4. "bundled vs embedded trade-offs"
5. "deprecation timeline embedded mode"

### Step 2 — First probe

Run the broadest sub-query first with `--format json --top-k 10`. Read:

- **Coverage label** (`strong` / `partial` / `edge` / `gap`) — tells you if the knowledge model can answer at all.
- **Top source files** — these are your reading targets.
- **Band focus** — topic-dominant → keep widening; entity-dominant → switch to grep.
- **expansion_hint** — if the coverage is thin, this is where to look next.

### Step 3 — Refine

For each gap or thin area, run one targeted follow-up. **Refinement is only refinement if the next query depends on the previous result** — if you could have written all the queries up front, you're just fanning out, not recursing. Good refinements:

- "Source file X surfaced but it mentioned concept Y I don't understand" → search for Y.
- "Two files gave conflicting answers" → `rlat search --with-contradictions` scoped to both.
- "Coverage is `edge` but expansion_hint points at topic Z" → search for Z in a neighbouring knowledge model via `--with`.

### Step 4 — Cross-check

If any two sources disagree, or if the question is about rationale / decisions (high stakes for getting wrong):

```bash
rlat search project.rlat "<topic>" --with-contradictions
# or
rlat contradictions project.rlat "<topic>"
```

Flag every contradiction in the synthesis — don't silently pick a side.

### Step 5 — Drill down

Once search has surfaced specific source files / symbols, switch to grep + Read for exact content. Pattern:

```bash
grep -rn "<exact_symbol_from_search_results>" src/
# then Read the specific lines
```

This is where semantic search hands off to exact search — don't re-run `rlat search` for symbols you already know.

### Step 6 — Synthesize

Write the answer with inline citations (`file:line` or `[source_id]`). Explicitly flag:

- What's well-supported (≥2 sources agree)
- What's single-sourced (say so)
- What's contradicted (show both sides)
- What the knowledge model didn't cover (admit the gap — don't extrapolate)

---

## Budget & stopping conditions

Default budget: **8 search calls + 4 grep + 4 Read**. Stop at the limit even if the answer is incomplete — it's better to hand back a partial synthesis with honest gaps than to spiral.

Hard stops (bail immediately):

- Coverage is `gap` on the first 2 probes → tell the user the knowledge model doesn't cover this, suggest `rlat init-project` or switching to grep-only.
- The answer becomes obvious after 2–3 hops → synthesize and stop; don't pad.
- The loop starts returning the same top results → the knowledge model has given you everything it has.

---

## Cost reality check

Each hop is an LLM call (or MCP tool roundtrip) plus a CPU-bound `rlat search` (~80ms warm). A full 8-hop loop costs ~5-10× a one-shot answer in tokens and latency. Worth it for multi-hop questions, wasteful for simple ones. If the user asked casually ("hey, quick question…"), skip this skill and answer one-shot.

---

## Handoff

- Foundational tool use (search flags, build, MCP setup, output formats) → [rlat skill](../rlat/SKILL.md) and [rlat playbook](../rlat/references/PLAYBOOK.md).
- Question about rlat itself → read rlat SKILL.md first, then this skill only if the rlat question is multi-hop.
- Memory / history recall across sessions → use `rlat memory recall` on the memory knowledge model; the same research loop applies.
