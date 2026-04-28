---
name: deep-research
description: >-
  Interactive multi-hop research over an rlat knowledge model, run by Claude
  in this conversation against `rlat search`. Plan → retrieve → refine →
  retrieve → synthesize loop, driven by the assistant turn-by-turn so the
  user can redirect at any hop. Bench-validated: 92.2% answerable accuracy
  at 0% hallucination on the Microsoft Fabric documentation (63 questions,
  Sonnet 4.6, relaxed rubric) — same loop the `rlat deep-search` CLI
  implements, exposed here as a free Claude Code workflow that uses the
  user's existing subscription instead of the Anthropic API. Trigger for
  cross-codebase synthesis questions whose answer requires evidence from
  multiple files: rationale ("why was X chosen over Y"), trade-offs,
  contradictions across sources, historical "why" questions, long-horizon
  investigation, "summarise the X across the codebase", "what's the full
  story of Y". Trigger for memory-recall research across past sessions
  (drive the same loop on a memory knowledge model). Trigger when a single
  `rlat search` returned thin or partial coverage and the user asked a
  follow-up. Single-shot factual lookups and exact-symbol grep are handled
  by the `rlat` skill instead.
---

# Deep Research (in-session, no API key)

This skill runs the same multi-hop research loop the `rlat deep-search` CLI
implements, but driven by **Claude natively in this session** rather than
calling Anthropic's API as a separate tool. The user pays for nothing on top
of their existing Claude subscription — every "LLM hop" in the loop is just
this conversation.

For users who want a programmatic / batch / agent surface and have an
Anthropic API key, the equivalent CLI verb is `rlat deep-search km.rlat
"<question>"` (see [docs/user/CLI.md](../../../docs/user/CLI.md#rlat-deep-search)).
The skill below and the CLI verb run the same loop with the same prompt
shape and the same termination semantics; the bench-validated numbers
apply to both surfaces.

## When to trigger

Trigger for any of:

- Question requires synthesising evidence from ≥3 sources: *"summarize the
  trade-offs across…"*, *"what's the full history of…"*, *"where have we
  contradicted ourselves on…"*
- Question asks for **rationale, not fact**: *"why did we pick X over Y?"*,
  *"explain the design constraints behind Z"*
- First `rlat search` returned partial or low-confidence results, or the
  user rejected the one-shot answer as incomplete
- Question explicitly spans workstreams, domains, or time ranges
- User says "dig into", "really understand", "research", "go deep on",
  "investigate", "what's the story behind"

**Don't trigger** for:

- Exact-symbol lookups (`EncoderConfig`, `DEFAULT_MAX_TOKENS`) → grep + Read
- One-shot lookups ("how do I run X?", "what's the command for Y?") → a
  single `rlat search` is cheaper and as accurate
- Trivially short answers
- Questions where the user already gave you a file path

## The loop

A four-hop budget. Each hop is one of: plan, search, decide, synth.

### Hop 1 — Plan

You are a research planner for fact extraction from a documentation corpus.
Given the user's question, output a SHORT initial search query (6-15 words)
that's likely to surface a relevant passage. Output ONLY the query — no
preamble, no explanation. Then run that query as `rlat search` (single-shot,
top-k 5).

```bash
rlat search km.rlat "<your planned query>" --top-k 5 --format context --mode augment
```

### Hop 2-4 — Refine

After each `rlat search`, you've collected one more block of evidence. Read
the original question + everything retrieved so far. Decide your next
action — output exactly one of these JSON objects on a single line, nothing
else:

```json
{"action": "answer", "answer": "<final answer with citations>"}
{"action": "search", "query": "<next short query>"}
{"action": "give_up"}
```

- Pick `answer` once you have enough evidence to answer with confidence —
  cite source files inline (`docs/foo.md`, `src/bar.py`). **Stop searching
  the moment you have a clear answer; don't burn hops on confirmation.**
- Pick `search` if you need more evidence and you have a specific next
  query in mind. The query should target a *facet* of the question that
  the previous hops haven't covered — refinement, not bag-of-words
  reformulation.
- Pick `give_up` if the corpus clearly doesn't have the answer. Output a
  brief refusal message that says "the corpus covers X but not Y; the
  question may be about a different entity." This is the correct response
  for distractor questions about things that don't exist in the corpus.

If hops 2, 3, and 4 all returned `search`, hop 5 is the synth hop —
write the answer from the union of evidence collected. If even the synth
hop's evidence doesn't cover the question, refuse explicitly.

### Name verification (always last)

Before delivering the synthesised answer, run a final check:

> **Did any distinctive proper noun, acronym, or alphanumeric ID from the
> question appear verbatim in any retrieved passage?**

If a token from the question (e.g. `MVE`, `F4096`, a quoted multi-word
product name) is missing from every passage, that's the **name-aliasing
trap**: the encoder may have surfaced an adjacent entity (`MLV`, `F32`)
and your answer is at risk of being about a different thing. In that
case, prepend a refusal directive to your answer:

> ⚠ **Name verification failed.** The question references `<token>`, but
> no retrieved passage contains this exact name. The corpus may describe
> an adjacent or differently-named entity. Either refuse explicitly or
> ask the user to confirm the name is correct.

The CLI verb (`rlat deep-search --strict-names`) does this check
mechanically; the skill version asks you to do it once before answering.

## Output shape

Match the shape `rlat deep-search` returns so the user gets a consistent
artefact regardless of whether they used the skill or the CLI:

```
<the synthesised answer with inline file:line citations>

[deep-research]
  hop 1 plan          '<initial query>'
  hop 2 search        '<query>' → N passages
  hop 3 decide_search '<refined query>'  (or decide_answer / decide_give_up)
  hop 4 search        '<query>' → N passages
  hop N answer (loop terminated)  (or synth_after_max_hops)

evidence union: <count> distinct passages across <count> source files
```

## Bench numbers

The same loop, run via the `rlat deep-search` CLI verb against the
Anthropic API, on the Microsoft Fabric documentation (63 questions,
Sonnet 4.6, relaxed rubric):

| Approach | Accuracy | Hallucination | Distractor refusal |
|---|---:|---:|---:|
| **Deep-research loop** (this skill, or `rlat deep-search`) | **92.2%** | **0.0%** (`--mode knowledge` variant) | 83.3% |
| Single-shot `rlat search` | 76.5% | 3.9% | 75.0% |
| LLM-only (no retrieval) | 56.9% | 19.6% | 50.0% |

These were measured on the API surface; the skill drives the same loop
through the user's Claude session, with the same retrieval primitive, the
same prompts, and the same hop semantics. Numbers should transfer with
small variance from differences in Sonnet version (Claude Code ships
Sonnet 4.5 or 4.6 depending on the user's setup) and tool-use mechanics.
Full methodology and the 11-lane matrix:
[docs/user/BENCHMARKS.md](../../../docs/user/BENCHMARKS.md).

## Stopping conditions

- Pick `answer` the moment you have evidence to answer. Padding with extra
  hops costs the user latency for no gain.
- Pick `give_up` if the corpus is genuinely thin — better to refuse
  explicitly than to invent.
- Hard stop at hop 5 (synth-after-max-hops). If the evidence still doesn't
  cover the question, your synthesised answer should say so explicitly.
- If retrieval keeps returning the same top results across hops, the
  knowledge model has given you everything it has — synthesise and stop.

## Cost reality

Each hop is one assistant turn in this session, plus one `rlat search`
subprocess (~80 ms warm). A full 4-hop loop is 4 turns. The user pays
for nothing extra — this is just the assistant doing its job inside the
existing conversation. Compare to the API-key surface
(`rlat deep-search`) at ~$0.009-0.025 per question depending on hop
count.

## Handoff

- Need to run this from a non-Claude-Code agent / CI / batch script?
  Use the CLI verb (requires Anthropic API key — see
  [docs/user/API_KEYS.md](../../../docs/user/API_KEYS.md)).
- Foundational `rlat` tool use (search flags, build, output formats) →
  [rlat skill](../rlat/SKILL.md).
- Memory recall across sessions → `rlat memory recall` on the memory
  knowledge model; the same loop applies.
