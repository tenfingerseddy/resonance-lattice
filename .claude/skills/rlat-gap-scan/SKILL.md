---
name: rlat-gap-scan
description: >-
  Find the gaps in an rlat knowledge model (.rlat) — questions the corpus
  cannot actually answer — using YOUR own reading of the retrieved context, with
  no external API key and no metered cost. Trigger when the user asks "what is my
  corpus missing?", "where are the gaps?", "what can't this .rlat answer?", "scan
  my knowledge base for holes", "what should I add to the docs?", or wants the
  .rlat to self-improve / notice what it lacks. Also the detection half of the
  closed learning loop: a confirmed, recurring gap is what earns a cloud-authored
  fill. NOT for: answering a question (use the `rlat` search skill); multi-hop
  research (use `deep-research`); gaps judged by retrieval score alone (proven
  blind — see below).
allowed-tools: Bash(rlat:*), Read, Write, Edit, Glob, Grep
---

# rlat-gap-scan — find what the corpus can't answer

A gap is a question the corpus **cannot actually answer** — not one with a low
retrieval score. Those are different things, and the difference is the whole point
of this skill.

**Why retrieval score is the wrong signal (measured, not assumed).** On a real,
well-covered corpus, removing the one doc that answers a question barely changes
the top retrieval score — a topically-adjacent *sibling* doc still matches high. In
the H1 §D measurement (Fabric docs, 131 held-out questions), the gap questions
scored top-1 cosine **0.774** vs **0.813** for non-gaps — a 0.04 gap, with **0 of
54** below any "retrieval failed" floor. A model trained on retrieval features
detected gaps no better than guessing (`review-log.md`, 2026-06-02). **The gap is
invisible to the score** because it lives in *answer quality*, not match strength.

**The signal that works is YOU reading the context.** The same measurement showed
that asking a model — given only the query + retrieved context, no gold answer —
*"is the specific answer actually present here?"* detects the gaps the score
cannot (ROC-AUC ~0.80, matching the gold-based ceiling; even a small model nails
it). You are that model, already in the session. So this runs on **your** judgment
over rlat's retrieved context — **no Anthropic API key, no per-query bill** (the
same reason `deep-research` beats the `rlat deep-search` CLI verb in Claude Code).

The `.rlat` stays self-contained: it provides the context and the memory; you
provide the reading. No new service to set up.

---

## The loop

### 1. Get candidate questions
In priority order, whichever is available:

- **The user gave a list** ("check these questions") — use it.
- **The user's own recent questions** in this session against this corpus — the
  most honest source (real intents, not synthetic).
- **The corpus's own weak zones** — `rlat profile <km>.rlat` and the categories
  the user cares about; generate a handful of specific, answer-seeking questions a
  real user would ask of this corpus (favour *specific* asks — a number, a
  condition, a step — over broad topics; broad topics are almost never gaps).

Find the `.rlat` with `Glob "*.rlat"`; if several, prefer the project-named one or
ask.

### 2. Retrieve the context each question would get
```bash
rlat search <km>.rlat "<question>" --top-k 8 --format context
```
This is exactly what a user querying the corpus would see. Read it as the corpus's
best attempt to answer.

### 3. Judge answerability — the rubric (gold-free, the validated signal)
For each question, decide from the retrieved context **alone** (do not use your own
outside knowledge of the answer — that would hide the gap):

> Does the context contain the **specific** information needed to fully and
> correctly answer the question — the exact fact, number, condition, mechanism, or
> step asked for? Merely covering the general *topic* is **not** enough.

Classify each:
- **Answered** — the specific answer is present in the context. Not a gap.
- **Gap** — the context is on-topic but the specific answer is **absent** (the
  classic case: a sibling doc matches but doesn't contain the asked-for detail).
- **Off-topic** — retrieval returned little relevant; usually a phrasing problem,
  not a corpus gap. Note it but don't treat it as a gap.

Be strict about "specific". The failure mode that makes gap-detection useless is
calling a topically-close-but-incomplete answer "answered".

### 4. Recurrence gate — confirm before it counts
A single unanswered question is noise; a **recurring** one is a real gap worth
filling (the runaway-fill guard). Promote a gap to **confirmed** only when the same
intent shows up more than once — across the user's repeated/rephrased asks, or
across sessions. Keep a running gap log (a small markdown file the user can see,
e.g. `.rlat-gaps.md`, or session memory via the `rlat` memory workflow) recording
each gap, how many times it recurred, and the questions that hit it. One-off gaps
stay **candidate**, not confirmed.

### 5. Report — and optionally author the fill
Report confirmed gaps plainly: the question, what specific thing is missing, and
the closest doc the corpus *did* return (so the user knows where it would belong).

If the user wants to **close** a confirmed gap (the "create" tier, the only step
that grows the corpus): author a grounded fill — research the specific missing fact
from an authoritative source, write it as a short, self-contained, **cited** claim,
and add it to the corpus's earned layer **low-trust** (it earns its place by later
outcomes, it is never written as settled truth). Faithfulness first: every
sentence of the fill must trace to a cited source; if you can't ground it, say so
rather than guessing. (This authoring half is the heavier, irreversible step — do
it only on the user's go, one gap at a time.)

---

## Honest limits (state them, never paper over)

- **Telemetry is blind to the un-asked.** This finds gaps for questions someone
  *phrased*. A need no one queried for is invisible — the corpus can calcify around
  what it nearly covers. Don't claim "no gaps found" means "complete".
- **Your verdict is a probability, not a proof.** Answer-quality judgment is ~80%
  reliable at best; a confirmed gap is a strong candidate, not a certainty. That is
  why fills are born low-trust and earn over time, and why recurrence gates the
  cloud touch.
- **Refinement isn't always a gap.** A user rephrasing may be progress, not a miss.
  Weigh recurrence + your answerability read together, not either alone.

## Why not just check the retrieval score?
Because it doesn't work — measured, not assumed (see the top of this file). A weak
score is neither necessary nor sufficient for a gap on a dense corpus. Reading the
context is what separates "found a topical neighbour" from "found the answer".

## Reference
- `.claude/plans/insight-engine/review-log.md` (2026-06-02) — the measurement
  behind this skill: retrieval-score gap-detection KILLed; gold-free answerability
  validated (AUC ~0.80) and shown to work on a cheap model.
- The `rlat` skill — for the search/retrieve/memory CLI workflows this skill drives.
