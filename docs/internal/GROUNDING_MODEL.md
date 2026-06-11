# Grounding Model — what rlat promises, and how knowledge earns trust

Status: current truth as of 2026-05-16. Outcome of a first-principles design
review. Resolves two questions the prior docs were self-contradictory on:
is rlat retrieval or answer-improvement, and how should an answer earn its
way into the durable insight layer.

Supersedes the "no reader" thesis and the accept/reject verdict-gated
promotion design (see Documentation debt, below).

## Confirmed goals

The fixed points the model is built on.

**A — Purpose & positioning**
- A1. rlat is a semantic layer over the user's own corpus — a controlled,
  owned body of knowledge.
- A2. The corpus is user-gated, not closed. Anything may feed it (including
  the internet); only user approval admits source content.
- A3. rlat's honesty scope is "what your corpus says," not "what is true in
  the world." It represents the corpus faithfully; it is not an oracle.

**B — User experience**
- B4. Maximum value for minimum user effort — defaults over knobs; the user
  is never required to do labour.
- B5. Interrupt the user only when their input changes behaviour.
- B6. Self-explaining — every answer exposes its grounding and provenance.

**C — Trust**
- C7. Trust is earned and visible, never demanded.
- C8. Honest about limits — a gap is surfaced as a gap, never hidden.
- C9. Every claim is traceable to a specific source.

**D — Engine**
- D10. Compounding — corpus + agent get measurably better with use.
- D11. The improvement loop must not depend on user labour.

## What rlat is

rlat is an answer-improvement layer. Its product is a faithfully-grounded,
outcome-validated answer. "Retrieval" is the mechanism; the grounded answer
is the product. The v2.0 "no reader — return passages" thesis described the
retrieval library rlat used to be; it does not describe the agent harness
rlat became, and it is retired.

rlat promises two things on two timescales:

- **Faithfulness** — immediate, machine-checkable. Every claim in an answer
  traces to a corpus source; citations are on-topic for the question; gaps
  are stated as gaps.
- **Truth** — accumulating, outcome-driven. The corpus itself is dragged
  toward truth over time by results.

rlat owns faithfulness directly. It owns truth indirectly — by evolving the
corpus. It never claims to *know* truth; it maintains a calibrated, visible
confidence.

## The confidence lifecycle

Replaces `candidate → /accept → promote`.

1. `deep-search` produces an answer.
2. **Faithfulness gate** (machine judge): is every claim grounded in a cited
   source, and are the citations on-topic for the question? Faithful → the
   answer enters as a **provisional insight** with a modest starting
   confidence. Not faithful → not admitted (ungrounded synthesis).
3. The provisional insight is retrievable immediately — it is grounded, so
   it is safe to serve — and carries a **visible provisional confidence**.
4. **Outcomes update confidence.** As skill-mediated agent tasks use the
   insight, task outcomes — errors encountered, task success, whether it
   could have been done better — corroborate or falsify it. Confidence is
   multi-signal: outcomes weighted heaviest, then cross-source corroboration,
   then source drift / contradiction, then post-hoc correction.
5. Corroborated insights harden; falsified ones retire. The corpus evolves.

Truth is never determined; it is estimated as an accumulating confidence,
always provisional.

## Confidence & attribution

How step 4 — "outcomes update confidence" — actually works. Outcome of the
2026-05-16 design review.

**Outcome signal.** Two observables per agent session:

- *intent resolution* — an intent marked satisfied (good) or abandoned
  (bad).
- *retry count* — anything re-attempted means the knowledge wasn't good
  enough to get it right the first time. A first-try success is clean;
  friction is graded by how much had to be redone.

**Raw outcome log.** Per session, record the raw facts: insights retrieved
(with retrieval rank), the intent outcome, the retry count, and the
agent-reported load-bearing set. Attribution is computed *from* this log,
never baked into it.

**Attribution is a pluggable reducer** — a pure function from the outcome
log to per-insight corroboration / falsification weight. Three reducers,
all built, compared empirically:

- *diffuse, rank-weighted* — every insight a session retrieved shares the
  session's outcome, weighted by retrieval rank and **signed** by the
  outcome (clean success → +, abandoned / retry-heavy → −). The sign is
  what stops it being a popularity count. Unbiased-but-noisy; disentangles
  over volume.
- *agent-reported* — at session end (Stop hook) one cheap LLM call asks
  the agent which retrieved insights its solution actually depended on;
  credit only those. Recognition over the session's retrieval list, not
  free recall. Zero user impact — it asks the agent, not the user.

No reducer is hard-picked. We measure which one's confidence numbers best
predict held-out outcomes — high confidence → clean success, low → retries
— and let measurable results pick the winner.

**Confidence = Beta accumulation.** Each insight carries two tallies:
corroboration weight and falsification weight. The faithfulness score
seeds the prior pseudocounts (a faithful insight starts modestly
positive). `confidence = corroboration / (corroboration + falsification)`.
Bounded 0..1, naturally slow — a well-observed insight barely moves on one
noisy outcome — and calibrated. A reducer just adds weight to one side.

**What confidence does.** It multiplies the insight's retrieval score, so
corroborated insights float above provisional ones at similar relevance.
All insights still surface, labelled with visible confidence. Below a
retire floor, a falsified insight drops out of retrieval (→ `retired`).

**Residual risk.** Attribution stays imperfect — every reducer is noisy.
Robustness comes not from precise attribution but from three things: small
per-outcome updates, Beta's slowness, and a multi-session requirement
before an insight hardens or retires. We do not claim clean per-insight
causal attribution; we claim the estimate converges over volume.

## The user's role

One act: **approving raw source content into the corpus** — the curation
act, where trust originates (A2). Everything downstream — the faithfulness
gate, provisional promotion, confidence accrual — is autonomous (D11).

`/accept` and `/reject` are no longer the insight-promotion gate. `/accept`
on an *intent* survives: an intent marked satisfied is a task outcome, and
feeds confidence. Post-hoc correction stays available, never required.

## The three planes

- `deep-research` → better answers (knowledge retrieval + synthesis).
- memory → better actions (what the agent learned to do).
- skills → the workflow: the application of knowledge to produce an outcome.
  Skill-mediated tasks are where knowledge becomes action becomes the
  measurable result.

## What this means for Claim 1

Claim 1 ("repeat-query latency and quality strictly improve") splits by what
each setting can measure:

- **Speed + faithfulness compounding** — measurable in a controlled run
  (`deep-search` over a fixed corpus). Fully autonomous: speed is wall-clock,
  faithfulness is machine-judged. No user verdicts.
- **Truth compounding** — a separate claim, measurable only where real task
  outcomes exist: skill-mediated agent tasks, via the harness outcome ledger
  in normal operation.

## What this reuses

The redesign rewires; it does not rebuild. Reused as-is:

- `InsightPassage.confidence` — now the derived Beta-mean truth estimate
  over the corroboration / falsification tallies.
- The compression test — folds into / alongside the faithfulness gate.
- The drift cascade — one falsification signal.
- The memory outcome ledger + intent graph — the task-outcome capture.
- `rlat probe` — weak-zone detection becomes low-confidence / gap detection.

What changes: what *feeds* confidence (outcomes, not user verdicts) and what
*gates* promotion (faithfulness, not `/accept`).

## Open problems

- **Attribution under confounding** — *designed* (see Confidence &
  attribution, above). Residual risk acknowledged there: attribution stays
  noisy; robustness rests on slow Beta updates + a multi-session bar, not
  on precise causal attribution.
- **Faithfulness-judge calibration.** The faithfulness gate is itself an LLM.
  Defensible — it is a verifiable matching task (does claim X trace to
  citation Y), not a vibe, and is spot-checkable — but it needs a calibration
  pass before its scores gate anything.

## Documentation debt

These assert the retired "no reader" thesis or the retired verdict-gated
promotion, and must be reconciled against this document:

- `CLAUDE.md` — "Current Thesis" point 3; "Conventions" no-reader line.
  (Reconciled 2026-05-16.)
- `.claude/plans/lensed-knowledge-manifesto.md` — no-reader positioning.
- `.claude/plans/lensed-knowledge-architecture.md` — §1 (no reader layer),
  §4.4 + §9.3 + §10 (verdict-gated promotion).
- `benchmarks/bench_lensed_dogfood.py` — accept/reject as the quality axis.
