# The context block is a curriculum: from retrieval to reasoning support

**Status:** design + machine-checked component receipts; the decisive experiment is
pre-registered, not run (it needs the production pipeline) · **Date:** 2026-08-10 ·
**Receipts:** [`demos/demo9_evidence_curriculum.py`](demos/demo9_evidence_curriculum.py)
(all checks pass) · **Builds on:** the quotient band
([`ORBIT_RETRIEVAL.md`](ORBIT_RETRIEVAL.md)) and the session's theory report
([`THE_NEXT_BREAKTHROUGH.md`](THE_NEXT_BREAKTHROUGH.md)).

## 1. The reframe, taken literally

Owner's framing: rlat is not just retrieval — it is a **knowledge source**, and
should facilitate better **reasoning** over the content.

The session's theory says exactly where that leverage lives. A consumer LLM's
in-context processing behaves like regression over the items in its window
(test-time regression; demo 1 verifies the identities). Therefore:

> **The context block rlat serves is the training set of the consumer model's
> in-context learner.** Retrieval selects the k passages; reasoning quality is
> governed by what is served *around and between* them. A knowledge source that
> wants better reasoning designs that training set — a curriculum, not a bag of
> hits.

And what makes a training set good for an interference-prone online learner is
known mathematics (demo 1d measured the interference): deduplication, explicit
keys, labelled conflicts, and calibrated coverage — every one of which rlat can
compute **without an LLM in the loop**, from geometry, hashes, and the build-time
self-audit it already runs. The consumer LLM currently re-derives all of this per
query, at inference cost, with hallucination risk concentrated exactly where the
relations between passages are implicit.

This is also the natural completion of the product's own arc: the claims layer
already serves unary knowledge (facts, constraints, falsified findings — each
proven to change answers, with receipts). The curriculum extends the same
philosophy to **binary** knowledge (what served passages are to each other) and
**epistemic** knowledge (whether the corpus can support an answer at all).

## 2. The four served surfaces (all LLM-free, all with receipts)

Computed at query time over the k served passages only — k² pairs, no stored
graph, honouring the verdict that killed edge artefacts:

**S1 — Pair verdicts.** Every pair of served passages gets two numbers: `s_raw`
(the retrieval band) and `s_quot` (the quotient band of ORBIT_RETRIEVAL.md,
register nuisance removed). Measured on real text (demo 9a): the quotient
separates *same-topic-across-register* pairs from unrelated ones at **AUC 0.828 vs
0.681 for raw** — so the assembler can label `near-duplicate of`, `continues`,
`same topic in another register` from geometry alone, each label carrying its two
numbers as receipt. (A first hypothesis — mean similarity *deltas* would single
out cross-register pairs — was wrong and stays printed in the demo.) Build-time
self-audit contributes the strongest labels where it has them: `contradicts`
(already computed and stored), `supersedes` (hash lineage across refresh deltas).

**S2 — A calibrated coverage verdict.** "Can this corpus support an answer here?"
is a geometric quantity. Protocol (demo 9b, with a correction that matters:
held-out *files* are only valid "uncovered" labels if their content is not
duplicated elsewhere — redundancy-filtered hold-out): retrieval geometry alone
separates covered from uncovered queries at **AUC 0.849** (margin scorer) at LSA
quality. Served as a per-query epistemic header, this turns `--mode constrain`'s
refusal from a directive into a **calibrated, auditable decision** — the number
and its calibration curve are the receipt. It also makes absence first-class: the
map declares its blank spots, which is the strongest anti-hallucination object a
knowledge source can serve.

**S3 — Join keys.** The hooks a reasoner needs to compose facts across passages —
shared rare terms with exact offsets — served instead of rediscovered (demo 9c:
125/200 same-topic pairs carry at least one; pure text statistics).

**S4 — Curriculum order and dedup.** Near-duplicates collapsed (crosstalk, demo
1d), remainder ordered dependency-first (definitions before uses, via join-key
direction and file structure) rather than by raw score. This surface is design
only — no local receipt exists without a consumer model; it rides on the A/B
below.

Demo 9d prints the artefact end to end: same six passages a flat block would
serve, plus a coverage header, three labelled relations, and join keys — every
line carrying its numbers and coordinates.

## 3. The decisive experiment (pre-registered; the only one that counts)

The claim "facilitates better reasoning" is a claim about the *consumer*, so the
test isolates the consumer: **same retrieved passages, two assemblies**.

- **Data**: the 63-question Fabric hallucination bench, existing harness and
  rubric, plus the cross-domain constraint sets.
- **Arms**: (A) current `--format context`; (B) curriculum block — identical
  passage set, plus S1–S4. Optionally (C) = B with coverage-gated refusal in
  `constrain` mode.
- **Metrics**: answerable accuracy, answerable-hallucination rate, refusal
  correctness on unanswerables (S2's calibration measured directly), and token
  cost of the consumer's reasoning (structure should *reduce* re-derivation
  tokens).
- **Kill criteria, stated now**: if B does not beat A on hallucination rate or
  accuracy by a pre-agreed margin at ≤10% token overhead, the curriculum joins
  the falsification ledger with its numbers. If only S2 (coverage gating) moves
  the needle, ship S2 alone — it is independently the cheapest surface.
- **Secondary calibration**: S2's scorer against answerability labels on the same
  63 questions (AUC, reliability curve) — the production version of demo 9b.

## 4. Fit to the product (why this is core, not garnish)

- It upgrades the product's **central output contract** — the context block — not
  a side feature. The assembler (`src/resonance_lattice/assembler/`) is the
  natural home; everything upstream (bands, registry, audit, claims) already
  produces the inputs.
- **No LLM in the loop, no API key**, consistent with the retrieval contract;
  every served structure line is receipt-bearing and hence consistent with the
  trust surface (`augment`/`knowledge`/`constrain` extend naturally: the coverage
  header is the quantitative backbone `constrain` has been missing).
- **No stored graph.** k² pair computations at query time over served passages;
  the quotient band is the only persistent object, and it is a band.
- Cost: at k = 10, 45 pair scores (two dot products each), one coverage score,
  token-set intersections — microseconds next to encoding the query.

## 5. Honest limits

- Component receipts here are LSA-scale; S1/S2 numbers must be re-measured on a
  production band (the demo's `.rlat` loader exists for exactly this).
- S4 (ordering) has no local validation — it is theory-motivated (curriculum for
  an interference-prone learner) and rides entirely on the A/B.
- Whether consumer LLMs *use* served structure is an empirical question about
  consumers, not about rlat; the A/B answers it per model class, and the answer
  may differ between strong and weak consumers (worth stratifying).
- The coverage scorer's threshold is corpus-dependent; it ships as a calibrated
  number with its curve, never as a bare boolean.
- Contradiction and supersession labels inherit the self-audit's precision —
  already receipted machinery, but their serving frequency at k = 10 is unknown
  until measured on production corpora.
