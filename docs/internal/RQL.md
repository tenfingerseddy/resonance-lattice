# RQL — Resonance Query Language (curated v2.0 reference)

The Python API surface for power users + integrators who want more than `rlat
search`. CLI subcommands wrap a subset of these ops; the rest are
library-callable from any consumer.

> Source-of-truth code: `src/resonance_lattice/rql/`.
> Public surface: 14 ops, 5 groups. Every op returns a typed result —
> never a raw ndarray or untyped tuple.

## Why these ops, and not others

The single architectural advantage Resonance Lattice has end-to-end is
**lossless verified retrieval**: every passage carries `(source_file,
char_offset, char_length, content_hash, drift_status)`. Most retrievers
store opaque embeddings and lose source. We don't.

That fact shapes RQL. Every op:

- Returns Citations (full provenance), not opaque passage IDs.
- Respects the cross-model rule when crossing knowledge models (base
  band only — optimised bands are corpus-specific projections, not
  comparable across KMs).
- Treats `drift_status` as a first-class data dimension where relevant
  (`evidence`, `drift`, `audit`).

Ops that DIDN'T make the cut were either: thin numpy wrappers (most of
v0.11's algebra ops), research artifacts (dynamics, geometric, EML
retrieval-time scoring — falsified), or intent-only (boost, suppress,
cascade — defer to v2.1 alongside Intent operators).

## The 14 curated ops

```
Foundation (4):       locate, profile, neighbors, near_duplicates
Comparison (3):       compare, unique, intersect
Composition (2):      compose, merge
Evidence + drift (3): evidence, drift, corpus_diff
Experimental (2):     contradictions, audit
```

---

## Foundation — within one corpus

### `locate(contents, passage_idx) -> Citation`
**Purpose**: Resolve a passage_idx to its source coordinates + build-time
content_hash. The single op that turns RAG black-box outputs into auditable
citations.

**Why kept**: foundational. Every other op that returns a passage reference
goes through `Citation.from_coord` ultimately; `locate` is the single-
passage public form of that lift.

**Why no `cite`**: would be a synonym. Folded.

### `profile(contents, *, store=None) -> Profile`
**Purpose**: typed corpus snapshot — source distribution, drift counts,
band info, encoder identity. The Python-API counterpart to `rlat profile`.

**Why kept**: structured access for dashboards, scripts, quality gates.
`drift_counts` is opt-in via `store=` because verifying every passage
requires reading every source file once.

### `neighbors(contents, passage_idx, top_k=10, prefer=None) -> list[NeighborHit]`
**Purpose**: k-nearest neighbours of a given passage by cosine, excluding
self. "Show me passages related to this one I'm reading."

**Why kept**: foundational graph navigation primitive. Powers
"related-content" UI patterns without requiring a separate query encode.

### `near_duplicates(contents, *, threshold=0.95, prefer=None) -> list[DuplicateCluster]`
**Purpose**: within-corpus near-duplicate clustering for ingest hygiene.

**Why kept**: every real corpus has duplicates from include-paths,
overlapping windows, or imported sub-corpora. Surfacing them is product
value; provenance preservation per cluster member is what makes this
useful (knowing both copies' source paths).

**Algorithm**: greedy single-linkage at cosine ≥ threshold via
`field.algebra.greedy_cluster` (lifted from `memory/consolidation` to
unify the threshold-tuning history).

---

## Comparison — between two corpora

All three ops use the **base band** per the cross-model rule. Optimised
bands are corpus-specific projections — their dimensions and orientations
aren't comparable across knowledge models.

### `compare(contents_a, contents_b, *, sample_size=512, seed=0) -> CompareResult`
**Purpose**: thematic alignment + asymmetric coverage in one struct.

**Fields returned**:
- `centroid_cosine` — single thematic-alignment number
- `coverage_a_in_b` / `coverage_b_in_a` — asymmetric mean max-cosine
  (sampled, deterministic via SeedSequence.spawn for honest sub-streams)
- `overlap_score` — symmetric average of the two coverages
- `revision_match` — true when both KMs were built at the same encoder
  revision

**Why folded `overlap` in**: `overlap` was originally a separate op
returning a single scalar. Folded as a `CompareResult` field — same
information, no duplicate op.

### `unique(contents_a, contents_b, *, threshold=0.7) -> list[Citation]`
**Purpose**: passages in A whose top-1 match in B falls below threshold —
"what's distinctive about A?"

**Why kept**: pairs naturally with `intersect` for set-algebra-like
comparison workflows. `threshold=0.7` is a reasonable "approximate
semantic match" floor for gte-mb-base.

### `intersect(contents_a, contents_b, *, threshold=0.7, max_pairs=10000) -> list[PassagePair]`
**Purpose**: passage pairs across two KMs with cosine ≥ threshold —
"what do these two corpora agree about?"

**Why kept**: cross-corpus consensus detection with full source
attribution. Caps result count via `max_pairs` because for two corpora
of N=10K passages at threshold=0.7, the cross product can be millions
of pairs.

---

## Composition — multi-KM federated + write-out

### `compose(**named_kms) -> ComposedKnowledgeModel`
**Purpose**: federated read-only view over N knowledge models. Each
`.search` runs against every member's base band; results carry a
`corpus_label` so attribution is preserved across the federation.

**Why kept**: most retrieval libs require you to manually merge collections
to query across them. Compose preserves attribution AND avoids the
write-out cost of merge.

**Backbone-revision mismatch**: emits `UserWarning` (not raise) — cosine
ordering still holds across distinct embedding distributions; magnitudes
are across different distributions but the user can decide whether that
matters.

### `merge(contents_a, contents_b, output_path, *, dedupe_threshold=0.92) -> int`
**Purpose**: physically merge two KMs into a new `.rlat` with semantic
dedupe. Returns kept-passage count.

**Why kept (alongside compose)**: compose is read-time; merge is write-time.
Both have legitimate workflows: compose for "search across many KMs
without committing to a merge"; merge for "consolidate two KMs into one
canonical artefact for distribution."

**Why no `diff`**: the original Phase 6 hypothesis included `diff`. After
the design pass, `diff` was identical to `unique` with a different
default threshold — same operation, different framing. Single op
(`unique`) is the honest one. Net 14 ops, not 15.

---

## Evidence + drift — exploiting verified retrieval

This group is where the verified-retrieval differentiator pays off most
directly.

### `evidence(contents, store, query_emb, *, top_k=10, prefer=None) -> EvidenceReport` ★ FLAGSHIP
**Purpose**: search + per-result citation + corpus-level **calibrated
confidence**.

**Why this is the flagship**: no other RAG library returns calibrated
confidence + verified citations as a first-class result. The
`ConfidenceMetrics` block surfaces:
- `top1_score` — cosine of the top-1 hit ("is retrieval working at all")
- `top1_top2_gap` — separation of best from runner-up (informational; paraphrase clusters invert its meaning, so `_grounding.py` gates on `top1_score`)
- `source_diversity` — unique sources / K (proxy for evidence breadth)
- `drift_fraction` — fraction of top-K with non-verified drift_status
- `band_used` — base or optimised

Drifted hits are NOT filtered — they're surfaced via `drift_fraction`
so the caller can decide whether to trust the evidence base. Pair with
`store.verified.filter_verified()` when drift must be excluded.

### `drift(contents, store) -> DriftReport`
**Purpose**: within-KM drift. Lists passages whose recorded
`content_hash` no longer matches the live source.

**Why kept**: drift-as-data. Most retrievers lose track once embedded;
we know exactly which passages have stale source. Powers maintenance
workflows (`rlat refresh` consumes the same drift signal at the CLI
level).

### `corpus_diff(contents_old, contents_new, *, threshold=0.95) -> CorpusDiff`
**Purpose**: cross-KM passage-level diff between two snapshots. Added /
removed / unchanged classification.

**Why kept (separate from `drift`)**: different question. `drift` asks
"which passages have stale source against the current filesystem?"
`corpus_diff` asks "what changed between these two .rlat snapshots
semantically?" Both are useful; folding them into one polymorphic op
muddied the contract.

**Threshold**: tight (0.95) — material content edits typically drop
cosine well below this.

---

## Experimental — heuristic surfaces

Both ops below are HEURISTIC and will produce noise. They ship under the
experimental banner because the underlying user stories are genuinely
novel under verified retrieval and even imperfect candidate sets are a
non-trivial product surface — but the lexical heuristic isn't a precise
oracle. Iterate post-launch on real corpora.

### `contradictions(contents, store, *, cosine_threshold=0.85, lexical_threshold=0.3, max_pairs=1000, prefer=None) -> list[ContradictionPair]`
**Purpose**: within-corpus pairs with high semantic similarity AND low
lexical overlap. Surfaces "passages saying the same thing in different
words" — true contradictions and benign paraphrases both appear; triage
required.

**Heuristic**: `cosine ≥ cosine_threshold` AND `Jaccard(token-3-gram) ≤
lexical_threshold`.

**Why ship**: no other RAG library surfaces this question at all. Even an
imperfect candidate set with verified citations is a meaningful surface,
especially for editorial review of large corpora.

### `audit(contents, store, claim_emb, *, support_threshold=0.7, top_k_support=10, contradiction_cosine=0.85, contradiction_lexical=0.3) -> AuditReport`
**Purpose**: fact-check a claim against a corpus. Returns supporting
passages, potentially-contradicting passages, source_count, and
drift_fraction.

**Why this isn't just `evidence` + `contradictions` separately**: the
question "did N independent sources back this claim, or 1 source repeated
N times?" requires a single op that surfaces `source_count` over the
SUPPORTING set (not the whole corpus). Composing `evidence` +
`contradictions` doesn't tie the contradicting candidates to THIS claim's
supporting set.

**Always uses base band**: audit results should be reproducible across
KMs built at the same revision; optimised bands break that contract.

---

## What didn't make the cut

The original Phase 6 hypothesis listed these ops; they were dropped after
the first-principles design pass:

- **`bands`** — folded into `Profile.bands` field. A separate op was
  redundant.
- **`overlap`** — folded into `CompareResult.overlap_score` field.
- **`cite`** — synonym for `locate`. Same op.
- **`subtract`** — algebraic-symmetry name for what users actually want
  to call `unique`. Folded.
- **`diff`** — `unique` with a different default threshold. Folded.
- **`bridges`** — boundary-spanning content op (passages with high cosine
  to two distinct clusters). Speculative; deferred to v2.1 if user
  demand emerges.
- **`drift_passages` / `drift_compare`** — single polymorphic `drift` was
  considered but split into `drift` + `corpus_diff` because the contracts
  differ enough that overloading would muddy both.

## What was dropped from v0.11

The legacy `src/resonance_lattice/rql/` directory had ~120 public ops
across 17 files. Of those, **0 ports cleanly to v2.0** — the v0.11 RQL
was framed against density-matrix algebra (eigendecompositions, phase
vectors, Frobenius norms) that v2.0's data model (768d L2-normalised
vectors + dense cosine) doesn't have.

Categories dropped wholesale:
- **EML retrieval-time scoring** (eml.py, ~13 ops): falsified 3× in
  benchmarks (memory: `project_eml_retrieval_falsified`). Exp dominates
  log; collapses to `exp_only` or NaN.
- **Differential geometry** (geometric.py, 15 ops): zero product use
  case, never benchmarked.
- **Dynamics** (dynamics_ops.py, 9 ops): research-phase time evolution,
  none in CORE_FEATURES.
- **Spectral algebra** (spectral.py, ~30 ops, plus algebra_ops.py and
  signal_ops.py): thin numpy wrappers built around density matrices that
  v2.0 doesn't construct.
- **Field-as-content DSL** (dsl.py: Field/.pipe/metabolise/cascade): v2.0
  is router+store, not field-as-content. Bands are float32 NPZ arrays,
  not chainable Field objects.

The audit doc at `docs/internal/audits/06_rql_classification.md` would
have classified all 120 ops if the audit hadn't gone off the rails on
v0.11 framing during Phase 6 — see the rebuild plan's Phase 6 entry for
the post-mortem.

## Property tests

Phase 7 will extend `tests/harness/property.py` with RQL invariants:

- `compare(a, b).overlap_score == compare(b, a).overlap_score` (symmetric)
- `len(unique(a, a)) == 0` (a passage near-matches itself)
- `len(intersect(a, a, threshold=1.0)) == n_passages` (every passage
  matches itself)
- `merge(a, b)` then `compare(merged, a).coverage_a_in_b == 1.0`
  (everything in `a` is in `merged`)
- `near_duplicates(km, threshold=1.0)` returns only exact-equal-vector
  clusters

## Cross-references

- [docs/internal/STORE.md](STORE.md) — verified retrieval contract that
  RQL ops depend on.
- [docs/internal/FIELD.md](FIELD.md) — band selection (`select_band`)
  semantics.
- [docs/internal/HONEST_CLAIMS.md](HONEST_CLAIMS.md) — falsification
  evidence for the v0.11 RQL ops dropped from v2.0 (EML scoring,
  multi-vector, asymmetric field, trained heads, etc.). Where a removal
  decision was driven by measurement, the evidence lives there.
- [src/resonance_lattice/rql/types.py](../../src/resonance_lattice/rql/types.py)
  — full typed-result reference.
