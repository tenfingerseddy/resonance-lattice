# Honest Claims

What `rlat` v2.0 claims, what it doesn't, and what's measured vs. projected. Calibration discipline — every public claim either has evidence behind it or is explicitly flagged as forward-looking.

## What we claim, with evidence

### Retrieval quality (base band, no optimised, no rerank)

> **gte-modernbert-base 768d CLS+L2 + FAISS HNSW** scores BEIR-5 mean **0.5144 nDCG@10** / **0.5666 R@10** at zero training cost.

Evidence: [`benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json`](../../benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json).

Per-corpus locked floor (matches [`BENCHMARK_GATE.md`](BENCHMARK_GATE.md) — single source of truth):

| Corpus | passages | nDCG@10 | R@10 |
|---|---:|---:|---:|
| nfcorpus | 3,633 | 0.3431 | 0.1640 |
| scifact | 5,183 | 0.7672 | 0.8926 |
| arguana | 8,674 | 0.7430 | 0.9637 |
| scidocs | 25,657 | 0.1946 | 0.2014 |
| fiqa | 57,638 | 0.5239 | 0.6114 |

Comparators on the v2.0 stack (same chunker / ANN / scoring — apples-to-apples):
- BGE-large-v1.5 (1024d): 0.4888 — `rlat` base wins by **+0.026 nDCG@10**.
- E5-large-v2 (1024d): 0.4331 — `rlat` base wins by **+0.081 nDCG@10**.
- Qwen3-Embedding-4B: loader-incompat in this run; retry pending v2.0.1.

Evidence: [`benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json`](../../benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json) (encoder comparison, run on Kaggle T4). Plus an off-stack reference: Qwen3-Embedding-8B (last-token pooling) measured 0.500 in earlier work — rlat base wins by **+0.014 nDCG@10** mean, but that's not apples-to-apples against rlat's chunker/ANN, so the v2.0-stack comparators above are the load-bearing numbers.

Honest framing: scidocs scores are intentionally low across the field because the corpus is short-passage retrieval where dense bi-encoders without domain adaptation generally floor at the 0.18–0.22 nDCG@10 band; nfcorpus is ~3K medical passages with long synonym chains. These are corpus characteristics, not method failings.

### Encoder reproducibility

> Determinism: bit-exact across CPU runs of the same revision; L2-norm error < 1.2e-7.

Evidence: Phase 1 close-out empirical runs against `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. Verified on Intel CPU + OpenVINO runtime.

### Latency

> Warm encode 12.1ms / cold 942ms (single query, T4 not warmed).

Evidence: Phase 1 close-out measurement on the standard T4 instance.

ANN search adds ~sub-millisecond. Cold start dominated by encoder load + tokenizer init.

### Storage modes

> All three storage modes — bundled, local, remote — ship in v2.0 and round-trip through the v4 `.rlat` ZIP format.

Evidence: round-trip + parity tests in `tests/harness/`; remote-mode end-to-end confirmed against a fixture HTTP server (the `incremental_sync` harness exercises build → sync-no-op → modify-upstream → sync-delta → catalog-mode-add → catalog-mode-remove → manifest-pin-advance, 5 guarantees, all hermetic). Remote manifest persists as `manifest.json` at the top of the ZIP. v2.0 reconciliation is read-only `rlat freshness` (CI gate) + `rlat sync` (incremental delta-apply on the same `store/incremental.py` pipeline as `rlat refresh`). The codex P0 manifest-only-sync mode is statically impossible — `apply_delta` requires the encoder, the only manifest-write path is `apply_delta`. Audit 07 is the design source of truth.

### Deep-search loop accuracy

> **`rlat deep-search`** scores **92.2% answerable accuracy at 0% hallucination, $0.009/q** on the Microsoft Fabric corpus 11-lane v4 bench (63 questions, Sonnet 4.6, relaxed rubric).

Evidence: [`benchmarks/results/user_bench/hallucination_v4.json`](../../benchmarks/results/user_bench/hallucination_v4.json). Same 11-lane matrix produced these comparators on the same test set:

- LLM+grep/glob/read_file (8 tool calls): 94.1% accuracy at $0.060/q — `rlat deep-search` within 2 pp at **6.5× lower spend** and faster wall-time.
- Single-shot `rlat search --mode augment`: 76.5% accuracy.
- LLM-only (no retrieval): 56.9% accuracy at 19.6% hallucination.
- `rlat search --mode constrain`: 91.7% distractor refusal — the compliance floor for wrong-but-confident-is-worse-than-no-answer workloads.

The bench is built around a real-world corpus the LLM partially knows: Microsoft Fabric documentation has been in market since 2023 (Sonnet has substantial training data) but contains 559 files dated post Sonnet's January 2026 cutoff. This is the right shape to measure whether grounding actually moves the needle vs the LLM's training prior.

### Namecheck distinctive-token verification

> **`--strict-names`** catches name-aliasing distractor failures where the encoder surfaces a similarly-named real entity for a fake-product-name question.

Evidence: 12 of the 63 hallucination-bench questions are distractors — fake F SKU codes, made-up product names, Power-BI-only features asked as if they were Fabric features. Without namecheck, a single-shot retrieval surfaces a similarly-named real entity and the LLM answers about that adjacent entity (a hallucination by user intent, even when the fact is correct about the wrong product). Wired into `rlat skill-context`, `rlat search --format context`, and `_grounding.py` distinctive-token verification on the grounding-emit boundary. Harness suite at [`tests/harness/name_check.py`](../../tests/harness/name_check.py) (16 guarantees).

### Session-start primer effectiveness

> **Code primer** (`rlat summary` → `.claude/resonance-context.md`) **and memory primer** (`rlat memory primer` → `.claude/memory-primer.md`) **have measurable but tier-specific value** — they shine on the tier their content was designed for and degrade to roughly cold elsewhere.

Evidence: [`benchmarks/results/user_bench/primer_effectiveness.json`](../../benchmarks/results/user_bench/primer_effectiveness.json). 25-scenario × 5-lane bench on `resonance-lattice.rlat` (3,506 passages / 126 files), Sonnet 4.6 + 4-state relaxed rubric. Per-tier turn-1 accuracy:

| Tier | cold | code primer | memory primer | both primers | rlat search |
|---|---:|---:|---:|---:|---:|
| 1 — orientation | 0/5 | **3/5** | 0/5 | **3/5** | 0/5 |
| 2 — specific factual | 0/10 | 2/10 | 2/10 | 3/10 | **8/10** |
| 3 — cross-reference | 0/5 | 0/5 | 1/5 | 1/5 | **2/5** |
| 4 — memory recall | 0/5 | 0/5 | **5/5** | **5/5** | 4/5 |

Aggregate turn-1 correct: cold 0% / code primer 20% / memory primer 32% / both primers 48% / rlat search 56%.

Token cost of each surface (Sonnet 4.6 input pricing):
- Code primer: ~1,708 tokens/call (~3 KB markdown).
- Memory primer: ~746 tokens/call (~2 KB markdown).
- Both concatenated: ~2,454 tokens/call.
- `rlat search` top-5: ~704 tokens of passages/call (dynamic per turn).

Combined-stack reading: load both primers at session start (free, ~5 KB) **and** keep `rlat search` available. Primers carry orientation + memory recall; per-turn search picks up specific facts. For session-start questions that turn out to need synthesis, escalate to `rlat deep-search`. Honest framing on the 25-scenario MVP sample: tier-level n=5 is small; per-tier numbers are directional, not precise.

### Source provenance

> Every passage carries `(source_file, char_offset, char_length, content_hash, drift_status)`. Cite-back is free; drift detection is free.

Evidence: ported from v0.11 WS3 #292; `Store.verify` walks the registry against live source bytes. Tested in `tests/harness/property.py:_check_rql_invariants` (drift-aware ops surface drift fraction in evidence reports).

## What we measured on `rlat optimise` — full honest framing

### Three-row table (Fabric / fiqa / nfcorpus)

| Corpus | source | passages | base | optimised | Δ |
|--------|--------|---------:|-----:|----------:|--:|
| Microsoft Fabric docs | private | 62,953 | 0.871 R@5 | 0.903 R@5 | **+0.032** |
| BEIR fiqa | public | 57,638 | 0.524 nDCG@10 | 0.482 nDCG@10 | **−0.042** |
| BEIR nfcorpus | public | 3,633 | 0.343 nDCG@10 | 0.300 nDCG@10 | **−0.043** |

Same locked v2.0 hparams across all three (250 steps, batch 128, 7 negs,
lr 5e-4, τ=0.02, MRL dims {64,128,256,512}, seed 0). Two of three regressed.

### Falsifications

> The base plan §C "+6.8 nDCG@10 mean lift on 3-corpus holdout" projection was **falsified**.

Source: `project_specialist_beir3_falsified.md`. v2.0 BEIR-3 specialist
soak v7 (kernel `kanesnyder/rlat2-beir-3-specialist-soak-v7`, ran
2026-04-26) regressed nfcorpus −0.043 nDCG@10 at d=512. Smaller MRL
slices recovered nothing.

> The "natural-language full-sentence queries lift" rule was **weaker than originally framed**.

fiqa is real natural-language Q&A (real StackExchange financial
questions on forum-post answers) and still regressed −0.042 nDCG@10
under the same locked hparams. Source: `project_specialist_beir3_falsified.md`
+ `benchmarks/results/optimised/beir_fiqa_probe_v1.json`. The actual
predictive signal is **distribution alignment between synth-generated
training queries and deployment-time test queries**, not the surface
"natural-language vs keyword" form. Fabric's win comes from
LLM-generated synth queries matching LLM-driven Sonnet workflows that
deploy against the corpus. fiqa's regression comes from real human
StackExchange queries differing from the synth distribution Sonnet
generates against the corpus.

### What we ship

`rlat optimise` ships **opt-in-with-real-caveat**. The
[user OPTIMISE.md](../user/OPTIMISE.md) "When to optimise / When not"
sections are explicit about the conditions under which lift is
expected vs unexpected. The README and BENCHMARKS.md surface both the
Fabric win and the BEIR-fiqa / nfcorpus losses in the same table.

### MRL hparam validity

The locked hparams are validated only on Fabric (where they produced
the +0.032 R@5 lift). On BEIR-3 nfcorpus and BEIR fiqa they trained
fine (train_dev_r1 0.94 / 0.67) but generalised badly to the test
distribution. Hparams are not at fault — the synth-query → deployment-
query distribution mismatch is. We do not ship hparam alternatives in
v2.0; the falsification suggests the right product move is to make the
distribution-alignment requirement clearer in OPTIMISE.md, not to
re-tune hparams blindly.

## What we do NOT claim

### Reranking

> No claim that adding a cross-encoder reranker improves quality.

Evidence: cross-encoder rerankers (bge-reranker-v2-m3, mxbai-rerank) regressed gte-mb-base on **4 of 5 BEIR corpora** in measured comparisons. The training-distribution mismatch between the reranker's MS MARCO/Natural Questions data and a strong-dense top-k is the documented mechanism. Memory: `project_ce_rerankers_hurt_strong_dense.md`. We do not ship a reranker; we don't recommend bolting one on.

### Lexical / hybrid

> No claim that adding a BM25 / sparse-lexical sidecar improves retrieval.

Evidence: lexical V1 band failed BEIR-5 parity on 4 of 5 corpora (avg -2.7 pt). Memory: `project_lexical_band_v1_parity.md`. We don't ship a lexical sidecar.

### Multi-vector / asymmetric / trained heads

> No claim that v0.11's multi-vector, asymmetric-field, or trained-head architectures improve retrieval.

Evidence: trained heads closed 0-for-9 in measured experiments; multi-vector showed catastrophic regressions; asymmetric field measured -0.8% nDCG@10. Memory: `project_asymmetric_field.md`. None ship in v2.0.

### Query-prefix tuning

> No claim that adding query/passage prefix tokens improves quality.

Measured null on strong-dense in Phase 0; not shipped.

### EML retrieval scoring

> No claim about EML-based scoring or fusion.

Evidence: falsified 3 times in independent benchmarks (exp dominates log; collapses to `exp_only` or NaN). Memory: `project_eml_retrieval_falsified.md`. Note: user-facing CLI EML transforms (`--sharpen` etc.) were a SEPARATE feature in v0.11; v2.0 doesn't ship either form.

## Known limits

### Scale

- **(N, N) cosine matrix ops** (`near_duplicates`, `merge` dedupe, `contradictions`) cap at ~50K passages per call — the float32 matrix is ~10 GB at that size. Above this, callers must pre-shard.
- **`compose` federated search** has no ANN attach point yet — high-QPS workloads with many member knowledge models will pay per-member matmul cost on every query. Adequate for single-user / small-team integration; an ANN extension hook ships when there's a real workload profile.

### Heuristic ops

- **`contradictions`** uses Jaccard over token-3-grams as the lexical-disjointness proxy. Will surface paraphrases AND true contradictions; triage required. Flagged experimental.
- **`audit`** composes `evidence` + the same lexical heuristic. Same caveat.
- Both ops ship anyway because no other RAG library addresses these questions, and a triagable candidate set with verified citations is meaningfully better than nothing — but consumers should not treat their output as ground truth.

### Storage modes

- **`merge` doesn't carry source/ across bundled-mode inputs** in v2.0. Raises `NotImplementedError` if either input is bundled. Workaround: rebuild inputs in local mode before merging.
- **`rlat refresh` for remote-mode KMs** prints a friendly error pointing at `rlat sync`. Refresh is the local-disk delta-apply path; sync is the remote delta-apply path. Both land on `store/incremental.py`.

### Optimised band

- Optimised is corpus-specific. Doesn't transfer to OTHER corpora. Doesn't help cross-knowledge-model `compare`/`unique`/`intersect` (those use base band by design).
- The optimised band re-projects from the new base on every `rlat refresh` and `rlat sync` — `optimised = (new_base @ W.T)` row-wise L2-normalised, sub-second for 50K passages, no LLM call, no GPU. The W matrix is preserved byte-identically (no retraining). Pass `--discard-optimised` to drop the band instead of re-projecting (rare).
- Below ~1000 passages: the InfoNCE training early-kills at step 100 if dev R@1 < 0.2. You'll spend $14-21 on Haiku and get a band no better than base.

### Memory

- **Tier pair-write recovery is implemented**, but a crash before the first `os.replace` of either tier can still leave an orphaned tmp file. The load-time self-heal handles this; no data loss; some tmp file cleanup may be required in pathological cases.
- **Embedding staleness contract**: `recall` and `all_entries` mutate `entry.embedding` in place to re-attach from the tier's NPY slice. Long-lived API consumers should `entry.embedding.copy()` if they want to hold it across re-queries.

### MCP / HTTP

- **No MCP server in v2.0.** CLI is the primary interface **by design** — v2.0 stabilises the CLI surface as the canonical entry point so a future MCP wrapper can passthrough without forcing a redesign. We declined to ship MCP early specifically to avoid coupling protocol decisions to an unstable CLI. The planned AI-assistant bridge ships post-v2.0 once the CLI contract is locked.
- **No HTTP server in v2.0.** `rlat serve` was dropped during the doc audit (no current consumer; the planned MCP bridge supersedes it). HTTP wrappers around the rlat Python API are user-buildable today (cf. `docs/direction/FABRIC_INTEGRATION.md`). NOTE: this is unrelated to remote *storage mode*, which DOES ship — remote mode is about how a `.rlat` resolves source files (HTTP-backed manifest), not about exposing rlat itself over HTTP.

## Calibration philosophy

- A **claim with measurement** uses present-tense factual framing ("scores 0.5144 nDCG@10").
- A **projection** uses explicit forward-looking framing ("projected to lift +6.8 pt").
- **No silent regressions** — anything we measured negative on (rerankers, lexical, multi-vector, EML scoring) is documented as such with the failure mode named.
- **No silent feature gates** — anything we promise in CLI help / docs is wired and shippable. Anything not wired is flagged as such ("remote mode isn't shippable in v2.0").

When in doubt, we prefer to under-promise. The product is better than its claims, not the other way around.
