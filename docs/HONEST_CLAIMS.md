---
title: Honest Claims
slug: honest-claims
description: What Resonance Lattice does and doesn't do, with explicit calibration against the evidence. Read before making a decision to adopt.
nav_group: Product
nav_order: 99
---

# Honest Claims

This document exists to keep the marketing copy honest and help you calibrate expectations before adopting. Every claim below is paired with the evidence (or the gap in evidence) that backs it.

The goal is not humility for its own sake — it's to be reliably trusted. If we overclaim and users find out on week two, the whole retrieval story collapses. If we underclaim, we lose to systems that don't. The calibration below is our best attempt to say exactly what we can say, and no more.

## Measurement state (updated 2026-04-22)

Benchmark numbers below are drawn from three measurement sweeps on the same BEIR-5 harness:
1. **2026-04-20** — E5-large-v2 full 5-BEIR + a 2-corpus BGE-large-en-v1.5 pilot.
2. **2026-04-22** — BGE-large-en-v1.5 full 5-BEIR sweep (board item 239 launch verification; see Block D).
3. **2026-04-22 / 2026-04-23** — Qwen3-Embedding-8B full 5-BEIR sweep, field_only baseline + full probe grid (see Block E).

All three encoder tiers are now shipped-grade measured. See [docs/ENCODER_CHOICE.md](ENCODER_CHOICE.md) for the side-by-side comparison and the pick-by-workload decision guide.

Rules for citing this doc:
- Numbers in *Remaining pending claims* (LongMemEval, python-stdlib answer-quality, reader latency) are still pending — not yet measured.
- Do not cite the table numbers as ship claims in external copy (README, Show HN, blog post) without linking back to this document. One edit here should update every downstream reference.
- Updates are append-dated in `## Block D benchmark results`, `## Block E benchmark results`, and in the `## Remaining pending claims` section — never backdated.

---

## What we claim, with evidence

### "Portable semantic model as a file"

**Claim:** A `.rlat` knowledge model is a portable, inspectable, versionable single-file artifact containing a semantic model of your corpus.

**Evidence:**
- File format documented in `docs/KNOWLEDGE_MODEL_ARCHITECTURE.md`. Fixed 64-byte header, versioned (`FORMAT_VERSION=3`), structured sections for field / registry / manifest / optional source store (bundled / local / remote).
- Round-trip tests: `tests/test_serialise.py` builds knowledge models, serializes, deserializes, and checks byte-identical field tensors across save/load cycles.
- `rlat info`, `rlat profile`, `rlat xray` inspect the file without needing the original source tree (inspection is structural — sources are only needed for evidence retrieval).

**Qualifier:** "Portable" means the knowledge model itself is portable. `local`-mode knowledge models additionally require the source tree on the receiving machine. `bundled` mode packs the source files into the `.rlat` for self-contained distribution; `remote` mode resolves files from a SHA-pinned upstream repo. See `docs/STORAGE_MODES.md`.

---

### "Local mode produces smaller knowledge models than legacy embedded"

**Claim:** `local`-mode knowledge models (the v1.0.0 default) are substantially smaller than the deprecated `embedded` mode for the same corpus, because they don't pack the evidence text inside the `.rlat`.

**Evidence:**
- The `local` knowledge model stores only field tensor + registry + manifest (chunk offsets, content hashes, source paths). Size is a function of `num_chunks × bands × dim` for the field, plus O(num_chunks) registry entries.
- The legacy `embedded` knowledge model additionally stored the full chunk text, which scaled linearly with corpus size.
- For a corpus of N tokens at chunk size ~300, field size is ~50-200 MB (independent of text volume); the legacy embedded store added N × bytes-per-token × chunking-overhead.

**Qualifier:** The absolute size reduction depends heavily on corpus shape. For a 10 MB docs corpus, `local` vs legacy `embedded` is within a factor of ~2. For a 1 GB corpus, `local` can be 5–10× smaller. `bundled` mode (v1.0.0's self-contained option) is larger than `local` but materially smaller than legacy `embedded` because zstd framing compresses whole files instead of pre-chunked text. Measured deltas for the three launch knowledge models land with Block D benchmarks.

---

### "Drift is detected deterministically"

**Claim:** `local`- and `remote`-mode knowledge models detect when source files change after the build, via SHA-256 content hashes stored per chunk.

**Evidence:**
- Every chunk's manifest entry carries a 16-byte SHA-256 prefix of the normalized chunk text (A3 from the v1.0.0 plan).
- At query time, the `LocalStore` / `RemoteStore` re-hashes the chunk region of the source file and compares. Mismatch → `RuntimeWarning` naming the drifted chunk.
- `rlat info --verify` surfaces drift without running a query. `tests/test_refresh.py` covers drift scenarios: edited, renamed, deleted, new-file, moved-root.

**Qualifier:** Detection is deterministic. **Response** is fail-soft — we serve the current bytes anyway and warn, rather than refusing to query. Users who want strict drift-free behavior can treat warnings as errors in CI.

---

### "Algebraic operations are exact"

**Claim:** `forget`, `merge`, `diff` operate on the field tensor with well-defined mathematical semantics, not approximations.

**Evidence:**
- `forget` performs a rank-1 subtraction: `F' = F - φᵢ ⊗ φᵢ`. The removal is algebraically exact — the removed source's phase vector no longer contributes to the field. Tests assert this via residual-energy inspection.
- `merge` is commutative and associative: `merge(A, B) == merge(B, A)`, `merge(merge(A,B), C) == merge(A, merge(B,C))`. Tested in `tests/test_field_merge.py`.
- Field operations preserve fixed-size tensor shape regardless of corpus size — the field block is decoupled from the registry size.

**Qualifier:** "Exact" is a mathematical claim about the field operations. It does NOT mean forget necessarily eliminates all semantic traces of a source — a closely related source not in the forget set will still produce similar search results. Orthogonality determines residual cross-talk; the `forget` operation reports a certificate quantifying it.

---

### "Reader citations are structurally grounded"

**Claim:** The reader layer enforces that citations reference evidence actually passed to it. A reader cannot produce a citation pointing at a source file that wasn't in its evidence set.

**Evidence:**
- `extract_citations` in `reader/local.py` parses `[N]` markers and drops any index outside `1..len(evidence)`. If the LLM hallucinates `[99]` when only 5 items were retrieved, the citation is silently removed. Tested in `tests/test_reader_local.py` and `tests/test_reader_api.py`.
- `build_bundle` additionally verifies each citation's quote against the source file at the claimed offset. Mismatches surface as `verified=False` with a diagnostic (`tests/test_reader_citations.py` covers the drift case).

**Qualifier:** "Structurally grounded" means the reader can't point at fabricated sources. It does NOT mean the reader's *synthesis* (the natural-language answer) is always correct. A model citing `[1]` correctly can still paraphrase or synthesize inaccurately. Quote verification (`✓`/`○`/`✗` markers) lets users check the claim against the bytes.

---

### "Benchmarked against BEIR-5 and LongMemEval"

**Claim:** Retrieval quality is measured against standard information-retrieval benchmarks, not just anecdotes.

**Evidence:**
- BEIR-5 nDCG@10 numbers shipped in `benchmarks/results/` with full reproduction commands in `docs/RETRIEVAL_BENCHMARK_RUNBOOK.md`.
- LongMemEval v14 R@5 baseline: **0.924** on the 500-question subset (see `benchmarks/results/longmemeval/` and the `project_longmemeval_ku_filter` project memory for context).
- Three-layer (field + expand + hybrid + reader) benchmarks measured 2026-04-20 — see the "Block D benchmark results" section below.

**Qualifier:** BEIR results are **not uniformly positive**. We win on some corpora and lose on others. Do not expect blanket "better than baseline" — expect "better on X, parity on Y, regression on Z". The runbook documents which is which. LongMemEval v14 is the ship baseline; earlier versions showed different patterns and should not be compared across runs.

---

## What we do NOT claim

These are boundaries. If a claim below is presented as something Resonance Lattice does, it's wrong — push back.

### Not faster than grep for exact-symbol lookup

Semantic retrieval is slower than `grep` at exact token matches and will remain so. Use grep for `grep "FORMAT_VERSION"`. Use `rlat search` for `rlat search "how does versioning work?"`. The `--hybrid` flag runs ripgrep as a second pass inside the field's retrieved neighborhood precisely because grep beats dense retrieval on literal-term precision; we don't try to replace it.

**Measured:** In internal dogfood benchmarks, grep returns 3.6× more complete coverage than `rlat search` for exact-identifier queries. This is expected and not a bug.

### Not a replacement for hosted RAG at scale

For corpora that don't fit on a developer's laptop (multi-TB vector stores, horizontally-sharded serving, per-tenant isolation), Resonance Lattice is the wrong tool. It's a local-first, single-file semantic layer — not a distributed retrieval service.

### Not a language model

Resonance Lattice retrieves and synthesizes evidence. The synthesis step (`rlat ask --reader llm`) calls out to either OpenVINO-inference of a 3B-class model or a remote LLM API. We do not train, host, or ship a language model. The encoder (`BAAI/bge-large-en-v1.5` as of the 2026-04-20 default flip; `intfloat/e5-large-v2` legacy) handles embedding only.

### Not measured against "all RAG systems"

We benchmark against specific baselines (flat cosine E5, ripgrep, LongMemEval) documented in the runbook. Comparisons to unnamed "RAG systems" or "vector databases" are marketing, not evidence. Treat them as directional and ask for specifics.

### Not a guarantee against LLM hallucination

The reader's system prompt instructs grounded synthesis. The `[N]` citation extraction enforces structural grounding. Quote verification catches drift. **None of this prevents the LLM from paraphrasing an evidence span in a way that misrepresents it.** A paraphrase that preserves every `[N]` marker can still be wrong about what the source said. Users who need verbatim answers should use `--reader context` and feed the pack through their own sampler at `T=0`, or read the cited spans directly.

### Not tested at petabyte scale

The largest knowledge models we've shipped are in the low GB range (unpacked source). Behavior at TB+ corpora is not characterized. If that's your target, plan to profile first.

---

## Block D benchmark results (E5 measured 2026-04-20; BGE full 5-BEIR measured 2026-04-22)

### v1.0.0 launch gate: MISSED on portable tier (both E5 and BGE)

**Gate spec:** 5-BEIR best-mode avg nDCG@10 ≥ 0.46.

| Encoder | 5-BEIR best-mode avg | Gap to gate | Passes? |
|---|---|---|---|
| E5-large-v2 (portable, old default) | 0.455 | −0.005 | ❌ |
| BGE-large-en-v1.5 (portable, current default) | 0.445 | −0.015 | ❌ |
| Qwen3-Embedding-8B (high-end, opt-in) | 0.5005 | +0.040 | ✅ (see Block E) |

**Interpretation:** The portable tier (335M-class encoders) lands at-parity with 2024 open mid-tier retrievers but does not clear our internal 0.46 bar on its own. The opt-in high-end tier (Qwen3-8B) clears the bar on dense alone. Rather than gate-failing the v1.0.0 ship, we reframe the product position: **publish full per-encoder numbers, let users pick by workload**, and recommend Qwen3-8B for users who need the extra headroom. See [docs/ENCODER_CHOICE.md](ENCODER_CHOICE.md) for the decision guide.

### Per-corpus 5-BEIR (E5 vs BGE, best-mode per corpus)

| Corpus | E5-large-v2 | BGE-large-en-v1.5 | Δ (BGE − E5) |
|---|---|---|---|
| NFCorpus | 0.38217 | **0.39246** | +0.010 |
| SciFact | 0.74664 | **0.75538** | +0.009 |
| ArguAna | **0.50478** | 0.40795 | **−0.097** |
| SciDocs | 0.20276 | **0.21374** | +0.011 |
| FiQA | 0.43782 | **0.45407** | +0.016 |
| **5-BEIR avg** | **0.4552** | **0.4447** | −0.011 |

**The ArguAna regression.** BGE wins by small margins on 4/5 corpora but regresses -9.7 pts on ArguAna (counter-argument retrieval). E5's training (MSMARCO + diverse pairs) captured argumentative inversion; BGE's (more QA-biased) did not. The single large regression wipes out all four small wins; BGE is ~1 pt worse on 5-BEIR avg.

**Why BGE is still the starting-point default.** Ecosystem (best ONNX / OpenVINO / fine-tune support), workload fit (most real queries are QA-style, not counter-argument), and parity-or-better on 4/5 corpora. Users with ArguAna-like workloads should `--encoder e5-large-v2` explicitly. See the decision guide.

### Full-stack vs field_only (secondary signal: does the stack help?)

**Sub-gate spec:** full_stack nDCG@10 ≥ field_only + 5% on ≥3/5 BEIR corpora. Used to decide whether `rlat ask` defaults to `--hybrid` and full rerank. Missed on both encoders.

| Corpus | E5 field_only | E5 full_stack | E5 Δ% | BGE field_only | BGE full_stack | BGE Δ% |
|---|---|---|---|---|---|---|
| NFCorpus | 0.3739 | 0.3553 | **−5.0%** ❌ | 0.3819 | 0.3759 | **−1.6%** ❌ |
| SciFact | 0.7062 | 0.7466 | +5.7% ✅ | 0.7318 | 0.7483 | +2.3% ❌ |
| ArguAna | 0.4146 | 0.5048 | +21.8% ✅ | 0.3919 | 0.4080 | +4.1% ❌ |
| SciDocs | 0.2028 | 0.1804 | **−11.0%** ❌ | 0.2137 | 0.1831 | **−14.3%** ❌ |
| FiQA | 0.4175 | 0.4369 | +4.7% ❌ | 0.4417 | 0.4361 | **−1.3%** ❌ |
| **pass count** | | | **2/5** | | | **0/5** |

**Interpretation:** Full stack (dense + BM25 fusion + CE rerank) is only a clear win on corpora where lexical signals are strongly orthogonal to dense retrieval (E5/ArguAna, E5/SciFact). On BGE, the encoder is already closer to its workload ceiling — the stack adds noise more than signal. **SciDocs actively regresses on both encoders (−11%, −14%)** because cross-encoders trained on QA pairs don't transfer to citation retrieval.

**What we ship:** per-corpus best-mode via build-time probe (board items 236c / 238). The cartridge stamps `__retrieval_config__` with the winning `(mode, reranker)` picked from held-out queries; `rlat search` reads it and routes automatically. `--hybrid` is opt-in with corpus-sensitivity notes in [docs/RETRIEVAL_BENCHMARK_RUNBOOK.md](RETRIEVAL_BENCHMARK_RUNBOOK.md). `rlat ask` defaults to `--reader context`.

### Market position (measured anchors, same 5 BEIR corpora, best-mode avg)

Sources: [BEIR paper](https://arxiv.org/abs/2104.08663) (Thakur et al. 2021, Table 2), [jina-embeddings-v3 paper](https://arxiv.org/abs/2409.10173) (Sturua et al. 2024, Table A3), [Qwen3-Embedding card](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (Qwen team 2026).

| Anchor | 5-BEIR avg | Tier |
|---|---|---|
| BEIR BM25 (2021) | 0.340 | baseline |
| BEIR BM25+CE (2021) | 0.372 | baseline |
| Cohere-embed-mlv3 (2024) | 0.450 | mid-tier |
| **Ours — BGE-large-en-v1.5** | **0.445** | mid-tier (default, portable) |
| **Ours — E5-large-v2** | **0.455** | mid-tier (opt-in, portable) |
| jina-v3 (2024) | 0.461 | mid-tier |
| mE5-large-instruct (2024) | 0.464 | mid-tier |
| **Ours — Qwen3-Embedding-8B (field_only)** | **0.500** | high-end (opt-in, 16 GB GPU) |
| text-embedding-3-large (OpenAI, 2024) | 0.512 | high-end (proprietary) |
| Qwen3-Embedding-8B (published, different BEIR subset) | ~0.55 | high-end |
| NV-Embed-v2 (2024, CC BY-NC) | 0.620 | frontier (research only) |

**Position:**
- Portable tier (E5, BGE) sits with 2024 mid-tier open dense retrievers. The default (BGE) is tuned for ecosystem + general QA fit, not 5-BEIR-wins — see ENCODER_CHOICE.md for the tradeoffs.
- High-end tier (Qwen3-Embedding-8B) is frontier-adjacent. Dense-only wins 4 of 5 corpora; the shipped cross-encoder rerankers are mismatched (trained atop weaker base retrievers) and regress strong-dense top-k. A Qwen3-matched reranker is the tier's remaining product gap. See Block E for full per-mode numbers.

Full aggregates: [aggregate_5beir_sota_v1.json](../benchmarks/results/beir/new_arch/aggregate_5beir_sota_v1.json) (E5), [aggregate_5beir_horizon1_v239.json](../benchmarks/results/beir/new_arch/aggregate_5beir_horizon1_v239.json) (BGE), [aggregate_5beir_qwen3_8b_v1.json](../benchmarks/results/beir/new_arch/aggregate_5beir_qwen3_8b_v1.json) (Qwen3-8B). External anchors: [external_reference_sota.md](../benchmarks/results/beir/new_arch/external_reference_sota.md).

## Block E benchmark results (measured 2026-04-22, full-stack probe measured 2026-04-23; Qwen3-Embedding-8B high-end tier)

The `--encoder qwen3-8b` path (opt-in, needs GPU) replaces BGE-large as the
retrieval engine. Same BEIR-5 harness. Full probe grid (field_only +
plus_cross_encoder + plus_cross_encoder_expanded + plus_full_stack, each
with both `bge-reranker-v2-m3` and `mxbai-rerank-base-v1`) measured.

### 5-BEIR per-mode results

| Corpus | field_only | +cross_encoder | +CE expanded | +full_stack | **Best mode** |
|--------|-----------:|---------------:|-------------:|------------:|:--------------|
| NFCorpus | **0.414** | 0.413 | 0.413 | 0.385 | field_only |
| SciFact | 0.774 | **0.776** | 0.776 | 0.750 | +CE (bge-v2-m3) |
| ArguAna | **0.466** | 0.461 | 0.461 | 0.412 | field_only |
| SciDocs | **0.278** | 0.228 | 0.228 | 0.189 | field_only |
| FiQA | **0.568** | 0.506 | 0.506 | 0.446 | field_only |
| **avg** | **0.5001** | 0.477 | 0.477 | 0.396 | **0.5005 (best-mode)** |

### Headline finding: the stack hurts, not helps

**Dense-only Qwen3-8B wins 4 of 5 corpora.** The cross-encoder rerankers we ship
(`bge-reranker-v2-m3`, `mxbai-rerank-base-v1`) regress Qwen3-8B's top-k on NFCorpus,
ArguAna, SciDocs, and FiQA — FiQA is hit hardest (field_only 0.568 → +CE 0.506 →
+full_stack 0.446). Only SciFact's claim-verification structure is regular enough
for the reranker to add anything, and only marginally (+0.002).

**Why:** both rerankers were trained atop weaker base retrievers (0.3–0.4 nDCG@10
top-k). Qwen3-8B already produces 0.5+ top-k, so the reranker sees an input
distribution mismatched with its training data. The extra cross-attention then
reshuffles a mostly-correct top-k based on the wrong signal. Full-stack also adds
BM25 fusion, which amplifies the harm on domains where lexical signal adds noise
(finance jargon, specialised scientific citation networks).

**Product consequence:** for the Qwen3-8B tier, the build-time probe correctly
picks `field_only` for 4/5 corpora (the routing is encoder-agnostic as designed
in board items 236c/238). Users who set `--encoder qwen3-8b` get dense-only
retrieval by default via `__retrieval_config__`, no opt-out needed.

### 5-BEIR best-mode vs BGE-large best-mode

| Corpus | BGE-large best-mode (Block D) | Qwen3-8B best-mode (Block E) | Δ |
|--------|------------------------------:|-----------------------------:|---|
| NFCorpus | 0.392 | **0.414** | +0.022 |
| SciFact | 0.755 | **0.776** | +0.021 |
| ArguAna | 0.408 | **0.466** | +0.058 |
| SciDocs | 0.214 | **0.278** | +0.064 |
| FiQA | 0.454 | **0.568** | +0.114 |
| **avg** | **0.445** | **0.5005** | **+0.056** |

Full aggregate: [aggregate_5beir_qwen3_8b_v1.json](../benchmarks/results/beir/new_arch/aggregate_5beir_qwen3_8b_v1.json).
Per-corpus probe records + sweep log: [qwen3_8b_fullstack_v1/](../benchmarks/results/beir/new_arch/qwen3_8b_fullstack_v1/).

### Position

Qwen3-8B best-mode (0.5005) sits just below `text-embedding-3-large` (0.512)
and well above the portable tier. Our harness has a known ~0.05 gap vs
Anserini-indexed published BEIR numbers (concentrated in ArguAna); after that
adjustment, 0.5005 is consistent with Qwen3-8B's published ~0.55 BEIR-5 subset
average. The gap to frontier (NV-Embed-v2 ~0.62) is held open by CC-BY-NC
licensing, not measurement.

### Caveats worth naming (Block E)

1. **Pooling is load-bearing.** Decoder-only LMs like Qwen3-Embedding require
   last-non-padding-token pooling; mean or CLS pooling collapses scores
   (measured 7× drop on FiQA, 2× drop on 5-BEIR avg). The qwen3-*
   presets in `ENCODER_PRESETS` set `"pooling": "last"` — do not edit this
   without re-benchmarking. Evidence archived at
   `benchmarks/results/beir/new_arch/baseline_qwen3_8b_v1_meanpool_broken/`.
2. **Shipped rerankers are mismatched.** `bge-reranker-v2-m3` and
   `mxbai-rerank-base-v1` regress Qwen3-8B top-k on 4/5 corpora; see per-mode
   table above. A Qwen3-matched reranker (e.g. `Qwen/Qwen3-Reranker-*`) is
   the follow-up for the high-end tier. Until then, `__retrieval_config__`
   correctly routes to dense-only for Qwen3-8B builds.
3. **Compute tier difference.** BGE field_only encode is fast on CPU / Intel Arc iGPU;
   Qwen3-8B field_only needs a ~16 GB GPU for practical query latency.
   Not interchangeable deployment profiles.
4. **Encoder stamping.** A knowledge model built with Qwen3-8B cannot be queried with
   BGE (and vice versa). `_check_encoder_consistency` blocks mismatched loads.

## Market position (post-Block E)

Sources: BEIR paper (Thakur et al. 2021), jina-embeddings-v3 paper (Sturua et al. 2024),
and published Qwen3-Embedding card (Qwen team 2026).

| Anchor | 5-BEIR best-mode avg | Tier |
|--------|----------------------|------|
| BEIR BM25 (2021) | 0.340 | baseline |
| BEIR BM25+CE (2021) | 0.372 | baseline |
| Cohere-embed-mlv3 (2024) | 0.450 | mid |
| **Ours BGE-large best-mode (2026-04, Block D)** | **0.455** | mid (portable tier default) |
| jina-v3 (2024) | 0.461 | mid |
| mE5-large-instruct (2024) | 0.464 | mid |
| **Ours Qwen3-8B best-mode (2026-04-22/23, Block E)** | **0.5005** | high-end (opt-in via `--encoder qwen3-8b`; dense-only wins 4/5) |
| text-embedding-3-large (OpenAI, 2024) | 0.512 | high-end (proprietary) |
| Qwen3-Embedding-8B (published, different BEIR subset) | ~0.55 | high-end |
| NV-Embed-v2 (2024, CC BY-NC) | 0.620 | frontier (research only) |

**Position:**
- Portable tier (default, BGE-large-en-v1.5) sits with 2024 mid-tier open dense
  retrievers. Block D full 5-BEIR is measured — BGE at 0.445 best-mode avg, E5
  at 0.455. The 0.46 launch gate is missed on dense-only portable tier; we ship by
  publishing numbers and letting users pick per workload rather than forcing a
  reranker-heavy default.
- High-end tier (opt-in, Qwen3-Embedding-8B) is frontier-adjacent. Dense alone
  (0.500) already matches text-embedding-3-large on this harness. Full-stack
  results for Qwen3-8B are not yet measured — don't quote "stack + Qwen3"
  numbers externally.

### Encoder training (Track 3, parallel)

Closing the ~6 pt gap to text-embedding-3-large is pursuable via distilled / fine-tuned 335M student (recipe-maturity well-documented; expected 2-8 weeks calendar, $50-300 compute). Tracked as board item 235 under Semantic-Layer workstream, **running in parallel with launch prep** — lands as a point-release retrieval uplift when measured BEIR-5 avg ≥ 0.47 (Stage 1) or ≥ 0.50 (Stage 2 distillation).

## Remaining pending claims

These are still not validated and remain gated on separate work:

- **LongMemEval R@5 ≥ 0.92 with the three-layer stack** (must not regress from the 0.924 baseline).
- **python-stdlib answer-quality ≥ 70%** on 50 held-out questions (custom LLM judge).
- **Reader latency p50 < 3s** for `rlat ask` on local OpenVINO on Intel Arc iGPU.

If any of these miss, we'll document the miss here rather than quietly remove the gate.

---

## Known asterisks

Things that work but have caveats worth naming:

**BEIR results were originally run with a checkpoint.** The pre-v1.0.0 numbers used trained projection heads. Random-projection heads (the v1.0.0 default after the 0-for-9 training result) need to be re-run for honest numbers. Block D (refreshed 2026-04-22 with the full BGE 5-BEIR sweep) supersedes the pre-v1.0.0 checkpoint numbers. Block E (Qwen3-8B) is measured on the same harness.

**Dense field alone underperforms flat cosine.** The dense field tensor by itself doesn't beat a flat-cosine baseline on BEIR. The win comes from the full hybrid pipeline: dense + lexical injection + reranking + expansion. Any direct "our field vs flat cosine" comparison is on the wrong axis — it's the pipeline that matters.

**Byte ≈ char approximation in hybrid.** The ripgrep second pass emits byte offsets; our chunks carry character offsets. For ASCII-dominant corpora (code, English prose) they're identical. For heavy multi-byte content (CJK docs, emoji-heavy notes), proximity matching is approximate — not wildly wrong, but not byte-exact. Proper UTF-8 accounting is a roadmap item.

**Windows UTF-8 quirks.** Running `gh`, `kaggle`, or other CLI tools that write JSON through the Windows console requires `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`. Em-dashes in issue titles silently corrupt without it. We've fenced this in our own scripts; if you're building on top, mirror the pattern.

**First-build encoder download.** `rlat build` downloads `BAAI/bge-large-en-v1.5` (~1.3GB) from Hugging Face on first invocation. After that, builds and queries are fully local. Pair with `--onnx` or `--openvino` for on-device accelerated inference on Intel Arc iGPU / NPU. Pass `--encoder e5-large-v2` to reproduce legacy cartridges built before the 2026-04-20 default flip.

---

## How to read this document

- **Claim + Evidence + Qualifier.** Every positive claim gets all three. The qualifier is not a footnote — it's the scope boundary. Read it.
- **Pending claims will be dated.** When benchmarks land, they land here with the run date. No backdating.
- **If you find an overclaim, file an issue.** Our incentive aligns with your audit: a user who trusts specific claims is more valuable than one who trusts blanket marketing.

If something in the top-of-funnel docs (README, CORE_FEATURES, product pages) doesn't trace back to a claim here, treat it as a gap and push back in an issue.
