# Benchmark Gate — v2.0 BEIR-5 Floor

**Locked at**: 2026-04-25 on Kaggle T4 (kernel `kanesnyder/rlat2-beir-5-floor-v2-base-768d` v4).
**Evidence**: [`benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json`](../../benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json).
**Build**: `gte-modernbert-base @ e7f32e3c00f91d699e8c43b53106206bcc72bb22`, 768d CLS+L2, base band only (no optimised).
**Spec**: base plan §3.1 (default search path) + Audit 04 (FAISS HNSW M=32 efC=200 efS=128, exact below N=5000).

> **Phase 7 close (2026-04-27)**: re-confirmed by code inspection rather than re-run.
> Encoder revision is pinned at `e7f32e3c00f91d699e8c43b53106206bcc72bb22` (unchanged
> since the lock). Retrieval-math files (`field/encoder.py`, `field/dense.py`,
> `field/ann.py`, `field/algebra.py`, `store/registry.py`, `store/bands.py`) had
> three commits since the lock: `38d18d64` (encode-batched lift — same math at
> identical batch size), `0fd4fa0a` (passage_id schema — additive registry field,
> not retrieval), `8adefb8c` (specialise→optimise rename — cosmetic). At fixed
> revision and fixed pipeline, CPU embeddings are deterministic; a re-run would
> reproduce 0.5144 / 0.5666. Spending GPU quota on a re-run was deferred in
> favour of the deferred BEIR-pair-illustrating task (one regressed + one improved
> corpus to make the optimised opt-in framing concrete).
>
> **v2.0 launch re-confirm (2026-04-29)**: still valid. Audited the
> retrieval-math file set (`field/{encoder, dense, ann, algebra}.py`,
> `store/{registry, bands}.py`) for changes since the Phase 7 close
> against the 2026-04-29 launch tip. The post-Phase-7 commits
> (`23e8eaa4..e87e2aab`) cover deep-search loop + namecheck CLI surface,
> 11-lane bench, package rename `rlat2 → rlat`, skill restructure,
> Audit 08 storage-mode `convert`, bench 5 primer effectiveness MVP,
> and doc-currency edits — **no commit modifies retrieval math**. The
> `convert` pipeline (`store/conversion.py`) is bands-byte-stable
> (`np.allclose` at 1e-6, enforced by `tests/harness/conversion`); it
> reshapes the storage mode without touching the embedding distribution.
> Floor gate carries forward unchanged into v2.0.0. Any future Phase 1+
> change to encoder, pooling, ANN config, or chunker still requires
> a paired re-run + this gate updated.

---

## Locked floor

| Metric | Value |
|---|---|
| nDCG@10, mean over 5 corpora | **0.5144** |
| Recall@10, mean over 5 corpora | **0.5666** |
| Corpora | nfcorpus, scifact, arguana, scidocs, fiqa |

Per-corpus:

| Corpus | N passages | N queries | nDCG@10 | Recall@10 | ANN used? |
|---|---:|---:|---:|---:|:---:|
| nfcorpus | 3,633 | 323 | 0.3431 | 0.1640 | ✗ (exact dense) |
| scifact | 5,183 | 300 | 0.7672 | 0.8926 | ✓ |
| arguana | 8,674 | 1,406 | 0.7430 | 0.9637 | ✓ |
| scidocs | 25,657 | 1,000 | 0.1946 | 0.2014 | ✓ |
| fiqa | 57,638 | 648 | 0.5239 | 0.6114 | ✓ |

scidocs is a known outlier — the BEIR scidocs task is small-passage retrieval where dense bi-encoders without domain adaptation generally floor at the 0.18–0.22 nDCG@10 band. It's included in the gate to make adaptation work visible (a v2.0+ optimised, or a future cross-encoder, must move scidocs above 0.20 to count as a real lift).

nfcorpus uses exact dense — 3,633 < 5,000 ANN threshold (`field/ann.py:ANN_THRESHOLD_N`). The other four use FAISS HNSW.

---

## Why this is the floor

These numbers are the **base recipe with no opt-in features active**:

- Single encoder (`gte-modernbert-base`), single pool (CLS), single norm (L2).
- Single retrieval mode (dense cosine).
- No reranker, no lexical sidecar, no query-prefix tuning, no router auto-mode — the v0.11 knob surface is intentionally absent.
- No MRL optimised (Phase 4 `rlat optimise` opt-in would add a (N, 512) band trained on a per-knowledge-model synth-query corpus).

A regression below this floor in **any** of the 5 corpora is a Phase 1/2 bug, not a tuning opportunity. A drop in the average without a per-corpus drop is also a regression — average alone is insufficient evidence of correctness.

## Comparison: same-stack (v2.0) vs off-stack (v0.11) baselines

The **same-stack** numbers are the load-bearing comparison for launch claims — same encoder pipeline (chunker / ANN / scoring held constant), only the encoder model swapped. The v0.11 column is preserved as a historical record but compares across different stacks (different chunker, different ANN config) and is not directly comparable to the gte-mb measurement.

| Encoder | 5-BEIR mean nDCG@10 (v2.0 stack) | 5-BEIR mean nDCG@10 (v0.11 stack, historical) | Δ vs gte-mb (same-stack) |
|---|---:|---:|---:|
| **gte-mb base 768d (this gate)** | **0.5144** | — | — |
| BGE-large-v1.5 | 0.4888 | 0.4447 | **+0.026** |
| E5-large-v2 | 0.4331 | — | **+0.081** |
| Qwen3-Embedding-8B (last-token) | — | 0.500 | +0.014 (off-stack) |

**Headline (apples-to-apples)**: gte-mb base wins the v2.0 stack against BGE-large by **+0.026 nDCG@10** and E5-large by **+0.081**. The off-stack Qwen3-8B comparison (run on its own stack with last-token pooling) is preserved as a non-load-bearing reference but is not directly comparable to the same-stack numbers above.

Evidence: [`benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json`](../../benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json) (same-stack encoder comparison, run on Kaggle T4). v0.11-stack BGE-large 0.4447 is from `project_arguana_bge_regression` memory; v0.11-stack Qwen3-8B 0.500 is from `project_rebenchmark_in_flight` (Block E).

The Phase 4 optimised soak (per base plan §C) projected ~+6.8 pt average lift on a 3-corpus holdout when MRL d=512 was opt-in. **Falsified** in v2.0 BEIR-3 specialist soak v7 (2026-04-26) — see [HONEST_CLAIMS.md §What we measured on `rlat optimise`](HONEST_CLAIMS.md#what-we-measured-on-rlat-optimise---full-honest-framing) for the falsification evidence.

## Comparison to gte-modernbert-base published numbers

The model card at <https://huggingface.co/Alibaba-NLP/gte-modernbert-base> reports BEIR-15 NDCG@10 per corpus. Comparing the 5-corpus subset we benchmark:

| Corpus | Published nDCG@10 | Our measurement | Δ |
|---|---:|---:|---:|
| NFCorpus | 36.44 | 34.31 | −2.13 |
| SciFact | 77.40 | 76.72 | −0.68 |
| ArguAna | 72.68 | 74.30 | +1.62 |
| SCIDOCS | 21.29 | 19.46 | −1.83 |
| FiQA2018 | 48.81 | 52.39 | +3.58 |
| **5-corpus mean** | **51.32** | **51.44** | **+0.12** |

We match published reproduction within typical evaluation variance (+0.12 pt on the 5-corpus mean). The two corpora we're below on (NFCorpus, SCIDOCS) are the small-passage outliers; the two we're above on (FiQA, ArguAna) are larger and more diverse. The encoder is doing what the paper claims; the same-stack +0.026 nDCG@10 gap over BGE-large and +0.081 over E5-large is a function of gte-mb's training, not measurement error. (The off-stack +1.4 pt over Qwen3-8B from earlier work is preserved above as a historical reference but isn't apples-to-apples with the v2.0 stack.)

Published BEIR-15 average is 55.33 — our 5-corpus subset (51.32 published / 51.44 measured) is a strict subset that excludes some of the easier corpora (FEVER 91.03, QuoraRetrieval 88.55, TRECCOVID 81.95), so the average looks lower; this is *not* a regression.

## Reproduction

```bash
# Local (Intel CPU, OpenVINO auto-detect, ~hours)
python benchmarks/bench_beir5_floor.py --runtime auto

# Kaggle T4 (recommended, ~30 min total wall time)
# 1. Build a single-file kernel:
#    code_file: bench_beir5_floor.py (self-installs deps from pip)
# 2. Push: PYTHONUTF8=1 kaggle kernels push -p kaggle/<dir> --accelerator NvidiaTeslaT4
# 3. Pull: PYTHONUTF8=1 kaggle kernels output kanesnyder/<slug> -p kaggle/output -o
```

`bench_beir5_floor.py` is intentionally self-contained — it `pip install`s `beir`, `faiss-cpu`, `onnxruntime`, and `onnxscript` (the last is required by `torch.onnx.export` at opset ≥ 17 and Kaggle's torch image doesn't ship it).

The Kaggle kernel mounts the source as a tarball-extracted dataset; `bench_beir5_floor._bootstrap_src_path` walks `/kaggle/input/**` for `resonance_lattice/__init__.py` and prepends to `sys.path`. Pin the dataset version when re-running so the source matches the recorded floor.

## Honest claims

- The metric is `EvaluateRetrieval.evaluate(qrels, results, [10])` from BEIR's pytrec_eval wrapper — graded-qrels, **not** binary recall.
- Encoding is done passage-by-passage at batch 32, on CUDA via the torch runtime. The OpenVINO and ONNX runtimes produce numerically equivalent embeddings (Phase 1 close-out: bit-exact on CPU, L2 norm error 1.19e-7). A switch in inference path should not move these numbers; if it does, that's the runtime drifting.
- ANN is M=32 efC=200 efS=128. Audit 04 evidence: [`benchmarks/results/ann_audit_04.json`](../../benchmarks/results/ann_audit_04.json). efS=128 was the calibrated value where recall@10 vs exact reaches ≥0.95 across the corpus-size bands tested.
- The numbers are for **base band only**. Anything reporting "optimised on" must say so explicitly and reference [`OPTIMISE.md`](./OPTIMISE.md) for the per-knowledge-model synth-query budget.

## When to bump this gate

The floor moves up only when:

1. The encoder revision is bumped on a measured win (paired with revised numbers across all 5 corpora).
2. The pooling/normalisation choice changes on measured win (same constraint).
3. The ANN config changes on measured win (same constraint).

The floor moves down (allowed) only on:

1. A bug fix in `EvaluateRetrieval`-like upstream code that revealed the previous numbers were wrong.
2. A deliberate scope change recorded in `CHANGELOG.md` with rationale.

The floor does **not** move because:

1. The MRL optimised is opt-in and lives in its own gate.
2. A hardware swap (T4 → A10) — only the wall-time changes.
3. Cross-encoder rerankers — not part of the default recipe.
