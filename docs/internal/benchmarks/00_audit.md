# Benchmark suite — Phase 0 audit

> Pre-implementation audit, dated 2026-04-27, for the user-facing benchmark suite plan.
> Authoritative reference for what already exists, what's reusable, and the rules every
> new benchmark in `benchmarks/user_bench/` must follow. Read before writing any
> `run.py` in this suite.

## Summary

The repo already carries a substantial measurement surface from v0.11 and the early v2 rebuild: BEIR loaders, optimise pipeline, LoCoMo + LongMemEval harnesses, and a body of locked numbers we can cite. Three things change the plan as written:

1. **`benchmarks/` is not a Python package** — no `benchmarks/__init__.py`. The plan's `python -m benchmarks.user_bench.<n>.run` invocation requires we add `__init__.py` at three levels (`benchmarks/`, `benchmarks/user_bench/`, `benchmarks/user_bench/<n>/`). Phase 6 deliverable.
2. **`benchmarks/results/optimised/fabric_probe_v1.json` was missing on disk** at audit time despite being referenced from `docs/user/OPTIMISE.md`. **Resolved 2026-04-27**: pulled from the Kaggle kernel `kanesnyder/rlat2-fabric-specialist-probe-v1` and committed to the canonical path. Numbers verified end-to-end: 62,953 passages, base R@5 0.871, optimised R@5 0.903 (best dim 256, tied with 512), Δ +0.032, cost $8.16, train_dev_r1 0.578, encoded in 382 s. Backstop for the optimise three-row table is now traceable.
3. **All Kaggle kernel slugs retain pre-rename `specialist`** (e.g. `kanesnyder/rlat2-beir-3-specialist-soak-v7`, `kanesnyder/rlat2-fabric-specialist-probe-v1`). Slugs are immutable; CLAUDE.md / REBUILD_PLAN already track this. Audit logs that reference Kaggle artefacts must use the `specialist`-form slug.

## Reusable scripts and modules

### Direct adapter targets

- **`benchmarks/bench_beir3_optimised_soak.py`** (585 lines) — full optimise pipeline (synth queries → train → multi-MRL-slice eval). **Direct template for benchmark 7 (BEIR fiqa optimise probe)**: the corpus loader is the only swap from {nfcorpus, scifact, arguana} to {fiqa}; everything else (locked hparams, synth cache, training loop, slice eval, JSON output schema) is reusable.
- **`benchmarks/bench_beir5_floor.py`** (276 lines) — BEIR loader (`_load_beir`), corpus encoder, query encoder, evaluation wrapper (`EvaluateRetrieval`), and the `_bootstrap_src_path()` Kaggle/local switcher. **Reusable across benchmarks 6 (build/query speed) and 7 (fiqa optimise) for the BEIR side**.
- **`benchmarks/bench_longmemeval.py`** (1073 lines) — LongMemEval 500-question runner with retrieval-recall metrics. **Wrap with an LLM-answering layer for benchmark 3**.
- **`benchmarks/bench_locomo.py`** (1440 lines) — LoCoMo multi-session conversational memory eval, Sonnet-judge integrated. **Reusable for benchmark 3 multi-session resume scoring**.
- **`benchmarks/bench_fabric_optimised_probe.py`** (442 lines) — closest existing recipe to the optimise three-row table's Fabric row. Validate it can re-emit the +0.032 R@5 result if the JSON's missing.

### Encoder + retrieval programmatic surfaces

```python
from resonance_lattice.field.encoder import Encoder
enc = Encoder(runtime="auto")              # auto-picks openvino (Intel) / onnx
embs = enc.encode(["q1", "q2"])            # (N, 768) float32 L2-normed
embs = enc.encode_batched(texts, batch_size=32)
```

`runtime="torch"` for build/optimise paths (requires `[build]` extra).

Cold-call ~942 ms; warm ~12 ms (`docs/internal/FIELD.md` + `docs/user/SKILLS.md`). Benchmark 6 measures these directly.

### LLM judge — API key + client

```python
from resonance_lattice.optimise.synth_queries import discover_api_key, default_client
key = discover_api_key()                   # RLAT_LLM_API_KEY_ENV → CLAUDE_API → ANTHROPIC_API_KEY → kaggle_secrets
client = default_client(key, model="claude-sonnet-4-6")
```

`discover_api_key()` returns `str | None`. Per `feedback_api_key.md`: the `CLAUDE_API` env var carries leading/trailing newlines that cause `httpcore.LocalProtocolError`; the resolver `.strip()`s. Reuse don't reimplement.

### Skill-context CLI surface (benchmarks 1, 4)

`rlat skill-context` flags: `--query` (repeat), `--top-k`, `--token-budget`, `--source-root`, `--strict`, `--mode {augment|knowledge|constrain}`. Output is markdown with HTML-comment headers carrying `top1_top2_gap`, `source_diversity`, `drift_fraction`. **No Python API** — invoke via subprocess and parse stdout. Token budget drops later queries first; strict mode exits with code 2 on drift.

### Search CLI surface (benchmarks 1, 2)

`rlat search` flags: `--query` (single, not repeat), `--top-k`, `--format {text|json|context}`, `--source-root`, `--verified-only`, `--mode {augment|knowledge|constrain}`. JSON format is the structured-result path (benchmark harnesses parse this); context format is the markdown-for-LLM path; text format is human-readable.

## Existing result artefacts (load-bearing only)

| Path | Headline number | Referenced from |
|---|---|---|
| `benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json` | BEIR-5 floor 0.5144 nDCG@10 / 0.5666 R@10 | BENCHMARK_GATE.md |
| `benchmarks/results/beir/new_arch/v2_specialist_soak_3beir.json` | BEIR-3 optimise regression −0.040 mean | OPTIMISE.md, HONEST_CLAIMS.md |
| `benchmarks/results/locomo/phase2_cartridge_e5_all10.json` | LoCoMo cartridge baseline 68.2% | `project_locomo_feature_transfer.md` |
| `benchmarks/results/locomo/phase2_cartridge_e5_all10_sonnet_judge.json` | LoCoMo Sonnet-judge sibling | `project_locomo_feature_transfer.md` |
| `benchmarks/results/longmemeval/v14_full500_800.json` | LongMemEval v14 R@5 0.924 / MRR 0.919 | `project_longmemeval_ku_filter.md` |
| `benchmarks/results/longmemeval/runpod_v15_bge/longmemeval_v15_bge_full500.json` | v15 BGE full-500 baseline | `project_arguana_bge_regression.md` |
| `benchmarks/results/ann_audit_04.json` | FAISS HNSW efS=128 lock | BENCHMARK_GATE.md |
| `benchmarks/results/optimised/fabric_probe_v1.json` | Fabric +0.032 R@5 (Fabric optimised lift) | OPTIMISE.md, README.md, BENCHMARKS.md plan |

The optimised three-row table is no longer blocked — fabric_probe_v1.json is committed at the published path with verified numbers.

## Locked baseline numbers (referenceable from BENCHMARKS.md)

> **Note on stack provenance.** The "comparison row" entries below for BGE-large (0.4447), Qwen3-8B (0.500), and E5-large (0.5351) are off-stack historical measurements (each on its own native pipeline). The **load-bearing same-stack comparison** for v2.0 launch claims lives in `benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json` (BGE-large 0.4888 / E5-large 0.4331 on the v2.0 chunker + ANN + scoring) and is published in [`docs/user/BENCHMARKS.md`](../../user/BENCHMARKS.md#retrieval-quality-beir-5-floor--encoder-comparison) and [`docs/internal/BENCHMARK_GATE.md`](../BENCHMARK_GATE.md). The off-stack rows below are preserved as the historical record that motivated the same-stack rerun.

| Recipe | Metric | Value | Notes |
|---|---|---|---|
| **gte-modernbert-base 768d (v2.0 base)** | BEIR-5 mean nDCG@10 | **0.5144** | locked floor |
| **gte-modernbert-base 768d (v2.0 base)** | BEIR-5 mean R@10 | **0.5666** | locked floor |
| **gte-modernbert-base 768d (v2.0 base)** | BEIR fiqa nDCG@10 | **0.5239** | will be the base column for benchmark 7 |
| **gte-modernbert-base 768d (v2.0 base)** | BEIR fiqa R@10 | **0.6114** | will be the base column for benchmark 7 |
| Qwen3-8B last-pool | BEIR-5 mean nDCG@10 | 0.500 | off-stack comparison row (historical) |
| BGE-large-en-v1.5 full-stack | BEIR-5 mean nDCG@10 | 0.4447 | off-stack comparison row (historical — v2.0 same-stack: 0.4888) |
| E5-large field-only | BEIR-5 mean nDCG@10 | 0.5351 | off-stack comparison row (historical — v2.0 same-stack: 0.4331) |
| LongMemEval v14 (BGE) | R@5 / MRR | 0.924 / 0.919 | retrieval recall, NOT task accuracy |
| LoCoMo cartridge-only | aggregate | 68.2% | LLM-judged ship number |
| Fabric optimised d=512 | R@5 | 0.903 | vs base 0.871, +0.032 — see `benchmarks/results/optimised/fabric_probe_v1.json` |
| BEIR-3 optimised d=512 (locked v2.0) | mean nDCG@10 | 0.300 | vs base 0.343, −0.043 |

## Falsified hypotheses (for BENCHMARKS.md "what we measured and didn't ship")

Each one ships in the user-facing doc as a 1-line claim + 1-line measurement + memory-entry link. Audit-confirmed list:

1. **EML retrieval-time scoring/fusion** — falsified 3× (E2 research, bench_eml_scifact, ablate_eml_cluster). `exp(α·x)` numerically dominates `−ln(y)`; collapses to exp_only or NaN. Source: `project_eml_retrieval_falsified.md`.
2. **Cross-encoder rerankers on strong-dense** — bge-reranker-v2-m3, mxbai-rerank-base-v1 regress Qwen3-8B on 4/5 BEIR; FiQA −0.062. Source: `project_ce_rerankers_hurt_strong_dense.md`.
3. **MRL +6.8 pt projection on BEIR** — base plan §C projection falsified by locked v2.0 hparams on nfcorpus −0.043 nDCG@10. Source: `project_specialist_beir3_falsified.md`.
4. **BGE-large-en-v1.5 default upgrade** — failed Gate 1 (0.4447 < 0.46). ArguAna catastrophic regression −9.7 pts. Source: `project_arguana_bge_regression.md`.
5. **Post-hoc session-vector filter (LongMemEval v16)** — inert; ranking-time damage can't be fixed at output-time. Source: `project_longmemeval_ku_filter.md`.
6. **LoCoMo "full-stack" features** — subgraph −30pt catastrophic, diversify −3.3pt, CE neutral. Cartridge baseline 68.2% is the ship recipe. Source: `project_locomo_feature_transfer.md`.
7. **Qwen3 mean pooling** — collapsed FiQA 7×; correct pooling is last-token. Source: `project_qwen3_last_pooling.md`.
8. **Stage 1 MS MARCO fine-tune** — regressed 4/5 BEIR avg −8.3pt; failed Gate 1. Source: `project_stage1_fine_tune_negative.md`.
9. **Lexical band V1 parity** — FAIL on 4/5 BEIR; FiQA −7.8pt. Stays as `--lexical-impl auto` fallback. Source: `project_lexical_band_v1_parity.md`.
10. **Structural rerank (opportunity #5)** — dead on BEIR prose; SciFact 0.6655→0.6266. Source: `project_structural_rerank_scifact.md`.
11. **Trained heads** — 0-for-10. Source: multiple memory entries.
12. **Multi-vector fields** — catastrophic. Source: `feedback_first_principles.md` + memory.
13. **Asymmetric dense fields** — −0.8%. Source: `project_asymmetric_field.md`.
14. **B3 ripgrep hybrid on short-prose** — 56% recovery but still −0.011 vs dense; disabled by default for short-prose corpora. Source: `project_b3_hybrid_corpus_sensitivity.md`.

## Hard rules every `benchmarks/user_bench/<n>/run.py` must follow

1. **Pre-research before designing.** Audit `benchmarks/results/` and prior memory entries. (Source: `feedback_review_benchmarks_first.md`, `feedback_benchmark_preresearch.md`.)
2. **Always Kaggle T4, never P100** — pass `--accelerator NvidiaTeslaT4` on push. P100 sm_60 fails with `cudaErrorNoKernelImageForDevice` 57s in. (Source: `feedback_kaggle_skill.md`.)
3. **`PYTHONUTF8=1` on Windows for every Kaggle CLI call** — without it, charmap decode crashes silently drop pushes. (Source: `feedback_kaggle_cli_utf8.md`.)
4. **Synth-query cache scoped to corpus name, not global** — cross-corpus reuse collapses training because `passage_idx` is position-specific. (Source: `feedback_synth_queries_corpus_scoped.md`.)
5. **Commit before RunPod uploads** — orchestrators upload via `git archive HEAD | tar`; working-tree edits silently dropped. (Source: `feedback_runpod_uploads_need_commit.md`.)
6. **Determine measurement surface first, LLM-judge later** — gate runs use deterministic search metrics; LLM-augmentation comes as a separate, later extension. (Source: `feedback_benchmark_gate.md`.)
7. **First-principles, not roadmap alignment** — defend novel claims as breaking structural assumptions; existing roadmap items have a poor track record (0-for-10 trained heads, multi-vector catastrophic). (Source: `feedback_first_principles.md`.)
8. **Inter-rater 10% sample for every LLM-judged metric** — hand-validate, report agreement, commit the judge prompt. (Plan-level rule.)
9. **Per-`run.py` `--budget-usd <cap>` flag** — print running total, abort on cap. (Plan-level rule.)
10. **Test sets committed in full** — auditability over compactness. (Plan-level rule.)

## Methodology gotchas (per-bench notes)

- **Benchmark 1, 4 — capture skill-context output via subprocess**, parse the HTML-comment headers (`top1_top2_gap`, `source_diversity`, `drift_fraction`) for ConfidenceMetrics. No Python API exists yet.
- **Benchmark 2 — drifted-subset construction**: post-build mutate source files; record original + mutated content for inter-rater verification. The drift signal is `[verified|drifted|missing]` in skill-context output.
- **Benchmark 3 — LongMemEval task accuracy ≠ retrieval R@5/MRR**. Existing `bench_longmemeval.py` measures recall; benchmark 3 must wrap retrieval with an LLM-answer layer + correct/incorrect rubric.
- **Benchmark 3 — recency tier policy and `prefer_recent` post-processing are mutually exclusive** (LayeredMemory recency bins double-count on knowledge-update queries). Pick one. (Source: `project_longmemeval_ku_filter.md`.)
- **Benchmark 6 — sparsify_mode is baked into encoder at build time**; A/B comparisons require rebuild. Default settings only — don't try to flip flags at eval time.
- **Benchmark 7 — corpus-aware synth anchors** for the optimise pipeline. Fabric-hardcoded anchors biased Claude toward conceptual prose on PowerShell corpus (cmdlet-heavy); v4→v5 flipped FAIL→PASS with `derive_style_anchors()` per-corpus. fiqa is closer to Fabric's natural-language-prose distribution but verify the anchor set isn't hardcoded for Fabric. (Source: `project_mrl_specialist_encoder.md`.)
- **Benchmark 7 — corpus-scoped synth cache fingerprint** — the existing optimise pipeline already does this correctly; just verify the fiqa run produces a different cache key than nfcorpus.

## Decisions surfaced from this audit

1. **Adopt `python -m benchmarks.user_bench.<n>.run`** as the invocation pattern. Add three `__init__.py` files in Phase 6.
2. **Fabric optimised result** — recovered from Kaggle and committed (no Stage 1 budget spent). Backstop traceable end-to-end.
3. **Benchmark 7 corpus-aware anchors** — verify fiqa run regenerates anchors via `derive_style_anchors()` before training. Add a 5-minute pre-run check to `run.py`.
4. **Locked Sonnet 4.6 judge** — confirmed available via `default_client(key, model="claude-sonnet-4-6")`; existing pattern, no rework.
5. **No new BEIR-loader code** — reuse `_load_beir` + `_encode_corpus` + `_bench_corpus` from `bench_beir5_floor.py` for benchmarks 6, 7.

## Audit close

This audit covers everything called for in plan §"Pre-implementation audit (Phase 0)". Three follow-up audits will land as separate methodology docs:

- `01_token_usage.md` — once the 50-task corpus is hand-written.
- `02_hallucination.md` — once the 100-question test set is hand-written + the drifted-subset protocol is locked.
- `07_optimise_beir_fiqa.md` — once `bench_beir3_optimised_soak.py` is forked + verified for fiqa.

Stage 1 begins after this audit lands.
