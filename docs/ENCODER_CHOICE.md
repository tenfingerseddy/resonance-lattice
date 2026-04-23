---
title: Encoder Choice — E5 / BGE / Qwen3-8B
status: shipping
decided: 2026-04-22
board_items: [237, 239, 240]
---

# Encoder Choice

Resonance Lattice ships three well-supported encoders. None is universally best — each wins on a different workload shape. This page gives you the data and a decision guide so you can pick the right one for your corpus.

If you're not sure, **BGE-large-en-v1.5** is the starting-point default. It's a solid middle-of-the-road choice that works on CPU / Intel Arc iGPU and has strong ecosystem support (ONNX / OpenVINO / sentence-transformers). If your workload is counter-argument retrieval, switch to E5. If you have a 16 GB GPU and want frontier quality, use Qwen3-8B — it wins 4 of 5 BEIR corpora (0.5005 5-BEIR best-mode avg vs BGE's 0.4447, E5's 0.4552), but note that the shipped cross-encoder rerankers regress it on 4/5 corpora so `__retrieval_config__` correctly routes Qwen3 builds to dense-only by default.

## TL;DR — which encoder, when

| If your corpus is… | Use | Why |
|---|---|---|
| General QA, science, docs, code | `bge-large-en-v1.5` (default) | Strong on QA-style queries, best ecosystem, fastest CPU inference |
| Counter-argument / debate retrieval (ArguAna-like) | `e5-large-v2` | BGE regresses 9.7 pts on ArguAna; E5 preserves argumentative structure |
| You have a 16 GB GPU and want SOTA quality | `qwen3-8b` | +5.6 pt lift on 5-BEIR best-mode avg (0.5005 vs BGE 0.4447); wins 4/5 corpora |
| You're deploying to Intel Arc / NPU / strict CPU-only | `bge-large-en-v1.5` | Well-validated OpenVINO path |
| You built cartridges before 2026-04-20 | whatever is stamped in the cartridge | Encoder is restored on load; no action needed |

## Full 5-BEIR comparison (best-mode nDCG@10)

Best-mode = the winning `(probe_mode, reranker)` combination for that corpus, as picked by the build-time probe infra (board items 236c / 238).

| Corpus | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| NFCorpus | 0.38217 | **0.39246** | 0.41381 |
| SciFact | 0.74664 | 0.75538 | **0.77615** |
| ArguAna | **0.50478** | 0.40795 | 0.46616 |
| SciDocs | 0.20276 | 0.21374 | **0.27823** |
| FiQA | 0.43782 | 0.45407 | **0.56835** |
| **5-BEIR avg** | 0.45523 | 0.44472 | **0.50054** |
| **v1.0.0 gate (≥ 0.46)** | ❌ −0.005 | ❌ −0.015 | ✅ +0.040 |

Bold = best encoder per corpus. Qwen3-8B wins 4/5. E5 owns ArguAna decisively. BGE is the rational middle — small wins on NFCorpus, loses elsewhere.

**Measurement note**: all three encoders were probed across the full grid (`field_only / plus_cross_encoder / plus_cross_encoder_expanded / plus_full_stack` × both `bge-reranker-v2-m3` and `mxbai-rerank-base-v1`). For Qwen3-8B, `field_only` won on 4/5 corpora; SciFact is the only corpus where the cross-encoder helped (and only marginally, +0.002 over dense). See "Per-corpus cross-mode data" below for the Qwen3-8B table.

## Per-corpus cross-mode data

### NFCorpus (medical abstracts, QA)

| Mode | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| `field_only` | 0.37387 | 0.38187 | **0.41381** |
| `plus_cross_encoder` | 0.38217 *(mxbai)* | **0.39246** *(mxbai)* | 0.41305 |
| `plus_full_stack` | 0.35531 | 0.37591 | 0.38517 |

Medical QA: cross-encoder rerank helps E5 and BGE (mxbai beats bge-v2-m3 on this corpus). On Qwen3-8B, field_only wins — the reranker adds nothing and full-stack hurts.

### SciFact (scientific claim verification)

| Mode | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| `field_only` | 0.70623 | 0.73178 | 0.77380 |
| `plus_cross_encoder` | 0.74046 | 0.75538 | **0.77615** |
| `plus_full_stack` | 0.74664 | 0.74826 | 0.75013 |

Science claims benefit from rerank across all three encoders. For Qwen3-8B, the CE lift is marginal (+0.002) — scifact is the only corpus where Qwen3 + reranker beats Qwen3 dense.

### ArguAna (counter-argument retrieval) — the regression

| Mode | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| `field_only` | 0.41459 | 0.39191 | **0.46616** |
| `plus_cross_encoder` | 0.40620 | 0.39162 | 0.46120 |
| `plus_full_stack` | **0.50478** | 0.40795 | 0.41155 |

**BGE regresses 9.7 pts vs E5 on best-mode** (0.408 vs 0.505) — the largest single-corpus gap on the portable tier. E5's full-stack win on ArguAna is unique: its training captured argumentative inversion, and the BM25 + reranker lift is large enough to overtake Qwen3-8B. If counter-argument retrieval is your workload, `--encoder e5-large-v2` still wins.

### SciDocs (citation prediction)

| Mode | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| `field_only` | 0.20276 | **0.21374** | **0.27823** |
| `plus_cross_encoder` | 0.19243 | 0.20627 | 0.22825 |
| `plus_full_stack` | 0.18044 | 0.18314 | 0.18875 |

**Cross-encoders actively hurt on SciDocs for all three encoders.** CE rerankers are trained on QA-style pairs; citation prediction is a different pair distribution. Auto-routing correctly picks `field_only` for all three.

### FiQA (finance QA)

| Mode | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| `field_only` | 0.41745 | 0.44168 | **0.56835** |
| `plus_cross_encoder` | 0.43782 | **0.45407** | 0.50615 |
| `plus_full_stack` | 0.43694 | 0.43610 | 0.44617 |

Domain QA: E5/BGE both benefit from reranking (BGE best-mode +CE). Qwen3-8B's dense retrieval is strong enough that CE rerank regresses 6 pts (0.568 → 0.506), and full-stack regresses 12 pts — the starkest rerank-mismatch case in the suite.

### Why the pattern flips for Qwen3-8B

Both shipped rerankers (`bge-reranker-v2-m3`, `mxbai-rerank-base-v1`) were trained atop weaker base retrievers producing 0.3–0.4 nDCG@10 top-k. Qwen3-8B already produces 0.5+ top-k — the reranker sees an input distribution mismatched with its training data, and the extra cross-attention reshuffles a mostly-correct top-k based on the wrong signal. Full-stack also adds BM25 fusion, which amplifies the harm on specialized-vocabulary domains (finance jargon, citation networks). **A Qwen3-matched reranker (e.g. `Qwen/Qwen3-Reranker-*`) is the tier's remaining product gap.**

## Deployment profile

| Dimension | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| Parameters | 335 M | 335 M | 8 B |
| Size (bf16 / fp16) | 0.7 GB | 0.7 GB | 16 GB |
| Min practical inference | CPU | CPU / Intel Arc iGPU | A100-class GPU (16 GB VRAM min) |
| Query latency (warm, 1 query) | ~30 ms CPU / ~10 ms Arc | ~25 ms CPU / ~8 ms Arc | ~60 ms A100 |
| Ecosystem | sentence-transformers, ONNX, OpenVINO | **best** (sentence-transformers, ONNX, OpenVINO, broad fine-tunes) | transformers; limited ONNX/OpenVINO |
| License | MIT | MIT | Apache-2.0 |
| Default pooling | `mean` | `cls` | **`last`** (required — see caveat) |

Resonance Lattice stamps the encoder into every cartridge's `__encoder__` block. Loading a cartridge restores the encoder preset (pooling, prefixes, max-length) automatically — you cannot accidentally query a BGE cartridge with E5.

## How to opt in / out

```bash
# Starting-point default (BGE-large-en-v1.5)
rlat build ./docs ./src -o project.rlat

# Explicit E5 (use for counter-argument / debate corpora)
rlat build ./docs ./src -o project.rlat --encoder e5-large-v2

# Qwen3-Embedding-8B (needs 16 GB GPU)
rlat build ./docs ./src -o project.rlat --encoder qwen3-8b

# Legacy cartridges keep working — the encoder is stamped at build time
rlat search legacy-e5-cartridge.rlat "how does auth work?"   # auto-loads E5
```

## Market position (same 5 BEIR corpora, best-mode avg)

| Anchor | 5-BEIR avg |
|---|---|
| BEIR BM25 (2021) | 0.340 |
| BEIR BM25+CE (2021) | 0.372 |
| Cohere-embed-mlv3 (2024) | 0.450 |
| **Ours — E5-large-v2 (full stack)** | **0.455** |
| jina-v3 (2024) | 0.461 |
| mE5-large-instruct (2024) | 0.464 |
| **Ours — BGE-large-en-v1.5 (full stack)** | **0.445** |
| **Ours — Qwen3-Embedding-8B (field_only)** | **0.500** |
| text-embedding-3-large (OpenAI, 2024) | 0.512 |
| Qwen3-Embedding-8B (published, different BEIR subset) | ~0.55 |
| NV-Embed-v2 (2024, CC BY-NC) | 0.620 |

**Position**: our portable tiers (E5, BGE) sit with 2024 mid-tier open dense retrievers. Qwen3-8B is frontier-adjacent on dense alone; still a ~1 pt gap to `text-embedding-3-large`, and another 5-10 pts below the published NV-Embed / Gemini-Embedding frontier.

Sources: [BEIR paper](https://arxiv.org/abs/2104.08663), [jina-embeddings-v3 paper](https://arxiv.org/abs/2409.10173), [Qwen3-Embedding card](https://huggingface.co/Qwen/Qwen3-Embedding-8B), our aggregates in [benchmarks/results/beir/new_arch/](../benchmarks/results/beir/new_arch/).

## Technical notes

### Qwen3-Embedding pooling is load-bearing

Qwen3-Embedding is a decoder-only LM. It **requires last-non-padding-token pooling** — the final position is the only one with full left-context. Mean or CLS pooling discards that; a sweep with `"pooling": "mean"` collapsed the 5-BEIR avg to 0.250 (FiQA dropped 7× from 0.568 to 0.092). The `qwen3-*` entries in `ENCODER_PRESETS` set `"pooling": "last"` — don't edit without re-benchmarking. Evidence archived at [benchmarks/results/beir/new_arch/baseline_qwen3_8b_v1_meanpool_broken/](../benchmarks/results/beir/new_arch/baseline_qwen3_8b_v1_meanpool_broken/).

### Build-time probe picks the right mode automatically

`rlat build` with `--probe-qrels` and `--probe-queries` (board item 236c) runs the mode × reranker sweep over held-out labeled queries and writes the winner to `__retrieval_config__` inside the cartridge. At query time, `rlat search` reads that config and routes automatically. You get per-corpus best-mode for free — no manual tuning, no need to memorize which reranker works where.

See [docs/CLI.md](CLI.md#rlat-build) for the probe flags.

### Encoder stamping + consistency check

Every cartridge carries `__encoder__` metadata (name, pooling, prefixes, max_length). `Lattice.load` restores the encoder from this stamp. `_check_encoder_consistency` blocks loads where the caller tries to force a different encoder — you can't accidentally mix encoders in a single query path.

### First-build download

`rlat build` downloads the encoder weights from Hugging Face on first invocation:
- `BAAI/bge-large-en-v1.5` — ~1.3 GB
- `intfloat/e5-large-v2` — ~1.3 GB
- `Qwen/Qwen3-Embedding-8B` — ~16 GB

After the first build, subsequent builds and queries are fully local. Pair with `--onnx` or `--openvino` on Intel Arc for accelerated inference. For `Qwen3-Embedding-8B`, a ~16 GB GPU is required for practical latency.

### Memory + lens default

Secondary code paths (`rlat memory save` with no existing lattice, `rlat lens build --topics`) also default to BGE-large-en-v1.5 for consistency with the primary build path.

## What's still open

1. **Qwen3-8B full-stack probe**. Currently only `field_only` measured. Cross-encoder rerankers in the shipped set (bge-reranker-v2-m3, mxbai-rerank-base-v1) are trained on weaker base retrievers and are expected to regress Qwen3-8B; a Qwen3-Reranker pairing is tracked under a follow-up board item.
2. **Per-corpus encoder routing**. Projected BGE-elsewhere + E5-on-ArguAna would average 0.464 — clears the v1.0.0 gate. Not yet implemented; would require a per-encoder build and a per-corpus routing layer. Filed as a v1.1.0 consideration.
3. **Encoder fine-tune at 335M**. Closing the ~6-pt gap to text-embedding-3-large is tractable with distillation or task-aware fine-tune (expected 2-8 weeks, $50-300 compute). Tracked under board item 235.

## Changelog

- **2026-04-22** — Rewrite. Three-encoder comparison (E5 / BGE / Qwen3-8B) with full 5-BEIR best-mode table. Item 239 launch verification sweep landed BGE full 5-BEIR numbers; reveals -9.7 pt ArguAna regression vs E5 and net -1 pt on 5-BEIR avg. BGE framed as starting-point default with explicit decision guidance.
- 2026-04-20 — Initial BGE default flip (item 237) based on 2-corpus pilot (NFCorpus + SciFact only).
