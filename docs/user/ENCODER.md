# The Encoder

`rlat` ships with **one encoder.** No options, no presets, no decision tree.

## What it is

| | |
|---|---|
| Backbone | `Alibaba-NLP/gte-modernbert-base` |
| Pooling | CLS |
| Output dimension | 768 |
| Max sequence length | 8192 tokens |
| Normalisation | L2 |
| Revision | pinned at install time (recorded in every knowledge model) |

Every knowledge model carries the embeddings produced by this encoder in its `base` band. Cross-model search and `rlat compare` always operate on these vectors, which is why the recipe is locked.

## Installation

The encoder is **not** included in the pip package — it's downloaded and converted on first use. You install it once per machine.

### Automatic (recommended)

The first time you run `rlat build` or `rlat optimise`, the encoder installs automatically. You don't need to do anything.

```bash
pip install rlat[build]
rlat build ./docs -o docs.rlat
# First run: downloads ~700MB and converts. Subsequent runs: cached.
```

### Manual

For offline environments, CI runners, or when you want to pre-stage:

```bash
pip install rlat[build]
rlat install-encoder
```

This downloads the HuggingFace weights, exports them to ONNX, and (on Intel CPUs with the `openvino` package available) also converts to OpenVINO IR.

### Pinning a specific revision

```bash
rlat install-encoder --revision 1c7b39da7c5a3f0c92c11d6f5cb6a6e7a3e84c2f
```

You can pass any HuggingFace ref (commit hash, branch name, or tag). Symbolic refs are resolved to the concrete commit hash before caching, so `rlat install-encoder --revision main` and `rlat install-encoder --revision <that-hash>` produce the same cache directory.

### Re-installing after a corrupted cache

If something went wrong (interrupted download, modified files, etc.):

```bash
rlat install-encoder --force
```

This regenerates everything in place.

## Cache location

The encoder is stored under:

- **Linux / macOS:** `~/.cache/rlat/encoders/<revision>/`
- **Windows:** `%USERPROFILE%\.cache\rlat\encoders\<revision>\`

`$XDG_CACHE_HOME` is honoured if set.

You can safely delete the cache; the next `rlat build` or explicit `rlat install-encoder` re-creates it.

## Inference runtime

`rlat` picks the inference runtime automatically based on what's installed and what hardware is available:

| Your system | Runtime used | Why |
|---|---|---|
| Intel CPU + `openvino` installed | OpenVINO Runtime | 1.5-2× faster than ONNX on Intel via AVX-512 + OpenMP |
| Other CPU (AMD, ARM, Apple Silicon) | ONNX Runtime | 2-4× faster than PyTorch on CPU |
| NVIDIA GPU + `[gpu]` extra installed | ONNX Runtime (CUDA provider) | Auto-preferred for build-time batch encoding; helps on the query path with very large batches |
| `rlat build` / `rlat optimise` | PyTorch | Build paths need transformers' forward pass; auto-uses CUDA if available |

You **don't normally choose.** The runtime is a function of what's installed and what hardware is available, and there are no flags exposing it on the search path.

If the OpenVINO IR isn't present in the cache (e.g. you pre-staged on an AMD host and copied the cache to an Intel host), `rlat` falls back to ONNX automatically — you'll keep working, just without the Intel speedup.

### NVIDIA GPU (CUDA)

If you have NVIDIA hardware:

```bash
pip install rlat[build,gpu]
rlat install-encoder
```

The `[gpu]` extra installs `onnxruntime-gpu`, which exposes `CUDAExecutionProvider`. `rlat`'s ONNX path discovers available providers at load time — CUDA gets auto-preferred when present. For `rlat build` / `rlat optimise`, the PyTorch path inside the encoder also calls `torch.cuda.is_available()` and lands the model on CUDA when it can.

A note about query-time GPU: for single queries (1 text × ~30 tokens) the host-to-device transfer typically dominates and CPU runtimes win. CUDA shines on batch encoding — build paths, optimise paths, anything encoding a corpus or many queries at once. The auto-selector never picks `runtime="torch"` for two reasons: (a) it tends to be slower than ONNX/OpenVINO for single-query workloads, and (b) `torch` is in the optional `[build]` extra so cannot be assumed available. If you want torch explicitly:

```python
from resonance_lattice.field.encoder import Encoder
enc = Encoder(runtime="torch")  # requires [build] extra; uses CUDA if available
```

**`[build]` alone is not enough for query-time CUDA.** With `[build]` you get a torch path that auto-uses CUDA *for builds and optimise*. Query-time queries still go through the auto-selected ONNX path, which on a CPU-only `onnxruntime` install runs on CPU regardless of whether torch sees a GPU. To get query-time CUDA, install `[gpu]` as well so the ONNX path picks up `CUDAExecutionProvider`.

### When OpenVINO is unavailable

`openvino` is in the `[build]` extra. If you only installed `rlat` (no `[build]`), the OpenVINO path is unavailable and `rlat` uses ONNX everywhere. That's fine for query-time use; you only need `[build]` to actually build or optimise knowledge models.

## What's in the cache

After a successful install, the revision directory contains:

```
<revision>/
├── revision.txt          # the pinned HF commit hash
├── tokenizer.json        # tokenizer used at query time
├── model.onnx            # ONNX export
├── torch/                # HuggingFace snapshot (used by build paths)
└── openvino/             # OpenVINO IR (only on Intel CPUs)
```

The directory name **is** the revision hash — meaningful for builds that record `backbone.revision` in `metadata.json` so a knowledge model built today is byte-comparable to one built tomorrow on a different machine, as long as the revision matches.

## Why it's locked

v0.11 of `rlat` had encoder presets, pooling toggles, and projection knobs. We measured them across BEIR-5 and LongMemEval and dropped every one of them: cross-knowledge-model search depends on the base band being byte-comparable, which is incompatible with per-build encoder choice.

If you want better in-corpus retrieval, the route is `rlat optimise` — it adds an MRL-trained 512-dim **optimised band** alongside the locked base band, in place. See [OPTIMISE.md](./OPTIMISE.md).

## Troubleshooting

**"No encoder cache at ..."** — you haven't run `rlat install-encoder` yet (or it's never been triggered automatically). Run it.

**"OpenVINO IR not found in ..."** — `runtime="openvino"` was requested explicitly but the IR isn't in the cache. Either install on an Intel host with the `openvino` package, or omit the explicit runtime so auto-fallback to ONNX kicks in.

**Slow first query after a long idle** — that's the lazy-load: tokenizer + runtime only initialise on the first `encode()` call. Subsequent calls reuse the cached state.

**Need a different encoder?** — there isn't one in v2.0. The only adjacent move is `rlat optimise`, which adds an optimised band on top of the locked base.
