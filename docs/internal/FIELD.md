# Field Layer — Technical Reference

The field layer is the **router** in the v2.0 three-layer thesis (field → store → no reader). It encodes text into 768-dimensional CLS-pooled embeddings, performs dense cosine retrieval, and exposes minimal algebra used by RQL ops. This document covers the encoder + the three inference runtimes + the install pipeline that produces their assets.

> Source-of-truth code: `src/resonance_lattice/field/`, `src/resonance_lattice/install/encoder.py`.
> Specification: [base-first-rebuild.md §1](../../.claude/plans/base-first-rebuild.md).

## Layer overview

```
text in ─── tokenize (HF Rust tokenizers) ──┐
                                            ▼
                       runtime.encode_batch(input_ids, attention_mask)
                                            │
                                  CLS pool [:,0,:]  (3-D → 2-D)
                                            │
                                       L2 normalise
                                            │
                                            ▼
                                     (N, 768) float32
```

Single recipe. No knobs. The retrieval pipeline (Phase 1 #11, `field/dense.py`) consumes these embeddings; algebra ops (Phase 1 #12) operate on band tensors of the same shape.

## Locked encoder recipe

| Property | Value | Authority |
|---|---|---|
| Backbone | `Alibaba-NLP/gte-modernbert-base` | `install.encoder.MODEL_ID` |
| Pooling | CLS (token index 0) | `field._runtime_common.cls_pool` |
| Output dim | 768 | `field.encoder.DIM` |
| Max sequence length | 8192 tokens | `field.encoder.MAX_SEQ_LENGTH` |
| Normalisation | L2 on each row, ε-guarded | `field.encoder.Encoder.encode` |
| Revision | pinned at install time | `install.encoder.PINNED_REVISION` |

These are **constants, not configuration.** Encoder presets, pooling toggles, projection knobs are intentionally absent. Cross-knowledge-model interop depends on every knowledge model's base band being byte-comparable, which forces a single recipe.

## Public encoder API

The orchestrator lives in `field.encoder`:

- **`Encoder(runtime="auto", revision=None)`** — construction is cheap; the runtime + tokenizer are lazy-loaded on first `encode()` call. Reuse one instance across calls.
- **`Encoder.encode(texts: list[str]) → np.ndarray`** — returns `(N, 768)` L2-normalised float32. Empty list returns `(0, 768)`.
- **`encode(texts, runtime="auto")`** (module-level) — singleton convenience, used by query-time paths that don't need explicit construction.
- **`get_pinned_revision() → str`** — delegates to `install.encoder.get_pinned_revision()`. The package-pinned hash wins over mtime when multiple revisions are cached.

`Encoder.encode` is the **query-time hot path.** Three implementation choices made here:

1. The singleton in module-level `encode()` resolves `"auto"` once per Encoder, not per call. The `_select_runtime("auto")` import probe is cached but `is_available()` does CPU vendor detection which we don't want firing per query.
2. CLS pooling happens inside the runtime export (graph output). The runtime returns `(N, seq_len, 768)` last-hidden-state and `_runtime_common.cls_pool` extracts `[:, 0, :]`. Dropping CLS pooling into the graph itself is a Phase 1 #10 deferred option.
3. L2 normalises in place (`cls /= norms`) — saves one `(N, 768)` float32 allocation per call. Safe because the runtime's output buffer is freshly allocated and not aliased by the caller.

## Runtime matrix

Each runtime exposes a uniform contract:

```python
load(asset_path: Path) -> handle
encode_batch(handle, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray  # (N, seq_len, 768)
```

| Runtime | Asset format | Used when | Implementation |
|---|---|---|---|
| **ONNX** | `model.onnx` | non-Intel CPU; auto fallback | `field/onnx_runtime.py` |
| **OpenVINO** | `openvino_model.{xml,bin}` (or `model.{xml,bin}`) | Intel CPU + `openvino` package | `field/openvino_runtime.py` |
| **PyTorch** | `torch/` snapshot dir | build & optimise paths only | `field/torch_runtime.py` |

### Auto-selection (`Encoder(runtime="auto")`)

`field.encoder._select_runtime`:
1. If `openvino` package not importable → `"onnx"`.
2. Else, if `field._runtime_common.is_intel_cpu()` is False → `"onnx"`.
3. Else → `"openvino"`.

`"torch"` is **never** auto-picked because it pulls in the optional `[build]` extra.

`Encoder._ensure_loaded` performs an additional asset-presence check on top of the runtime selection: even when auto resolved to `"openvino"`, if the OpenVINO IR (`.xml` + `.bin`) is missing from the cache (e.g. the cache was offline-staged from a non-Intel host), it falls back to `"onnx"` rather than crashing on `load()`. This is why the runtime selection cannot rely solely on `is_available()` — install state matters.

### Intel CPU detection

`is_intel_cpu()` lives in `field/_runtime_common.py` and is shared between the runtime selector and the install-time IR conversion gate. Detection is best-effort across Win/Linux/macOS:

- Windows: `PROCESSOR_IDENTIFIER` env var prefix.
- Linux: `/proc/cpuinfo` contains `GenuineIntel`.
- macOS: `platform.machine() == "x86_64"` (Intel Macs only — Apple Silicon is arm64).
- Fallthrough: `platform.processor()` containing `"Intel"`.

OpenVINO **runs** on AMD x86 and (newer versions) ARM. We prefer ONNX on those targets to keep query latency stable until Audit 04 measures otherwise. The detection function is intentionally conservative — false negatives (unrecognised Intel SKUs) downgrade to ONNX cleanly; false positives could push slow OV onto AMD hosts.

### Per-runtime details

**ONNX runtime** (`field/onnx_runtime.py`):
- `onnxruntime.InferenceSession` with `providers=ort.get_available_providers()`. The `[gpu]` extra installs `onnxruntime-gpu`, which announces `CUDAExecutionProvider`; ORT auto-prefers CUDA when present. CPU-only installs see only `CPUExecutionProvider`. Single-query latency on CPU still wins over CUDA because the host↔device transfer dominates a 30-token forward pass; CUDA matters at build / optimise / batched-rerun time.
- Validates `input_ids` / `attention_mask` are present in the session's declared inputs at `load()` time. A future export drift fails loudly at install rather than silently per-batch.

**OpenVINO runtime** (`field/openvino_runtime.py`):
- `openvino.Core().compile_model(xml_path, "CPU")`. The single-threaded `InferRequest` is created once at `load()` and reused across `encode_batch` calls — saves ~100-300μs per warm query versus creating a fresh request each call.
- Accepts both `openvino_model.xml` (Optimum convention) and `model.xml` (in-tree converter). `find_xml(model_dir)` returns the first match or `None` for the asset-presence fallback in `Encoder._ensure_loaded`.

**PyTorch runtime** (`field/torch_runtime.py`):
- `transformers.AutoModel.from_pretrained` + `model.eval()`, on CUDA when `torch.cuda.is_available()` else CPU. No flag — auto.
- Used at build / optimise time (corpus-scale batch encoding) where the larger batches pay back the dispatcher overhead. The auto-selector never picks `torch` for query path because ONNX/OpenVINO are 2-4× faster on small inputs even when CUDA is available; users who want torch explicitly construct `Encoder(runtime="torch")`.
- `torch.inference_mode` disables autograd. On CPU we skip `.cpu()` (no-op but goes through dispatcher) and call `.numpy()` directly; on CUDA the host copy is unavoidable.
- Types via `TYPE_CHECKING` so static analysis sees `torch.device` / `PreTrainedModel` without forcing a runtime torch import for callers of unrelated code.

### Shared helpers (`field/_runtime_common.py`)

Three private utilities used across the runtimes and the install pipeline:

- **`require_module(name, install_hint)`** — lazy-import + uniform RuntimeError. Replaces four duplicate try-import patterns.
- **`require_asset(path, label)`** — uniform missing-cache error pointing at `rlat install-encoder`.
- **`cls_pool(arr)`** — single CLS slice with shape assert (`(N, seq_len, 768)` → `(N, 768)`). The export shape is locked at install time.

## Install pipeline

`install.encoder.install(revision=None, force=False) → Path`

Pipeline:

1. **Resolve revision.** Symbolic refs (`"main"`, branches, tags) go through `HfApi().model_info()` and resolve to the concrete 40-char commit hash. Already-concrete hashes pass through. Short hex prefixes are not assumed to be hashes (a tag like `0123456789` would collide).
2. **Skip-if-installed.** `is_installed(concrete)` checks every required artefact for this host: `revision.txt`, `tokenizer.json`, `model.onnx`, `torch/config.json`, `torch/*.safetensors` (any), and (only on Intel + openvino) `openvino/openvino_model.{xml,bin}`. If all present and `force=False`, returns immediately.
3. **Download HF snapshot** into `<rev>/torch/` via `huggingface_hub.snapshot_download` with allow-pattern restricted to tokenizer + config + safetensors. snapshot_download is content-addressed — re-runs verify SHAs and skip files already present locally.
4. **Copy tokenizer.json** to the revision root for O(1) tokenizer loads at runtime.
5. **Export ONNX** via `torch.onnx.export` through a `_HiddenStateWrapper(nn.Module)` that strips the HF `BaseModelOutput` wrapper and returns a plain `last_hidden_state` tensor. Opset 17, `do_constant_folding=True`, `dynamic_axes` on batch + seq_len.
6. **Convert to OpenVINO IR** (only if Intel + openvino): `ov.convert_model(onnx_path)` → `ov.save_model(<rev>/openvino/openvino_model.xml)`.
7. **Atomically write `revision.txt`** via tmp + `os.replace`. This is the install-complete sentinel.

Crash-recovery: if a prior run died after writing partial conversion outputs but before `revision.txt`, the retry detects `revision.txt is missing` and **regenerates all conversion outputs** even when their files exist. This avoids blessing a possibly-truncated `model.onnx` as valid.

## Cache layout

```
$XDG_CACHE_HOME/rlat/encoders/<revision>/
├── revision.txt              # the concrete HF commit hash (atomic sentinel)
├── tokenizer.json            # Rust-tokenizers Tokenizer.from_file() reads this
├── model.onnx                # ONNX export, last_hidden_state output
├── torch/                    # HF snapshot — used by torch_runtime + as ONNX export source
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── model.safetensors     # (or model-NNNN-of-MMMM.safetensors for sharded)
└── openvino/                 # only on Intel CPU + openvino package installed
    ├── openvino_model.xml
    └── openvino_model.bin
```

`$XDG_CACHE_HOME` defaults to `~/.cache` on Linux/macOS and `~\.cache` on Windows (we honour the env var if set). Multiple revisions can coexist; `get_pinned_revision()` prefers `PINNED_REVISION` if set and cached, else most-recent-mtime.

## Failure modes

| Error | Cause | Fix |
|---|---|---|
| `No encoder cache at <path>. Run rlat install-encoder first.` | First call before any install | `rlat install-encoder` |
| `Encoder cache at <path> has no revision pinned.` | Cache dir exists but has no `revision.txt` | `rlat install-encoder` (will populate / regenerate) |
| `OpenVINO IR not found in <dir>` | OV runtime explicitly requested but IR not staged | Re-install on an Intel host with `openvino` package, or pass `runtime="onnx"` |
| `ONNX export at <path> is missing inputs [...]` | `model.onnx` was rebuilt with a different export wrapper | `rlat install-encoder --force` |
| `tokenizers is not installed` | Base dependency missing (corrupted env) | `pip install --force-reinstall rlat` |
| `transformers / torch is not installed` | `runtime="torch"` requested without `[build]` extra | `pip install rlat[build]` |

## Dense retrieval

`field/dense.py` is the single retrieval strategy applied uniformly. There is no router, no mode-selection flag, no auto vs explicit choice. The presence or absence of the optimised band is determined when the knowledge model is loaded and cached on the handle.

### `search(query, band, registry, projection_matrix, top_k)`

```
encode(query) → q  (D,)
       │
       └── if optimised: q_proj = L2(q @ W.T)  (d_native,)
             │
             └── scores = band @ q              (N,)
                   │
                   └── argpartition top-(top_k × 4) candidates
                         │
                         └── argsort the candidate slice
                               │
                               └── dedup_by_source(hits, registry)
                                     │
                                     └── return hits[:top_k]
```

Implementation choices:

- **Cosine == dot product.** The encoder L2-normalises every output and the bands are stored already-normalised (Phase 2). The retrieval kernel is `band @ q`, no division.
- **`np.argpartition` over `np.argsort`.** O(N) partition + O(K log K) sort of the candidate slice beats O(N log N) full sort for the typical `K << N` case. On a 50K-passage band, 10× faster.
- **Candidate budget loop.** Start with `top_k × 4` candidates (calibrated for the WS3 #292 observation that 10–30% of nearest-neighbour pairs share `(source_file, char_offset)`). If dedup leaves fewer than `top_k` distinct hits, **double the budget** and re-partition; loop until the budget covers the whole band. Guarantees the function honours its `top_k` contract on duplicate-heavy registries.
- **W projection norm.** When projecting through the optimised W, the result is *re-normalised* even though W is approximately orthonormal — eps-guarded so a degenerate projection can't produce NaN.

### `dedup_by_source(hits, registry)`

Two passages are query-time duplicates if they share `(source_file, char_offset)`. First-seen wins. The registry parameter is duck-typed — anything exposing `.source_file: str` and `.char_offset: int` works. The static type annotation is intentionally minimal (`Sequence`) until Phase 2 lands `store.registry.PassageCoord` as the canonical type.

Two duplication mechanisms exist; this query-time path handles both:
- An overlapping chunker emits passages whose embeddings are near-identical even though `char_offset` differs by a few bytes — the dedup key still matches.
- Boilerplate text recurs across files at different `(source_file, char_offset)` positions; query-time dedup doesn't suppress these (different keys), but build-time content-hash dedup does (handled by Phase 2 `store/registry.py`).

## ANN indexing

`field/ann.py` builds and queries a FAISS HNSW index over band embeddings. It mirrors `dense.search`'s contract so callers can route between exact and approximate paths without rewiring args.

### When ANN runs

```
should_build_ann(N) ⇔ N > ANN_THRESHOLD_N (= 5000)
```

Below the threshold, exact `dense.search` is fast enough on CPU and the index-build memory cost isn't justified. Above it, `rlat build` writes one FAISS index per band into `ann/<band>.faiss` inside the knowledge model.

### Locked configuration

| Constant | Value | Justification |
|---|---|---|
| `HNSW_M` | 32 | Connectivity tradeoff. 32 is the bge / e5 community default for retrieval at d=768 |
| `HNSW_EFCONSTRUCTION` | 200 | Build-time accuracy ceiling. Diminishing returns above 200 on mainstream corpora |
| `HNSW_EFSEARCH` | 128 | Query-time accuracy floor. Audit 04 measured efS=32 (base plan default) at ~13% recall@10 on synthetic 50K @ 768d; efS=128 clears the 0.95 audit gate at N=5K. Real-corpus calibration deferred to Phase 1 #15 (BEIR-5 floor lock) |
| `ANN_THRESHOLD_N` | 5000 | Below this, exact matmul on numpy is fast enough |

These are constants in code, not config knobs. Locked at Audit 04 (Phase 1 #14).

### Library: FAISS (Audit 04 lock)

`faiss-cpu` won the Audit 04 tertiary cross-platform-wheel gate before recall-vs-hnswlib could be measured: hnswlib has no precompiled wheel for Python 3.12 on Windows (source build needs Visual C++ Build Tools). FAISS has prebuilt wheels everywhere; ScaNN is Linux/macOS only. Evidence: [audits/03_format_ann_chunking.md §Audit 04 verdict](./audits/03_format_ann_chunking.md), `benchmarks/results/ann_audit_04.json`.

FAISS HNSW build is ~5-10× slower than hnswlib by published benchmarks (~75s for 50K @ M=32 efC=200 on Win11). The 5min/500K secondary gate likely passes (sub-linear scaling) but isn't validated.

### Cosine via METRIC_L2

FAISS HNSW + `METRIC_INNER_PRODUCT` has known quality issues. The canonical FAISS cosine recipe is `METRIC_L2` over already-L2-normalised vectors: for unit vectors `||a-b||² = 2 - 2<a,b>`, so L2 ranking is monotonic with cosine ranking. `search()` recovers cosine score as `1 - L2² / 2` so the score-ordering matches `dense.search`.

### `search(index, query, registry, projection_matrix, top_k)`

Same contract as `dense.search`:
- Optional W projection through optimised band (re-L2 via `common.l2_normalize`).
- Optional registry → first-seen-wins dedup with the doubling-budget loop.
- Returns list of `(passage_idx, score)` sorted descending.

The `efSearch` bump is wrapped in a `try/finally` so a search call that needed a temporary high `efSearch` doesn't leak that cost to subsequent callers.

### Persistence

- `save(index, path)` — `faiss.write_index`. Phase 2's store layer wires this into `ann/base.faiss` / `ann/optimised.faiss` inside the knowledge model.
- `load(path, dim)` — `faiss.read_index`. `dim` is unused at FAISS load time (the file records it) but accepted for API symmetry; Phase 2's store layer reads `dim` from `metadata.json` regardless.

## Field algebra

`field/algebra.py` is the minimal operator set used by Phase 6 RQL composition. The v0.11 surface (~271 ops) collapses to 5 here; the rest were either NumPy wrappers or relics of the field-as-content thesis (PSD projection, symplectic ops, reaction-diffusion heads).

### Operators

| Op | Definition | Symmetric? | Notes |
|---|---|---|---|
| `merge(a, b)` | `a + b` | yes | Strict-associative. Identity is `empty(...)` |
| `intersect(a, b)` | sign-aware bottleneck (see below) | yes | Same-sign smaller-magnitude wins; disagreeing signs zero out |
| `diff(a, b)` | `np.maximum(a - b, 0)` | no | Asymmetric residual (a but-not-b, ReLU style) |
| `subtract(a, b)` | `a - b` | no | Signed residual; additive inverse of `merge` |
| `empty(shape)` | `np.zeros(shape)` | — | Identity element |

All operators are **elementwise and shape-agnostic**: they work on a single `(D,)` concept vector or on an `(N, D)` band tensor as long as the operands share a shape. Operators are linear and intentionally do **not** L2-renormalise their output — callers that need a unit vector apply `_runtime_common.l2_normalize` themselves. Skipping in-op renormalisation keeps `merge` strictly associative under floating-point arithmetic.

`intersect` is the only non-`np.<op>` formulation: a naive `np.minimum` over L2-normalised cosine vectors inflates negative coordinates (`min(-0.2, -0.7) == -0.7`), which is the wrong reading for "shared signal". The implementation returns the smaller magnitude with the shared sign, or zero where signs disagree.

### Invariants (validated by `tests/harness/property.py`)

```
merge(a, merge(b, c))    == merge(merge(a, b), c)     # associativity
merge(a, empty(...))     == a                          # identity
merge(a, b)              == merge(b, a)                # commutativity
subtract(a, a)           == empty(...)
intersect(a, b)          == intersect(b, a)            # commutativity
diff(a, a)               == empty(...)
diff(a, b)               >= 0                          # non-negativity
```

The property suite runs 32 fixed-seed trials per invariant on `(D,)` random vectors. Hypothesis would be the natural fit but is in the `[dev]` extra; the deterministic loop covers the contract without that dependency.

`merge ∘ subtract` is approximately the identity: `merge(a, subtract(b, a)) ≈ b` to within float32 precision (`atol≈1e-5` on length-768 random normals). Not a strict invariant — float32 cancellation costs precision in the low bits — but documented here so consumers know the relationship is robust enough for retrieval-grade use.

## Cross-references

- User-facing single-recipe doc: [`docs/user/ENCODER.md`](../user/ENCODER.md).
- Knowledge-model format (where `metadata.json` records `backbone.revision`): Phase 2, [`docs/internal/KNOWLEDGE_MODEL_FORMAT.md`](./KNOWLEDGE_MODEL_FORMAT.md) (TBD).
- ANN configuration (separate concern, lives in `field/ann.py`): Phase 1 #13, [`audits/03_format_ann_chunking.md`](./audits/03_format_ann_chunking.md) (Audit 04 lock pending).
- RQL composition ops that consume this algebra: Phase 6 (`rql/compose.py`).
