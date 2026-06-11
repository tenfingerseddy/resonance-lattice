# Field Layer — Technical Reference

The field layer is the **router** in the three-layer thesis (field → store → grounded synthesis — see [GROUNDING_MODEL.md](./GROUNDING_MODEL.md)). It encodes text into 768-dimensional CLS-pooled embeddings, performs dense cosine retrieval (exact + ANN behind one entry point), observes every retrieval into the capture stream, and exposes the band-level geometry RQL and the curator use. This document covers the encoder + the three inference runtimes + the install pipeline that produces their assets, the retrieval entry points, capture + counters, and the geometry/text helpers.

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
- **`Encoder.encode_batched(texts, batch_size) → np.ndarray`** — fixed-size batches, per-batch outputs concatenated. Used by `cli/build.py` and `store/incremental.apply_delta` so a corpus-scale encode doesn't blow peak memory — the tokenizer and the ONNX/OpenVINO session both allocate an `(input_ids, attention_mask)` pair sized to the whole input list.
- **`encode(texts, runtime="auto")`** (module-level) — singleton convenience, used by query-time paths that don't need explicit construction.
- **`get_pinned_revision() → str`** — delegates to `install.encoder.get_pinned_revision()`. The package-pinned hash wins over mtime when multiple revisions are cached.

`Encoder.encode` is the **query-time hot path.** Three implementation choices made here:

1. The module-level `encode()` reuses a singleton Encoder. After the first call it still re-resolves the requested runtime per call (cheap: `find_spec` + vendor check) and rebuilds the Encoder only when the resolved concrete runtime differs from the cached one — so an explicit `runtime="torch"` after a default call swaps cleanly instead of silently serving the wrong runtime.
2. CLS pooling happens inside the runtime export (graph output). The runtime returns `(N, seq_len, 768)` last-hidden-state and `_runtime_common.cls_pool` extracts `[:, 0, :]`. Dropping CLS pooling into the graph itself is a Phase 1 #10 deferred option.
3. L2 normalises in place via `_runtime_common.l2_normalize` (ε-guarded `x /= norms`, shared with `algebra.centroid`) — saves one `(N, 768)` float32 allocation per call. Safe because the runtime's output buffer is freshly allocated and not aliased by the caller.

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
| **PyTorch** | `torch/` snapshot dir | build path only | `field/torch_runtime.py` |

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
- `onnxruntime.InferenceSession` with `providers=ort.get_available_providers()`. The `[gpu]` extra installs `onnxruntime-gpu`, which announces `CUDAExecutionProvider`; ORT auto-prefers CUDA when present. CPU-only installs see only `CPUExecutionProvider`. Single-query latency on CPU still wins over CUDA because the host↔device transfer dominates a 30-token forward pass; CUDA matters at build / batched-rerun time.
- Validates `input_ids` / `attention_mask` are present in the session's declared inputs at `load()` time. A future export drift fails loudly at install rather than silently per-batch.
- `enable_mem_pattern` is disabled. Memory-pattern reuse keys on the first input shape; with batch and seq_len both dynamic, the last batch of a non-divisible `encode_batched` run can crash ORT's attention allocator on some Linux builds. Small per-call alloc cost, no shape-related crashes.

**OpenVINO runtime** (`field/openvino_runtime.py`):
- `openvino.Core().compile_model(xml_path, "CPU")`. The single-threaded `InferRequest` is created once at `load()` and reused across `encode_batch` calls — saves ~100-300μs per warm query versus creating a fresh request each call.
- Accepts both `openvino_model.xml` (Optimum convention) and `model.xml` (in-tree converter). `find_xml(model_dir)` returns the first match or `None` for the asset-presence fallback in `Encoder._ensure_loaded`.

**PyTorch runtime** (`field/torch_runtime.py`):
- `transformers.AutoModel.from_pretrained` + `model.eval()`, on CUDA when `torch.cuda.is_available()` else CPU. No flag — auto.
- Used at build time (corpus-scale batch encoding) where the larger batches pay back the dispatcher overhead. The auto-selector never picks `torch` for query path because ONNX/OpenVINO are 2-4× faster on small inputs even when CUDA is available; users who want torch explicitly construct `Encoder(runtime="torch")`.
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
5. **Export ONNX** via `torch.onnx.export` through a `_HiddenStateWrapper(nn.Module)` that strips the HF `BaseModelOutput` wrapper and returns a plain `last_hidden_state` tensor. Opset 18 — not 17: the ModernBERT export emits `aten_split`, a domain-18 function in onnxscript's torch_lib, and opset 17 fails to inline it. `do_constant_folding=True`, `dynamic_axes` on batch + seq_len.
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

## Retrieval entry points

`field/__init__.py` exposes the two functions every retrieval consumer routes through (`cli/search.py`, `cli/summary.py`, `cli/memory.py`, `cli/skill_context.py`, `deep_search/loop.py`, `fabric/_runtime.py`, `curator/author.py`, `store/verified.py`):

- **`retrieve(query_emb, handle, ann_index, registry, top_k)`** — source-band retrieval: ANN when an index is bound, exact `dense.search` otherwise. Lifted from `cli/search.py` + `cli/summary.py`, where the if/else dispatch was duplicated. Both paths return `[(passage_idx, score), ...]` descending.
- **`retrieve_insight(query_emb, insight_band, ann_index, top_k, km_id=None)`** — the parallel entry for the insight band. Takes **no registry**: insight rows are unique by `insight_id`, so there is no source-file dedup; pre-promotion semantic duplicates are filtered by the compression test, not at query time. Empty band → empty list, same contract as `retrieve()` on an empty corpus.

Every call through either entry is **observed** by the capture heart before returning — `capture.observe(km_id, query_emb, result, layer)`, with `km_id` carried on the `BandHandle` (or passed explicitly for the insight band). Observation never raises and never alters the result. One user search produces two observations (source + insight) sharing the same `query_emb` — the natural join key downstream.

## Dense retrieval

`field/dense.py` is the single retrieval strategy applied uniformly. There is no router, no mode-selection flag, no auto vs explicit choice. Retrieval always runs against the base band (`select_band()` resolves to `base`); the optional insight band is a separate, additive surface, not a mode `dense.search` branches on.

### `search(query_embedding, band, registry, top_k)`

```
query_embedding → q  (D,)   # encoded upstream — dense.py never touches the encoder
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
- **Candidate budget loop.** Start with `top_k × 4` candidates (calibrated for the WS3 #292 observation that 10–30% of nearest-neighbour pairs share `(source_file, char_offset)`). If dedup leaves fewer than `top_k` distinct hits, **double the budget** and re-partition; loop until the budget covers the whole band. Guarantees the function honours its `top_k` contract on duplicate-heavy registries. The `×4` is `_runtime_common.CANDIDATE_MULTIPLIER`, shared with `ann.search`.
- **`registry=None` skips dedup entirely** — one partition + sort returns exactly `top_k`. The insight band takes this path (`retrieve_insight` — insight rows are unique by `insight_id`, so there is no source-file dedup).

### `dedup_by_source(hits, registry)`

Two passages are query-time duplicates if they share `(source_file, char_offset)`. First-seen wins. The registry parameter is duck-typed — anything exposing `.source_file: str` and `.char_offset: int` works. The static type annotation stays minimal (`Sequence`) rather than importing `store.registry.PassageCoord` — the canonical type in practice — because a module-scope store import from field/ is circular (store imports field during its own init; see the TYPE_CHECKING note in `field/__init__.py`).

Two duplication mechanisms exist; this query-time path handles both:
- An overlapping chunker emits passages whose embeddings are near-identical even though `char_offset` differs by a few bytes — the dedup key still matches.
- Boilerplate text recurs across files at different `(source_file, char_offset)` positions; query-time dedup doesn't suppress these (different keys), and the build pipeline doesn't either — `content_hash` on `PassageCoord` serves verified-retrieval drift checks and incremental change detection, not dedup. Recurring boilerplate costs band rows today.

### Band-vs-band kernels

Three more public helpers in `dense.py` serve the compare/RQL surfaces:

- **`topk_indices(scores, k)`** — the argpartition + slice-sort idiom, public so RQL ops (`navigate.neighbors` etc.) reuse it rather than reimplementing it.
- **`max_cosines_against(query_band, target_band)`** — per-row max cosine of one band against another. Memory-bounded: the `(N_query, N_target)` similarity matrix is chunked along the target axis so peak RSS stays under `COSINE_CHUNK_BYTES` (512 MB) regardless of corpus size. Empty target → `-inf` per row ("no neighbour at any threshold"), not a silent zero. Consumers: `rql/compare.unique`, `rql/navigate.corpus_diff`, and `sampled_mean_max_cosine` below.
- **`sampled_mean_max_cosine(src, dst, sample_size, seed=0)`** — deterministic sample-then-mean-max (seeded `default_rng`); identical inputs produce identical numbers across runs. Single home for the idiom shared by `cli/compare` and `rql/compare`.

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

### `search(index, query, registry, top_k)`

Same contract as `dense.search`:
- Optional registry → first-seen-wins dedup with the doubling-budget loop.
- Returns list of `(passage_idx, score)` sorted descending.

When the doubling budget exceeds `HNSW_EFSEARCH`, the bump is passed per-query via `faiss.SearchParametersHNSW` (FAISS requires `efSearch ≥` the candidate budget). The shared index is never mutated, so a cached/reused index (the Fabric warm path) stays correct under concurrent calls — FAISS releases the GIL during search.

### Persistence

- `serialize(index) -> bytes` / `deserialize(blob)` — the store layer embeds the FAISS index as bytes inside the `.rlat` ZIP (`ann/*.faiss` members) via `archive.write(ann_blobs=...)`; `deserialize` reapplies `efSearch=128` (query-time knob, not persisted by FAISS). The earlier file-path `save`/`load` pair was never wired and was removed in the 2026-06 review.

## Capture — the self-aware heart

`field/capture.py` observes every retrieval through `retrieve` / `retrieve_insight`: the query **fingerprint** (rounded embedding) and per-rank scores land in an in-memory ring buffer keyed by corpus identity (`BandHandle.km_id` — the resolved `.rlat` path). Observing is fused into retrieving — using a `.rlat` *is* being seen by it; no hook or caller cooperation enforces it.

Properties (`.claude/plans/insight-engine/capture.md` §3):

- **Fingerprints, not words.** Embedding + scores only — never the query text (the heart has none; the text lives above it). Privacy by construction.
- **In-memory and bounded.** One `deque(maxlen=4096)` per corpus. The fold into the `.rlat` lives in the store layer — `store/telemetry.py` peeks via `capture.buffered`, persists to the `insight/telemetry.jsonl` member, then `drain(n)`s exactly the persisted snapshot so a concurrent observation is never cleared unpersisted. This module keeps no store/disk dependency, so `field.retrieve` imports it with no cycle.
- **Raise-your-hand to be ignored.** `is_user_query` defaults True; internal machinery (summary probes, deep-search hops, skill-context batches, recall, verified retrieval) wraps its calls in the `internal_retrieval()` context manager so it doesn't pollute the user-intent stream.
- **Never breaks retrieval.** Every entry point swallows its own errors — a lost observation is never a failed query.

`session_id()` buckets observations by `RLAT_DOGFOOD_SESSION` when set (controlled multi-batch runs), else the UTC calendar day.

## Counters — the Tier-0 count layer

`field/counters.py` is the closed-form "count with no model" tier over the capture stream. `access_reinforcement(observations)` maps each retrieved unit `(layer, idx)` to a `ReinforceStat`: the raw user-intent hit tally, its Ebbinghaus-decayed sum (`exp(-Δt/τ)`, τ = 30 days default), and the log-damped reinforcement (`log1p` — a unit hit ten times is worth more than one hit, not ten times more). "Now" defaults to the latest observation timestamp, so a replay is deterministic with no wall clock. Only `is_user_query` rows count; a malformed row is skipped, never raised on — mirroring the hot path's never-break contract. Pure, no I/O, no persistence.

## Field algebra

`field/algebra.py` holds the band-level geometry helpers production uses:

- **`centroid(band)`** — mean of an `(N, D)` band, L2-renormalised: the mean of unit vectors is not unit; renormalising lets two centroids compare by raw dot as cosine, and lets one centroid serve as a synthetic "what is this corpus about?" query. Empty band → zero vector, so callers short-circuit cosine to 0.0 rather than leaking NaN into JSON. Consumers: `cli/compare`, `cli/summary`, `rql/compare`.
- **`greedy_cluster(embeddings, threshold)`** — single-linkage clustering by cosine ≥ threshold, union-find over the upper-triangle threshold graph. Honours transitive chains (A↔B and B↔C merge even when A↔C is below threshold) — right for near-duplicate dedup at 0.95. O(N²); the full-precision `(N, N)` cosine matrix caps a single call at ~50K rows. Consumers: `rql/inspect.near_duplicates`, `rql/compose` semantic dedupe.
- **`complete_linkage_cluster(embeddings, threshold)`** — the chaining-resistant counterpart: merges two clusters only when **every** cross pair clears the threshold, so a cluster is a genuine clique. Needed for intent clustering at 0.70, where single-linkage chains same-frame queries on different topics into one mega-cluster (measured: single @ 0.70 → 1 cluster / pairwise-F1 0.13; complete @ 0.70 → F1 1.0 — `benchmarks/bench_intent_clustering.py`). Same return contract as `greedy_cluster`. Consumer: `curator/signals` intent clustering.

The five elementwise ops the v0.11 collapse had kept (`merge`/`intersect`/`diff`/`subtract`/`empty`) were removed in the 2026-06 review — no production caller; RQL composition uses a physical archive merge and `dense.max_cosines_against` instead.

## Text utilities

`field/text.py` holds the regex sentence splitter. `iter_sentence_spans(text)` returns non-overlapping `(offset, length)` spans — greedy on `[.!?]` followed by whitespace, with an unterminated trailing sentence emitted as the final span — and `split_sentences` is its stripped-string view. Consumer: `store/chunker.py`, for paragraph→sentence splits when a paragraph exceeds `max_chars`. One module so future text handling reuses one regex instead of cloning it inline.

## Cross-references

- User-facing single-recipe doc: [`docs/site/encoder.html`](../site/encoder.html).
- Knowledge-model format (where `metadata.json` records `backbone.revision`): [`docs/internal/KNOWLEDGE_MODEL_FORMAT.md`](./KNOWLEDGE_MODEL_FORMAT.md).
- ANN library + constants lock: [`audits/03_format_ann_chunking.md`](./audits/03_format_ann_chunking.md) (Audit 04, locked).
- Grounding thesis this layer routes for: [`GROUNDING_MODEL.md`](./GROUNDING_MODEL.md).
- Algebra consumers: `rql/inspect.py`, `rql/compose.py`, `rql/compare.py`, `curator/signals.py`, `cli/compare.py`, `cli/summary.py`.
- Capture/telemetry design: `.claude/plans/insight-engine/capture.md` §3; the fold into the `.rlat` lives in `store/telemetry.py`.
