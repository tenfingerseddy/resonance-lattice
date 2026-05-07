# Architecture

The v2.0 thesis is a **three-layer split**, all of it CLI-first:

```
┌────────────────────────────────────────────────────────────┐
│  field/    ROUTER      gte-modernbert-base 768d, dense    │
│  store/    AUTHORITY   3 modes — bundled / local / remote │
│  no reader             rlat returns passages, not prose   │
└────────────────────────────────────────────────────────────┘
```

Single-recipe by design. No reranker, no lexical sidecar, no auto-mode router, no encoder presets. The v0.11 retrieval-knob surface is intentionally absent — if you came looking for `--rerank`, `--retrieval-mode`, or `--cascade`, they were measured null/negative on strong-dense and removed (see `audits/01_feature_audit.md`).

> Source-of-truth code: `src/resonance_lattice/`.
> Companion docs: [FIELD.md](./FIELD.md), [STORE.md](./STORE.md), [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md), [BENCHMARK_GATE.md](./BENCHMARK_GATE.md).

## The three layers

### Field (router) — `src/resonance_lattice/field/`

Turns a query into `[(passage_idx, score), ...]`. Single encoder, three swappable inference runtimes.

| File | Role |
|---|---|
| `encoder.py` | gte-modernbert-base 768d, CLS+L2, max-seq 8192. Runtime auto-picks ONNX/OpenVINO at query time, torch on the build path. |
| `onnx_runtime.py` | Default non-Intel CPU runtime. |
| `openvino_runtime.py` | Intel CPU runtime — auto-detected. |
| `torch_runtime.py` | Build + optimise only (pulls torch via `[build]`). |
| `dense.py` | Exact cosine retrieval, dedup by `(source_file, char_offset)`. |
| `ann.py` | FAISS HNSW M=32 efC=200 efS=128 above N=5000 passages. Audit 04 locked. |
| `algebra.py` | `merge` / `intersect` / `diff` / `subtract` / `centroid` — minimal field algebra used by RQL ops + cli/compare + cli/summary. |
| `__init__.py` | `retrieve(query_emb, handle, ann_index, registry, top_k)` — the canonical retrieval entry point used by every cli/* path. |

**Locked invariants**: pinned HF revision, CLS pooling, L2-norm output, no swap of pool/norm without a paired BENCHMARK_GATE update.

### Store (authority) — `src/resonance_lattice/store/`

Resolves `(source_file, char_offset, char_length)` triples back to authoritative text, with drift detection.

| File | Role |
|---|---|
| `archive.py` | ZIP v4 read/write/in-place. `ArchiveContents` holds metadata + registry + bands + ANN bytes. `select_band(prefer="base"|None)` returns a `BandHandle`. |
| `metadata.py` | JSON schema + serialise. `FORMAT_VERSION = 4` is the single source of truth; archive imports it. Forward-compat `_extras` dicts at top-level and per-band. |
| `bands.py` | NPZ I/O for base + (optional) optimised + W projection. Half-present optimised raises. |
| `registry.py` | `PassageCoord` (frozen dataclass), `passages.jsonl` reader/writer; line-implicit `passage_idx`, blank-line-rejecting. |
| `chunker.py` | `passage_v1` strategy — paragraph→sentence→hard split, undersized-tail emission for full coverage. |
| `base.py` | `Store` ABC + `compute_hash`. Concrete `fetch` / `verify` / cache live here; subclasses implement only `_read_full_text_uncached`. |
| `bundled.py` | zstd-framed `source/` inside the .rlat ZIP. Module-level `_DCTX` for one-shot decompression. |
| `local.py` | FS-resolved via `--source-root`, per-instance text cache, path-traversal guard. |
| `remote.py` | HTTP-pinned manifest + on-disk cache, default 30s timeout, injectable opener for tests. |
| `verified.py` | `VerifiedHit` dataclass + `verify_hits(hits, store, registry)` — bridges `field.dense.search`'s tuples to authoritative-text rows with drift status. Re-exports `compute_hash` and `DriftStatus`. |
| `__init__.py` | `open_store(km_path, contents, source_root)` factory used by every CLI path. |

**Locked invariants**: per-passage `content_hash`, atomic ZIP writes via tmp+rename, ZIP_STORED outer compression (NPZ is already compressed), cross-knowledge-model ops always run on the base band.

### No reader

`rlat search` returns passages — `(score, source_file:offset+length, drift_status, text)` tuples. Synthesis is the consumer's job; we don't ship a "reader" layer that does its own LLM call.

This is a product decision, not a missing feature. Three reasons:
1. **Honesty about scope**: the retrieval is sound; the synthesis would be brittle and not what differentiates rlat.
2. **Cost**: every reader call is an LLM bill the consumer didn't ask for.
3. **Composability**: AI assistants and RAG pipelines already have synthesis paths; bolting on another one degrades the integration story.

`rlat search --format context` materialises a token-budgeted block of verified passages with HTML-comment delimiters — designed to be piped into an LLM call you're already making, not invoked as one.

## CLI surface — `src/resonance_lattice/cli/`

Every subcommand is a thin orchestrator: load archive, pick band, retrieve, format. The shared helpers (lifted during Phase 3 retroactive simplify) keep the dispatchers ~100-150 LOC each.

| Command | File | Role |
|---|---|---|
| `rlat install-encoder` | `install_encoder.py` | HF download + ONNX export + OpenVINO IR conversion. |
| `rlat init-project` | `init.py` | Auto-detect sources → `cmd_build` → `cmd_summary`. Sugar over build+summary. |
| `rlat build` | `build.py` | Walk sources, chunk, encode (torch), build FAISS index, write v4 ZIP. |
| `rlat search` | `search.py` | encode → `field.retrieve` → `verify_hits` → format text/json/context. |
| `rlat profile` | `profile.py` | Backbone + bands + drift summary. JSON status-discriminator. |
| `rlat compare` | `compare.py` | Centroid cosine + asymmetric mutual coverage. Base band only. |
| `rlat summary` | `summary.py` | Extractive primer (Landscape / Structure / Evidence). |
| `rlat refresh` | `maintain.py` + `store/incremental.py` | Local-mode incremental delta-apply: walk source_paths → bucketise on stable passage_id → re-encode updated+added → preserve unchanged rows + re-project optimised band → atomic write. |
| `rlat watch` | `watch.py` + `maintain.py` semantics + `watchdog` | Live, silent, self-discovering refresh loop on top of the same `incremental.apply_delta` pipeline. Per-archive `_DebouncedRefresher` + `threading.Lock` (closes the `<archive>.tmp` race that two concurrent refreshes would otherwise lose). Mental model: events are hints to reconcile, not the unit of correctness — every fire does a full source-tree walk + bucketise. `--once` is the synchronous CI / pre-commit shape (no observer, no event wait). Skipped-file preservation defends against transient read failures becoming silent deletes. Local mode only; bundled / remote rejected at preflight. Requires `[watch]` extra. |
| `rlat sync` | `maintain.py` + `store/incremental.py` + `store/remote_index.py` | Remote-mode incremental delta-apply: `RemoteIndex.changed_files_since(pinned_ref)` → fetch deltas only → same `incremental.apply_delta` pipeline as refresh. Rewrites the in-archive `manifest.json` per-entry `sha256` and `metadata.build_config["pinned_ref"]` atomically with the band write (codex P0 correctness gate baked in: `apply_delta` requires the encoder by signature). |
| `rlat convert` | `cli/convert.py` + `store/conversion.py` | Switch storage modes (bundled / local / remote) without rebuilding. `Store.fetch_all()` materialises bytes in the source mode; conversion validates per-passage `content_hash` against live bytes (drift abort if any mismatch); composes target-mode payload (`source/` zstd / `manifest.json` / disk files); rewrites metadata atomically. Bands + registry + projections + ANN preserved (`np.allclose` at 1e-6). Audit 08. |
| `rlat freshness` | `maintain.py` + `store/remote.py` | Remote-mode read-only drift check: walks the manifest, downloads each entry, hashes it, reports per-entry status. CI-friendly gate before running sync. |

Shared helpers:

- `cli/_load.py:load_or_exit(km_path)` — friendly archive read.
- `cli/_load.py:open_store_or_exit(km_path, contents, source_root)` — friendly Store construction.
- `cli/_load.py:load_build_spec(contents, *overrides)` — single owner of "read provenance from `build_config` (source_root / source_paths / extensions / min_chars / max_chars) with fallbacks + CLI overrides." Returns `BuildSpec` dataclass; used by `cmd_refresh` and `_preflight_archive` (watch).
- `cli/build.py:_DEFAULT_MIN_CHARS` / `_DEFAULT_MAX_CHARS` — chunker bounds, single source of truth across build / refresh / sync / watch.
- `cli/app.py` — `argparse` dispatch; each subcommand registers via its own `add_subparser(sub)`.

## Data flow

### Build

```
sources → walk → utf-8 decode (skip non-text) → chunk_text → encode (torch, L2) →
  PassageCoord registry (sha256 per passage) →
  metadata.json + bands/base.npz + (optional) ann/base.faiss + (optional) source/ →
  archive.write (atomic tmp + os.replace)
```

### Query

```
"<query>" → Encoder.encode (auto runtime) →
  archive.read (eager bands + registry + ann_blob bytes) →
  contents.select_band() → BandHandle →
  field.retrieve(query, handle, ann_index, registry, top_k) → [(passage_idx, score)] →
  open_store(km_path, contents, source_root) → Store →
  verify_hits(hits, store, registry) → [VerifiedHit] →
  filter_verified (if --verified-only) → format text/json/context
```

Both flows go through the same shared helpers; the asymmetry is encoder runtime (build=torch, query=auto) and store population (build writes, query reads).

## Configuration surface

There **is no retrieval configuration**. The full surviving config from v0.11's preset registries is two enums and two frozen dataclasses (`config.py`):

```python
class StoreMode(Enum):    BUNDLED = "bundled"; LOCAL = "local"; REMOTE = "remote"
class Kind(Enum):         CORPUS = "corpus"; INTENT = "intent"

@dataclass(frozen=True)
class MaterialiserConfig:    # token budgets for context assembly
    token_budget: int = 3000
    sections_landscape: int = 600
    sections_structure: int = 800
    sections_evidence: int = 1600
    chars_per_token: int = 4

@dataclass(frozen=True)
class BuildConfig:
    chunker: str = "passage_v1"
    min_chars: int = 200
    max_chars: int = 3200
    store_mode: StoreMode = StoreMode.LOCAL
    kind: Kind = Kind.CORPUS
```

Anything else you might think should be a knob — encoder choice, pooling, normalisation, ANN params, retrieval mode — is locked in code or pinned in the archive's `metadata.json`. Per the audit (`audits/02_deps_and_presets.md`), the v0.11 preset registries collapsed ~80%.

## Cross-references

- Encoder + runtimes + ANN: [FIELD.md](./FIELD.md).
- Store layer + format: [STORE.md](./STORE.md), [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md).
- Locked benchmark numbers: [BENCHMARK_GATE.md](./BENCHMARK_GATE.md).
- Per-feature audits: [audits/](./audits/).
- Long-horizon tracking: [REBUILD_PLAN.md](./REBUILD_PLAN.md).
