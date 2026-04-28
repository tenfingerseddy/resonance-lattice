# Knowledge-Model Format v4

`.rlat` is a **ZIP archive** containing JSON metadata and NPZ numeric arrays. v4 is a clean break from v0.11's custom binary header; nothing in v0.11 has shipped to users so backward-compat is not a concern.

> Source-of-truth code: `src/resonance_lattice/store/archive.py`, `store/metadata.py`, `store/registry.py`, `store/bands.py`.
> Specification: [`base-first-rebuild.md §2`](../../.claude/plans/base-first-rebuild.md).

## File layout

```
my-corpus.rlat                    (ZIP archive, no internal compression)
├── metadata.json                 -- Metadata (see below)
├── passages.jsonl                -- one PassageCoord per line, in passage_idx order
├── bands/
│   ├── base.npz                  -- (N, 768) float32, L2-normalised
│   ├── optimised.npz            -- (N, 512) float32, L2-normalised   (optional)
│   └── optimised_W.npz          -- (512, 768) float32 MRL projection (paired with optimised)
├── ann/
│   ├── base.faiss                -- FAISS HNSW index for base band
│   └── optimised.faiss          -- FAISS HNSW index for optimised band (when present)
└── source/                       -- only if metadata.store_mode == "bundled"
    └── ...                       -- zstd-framed source files, flat layout
```

ZIP internal compression is **disabled** — NPZ files are already deflate-compressed and ZIP-on-ZIP is wasted CPU. Central-directory overhead is a few KB.

The file extension stays `.rlat` (carryover from v0.11). The format version bump (3 → 4) is recorded in `metadata.format_version` for any future reader that needs to dispatch.

## metadata.json schema

The example below is **shown logically**; on disk `to_json` emits keys in alpha-sorted order (`sort_keys=True` for stable diffs across rebuilds).

```json
{
  "format_version": 4,
  "kind": "corpus",
  "backbone": {
    "name": "Alibaba-NLP/gte-modernbert-base",
    "revision": "<40-char HF commit hash>",
    "dim": 768,
    "pool": "cls",
    "max_seq_length": 8192
  },
  "bands": {
    "base": {
      "role": "retrieval_default",
      "dim": 768,
      "l2_norm": true,
      "passage_count": 39235
    },
    "optimised": {
      "role": "in_corpus_retrieval",
      "dim": 512,
      "l2_norm": true,
      "passage_count": 39235,
      "dim_native": 512,
      "w_shape": [512, 768],
      "nested_mrl_dims": [64, 128, 256, 512],
      "trained_from": "bands/base.npz"
    }
  },
  "store_mode": "bundled",
  "ann": {
    "base":       { "type": "hnsw", "M": 32, "efConstruction": 200, "efSearch": 128 },
    "optimised": { "type": "hnsw", "M": 32, "efConstruction": 200, "efSearch": 128 }
  },
  "build_config": {
    "chunker": "passage_v1",
    "min_chars": 200,
    "max_chars": 3200,
    "passage_count": 39235,
    "source_fingerprint": "sha256:..."
  },
  "created_utc": "2026-04-25T19:30:00Z",
  "rlat_version": "2.0.0a1"
}
```

### Key invariants

- **`format_version`** — single integer, bumped only for breaking on-disk-layout changes. v4 is current.
- **`backbone.revision`** — 40-char HF commit hash, populated from `install.encoder.PINNED_REVISION` at build. **Cross-knowledge-model retrieval depends on this matching across models** (the base bands must be byte-comparable). Two knowledge models built at different revisions can both load, but `rlat compare` should refuse them or downgrade — Phase 3 #29 nails the exact policy.
- **`bands.<name>`** — keyed by band name. `base` is required for any usable knowledge model. `optimised` is absent when `rlat optimise` hasn't run; reader code branches on dict presence, not on a flag.
- **`bands.optimised.w_shape`** — recorded explicitly so the W matrix shape is authoritative. JSON has no tuple type; `from_json` restores `tuple` from the JSON array invariant.
- **`store_mode`** — single string per the three options. Mode-switching after build requires rewriting the file.
- **`ann.<band>`** — present iff `bands.<band>.passage_count > ANN_THRESHOLD_N` (=5000). Phase 1 #14 audit locked `{type: "hnsw", M: 32, efC: 200, efS: 128}`; the field stays declarative so a future ANN swap (Phase 7+) can record different params per band.
- **`build_config`** — `dict[str, Any]` for forward-compat. Phase 2 audits 03 + 05 will populate the canonical chunker / format keys; future build-time options land here without bumping the format.

## bands/*.npz format

Each band is a single NPZ archive with one array under key `"embeddings"`:

```python
np.savez_compressed(
    "bands/base.npz",
    embeddings=base_embeddings,        # (N, 768) float32, L2-normalised
)
```

The L2-normalisation invariant is enforced at write time (`store.bands.write_band` runs `_runtime_common.l2_normalize` defensively on a copy of the caller's array); it's not re-checked at load to keep the load path fast. Consumers can `assert np.allclose(np.linalg.norm(band, axis=1), 1.0, atol=1e-5)` if they want to verify.

The optimised W matrix lives in its own NPZ at `bands/optimised_W.npz` so `store.bands.load_optimised` can return `(band, W)` atomically — both are needed for query-time projection.

## passages.jsonl format

One JSON object per line, in `passage_idx` order:

```jsonl
{"char_length": 200, "char_offset": 0, "content_hash": "sha256:aaa", "id": "a3f1c2d4e5f6a7b8", "source_file": "src/a.py"}
{"char_length": 180, "char_offset": 200, "content_hash": "sha256:bbb", "id": "9b8a7c6d5e4f3210", "source_file": "src/a.py"}
{"char_length": 500, "char_offset": 0, "content_hash": "sha256:ccc", "id": "1122334455667788", "source_file": "src/b.md"}
```

`passage_idx` is **the line index**, not stored in the JSON. JSON keys are `sort_keys=True` for stable diffs across rebuilds.

`content_hash` is `sha256:<hex>` of the passage text at build time. Drift detection compares to current source hash.

`id` is the **stable passage identity** — `sha256(source_file + char_offset + char_length)[:16]` — that survives `rlat refresh` / `rlat sync` deltas. A passage that changes content but keeps its source-file slice keeps the same `id`; a passage that moves within a file (different `char_offset`) gets a new `id`. Audit 07 §"Identity decision" for the design rationale; without stable ids, deletes shift every later `passage_idx` and silently break verified-retrieval citation chains, `corpus_diff` continuity, and any consumer that bookmarks `passage_idx`.

**Forward-compat** — archives written before the `id` field shipped (legacy v4) load through `registry.compute_id`, which derives the same value from the stored coordinates. Either format reads cleanly; v4.1+ archives emit the field explicitly so consumers don't have to recompute.

## ann/*.faiss format

FAISS `IndexHNSWFlat` written via `faiss.write_index`. Loaded via `faiss.read_index` — FAISS records `dim` and `space` in the file, so Phase 2's store layer doesn't need to pass them.

The `.faiss` filename extension distinguishes them from the deprecated v0.11 `.hnsw` files (which were hnswlib-format). Phase 1 #14 audit locked FAISS; the field/ann.py API surface is library-agnostic so a future swap is one-file change.

## source/ tree (bundled mode only)

When `store_mode == "bundled"`, source files are zstd-framed and stored under `source/` inside the ZIP. Layout is **flat** — files are keyed by their `source_file` value from `passages.jsonl`, with directory separators preserved as ZIP path components.

`local` and `remote` modes leave `source/` empty (or absent). The mode determines where `Store.fetch(source_file, char_offset, char_length)` reads from.

## Forward compatibility

The format admits extension without a version bump in two specific ways:

1. **New top-level keys** under `metadata.build_config` and `metadata.ann` are typed `dict[str, Any]` — readers preserve them on round-trip.
2. **New band slots** (e.g. a future `lexical` or `quantised` band) just appear as new keys under `metadata.bands` + new files under `bands/`. Readers branch on dict presence; absent bands are skipped.

Anything that breaks the layout above (rename a top-level dir, change band file format, swap JSON for protobuf) bumps `format_version` to 5 and ships with explicit migration tooling. v0.11 → v2 has no migration because no v1-era models reached users.

## Cross-references

- Store-layer technical reference: [STORE.md](./STORE.md).
- Field-layer technical reference: [FIELD.md](./FIELD.md) (encoder + retrieval pipeline that produces / consumes these bands).
- Audit 03 format methodology: [audits/03_format_ann_chunking.md](./audits/03_format_ann_chunking.md).
- Audit 05 chunking methodology: same file.
