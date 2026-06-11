# Store Layer — Technical Reference

The store layer is the **authority** layer of the three-layer thesis (field routes → store serves authoritative text → grounded synthesis builds answers on top; see [GROUNDING_MODEL.md](./GROUNDING_MODEL.md)). The field layer routes a query to a list of `(passage_idx, score)` tuples; the store layer turns each `passage_idx` back into source text the consumer can use, with verification that the source hasn't drifted from what was indexed.

> Source-of-truth code: `src/resonance_lattice/store/`.
> Format spec: [`docs/internal/KNOWLEDGE_MODEL_FORMAT.md`](./KNOWLEDGE_MODEL_FORMAT.md).
> User-facing modes: [`docs/site/storage-modes.html`](../site/storage-modes.html).
> Specification: [`base-first-rebuild.md §2`](../../.claude/plans/base-first-rebuild.md).

## Layer overview

```
field.dense.search(...) → [(passage_idx, score), ...]
                                  │
                                  ▼
                store.registry.PassageCoord[passage_idx]
                                  │
                          (source_file, char_offset, char_length, content_hash)
                                  │
                                  ▼
                          store.<mode>.fetch(...)
                                  │
                                  ▼
                           authoritative text
```

The mode (`bundled` / `local` / `remote`) is recorded in `metadata.json` and resolved at knowledge-model load time, not at query time.

## Foundational types

### `Metadata` (`store.metadata`)

Dataclass mirror of `metadata.json`. Round-trip via `to_json` / `from_json`.

```python
@dataclass
class Metadata:
    format_version: int = FORMAT_VERSION       # 4 — single-sourced in store.metadata
    kind: Literal["corpus", "intent"] = "corpus"
    backbone: BackboneInfo
    bands: dict[str, BandInfo]
    store_mode: Literal["bundled", "local", "remote"] = "local"
    ann: dict[str, Any]
    build_config: dict[str, Any]
    created_utc: str
    rlat_version: str = "2.0.0a1"
    insight_layer_last_reverify_utc: str = ""  # heartbeat stamped by `rlat reverify`
    _extras: dict[str, Any]                    # unknown top-level keys, round-tripped
```

`backbone.revision` is the pinned HF commit hash from `install.encoder` — guarantees a knowledge model retrieved via a different revision of the encoder fails loud rather than producing silently-misaligned scores.

`bands` holds one entry per band keyed by name. `base` (role `retrieval_default`, 768d, CLS+L2) is always present and is the only retrieval band. An optional `insight` band (role `insight_layer`, 768d) carries the per-corpus insight/claims layer when one has been promoted. Each `BandInfo` records `role`, `dim`, `l2_norm`, `passage_count` (plus `_extras` for forward-compat round-trip).

`build_config` and `ann` are typed `dict[str, Any]` for forward-compat. Builds populate `ann` with the locked HNSW params per band (`{type, M, efConstruction, efSearch}`) and `build_config` with chunker provenance (`chunker`, `row_mode`, `min_chars`, `max_chars`, `passage_count`, `file_count`, `source_root`, `batch_size`) — what `rlat refresh` needs to replay a build faithfully; deltas update only the live counts. Unknown keys, top-level and per-band, round-trip via `_extras` so a newer writer's keys survive an older reader's rewrite.

### `PassageCoord` (`store.registry`)

```python
@dataclass(frozen=True)
class PassageCoord:
    passage_idx: int
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str
    passage_id: str          # stable identity — survives refresh/sync renumbering
    key: str | None = None   # business key — row-mode (slicer) builds only
```

This is **the canonical type** that `field.dense.search`'s `registry` parameter is structurally-typed against (a `Sequence` with duck-typed `.source_file` / `.char_offset`). `build/pipeline.py` builds the `passages.jsonl` from chunker output — or one passage per walker entry in row mode, where `passage_id` pins to the business `key` via `compute_key_id` so a row keeps its identity across text edits. Elsewhere `passage_id` is `compute_id(source_file, char_offset, char_length)` (16-hex SHA-256 slice): a passage that changes content but keeps its slice keeps its id — the pivot `rlat refresh` / `rlat sync` deltas reorder on. RQL `cite` / `evidence` ops consume the registry.

### `passages.jsonl` format

One JSON object per line, in `passage_idx` order:

```jsonl
{"char_length": 200, "char_offset": 0, "content_hash": "sha256:aaa", "id": "a3f1c2d4e5f6a7b8", "source_file": "src/a.py"}
{"char_length": 180, "char_offset": 200, "content_hash": "sha256:bbb", "id": "9b8a7c6d5e4f3210", "source_file": "src/a.py"}
```

`passage_idx` is **not** stored — it's the line index. `write_jsonl` validates the input list is contiguous-from-0 before serialising; `load_jsonl` does **not** silently skip blank lines (a mid-file blank would renumber every downstream passage and break the `passage_idx ↔ band row` join), so it raises `JSONDecodeError` instead. Standard line iteration over a file or `splitlines()` produces no trailing empties, so well-formed archives parse cleanly.

`content_hash` is `sha256:<hex>` of the passage text at build time. The drift check compares the source's current hash against this stored value to detect post-build edits. `id` is the stable passage identity (`write_jsonl` emits it; legacy archives without it load through `compute_id`, deriving the same value). Row-mode builds also emit `key` — the per-row business key; `write_jsonl` omits the slot entirely for chunked corpora, so their on-disk bytes are unchanged by the field's existence.

## Store modes (Phase 2 #20-#22)

Three modes; one rule per knowledge model. Recorded in `metadata.store_mode`.

| Mode | Source location | When to use |
|---|---|---|
| `bundled` | inside the .rlat ZIP under `source/` (zstd-framed) | self-contained models; CI artefacts; offline / archival |
| `local` | filesystem at `--source-root` relative paths | the default; live-edit workflow; corpus on disk |
| `remote` | HTTP(S) URL with SHA-pinned manifest + lockfile | published knowledge models; team-shared corpora |

All three subclass `store.base.Store` (an ABC, not a Protocol — concrete `fetch`/`verify`/cache live on the base; subclasses implement only one primitive):

```python
class Store(ABC):
    def __init__(self): self._text_cache: dict[str, str] = {}

    @abstractmethod
    def _read_full_text_uncached(self, source_file: str) -> str: ...

    # concrete:
    def _read_full_text(...): ...               # cached wrapper
    def fetch(source_file, char_offset, char_length) -> str: ...
    def verify(source_file, char_offset, char_length, expected_hash) -> DriftStatus: ...
```

The "ABC with concrete fetch/verify" shape replaces the earlier per-store duplication (rule-of-three crossed when the third store landed; the simplify-skill review flagged ~40 LOC of identical method bodies across bundled/local/remote). Subclasses provide only the per-mode read primitive — `BundledStore` opens the ZIP and decompresses; `LocalStore` reads from disk via `_resolve_safe`; `RemoteStore` fetches via `_fetch_bytes` with SHA-pin verification. The cache, slice, and hash-compare are written once.

Both `fetch` and `verify` take the **same passage triple** (source_file, char_offset, char_length). `verify` re-hashes that range and compares to the per-passage `content_hash` recorded on `PassageCoord` at build time. Per-passage (not per-file) hashing is intentional — a one-line edit elsewhere in a 5K-line file shouldn't mark every passage in the file as drifted; only the passages whose char-range actually changed.

`DriftStatus = Literal["verified", "drifted", "missing"]` and `compute_hash(text)` (canonical `sha256:<hex>` of utf-8) both live in `store.base` as the single source of truth and are re-exported from `store.verified` for callers that already import that module.

Cross-cutting code (CLI, RQL `cite` / `evidence`) goes through `Store` — the consumer never sees the mode.

## Local store (`store.local`)

```python
class LocalStore(Store):
    def __init__(self, source_root: str | Path): ...
    def _read_full_text_uncached(source_file): ...   # only override
```

The default mode for the live-edit workflow: corpus stays on disk, the knowledge model only records `(source_file, char_offset, char_length)` pointers. The base-class text cache amortises across "fetch+verify on N hits from the same source file" — each file reads once, not 2N times.

Path-traversal hazard: `source_file` keys come from inside the .rlat (a tampered archive could inject `"../../etc/passwd"`). `_resolve_safe` resolves both `source_root / source_file` and `source_root` and rejects any target that isn't `relative_to` the root. Symlinks are followed (a within-root symlink that points outside the root is correctly rejected).

Newline handling is **invariant-coupled**: build and verify must both read via `Path.read_text()` (universal-newlines mode, `\r\n`/`\r` → `\n`). If the build pipeline ever switches to `read_bytes()` or `open(..., newline="")`, recorded hashes diverge across platforms.

## Bundled store (`store.bundled`)

```python
def pack_source_files(files: dict[str, str]) -> dict[str, bytes]: ...
class BundledStore(Store):
    def __init__(self, zip_path: str | Path): ...
    def _read_full_text_uncached(source_file): ...   # only override
```

`pack_source_files` is the build-side helper: it zstd-frames each text file independently and returns a `dict[str, bytes]` ready to pass through to `archive.write(source_files=...)`. Per-file framing (not a shared dictionary) is deliberate — random-access `fetch` of one file shouldn't depend on having decompressed any other file first. The space cost vs a shared dict is small for source-code corpora because individual files compress well in isolation.

The read primitive re-opens the ZIP per call (no held handle) and decompresses the full source file. The decompressor instance is module-level (`_DCTX`) since `ZstdDecompressor.decompress(blob)` is reusable for one-shot calls. The base class's text cache means each file is decompressed once per query regardless of how many hits land in it.

The `source/` ZIP prefix is owned by `store.archive.SOURCE_DIR`; bundled.py imports it so the layout has exactly one source of truth.

## Remote store (`store.remote`)

```python
class RemoteStore(Store):
    def __init__(self, manifest, cache_dir, opener=None): ...  # manifest = parsed {source_file: {url, sha256}}
    def _read_full_text_uncached(source_file): ...   # read primitive + SHA-pin check
    def fetch_all(source_files): ...                 # parallel-download override (8 threads)
    def freshness() -> dict[str, DriftStatus]: ...   # read-only upstream-vs-pin poll
```

There is no `sync()` method — reconciliation lives in `cli/maintain.cmd_sync` + `store/remote_index.py`, landing on `store/incremental.apply_delta`. `freshness()` walks every manifest entry, downloads, hashes, and compares to the pin (`verified` / `drifted`; network errors → `missing`) without touching the cache or the manifest. Cache files are keyed on `(source_file, expected_sha)`, so a sync that re-pins a path to a new SHA naturally falls through to a fresh download — old cached bytes are inert, never trusted.

```python
def compose_manifest(source_files, url_base) -> dict[str, dict[str, str]]: ...
# single owner of the {source_file: {url, sha256}} contract — shared by
# `rlat build --store-mode remote` and `rlat convert --to remote`
```

HTTP-backed; the `.rlat` carries a manifest (`{source_file: {url, sha256}}`) recorded at build time so query never trusts the network unconditionally. First access to a file downloads to a persistent on-disk cache; the SHA-pin is re-verified before every read so cached bytes that drifted post-write fail loud rather than silently serving wrong text.

Atomic cache writes (tmp + `Path.replace`) so a crash mid-download doesn't leave a half-file that future reads would trust. The `urllib.request.urlopen` opener is wrapped with a default 30-second timeout so a stalled upstream can't hang `rlat search` indefinitely. The `opener` parameter is injectable for tests — production uses `_default_opener`, tests substitute a callable returning `BytesIO`.

**Remote mode ships in v2.0** (Audit 07 promoted it from deferred — see [`audits/07_incremental_sync.md`](audits/07_incremental_sync.md)). `rlat build --store-mode remote --remote-url-base <prefix>` writes a remote-mode archive; `rlat freshness` is the read-only drift check; `rlat sync` is the incremental delta-apply reconciler. Both `freshness` and `sync` route through `RemoteIndex` (`HttpManifestIndex` for catalog-mode + poll-mode) and land on the shared `store/incremental.py` delta-apply pipeline as local-mode `rlat refresh`. The codex P0 manifest-only-sync mode is statically impossible — `apply_delta` requires the encoder, and the only manifest-write path is `apply_delta`. End-to-end harness coverage at `tests/harness/incremental_sync.py` (6 hermetic guarantees, no live network). `GitHubCompareIndex` (uses GitHub's compare API for git-hosted corpora) is deferred to v2.1.

## Incremental delta-apply (`store.incremental` — Audit 07)

The shared reconciliation pipeline both `rlat refresh` (local disk walk) and `rlat sync` (`RemoteIndex` poll) land on. `bucketise` classifies every candidate passage against the old registry on stable `passage_id`: id + hash match → `unchanged` (band row lifted, no re-encode); id matches, hash differs → `updated`; new id → `added`; id absent from live → `removed`. `apply_delta` then encodes only `updated + added`, composes the new band (kept rows + new rows), renumbers `passage_idx` line-implicitly (ids stay stable, so external references survive), rebuilds ANN when N crosses the threshold, and lands one atomic `archive.write`.

Two invariants (Audit 07, codex P0): the `encoder` parameter is **required** — no manifest-only path can update the archive without re-encoding deltas; and every kept passage's `content_hash` is validated against live bytes during bucketise. Detection is content-hash-based, never mtime — a touched-but-unchanged file re-encodes nothing. The insight layer and the telemetry member are explicitly passed through the rewrite. Bundled-mode archives are rejected (source bytes are baked in at build time — route to `rlat build`).

## Storage-mode conversion (`store.conversion` — Audit 08)

`rlat convert` reshapes a knowledge model between `bundled` / `local` / `remote` without rebuilding embeddings — bands, registry, and ANN bytes flow through unchanged; only `metadata.store_mode` plus the supporting payload (`source/` for bundled, `manifest.json` for remote, neither for local) changes, atomically with the metadata swap. Every passage's `content_hash` is re-validated against bytes resolved via the *source* mode's Store first; any drift aborts with `ConversionDriftError` (reconcile via `rlat refresh` / `rlat sync`, then convert). Row-mode (slicer) models convert only **to** `bundled` — row text has no on-disk source to point at.

## Verified retrieval (`store.verified` — WS3 #292 port)

```python
@dataclass(frozen=True)
class VerifiedHit:
    passage_idx: int
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str
    drift_status: DriftStatus
    score: float
    text: str
    key: str | None = None              # business key — row-mode models only
    layer: Literal["source"] = "source"
```

`InsightHit` is the sibling type for the insight layer — structurally distinct on purpose (source and insight must look different at every output surface), with `drift_status` derived from claim state (active → verified, stale/candidate → drifted, retired → missing) so `filter_verified` works on a mixed list. `verify_insight_hits` renders corpus claims (score = cosine × confidence factor); `rank_insight_band` and `serve_band_attributes` are the recall-time band readers. The `layer` field lets consumers filter without isinstance checks.

```python
def verify_hits(hits: list[tuple[int, float]], store: Store,
                registry: list[PassageCoord]) -> list[VerifiedHit]: ...
def filter_verified(hits) -> list: ...   # uniform across source + insight layers
```

`verify_hits` is the glue between `field.dense.search` (which returns `(passage_idx, score)` tuples) and the `Store` ABC (which resolves coordinates to text + drift status). Each hit goes through `store.verify` (re-hashes the slice against the build-time `content_hash`) then `store.fetch` (returns current authoritative text — skipped on `"missing"` and `"drifted"`, where text becomes the empty string so the row stays in output without serving bytes that no longer match the recorded hash). A `RemoteShaMismatch` raised during the fetch itself is demoted to drifted+empty rather than crashing the retrieval path. The base-class text cache amortises both calls to a single full-file read per source file.

Output is in **input order** — `dense.search` already returns descending-by-score, so threading `search → verify_hits → filter_verified` preserves rank order naturally. Re-sorting is the caller's responsibility, not this layer's.

Drift status:

- `"verified"` — current slice hashes to the stored value.
- `"drifted"` — source exists but the slice hash has changed since build.
- `"missing"` — source file no longer exists (text field is `""`).

`rlat search --verified-only` filters via `filter_verified`. The default surface returns hits regardless and exposes `drift_status` on each so the consumer can decide. An out-of-range `passage_idx` raises `IndexError` — that means hits from a different knowledge model's registry, which is a programming error, not a runtime drift.

## ZIP archive orchestrator (`store.archive`)

`store.archive` owns the on-disk container. The format itself is documented in [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md); this section is the **API contract** consumed by the build / search code paths.

### Public surface

```python
@dataclass
class ArchiveContents:
    metadata: Metadata
    registry: list[PassageCoord]
    bands: dict[str, np.ndarray]               # eager (search hot path)
    ann_blobs: dict[str, bytes]                # raw FAISS bytes; deserialised by field/ann
    remote_manifest: dict[str, dict[str, str]] # parsed manifest.json (remote mode); else {}
    insights: list[Claim]                      # loaded insight layer; [] when none promoted
    source_path: Path | None                   # resolved .rlat path — the corpus identity telemetry keys by
    def select_band(prefer=None) -> BandHandle: ...   # base is the only retrieval band
    def insight_band() -> BandHandle | None: ...

def read(path, *, defer_base_band=False) -> ArchiveContents: ...
def read_insight_layer(path) -> tuple[list[Claim], np.ndarray] | None: ...
```

`read(defer_base_band=True)` skips materialising the base band when an ANN index serves it — ANN-mode `field.retrieve` never dereferences `handle.band`, so a memory-bounded host (the Fabric UDF) avoids holding the full (N, 768) matrix. `read_insight_layer` opens the ZIP and reads only `insight.jsonl` + `bands/insight.npz` for the prompt-time band-recall path. `read` validates the insight join on load: `insight.jsonl` with no declared insight band, or a row/band count mismatch, raises — a half-promoted archive fails loud.

```python
def write(path, metadata, bands, registry, ann_blobs=None, source_files=None,
          remote_manifest=None, insights=None, self_audit=None,
          telemetry=None) -> None: ...
def write_band_in_place(path, band_name, band_info, band_data,
                        ann_blob=None) -> None: ...
def write_insight_layer_in_place(path, insights, insight_band, ann_blob=None,
                                 *, mark_reverified_utc=None) -> None: ...
def read_telemetry(path) -> list[dict]: ...        # [] when no telemetry member
def append_telemetry_in_place(path, rows) -> int: ...
def read_self_audit(path) -> dict: ...             # {} when absent; never raises
def write_self_audit_in_place(path, report) -> None: ...
```

### `FORMAT_VERSION` is single-sourced

The literal `FORMAT_VERSION = 4` lives in `store.metadata`; `store.archive` re-imports it and `Metadata.format_version` defaults to it. A v5 bump is a one-line change in `metadata.py` and every version-mismatch check stays consistent.

### Atomic write (`write`)

A fresh archive is written to a per-writer-unique tmp path
(`<path>.<pid>.<rand>.tmp` via `_unique_tmp_path`) and then
`os.replace`-d onto `<path>`. A crash mid-write leaves the original (or
absence) untouched. The `.tmp` file is `unlink`-ed on any exception so
we don't accumulate orphans on disk-full / SIGKILL.

The same per-writer-unique tmp scheme protects `write_insight_layer_in_place`
and `write_band_in_place`. Two concurrent OS processes mutating the same
archive — e.g. `rlat watch` running in the background and a manual
`rlat reverify` — each write to their own tmp file, so no in-flight tmp
collision tears the ZIP and reads of the on-disk archive remain
consistent (the only mutator is `os.replace`, which is atomic).
*Lost updates* are not prevented: both writers read the same pre-state,
and one's `os.replace` lands after the other, dropping the loser's
delta. Callers running overlapping mutators must serialise themselves;
`cli/watch.py`'s per-archive `threading.Lock` is the canonical example.

The outer ZIP uses `ZIP_STORED` (no compression) — NPZ files are already deflate-compressed and ZIP-on-ZIP wastes CPU. Internal ANN bytes are passed through verbatim (FAISS chooses its own on-disk layout via `faiss.serialize_index`).

### Eager band load + lazy source

`read()` eagerly slurps all declared bands and ANN blobs into RAM because retrieval requires the full `(N, D)` matrix resident; lazy bands would make the first query unboundedly slow. Source files (`source/` tree, bundled mode only) are **not** loaded — Store classes (`bundled` / `local` / `remote`) reopen the ZIP and resolve `source_file` keys on demand, so a 50K-passage knowledge model with 1 GB of source/ pays only the ~200 MB band cost up front.

### Insight, telemetry, and self-audit members

Beyond the format-spec slots, `store.archive` owns three insight-engine members:

- **`insight.jsonl` + `bands/insight.npz` (+ optional `ann/insight.faiss`)** — the promoted insight layer: one corpus `Claim` per line, positionally joined to the band rows. Written whole by `write_insight_layer_in_place`; empty `insights` clears the layer and deregisters the band. `mark_reverified_utc` stamps the metadata heartbeat `rlat profile` surfaces.
- **`insight/telemetry.jsonl`** — append-only capture log: one redacted `field.capture` observation per line (a query *fingerprint* + per-rank scores, never query text). `append_telemetry_in_place` is a true byte-append — existing bytes carry through verbatim, never re-parsed (a torn final line is newline-healed first), so a foreign or future writer's rows survive this build's writer. `store.telemetry.flush` is the fold: drain the in-process capture ring, redact, append — at session boundaries, gated by `RLAT_CAPTURE_PERSIST` / `RLAT_DOGFOOD_SESSION`, so a bare one-shot search never surprise-rewrites the archive. The fold never raises; a failed write leaves the rows buffered for the next fold (peek, persist, then drain exactly what was written).
- **`insight/self_audit.json`** — the LLM-free corpus shape-report (`store.self_audit`): high-cosine cross-document *candidate* pairs for stance judgement (geometry guarantees same-topic only, so the pass is hard-capped — `per_row_cap` per passage plus a global `max_pairs` heap, and skipped entirely above 60K passages), demand gaps, and drift hints (demoted — `rlat refresh` clears corpus drift). Recomputed whole: `write_self_audit_in_place` replaces the member; builds fold the report into the single `archive.write` so no second rewrite is needed. `read_self_audit` returns `{}` on anything unreadable.

All three ride the same per-writer-unique tmp + `os.replace` atomicity, and `apply_delta` (refresh/sync) explicitly passes insights + telemetry through the rewrite so neither is silently dropped on drift reconciliation.

### Streaming in-place band write (`write_band_in_place`)

`write_band_in_place` adds or replaces a band slot — `bands/<name>.npz` (and optional `ann/<name>.faiss`) — on a knowledge model whose base band was built earlier; the insight band (`bands/insight.npz` + optional `ann/insight.faiss`) is the live use, filled on a model whose base band was built earlier. ZIP archives don't support member mutation, so the implementation rewrites: read every preserved member → write to a tmp ZIP → `os.replace`. The "write every preserved member" step uses `shutil.copyfileobj` with a 1 MB buffer (not `zf.read(name)`) so peak RSS stays bounded by the buffer regardless of source-file size — critical for bundled-mode archives that may aggregate >1 GB of source. The metadata's `bands.<name>` entry is updated to the new `BandInfo` before the rewrite; everything else (other bands, source/, other ANN, registry, build_config, `_extras`) round-trips untouched.

Forward-compat: any ZIP member archive.py doesn't recognise — future band slots, lexical sidecars, alternative ANN files — is preserved bit-for-bit through `write_band_in_place`. `metadata.json` round-trips its top-level and per-band `_extras` dicts, so an older `archive.py` rewriting a newer-format archive doesn't drop unknown keys.

## Streaming row-mode serve (`store.streaming`)

The OOM-safe serve path for a large row-mode corpus in a memory-bounded host (the Fabric UDF worker — loading the band or a FAISS index peaks ~1.5 GB on the 94k slicer corpus, over the worker ceiling). Two paths share one ranking contract — raw cosine descending, ties broken by key — so they agree row-for-row:

- `stream_topk` — the low-memory primitive: reads `bands/base.npz` row-chunk by row-chunk straight out of the `.rlat` (the outer ZIP is `ZIP_STORED`, so the member is seekable; the inner `.npy` is DEFLATE, read sequentially) keeping only a top-k heap. Exact full scan, no approximation; peak resident ≈ one 2048-row chunk + the heap.
- `materialize_band` + `topk_over_band` — the fast serve: decompress the band once on cold start to an uncompressed `/tmp` `.npy` (atomic tmp + replace), `mmap` it, then every warm query is one GEMV + `argpartition` (~ms on 94k).

Both return `[(key, score)]` for keyed rows only — chunked corpora yield nothing. `read_keys` loads the row-aligned business keys from `passages.jsonl` (cached once per worker; ~0.5 s to parse 94k lines). `SourceSnippets` holds the ZipFile open to serve per-key zstd-framed source text (~µs per read from an open handle vs ~236 ms per re-open, measured), degrading to keys-only on any failure. No sidecar, no FAISS, no SQL — everything reads from the one `.rlat`.

## Promotion gate (`store.promotion` + `store.faithfulness`)

How earned insight enters the archive. `promote_candidates` runs each synthesis candidate through the compression test against the source band + the running insight layer, hands passers to the claim-lifecycle spine (`consolidate_corpus` — only a claim the spine transitions to `active` is written), and lands all survivors in one atomic `write_insight_layer_in_place`: every survivor lands together or none. Re-running with the same candidate is idempotent — a content fingerprint already in the layer is skipped, not duplicated.

`promote_if_faithful` is the autonomous entry: a deep-search answer is gated by `assess_faithfulness` on two axes — `claim_support` (fraction of atomic claims entailed by a cited passage, floor 0.8) and `question_relevance` (floor 0.6); the single faithfulness score is the *weaker* axis. With `client=None` (the free agent/human path) the LLM gate is skipped, but an explicit caller `faithfulness` score is required — no silent unverified write — and every downstream guard still applies: citations must carry `passage_id` + `content_hash`, the compression test gates the write, the spine holds its trust floor. Faithfulness judges grounding, not truth — whether the answer honestly rests on its passages ([GROUNDING_MODEL.md](./GROUNDING_MODEL.md)).

## Open audits

- **Audit 03**: format benchmark (ZIP+JSON+NPZ vs single binary vs HDF5). Methodology in `audits/03_format_ann_chunking.md` §Audit 03. The benchmark never ran; the default shipped in v2.0 and is the live format.
- **Audit 05**: chunking defaults benchmark (200-3200 chars vs alternatives). Never ran; the defaults shipped (`store/chunker.py` locks `min_chars=200`, `max_chars=3200`).

## Cross-references

- Field-layer doc: [FIELD.md](./FIELD.md).
- User-facing storage modes guide: [`docs/site/storage-modes.html`](../site/storage-modes.html).
- Format spec: [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md).
- Audit methodology: [audits/03_format_ann_chunking.md](./audits/03_format_ann_chunking.md).
- Failure-mode contract for archive I/O (atomic write, concurrent writers, half-promotion handling): [FAILURE_MODES.md §"Archive I/O"](./FAILURE_MODES.md).
