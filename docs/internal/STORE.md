# Store Layer — Technical Reference

The store layer is the **authoritative content** half of the v2.0 three-layer thesis (field → store → no reader). The field layer routes a query to a list of `(passage_idx, score)` tuples; the store layer turns each `passage_idx` back into source text the consumer can use, with verification that the source hasn't drifted from what was indexed.

> Source-of-truth code: `src/resonance_lattice/store/`.
> Format spec: [`docs/internal/KNOWLEDGE_MODEL_FORMAT.md`](./KNOWLEDGE_MODEL_FORMAT.md).
> User-facing modes: [`docs/user/STORAGE_MODES.md`](../user/STORAGE_MODES.md).
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
    format_version: int = 4
    kind: Literal["corpus", "intent"] = "corpus"
    backbone: BackboneInfo
    bands: dict[str, BandInfo]
    store_mode: Literal["bundled", "local", "remote"] = "local"
    ann: dict[str, Any]
    build_config: dict[str, Any]
    created_utc: str
    rlat_version: str = "2.0.0a1"
```

`backbone.revision` is the pinned HF commit hash from `install.encoder` — guarantees a knowledge model retrieved via a different revision of the encoder fails loud rather than producing silently-misaligned scores.

`bands.optimised.w_shape` is the MRL projection shape (`(d_native, backbone_dim)`, e.g. `(512, 768)` for gte-mb). JSON has no tuple type, so `from_json` restores the `tuple` invariant from the JSON array on parse.

`build_config` and `ann` are typed `dict[str, Any]` for forward-compat — Phase 2 audits 03 + 05 will flesh these out without bumping the format version.

### `PassageCoord` (`store.registry`)

```python
@dataclass(frozen=True)
class PassageCoord:
    passage_idx: int
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str
```

This is **the canonical type** that `field.dense.search`'s `registry` parameter is currently structurally-typed against (Phase 1 used a `Sequence` with duck-typed `.source_file` / `.char_offset`). Phase 3's `cli/build.py` builds the `passages.jsonl` from chunker output; Phase 6 RQL `cite` / `evidence` ops consume it.

### `passages.jsonl` format

One JSON object per line, in `passage_idx` order:

```jsonl
{"char_length": 200, "char_offset": 0, "content_hash": "sha256:aaa", "source_file": "src/a.py"}
{"char_length": 180, "char_offset": 200, "content_hash": "sha256:bbb", "source_file": "src/a.py"}
```

`passage_idx` is **not** stored — it's the line index. `write_jsonl` validates the input list is contiguous-from-0 before serialising; `load_jsonl` does **not** silently skip blank lines (a mid-file blank would renumber every downstream passage and break the `passage_idx ↔ band row` join), so it raises `JSONDecodeError` instead. Standard line iteration over a file or `splitlines()` produces no trailing empties, so well-formed archives parse cleanly.

`content_hash` is `sha256:<hex>` of the passage text at build time. The drift check (Phase 2 #23) compares the source's current hash against this stored value to detect post-build edits.

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
    def __init__(self, manifest_path, cache_dir, opener=_default_opener): ...
    def _read_full_text_uncached(source_file): ...   # only override
    def freshness() -> dict[str, DriftStatus]: ...   # Phase 3 stub
    def sync() -> None: ...                          # Phase 3 stub
```

HTTP-backed; the `.rlat` carries a manifest (`{source_file: {url, sha256}}`) recorded at build time so query never trusts the network unconditionally. First access to a file downloads to a persistent on-disk cache; the SHA-pin is re-verified before every read so cached bytes that drifted post-write fail loud rather than silently serving wrong text.

Atomic cache writes (tmp + `Path.replace`) so a crash mid-download doesn't leave a half-file that future reads would trust. The `urllib.request.urlopen` opener is wrapped with a default 30-second timeout so a stalled upstream can't hang `rlat search` indefinitely. The `opener` parameter is injectable for tests — production uses `_default_opener`, tests substitute a callable returning `BytesIO`.

**Remote mode ships in v2.0** (Audit 07 promoted it from deferred — see [`audits/07_incremental_sync.md`](audits/07_incremental_sync.md)). `rlat build --store-mode remote --remote-url-base <prefix>` writes a remote-mode archive; `rlat freshness` is the read-only drift check; `rlat sync` is the incremental delta-apply reconciler. Both `freshness` and `sync` route through `RemoteIndex` (`HttpManifestIndex` for catalog-mode + poll-mode) and land on the shared `store/incremental.py` delta-apply pipeline as local-mode `rlat refresh`. The codex P0 manifest-only-sync mode is statically impossible — `apply_delta` requires the encoder, and the only manifest-write path is `apply_delta`. End-to-end harness coverage at `tests/harness/incremental_sync.py` (6 hermetic guarantees, no live network). `GitHubCompareIndex` (uses GitHub's compare API for git-hosted corpora) is deferred to v2.1.

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

def verify_hits(hits: list[tuple[int, float]], store: Store,
                registry: list[PassageCoord]) -> list[VerifiedHit]: ...
def filter_verified(hits: list[VerifiedHit]) -> list[VerifiedHit]: ...
```

`verify_hits` is the glue between `field.dense.search` (which returns `(passage_idx, score)` tuples) and the `Store` Protocol (which resolves coordinates to text + drift status). Each hit goes through `store.verify` (re-hashes the slice against the build-time `content_hash`) then `store.fetch` (returns current authoritative text — skipped on `"missing"`, where text becomes the empty string so the row stays in output). The base-class text cache amortises both calls to a single full-file read per source file.

Output is in **input order** — `dense.search` already returns descending-by-score, so threading `search → verify_hits → filter_verified` preserves rank order naturally. Re-sorting is the caller's responsibility, not this layer's.

Drift status:

- `"verified"` — current slice hashes to the stored value.
- `"drifted"` — source exists but the slice hash has changed since build.
- `"missing"` — source file no longer exists (text field is `""`).

`rlat search --verified-only` filters via `filter_verified`. The default surface returns hits regardless and exposes `drift_status` on each so the consumer can decide. An out-of-range `passage_idx` raises `IndexError` — that means hits from a different knowledge model's registry, which is a programming error, not a runtime drift.

## ZIP archive orchestrator (`store.archive`)

`store.archive` owns the on-disk container. The format itself is documented in [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md); this section is the **API contract** consumed by the build / optimise / search code paths.

### Public surface

```python
@dataclass
class ArchiveContents:
    metadata: Metadata
    registry: list[PassageCoord]
    bands: dict[str, np.ndarray]               # eager (search hot path)
    projections: dict[str, np.ndarray]         # optimised W per band; empty for base-only
    ann_blobs: dict[str, bytes]                # raw FAISS bytes; deserialised by field/ann

def read(path) -> ArchiveContents: ...
def write(path, metadata, bands, registry, projections=None, ann_blobs=None,
          source_files=None) -> None: ...
def write_band_in_place(path, band_name, band_info, band_data,
                        projection=None, ann_blob=None) -> None: ...
```

### `FORMAT_VERSION` is single-sourced

The literal `FORMAT_VERSION = 4` lives in `store.metadata`; `store.archive` re-imports it and `Metadata.format_version` defaults to it. A v5 bump is a one-line change in `metadata.py` and every version-mismatch check stays consistent.

### Atomic write (`write`)

A fresh archive is written to `<path>.tmp` and then `os.replace`-d onto `<path>`. A crash mid-write leaves the original (or absence) untouched. The `.tmp` file is `unlink`-ed on any exception so we don't accumulate orphans on disk-full / SIGKILL.

The outer ZIP uses `ZIP_STORED` (no compression) — NPZ files are already deflate-compressed and ZIP-on-ZIP wastes CPU. Internal ANN bytes are passed through verbatim (FAISS chooses its own on-disk layout via `faiss.serialize_index`).

### Eager band load + lazy source

`read()` eagerly slurps all declared bands and ANN blobs into RAM because retrieval requires the full `(N, D)` matrix resident; lazy bands would make the first query unboundedly slow. Source files (`source/` tree, bundled mode only) are **not** loaded — Store classes (`bundled` / `local` / `remote`) reopen the ZIP and resolve `source_file` keys on demand, so a 50K-passage knowledge model with 1 GB of source/ pays only the ~200 MB band cost up front.

### Streaming in-place band write (`write_band_in_place`)

`rlat optimise` adds a `bands/optimised.npz` + `bands/optimised_W.npz` (and optional `ann/optimised.faiss`) to a knowledge model whose base band was built earlier. ZIP archives don't support member mutation, so the implementation rewrites: read every preserved member → write to a tmp ZIP → `os.replace`. The "write every preserved member" step uses `shutil.copyfileobj` with a 1 MB buffer (not `zf.read(name)`) so peak RSS stays bounded by the buffer regardless of source-file size — critical for bundled-mode archives that may aggregate >1 GB of source. The metadata's `bands.<name>` entry is updated to the new `BandInfo` before the rewrite; everything else (other bands, source/, other ANN, registry, build_config, `_extras`) round-trips untouched.

Forward-compat: any ZIP member archive.py doesn't recognise — future band slots, lexical sidecars, alternative ANN files — is preserved bit-for-bit through `write_band_in_place`. `metadata.json` round-trips its top-level and per-band `_extras` dicts, so an older `archive.py` rewriting a newer-format archive doesn't drop unknown keys.

## Open Phase 2 questions

- **Audit 03**: format benchmark (ZIP+JSON+NPZ vs single binary vs HDF5). Methodology in `audits/03_format_ann_chunking.md` §Audit 03. Runs in parallel with the implementation work; default proceeds unless the gate fails.
- **Audit 05**: chunking defaults benchmark (200-3200 chars vs alternatives). Same parallelism.

## Cross-references

- Field-layer doc: [FIELD.md](./FIELD.md).
- User-facing storage modes guide: [`docs/user/STORAGE_MODES.md`](../user/STORAGE_MODES.md).
- Format spec: [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md).
- Audit methodology: [audits/03_format_ann_chunking.md](./audits/03_format_ann_chunking.md).
