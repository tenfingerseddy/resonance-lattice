# Storage Modes

A knowledge model points its passages back at source files. **Where those source files live** is the storage mode — and the mode is baked into the `.rlat` at build time.

Three modes ship in v2.0:

| Mode | Source lives in | Pick when |
|---|---|---|
| **bundled** | inside the `.rlat` (zstd-framed) | shipping a self-contained artefact (HF Hub, archive, offline) |
| **local** *(default)* | filesystem at `--source-root` | you're indexing a corpus on disk you'll edit |
| **remote** | HTTP(S) URLs with SHA-pin manifest | source lives at canonical URLs (e.g. a tagged GitHub release) |

You pick the mode at build time with `--store-mode`. To switch modes later, use `rlat convert <km> --to {bundled|local|remote}` — reshapes the storage mode in place without re-running the encoder. Bands, registry, ANN, and the optimised W projection are preserved.

## Bundled

```bash
rlat build ./my-corpus -o my-corpus.rlat --store-mode bundled
```

Source files are zstd-compressed individually inside the `.rlat`. The result is one self-contained file. Drop it on HF Hub, mail it as an attachment, ship it in CI — there's nothing else to fetch.

| Pro | Con |
|---|---|
| One file, fully reproducible | Largest on disk (full source + bands) |
| Works offline | No live edits — rebuild to re-ingest |
| Drift can only happen via tampering with the `.rlat` itself | Bigger to push around |

When `rlat search` returns a hit from a bundled model, `drift_status` is almost always `verified`. A `drifted` or `missing` row means the `.rlat` was edited after build — investigate.

## Local *(default)*

```bash
# build
rlat build ./my-corpus -o my-corpus.rlat        # default --store-mode local

# query (source-root defaults to where you built from; override with --source-root)
rlat search my-corpus.rlat "..."
rlat search my-corpus.rlat "..." --source-root /different/path
```

The `.rlat` records `(source_file, char_offset, char_length)` triples and re-resolves them on disk at query time. The corpus stays in your filesystem; the `.rlat` just indexes it.

| Pro | Con |
|---|---|
| Smallest `.rlat` (no source bytes) | The source-root path is part of the runtime contract |
| Live edits — drift detection tells you what changed | Move/delete a file → that hit shows up as `missing` |
| Cheap to rebuild incrementally (`rlat refresh`) | Source isn't portable with the `.rlat` |
| Stays live as you edit (`rlat watch`) | Watch needs the `[watch]` extra (`pip install rlat[watch]`) |

`drift_status` tells the truth here. If you want to see only hits whose source matches the build:

```bash
rlat search my-corpus.rlat "..." --verified-only
```

To re-ingest drifted files in place, use `rlat refresh`. To keep the archive live as you edit, use `rlat watch` — same incremental pipeline, debounced by filesystem events. `rlat watch --once` is the synchronous CI / pre-commit shape.

## Remote

```bash
rlat build ./my-corpus -o my-corpus.rlat \
  --store-mode remote --remote-url-base https://example.com/corpus/v1
```

The `.rlat` carries a manifest (`manifest.json` at the top of the ZIP) mapping each `source_file` to a `{url, sha256}` pair, recorded at build time. URL = `<remote-url-base>/<source_file>` (forward-slash-joined). First time a query touches a passage, the source file downloads to a per-knowledge-model on-disk cache (default location: `~/.cache/rlat/remote/<km-sha>/`) and the SHA is pin-verified against the manifest. Subsequent reads hit the cache.

| Pro | Con |
|---|---|
| Source is centrally hosted — multiple consumers share it | First query is slower (downloads on miss) |
| SHA pin means a swapped-out URL fails loud, not silent | Needs network on first access |
| Survives `git rm` of local files | Cache verification on every read |

Default download timeout is 30 seconds per file — a stalled upstream can't hang `rlat search` indefinitely. SHA-pin mismatches raise an actionable error pointing at the cache file to delete; if upstream genuinely moved, run `rlat build --remote-url-base ...` again to regenerate against current content.

```bash
rlat freshness my-corpus.rlat   # read-only: are upstream SHAs still what we pinned?
```

`rlat freshness` walks every manifest entry, downloads the upstream bytes, hashes them, and reports per-entry status (`verified` / `drifted` / `missing`). Read-only; suitable for CI gating (exits non-zero when any drift detected). When freshness reports drift, run `rlat sync` — it lands on the same incremental delta-apply pipeline `rlat refresh` uses (`store/incremental.apply_delta`), bucketises on stable `passage_id`, re-encodes only the updated + added passages, re-projects the optimised band from the new base for free, and writes atomically. Manifest, bands, and registry stay internally consistent end-to-end.

## Cross-mode rules

- `rlat compare` and any RQL `compare` / `intersect` / `unique` op always use the **base band**, regardless of mode. Two knowledge models built in different modes can be compared as long as their base bands were produced by the same pinned encoder revision.
- Switching modes is `rlat convert <km> --to {bundled|local|remote}` (Audit 08) — preserves embeddings + drift hashes, no rebuild. See [CLI.md §rlat convert](CLI.md#rlat-convert-shipped-audit-08).
- Drift is per-passage, not per-file: editing one line in a 5K-line file marks only the passages whose char-range actually changed.

## Picking a mode

If you're not sure:

- **You're building a knowledge model for yourself, against your own working repo.** Use `local`. Get drift detection. Rebuild as the corpus evolves.
- **You're publishing a knowledge model for others to download.** Use `bundled` if the corpus is self-contained (consumers get a single file); use `remote` if the source belongs at canonical URLs (e.g. a tagged GitHub release — the manifest pins SHAs so a moved or rewritten upstream fails loud).
- **You're shipping a knowledge model in CI / a Docker image / an air-gapped install.** Use `bundled`.

For the technical layout (ZIP internals, NPZ format, manifest schema), see [docs/internal/STORE.md](../internal/STORE.md) and [docs/internal/KNOWLEDGE_MODEL_FORMAT.md](../internal/KNOWLEDGE_MODEL_FORMAT.md).
