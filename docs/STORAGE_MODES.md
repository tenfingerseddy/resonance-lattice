# Storage Modes

Resonance Lattice ships a **three-layer semantic router** (the 2026-04-19 pivot):

1. The **field** routes queries to candidate chunks.
2. A **lossless store** serves the evidence text.
3. A **reader** (LLM) synthesises the final answer.

The store is where the bytes actually live. There are three modes, sharing one abstraction (`LosslessStore`) and differing only in *where* the raw source files come from.

| Mode | Raw files live in | Pick when… | Freshness |
|---|---|---|---|
| **`bundled`** | the `.rlat` knowledge model itself (zstd frames) | shipping a self-contained artifact — HF Hub demos, CI, offline use, a single file to hand someone | rebuild the knowledge model |
| **`local`** | a local directory resolved via `--source-root` | developing against a working copy; large corpora where the knowledge model should stay thin | `rlat refresh <cart> --source-root <dir>` |
| **`remote`** | a public HTTP origin with a SHA-pinned local cache | pointing at an upstream repo you don't own (docs, public codebases) | `rlat freshness <cart>` / `rlat sync <cart>` |

The canonical CLI flag is `--store-mode {bundled,local,remote}`. The historical name `external` is still accepted as a synonym for `local` — old knowledge models and in-flight branches keep working unchanged. The deprecated `embedded` mode (pre-chunked SQLite store, legacy) is a **separate** thing from `bundled`; see [Bundled vs. legacy embedded](#bundled-vs-legacy-embedded) below.

---

## Build

```bash
# bundled — self-contained
rlat build ./docs ./src -o project.rlat --store-mode bundled

# local (default) — thin knowledge model + --source-root at query time
rlat build ./docs ./src -o project.rlat

# remote — URL auto-detected; pins to the default branch HEAD
rlat build https://github.com/MicrosoftDocs/fabric-docs -o fabric-docs.rlat

# remote, pinned to a specific ref
rlat build https://github.com/MicrosoftDocs/fabric-docs#release-branch -o fabric-docs.rlat
rlat build https://github.com/MicrosoftDocs/fabric-docs@abc1234 -o fabric-docs.rlat

# remote, scoped to a subdirectory of a larger repo
#   --path is persisted into __remote_origin__ so freshness + sync
#   only see drift under that prefix
rlat build https://github.com/python/cpython --path Doc -o python-stdlib.rlat
rlat build https://github.com/MicrosoftDocs/sql-docs --path docs/t-sql -o t-sql.rlat
```

## Query

All three modes answer queries identically — the retriever doesn't know which backend is serving bytes:

```bash
rlat search project.rlat "how does auth work?" --source-root .    # local
rlat search fabric-docs.rlat "how do I create a lakehouse"        # bundled OR remote
rlat ask fabric-docs.rlat "what is a fabric workspace?"
```

Remote knowledge models consult `~/.cache/rlat/remote/<origin>/<commit_sha>/<path>` before hitting the network. Once a blob is in the disk cache, a warm remote query is one disk read — identical cost to local/bundled.

## Inspect

```bash
rlat info project.rlat                     # shows mode, encoder, source count, size
rlat info fabric-docs.rlat                 # remote knowledge models additionally report origin + pinned sha + cache size
```

## Freshness (remote only)

```bash
# Read-only drift check — one GitHub API call
rlat freshness fabric-docs.rlat
# => pinned at abc1234aa
#    upstream  def5678bb
#    diff      +3 added, ~12 modified, -1 removed
#    run `rlat sync fabric-docs.rlat` to apply.
#
# exit 0 = up-to-date, exit 1 = drift detected (useful for CI gates)
```

## Sync (remote only)

```bash
rlat sync fabric-docs.rlat
# fetches only changed files via the GitHub compare API,
# routes them through the same chunk-reconciliation pipeline
# `rlat refresh` uses, atomically bumps __remote_origin__.commit_sha
```

The cost is proportional to the **diff**, not the corpus. A fabric-docs resync against a single-commit change pulls only the handful of changed markdown files, not the whole tree.

## Repoint — switch modes without re-encoding

Because the three modes share a manifest schema (posix-relative `source_file` per chunk), you can flip between them without rebuilding the field or re-encoding any text:

```bash
# local → remote (pin a locally-built knowledge model at an upstream GitHub repo)
rlat repoint fabric.rlat --to remote \
    --url https://github.com/MicrosoftDocs/fabric-docs

# remote → local (unpin, useful for offline queries against a local checkout)
rlat repoint fabric-docs.rlat --to local

# local → bundled (pack source files into a self-contained artifact)
rlat repoint fabric.rlat --to bundled --source-root ./fabric-docs -o fabric-bundled.rlat

# remote → bundled (stage via fetcher + disk cache, then pack)
# If the knowledge model has been queried, hot blobs are already cached and no network
# is needed. Missing blobs are fetched from the pinned upstream SHA.
rlat repoint fabric-docs.rlat --to bundled -o fabric-bundled.rlat
```

All transitions validate before writing. The `→ remote` direction rejects zero path overlap with the upstream tree (hard fail) and warns on partial overlap (<80%). The `→ bundled` directions skip unreachable files gracefully — warnings get printed, but the knowledge model still ships without them.

All the expensive parts (encoder load, passage encoding, field accumulation) are skipped across every transition — repoint is cheap and interactive, typically seconds for a mid-sized knowledge model.

Supported transitions: `local ↔ remote`, `local → bundled`, `remote → bundled`. Bundled → anything requires a rebuild (`rlat build`) since the bundled blob is immutable once packed.

## Refresh (local only)

```bash
rlat refresh project.rlat --source-root .
# preserves field entries whose chunk hashes still match; runs
# forget+superpose only on drifted / added / removed chunks
```

---

## Lossless parity

Every retrieval feature that works in one mode works in every other mode. The knowledge model stores **whole files**, not pre-chunked text, so:

- The retriever can re-chunk on demand with different chunker parameters.
- Window widening / heading expansion operates over full file context.
- Content-hash drift detection runs the same way in bundled, local, and remote.
- Format dispatch (markdown / python / csv / pdf / docx / xlsx / …) is identical across modes.

The parity test in `tests/test_bundled_store.py` locks this in: a corpus built in `local` and `bundled` modes returns **identical `source_id`s AND identical `full_text`** in the top-K.

## Bundled vs. legacy embedded

Do not confuse `bundled` with the deprecated `embedded` mode:

| | `embedded` (legacy, deprecated) | `bundled` (v3+) |
|---|---|---|
| What's stored in the `.rlat` | **pre-chunked** text rows in an in-knowledge model SQLite database | **raw whole files**, zstd-compressed, individually framed |
| Can the retriever re-chunk? | No — chunker is baked in at build time | Yes — re-chunks on every retrieve, same as local/remote |
| Window widening / format dispatch | Limited / broken | Works identically to local/remote |
| Status | Deprecated; removal target v2.0.0 | Supported; first-class |
| Warning on load | `DeprecationWarning` pointing at `--store-mode bundled` | None |

If you have old `embedded` knowledge models, rebuild them with `--store-mode bundled` (if you want self-contained) or the default `--store-mode local` (thinner knowledge model, source files on disk).

## Cache layout (remote)

```
~/.cache/rlat/remote/
  github__MicrosoftDocs__fabric-docs/
    abc1234.../docs/fabric/index.md
    abc1234.../docs/lakehouse/overview.md
    ...
    def5678.../docs/fabric/index.md    # new SHA = new dir; old entries age out by LRU
```

Keys include the commit SHA, so cached bytes are immutable — no conditional GETs, no ETag revalidation on the hot path. Defaults to a 500 MB budget with atime-LRU eviction; tune via `DiskCache(budget_bytes=...)` in code.

## Format layout on disk

All three modes share the `.rlat` v3 format header:

```
[0x00] "RLAT" magic
[0x04] version = 3
...
[0x39] store_mode byte — 0=embedded, 1=local/external, 2=remote, 3=bundled
```

The source-store section (after the field tensor + registry) varies by mode:

- **embedded** — SQLite database with per-chunk text rows (legacy)
- **local** — metadata-only SQLite: encoder config, source manifest, retrieval config, profile
- **remote** — metadata-only SQLite as above, plus `__remote_origin__` (type / org / repo / ref / commit_sha)
- **bundled** — wrapped payload starting with `"RLBD"` magic: `[meta sqlite][index json][concatenated zstd frames]`

## Related

- [CLI.md](CLI.md) — full command reference for `build`, `search`, `refresh`, `freshness`, `sync`, `info`.
- [SEMANTIC_MODEL.md](SEMANTIC_MODEL.md) — the three-layer router architecture.
- [CORE_FEATURES.md](CORE_FEATURES.md) — seven launch use cases; knowledge model portability maps onto bundled, always-fresh maps onto remote.
