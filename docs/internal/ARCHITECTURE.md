# Architecture

The thesis is a **three-layer split**, all of it CLI-first:

```
┌────────────────────────────────────────────────────────────┐
│  field/    ROUTER      gte-modernbert-base 768d, dense    │
│  store/    AUTHORITY   3 modes — bundled / local / remote │
│  synthesis GROUNDED    passages and grounded answers —    │
│                        every claim traces to the corpus   │
└────────────────────────────────────────────────────────────┘
```

Single-recipe by design. No reranker, no lexical sidecar, no auto-mode router, no encoder presets. Retrieval knobs — `--rerank`, `--retrieval-mode`, `--cascade`, and kin — were measured null/negative on strong-dense and removed (see `audits/01_feature_audit.md`).

> Source-of-truth code: `src/resonance_lattice/`.
> Companion docs: [FIELD.md](./FIELD.md), [STORE.md](./STORE.md), [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md), [GROUNDING_MODEL.md](./GROUNDING_MODEL.md), [MEMORY.md](./MEMORY.md), [RQL.md](./RQL.md), [BENCHMARK_GATE.md](./BENCHMARK_GATE.md).

## Package map

The three layers below are the retrieval-critical spine. The full package surface of `src/resonance_lattice/`:

| Package | Role |
|---|---|
| `field/` | Router — single encoder, three auto-selected runtimes, dense + ANN retrieval, capture heart. |
| `store/` | Authority — v4 archive I/O, three storage modes, verified retrieval, incremental delta-apply, insight layer, self-audit. |
| `deep_search/` | Multi-hop plan → search → refine → synthesize — the grounded-answer loop. |
| `curator/` | Closed-form signals + decide/act layers (gap fill, contradiction reconcile) over the capture stream. |
| `memory/` | Per-user earned-experience memory — flat `ExperienceClaimStore`, polarity-tagged + confidence-graded. |
| `rql/` | 14 curated ops in 5 groups; Python API, CLI wraps a subset. See [RQL.md](./RQL.md). |
| `intent/` | `--kind intent` tag only in v2.0; operators v2.1+. |
| `lens/` | Portable user/team perspective overlay — corpus-agnostic schema. |
| `state/` | Agent-state primitives — workspace identity, live intent graph, outcome ledger, claim spine. |
| `expertise/` | The fourth context layer — corpus + memory + intent distilled into a session-start primer. |
| `assembler/` | Relevance-gated per-query assembly of the personal-context sources into one block. |
| `build/` | Pure-Python build/refresh pipeline; `cli/build.py` + `cli/maintain.py` are thin wrappers; the Fabric UDF imports it directly. |
| `install/` | Encoder download + ONNX/OpenVINO conversion (`rlat install-encoder`). |
| `fabric/` | Microsoft Fabric UDF runtime helpers (bootstrap + loaders); template at `fabric/udf/`. |
| `cli/` | 28 commands; the `add_subparser(sub)` call sequence in `cli/app.build_parser()` is the authoritative registration list. |

## The three layers

### Field (router) — `src/resonance_lattice/field/`

Turns a query into `[(passage_idx, score), ...]`. Single encoder, three swappable inference runtimes.

| File | Role |
|---|---|
| `encoder.py` | gte-modernbert-base 768d, CLS+L2, max-seq 8192. Runtime auto-picks ONNX/OpenVINO at query time, torch on the build path. |
| `onnx_runtime.py` | Default non-Intel CPU runtime. |
| `openvino_runtime.py` | Intel CPU runtime — auto-detected. |
| `torch_runtime.py` | Build path only (pulls torch via `[build]`). |
| `dense.py` | Exact cosine retrieval, dedup by `(source_file, char_offset)`. |
| `ann.py` | FAISS HNSW M=32 efC=200 efS=128 above N=5000 passages. Audit 04 locked. |
| `algebra.py` | `centroid` / `greedy_cluster` / `complete_linkage_cluster` — minimal field algebra used by RQL ops + cli/compare + cli/summary + curator/signals. |
| `capture.py` | The capture heart — every retrieval is observed (query fingerprint + per-rank scores, never the query text) into a bounded in-memory ring keyed by corpus identity. |
| `counters.py` | Tier-0 closed-form count layer — Ebbinghaus-decayed access-reinforcement over the capture stream, no model. |
| `text.py` | Sentence splitter shared with `store/chunker.py`. |
| `__init__.py` | `retrieve(query_emb, handle, ann_index, registry, top_k)` + `retrieve_insight(query_emb, insight_band, ann_index, top_k)` — the canonical retrieval entry points used by every cli/* path; both feed `capture.observe`. |

**Locked invariants**: pinned HF revision, CLS pooling, L2-norm output, no swap of pool/norm without a paired BENCHMARK_GATE update.

### Store (authority) — `src/resonance_lattice/store/`

Resolves `(source_file, char_offset, char_length)` triples back to authoritative text, with drift detection.

| File | Role |
|---|---|
| `archive.py` | ZIP v4 read/write/in-place. `ArchiveContents` holds metadata + registry + bands + ANN bytes. `select_band(prefer="base"|None)` returns a `BandHandle`. |
| `metadata.py` | JSON schema + serialise. `FORMAT_VERSION = 4` is the single source of truth; archive imports it. Forward-compat `_extras` dicts at top-level and per-band. |
| `bands.py` | NPZ I/O for the base band (+ optional insight band — same `write_band` slot). Exports `load_base` + `write_band`. |
| `registry.py` | `PassageCoord` (frozen dataclass), `passages.jsonl` reader/writer; line-implicit `passage_idx`, blank-line-rejecting. v4.1 adds a stable `id` (sha256-short of the coordinate triple) that survives refresh/sync deltas; legacy v4 lines load through `compute_id`. |
| `chunker.py` | `passage_v1` strategy — paragraph→sentence→hard split, undersized-tail emission for full coverage. |
| `base.py` | `Store` ABC + `compute_hash`. Concrete `fetch` / `verify` / cache live here; subclasses implement only `_read_full_text_uncached`. |
| `bundled.py` | zstd-framed `source/` inside the .rlat ZIP. Module-level `_DCTX` for one-shot decompression. |
| `local.py` | FS-resolved via `--source-root`, per-instance text cache, path-traversal guard. |
| `remote.py` | HTTP-pinned manifest + on-disk cache, default 30s timeout, injectable opener for tests. |
| `verified.py` | `VerifiedHit` dataclass + `verify_hits(hits, store, registry)` — bridges `field.dense.search`'s tuples to authoritative-text rows with drift status. Re-exports `compute_hash` and `DriftStatus`. |
| `incremental.py` | Shared delta-apply pipeline behind `rlat refresh` + `rlat sync`. |
| `conversion.py` | Storage-mode conversion behind `rlat convert` (Audit 08). |
| `remote_index.py` | `RemoteIndex` Protocol — upstream-state oracle for `rlat sync`. |
| `streaming.py` | OOM-safe streaming top-k over the base band — never materialises the (N, D) matrix. |
| `insight.py` + `insight_lifecycle.py` + `insight_attribution.py` + `corpus_claim_io.py` | Corpus-claim primitives (citations, verdict signals, trust math), lifecycle orchestrators, outcome→Beta attribution, JSONL I/O. |
| `self_audit.py` | Corpus self-audit — contradiction candidates + demand gaps, LLM-free, stored at build/refresh. |
| `promotion.py` + `compression_test.py` | Insight promotion — candidates earn the layer only by passing the compression test. |
| `reverification.py` + `external_freshness.py` | Re-verify drifted insights; re-check external fills against the live world. |
| `faithfulness.py` | Machine grounding check for deep-search answers — entry gate of the confidence lifecycle. |
| `telemetry.py` | Folds the capture heart's in-memory stream into the `.rlat`. |
| `audit.py` | Trust-contract surface behind `rlat audit` / `rlat trace`. |
| `__init__.py` | `open_store(km_path, contents, source_root)` factory used by every CLI path. |

**Locked invariants**: per-passage `content_hash`, atomic ZIP writes via tmp+rename, ZIP_STORED outer compression (NPZ is already compressed), cross-knowledge-model ops always run on the base band.

### Grounded synthesis — `src/resonance_lattice/deep_search/` + the insight layer

rlat returns passages *and* faithfully-grounded answers. The "no reader" thesis is retired — see [GROUNDING_MODEL.md](./GROUNDING_MODEL.md). The product is a grounded answer: every claim traces to a corpus source; citations are on-topic; gaps are stated as gaps. rlat owns *grounding*, not truth — it represents what the corpus says, never what is true in the world.

Two synthesis surfaces:
1. **`rlat deep-search`** — multi-hop plan → search → refine → synthesize loop (`deep_search/`). Each hop reuses the same single-recipe retrieval on the base band; name-verification over the union of evidence gates the synth call. Fabric 5-lane bench (Sonnet 4.6, 63 questions): 92.2% acc / 2.0% halluc vs 74.5% / 3.9% single-shot.
2. **The insight layer** — earned, cited claims inside the archive (`store/insight.py`; `bands/insight.npz` + `insight.jsonl`), promoted only through the compression test (`store/compression_test.py`), re-verified on drift (`store/reverification.py`). `rlat search` serves insight hits alongside source passages.

`rlat search --format context` materialises a token-budgeted block of verified passages with HTML-comment delimiters — designed to be piped into an LLM call you're already making — under a grounding-mode header (`--mode augment|knowledge|constrain`, `cli/_grounding.py`).

## CLI surface — `src/resonance_lattice/cli/`

28 commands; the `add_subparser(sub)` call sequence in `cli/app.build_parser()` is the authoritative registration list. Retrieval commands are thin orchestrators — load archive, pick band, retrieve, format — over the shared helpers below. The table covers the retrieval-critical commands; the complete reference is `docs/site/cli.html`.

| Command | File | Role |
|---|---|---|
| `rlat install-encoder` | `install_encoder.py` | HF download + ONNX export + OpenVINO IR conversion. |
| `rlat init-project` | `init.py` | Auto-detect sources → `cmd_build` → `cmd_summary`. Sugar over build+summary. |
| `rlat build` | `build.py` | Walk sources, chunk, encode (torch), build FAISS index, write v4 ZIP. |
| `rlat search` | `search.py` | encode → `field.retrieve` → `verify_hits` → format text/json/context. |
| `rlat profile` | `profile.py` | Backbone + bands + drift summary. JSON status-discriminator. |
| `rlat compare` | `compare.py` | Centroid cosine + asymmetric mutual coverage. Base band only. |
| `rlat summary` | `summary.py` | Extractive primer (Landscape / Structure / Evidence). |
| `rlat refresh` | `maintain.py` + `store/incremental.py` | Local-mode incremental delta-apply: walk source_paths → bucketise on stable passage_id → re-encode updated+added → preserve unchanged rows → atomic write. |
| `rlat watch` | `watch.py` + `maintain.py` semantics + `watchdog` | Live, silent, self-discovering refresh loop on top of the same `incremental.apply_delta` pipeline. Per-archive `_DebouncedRefresher` + `threading.Lock` (closes the `<archive>.tmp` race that two concurrent refreshes would otherwise lose). Mental model: events are hints to reconcile, not the unit of correctness — every fire does a full source-tree walk + bucketise. `--once` is the synchronous CI / pre-commit shape (no observer, no event wait). Skipped-file preservation defends against transient read failures becoming silent deletes. Local mode only; bundled / remote rejected at preflight. Requires `[watch]` extra. |
| `rlat sync` | `maintain.py` + `store/incremental.py` + `store/remote_index.py` | Remote-mode incremental delta-apply: `RemoteIndex.changed_files_since(pinned_ref)` → fetch deltas only → same `incremental.apply_delta` pipeline as refresh. Rewrites the in-archive `manifest.json` per-entry `sha256` and `metadata.build_config["pinned_ref"]` atomically with the band write (codex P0 correctness gate baked in: `apply_delta` requires the encoder by signature). |
| `rlat convert` | `cli/convert.py` + `store/conversion.py` | Switch storage modes (bundled / local / remote) without rebuilding. `Store.fetch_all()` materialises bytes in the source mode; conversion validates per-passage `content_hash` against live bytes (drift abort if any mismatch); composes target-mode payload (`source/` zstd / `manifest.json` / disk files); rewrites metadata atomically. Bands + registry + ANN preserved (`np.allclose` at 1e-6). Audit 08. |
| `rlat freshness` | `maintain.py` + `store/remote.py` | Remote-mode read-only drift check: walks the manifest, downloads each entry, hashes it, reports per-entry status. CI-friendly gate before running sync. |
| `rlat deep-search` | `deep_search.py` | Multi-hop grounded answer (plan → search → refine → synthesize). Needs an Anthropic API key. |
| `rlat memory` | `memory.py` | Per-user flat-memory subcommands (add / recall / capture / consolidate / gc / …). |
| `rlat skill-context` | `skill_context.py` | Markdown context block for Anthropic skill `!command` blocks. |
| `rlat intent` / `rlat workspace` | `intent.py` / `workspace.py` | Live-intent graph + workspace identity (state layer). |
| `rlat expertise` | `expertise.py` | Write the expertise primer — the fourth context layer. |
| `rlat audit` / `rlat trace` | `audit.py` / `trace.py` | Trust-contract audit + full provenance chain for an insight or source passage. |
| `rlat lens` | `lens.py` | Manage lens artefacts. |
| `rlat reverify` | `reverify.py` | LLM re-verification of stale insights. |
| `rlat probe` | `probe.py` | Idle-cycle self-probe — re-attempt unmet intents against the current corpus. |
| `rlat capture-env` / `rlat capture-attribute` | `capture_env.py` | Land user-environment attributes in the insight band. |
| `rlat consolidate-insights` | `consolidate_insights.py` | Fold resolved-intent outcomes into insight-layer Beta confidence. |
| `rlat fabric` | `fabric.py` | Manage `fabric://` aliases + the search-skill scaffold. |

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
  metadata.json + passages.jsonl + bands/base.npz + (optional) ann/base.faiss + (optional) source/ + (optional) manifest.json (remote) →
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
  + field.retrieve_insight → verify_insight_hits (when an insight band is present) →
  filter_verified (if --verified-only) →
  format text/json/context (context adds the --mode grounding header + suppression,
  then the --strict-names gate)
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
- Grounding model — what rlat promises, how knowledge earns trust: [GROUNDING_MODEL.md](./GROUNDING_MODEL.md).
- Memory design: [MEMORY.md](./MEMORY.md). Per-op RQL rationale: [RQL.md](./RQL.md).
- Locked benchmark numbers: [BENCHMARK_GATE.md](./BENCHMARK_GATE.md).
- Per-feature audits: [audits/](./audits/).
- Long-horizon tracking: [REBUILD_PLAN.md](./REBUILD_PLAN.md).
