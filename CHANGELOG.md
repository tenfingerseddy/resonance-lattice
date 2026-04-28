# Changelog

All notable changes to Resonance Lattice. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-04-29

The v2.0.0 rebuild collapses the v0.11 surface to an evidence-backed minimum.
Breaking changes are extensive — full list compiled per phase as work lands.
Migration path from v0.11: there isn't one. Build fresh knowledge models with v2.0.

### Added (2026-04-28 — Bench 5 primer effectiveness MVP)

- **`benchmarks/user_bench/primer_effectiveness/`** ships — 5-lane × 25-scenario harness measuring code primer (`rlat summary`) vs memory primer (`rlat memory primer`) vs both-primers vs per-turn `rlat search` vs cold. Per-tier coverage profile: code primer wins orientation (3/5), memory primer wins memory recall (5/5), `rlat search` wins specific-factual (8/10). `both_primers` carries 48% turn-1 correct vs cold's 0%. Bench cost: $2.31. Result JSON committed at `benchmarks/results/user_bench/primer_effectiveness.json`.
- **Token cost table** for primers added to `docs/user/BENCHMARKS.md` and `docs/internal/benchmarks/05_primer_effectiveness.md`: code primer ~1,708 tokens/call, memory primer ~746 tokens/call, both ~2,454 tokens/call. ~1,400× smaller than a full-corpus dump.
- BENCHMARKS.md "deferred to v2.0.1" placeholder under § Session-start primer replaced with the real numbers + per-tier headline finding + honest framing on the 25-scenario sample size.

### Added (2026-04-28 — Skill restructure 3 → 2 + composite workflow)

- **`.claude/skills/rlat/`** restructured as a workflow-orchestration composite. Frontmatter description names all 9 sub-workflows (init / refresh / search / skill-context / memory / compare / convert / optimise / programmatic deep-search) for positive trigger specificity. `allowed-tools: Bash(rlat:*), Read, Write, Edit, Glob, Grep`. Pre-launch fix-up at commit `e493f050+` corrected three classes of stale commands flagged in the launch-readiness audit: removed `rlat install-encoder --check` (no `--check` flag), corrected memory subcommand syntax to `rlat memory --memory-root <path> {add|recall|primer|...} [args]` (the `--memory-root` flag goes on the parent command, not the subcommand), and replaced the non-existent `rlat rql ...` CLI surface with a Python-API reference (RQL ops are Python-only in v2.0). Replaced 500-line v0.11-stale `references/CLI_REFERENCE.md` with a slim pointer to the canonical [docs/user/CLI.md](docs/user/CLI.md).
- **`.claude/skills/rlat-deep-research/`** removed — folded into the rlat skill's "Programmatic deep-search" workflow section. Eliminates 3-skills redundancy where `rlat deep-search` was both a CLI verb and a separate skill.
- **`.claude/skills/deep-research/`** retained — Claude-driven multi-hop research over an rlat KM (uses the user's Claude Code subscription instead of the API key the CLI verb requires).
- 15 evals at `.claude/skills/rlat/evals.json` covering should-trigger × 8 sub-workflows + should-defer-to-deep-research × 2 + should-NOT-trigger × 4 (exact-symbol-rename, specific-file-edit, other-vector-DB, training-knowledge).
- `docs/user/SKILLS.md` documents the 2-skill structure with citations to Anthropic skill design guidance.

### Added (2026-04-27 — `rlat deep-search` CLI verb + namecheck)

- **`rlat deep-search km.rlat "<question>"`** ships. Multi-hop research loop (plan → retrieve → refine → maybe re-retrieve → synthesize) returning a final answer plus the union of evidence. Bench-validated headline: **92.2% answerable accuracy at 0% hallucination, $0.009/q** on the Microsoft Fabric corpus 11-lane v4 bench (63 questions, Sonnet 4.6, relaxed rubric). Within 2 pp of an LLM+grep/glob baseline at 6.5× lower spend.
- **Namecheck** (`--strict-names`) — distinctive-token verification on the grounding-emit boundary. Catches name-aliasing distractor failures where the encoder surfaces a similarly-named real entity for a fake-product-name question. Wired through `rlat skill-context`, `rlat search --format context`, and `_grounding.py`. Harness suite at `tests/harness/name_check.py` (10 + 6 guarantees).
- **CLI surface 15 → 16** (added `deep-search`).
- **11-lane v4 bench results** (`benchmarks/results/user_bench/hallucination_v4.json`): 3 modes × 3 retrieval shapes + LLM-only + LLM+grep/glob. `rlat deep-search --mode knowledge` and default `augment` both hit 92.2% / 0% halluc / $0.009/q. `rlat search --mode constrain` is the compliance floor at 91.7% distractor refusal / 2.0% answerable hallucination.
- New `src/resonance_lattice/deep_search/` module — `loop.py`, `prompts.py`, `types.py` (composable; `rlat deep-search` is a thin CLI wrapper).

### Renamed (2026-04-28 — package distribution name)

- PyPI distribution name `rlat2` → `rlat`. Was reserved as `rlat2` during the rebuild; now claims the canonical `rlat` namespace for v2.0 launch. `pip install rlat` works; `pip install rlat2` no longer publishes new versions.
- README + all docs updated to `pip install rlat[bench]`, `pip install rlat[build]`, etc.

### Added (2026-04-27 — Audit 08: storage-mode conversion)

- **`rlat convert <km> --to {bundled|local|remote}`** ships. Switches a knowledge model between storage modes WITHOUT rebuilding embeddings — bands, registry, ANN, and the optimised W projection are preserved (`np.allclose` at 1e-6). All six pairwise transitions supported. Atomic in-place via `tmp + os.replace`.
- **`Store.fetch_all(source_files)`** primitive on the ABC — bulk-reads every requested source file via the cached `_read_full_text`. Default impl works for all three subclasses; specific stores can override with parallel-fetch paths in v2.1+.
- **`ConversionDriftError`** typed exception. Conversion validates every passage's `content_hash` against the live bytes resolved via the source mode's Store BEFORE write; if any drift, raises this error and does NOT write a new archive. The user runs `rlat refresh` (local) or `rlat sync` (remote) to reconcile, then retries convert. Same correctness pattern as Audit 07's codex P0 fix.
- **"Optimise on remote" workflow** is now a clean two-command flow: `rlat convert upstream.rlat --to local --source-root <dir> -o working.rlat` then `rlat optimise working.rlat`. Documented in [docs/user/OPTIMISE.md](docs/user/OPTIMISE.md) and [docs/user/FAQ.md](docs/user/FAQ.md). The optimise pipeline stays storage-mode-agnostic.
- **`tests/harness/conversion`** — 8 hermetic guarantees (3 round-trips × bands `np.allclose`; passage_id stable; content_hash stable; drift abort; idempotent no-op; error-shape).
- **CLI surface count 14 → 15** (added `convert`).

### Added (2026-04-26 — Audit 07: incremental refresh + sync)

- **`rlat sync`** ships. Remote-mode incremental delta-apply: discover upstream changes via `RemoteIndex.changed_files_since(pinned_ref)`, fetch only the deltas, re-encode them, atomically write the new archive with the new manifest pinned. Two `RemoteIndex` modes:
  - **Catalog mode** (`--upstream-manifest <url>`) — upstream serves a stable `{source_file: {url, sha256}}` endpoint; sync diffs in O(1) network calls and detects added + modified + removed.
  - **Poll mode** (default) — re-fetches every URL in the existing manifest, diffs SHAs against the pinned values; detects modified + removed only.
- **`rlat refresh`** rewritten as incremental delta-apply (was: full rebuild from `metadata.build_config.source_root`). Unchanged passages now keep their band rows untouched; only modified/added passages are re-encoded.
- **Optimised band re-projection** in both refresh + sync. After a delta-apply, the optimised band is re-projected from the new base via `optimised = (new_base @ W.T)` row-wise L2-normalised. **Free** — no LLM call, no GPU. The earlier "refresh discards the optimised band, pay $14-21 + 30 min to regenerate" footgun is gone.
- **Stable `passage_id`** in `passages.jsonl` (additive v4 → v4.1 schema bump). Each passage now carries `id = sha256(source_file + char_offset + char_length)[:16]` so a passage's identity survives across refresh/sync deltas. Verified-retrieval citations, `corpus_diff` continuity, and external bookmark consumers stay valid through reconciliation. Legacy v4 archives load through `registry.compute_id` — back-compat read.
- **`--dry-run`** on both `refresh` and `sync` — walk + bucketise + report the four-bucket delta counts, no fetch, no write.
- **`tests/harness/incremental_refresh`**, **`incremental_sync`**, **`optimised_reproject`** — three new harness suites enforcing the contracts (14 guarantees total, all hermetic — no live network).

### Renamed (2026-04-26)

- CLI `rlat specialise` → `rlat optimise`. Better word: the command both reduces
  embedding dim (768 → 512 via the MRL projection) AND improves in-corpus
  retrieval. "Specialise" was opaque about what it did; "optimise" is clearer
  to users who haven't read the docs.
- Module `src/resonance_lattice/specialise/` → `src/resonance_lattice/optimise/`.
- pyproject extras `[specialise]` → `[optimise]`.
- Band name in archive: `bands["specialist"]` → `bands["optimised"]`. v2.0
  unreleased — no migration path needed for in-flight `.rlat` files; rebuild
  with the new code.
- Docs: `docs/user/SPECIALISE.md` → `OPTIMISE.md`, `docs/internal/SPECIALISE.md`
  → `OPTIMISE.md`.
- Harness suite: `tests/harness/specialise_roundtrip.py` →
  `optimise_roundtrip.py`. Runner matcher updated.
- Bench scripts: `bench_beir3_specialist_soak.py` → `bench_beir3_optimised_soak.py`,
  `bench_fabric_specialist_probe.py` → `bench_fabric_optimised_probe.py`.

Clean break — no aliases, no deprecation shim. Frozen artifacts under
`benchmarks/results/` retain their original "specialist"/"specialise"
filenames as historical-run records. Kaggle kernel slugs (e.g.
`rlat2-beir-3-specialist-soak-v7`) likewise frozen as historical.

### Removed (compared to v0.11)

CLI commands deleted (21):
- `ask` — LLM coupling broke LLM-free positioning. Use `rlat search --format context` and your own assistant.
- `resonate` — redundant with `search --format context`.
- `merge` — needs Intent operators; deferred to v2.1+.
- `mcp` — MCP server dropped from v2.0; CLI is primary interface. Candidate for v2.1.
- `query`, `ingest`, `add`, `ls`, `info`, `diff`, `encoders`, `export`, `probe`, `topology`, `xray`, `negotiate`, `forget` — replaced by `build` / `sync` / `profile` / RQL ops.
- `contradictions` (top-level), `primer` (top-level), `locate` (top-level), `compose` (top-level) — folded into `rlat` RQL dispatch.
- All `skill *` subcommands (10) — skills reference existing knowledge models via SKILL.md frontmatter.
- All `lens *` subcommands (4) — replaced by RQL subspace algebra.

CLI flags / knobs deleted (~50+):
- Cross-encoder rerank: `--rerank`, `--probe-rerankers`. Measured null/negative on strong-dense.
- Lexical: `--hybrid`, `--lexical-impl`, `--bm25-index`. BEIR-5 parity failure (4/5 corpora).
- Encoder selection: `--encoder` (single recipe), `--bands`, `--dim`, `--field-type`, `--precision`, `--compression`, `--sparsify-mode`, `--soft-topk-tau`, `--sparsemax-scale`, `--quantize-registry`, `--compact`.
- Routing: `--retrieval-mode`, `--mode`, `--cascade`, `--cascade-depth`, `--subgraph`, `--subgraph-k`, `--expand`. Dense-only thesis.
- Inference: `--onnx`, `--openvino`, `--openvino-device`, `--openvino-static-seq-len`. Auto-detected.
- Misc: `--no-worker`, `--probe-*`, `--contextual-chunking`, `--with-contradictions`, `--contradiction-threshold`.

Modules deleted:
- `reranker.py`, `query_router.py`, `lens_router.py`, `cascade.py`, `reversible_cascade.py`, `lens.py`, `projector.py`, `skill_projector.py`, `mcp_server.py`, `stream.py`.
- `field/asymmetric_dense.py`, `field/multi_vector.py`, `field/factored.py`, `field/pq.py`.
- `training/heads.py`, `training/trainer.py`, `training/asymmetric_*.py`. Trained heads closed 0-for-9.
- `rql/eml*.py`. EML retrieval falsified 3×.
- `_experimental/*` (9 modules). Stubs and prototypes.
- `temporal.py`, `temporal_algebra.py`, `quantize.py`, `subspace.py`, `consciousness.py`, `quantum.py`, `symplectic.py`, `sculpting.py`, `pattern_injection.py`, `metabolism.py`, `interference.py`, `confidence.py`.
- `reader/*` — no reader layer in v2.0; consumer synthesizes.

Dependencies dropped from base install:
- `torch`, `transformers` → moved to `[build]` / `[optimise]` / `[gpu]` extras.
- `mcp` → MCP dropped from v2.0.
- `datasketch`, `watchdog`, `questionary`, `tree-sitter-*` → unused in v0.11 src/, removed entirely.
- Base install drops from ~2.5 GB to ~250 MB.

### Added

CLI commands new (2):
- `rlat install-encoder` — one-time HF download → ONNX export → optional OpenVINO conversion.
- `rlat optimise <km.rlat>` — opt-in in-place MRL optimised band (~$14-21 + 30 min GPU).

CLI flags new:
- `rlat build --kind corpus|intent` — Intent Lattice kind tag (no operators in v2.0).

API key discovery:
- `RLAT_LLM_API_KEY_ENV` → `CLAUDE_API` → `ANTHROPIC_API_KEY`.

### Changed

- **Default encoder**: `gte-modernbert-base` 768d (was `e5-large-v2` / `bge-large-en-v1.5` depending on point release). One recipe, no presets.
- **Knowledge-model format**: v4 — ZIP + JSON + NPZ multi-band slots. v0.11's binary format is no longer readable.
- **Default storage mode**: `local` (`--source-root`).
- **Python minimum**: 3.12 (was 3.11).
- **Terminology**: "knowledge model" replaces "cartridge" in all surfaces.

### Migration

There is no in-place migration tool. v0.11 knowledge models cannot be read by v2.0. Rebuild from source:

```bash
pip install rlat
rlat install-encoder
rlat build ./your-source -o new.rlat
```

The `legacy/v0.11.0` tag preserves the v0.11 codebase if you need to read old `.rlat` files.

---

## Pre-2.0.0 history

See git log on the `legacy/v0.11.0` tag.
