# Changelog

All notable changes to Resonance Lattice. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-06-12

**A knowledge model that knows its own world.** One portable `.rlat` carries the corpus, what's been learned about the world that corpus covers — stable facts, standing constraints, what was tried and failed — and the receipts for all of it. The three content classes are serve-proven on pre-registered benchmarks (constraints: violations 62% → 7%, generalised to a garden and a law practice; falsified findings: 0/7 repeat recommendations vs 6/7 on a topical control), capture is validated behind a privacy gate (precision 0.86, recall 1.00, zero person-fact leaks), and the honest boundary is published with the same prominence: automatic suppression of disproven facts was tested three ways and falsified — corrections stay explicit. Every claim links its design, verdict, and raw run artifacts under `benchmarks/` ("no claim without a public receipt").

Last published build was `2.1.0a13`; this entry also covers the a14–a20 pre-release work.

### Added — the world-knowledge layer (three content classes: capture → review → serve)

- **`constraint` claim kind + `negation` (tried-and-falsified) adopted** alongside `attribute` world facts — the three serve-proven content classes, all living in the archive's insight band with full provenance.
- **Explicit capture**: `rlat capture-attribute <km> "..." --kind attribute|constraint|falsified [--attribute-key K] [--criticality ...]` (the falsified convention: evidence pointer in the text); `rlat capture-env <km>` auto-probes machine-readable environment attributes — zero confabulation, a read value can't be invented.
- **Passive capture, opt-in**: `rlat memory install-hooks [--user|--project-dir] [--mine]` wires the session hooks idempotently (foreign hooks preserved); `--mine` sets `RLAT_MINE_ATTRIBUTES=1`, turning on the 4-gate world-fact miner over what *you* said in a session. GATE 4 drops facts about the person — only facts true of the shared world land (validated: precision 0.86, recall 1.00, 0 person-fact leaks across every trap; `benchmarks/attribute_gate_e2c/`). Fails open without an LLM key. A visible receipt prints per session: `[rlat] Learned N world fact(s) into <km> (review: rlat lens / rlat profile)`.
- **Serve-all standing constraints + falsified findings** — no cosine floor, no top-k (the measured zero-over-blocking design, `benchmarks/constraint_band/`): served into agent-session injection and `rlat search --format context` with the measured kind framings ("Standing constraints for this environment:" / "Tried and falsified in this environment:", `store/serve_framing.py`).
- **World claims visible to `rlat search`** — experience-band claims render as `[INSIGHT]` hits in text/json/context output; a superseded keyed value (newest-wins per subject, per kind) never serves through any path, while older values stay on disk.
- **`rlat lens`** — `create | show | compose | set-trust`: the portable per-person perspective + the review surface over landed claims.

### Added — `rlat grow` (opt-in corpus self-improvement pass)

- **`rlat grow <km> [--max-fills N] [--dry-run]`** — fills the most-demanded, worst-covered gaps from the archive's own telemetry: demand-ranked gap selection, LLM-authored grounded fills, landing gated by faithfulness + compression + regression checks. Needs an API key; `--dry-run` is free.

### Added — per-user memory + session hooks (the experience loop)

- **Flat `ExperienceClaimStore`** — per-user, polarity-tagged, confidence-graded claims (no tier hierarchy); capture / recall / distil / confidence / forget / redactor, with privacy scrubbing and an append-only audit log.
- **Recall daemon + prompt injection** — a per-user daemon ranks the store and the workspace's corpus insight band on every prompt (`UserPromptSubmit` hook): experience lessons inject as `<rlat-memory>`, world facts and constraints as `<rlat-context>` blocks, all delimiter-spoof-safe and budgeted; `SessionEnd` captures the transcript. Fail-open everywhere — a daemon failure never blocks the prompt.
- **Intent + workspace state** (`rlat intent`, `rlat workspace`, `rlat expertise`) — outcome attribution moves claim confidence from real task outcomes; `rlat trace` / `rlat audit` expose the trail; `rlat reverify` re-checks stale claims; `rlat consolidate-insights` re-derives confidence idempotently.

### Added — row-mode knowledge models (semantic slicing for tabular worlds)

- **Row-mode builds** — one passage per business row with the caller's key pinned as passage identity (`key` carried on verified hits and the Fabric UDF surface), powering the rlat-native semantic-slicer Data App for Microsoft Fabric (`fabric/slicer/`): semantic slice → business keys + snippet receipts, OOM-safe streaming over OneLake, no SQL database required.

### Removed — experimental RQL ops

- The experimental `contradictions` / `audit` RQL ops are removed (2026-06) — superseded by the shipped corpus self-audit (`store/self_audit.py`) and the no-key skills. The curated 12-op suite (Foundation / Comparison / Composition / Evidence) stands.

### Changed — release engineering

- CLI surface is now **28 commands** (see the [CLI reference](https://tenfingerseddy.github.io/resonance-lattice/cli.html)).
- The v3 world-knowledge evidence dirs (`benchmarks/constraint_band*`, `falsification_ledger`, `attribute_gate_e2c`, `r4_continuous_credit`) ship in the public repo so every doc claim's receipt link resolves publicly — including the R4 honest-failure record.
- `tests/harness/cli_smoke.py` K6 enforces pyproject ↔ `__version__` parity (pyproject is the source of truth).

### Added — corpus self-audit + external-fact freshness (the insight-band loop)

A `.rlat` now **audits its own shape** and can **grow from outside facts**, all self-contained and (via the skills) with no API key.

- **Corpus self-audit, stored in the archive.** Every `build`/`refresh` computes an LLM-free shape report (`insight/self_audit.json`): **demand gaps** (under-served intents from telemetry) and **same-topic cross-document contradiction candidates** (high-cosine pairs, bounded `per_row_cap` + heap so it stays cheap on a redundant corpus). Folded into the single build write — no second archive rewrite. See `store/self_audit.py`.
- **`rlat audit --shape`** surfaces the report; `--min-cosine <f>` / `--with-text` recompute the contradiction candidates live at a chosen cosine floor with resolved text (judge-ready), **demand-ranked** so conflicts in the path of real query traffic come first. `rlat audit --external` enumerates web-fetched claims + their source URLs (the input to a world-freshness re-check).
- **Contradiction act layer** (`curator/reconcile.py`): a stance judge confirms which same-topic candidates genuinely contradict; `reconcile_contradiction` records a non-destructive, policy-gated resolution claim that cites both sides (never edits corpus source files).
- **External-fact freshness** (`store/external_freshness.py`): re-fetch a web-fetched claim's cited URLs and re-judge whether the live world still supports it (`fresh`/`stale`/`unknown`) — the staleness `rlat refresh` cannot catch. Surfaces only; a failed fetch is `unknown`, never `stale`.
- **External-fill landing + provenance trust tiers.** A verified external fact (≥2 independent agreeing sources) can land in the band with `source_url` provenance, durable across the corpus drift cascade. Trust is now tiered at the seed: **user ≥ verified-external ≥ single-external ≥ corpus** (`store/insight.py::seed_confidence(provenance=…)`). A caller-verified landing path (`promote_if_faithful(client=None, faithfulness=…)` + `curator/agent_fill.py::land_external_fact`) lets the free agent/human path land an already-verified fact without a metered judge — every other guard (compression test, ≥2-citation, trust floor) still applies.
- **Four no-API-key skills** drive the loop on subscription agents: `rlat-contradictions` (cross-doc conflicts), `rlat-gap-scan` (corpus gaps), `rlat-refresh-facts` (re-check web-fetched facts), `rlat-curate` (human-in-the-loop approval, hand-provide an authoritative source at the user trust tier).

### Removed — MRL optimise path

- **`rlat optimise` and the MRL optimised-band trainer are retired.** Deleted `cli/optimise.py`, the whole `src/resonance_lattice/optimise/` package (synth-query generation, hard-negative mining, InfoNCE training, in-place band write), `docs/internal/OPTIMISE.md`, and `docs/site/optimise.html`. The `[optimise]` install extra is renamed `[llm]` (just the Anthropic SDK — no torch, which only the trainer needed). The shipping recipe is the single `gte-modernbert-base` 768d base band; an opt-in specialist projection is no longer part of the CLI.

### Added — Microsoft Fabric UDF integration

- **Host a `.rlat` on Microsoft Fabric**, query it from anywhere with `rlat search fabric://<alias>[/<km>] "..."`. The maintainer publishes one User Data Function bound to a Lakehouse holding `Files/rlat/<km>.rlat`; team members run `rlat fabric add <alias>=<udf-url>` once and query through the existing `rlat search` surface. Single hosted copy, no per-user `.rlat` download, no encoder install on each machine.
- **Server-side runtime helpers** at `src/resonance_lattice/fabric/`: `_runtime.bootstrap()` (mtime cache-bust + LRU(8) over `(kmName, OneLake last_modified)` so a re-uploaded `.rlat` propagates within one warm call), `lakehouse_loader` (DataLake reads), `hf_loader` (HuggingFace revision-pinned encoder fetch).
- **UDF template** at `fabric/udf/` — `function_app.py` + `requirements.txt` + publish recipe README. Two endpoints: `search(kmName, query, ...)` and `list_kms()` for discovery.
- **Client URL dispatch** — `rlat search` accepts `fabric://<alias>[/<km>]`. `fabric://<alias>` (no km) hits the discovery endpoint and lists available KMs; `fabric://<alias>/<km>` runs single-shot retrieval via the same `--top-k` / `--format` / `--mode` / `--verified-only` flags.
- **`rlat fabric add | list | remove`** subcommand — alias storage at `~/.config/rlat/fabric.toml`, plus a per-cwd skill scaffold at `.claude/skills/rlat-fabric-search/SKILL.md` so Claude Code's skill list reflects the registered alias set.
- **Auth** — Microsoft Entra device-code flow by default (URL + code printed to stderr; Claude Code surfaces it through Bash tool stdout, the user completes the browser sign-in inline). Service-principal env-var fallback (`AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` / `AZURE_TENANT_ID`) for silent CI / shared-machine setups. Token cache via msal-extensions to OS keyring (`rlat-fabric` cache name).
- **`[fabric]` extra** in `pyproject.toml` — `azure-identity>=1.15`, `azure-storage-file-datalake>=12.14`. `requires-python` relaxed from `>=3.12` to `>=3.11` (Fabric UDFs run 3.11.9 only); ruff + pyright targets adjusted accordingly.
- **CLI surface 16 → 17** (added `fabric` subcommand).
- New harness suites: [tests/harness/fabric_bootstrap.py](tests/harness/fabric_bootstrap.py) (7 server-side guarantees: cold/warm bootstrap, mtime drift, LRU(8) cap, list_kms_for, missing-rlat error, blank-revision error) and [tests/harness/fabric_client.py](tests/harness/fabric_client.py) (10 client-side guarantees against a fake `http.server` UDF: URL parse, search/list_kms dispatch, unknown alias, HTTPError surfacing, `rlat fabric add/list/remove` round-trip, SP env detection, missing `azure-identity`).
- Docs: end-to-end walkthrough at [fabric.html](https://tenfingerseddy.github.io/resonance-lattice/fabric.html); "Hosted (Fabric UDF)" section in [storage-modes.html](https://tenfingerseddy.github.io/resonance-lattice/storage-modes.html); `fabric://` URL form + `rlat fabric` reference in [cli.html](https://tenfingerseddy.github.io/resonance-lattice/cli.html).

### Added — `rlat watch`

- **`rlat watch`** ships. Live, silent, self-discovering refresh loop on top of the Audit 07 incremental delta-apply pipeline. Zero-arg invocation auto-discovers `*.rlat` files in cwd and watches every recorded source root concurrently; explicit path overrides discovery. `--once` for CI / pre-commit; `--verbose` for per-refresh status lines. Default UX is silent — only errors and the startup + Ctrl-C summary are printed.
- Per-archive `threading.Lock` serialises `apply_delta` calls so concurrent FS events can't race the `<archive>.tmp` write path.
- `[watch]` extra in `pyproject.toml` (composes `[build]` + `watchdog>=4.0`); folded into `[all]`. Cross-platform FS event delivery via watchdog (inotify / FSEvents / ReadDirectoryChangesW).
- Pre-flight rejection of bundled-mode (with `rlat convert` hint) and remote-mode (with `rlat sync` hint) archives at startup.
- Path-prefix hygiene blocks `.git/`, `node_modules/`, `__pycache__/`, `.venv/` + 6 more known-noisy dirs.
- Debounce default `1000ms` (override via `RLAT_WATCH_DEBOUNCE_MS`).
- New harness suite [tests/harness/watch_loop.py](tests/harness/watch_loop.py): 9 contracts + a debounce-coalescing sanity check.
- **CLI surface 15 → 16** (added `watch`).
- Docs: `rlat watch` reference section in [cli.html](https://tenfingerseddy.github.io/resonance-lattice/cli.html).

### Fixed — `rlat watch` review pass

- **`--once` was hanging in CI / pre-commit** despite docs claiming CI-friendliness — it waited for a future FS event before refreshing, and pre-commit hooks always run *after* the edit. Rewrote `--once` as a synchronous one-shot: walks every preflighted archive, runs `bucketise` + `apply_delta` against current disk state, exits. No observer, no event wait. Matches user mental model + actually works in pre-commit hooks.
- **Renames / moves out of the watched suffix were leaving stale passages.** `foo.md → foo.bak` only fired `on_moved` with the dest path; the dest's suffix wasn't in `extensions`, the suffix pre-filter dropped it, the original `foo.md` passages stayed indexed forever. Added a `force` kwarg to the event dispatch that bypasses the suffix filter; move events now dispatch on **both** `src_path` and `dest_path`. Directory deletes also use `force=True` so the bucketise reconciliation can drop orphaned passages.
- **Transient read failures could silently delete archive content.** Windows file locks during atomic save (or mid-write UTF-8 decode failures) make a real source file disappear from `_walk_sources`'s output, which would make `bucketise` emit a destructive removal for every passage of that file. Added `_filter_skipped_removals`: any removal whose `source_file` is in the post-walk `skipped` set gets demoted back to `unchanged`, with a stderr warning. The next FS event drives the next refresh that reconciles for real. The "events are hints to reconcile, not the unit of correctness" mental model is now baked into the implementation contract.
- Three new harness contracts (7 / 8 / 9) cover --once with prior drift, force-dispatch bypass, and skip preservation.

## [2.0.0] — 2026-04-29

The v2.0.0 rebuild collapses the v0.11 surface to an evidence-backed minimum.
Breaking changes are extensive — full list compiled per phase as work lands.
Migration path from v0.11: there isn't one. Build fresh knowledge models with v2.0.

### Added (2026-04-28 — Bench 5 primer effectiveness MVP)

- **`benchmarks/user_bench/primer_effectiveness/`** ships — 5-lane × 25-scenario harness measuring code primer (`rlat summary`) vs memory primer (`rlat memory primer`) vs both-primers vs per-turn `rlat search` vs cold. Per-tier coverage profile: code primer wins orientation (3/5), memory primer wins memory recall (5/5), `rlat search` wins specific-factual (8/10). `both_primers` carries 48% turn-1 correct vs cold's 0%. Bench cost: $2.31. Result JSON committed at `benchmarks/results/user_bench/primer_effectiveness.json`.
- **Token cost table** for primers added to `docs/user/BENCHMARKS.md` and `docs/internal/benchmarks/05_primer_effectiveness.md`: code primer ~1,708 tokens/call, memory primer ~746 tokens/call, both ~2,454 tokens/call. ~1,400× smaller than a full-corpus dump.
- BENCHMARKS.md "deferred to v2.0.1" placeholder under § Session-start primer replaced with the real numbers + per-tier headline finding + honest framing on the 25-scenario sample size.

### Added (2026-04-28 — Skill restructure 3 → 2 + composite workflow)

- **`.claude/skills/rlat/`** restructured as a workflow-orchestration composite. Frontmatter description names all 9 sub-workflows (init / refresh / search / skill-context / memory / compare / convert / optimise / programmatic deep-search) for positive trigger specificity. `allowed-tools: Bash(rlat:*), Read, Write, Edit, Glob, Grep`. Pre-launch fix-up at commit `e493f050+` corrected three classes of stale commands flagged in the launch-readiness audit: removed `rlat install-encoder --check` (no `--check` flag), corrected memory subcommand syntax to `rlat memory --memory-root <path> {add|recall|primer|...} [args]` (the `--memory-root` flag goes on the parent command, not the subcommand), and replaced the non-existent `rlat rql ...` CLI surface with a Python-API reference (RQL ops are Python-only in v2.0). Replaced 500-line v0.11-stale `references/CLI_REFERENCE.md` with a slim pointer to the canonical `docs/user/CLI.md`.
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
- **"Optimise on remote" workflow** is now a clean two-command flow: `rlat convert upstream.rlat --to local --source-root <dir> -o working.rlat` then `rlat optimise working.rlat`. Documented in `docs/user/OPTIMISE.md` and `docs/user/FAQ.md`. The optimise pipeline stays storage-mode-agnostic.
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
