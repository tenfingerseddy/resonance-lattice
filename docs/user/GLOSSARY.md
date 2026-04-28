# Glossary

The terms `rlat` uses, in one place.

---

## Key terms first

If you're reading the docs cold, these six concepts carry most of the weight:

- **[Knowledge model](#knowledge-model)** — the canonical name for a `.rlat` archive (replaces v0.11 "cartridge").
- **[Verified retrieval](#verified-retrieval)** — every passage carries source provenance + content hash, so query-time results can be cited back to source and drift can be detected.
- **[Drift](#drift)** — divergence between a passage's recorded `content_hash` and the live source bytes. Three statuses: `verified` / `drifted` / `missing`.
- **[Grounding mode](#grounding-mode)** — the `--mode {augment|knowledge|constrain}` directive stamped at the top of LLM-facing markdown output, telling the consumer LLM how to weight passages vs training.
- **[Deep-research / deep-search](#deep-research--deep-search)** — the multi-hop research loop. Two surfaces (skill + CLI) running the same plan → retrieve → refine → synthesize loop.
- **[Three storage modes](#bundled-mode)** — `bundled` / `local` / `remote`, switchable in place via `rlat convert` without rebuilding.

Full alphabetical glossary below.

---

**ANN (Approximate Nearest Neighbour)** — fast retrieval index used when a knowledge model's passage count crosses a threshold. `rlat` uses FAISS HNSW (M=32, efConstruction=200, efSearch=128); index built when N > 5000 passages. Below that, exact dense matmul is sub-millisecond and the ANN overhead doesn't pay.

**Augment mode** — default grounding mode for `rlat skill-context` and `rlat search --format context`. Tells the consumer LLM to use the passages as primary context and prefer them over training when they conflict. Suppresses the dynamic body when retrieval is weak (`top1_score < 0.30` OR `drift_fraction > 0.30`). Bench 2 v4 (Microsoft Fabric, single-shot): 76.5% answerable accuracy / 3.9% hallucination, vs 56.9% / 19.6% LLM-only. The right default for broad documentation corpora where the LLM has solid prior knowledge. For fact extraction / compliance prefer `constrain`. See [Grounding mode](#grounding-mode).

**Backbone** — the encoder model + revision baked into a knowledge model at build time. Recorded in `metadata.json` (`backbone.name`, `backbone.revision`). Cross-knowledge-model operations require matching backbones (different revisions yield non-comparable embeddings).

**Band** — the embedding tensor inside a knowledge model. v2.0 has two roles: `base` (768d, always present, gte-modernbert-base output) and `optimised` (512d, optional, MRL-projected after `rlat optimise`). Bands are float32 NPZ-compressed inside the `.rlat` ZIP.

**Base band** — the universal 768d gte-modernbert-base output. Every knowledge model has one. Cross-knowledge-model operations always use it (the cross-model rule).

**Bundled mode** — storage mode where source files are zstd-compressed inside the `.rlat` archive. Self-contained — drop the file on a USB stick, ship it to a colleague, query offline. See [STORAGE_MODES.md](STORAGE_MODES.md).

**Citation** — typed RQL result: `(passage_idx, source_file, char_offset, char_length, content_hash)`. Returned by `rql.locate()`, embedded in every `CitationHit` from `rql.evidence()` etc. The build-time content_hash; live drift status requires a separate `Store.verify()` call.

**Content hash** — SHA-256 of a passage's source bytes, recorded at build time. Used at query time to detect drift (live source bytes hashed and compared).

**ConfidenceMetrics** — calibration data returned by `rql.evidence()`: `top1_score` (cosine of the top-1 hit — "is retrieval working at all"), `top1_top2_gap` (separation of best from runner-up — informational, not gated on because paraphrase clusters invert its meaning), `source_diversity` (unique sources / K), `drift_fraction` (non-verified hits / K), `band_used` (base or optimised).

**Connected components clustering** — single-linkage clustering algorithm used by `near_duplicates` + `merge` dedupe + memory consolidation. A and C cluster together if there's any path of pairwise-above-threshold edges connecting them, even if A↔C is below threshold directly. Lifted to `field/algebra.greedy_cluster` (one home for the algorithm + threshold-tuning history).

**Constrain mode** — opt-in grounding mode that tells the consumer LLM to answer ONLY from the retrieved passages and refuse explicitly when the passages don't cover the question. The mode-gate is disabled (no body suppression) — the LLM does the refusal. Bench 2 v4 (Microsoft Fabric, single-shot): 66.7% accuracy / 2.0% hallucination / **91.7% distractor refusal** (highest in the suite) — trades 10 pp answerable accuracy for halving the hallucination rate vs the default `augment`. The recommended posture for fact extraction, compliance, regulatory, and audit work where wrong-but-confident is worse than no answer. Pair with `--strict` so drifted source aborts the skill load. See [Grounding mode](#grounding-mode).

**Deep-research / deep-search** — multi-hop research loop (plan → retrieve → refine → synthesize) that returns a synthesised answer plus an evidence union with citations. Two surfaces ship in v2.0, both running the same loop:

- **`deep-research` skill** (`.claude/skills/deep-research/SKILL.md`) — drives the loop natively in your Claude Code session. **No API key required**; your Claude subscription covers the LLM hops. The right pick for nearly every interactive scenario.
- **`rlat deep-search` CLI verb** — same loop, exposed as a CLI command for non-Claude-Code agents, CI pipelines, batch jobs. **Requires an Anthropic API key** (~$0.009-0.025/q; calls Sonnet 4.6 internally). See [CLI.md §rlat deep-search](CLI.md#rlat-deep-search) and [API_KEYS.md](API_KEYS.md).

Bench (Microsoft Fabric, relaxed rubric, measured on the API surface): 92.2% accuracy / 0% hallucination — same numbers apply to the skill version with small variance from Sonnet-version + tool-use mechanic differences.

**Corpus description** — short string passed to `rlat optimise --corpus-description "..."` (e.g. `"Microsoft Fabric documentation"`). Conditions both the style-anchor and per-query LLM calls so synth queries match the register of real users on that corpus. Omitting it is the most common cross-corpus replication failure mode — always pass it for non-trivial corpora.

**Cross-model rule** — operations spanning two knowledge models always use the base band. Optimised bands are corpus-specific projections; their dimensions and orientations aren't comparable across knowledge models. `corpus_diff` and `merge` additionally require backbone-revision match (raise on mismatch); `compare` warns but proceeds.

**CLS pooling** — taking the [CLS] token's hidden state as the sentence embedding. The encoder convention `rlat` uses; matches the gte-modernbert-base recipe.

**Drift** — divergence between a passage's recorded `content_hash` and the live source bytes. Three statuses: `verified` (hash matches), `drifted` (source exists but hash mismatches), `missing` (source file no longer exists). Computed at query time by `Store.verify()`.

**Encoder** — the gte-modernbert-base model that produces 768d L2-normalised embeddings. Single recipe — no preset, no pooling toggle, no projection knob. Pinned to a specific HF revision (`PINNED_REVISION` in `install/encoder.py`).

**Episodic tier** — middle layer of `LayeredMemory`: per-session records, 14-day half-life, 2,000-entry cap. Promotes to semantic via `rlat memory consolidate` when a record recurs ≥3 times.

**Field** — the router layer of `rlat`'s three-layer architecture. Encoder + dense cosine retrieval + ANN. See [FIELD.md](../internal/FIELD.md).

**Grounding mode** — directive stamped at the top of LLM-facing markdown by `rlat skill-context` and `rlat search --format context`. Three modes: `augment` (default — passages are primary context blended with LLM training; bench 2 v4 single-shot: 76.5% accuracy / 3.9% hallucination on Microsoft Fabric docs), `constrain` (passages are the only source of truth, refuse on thin evidence; 66.7% / 2.0% / 91.7% distractor refusal, pick for compliance / audit work), `knowledge` (passages supplement training, lighter gate — single-shot 70.6% / 5.9%; under `deep-search` ties augment at 0% hallucination). Selected via `--mode`. The directive header always ships even when the dynamic body is suppressed by the gate. See [SKILLS.md §Grounding modes](SKILLS.md#grounding-modes).

**Knowledge mode** — opt-in grounding mode that tells the consumer LLM to use the passages as a supplement to training knowledge for the corpus's domain. Lighter gate than `augment` (suppresses only on `top1_score < 0.15`; drift is allowed) because the directive already invites training-blending. Choose this for domains where training data may be stale but partial knowledge is still useful. See [Grounding mode](#grounding-mode).

**Knowledge model** — the canonical name for a `.rlat` archive. Replaces the v0.11 term "cartridge."

**LayeredMemory** — three-tier append-only memory (working / episodic / semantic). Independent from knowledge models. See [MEMORY.md](../internal/MEMORY.md).

**Local mode** — default storage mode. Source files stay on disk; the `.rlat` archive carries embeddings + coordinates + a recorded `source_root`. Drift detection works against live filesystem state.

**MRL (Matryoshka Representation Learning)** — the technique behind the optimised band. Trains a (768 → 512) projection where the first 64/128/256 dims are also valid embeddings. Locked dim slate: {64, 128, 256, 512}.

**Name-check / `--strict-names`** — distinctive-token verification applied at the grounding emit boundary. Extracts proper nouns, acronyms, and alphanumeric IDs from the question, presence-checks each against the rendered passage text. Default behaviour: prepend a refusal directive when a token is missing from all passages. Under `--strict-names`: exit non-zero (rc=3). Catches the name-aliasing distractor failure mode that score-based gating cannot — canonical case from bench 2 distractor analysis (fb37: question said `MVE`, corpus said `MLV`, all single-shot rlat lanes hallucinated). Wired into `rlat skill-context`, `rlat search --format context`, and `rlat deep-search`. See [docs/internal/benchmarks/02_distractor_floor_analysis.md](../internal/benchmarks/02_distractor_floor_analysis.md).

**Passage** — a chunked text fragment from a source file. Built by the `passage_v1` chunker (paragraph → sentence → hard split fallback). Defaults: 200-3200 chars per passage. Tracked in `passages.jsonl` inside the archive with line-implicit `passage_idx`.

**Passage_idx** — zero-based row index identifying a passage. Maps to a row in every band tensor and a line in `passages.jsonl`. The bridge between the field layer (band rows) and the store layer (source coordinates).

**Recall** — what `rlat memory recall` does. Fuses scores from all three memory tiers via `cosine × tier_weight × salience`. Default tier weights: working=0.5, episodic=0.3, semantic=0.2.

**Registry** — the list of `PassageCoord` objects mapping passage_idx → source coordinates. Stored as `passages.jsonl` inside the `.rlat` ZIP.

**rlat** — the CLI entry point. Also the package name on import (`resonance_lattice` is the Python module, `rlat` is the command).

**RQL (Resonance Query Language)** — the typed Python API surface. 14 ops grouped Foundation / Comparison / Composition / Evidence / Experimental. CLI subcommands wrap a subset; the rest are library-callable. See [RQL.md](../internal/RQL.md).

**Salience** — per-entry weighting in `LayeredMemory` (0.0-2.0 typical). Multiplies the cosine score at recall time so a high-salience entry surfaces above an equally-relevant low-salience entry. Default 1.0.

**Semantic tier** — top layer of `LayeredMemory`: consolidated knowledge, infinite half-life, 20,000-entry cap. Cap-pruned by relevance only — never decays.

**Skill-context** — `rlat skill-context km.rlat --query Q [--query Q ...]`. CLI command designed as the body of an Anthropic skill `!command` block — its stdout replaces the placeholder before the model sees the skill, injecting query-relevant grounded passages. Output carries citation anchors, drift status per passage, a `ConfidenceMetrics` header, and a `--mode` directive. See [SKILLS.md](SKILLS.md).

**Source coordinates** — the `(source_file, char_offset, char_length)` triple identifying a passage's exact byte span in its source file. Plus `content_hash` for drift detection, all together = `PassageCoord`.

**Source root** — the filesystem directory `source_file` paths in a knowledge model are relative to. Local mode requires it (recorded in `build_config.source_root` at build time; CLI `--source-root` overrides).

**Optimised band** — the optional 512d projection added by `rlat optimise`. Trained per-corpus from LLM-generated synth queries. Verified +3.2 pt R@5 lift on Microsoft Fabric documentation (62K passages); regresses on BEIR nfcorpus (−0.043 nDCG@10) where test queries are register-shifted from the synth distribution. Opt-in; ships base-only by default. See [OPTIMISE.md](OPTIMISE.md).

**Store** — the authoritative content layer of `rlat`'s three-layer architecture. Three modes ship in v2.0: bundled / local / remote. Local archives reconcile via `rlat refresh`; remote archives via `rlat sync` (both incremental, both land on `store/incremental.py`). `rlat freshness` is the read-only drift check on remote archives — useful as a CI gate before running sync. Resolves `(source_file, char_offset, char_length)` to text + computes drift status. See [STORE.md](../internal/STORE.md).

**Stable passage_id** — `sha256(source_file + char_offset + char_length)[:16]`. The identity that survives `rlat refresh` / `rlat sync` deltas. A passage that changes content but keeps its source-file slice keeps the same `id`; a passage that moves within a file gets a new `id`. Without stable ids, refresh/sync deletes shift every later `passage_idx` and silently break verified-retrieval citation chains, `corpus_diff` continuity, and any consumer that bookmarked `passage_idx`. Audit 07 §"Identity decision" for the design rationale. v4.1 archives emit it explicitly; legacy v4 archives compute it on load.

**Synth queries** — LLM-generated (Claude Sonnet 4.6) natural-language questions about each passage in a corpus. The training set for `rlat optimise`. Cached per-corpus at `~/.cache/rlat/optimise/<corpus_fingerprint>/` (key includes the full passage text, source files, corpus_description, model, and a schema version) so re-runs of the same corpus hit zero LLM cost.

**Tier (memory)** — one of `working` / `episodic` / `semantic`. Each tier has its own retention policy (half-life + cap) and recall weight.

**Verified retrieval** — the contract that every passage in every `.rlat` carries source provenance + content hash, so query-time results can be cited back to source and drift can be detected. `rlat`'s single biggest architectural advantage vs. opaque-embedding-store retrievers.

**Working tier** — bottom layer of `LayeredMemory`: session-local scratch, 1-day half-life, 200-entry cap. The "what we just talked about" layer; gets `gc`'d frequently.
