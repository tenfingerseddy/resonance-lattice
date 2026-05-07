# CLI Reference

`rlat` is the command-line interface to a knowledge model. It's the only one shipping in v2.0 — there's no SDK surface, no MCP server.

The complete v2.0 surface, in dispatch order:

```bash
rlat install-encoder        # one-time: download + convert gte-modernbert-base
rlat build <sources...>     # build a knowledge model from text sources
rlat optimise km.rlat       # opt-in: train an in-corpus MRL optimised band
rlat search km.rlat "..."   # single-shot top-k retrieval
rlat deep-search km.rlat "..."   # multi-hop research loop (Anthropic API key required; the in-session `deep-research` skill is the free equivalent)
rlat profile km.rlat        # corpus shape / cluster / drift summary
rlat compare a.rlat b.rlat  # cross-knowledge-model comparison (base band)
rlat summary km.rlat        # generate a context primer
rlat memory <subcmd>        # flat memory (add | list | recall | train | doctor | hook | capture | migrate | gc; distil/feedback stubbed)
rlat refresh km.rlat        # local-mode incremental delta-apply
rlat sync km.rlat           # remote-mode incremental delta-apply
rlat freshness km.rlat      # remote: read-only drift check
rlat skill-context km.rlat --query "..."  # LLM-skill !command context block
rlat convert km.rlat --to {bundled|local|remote}   # change storage mode in place
rlat init-project           # one-command project setup
```

This page documents the full v2.0 surface. Every subcommand listed above is wired and ships.

---

## `rlat build` *(shipped Phase 3 #25)*

Build a knowledge model from one or more source paths.

```bash
rlat build <source...> -o knowledge_model.rlat \
  [--store-mode bundled|local|remote] \
  [--kind corpus|intent] \
  [--source-root PATH] \
  [--min-chars N] [--max-chars N] \
  [--batch-size N] \
  [--ext EXT [--ext EXT ...]]
```

### Examples

```bash
# Default: local-mode (smallest .rlat, source stays on disk)
rlat build ./docs ./src -o my-corpus.rlat

# Bundled (self-contained, ship anywhere)
rlat build ./docs ./src -o my-corpus.rlat --store-mode bundled

# Restrict to specific extensions (override the default text-file allowlist)
rlat build ./notebooks -o nb.rlat --ext .ipynb --ext .md

# Pin source-root explicitly when you want passage_file paths relative to a
# different ancestor than the auto-detected common root
rlat build ./repo/src -o code.rlat --source-root ./repo
```

### Flags

| Flag | Default | Effect |
|---|---|---|
| `<sources...>` | (required) | One or more files or directories. Walked recursively for directories. |
| `-o`, `--output` | (required) | Output `.rlat` path. |
| `--store-mode` | `local` | How the knowledge model resolves source files at query time. See [STORAGE_MODES.md](./STORAGE_MODES.md). |
| `--kind` | `corpus` | Knowledge-model kind tag. v2.0 ships the tag only; intent operators are deferred. |
| `--source-root` | common ancestor of `<sources>` | Root for `source_file` paths recorded in the registry. |
| `--min-chars` | `200` | Chunker minimum chunk size. |
| `--max-chars` | `3200` | Chunker maximum chunk size. |
| `--batch-size` | `32` | Encoder batch size. Smaller for tight RAM, larger for GPU throughput. |
| `--ext EXT` | built-in text allowlist | Override the file-extension allowlist. Repeatable. Leading dot optional. |

### What gets built

1. Sources are walked recursively, sorted for cross-platform stability.
2. Each file is decoded as UTF-8 (universal newlines — `\r\n` and `\r` normalised to `\n`); non-UTF-8 files are skipped with a one-line summary at the end.
3. Each file is chunked via `passage_v1`: paragraph splits → sentence splits → hard splits at `--max-chars`. Chunks below `--min-chars` are merged into adjacent ones.
4. Passages are encoded by `Alibaba-NLP/gte-modernbert-base` (CLS-pooled, L2-normalised, 768d) on the **torch** runtime — the build path always uses torch via `[build]` extras.
5. A FAISS HNSW index is built when N > 5,000 passages (M=32, efConstruction=200, efSearch=128 — see [BENCHMARK_GATE.md](../internal/BENCHMARK_GATE.md)).
6. If `--store-mode bundled`, source files are zstd-framed and stored under `source/` inside the ZIP.
7. The `.rlat` is written atomically (tmp + rename).

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Build succeeded. |
| `1` | Runtime error (no text files found under `<sources>`, chunker produced no passages). |
| `2` | Usage error (missing `<sources>`). |

### Skip summary

```text
[build] skipped 5 files: 3 decode, 2 PermissionError
```

`decode` = file isn't valid UTF-8 (binary blob misnamed `.txt`, or a file in a non-UTF-8 encoding). `PermissionError` / `FileNotFoundError` on Windows often means a file is locked or the path is a broken symlink. Silent skips would be a footgun on first builds, so the summary is unconditional.

### Building on a remote GPU (large corpora, no local GPU)

Encoding `gte-modernbert-base` on CPU runs at ~80-150 passages/sec; on a CUDA T4 it runs at ~1500-2500 passages/sec. For corpora over ~10K passages where the local machine has no GPU, the [`rlat-build-on-kaggle`](../../.claude/skills/rlat-build-on-kaggle/SKILL.md) skill walks through using Kaggle's free T4 — account setup, kernel push, polling, and pulling the `.rlat` back. The skill is opt-in; small corpora finish faster on CPU than the Kaggle round-trip would take.

---

## `rlat search` *(shipped Phase 3 #26)*

Top-k retrieval against a knowledge model. Single recipe — no rerank flag, no hybrid mode, no cascade. The optimised band is used if present (added by `rlat optimise`); otherwise the base band.

```bash
rlat search km.rlat "<query>" \
  [--top-k N] \
  [--format text|json|context] \
  [--mode augment|knowledge|constrain] \
  [--source-root DIR] \
  [--verified-only] \
  [-q | --quiet]
```

### Examples

```bash
# Quick top-10 (default), human-readable
rlat search my-corpus.rlat "how does retrieval work"

# Synthesis-ready context block — pipe straight into your LLM call
rlat search my-corpus.rlat "auth flow" --format context --top-k 5

# JSON for scripts; suppress the stderr banner
rlat search my-corpus.rlat "..." --format json -q

# Only return hits whose source hasn't drifted since build
rlat search my-corpus.rlat "..." --verified-only

# Constrain mode — context block instructs the LLM to refuse on thin evidence
rlat search compliance.rlat "what's our SOX retention policy?" \
  --format context --mode constrain
```

### Output formats

- `text` (default): `score  source_file:offset+length  [drift_status]  preview`
- `json`: list of objects with all `VerifiedHit` fields including `text` and `content_hash`
- `context`: passages concatenated with `<!-- src:offset+length score=… drift -->` delimiters until `MaterialiserConfig.token_budget` (3000 × 4 chars/token = 12K chars by default) is met. Output is prefixed by a grounding-mode directive (see `--mode`) instructing the consumer LLM how to treat the passages. Designed to be pasted/piped into a downstream LLM as primer context.

### Grounding modes (`--format context` only)

`--mode {augment|knowledge|constrain}` (default: `augment`) controls *how the consumer LLM should treat the retrieved passages*. The mode stamps a directive header at the top of the markdown output, paired with a confidence-gate that suppresses the dynamic body on weak retrieval.

- `augment` (default) — passages are primary context blended with the LLM's training. Gate fires when `top1_score < 0.30` (retrieval has genuinely failed on this query) or `drift_fraction > 0.30`; on weak retrieval the body is replaced by a *no confident evidence* marker so the LLM falls back to training instead of grounding on noise. **Default — bench 2 v4 (Microsoft Fabric, single-shot) measured 76.5% answerable accuracy / 3.9% hallucination, vs 56.9% / 19.6% for the LLM alone.** The right shape for broad documentation corpora where the LLM already has solid prior knowledge.
- `constrain` — passages are the only source of truth. No gate; the LLM is told to refuse when evidence is thin. Bench 2 v4 (single-shot): 66.7% accuracy / 2.0% hallucination / 91.7% distractor refusal — trades 10 pp answerable accuracy for halving the hallucination rate vs augment and the highest distractor refusal in the suite. Recommended for fact extraction, compliance, regulatory, and audit work where wrong-but-confident is worse than no answer.
- `knowledge` — passages supplement training. Lighter gate (`top1_score < 0.15` only — drift is allowed) because the directive already tells the LLM to lean on training. Bench 2 v4 (single-shot): 70.6% / 5.9% — under `rlat deep-search` ties augment at 0% hallucination. Choose for partial-coverage corpora where the LLM already knows the surrounding domain reasonably well; the launch recommendation when the deep-search loop runs.

The mode header always ships, even when the body is suppressed — the directive is non-negotiable.

### Drift status on every hit

Every hit carries `drift_status ∈ {verified, drifted, missing}` derived by re-hashing the slice against `PassageCoord.content_hash` recorded at build. `--verified-only` filters to verified hits only; the default surface returns everything so a consumer can decide.

### Banner

By default a one-line banner is printed to **stderr**:

```text
[search] band=base ann=yes hits=10 (3633 passages)
```

`band=base|optimised`, `ann=yes|no` (FAISS HNSW path vs exact dense), and the corpus passage count. Stderr placement keeps stdout parseable for `--format json|context` consumers. Use `-q` to suppress.

### `--strict-names` (name-aliasing safety)

Single-shot `rlat search --format context` honours the same name-verification check as `rlat skill-context`. When a distinctive proper noun, acronym, or alphanumeric ID from the query does not appear in any retrieved passage, a refusal directive is prepended to the rendered context body. Under `--strict-names`, the same condition exits non-zero (rc=3) so a calling tool can gate.

This addresses the name-aliasing distractor failure mode score-based gating cannot — see `docs/internal/benchmarks/02_distractor_floor_analysis.md`.

---

## `rlat deep-search` *(shipped Phase 7)*

> ⚠ **Requires an Anthropic API key.** This CLI verb calls Sonnet 4.6 via
> the Anthropic API for the planner / refiner / synthesizer hops; each
> invocation costs ~$0.009-0.025. **For interactive Claude Code use, the
> `deep-research` skill at `.claude/skills/deep-research/SKILL.md` runs
> the same loop natively in your session with no extra cost** — your
> Claude subscription covers the LLM hops. Same retrieval, same
> prompts, same hop budget, same name-verification check. Prefer the
> skill in nearly every Claude Code scenario; this CLI verb is the
> right pick for non-Claude-Code agents, CI pipelines, batch jobs,
> and any consumer that needs a programmatic / non-interactive
> surface. API-key setup: [docs/user/API_KEYS.md](API_KEYS.md).

Multi-hop research loop: plan → search → refine → synthesize. Returns a synthesised answer + an evidence union with anchors, in one call. Use this when correctness matters more than latency / spend AND a programmatic surface is needed.

```bash
rlat deep-search km.rlat "<question>" \
  [--max-hops N]               # default 4
  [--top-k K]                  # default 5 per hop
  [--format text|json|markdown]   # default text
  [--source-root DIR]
  [--strict-names]             # rc=3 on name-aliasing refusal
```

### What it does

1. **Plan** — Sonnet generates an initial 6-15 word query from the question.
2. **Search** — `rlat` retrieves top-k passages for the current query.
3. **Refine** — Sonnet sees the question + accumulated evidence and decides one of: `answer` (final answer ready), `search` (next query), `give_up` (corpus does not cover the question).
4. **Repeat** until the refiner picks `answer` or `max_hops` exhausts.
5. **Name-check** runs across the union of all hops' evidence vs the original question; missing distinctive tokens trigger a refusal directive (or, under `--strict-names`, replace the answer entirely).

### Bench (Microsoft Fabric corpus, 63 questions, Sonnet 4.6, relaxed rubric)

| Approach | Accuracy | Hallucination | Mean cost / q |
|---|---:|---:|---:|
| LLM alone (no retrieval) | 56.9% | 19.6% | $0.002 |
| `rlat search --format context` (single-shot, `--mode augment`) | 76.5% | 3.9% | $0.004 |
| **`rlat deep-search` (4-hop, default `--mode augment`)** | **92.2%** | **2.0%** | **$0.010** |
| **`rlat deep-search --mode knowledge`** (compliance variant) | **92.2%** | **0.0%** | **$0.009** |

Worth ~2.5× the cost of single-shot for ~16 pp accuracy lift and 2× lower hallucination on hard questions; the `knowledge`-mode variant achieves 0% answerable hallucination at the same accuracy. See `docs/user/BENCHMARKS.md` for the full 11-lane matrix.

### Anthropic API key

Resolved via the same discovery order as `rlat optimise`:

1. `$RLAT_LLM_API_KEY_ENV` (indirected — names another env var holding the key)
2. `$CLAUDE_API`
3. `$ANTHROPIC_API_KEY`

Anthropic-only for v2.0 (Sonnet 4.6 hardcoded). OpenAI / local-LLM adapters land post-launch on demand.

### Output

`text` (default): answer + cost summary + per-hop log.
`json`: full machine-readable result — answer, hops, evidence passages with anchors + drift status, name-check status.
`markdown`: drop-in shape for piping into another model — `# Answer` block + `# Evidence` block in the same shape as `rlat skill-context`.

### Example

```bash
rlat deep-search project.rlat "What is the rationale for picking gte-modernbert-base as the v2.0 encoder?"
```

```text
The encoder choice was driven by a head-to-head BEIR-5 comparison: gte-mb
hit 0.5144 mean nDCG@10 zero-cost, beating BGE-large-v1.5 by +0.026 and
E5-large-v2 by +0.081 on the same v2.0 stack (chunker / ANN / scoring
held constant). (docs/internal/BENCHMARK_GATE.md, docs/user/BENCHMARKS.md)

[deep-search] hops=4 in=2143 out=312 cost=$0.0118 evidence=8 passages
  hop 1 plan           'gte-modernbert-base BEIR comparison rationale'
  hop 2 search 'gte-modernbert-base BEIR comparison rationale' → 5 passages
  hop 3 decide_search  'BGE-large vs gte-mb 5-corpus average'
  hop 4 search 'BGE-large vs gte-mb 5-corpus average' → 5 passages
  hop 4 answer (loop terminated)
```

---

## `rlat profile` *(shipped Phase 3 #27)*

Knowledge-model diagnostics — what's in the `.rlat` without running a query. Reports backbone, bands, store mode, build config, ANN params, and a drift summary across the registry.

```bash
rlat profile km.rlat \
  [--format text|json] \
  [--source-root DIR] \
  [--no-drift]
```

### Examples

```bash
# Default text report
rlat profile my-corpus.rlat

# JSON for dashboards / scripts
rlat profile my-corpus.rlat --format json

# Skip the drift walk on huge corpora (drift verifies all N passages)
rlat profile huge-corpus.rlat --no-drift
```

### What's reported

- **Backbone**: model id, full pinned HF revision (40 chars — needed for encoder-mismatch debug), dim, pooling, max-seq-length.
- **Bands**: name → role, dim, passage_count, MRL projection shape if optimised is present.
- **Storage**: recorded `store_mode` and (when relevant) the source_root.
- **Build config**: chunker name, min/max chars, file count, build timestamp.
- **ANN**: per-band index params if a FAISS index is present.
- **Registry size**: total passage count.
- **Drift**: `verified` / `drifted` / `missing` counts plus `verified` percentage.

### Drift discriminator

JSON output's `drift` field carries an explicit `status` tag:

```json
"drift": { "status": "computed", "verified": 9234, "drifted": 12, "missing": 0 }
"drift": { "status": "skipped", "reason": "--no-drift" }
"drift": { "status": "skipped", "reason": "local-mode knowledge model has no recorded source_root..." }
```

Consumers branch on `status`, not on key presence — no sniffing required.

### When to use `--no-drift`

The drift walk re-hashes every passage (`O(N_passages)` SHA-256 calls) and reads every distinct source file. For a 100K-passage corpus across 5K files that's a few seconds; on remote-mode where every read is an HTTP miss this can be much longer. `--no-drift` returns immediately with metadata only — useful for scripted profile dumps or sanity checks on huge models.

---

## `rlat compare` *(shipped Phase 3 #28)*

Cross-knowledge-model comparison. Always uses the **base band** on both sides — optimised bands are not interoperable across knowledge models by design (each is trained against its own corpus). A optimised + non-optimised pair compares on bases just fine.

```bash
rlat compare a.rlat b.rlat \
  [--format text|json] \
  [--sample N]
```

### Examples

```bash
# Default text report — side-by-side stats + 3 similarity numbers
rlat compare project-old.rlat project-new.rlat

# JSON for dashboards
rlat compare a.rlat b.rlat --format json

# Smaller sample on huge corpora (cheaper but noisier coverage estimate)
rlat compare big-a.rlat big-b.rlat --sample 128
```

### Reported metrics

| Metric | Range | Meaning |
|---|---|---|
| `centroid_cosine` | [-1, 1] | Cosine of the two corpora's mean vectors. Single thematic-alignment number. |
| `coverage_a_in_b` | [-1, 1] | Mean of max-cosine for `--sample` rows from A vs all of B. "How well are A's passages mirrored in B?" |
| `coverage_b_in_a` | [-1, 1] | Same direction reversed. Asymmetric — corpora differ in size/focus. |

`coverage_*` is the more useful number when the corpora differ substantially in size; `centroid_cosine` is the quick thematic check.

### Backbone revision check

If A and B were built at different `backbone.revision` values (e.g. one was built before an encoder upgrade), the cosine numbers are still **ordinally meaningful** (both bands are unit vectors) but the magnitude comparison is across different embedding distributions. A warning is printed to **stderr** so JSON consumers also see it without the warning leaking into stdout.

### Determinism

Coverage sampling uses `numpy.random.default_rng(seed=0)` so identical inputs produce identical numbers across runs — required for benchmark reproductions and doc-example tests.

### Memory bound

`coverage_*` materialises a `(--sample, N_dst)` similarity matrix. For sample=512 and N=100K that's 200 MB; for huge corpora the destination is chunked along axis 0 so peak RSS stays bounded by `_COVERAGE_CHUNK_BYTES = 512 MB` regardless of corpus size.

---

## `rlat summary` *(shipped Phase 3 #29)*

Extractive primer generation. Produces a structured markdown document that captures what's in a knowledge model — designed to be loaded into an AI assistant's system prompt so it knows the corpus shape without spending tokens on a full search.

```bash
rlat summary km.rlat \
  [-o output.md] \
  [--query "..." [--query "..." ...]] \
  [--source-root DIR]
```

### Examples

```bash
# Default: print primer to stdout
rlat summary my-corpus.rlat

# Save to a file (overwrites). Common pattern for assistant integration:
rlat summary my-corpus.rlat -o .claude/resonance-context.md

# Add themed evidence sections — repeat --query per topic
rlat summary my-corpus.rlat -o primer.md \
  --query "how does retrieval work" \
  --query "storage modes" \
  --query "encoder details"
```

### Sections

| Section | Source | Token budget |
|---|---|---|
| **Landscape** | top-10 hits against the corpus centroid | `MaterialiserConfig.sections_landscape = 600` |
| **Structure** | source-file breakdown sorted by passage count | `sections_structure = 800` |
| **Evidence** | per-query top-5 hits (only when `--query` given) | `sections_evidence = 1600` |

Total budget defaults to `MaterialiserConfig.token_budget = 3000` tokens (~12K chars at 4 chars/token). The materialiser stops adding hits to a section once that section's char-budget is met; the first hit in each section is always included even if oversized — better one truncated primary passage than an empty section.

### What landscape captures

The "landscape" section runs a single retrieval **against the corpus centroid** (mean of base band, L2-normalised). The hits are the corpus's most semantically representative passages — they answer "what is this knowledge model about?" without an LLM call. Cheap, deterministic, and reproducible.

### Common workflow

The `.claude/resonance-context.md` pattern is the canonical use:

```bash
# Run after every meaningful corpus change so the assistant primer stays fresh
rlat build ./docs ./src -o my-corpus.rlat
rlat summary my-corpus.rlat -o .claude/resonance-context.md
```

The primer is markdown, not JSON — designed for direct paste into a prompt. AI assistants (including Claude Code) can read it as context without extra parsing.

---

## `rlat init-project` *(shipped Phase 3 #29)*

One-command project setup. Auto-detects source directories in the current working directory, builds a knowledge model with sensible defaults, and (unless `--no-primer`) writes a context primer to `.claude/resonance-context.md`.

```bash
cd my-project/
rlat init-project                                # auto-detects docs/, src/, README.md, etc.
rlat init-project -o custom.rlat                 # override output path
rlat init-project --source ./notes --source ./code   # explicit sources, skip auto-detect
rlat init-project --no-primer                    # skip .claude/resonance-context.md
```

Equivalent to running:

```bash
rlat build <detected-sources> -o <cwd>.rlat
rlat summary <cwd>.rlat -o .claude/resonance-context.md
```

Default auto-detection looks for `docs/`, `src/`, `lib/`, `notebooks/`, `examples/` and top-level `*.md`/`*.rst`/`*.txt`. If none of those exist, the command errors and asks you to be explicit with `--source` rather than producing an empty knowledge model.

## `rlat install-encoder` *(shipped Phase 1 / Phase 3 wiring)*

Pre-stage the gte-modernbert-base encoder cache. Most users never run this directly — the first `rlat build` triggers it automatically. Useful for offline-build environments and CI.

```bash
rlat install-encoder                  # default pinned revision
rlat install-encoder --force          # re-download + re-convert even if cache exists
rlat install-encoder --revision <sha>  # pin to a different HF revision
```

The cache lives at `~/.cache/rlat/encoders/<revision>/`. ONNX is always exported; OpenVINO IR is exported additionally on Intel CPUs.

---

## `rlat convert` *(shipped Audit 08)*

Switch a knowledge model between storage modes (`bundled` / `local` / `remote`) **without rebuilding embeddings**. The bands, registry, ANN index, and optimised W projection are storage-mode-independent — conversion is just a metadata + payload-shuffle, atomic in-place by default.

```bash
rlat convert km.rlat --to {bundled|local|remote} \
  [--source-root DIR] \
  [--remote-url-base URL] \
  [-o output.rlat] \
  [--dry-run]
```

### Examples

```bash
# Hand off a working knowledge model as a self-contained artefact
rlat convert my-corpus.rlat --to bundled

# Open a downloaded knowledge model so you can edit the underlying files
rlat convert handout.rlat --to local --source-root ./extracted/

# Publish a private corpus to an HTTP endpoint
rlat convert internal.rlat --to remote --remote-url-base https://docs.example.com/v1/

# Run optimise on a remote corpus by materialising locally first
rlat convert upstream.rlat --to local --source-root ./local-mirror/ -o working.rlat
rlat optimise working.rlat --corpus-description "..."
```

### Pairwise transitions

| From → To | Required flags | What happens |
|---|---|---|
| `local → bundled` | (none) | Read each source file from disk, pack as zstd frames into `source/`. |
| `local → remote` | `--remote-url-base` | Compose `manifest.json` from URL prefix + `sha256_hex(text)` per file; drop `source/`. |
| `bundled → local` | `--source-root` | Extract `source/` zstd frames to disk; drop `source/` from archive. |
| `bundled → remote` | `--remote-url-base` | Extract zstd frames; compose manifest; drop `source/`. |
| `remote → local` | `--source-root` | Walk manifest, download (cache-first, SHA-pin verified), materialise to disk. |
| `remote → bundled` | (none) | Walk manifest, download into cache, pack zstd frames. |

`--to <current_mode>` is a friendly idempotent no-op (rc=0, no write).

### Three correctness invariants

1. **Drift abort.** Every kept passage's `content_hash` is re-validated against live bytes resolved via the source mode's Store before write. If any drift, conversion **aborts** with the drift report (rc=2) — does not silently carry stale bytes into the new mode. Run `rlat refresh` (local) or `rlat sync` (remote) first to reconcile, then retry.

2. **Bands semantically identical pre/post** (`np.allclose` at 1e-6). Storage mode doesn't change corpus identity. Strict byte equality does not hold — `archive.write_band` defensively re-runs L2-normalisation, which introduces ULP-level float32 drift (~6e-8). Cosine retrieval is unaffected at this precision.

3. **`metadata.store_mode` advances atomically with the payload swap** via `tmp + os.replace`. A crash mid-convert leaves the original archive untouched.

`passage_id` and `content_hash` are preserved across every conversion — verified-retrieval citations and external `passage_id` bookmarks survive intact.

`--dry-run` reports the transition + passage/file counts without reading source bytes or writing.

---

## `rlat refresh` *(shipped Phase 3, rewritten as incremental delta-apply in Audit 07)*

Bring a local-mode knowledge model back in sync after you edit source files. Incremental — unchanged passages keep their band rows untouched; only modified/added passages are re-encoded.

```bash
rlat refresh my-corpus.rlat                              # apply all deltas
rlat refresh my-corpus.rlat --dry-run                    # report counts only
rlat refresh my-corpus.rlat --discard-optimised          # rare opt-out
```

Local mode only. Bundled archives are immutable post-build (re-run `rlat build`). Remote archives route to `rlat sync`.

The pipeline:

1. Walk the source paths recorded in `metadata.build_config` (or `--source` / `--source-root` overrides) — same chunker as the build path.
2. Bucketise candidates against the existing registry on stable `passage_id`: unchanged / updated / added / removed.
3. Re-encode `updated + added` passages once, batched.
4. Compose the new band by `np.vstack`-ing kept rows (byte-identical lifts from the old band) with newly-encoded rows.
5. Rebuild the FAISS HNSW index if `N` crosses the threshold.
6. Re-project the optimised band (if present) from the new base — `optimised = (new_base @ W.T)` row-wise L2-normalised. **Free** — no LLM call, no GPU. The W matrix is preserved byte-identically.
7. Atomic in-place write via `tmp + os.replace`.

`--dry-run` walks + bucketises + reports the four delta counts without writing — useful as a CI "would refresh do anything?" gate.

`--discard-optimised` drops the optimised band on refresh instead of re-projecting. Rare — the re-projection is free, so this opt-out is for users who specifically want a base-only archive afterwards.

Override flags: `--source <dir>` (repeatable) ingests a different set of paths than recorded; `--source-root <dir>` overrides the registered root; `--ext <ext>` (repeatable) overrides the extension allowlist; `--batch-size N` for tight RAM.

## `rlat watch` *(shipped post-2.0)*

Live, silent, self-discovering refresh loop. Sits on top of `rlat refresh` — runs the incremental delta-apply pipeline on every save after a debounce window, so the archive stays current as you edit.

```bash
rlat watch                          # zero-arg: discover *.rlat in cwd, watch all
rlat watch my-corpus.rlat           # explicit single archive
rlat watch --once                   # CI / pre-commit: synchronous one-shot reconcile
rlat watch --verbose                # print one line per refresh (default: silent)
```

Default UX is silent — errors are loud, success is invisible. A short startup line confirms what's being watched, then the loop falls quiet. On `Ctrl-C`, prints a single summary: elapsed time, total refreshes, error count.

Multi-archive: zero-arg invocation auto-discovers `*.rlat` files in the current directory (non-recursive) and watches every one of their recorded source roots. FS events fan out to every archive whose `build_config.source_paths` covers the changed path — overlapping source roots get a single watchdog subscription rather than N copies.

Local mode only. Bundled archives are immutable post-build (`rlat convert <km> --to local` first). Remote archives route to `rlat sync` for upstream reconciliation.

Pre-flight checks at startup, all fail-fast with actionable hints:

- archive loads cleanly + has recorded `source_paths` provenance
- store mode is `local`
- every recorded source path exists on disk

The mental model is **events are hints to reconcile, not the unit of correctness.** Each FS notification debounces a refresh; the refresh itself does a full source-tree walk + `bucketise` against the archive registry. Over-triggering is bounded (empty delta = no archive write); under-triggering would be a correctness hole.

Filtering on every event:

1. Block known-noisy directory components (`.git/`, `node_modules/`, `__pycache__/`, `.venv/`, `.tox/`, `.mypy_cache/`, `.pytest_cache/`, `.idea/`, `.vscode/`, `.ruff_cache/`).
2. For modify/create events: drop paths whose extension isn't in the archive's recorded `extensions` allowlist (cheap pre-filter; also drops editor swap files like `.swp` / `.tmp`).
3. For deletes, moves, and directory events: skip the suffix pre-filter (rename-out scenarios like `foo.md → foo.bak` and directory deletes still need reconciliation; the path's *current* suffix is uninformative). Move events dispatch on both `src_path` and `dest_path`.
4. Re-arm the per-archive debounce timer.

After `RLAT_WATCH_DEBOUNCE_MS` (default `1000`) of idleness, refresh fires. Concurrency hazard: per-archive `threading.Lock` serialises refreshes — `apply_delta` writes via `<archive>.tmp` + `os.replace`, and two concurrent calls share that tmp path. The lock closes the race.

**Transient read failures are NOT silent deletes.** If `_walk_sources` skips a file (Windows file lock during atomic save, mid-write UTF-8 decode error), `bucketise` would normally classify its passages as `removed` and `apply_delta` would write that deletion. The watch path post-processes the delta: any removal whose `source_file` is in the skipped set is demoted back to `unchanged`, and the next FS event drives the next refresh that reconciles for real.

The encoder pre-loads at startup so the first event refreshes immediately (no model-load latency on the first save). Optimised bands re-project for free from the new base on every refresh, same as `rlat refresh`.

`--once` is the CI / pre-commit shape — **synchronous one-shot reconciliation, no observer.** Walks the tree of every preflighted archive, runs `bucketise` + `apply_delta` against current disk state, prints a summary, and exits. No waiting for events: in CI / pre-commit / formatter-on-save / `git checkout` flows the files are already changed before the command runs, so an event-waiting `--once` would hang. If the archives are already up to date, exits silently with code 0.

Requires the `[watch]` extra: `pip install rlat[watch]`. Pulls in `watchdog` (cross-platform FS event delivery) and the `[build]` deps (transformers + torch) for re-encoding.

## `rlat sync` *(shipped Audit 07)*

Bring a remote-mode knowledge model back in sync against upstream. Same incremental delta-apply as `refresh`, with the upstream-state oracle replaced by a `RemoteIndex` that polls the network.

```bash
rlat sync my-corpus.rlat                                 # poll mode (default)
rlat sync my-corpus.rlat --upstream-manifest <url>       # catalog mode
rlat sync my-corpus.rlat --dry-run                       # list paths only
```

Two modes:

- **Catalog mode** (`--upstream-manifest <url>`) — upstream serves a stable JSON manifest endpoint listing `{source_file: {url, sha256}}` for the current corpus. Sync diffs upstream against the archive's pinned manifest in O(1) network calls; detects `added` + `modified` + `removed` paths.
- **Poll mode** (default) — re-fetches every URL in the existing manifest, hashes the bytes, diffs against the pinned `sha256`. Detects `modified` + `removed` only — cannot enumerate NEW files because there's no upstream signal for that. Use catalog mode if your corpus adds files often.

After sync, the in-archive `manifest.json` and `metadata.build_config.pinned_ref` advance atomically with the band write. There is no window where the manifest is current but the bands are stale (codex P0 correctness gate, baked in by signature — `apply_delta` requires the encoder).

`--dry-run` lists added/modified/removed paths with `+/~/-` prefixes without fetching or writing — CI-friendly.

`--discard-optimised` is the same opt-out as `refresh`.

`GitHubCompareIndex` (uses GitHub's compare API for git-hosted corpora — cheaper than poll mode, no need for a stable manifest endpoint) is a v2.1 follow-up.

## `rlat freshness` *(remote mode only — shipped Phase 7)*

```bash
rlat freshness my-corpus.rlat                # text report (per-entry status)
rlat freshness my-corpus.rlat --format json  # structured (counts + per-entry map)
```

Remote-mode only — bundled / local archives print a friendly error pointing at `rlat build` / `rlat refresh`.

`freshness` walks every entry in the in-archive manifest, downloads the upstream bytes, hashes them, and compares to the pinned SHA. Per-entry status is one of:

- `verified` — upstream still serves the bytes we built against
- `drifted` — upstream exists but content has changed
- `missing` — upstream returned an error (404 / timeout / etc.)

Read-only. Exits with code `2` if any entry is non-verified — suitable for CI gating before deciding to run `rlat sync`.

## `rlat optimise` *(shipped Phase 4)*

Adds the MRL optimised band to a knowledge model. Opt-in. Costs ~$8-15 in LLM calls (Claude Sonnet 4.6) per 40K-passage corpus; runs once and writes back in place.

```bash
rlat optimise my-corpus.rlat --estimate            # dry run — print cost + duration
rlat optimise my-corpus.rlat                       # interactive (confirms cost)
rlat optimise my-corpus.rlat --yes                 # skip the prompt
rlat optimise my-corpus.rlat --cache-dir ./cache   # custom synth-query cache
```

Requires `[optimise]` extras (torch + anthropic) and an LLM API key via `CLAUDE_API`, `ANTHROPIC_API_KEY`, or the `RLAT_LLM_API_KEY_ENV` indirection. After a successful run, every `rlat search` automatically uses the optimised band — no flag needed.

See [OPTIMISE.md](./OPTIMISE.md) for the full guide (when to optimise, cost calibration, caching, idempotency).

---

## `rlat memory <sub>` *(v2.1 flat-memory; MVP)*

Per-user flat memory at `~/.rlat/memory/<user-id>/` — one `memory.npz`
embedding band + one `sidecar.jsonl` row file per user. Polarity-tagged
rows (`prefer | avoid | factual` plus scope tags `workspace:<hash>` /
`cross-workspace`) drive the §0.6 retrieval pipeline. The full §0.7
surface is documented below; **add**, **list**, **gc**, **recall**
(both daemon mode AND synchronous one-shot), **train** (operator flags),
**doctor**, **hook** (UserPromptSubmit), **capture** (SessionEnd), and
**migrate** ship today; **distil** and **feedback** print a "ships in
v2.1 MVP" banner with exit code 3 pending post-MVP work.

```bash
# Add a manual row — defaults to polarity=factual + cwd workspace tag
rlat memory add "prefer pytest -xvs <path> for debugging" --polarity prefer
rlat memory add "avoid 'from rlat import *' in code"      --polarity avoid
rlat memory add "encoder revision pinned"                  --polarity factual

# Cross-workspace habits — visible in any cwd
rlat memory add "prefer Sonnet 4.6 over Haiku 4.5 for distil" \
    --polarity prefer --scope cross-workspace

# Read from stdin
cat note.md | rlat memory add - --polarity factual

# List — tabular view of the sidecar
rlat memory list                              # all (is_bad rows hidden)
rlat memory list --polarity prefer            # filter by primary tag
rlat memory list --min-recurrence 3           # only well-corroborated rows
rlat memory list --include-bad                # show is_bad rows too
rlat memory list --format json                # machine-parseable

# Garbage collect — manual escape hatch only (§0.5).
# Refuses to run without at least one filter; `--dry-run` previews.
rlat memory gc --is-bad --dry-run                       # what would go
rlat memory gc --is-bad                                 # delete is_bad rows
rlat memory gc --polarity factual --max-age-days 90     # stale lookup-cache
rlat memory gc --min-recurrence 1                       # one-off rows

# Per-user override — defaults to $RLAT_MEMORY_USER → $USER → $USERNAME
rlat memory --user alice list

# Custom base directory — composes with --user as <root>/<user>/
rlat memory --memory-root ./team-mem --user kane add "..."
```

### Subcommand surface (§0.7)

| Subcommand | Status | Purpose |
|------------|--------|---------|
| `add`        | shipped (Sub-MVP) | Append a manual row; auto-stamps cwd workspace tag. |
| `list`       | shipped (Sub-MVP) | Tabular sidecar view with optional filters. |
| `gc`         | shipped (Sub-MVP) | Manual escape-hatch deletion; never automatic (§0.5). |
| `recall`     | shipped (MVP D7-9) | `--daemon` boots the long-lived recall server; one-shot synchronous body honours `--top-k`/`--polarity`/`--format`/`--explain`. |
| `train`      | shipped (MVP D5-6; operator flags) | `--bad-vote`/`--good-vote`/`--corroborate` mutate single rows; full §8 GRPO loop runs via `/rlat-train`. |
| `doctor`     | shipped (MVP D7-8) | Probe per-user store + daemon (POSIX socket / Windows pipe) + encoder revision. |
| `hook`       | shipped (MVP D9-10) | UserPromptSubmit hook entry point. Reads JSON envelope from stdin, emits §0.4 `<rlat-memory>` block via `additionalContext` to stdout. Always rc=0 (fail-open). |
| `migrate`    | shipped (MVP D11-12) | One-shot v2.0-tiered → v2.1-flat migration; lossy by design (§14.5). Deleted in v2.2. |
| `distil`     | pending (post-MVP, #88)           | LLM distil over recent transcripts → expertise rows. (Library API ships via `memory.distil`; CLI body deferred.) |
| `feedback`   | pending (post-MVP, #88)           | Log thumbs-up/down on the most recent injection. |
| `consolidate`| **removed**       | Replaced by `rlat memory distil`. Banner + exit 2. |
| `primer`     | **removed**       | Replaced by per-prompt `UserPromptSubmit` hook (no static primer in v2.1; see §17.3). |

### Flags

| Subcommand | Flag | Default | Effect |
|---|---|---|---|
| (top-level) | `--memory-root` | `~/.rlat/memory/` | Base directory; per-user subdir always nests inside (`<root>/<user>/`). |
| (top-level) | `--user`        | `$RLAT_MEMORY_USER` → `$USER` → `$USERNAME` | User id picking the per-user store. |
| `add`     | `text` (positional) | (required) | Row text. Pass `-` to read stdin. |
| `add`     | `--polarity`        | `factual`  | One of `prefer`, `avoid`, `factual` (§0.3 closed primary). |
| `add`     | `--scope`           | (none)     | Set to `cross-workspace` to add the cross-workspace tag. The cwd workspace tag is **always** stamped automatically. |
| `list`    | `--polarity`        | (none)     | Filter to rows containing this tag. |
| `list`    | `--min-recurrence`  | (none)     | Show only rows with `recurrence_count >= N`. |
| `list`    | `--limit`           | (none)     | Cap the row count after filtering. |
| `list`    | `--include-bad`     | off        | Show `is_bad: true` rows (hidden by default). |
| `list`    | `--format`          | `text`     | `text` (Row.summary form) or `json` (full sidecar dict). |
| `gc`      | `--polarity`        | (none)     | Filter targets to rows containing this tag. |
| `gc`      | `--min-recurrence`  | (none)     | Delete rows with `recurrence_count <= N` (note: opposite sense from `list`). |
| `gc`      | `--max-age-days`    | (none)     | Delete rows whose `last_corroborated_at` is older than N days (per §15.2 — corroboration resets the clock so a periodically-bumped row never ages out). |
| `gc`      | `--is-bad`          | off        | Delete rows tagged `is_bad: true`. |
| `gc`      | `--dry-run`         | off        | Print what would go; don't write. |
| `recall`  | `query` (positional) | (required for one-shot) | Query text; ignored under `--daemon`. |
| `recall`  | `--daemon`          | off        | Boot the long-lived recall server (idle-exits per §0.8). |
| `recall`  | `--top-k`           | `5`        | Maximum hits to return on synchronous one-shot. |
| `recall`  | `--polarity`        | (none)     | Post-filter to hits with this primary polarity tag. |
| `recall`  | `--format`          | `text`     | `text` (Row.summary lines) or `json` (full hits dict). |
| `recall`  | `--explain`         | off        | Append per-hit cosine score to text output. |
| `train`   | `--bad-vote ROW_ID` | (none)     | Mark a row `is_bad=True` (drops from recall). |
| `train`   | `--good-vote ROW_ID`| (none)     | Reverse a bad-vote: `is_bad=False`. |
| `train`   | `--corroborate ROW_ID` | (none)  | Bump `recurrence_count` + refresh `last_corroborated_at`. |
| `train`   | `--why TEXT`        | (none)     | Optional rationale recorded in the train audit log alongside `--bad-vote`. |
| `migrate` | `v20_root` (positional) | (required) | Path to the v2.0 LayeredMemory root to migrate from. |
| `migrate` | `--to PATH`         | (required) | v2.1 base directory — the per-user subdir is created inside. |
| `migrate` | `--migrate-user ID` | (required) | User id for the v2.1 store (`<to>/<id>/`); distinct from the top-level `--user`. |
| `migrate` | `--dry-run`         | off        | Preview the migration without writing or archiving the v2.0 root. **Recommended first invocation.** |
| `migrate` | `--polarity-default`| `factual`  | Polarity for rows the verb-scan heuristic doesn't classify (§14.5). |

`gc` requires **at least one** filter — bare `rlat memory gc` returns
exit 1 with the explicit "manual escape hatch, not a sweep" message.

### Exit codes

- `0` — success.
- `1` — usage / runtime error: empty text on `add`, missing filter on `gc`, invalid polarity, mutually-exclusive `train` operator flags.
- `2` — deprecated subcommand (`consolidate`, `primer`); banner points at the v2.1 successor.
- `3` — pending post-MVP subcommand (`distil`, `feedback`, plus `train <task>` GRPO body); banner names the tracking issue.

The split between exit 2 (removed forever) and exit 3 (ships in MVP)
lets shell scripts distinguish "you typed something wrong" from "wait
for the next release" from "this command is gone".

### Storage layout

```
~/.rlat/memory/<user-id>/
├── memory.npz              # (N, 768) float32 embedding band
├── sidecar.jsonl           # one JSON row per band row, §0.2 schema
├── .lock                   # portalocker advisory lock for concurrent writes
├── .recall.sock            # (POSIX) IPC socket; Windows uses \\.\pipe\rlat-memory-<hash>
├── .daemon_authkey         # 32-byte random authkey (POSIX 0o600); per-root cross-tenant guard
├── redaction.log           # append-only Layer-1 audit log
├── train_audit.log         # append-only train-operator audit (bad/good vote + corroborate)
└── .distil_watermark.json  # (D11-12) last_processed timestamp + transcript_hash
```

Empty stores are auto-created on first call. `rlat memory list` against
an empty store prints `(no rows match)`.

For the §0.2 9-field row schema, retrieval pipeline (§0.6), polarity
vocabulary (§0.3), redaction layers (§6.2), and the full design
context, see
[.claude/plans/fabric-agent-flat-memory.md](../../.claude/plans/fabric-agent-flat-memory.md).

---

## `rlat skill-context` *(shipped Phase 7)*

Markdown context block for the Anthropic skill `!command` dynamic-injection primitive. Designed to land directly in a skill body via:

```markdown
!`rlat skill-context km.rlat --query "$user_query" --top-k 5`
```

The shell command runs **before the model sees the skill**; stdout replaces the placeholder. Output carries citation anchors, drift status per passage, and a `ConfidenceMetrics` header line so the model can self-judge. See [SKILLS.md](SKILLS.md) for the integration walkthrough.

### Examples

```bash
# Single dynamic query
rlat skill-context fabric-docs.rlat --query "how does OneLake work" --top-k 5

# Preset + user query in one call
rlat skill-context fabric-docs.rlat \
   --query "Fabric workspace fundamentals" \
   --query "$user_query" --top-k 3

# Strict mode — abort if any source has drifted
rlat skill-context fabric-docs.rlat --strict --query "..."

# Constrain mode — instruct the LLM to refuse when evidence is thin
rlat skill-context compliance.rlat --mode constrain --strict --query "$user_query"
```

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `knowledge_model` (positional) | (required) | Path to a `.rlat` knowledge model |
| `--query` | (required, repeatable) | Retrieval query. Pass multiple flags for preset + user-query injection in one call. |
| `--top-k` | `5` | Top-k passages per query. |
| `--token-budget` | `4000` | Cap on per-query block output. Lower-priority (later) blocks dropped first; the first block always survives. The grounding-mode directive header and drift banner ship outside the budget — the directive is a small fixed-size instruction the consumer LLM must always see. |
| `--source-root` | (recorded) | Override stored `source_root` (local mode only). |
| `--strict` | (off) | Exit code 2 if any retrieved passage has drifted or missing source. Default warn-mode prepends a `⚠ DRIFT WARNING` banner and proceeds. |
| `--strict-names` | (off) | Exit code 3 if a distinctive proper noun, acronym, or alphanumeric ID from a query does not appear in any retrieved passage. Default warn-mode prepends a `<!-- rlat-namecheck: missing X -->` block + ⚠ Name verification failed directive to the body and proceeds. Catches the name-aliasing distractor failure mode. |
| `--mode` | `augment` | Grounding mode for the consumer LLM. `augment` (default — passages are primary context blended with LLM training; bench 2 v4 single-shot: 76.5% accuracy / 3.9% hallucination on Microsoft Fabric docs), `constrain` (passages are the only source of truth, refuse on thin evidence — 66.7% / 2.0% / 91.7% distractor refusal, pick for compliance / audit work), or `knowledge` (passages supplement training, lighter gate; under `deep-search` ties augment at 0% hallucination). See [SKILLS.md §Grounding modes](SKILLS.md#grounding-modes). |

### Output format

The output starts with a grounding-mode directive (controlled by `--mode`), then one block per query:

```markdown
<!-- rlat-mode: augment -->
> **Grounding mode: augment.** Use the passages below as primary context …

<!-- rlat skill-context query='...' band=base mode=augment
     top1_score=0.612 top1_top2_gap=0.124 source_diversity=0.83 drift_fraction=0.00 -->
## Context for: "..."

> **source: `docs/foo.md:1234+512`** — score 0.812 `[verified]`
>
> Passage text...
```

Multiple `--query` flags emit blocks in submission order. Drifted source prepends a `⚠ DRIFT WARNING` banner; strict + drift aborts non-zero. Under `--mode augment` (default) or `--mode knowledge`, the dynamic body of any query whose `ConfidenceMetrics` fail the gate is replaced by a *no confident evidence* marker. Under `--mode constrain` the body always renders (refusal is the LLM's job). The directive header always ships regardless.

### Exit codes

- `0` — success (including warn-mode drift / name-check detection).
- `2` — `--strict` and one or more passages have drifted or missing source.
- `3` — `--strict-names` and a distinctive query token does not appear in any retrieved passage.

For format contracts (the harness suite enforces 10 of them — including name-check warn + strict), see [SKILL_INTEGRATION.md](../internal/SKILL_INTEGRATION.md).

