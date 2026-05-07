# FAQ

## What is rlat?

A retrieval layer for AI assistants. You point it at a corpus (docs, code, notes, anything text), it builds a `.rlat` knowledge model, and you query that model — the assistant gets back ranked passages with full source citations + drift status.

Three layers under the hood: **field** (the encoder + dense cosine retrieval), **store** (the on-disk knowledge-model archive + verified-retrieval contract), and a deliberately empty reader layer (you compose synthesis on top).

## How is it different from a vanilla embedding store?

**Lossless verified retrieval.** Every passage in every `.rlat` carries `(source_file, char_offset, char_length, content_hash, drift_status)`. Most retrievers store opaque embeddings and lose source attribution. We don't. Citations are free; drift detection is free; fact-checking workflows compose on top without extra bookkeeping.

## What encoder do you use?

`Alibaba-NLP/gte-modernbert-base` at 768d, CLS-pooled, L2-normalised, max sequence length 8192 tokens. Pinned to commit `e7f32e3c00f91d699e8c43b53106206bcc72bb22` so a fresh box gets the same embedding distribution that produced the locked BEIR-5 floor.

Validated on the v2.0 stack against BGE-large-v1.5 (we win by +0.026 nDCG@10) and E5-large-v2 (+0.081 nDCG@10) — apples-to-apples, same chunker / ANN / scoring. Plus Qwen3-Embedding-8B at +0.014 nDCG@10 mean (run on its own stack with last-token pooling). See [BENCHMARKS.md](BENCHMARKS.md#retrieval-quality-beir-5-floor--encoder-comparison) for the comparison table and [BENCHMARK_GATE.md](../internal/BENCHMARK_GATE.md) for the locked floor.

## Why no rerankers / lexical sidecar / auto-router?

Measured. Cross-encoder rerankers regressed gte-mb-base on 4 of 5 BEIR corpora (training-distribution mismatch with strong dense top-k). Lexical V1 band failed parity on 4 of 5 corpora. Auto-mode routers add latency without quality gain. Single recipe wins on quality + simplicity + reproducibility.

The `rlat optimise` MRL optimised (opt-in) is the one knob that lifted quality without regressions — see "When should I run optimise?"

## When should I run `rlat optimise`?

When you want a corpus-tuned projection that lifts in-corpus retrieval quality. Costs ~$8-15 in LLM calls (Claude Sonnet 4.6) per 40K-passage corpus + ~50 min on a T4 GPU. Verified +3.2 pt R@5 lift on Microsoft Fabric documentation; regresses on register-shifted benchmarks like BEIR nfcorpus. Run `rlat optimise <km>.rlat --estimate` first to get a real cost number for your corpus before committing.

**Skip optimise when**: corpus < 1000 passages (training won't converge), corpus is heterogeneous (one projection averages topics), queries are short keyword-style ("auth", "deploy" — out of distribution from the synth queries), base band already saturated on your benchmark, or you don't want to spend $20.

The lift is **corpus-specific, not user-specific**: any natural-language querier of the corpus benefits, but the projection doesn't transfer to OTHER corpora. See [OPTIMISE.md](OPTIMISE.md).

## Bundled vs. local vs. remote storage mode?

- **Local** (default): source files stay on disk; `.rlat` carries embeddings + coordinates. Use when you're indexing a working repo you'll edit.
- **Bundled**: source files zstd-compressed inside the `.rlat`. Use when shipping a self-contained artefact (HF Hub, archive, offline distribution).
- **Remote**: source resolved over HTTP(S) against a SHA-pinned manifest baked into the `.rlat` at build time (`--store-mode remote --remote-url-base <prefix>`). Use when source belongs at canonical URLs (e.g. a tagged GitHub release). First query downloads to a per-knowledge-model cache; SHA-pin verification on every read. `rlat freshness` is the read-only drift check; `rlat sync` is incremental reconciliation — see "What if my source repo (remote) changes?" below. Both land on the same `store/incremental.py` delta-apply pipeline as local-mode `rlat refresh`.

## How do I switch storage modes without rebuilding?

`rlat convert <km.rlat> --to {bundled|local|remote}` (Audit 08). Reshapes the storage mode in place without re-running the encoder — bands, registry, ANN, and the optimised W projection are storage-mode-independent and flow through unchanged.

```bash
# Hand off a working knowledge model as self-contained
rlat convert my.rlat --to bundled

# Open a downloaded handout for editing
rlat convert handout.rlat --to local --source-root ./extracted/

# Publish a private corpus to HTTP
rlat convert internal.rlat --to remote --remote-url-base https://docs.example.com/v1/
```

Conversion validates every passage's `content_hash` against live bytes before write. If any have drifted, conversion aborts (rc=2) with the drift report; run `rlat refresh` or `rlat sync` first to reconcile, then retry. See [CLI.md §rlat convert](CLI.md#rlat-convert-shipped-audit-08) for the full pairwise-transition table.

## How do I run `rlat optimise` on a remote-served corpus?

Materialise it locally first, then optimise normally. The optimise pipeline reads passage texts via the Store; running it directly against a remote-mode archive streams every passage over HTTP one-at-a-time during a $8-15 LLM run, so a network blip wastes the bill. Two clean commands:

```bash
rlat convert upstream.rlat --to local --source-root ./local-mirror/ -o working.rlat
rlat optimise working.rlat --corpus-description "Microsoft Fabric documentation"
# Optionally back to remote:
rlat convert working.rlat --to remote --remote-url-base https://upstream.example.com/v1/
```

## How do I tell the LLM how to use the retrieved passages?

`rlat skill-context` and `rlat search --format context` both take a `--mode` flag that stamps a directive at the top of the markdown output. Three modes:

- **`augment`** (default) — passages are primary context blended with the LLM's training. Body is suppressed on weak retrieval (`top1_score < 0.30` OR `drift_fraction > 0.30`) so the LLM falls back to training instead of grounding on noise. **Default — bench 2 v4 (Microsoft Fabric, single-shot) measured 76.5% answerable accuracy / 3.9% hallucination, vs 56.9% / 19.6% for the LLM alone.** The right shape for broad documentation corpora where the LLM has solid prior knowledge.
- **`constrain`** — passages are the ONLY source of truth. No gate; the LLM is told to refuse when evidence is thin. Bench 2 v4 (single-shot): 66.7% accuracy / 2.0% hallucination / 91.7% distractor refusal — trades 10 pp answerable accuracy for halving the hallucination rate and the highest distractor-refusal in the suite. Pair with `--strict` so drifted source aborts the skill load. Pick for fact extraction, compliance, regulatory, and audit work where wrong-but-confident is worse than no answer.
- **`knowledge`** — passages supplement training. Lighter gate (`top1_score < 0.15` only — drift is allowed) because the directive already tells the LLM to lean on training. Bench 2 v4 (single-shot): 70.6% accuracy / 5.9% hallucination. Choose this for partial-coverage corpora where the LLM already knows the surrounding domain reasonably well; under `rlat deep-search` it ties augment on accuracy at 0% hallucination — the launch recommendation when the loop runs.

```bash
rlat skill-context fabric.rlat --query "$user_query" --mode constrain --strict
rlat search compliance.rlat "what's our SOX retention policy?" --format context --mode constrain
```

The directive header always ships, even when the gate suppresses the body — the LLM always sees the instruction. See [SKILLS.md §Grounding modes](SKILLS.md#grounding-modes).

## When should I use `rlat deep-search` vs `rlat search`?

> **Heads up**: `rlat deep-search` requires an Anthropic API key. If you're in
> a Claude Code session and don't have a key, the `deep-research` skill at
> `.claude/skills/deep-research/SKILL.md` runs the same loop natively in your
> conversation with no extra cost — same prompts, same hop budget, same name-
> verification check, same output shape. Use the skill in nearly every
> interactive scenario. The CLI verb is the right pick when you need a
> programmatic / batch / non-Claude-Code surface. See
> [API_KEYS.md](API_KEYS.md).

`rlat search` is single-shot — one query, top-k passages, return. **No LLM at all** (just the deterministic encoder + cosine + ANN), fully local, cheap (~$0.005 with a consumer LLM downstream), instant, fine for one-hop fact lookups.

`rlat deep-search` (CLI) or the `deep-research` skill runs an internal multi-hop loop — plan, retrieve, refine, retrieve again, synthesize — and returns a final answer plus the evidence union. Pricier (~$0.010/q on the API surface; free in-session via the skill) and slower (~10s per question), but on the Microsoft Fabric 5-lane bench (relaxed rubric):

| | Accuracy | Hallucination | Cost / q |
|---|---:|---:|---:|
| `rlat search --format context` (single-shot, `--mode augment`) | 76.5% | 3.9% | $0.004 |
| `rlat deep-search` | **92.2%** | **2.0%** | $0.010 |

Pick `deep-search` when:

- the question spans multiple files (the loop can issue follow-up queries),
- correctness matters more than latency / spend (compliance, research, audit),
- the user wants a synthesised answer with citations rather than a list of passages.

Stick with `search` when:
- the question is a single-hop fact lookup,
- you're driving the synthesis yourself in a downstream LLM,
- latency / cost dominates (chatbots, real-time UX).

See [CLI.md §rlat deep-search](CLI.md#rlat-deep-search) for the full surface.

## What's `--strict-names`?

A second safety check (independent of grounding mode) on `rlat skill-context`, `rlat search --format context`, and `rlat deep-search`. Extracts distinctive proper nouns / acronyms / alphanumeric IDs from the question, then verifies each appears verbatim in at least one retrieved passage. When a distinctive token is missing from all passages:

- **default**: prepend a refusal directive to the body (the LLM sees a "the corpus may describe a different entity" warning).
- **`--strict-names`**: exit non-zero (rc=3). Skill loaders / scripts gate on this.

This catches the name-aliasing failure mode score-based gating cannot — e.g. a question about `MaterializedViewExpress (MVE)` when the corpus only describes the adjacent `MaterializedLakeView (MLV)`. Bench-2 v3 distractor analysis showed score and gap distributions overlap completely between answerable and distractor questions on a paraphrase-rich documentation corpus, so any score-floor gate is statistically falsified. Name-check is the right mitigation.

```bash
rlat search fabric.rlat "How does MVE refresh work?" --format context --strict-names
# rc=3 if MVE doesn't appear in any retrieved passage
```

## How do I cite a search result?

Citations are built into every result. CLI:

```bash
rlat search my.rlat "query" --format json | jq '.[] | {file:.source_file, offset:.char_offset, drift:.drift_status, score:.score}'
```

Python:

```python
from resonance_lattice.rql import evidence
report = evidence(contents, store, query_emb)
for hit in report.hits:
    c = hit.citation
    print(f"{c.source_file}:{c.char_offset}-{c.char_offset + c.char_length}  ({hit.score:.3f})")
```

The `evidence` op also returns `ConfidenceMetrics` (top1-top2 gap, source diversity, drift fraction) — calibration data the caller can act on (e.g. abstain when top1-top2 gap < 0.05).

## What if my source files change?

Three options:

**Live (recommended for active work):** `rlat watch` — runs the same incremental delta-apply as `refresh` but on a debounced timer triggered by filesystem events. Default UX is silent. Zero-arg invocation auto-discovers `*.rlat` in cwd and watches every recorded source root. Per-archive `threading.Lock` serialises refreshes so concurrent saves can't race the atomic-write path. Optimised bands re-project for free. Local mode only — bundled is immutable, remote routes to `rlat sync`. Requires the `[watch]` extra: `pip install rlat[watch]`.

**Manual:** `rlat refresh my.rlat` — applies an incremental delta against the existing archive: unchanged passages keep their band rows untouched, modified/added passages are re-encoded, removed passages are dropped. Atomic in-place write. Local mode only.

**CI / pre-commit:** `rlat watch --once` — synchronous one-shot reconciliation. Walks every preflighted archive, runs `bucketise` + `apply_delta` against current disk state, exits. No observer, no event wait — files are typically already changed before the command runs (formatter pass, applied patch, `git checkout`).

If your knowledge model has an optimised band, refresh **re-projects it from the new base for free** — no LLM call, no GPU. The refresh path runs `optimised = (new_base @ W.T)` row-wise L2-normalised; W is preserved byte-identically. Pass `--discard-optimised` only if you specifically want a base-only archive afterwards (rare).

`rlat refresh --dry-run` walks sources and reports the delta counts (unchanged/updated/added/removed) without writing — useful as a "would refresh do anything?" check.

`rlat profile` reports drift status without rebuilding — use it to decide whether you need to refresh.

## What if my source repo (remote) changes?

Run `rlat sync my.rlat` — same incremental delta-apply as `refresh`, but the upstream-state oracle is your remote source instead of the local filesystem.

Two modes:

- **Catalog mode** (`--upstream-manifest <url>`) — upstream serves a stable JSON manifest endpoint listing `{source_file: {url, sha256}}` for the current corpus. Sync diffs upstream against the archive's pinned manifest in O(1) network calls; detects added + modified + removed paths.
- **Poll mode** (default) — re-fetches every URL in the existing manifest, hashes the bytes, diffs against the pinned `sha256`. Detects `modified` only — cannot enumerate NEW files because there's no upstream signal for added paths. URLs that error (timeout/5xx/connection refused) go into an `unavailable` bucket; sync **aborts** by default rather than treating an outage as a deletion. Pass `--treat-unreachable-as-removed` only when you've confirmed upstream is healthy and the unreachable paths are genuinely gone. **Catalog mode is the safe mode for removals** — only an authoritative upstream catalog can tell you a file is gone vs. just unreachable.

`rlat sync --dry-run` lists the changed paths with `+/~/-` prefixes without fetching or writing — a CI-friendly "would sync change anything?" check.

After sync, the in-archive manifest's pinned `sha256` and `pinned_ref` advance atomically with the band write. There's no window where the manifest is current but the bands are stale.

`rlat freshness my.rlat` is the read-only check — walks the manifest, downloads each entry, hashes, reports per-entry status (`verified`/`drifted`/`missing`). No mutation; useful as a CI gate.

## What's "memory"?

Three-tier append-only memory: **working** (1d half-life, 200-entry cap), **episodic** (14d, 2K), **semantic** (∞, 20K). Recall fuses tiers via `cosine × tier_weight × salience`. `rlat memory consolidate` promotes facts that recurred ≥3 times in episodic to semantic. `rlat memory primer` synthesises a markdown context block for the assistant to load at session start.

Independent from knowledge models — memory is per-user, knowledge models are per-corpus.

## How big can a knowledge model get?

Tested up to ~50K passages comfortably. The (N, N) cosine matrix used by `near_duplicates` and `merge` dedupe scales O(N²) — at 50K rows that's ~10 GB float32, the practical ceiling for those operations. Plain `rlat search` scales further (FAISS HNSW) but cross-KM ops above ~50K need pre-sharding.

Single `.rlat` files in the wild range from ~10 MB (a small docset) to ~2 GB (a large code repo + bundled source).

## I don't have a GPU. Can I still build a big corpus?

Yes — and the `rlat-build-on-kaggle` Claude skill at `.claude/skills/rlat-build-on-kaggle/SKILL.md` walks you through using Kaggle's free T4 GPU. T4 encodes `gte-modernbert-base` ~10-20× faster than ONNX-CPU, so anything over ~10K passages is meaningfully faster on Kaggle than locally. The skill covers Kaggle account setup, CLI install + auth, the kernel script (sparse-clones source from GitHub or mounts a Kaggle dataset), polling, and pulling the `.rlat` back. Free tier gives you 30 GPU-hours/week — enough to rebuild a ~500K-passage corpus several times.

Skip Kaggle if your corpus is small (<2K passages — the round-trip overhead exceeds the encode time on CPU) or if you already have a CUDA GPU locally.

## Is the API key required?

Only for two opt-in CLI commands: `rlat optimise` and `rlat deep-search`. The rest of the surface (build, search, profile, compare, summary, skill-context, memory, refresh, sync, freshness, convert, init-project) needs no LLM access — `rlat` is offline and LLM-free for the standard retrieval surface.

If you're in a Claude Code session, the `deep-research` skill at `.claude/skills/deep-research/SKILL.md` runs the same multi-hop loop as `rlat deep-search` natively in your conversation with no API key — your Claude subscription covers the LLM hops. See [API_KEYS.md](API_KEYS.md) for when each surface is the right pick.

API key discovery: `RLAT_LLM_API_KEY_ENV` (indirection) → `CLAUDE_API` → `ANTHROPIC_API_KEY` → Kaggle Secrets vault.

## Does it ship to PyPI?

Yes — `pip install rlat`. The base install is ~250 MB; full encoder + build dependencies (`pip install rlat[build]`) adds torch + transformers (~2 GB). Optimise extras add the Anthropic SDK.

For most consumers: `pip install rlat[build]` then `rlat install-encoder` once.

## Where do I report bugs / feature requests?

GitHub: [`tenfingerseddy/resonance-lattice`](https://github.com/tenfingerseddy/resonance-lattice). Issues + discussions enabled.

## What's coming after v2.0?

- MCP bridge for AI assistants
- Intent operators (boost, suppress, cascade)
- HTTP server / SDK surface (currently CLI-only)
- Per-passage origin tracking in merge output
- ANN attach point on `ComposedKnowledgeModel` for high-QPS federated search

These are roadmap-stable but not committed dates. v2.0 is the foundation; the missing pieces compose on top of it.
