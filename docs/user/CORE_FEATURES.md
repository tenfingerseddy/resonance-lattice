# Core Use Cases

Resonance Lattice is a knowledge model: a single `.rlat` file containing the embedded representation of your corpus plus every coordinate needed to find, cite, and verify the underlying source. One artifact, three layers — the **field** routes queries, the **store** serves evidence, and your assistant composes the answer.

Seven use cases this enables. They share the same artifact and the same architecture; they differ in which surface of `rlat` they use most.

---

## 1. Ground an LLM with verified citations

**Trust the answer without checking the LLM's work.** Every passage `rlat` returns carries the source file, character offset, content hash, and a live drift status against the file on disk. Your assistant composes the response from passages that have been mechanically verified — not from what the model remembers.

### What you get

- **Hits with full provenance.** Every result is `(text, source_file, char_offset, char_length, content_hash, drift_status)`. The reader sees the bytes from your file, not a paraphrase. `text` is sliced live from the source — drift is detected because hashes are checked, not assumed.
- **Drop-the-drifted with one flag.** `rlat search --verified-only` filters out any hit whose content_hash no longer matches the source. Use it whenever the corpus is older than the underlying files and you'd rather get fewer-but-truthful hits than more-but-stale.
- **Honest confidence on every retrieval.** The `evidence()` RQL op returns `ConfidenceMetrics`: `top1_score` (cosine of the top-1 hit — "is retrieval working at all"), `top1_top2_gap` (margin between the best and second-best hit — informational, paraphrase clusters invert its meaning), `source_diversity` (how many distinct sources back the answer), `drift_fraction` (how stale the supporting evidence is), and `band_used`. Use these to gate whether a downstream LLM should answer or refuse.
- **Grounding modes for LLM-facing output.** `rlat skill-context --mode {augment|knowledge|constrain}` and `rlat search --format context --mode ...` stamp a directive at the top of the markdown telling the consumer LLM how to treat the passages: `augment` (default — primary context blended with LLM training; bench 2 v4 single-shot: 76.5% accuracy / 3.9% hallucination on Microsoft Fabric docs vs 56.9% / 19.6% LLM-only), `constrain` (only source of truth, refuse on thin evidence; 66.7% / 2.0% / 91.7% distractor refusal — pick for compliance / audit work), or `knowledge` (supplement training, lighter gate; under `rlat deep-search` ties augment at 0% hallucination). The mode header always ships; the body is gated by `ConfidenceMetrics` thresholds when applicable.
- **Skills that refuse on thin evidence.** `rlat skill-context --strict` exits non-zero when any hit drifted, so a skill `!command` block aborts before the assistant ever sees stale evidence. Pair with `--mode constrain` for compliance and audit work where wrong-but-confident is worse than refusal.
- **Name-aliasing protection.** `--strict-names` extracts distinctive proper nouns / acronyms / alphanumeric IDs from the question and verifies each appears in at least one retrieved passage. Default behaviour prepends a refusal directive when a token is missing; `--strict-names` aborts non-zero. Catches the failure mode where the question references a fake or adjacent product (e.g. `MaterializedViewExpress (MVE)`) but the corpus only describes a similarly-named real entity (`MaterializedLakeView (MLV)`) — score-based gating cannot tell these apart on a paraphrase-rich corpus, the name-check can.
- **Multi-hop research, two surfaces, one loop.** For cross-file synthesis questions, the **`deep-research` skill** at `.claude/skills/deep-research/SKILL.md` drives a plan → retrieve → refine → synthesize loop natively in your Claude Code session — no API key, your subscription covers the LLM hops. The same loop is also exposed as `rlat deep-search` for programmatic / agent / CI consumers (Anthropic API key required, ~$0.009-0.025/q). Bench (Microsoft Fabric, relaxed rubric): 92.2% accuracy / 0% hallucination — same numbers apply to both surfaces because it's the same loop.

### Run

```bash
rlat search my-project.rlat "how does the encoder cache work" --verified-only
rlat search my-project.rlat "auth flow" --format context        # markdown for an LLM prompt
rlat deep-search my-project.rlat "explain the auth flow with rationale"  # multi-hop loop, synthesised answer
```

```python
from resonance_lattice.store import archive, open_store
from resonance_lattice.rql import evidence

contents = archive.read("my-project.rlat")
store = open_store("my-project.rlat", contents, source_root="./")
report = evidence(contents, store, query_emb, top_k=10)

if report.confidence.drift_fraction > 0.2:
    print("warning: 20%+ of supporting evidence has drifted")
if report.confidence.top1_score < 0.30:
    print("warning: retrieval has no relevant match — consider refusing")
```

### Why it matters

Hallucinated citations are the failure mode that consequences hang on — legal briefs, medical advice, regulatory filings, audit reports. `rlat` makes them **structurally detectable**: every retrieved passage carries source path, character offset, and content hash, so the consumer can verify where evidence came from and whether it has drifted from the live source. The retrieval contract is hard; the consumer LLM still has to follow the directive, so pair `--mode constrain` with `--strict` and `--strict-names` for compliance workflows where wrong-but-confident is worse than no answer.

---

## 2. Bootstrap any assistant in seconds

**Stop re-explaining your project on every new chat.** `rlat` generates two extractive primers — one of the project itself, one of the work that's gone into it — and your assistant loads them at session start. Conversations don't begin cold and the assistant doesn't invent details to fill the gap.

### What you get

- **One command from clone to full context.** `rlat init-project` auto-detects `docs/`, `src/`, and top-level `*.md` / `*.rst` / `*.txt`, builds the knowledge model, generates a primer, and (optionally) writes a memory primer too. A new contributor — or your future self after weeks away — gets the assistant fully briefed in under a minute.
- **Two primers, no duplication.** A *code primer* (`rlat summary`) captures what the project IS — landscape, file structure, evidence per topical query. A *memory primer* (`rlat memory primer`) captures how the work has unfolded — settled facts, in-flight items, recurring themes. They're queried independently so the assistant gets denser context per token.
- **Targeted evidence sections.** `rlat summary --query "auth" --query "billing"` adds an Evidence block per query so a primer covers the topics you actually ask about, not just the corpus centroid.
- **Stays current — manually or live.** `rlat refresh` re-ingests local-mode knowledge models against the recorded source paths atomically. `rlat watch` runs the same refresh on every save (debounced, silent by default, auto-discovers `*.rlat` in cwd) so the archive stays fresh without you remembering to refresh after each edit. Primers regenerate from the refreshed archive.
- **Works with any assistant.** Claude Code, Cursor, command-line LLMs, agents in any IDE — anywhere a project-context file gets loaded.

### Run

```bash
rlat init-project                                # auto-detect + build + primer
rlat summary my-project.rlat -o .claude/resonance-context.md
rlat summary my-project.rlat \
  --query "encoder caching" --query "drift detection" \
  -o .claude/resonance-context.md
rlat memory primer ./memory/ -o .claude/memory-primer.md
```

### Why it matters

Every cold session is a tax. Two primers paid once and refreshed on demand collapse the cost of context-rebuild from per-conversation to per-corpus-change. The assistant arrives knowing what the project is and what's been decided, so the conversation starts where the last useful one left off.

---

## 3. Make skills knowledge-aware

**Anthropic skills are a static instruction bundle by default.** `rlat skill-context` makes them adaptive — each invocation injects markdown evidence retrieved against the user's actual question, drawn from a knowledge model the skill author chose.

### What you get

- **First-class Anthropic-skill primitive.** `rlat skill-context` is designed to be the body of a skill `!command` block ([anthropic skills docs — inject dynamic context](https://code.claude.com/docs/en/skills.md#inject-dynamic-context)). It returns markdown so the skill SKILL.md doesn't post-process anything.
- **Multi-query in one call.** `--query` is repeatable — pass a skill-author preset query AND the user's live query in one invocation. Token budget truncates later blocks first, so the preset survives if the budget shrinks.
- **Three grounding modes.** `--mode augment` (default) tells the LLM to treat passages as primary context blended with training — bench 2 v4 single-shot: 76.5% accuracy / 3.9% hallucination on Microsoft Fabric docs. `--mode constrain` tells it to answer ONLY from the passages and refuse otherwise — load-bearing for fact extraction (66.7% / 2.0% / 91.7% distractor refusal), compliance, regulatory, and audit skills. `--mode knowledge` treats passages as a supplement to training (single-shot 70.6% / 5.9%; ties augment under `deep-search` at 0% hallucination).
- **Drift-aware injection.** Every passage carries a drift indicator. `--strict` aborts non-zero when any retrieved passage has drifted, so a skill never silently serves stale knowledge.
- **Per-passage citation anchors.** Output is `## Passage N — source_file:char_offset` so the assistant can cite back into your source files using the exact coordinates `rlat` retrieved.
- **Confidence header.** Output prepends a `ConfidenceMetrics` block (top1 score, top1-top2 gap, source diversity, drift fraction, band) so the LLM downstream can refuse on thin evidence.
- **Works on existing knowledge models.** Any `.rlat` file already serves a skill — no per-skill encoder, no skill-specific build, no separate index.

### Run

```bash
# Inside a skill SKILL.md, in a !command block:
rlat skill-context fabric.rlat \
  --query "Fabric workspace authentication and service principal patterns" \
  --query "$USER_QUERY" \
  --top-k 8 --token-budget 2000 --strict
```

### Why it matters

A skill about Fabric lakehouses without `rlat` injects the same instructions whether the question is about workspace setup, ingestion patterns, or Delta merge semantics. With `rlat skill-context`, each question retrieves different evidence from the same knowledge model — workspace auth for the first, PySpark templates for the second, upsert semantics for the third. The skill's static instructions frame the workflow; the knowledge model supplies the specifics.

→ Full walkthrough: [SKILLS.md](SKILLS.md).

---

## 4. Give an assistant durable memory

**Conversations that accumulate, not conversations that vanish.** `rlat memory` keeps your interaction history as a queryable three-tier knowledge model with retention, consolidation, and primer generation.

### What you get

- **Three tiers by recency.** *Working* (session-local, 1d half-life), *episodic* (per-session, 14d), *semantic* (consolidated, no decay). A question about today's work pulls from working; a question about long-running themes pulls from semantic.
- **Tier-weighted recall.** `score = cosine × tier_weight × salience`. Defaults favour fresh context (working 0.5, episodic 0.3, semantic 0.2) so today's facts dominate older ones, but salience overrides for facts you've explicitly marked load-bearing.
- **Mechanical write path.** `rlat memory add` stores text + tier + salience exactly as given. No LLM is summarising in the background, no extractor is classifying — what you wrote is what gets remembered. The "photocopy of a photocopy" failure mode of LLM-derived memory tools cannot accumulate.
- **Episodic-to-semantic consolidation.** `rlat memory consolidate` promotes facts that recurred ≥3 times in episodic to semantic. Idempotent on stable input.
- **Memory primer.** `rlat memory primer` synthesises across all tiers into the same markdown shape the corpus primer uses — drop-in for `.claude/memory-primer.md` so the assistant loads it at session start.
- **Garbage collection.** `rlat memory gc --tier working` drops expired entries; the rest of the store stays untouched.

### Run

```bash
rlat memory add "user prefers TypeScript over Python" --tier semantic --salience 1.5
rlat memory add "shipping v2.0 launch on 2026-06-08" --tier episodic
rlat memory recall "what tech stack does the user prefer"
rlat memory consolidate                          # episodic → semantic for stable items
rlat memory primer -o .claude/memory-primer.md
rlat memory gc --tier working
```

### Why it matters

The assistant remembers across sessions without re-reading transcripts. Because the write path is mechanical, the searchable index is rebuildable from your raw entries at any time — derivation drift cannot accumulate the way it does in summary-of-summary chains.

→ Storage shape + retention math: [docs/internal/MEMORY.md](../internal/MEMORY.md).

---

## 5. Keep your knowledge portable, private, local-first

**Your knowledge stays yours.** A knowledge model is a single `.rlat` file, your source files stay where they are, and nothing leaves your machine unless you explicitly send it.

### What you get

- **Single-file portability.** One `.rlat` packages embeddings, source coordinates, drift hashes, optional ANN index, and (in bundled mode) the source text itself. Copy it, version it, share it — it works anywhere Python 3.12+ runs.
- **Three storage modes for three trust models.**
  - `bundled` — source text is zstd-framed inside the archive. Fully self-contained; ship it without the source files.
  - `local` (default) — source text stays on disk; the archive resolves it via `--source-root`. No duplication; ideal when source is a git repo you already version. `rlat watch` keeps the archive live as you edit (debounced refresh on every save, zero-arg auto-discovery).
  - `remote` — source text lives behind an HTTP base URL with SHA-pinned manifests. Read-only `rlat freshness` confirms upstream hasn't changed; `rlat sync` applies an incremental delta when it has (Audit 07 — same `store/incremental.py` pipeline as `rlat refresh`).
- **Fully offline after first install.** `rlat install-encoder` downloads gte-modernbert-base once. After that, `build` / `search` / `optimise` need zero network access. Pair with a local LLM (Ollama, llama.cpp) for an end-to-end private stack.
- **Field-only sharing.** Ship a `.rlat` containing `bands/base.npz` without `source/` — recipients see the semantic structure (and can search against it) without reading your raw text. Useful for publishing the structure of a private corpus.
- **No telemetry, no accounts, no cloud.** Everything is a local Python package.

### Run

```bash
rlat install-encoder                                       # one-time
rlat build ./docs ./src -o my-project.rlat                 # local mode (default)
rlat build ./docs -o handout.rlat --store-mode bundled     # self-contained
```

### Why it matters

The same artifact serves a hobbyist on a laptop, a consultant carrying client knowledge between engagements, a research team archiving a paper corpus, and a regulated-industry team where data egress is policy-restricted. The `.rlat` format is a single-file unit of distribution — versionable, diffable, hand-offable, archivable.

→ Decision guide: [STORAGE_MODES.md](STORAGE_MODES.md).

---

## 6. Inspect what your knowledge actually contains

**See the structure of what you know — and what you don't.** Most retrieval tools are opaque: documents in, results out, no way to understand the model in between. `rlat` exposes the semantic structure of the corpus as a first-class interface.

### What you get

- **`rlat profile`** — Per-source-file passage counts, drift counts grouped by `verified` / `drifted` / `missing`, encoder identity, band info. "How healthy is this knowledge model?" in one command.
- **`rlat compare`** — Cross-knowledge-model centroid alignment + asymmetric coverage. Returns `centroid_cosine` (single thematic-alignment number), `coverage_a_in_b` and `coverage_b_in_a` (asymmetric), plus a symmetric overlap score. Always uses the base band per the cross-model rule.
- **`locate(contents, query_emb)`** — Find the closest passage to a point in semantic space and report how concentrated the surrounding neighbourhood is. Answers "is the corpus dense in this topical region or thin?"
- **`neighbors(contents, passage_idx)`** — k-nearest passages to a given passage. Useful for "what else is the corpus saying about the thing this passage discusses?"
- **`near_duplicates(contents, threshold=0.92)`** — Find passages so similar they're effectively duplicates. Surfaces accidental redundancy in a corpus.
- **`contradictions(contents)`** — Heuristic: high cosine similarity + low Jaccard overlap pairs. Flags potential contradictions for editorial review. Experimental.
- **`corpus_diff(old_contents, new_contents)`** — Per-passage added / removed / unchanged across two snapshots of the same corpus. Queryable change tracking.
- **`drift(contents, store)`** — Lists every passage whose live source content_hash no longer matches the recorded one. Drives maintenance workflows.
- **`audit(contents, store, claim_emb)`** — Given a claim embedding, retrieves supporting + contradicting passages with confidence metrics. Experimental fact-check primitive.

### Run

```bash
rlat profile my-project.rlat                       # text summary
rlat profile my-project.rlat --format json         # structured
rlat compare a.rlat b.rlat --format json --sample 1024
```

```python
from resonance_lattice.store import archive
from resonance_lattice.rql import locate, neighbors, near_duplicates, contradictions, corpus_diff

contents = archive.read("my-project.rlat")
result = locate(contents, query_emb)
print(f"closest passage: {result.passage_idx}, neighbourhood density: {result.local_density:.3f}")

dupes = near_duplicates(contents, threshold=0.92)
print(f"{len(dupes.clusters)} clusters of near-duplicate passages")
```

### Why it matters

Inspecting the corpus before trusting it is the discipline that keeps retrieval systems honest. `rlat`'s inspection ops are derived mechanically from the embeddings themselves — cosine, content_hash equality, connected-component clustering — not from heuristics or LLM-judged labels.

→ Per-op rationale: [docs/internal/RQL.md](../internal/RQL.md).

---

## 7. Compose and optimise without rebuilding from scratch

**Reshape what you have, don't rebuild from raw text every time.** `rlat` ships two ways to change retrieval behaviour without re-encoding the corpus: multi-knowledge-model composition at query time, and an opt-in per-corpus optimisation pass.

### What you get

- **`compose(contents_a, contents_b, ...)`** — Federated read-only view across multiple knowledge models. Query routes against all of them, results unified. Backbone-revision mismatch warns. Useful when you want to query across corpora without merging them on disk.
- **`merge(contents_a, contents_b, ...)`** — Write-out merge that semantically dedupes. Backbone-revision mismatch raises (write-out can't have mixed embedding distributions). Useful when you want a single permanent artefact spanning multiple sources.
- **`unique(contents_a, contents_b)` / `intersect(contents_a, contents_b)`** — Set-algebra over knowledge models. Returns the passages from A that aren't in B (or are in both), based on cosine threshold. Cross-KM, base-band only.
- **`rlat optimise <km.rlat>`** — Trains a corpus-specific 768→512 MRL projection in-place. Adds an `optimised` band that lifts retrieval for natural-language queries against the corpus (verified +3.2 pt R@5 on Microsoft Fabric docs). Opt-in: one LLM bill (~$8–15) and ~50 min on a T4. The base band stays in the archive — cross-KM `compare` always uses base.
- **`--kind intent` tag** — Mark a knowledge model as carrying intent vectors rather than corpus content. v2.0 ships the tag (so the registry can disambiguate); intent-specific operators (boost, suppress, cascade at query time) ship in v2.1.

### Run

```bash
rlat optimise my-project.rlat --corpus-description "Microsoft Fabric documentation"
rlat optimise my-project.rlat --estimate                     # cost dry-run before paying
```

```python
from resonance_lattice.store import archive
from resonance_lattice.rql import compose, merge, unique, intersect

a = archive.read("docs.rlat")
b = archive.read("src.rlat")

# Federated view — query both corpora at once, no write
view = compose(a, b)

# Permanent merge with dedupe — single artefact going forward
merged = merge(a, b, dedupe_threshold=0.95)
archive.write("everything.rlat", merged)

# What's in docs that isn't in src?
docs_only = unique(a, b, threshold=0.85)
```

### Why it matters

A corpus is rarely just one source. `compose` lets you query across knowledge models without manually maintaining a meta-archive; `merge` lets you commit to a single one when that's what your workflow needs. `optimise` lets you trade an LLM bill for measurable retrieval lift — verified, optional, idempotent. Together: composition for the federated case, optimisation for the single-corpus case.

→ Optimise pipeline + cost table: [OPTIMISE.md](OPTIMISE.md). RQL ops: [docs/internal/RQL.md](../internal/RQL.md).

---

## What `rlat` does not do

Honest about scope so you can plan around it:

- **No reader / synthesis.** `rlat` returns passages with verified citations; your assistant composes the response on top. There is no `rlat ask` command.
- **No retrieval-time knobs.** No reranking, no lexical sidecar, no query-prefix tuning, no auto-mode router. Single recipe — `gte-modernbert-base` 768d CLS+L2 dense cosine, optional MRL `optimised` band per corpus. Empirically validated to match or beat tuned alternatives ([BENCHMARK_GATE.md](../internal/BENCHMARK_GATE.md)).
- **No HTTP server.** CLI is the primary surface. An MCP bridge for AI assistants is planned for a future version.
- **No intent operators.** The `--kind intent` flag tags a knowledge model; query-time intent operators (boost, suppress, cascade) ship in v2.1.
- **No algebraically-exact `forget`.** To remove specific knowledge, rebuild with the source subset you want and ship the new `.rlat` — atomic and auditable, just at the corpus granularity rather than the per-passage granularity.

---

## Where to go next

- 15-minute first build: [GETTING_STARTED.md](GETTING_STARTED.md)
- Full CLI reference: [CLI.md](CLI.md)
- Storage-mode decision guide: [STORAGE_MODES.md](STORAGE_MODES.md)
- When to run `rlat optimise`: [OPTIMISE.md](OPTIMISE.md)
- Skill integration walkthrough: [SKILLS.md](SKILLS.md)
- Per-op RQL rationale: [docs/internal/RQL.md](../internal/RQL.md)
