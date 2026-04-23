# Resonance Lattice

**A portable semantic model for knowledge.**

Resonance Lattice turns docs, code, notebooks, and any other files into a `.rlat` knowledge model — a single-file semantic model of your corpus that you can search, inspect, compose, and share like any other artifact. One file. Local by default. Works across any LLM.

```bash
pip install resonance-lattice
rlat build ./docs ./src -o project.rlat
rlat search project.rlat "how does auth work?" --format text
rlat ask    project.rlat "how does auth work?" --reader llm
```

See [docs/CORE_FEATURES.md](docs/CORE_FEATURES.md) for the seven use cases the knowledge model unlocks — grounding, primers, skills, memory, portability, inspection, and retrieval shaping.

---

## Why a semantic model

Data professionals already know the pattern. In BI, a semantic model sits between raw tables and the reports that query them — it names entities, defines measures, and makes the underlying data queryable without every report having to stitch joins itself. You build it once. Everything downstream gets the same honest answer.

Resonance Lattice does the same thing, but for unstructured knowledge: the docs, notebooks, meeting notes, PDFs, spec sheets, and source code that sit between you and the answer you actually need. `rlat build` reads those files and writes a **knowledge model** — a compact semantic structure that any LLM, any tool, any workflow can query consistently. The model is the artifact. Retrieval is one thing the model does, not the whole product.

This is why we call it a *semantic model for knowledge* rather than a retrieval library or a RAG framework. The framing matters because what you get is an inspectable, versionable, mathematically-structured object — not a pipeline.

---

## Three-layer retrieval

Under the hood, the knowledge model is a three-layer object. Each layer is independently auditable so you can stop at whichever one fits the workflow.

| Layer | Question it answers | Command |
|-------|---------------------|---------|
| **1. Field** | *Which region of the corpus is relevant?* — fixed-size semantic router over your files | `rlat search` |
| **2. Retrieval** | *What bytes do we return?* — adaptive expansion + optional lexical (ripgrep) hybrid pass | `rlat search --expand natural --hybrid on` |
| **3. Reader** | *What answer does the user see?* — grounded synthesis with verified citations | `rlat ask --reader llm` |

`rlat ask --reader context` returns the exact evidence pack a reader would see — cheap, deterministic, LLM-free. Citations link back to source files with line numbers and verification markers (✓/○) so every claim can be checked against the bytes it came from.

The reader supports an on-device OpenVINO backend (Intel Arc iGPU / CPU / NPU) and Anthropic / OpenAI-compatible HTTP APIs. Pick via `--reader-backend` or set a project default in `.rlat.toml`.

---

## How it differs from RAG

Standard RAG is a pipeline: embed chunks, store vectors, retrieve top-k, stuff into a prompt. It solves a narrow problem and leaves you with plumbing — and usually a service dependency.

Resonance Lattice ships a different primitive. Because the knowledge model is a real object with algebraic structure, you get operations that pipelines don't offer:

- **Diff two versions** of a corpus and query the delta — not "what files changed" but "what *knowledge* changed".
- **Forget a subset** cleanly, with a mathematical guarantee the removed source no longer contributes to the field.
- **Merge** per-domain knowledge models into a unified view — order-independent, reproducible.
- **Project** one model through the lens of another (*"show me code, but only through the compliance lens"*).
- **Contradict** — find where two models disagree.
- **X-ray** the corpus itself (signal quality, saturation, coverage gaps) before you even query.

None of this requires running an LLM. The knowledge model is the analytical artifact; the LLM reader is optional on top.

| If you use | What it is good at | What Resonance Lattice adds |
|------------|--------------------|-----------------------------|
| **grep** | Exact text match | Semantic retrieval, inspection, portable artifacts |
| **Standard RAG / vector DB** | Hosted or index-backed retrieval | A portable `.rlat` file, local control, algebraic composition |
| **LLM direct** | Reasoning and generation | Grounded evidence with verified citations |
| **Note vaults / wikis** | Human-authored organization | Automatic semantic modeling over the files you already have |

---

## Mathematical (and analytical, by consequence)

The field has algebraic structure. That's not a marketing line — it's what makes the product different.

- **Adding a source** accumulates onto the field; removing it subtracts cleanly. The same inputs produce the same field every time. No temperature, no sampling, no drift.
- **Merging** is commutative and associative. `merge(merge(A, B), C) == merge(A, merge(B, C))`. You can compose by team, by domain, by time period — the result is stable.
- **Diffing** is a signed semantic delta. Query the diff and get what changed in *meaning*, not what changed in text.
- **Forget** is an exact rank-1 subtraction. The removed source's phase vector no longer contributes, and the operation returns a certificate quantifying any residual cross-talk.

Because the field is a well-defined mathematical object, inspection is cheap and meaningful:

| Command | What it tells you |
|---------|-------------------|
| `rlat xray corpus.rlat` | Corpus-level health: signal quality, saturation, diagnostic flags |
| `rlat locate corpus.rlat "query"` | Where a query sits in the knowledge landscape, and what the field does not cover |
| `rlat probe corpus.rlat <recipe>` | Pre-built insight recipes: novelty, saturation, coverage gaps, contradictions |
| `rlat profile corpus.rlat` | Semantic shape of the corpus: per-band energy, rank, source distribution |

The analytical surface is a consequence of the mathematical foundation. You can answer "what does my corpus *know*?" as cleanly as you can answer "find me a passage about X" — with the same object, the same commands, no extra infrastructure. See [docs/SEMANTIC_MODEL.md](docs/SEMANTIC_MODEL.md) for the fuller story.

---

## Storage modes

One lossless-store abstraction, three backends — pick the shape that fits the deployment. See [docs/STORAGE_MODES.md](docs/STORAGE_MODES.md) for details.

| Mode | Where the source files live | Best for |
|------|------------------------------|----------|
| **`local`** (default) | On disk, resolved via `--source-root` at query time | Developing against a working copy; large corpora where the `.rlat` should stay thin |
| **`bundled`** | Packed inside the `.rlat` as zstd frames | Self-contained artifacts — HF Hub demos, CI, offline distribution |
| **`remote`** | Public HTTP origin, SHA-pinned against a commit, cached locally on first fetch | Pointing at an upstream repo you don't own — `rlat freshness` and `rlat sync` manage upgrades lockfile-style |

All three serve the same retrieval pipeline. Re-chunking, drift detection, window expansion, and format dispatch work identically across modes.

Legacy `embedded` mode (pre-chunked SQLite) is deprecated and scheduled for removal in v2.0.0; `bundled` is the canonical self-contained replacement.

---

## Headline benchmarks

Numbers are pointers — fit to your own data matters more than any leaderboard. Calibration rules, methodology, and caveats live in [docs/HONEST_CLAIMS.md](docs/HONEST_CLAIMS.md).

- **Microsoft Fabric docs** (24,635-chunk corpus, 100 evaluation questions) — full `rlat` pipeline: **Recall@5 1.00**, **MRR 0.93**, **0% failed retrieval**.
- **Token efficiency** (2,266-file codebase) — `rlat search` returned **24.6× fewer tokens** than a `grep + read top 5 files` workflow while keeping ranked passages and source attribution intact.
- **Grounding quality** (LLM hallucination eval) — feeding `rlat` context reduced hallucinations from **78% → 16%** and raised fact recall from **0.27 → 0.91**.
- **BEIR-5 best-mode avg (three encoders measured, 2026-04-22)**:
  - `BGE-large-en-v1.5` (default, portable) — **0.445 nDCG@10**
  - `E5-large-v2` (opt-in, portable) — **0.455 nDCG@10**
  - `Qwen3-Embedding-8B` (opt-in, needs 16 GB GPU) — **0.500 nDCG@10** (field_only; stack probe deferred)

  Portable tiers sit with 2024 mid-tier open dense retrievers (Cohere-embed-v3, jina-v3, mE5-large-instruct). Qwen3-8B is frontier-adjacent — ~1 pt below `text-embedding-3-large` on this harness.

> **Pick by workload.** None of the three encoders wins everywhere. BGE is the rational middle; E5 wins on counter-argument / debate retrieval (ArguAna); Qwen3-8B wins in raw quality when you have the GPU. [docs/ENCODER_CHOICE.md](docs/ENCODER_CHOICE.md) has the full per-corpus table and the decision guide.

The full story — including misses, per-corpus regressions, and what doesn't work — is in [HONEST_CLAIMS.md](docs/HONEST_CLAIMS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Quick start

Python `>=3.11`. First build downloads the default encoder (`BAAI/bge-large-en-v1.5`, ~1.3 GB). Builds and queries run fully locally after that.

```bash
pip install resonance-lattice
pip install onnxruntime  # optional — 2-5× encoding speedup
```

**Build** a knowledge model from your files:

```bash
rlat build ./docs ./src -o project.rlat
```

Or against a public GitHub repo — the knowledge model pins to the current commit SHA and serves from a local cache thereafter:

```bash
rlat build https://github.com/MicrosoftDocs/fabric-docs -o fabric-docs.rlat
```

**Search** it:

```bash
rlat search project.rlat "how does auth work?" --format text
```

**Ask** a grounded question:

```bash
# Full pipeline: expand + hybrid + LLM reader with cited answer
rlat ask project.rlat "how does auth work?" --reader llm

# Evidence over synthesis — no LLM, deterministic:
rlat ask project.rlat "how does auth work?" --reader context
```

**Inspect** the shape of what you built:

```bash
rlat profile project.rlat
rlat xray    project.rlat
```

For a one-command project setup that wires in `.claude/resonance-context.md`, `.rlat/manifest.json`, MCP config, and assistant integration:

```bash
rlat init-project --auto-integrate
```

The full walkthrough — profiling, composition, MCP, HTTP serving — is in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

---

## Prebuilt knowledge models

Skip `rlat build` entirely — we publish five production-ready knowledge models on HuggingFace Hub. Each ships as both a `bundled` variant (self-contained, works offline) and a `remote` variant (SHA-pinned, `rlat sync`-able). Identical retrieval quality; the trade-off is network dependency at query time.

| Corpus | HuggingFace repo |
|---|---|
| Microsoft Fabric docs | [`tenfingers/fabric-docs-rlat`](https://huggingface.co/datasets/tenfingers/fabric-docs-rlat) |
| Power BI Developer docs | [`tenfingers/powerbi-developer-rlat`](https://huggingface.co/datasets/tenfingers/powerbi-developer-rlat) |
| PowerShell docs | [`tenfingers/powershell-docs-rlat`](https://huggingface.co/datasets/tenfingers/powershell-docs-rlat) |
| Python standard library | [`tenfingers/python-stdlib-rlat`](https://huggingface.co/datasets/tenfingers/python-stdlib-rlat) |
| T-SQL docs | [`tenfingers/tsql-docs-rlat`](https://huggingface.co/datasets/tenfingers/tsql-docs-rlat) |

```bash
huggingface-cli download tenfingers/fabric-docs-rlat fabric-docs-bundled.rlat --local-dir .
rlat search fabric-docs-bundled.rlat "how do I create a lakehouse"
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for the three workflows — **keep bundled** (offline-ready), **switch to repo-backed** (track upstream with `rlat freshness` / `rlat sync`), or **repoint to a local synced folder** (for vendored / air-gapped / editable checkouts).

---

## Assistant integration

Three composable entry points. Use what fits — they work side by side.

**MCP server** — native tool integration in Claude Code, Copilot, Cursor, and any MCP-compatible client. Single-line config:

```json
{ "mcpServers": { "rlat": { "command": "rlat", "args": ["mcp", "project.rlat"] } } }
```

**CLI** — zero-config, works with any assistant that can run shell commands. `--format json|context|prompt` controls how output is framed.

**Knowledge-model-backed skills** — skills keep their workflow structure while their reference knowledge adapts to each request. Four-tier injection (static, foundational, user-query, derived). Skills without knowledge-model fields work exactly as they did.

See [docs/MCP.md](docs/MCP.md), [docs/CLI.md](docs/CLI.md), and [docs/SKILL_INTEGRATION.md](docs/SKILL_INTEGRATION.md) for the detailed interfaces.

---

## Common workflows

**Keep a knowledge model fresh as sources change:**

```bash
rlat add  project.rlat ./new_docs              # append new sources
rlat sync project.rlat ./docs ./src            # track additions, updates, deletions
rlat refresh project.rlat --source-root .      # re-encode drifted chunks
```

**Compose across domains:**

```bash
rlat search docs.rlat "auth flow" --with code.rlat
rlat search code.rlat "data handling" --through compliance.rlat
rlat search current.rlat "what changed?" --diff baseline.rlat
```

**Shape retrieval without tuning knobs:**

```bash
# Task-matched presets
rlat search corpus.rlat "exact error code" --tune focus
rlat search corpus.rlat "design trade-offs" --tune explore

# Asymmetric contrast: what does my corpus know that a baseline doesn't?
rlat search internal-api.rlat "authentication" --contrast vendor-docs.rlat
```

**Generate an assistant primer** (compact, query-derived, auto-refreshes on build):

```bash
rlat summary project.rlat -o .rlat/resonance-context.md
# Reference from CLAUDE.md / .github/copilot-instructions.md / .cursorrules
```

---

## Practical limits

| What | Detail | What to do |
|------|--------|------------|
| **Initial build is CPU-intensive** | First build encodes every chunk through BGE-large-en-v1.5 | Incremental sync only re-processes changed files. ONNX runtime (`pip install onnxruntime`) gives 2-5× CPU speedup. CUDA GPU supported if available. |
| **Default encoder is English-optimised** | Non-English retrieval less reliable with the default setup | `--encoder` is configurable; multilingual alternatives should be revalidated on your corpus. |
| **Best numbers use the full pipeline** | Lexical injection + reranking helps on factual / technical corpora; short-prose corpora sometimes regress | `rlat search` auto-selects mode from signal-separation; `--hybrid` is opt-in with documented per-corpus sensitivity. |

See [docs/STATUS_AND_BOUNDARIES.md](docs/STATUS_AND_BOUNDARIES.md) for shipped surfaces, experimental areas, and current limits.

---

## Status

**`0.11.0`** — heading to v1.0.0 on 2026-06-08. Roadmap and live workstreams are on the [public GitHub project board](https://github.com/users/tenfingerseddy/projects/1).

## Docs

- [Core Use Cases](docs/CORE_FEATURES.md) — the seven workflows the knowledge model unlocks
- [Overview](docs/OVERVIEW.md) — product positioning and comparison matrix
- [Getting Started](docs/GETTING_STARTED.md) — install → first query
- [Knowledge Model Architecture](docs/KNOWLEDGE_MODEL_ARCHITECTURE.md) — file format + internals
- [Storage Modes](docs/STORAGE_MODES.md) — bundled / local / remote, when to pick each
- [Semantic Model](docs/SEMANTIC_MODEL.md) — the three-layer retrieval story in depth
- [Honest Claims](docs/HONEST_CLAIMS.md) — calibration document, numbers with methodology
- [Encoders](docs/ENCODERS.md) — encoder guide (backbones, projection heads, warm path); [Encoder Choice](docs/ENCODER_CHOICE.md) for the per-workload decision
- [CLI Reference](docs/CLI.md), [MCP](docs/MCP.md), [Skill Integration](docs/SKILL_INTEGRATION.md)
- [RQL Reference](docs/RQL_REFERENCE.md) — programmable field operations
- [Benchmarks](docs/BENCHMARKS.md), [FAQ](docs/FAQ.md)

## License

[Business Source License 1.1](LICENSE.md) — broad permitted use for internal development, CI/CD, coding assistants, and your own products. Each release converts to MPL 2.0 four years after it's first published. Plain-language summary: [LICENSE_FAQ.md](docs/LICENSE_FAQ.md).
