---
name: rlat
description: >-
  Use rlat for semantic search, knowledge model building, profiling, comparison,
  and LLM context injection. Trigger on: rlat, .rlat files, knowledge models,
  semantic search, grep-vs-rlat, build/sync, injection modes. ALSO trigger
  on conceptual questions about any indexed corpus — codebases, technical
  docs, knowledge bases: "how does X work", "why did we choose Y",
  "what is the strategy for Z", "what evidence exists", "explain the design",
  research tasks, understanding rationale, cross-file synthesis, finding
  benchmark results, or any question about meaning/architecture/history that
  spans multiple files. Works on any .rlat knowledge model in the workspace —
  codebases, technical docs, knowledge bases, or any other corpus packaged
  as a knowledge model. Use rlat search as the starting point for research,
  then follow up with grep for exact symbols.
argument-hint: "[command] [query] or a natural-language question about an indexed corpus"

# rlat skill-loader extensions (below) — NOT part of Anthropic's skill spec.
# These fields are consumed by `rlat skill inject` / `rlat skill route` to wire
# dynamic knowledge-model context into the skill at activation time. The standard
# Claude skill loader ignores them harmlessly. Points at project.rlat — the BGE
# knowledge model built from ./docs and ./src. Same pattern as fabric-lakehouse-
# medallion → fabric.rlat. See memory/feedback_skill_cartridge_wiring.md.
knowledge models:
  - project.rlat
knowledge model-queries:
  - "rlat CLI commands build sync search ask"
  - "knowledge model architecture field registry store three-layer"
  - "retrieval lens selection search locate profile compare negotiate compose"
knowledge model-mode: augment
knowledge model-budget: 2000
knowledge model-derive: true
knowledge model-derive-count: 3
---

# Resonance Lattice CLI (rlat)

Resonance Lattice is a **portable semantic model for knowledge**. It packages a corpus into a single `.rlat` knowledge model containing three layers:

- **Field** — a semantic model of what the corpus knows
- **Registry** — maps semantic hits back to ranked source files
- **Store** — returns evidence text, passages, and metadata

The CLI (`rlat`) is the primary interface. Build a knowledge model from local files, query it semantically, inspect its shape, compare versions, and generate LLM-ready context — all without an LLM, API keys, or network calls. The encoder runs locally and the knowledge model is a portable file.

```bash
pip install resonance-lattice
rlat build ./docs ./src -o project.rlat
rlat search project.rlat "how does auth work?"
```

---

## When to Use rlat vs Other Tools

| Task | Best tool | Why |
|------|-----------|-----|
| Semantic/conceptual search ("how does auth work?") | **rlat search** | Finds meaning, not keywords |
| Exact string or symbol lookup (`getUser`, `TODO`) | **Grep** | Exact match is faster and precise |
| Read a specific known file | **Read** | Direct file access |
| Find files by name pattern | **Glob** | Pattern matching |
| Multi-document synthesis ("what does this project do?") | **rlat search** | Cross-file semantic understanding |
| Fast orientation in an unfamiliar repo | **rlat summary** | Compressed semantic primer |
| Compare what two codebases know | **rlat compare** | Semantic diff, not file diff |
| Check if docs contradict each other | **rlat contradictions** | Interference detection |
| Find a function definition | **Grep** | Exact symbol match |
| Understand coverage gaps in documentation | **rlat xray** / **probe** | Field diagnostics |
| Generate grounded LLM context | **rlat resonate** | Evidence-backed injection |
| Check current git state | **Bash** (git) | Git operations |

**Rule of thumb**: Use rlat when you need to understand *meaning* across files. Use Grep when you know the *exact text* to find. They are complementary — rlat for semantic exploration, Grep for precision lookup.

When the user asks a conceptual question about their codebase or docs, prefer `rlat search` if a knowledge model exists. When they ask for a specific function, variable, or string, prefer Grep.

---

## Quick Command Reference

### Primary — Interactive query and inspection

| Command | Purpose | Example |
|---------|---------|---------|
| `search` | Enriched semantic query with coverage, topics, contradictions | `rlat search project.rlat "auth flow"` |
| `ask` | Smart query: auto-selects best retrieval lens | `rlat ask project.rlat "how does auth work?"` |
| `profile` | Inspect knowledge model shape: bands, energy, rank, coverage | `rlat profile project.rlat` |
| `compare` | Semantic diff between two knowledge models | `rlat compare old.rlat new.rlat` |
| `ls` | List sources in a knowledge model | `rlat ls project.rlat --grep "auth" -v` |
| `info` | Knowledge Model metadata: size, bands, dim, sources | `rlat info project.rlat` |

### Build — Create and update knowledge models

| Command | Purpose | Example |
|---------|---------|---------|
| `build` | Build knowledge model from files/directories | `rlat build ./docs ./src -o project.rlat` |
| `add` | Append new files to existing knowledge model | `rlat add project.rlat ./new_docs` |
| `sync` | Smart incremental update (add/update/remove) | `rlat sync project.rlat ./docs ./src` |
| `init` | Create empty knowledge model with custom params | `rlat init -o empty.rlat --dim 1024` |
| `ingest` | Lower-level single-path ingestion | `rlat ingest project.rlat ./file.md` |
| `init-project` | One-command project setup | `rlat init-project` |

### Serve & Context — Expose the model

| Command | Purpose | Example |
|---------|---------|---------|
| `serve` | HTTP server for network queries | `rlat serve project.rlat --port 8080` |
| `summary` | Generate code primer from knowledge model | `rlat summary project.rlat -o primer.md` |
| `mcp` | MCP server for Claude Code integration | `rlat mcp project.rlat` |

### Memory — Layered conversation memory

| Command | Purpose | Example |
|---------|---------|---------|
| `memory init` | Initialize 3-tier memory root | `rlat memory init ./memory/` |
| `memory write` | Write transcripts to a memory tier | `rlat memory write ./memory/ --input-file t.jsonl --tier working` |
| `memory recall` | Query across memory tiers with fusion | `rlat memory recall ./memory/ "auth decisions"` |
| `memory primer` | Generate memory primer for CLAUDE.md | `rlat memory primer ./memory/ -o .claude/memory-primer.md` |
| `memory consolidate` | Promote between tiers | `rlat memory consolidate ./memory/` |
| `memory gc` | Enforce TTL and capacity limits | `rlat memory gc ./memory/` |
| `memory profile` | Per-tier memory stats | `rlat memory profile ./memory/` |

### Query & Analysis — Specialized retrieval and diagnostics

| Command | Purpose | Example |
|---------|---------|---------|
| `query` | Basic ranked passages (no enrichment) | `rlat query project.rlat "storage"` |
| `resonate` | LLM-ready context with injection modes | `rlat resonate project.rlat "auth" --mode constrain` |
| `contradictions` | Find conflicting sources | `rlat contradictions project.rlat "auth"` |
| `topology` | Knowledge topology and cluster analysis | `rlat topology project.rlat --band 0` |
| `xray` | Corpus-level health diagnostics | `rlat xray project.rlat --deep` |
| `locate` | Where a query sits in the knowledge landscape | `rlat locate project.rlat "rate limiting"` |
| `probe` | Quick insight recipes (health, novelty, gaps...) | `rlat probe project.rlat health` |

### Algebra — Compose and transform knowledge models

| Command | Purpose | Example |
|---------|---------|---------|
| `merge` | Combine two knowledge models (commutative) | `rlat merge a.rlat b.rlat -o merged.rlat` |
| `forget` | Remove a source (exact algebraic operation) | `rlat forget project.rlat --source "old_file"` |
| `diff` | Algebraic delta between versions | `rlat diff v2.rlat v1.rlat -o delta.rlat` |

### Composition — Multi-knowledge model search and context control

| Command/Flag | Purpose | Example |
|---------|---------|---------|
| `search --with` | Compose with additional knowledge models | `rlat search docs.rlat "auth" --with code.rlat` |
| `search --through` | Project through a lens knowledge model | `rlat search code.rlat "data" --through compliance.rlat` |
| `search --diff` | Search what changed vs baseline | `rlat search current.rlat "auth" --diff baseline.rlat` |
| `search --boost` | Boost a topic during search | `rlat search docs.rlat "overview" --boost "security"` |
| `search --suppress` | Suppress a topic on the fly | `rlat search docs.rlat "news" --suppress "politics"` |
| `search --lens` | Apply a knowledge lens | `rlat search docs.rlat "config" --lens denoise` |
| `compose` | Expression-based composition | `rlat compose "docs.rlat ^ code.rlat" "auth"` |

### Discovery — Find and inspect knowledge models

| Command/Flag | Purpose | Example |
|---------|---------|---------|
| `init-project --auto-integrate` | Build + manifest + wire MCP + inject CLAUDE.md | `rlat init-project --auto-integrate` |
| `skill build` | Build knowledge model from a skill's references | `rlat skill build .claude/skills/my-skill/` |
| `skill inject` | Four-tier adaptive context injection | `rlat skill inject my-skill "query" --derived "q1;q2"` |
| `skill route` | Rank skills by resonance energy | `rlat skill route "query"` |
| `skill search` | Search a skill's knowledge models | `rlat skill search my-skill "query"` |
| `skill profile` | Semantic profile of a skill's knowledge model | `rlat skill profile my-skill` |
| `skill freshness` | Check staleness of skill knowledge models | `rlat skill freshness` |
| MCP: `rlat_discover` | List available knowledge models (use FIRST) | Via MCP tool call |
| MCP: `rlat_freshness` | Check if knowledge model needs rebuild | Via MCP tool call |
| MCP: `rlat_compose_search` | Composed multi-knowledge model search | Via MCP tool call |

### Export

| Command | Purpose | Example |
|---------|---------|---------|
| `export` | Export knowledge model (optionally field-only) | `rlat export project.rlat -o shared.rlat --field-only` |

For full flag tables and detailed options, read [references/CLI_REFERENCE.md](references/CLI_REFERENCE.md).

---

## Knowledge Model Discovery

Before using rlat, check what knowledge models are available:

1. **If MCP is configured**: Call `rlat_discover` first. It returns all available knowledge models, their domains, and freshness.
2. **If using CLI**: Check `.rlat/manifest.json` or glob for `.rlat/*.rlat` files.
3. **If neither exists**: Suggest `rlat init-project --auto-integrate` to the user.

When a user asks a conceptual question:

1. Check `rlat_discover` for available knowledge models.
2. Pick the knowledge model whose domain matches the question.
3. For cross-domain questions, use `--with` to compose multiple knowledge models.
4. For "what changed" questions, use `--diff` against a baseline.

---

## Output Formats

Most query commands accept `--format`:

| Format | Use for | What you get |
|--------|---------|-------------|
| `text` (default) | Terminal reading | ANSI color, coverage bars, source paths, timing |
| `json` | Scripts, pipelines | Full metadata: scores, band_scores, passages, latency |
| `context` | LLM injection (dense) | Compressed line-per-passage: `- [score] passage...` |
| `prompt` | Copy-paste into LLM | Rich markdown with coverage, passages, related topics |

## Injection Modes

When using `--format context` or `--format prompt`, control how an LLM should use the evidence with `--mode`:

| Mode | LLM behavior | Best for |
|------|-------------|----------|
| `augment` | Use own knowledge + cite sources for detail | General-purpose assistants |
| `constrain` | Answer ONLY from provided sources, cite [1][2] | High-stakes, compliance, zero hallucination |
| `knowledge` | Base answer on this context, be transparent about gaps | Domain-specific or proprietary content |
| `custom` | Your own system prompt via `--custom-prompt` | Full control |

```bash
rlat resonate project.rlat "query" --mode constrain --format context
rlat search project.rlat "query" --format prompt --mode augment
```

## Key Flags

| Flag | Commands | Effect |
|------|----------|--------|
| `--top-k N` | search, query, resonate | Number of results (default: 10) |
| `--format` | most commands | Output format (text/json/context/prompt) |
| `--cascade` | search | Enable related topics discovery |
| `--with-contradictions` | search | Enable contradiction detection |
| `--mode` | search, query, resonate | Injection framing (augment/constrain/knowledge/custom) |
| `--source-root PATH` | search, resonate, summary | Resolve evidence from local files (external store) |
| `--onnx PATH` | search, resonate, mcp | ONNX backbone for 2-5x faster encoding |
| `--no-worker` | search | Disable background warm worker |
| `--store-mode` | build | `local` (default), `bundled` (self-contained), or `remote` (HTTP-backed). `embedded` is deprecated. |
| `--compression` | build | `none`, `zstd`, `lz4` |
| `--quantize-registry` | build | 0 (off), 8 (~50% smaller), 4 (~87% smaller) |
| `--deep` | xray | Add topological analysis |
| `-v` | search, ls, resonate | Verbose output with raw scores |

---

## Knowledge Model Concepts

A `.rlat` knowledge model has three layers: a fixed-size **field** (semantic router), a **registry** (source coordinates, scales with source count), and a **store** (evidence text, scales with corpus size). See the project `CLAUDE.md` / `resonance-context.md` primer for the architectural detail.

### Store Modes

Three modes share one abstraction (`LosslessStore`) and differ only in where raw source files live. The canonical flag is `--store-mode {bundled,local,remote}`; `external` is still accepted as a synonym for `local`. See [docs/STORAGE_MODES.md](../../../docs/STORAGE_MODES.md) for the full reference.

| Mode | Raw files live in | Pick when | Freshness command |
|------|-------------------|-----------|-------------------|
| `local` (default) | local directory resolved via `--source-root` | developing against a working copy; thin knowledge model | `rlat refresh` |
| `bundled` | inside the `.rlat` (zstd frames) | shipping a self-contained artifact — demos, CI, offline | rebuild |
| `remote` | HTTP origin with SHA-pinned local cache | pointing at an upstream repo you don't own (docs, public codebases) | `rlat freshness` / `rlat sync` |

The legacy `embedded` mode (pre-chunked SQLite store) is **deprecated and scheduled for removal at v2.0.0** — it is a different thing from `bundled`. Rebuild old `embedded` knowledge models with `--store-mode bundled` (self-contained) or the default `--store-mode local` (thin).

### Bands

The field is organised into multiple semantic bands, each capturing a different level of meaning (broad subject area through to close lexical matches). These are working labels, not guaranteed ontology categories.

### Encoder

Resolution order: (1) `--encoder` flag, (2) stored encoder in knowledge model, (3) default. Most users never need `--encoder` — the knowledge model stores its encoder at build time.

---

## Recipes

Six canonical flows. For variations (profiling pipelines, privacy-preserving export, layered memory, dual primers, skill injection, adaptive derived queries), read [references/RECIPES.md](references/RECIPES.md).

### 1. Bootstrap a new project

```bash
rlat init-project
# Or manually:
rlat build ./docs ./src -o .rlat/project.rlat
rlat summary .rlat/project.rlat -o .claude/resonance-context.md
```

### 2. Search with different formats

```bash
rlat search project.rlat "how does auth work?"                    # terminal
rlat search project.rlat "how does auth work?" --format json      # scripting / agent research
rlat search project.rlat "how does auth work?" --format prompt    # LLM paste
```

### 3. Keep a knowledge model fresh after file changes

```bash
rlat sync project.rlat ./docs ./src     # local mode — incremental add/update/remove
rlat freshness fabric-docs.rlat         # remote mode — drift check (read-only)
rlat sync fabric-docs.rlat              # remote mode — apply upstream diff
```

### 4. Grounded LLM injection

```bash
# Zero-hallucination: answer only from retrieved evidence
rlat resonate project.rlat "design constraints" --mode constrain --format context
# Augmented: LLM blends own knowledge with citations
rlat search project.rlat "design constraints" --format prompt --mode augment
```

### 5. MCP server for Claude Code

```bash
rlat mcp project.rlat
```

Add to Claude Code settings:

```json
{ "mcpServers": { "resonance": { "command": "rlat", "args": ["mcp", "project.rlat"] } } }
```

### 6. Research a subsystem (hybrid workflow)

```bash
# Semantic orientation first — find which files discuss the topic and why
rlat search project.rlat "scoring strategy rationale" --format json --top-k 5
# Then grep for exact symbols found in results
grep -rn "score_function" src/
# Then Read the specific implementation
```

Use this pattern when the task involves understanding *why* something was built a certain way, then making changes. rlat finds the rationale and evidence; grep finds the exact code. For deep multi-hop investigation across many files, see the [deep-research](../deep-research/SKILL.md) skill.

---

## Invocation

Use the `rlat` CLI entry point, not `python -m resonance_lattice.cli` — the module has no `__main__` entry and the module form fails with `ModuleNotFoundError`.

```bash
rlat search project.rlat "query"
```

### Format Selection

Choose the output format based on **who consumes the output**:

| Consumer | Format | Why |
|----------|--------|-----|
| Showing results to user in terminal | `--format text` | Human-readable with ANSI color, timing, paths |
| Agent consuming results as intermediate research | `--format json` | Full passage text, scores, band_scores — machine-parseable |
| Feeding context into an LLM prompt | `--format context` or `--format prompt` | Compressed/rich evidence for injection |

**Default to `--format json` when searching as a research step.** The `text` format truncates passages and adds ANSI noise that wastes context tokens. Only use `text` when piping output directly to the user's terminal.

### Prerequisites

Before running any `rlat` command, ensure the repo-local venv is activated. The `rlat` entry point is installed into the venv by `pip install -e ".[dev]"`.

On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Verify availability:
```bash
rlat --version
```

### Error Recovery

| Symptom | Fix |
|---------|-----|
| `rlat: command not found` / `not recognized` | Activate the venv: `.\.venv\Scripts\Activate.ps1` |
| Still missing after activation | Reinstall: `pip install -e ".[dev]"` then retry |
| `ModuleNotFoundError: resonance_lattice` | You're bypassing the entry point — use `rlat`, not `python -m` |
| Import errors on first run | Check `pip show resonance-lattice` to confirm the package is installed |

---

## Context Control Playbook

Decision rules for *how* to search — which composition operation, how to read the coverage assessment, which injection mode to pick, how to interpret `xray` / `profile` diagnostics, and when to compose proactively. Read [references/PLAYBOOK.md](references/PLAYBOOK.md) when the user's question involves multiple knowledge models, confidence assessment, or LLM grounding framing.

Quick rules of thumb:

- Cross-domain question → `--with` (merge). Specific perspective → `--through` (project). Change tracking → `--diff` against baseline. Noisy results → `--boost` / `--suppress`.
- Coverage label: `strong` answer confidently, `partial` flag thin aspects, `edge` add heavy caveats + suggest expansion, `gap` admit the knowledge model doesn't cover this.
- Injection mode: default `knowledge`, escalate to `constrain` for accuracy-critical, drop to `augment` for exploration.

---

## Standing Instructions

When using rlat in conversations:

- **Check for a knowledge model first.** Before suggesting `rlat search`, verify a `.rlat` file exists (check for `.rlat/` directory or `*.rlat` files in the workspace root). If none exists, suggest `rlat build` or `rlat init-project`.
- **Discover available knowledge models.** Don't assume knowledge model names — look for `*.rlat` files in the workspace. Each knowledge model indexes a different corpus. Choose the right one based on what the user is asking about.
- **Prefer rlat for conceptual queries.** When the user asks about meaning, architecture, relationships, or "how does X work?" — use `rlat search` if a knowledge model is available.
- **Prefer Grep for exact strings.** When the user wants a specific function name, variable, error message, or exact text — use Grep or Read.
- **Default format by context.** Use `--format text` for displaying to the user, `--format json` for scripting, `--format context` or `--format prompt` for feeding into an LLM. When using search results as an intermediate research step (not showing directly to the user), prefer `--format json` so scores, band_scores, and passage text are machine-parseable.
- **Use `--no-worker` in non-interactive contexts.** When running search from a script or subagent, disable the warm worker.
- **Always show the command.** When running rlat commands, show the user the exact command being executed.
- **Summarize results before acting.** After search, always summarize findings in 2-3 sentences before following up with grep or read_file. Do not silently consume results and jump to the next step — the user should see what the semantic search found and why you are drilling into specific files.
- **Focus your queries.** For broad research questions, prefer 2-3 focused queries over one overloaded query string. Each query should target a single facet (e.g. one for rationale/design, one for implementation details). Overloaded queries dilute retrieval precision.
- **Confirm destructive operations.** Before `forget`, `merge`, `build` (overwrite), or `sync` (which removes deleted files), confirm with the user.
- **Generate derived queries for skill inject (when enabled).** When using `rlat skill inject` on a skill that has `knowledge model-derive: true` (or doesn't set it, since true is the default), generate 2-3 short (under 15 words), specific search queries targeting knowledge the user didn't explicitly ask for but will need. Pass them via `--derived "q1;q2;q3"`. Do not restate the user's query — focus on implicit prerequisites, error handling patterns, related conventions, or integration requirements the user would otherwise miss. **Skip this if the skill sets `knowledge model-derive: false`.**
- **Use skill route for ambiguous requests.** When a user's question could match multiple knowledge model-backed skills, run `rlat skill route "query"` first to see which skill has the strongest knowledge match before choosing which knowledge model to search.
- **Route $ARGUMENTS.** When the user passes arguments:
  - If it looks like a subcommand name (`search`, `build`, `profile`...), route to that command's documentation.
  - If it's a natural-language question, use `rlat search` with the knowledge model.
  - If unclear, ask the user what they need.

---

## Benchmarks & Performance

**Retrieval quality** (24,635-chunk corpus, 100 questions):

| System | R@5 | MRR | Failed |
|--------|-----|-----|--------|
| BM25 | 0.84 | 0.72 | 16% |
| Flat dense (cosine) | 0.93 | 0.80 | 7% |
| Hybrid RRF (BM25 + Dense) | 0.94 | 0.77 | 6% |
| **rlat reranked** | **1.00** | **0.92** | **0%** |

**LLM grounding**: Feeding rlat context to an LLM reduced hallucinations from 76% to 14%, lifted fact recall from 0.28 to 0.88.

**Speed**: ~8ms warm field resonance, ~80ms warm full pipeline, 50-500ms cold (first query loads knowledge model).

**Honest caveats**: The dense field alone underperforms flat cosine — the win comes from the full retrieval pipeline. These benchmarks are from one corpus; cross-corpus evaluation is ongoing.

**Session primers**: `rlat summary` produces ~25x less context than grep+read workflows for meaningful session startup.

For detailed reference, see:
- [references/CLI_REFERENCE.md](references/CLI_REFERENCE.md) — full flag tables for all 25 commands
- [references/FAQ.md](references/FAQ.md) — common questions about RL
- [references/RECIPES.md](references/RECIPES.md) — extended recipes with variations
