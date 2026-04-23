# rlat CLI Reference

Complete flag reference for all commands. For quick lookup, see the command tables in SKILL.md.

## Contents

- [Primary Commands](#primary-commands) — search, ask, profile, compare, ls, info
- [Build Commands](#build-commands) — build, add, sync, init, ingest, init-project
- [Serve & Context Commands](#serve--context-commands) — serve, summary, mcp
- [Query & Analysis Commands](#query--analysis-commands) — query, resonate, contradictions, topology, xray, locate, probe
- [Algebra Commands](#algebra-commands) — merge, forget, diff
- [Export Commands](#export-commands) — export (optional field-only)
- [Choosing the Right Command](#choosing-the-right-command) — decision tree
- [Warm Worker System](#warm-worker-system) — background worker semantics and `--no-worker`
- [Skill Commands](#skill-commands) — skill build, inject, route, search, profile, freshness
- [Environment Variables](#environment-variables)

---

## Primary Commands

### `rlat search <knowledge model> <query> [options]`

Full enriched semantic query. The main product surface.

**Core options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | `text`, `json`, `context`, `prompt` |
| `-v, --verbose` | flag | off | Show raw scores and detailed timings |

**Enrichment options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cascade` | flag | off | Enable related topics cascade |
| `--cascade-depth` | int | 2 | Cascade hop depth |
| `--with-contradictions` | flag | off | Enable contradiction detection |
| `--contradiction-threshold` | float | -0.3 | Destructive interference threshold |
| `--subgraph` | flag | off | Expand results with related neighbours |
| `--subgraph-k` | int | 3 | Neighbours per result for subgraph expansion |

**Encoder options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override (default, `random` for testing, or HuggingFace model) |
| `--checkpoint` | str | None | Trained heads checkpoint (experimental, not recommended) |
| `--source-root` | str | None | External source root for file-backed store |
| `--onnx` | str | None | ONNX backbone directory (auto-detects `<stem>_onnx/`) |

**Performance options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-worker` | flag | off | Disable background warm worker |

**Injection options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | None | `augment`, `constrain`, `knowledge`, `custom` |
| `--custom-prompt` | str | None | Custom system prompt (requires `--mode custom`) |

**Examples:**

```bash
rlat search project.rlat "how does auth work?"
rlat search project.rlat "auth" --format json --cascade --with-contradictions
rlat search project.rlat "auth" --format prompt --mode constrain
rlat search project.rlat "auth" --onnx ./onnx_backbone/ --no-worker
```

---

### `rlat profile <knowledge model> [options]`

Semantic profile of a knowledge model. Shows per-band energy, signal quality metrics, and source count.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | `text`, `json` |

```bash
rlat profile project.rlat
rlat profile project.rlat --format json
```

---

### `rlat compare <model_a> <model_b> [options]`

Compare two knowledge models semantically. Reports overlap, unique coverage, per-band energy differences.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | `text`, `json` |

```bash
rlat compare baseline.rlat current.rlat --format json
```

---

### `rlat ls <knowledge model> [options]`

List sources in a knowledge model.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | `text`, `json` |
| `-v, --verbose` | flag | off | Show summaries for each source |
| `--head` | int | None | Show only first N sources |
| `--grep` | str | None | Filter sources by substring match |

```bash
rlat ls project.rlat --grep "auth" --head 5 -v
```

---

### `rlat info <knowledge model>`

Display knowledge model metadata: source count, field type, dimensions, bands, compression, encoder.

```bash
rlat info project.rlat
```

---

## Build Commands

### `rlat build <inputs...> -o <output> [options]`

Build a knowledge model from source files. Discovers files recursively, chunks, encodes, and builds the field.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output `.rlat` file |
| `--bands` | int | (default) | Number of semantic bands |
| `--dim` | int | (default) | Field dimension per band |
| `--field-type` | choice | dense | `dense`, `factored`, `pq` |
| `--precision` | choice | f32 | `f16`, `bf16`, `f32` |
| `--compression` | choice | none | `none`, `zstd`, `lz4` |
| `--encoder` | str | None | Encoder choice (default or `random` for testing) |
| `--checkpoint` | str | None | Trained heads checkpoint (not recommended) |
| `--quantize-registry` | int | 0 | 0=off, 8=~50% compression, 4=~87% compression |
| `--store-mode` | choice | embedded | `embedded` (full store) or `external` (field+registry only) |

**Supported file types:**
- Text/markup: `.txt`, `.md`, `.rst`, `.html`, `.xml`
- Code: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.c`, `.cpp`, `.go`, `.rb`, `.rs`, `.swift`, `.kt`, `.lua`, `.r`, `.php`, `.sh`, `.sql`, `.css`, `.scss`
- Data/config: `.json`, `.yaml`, `.yml`, `.toml`, `.csv`, `.tsv`
- Binary (optional deps): `.pdf` (pdfplumber), `.docx` (python-docx), `.xlsx` (openpyxl)

**Auto-skipped directories:** `__pycache__`, `.git`, `node_modules`, `.venv`, `venv`, `dist`, `build`, `.env`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.egg-info`, `htmlcov`, `.ipynb_checkpoints`

```bash
rlat build ./docs ./src -o project.rlat
rlat build ./docs -o project.rlat --compression zstd --quantize-registry 8
rlat build ./docs -o project.rlat --store-mode external
```

---

### `rlat add <knowledge model> <inputs...> [options]`

Incrementally add files to an existing knowledge model. Skips already-present files by content hash.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override |
| `--checkpoint` | str | None | Trained heads checkpoint (not recommended) |

```bash
rlat add project.rlat ./new_docs
```

---

### `rlat sync <knowledge model> <inputs...> [options]`

Smart incremental update. Detects added, modified, and deleted files. Adds new, re-encodes changed, forgets removed.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override |
| `--checkpoint` | str | None | Trained heads checkpoint (not recommended) |

```bash
rlat sync project.rlat ./docs ./src
```

---

### `rlat init -o <output> [options]`

Create a new empty knowledge model with specified parameters.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output `.rlat` file |
| `--bands` | int | (default) | Number of semantic bands |
| `--dim` | int | (default) | Field dimension per band |
| `--field-type` | choice | dense | `dense`, `factored`, `pq` |
| `--pq-subspaces` | int | (default) | PQ subspaces (for `pq` field type) |
| `--pq-codebook-size` | int | (default) | PQ codebook size (for `pq` field type) |
| `--svd-rank` | int | (default) | Rank (for `factored` field type) |
| `--precision` | choice | f32 | `f16`, `bf16`, `f32` |
| `--compression` | choice | none | `none`, `zstd`, `lz4` |

```bash
rlat init -o empty.rlat --field-type factored
```

---

### `rlat ingest <knowledge model> <input> [options]`

Lower-level ingestion of a single file or directory.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override |
| `--checkpoint` | str | None | Trained heads checkpoint (not recommended) |

```bash
rlat ingest project.rlat ./new_document.md
```

---

### `rlat init-project [inputs...] [options]`

One-command project setup. Auto-detects `docs/`, `src/`, `lib/`, `README.md`, `CLAUDE.md`, `AGENTS.md`. Builds knowledge model, generates summary, prints integration hints.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | `.claude/resonance-context.md` | Summary output path |
| `--encoder` | str | None | Encoder choice |
| `--checkpoint` | str | None | Trained heads checkpoint (not recommended) |

```bash
rlat init-project
rlat init-project ./docs ./src --encoder random
```

---

## Serve & Context Commands

### `rlat serve <knowledge model> [options]`

HTTP server. Endpoints: `GET /health`, `GET /info`, `POST /query`, `POST /search`, `POST /add`, `POST /remove`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | int | 8080 | Port number |
| `--host` | str | 0.0.0.0 | Bind address |
| `--encoder` | str | None | Encoder override |

```bash
rlat serve project.rlat --port 9090
```

HTTP query:
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"text": "how does auth work", "top_k": 10}'
```

---

### `rlat summary <knowledge model> [options]`

Generate a pre-injection context primer. Bootstraps by running internal queries to sample the knowledge model's knowledge.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | stdout | Output file path |
| `--format` | choice | context | `context` (rich primer) or `stats` (field metadata only) |
| `--queries` | str | None | Custom bootstrap queries separated by semicolons |
| `--top-k` | int | 20 | Results per bootstrap query |
| `--encoder` | str | None | Encoder override |
| `--source-root` | str | None | External source root |
| `--budget` | int | 2500 | Target token budget for output |

```bash
rlat summary project.rlat -o .claude/resonance-context.md
rlat summary project.rlat --queries "architecture;testing;deployment" --top-k 30
rlat summary project.rlat --format stats
```

---

### `rlat mcp <knowledge model> [options]`

MCP server over stdio. Enables Claude Code and MCP-compatible tools to query directly.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source-root` | str | None | External source root |
| `--onnx` | str | None | ONNX backbone directory |

```bash
rlat mcp project.rlat --source-root .
```

Claude Code settings:
```json
{
  "mcpServers": {
    "resonance": {
      "command": "rlat",
      "args": ["mcp", "project.rlat"]
    }
  }
}
```

---

## Query & Analysis Commands

### `rlat query <knowledge model> <query> [options]`

Basic ranked passages. No enrichment (no coverage, cascade, or contradictions). Lighter than `search`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | `text`, `json`, `context`, `prompt` |
| `--encoder` | str | None | Encoder override |
| `--mode` | choice | None | `augment`, `constrain`, `knowledge`, `custom` |
| `--custom-prompt` | str | None | Custom system prompt |

```bash
rlat query project.rlat "storage architecture" --format json
```

---

### `rlat resonate <knowledge model> <query> [options]`

LLM-ready context output. Same retrieval as `query` but defaults to `context` format with injection mode support.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | context | `text`, `json`, `context`, `prompt` |
| `--mode` | choice | None | `augment`, `constrain`, `knowledge`, `custom` |
| `--custom-prompt` | str | None | Custom system prompt |
| `--source-root` | str | None | Resolve from local files |
| `--onnx` | str | None | ONNX backbone directory |
| `-v, --verbose` | flag | off | Show raw scores |

```bash
rlat resonate project.rlat "how does auth work?" --mode constrain
rlat resonate project.rlat "auth" --source-root ./docs --format prompt
```

---

### `rlat contradictions <knowledge model> <query> [options]`

Find contradicting sources via destructive interference detection.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--band` | int | 2 | Band to check (default: relations) |
| `--threshold` | float | 0.7 | Anti-correlation threshold |
| `--top-k` | int | 20 | Candidates to check |
| `--encoder` | str | None | Encoder override |

```bash
rlat contradictions project.rlat "auth requirements" --threshold 0.5
```

---

### `rlat topology <knowledge model> [options]`

Knowledge topology and cluster analysis. Identifies clusters and connectivity.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--band` | int | 0 | Band to analyze |
| `--top-k` | int | 20 | Number of top components to analyse |
| `--output, -o` | str | None | Output JSON file |

```bash
rlat topology project.rlat --band 0 -o topology.json
```

---

### `rlat xray <knowledge model> [options]`

Corpus-level semantic diagnostics. Per-band health and signal quality metrics. With `--deep`, adds topological analysis. Requires DenseField.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | `text`, `json`, `prompt` |
| `--deep` | flag | off | Add topological analysis and community detection |

```bash
rlat xray project.rlat
rlat xray project.rlat --deep --format json
```

---

### `rlat locate <knowledge model> <query> [options]`

Query positioning: where a question sits in the knowledge landscape. Reports per-band energy, coverage labels (strong/partial/edge/gap), and expansion hints. Requires encoder + DenseField.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | `text`, `json`, `prompt` |
| `--encoder` | str | None | Encoder override |

```bash
rlat locate project.rlat "how do we handle rate limiting?"
```

---

### `rlat probe <knowledge model> <recipe> [query] [options]`

RQL quick insight recipes. Each composes 2-5 operations into a named analysis.

**Recipes:**

| Recipe | Query needed | What it reports |
|--------|-------------|-----------------|
| `health` | No | Signal/noise split, SNR, effective rank per band |
| `novelty` | Yes | How novel content is relative to corpus (0-1 with ADD/OPTIONAL/SKIP) |
| `saturation` | No | Field capacity usage, remaining source capacity |
| `band-flow` | No | Inter-band mutual information, strongest/weakest couplings |
| `anti` | Yes | What the field does NOT know about the query |
| `gaps` | No | Topological gap analysis: clusters, loops, robustness |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | `text`, `json`, `prompt` |
| `--encoder` | str | None | Encoder override |

```bash
rlat probe project.rlat health
rlat probe project.rlat novelty "quantum computing concepts"
rlat probe project.rlat saturation --format json
rlat probe project.rlat anti "distributed consensus" --format prompt
```

---

## Algebra Commands

### `rlat merge <model_a> <model_b> -o <output>`

Merge two knowledge models. Commutative: `merge(A, B) = merge(B, A)`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output merged `.rlat` file |

```bash
rlat merge frontend.rlat backend.rlat -o fullstack.rlat
```

---

### `rlat forget <knowledge model> --source <source_id>`

Remove a source. Algebraically exact operation. Use `rlat ls` to find source IDs.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source` | str | **required** | Source ID to remove |

```bash
rlat forget project.rlat --source "deprecated_auth_md"
```

---

### `rlat diff <model_a> <model_b> [options]`

Algebraic delta: what's in A but not in B. Self-diff produces near-zero energy.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | None | Output delta `.rlat` file |

```bash
rlat diff v2.rlat v1.rlat -o whats_new.rlat
```

---

## Export Commands

### `rlat export <knowledge model> -o <output> [options]`

Export a knowledge model. `--field-only` strips the evidence store for privacy-preserving sharing.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output path |
| `--field-only` | flag | off | Export field+registry without source store |

```bash
rlat export project.rlat -o shared.rlat --field-only
```

---

## Choosing the Right Command

### search vs query vs resonate

| | `search` | `query` | `resonate` |
|---|---------|---------|------------|
| **Purpose** | Full enriched retrieval | Basic ranked passages | LLM-ready context |
| **Default format** | text | text | context |
| **Coverage profile** | Yes | No | No |
| **Topic cascade** | Optional | No | No |
| **Contradictions** | Optional | No | No |
| **Warm worker** | Yes | No | No |
| **Best for** | Interactive exploration | Scripts, debugging | Feeding an LLM |

### Which command for my use case?

| Use case | Command |
|----------|---------|
| Find information in a knowledge model | `search` |
| Generate context for an LLM | `resonate` or `search --format context` |
| Understand what a knowledge model covers | `profile` |
| Compare two codebase versions | `compare` |
| Check field health and signal quality | `probe health` |
| See if new content would add value | `probe novelty "content"` |
| Find conflicting information | `contradictions` or `search --with-contradictions` |
| Understand knowledge topology | `topology` or `xray --deep` |
| See where a question sits in the field | `locate` |
| List what's in the knowledge model | `ls` |
| Get knowledge model metadata | `info` |
| Build from scratch | `build` |
| Update after file changes | `sync` |
| Set up a new project quickly | `init-project` |
| Generate an assistant primer | `summary` |
| Share without source text | `export --field-only` |
| Serve over HTTP | `serve` |
| Integrate with Claude Code | `mcp` |
| Combine team knowledge | `merge` |
| Remove outdated content | `forget` |
| See what changed between versions | `diff` |

---

## Warm Worker System

The `search` command spawns a background HTTP worker that keeps the knowledge model loaded in memory.

1. First `rlat search` runs the query in-process (cold path).
2. After returning results, it spawns a background worker.
3. Subsequent queries hit the warm worker (~0ms load time).

Worker key includes knowledge model path + modification time. Changed knowledge models invalidate the worker. Workers shut down after 30 minutes idle.

Disable: `--no-worker` flag or `RLAT_NO_WORKER=1` environment variable.

---

## Skill Commands

Skill commands manage knowledge model-backed skills — build, search, inject adaptive context, and inspect quality.

### `rlat skill build <skill-dir>`

Build a knowledge model from a skill's `references/` directory (or paths listed in `knowledge model-sources`).

```bash
rlat skill build .claude/skills/my-skill/
```

Output: `<skill-dir>/knowledge model/<name>.rlat` + `primer.md`

### `rlat skill sync <skill-dir>`

Incrementally update a skill's knowledge model. Skips unchanged files.

```bash
rlat skill sync .claude/skills/my-skill/
```

### `rlat skill search <name> <query> [options]`

Search a skill's knowledge models. Single-knowledge model skills support all formats; multi-knowledge model skills support `text` and `json`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | `text`, `json` (single-knowledge model also: `context`, `prompt`) |

```bash
rlat skill search my-skill "how does auth work?"
```

### `rlat skill inject <name> <query> [options]`

Four-tier adaptive context injection. Returns the **dynamic body** (Tiers 2-4) for a given query. The static Tier 1 header (SKILL.md instructions) is not included in the output — it loads separately through the skill system. Use `--format context` to get the injectable passages for piping into prompts.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | `text` (diagnostic), `json` (structured), `context` (injectable body) |
| `--derived` | str | None | Tier 4 queries, semicolon-separated |

```bash
# Tiers 1-3 only (static + foundational + user query)
rlat skill inject my-skill "create a notebook for REST API ingestion"

# With Tier 4 derived queries
rlat skill inject my-skill "create a notebook for REST API ingestion" \
  --derived "pyspark pagination retry patterns;Delta Lake merge upsert"

# Injectable body only (for piping into prompts)
rlat skill inject my-skill "query" --format context
```

### `rlat skill route <query> [options]`

Rank all knowledge model-backed skills by resonance energy for a query.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-n` | int | 5 | Max skills to return |
| `--format` | choice | text | `text`, `json` |

```bash
rlat skill route "how do I configure autoscaling?"
```

### `rlat skill profile <name>`

Semantic profile of a skill's knowledge model (per-band energy, effective rank, spectral entropy, communities).

### `rlat skill freshness [name]`

Check freshness of skill knowledge models. Omit name to check all.

### `rlat skill gaps <name>`

Detect knowledge gaps in a skill's knowledge model (cluster analysis, robustness).

### `rlat skill compare <skill-a> <skill-b>`

Compare two skills' knowledge models for semantic overlap and unique coverage.

### `rlat skill info [name]`

Show knowledge model configuration and status. Omit name to list all skills.

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `RLAT_VERBOSE=1` | Show encoder loading details, timing breakdown |
| `RLAT_NO_WORKER=1` | Disable background warm worker |
| `NO_COLOR` | Disable ANSI color output |
| `FORCE_COLOR` | Force ANSI color in non-TTY environments |
