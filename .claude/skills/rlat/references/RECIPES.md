# rlat Recipes

Extended recipes for common workflows. Each includes the commands, expected behavior, and useful variations.

## Contents

- [Getting Started](#getting-started) — bootstrap, first build, quick-start
- [Searching](#searching) — format variants, enriched vs. raw, contradictions
- [Keeping Knowledge Models Updated](#keeping-knowledge-models-updated) — sync, refresh, forget
- [Inspection and Diagnostics](#inspection-and-diagnostics) — profile, xray, probe, topology
- [Team and Version Workflows](#team-and-version-workflows) — compare, diff, merge
- [LLM Integration](#llm-integration) — injection modes, primers, custom prompts
- [External Store Workflow](#external-store-workflow) — thin knowledge model + `--source-root`
- [Quick Reference: Common Flag Combinations](#quick-reference-common-flag-combinations)

---

## Getting Started

### Bootstrap a new project

Auto-detect sources, build a knowledge model, and generate an assistant primer in one step:

```bash
rlat init-project
```

This detects `docs/`, `src/`, `lib/`, `README.md`, `CLAUDE.md`, `AGENTS.md` in the current directory, builds `.rlat/project.rlat`, generates `.claude/resonance-context.md`, and prints integration hints.

**Manual equivalent:**

```bash
rlat build ./docs ./src -o .rlat/project.rlat
rlat summary .rlat/project.rlat -o .claude/resonance-context.md
```

**Variations:**
- Custom sources: `rlat init-project ./my-docs ./lib`
- Fast test (no model download): `rlat init-project --encoder random`
- Specify summary output: `rlat init-project -o ./custom-primer.md`

---

### Build with optimization options

For larger corpora, compress the knowledge model and registry:

```bash
rlat build ./docs ./src -o project.rlat \
  --compression zstd \
  --quantize-registry 8
```

**Variations:**
- Maximum compression: `--compression zstd --quantize-registry 4`
- Fastest decompression: `--compression lz4`
- External store (no text in knowledge model): `--store-mode external`
- Lower dimensions for faster queries: `--dim 1024`
- Factored field for memory savings: `--field-type factored`

---

## Searching

### Iterative search with different formats

Start with terminal-readable output, then refine:

```bash
# 1. Human-readable exploration
rlat search project.rlat "how does authentication work?"

# 2. With all enrichment
rlat search project.rlat "auth flow" --cascade --with-contradictions

# 3. JSON for scripting
rlat search project.rlat "auth flow" --format json | jq '.results[:3]'

# 4. Rich markdown for LLM paste
rlat search project.rlat "auth flow" --format prompt --mode augment

# 5. Dense context for automated injection
rlat resonate project.rlat "auth flow" --mode constrain --format context
```

**Variations:**
- More results: `--top-k 20`
- Verbose scores: `-v`
- ONNX speedup: `--onnx ./onnx_backbone/`
- No background worker: `--no-worker`

---

### Multi-hop search

Use subgraph expansion and cascade for queries that span multiple concepts:

```bash
rlat search project.rlat "how does the auth middleware interact with the config system?" \
  --subgraph --subgraph-k 5 \
  --cascade --cascade-depth 2
```

Subgraph expansion adds related neighbours to each result. Cascade discovers related topics.

---

### Find and analyze contradictions

Check if your documentation contradicts itself on a topic:

```bash
# Quick contradiction check
rlat contradictions project.rlat "authentication requirements"

# Lower threshold to catch more subtle conflicts
rlat contradictions project.rlat "config precedence" --threshold 0.3

# Integrated into search
rlat search project.rlat "deployment process" --with-contradictions
```

---

## Keeping Knowledge Models Updated

### Incremental sync after file changes

After editing source files, sync the knowledge model:

```bash
rlat sync project.rlat ./docs ./src
```

Reports: files added, updated, removed, unchanged. Only re-encodes what changed.

**Schedule in a workflow:**

```bash
# In a git hook or CI step:
rlat sync project.rlat ./docs ./src
rlat summary project.rlat -o .claude/resonance-context.md
```

---

### Add new files without touching existing content

```bash
rlat add project.rlat ./new_module/
```

Skips files already present (by content hash). Encoder mismatch triggers a warning.

---

### Remove a specific source

```bash
# Find the source ID
rlat ls project.rlat --grep "deprecated"

# Remove it (exact algebraic inverse)
rlat forget project.rlat --source "deprecated_auth_md"
```

---

## Inspection and Diagnostics

### Full profiling pipeline

```bash
# 1. Semantic shape and coverage
rlat profile project.rlat

# 2. Corpus-level health diagnostics
rlat xray project.rlat

# 3. Deep topological analysis
rlat xray project.rlat --deep --format json

# 4. Signal/noise analysis
rlat probe project.rlat health

# 5. Capacity check
rlat probe project.rlat saturation

# 6. Knowledge gaps
rlat probe project.rlat gaps
```

---

### Check if new content is worth adding

Before adding new documentation, check if the field already covers it:

```bash
rlat probe project.rlat novelty "kubernetes deployment patterns"
```

Returns a 0-1 score with ADD/OPTIONAL/SKIP recommendation.

---

### Understand where a question sits

```bash
rlat locate project.rlat "how do we handle rate limiting?"
```

Reports per-band energy, coverage labels (strong/partial/edge/gap), and an expansion hint pointing to a richer nearby query.

---

### Band flow analysis

See how information couples between the five semantic bands:

```bash
rlat probe project.rlat band-flow
```

Shows how information couples between bands and identifies strongest/weakest band couplings.

---

## Team and Version Workflows

### Compare knowledge model versions

```bash
# Semantic comparison
rlat compare v1.rlat v2.rlat

# Algebraic delta (queryable)
rlat diff v2.rlat v1.rlat -o whats_new.rlat

# Search the delta to see what's new
rlat search whats_new.rlat "new features"
```

---

### Merge team knowledge models

```bash
# Combine two team knowledge models
rlat merge frontend.rlat backend.rlat -o fullstack.rlat

# Inspect overlap
rlat compare frontend.rlat fullstack.rlat

# Three-way merge
rlat merge frontend.rlat backend.rlat -o temp.rlat
rlat merge temp.rlat infra.rlat -o fullstack.rlat
```

Merge is commutative and associative — order doesn't matter.

---

### Privacy-preserving export

Share the semantic model without source text:

```bash
rlat export project.rlat -o shared.rlat --field-only
```

The receiver can run `profile`, `compare`, `locate`, and `probe` against the model, but cannot read original source passages.

---

## LLM Integration

### Grounded context injection

Generate evidence-backed context for any LLM:

```bash
# Zero-hallucination mode
rlat resonate project.rlat "design constraints" --mode constrain --format context

# Augmented (LLM uses own knowledge + cites sources)
rlat search project.rlat "design constraints" --format prompt --mode augment

# Full control with custom prompt
rlat resonate project.rlat "auth" --mode custom \
  --custom-prompt "You are a security auditor. Evaluate these passages for vulnerabilities."
```

---

### Generate session primer

Produce a compressed context document for assistant startup:

```bash
# Default primer
rlat summary project.rlat -o .claude/resonance-context.md

# Custom bootstrap queries
rlat summary project.rlat --queries "architecture;testing;deployment;security" --top-k 30

# Larger budget for deeper primer
rlat summary project.rlat --budget 5000 -o primer.md

# Metadata-only stats
rlat summary project.rlat --format stats
```

---

### Claude Code integration

**Option A: Static primer** (include in CLAUDE.md)

```bash
rlat summary project.rlat -o .claude/resonance-context.md
```

In your `CLAUDE.md`:
```markdown
@.claude/resonance-context.md
```

**Option B: MCP server** (live queries from Claude Code)

```bash
rlat mcp project.rlat
```

In `.claude/settings.json` or `~/.claude/settings.json`:
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

### HTTP server for app integration

```bash
rlat serve project.rlat --port 8080
```

Query via REST:
```bash
# Basic search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"text": "how does auth work", "top_k": 10}'

# With contradiction detection
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"text": "auth", "top_k": 10, "enable_contradictions": true}'
```

---

## External Store Workflow

Build a lightweight knowledge model that reads evidence from local files at query time:

```bash
# Build with external store
rlat build ./docs ./src -o project.rlat --store-mode external

# Query with source resolution
rlat search project.rlat "auth flow" --source-root .

# Generate context with source resolution
rlat resonate project.rlat "auth flow" --source-root . --mode constrain
```

Useful for: reducing knowledge model size, avoiding embedded PII, keeping source text always up-to-date.

---

## Quick Reference: Common Flag Combinations

```bash
# Fast local test (no model download)
rlat build ./docs -o test.rlat --encoder random

# Production build (compressed + quantized)
rlat build ./docs ./src -o project.rlat --compression zstd --quantize-registry 8

# Full enriched search
rlat search project.rlat "query" --cascade --with-contradictions --format prompt

# Non-interactive search (for scripts/subagents)
rlat search project.rlat "query" --format json --no-worker

# LLM grounding (zero hallucination)
rlat resonate project.rlat "query" --mode constrain --format context

# Complete diagnostic suite
rlat profile project.rlat && rlat xray project.rlat --deep && rlat probe project.rlat health
```
