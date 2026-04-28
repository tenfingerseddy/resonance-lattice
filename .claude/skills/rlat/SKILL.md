---
name: rlat
description: >-
  Orchestrate rlat CLI workflows on the user's behalf for any task involving
  an rlat knowledge model (.rlat file). Triggers cover: setting up rlat for a
  fresh project (init), building or refreshing a knowledge model on drift,
  running grounded retrieval to ground an answer in citations, generating
  skill-context for an Anthropic skill !command block, recalling past
  decisions from memory, comparing two corpora, switching storage modes
  (bundled / local / remote), and the opt-in MRL optimise pipeline. Trigger
  when the user mentions rlat by name, references a .rlat file, asks to
  build/query/refresh/sync a knowledge model, asks "what does the project
  know about X" on a project with a .rlat file, or describes any task that
  maps onto an rlat workflow. This skill executes the workflow on the
  user's behalf — it does not just document the CLI. For multi-hop research
  with cross-file synthesis (plan → retrieve → refine → synthesize loop),
  defer to the deep-research skill instead. Not for: Pinecone, Chroma,
  FAISS, or other vector databases; raw embedding ops without rlat.
allowed-tools: Bash(rlat:*), Read, Write, Edit, Glob, Grep
---

# rlat — workflow orchestration

You are running an `rlat` workflow on the user's behalf. The CLI is
deterministic; your job is to (1) identify which workflow fits the user's
request, (2) execute it with the right flags for their context, and (3)
surface what changed and what's next.

## Step 1: Identify the workflow

| Signal in the user's request | Workflow | Section |
|---|---|---|
| "set up rlat", "index this codebase", project has no `.rlat` file yet | **Init** | [§ Init](#init) |
| Source files changed since build, drift detected on a search, build is >3 days old | **Refresh** | [§ Refresh](#refresh) |
| Single-hop fact question about an indexed corpus ("what does X do?", "where is Y defined?") | **Search** | [§ Search](#search) |
| Cross-file synthesis question ("why was X chosen?", "trade-offs across Y") | **Defer** to the `deep-research` skill |
| Authoring a Claude skill that needs corpus context | **Skill-context** | [§ Skill-context](#skill-context) |
| "have we discussed X?", "what did we decide about Y?", "what was the rationale we agreed on?" | **Memory recall** | [§ Memory](#memory) |
| User wants to compare two `.rlat` files | **Compare** | [§ Compare](#compare) |
| User wants a `.rlat` to be self-contained / on disk / HTTP-served | **Convert** | [§ Convert](#convert) |
| User wants per-corpus retrieval lift (opt-in, costs API + GPU time) | **Optimise** | [§ Optimise](#optimise) |
| User asks how `rlat` works conceptually | Read [docs/user/CLI.md](../../../docs/user/CLI.md) and answer; do not run a workflow |

## Step 2: Pre-flight (before any workflow that builds or runs)

- **Encoder installed?** First-time builds need a one-shot encoder install (~2 min on CPU). If `rlat build` or `rlat search` errors with "encoder not found", run `rlat install-encoder` once and tell the user it's a one-time setup (cached at the standard XDG cache location).
- **Find the right `.rlat` file.** Use `Glob "*.rlat"` to discover. If multiple, prefer the one whose name matches the project (`resonance-lattice.rlat`, `project.rlat`) or ask the user.
- **Check freshness** when the workflow assumes a current build: `rlat profile <km>.rlat` reports drift count.

If a precondition fails, fix it first or surface it clearly to the user before running the workflow.

---

## Init

**When**: project has no `.rlat` file or the user explicitly asks to set up rlat for this project.

**Run**:
```bash
rlat init-project
```

Auto-detects `docs/`, `src/`, top-level `*.md`/`*.rst`/`*.txt`. Builds the knowledge model and generates `.claude/resonance-context.md` (the project primer) in one command.

For non-default source paths:
```bash
rlat init-project --source ./<dir1> --source ./<dir2>
```

**After**:
- Verify the `.rlat` and `.claude/resonance-context.md` both exist.
- Tell the user the next-step paths: single-shot retrieval is now `rlat search ./<km>.rlat "<query>"`; for multi-hop research, the `deep-research` skill takes over.

---

## Refresh

**When**: source files changed since the last build, drift detected on a recent search, or `rlat profile` reports any non-zero drift count.

**Run** (local-mode KMs):
```bash
rlat refresh <km>.rlat
```

Atomic. Re-encodes only the changed passages; preserves any existing optimised band by reprojecting it from the new base for free. If the optimised band would be invalidated by the changes, the command aborts unless `--discard-optimised` is passed — surface this to the user before deciding.

**Run** (remote-mode KMs):
```bash
rlat sync <km>.rlat
```

Same shape; fetches the manifest delta over HTTP first.

**After**:
- Run `rlat profile <km>.rlat` — drift count should be 0.
- If a primer was generated from this KM, regenerate: `rlat summary <km>.rlat -o .claude/resonance-context.md`.

---

## Search

**When**: user has a one-shot question whose answer plausibly lives in the indexed corpus.

**Run**:
```bash
rlat search <km>.rlat "<question>" --top-k 5 --format context --mode <mode>
```

**Pick the mode** based on the user's risk tolerance:
- `--mode augment` (default) — passages as primary context, blended with training. **The right pick for general-purpose use.**
- `--mode constrain` — passages are the only source of truth, refuse on thin evidence. **Pick for compliance / audit / regulatory work** where wrong-but-confident is unacceptable. Bench: 91.7% distractor refusal at 2.0% answerable hallucination — the safety floor.
- `--mode knowledge` — passages supplement training. Pick when the LLM has solid base domain knowledge.

Add `--strict-names` when the question references specific proper nouns / acronyms / SKUs and a name-mismatch should refuse rather than silently substitute.

**After**:
- Use the returned passages as the basis for your answer; cite source paths inline (`docs/foo.md`, `src/bar.py`).
- If retrieval coverage looks thin (top-1 score < 0.30, all hits drifted), surface the gap honestly; don't extrapolate.

---

## Skill-context

**When**: user is authoring an Anthropic skill that needs corpus-grounded context injected via `!command` blocks.

**Run** (preset + dynamic-query pattern):
```bash
rlat skill-context <km>.rlat \
  --query "<skill-author preset>" \
  --query "$user_query" \
  --top-k 5 --token-budget 4000
```

Multi-query order matters: skill-author preset first, `$user_query` last. When the token budget runs out, later queries drop first; the first preset always survives.

**Compliance variant** (drift + name-mismatch are hard refusals):
```bash
rlat skill-context <km>.rlat --strict --strict-names --mode constrain \
  --query "$user_query"
```

`--strict` aborts on drift (rc=2). `--strict-names` aborts on name-mismatch (rc=3). The skill loader gates on the exit code.

---

## Memory

**When**: user asks about past sessions, prior decisions, recurring themes — *"have we discussed X before?"*, *"what did we decide about Y?"*, *"what was the rationale agreed for Z?"*.

The memory directory is selected with `--memory-root` on the parent `memory` command. Default: `./memory/` (project-local). For Claude Code session memory, use `.claude/memory`.

**Run** (recall):
```bash
rlat memory --memory-root ".claude/memory" recall "<query>"
```

Returns ranked memory entries fused across the working / episodic / semantic tiers (recency-weighted retention; cross-session knowledge survives in the semantic tier).

**Add a memory** (when the user wants to record a decision for future sessions):
```bash
rlat memory --memory-root ".claude/memory" add "<text>" --tier semantic
```

**Generate a session-start primer** from the memory state:
```bash
rlat memory --memory-root ".claude/memory" primer -o .claude/memory-primer.md
```

**Consolidate** episodic near-duplicates into the semantic tier:
```bash
rlat memory --memory-root ".claude/memory" consolidate
```

**Garbage-collect** entries past their retention half-life:
```bash
rlat memory --memory-root ".claude/memory" gc
```

---

## Compare

**When**: user wants to compare two corpora — typically two snapshots of the same project, or two related products.

**Run**:
```bash
rlat compare a.rlat b.rlat
```

Always uses the base band (cross-knowledge-model rule). Outputs centroid cosine + asymmetric mutual coverage (both directions).

**For deeper analysis** of a single KM (cross-passage relationships, contradictions, evidence per claim, drift detection, corpus diff across snapshots), use the RQL ops via the Python API:

```python
from resonance_lattice.store import archive
from resonance_lattice.rql import evidence, contradictions, corpus_diff, drift

contents_a = archive.read("a.rlat")
contents_b = archive.read("b.rlat")

# Evidence + ConfidenceMetrics for a claim:
evidence(contents_a, "<claim>")
# Within-KM stale passages:
drift(contents_a)
# Added / removed / unchanged across two snapshots:
corpus_diff(contents_a, contents_b)
# High-similarity, low-Jaccard pair candidates (experimental — heuristic):
contradictions(contents_a, "<topic>")
```

These ops are Python-only in v2.0; CLI verbs for them are not in scope.

---

## Convert

**When**: user wants to switch a `.rlat` between storage modes (`bundled` / `local` / `remote`).

**Run**:
```bash
rlat convert <km>.rlat --to <mode>
# remote requires:
rlat convert <km>.rlat --to remote --remote-url-base <url>
```

In place; embeddings preserved (semantically identical pre/post). Conversion validates every kept passage's `content_hash` against live bytes; if any have drifted, conversion aborts. Run `rlat refresh` (local) or `rlat sync` (remote) first if so.

---

## Optimise (opt-in, requires Anthropic API key)

> ⚠ **Cost**: ~$2-8 in Anthropic API calls + ~20-50 min Kaggle T4 GPU time per corpus. Two of three publicly-tested corpora regressed (BEIR fiqa: -0.042 nDCG@10, BEIR nfcorpus: -0.043). Lift correlates with synth-query/test-query distribution alignment, not "natural-language vs keyword" surface form. Per-corpus decision; run `--estimate` first.

**When**: user explicitly asks for the optimised band, OR retrieval recall is the bottleneck on a natural-language Q&A corpus where this can pay off.

**Run**:
```bash
export CLAUDE_API=sk-ant-...                        # or ANTHROPIC_API_KEY
rlat optimise <km>.rlat --estimate                  # cost preview FIRST
rlat optimise <km>.rlat --corpus-description "<one-line corpus description>"
```

The corpus description conditions the synth-query generator. Without it, lift drops materially — always pass it for non-trivial corpora.

API key setup if missing: see [docs/user/API_KEYS.md](../../../docs/user/API_KEYS.md).

---

## Programmatic deep-search (CLI verb, requires API key)

The `rlat deep-search` CLI verb runs the same multi-hop research loop the
`deep-research` skill drives interactively, but as a standalone CLI command
suitable for non-Claude-Code agents, CI pipelines, batch scripts, or any
consumer that needs a programmatic surface with structured output.

> ⚠ **Requires an Anthropic API key** (~$0.009-0.025/q). For interactive
> Claude Code research, the `deep-research` skill drives the same loop with
> no API cost — prefer that skill in nearly every Claude Code scenario.

**When to invoke this CLI verb instead of the skill:**
- The deliverable is a machine-readable artefact (JSON, markdown for piping into another LLM, file output)
- The user is authoring a CI pipeline, agent, or batch script
- The user explicitly asks for `rlat deep-search`

**Run**:
```bash
rlat deep-search <km>.rlat "<question>" \
  [--max-hops 4] [--top-k 5] \
  [--format text|json|markdown] \
  [--strict-names]
```

Exit codes: `0` success, `2` no-answer, `3` name-mismatch refusal.

---

## When NOT to use this skill

- Question is answerable from training alone (no `.rlat` involved). Just answer.
- User is asking about Pinecone, Chroma, FAISS, or other vector DBs — different products.
- User wants to read or edit specific source files. Use `Read` / `Edit` / `Grep` directly.
- Question requires multi-hop synthesis across files — defer to the `deep-research` skill.
- User just wants a one-line answer that doesn't need grounding — answer directly without the retrieval round-trip.

## Reference

- [docs/user/CLI.md](../../../docs/user/CLI.md) — full CLI reference (every command, every flag — the canonical source)
- [docs/user/CORE_FEATURES.md](../../../docs/user/CORE_FEATURES.md) — seven things rlat enables
- [docs/user/STORAGE_MODES.md](../../../docs/user/STORAGE_MODES.md) — bundled / local / remote decision guide
- [docs/user/API_KEYS.md](../../../docs/user/API_KEYS.md) — only for `optimise` and the `deep-search` CLI verb
- [docs/user/FAQ.md](../../../docs/user/FAQ.md) — common questions including licence
