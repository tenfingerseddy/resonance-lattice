---
title: Getting Started
slug: getting-started
description: Install Resonance Lattice, build your first knowledge model, and choose the right interface for the job.
nav_group: Foundations
nav_order: 20
aliases:
---

# Getting Started

## What Is This?

This guide gets a new user from install to a working knowledge model. It is the fastest path to understanding the product shape before you go deeper into the architecture or reference docs.

## Why Start Here?

The main mental model of Resonance Lattice becomes clear only after you build and query a knowledge model once. A short first-run loop makes the later architecture and reference material easier to place.

## How Does The First Workflow Work?

The first workflow has four steps:

1. install the package and optional fast path
2. build a knowledge model from docs, code, or both
3. query or profile it
4. choose the interface you want to keep using

## How Do I Use It?

### Install

```bash
pip install resonance-lattice
pip install onnxruntime  # optional CPU acceleration
```

Python `>=3.11` is required. The first build downloads the default encoder (`BAAI/bge-large-en-v1.5`, ~1.3 GB) unless you pass `--encoder` to pick another. Three encoders are well-supported: BGE (default, portable), E5 (opt-in — stronger on counter-argument corpora), Qwen3-8B (opt-in — frontier quality, needs 16 GB GPU). See [Encoder Choice](/docs/encoder-choice) for the decision guide.

### Use A Prebuilt Launch Knowledge Model (Skip The Build)

If your corpus is one we publish — Microsoft Fabric, Power BI Developer, PowerShell, Python stdlib, or T-SQL — download the prebuilt `.rlat` from HuggingFace Hub and start querying immediately.

| Corpus | Audience | HF repo |
|---|---|---|
| Microsoft Fabric docs | Data engineers, Fabric users | [tenfingers/fabric-docs-rlat](https://huggingface.co/datasets/tenfingers/fabric-docs-rlat) |
| Power BI Developer docs | App developers embedding Power BI | [tenfingers/powerbi-developer-rlat](https://huggingface.co/datasets/tenfingers/powerbi-developer-rlat) |
| PowerShell docs | Windows admins, DevOps, scripting | [tenfingers/powershell-docs-rlat](https://huggingface.co/datasets/tenfingers/powershell-docs-rlat) |
| Python standard library docs | Python developers | [tenfingers/python-stdlib-rlat](https://huggingface.co/datasets/tenfingers/python-stdlib-rlat) |
| T-SQL docs | SQL Server / Azure SQL / Fabric Warehouse | [tenfingers/tsql-docs-rlat](https://huggingface.co/datasets/tenfingers/tsql-docs-rlat) |

Every repo ships **two variants** — identical retrieval quality (same field, same registry, same encoder), different source-serving backend. The storage-mode choice is about *where the source bytes live at query time*, not about retrieval behaviour.

#### Path 1 — Keep it bundled (offline-ready, recommended for most users)

```bash
huggingface-cli download tenfingers/fabric-docs-rlat fabric-docs-bundled.rlat --local-dir .
rlat search fabric-docs-bundled.rlat "how do I create a lakehouse"
```

The `-bundled.rlat` variant packs upstream source files inside the knowledge model as zstd frames. **No network after download**, works offline forever, pinned to a specific upstream commit SHA. Roughly 2× the size of the remote variant (137 MB – 737 MB depending on corpus) but no cold-fetch surprises in CI, demos, or air-gapped environments. Pick this when reliability matters more than freshness.

#### Path 2 — Switch to the repo-backed variant (track upstream, smaller download)

```bash
huggingface-cli download tenfingers/fabric-docs-rlat fabric-docs.rlat --local-dir .
rlat search fabric-docs.rlat "how do I create a lakehouse"

# Check for upstream drift (read-only, one GitHub API call):
rlat freshness fabric-docs.rlat

# Pull the diff into your local knowledge model (lockfile-style):
rlat sync fabric-docs.rlat
```

The bare `.rlat` (no `-bundled`) is pinned to a GitHub commit SHA and fetches source text on the first query touching each file — cached locally under `~/.cache/rlat/remote/` thereafter. Smaller download (120 MB – 340 MB), but queries need network on cold cache, and you choose when to pick up upstream changes via `rlat sync`. Pick this when you want automatic freshness and don't mind the first-query fetch latency.

**Transitioning bundled → remote isn't supported** — bundled blobs are immutable. Download the remote variant directly instead.

#### Path 3 — Repoint to a local synced folder (vendoring, offline mirror, source edits)

If you want to serve queries from your own checkout of the upstream repo — for vendoring, air-gapped internal mirrors, or when you want to edit docs locally and see retrieval against your edits — start from the remote variant, repoint it to local mode, and point `--source-root` at your clone:

```bash
# 1. Clone the upstream repo (or sync your existing fork):
git clone https://github.com/MicrosoftDocs/fabric-docs ./fabric-docs-clone

# 2. Repoint the knowledge model from remote to local (strips __remote_origin__,
#    keeps field + registry + manifest unchanged — takes seconds, no re-encoding):
rlat repoint fabric-docs.rlat --to local

# 3. Query against your clone:
rlat search fabric-docs.rlat "lakehouse" --source-root ./fabric-docs-clone
```

Handy for scoped rlats too — `python-stdlib.rlat` is scoped to CPython's `Doc/` subdirectory, so `--source-root` should point at `./cpython-clone/Doc`. The manifest paths match the scope root, not the repo root.

When upstream drifts and you want to re-sync chunks, `rlat refresh fabric-docs.rlat --source-root ./fabric-docs-clone` re-encodes only the files whose content hash changed. Or repoint back to `remote` (`rlat repoint fabric-docs.rlat --to remote --url https://github.com/MicrosoftDocs/fabric-docs`) and let `rlat sync` do the diff-based upgrade.

See [Storage Modes](/docs/storage-modes) for the full comparison table.

### Build Your First Knowledge Model

If your corpus isn't in the prebuilt list above, build one yourself:

```bash
rlat build ./docs ./src -o project.rlat
```

This chunks your sources, encodes them, and writes a knowledge model with the semantic field, source registry, and lossless store. The default `--store-mode local` keeps the knowledge model thin and resolves source files from disk at query time. Use `--store-mode bundled` to pack source files inside the `.rlat` for a self-contained artifact, or build from a GitHub URL for `--store-mode remote` with SHA-pinned upstream tracking. For a large monorepo where only one subdirectory matters (e.g. just the `Doc/` folder of CPython), pass `--path SUBDIR` to scope the build. See [Storage Modes](/docs/storage-modes) for the full comparison.

### Search, Ask, Profile, And Compare

```bash
rlat search project.rlat "how does authentication work?"
rlat ask project.rlat "how does authentication work?"
rlat profile project.rlat
rlat compare baseline.rlat project.rlat
```

Use `search` for grounded evidence, `ask` when you want the auto-lens dispatcher to pick the best retrieval mode for the question, `profile` for knowledge model shape, and `compare` when you need overlap and drift between snapshots.

### Generate Assistant Primers

```bash
rlat summary project.rlat -o .claude/resonance-context.md
rlat memory primer ./memory/ -o .claude/memory-primer.md
```

The dual primer system produces two complementary context documents: a **code primer** (what the project IS — structure, conventions, patterns) and a **memory primer** (how the work has unfolded — settled decisions, reversals, active threads). They de-duplicate against each other so an assistant does not pay twice for the same fact. Reference both from `CLAUDE.md` or your equivalent system prompt.

### Wire Up Claude Code (Skill + Hooks)

Resonance Lattice ships a first-class Claude Code integration — a **skill** that gives Claude direct access to `rlat` commands, plus **two lifecycle hooks** that keep your primer fresh and ingest each session into persistent memory. The files live at `.claude/skills/rlat/` in this repo; drop them into a consumer project to activate.

#### What you get

- **`rlat` skill** (`.claude/skills/rlat/SKILL.md`) — auto-triggers when Claude sees questions about your corpus ("how does X work", "why did we choose Y", "find the benchmark for Z"). The skill teaches Claude to use `rlat search` as its primary research tool for semantic / cross-file questions, and reserves `grep` for exact-symbol lookup. It also loads a dynamic slice of `project.rlat` into context at skill-invocation time so Claude has corpus-grounded background before it answers.

- **SessionStart hook** (`.claude/skills/rlat/hooks/session-start.py`) — runs when Claude Code opens a session. Warns if any `*.rlat` knowledge model in the project root is older than `RLAT_STALE_HOURS` (default 72h) so Claude knows to rebuild before trusting results, and fires a background `rlat primer refresh` if the code primer (`.claude/resonance-context.md`) is stale or git-diverged. The current session never blocks on rebuild; the next session picks up the regenerated primer.

- **SessionEnd hook** (`.claude/skills/rlat/hooks/memory-session-end.py`) — runs when a Claude Code session closes. Reads the session transcript and calls `rlat memory write` to append it to the project's layered memory store (working tier). First use auto-initializes the memory root at `./memory/`. Subsequent sessions recall prior context via the `memory-primer.md` generated in the previous step.

Both hooks are **fail-soft**: exceptions go to stderr and the hook exits 0, so a broken rlat install never blocks Claude Code from opening or closing a session.

#### Install into a consumer project

From inside the repo (or any project where you want rlat-aware Claude Code):

```bash
# 1. Copy the skill + hooks into the project's .claude/ directory.
#    If cloning this repo, they are already there.
cp -r .claude/skills/rlat /your/project/.claude/skills/

# 2. Merge the hook registration into .claude/settings.json.
#    The example file shows the exact snippet:
cat .claude/skills/rlat/hooks/settings.example.json
```

The `settings.example.json` block registers the two hooks via `${CLAUDE_PROJECT_DIR}` so it works from any working directory:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "python \"${CLAUDE_PROJECT_DIR}/.claude/skills/rlat/hooks/session-start.py\""
      }]
    }],
    "SessionEnd": [{
      "hooks": [{
        "type": "command",
        "command": "python \"${CLAUDE_PROJECT_DIR}/.claude/skills/rlat/hooks/memory-session-end.py\""
      }]
    }]
  }
}
```

Merge this into your existing `.claude/settings.json` under the `hooks` key. Then build `project.rlat` once (`rlat build ./docs ./src -o project.rlat`) so the skill has a knowledge model to search — the skill frontmatter declares `knowledge models: [project.rlat]` and expects it in the project root.

#### Optional environment controls

All env vars are optional; sensible defaults work for most projects.

| Variable | Purpose | Default |
|---|---|---|
| `RLAT_STALE_HOURS` | Knowledge-model freshness threshold (SessionStart warning) | `72` |
| `RLAT_MEMORY_ROOT` | Override memory root path | `<cwd>/memory` |
| `RLAT_OPENVINO_DIR` | OpenVINO IR directory for fast encoding on Intel Arc iGPU / NPU | unset (PyTorch encode) |
| `RLAT_OPENVINO_DEVICE` | `CPU` \| `GPU` \| `NPU` \| `AUTO` | `AUTO` when `RLAT_OPENVINO_DIR` is set |
| `RLAT_ONNX_DIR` | ONNX backbone directory (alternative accelerator, Intel / NVIDIA CPUs) | unset (PyTorch encode) |
| `RLAT_REBUILD_PRIMER` | Set to `1` to regenerate the memory primer after each SessionEnd ingest | unset |
| `RLAT_CODE_CARTRIDGE` | Code-knowledge-model path passed to primer regen for cross-primer novelty filtering | unset |

For on-device acceleration on Intel Arc, install the fast extra and export the backbone once:

```bash
pip install 'resonance-lattice[fast]'
rlat export-onnx ./onnx_backbone/         # ONNX path
# or
rlat export-openvino ./openvino_backbone/ # OpenVINO path (Arc iGPU / NPU)
export RLAT_OPENVINO_DIR=./openvino_backbone/
```

Encoding latency on the SessionEnd ingest drops 2–5× vs pure PyTorch CPU.

#### Verify the wiring

Open a Claude Code session in the project. You should see a SessionStart notification in the conversation transcript only if there is something to flag (stale knowledge model or primer). On session close, `./memory/` gets a new `working/` entry — inspect it with:

```bash
rlat memory profile ./memory/
```

If the hook didn't fire, the failure is silent by design (fail-soft). Check Claude Code's hook-error log or run the hook script manually to surface the exception:

```bash
echo '{"transcript_path": "/tmp/fake.jsonl"}' | python .claude/skills/rlat/hooks/memory-session-end.py
```

### Choose CLI vs MCP vs HTTP

#### CLI

Use the CLI when you want direct terminal workflows, scripts, or local automation.

#### MCP

Use MCP when your assistant supports native tool calling and you want the knowledge model to stay warm in the conversation.

#### HTTP

Use the local HTTP server when another app or plugin needs a simple request/response interface.

## What Should I Do Next?

- Read [Knowledge Model Architecture](/docs/knowledge-model-architecture) to understand the artifact itself.
- Read [Storage Modes](/docs/storage-modes) to pick between `local` (default), `bundled`, and `remote`.
- Read [Encoder Choice](/docs/encoder-choice) before switching off the default BGE encoder.
- Read [CLI](/docs/cli) if terminal workflows are primary.
- Read [MCP](/docs/mcp) if assistant-native integration is primary.
- Read [Semantic Model](/docs/semantic-model) if you want to understand retrieval quality, coverage, and limits.

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
