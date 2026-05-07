# Getting Started

This walks you through the first build and the first query in under 15 minutes. By the end you'll have a `.rlat` knowledge model of your project and a context primer that AI assistants can read.

## Install

`rlat` requires **Python 3.12+**. The base install is small (~250 MB) and CPU-only — only the build path needs the heavyweight ML extras.

```bash
pip install rlat[build]
```

The `[build]` extra pulls in `transformers`, `torch`, and `onnxscript`. Once you have a `.rlat`, query-time only needs the base install — you can `pip install rlat` (no extras) on a different machine and search a knowledge model someone else built.

> If you only ever search and never build, use the base install and have someone else build the `.rlat` for you.

## Don't want to build yet? Try a prebuilt `.rlat`

Four launch-day knowledge models live on HuggingFace, ready to query without any encoder install or build step. They use **remote storage mode** — the .rlat itself only contains embeddings + source coordinates + a SHA-pinned URL manifest; the source files are fetched on demand from `raw.githubusercontent.com` at the pinned commit and SHA-verified locally.

| Dataset | Source repo | Files | Passages | Size |
|---|---|---:|---:|---:|
| [`tenfingers/powerbi-developer-rlat`](https://huggingface.co/datasets/tenfingers/powerbi-developer-rlat) | [MicrosoftDocs/powerbi-docs](https://github.com/MicrosoftDocs/powerbi-docs) `powerbi-docs/developer` | 176 | 5,684 | 34 MB |
| [`tenfingers/powershell-docs-rlat`](https://huggingface.co/datasets/tenfingers/powershell-docs-rlat) | [MicrosoftDocs/PowerShell-Docs](https://github.com/MicrosoftDocs/PowerShell-Docs) `reference` | 2,647 | 107,033 | 640 MB |
| [`tenfingers/python-stdlib-rlat`](https://huggingface.co/datasets/tenfingers/python-stdlib-rlat) | [python/cpython](https://github.com/python/cpython) `Doc` | 617 | 49,179 | 293 MB |
| [`tenfingers/tsql-docs-rlat`](https://huggingface.co/datasets/tenfingers/tsql-docs-rlat) | [MicrosoftDocs/sql-docs](https://github.com/MicrosoftDocs/sql-docs) `docs/t-sql` | 1,209 | 33,282 | 198 MB |

```bash
# pip install rlat (base install is enough — no [build] extras needed for query)
pip install rlat huggingface_hub

# Pick the corpus that fits your work
huggingface-cli download tenfingers/python-stdlib-rlat python-stdlib.rlat --local-dir .

# Query it — first hit caches the cited source files locally; subsequent queries are sub-20ms warm
rlat search python-stdlib.rlat "asyncio Task cancellation" --top-k 5
```

All four are encoded with the same `gte-modernbert-base` 768d recipe documented in [BENCHMARK_GATE.md](../internal/BENCHMARK_GATE.md), so retrieval quality is identical to what you'd get building locally. The `.rlat` archive itself is the source of truth for the source-repo SHA — the README on each HuggingFace repo records it and the build date.

If you outgrow the prebuilt set or want to index your own project, keep reading.

## Stage the encoder (one-time)

The first `rlat build` will trigger this automatically. To pre-stage (offline-build environments, CI):

```bash
rlat install-encoder
```

This downloads `Alibaba-NLP/gte-modernbert-base` from Hugging Face, exports it to ONNX, and (on Intel CPUs) converts it to OpenVINO IR. Cache lives at `~/.cache/rlat/encoders/<revision>/`.

## Build your first knowledge model

Two ways. Pick one.

### Option A — `rlat init-project`

Run this in the root of a project that has `docs/`, `src/`, or top-level Markdown files:

```bash
cd my-project/
rlat init-project
```

It auto-detects sources, builds `<cwd-name>.rlat` in the current directory, and writes a context primer to `.claude/resonance-context.md` for AI-assistant integration.

```
[init] detected sources:
  - docs
  - src
  - README.md
[init] output: my-project.rlat
[build] walking sources rooted at .
[build] 14 files; chunking …
[build] 39 passages; encoding (runtime=torch, batch=32) …
[build] wrote my-project.rlat (1.20 MB, 39 passages from 14 files)
[summary] wrote .claude/resonance-context.md (1856 chars, 464 tokens approx)
```

### Option B — `rlat build` (full control)

```bash
rlat build ./docs ./src -o my-project.rlat \
  --store-mode local \
  --kind corpus
```

Three storage modes (`local` is the default) — see [STORAGE_MODES.md](./STORAGE_MODES.md) for the trade-offs.

> **Big corpus, no local GPU?** CPU-only encoding of `gte-modernbert-base` runs at ~80-150 passages/sec; corpora over ~10K passages take an unpleasant amount of time. The [`rlat-build-on-kaggle`](../../.claude/skills/rlat-build-on-kaggle/SKILL.md) Claude skill walks through using Kaggle's free T4 GPU instead — account setup, kernel push, polling, and pulling the `.rlat` back. Just ask "can we build this on Kaggle?" inside Claude Code.

## Run your first query

```bash
rlat search my-project.rlat "how does retrieval work"
```

Output:

```
0.836  docs/architecture.md:84+12  [verified]  ## Indexing
0.820  docs/architecture.md:12+73  [verified]  This project does dense retrieval with cosine similarity over unit-norm embeddings.
0.805  src/retrieval.py:0+50  [verified]  def build_index(passages):  return faiss.HNSW(passages)
...
```

Three columns: cosine score, source coordinate, drift status, preview. The status is per-passage — if you edit `architecture.md`, that file's hits will show `drifted` until you `rlat refresh`.

For machine-readable JSON or LLM-ready context blocks:

```bash
rlat search my-project.rlat "..." --format json
rlat search my-project.rlat "..." --format context --top-k 5
```

## Hand the corpus to your assistant

Two integrations.

### Static primer

`rlat init-project` already wrote `.claude/resonance-context.md`. That's a markdown document with three sections (Landscape / Structure / Evidence) that you paste into your assistant's system prompt. Regenerate after meaningful corpus changes:

```bash
rlat summary my-project.rlat -o .claude/resonance-context.md
```

## Keep things fresh

When you edit source files, the on-disk content drifts from what was indexed. Three recovery paths:

```bash
# Live: silent watch loop — debounced refresh on every save. Run once, forget about it.
pip install rlat[watch]            # one-time
rlat watch                         # zero-arg auto-discovers *.rlat in cwd

# Manual: re-ingest changed files in place. Atomic — old archive intact until new write succeeds.
rlat refresh my-project.rlat

# Query-time guard: filter to only verified hits
rlat search my-project.rlat "..." --verified-only
```

`rlat watch` runs the same incremental delta-apply pipeline as `rlat refresh`, but on a debounce timer triggered by filesystem events. Default UX is silent — startup confirms what's being watched, then the loop falls quiet. Errors are loud. On `Ctrl-C`, prints a one-line summary. `--once` is the CI / pre-commit shape.

`rlat refresh` re-runs the build pipeline against the source paths recorded at build time. The chunker constants (min/max chars) and kind tag come from `metadata.build_config` so passage_idx layout stays the same.

## Inspect a knowledge model

```bash
# Human-readable summary
rlat profile my-project.rlat

# JSON for scripts / dashboards
rlat profile my-project.rlat --format json

# Skip the drift walk on huge corpora
rlat profile big-corpus.rlat --no-drift
```

## Compare two corpora

```bash
rlat compare project-old.rlat project-new.rlat
```

Reports centroid_cosine (single thematic-alignment number) and asymmetric mutual coverage. Always uses the **base band** so two corpora built with different optimiseds still compare correctly.

## What's next

You now have the day-one workflow. From here:

- **CLI reference**: [CLI.md](./CLI.md) — every flag for every subcommand.
- **Storage modes deep-dive**: [STORAGE_MODES.md](./STORAGE_MODES.md) — pick `bundled` if you want to ship the source inside the `.rlat`, or `remote` if you want consumers to pull source from upstream URLs.
- **Encoder details**: [ENCODER.md](./ENCODER.md) — single-recipe explainer.
- **Internal architecture**: [ARCHITECTURE.md](../internal/ARCHITECTURE.md) — three-layer thesis, module map.
- **Honest claims about retrieval quality**: [BENCHMARK_GATE.md](../internal/BENCHMARK_GATE.md) — locked floor + published-vs-measured comparison.

## Where things go wrong

- **`rlat: command not found`** — pip install didn't drop the script on PATH. Re-run with `pip install -e .[build]` if you're working from source, or check `python -m resonance_lattice.cli.app --help` works.
- **`error: <path> is not a valid v4 knowledge model`** — the file is corrupted, was built by a pre-v2 rlat, or isn't a `.rlat` at all. Check `file <path>` to confirm it's a ZIP archive.
- **`error: local-mode knowledge model has no recorded source_root`** — the model was built without recording where its sources came from. Pass `--source-root <dir>` explicitly to override.
- **First query is slow (~1 second)** — encoder cold-start. Subsequent queries in the same Python process are sub-20ms warm.
- **A search hit shows `drifted`** — the source file has been edited since build. Run `rlat refresh` to re-ingest.
- **A search hit shows `missing`** — the source file has been deleted. `rlat refresh` will rebuild without it.
