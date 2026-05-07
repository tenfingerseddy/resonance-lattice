<p align="center">
  <img src="docs/assets/lockup-horizontal.svg" alt="Resonance Lattice" width="640">
</p>

# Resonance Lattice (`rlat`)

[![PyPI](https://img.shields.io/pypi/v/rlat?label=pypi%20rlat)](https://pypi.org/project/rlat/)
[![Python](https://img.shields.io/pypi/pyversions/rlat)](https://pypi.org/project/rlat/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)](LICENSE.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20datasets-tenfingers-yellow)](https://huggingface.co/tenfingers)

> **Give your AI assistant a local, citeable, drift-checked knowledge model of your docs, code, and notes.** One file you own. Every passage cited. No hosted index, no LLM in the retrieval loop.

`rlat` packages a corpus — your codebase, your documentation, your research notes — into a single `.rlat` file. Where a semantic data model captures the meaning of a database, a knowledge model captures the semantic shape of unstructured text plus every coordinate needed to find, cite, and verify the underlying source. You query it with a CLI command and feed the results to whichever AI assistant you're already using (Claude Code, Cursor, Aider, Continue, your own agent).

## What it does

- **One `.rlat` file you own.** Embeddings + source coordinates + content hashes — all in one ZIP. Open it with `unzip`, inspect the registry with `jq`. No hosted index, no proprietary binary, no vendor lock-in.
- **Every passage carries provenance.** Each result is `(text, source_file, char_offset, char_length, content_hash, drift_status)`. Citation is free. Drift detection is free. Refusal on stale evidence is one flag away.
- **No LLM in the retrieval loop.** `rlat search`, `rlat profile`, `rlat compare`, `rlat summary`, `rlat refresh`, `rlat watch`, `rlat skill-context`, `rlat memory`, and the entire RQL surface need no API key and no model call. After a one-time `rlat install-encoder` (downloads the 768d encoder), local-mode build/search workflows run offline. Remote-mode `rlat sync` and `rlat freshness` are network-backed by design (HTTP + SHA-pin verify); `rlat optimise` and the `rlat deep-search` CLI verb are the two opt-in commands that need an Anthropic API key — see [docs/user/API_KEYS.md](docs/user/API_KEYS.md).
- **A grounding directive your LLM sees.** `rlat search --format context` and `rlat skill-context` stamp an explicit instruction at the top of the markdown they emit (`augment` / `knowledge` / `constrain`) telling the consumer LLM how to weight the passages vs its training. The directive is non-negotiable — every consumer of the corpus sees the same rule.
- **Three storage modes, switchable in place.** `bundled` (zstd-framed source inside the file, fully self-contained), `local` (default — source on disk, reconcile via `rlat refresh`), `remote` (HTTP-pinned, SHA-verified, reconcile via `rlat sync`). Switch with `rlat convert` — no rebuild, embeddings preserved.
- **Stays live as you edit.** `rlat watch` (zero-arg, auto-discovers `*.rlat` in cwd) runs a silent debounced refresh on every save. No manual `rlat refresh` after each edit, no "did I forget to refresh?" tax. Local mode only — bundled is immutable, remote routes to `rlat sync`. Same incremental delta-apply pipeline; optimised bands re-project for free.
- **Multi-hop research, free with your Claude subscription.** A `deep-research` skill ships in `.claude/skills/` that drives a plan → retrieve → refine → synthesize loop natively in your Claude Code session — same loop and prompts as the `rlat deep-search` CLI verb, but no API key required because the LLM hops run through your existing Claude subscription. The CLI verb is the same loop exposed for non-Claude-Code agents / CI / batch consumers (Anthropic API key required).

## Quickstart (~2 min)

```bash
pip install rlat[build]                 # base + transformers/torch for first build
rlat install-encoder                    # one-time, ~2 min on CPU
rlat init-project                       # auto-detects docs/ + src/, builds + writes a primer
rlat search myproject.rlat "how does auth work?"
```

Once you have a `.rlat`, query-time only needs the base install — `pip install rlat` (no extras) on a different machine is enough to search a knowledge model someone else built. Full walkthrough: [docs/user/GETTING_STARTED.md](docs/user/GETTING_STARTED.md). Full CLI reference: [docs/user/CLI.md](docs/user/CLI.md).

### Don't want to build? Try a prebuilt `.rlat` first.

Four launch-day knowledge models live on HuggingFace, ready to query in seconds — no encoder install, no build step. They use **remote storage mode** (passages reference the source repo at a pinned commit; sources are fetched on demand and SHA-verified at query time):

| Corpus | Source | Files | Passages |
|---|---|---:|---:|
| [`tenfingers/powerbi-developer-rlat`](https://huggingface.co/datasets/tenfingers/powerbi-developer-rlat) | [MicrosoftDocs/powerbi-docs](https://github.com/MicrosoftDocs/powerbi-docs) `powerbi-docs/developer` | 176 | 5,684 |
| [`tenfingers/powershell-docs-rlat`](https://huggingface.co/datasets/tenfingers/powershell-docs-rlat) | [MicrosoftDocs/PowerShell-Docs](https://github.com/MicrosoftDocs/PowerShell-Docs) `reference` | 2,647 | 107,033 |
| [`tenfingers/python-stdlib-rlat`](https://huggingface.co/datasets/tenfingers/python-stdlib-rlat) | [python/cpython](https://github.com/python/cpython) `Doc` | 617 | 49,179 |
| [`tenfingers/tsql-docs-rlat`](https://huggingface.co/datasets/tenfingers/tsql-docs-rlat) | [MicrosoftDocs/sql-docs](https://github.com/MicrosoftDocs/sql-docs) `docs/t-sql` | 1,209 | 33,282 |

```bash
pip install rlat
huggingface-cli download tenfingers/python-stdlib-rlat python-stdlib.rlat --local-dir .
rlat search python-stdlib.rlat "asyncio Task cancellation" --top-k 5
```

All four are encoded with `gte-modernbert-base` 768d at the pinned revision documented in [`docs/internal/BENCHMARK_GATE.md`](docs/internal/BENCHMARK_GATE.md), so retrieval quality matches anything you build locally with the same recipe.

## A real query, end-to-end

```bash
rlat build ./docs ./src -o resonance-lattice.rlat
rlat search resonance-lattice.rlat "How does verified retrieval work?" \
  --top-k 2 --format context
```

```markdown
<!-- rlat-mode: augment -->
> **Grounding mode: augment.** Use the passages below as primary context
> for this corpus's domain. Cite them when answering; prefer them over
> your training knowledge when the two conflict.

<!-- docs/internal/STORE.md:10094+56 score=0.832 verified -->
## Verified retrieval (`store.verified` — WS3 #292 port)

<!-- docs/user/GLOSSARY.md:10864+279 score=0.758 verified -->
**Verified retrieval** — the contract that every passage in every `.rlat` carries
source provenance + content hash, so query-time results can be cited back to
source and drift can be detected. `rlat`'s single biggest architectural
advantage vs. opaque-embedding-store retrievers.
```

Zero LLM calls, zero network round-trip, all local. That block drops straight into a Claude / Cursor / agent prompt. Each comment line is a stable citation anchor your assistant can preserve in its answer; the `verified` tag is its drift contract; the grounding-mode header is the rule the LLM is being asked to follow.

## Use it with your AI assistant

`rlat` is assistant-neutral — every result is portable markdown your assistant of choice can read.

### Claude Code

A `deep-research` skill ships in `.claude/skills/`. For cross-file synthesis questions ("why did we pick X over Y?", "what are the trade-offs across Y?"), it drives a plan → retrieve → refine → synthesize loop natively in your Claude Code session. No API key required — your Claude subscription covers the LLM hops. For one-shot fact lookups, use `rlat search`. See [docs/user/SKILLS.md](docs/user/SKILLS.md) for skill `!command` integration with citations + drift status pre-baked.

### Cursor, Aider, Continue, CLI-only, CI

```bash
# Pipe the markdown straight into your assistant's prompt
rlat search myproject.rlat "how does auth work?" --format context > context.md

# Or for compliance / audit work where wrong-but-confident is unacceptable:
rlat search myproject.rlat "what's our SOX retention policy?" \
  --format context --mode constrain --strict-names --verified-only
```

The output is plain markdown with citation anchors and a grounding-mode directive header. Paste it into your assistant's system prompt or include it via your IDE's context-injection mechanism. For programmatic multi-hop research outside Claude Code, `rlat deep-search` runs the same loop as the skill but as a CLI command (Anthropic API key required, ~$0.009-0.025 per question — see [API_KEYS.md](docs/user/API_KEYS.md)).

## What's measured

Every claim against named alternatives on committed test sets — methodology and reproducibility recipes in [docs/user/BENCHMARKS.md](docs/user/BENCHMARKS.md).

| Bench | Result |
|---|---|
| **Hallucination** (Microsoft Fabric, 63 hand-written questions, Sonnet 4.6, relaxed rubric) | `rlat deep-search` (default `--mode augment`): **92.2% accuracy / 2.0% answerable hallucination** at $0.009/q, vs LLM-only **56.9% / 19.6%**. `--mode knowledge` variant drops to **0% answerable hallucination** at the same accuracy; `--mode constrain` hits **91.7% distractor refusal** — pick for compliance / audit. Full 11-lane matrix in [BENCHMARKS.md](docs/user/BENCHMARKS.md#hallucination-reduction). The `--strict-names` namecheck gate catches the failure mode where the encoder surfaces a similarly-named real entity for a fake-product-name distractor and the LLM confidently answers about the wrong entity. |
| **Token spend** (rlat repo, 20 questions, **single-shot** retrieval — note: predates `deep-search`) | `rlat skill-context --mode constrain` $0.012 per correct answer vs grep+read $0.044 (3.7× cheaper) vs full-corpus dump $0.796 (67× cheaper). Absolute accuracy is lower (35% vs 85%) — single-shot trades accuracy for cost. Deep-search lanes pending v2.0.1. |
| **Session-start primer** (`resonance-lattice.rlat`, 25 scenarios × 5 lanes, Sonnet 4.6) | Code primer (`rlat summary`) wins **3/5 orientation**; memory primer (`rlat memory primer`) wins **5/5 memory recall**; `both_primers` loaded carries **48% turn-1 correct** vs cold's **0%**. Code primer ~1,708 tok/call, memory primer ~746 tok/call, both ~2,454 tok/call (~1,400× smaller than a full-corpus dump). [BENCHMARKS.md § Session-start primer](docs/user/BENCHMARKS.md#session-start-primer). |
| **Query latency + on-disk size** (1K-passage corpus, Intel CPU + OpenVINO) | Warm query p50 **17 ms** vs Chroma **145 ms** (8.5× faster); **2.7 MB** on disk vs Chroma **8.6 MB** (3.2× smaller). |
| **Retrieval quality** (BEIR-5 mean nDCG@10, gte-modernbert-base 768d) | **0.5144** locked floor — beats BGE-large by **+0.026** and E5-large by **+0.081** on the same `rlat` stack (apples-to-apples). [BENCHMARK_GATE.md](docs/internal/BENCHMARK_GATE.md). |

## How it's built

Three layers: the **field** (`gte-modernbert-base` 768d CLS+L2, dense cosine, FAISS HNSW) routes the query to ranked passage IDs; the **store** (the `.rlat` archive) resolves IDs back to source bytes and verifies drift; and **your assistant** composes the synthesis. The third layer is deliberately empty on `rlat`'s side — there is no reader. That keeps `rlat` LLM-free at retrieval time and lets every consumer impose its own grounding rules. Single recipe — no rerank, no lexical sidecar, no query-prefix tuning, no auto-mode router. Empirically validated to match or beat tuned alternatives. Deep dive: [docs/internal/ARCHITECTURE.md](docs/internal/ARCHITECTURE.md).

## Documentation

- [docs/user/GETTING_STARTED.md](docs/user/GETTING_STARTED.md) — first knowledge model in 15 minutes
- [docs/user/CLI.md](docs/user/CLI.md) — every command, every flag
- [docs/user/CORE_FEATURES.md](docs/user/CORE_FEATURES.md) — seven things `rlat` enables
- [docs/user/BENCHMARKS.md](docs/user/BENCHMARKS.md) — measured numbers vs named baselines
- [docs/user/SKILLS.md](docs/user/SKILLS.md) — Anthropic-skill `!command` integration
- [docs/user/STORAGE_MODES.md](docs/user/STORAGE_MODES.md) — bundled / local / remote decision guide
- [docs/user/API_KEYS.md](docs/user/API_KEYS.md) — when an Anthropic API key is needed (and the free alternatives)
- [docs/user/FAQ.md](docs/user/FAQ.md) — common questions, including licence
- [docs/user/GLOSSARY.md](docs/user/GLOSSARY.md) — terminology
- [docs/internal/ARCHITECTURE.md](docs/internal/ARCHITECTURE.md) — three-layer thesis + module map
- [docs/internal/KNOWLEDGE_MODEL_FORMAT.md](docs/internal/KNOWLEDGE_MODEL_FORMAT.md) — `.rlat` v4.1 ZIP format spec (open it with `unzip` and `jq`; nothing proprietary)
- [docs/internal/HONEST_CLAIMS.md](docs/internal/HONEST_CLAIMS.md) — calibrated claims, known limits
- [docs/internal/BENCHMARK_GATE.md](docs/internal/BENCHMARK_GATE.md) — locked floor + reproduction recipe
- [docs/VISION.md](docs/VISION.md) — the why

## License

[BSL-1.1](LICENSE.md). Source-available, commercial-use restricted during the change-licence window. See the [licence FAQ](docs/user/FAQ.md) for the change date and what it means in practice.

Issues, contributions, and corrections welcome at [tenfingerseddy/resonance-lattice](https://github.com/tenfingerseddy/resonance-lattice).
