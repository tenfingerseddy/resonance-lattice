<p align="center">
  <img src="docs/assets/lockup-horizontal.svg" alt="Resonance Lattice" width="640">
</p>

# Resonance Lattice (`rlat`)

[![PyPI](https://img.shields.io/pypi/v/rlat?label=pypi%20rlat)](https://pypi.org/project/rlat/)
[![Python](https://img.shields.io/pypi/pyversions/rlat)](https://pypi.org/project/rlat/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)](LICENSE.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20datasets-tenfingers-yellow)](https://huggingface.co/tenfingers)

> **A knowledge model that knows its own world.** One portable `.rlat` file carries your corpus — docs, code, notes — plus what's been learned about the world that corpus covers: stable facts, standing constraints, and what was tried and failed. Every learned item carries a receipt. Local, citeable, drift-checked. Proven in three unrelated worlds: a Fabric tenant, a garden, a law practice.

`rlat` is a semantic layer over unstructured data. It packages a corpus into a single `.rlat` file you own; where a semantic data model captures the meaning of a database, a knowledge model captures the meaning of unstructured text — plus every coordinate needed to find, cite, and verify the source. You query it with one CLI command and feed the results to whichever AI assistant you already use (Claude Code, Cursor, Aider, your own agent). No hosted index, no LLM in the retrieval loop, no API key to search.

## What you get

- **One file you own.** Embeddings + source coordinates + content hashes + learned facts, all in one ZIP. Open it with `unzip`, inspect it with `jq`. Three storage modes — `bundled` (source inside the file), `local` (source on disk, reconcile with `rlat refresh` or live with `rlat watch`), `remote` (HTTP-pinned, SHA-verified, reconcile with `rlat sync`) — switchable in place with `rlat convert`, no rebuild.
- **Every passage cited, drift-checked.** Each result is `(text, source_file, char_offset, char_length, content_hash, drift_status)`. If a source file changed since the build, the result says so instead of silently returning stale text. `--verified-only` refuses unverified evidence outright.
- **It knows its own world.** Three kinds of learned content, each proven to change answers for the better, each with a published receipt:
  - **World facts** — stable facts about your environment that no document states ("the tenant runs an F64 capacity", "watering is restricted to twice a week"). The capture path is validated: 0.86 precision, 1.00 recall, and zero personal facts captured across seven deliberate traps. [Receipt](https://github.com/tenfingerseddy/resonance-lattice/blob/main/benchmarks/attribute_gate_e2c/DESIGN.md)
  - **Standing constraints** — rules every answer must respect ("no synthetic fertiliser", "we don't take family-law matters"). Tested in three unrelated worlds — a Fabric tenant, a home garden, a law practice. Without the rules served, answers broke them most of the time (15 of 24 Fabric items; 23 of 24 garden and legal items). With them served, violations nearly disappeared (1 of 15; 3 of 23), and no unrelated question lost its answer. Receipts: [Fabric](https://github.com/tenfingerseddy/resonance-lattice/blob/main/benchmarks/constraint_band/VERDICT.md), [garden + law practice](https://github.com/tenfingerseddy/resonance-lattice/blob/main/benchmarks/constraint_band_xdomain/VERDICT.md)
  - **Tried-and-falsified findings** — "we tested X; it failed." Serving the verdict stopped every repeat recommendation of a falsified approach (0 of 7, against 6 of 7 when the same approach was described without its verdict — an 86-point gap). [Receipt](https://github.com/tenfingerseddy/resonance-lattice/blob/main/benchmarks/falsification_ledger/VERDICT.md)
- **Capture is low-friction and consent-gated.** Say "remember this" and the bundled `rlat-learn` skill captures it on your Claude subscription — you approve every fact, no API key. Explicit: `rlat capture-attribute km.rlat "..." --kind attribute|constraint|falsified` (or `rlat capture-env` to auto-read machine facts). Unattended: `rlat memory install-hooks --mine` opts a workspace into a four-gate session miner that learns world facts as you work. Every path drops facts about a *person* by design — that privacy gate held at zero leaks in validation.
- **A corpus that fills its own gaps — opt-in, gated.** The archive records what was asked and poorly covered. `rlat grow` fills the most-demanded, worst-covered gap; nothing lands without passing its gates. `--dry-run` shows the candidates first.
- **Nothing is forgotten silently.** Automatic cleanup of learned facts is **not** shipped: we built it, tested it live, and it failed its pre-registered safety bar — it suppressed a true fact and added no measurable value. The full failure record is published. [Receipt](https://github.com/tenfingerseddy/resonance-lattice/blob/main/benchmarks/r4_continuous_credit/DESIGN_CONFIRM.md) Corrections stay explicit: the newest value per subject wins, and `rlat lens` is the review/delete surface.
- **Trust surfaces throughout.** Verified retrieval, corpus self-audit (contradictions + demand gaps, stored at build), and a grounding directive at the top of every context block telling the consumer LLM how to weight the passages (`augment` / `knowledge` / `constrain`).
- **No API key required — bring your Claude subscription.** Retrieval never calls an LLM: `rlat search`, `profile`, `compare`, `summary`, `refresh`, `watch`, and `skill-context` run offline after a one-time `rlat install-encoder`. The LLM-driven loops run two ways, your choice. **Free on the subscription you already have**: bundled Claude Code skills drive deep research (`deep-research`), teach the file your world in-session with you approving every fact (`rlat-learn`), and run the whole curation loop (`rlat-gap-scan`, `rlat-contradictions`, `rlat-refresh-facts`, `rlat-curate`) — your assistant's own model does the reading and judging, which is typically a stronger model than a budget API call. **Or unattended with an API key** for automation and CI: `rlat deep-search` (~$0.009/question), `rlat grow`, the background capture hook. See [API keys](https://tenfingerseddy.github.io/resonance-lattice/api-keys.html).

## Install

```bash
pip install rlat[build]      # Python 3.11+; [build] adds the encoder deps for building
rlat install-encoder         # one-time, ~2 min on CPU
```

Query-only machines need just `pip install rlat` — no extras — to search a knowledge model someone else built.

## 60-second quickstart

```bash
cd my-project/
rlat init-project                       # auto-detects docs/ + src/, builds + writes a primer
rlat search my-project.rlat "how does auth work?" --format context
```

```markdown
<!-- rlat-mode: augment -->
> **Grounding mode: augment.** Use the passages below as primary context
> for this corpus's domain. Cite them when answering; prefer them over
> your training knowledge when the two conflict.

<!-- SOURCE docs/auth.md:1042+118 score=0.810 verified -->
Authentication is handled by ...
```

Now teach it a standing rule of your world, and see it served:

```bash
rlat capture-attribute my-project.rlat \
  "All schema changes go through the staging workspace first" --kind constraint

rlat search my-project.rlat "how do I deploy a schema change?" --format context
# → your standing constraints now render right after the grounding
#   directive, before any cited passage
```

`--format context` prints LLM-ready markdown — citations, drift status, the grounding directive, and the learned facts and rules of your world. Paste it into any assistant's prompt. Inputs are plain text (`.md`, `.txt`, source code, prose); convert PDF/DOCX to markdown first.

If your docs fit in your assistant's context window, `cat docs/*.md` is fine. `rlat` earns its keep when they don't — or when citations, drift checks, and a world that's learned over time matter. Full walkthrough: [Getting started](https://tenfingerseddy.github.io/resonance-lattice/getting-started.html).

### Don't want to build? Try a prebuilt `.rlat` first.

Five prebuilt knowledge models live on HuggingFace, ready to query in seconds — no encoder install, no build step. Remote-mode rlats pin to the source repo at a commit SHA and fetch source on demand (SHA-verified at query time); the bundled-mode rlat packs source inside the archive:

| Corpus | Source | Files | Passages | Filename | Mode |
|---|---|---:|---:|---|---|
| [`tenfingers/fabric-docs-rlat`](https://huggingface.co/datasets/tenfingers/fabric-docs-rlat) | [MicrosoftDocs/fabric-docs](https://github.com/MicrosoftDocs/fabric-docs) `docs` | 2,435 | 67,503 | `fabric-docs-bundled.rlat` | bundled |
| [`tenfingers/powerbi-developer-rlat`](https://huggingface.co/datasets/tenfingers/powerbi-developer-rlat) | [MicrosoftDocs/powerbi-docs](https://github.com/MicrosoftDocs/powerbi-docs) `powerbi-docs/developer` | 176 | 5,684 | `powerbi-developer.rlat` | remote |
| [`tenfingers/powershell-docs-rlat`](https://huggingface.co/datasets/tenfingers/powershell-docs-rlat) | [MicrosoftDocs/PowerShell-Docs](https://github.com/MicrosoftDocs/PowerShell-Docs) `reference` | 2,647 | 107,033 | `powershell-docs.rlat` | remote |
| [`tenfingers/python-stdlib-rlat`](https://huggingface.co/datasets/tenfingers/python-stdlib-rlat) | [python/cpython](https://github.com/python/cpython) `Doc` | 617 | 49,179 | `python-stdlib.rlat` | remote |
| [`tenfingers/tsql-docs-rlat`](https://huggingface.co/datasets/tenfingers/tsql-docs-rlat) | [MicrosoftDocs/sql-docs](https://github.com/MicrosoftDocs/sql-docs) `docs/t-sql` | 1,209 | 33,282 | `tsql-docs.rlat` | remote |

```bash
pip install rlat huggingface_hub
hf download tenfingers/python-stdlib-rlat python-stdlib.rlat --local-dir .
rlat search python-stdlib.rlat "asyncio Task cancellation" --top-k 5 --format context
```

All five are encoded with `gte-modernbert-base` 768d at the pinned revision documented in [`docs/internal/BENCHMARK_GATE.md`](docs/internal/BENCHMARK_GATE.md), so retrieval quality matches anything you build locally.

## Use it with your AI assistant

`rlat` is assistant-neutral — every result is portable markdown. Pipe `rlat search ... --format context` into any prompt, or use `rlat skill-context` for Anthropic-skill `!command` blocks ([Skills](https://tenfingerseddy.github.io/resonance-lattice/skills.html)). For compliance work where wrong-but-confident is unacceptable: `--mode constrain --strict-names --verified-only`.

In Claude Code, a bundled `deep-research` skill drives a plan → retrieve → refine → synthesize loop for cross-file questions — no API key, your subscription covers the LLM hops. The same loop runs as `rlat deep-search` for other agents and CI (Anthropic API key required — [API keys](https://tenfingerseddy.github.io/resonance-lattice/api-keys.html)).

## What's measured

Every claim against named alternatives on committed test sets, designs pre-registered before the runs — methodology and reproduction in [Benchmarks](https://tenfingerseddy.github.io/resonance-lattice/benchmarks.html). The world-learning receipts are linked inline above; headline retrieval numbers:

| Bench | Result |
|---|---|
| **Hallucination** (Microsoft Fabric, 63 hand-written questions, Sonnet 4.6, relaxed rubric) | `rlat deep-search`: **92.2% accuracy / 2.0% answerable hallucination** vs LLM-only **56.9% / 19.6%**. `--mode knowledge` drops to **0%** answerable hallucination at the same accuracy. [Full matrix](https://tenfingerseddy.github.io/resonance-lattice/benchmarks.html#hallucination). |
| **Query latency + on-disk size** (1K-passage corpus, Intel CPU + OpenVINO) | Warm query p50 **17 ms** vs Chroma **145 ms**; **2.7 MB** on disk vs Chroma **8.6 MB**. |
| **Retrieval quality** (BEIR-5 mean nDCG@10, gte-modernbert-base 768d) | **0.5144** locked floor — beats BGE-large by **+0.026** and E5-large by **+0.081** on the same `rlat` stack. [BENCHMARK_GATE.md](docs/internal/BENCHMARK_GATE.md). |

## Documentation

User docs — rendered — at **https://tenfingerseddy.github.io/resonance-lattice/**:

- [Getting started](https://tenfingerseddy.github.io/resonance-lattice/getting-started.html) — first knowledge model in 15 minutes
- [CLI reference](https://tenfingerseddy.github.io/resonance-lattice/cli.html) — every command, every flag
- [Core features](https://tenfingerseddy.github.io/resonance-lattice/core-features.html) — what `rlat` enables
- [Claims](https://tenfingerseddy.github.io/resonance-lattice/claim-system.html) — how learned facts are captured, trusted, served, and corrected
- [Benchmarks](https://tenfingerseddy.github.io/resonance-lattice/benchmarks.html) — measured numbers vs named baselines, with run artifacts
- [Skills](https://tenfingerseddy.github.io/resonance-lattice/skills.html) — Anthropic-skill `!command` integration
- [Storage modes](https://tenfingerseddy.github.io/resonance-lattice/storage-modes.html) — bundled / local / remote decision guide
- [API keys](https://tenfingerseddy.github.io/resonance-lattice/api-keys.html) — when a key is needed (and the free alternatives)
- [FAQ](https://tenfingerseddy.github.io/resonance-lattice/faq.html) — common questions, including licence
- [Glossary](https://tenfingerseddy.github.io/resonance-lattice/glossary.html) — terminology

Engineering reference (in-repo): [ARCHITECTURE.md](docs/internal/ARCHITECTURE.md) — three-layer thesis + module map · [STORE.md](docs/internal/STORE.md) — `.rlat` ZIP format spec (nothing proprietary) · [HONEST_CLAIMS.md](docs/internal/HONEST_CLAIMS.md) — calibrated claims, known limits · [VISION.md](docs/VISION.md) — the why.

## License

[BSL-1.1](LICENSE.md). Source-available, commercial-use restricted during the change-licence window. See the [licence FAQ](https://tenfingerseddy.github.io/resonance-lattice/faq.html) for the change date and what it means in practice.

Issues, contributions, and corrections welcome at [tenfingerseddy/resonance-lattice](https://github.com/tenfingerseddy/resonance-lattice).
