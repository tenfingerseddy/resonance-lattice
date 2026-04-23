---
title: Core Use Cases
slug: core-features
description: Seven high-value use cases Resonance Lattice ships with — each unlocked by the same portable knowledge model.
nav_group: Product
nav_order: 5
---

# Core Use Cases

Resonance Lattice is a knowledge model — a `.rlat` file containing the semantic structure of your corpus, paired with your original source files as the lossless evidence layer. One artifact, three layers (the field routes queries, the files serve evidence, any reader synthesizes the answer), portable across any LLM, cloud or local.

> **"Is this RAG?"** Not quite. RAG retrieves passages and hands them to an LLM. Resonance Lattice builds a queryable *model of* your corpus — retrieval is one thing the model does, not the whole product. The seven use cases below are what falls out when the model itself is the artifact you ship.

What follows are seven use cases we ship in v1.0 — the highest-value workflows we know the technology unlocks today. They share one artifact and one architecture; they differ in which surfaces of the technology they exercise. As the model and the ecosystem mature, more use cases will emerge from the same foundation.

---

## 1. Ground an LLM with verified citations

**Trust the answer without checking the LLM's work.** Resonance Lattice grounds every reply in your actual files and checks every citation against the source itself — so fabricated references, misquoted passages, and made-up facts can't reach the answer in the first place. The categories of hallucination that matter most for serious work are the ones you can verify mechanically. Resonance Lattice verifies them.

### What you get

- **Answers built from your files, not from what the LLM remembers.** Ask a question. Every claim in the response traces to a passage retrieved from your corpus. Nothing slips in from the LLM's pretraining — if it isn't in your documents, it doesn't reach the answer.

- **Every citation checked against the source itself.** When the answer cites a passage, the system has already opened the file at the cited line and confirmed the quote matches what's there. Verified citations show with a ✓; anything unverified shows with a ○ — so a researcher, lawyer, consultant, or clinician reading the answer knows at a glance which claims have been checked and which still need their eyes.

- **Fabricated references can't slip through.** If the LLM invents a `[7]` pointing to a passage that doesn't exist, the system strips it before you see the answer. You'll never get a confident-sounding citation that points to nothing — the failure mode that has real consequences in legal, medical, regulatory, and academic work.

- **Drift detection.** When a source file changes after the knowledge model was built, the verification flags the mismatch. You won't unknowingly cite a quote that no longer exists in the source.

- **Honest "I don't know" when evidence is thin.** The default behavior tells the LLM to say so plainly when the corpus doesn't cover the question, rather than guess. A compliance officer or a research analyst gets a useful "no" instead of a confident-sounding answer that might be wrong.

- **Your LLM, your choice.** Works with Claude or any frontier model in the cloud. Works with a local LLM running on your own machine for sensitive corpora — air-gapped environments, regulated industries, client engagements where data egress is prohibited. Same answer shape, same citation guarantees, same verification.

- **Two ways to use the answer.** Get the prose with cited evidence (`rlat ask`), or skip the LLM entirely and just get the evidence pack with citations (`rlat ask --reader context`) — useful when you want to read the source passages yourself or feed them into a different tool.

### Evidence

Hallucinated citations are a structural problem with a structural fix. The system reads the file, finds the cited line, and compares the quote — every time. If the LLM invents a reference that points to nothing, it is filtered. If a file changed after the model was built, the mismatch is surfaced. Tested across frontier cloud models and local open-weight ones, the default behavior is to refuse to guess: when the evidence is thin, you get an honest *"I don't have enough information"* instead of a confident-sounding answer that might be wrong.

**Honest refusal is a measured capability.** On the LoCoMo memory benchmark, rlat correctly refuses to answer on **91.3% of the 446 questions deliberately designed to be unanswerable** from the conversation. That is the highest-performing category in the whole 1986-question benchmark — the reader gets "I don't have enough information" right more often than it answers any category where the answer is present. For legal, medical, regulatory, and compliance work where a wrong answer is worse than no answer, this is the default behaviour you want. See [Benchmarks — LoCoMo](/docs/benchmarks#locomo) for details.

---

## 2. Bootstrap any assistant in seconds

**Stop re-explaining your project on every new chat.** Resonance Lattice generates a compressed context document that captures what an assistant needs to know — what the project is, what's settled, what's in flight — and includes it automatically in every session. The conversation doesn't start cold and the assistant doesn't invent details about your work to fill the gap.

### What you get

- **The assistant arrives knowing your project.** One command builds a structured context document covering architecture, conventions, design decisions, and active work threads. The assistant reads it before the first message, so a new conversation starts where the last useful one left off.

- **Two primers, one for the project and one for the work.** A *code primer* captures what the project IS — structure, conventions, patterns. A *memory primer* captures how the work has unfolded — what's settled, what was tried and reversed, what's currently in flight. Together they give an assistant both the static project shape and the live decision trail.

- **No redundant context between them.** The two primers are compared; topics covered in one are skipped in the other. You don't burn context tokens on the same fact showing up twice — the assistant gets denser, more useful context per token.

- **One command from clone to full context.** `rlat init-project` auto-detects your project structure, builds the knowledge model, generates both primers, and wires up the assistant integration. A new team member, a contractor joining a project, or your future self returning after weeks away gets the assistant fully up to speed in under a minute.

- **Stays current as your work evolves.** When source files change, an incremental sync re-encodes only what changed — no full rebuild for routine updates. Freshness checks warn you when a primer is aging, so the assistant never quietly serves stale context.

- **Works with any AI assistant.** Claude Code, Cursor, command-line LLM tools, agents inside other IDE integrations — anywhere a project-context file gets loaded into the conversation.

### Evidence

The two-primer split with cross-primer deduplication addresses the most common failure mode in assistant context — code documentation and conversation memory repeating the same facts, wasting space the assistant could have used for the actual question. Token budgeting keeps each primer within practical context window limits. Freshness detection compares file modification times against build timestamps so an aging primer surfaces a warning before it serves a stale picture of the project.

---

## 3. Make skills knowledge-aware on any LLM platform

**Skills have become the standard pattern for extending LLM assistants — a declarative bundle of capabilities that's largely replaced ad-hoc subagent orchestration across Anthropic, OpenAI, and the agent-framework ecosystem.** A conventional skill is a fixed block of instructions loaded identically on every invocation regardless of what the user actually asked. Resonance Lattice makes skills *adaptive*: each one declares which knowledge models it draws from, and the retrieval pipeline assembles precisely targeted context for each request — different question, different evidence, same skill.

### What you get

- **Four-tier adaptive injection.** When a skill triggers, the pipeline assembles context across four tiers, each with its own token budget, freshness signal, and confidence gating:
  - *Tier 1 (static)* — the skill's SKILL.md header: workflow instructions, templates, scripts. Loaded every time.
  - *Tier 2 (foundational)* — queries declared in skill frontmatter that the skill always needs. Cached across invocations.
  - *Tier 3 (user query)* — the user's actual question resonated against the skill's knowledge models. Different every request.
  - *Tier 4 (LLM-derived)* — optional additional queries targeting implicit needs the model anticipates (e.g., workspace separation patterns when building a lakehouse).

- **Declarative knowledge binding.** Skills declare their knowledge sources and retrieval preferences in frontmatter — model paths, foundational queries, injection mode, token budget, and whether to derive additional queries. No code changes required to make an existing skill knowledge-aware.

  ```yaml
  ---
  knowledge models:
    - .rlat/fabric-docs.rlat
    - .rlat/engineering-practices.rlat
  knowledge model-queries:
    - "Fabric workspace authentication and service principal patterns"
    - "Medallion architecture naming conventions bronze silver gold"
  knowledge model-mode: augment
  knowledge model-budget: 2000
  knowledge model-derive: true
  knowledge model-derive-count: 3
  ---
  ```

- **Injection modes per skill.** Each skill chooses how aggressively context is gated:
  - `augment` — suppress low-relevance context (most skills)
  - `knowledge` — suppress only very low relevance (domain expertise where training data may be stale)
  - `constrain` — always inject, no gating (compliance, regulatory — zero hallucination tolerance)

- **Skill routing.** Given a query and a manifest of available skills, `rlat skill route` ranks skills by resonance energy to determine which skill's knowledge is most relevant. Multi-skill projects surface the right expertise without manual selection.

- **Per-skill diagnostics.** A full suite of skill-scoped commands exposes the knowledge backing each skill:
  - `rlat skill search` — search within a skill's knowledge model
  - `rlat skill profile` — semantic shape of the skill's knowledge
  - `rlat skill compare` — compare two skills' knowledge coverage
  - `rlat skill freshness` — how stale is the skill's model relative to its reference materials
  - `rlat skill gaps` — detect uncovered areas in the skill's knowledge

- **Freshness-aware injection.** Each tier reports the age of the model it drew from. Stale models trigger warnings and can auto-sync before injection. Skills never silently serve outdated knowledge.

- **Deduplication and confidence gating.** When multiple tiers or models return overlapping passages, the pipeline deduplicates by content hash and keeps the highest-scoring instance. A coverage confidence metric from enriched retrieval gates whether the dynamic body is included at all — if the models have nothing relevant, the skill falls back to its static instructions rather than injecting noise.

### Evidence

Without dynamic context, a skill about Fabric lakehouse architecture injects the same static instructions whether the user asks about workspace setup, ingestion notebooks, or Delta Lake merge patterns. With four-tier injection, each of those questions retrieves different evidence from the same models — workspace authentication patterns for the first, PySpark ingestion templates for the second, upsert semantics for the third. The skill's static instructions frame the workflow; the knowledge models supply the specifics.

When grounding an LLM with Resonance Lattice context, hallucination rates dropped from **0.78 to 0.16** (80% reduction) and fact recall improved from **0.27 to 0.91** (237% improvement) on internal evaluation.

---

## 4. Give an assistant durable memory

**Conversations that accumulate, not conversations that vanish.** Every chat session you have with an LLM ends and the context is lost. Resonance Lattice keeps your conversations as a queryable knowledge model — three layers of memory by recency, decisions tracked across time, contradictions surfaced when newer turns supersede older ones, and verifiable removal when you need to forget something cleanly.

### What you get

- **Three layers of memory by how fresh it is.** *Working memory* covers what was just said in the current session. *Episodic memory* covers past sessions, with their dates and context. *Semantic memory* covers long-running themes and settled patterns. A question like *"what did we decide last Tuesday?"* pulls from the right layer; a question like *"what's our overall approach?"* pulls from another. The assistant gets the right kind of recall for the question being asked.

- **Memories don't drift.** Most memory systems summarize your conversations using an LLM, then summarize the summaries, until the originals are unrecognizable months later. Resonance Lattice never rewrites. Your conversation text is preserved; the searchable index is built mechanically from it. The photocopy-of-a-photocopy failure mode that breaks long-running memory tools cannot accumulate, because nothing is being rewritten in the first place.

- **Decision trails with reversals tracked.** When a newer entry contradicts an older one, the contradiction is surfaced and the reversal is marked. Ask *"what did we settle on for X?"* and you get the current answer with the history of how it changed — not a confused mash-up of old and new positions presented as if they were the same.

- **Recency-aware recall.** Filter by session, by date range, by who said it, or weight recent context heavier than older — so a question about today's work doesn't surface a decision from six weeks ago, and a question about long-term direction isn't drowned out by yesterday's chatter.

- **Forget specific things, verifiably.** When you need to remove a piece of information — a sensitive detail, a deprecated decision, a client engagement that ended — the removal is mathematically exact and produces proof that it has been excised. That matters for compliance, GDPR right-to-erasure, and any context where *"we deleted it"* needs to be auditable, not just asserted.

- **No AI in the background guessing what to remember.** Nothing is being summarized, classified, or selectively forgotten by another LLM. There is no extraction model that could mis-categorize an entry, no summarizer that could lose nuance, no over-eager classifier that could store something private without your knowledge. The write path is mechanical — what you said is what gets remembered.

### Evidence

The memory architecture is benchmarked against two standard long-conversation evaluations.

**LongMemEval** (retrieval-only): Phase 2 v14 baseline at **Recall@5 0.924 / MRR 0.919** on 500 questions — the correct prior session is in top-5 for 92% of long-horizon memory queries, across six question types (single-session, multi-session, knowledge-update, temporal).

**LoCoMo** (end-to-end QA, the headline Mem0/Letta/Zep benchmark): Phase 2 baseline at **66.23% leaderboard-style (excl. adversarial) / 71.85% overall** on 1986 questions across 10 real multi-month conversations (Claude Sonnet 4.6 judge). That lands rlat in Mem0 / Letta territory on the full retrieve → read → judge pipeline. Standout category: **91.3% correct refusal on adversarial questions** — unanswerable-by-design questions where the reader must decline. See [Benchmarks — LoCoMo](/docs/benchmarks#locomo).

**What's measured and what isn't.** Both benchmarks exercise *retrieval over a fixed haystack* — LoCoMo loads the whole conversation at once, LongMemEval loads all prior sessions at once. They do not exercise the write-path differentiator: tier migration, retention decay, stale-fact overwrite, algebraic-exact forgetting. That validation (on dogfooded Claude Code transcripts aged over months) is explicit pending work in [Honest Claims](/docs/honest-claims#remaining-pending-claims). The shipped tier-weighted `memory recall` path regresses vs cartridge-per-context on both static-haystack benchmarks; for static QA, cartridge retrieval is the measured-best surface. The retention / consolidation / forget primitives remain the architectural claim — they're just not what these benchmarks measure.

The mechanical-write-path approach avoids the failure modes documented across the major LLM-driven memory systems — each of which relies on an LLM to derive memories at write time, introducing compounding error over long interaction histories. Because the searchable index is rebuildable from your raw conversation text at any time, derivation drift cannot accumulate the way it does in summary-of-summary chains.

---

## 5. Keep your knowledge portable, private, and local-first

**Your knowledge stays yours.** A knowledge model is a single `.rlat` file you own, and your source files stay where they are — on your machine, in your repo, behind your firewall. Nothing leaves your environment unless you explicitly send it. That makes the same artifact serve a hobbyist on a laptop, a consultant carrying client knowledge between engagements, a research team archiving a paper corpus, and a regulated-industry team where data egress is restricted by policy.

### What you get

- **Single-file portability.** One `.rlat` file packages the semantic model and source registry. Copy it, version it, share it — it works anywhere Python runs.

- **Fully offline after first build.** The encoder downloads once from Hugging Face. After that, builds and queries need no network access — useful for air-gapped environments, regulated industries, and sensitive corpora where data egress is restricted by policy.

- **Field-only sharing.** Share the semantic model without exposing source text. A research team can publish the structure of a paper corpus without publishing the papers; a consultant can hand off a knowledge model without exposing client data; an internal team can ship a model of a sensitive codebase to a vendor without shipping the code itself.

- **Algebraically exact removal.** The `forget` operation performs a mathematically precise rank-1 subtraction. It produces a removal certificate quantifying residual cross-talk — verifiable proof that specific knowledge has been excised from the field. This is not approximate deletion; it is algebraically exact, and that matters for compliance, GDPR right-to-erasure, and audit trails.

- **Lossless source resolution.** The semantic model routes queries to the relevant region of your corpus; your original files serve as the evidence layer at query time. Nothing is summarized, paraphrased, or re-encoded — what an LLM sees is the bytes from your file, line numbers and all.

- **No telemetry, no accounts, no cloud.** The entire system is a local Python package. Pair it with a local LLM (Ollama, llama.cpp, on-device inference) for a fully private end-to-end stack — useful when even a hosted vector database is one too many third parties.

### Evidence

The `.rlat` format uses a fixed-size field block (independent of corpus size) with a 64-byte header, format versioning, and configurable compression (zstd, lz4, or none). Field precision options include f16, bf16, and f32. Registry quantisation at 4-bit achieves ~87% compression with minimal retrieval degradation.

---

## 6. Inspect what your knowledge actually contains

**See the structure of what you know — and what you don't.** Most retrieval tools are opaque: you put documents in, you get results out, and there's no way to understand the model in between. Resonance Lattice exposes the semantic structure of your corpus as a first-class interface — useful when you want to audit a documentation set for gaps, compare two snapshots of a knowledge base, or know which questions a corpus actually has the depth to answer before you trust an LLM's response.

### What you get

- **`profile`** — Semantic shape of a knowledge model: per-band energy distribution, effective rank, spectral entropy, coverage summary. Answers "what does this corpus actually contain and how is the knowledge distributed?"

- **`compare`** — Side-by-side structural comparison of two knowledge models: shared knowledge, unique coverage in each, per-band energy deltas, structural similarity score. Answers "how do these two bodies of knowledge relate?"

- **`xray`** — Deep corpus diagnostics: per-band signal-to-noise ratio, saturation (how full is the field?), spectral gap analysis, condition number, purity metrics, effective rank. Answers "is this corpus healthy, or is it noisy, saturated, or thin?"

- **`locate`** — Query positioning analysis: where does a question sit in the knowledge landscape? Returns per-band energy at the query point, band focus, anti-resonance gap (what the field lacks), Mahalanobis distance from corpus centre, coverage classification (strong / partial / edge / gap), and an expansion hint pointing toward richer semantic territory.

- **`probe`** — Named insight recipes for specific diagnostic questions:
  - `health` — signal/noise split via Marchenko-Pastur threshold
  - `novelty "content"` — how novel is this content relative to the corpus (0–1 with recommendation)
  - `saturation` — capacity usage per band with remaining headroom
  - `band-flow` — inter-band mutual information matrix
  - `anti "query"` — what the field does NOT know (per-band gap analysis)
  - `gaps` — topological analysis of knowledge clusters and structural robustness

- **`diff`** — Queryable semantic delta between knowledge model versions. Not a file diff — a signed, per-band energy change that you can search against. Answers "what changed in the knowledge between these two snapshots?"

- **`contradictions`** — Destructive interference detection across sources or across knowledge models. Surfaces passages that disagree on the same topic — useful for editorial review, conflicting documentation, or merging two corpora that disagree on terminology.

- **`topology`** — Eigendecomposition analysis revealing knowledge clusters, connectivity patterns, and structural properties of the semantic field.

### Evidence

Every diagnostic metric is derived from the field tensor itself — eigenvalues, Frobenius norms, Shannon entropy, Von Neumann entropy, Marchenko-Pastur thresholds. These are not heuristic labels; they are mathematically grounded measurements of semantic structure. The system exposes 271 typed RQL operations across 10 mathematical domains.

---

## 7. Steer retrieval without rebuilding

**Shape what the model sees, not just how much.** Most retrieval systems give you a ranked list and a top-k cutoff. Resonance Lattice gives you mathematical tools to reshape the semantic field itself before retrieval — controlling not just which results appear, but from what perspective, with no rebuild required.

### What you get

- **EML corpus transforms.** Reshape the semantic field at query time:
  - `--sharpen` amplifies dominant topics and compresses noise (factoid queries)
  - `--soften` surfaces buried topics under dominant ones (exploratory queries)
  - `--contrast baseline.rlat` highlights what's distinctive about your corpus relative to another
  - `--tune focus|explore|denoise` applies task-matched presets in a single flag

- **Topic boost and suppress.** Amplify or attenuate specific semantic directions by name. `--boost "security"` foregrounds authentication and encryption patterns; `--suppress "boilerplate"` attenuates scaffolding noise.

- **Retrieval lenses.** Reusable semantic perspectives (`.rlens` files) that project the field into learned subspaces:
  - *Eigenspace lenses* — built from exemplar texts, project the field into a learned semantic subspace
  - *Spectral lenses* — reweight eigenvalues by function for structural emphasis
  - *Compound lenses* — chain and compose existing operators
  - All lenses are deterministic, PSD-preserving, composable, invertible, and serialisable

- **Multi-model composition at query time.** No rebuild required:
  - `--with other.rlat` merges fields and dispatches to each registry independently
  - `--through policy.rlat` views your corpus through another's semantic lens
  - `--diff baseline.rlat` queries only what's new since the baseline
  - `--explain` shows composition diagnostics (overlap, novelty, contradiction ratio)

- **Injection modes.** Control how aggressively context is gated:
  - `augment` — full gate, suppress low-relevance context (default for most use cases)
  - `knowledge` — soft gate, suppress only very low relevance (domain expertise, possibly stale training data)
  - `constrain` — no gate, always inject (compliance, regulatory — zero hallucination tolerance)

- **Band-selective retrieval.** The multi-band field decomposes meaning across abstraction levels. The same knowledge model serves conceptual queries, factual lookups, structural questions, and entity searches — each activating different bands of the spectrum.

- **Knowledge model algebra.** Merge, diff, forget, intersect, and project are all O(B×D²) operations — independent of corpus size, with algebraic guarantees: merge is commutative and associative, forget is exact, diff is queryable. This is what makes composition behave like infrastructure rather than a heuristic — the same artifact composes with another and the result has the mathematical properties of both.

### Evidence

The full hybrid pipeline (dense + lexical + reranking) achieves **Recall@5 of 1.00** and **MRR of 0.93** with **0% failed retrieval** on a 24,635-chunk technical documentation benchmark. Context output is **24.6x more token-efficient** than a grep-plus-read workflow (1,518 tokens vs 37,154 for equivalent coverage), while delivering ranked passages with source attribution instead of raw file text.

On the 2026-04-22 5-BEIR rebench, the production default encoder (`BAAI/bge-large-en-v1.5`, flipped from E5 on 2026-04-20) wins 4 of 5 corpora vs flat E5, with strongest gains on technical and financial question-answering. The frontier-tier `qwen3-8b` reaches a 5-BEIR average of **0.500** (vs BGE 0.445 provisional, E5 0.455) when 16 GB of GPU headroom is available. See [Encoder Choice](/docs/encoder-choice) for the per-workload guide.

---

## What this means in practice

**Before Resonance Lattice:**
Your assistant reads raw files. A question about authentication pulls in 37,000 tokens of full source files, most of it irrelevant. The model hallucinates details it cannot find. Each new session starts blank — no memory of yesterday's decisions, no awareness of project conventions, no sense of what the corpus actually covers.

**After Resonance Lattice:**
The same question retrieves 1,500 tokens of ranked, attributed evidence. The model answers from your actual documentation, with citations verified against the source bytes. Session primers carry forward settled decisions and active work threads. Skills adapt their context to each question. The assistant knows what your knowledge contains, where the gaps are, and which surfaces of the model best serve the current task. Knowledge is a file you control — portable, private, inspectable, composable.

The gap is not incremental. It is the difference between an assistant that guesses and one that knows.

---

These seven are the launch use cases — the highest-value workflows we ship in v1.0. What's planned next lives on the [public roadmap](https://github.com/users/tenfingerseddy/projects/1).

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
