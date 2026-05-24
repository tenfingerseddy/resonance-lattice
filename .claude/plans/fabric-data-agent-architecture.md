# rlat × Fabric Data Agent — Architecture

**Status**: v1. The research foundation, locked decisions, mechanism, and schemas.
**Last updated**: 2026-05-17.
**Companions**: [fabric-data-agent-manifesto.md](fabric-data-agent-manifesto.md) (thesis, value claims, scope); [fabric-data-agent-roadmap.md](fabric-data-agent-roadmap.md) (build sequence).

---

## What this document is

The manifesto says *what* this is and *why*. This document records the *research* it rests on, the *decisions* taken (with the alternatives rejected), and the *mechanism* — components, data flow, schemas. The roadmap says *when* and in what order it gets built.

---

## 1. Research foundation

Every decision below derives from these verified facts. Each was checked against primary Microsoft documentation; the rushed-claim corrections in §1.7 are recorded so they are not repeated.

### 1.1 Fabric data agents

- A data agent supports these data sources: **lakehouse, warehouse, Power BI semantic model, KQL database, mirrored database, ontology, Microsoft Graph**. Up to **5** sources per agent, in any combination.
- For a KQL database the agent uses **NL2KQL** — it generates Kusto Query Language from natural language and executes it. It does **not** compute embeddings itself.
- Docs: [Fabric data agent concept](https://learn.microsoft.com/en-us/fabric/data-science/concept-data-agent) · [Create a data agent](https://learn.microsoft.com/en-us/fabric/data-science/how-to-create-data-agent) · [Data agent configurations](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-configurations) · [Example queries](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-example-queries) · [Consume a data agent in Foundry](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-foundry).

**Consequence**: a KQL database is the chosen façade. It is the only data-agent source that can store vectors and compute similarity in-query.

### 1.2 Eventhouse vector search

- KQL function **`series_cosine_similarity()`** computes vector similarity. Vectors are stored in a `dynamic` column.
- The **`Vector16`** encoding stores floats as Bfloat16 — 4× smaller, similarity functions "orders of magnitude" faster.
- Search is **brute-force** — there is no ANN / HNSW index. It parallelises across shards. Documented scale target: **~1 million vectors** with `Vector16` plus tuned sharding/merging policies (e.g. ≤3,125 rows/shard on a 20-node cluster).
- Docs: [Vector similarity search with Eventhouse](https://blog.fabric.microsoft.com/en-US/blog/empowering-real-time-searches-vector-similarity-search-with-eventhouse/) · [Vector database (Fabric)](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/vector-database) · [Eventhouse as a vector database tutorial](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/vector-database-eventhouse).

**Consequence**: a ~63K-passage Fabric docs corpus sits comfortably inside the scale ceiling. Exact brute-force cosine means **retrieval parity** with local rlat is achievable.

### 1.3 KQL embedding paths — both are Azure-OpenAI-locked

- **`ai_embeddings` / `ai_embed_text`** plugins generate embeddings inside a KQL query. Connection string is an Azure OpenAI deployment URI; the `azure_openai` callout policy is regex-locked to `*.openai.azure.com` / `cognitiveservices.azure.com` / `services.ai.azure.com` domains. Embedding model and dimension are configurable via `ModelParameters`. Requires the `Cognitive Services OpenAI User` role. `RecordsPerRequest` defaults to **1**; Azure OpenAI embeddings are "subject to heavy throttling".
- **`slm_embeddings_fl()`** runs an embedding model (`e5-small-v2`) **locally inside the Eventhouse Python sandbox** via the `python()` plugin — **no external service**.
- Docs: [ai_embeddings plugin](https://learn.microsoft.com/en-us/kusto/query/ai-embeddings-plugin?view=microsoft-fabric) · [ai_chat_completion plugin](https://learn.microsoft.com/en-us/kusto/query/ai-chat-completion-plugin?view=microsoft-fabric) · [New OpenAI plugins for Eventhouse](https://blog.fabric.microsoft.com/en-US/blog/introducing-new-openai-plugins-for-eventhouse-preview/) · [SLM embeddings in Eventhouse](https://blog.fabric.microsoft.com/en-US/blog/create-embeddings-in-fabric-eventhouse-with-built-in-small-language-models-slms/).

**Consequence**: Fabric offers exactly two ways to embed a query in-KQL. `ai_embeddings` requires a provisioned external service. `slm_embeddings_fl()` proves a local model in the sandbox needs none — this is the pattern we adopt, with rlat's own encoder.

### 1.4 The Kusto Python sandbox

- The `python()` plugin runs user code in a sandbox that is **network-isolated**: *"A sandbox can't interact with any resource on the virtual machine (VM) or outside of it."* It cannot make outbound HTTP calls.
- The sandbox is **single-use** — disposed after each query; there is no warm cache.
- Memory: **1 GB** (Hyper-V sandboxes) default; CPU capped at 50% of host.
- **External artifacts**: files (packages, model files) are staged into the sandbox by the **Kusto engine** — not by sandbox code — gated by the cluster **callout policy**. The engine fetches; the sandbox only reads local files.
- The plugin is **disabled by default**; enabled per-Eventhouse via *Eventhouse → Plugins*.
- Docs: [Sandboxes (Kusto)](https://learn.microsoft.com/en-us/kusto/concepts/sandboxes?view=azure-data-explorer) · [Python plugin (Kusto)](https://learn.microsoft.com/en-us/kusto/query/python-plugin?view=microsoft-fabric) · [Enable the Python plugin in Real-Time Intelligence](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/python-plugin) · [Python package reference](https://learn.microsoft.com/en-us/kusto/query/python-package-reference?view=microsoft-fabric).

**Consequence**: a KQL function cannot call out to the rlat UDF (no network). But the encoder ONNX *can* be staged as an external artifact from OneLake — engine-fetched, sandbox-local. This is the no-external-service embedding path.

### 1.5 Fabric AI functions

- The `ai.*` family: `ai.embed`, `ai.classify`, `ai.extract`, `ai.summarize`, `ai.generate_response`, `ai.similarity`, `ai.analyze_sentiment`, `ai.translate`, `ai.fix_grammar`.
- They attach to **pandas** DataFrames (in both the Python and PySpark runtimes) and to Spark DataFrames. They are **included in paid Fabric SKUs** with a small default model — no separate Azure OpenAI provisioning. Generative functions also support Foundry models (Claude, LLaMA); embeddings are Azure OpenAI.
- Docs: [AI functions overview](https://learn.microsoft.com/en-us/fabric/data-science/ai-functions/overview) · [AI functions enhancements (GA)](https://blog.fabric.microsoft.com/en-US/blog/29826/) · [Use Foundry tools in Fabric](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview).

**Consequence**: the consolidation notebook can use `ai.summarize` / `ai.classify` / `ai.extract` / `ai.generate_response` for distillation and insight generation with no provisioned service. They are an *optional enhancement* — the loop degrades gracefully without them.

### 1.6 Pure-Python notebooks

- Fabric **Python notebooks** run single-node (default 2 vCores / 16 GB), no Spark. **Polars, DuckDB, pandas, scikit-learn** are preinstalled.
- Docs: [Use the Python experience on a notebook](https://learn.microsoft.com/en-us/fabric/data-engineering/using-python-experience-on-notebook) · [Choosing between Python and PySpark notebooks](https://learn.microsoft.com/en-us/fabric/data-engineering/fabric-notebook-selection-guide).

**Consequence**: both notebooks (deploy, consolidation) are pure-Python with Polars / DuckDB. `ai.*` calls bridge via `.to_pandas()` at the AI-call boundary only. This matches the existing `notebooks/examples/fabric_build.ipynb`.

### 1.7 Rushed-claim corrections (recorded so they are not repeated)

| Earlier loose claim | Verified correction |
|---|---|
| "Just add a Fabric band to the `.rlat`." | The format records **one `backbone` per archive** (`store/metadata.py`); `BandInfo` has no per-band encoder field. A foreign-encoder band needs schema changes and a `FORMAT_VERSION` bump. |
| "The optimised band ≈ a Fabric band." | The optimised band is a learned **projection** of base (`base @ Wᵀ`), recomputed sub-second on refresh. A re-encoding is a different object — cannot be recomputed from base. |
| "`ai_embed_text` uses text-embedding-3-small." | It targets *whichever Azure OpenAI deployment is configured*; dimension is set via `ModelParameters`. Azure OpenAI Service only. |
| "The whole loop runs in Fabric, no external service." | The `ai_embeddings` KQL plugin **requires** a BYO Azure OpenAI deployment. Only the `python()`-sandbox path is truly external-service-free. |
| "Foundry / Claude can back the AI functions." | Only the *generative* AI functions. Embeddings are Azure OpenAI. |

### 1.8 Internal rlat surface (reuse — do not re-implement)

- [`src/resonance_lattice/store/metadata.py`](../../src/resonance_lattice/store/metadata.py) — `Metadata`, `BackboneInfo`, `BandInfo`. `FORMAT_VERSION = 4`. Backbone is per-archive.
- [`src/resonance_lattice/store/bands.py`](../../src/resonance_lattice/store/bands.py) / `store/archive.py` — band I/O; bands keyed by name in `metadata.bands`; base band `(N, 768)` L2-normalised float32.
- [`src/resonance_lattice/store/incremental.py`](../../src/resonance_lattice/store/incremental.py) — delta-apply refresh; content-hash drift detection.
- [`src/resonance_lattice/field/encoder.py`](../../src/resonance_lattice/field/encoder.py) — `Encoder`, gte-modernbert, ONNX runtime, CLS+L2.
- [`src/resonance_lattice/fabric/`](../../src/resonance_lattice/fabric/) — UDF runtime helpers (`bootstrap`, `lakehouse_loader`, `hf_loader`); the external-assistant consumer.
- [`src/resonance_lattice/optimise/`](../../src/resonance_lattice/optimise/) — to be removed (see Decision 5).
- [`docs/internal/HONEST_CLAIMS.md`](../../docs/internal/HONEST_CLAIMS.md) — records optimise: **+0.032 R@5 on Fabric docs, −0.042 / −0.043 nDCG@10 on two BEIR corpora**; the projected lift was *falsified*.
- [`docs/internal/STORE.md`](../../docs/internal/STORE.md) — `.rlat` v4 format spec. [`docs/internal/GROUNDING_MODEL.md`](../../docs/internal/GROUNDING_MODEL.md) — grounding-not-truth.

---

## 2. Decisions locked

1. **KQL database / Eventhouse is the data-agent façade.** It is the only data-agent source that stores vectors and computes similarity in-query (§1.1, §1.2). *Rejected*: lakehouse / semantic model — they cannot embed or do cosine in-query.

2. **`.rlat` stays canonical; Fabric is a compile target.** "rlat compiles, Fabric runs." The `.rlat` is the source of truth (chunking, hashing, drift, refresh); a deploy step projects it into Eventhouse; the Eventhouse table is a derived materialisation. *Rejected*: running rlat's retrieval engine as the Fabric runtime.

3. **No external services — embed the query in the `python()` sandbox with gte-modernbert.** This is the original "Path A". The encoder ONNX is staged as an external artifact from OneLake (§1.4). *Rejected*: "Path B" — `ai_embeddings` + a BYO Azure OpenAI deployment.

4. **Path B is reversed; there is no Fabric band.** Path B's only advantage was avoiding the sandbox, bought with an external service the manifesto forbids. The Eventhouse table holds the **gte-modernbert base band** directly. One encoder everywhere; the `.rlat` is byte-identical across all targets.

5. **Remove the `optimise/` machinery.** Independent of Fabric: [HONEST_CLAIMS.md](../../docs/internal/HONEST_CLAIMS.md) shows it is net-negative on public corpora and its projected lift was falsified. Blast radius is bounded (~13 call sites; see roadmap Phase 0).

6. **The learning loop is the lens + harness-memory designs deployed on Fabric primitives** — not a bespoke cache. The query log is harness memory; the earned FAQ is the insight layer; per-team perspective is the lens. See [lensed-knowledge-architecture.md](lensed-knowledge-architecture.md) and [agent-harness-architecture.md](agent-harness-architecture.md).

7. **Pure-Python notebooks** (Polars / DuckDB), never PySpark (§1.6).

8. **The data agent owns the multi-hop loop.** A data agent is already a plan→retrieve→refine→synthesize loop. `rlat_search()` is one hop; the agent calls it repeatedly. Deep-search runs **offline** to pre-build the insight layer, not as a server-side KQL loop.

9. **Latency trade-off accepted.** Per-query encoder init in the sandbox adds ~1–3 s (the `slm_embeddings_fl()` tax). Accepted "for now"; the semantic cache softens repeat queries.

10. **One `.rlat`, two consumers.** The data agent (KQL façade) and external LLM assistants (the shipped UDF, [fabric-udf-integration.md](fabric-udf-integration.md)) read the same knowledge model. The deploy pipeline keeps them consistent.

---

## 3. Architecture overview — "rlat compiles, Fabric runs"

```
  BUILD (rlat, local / CI)                COMPILE TARGET: FABRIC TENANT
  ────────────────────────                ─────────────────────────────────────
                                          ┌─────────────── OneLake ───────────┐
  sources ──rlat build──▶ corpus.rlat ───▶ │  corpus.rlat   gte-modernbert.onnx │
                          (gte-mb base     │  lens/*.json                       │
                           band, canonical)└────────────┬───────────────────────┘
                                                        │  deploy notebook (pure Python)
                                                        ▼
                          ┌──────────────────── Eventhouse (KQL DB) ───────────────┐
                          │  passages   (source + insight layers, Vector16)         │
                          │  memory_events   (13-field memory rows + query events)  │
                          │  outcomes        (outcome ledger)                       │
                          │  lens            (trust weights, stance per scope)      │
                          │  fn rlat_search(query, lens_id)  ── python() embed +    │
                          │                                    series_cosine_sim    │
                          └───────┬───────────────────────────────┬─────────────────┘
                                  │                               │
                   Fabric data agent (NL2KQL)        consolidation notebook (scheduled)
                                  │                               │  distil • insight • forget
                          end users / M365 Copilot      Activator: gap alerts, FAQ promotion
                                                                  │
  external LLM assistants ──▶ Fabric UDF (shipped) ──▶ same corpus.rlat
```

Build-time and governance are rlat's job (chunking, hashing, drift, refresh, the `.rlat` format). The Fabric runtime is native Fabric primitives. The two are joined by one deploy step.

---

## 4. The three layers

From [lensed-knowledge-manifesto.md](lensed-knowledge-manifesto.md), realised as rows in the Eventhouse `passages` table, distinguished by a `layer` column:

- **Source layer** — the corpus passages from the `.rlat` base band. Ground truth, hash-addressable. `layer = "source"`.
- **Insight layer** — earned answers from accepted use: cited, verdict-stated, grown by the consolidation notebook. `layer = "insight"`.
- **Lens** — a per-user / role / team overlay (trust weights, stance, private insights). Stored in the `lens` table; applied as native-KQL re-ranking. Carries **no corpus-specific identifiers** — trust weights are source-path patterns — so it is portable across corpora.

---

## 5. Build & deploy pipeline

1. **Build** — `rlat build <sources> -o corpus.rlat`. Unchanged. Produces the canonical `.rlat` with the gte-modernbert base band. Refresh via `rlat refresh` (delta-apply, drift detection).
2. **Stage** — upload `corpus.rlat` and the gte-modernbert ONNX encoder to OneLake. Add the OneLake path to the Eventhouse **callout policy** so the engine may fetch the encoder as an external artifact.
3. **Deploy** — a pure-Python notebook (`notebooks/examples/fabric_deploy.ipynb`):
   - reads `passages.jsonl` + `bands/base.npz` from the `.rlat` (rlat library API);
   - writes one row per passage into the Eventhouse `passages` table, `layer = "source"`, vector under a `Vector16` policy;
   - sets the table's sharding/merging policy for vector search (§1.2);
   - is idempotent and incremental — re-running after `rlat refresh` upserts only changed passages (keyed by `content_hash`).

The `.rlat` is never read at query time. It is the build artifact; the Eventhouse table is the runtime.

---

## 6. Eventhouse schema

### 6.1 `passages` — source + insight layers

| Column | Type | Notes |
|---|---|---|
| `passage_id` | `string` | stable; `idx` for source rows, ULID for insight rows |
| `layer` | `string` | `source` \| `insight` |
| `text` | `string` | passage content |
| `source_file` | `string` | corpus-relative path (drives lens trust patterns) |
| `char_start`, `char_end` | `int` | provenance offsets |
| `content_hash` | `string` | drift detection; insight rows store hashes of cited passages |
| `vector` | `dynamic` | 768-d gte-modernbert embedding, `Vector16`-encoded |
| `confidence` | `string` | insight rows: `low`\|`medium`\|`high`\|`verified` |
| `cited_passage_ids` | `dynamic` | insight rows: provenance chain |
| `created_utc` | `datetime` | |

### 6.2 `memory_events` — query log = harness memory

The 13-field memory schema from [agent-harness-architecture.md](agent-harness-architecture.md), plus query-event fields. One row per `rlat_search()` call.

| Column | Type | Notes |
|---|---|---|
| `row_id` | `string` | ULID |
| `event_utc` | `datetime` | recency → the *strength* axis |
| `query_text` | `string` | |
| `query_vector` | `dynamic` | for semantic-cache lookup |
| `lens_id` | `string` | |
| `intent_kind` | `string` | `debug`\|`design`\|`implement`\|`review`\|`explain`\|`refactor` (set by `ai.classify` in consolidation) |
| `retrieved_ids` | `dynamic` | passage ids returned |
| `top_score` | `real` | uniformly-low → corpus-gap signal |
| `cache_hit` | `bool` | insight-layer hit? |
| `level` | `string` | `event`\|`pattern`\|`learning`\|`principle` — the *development* axis |
| `recurrence_count` | `int` | incremented on near-duplicate queries |
| `criticality` | `string` | `low`\|`normal`\|`high`\|`severe` |
| `confidence` | `string` | the *truth* axis |
| `parent_ids` | `dynamic` | distillation lineage |
| `origin` | `string` | `auto`\|`distilled`\|`outcome_derived` |

### 6.3 `outcomes` — the outcome ledger

Append-only, one row per resolved query. `row_id`, `memory_row_id` (FK), `verdict` (`satisfied`\|`not_satisfied`\|`unknown`\|`pending`), `signal_source` (`mechanical`\|`user`\|`llm`), `event_utc`.

### 6.4 `lens` — per-scope perspective

`lens_id`, `scope` (`user`\|`role`\|`team`\|`project`), `trust_weights` (`dynamic`: source-path-pattern → float), `declared_stance` (string, optional), `created_utc`, `last_active_utc`. Mirrors the `lensed-knowledge-architecture.md` lens schema, minus corpus-specific identifiers.

---

## 7. `rlat_search()` — the stored function

A single KQL stored function. One hop. The data agent calls it (via NL2KQL) and may call it repeatedly.

```
rlat_search(query: string, lens_id: string = "", top_k: int = 8)
```

Steps:

1. **Embed** — `python()` plugin embeds `query` with the staged gte-modernbert ONNX. The encoder file arrives via `external_artifacts`; the sandbox reads it locally (no network). Output: a 768-d L2-normalised vector.
2. **Retrieve** — `series_cosine_similarity` between the query vector and `passages.vector`, over `layer in ("source","insight")`, `top top_k`.
3. **Lens re-rank** — if `lens_id` is set, join `lens.trust_weights` and `extend score = score * trust_weight_for(source_file)`. Pure KQL; no sandbox.
4. **Contradiction flag** — lightweight: surface passages above threshold whose text opposes the top hit (a simple signal the data agent can act on; full deliberation is the agent's job — Decision 8).
5. **Return** — passage `text`, `source_file`, offsets, `score`, `layer`, `confidence`. The data agent synthesises the grounded answer.
6. **Capture** — the function (or a paired update policy) writes a `memory_events` row. This is the *capture* operation of the learning loop.

Insight-layer rows are searched alongside source rows in step 2 — a strong insight hit *is* the semantic cache.

---

## 8. Query embedding in the sandbox

- The gte-modernbert ONNX encoder is staged in OneLake and referenced via the `python()` plugin's `external_artifacts`. The Kusto engine fetches it (callout policy must allow the OneLake path); the artifact is **cached on the node**, so steady-state per-query cost is process init, not download.
- The sandbox loads the ONNX with `onnxruntime`, tokenises, runs CLS pooling + L2 — the exact recipe of [`field/encoder.py`](../../src/resonance_lattice/field/encoder.py). The query vector therefore matches the `.rlat` base band exactly → **retrieval parity**.
- Sandbox memory: encoder (~150 MB) + `onnxruntime` fits the 1 GB limit because the band itself lives in the Eventhouse table, not the sandbox — the sandbox only embeds one short string.
- Cost: ~1–3 s per query (Decision 9). Pattern precedent: `slm_embeddings_fl()` (§1.3).

---

## 9. The learning loop

The closed loop from [agent-harness-architecture.md](agent-harness-architecture.md): `traces → memory → expertise → context → action → outcomes → traces`, deployed on Fabric primitives.

- **Capture** — `rlat_search()` writes a `memory_events` row per call (§7 step 6).
- **Recurrence** — a KQL update policy / materialised view increments `recurrence_count` when a near-duplicate query (high `series_cosine_similarity` to an existing `query_vector`) arrives. This is the *strength* axis.
- **Consolidation notebook** (scheduled, pure-Python — see roadmap Phase 7): reads `memory_events` with Polars/DuckDB and runs the harness operations —
  - *distil* — `ai.summarize` promotes recurring `event` rows → `pattern` → `learning` (the *development* axis), writing `parent_ids`;
  - *insight generation* — `ai.generate_response` synthesises an insight-layer passage from a cluster of accepted, high-score query→answer events, with citations; writes it to `passages` with `layer = "insight"`;
  - *classify* — `ai.classify` / `ai.extract` set `intent_kind`, `criticality`, `level`;
  - *forget* — removes weak rows (decay, redundant-after-promotion, falsified-by-outcomes, corpus-drift);
  - *drift invalidation* — insight rows whose `cited_passage_ids` hashes no longer match `source` rows are demoted/removed. Drift detection is free — it reuses rlat's content hashes.
- **Deep-search offline** — the consolidation notebook runs the rlat deep-search loop over the top recurring unanswered queries, pre-populating the insight layer. Deep-search is a corpus builder here, not a query path.
- **Activator** — rules on `memory_events`: uniformly-low `top_score` → corpus-gap alert; `recurrence_count` over threshold → FAQ-promotion candidate; `not_satisfied` outcomes → demote an insight row.
- **Gap dashboard** — a Power BI report over `memory_events` / `outcomes`: cache-hit trend, retrieval-quality trend, top questions, the live gap list. Docs: [Data Activator](https://learn.microsoft.com/en-us/fabric/data-activator/data-activator-introduction).

---

## 10. The lens in Fabric

- A lens is a small artifact — a row in the `lens` table (and/or a JSON file in OneLake). Corpus-agnostic by design.
- Applied as **native KQL re-ranking** in `rlat_search()` step 3 — `score * trust_weight`. No sandbox, no Python.
- **Portability**: because a lens carries only source-path patterns, the same lens file applies to a different corpus's Eventhouse table. In a tenant, a team's lens travels from the Fabric-docs knowledge model to that team's own-codebase knowledge model.
- **Composition**: a workspace lens is a KQL aggregation over member lenses.
- Lens scope maps to Fabric structure: `team` ≈ workspace, `role`/`user` ≈ Entra group / principal. (Exact mapping — see roadmap open questions.)

---

## 11. The two consumers

| | Fabric data agent | External LLM assistants |
|---|---|---|
| Surface | KQL database source → NL2KQL → `rlat_search()` | Fabric UDF + `fabric://` skill (shipped — [fabric-udf-integration.md](fabric-udf-integration.md)) |
| Multi-hop loop | the data agent | the `deep-research` skill, client-side |
| Synthesis | the data agent's LLM | the assistant |
| Source of truth | the same `corpus.rlat` in OneLake | the same `corpus.rlat` in OneLake |

The deploy pipeline (§5) is the single point that keeps both consistent: it both populates the Eventhouse table and is the `.rlat` the UDF serves.

---

## 12. Honest limits

- **Latency** — ~1–3 s per-query encoder init in the sandbox. The only way to remove it is `ai_embeddings` + Azure OpenAI, i.e. an external service. Accepted (Decision 9).
- **Python plugin prerequisite** — must be enabled per-Eventhouse; some security-locked tenants disable it. Lighter than provisioning Azure OpenAI, but a real prerequisite.
- **Sandbox memory** — 1 GB. Fine because the sandbox only embeds the query; the band is in the table.
- **Scale ceiling** — ~1 M vectors per Eventhouse table with tuned sharding. Beyond that, retrieval slows (brute-force, no ANN).
- **Lens compounding** — a benchmark-gated claim; not quantified until the lensed-knowledge dogfood passes (manifesto Falsifiable claim).
- **`ai.*` in consolidation** — included in the Fabric SKU but consumes capacity; the loop is designed to degrade gracefully without them.
