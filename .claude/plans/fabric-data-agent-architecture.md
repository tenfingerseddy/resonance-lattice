# rlat × Fabric Data Agent — Architecture

**Status**: v1. The research foundation, locked decisions, mechanism, and schemas.
**Last updated**: 2026-05-17 (companions list updated 2026-05-24).
**Companions**: [fabric-data-agent-manifesto.md](fabric-data-agent-manifesto.md) (thesis, value claims, scope); [fabric-data-agent-goal-and-success.md](fabric-data-agent-goal-and-success.md) (the contract); [fabric-data-agent-roadmap.md](fabric-data-agent-roadmap.md) (build sequence); [fabric-data-agent-methodology.md](fabric-data-agent-methodology.md) (operating manual); [fabric-data-agent-research.md](fabric-data-agent-research.md) (research record — §1 below is the seed; further findings land there).

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

**Consequence** *(revised 2026-05-25 — see Amendment)*: **Fabric SQL DB is the chosen source-layer façade** (data agent uses NL2SQL). It's not in the §1.1 list above because the Fabric Data Agent product gained SQL DB as a source type after the original list was written; SQL DB has its own native `VECTOR(768)` type and `VECTOR_DISTANCE` function and can call out to REST endpoints via `sp_invoke_external_rest_endpoint`, both of which the source-layer architecture relies on. **Eventhouse retained for the learning loop** (memory_events + Activator integration, §6.2-§6.4, §9).

### 1.2 Eventhouse vector search

- KQL function **`series_cosine_similarity()`** computes vector similarity. Vectors are stored in a `dynamic` column.
- The **`Vector16`** encoding stores floats as Bfloat16 — 4× smaller, similarity functions "orders of magnitude" faster.
- Search is **brute-force** — there is no ANN / HNSW index. It parallelises across shards. Documented scale target: **~1 million vectors** with `Vector16` plus tuned sharding/merging policies (e.g. ≤3,125 rows/shard on a 20-node cluster).
- Docs: [Vector similarity search with Eventhouse](https://blog.fabric.microsoft.com/en-US/blog/empowering-real-time-searches-vector-similarity-search-with-eventhouse/) · [Vector database (Fabric)](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/vector-database) · [Eventhouse as a vector database tutorial](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/vector-database-eventhouse).

**Consequence** *(revised 2026-05-25 — see Amendment)*: this remains the foundation for the **learning-loop substrate** (memory_events near-duplicate detection via `series_cosine_similarity`, recurrence aggregations, outcome ledgers). The **source-layer cosine search** moved to Fabric SQL DB's native `VECTOR_DISTANCE` (measured 18-23 ms warm on 1,563 rows; same exact brute-force semantics, different surface).

### 1.3 KQL embedding paths — both are Azure-OpenAI-locked

- **`ai_embeddings` / `ai_embed_text`** plugins generate embeddings inside a KQL query. Connection string is an Azure OpenAI deployment URI; the `azure_openai` callout policy is regex-locked to `*.openai.azure.com` / `cognitiveservices.azure.com` / `services.ai.azure.com` domains. Embedding model and dimension are configurable via `ModelParameters`. Requires the `Cognitive Services OpenAI User` role. `RecordsPerRequest` defaults to **1**; Azure OpenAI embeddings are "subject to heavy throttling".
- **`slm_embeddings_fl()`** runs an embedding model (`e5-small-v2`) **locally inside the Eventhouse Python sandbox** via the `python()` plugin — **no external service**.
- Docs: [ai_embeddings plugin](https://learn.microsoft.com/en-us/kusto/query/ai-embeddings-plugin?view=microsoft-fabric) · [ai_chat_completion plugin](https://learn.microsoft.com/en-us/kusto/query/ai-chat-completion-plugin?view=microsoft-fabric) · [New OpenAI plugins for Eventhouse](https://blog.fabric.microsoft.com/en-US/blog/introducing-new-openai-plugins-for-eventhouse-preview/) · [SLM embeddings in Eventhouse](https://blog.fabric.microsoft.com/en-US/blog/create-embeddings-in-fabric-eventhouse-with-built-in-small-language-models-slms/).

**Consequence** *(revised 2026-05-25 — see Amendment)*: neither of these in-KQL embedding paths is used for the source-layer data-agent embed. `ai_embeddings` is rejected because the manifesto forbids external services. `slm_embeddings_fl()`-style sandbox embed measured 18.7 s warm per call on F2 — structurally unusable for interactive UX. The chosen embed path is the **existing Fabric UDF** (warm Python process, gte-modernbert in memory, ~100 ms per call) called from a T-SQL stored procedure via `sp_invoke_external_rest_endpoint`. Both `ai_embeddings` and `slm_embeddings_fl` remain valid research foundation for potential future learning-loop work (e.g., embedding inside Eventhouse for memory_events processing where the per-call cost amortises across a batch).

### 1.4 The Kusto Python sandbox

- The `python()` plugin runs user code in a sandbox that is **network-isolated**: *"A sandbox can't interact with any resource on the virtual machine (VM) or outside of it."* It cannot make outbound HTTP calls.
- The sandbox is **single-use** — disposed after each query; there is no warm cache.
- Memory: **1 GB** (Hyper-V sandboxes) default; CPU capped at 50% of host.
- **External artifacts**: files (packages, model files) are staged into the sandbox by the **Kusto engine** — not by sandbox code — gated by the cluster **callout policy**. The engine fetches; the sandbox only reads local files.
- The plugin is **disabled by default**; enabled per-Eventhouse via *Eventhouse → Plugins*.
- Docs: [Sandboxes (Kusto)](https://learn.microsoft.com/en-us/kusto/concepts/sandboxes?view=azure-data-explorer) · [Python plugin (Kusto)](https://learn.microsoft.com/en-us/kusto/query/python-plugin?view=microsoft-fabric) · [Enable the Python plugin in Real-Time Intelligence](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/python-plugin) · [Python package reference](https://learn.microsoft.com/en-us/kusto/query/python-package-reference?view=microsoft-fabric).

**Consequence** *(revised 2026-05-25 — see Amendment)*: the no-network sandbox property is the structural reason the source-layer embed cannot live in a KQL function. The Fabric SQL DB's `sp_invoke_external_rest_endpoint` (which DOES have network and a documented allowlist that empirically permits `*.userdatafunctions.fabric.microsoft.com`) is the bridge from the data-agent's source query to the UDF embed call. The `external_artifacts` path through sandbox python() works (G1 PASS) but the per-call cost is 18.7 s — recorded as a research finding, not a chosen path.

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

1. **Fabric SQL DB is the source-layer façade for the data agent.** *(Revised 2026-05-25 — see Amendment.)* SQL DB's native `VECTOR(768)` type + `VECTOR_DISTANCE` function provide in-query cosine search; a T-SQL stored procedure `dbo.rlat_search(@query)` calls the existing Fabric UDF for embedding via `sp_invoke_external_rest_endpoint`. The data agent uses NL2SQL. **Eventhouse is retained** for the learning loop (memory_events, outcomes, lens — §6.2, §6.3, §6.4). *Rejected*: KQL database as the source surface — `python()`-sandbox per-call cost (measured 18.7 s warm) exceeds usable interactive UX.

2. **`.rlat` stays canonical; Fabric is a compile target.** "rlat compiles, Fabric runs." The `.rlat` is the source of truth (chunking, hashing, drift, refresh); a deploy step projects it into the Fabric SQL DB `passages` table; the SQL DB table is a derived materialisation. *Rejected*: running rlat's retrieval engine as the Fabric runtime.

3. **No external services — embed via the existing Fabric UDF.** *(Revised 2026-05-25 — see Amendment.)* The data-agent embed step runs in the shipped Fabric UDF (`rlat-search`, [fabric-udf-integration.md](fabric-udf-integration.md)) which holds gte-modernbert ONNX warm. The UDF gets a tiny new `embed(query) -> vec[768]` function (~10 lines) so the SQL stored procedure can request just the vector. *Rejected*: `python()` sandbox embed (Decision 3 original) — measured 18.7 s warm per call, structurally bounded by `external_artifacts` resolution overhead; *Rejected*: `ai_embeddings` + Azure OpenAI (external service the manifesto forbids).

4. **There is no Fabric band; same gte-modernbert base band everywhere.** The SQL DB `passages` table holds the same float32 vectors that the local `.rlat` archive holds in `bands/base.npz` and that the UDF returns. One encoder everywhere; the `.rlat` is byte-identical across all targets.

5. **Remove the `optimise/` machinery.** Independent of Fabric: [HONEST_CLAIMS.md](../../docs/internal/HONEST_CLAIMS.md) shows it is net-negative on public corpora and its projected lift was falsified. Blast radius is bounded (~13 call sites; see roadmap Phase 0). **CLOSED AS NO-OP 2026-05-24**: P0 premise gate found optimise is net-positive on Fabric docs (+0.032 R@5); the negative numbers were misattributed BEIR corpora. Decision stands; the machinery survives.

6. **The learning loop is the lens + harness-memory designs deployed on Fabric primitives** — not a bespoke cache. The query log is harness memory; the earned FAQ is the insight layer; per-team perspective is the lens. The learning loop continues to use **Eventhouse + Eventstream + Activator** (§6.2-§6.4, §9). See [lensed-knowledge-architecture.md](lensed-knowledge-architecture.md) and [agent-harness-architecture.md](agent-harness-architecture.md).

7. **Pure-Python notebooks** (Polars / DuckDB), never PySpark (§1.6).

8. **The data agent owns the multi-hop loop.** A data agent is already a plan→retrieve→refine→synthesize loop. `dbo.rlat_search()` is one hop; the agent calls it repeatedly. Deep-search runs **offline** to pre-build the insight layer, not as a server-side T-SQL loop.

9. **Latency measured and bounded.** *(Revised 2026-05-25 — see Amendment.)* Per-query end-to-end: **~600-1000 ms typical, ~3 s tail, ~5 s cold**. Decomposition (measured 2026-05-25): UDF embed call from SQL DB via `sp_invoke_external_rest_endpoint` 379-2749 ms (median ~600 ms, one 4.7 s outlier), native `VECTOR_DISTANCE` cosine top-10 on 1,563 rows 18-23 ms. The python() sandbox path's 18.7 s warm cost is no longer relevant. Phase 6 semantic cache softens the tail further (existing design).

10. **One `.rlat`, two consumers — and one warm encoder.** *(Strengthened 2026-05-25 — see Amendment.)* The data agent (Fabric SQL DB façade) and external LLM assistants (the shipped UDF) read the same knowledge model AND share literally the same encoder process for query embedding. Encoder parity is mechanical, not just bit-equivalence. The deploy pipeline keeps the SQL DB `passages` table and the `.rlat` artefact consistent.

---

## 3. Architecture overview — "rlat compiles, Fabric runs" *(revised 2026-05-25)*

```
  BUILD (rlat, local / CI)                COMPILE TARGET: FABRIC TENANT
  ────────────────────────                ─────────────────────────────────────
                                          ┌─────────────── OneLake ───────────┐
  sources ──rlat build──▶ corpus.rlat ───▶ │  corpus.rlat                       │
                          (gte-mb base     │  (one canonical artefact;          │
                           band, canonical)│   UDF reads it for embed-and-search│
                                           │   deploy notebook reads it for     │
                                           │   the SQL DB passages projection)  │
                                           └─────┬────────────────────┬─────────┘
                                                 │                    │
                                  deploy notebook│                    │ UDF (warm)
                                  (pure Python)  │                    │ loads rlat
                                                 ▼                    │ + gte-modernbert
              ┌──────────── Fabric SQL DB (source layer) ───────────┐ │ on cold start
              │ passages   — VECTOR(768), source_file, char_offset, │ │
              │              content_hash, drift_status (insert via │ │
              │              CAST(N'[…]' AS VECTOR(768)))           │ │
              │ rlat_search(@query)  — T-SQL stored procedure:      │ │
              │     sp_invoke_external_rest_endpoint ──HTTPS──┐     │ │
              │     JSON_VALUE qvec; VECTOR_DISTANCE('cosine')│     │ │
              │     TOP @top_k by ASC; emit capture event     │     │ │
              └─────────────────────────────────────┬─────────┘     │ │
                                                    │               │ │
                                                    │POST embed     │ │
                                                    ▼               │ │
              ┌──────────────────────── Fabric UDF (rlat-search) ───┴─┘
              │ embed(query) -> list[float]   — warm gte-modernbert  │
              │ search(...)                   — existing, for external│
              │                                 assistants (Claude    │
              │                                 Code via fabric://)   │
              └──────────────┬────────────────────────┬───────────────┘
                             │                        │
        ┌────────────────────┴────────┐   ┌───────────┴───────────────────┐
        │ Fabric data agent           │   │ external LLM assistants       │
        │  (NL2SQL → rlat_search)     │   │  Claude Code / Cursor          │
        │  end users / Copilot        │   │  rlat search fabric://         │
        └─────────────────────────────┘   └───────────────────────────────┘

  ────────────────────────────────────────────────────────────────────────
  LEARNING LOOP (unchanged — Eventhouse retained)
  ────────────────────────────────────────────────────────────────────────
                                                ┌─ Eventstream — capture spine ──┐
              rlat_search emits event ─────────▶│   ├─▶ Eventhouse                │
                                                │   │     memory_events_raw       │
                                                │   │     memory_events (mv)      │
                                                │   │     outcomes                │
                                                │   │     lens (v2)               │
                                                │   └─▶ Activator                 │
                                                │         (gap alerts, FAQ)       │
                                                └────────────┬───────────────────┘
                                                             │
                                            consolidation notebook (scheduled)
                                            distil • insight • forget
```

Build-time and governance are rlat's job (chunking, hashing, drift, refresh, the `.rlat` format). The Fabric runtime is native Fabric primitives. The two are joined by one deploy step.

**Two surfaces, one encoder**: the data agent's source-layer embed and the external-assistant embed both run in the same warm UDF process, hitting the same gte-modernbert weights, returning the same float32 vectors. Encoder parity is mechanical, not just bit-equivalent.

**Why Eventstream is the spine for the learning loop**: Activator cannot subscribe directly to a KQL table (research entry 2026-05-24). The supported pattern is the source-layer stored procedure emits the capture event to Eventstream, which tees it to both the Eventhouse `memory_events` destination AND the Activator destination. The consolidation notebook reads only the materialised Eventhouse view (not the stream), keeping its pure-Polars/DuckDB shape.

---

## 4. The three layers

From [lensed-knowledge-manifesto.md](lensed-knowledge-manifesto.md), realised as rows in the Eventhouse `passages` table, distinguished by a `layer` column:

- **Source layer** — the corpus passages from the `.rlat` base band. Ground truth, hash-addressable. `layer = "source"`.
- **Insight layer** — earned answers from accepted use: cited, verdict-stated, grown by the consolidation notebook. `layer = "insight"`.
- **Lens** — a per-user / role / team overlay (trust weights, stance, private insights). Stored in the `lens` table; applied as native-KQL re-ranking. Carries **no corpus-specific identifiers** — trust weights are source-path patterns — so it is portable across corpora.

---

## 5. Build & deploy pipeline *(revised 2026-05-25)*

1. **Build** — `rlat build <sources> -o corpus.rlat`. Unchanged. Produces the canonical `.rlat` with the gte-modernbert base band. Refresh via `rlat refresh` (delta-apply, drift detection).
2. **Stage** — upload `corpus.rlat` to OneLake (`Files/rlat/<kmName>.rlat`). This is the **only** OneLake stage step under the amendment — no encoder/tokenizer/wheel staging is required because the UDF holds the encoder warm (it fetches gte-modernbert from HuggingFace on cold start and caches it on `/tmp` per [src/resonance_lattice/fabric/_runtime.py](../../src/resonance_lattice/fabric/_runtime.py)). The OneLake-encoder-staging path (P2 work) is **retired** as the active design but the scripts remain in tree as the supported fallback if the UDF surface ever becomes unavailable.
3. **Deploy** — a pure-Python notebook (`notebooks/examples/fabric_deploy.ipynb`):
   - reads `passages.jsonl` + `bands/base.npz` from the `.rlat` (rlat library API);
   - writes one row per passage into the Fabric SQL DB `dbo.passages` table — columns: `passage_id`, `passage_idx`, `text`, `source_file`, `char_offset`, `char_length`, `content_hash`, `vec VECTOR(768)`, `confidence`, `created_utc`;
   - INSERTs use inline literal form `CAST(N'[v1,v2,…,v768]' AS VECTOR(768))` (verified pattern — `pyodbc` binds JSON strings as `ntext` which has no implicit cast to `VECTOR`; inline literal sidesteps this);
   - **idempotence** is content-hash–based: the deploy procedure does an UPSERT-style MERGE on `content_hash` (T-SQL has native MERGE, unlike KQL). Re-running after `rlat refresh` updates only the changed rows;
   - the Eventhouse `passages_raw` table (P1 deliverable, 271 rows from `team-docs.rlat`) is retained as **sample data for the learning-loop work** but is no longer the data-agent source surface.

The `.rlat` is never read at query time. It is the build artifact; the SQL DB `passages` table is the runtime source-layer surface. The Eventhouse retains `memory_events`, `outcomes`, `lens` (§6.bis below) for the learning loop.

---

## 6. Source-layer schema — Fabric SQL DB *(revised 2026-05-25)*

### 6.1 `dbo.passages` — source + insight layers

T-SQL has native MERGE, so the storage is a single table (no view-dedupe trick needed).

```sql
CREATE TABLE dbo.passages (
    passage_idx       INT             NOT NULL,
    passage_id        NVARCHAR(64)    NOT NULL,
    layer             NVARCHAR(16)    NOT NULL DEFAULT N'source',  -- 'source' | 'insight'
    text              NVARCHAR(MAX)   NOT NULL,
    source_file       NVARCHAR(512)   NOT NULL,
    char_offset       INT             NOT NULL,
    char_length       INT             NOT NULL,
    content_hash      NVARCHAR(128)   NOT NULL,
    vec               VECTOR(768)     NOT NULL,
    confidence        NVARCHAR(16)        NULL,                    -- 'low'|'medium'|'high'|'verified'
    cited_passage_ids NVARCHAR(MAX)       NULL,                    -- JSON array; insight rows only
    created_utc       DATETIMEOFFSET  NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_passages PRIMARY KEY (content_hash)
);
CREATE INDEX ix_passages_source_file ON dbo.passages (source_file) INCLUDE (passage_idx);
CREATE INDEX ix_passages_layer ON dbo.passages (layer);
```

Query time reads `dbo.passages` directly. Deploy uses `MERGE INTO dbo.passages USING (VALUES (...)) AS src ON tgt.content_hash = src.content_hash WHEN MATCHED THEN UPDATE ... WHEN NOT MATCHED THEN INSERT ...`. The `content_hash` primary key gives hash-keyed idempotence; the deploy notebook does not need to know whether each row is new or updated.

**VECTOR_DISTANCE index**: Fabric SQL DB native vector type supports brute-force cosine via `VECTOR_DISTANCE('cosine', vec_a, vec_b)`. Measured 18-23 ms warm on 1,563 rows; should scale roughly linearly for our v1 corpora (≤67K passages — fabric-docs-rlat v2 size). ANN index policy is deferred to v2; the v1 floor is exact brute-force matching local `rlat search` cosine.

## 6.bis Learning-loop schema — Eventhouse *(unchanged from original §6.2-§6.4)*

`memory_events_raw` (append-only) + `memory_events` (materialised view = `arg_max(ingest_time, *) by row_id`), `outcomes`, and `lens` (v2) remain in the Eventhouse KQL database (item: `rlat-data-agent`). The Eventstream → Eventhouse + Activator capture spine (§9.1) is unchanged. The consolidation notebook (§9.2) reads the Eventhouse `memory_events` view via Polars/DuckDB.

The only schema change vs the original design: insight rows (`layer = 'insight'`) live in `dbo.passages` (SQL DB) alongside source rows — same MERGE-by-content_hash flow. The consolidation notebook writes new insight rows back to the SQL DB instead of to an Eventhouse table.

(For full Eventhouse-side columns, see the original §6.2 `memory_events`, §6.3 `outcomes`, §6.4 `lens` — those table definitions stand unchanged.)

### 6.2 `memory_events` — query log = harness memory

The 13-field memory schema from [agent-harness-architecture.md](agent-harness-architecture.md), plus query-event fields. One row per `rlat_search()` call.

| Column | Type | Notes |
|---|---|---|
| `row_id` | `string` | ULID |
| `event_utc` | `datetime` | recency → the *strength* axis |
| `query_text` | `string` | |
| `query_vector` | `dynamic` | for semantic-cache lookup |
| `lens_id` | `string` | v1 default: `"default"` (lens deferred to v2 — §10). v2 distillation re-keys pre-v2 rows by inferring lens from `source_file` patterns. |
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

## 7. `dbo.rlat_search` — the T-SQL stored procedure *(revised 2026-05-25)*

A single T-SQL stored procedure on the Fabric SQL DB. One hop. The data agent calls it (via NL2SQL → `EXEC dbo.rlat_search`) and may call it repeatedly.

```sql
CREATE OR ALTER PROCEDURE dbo.rlat_search
    @query        NVARCHAR(MAX),
    @top_k        INT = 8,
    @lens_id      NVARCHAR(64) = N''  -- unused in v1; v2 plugs in here
AS
BEGIN
    -- 1. Embed via UDF (warm Python process; ~100-600 ms typical)
    DECLARE @response NVARCHAR(MAX);
    -- Bearer token embedded inline as @headers (see §5 for why @credential
    -- is blocked on Fabric SQL DB today). Token is rotated by CREATE OR
    -- ALTER PROCEDURE on each refresh cycle (scripts/fabric_e2e_init_sqldb.py).
    DECLARE @headers NVARCHAR(MAX) = N'{"Authorization":"Bearer <token>"}';
    EXEC sp_invoke_external_rest_endpoint
         @url      = N'<udf-base>/functions/embed/invoke',
         @method   = N'POST',
         @headers  = @headers,
         @payload  = (SELECT @query AS query FOR JSON PATH, WITHOUT_ARRAY_WRAPPER),
         @timeout  = 230,
         @response = @response OUTPUT;

    -- 2. Parse the qvec from the UDF response and cast to VECTOR(768)
    DECLARE @qvec VECTOR(768) = CAST(
         JSON_QUERY(@response, '$.result.output')
         AS VECTOR(768));

    -- 3. Cosine top-k over source + insight layers
    SELECT TOP (@top_k)
        passage_id, source_file, char_offset, char_length,
        content_hash, text, layer, confidence,
        1.0 - VECTOR_DISTANCE('cosine', vec, @qvec) AS score
    FROM dbo.passages
    WHERE layer IN (N'source', N'insight')
    ORDER BY VECTOR_DISTANCE('cosine', vec, @qvec) ASC;

    -- 4. Emit capture event to Eventstream for the learning loop (§9.1)
    --    (separate sp_invoke_external_rest_endpoint to the Eventstream ingest endpoint)
END
```

Steps explained:

1. **Embed** — `sp_invoke_external_rest_endpoint` POSTs to the UDF `embed(query)` function. The UDF runs gte-modernbert ONNX (warm), returns the 768-d L2-normalised vector. Auth: bearer token embedded **inline as `@headers`** in the procedure body — the documented `DATABASE SCOPED CREDENTIAL WITH IDENTITY = 'HTTPEndpointHeaders'` pattern is blocked because SP-initiated CREATE CREDENTIAL needs server identity (probed 2026-05-25; see §5). Token rotation: re-run `scripts/fabric_e2e_init_sqldb.py` (CREATE OR ALTER PROCEDURE embeds the fresh token). SP tokens for the Power BI scope have ~1 hour TTL.
2. **Parse** — extract the vector from the UDF's JSON response (`$.result.output` is the float array; Fabric UDF wire format wraps under `result.output`); cast to `VECTOR(768)`.
3. **Retrieve** — native `VECTOR_DISTANCE('cosine', ...)` brute-force top-k over `dbo.passages` (both source and insight layers); ~20 ms warm at v1 scale.
4. **Lens re-rank** — (v2 — deferred) when `@lens_id` is non-empty, join the `lens` table from Eventhouse via cross-database query (preview feature; alternative: cache lens trust weights as a SQL DB lookup table). v1 ignores `@lens_id`.
5. **Capture** — separate `sp_invoke_external_rest_endpoint` call to the Eventstream ingest endpoint (or to a workspace-bound destination), payload = the capture event row. Same allowlist permits `*.fabric.microsoft.com`. The capture path keeps the learning loop's Eventstream → Eventhouse + Activator wiring intact (§9.1).
6. **Return** — passage rows with `text`, `source_file`, offsets, `score`, `layer`, `confidence`. The data agent synthesises the grounded answer with citations.

Insight-layer rows are searched alongside source rows in step 3 — a strong insight hit *is* the semantic cache.

**Per-call latency budget** (measured 2026-05-25 on F2):

| Step | Time | Notes |
|---|---|---|
| sp_invoke → UDF embed → response | 379-2749 ms (median ~600 ms) | UDF warm; SQL DB → REST overhead is the dominant cost |
| JSON parse + CAST VECTOR(768) | <10 ms | T-SQL native |
| `VECTOR_DISTANCE` TOP @top_k | 18-23 ms | brute-force; scales near-linearly with row count |
| Capture event emit (async target) | ~50-100 ms estimated | not yet measured |
| **End-to-end** | **~600-1000 ms typical; ~3 s tail; ~5 s cold** | usable interactive UX |

---

## 8. Query embedding in the UDF *(revised 2026-05-25)*

- The gte-modernbert ONNX encoder lives in the existing Fabric UDF's process. Loaded on cold start via `huggingface_hub.snapshot_download` (revision-pinned, cached on `/tmp/rlat`); resident across calls in the warm UDF Python process (LRU(8) cache in [src/resonance_lattice/fabric/_runtime.py](../../src/resonance_lattice/fabric/_runtime.py)).
- The UDF gains a new top-level function `embed(query: str) -> list[float]` (alongside the existing `search` and `list_kms`) — ~10 lines: tokenize → ONNX inference → CLS pool → L2 normalize → return the float list. The recipe matches [`field/encoder.py`](../../src/resonance_lattice/field/encoder.py) exactly, so the query vector matches the `.rlat` base band exactly → **retrieval parity (G1 PASS, cosine 1.000000 — already verified 2026-05-24 via the UDF for `search`; `embed` shares the same encoder process so the parity property transfers mechanically).**
- Per-call cost: ~100 ms warm in the UDF process; ~5-10 s on cold start (HF download + encoder load). The UDF's LRU(8) means even with multiple KMs in rotation, the encoder weights stay resident.
- The python() sandbox path (Decision 3 original, P2 deliverable) is **retained in tree** as a research artefact but not used by the production source-layer path. The work that landed (encoder staged in OneLake, `embed_query()` KQL function on `rlat-data-agent`, G1 PASS at cosine 1.000000) is preserved; the function survives unchanged in case a future scenario requires sandbox embed (e.g., a Fabric tenant that disables UDF item creation).

---

## 9. The learning loop

The closed loop from [agent-harness-architecture.md](agent-harness-architecture.md): `traces → memory → expertise → context → action → outcomes → traces`, deployed on Fabric primitives.

### 9.1 Capture data flow — Eventstream is the spine

```
rlat_search() ──emit capture event──▶ Eventstream
                                         ├──▶ Eventhouse destination ──▶ memory_events_raw (table)
                                         │                                ├──▶ memory_events (mv: arg_max by row_id)
                                         │                                └──▶ recurrence_mv (mv: count near-duplicate vectors)
                                         └──▶ Activator destination ──▶ rules (gap alerts, FAQ promotion)
```

Activator cannot subscribe directly to a KQL table (research 2026-05-24). The Eventstream tee makes a single capture event reach both the durable table (for consolidation) and Activator (for reactivity), with no double-write contract in `rlat_search()`.

### 9.2 Loop operations

- **Capture** — `rlat_search()` emits a capture event to the Eventstream (§9.1).
- **Recurrence** — a KQL materialised view increments `recurrence_count` when a near-duplicate query (high `series_cosine_similarity` to an existing `query_vector`) arrives. This is the *strength* axis.
- **Consolidation notebook** (scheduled, pure-Python — see roadmap Phase 7): reads the `memory_events` view with Polars/DuckDB and runs the harness operations —
  - *distil* — `ai.summarize` promotes recurring `event` rows → `pattern` → `learning` (the *development* axis), writing `parent_ids`;
  - *insight generation* — `ai.generate_response` synthesises an insight-layer passage from a cluster of accepted, high-score query→answer events, with citations; writes it to `passages_raw` with `layer = "insight"` (the view's `arg_max` keeps the latest revision by content_hash);
  - *classify* — `ai.classify` / `ai.extract` set `intent_kind`, `criticality`, `level`;
  - *forget* — removes weak rows (decay, redundant-after-promotion, falsified-by-outcomes, corpus-drift);
  - *drift invalidation* — insight rows whose `cited_passage_ids` hashes no longer match `source` rows are demoted/removed. Drift detection is free — it reuses rlat's content hashes.
- **Deep-search offline** — the consolidation notebook runs the rlat deep-search loop over the top recurring unanswered queries, pre-populating the insight layer. Deep-search is a corpus builder here, not a query path.
- **Activator** — rules on the Eventstream Activator destination: uniformly-low `top_score` → corpus-gap alert; `recurrence_count` over threshold → FAQ-promotion candidate; `not_satisfied` outcomes → demote an insight row (via a UDF action — research 2026-05-24 — *UDF action is the cleanest fit for the demote-row write-back*).
- **Gap dashboard** — a Power BI report over `memory_events` / `outcomes`: cache-hit trend, retrieval-quality trend, top questions, the live gap list. Docs: [Data Activator](https://learn.microsoft.com/en-us/fabric/data-activator/data-activator-introduction).

### 9.3 Activator setup constraint

**Reflex (Activator) item CRUD does NOT support service principals** (research 2026-05-24). The setup wizard cannot SP-automate Activator. The HTML guide's `operations.html` includes step-by-step portal instructions for the user to create the rules manually. This is unavoidable until Microsoft adds SP support; documented as a `Microsoft roadmap-watch` item.

---

## 10. The lens in Fabric

**Note (2026-05-24, Goal & Success grilling outcome)**: lens is **deferred to v2**. The architectural design below stays accurate; v2 plugs in mechanically using the `lens_id` parameter on `rlat_search()` that ships in v1 (P3) but goes unused.

**v1 → v2 data-migration contract for `lens_id`**: In v1, every captured event in `memory_events` carries `lens_id = "default"` (the literal string). When v2 lens ships, the v2 distillation re-keys pre-v2 rows by inferring lens from `source_file` patterns (e.g. files under `Files/team-finance/*` → `lens_id = "team-finance"`). Pre-v2 recurrence/insight clusters built under `"default"` are NOT discarded; they're treated as the "tenant baseline" lens that all team lenses inherit from. This is the only state Carbon-cost v2 lens-adoption needs.

- A lens is a small artifact — a row in the `lens` table (and/or a JSON file in OneLake). Corpus-agnostic by design.
- Applied as **native KQL re-ranking** in `rlat_search()` step 3 — `score * trust_weight`. No sandbox, no Python.
- **Portability**: because a lens carries only source-path patterns, the same lens file applies to a different corpus's Eventhouse table. In a tenant, a team's lens travels from the Fabric-docs knowledge model to that team's own-codebase knowledge model.
- **Composition**: a workspace lens is a KQL aggregation over member lenses.
- Lens scope maps to Fabric structure: `team` ≈ workspace, `role`/`user` ≈ Entra group / principal. (Exact mapping — see roadmap open questions.)

---

## 11. The two consumers *(revised 2026-05-25)*

| | Fabric data agent | External LLM assistants |
|---|---|---|
| Surface | Fabric SQL DB source → NL2SQL → `EXEC dbo.rlat_search(@query)` | Fabric UDF `search()` directly via `fabric://` skill (shipped — [fabric-udf-integration.md](fabric-udf-integration.md)) |
| Embed runtime | UDF `embed()` (called from SQL stored proc via `sp_invoke_external_rest_endpoint`) | UDF `search()` (embed-and-search in one call) |
| Multi-hop loop | the data agent | the `deep-research` skill, client-side |
| Synthesis | the data agent's LLM | the assistant |
| Source of truth | the same `corpus.rlat` in OneLake (deploy notebook reads it and writes `dbo.passages`) | the same `corpus.rlat` in OneLake (UDF reads it directly at query time) |

The deploy pipeline (§5) keeps both consistent: it materialises `.rlat` → `dbo.passages` for the data-agent surface, and the UDF reads the same `.rlat` for the external-assistant surface. **Encoder parity is mechanical** — both surfaces send their text through the same UDF process holding the same gte-modernbert weights.

---

## Amendment — 2026-05-25 (approved by Kane 2026-05-25)

**Status**: **approved and authoritative**. Decisions 1, 3, 9 revised below; cascading edits applied to §§1-12 in this commit. Manifesto wording edits applied separately. The amendment block stays as the audit record of why and when the change happened.

**Trigger**: P2 measurement closed (G1 PASS, cosine = 1.000000 bit-exact). But the steady-state per-query latency for the `python()` sandbox embed path landed at **18.7 s warm / 21.4 s cold** — structurally bounded by Fabric's `external_artifacts` resolution overhead (~10-15 s per call regardless of artefact size), not by anything we control. Methodology §6 trigger: an empirical measurement falsifies the latency premise of Decision 9 ("accepted for v1 regardless of magnitude") in the strong sense — 18.7 s makes O2's grounded-answer outcome unusable in practice for cold/paraphrased questions.

### Empirical evidence supporting the change

All measured 2026-05-24 / 2026-05-25 against the `Kane-Test-Personal` workspace (F2, Australia East), recorded in detail in [fabric-data-agent-research.md](fabric-data-agent-research.md):

| Measurement | Value | Path |
|---|---|---|
| Python-direct UDF embed-and-search (the existing rlat-search UDF) | 917 ms warm | UDF (current shipped) |
| `sp_invoke_external_rest_endpoint` allowlist for `*.userdatafunctions.fabric.microsoft.com` | PASS (HTTP 401 in 84 ms with no auth) | Fabric SQL DB → UDF |
| End-to-end SQL DB → UDF round-trip (5 calls) | 379-2749 ms (median ~600 ms; one 4.7 s outlier on call 1) | Fabric SQL DB → UDF |
| Fabric SQL DB `VECTOR_DISTANCE('cosine', …)` on 1,563-row passages table | 18-23 ms warm | SQL DB native |
| Sandbox python() embed_query() | 18,700 ms warm; 21,400 ms cold | Current Decision 3 path |

The python() sandbox path works but is unusable for interactive data-agent traffic. The Fabric SQL DB + UDF path is **~20-30× faster** at the per-call level and uses the **same gte-modernbert encoder**, so G1 parity (cosine = 1.000000) is preserved.

### What changes

**Decision 1 — REVISED**:
- **Was**: KQL database / Eventhouse is the data-agent façade.
- **Becomes**: **Fabric SQL DB is the source-layer façade for the data agent.** It holds the passages table with native `VECTOR(768)` column and exposes a T-SQL stored procedure `dbo.rlat_search(@query)` that the data agent calls via NL2SQL. Eventhouse is **retained** for the learning loop (memory_events, outcomes, Activator wiring — §6.2, §6.3, §9 unchanged).

**Decision 3 — REVISED**:
- **Was**: No external services — embed the query in the python() sandbox with gte-modernbert.
- **Becomes**: **No external services — embed the query via the existing Fabric UDF** (`rlat-search`, already shipped — [fabric-udf-integration.md](fabric-udf-integration.md)), called from a T-SQL stored procedure via `sp_invoke_external_rest_endpoint`. The UDF runs the same gte-modernbert encoder used by the `.rlat` build pipeline, the local CLI, and external assistants — encoder parity is the constraint, not the runtime surface. The UDF gets a new tiny `embed(query) -> vec[768]` function (~5 lines); `search()` stays for the external-assistant consumer.

**Decision 4 — UNCHANGED**: There is no Fabric band; the table holds the gte-modernbert base band directly. (The same is true under the SQL DB schema — `VECTOR(768)` column holds the same float32 vectors that `Vector16`-encoded would in Eventhouse.)

**Decision 9 — REVISED**:
- **Was**: Latency trade-off accepted (~1-3 s per-query encoder init in the sandbox, unmeasured).
- **Becomes**: **Latency measured and bounded.** Typical per-query end-to-end: **~600-1000 ms** (UDF embed via SQL invoke + native `VECTOR_DISTANCE`). Tail: **~3 s** (95th percentile from 5-call samples; root cause is `sp_invoke_external_rest_endpoint` variance, not encoder). Cold-first call after extended idle: ~5 s. The python() sandbox path's 18.7 s is no longer the operating cost. The Phase 6 semantic cache softens the tail further (existing design).

**Decisions 2, 5, 6, 7, 8, 10 — UNCHANGED**:
- `.rlat` stays canonical (D2); optimise/ stays (D5 — P0 closed as no-op); lens still deferred to v2 (D6); pure-Python notebooks (D7); data agent owns multi-hop (D8); one `.rlat`, two consumers (D10 — and now the encoder is *literally* the same Python process for both consumers, strengthening the claim).

### What this preserves

| Manifesto Tier-1 claim | Status under Path #2 |
|---|---|
| Grounded, traceable answers (provenance via `source_file`, `char_offset`, `content_hash`) | ✓ preserved |
| One knowledge model, many consumers (same `.rlat`, both surfaces) | ✓ **strengthened** — the data-agent's embed step is now literally the same UDF the external assistants use, so encoder parity is mechanical, not just bit-equivalence |
| No external services (everything in Fabric tenant) | ✓ preserved (Fabric SQL DB is in-capacity; UDF is in-capacity; Eventhouse is in-capacity) |
| Data never leaves the tenant | ✓ preserved |
| Drift-aware (content hashes flag stale answers) | ✓ preserved |
| Learning loop compounds | ✓ preserved (Eventhouse + Activator retained for memory_events / Activator) |

### Architecture-cascading edits

If approved, the following sections require text updates (mechanical; logic survives):

| Section | Edit |
|---|---|
| §1.1 "Fabric data agents" — *Consequence* line | Was: "a KQL database is the chosen façade." → "a Fabric SQL DB is the chosen source-layer façade; KQL DB retained for the learning loop." |
| §1.2 "Eventhouse vector search" | Retain as research foundation; add note "used for memory_events recurrence (P5) and outcome aggregations, not source-layer search." |
| §1.3 "KQL embedding paths" | Add note: "The data-agent embed path is now the UDF, not these in-KQL options." Both stay as research foundation. |
| §1.4 "The Kusto Python sandbox" | Retain as research foundation; mark as not used in the source-layer path. |
| §3 Architecture diagram | Source-layer block becomes Fabric SQL DB; learning-loop block stays Eventhouse |
| §5 Build & deploy pipeline | Step 3 changes: deploy now writes the `passages` table to **Fabric SQL DB** (CREATE TABLE with `VECTOR(768)`; bulk INSERT with `CAST(N'[…]' AS VECTOR(768))`). The deploy notebook gains a SQL DB write path alongside the Eventhouse-write path used for `memory_events`. |
| §6 Eventhouse schema | Renames to "§6 Source-layer schema (Fabric SQL DB)" + "§6.bis Learning-loop schema (Eventhouse)". The `passages` table moves to SQL DB; `memory_events` / `outcomes` / `lens` stay in Eventhouse. |
| §7 `rlat_search()` | Becomes a T-SQL stored procedure. Step 1 calls UDF `embed` via `sp_invoke_external_rest_endpoint`; step 2 is `VECTOR_DISTANCE`; the capture event in step 6 still goes to Eventstream → Eventhouse `memory_events` + Activator. |
| §8 Query embedding in the sandbox | Becomes "§8 Query embedding in the UDF" — the embed runs in a warm Python process, not the per-query sandbox. The G1 parity gate runs against the UDF (already passes today). |
| §11 Two consumers table | "Surface" cell for the data agent changes from "KQL database source → NL2KQL → rlat_search()" to "Fabric SQL DB source → NL2SQL → dbo.rlat_search". Other rows unchanged. |
| §12 Honest limits | Latency entry updates with measured numbers. Sandbox-memory limit entry removed (not applicable). Scale-ceiling entry updates: Fabric SQL DB `VECTOR_DISTANCE` is brute-force at our scale; ANN index policy is a P10 question, not v1. |

### Manifesto edits required (smaller scope)

Two phrases in `fabric-data-agent-manifesto.md` reference "KQL database / Eventhouse" as the bridge:

- §"Motivation" para 4: "The bridge is a **KQL database / Eventhouse** — the one data-agent source that can natively store vectors and compute similarity in-query." → "The bridge is a **Fabric SQL DB** — its native `VECTOR` type and `VECTOR_DISTANCE` function provide in-query cosine search; a tiny T-SQL stored procedure calls the existing Fabric UDF to embed the query at sub-second latency. Eventhouse is retained for the learning loop's memory events and Activator wiring."
- §"North Star alignment" table, row 1: "The query encoder (gte-modernbert) runs locally inside the Eventhouse Python sandbox." → "The query encoder (gte-modernbert) runs in the Fabric UDF (in-tenant, in-capacity, warm process). The same UDF serves the data agent (via a T-SQL stored proc) and external assistants directly."

Manifesto Tier-1 claims all survive; "no external services" is preserved (UDF is first-party Fabric, in-capacity).

### Falsifiable-claim updates

| Gate | Was | Becomes |
|---|---|---|
| **G1 — parity** | Top-8 KQL `rlat_search()` == local top-8, cosine ≥ 0.999 | Top-8 from SQL DB `dbo.rlat_search()` == local top-8 (still bit-exact at the encoder; SQL `VECTOR_DISTANCE` is deterministic) |
| **G3 — setup time** | ≤ 30 min on demo corpus | unchanged target; setup-wizard adds a SQL DB provisioning step (~30s LRO) + inline-`@headers` procedure with an initial token embedded at CREATE OR ALTER PROCEDURE time. Server-assigned MI is NOT available on Fabric SQL DB today and even the documented HTTPEndpointHeaders credential is blocked for SP-initiated DDL (research + probe 2026-05-25); the wizard automates token refresh via a Data Pipeline cron so the user-visible flow stays one-shot. |
| **Latency soft target** | unmeasured "~3 s" | measured ~600-1000 ms typical, ~3 s tail, ~5 s cold |
| **G4 — consumer parity** | UDF and KQL function return same top-k | **strengthened** — both consumers literally share the same embed call to the UDF; difference is only `VECTOR_DISTANCE` (SQL DB) vs `series_cosine_similarity` (Eventhouse) over the same vectors, and the SQL DB path is the only one that uses the UDF embed |
| G2, G5, G6 | unchanged | unchanged |

### Open work before P3 ships under this amendment

1. Add `embed(query: str) -> list[float]` function to the UDF (`fabric/udf/function_app.py` and `src/resonance_lattice/fabric/_runtime.py`). ~10 lines.
2. SQL DB server-identity assignment — find the right portal / REST path (or document the explicit-bearer-token fallback that works today).
3. Write `dbo.rlat_search` T-SQL stored procedure that wraps `sp_invoke_external_rest_endpoint` → `embed` → `VECTOR_DISTANCE` → return top-K with provenance columns.
4. Update the deploy notebook (`fabric_deploy.ipynb`) to write the `passages` table to Fabric SQL DB (in addition to or instead of the Eventhouse `passages_raw` ingest done in P1).
5. P1 Eventhouse passages table is retained for the learning-loop substrate (P5-P9 unchanged). The 271 rows from team-docs.rlat that landed there during P1 stay where they are; they're not the data-agent source anymore but are useful sample data for `memory_events` testing.

### What needs explicit sign-off

1. **Decision 1 revision** — source-layer façade moves Eventhouse KQL DB → Fabric SQL DB.
2. **Decision 3 revision** — embed path moves python() sandbox → existing Fabric UDF (no manifesto Tier-1 claim change; just runtime change).
3. **Decision 9 revision** — latency replaced with measured numbers.
4. **Manifesto wording edits** (two phrases) — confirming "KQL database / Eventhouse" → "Fabric SQL DB + Fabric UDF".
5. **Cascading section rewrites** authorised in a follow-up commit (this amendment stays as the audit record).

If approved: I write the cascading edits as a single commit, restructure P3 in the roadmap to target the SQL DB stored procedure + UDF `embed` function, and proceed.

If not approved: the amendment stays here as a recorded option; we either accept the 18.7 s sandbox latency, fall back to Path #1 (AI Search + Custom Vectorizer + UDF), or revisit.

---

## 12. Honest limits *(revised 2026-05-25)*

- **Latency** — measured 2026-05-25: per-query end-to-end ~600-1000 ms typical (`sp_invoke_external_rest_endpoint` → UDF embed + `VECTOR_DISTANCE`), ~3 s tail (median+2σ from a 5-call sample), ~5 s cold (first call after extended idle, when UDF cold-starts the encoder). Phase 6 semantic cache softens the tail.
- **UDF prerequisite** — the Fabric UDF item must be provisioned and reachable from the SQL DB via `sp_invoke_external_rest_endpoint`. The allowlist for `api.fabric.microsoft.com` permits it without tenant changes (verified 2026-05-25). Setup-wizard step. Lighter than provisioning Azure OpenAI; both UDF and SQL DB are in-capacity on F2.
- **SQL DB credential auth** — Fabric SQL DB has no path today (REST, T-SQL, or portal) to enable a server-assigned MI on its underlying logical server (research 2026-05-25). Even the documented `IDENTITY = 'HTTPEndpointHeaders'` fallback fails because **SP-initiated `CREATE DATABASE SCOPED CREDENTIAL` triggers a server-side Entra-principal resolution that itself needs server identity** (probed 2026-05-25 — table DDL, DML, and `sp_invoke_external_rest_endpoint` calls all work from SP; only credential DDL fails). The shipping pattern is therefore **inline `@headers` in the procedure body**, with the bearer token embedded as a literal at `CREATE OR ALTER PROCEDURE` time. Rotation: re-run `scripts/fabric_e2e_init_sqldb.py` (or its P10 wizard equivalent) on a sub-hour cron via a Fabric Data Pipeline. SP tokens for the Power BI scope have ~1 hour TTL.
- **Scale ceiling** — Fabric SQL DB `VECTOR_DISTANCE` is brute-force at v1 scale (~18-23 ms on 1.5K rows, ~roughly linear to 67K passages — the fabric-docs-rlat v2 size). ANN index policy is a v2 question; v1 ships brute-force which preserves G1 parity exactly.
- **Lens compounding** — a benchmark-gated claim; not quantified until the lensed-knowledge dogfood passes (manifesto Falsifiable claim).
- **`ai.*` in consolidation** — included in the Fabric SKU but consumes capacity; the loop is designed to degrade gracefully without them.
- **Tail-latency variance** — the median 600 ms call sometimes spikes to 2.7-4.7 s. Root cause not yet isolated; `sp_invoke_external_rest_endpoint` connection establishment is the likely contributor. P4 latency probe under realistic load will characterise this.
