# rlat × Fabric Data Agent — Roadmap

**Status**: v2. The phased build sequence, premise gates, rigor binding, and acceptance — restructured to align with [fabric-data-agent-goal-and-success.md](fabric-data-agent-goal-and-success.md) and [fabric-data-agent-methodology.md](fabric-data-agent-methodology.md).
**Last updated**: 2026-05-24.
**Companions**: [manifesto](fabric-data-agent-manifesto.md), [architecture](fabric-data-agent-architecture.md), [goal & success](fabric-data-agent-goal-and-success.md), [methodology](fabric-data-agent-methodology.md), [research](fabric-data-agent-research.md).

---

## What this document is

The manifesto says *what* and *why*; the architecture says *how*; the goal & success contract says *done*; the methodology says *how the work runs*. **This roadmap says *when* and *in what order*** — with per-phase premise gates, rigor binding, HTML doc tasks, and Fabric Framework skill consults so any session can pick up any unstarted phase cold.

It is the most volatile of the planning docs; phases get re-scoped as engineering surfaces real constraints. The manifesto, architecture, and goal & success contract should not move with these changes.

---

## How to resume (session-resumption protocol)

A session continuing this work should:

1. Read [goal & success](fabric-data-agent-goal-and-success.md) end to end — re-confirm the contract.
2. Read [methodology](fabric-data-agent-methodology.md) §1 (operating principles) and §3 (the per-phase loop).
3. Read the manifesto §"Non-negotiables" and the architecture §2 "Decisions locked" — these are the invariants.
4. Find the first phase below whose **Status** is not `done`.
5. Enter the methodology §3 loop at step 1 (premise gate) for that phase.
6. Each phase has its own *Premise gate*, *Tasks*, *Acceptance (§D measurement)*, *HTML doc task*, and *Skill consults* — everything needed is there.

**Status legend**: `not started` · `in progress` · `blocked` · `done`. Update the line, in this file, as part of the phase's final commit.

**Phase dependency graph** (P8 deferred to v2 per grilling pass 2026-05-24):

```
P-1 ─▶ P0 ─▶ P1 ─▶ P2 ─▶ P3 ─▶ P4
                          │
                          └─▶ P5 ─▶ P6 ─▶ P7 ─▶ P9
                                                  │
                                                  ▼
                                                  P10 (after P4 + P7 + P9)

P8 (lens) — deferred to v2 release; the architecture's `lens_id` parameter
in `rlat_search()` stays in place so v2 plugs in mechanically.
```

---

## Phase −1 — Deep research seed

**Status**: **done** (2026-05-24). Phase log: [`phase-logs/P-1.md`](phase-logs/P-1.md). §D acceptance PASS (44 entries across all 9 seed topics). §B audit ran (11 findings — 4 P1 / 5 P2 / 2 P3 — all addressed; see phase log).
**Goal**: A populated [fabric-data-agent-research.md](fabric-data-agent-research.md) covering every open area listed in its "Seed entries" section, before any code touches anything.

**Premise gate (rigor §C — answer here, not in chat)**:
- *Why must this exist?* Because the existing plans cite MS docs inline but do not survive a session compact, and Phase 0+ make decisions that depend on facts not yet verified on the test workspace.
- *Evidence it's needed?* Architecture §1.7 already had to record five "rushed-claim corrections"; the same pattern will repeat without a persistent research record.
- *Simplest thing that could work?* Append the research findings as we go, into the doc, with the format the research doc specifies. No tooling. No automation.
- *What would falsify the need?* If every fact the build needs is already in architecture §1 verbatim, this phase is redundant. Verify by spot-checking three Phase 1+ tasks against architecture §1; if all three are fully covered with no inference gaps, skip P−1.

**Depends on**: nothing.

**Tasks**:
- Confirm Kane has provided the §2 inputs from methodology (SP creds, workspace, framework path).
- For each seed-entries section in `fabric-data-agent-research.md`, spawn a research subagent or do directed reading. Append findings to the research doc, dated, in the documented format.
- Verify findings against the test workspace where possible (e.g. SP can list lakehouses, Eventhouse plugin is enableable, encoder ONNX is fetchable from HF inside the workspace's network).
- Read the Fabric Framework skills listed in the *Skill consults* row below; cite which patterns are relevant for which phase. Append summaries to the research doc.

**Acceptance (§D measurement)**: every open question listed in the research doc's "Seed entries" section has at least one dated, sourced entry. Confidence is recorded for each.

**HTML doc task**: stub `docs/site/fabric/index.html` (landing page skeleton + table of contents — exact subpage list per the methodology §7 structure). Content TBD per phase; first version is the skeleton.

**Skill consults**: kb-resonance-lattice, kb-fabric-framework, kb-fabric-item-type-registry, kb-architecture-decision-guides, kb-user-data-functions.

---

## Phase 0 — Remove the `optimise/` machinery — CLOSED AS NO-OP

**Status**: **done — no-op (premise gate failed)** (2026-05-24). Phase log: [`phase-logs/P0.md`](phase-logs/P0.md).

**Why no-op**: the §C premise gate ran and falsified the phase's own justification:

1. The roadmap's original "why" cited "−0.042 / −0.043 nDCG@10 on Fabric docs" — this was a misread of the [HONEST_CLAIMS.md:108-119](../../docs/internal/HONEST_CLAIMS.md) three-row table. Those numbers are BEIR-fiqa and BEIR-nfcorpus; **Fabric docs is the corpus where optimise wins (+0.032 R@5)**.
2. The roadmap's "what would falsify removal" test ("a measurement showing optimise is net-positive on the Fabric docs corpus we'll deploy") was **already satisfied** by available evidence at the time of scoping.
3. Memory `project_mrl_specialist_encoder.md` names "MRL specialist @ d=512" (= the optimise output) as the SHIPPING RECIPE — removing it would change the named public product, not just the Fabric pipeline.
4. `src/resonance_lattice/fabric/_runtime.py:125` — the existing shipped Fabric UDF (`fabric://` consumer surface) calls `contents.select_band()`, which prefers optimised when present. Removing optimise changes the existing UDF behaviour, against which Goal & Success G4 (consumer parity) gates.
5. Blast radius probe: ~217 occurrences across 48 files (roadmap estimated "~13"). Order-of-magnitude misestimate.

The Fabric Eventhouse pipeline (architecture Decision 4) ingests **only** `bands/base.npz`. That is already true today and does not require deleting the optimise machinery elsewhere in the stack.

**Kane's resolution (2026-05-24)**: *"Optimise works on fabric docs that's confirmed. It stays for now. It was originally flagged to be replaced if we needed a specific band for fabric. We can go A."* (Option A = skip P0.)

**Lesson surfaced**: a §C premise gate citing a measurement number must verify the number against the cited source. The §C discipline caught this; left unchecked, P0 would have deleted a feature with measured value on the very corpus class the Fabric persona uses.

**Forward implication for P1+**: nothing changes downstream. P1 was always going to deploy only the base band into Eventhouse; that path stays. The optimise machinery survives unchanged for the local CLI and the existing `fabric://` UDF consumer.

---

## Phase 1 — `.rlat` → Eventhouse deploy path

**Status**: **done** (2026-05-24). Phase log: [`phase-logs/P1.md`](phase-logs/P1.md). §D acceptance PASS via local harness AND in-Fabric notebook (after §B-driven bisection caught two notebook-generator bugs). §B audit ran (12 findings — 2 P1 / 7 P2 / 3 P3 — P1s fixed in-phase, P2s carried forward per the phase log table).

**Premise gate**:
- *Why must this exist?* The data agent's only path to retrieval-against-rlat is via KQL (architecture Decision 1). The KQL surface requires the data to be in an Eventhouse table.
- *Evidence?* Architecture §1.1 (data-agent sources), §1.2 (Eventhouse as the only data-agent source with vectors), §6.1 (passages schema).
- *Simplest thing that could work?* A single pure-Python notebook that reads `.rlat`, builds a Polars frame, ingests into the Eventhouse table. Idempotent on `content_hash`.
- *What would falsify?* If the deploy step is so slow / error-prone that re-deploy isn't usable as the refresh path (Goal & Success O3, G5). Measurement: time a re-deploy on a 5K-passage corpus end-to-end.

**Depends on**: P0.

**Tasks**:
- Provision (or document provisioning of) an Eventhouse + KQL database in the test workspace. Docs: [Create a KQL database](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/create-database). (P-1 verified the SP can do this via REST — POST 201 sync.)
- Define the **two-object** schema (architecture §6.1):
  - `passages_raw` (append-only base table) with `Vector16` policy on the `vector` column;
  - `passages` (materialised view = `arg_max(ingest_time, *) by content_hash` over `passages_raw`).
  Apply a sharding/merging policy tuned for the F2 single-backend-node fanout (~8000 rows/shard per research entry 2026-05-24 — *Eventhouse vector — sharding/merging policy*).
- Write `notebooks/examples/fabric_deploy.ipynb` — pure-Python notebook (Polars/DuckDB; mirror the style of `notebooks/examples/fabric_build.ipynb`): read `passages.jsonl` + `bands/base.npz` from the `.rlat` via the rlat library API; build a Polars frame; **append** rows into `passages_raw` with `layer = "source"`. Idempotence is achieved by the materialised view's `arg_max by content_hash` — every re-run appends the full corpus to `_raw`; the view dedupes lazily in its background materialisation.
- Decide and document whether `rlat fabric deploy` CLI verb wraps the notebook (decide at P4 review, but capture the decision criterion now).

**Acceptance (§D)**: a built `.rlat` deploys (use a small corpus, ~100 passages); `passages | count` (the **view**) equals the `.rlat` passage count after the materialised view catches up; spot-checked rows have correct `text`, `source_file`, 768-element `vector`; re-deploy on an unchanged corpus appends N rows to `passages_raw` (expected) but the view's row count stays constant (no growth); CU-overhead of the materialised-view rebuild is measured and recorded (closes part of G5 propagation timing too).

**HTML doc task**: draft `docs/site/fabric/setup.html` "Step 2 — Deploy the knowledge model" section (placeholder where Step 1 — build — will land). Plain language, define every term.

**Skill consults**: kb-kql-patterns, kb-real-time-patterns, kb-notebook-patterns, kb-polars-development, wf-create-notebook.

**Rigor binding**: §A per commit (especially the notebook). §B at phase end. §D acceptance is the row-count + spot-check measurement above.

---

## Phase 2 — Query embedding in the sandbox

**Status**: **done — function shape verified; path superseded by amendment 2026-05-25** (2026-05-24/25). Phase log: [`phase-logs/P2.md`](phase-logs/P2.md). §D acceptance G1 PASS — 10/10 strings cosine=1.000000 (bit-exact vs local encoder, exceeding the ≥0.999 floor). Cold first call 21.4s, warm mean 18.7s. Subsequent measurement (2026-05-25) confirmed the per-call cost is structurally bounded by `external_artifacts` resolution (~10-15s overhead regardless of artefact size); the python() sandbox is not viable for interactive UX. **The architecture amendment 2026-05-25 (architecture.md) moves the source-layer embed path to the existing Fabric UDF**, called from a T-SQL stored procedure via `sp_invoke_external_rest_endpoint` — measured ~600-1000 ms typical. The P2 sandbox path artefacts (`embed_query()` KQL function, OneLake-staged encoder, G1 probe script) stay in tree as research artefacts and a documented fallback. Took 2 sessions; session 2 contained a methodology failure (two rounds of mis-framed "platform bug" → 5-pivot proposals, both retracted) corrected by Kane pushing back. Lessons recorded to `feedback_microsoft_tutorials_work.md`.

**Premise gate**:
- *Why must this exist?* Architecture Decision 3 — "no external services". The sandbox is the only no-external-service embed path (architecture §1.3, §1.4).
- *Evidence?* The `slm_embeddings_fl()` precedent (§1.3) proves the pattern works for `e5-small-v2`. Gte-modernbert is larger; verify in P2.
- *Simplest thing that could work?* Stage the ONNX + tokenizer.json + tokenizers wheel in OneLake; write the `python()` block that loads + tokenises + CLS-pools + L2-normalises. **No callout policy change is needed** for OneLake artefacts in the same tenant (research entry 2026-05-24).
- *What would falsify?* G1 fails — cosine between sandbox-embedded and locally-embedded vectors < 0.999 on a fixed string set. **This is the manifesto Falsifiable-claim gate 1.** If G1 fails, stop the entire plan — the deployment thesis is wrong.

**Depends on**: P1.

**Tasks**:
- Stage the gte-modernbert ONNX encoder, `tokenizer.json`, and `tokenizers-0.22.1-cp39-abi3-win_amd64.whl` (the wheel — `tokenizers` is NOT in the sandbox image per research 2026-05-24) in OneLake under `Files/rlat/`. **No callout-policy change needed** for OneLake artefacts (research 2026-05-24).
- Enable the Python plugin on the Eventhouse (*Eventhouse → Plugins*). Docs: [Enable the Python plugin](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/python-plugin).
- Write the `python()`-plugin embed code per the verbatim `rlat_search` shape in research entry 2026-05-24: load the ONNX with `onnxruntime`, install the tokenizers wheel via `pip install`, tokenise, CLS-pool, L2-normalise — the exact recipe of [`field/encoder.py`](../../src/resonance_lattice/field/encoder.py). Reference the artefacts via `external_artifacts` with `;impersonate` URL form. Docs: [Python plugin](https://learn.microsoft.com/en-us/kusto/query/python-plugin?view=microsoft-fabric).

**Acceptance (§D) — G1 parity floor**: embed 10 fixed strings in the sandbox; embed the same strings locally with `field/encoder.py`; cosine between each pair ≥ 0.999. **If this fails, stop — the deployment is wrong.** Result logged in research doc.

**HTML doc task**: `docs/site/fabric/how-it-works.html` — "How retrieval works inside Fabric" section. Plain-language explainer of the embed-in-sandbox pattern and why it means no external services.

**Skill consults**: kb-kql-patterns (python plugin section), kb-real-time-patterns.

**Rigor binding**: §A per commit. §D is the G1 measurement — non-negotiable. Surface to Kane on fail (methodology §6).

---

## Phase 3 — `dbo.rlat_search` T-SQL stored procedure (source layer) *(restructured 2026-05-25)*

**Status**: **done** (2026-05-25). Phase log: [`phase-logs/P3.md`](phase-logs/P3.md). §D acceptance PASS — G1 20/20 set-equivalence at K=8, latency soft gate PASS (p50 539 ms, p95 2.2 s). §B audit ran (5 P1 / 4 P2 findings — P1s addressed in-phase; P2s carried forward; see phase log).

**Premise gate**:
- *Why must this exist?* The data agent's NL2SQL needs a callable T-SQL stored procedure that wraps the embed-and-cosine pipeline. Without it the agent is asked to write the whole `sp_invoke_external_rest_endpoint` + JSON-parse + `VECTOR_DISTANCE` query — too much surface to generate reliably from NL.
- *Evidence?* Architecture §7 (the procedure spec, revised 2026-05-25); Decision 8 (data agent owns multi-hop, calls `dbo.rlat_search` per hop); empirical bridge proven 2026-05-25 (SQL DB → UDF via `sp_invoke_external_rest_endpoint`, allowlist permits, ~600 ms typical, ~20 ms `VECTOR_DISTANCE`).
- *Simplest thing that could work?* (1) Add `embed(query)` function to UDF (~10 lines). (2) Provision a Fabric SQL DB; assign server identity. (3) Create `dbo.passages` table with `VECTOR(768)` + MERGE-on-content_hash deploy. (4) Create `dbo.rlat_search` stored procedure that does `sp_invoke → embed → CAST VECTOR → VECTOR_DISTANCE TOP @top_k`. (5) Verify G1 parity vs local `rlat search`.
- *What would falsify?* G1 retrieval parity fails on the 20-query set (top-8 sets disagree). Encoder runs in the same UDF process for both surfaces, so the only failure mode is a `VECTOR_DISTANCE` vs local-cosine numerical mismatch beyond float-tolerance. If results disagree, debug the SQL DB vector storage / arithmetic path.

**Depends on**: P1 (Eventhouse passages — retained as learning-loop substrate, NOT the data-agent surface).

**Status update 2026-05-25**: P3 closed — G1 PASS 20/20, latency soft gate PASS (p50 539 ms / p95 2.2 s). See [P3 phase log](phase-logs/P3.md).

**Tasks** *(as completed; reflects 2026-05-25 amendment to inline `@headers` after the SAMI + credential-DDL findings; see research doc 2026-05-25 entries 1 & 2)*:
- Add `embed(query: str) -> list[float]` function to `fabric/udf/function_app.py` + `src/resonance_lattice/fabric/_runtime.py` — re-uses the warm `Encoder()` via a revision-keyed LRU shared with `bootstrap()`. Shipped in `rlat-2.1.0a14` wheel.
- Provision a Fabric SQL DB (`POST /v1/workspaces/{ws}/sqlDatabases`). `scripts/fabric_e2e_provision_sqldb.py` does it idempotently. **Server-identity assignment is NOT possible on Fabric SQL DB today** (research 2026-05-25); inline `@headers` in the procedure body is the working pattern.
- Define `dbo.passages` schema per architecture §6.1 (revised); primary key on `content_hash` (VARCHAR(80) — `sha256:<64-hex>` = 71 chars). No nonclustered index — `source_file` is `NVARCHAR(MAX)` which can't be in a key; `layer` has only 2 distinct values so an index on it isn't useful. Brute-force `VECTOR_DISTANCE` is the dominant cost.
- Author `dbo.rlat_search` stored procedure per architecture §7 (revised). `sp_invoke_external_rest_endpoint` uses inline `@headers = N'{"Authorization":"Bearer <token>"}'` (not `@credential`); token embedded at CREATE OR ALTER PROCEDURE time; rotated by re-running `scripts/fabric_e2e_init_sqldb.py`. Parse via `JSON_QUERY(@response, '$.result.output')` (sp_invoke wraps the UDF body under `$.result`); CAST to `VECTOR(768)`; `VECTOR_DISTANCE('cosine', vec, @qvec)` top-k.
- Deploy `.rlat` to `dbo.passages`: `scripts/fabric_e2e_deploy_rlat_sqldb.py`. Inline `CAST(N'[…]' AS VECTOR(768))` literals for the vec column (pyodbc binds JSON as ntext which won't implicit-cast); parameter binds for non-vector columns. Dedup on content_hash matches PK semantics.
- Capture event emission to the Eventstream destination is **deferred to P5** (memory event log capture). P3 ships the read path; capture wiring is part of the learning-loop phase.
- `fabric_deploy.ipynb` notebook update **deferred to P10 wizard unification** — combined with the Eventhouse ingest into a single user-facing wizard notebook there.

**Acceptance (§D) — G1 retrieval parity**: for 20 fixed queries, the top-8 `passage_id` set from `EXEC dbo.rlat_search(@query)` equals the top-8 from local `rlat search` on the same `.rlat` (order-insensitive; ties at the score boundary tolerated). This is the Goal & Success G1 gate. Encoder parity is mechanical (same UDF process as the local CLI uses); only the cosine-arithmetic path differs.

**Acceptance (§D, soft) — latency**: per-query end-to-end ≤ 1500 ms typical (50th percentile), ≤ 5 s tail (95th percentile) on F2. Recorded; not a hard ship gate but a UX-floor for P4 (data-agent integration).

**HTML doc task**: `docs/site/fabric/how-it-works.html` — add the "What `dbo.rlat_search` does" section (one stored procedure, one hop, plain language). Show the SQL inline so a Fabric Developer can read what the data agent generates under NL2SQL.

**Skill consults**: kb-fabric-sql-db (if it exists), kb-data-agent-patterns. The legacy kb-kql-patterns is no longer relevant for the source-layer surface (still relevant for the learning loop in P5).

**Rigor binding**: §A per commit. §D is G1 (hard) + the latency soft gate. §B at phase end (whole-solution review of the SQL DB substrate + UDF embed function).

---

## Phase 4 — Data agent integration

**Status**: not started

**Premise gate**:
- *Why must this exist?* Without it, all of P1–P3 is invisible to the user-facing surface (the Fabric data agent). The Goal & Success O2 requires this to be reached.
- *Evidence?* Architecture Decision 8 (data agent owns the multi-hop); §11 (two consumers).
- *Simplest thing that could work?* Create a data agent in the test workspace, add the KQL DB, configure instructions + example queries to call `rlat_search()`, test 10 questions.
- *What would falsify?* The agent ignores `rlat_search()` and answers from training data, OR fabricates citations. Either failure → debug agent instructions / example queries, not the function.

**Depends on**: P3.

**Tasks** *(updated 2026-05-25 for SQL DB source per architecture amendment)*:
- Create a Fabric data agent; add the **Fabric SQL DB** as a data source (NL2SQL). Docs: [Create a data agent](https://learn.microsoft.com/en-us/fabric/data-science/how-to-create-data-agent).
- Configure the agent's **instructions** (≤15,000 char limit per research entry "*Two-level instructions*"). Include explicit routing rule: "for conceptual / why-how / definitional questions, always call `EXEC dbo.rlat_search(@query)` and cite `source_file`."
- Configure the agent's **example queries** on the rlat data source (≤100 per source). Each example references `EXEC dbo.rlat_search(...)` so the NL2SQL retrieval surfaces them as few-shots (research entry "*Example queries are top-K retrieved few-shots*").
- **Author the refuse-cleanly instruction block** per research entry 2026-05-24 "*Data agent integration — Zero-result behaviour*": *"If `dbo.rlat_search` returns zero rows, reply exactly: 'No matching passages were retrieved. I cannot answer this without grounding.' Do not infer or supplement from prior knowledge."*
- Verify the multi-hop behaviour: the agent issues follow-up `EXEC dbo.rlat_search` calls when one hop is insufficient (architecture Decision 8).

**Acceptance (§D)**:
1. **G2 floor (10-question on-corpus probe)** — 10 conceptual Fabric-docs questions answered, each citing at least one real `source_file`; no fabricated citations on spot-check. Full G2 measurement (30-question blind-judged) runs in P10.
2. **G2.c floor (5-question off-corpus refuse probe)** — 5 deliberately-unanswerable questions all trigger the refuse-cleanly behaviour; recorded informationally per Goal & Success §3.

**HTML doc task**:
- `docs/site/fabric/setup.html` — "Step 4: create your data agent" section.
- `docs/site/fabric/using-the-agent.html` — first version: asking questions, reading citations, what a refusal means.
- `docs/site/fabric/external-assistants.html` — **lift from existing `docs/user/FABRIC.md` steps 5-7** (CLI install, `rlat fabric add`, `rlat search fabric://`); preserve the SOP for adding the alias once. P10 polishes. Per §B Finding 11.

**Skill consults**: kb-agent-patterns, kb-architecture-decision-guides.

**Rigor binding**: §A per commit (mostly KQL + agent config). §D is the 10-question floor + the 5-question off-corpus probe.

---

## Phase 5 — Memory event log (capture)

**Status**: not started — **substrate decided 2026-05-25 (research entry); P5 §C still pending**

**Substrate decision**: Fabric SQL DB (single-artefact). Eventstream + Eventhouse is NO LONGER REQUIRED for v1. Activator's March 2026 GA of scheduled-SQL-query-result rules collapses the original three-artefact pipeline (SQL DB + Eventstream + Eventhouse) into one (SQL DB). See research doc entry "2026-05-25 — P5 memory event log substrate" for the full comparison + sources + open questions.

**One open empirical validation** before P9 lock: Activator's scheduled-SQL-query rule is documented for "Fabric Data Warehouse SQL query results"; whether Fabric SQL DB (the relational workload, distinct from Warehouse) is treated as Warehouse-equivalent is not stated on MS Learn. Plan: 30-minute probe in P5a creating a trivial Activator rule against `dbo.passages` SELECT; if it works, lock single-artefact; if not, evaluate SQL-analytics-endpoint or SQL→Eventstream-shim fallback.

**Premise gate** *(below is the legacy framing under the old assumption; rewrite during P5a after substrate decision):*
- *Why must this exist?* The learning loop (manifesto Bet 3) starts here. Without capture, there's no compounding to measure.
- *Evidence?* Architecture §6.2 (`memory_events` schema), §9.1 (Eventstream capture spine — under review), §9.2 (loop ops).
- *Simplest thing that could work?* TBD post-substrate-research. The candidate paths above each have a different "simplest" answer.
- *What would falsify?* If writing per-call has measurable user-facing latency impact (>100ms added), or if the chosen substrate cannot keep up with the capture rate. Measurement: query latency with and without the capture emit.

**Depends on**: P3.

**Tasks**:
- Create the Eventstream (research entry — Activator needs Eventstream spine). Configure two destinations: Eventhouse `memory_events_raw` table + Activator (Activator setup deferred to P9 portal task; the destination wiring is metadata-only).
- Create `memory_events_raw` (append-only) + `memory_events` (materialised view = `arg_max(ingest_time, *) by row_id`) per architecture §6.2 pattern. Also create `outcomes` table (architecture §6.3).
- Extend `rlat_search()` to **emit a capture event to the Eventstream** per call. Event payload: `row_id` (ULID), `query_text`, `query_vector`, `lens_id` (default `"default"`), `retrieved_ids`, `top_score`, `cache_hit`, `event_utc`, defaults for the development/truth-axis fields. **NOT a direct KQL write to `memory_events`** — that path cannot trigger Activator.
- Add a recurrence materialised view: increment `recurrence_count` on near-duplicate `query_vector` (high `series_cosine_similarity` to existing rows in `memory_events`). Docs: [Materialized views](https://learn.microsoft.com/en-us/kusto/management/materialized-views/materialized-view-overview?view=microsoft-fabric).

**Acceptance (§D)**: issue N queries → N events through Eventstream → N rows in `memory_events_raw` → N distinct rows in `memory_events` view (post-arg_max); a repeated query shows `recurrence_count > 1` in the recurrence view; measure per-query latency overhead introduced by the Eventstream emit (target: <100ms added).

**HTML doc task**: `docs/site/fabric/operations.html` — "What gets logged and where" section.

**Skill consults**: kb-kql-patterns (materialised views), kb-real-time-patterns.

**Rigor binding**: §A per commit. §D includes the latency-overhead measurement.

---

## Phase 6 — Semantic cache (insight-layer read path)

**Status**: not started

**Premise gate**:
- *Why must this exist?* The insight layer is the user-visible payoff of the learning loop — a repeat / paraphrased question hits a cached cited answer.
- *Evidence?* Architecture §4 (three layers); §7 step 2 (search source + insight together).
- *Simplest thing that could work?* Allow `layer == "insight"` rows in `passages`; extend the function to search both layers; set `cache_hit = true` when an insight wins.
- *What would falsify?* If insight rows can't compete with source rows on cosine (e.g. they're too short / too generic), the cache never fires. Test with seeded rows in this phase before P7 grows them automatically.

**Depends on**: P5.

**Tasks**:
- Allow `layer == "insight"` rows in `passages` (schema already supports it).
- Extend `rlat_search()` step 2 to search `source` + `insight` together; set `cache_hit = true` when the top hit is an insight row above a threshold.
- Seed a handful of hand-written insight rows to exercise the path before P7 grows them automatically.

**Acceptance (§D)**: with seeded insight rows, a paraphrased query returns the insight row as top hit and `cache_hit = true`; retrieval parity (G1) still holds for source-only queries. Threshold for "insight wins" is set, measured, and recorded in research doc.

**HTML doc task**: `docs/site/fabric/how-it-works.html` — add "The learning layer" section (plain-language: cached answers from past questions).

**Skill consults**: kb-kql-patterns.

**Rigor binding**: §A per commit. §D requires re-running the G1 measurement to confirm no regression.

---

## Phase 7 — Consolidation notebook (the learning loop)

**Status**: not started

**Premise gate**:
- *Why must this exist?* Without it, the insight layer never grows automatically and the loop never compounds.
- *Evidence?* Architecture §9 (the loop spec).
- *Simplest thing that could work?* A scheduled pure-Python notebook that reads `memory_events`, runs the four harness operations (distil / insight generation / classify / forget), writes results back.
- *What would falsify?* If the `ai.*` functions can't produce cited insight rows that retain citation accuracy. Measurement: spot-check 10 generated insight rows for citation correctness; reject the approach if accuracy <90%.

**Depends on**: P5 (and P6 for the read path to benefit).

**Tasks**:
- Write `notebooks/examples/fabric_consolidate.ipynb` — pure-Python (Polars/DuckDB), scheduled. Read `memory_events` with DuckDB/Polars; bridge to pandas only at `ai.*` boundaries.
- Implement the harness operations (architecture §9): *distil* (`ai.summarize` — event→pattern→learning, write `parent_ids`); *insight generation* (`ai.generate_response` — synthesise cited insight passages from accepted query clusters, write `layer = "insight"`); *classify* (`ai.classify` / `ai.extract` — set `intent_kind`, `criticality`, `level`); *forget* (remove weak rows); *drift invalidation* (demote insight rows whose cited-passage hashes drifted). Docs: [AI functions overview](https://learn.microsoft.com/en-us/fabric/data-science/ai-functions/overview).
- Add the **deep-search offline batch**: run the rlat deep-search loop over top recurring unanswered queries; write results into the insight layer.
- Schedule the notebook (Fabric pipeline / scheduler). Docs: [Schedule and run notebooks](https://learn.microsoft.com/en-us/fabric/data-engineering/how-to-use-notebook).
- Make every `ai.*` step optional — the loop must still run (recurrence, forget, drift) with the LLM steps disabled.

**Acceptance (§D)**: after a seeded batch of accepted recurring queries, a scheduled run produces ≥1 new cited `insight` row; a subsequent paraphrased query hits it (`cache_hit = true`); a drifted insight row is demoted on the next run. Spot-check 10 generated insight rows for citation correctness.

**HTML doc task**: `docs/site/fabric/refreshing.html` (refresh / consolidation cadence). Update `docs/site/fabric/operations.html` (what the scheduled consolidation costs).

**Skill consults**: kb-ai-functions, kb-notebook-patterns, wf-create-notebook, kb-polars-development.

**Rigor binding**: §A per commit. §B at phase end (whole-loop review). §D is the produces-1-new-cited-insight + spot-check.

---

## Phase 8 — The lens — DEFERRED to v2

**Status**: deferred to v2 (Goal & Success Q2 grilling outcome, 2026-05-24).

**Why deferred**: v1's defensible claim ("grounded + traceable + no external services + one .rlat two consumers") is complete without lens. Lens adds a separate claim that itself depends on a G6 measurement that hasn't run. A half-lens (one example, no creation UX) would violate "no incomplete features in stable paths".

**What stays in v1 to accommodate v2 plug-in**:
- The `lens_id` parameter on `rlat_search()` (P3) stays in the function signature, unused.
- The architecture §6.4 `lens` table schema stays documented.
- v2 mechanical add-on: create the lens table, extend step 3 with the trust-weight join, integrate with the `rlat lens` CLI from [lensed-knowledge-architecture.md](lensed-knowledge-architecture.md).

**v2 phase outline** (preserved for future-Claude):

- Create the `lens` table (architecture §6.4). Decide lens-as-OneLake-file vs lens-as-table.
- Extend `rlat_search()` step 3: when `lens_id` is set, join `trust_weights` and re-rank `score * trust_weight_for(source_file)`. Pure KQL.
- Integrate with `rlat lens` CLI.
- Map lens scope to Fabric structure.
- Acceptance: two lenses with divergent trust_weights produce measurably different top-8 orderings for the same query (≥3 of 20 queries reorder); the no-lens path is unchanged from G1.

---

## Phase 9 — Activator + gap dashboard

**Status**: not started

**Premise gate**:
- *Why must this exist?* The "knowledge gap visible to the operator" claim (Goal & Success O5 + manifesto Tier 1 / 2). Without Activator + dashboard, the loop is silent.
- *Evidence?* Architecture §9.2 (loop ops); §9.3 (Activator setup constraint); goal & success O5.
- *Simplest thing that could work?* Activator rules attached to the Eventstream destination set up in P5 (low score → gap alert, high recurrence → FAQ candidate, not_satisfied → demote via UDF action) + a starter Power BI semantic model + measure pack.
- *What would falsify?* If Activator latency is too high to make alerts useful (>10 min worst case per research entry on Activator latency). Goal & Success O3/G5 cares about refresh propagation, not alert latency — but a >10-min alert pipeline would defeat the gap-discovery story.

**Depends on**: P5 (Eventstream + Activator destination wiring done in P5; rules ride on it) + P7 (meaningful data to alert on).

**Tasks**:
- **Author Activator rules in the portal** (NOT via SP REST — research entry 2026-05-24 "*Reflex item: REST CRUD shipped, but service principal NOT supported*"). The rules subscribe to the Eventstream Activator destination set up in P5. Three rules: uniformly-low `top_score` → corpus-gap alert; `recurrence_count` over threshold → FAQ-promotion candidate; `not_satisfied` outcomes → demote insight (UDF action — research entry "*UDF action is the cleanest fit for the demote-row write-back*").
- **Microsoft roadmap-watch item**: when Reflex CRUD adds SP support, automate this step into the setup wizard. Until then, the HTML guide carries the portal step-by-step.
- A starter Power BI semantic model + measure pack over `memory_events` / `outcomes`: cache-hit trend, retrieval-quality trend, top questions, the live knowledge-gap list. **No bespoke report ships** — measures only; users build their own visuals (Goal & Success §7 non-goal).
- Empirical probe queue (research entry — *Items to verify empirically on the test workspace before Phase 9 ships*): UI surface for composite AND/OR; aggregation primitives; end-to-end latency on F2; UDF action payload contract; per-rule CU consumption; Real-Time Dashboard fallback latency. Half-day timebox before architectural commitment between Eventstream-tee and RTD-poll paths.

**Acceptance (§D)**: a deliberately unanswerable query, asked repeatedly, raises a gap alert and appears on the dashboard's gap list. End-to-end latency from query → alert measured and recorded (target ≤10 min worst-case per research).

**HTML doc task**: `docs/site/fabric/operations.html` — Activator setup + measure pack sections. **Portal-only step-by-step** (with screenshots if practical) since SP automation isn't available.

**Skill consults**: kb-fabric-admin-api, kb-real-time-patterns, kb-powerbi-report-patterns, kb-semantic-model-patterns.

**Rigor binding**: §A per commit. §D includes the alert-latency measurement.

**Methodology §6 carry-forward from §B audit**: Activator is the only v1 component whose install path cannot be SP-automated. Goal & Success §3 G3 explicitly excludes Activator setup from the 30-min timer to avoid the portal-step penalty. The HTML guide must surface this honestly so the user knows their setup is "almost-done" after the wizard finishes; Activator is the deliberate manual step.

---

## Phase 10 — Unification, validation, docs (the "rlat in Fabric" v1 release)

**Status**: not started

**Premise gate**:
- *Why must this exist?* Goal & Success O1, O4, O5 all require a unified setup and a coherent user surface. The existing fabric work (UDF + 3 notebooks) and the new data-agent work (KQL DB + new notebooks) need to be one thing, not two.
- *Evidence?* Goal & Success outcomes; manifesto §11 (one .rlat, two consumers); user-experience-and-abstraction priority from Kane's brief.
- *Simplest thing that could work?* (a) A single setup wizard that drives both consumer surfaces from one entry point; (b) the HTML doc covers both surfaces from one landing page; (c) the eval (G1–G6) runs and is recorded.
- *What would falsify?* If Kane (or another fresh tester) walking the HTML guide can't complete O1 in <30 minutes (G3 fails), the unification isn't working — iterate the wizard / docs / surface.

**Depends on**: P4 + P7 + P9 (P8 deferred to v2).

**Tasks** *(restructured 2026-05-25 — clean-tenant rebuild is the explicit centrepiece, not the acceptance gate)*:

1. **Workspace teardown** — delete all probe/test items from the build workspace (`Kane-Test-Personal`):
   - throwaway Fabric SQL DB `rlat-allowlist-probe` (item id `81e8e1a7-…`)
   - throwaway ML experiment `rlat-p2-mlflow-probe` (item id `e4cd6b5d-…`) and the associated notebook `rlat-p2-mlflow-probe`
   - all `rlat-diag-*` notebooks
   - the `_p2_probe/` folder in OneLake Files
   - leave the Eventhouse + KQL DB + Lakehouse alone IF the design is "second workspace is the clean target"; otherwise also tear those down. **Decision point**: do the clean rebuild in a *separate* clean workspace OR in the same workspace after teardown. The clean-workspace option is preferable for the dress rehearsal because it doesn't risk leaving stale state behind.

2. **Unified setup wizard** — a single notebook (`notebooks/examples/fabric_setup.ipynb`) per grilling outcome Q3 (Goal & Success §8). **No v1 CLI verb** (non-goal per Goal & Success §7). The notebook absorbs today's `fabric_build.ipynb` and drives both consumer surfaces. Wraps: build/refresh `.rlat`, provision Fabric SQL DB + server-identity assignment, deploy passages with VECTOR(768) via MERGE-on-content_hash, create `dbo.rlat_search` stored procedure, provision Eventhouse + Eventstream + memory_events tables for the learning loop, publish/refresh UDF + its new `embed()` function, print the data-source URL and the `fabric://` alias. Abstracts every config choice that has a reasonable default. Activator setup is documented as a portal-only follow-up (P9 task; SP-unsupported).

3. **HTML guide complete** — `docs/site/fabric/` is the canonical user surface. **All nine pages** (per methodology §7: index + what-this-is + setup + using-the-agent + refreshing + external-assistants + how-it-works + operations + troubleshooting) reviewed by a fresh subagent against each of the three personas in Goal & Success §1. Findings addressed.

4. **Clean-tenant rebuild dress rehearsal** *(the G3 measurement — non-negotiable)*:
   - Kane provisions a fresh Fabric workspace (or uses one he hasn't been driving the build on).
   - Kane runs the published HTML setup guide end-to-end, stopwatch each step.
   - **No Claude assistance during the rehearsal** — the goal is to validate that the documentation is self-contained for a Fabric Developer who's never seen rlat.
   - Any step that requires Claude help → that step's docs are wrong → rewrite that section → re-run the rehearsal from that step.
   - Loop until: full setup → first cited answer from the data agent in ≤ 30 minutes (per G3), AND fabric:// search works from Claude Code against the same `.rlat` (G4 parity).
   - **Tear down the rehearsal workspace at the end** so the next rehearsal starts genuinely clean.

5. **Full eval suite** — run G1, G2.a, G2.b, G2.c, G3 (from the rehearsal), G4, G5, G6 measurements on the canonical test corpus (`fabric-docs-rlat` v2); record in research doc with deploy commit SHA + workspace ID + conditions.

6. **External-assistant parity confirmed** — verify the shipped UDF / `fabric://` path ([fabric-udf-integration.md](fabric-udf-integration.md)) reads the same `corpus.rlat` the SQL DB `dbo.passages` table was deployed from; document the single deploy pipeline that keeps them consistent. Run G4 (consumer parity).

7. **Docs alignment** — write or replace `docs/user/FABRIC.md` to point users at the HTML guide; update [CLAUDE.md](../../CLAUDE.md) (Primary Interfaces, Architecture Map, Fabric row); update [docs/internal/HONEST_CLAIMS.md](../../docs/internal/HONEST_CLAIMS.md) with measured results and the claim tier of each.

8. **Release** — version `2.2.0` per grilling Q11 (Goal & Success §8); PyPI push via the trusted-publisher workflow; sync to public repo (`sync-public.sh`).

**Acceptance (§D)**: every checkbox in Goal & Success §6 (Done definition) is true and recorded. **The clean-tenant rebuild (task 4) IS the §D measurement for G3 and the dress rehearsal for everything else.** External-validation gate per grilling Q10 (Goal & Success §8): Kane personally runs the HTML guide end-to-end on a clean tenant (one he hasn't been driving the build on) and confirms O1–O5; no external-tester gate, no v1.1 commitment.

**HTML doc task**: full pass on every page; fresh-subagent review against the §1 persona; address findings. The rehearsal in task 4 is the ultimate test of whether the HTML guide is fit for purpose.

**Skill consults**: wf-execute-tests, wf-execute-deployment, wf-handover-documentation, frontend-design (for HTML guide polish).

**Rigor binding**: §A per commit (especially the wizard notebook — its failure modes matter on a clean tenant). §B applied to the whole solution (the full v1 surface) — a fresh subagent audits the wizard + HTML guide + research record together. §D is the full G1–G6 measurement sweep, with G3 specifically driven by the task-4 clean rebuild.

---

## The eval — claim gates summary

No quantified claim ships before its gate passes. (Inherited and expanded from manifesto Falsifiable claim; full per-gate detail in Goal & Success §3.)

| Gate | Measures | Unlocks | Phase that gates it |
|---|---|---|---|
| **G1 — parity** | top-8 SQL `dbo.rlat_search` == top-8 local `rlat search`, 20 queries | "retrieval quality preserved exactly" | P3 (stored proc) — measured against the SQL DB surface per architecture amendment |
| **G2 — agent quality** | answerable-rate + fabrication-rate of the *data agent* on a fixed Fabric-docs question set, blind-judged | end-to-end quality number | P4 (10-q floor) / P10 (full 30-q) |
| **G3 — setup time** | wall-clock for Kane (or a fresh tester) to complete O1 end to end on a clean Fabric workspace, no Claude assistance | "30 minutes to first answer" | P10 task 4 — the clean-tenant rebuild rehearsal |
| **G4 — consumer parity** | top-k from `rlat search fabric://` matches `dbo.rlat_search` (both share the same UDF embed process — should be mechanically equivalent) | "one knowledge model, two surfaces" | P10 |
| **G5 — refresh propagation** | time from "user re-runs wizard" to "agent sees new content" | "stays fresh on demand" | P10 |
| **G6 — compounding** | insight-layer hit-rate over a 20-question repeat-and-vary session | "the loop compounds" (number not published until measured) | P7 (read path) / P10 (full) |

---

## Open questions (most resolved in P-1 research; remaining are tunable in-phase)

Roadmap-specific. Goal & Success §8 carries the user-facing list (locked).

1. **Format version** — bump to `5` or keep `4` with optimised fields unused? (P0; recommendation: bump.)
2. **Lens scope ↔ Fabric structure** — `team` ≈ workspace; `role`/`user` ≈ Entra group/principal. (v2 — P8 deferred.)
3. **Recurrence / promotion thresholds** — what `series_cosine_similarity` counts as a "near-duplicate" query; what `recurrence_count` triggers FAQ promotion. Tune with real data. (P5/P7.)
4. ~~**Encoder staging from OneLake** — confirm the Eventhouse callout policy can fetch the ONNX from OneLake directly, or whether same-tenant blob storage is needed.~~ **CLOSED 2026-05-24** (P-1 research): OneLake artefacts in the same tenant **bypass callout policy**; `external_artifacts` with `;impersonate` is sufficient. Verified in P2 G1.
5. **Deploy as a CLI verb (`rlat fabric deploy`) vs notebook-only** — decide after P4. (P1/P4.) **Note**: the unified setup wizard (P10) is notebook-only per Q11 below; a separate `rlat fabric deploy` CLI verb for power users / CI is still on the table for v2.
6. **Tokeniser in the sandbox** — ~~bundled artifact vs sandbox-available package~~ **CLOSED 2026-05-24** (P-1 research): `tokenizers` is NOT in the sandbox image; must ship as `tokenizers-0.22.1-cp39-abi3-win_amd64.whl` external artifact, installed via `pip install` in the `python()` block. Fits the 1 GB sandbox with ~400 MB headroom (research entry "*Sandbox encoder — lifecycle, artifact cache, 1 GB ceiling, init cost*").
7. **`ai.*` capacity cost** — ~~measure consolidation-run cost; confirm graceful no-LLM degradation~~ **PARTIALLY CLOSED 2026-05-24** (P-1 research forecast: ~18,800 CU-s per run on gpt-4.1-mini ≈ 10% F2 daily budget; throttling is staged not hard-reject). Final empirical measurement in P7.
8. ~~**Unified wizard form** — notebook, CLI verb, or both?~~ **CLOSED 2026-05-24** (grilling Q3): notebook primary, no v1 CLI verb.

**Residuals carried forward from P-1 to depending phases** (per §B Finding 8):

- **P2 / G1 verification**: `%%configure vCores=8` actually allocates on F2; `tokenizers` wheel installs and loads in the sandbox; sandbox encoder init cost measured; callout-bypass confirmed; first cold call measured.
- **P9 verification spike**: Activator UI surface for composite AND/OR; aggregation primitives; end-to-end latency on F2; UDF action payload contract; per-rule CU consumption; Real-Time Dashboard fallback latency.
- **P-1 G3 measurement** deferred to P10 — the components don't exist yet to measure end-to-end. P10 G3 acceptance carries the empirical-vs-30-min check.

**Residuals carried forward from P1 to later phases** (per P1 §B audit):

- **P2: extract `scripts/_fabric_sp.py`** when the first new `fabric_e2e_*.py` script lands (avoid refactor-without-driver). Consolidates auth helpers / OAuth POST / token / headers / poll_lro / WORKSPACE_DEFAULT duplicated across 4 P1 scripts.
- **P5: phase-log §C must call out** "do NOT copy `scripts/fabric_e2e_deploy_rlat.py`'s direct-Kusto-ingest pattern for `memory_events_raw` — Activator (P9) requires Eventstream as the spine."
- **P5: extract `resonance_lattice.fabric.deploy.passage_to_row(passage, base_vector, now)`** library function when the second row-writer surfaces (consolidation notebook); removes the row-dict duplication between harness + notebook generator.
- **P10: simplify runner** — drop `_inject_override_cell` and rely on native `executionData.parameters` binding once the diagnostic-mistake is understood (Kane confirms binding works for Fabric pure-Python; my P1 diagnostic must have had a defect — likely converter tag-emission shape or executionData payload shape).
- **P10: setup-wizard config UX** — replace hardcoded test-workspace IDs (Kane's tenant defaults in 4 scripts) with proper config surface.
- **General research housekeeping**: append a standalone "streaming REST ingest is broken on Fabric — use Kusto SDK" entry to research doc on next touch (currently only in skill-consult table).

---

## Risks

| Risk | Mitigation |
|---|---|
| G1 fails — sandbox encoder diverges from `field/encoder.py`. | P2 is an explicit gate before any further phase. Replicate the recipe exactly; pin the ONNX revision. Methodology §6 trigger: surface to Kane on fail. |
| Latency unacceptable in practice (>3 s felt). | Decision 9 accepted it "for now"; the semantic cache (P6) softens repeats; revisit only if G2 shows it harms usability. |
| Python plugin disabled by tenant policy. | Documented prerequisite in HTML guide; no code workaround — the no-external-service constraint requires it. P-1 confirms it's enable-able on the test workspace. |
| Corpus exceeds the ~1 M-vector Eventhouse ceiling. | Out of scope at current corpus sizes; note in honest limits; partition per knowledge model if it ever arises. |
| Lens compounding (G6) does not materialise. | Claims are pre-gated to Tier 2; the loop's Tier-1 value (grounded answers, no external service) stands regardless. |
| Scope creep across 11 phases. | Each phase is independently committable with its own acceptance check; the methodology §3 loop keeps sessions bounded; §6 trigger fires on order-of-magnitude phase-size blowouts. |
| SP loses workspace scope mid-build. | Methodology §6 trigger: stop, surface, do not improvise auth. |
| HTML guide drifts from the code surface. | Methodology §7 discipline: HTML doc updated in the same commit set as the code change. Phase doesn't close until docs read true. |
| Reviewer subagents share Claude's training distribution and miss the same things. | Rigor preamble admits this honestly; §D empirical gate is the real check; Kane is the genuinely independent reviewer at phase boundaries. |
