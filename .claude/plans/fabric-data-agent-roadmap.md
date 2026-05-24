# rlat × Fabric Data Agent — Roadmap

**Status**: v1. The phased build sequence, acceptance gates, eval, and open questions.
**Last updated**: 2026-05-17.
**Companions**: [fabric-data-agent-manifesto.md](fabric-data-agent-manifesto.md) (thesis, value claims); [fabric-data-agent-architecture.md](fabric-data-agent-architecture.md) (research, decisions, schemas).

---

## What this document is

The manifesto says *what* and *why*; the architecture says *how*. This roadmap says *when* and *in what order*, with enough per-phase detail that **any session can pick up any unstarted phase cold** — no mid-session improvisation. It is the most volatile of the three documents; phases get re-scoped as engineering surfaces real constraints. The manifesto and architecture should not move with these changes.

This is a long-horizon plan spanning many sessions. It is not expected to be done in one.

---

## How to resume (session-resumption protocol)

A session continuing this work should:

1. Read the manifesto §"Non-negotiables" and the architecture §2 "Decisions locked" — these are the invariants.
2. Find the first phase below whose **Status** is not `done`.
3. Read that phase's *Goal*, *Depends on*, *Tasks*, and *Acceptance*. Everything needed is there.
4. Do the phase. Run its *Acceptance* check. Update its *Status* line in this file. Commit.
5. Each phase is independently testable and independently committable. Do not start a phase whose *Depends on* is unmet.

**Status legend**: `not started` · `in progress` · `blocked` · `done`. Update the line, in this file, as part of the phase's commit.

**Phase dependency graph**:

```
P0 ─▶ P1 ─▶ P2 ─▶ P3 ─▶ P4
                   │
                   ├─▶ P5 ─▶ P6 ─▶ P7
                   │              │
                   └─▶ P8         └─▶ P9
                                  P10 (after P4 + P7)
```

---

## Phase 0 — Remove the `optimise/` machinery

**Status**: not started
**Goal**: A clean base band with no optimised-band code paths. Justified independently of Fabric by [docs/internal/HONEST_CLAIMS.md](../../docs/internal/HONEST_CLAIMS.md) (architecture §1.7, Decision 5).
**Depends on**: nothing.

**Tasks**:
- Delete `src/resonance_lattice/optimise/` and `src/resonance_lattice/cli/optimise.py`; drop the `optimise` command from the CLI registry.
- Remove the optimised-band code paths (~13 sites): `store/metadata.py` (`BandInfo` fields `dim_native`, `w_shape`, `nested_mrl_dims`, `trained_from`), `store/bands.py` (`load_optimised`, `write_projection`), `store/archive.py` (the `w_shape`-present branch; `select_band`), `store/incremental.py` (re-projection block), `field/dense.py` (the `projection_matrix` branch), `cli/maintain.py` (`--discard-optimised` flags), `rql/` band-preference logic.
- Decide and execute: keep `FORMAT_VERSION = 4` with optimised fields simply unused, or bump to `5` dropping them. Recommendation: **bump to 5** — a clean spec beats a deprecated field. Update [docs/internal/STORE.md](../../docs/internal/STORE.md).
- Delete / adapt `tests/harness/optimise_roundtrip.py` and any optimise benchmarks.
- Update docs: remove `docs/user/OPTIMISE.md`, `docs/internal/OPTIMISE.md`; prune the `optimise` row from [CLAUDE.md](../../CLAUDE.md) Primary Interfaces and the Optimise architecture-map row; update [docs/internal/HONEST_CLAIMS.md](../../docs/internal/HONEST_CLAIMS.md) to record the removal and why.

**Acceptance**: `python -m tests.harness.runner` green; `rlat build` produces a base-only `.rlat`; `rlat search` works; `grep -ri optimis src/` returns nothing.

---

## Phase 1 — `.rlat` → Eventhouse deploy path

**Status**: not started
**Goal**: A `.rlat` corpus lands as a queryable Eventhouse `passages` table.
**Depends on**: P0.

**Tasks**:
- Provision (or document provisioning of) an Eventhouse + KQL database in a test workspace. Docs: [Create a KQL database](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/create-database).
- Define the `passages` table schema (architecture §6.1). Apply a `Vector16` encoding policy on `vector` and a sharding/merging policy tuned for vector search. Docs: [Vector database tutorial](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/vector-database-eventhouse).
- Write `notebooks/examples/fabric_deploy.ipynb` — a **pure-Python** notebook (Polars/DuckDB; mirror the style of `notebooks/examples/fabric_build.ipynb`): read `passages.jsonl` + `bands/base.npz` from the `.rlat` via the rlat library API; build a Polars frame; ingest into `passages` with `layer = "source"`. Make ingestion **idempotent / incremental** — upsert keyed on `content_hash` so re-running after `rlat refresh` only touches changed rows.
- Optionally add a thin `rlat fabric deploy` CLI verb wrapping the same library calls (decide in Phase 4 review).

**Acceptance**: a built `.rlat` (use a small corpus) deploys; `passages | count` in the KQL database equals the `.rlat` passage count; spot-checked rows have correct `text`, `source_file`, 768-element `vector`.

---

## Phase 2 — Query embedding in the sandbox

**Status**: not started
**Goal**: A KQL fragment embeds an arbitrary query string into a 768-d gte-modernbert vector, with no external service.
**Depends on**: P1.

**Tasks**:
- Stage the gte-modernbert ONNX encoder in OneLake. Add its path to the Eventhouse **callout policy** so the engine may fetch it as an external artifact.
- Enable the Python plugin on the Eventhouse (*Eventhouse → Plugins*). Docs: [Enable the Python plugin](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/python-plugin).
- Write the `python()`-plugin embed code: load the ONNX with `onnxruntime`, tokenise, CLS-pool, L2-normalise — the exact recipe of [`field/encoder.py`](../../src/resonance_lattice/field/encoder.py). Reference the encoder via `external_artifacts`. Docs: [Python plugin](https://learn.microsoft.com/en-us/kusto/query/python-plugin?view=microsoft-fabric) · [Sandboxes](https://learn.microsoft.com/en-us/kusto/concepts/sandboxes?view=azure-data-explorer).
- Decide tokeniser packaging: bundle as an external artifact or use a sandbox-available package. Docs: [Python package reference](https://learn.microsoft.com/en-us/kusto/query/python-package-reference?view=microsoft-fabric).

**Acceptance** — **parity floor**: embed 10 fixed strings in the sandbox; embed the same strings locally with `field/encoder.py`; cosine between each pair ≥ 0.999. If this fails, stop — the deployment is wrong.

---

## Phase 3 — `rlat_search()` stored function (source layer)

**Status**: not started
**Goal**: One KQL stored function: embed → cosine → top-k, source layer only.
**Depends on**: P2.

**Tasks**:
- Compose Phase 2 embedding + `series_cosine_similarity` over `passages` (`layer == "source"`) + `top top_k`. Docs: [series_cosine_similarity](https://learn.microsoft.com/en-us/kusto/query/series-cosine-similarity-function?view=microsoft-fabric).
- Register as a KQL stored function `rlat_search(query, lens_id, top_k)` (lens_id unused until P8). Return `text`, `source_file`, offsets, `score`, `layer`.

**Acceptance** — **retrieval parity**: for 20 fixed queries, the top-8 `passage_id` set from `rlat_search()` equals the top-8 from local `rlat search` on the same `.rlat` (order-insensitive; ties at the score boundary tolerated). This is the manifesto Falsifiable-claim gate 1.

---

## Phase 4 — Data agent integration

**Status**: not started
**Goal**: A Fabric data agent answers a Fabric-docs question, grounded, with citations.
**Depends on**: P3.

**Tasks**:
- Create a Fabric data agent; add the KQL database as a data source. Docs: [Create a data agent](https://learn.microsoft.com/en-us/fabric/data-science/how-to-create-data-agent).
- Configure the agent's **instructions** and **example queries** to call `rlat_search()` for conceptual questions and to cite `source_file`. Docs: [Data agent configurations](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-configurations) · [Example queries](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-example-queries).
- Verify the multi-hop behaviour: the agent issues follow-up `rlat_search()` calls when one hop is insufficient (architecture Decision 8).

**Acceptance**: 10 conceptual Fabric-docs questions answered, each citing at least one real `source_file`; no fabricated citations on spot-check.

---

## Phase 5 — Memory event log (capture)

**Status**: not started
**Goal**: Every `rlat_search()` call is logged with the full memory schema.
**Depends on**: P3.

**Tasks**:
- Create `memory_events` and `outcomes` tables (architecture §6.2, §6.3).
- Extend `rlat_search()` (or add a paired KQL update policy) to write a `memory_events` row per call: `query_text`, `query_vector`, `lens_id`, `retrieved_ids`, `top_score`, `cache_hit`, `event_utc`, defaults for the development/truth-axis fields. Docs: [Update policy](https://learn.microsoft.com/en-us/kusto/management/update-policy?view=microsoft-fabric).
- Add a recurrence update policy / materialised view: increment `recurrence_count` on near-duplicate `query_vector`. Docs: [Materialized views](https://learn.microsoft.com/en-us/kusto/management/materialized-views/materialized-view-overview?view=microsoft-fabric).

**Acceptance**: issuing N queries yields N `memory_events` rows with all fields populated; a repeated query shows `recurrence_count > 1`.

---

## Phase 6 — Semantic cache (insight-layer read path)

**Status**: not started
**Goal**: A near-duplicate of a previously-answered question returns a cached insight.
**Depends on**: P5.

**Tasks**:
- Allow `layer == "insight"` rows in `passages` (schema already supports it).
- Extend `rlat_search()` step 2 to search `source` + `insight` together; set `cache_hit = true` when the top hit is an insight row above a threshold.
- Seed a handful of hand-written insight rows to exercise the path before P7 grows them automatically.

**Acceptance**: with seeded insight rows, a paraphrased query returns the insight row as top hit and `cache_hit = true`; retrieval parity (P3) still holds for source-only queries.

---

## Phase 7 — Consolidation notebook (the learning loop)

**Status**: not started
**Goal**: The closed loop runs end to end: events distil into earned insight.
**Depends on**: P5 (and P6 for the read path to benefit).

**Tasks**:
- Write `notebooks/examples/fabric_consolidate.ipynb` — **pure-Python** (Polars/DuckDB), scheduled. Read `memory_events` with DuckDB/Polars; bridge to pandas only at `ai.*` boundaries.
- Implement the harness operations (architecture §9): *distil* (`ai.summarize` — event→pattern→learning, write `parent_ids`); *insight generation* (`ai.generate_response` — synthesise cited insight passages from accepted query clusters, write `layer = "insight"`); *classify* (`ai.classify` / `ai.extract` — set `intent_kind`, `criticality`, `level`); *forget* (remove weak rows); *drift invalidation* (demote insight rows whose cited-passage hashes drifted). Docs: [AI functions overview](https://learn.microsoft.com/en-us/fabric/data-science/ai-functions/overview).
- Add the **deep-search offline batch**: run the rlat deep-search loop over top recurring unanswered queries; write results into the insight layer.
- Schedule the notebook (Fabric pipeline / scheduler). Docs: [Schedule and run notebooks](https://learn.microsoft.com/en-us/fabric/data-engineering/how-to-use-notebook).
- Make every `ai.*` step optional — the loop must still run (recurrence, forget, drift) with the LLM steps disabled.

**Acceptance**: after a seeded batch of accepted recurring queries, a scheduled run produces ≥1 new cited `insight` row; a subsequent paraphrased query hits it (`cache_hit = true`); a drifted insight row is demoted on the next run.

---

## Phase 8 — The lens

**Status**: not started
**Goal**: Two lenses produce appropriately different rankings for the same query.
**Depends on**: P3.

**Tasks**:
- Create the `lens` table (architecture §6.4). Decide lens-as-OneLake-file vs lens-as-table (or both — file is the portable artifact, table is the query-time copy).
- Extend `rlat_search()` step 3: when `lens_id` is set, join `trust_weights` and re-rank `score * trust_weight_for(source_file)`. Pure KQL.
- Integrate with `rlat lens` CLI ([lensed-knowledge-architecture.md](lensed-knowledge-architecture.md) §12.2) — a `lens` export targeting Fabric.
- Map lens scope to Fabric structure (see Open questions).

**Acceptance**: two lenses with divergent `trust_weights` produce measurably different top-8 orderings for the same query; the no-lens path is unchanged from P3.

---

## Phase 9 — Activator + gap dashboard

**Status**: not started
**Goal**: The loop becomes reactive and observable.
**Depends on**: P5 (Activator) and P7 (meaningful data).

**Tasks**:
- Activator rules on `memory_events`: uniformly-low `top_score` → corpus-gap alert; `recurrence_count` over threshold → FAQ-promotion candidate; `not_satisfied` outcomes → demote insight. Docs: [Data Activator introduction](https://learn.microsoft.com/en-us/fabric/data-activator/data-activator-introduction).
- A Power BI report over `memory_events` / `outcomes`: cache-hit trend, retrieval-quality trend, top questions, the live knowledge-gap list.

**Acceptance**: a deliberately unanswerable query, asked repeatedly, raises a gap alert and appears on the dashboard's gap list.

---

## Phase 10 — External-assistant parity, validation, docs

**Status**: not started
**Goal**: One `.rlat` cleanly serves both consumers; claims are validated; docs updated.
**Depends on**: P4 and P7.

**Tasks**:
- Verify the shipped UDF / `fabric://` path ([fabric-udf-integration.md](fabric-udf-integration.md)) reads the same `corpus.rlat` the data agent's Eventhouse table was deployed from; document the single deploy pipeline that keeps them consistent.
- Run the eval (below).
- Write `docs/user/FABRIC_DATA_AGENT.md` (setup → first grounded answer); update [CLAUDE.md](../../CLAUDE.md) (Primary Interfaces, Architecture Map, Fabric row); update [docs/internal/HONEST_CLAIMS.md](../../docs/internal/HONEST_CLAIMS.md) with measured results and the claim tier of each.

**Acceptance**: a new user follows `FABRIC_DATA_AGENT.md` end to end; the eval has run; no Tier-3 claim is published without a number from this deployment.

---

## The eval — claim gates

No quantified claim ships before its gate passes. Mirrors [lensed-knowledge-roadmap.md](lensed-knowledge-roadmap.md) discipline.

| Gate | Measures | Unlocks | Phase |
|---|---|---|---|
| **G1 — parity** | top-8 `rlat_search()` == top-8 local `rlat search`, 20 queries | "retrieval quality preserved exactly" | P3 |
| **G2 — agent quality** | answerable-rate + hallucination-rate of the *data agent* on a fixed Fabric-docs question set | an end-to-end quality number | P4 / P10 |
| **G3 — compounding** | insight-layer hit-rate over a 20-question repeat-and-vary session | "the knowledge model compounds" | P7 / P10 |
| **G4 — lens** | the lensed-knowledge dogfood + a Fabric-specific run | lens portability / composition claims | P8 / P10 |

Until a gate passes, the corresponding claim stays in manifesto Tier 2/3.

---

## Open questions

1. **Format version** — bump to `5` or keep `4` with optimised fields unused? (P0; recommendation: bump.)
2. **Lens scope ↔ Fabric structure** — `team` ≈ workspace; `role`/`user` ≈ Entra group/principal. Confirm and document. (P8.)
3. **Recurrence / promotion thresholds** — what `series_cosine_similarity` counts as a "near-duplicate" query; what `recurrence_count` triggers FAQ promotion. Tune with real data. (P5/P7.)
4. **Encoder staging from OneLake** — confirm the Eventhouse callout policy can fetch the ONNX from OneLake directly, or whether same-tenant blob storage is needed. (P2.)
5. **Deploy as a CLI verb (`rlat fabric deploy`) vs notebook-only** — decide after P4. (P1/P4.)
6. **Tokeniser in the sandbox** — bundled artifact vs sandbox-available package; confirm size against the 1 GB limit. (P2.)
7. **`ai.*` capacity cost** — measure consolidation-run cost; confirm graceful no-LLM degradation is acceptable as the default. (P7.)

---

## Risks

| Risk | Mitigation |
|---|---|
| Parity (G1) fails — sandbox encoder diverges from `field/encoder.py`. | P2 is an explicit gate before any further phase. Replicate the recipe exactly; pin the ONNX revision. |
| Latency unacceptable in practice (>3 s felt). | Decision 9 accepted it "for now"; the semantic cache (P6) softens repeats; revisit only if G2 shows it harms usability. |
| Python plugin disabled by tenant policy. | Documented prerequisite; no code workaround — the no-external-service constraint requires it. Surface early in `FABRIC_DATA_AGENT.md`. |
| Corpus exceeds the ~1 M-vector Eventhouse ceiling. | Out of scope at current corpus sizes; note in honest limits; partition per knowledge model if it ever arises. |
| Lens compounding (G3/G4) does not materialise. | Claims are pre-gated to Tier 2; the loop's Tier-1 value (grounded answers, no external service) stands regardless. |
| Scope creep across 11 phases. | Each phase is independently committable with its own acceptance check; the resumption protocol keeps sessions bounded. |
