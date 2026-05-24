# rlat × Fabric Data Agent — Manifesto

**Status**: v1. The thesis, motivations, value claims, scope, and falsifiable claim that constrain every downstream decision.
**Last updated**: 2026-05-17.
**Companions**: [fabric-data-agent-architecture.md](fabric-data-agent-architecture.md) (research, locked decisions, mechanism, schemas); [fabric-data-agent-roadmap.md](fabric-data-agent-roadmap.md) (phased build sequence, acceptance gates, open questions).
**Related prior plans**: [fabric-udf-integration.md](fabric-udf-integration.md) (the external-assistant consumer — shipped); [lensed-knowledge-manifesto.md](lensed-knowledge-manifesto.md) (source / insight / lens model); [agent-harness-manifesto.md](agent-harness-manifesto.md) (memory, intent, outcome ledger, closed-loop learning).

---

## The pitch

> **rlat as a grounded knowledge source for Microsoft Fabric data agents — and the same `.rlat`, unchanged, for external LLM assistants. It runs entirely inside the tenant with no external services, and it learns from every question asked of it.**

Three claims worth leading with:

1. **Grounded, traceable answers.** Every claim a data agent makes from rlat traces to a specific source passage. rlat owns *grounding* — faithful representation of what the corpus says — never *truth*.
2. **One knowledge model, many consumers.** The same `.rlat` serves a Fabric data agent (via a KQL database) and external LLM assistants (via the shipped UDF / `fabric://` skill). Build once, deploy to both.
3. **A learning loop that compounds.** The query log becomes memory; memory distils into an earned insight layer; the knowledge model is measurably more useful at session N+1 than at session N.

---

## Motivation

Microsoft Fabric is the primary niche. Inside Fabric, the **data agent** is how natural-language Q&A reaches users — it translates a question into a query (NL2KQL / NL2SQL / NL2DAX), runs it against a data source, and synthesises an answer.

A data agent has a structural blind spot. It only queries **structured rows**. Ask it "how should I design a medallion lakehouse for slowly-changing dimensions" or "why would I choose a warehouse over a lakehouse here" and it has nothing — that answer is not a row in a table. It is conceptual, grounded knowledge.

rlat is exactly that missing organ. A `.rlat` built over Fabric documentation plus the user's own artifacts (ADRs, notebook code, lakehouse schemas) gives the data agent the ability to answer the *why / how / which-pattern* class of question — with every claim traceable to a source passage.

The obstacle is a format mismatch: a data agent consumes **Fabric data sources** (lakehouse, warehouse, semantic model, KQL database, mirrored database), not a retrieval library. The bridge is a **KQL database / Eventhouse** — the one data-agent source that can natively store vectors and compute similarity in-query.

And the same bridge does not have to be single-purpose. The `.rlat` that backs the data agent is the same artifact that the shipped Fabric UDF integration ([fabric-udf-integration.md](fabric-udf-integration.md)) exposes to external LLM assistants (Claude Code, Cursor). One knowledge model, two consumer surfaces.

---

## North Star alignment

This plan is a direct expression of the three [CLAUDE.md](../../CLAUDE.md) North Star principles:

| Principle | How this plan embodies it |
|---|---|
| **1 — Target: maximum value, minimum effort. No required external services.** | The design uses **zero external services**. The query encoder (gte-modernbert) runs locally inside the Eventhouse Python sandbox. No Azure OpenAI deployment, no API keys, no third-party vector database. Data never leaves the tenant. |
| **2 — Engine: continuous self-improvement.** | The learning loop — query log → distillation → earned insight layer — is the closed-loop learning engine, deployed onto Fabric primitives (Eventhouse, Activator, notebooks). |
| **3 — Structure: context × tools.** | Context = the source / insight / lens layers of a `.rlat`. Tools = the Fabric data agent and external LLM assistants that leverage it. The product wins on the intersection: grounded retrieval that compounds. |

---

## What we are betting on

**Bet 1 — Grounding is the differentiator, not retrieval.** Every Fabric "RAG tutorial" wires up a vector store. The defensible product is not a better retriever; it is answers whose every claim traces back to the corpus, surfaced inside the tool users already have (the data agent). Traceability is the feature.

**Bet 2 — One rlat, many targets, is worth more than a Fabric-only build.** rlat is a product with many deployment targets; Fabric is its best-fitting one. The `.rlat` stays canonical and encoder-pure so the *same file* serves the data agent, the external-assistant UDF, and the local CLI. A Fabric-only fork would be smaller but strategically dead.

**Bet 3 — The learning loop must actually work.** A static knowledge model is commodity. A knowledge model that turns every asked question into earned, cited insight is not. This is the hardest part to get right and the part most worth getting right.

---

## Value claims — calibrated

Claims are tiered by what evidence currently supports them. This discipline is deliberate: it is the same discipline that kept the `rlat optimise` overclaim out of [docs/internal/HONEST_CLAIMS.md](../../docs/internal/HONEST_CLAIMS.md). **No quantified claim ships before it is measured on the actual Fabric deployment.**

### Tier 1 — claimable now (true by construction)

- **No external services.** No Azure OpenAI, no API keys, no third-party vector DB. Verifiable yes/no.
- **Data never leaves the tenant.** Corpus, queries, embeddings all stay in OneLake / Eventhouse.
- **No new infrastructure, no per-query embedding API cost.** Native Eventhouse; embedding is sandbox compute already paid for via capacity.
- **The data agent works unmodified.** rlat appears as an ordinary KQL database source; NL2KQL just works.
- **Every answer traces to source passages.** Grounding, not truth.
- **Retrieval quality is preserved exactly.** Eventhouse computes exact brute-force cosine over the *same* gte-modernbert vectors as the `.rlat` — no deployment-induced degradation.
- **One portable `.rlat`, many consumers.** Build once; serve the data agent and external assistants.
- **Drift-aware.** Content-hashed passages make stale answers detectable when source documents change.

### Tier 2 — claimable as a capability, never quantified yet

- **Closed-loop learning.** The query-log → distillation → insight-layer pipeline is real. The *magnitude* of improvement is benchmark-gated (see Falsifiable claim).
- **Per-team perspective (lens).** Trust weights and per-team re-ranking are real features. Portability and composition claims are gated on the lensed-knowledge benchmarks.
- **Knowledge-gap dashboard.** Frequent query + uniformly weak retrieval = a documented blind spot. True by construction.
- **Semantic cache.** Repeated questions return faster and *consistently*.

### Tier 3 — must be measured on this deployment before any number is stated

- End-to-end answer quality of the Fabric **data agent**. (rlat's own deep-search loop measured 92.2% answerable / 0% hallucination on the Fabric docs corpus — *indicative of the grounded approach*, but a different orchestrator; not transferable.)
- Any recall / precision figure for the Eventhouse deployment.
- Lens compounding deltas.

### Never claim

- That answers are "accurate" or "correct." rlat represents what the corpus says, not what is true. See [docs/internal/GROUNDING_MODEL.md](../../docs/internal/GROUNDING_MODEL.md).

---

## Scope

**In scope**

- A KQL database / Eventhouse façade over a `.rlat`, queryable by a Fabric data agent.
- A pure-Python deploy pipeline: `.rlat` → Eventhouse vector table.
- Query embedding inside the Eventhouse Python sandbox using rlat's gte-modernbert encoder.
- The learning loop: query-event capture, a scheduled consolidation notebook, an earned insight layer.
- The lens: per-team / per-role perspective applied as native KQL re-ranking.
- Consistency with the shipped external-assistant consumer ([fabric-udf-integration.md](fabric-udf-integration.md)).

**Out of scope**

- Replacing the data agent's structured sources. rlat is an *additional* source, not a substitute.
- A bespoke reader / answer generator. Synthesis is done by the data agent (its LLM) and by external assistants.
- PySpark. All notebooks are pure-Python (Polars / DuckDB).
- Path B (re-encoding the corpus with Azure OpenAI). Reversed — see architecture decision log.

---

## Falsifiable claim

> On the Microsoft Fabric documentation corpus, the Eventhouse-deployed `rlat_search()` returns a top-k set **identical** to local `rlat search` over the same `.rlat` (retrieval parity), and across a 20-question repeat-and-vary session the **insight-layer hit-rate rises measurably** with no loss of grounding fidelity.

Two halves, two gates:

1. **Parity** — falsified if Eventhouse retrieval diverges from local `rlat search` beyond float tolerance. This is the floor; if parity fails, the deployment is wrong.
2. **Compounding** — falsified if insight-layer hit-rate does not rise with repeated/varied use. This claim inherits the gate from [lensed-knowledge-roadmap.md](lensed-knowledge-roadmap.md): it is **not stated publicly with a number** until the lensed-knowledge dogfood benchmark passes and a Fabric-specific run confirms it.

---

## Non-negotiables

These constrain every downstream decision. Changing one requires revising this manifesto.

1. **The `.rlat` stays canonical and encoder-pure.** Fabric is a *compile target*, not a host. rlat compiles; Fabric runs.
2. **No external services.** If a feature requires Azure OpenAI or any provisioned third-party service, it does not ship in the default path.
3. **One encoder everywhere.** gte-modernbert in the CLI, the UDF, and Fabric. No Fabric-special encoder.
4. **Pure-Python notebooks.** No PySpark dependency.
5. **Calibrated claims only.** Tiered as above; measured before quantified.
