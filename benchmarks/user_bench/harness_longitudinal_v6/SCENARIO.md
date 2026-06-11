# Stratus — running scenario for v6 bench

**Product.** Stratus, a fictional device-telemetry platform run on Microsoft Fabric.

**Volume.** 5 TB/day of device events. ~1B rows/day. Bursty (10× peaks during outages and patch rollouts).

**Sources.**
- Primary: Azure Event Hubs (JSON envelopes, schema evolves quarterly).
- Secondary: partner uploads — daily CSV drops to a Files folder under the lakehouse (one folder per partner; ~30 partners; varies in size from 1 MB to 5 GB).

**Storage.** A single Fabric workspace `ws-stratus-prod`. One lakehouse `lh-stratus`. Bronze / Silver / Gold tables in Delta.

**Three downstream consumers.**
1. **Ops dashboard** — Power BI on DirectLake. Refresh expectation: < 15 min behind real-time. Used by 24×7 on-call to spot device-fleet incidents.
2. **Weekly exec report** — Power BI. Refresh expectation: every Monday morning, < 60s p95 page load. Used by leadership.
3. **DS feature store** — Spark notebooks reading silver/gold parquet snapshots. Refresh expectation: daily; reproducibility for any past day is non-negotiable.

**The agent's role.** Senior Fabric data-engineering judgement. The agent is making design and operational decisions across a build arc — not writing all the code, but deciding what to build, what to change, what to leave alone. Trade-offs are required; pure facts aren't enough.

**Failure beats.** Five sessions present a problem caused by earlier choices. The agent must (a) trace it to the contributing decision, (b) decide whether to revisit or work around, (c) commit. These are the outcome-attribution beats — they give arrow2 a real signal to surface "this earlier decision led to this failure" type learnings.

**Prior-session context discipline.** Every prompt embeds an "earlier sessions established: …" block listing the *minimum* shared decisions needed. This is the floor; the agent's own memory (in arm_on) sits on top of this floor and can supply more nuance.
