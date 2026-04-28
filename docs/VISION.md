# Resonance Lattice — Vision

> Resonance Lattice exists to make human knowledge more useful, portable, inspectable, and user-controlled.

We believe the future of AI is not about replacing knowledge with large language models, but about giving people better ways to structure, retrieve, share, and reason over their own knowledge.

Users should have fine-grained control over **what** knowledge an LLM can access, **when** it can access it, and **how** it is allowed to use it. Knowledge should not be passively absorbed into opaque systems or treated as a generic input to model behaviour. It should remain visible, intentional, permissioned, and owned by the people who create it.

This is the core product philosophy. Everything in v2.0 — the single-recipe encoder, the absent reader layer, the verified-retrieval contract, the grounding-mode directive, the absence of a hosted vector store — flows from it.

---

## Why this exists

The dominant pattern today is to upload your knowledge to an LLM provider, pay per token to query it, and trust that whatever the model produces in response is grounded in what you sent. Three things break under that pattern:

1. **Authoritativeness.** LLMs are trained on the internet. The internet is a sub-par knowledge source for any specific domain — your codebase, your compliance rules, your medical records, your research notes. Treating the LLM as the authoritative voice means deferring to that sub-par source by default, then trying to override it with retrieval that the model is free to ignore.
2. **Ownership.** Knowledge sent to a hosted service becomes legible to that service forever. The structure (what you indexed, how you chunked it, which embeddings represent it) is opaque to you and proprietary to them. You rent intelligence, you don't own it.
3. **Control.** Once knowledge is in the prompt, the model uses it however its training disposes it to. You cannot say "use this for citations only", "refuse to answer if these passages drift", "never blend this with your training". The LLM holds the steering wheel.

Resonance Lattice inverts all three. The knowledge model (`.rlat`) is a single file you own. Embeddings are produced by an encoder that runs on your machine with no API key. Retrieval surfaces verified passages with source coordinates and drift status — the LLM gets *evidence*, not *truth*. Grounding modes (`augment` / `knowledge` / `constrain`) are an explicit directive to the consumer LLM about how strictly to ground in those passages, including the option to refuse on thin evidence. The user holds the steering wheel.

---

## Core principles

### Human-centred

Every innovation must serve a real user need, improve an experience, reduce friction, or create a better outcome. We are not chasing scientific novelty for its own sake. We build useful systems first.

### Simple and useful over novel

Complexity is only valuable when it can be distilled, and innovation matters only when it creates value. The best product is the one that makes a difficult thing feel obvious, elegant, and effortless — and ships only what measurement proves works.

### User-owned knowledge

Users create knowledge. Organisations hold knowledge. Communities refine knowledge. AI should reason over that knowledge, under user direction, not own it.

### Fine-grained control

Users should decide exactly what knowledge an LLM can use, how it can use it, and for what purpose.

---

## What we build

A new layer for knowledge: one that helps people encode, inspect, retrieve, govern, and share what they know without surrendering control to closed models, hidden infrastructure, or rented intelligence.

The shape today (v2.0):

- **Encode**: `rlat build` packages a corpus into a single `.rlat` file. One encoder, no knobs, runs locally.
- **Inspect**: `rlat profile`, `rlat compare`, RQL `evidence` / `locate` / `corpus_diff` / `near_duplicates` / `drift` / `contradictions` — every angle on what a knowledge model contains, what's drifted, what conflicts.
- **Retrieve**: `rlat search` with verified citations and drift status on every hit. Cross-knowledge-model `compose` and `merge` for federated queries. `--verified-only` for hard-evidence workflows.
- **Govern**: `--mode {augment|knowledge|constrain}` directs how the consumer LLM treats the evidence. `--strict` aborts on drift. `rlat freshness` is the read-only gate before sync. The user controls the contract.
- **Share**: `rlat convert <km> --to {bundled|local|remote}` reshapes storage modes without rebuilding. Hand off a knowledge model as a single file, ship it via HTTP, or open it for editing — the choice is the user's, not the platform's.
- **Refresh**: `rlat refresh` (local) and `rlat sync` (remote) apply incremental deltas. Edit your source; sync your knowledge model in seconds; the optimised band re-projects from the new base for free.

The shape after v2.0 is set out under [Post-v2 priorities](#post-v2-priorities) below.

---

## Post-v2 priorities

After v2.0 launches, the focus shifts from a powerful retrieval concept to a deeply useful knowledge platform:

1. **MCP server.** Make Resonance Lattice directly usable inside modern AI agent workflows. The current CLI-first surface ships first because it's simple, scriptable, and provider-agnostic; an MCP wrapper extends the same `rlat skill-context` and `rlat search --format context` primitives into the tool-use protocol every major LLM provider is converging on.
2. **More source material types.** Expand beyond text documents into richer forms of user and organisational knowledge — chat transcripts, structured records, audio, code with semantic structure, design files. The encoder is the bottleneck; the format is general.
3. **Robust fundamentals.** Reliability, explainability, repeatability, and trust. Audit-driven engineering continues; every shipped feature pairs with a harness suite that locks the contract.
4. **Deep workflow integration.** Embed Resonance Lattice where people already work, not as another isolated tool. IDE extensions, document-editing surfaces, ticketing systems, code review.
5. **Free knowledge sharing platform.** Create a place where users can share portable knowledge structures openly and freely. The `.rlat` format is already designed for it — the platform is the missing piece.

Each of these inherits the principles above. We will measure before we ship. We will keep the surface small. We will keep ownership and control with the user.

---

## Condensed

Resonance Lattice exists to make human knowledge portable, useful, inspectable, and user-controlled.

AI should not be the source of truth. People, organisations, and communities create knowledge. AI should help reason over it, retrieve it, inspect it, and act on it — but only within the boundaries users define.

Our products are human-centred and simple by design. We pursue innovation only when it creates clearer workflows, better outcomes, and more elegant ways for people to work with what they know.

At the core of Resonance Lattice is a simple belief: users should control not only their knowledge, but also how AI systems are allowed to use it.
