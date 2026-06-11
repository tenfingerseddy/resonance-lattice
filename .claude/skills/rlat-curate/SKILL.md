---
name: rlat-curate
description: >-
  Human-in-the-loop curation of an rlat knowledge model (.rlat): surface the
  self-audit findings (gaps, contradictions, stale external facts), then let the
  USER decide each one — approve a drafted fill, hand-provide an authoritative
  source, or skip — so nothing the user hasn't sanctioned ever lands. Trigger when
  the user says "curate my knowledge base", "let me review before adding", "I'll
  approve what gets added", "human in the loop", "I'll provide the source", "review
  the findings with me", or wants final say over what enters the corpus's earned
  layer. This is the approval/authority front-end over `rlat-gap-scan`,
  `rlat-contradictions`, and `rlat-refresh-facts`. NOT for: autonomous filling (the
  loop does that unattended); answering questions (use the `rlat` skill).
allowed-tools: Bash(rlat:*), WebFetch, WebSearch, Read, Write, Edit, Glob, Grep
---

# rlat-curate — the human decides what enters the corpus

The self-audit and its skills can run unattended, but some `.rlat`s want a human in
the loop: the user reviews each finding and either **approves** a drafted fill,
**hand-provides** the authoritative fact/source, or **skips** it. This skill is that
front-end. Its whole point is **authority + trust tier**: what the user vouches for
lands at the highest trust; nothing lands without the user's say-so.

The trust model is explicit and ordered (the seed prior in
`store/insight.py::seed_confidence`):

> **user** ≥ **verified-external** (≥2 agreeing web sources) ≥ **single-external** ≥ **corpus** (synthesis)

A user-vouched fact seeds the band at the *user* tier — the highest starting trust —
because the user is the authority. Every fill, even a user-vouched one, is still
**faithfulness-gated**: the user provides a *source*, not just an assertion, and the
claim must trace to it. A fill is born at its tier's trust and then **earns or loses**
rank by outcomes; the tier is the *prior*, not a permanent verdict.

---

## The loop

### 1. Surface the findings (free)
Find the `.rlat` (`Glob "*.rlat"`; prefer the project-named one or ask). Gather what
the corpus's self-audit flags, by driving the sibling skills / CLI:

- **Gaps** — `rlat-gap-scan` (questions the corpus can't answer).
- **Contradictions** — `rlat audit <km> --shape --min-cosine 0.85 --with-text --format json`,
  then judge stance yourself (the `rlat-contradictions` rubric).
- **Stale external facts** — `rlat audit <km> --external --format json`, then re-fetch
  + re-judge (the `rlat-refresh-facts` rubric).

Present them to the user as a short, plain list — what's missing / conflicting /
out-of-date, and where it would belong. Don't fix anything yet.

### 2. Let the user decide each finding
For each, offer the three choices and **wait for the user**:

- **Approve a draft** — you propose a grounded, cited fill (for a gap: researched
  from authoritative sources with cross-source agreement; for a stale fact: the
  corrected value from the re-fetched source). The user okays it → it lands at
  **verified-external** (≥2 agreeing sources) or **single-external** (one source).
- **Hand-provide the source** — the user gives the authoritative fact *and* a source
  (a URL, a doc, a quote). You verify the claim traces to that source (read it —
  free), then it lands at the **user** tier (highest trust).
- **Skip** — nothing lands. A finding the user doesn't sanction never enters.

If the user gives a fact with **no** source, ask for one. The user is the authority
on *what's true*, but the band still records *where it traces* — a sourceless
assertion can't be faithfulness-checked and shouldn't land as settled.

### 3. Land it at the right tier (only on the user's go)
A sanctioned fill lands via the promotion path with its provenance set. The FREE
(no-API) way — you verified it yourself, so no judge client is needed:

- **Approved web draft (≥2 agreeing sources):**
  `curator.agent_fill.land_external_fact(km, question, claim, sources, provenance="verified_external", faithfulness=<your confidence>)`
  where `sources` is the `[{"url","text"}, …]` you fetched + judged to agree. It
  builds the external evidence and lands via the caller-verified path; it requires
  ≥2 DISTINCT sources structurally.
- **User-vouched fact (with the user's source):** the same helper with
  `provenance="user"` (the user is the authority), or
  `store.promotion.promote_if_faithful(km, question=…, answer=<claim>, evidence_passages=[…], client=None, faithfulness=<conf>, provenance="user")`
  directly. `client=None` skips the LLM gate because YOU verified the claim traces
  to the user's source; an explicit `faithfulness` is required (no silent unverified
  write), and the compression + ≥2-citation + trust gates still apply.

(If you'd rather have the metered model do the grounding check, pass an
`anthropic`-shaped `client` instead of `None` — a one-at-a-time user-initiated write,
so the cost is negligible.)

Land **one finding at a time**, on the user's explicit go. Report what landed, at
what tier, and from what source.

---

## Honest limits (state them, never paper over)

- **Authority is not infallibility.** A user-vouched fact seeds *high* trust, not
  *permanent* truth — it still earns or loses rank by outcomes, and a later
  authoritative correction supersedes it. The tier is a prior.
- **Every fill is still faithfulness-gated.** "The user said so" is not enough to
  land a *sourceless* claim — require a source so the band stays traceable. This
  protects the user from their own typo as much as anything.
- **Surfacing is probabilistic.** The findings (gaps/contradictions/stale) are
  candidates, not certainties (see each sibling skill's limits). Curation is
  exactly where a human corrects those misses — which is the point of the loop.
- **One at a time, reversible-minded.** Landing grows the earned layer; do it
  deliberately, per finding, on the user's word — never in a silent batch.

## Reference
- `store/insight.py::seed_confidence(provenance=…)` + `provenance_tier` — the trust-tier
  prior (user ≥ verified-external ≥ single-external ≥ corpus).
- `store/promotion.py::promote_if_faithful(provenance=…)` — the gated landing path.
- `rlat-gap-scan` / `rlat-contradictions` / `rlat-refresh-facts` — the finding sources.
- `curator/external_fill.py` — the cross-source-agreement fill (the verified-external tier).
