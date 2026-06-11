---
name: rlat-refresh-facts
description: >-
  Check whether the EXTERNAL (web-fetched) facts inside an rlat knowledge model
  (.rlat) are still true in the live world, by re-fetching each fact's source and
  re-reading it yourself — no API key, no metered cost. Trigger when the user asks
  "are my fetched facts still current?", "did anything I added from the web go
  stale?", "re-check my external facts", "refresh the facts in my .rlat", "is the
  external info in my knowledge base out of date?", or wants the corpus's outside
  facts re-verified against the live world. This is the staleness `rlat refresh`
  CANNOT fix — refresh only re-syncs corpus files to disk; an external fact's truth
  lives on the web. NOT for: corpus drift (use `rlat refresh` / `rlat reverify`);
  finding contradictions (use `rlat-contradictions`); finding gaps (`rlat-gap-scan`).
allowed-tools: Bash(rlat:*), WebFetch, Read, Write, Edit, Glob, Grep
---

# rlat-refresh-facts — re-check external facts against the live world

A `.rlat` can carry **external facts** — claims fetched from the web for questions
the corpus couldn't answer (the `external_fill` path). They were verified true *when
fetched*. But the world moves on: "the latest version is X", "the current price is
Y", "the record holder is Z" all go stale, and **`rlat refresh` cannot catch it** —
refresh only re-syncs corpus files to their local sources; it never touches the web.

This skill is that missing check. It re-fetches each external fact's cited source
and asks: **does the source still say this?** It runs free, on **your** reading of
the re-fetched page — no Anthropic API key (the same agent-as-judge basis as
`rlat-contradictions` and `rlat-gap-scan`).

It **surfaces** stale facts for review; it does **not** edit the corpus. Re-authoring
from the fresh source or retiring the stale fact is a separate, user-approved step.

---

## The loop

### 1. List the external facts + their sources (free, no fetch)
Find the corpus with `Glob "*.rlat"`; if several, prefer the project-named one or
ask. Then enumerate what was fetched from outside:

```bash
rlat audit <km>.rlat --external --format json
```

Output is a list of `{claim_id, state, content, source_urls}`. If it's empty, the corpus
has no external facts — report that and stop (nothing to re-check; a corpus-only
`.rlat` is fine). Each external fact is backed by **two or more** cited URLs (the
fill required cross-source agreement).

### 2. Re-fetch each source and judge freshness (you, free)
For each external fact, `WebFetch` **each** of its `source_urls` to get the page's
**current** content. Then decide, from the re-fetched content alone:

> Does the current content still support the claim — the same value/fact it was
> verified against? Wording may differ; additions that don't contradict are fine.

Classify each fact:
- **fresh** — the current sources still support it. No action.
- **stale** — the current sources contradict it, no longer state it, or have moved
  past it (a newer version/value/result). The fact is out of date — **surface it**.
- **unknown** — a source 404'd, was blocked, paywalled, or moved, so you could not
  read the current content. Report it as *uncheckable*, **never** as stale (a failed
  fetch is not evidence the fact changed).

A fact is **stale** only if the re-fetched content actively disagrees. If you can't
re-fetch enough of its sources to tell, it's **unknown**, not stale.

### 3. Report (surface, don't fix)
List each **stale** fact: the claim, the URL(s) that no longer support it, and what
the source now says instead (the likely new value). List **unknown** facts
separately as "couldn't re-check". Leave **fresh** ones unmentioned unless asked.

If the user wants to **act** on a stale fact, that's the create/fix tier (one fact
at a time, on their go): re-fetch and re-author the corrected fact from the now-
current sources (it lands low-trust, cited, and earns its place over time), or
retire the stale one. **Do not** silently edit the corpus as part of this scan.

---

## Honest limits (state them, never paper over)

- **A failed fetch is not staleness.** 404 / blocked / paywalled / moved → *unknown*,
  not stale. Never flag a fact stale because you couldn't read its source.
- **Your freshness verdict is a probability.** A page can support a fact in wording
  you misread, or drop it for unrelated layout reasons. Surface stale facts as
  *candidates to re-check*, not settled retirements — which is why this scan never
  auto-edits.
- **Only what was fetched can be re-checked.** This re-verifies external facts; it
  says nothing about corpus drift (use `rlat refresh`) or facts no one fetched.
- **Cost is your fetch budget.** One `WebFetch` per cited URL; a fact with many
  sources costs more. Re-check the facts most likely to age (versions, prices,
  "current"/"latest" anything) first.

## Why this is separate from `rlat refresh`
`rlat refresh` re-syncs corpus passages to their local source files — it fixes
*corpus drift*. An external fact has no local source to re-sync; its truth is on the
web. This skill is the only thing that re-checks it. (Corpus drift was deliberately
demoted in the self-audit for exactly this reason: refresh already fixes it; world-
staleness is the freshness problem that actually needs a re-fetch.)

## Reference
- `src/resonance_lattice/store/external_freshness.py` — the library mechanism:
  `external_claims` (the free enumeration this skill drives) and
  `recheck_external_freshness` (the metered/automated equivalent of the judge step).
- `rlat audit <km> --external` — the enumeration surface.
- `src/resonance_lattice/curator/external_fill.py` — how external facts get ADDED
  (the cross-source-agreement fill this skill later re-checks).
- `rlat-contradictions` / `rlat-gap-scan` — the sibling free self-audit skills.
