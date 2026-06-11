---
name: rlat-contradictions
description: >-
  Find conflicting facts INSIDE an rlat knowledge model (.rlat) — two documents
  that disagree about the same thing — using cheap corpus geometry to surface the
  candidates and YOUR own reading to judge them, with no external API key and no
  metered cost. Trigger when the user asks "are there contradictions in my
  corpus?", "does my knowledge base disagree with itself?", "find conflicting
  facts", "is my documentation inconsistent?", "which docs contradict each
  other?", "audit my .rlat for conflicts", or wants the corpus to check its own
  consistency. This is the contradiction half of the corpus self-audit (the gap
  half is `rlat-gap-scan`). NOT for: answering a question (use the `rlat` search
  skill); finding what's MISSING (use `rlat-gap-scan`); auto-fixing a conflict
  (this v1 SURFACES for review, it does not edit the corpus).
allowed-tools: Bash(rlat:*), Read, Write, Edit, Glob, Grep
---

# rlat-contradictions — find where the corpus disagrees with itself

A contradiction is **two different source documents making incompatible claims
about the same thing** — different values for the same quantity, or mutually
exclusive statements. The classic real-world case: a doc was updated but the old
copy was never pruned, so the knowledge base now answers the same question two
ways. An agent retrieving from that corpus gets a coin-flip.

This skill is the corpus **auditing its own shape** — a property invisible to any
single search (it's a fact about the *set* of documents, not one passage). It runs
in two cheap stages:

1. **Geometry (free math, no LLM)** narrows the O(n²) pair space to a handful of
   high-cosine **cross-document** candidates — passages close enough to be *about
   the same thing*. This is the only scalable part; it's already what
   `store.self_audit` computes and stores in the `.rlat` at build/refresh.
2. **You judge stance (free — your reading, not an API).** Geometry only proves
   *same topic*; **most** high-cosine pairs are RESTATEMENTS, not contradictions.
   You rule which pairs genuinely conflict. The proven reason this is free: a
   subscription agent judges stance as well as a metered model on this task — so
   the in-session you *are* the judge (the same logic as `rlat-gap-scan` and
   `deep-research`). No Anthropic API key, no per-pair bill.

The `.rlat` stays self-contained: it provides the candidates and the text; you
provide the reading. No new service to set up.

---

## The loop

### 1. Find the `.rlat` and surface the candidates (free)
Find the corpus with `Glob "*.rlat"`; if several, prefer the project-named one or
ask. Then emit the **judge-ready** candidates — geometry pairs with their text
resolved — at a chosen cosine floor:

```bash
rlat audit <km>.rlat --shape --min-cosine 0.85 --with-text --format json
```

- **`--min-cosine 0.85`** is the right floor for prose. The stored default (0.92)
  is tuned for near-duplicates and **misses genuine prose contradictions**, which
  measured at ~0.81–0.92 (see the value-proof reference below). Lower → more
  candidates (more recall); you supply the precision, so a low floor is fine on a
  clean corpus. On a corpus dense with **metric tables**, keep the floor higher —
  table cells that share a number are the main false-positive source.
- The output is a JSON object; `high_cosine_pairs` is a list of
  `{cosine, a:{source_file,text}, b:{source_file,text}}`, **ordered by query demand**
  (`demand_ranked: true`) — the geometry ranks conflicts in the path of real query
  traffic (the corpus's stored telemetry) first, so you judge the ones users
  actually hit before academic ones. Judge from the top down. If it's empty, the
  corpus has no same-topic cross-doc pairs at that floor — report that plainly
  (a consistent corpus correctly yields ~0; don't manufacture conflicts).
- Large corpus? The pass is bounded; if it reports the contradiction pass was
  skipped (too large), say so rather than implying "no conflicts".

### 2. Judge each candidate's stance — the rubric (this is the whole point)
For **each** pair, decide from the two texts **alone**:

> Do A and B state **incompatible** values/facts for the **same quantity, measured
> the same way**? A restatement, paraphrase, summary, or a *different metric /
> column / measurement condition* that merely shares a subject or a number is
> **NOT** a contradiction. The same value reported twice is agreement.

Classify each pair:
- **contradict** — a single shared quantity, two values that cannot both be true.
  Name the quantity and the two values. If discernible, note which side looks more
  **authoritative or recent** (an official/primary source, a newer date).
- **restatement / agree** — same fact, compatible framings. Not a conflict.
- **different-thing** — they share a word/number but describe *different*
  quantities (the #1 false positive — be strict here, especially with tables).

Be strict about "same quantity, same measurement". The failure mode that makes
this useless is calling two *different* measurements that share a number a
"contradiction" (e.g. an oracle-ceiling value vs a net-of-echo value; "10/10 on
test X" vs "7/7 on test Y").

### 3. Report the confirmed contradictions (surface, don't fix)
For each **contradict** pair, report plainly:
- the two source files,
- the shared quantity and the two incompatible values,
- which side looks more authoritative/recent (or "unclear"),
- the cosine (how close the pair was).

Optionally write the confirmed list to a small markdown file the user can act on
(e.g. `.rlat-contradictions.md`). **Do not edit the corpus source files.**

### Resolving a confirmed contradiction (the gated ACT — only on the user's go)
Once the user agrees on which side is authoritative and what the resolved value is,
record it **non-destructively**:

- `curator.reconcile.reconcile_contradiction(km, passage_idx_a, passage_idx_b, resolution, provenance="user")`

This lands a high-trust **resolution claim** in the band's earned layer — the
authoritative value *and* that it supersedes the losing side — citing **both**
conflicting passages. It does **not** touch the corpus source files: the
conflicting passages stay; serve simply ranks the resolution first (it lands at a
high provenance tier). Write `resolution` as a genuine **synthesis** (the winning
value + the supersede note) — a verbatim copy of a cited passage is rejected (it
adds no decision), but beyond that the synthesis quality is your responsibility (on
the free path the tool trusts your resolution; pass a judge `client` to gate it
against the passages). Do this **one conflict at a time, on the user's explicit
decision** — the user picks the winner; the tool only records it.

---

## Honest limits (state them, never paper over)

- **Geometry only catches surface-similar conflicts.** A genuine contradiction
  whose two sides are phrased in *very different words* can fall below the cosine
  floor and never be surfaced. The floor is a recall/precision knob, not a
  guarantee of completeness — "no contradictions found" means "none among the
  same-topic pairs", not "the corpus is fully consistent".
- **Your stance verdict is a probability, not a proof.** On corpora dense with
  similar-looking numbers (benchmark tables, metrics), expect false positives;
  a stronger model is *not* reliably better here (measured). That is exactly why
  v1 **surfaces for review** and never auto-edits.
- **Same-file conflicts are skipped by design.** A single document evolving within
  itself ("we used to think X, now Y") is the author's documented history, not two
  sources disagreeing. This scan is cross-document only.
- **Exact duplicates are excluded.** Identical text repeated across docs
  (boilerplate/includes, cosine ≈ 1.0) is redundancy, not contradiction.

## Why not just trust the cosine?
Because a high cosine only means *same topic* — most high-cosine cross-doc pairs
are restatements. The cosine narrows the search cheaply; **your reading** is what
separates "two docs about the same thing" from "two docs that disagree". That
split — cheap geometry filter, free agent judge — is the whole design.

## Reference
- `.claude/plans/insight-band-outcome/NIGHT_LOG.md` → "VALUE PROOF — the
  contradiction moat" (2026-06-06): the measurement behind this skill — it found a
  real stale-summary contradiction in the project's own notes; the candidate floor
  is a tunable recall/precision knob (4/4 recall + 2/2 safety at 0.80 on a clean
  corpus); ~50% judge precision on metric-dense prose (Sonnet no better).
- `rlat audit <km> --shape` — the geometry surface this skill drives.
- `rlat-gap-scan` — the sibling skill for what the corpus is *missing*.
- `curator.reconcile.judge_contradictions` — the metered (API) equivalent of the
  judge step, for non-interactive/batch use.
