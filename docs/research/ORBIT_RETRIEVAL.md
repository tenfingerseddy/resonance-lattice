# Orbit retrieval: the graph, without the graph

**Status:** invention under a design constraint set by the owner's verdict on earlier
graph work — with machine-checked receipts and a decisive ablation ·
**Date:** 2026-08-10 · **Receipts:**
[`demos/demo8_orbit_retrieval.py`](demos/demo8_orbit_retrieval.py) (all checks pass) ·
**Supersedes:** the `graph/` sidecar proposal in [`LATENT_GRAPH.md`](LATENT_GRAPH.md) §5.

## 0. The concession, first

The owner's verdict: earlier graph attempts were discarded — no meaningful
improvement, an add-on rather than core. Applying that bar to my own previous
turn: kNN + small-world navigation is textbook (HNSW already does it inside
`field/ann.py`); typed gates are metadata filtering; and a `graph/` sidecar is
precisely the add-on shape that failed before — new state, new surface, same
estimator. Withdrawn. What survives from that investigation is two *tools*, not
features: the chain-curvature regime diagnostic, and k-planes mining of
displacement fields. This document is what they build when the verdict is taken
as a constraint:

> **No stored edges. No new query surface. If corpus structure is to help, it must
> enter the two things the product already is — the score function or the
> coordinates.** Edges are instances; transformations are laws. Store the group,
> not the graph.

## 1. The object

At build time, mine the corpus's recurring displacement structure once: k-planes
clustering (validated in demo 7) over a structural displacement field — here,
cross-file nearest-neighbour displacements `e_j − e_i` — yields K prototypes, each
a 2-D plane with an angle distribution, a support count, and exemplar edge receipts.
Total state: K·(2d+1) floats (~6 KB at K=4, d=768). Every edge is then discarded.

A mined family can be one of two kinds, and **the mined angle statistics say
which** (the same measure-first philosophy as demo 7's curvature diagnostic):

- **Directed** (median |θ| large, consistent sign): the family is a *move* the
  corpus makes. Retrieval mode: **orbit-max** — score every query variant in the
  orbit {q} ∪ {R_c^{±1} q}, take the max, tag each hit with the rotor that found
  it. 2K+1 matvecs through the existing `band @ q` path; per-hit receipts.
- **Symmetric** (median |θ| ≈ 0): the family is *spread*, not motion — a nuisance
  subspace along which the corpus varies without meaning it (register, boilerplate,
  section style). Retrieval mode: **quotient** — remove the mined planes from the
  coordinates and retrieve flat. Zero query-time overhead.

The two modes are one object. **Proposition** (checked numerically to 1e-12 in
demo 8a): averaging a plane rotor over the cyclic group it generates annihilates
its plane exactly and is an orthogonal projection —

```
(1/m) Σ_j R(θ_j)  =  I − uuᵀ − vvᵀ,      θ_j = 2πj/m
```

so group-invariant retrieval **is** flat retrieval in a projected band: the graph
collapses into the coordinates. In rlat terms the quotient is just another band —
the product's central object — computed at build, listed in `metadata.bands`,
served through the unchanged retrieval path.

## 2. What the measurements say (real text; LSA stand-in — caveat in corpus.py)

**The mode selector fired "symmetric"** on this corpus: mined cross-file angles
{0.01, 0.03, 0.07, 0.19} rad. Transport should therefore buy nothing and the
quotient should carry the value. Both predictions held:

**Cross-register retrieval** (query = a DESIGN passage of a benchmark, target = the
same bench's VERDICT passages, own file excluded; 4 bench directories — a small,
direction-of-effect eval, stated as such):

| mode | verdict-recall@10 | MRR | topical precision@10 (guard) |
|---|---:|---:|---:|
| flat | 75.9 | 0.306 | 83.6 |
| orbit-max (transport) | 75.9 | 0.303 | 83.2 |
| random-direction removal (control) | 75.9 | 0.312 | 83.7 |
| global-PCA removal ("all-but-the-top", 8 dims) | 83.3 | 0.334 | 85.1 |
| **quotient (displacement-mined, 8 dims)** | **88.9** | **0.333** | **85.6** |

Three things the ablation ladder establishes:

1. **The gain is real and the guard improves with it** (+13.0 recall over flat;
   topical precision up, not down).
2. **Targeting is load-bearing twice over**: random directions reproduce nothing
   (75.9), and global PCA — the known trick, disclosed in §4 — captures barely
   half the gain (83.3). The displacement field of a *structural edge family*
   knows which directions are nuisance better than global variance does.
3. **Transport honestly buys nothing here** — as the near-zero mined angles said
   it wouldn't. Orbit-max stays in the design for corpora whose mined families are
   directed (demo 7c shows recovered directed rotors apply 200/200 on unseen
   nodes); it is not claimed for this one.

**Reading-continuation** (demo 7's hard task, held-out files): flat 27.0 vs
orbit 26.8 hit@5 — no rescue, exactly as the cloud-regime diagnostic requires.
Negative kept on display.

## 3. Why this is core, by this repository's own definition

- **It is a band.** The quotient ships as a band variant computed at
  `rlat build`/`refresh` (the format already supports band slots; deltas re-project
  as they do today). Not a sidecar, not a new query language — the estimator itself
  changes.
- **Zero marginal cost at query time** for the quotient mode; the orbit mode, where
  a corpus earns it, is 2K+1 matvecs on the existing code path.
- **Receipts throughout, in both directions.** Each removed plane carries its
  support count and exemplar edge pairs ("this direction separates DESIGN.md from
  VERDICT.md registers — 339 edges — examples: …"); each orbit hit carries the
  rotor that found it. Nothing silent, everything inspectable — and the removal is
  exactly invertible (the plane is stored), so `rlat lens`-style review/undo
  applies.
- **It answers the falsified-graph history** rather than repeating it: what failed
  was edges-as-artifacts (state added beside the estimator). What is proposed is
  laws-as-coordinates (the estimator improved, no state to maintain beyond K
  planes that rebuild with the band).

## 4. Prior art, disclosed from memory

Removing dominant or common components from embeddings is a known family
("all-but-the-top" post-processing, common-component removal, whitening) — that is
precisely why it sits in the ablation as a named alternative, and it loses to
displacement-mined targeting by 5.6 recall points here. Nuisance-direction removal
supervised by *labels* is classical (LDA's within-class scatter). To my knowledge
the specific composition — nuisance/relation subspaces mined **unsupervised from
structural displacement fields** (cross-document neighbour displacements), unified
with directed transport under one group object, with a measured statistic selecting
quotient vs orbit, and per-plane receipts — is new. No literature search was run
this session (per the standing instruction); a pass is required before any external
novelty claim.

## 5. Pre-registered production bench (kill criteria stated now)

To run where the pinned encoder lives, before any product decision:

1. **Floor**: BEIR-5 with the quotient band must hold the locked 0.5144 mean
   nDCG@10 within ±0.002. A larger drop kills the mode outright regardless of
   other gains.
2. **Register eval**: on `fabric-docs` (or `powershell-docs`), define register
   families from existing structure (how-to vs reference vs troubleshooting paths);
   queries drawn from one register, relevant targets in another (same topic).
   Quotient must beat BOTH flat and global-PCA removal by ≥ 2 nDCG@10 points.
   Beating flat but not PCA shrinks the claim to "ship the simpler PCA variant".
3. **Angle audit**: publish the mined angle distributions per corpus. If some
   production corpus mines a *directed* family (median |θ| > 0.3 with consistent
   sign), run the orbit-max arm there; otherwise orbit stays research.
4. **The product test**: the 63-question Fabric hallucination bench with quotient
   retrieval feeding the existing pipeline — answerable accuracy and hallucination
   rate must not regress, and cross-document questions should improve. If nothing
   moves, this joins the falsification ledger with its numbers, next to the
   previous graph attempts.

## 6. Limits

- LSA-scale evidence; the cross-register eval is small (4 bench directories) and
  is claimed only as direction-of-effect. §5 is the real test.
- K=4 and the cross-file edge family were the first reasonable choices, not tuned;
  the production bench should sweep K ∈ {2,4,8} and edge families (cross-file,
  cross-directory) — with the sweep pre-registered, not fished.
- The quotient deletes 2K of d dimensions for *all* queries; a corpus whose
  meaning genuinely lives in a mined plane would be harmed — the BEIR floor (§5.1)
  is the guard, and per-plane receipts make any harm attributable.
- The corpus here includes this research folder itself (self-reference quirk of
  measuring on the host repository); benchmarks/ evals are unaffected, but
  production numbers are the ones that count.
