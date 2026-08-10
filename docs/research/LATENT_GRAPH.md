# The latent graph: an ontology from vectors, with receipts

**Status:** investigation with machine-checked receipts — no product code changed ·
**Date:** 2026-08-10 · **Receipts:**
[`demos/demo7_latent_graph.py`](demos/demo7_latent_graph.py) (all checks pass;
re-runnable against any production `.rlat`) · **Companion:**
[`ROTOR_LENS.md`](ROTOR_LENS.md), [`ROTOR_DELTA.md`](ROTOR_DELTA.md).

The question, as posed by the repository owner: *"whether we can create a graph from
vectors. Connect vectors somehow in a way that makes them easier to traverse"* — and
ontology.

> **Addendum (same date, after owner feedback).** Earlier graph attempts on this
> product were discarded — no meaningful improvement, add-on rather than core. By
> that bar, the `graph/` sidecar proposed in §5 below is the wrong shape and is
> **withdrawn**. What survives from this investigation is its *tools* — the
> chain-curvature regime diagnostic (§2) and k-planes displacement mining (§3) —
> which feed the edge-free reformulation in
> [`ORBIT_RETRIEVAL.md`](ORBIT_RETRIEVAL.md): corpus structure entering the score
> function or the coordinates themselves, with no stored edges and no new query
> surface. The measurements in this document remain valid as measurements.

## The verdict

**Yes — and the graph is already latent in the band.** Nodes exist (passages);
edges, edge types, traversal laws, and a concept hierarchy are all *geometric
objects* recoverable from the embedding matrix plus the metadata the `.rlat` registry
already stores. But each layer needs its own mathematics, two of my starting
hypotheses died on contact with the data (both kept on display), and the single most
useful artefact of the investigation is a **one-number diagnostic that selects the
right traversal law per corpus** — so the design below measures before it assumes,
in this repository's own tradition.

What already exists in-repo and is *not* re-proposed here: a navigability graph for
**speed** (FAISS HNSW inside [`field/ann.py`](../../src/resonance_lattice/field/ann.py)),
single-hop `neighbors` in RQL, and clustering primitives in
[`field/algebra.py`](../../src/resonance_lattice/field/algebra.py). This document is
about the **semantic layer**: navigation, typed edges, and hierarchy as user-facing,
receipts-carrying objects.

All numbers below are from this repository's own docs (1,011 passages, LSA stand-in
embeddings — [`demos/corpus.py`](demos/corpus.py) states the caveat); the demo
accepts a `.rlat` path and re-runs every measurement against a production band.

## 1. Connectivity and navigation (7a): symmetry is load-bearing

A k-nearest-neighbour graph (k=8) over the band is one connected component, plus two
random long-range links per node (the small-world shortcut). Findings:

- **Directed kNN strands walkers.** Greedy routing on out-edges only reaches a
  random target 67.3% of the time — similarity says who *your* neighbours are, not
  who counts *you* as one; low in-degree nodes become unreachable.
- **Symmetrise the links and greedy reaches 100%.** This is precisely what HNSW
  does with bidirectional links; measured here in the open.
- **Backtracking buys efficiency**: HNSW-style best-first search reaches every
  target while touching a mean of **6 of 1,011 nodes (~1% of the corpus)**.

Design consequence: an RQL-level traversal surface needs a *symmetric* adjacency
sidecar; production point-search stays FAISS.

## 2. The regime diagnostic (7b): are documents paths or clouds?

My starting hypothesis — imported from the rotor work — was that traversal should
carry *momentum*: continue the geodesic that took you from passage k−1 to k to
predict k+1. **It failed, monotonically in momentum strength** (hit@5: greedy 24.7,
momentum-at-0.35 21.6, full momentum 13.1). The diagnostic that explains it:

> **Chain curvature** = mean cos(δ_k, δ_{k+1}) over reading-order chains, where
> δ_k = e_k − e_{k−1}. If passages are i.i.d. around a per-document centre, the
> shared middle term forces the value to exactly **−½**. A genuinely directional
> path gives **> 0**.

Measured: **−0.475 ± 0.005** (918 chain steps). Documents in this band are *clouds*,
not paths — there is almost no sequential geometry to exploit, and the ~0.025 gap
from −½ is all the drift there is. The regime predicts everything observed: momentum
must lose; the pure cloud tool (document centroid) can identify *which* document but
cannot rank *within* it (15.9); and the winner is a **typed metadata edge gating a
local similarity ranking**:

| next-passage predictor | hit@1 | hit@5 | MRR |
|---|---:|---:|---:|
| greedy (similarity to current) | 8.3 | 24.7 | 0.164 |
| momentum t=1.0 (path tool) | 3.9 | 13.1 | 0.087 |
| document centroid, ungated (cloud tool) | 4.4 | 15.9 | 0.105 |
| **typed gate (same document) + greedy** | **14.7** | **46.3** | **0.299** |
| typed gate + centroid (membership only) | 5.9 | 23.3 | 0.161 |

The gate is metadata the registry already stores (`source_file`, `char_offset`) —
**the cheapest edges in the graph are the ones rlat has carried all along.** Gated
greedy at 46.3 vs gated centroid at 23.3 also shows a real local-adjacency signal
beyond mere membership — weak, but exploitable once gated.

The diagnostic transfers: run the demo against a production `.rlat` and it
re-measures curvature and re-scores every law. Decision rule, registered now:
curvature > +0.1 on a production band → momentum ops earn a place; ≈ −½ → cloud
regime confirmed, ship gates + local ranking only.

## 3. Relations from raw pairs (7c): an emergent ontology's edges

If a relation is a consistent *transformation* — in the rotor-gate framing
([`ROTOR_DELTA.md`](ROTOR_DELTA.md)), a plane rotation — then displacement vectors
`y − x` of edge instances of one relation all lie in that relation's **2-D plane**.
Consequences, all machine-checked:

- **Mining tool**: relation discovery is **k-planes (subspace) clustering** of edge
  displacements, *not* k-means on directions (a first version used k-means: 57%
  purity; displacements of one relation fill a great circle, not a point — kept as
  a design lesson).
- **Planted-relation recovery**: 4 random rotor-relations in 64-d, 150 noisy edge
  instances each → k-planes (restart selection by the unsupervised objective, never
  by labels) recovers them at **99% purity**, recovers each plane to spectral error
  ~0.25, and the recovered rotors **apply correctly to unseen nodes 200/200** —
  typed neighbours can be *generated* from k stored prototypes rather than stored
  as O(N·k) labelled edges ("virtual edges").
- **An identifiability boundary, measured**: with fully random concepts the
  in-plane mass of a 64-d unit vector is ~√(2/64) ≈ 0.18, the rotational
  displacement barely clears the noise floor, and purity degrades to 85%. Relations
  are recoverable *where they act on entities that carry the relevant feature* —
  stated as a limit, not hidden.
- **Real corpus, no labels**: k-planes clustering of the 4,044 kNN-edge
  displacements finds a cluster **2.3× enriched** for adjacent-in-document edges —
  structural edge kinds the clustering never saw. Weak-but-real at LSA quality;
  the production-band re-run is pre-registered below.

Naming the discovered relations is deliberately out of scope for geometry: classes
and relations arrive as *sets of exemplar edges with passage receipts*, and naming
is a curation act — the `rlat-curate` loop (human-approved) is the natural namer,
consistent with this project's philosophy that nothing lands without consent.

## 4. Hierarchy for free (7d): broader/narrower with receipts

Single-linkage components across a similarity-threshold sweep (quantiles of the
kNN-edge similarity distribution) yield concept levels that **nest exactly** —
verified, not assumed — giving SKOS-style broader/narrower structure directly from
the band: 847 fine concepts → 573 → 352 coarse on this corpus. Every concept is a
set of passages with exact `source_file:offset+length` receipts. Production variant:
complete-linkage (already in `field/algebra.py`) resists single-linkage chaining,
at the cost of the strict-nesting-by-threshold property (a dendrogram restores it).

## 5. Proposed shape in rlat (design sketch — no code shipped)

The `.rlat` format is explicitly forward-compatible with new members
(`STORE.md`: new band slots "just appear"). One optional sidecar, built at
`rlat build`/`refresh` time from the band + registry, keyed by `passage_id` so
deltas survive refresh/sync renumbering:

```
graph/
├── edges.npz            bidirectional kNN + long-range links (uint32 pairs)
├── diagnostics.json     {chain_curvature, regime: "cloud"|"path"|"memoryless",
                          knn_k, built_from_band, build_params}
├── relations.json       k-planes prototypes: {plane (2 x d), angle, size,
                          exemplar edges as passage-coord pairs}   [opt-in]
└── hierarchy.json       nested concept levels with member passage_ids [opt-in]
```

RQL surface sketch: `navigate.walk(start, steps, law="auto")` (law chosen by
`diagnostics.regime`), `navigate.related(passage, relation=...)` (virtual typed
edges via prototypes), `navigate.concept(passage, level=...)`. Costs: build is
O(N·k) reusing the existing ANN; storage ≈ N·(k+2)·4 bytes (~40 KB per 1,000
passages) plus prototypes; every returned edge/concept carries passage receipts, so
the trust surface is unchanged.

Payoff hypothesis (the reason to build any of it): `deep-research` currently
multi-hops by re-querying flat retrieval; hop-by-graph (gates, concepts, relations)
should reduce hops and token cost for cross-document questions. That is the A/B that
justifies or kills the feature.

## 6. Pre-registered evaluation and falsification

1. **Regime**: measure chain curvature on ≥2 production bands (e.g. fabric-docs,
   python-stdlib). Prediction: still cloud-like (< 0) for documentation corpora;
   momentum ops ship only if some corpus measures > +0.1.
2. **Gates**: typed-gate next-passage prediction ≥ 1.5× ungated hit@5 on a
   production band (matches the 1.9× here), else the traversal surface ships
   metadata-gates-only with the numbers published.
3. **Relations**: k-planes enrichment ≥ 2× for a structural edge kind on a
   production band, else relation mining is shelved (virtual-edge machinery stays
   research).
4. **The payoff test**: `deep-research` A/B on the 63-question Fabric set —
   graph-assisted hops vs flat re-query, scored on the existing hallucination
   harness. If accuracy and token cost don't both improve, the graph remains an
   inspection tool (`rlat profile`-adjacent), not a retrieval path.

## 7. Honest limits

- Every number here is LSA-stand-in scale; the demo's `.rlat` mode exists precisely
  so the measurements can be repeated where the pinned encoder lives.
- Documents-as-clouds may be a property of *this* chunker + embedding; the
  diagnostic, not the conclusion, is the transferable artefact.
- The ontology extracted is *structural* (concepts as passage sets, relations as
  recurring displacements) — it is not a curated domain ontology and is not claimed
  to be one; naming and sanctioning stay human, via the existing curation loop.
- Graph state must survive `refresh`/`sync`; keying by `passage_id` is the design
  answer, but delta-maintenance of kNN edges (avoid full rebuild) is unsolved here.
- Two failed hypotheses are retained in the demo output by design: momentum
  traversal (killed by the cloud regime) and k-means relation mining (wrong tool
  for planes). The failures carried more design information than the successes.
