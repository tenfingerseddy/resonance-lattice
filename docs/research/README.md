# Research notes

Standalone research documents. Nothing in this directory is part of the `rlat`
package, its public API, or its test suite.

## Provenance

Produced 2026-08-10 by a Claude Code research session, in three parts. First, in
response to the repository task *"The next breakthrough in transformer architecture
will be mathematical. Find it."* — a research synthesis with runnable receipts.
Second, in response to the follow-up instruction *"invent it, don't look it up"* —
an original architecture component (the rotor gate) derived and machine-checked in
the same session, with no literature search performed for that part. Third, in
response to *"how can it apply to resonance lattice?"* and *"can we create a graph
from vectors?"* — two application investigations (rotor intent operators for
retrieval; the latent graph/ontology) run against real text from this repository's
own documentation, with a loader for production `.rlat` bands so every measurement
can be repeated where the pinned encoder lives.

Everything follows this repository's receipts-before-claims convention: every
load-bearing identity or construction is machine-checked by a script in `demos/`;
claims that cannot be checked here (anything requiring frontier-scale training) are
labelled as such, with pre-registered predictions so the calls can be scored later;
and negative results from the runs are published alongside the positive ones.

## Contents

- [`THE_NEXT_BREAKTHROUGH.md`](THE_NEXT_BREAKTHROUGH.md) — the synthesis report.
  Verdict up front; four mathematical pillars; production evidence as of mid-2026;
  five open problems; scoreable predictions (2027-12-31 / 2028-12-31); honest limits.
- [`ROTOR_DELTA.md`](ROTOR_DELTA.md) — the invention: a data-dependent plane-rotation
  ("rotor") transition family in closed form — exactly orthogonal at every raw
  parameter value, NC¹-reaching in the interior of its parameter space,
  chunk-parallelisable, containing RoPE / the Mamba-3 torus / DeltaProduct's corner
  as special cases — plus what training actually showed, including two corrections
  to the inventor's own claims.
- [`ROTOR_LENS.md`](ROTOR_LENS.md) — the invention applied to rlat retrieval: the
  Intent Lattice's deferred operators (`--toward`, `anti`, composition) as rotor
  operations on the query side of `band @ q`; the contrast-not-centroid design
  finding; the lens as a phase dial complementing trust weights' amplitude dial;
  serialisation sketch and a pre-registered production bench with falsification
  criteria.
- [`LATENT_GRAPH.md`](LATENT_GRAPH.md) — the graph/ontology investigation: edges,
  traversal, relation types, and hierarchy as geometric objects latent in the band;
  the chain-curvature regime diagnostic (documents here are clouds, not paths —
  the momentum hypothesis failed and is kept on display); typed metadata gates as
  the traversal win; k-planes relation mining with an identifiability boundary;
  nested concept levels with receipts; a `graph/` sidecar design sketch and
  pre-registered production evaluations.
- [`demos/`](demos/) — runnable receipts, Python 3.11+ stdlib only (no numpy, no
  torch), deterministic (fixed seeds, exact identities):
  - `demo1_layers_are_online_learners.py` — softmax attention ≡ kernel regression;
    linear attention ≡ Hebbian fast weights; DeltaNet ≡ online SGD on a memory loss;
    gating ≡ weight decay; crosstalk measurements (why the delta rule won).
  - `demo2_transition_algebra.py` — expressivity as a design dial: parity needs
    eigenvalue −1; modular counting needs the unit circle; diagonal transitions
    commute (the TC⁰ flattening); streamed DeltaNet steps solve the S₅ word problem
    exactly.
  - `demo3_one_dual_map_two_timescales.py` — Newton–Schulz → polar factor; the
    spectral-norm steepest-descent map behind both Muon (outer loop) and ATLAS's
    test-time memory (inner loop).
  - `demo4_rotor_gate.py` — the invention's derivation receipts: closed form ==
    matrix exponential; structural stability; single-step transpositions and
    3-cycles (with the delta-family det no-go); containments; chunk algebra;
    fp32 drift + Newton–Schulz self-repair.
  - `demo5_interior_vs_boundary.py` — what training shows: the angle-landscape
    trap (negative control), measured k^(−1/2) rate tie on parity, phase-vs-
    amplitude failure signatures (including aliasing back to perfect accuracy),
    end-to-end S₃ learning with a discovered double cover, and the delta
    baseline's representability ceiling.
  - `demo6_rotor_intent_ops.py` — rotor intent operators on real text: exactness
    receipts (SLERP identity, exact inverse, isometric lens, BCH composition law)
    and the contrast-vs-centroid retrieval experiment. **Requires numpy.**
  - `demo7_latent_graph.py` — the latent graph: edge symmetry and best-first
    navigation, the chain-curvature regime diagnostic, typed-gate traversal,
    k-planes relation recovery with virtual typed edges, nested concept
    hierarchy. Accepts an optional `.rlat` path to re-run every measurement on a
    production band. **Requires numpy.**
  - `corpus.py` — shared corpus utilities for demos 6-7: chunks this repository's
    markdown into passages in reading order (mirroring `passages.jsonl`
    semantics), embeds with hashed-ngram TF-IDF + LSA (a stand-in, not the
    production encoder — stated caveat), and loads real `.rlat` archives
    (`bands/base.npz` + `passages.jsonl`).
  - `linalg.py` — the ~140-line dense linear algebra demos 1-5 share.
  - `RESULTS.md` — captured output of all seven demos (all checks pass).

  Demos 1-5 are stdlib-only. Demos 6-7 need `pip install numpy` (corpus-scale
  linear algebra); they still use no other dependency and no network.

## Running the demos

```bash
cd docs/research/demos
python3 demo1_layers_are_online_learners.py
python3 demo2_transition_algebra.py
python3 demo3_one_dual_map_two_timescales.py
python3 demo4_rotor_gate.py
python3 demo5_interior_vs_boundary.py
pip install numpy                      # demos 6-7 only
python3 demo6_rotor_intent_ops.py
python3 demo7_latent_graph.py          # optionally: demo7 ... /path/to/model.rlat
```

Each script exits 0 and prints `ALL PASS` when every check holds. Demo 5 trains
small models in pure Python and takes ~30 seconds; the rest run in seconds.
