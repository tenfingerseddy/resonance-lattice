# Research notes

Standalone research documents. Nothing in this directory is part of the `rlat`
package, its public API, or its test suite.

## Provenance

Produced 2026-08-10 by a Claude Code research session, in two parts. First, in
response to the repository task *"The next breakthrough in transformer architecture
will be mathematical. Find it."* — a research synthesis with runnable receipts.
Second, in response to the follow-up instruction *"invent it, don't look it up"* —
an original architecture component (the rotor gate) derived and machine-checked in
the same session, with no literature search performed for that part.

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
  - `linalg.py` — the ~140-line dense linear algebra the demos share.
  - `RESULTS.md` — captured output of all five demos (all checks pass).

## Running the demos

```bash
cd docs/research/demos
python3 demo1_layers_are_online_learners.py
python3 demo2_transition_algebra.py
python3 demo3_one_dual_map_two_timescales.py
python3 demo4_rotor_gate.py
python3 demo5_interior_vs_boundary.py
```

Each script exits 0 and prints `ALL PASS` when every check holds. Demo 5 trains
small models in pure Python and takes ~30 seconds; the rest run in seconds.
