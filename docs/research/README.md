# Research notes

Standalone research documents. Nothing in this directory is part of the `rlat`
package, its public API, or its test suite.

## Provenance

Produced 2026-08-10 by a Claude Code research session, in response to the repository
task: *"The next breakthrough in transformer architecture will be mathematical. Find
it."* The write-up follows this repository's receipts-before-claims convention: every
load-bearing identity or construction in the report is machine-checked by a script in
`demos/`, and the claims that cannot be checked here (anything requiring frontier-scale
training) are labelled as such, with pre-registered predictions so the call can be
scored later.

## Contents

- [`THE_NEXT_BREAKTHROUGH.md`](THE_NEXT_BREAKTHROUGH.md) — the report. Verdict up
  front; four mathematical pillars; production evidence as of mid-2026; five open
  problems; scoreable predictions (2027-12-31 / 2028-12-31); honest limits.
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
  - `linalg.py` — the ~120-line dense linear algebra the demos share.
  - `RESULTS.md` — captured output of all three demos (all checks pass).

## Running the demos

```bash
cd docs/research/demos
python3 demo1_layers_are_online_learners.py
python3 demo2_transition_algebra.py
python3 demo3_one_dual_map_two_timescales.py
```

Each script exits 0 and prints `ALL PASS` when every check holds; deviations shown
are at float precision (~1e-9 or better).
