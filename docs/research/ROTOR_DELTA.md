# The rotor gate: free the plane, not just the angle

**Status:** an invented architecture component with machine-checked mechanism receipts —
not a validated breakthrough · **Date:** 2026-08-10 ·
**Receipts:** [`demos/demo4_rotor_gate.py`](demos/demo4_rotor_gate.py) (derivations),
[`demos/demo5_interior_vs_boundary.py`](demos/demo5_interior_vs_boundary.py) (training) —
all checks pass, including the ones that came out against the inventor's first claims.

This document was produced on the instruction *"invent it, don't look it up."*
Accordingly: everything below was derived and tested in this session; no literature
search was performed for this document (the prior-art section is from memory and says
so). It attacks open problems **9.1** (stable + expressively complete + chunk-parallel
transitions) and **9.5** (position as group action) of the companion report,
[`THE_NEXT_BREAKTHROUGH.md`](THE_NEXT_BREAKTHROUGH.md).

---

## 1. The gap, restated as one sentence

The 2024–26 architecture wave rediscovered that a recurrent layer's power is set by
the group its transition operators generate (companion report, §4) — but every shipped
family reaches the group elements that hard state tracking needs either **not at all**
(gates in [0,1]), **only on an abelian torus** (rotations in *fixed* planes:
Mamba-3's complex states, RoPE), **only at a parameter boundary** (DeltaNet's
reflection at β = 2, DeltaProduct's orthogonal corner), or **without a structural
stability guarantee** (free-spectrum diagonal-plus-rank-1). Mamba-3 made the rotation
*angles* data-dependent and kept the planes fixed — and fixed planes commute, which
is exactly the abelian/TC⁰ ceiling.

So: **free the plane.**

## 2. The invention

### 2.1 The rotor gate

Give the memory transition a data-dependent *bivector* — a plane *and* a magnitude —
and move on the rotation group by its exponential:

```
a_t = W_a x_t          b_t = W_b x_t                       (raw, unconstrained)
Z_t = a_t b_tᵀ − b_t a_tᵀ                                  (rank-2 skew)
R_t = exp(Z_t)                                             (the rotor gate)
```

with the closed form (no matrix exponential, no iteration, no normalisation — derived
from Z³ = −λ²Z, which holds for every rank-2 skew matrix):

```
λ   = √( ‖a‖²‖b‖² − (aᵀb)² )        — the angle IS the Gram area of (a, b)
R   = I + (sin λ / λ) Z + ((1 − cos λ) / λ²) Z²
```

`R` is **exactly orthogonal with det +1 for every raw value of (a, b)** — smooth
everywhere (series limits at λ → 0), periodic in λ, and it reaches *every* plane
rotation, including angle π. Verified in demo 4a against a reference matrix
exponential at mixed scales.

### 2.2 The layer (RotorDelta)

Compose with the shipped delta-rule machinery — decay chooses what to forget, the
rotor chooses how to *re-express* what is kept, the delta write stores new content:

```
S_t = γ_t · S_{t−1} R_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,      o_t = S_t q_t
```

- At `a, b → 0`: `R = I` exactly and the layer **is** Gated DeltaNet — a smooth
  degeneration, so it can be initialised as the incumbent and grow rotors only where
  the task demands them.
- Transition norm: `‖γ R (I − βkkᵀ)‖₂ ≤ γ ≤ 1` at every parameter value — stability
  is a *structural* product-of-contractions fact, not a clamped constraint (demo 4b
  exhibits a free-spectrum diagonal-plus-rank-1 setting whose 100-step product reaches
  norm ~10¹⁴; nothing of the kind is expressible here).
- Semantics: the state carries old keys in a frame that subsequent rotors keep
  re-indexing, so retrieval addresses the past *through the accumulated context
  transformation* — content-dependent relative positioning. With a **constant** rotor
  this is exactly RoPE (demo 4e, an equality, not an analogy); with data-dependent
  planes it is what RoPE could not be: non-abelian.

### 2.3 Why rank 2 is the natural next primitive

Skew-symmetric matrices have even rank, so the minimal nontrivial data-dependent
element of the Lie algebra is rank-2 — one plane. DeltaNet's `I − βkkᵀ` is the minimal
*contraction* step (rank-1 symmetric); the rotor is the minimal *isometry* step
(rank-2 skew, exponentiated). There is nothing in between.

## 3. What is proved, and by which check

| Claim | Where | Statement |
|---|---|---|
| Closed form correct | 4a | `R = expm(Z)` to 1e-14; orthogonal, det +1, at all raw parameters |
| Structural stability | 4b | isometry at every parameter; contrast family explodes; β<2 reflections decay as \|1−β\|^m |
| Single-step group reach | 4c | transposition = one rotor (θ=π); 3-cycle = one rotor (θ=2π/3) |
| Delta-family no-go | 4c | `I − βkkᵀ` orthogonal ⟺ β‖k‖² ∈ {0,2} ⟹ det ∈ {+1 (only I), −1}: **no single generalized-Householder step is a proper rotation** — 3-cycles are unreachable at any parameter, not merely at a boundary |
| Two micro-steps suffice for S₅ | 4c | every one of the 120 embedded elements is a product of ≤ 2 plane rotations (eigenvalue count) |
| Absorbs the incumbents | 4d, 4e | two reflections = one rotor (DeltaProduct(2)'s orthogonal corner); fixed planes commute (the Mamba-3/RoPE torus); constant rotor ≡ RoPE exactly |
| Escapes the torus | 4d | free-plane rotors do not commute — the non-abelian door TC⁰-capped families cannot open |
| Chunk-parallel algebra | 4f | `R = I + X N Xᵀ` (X = [a\|b], N 2×2); chunk product = scalar-cumprod × (I + rank-2C correction), assembled from n×2 matmuls; equality to the dense product at 1e-15 — the same WY-style shape Gated DeltaNet kernels exploit at rank 1 |
| Precision self-repair | 4g | fp32-rounded product of 5000 rotors drifts to ~2e-6 off orthogonality; two Newton–Schulz steps (the companion report's demo-3 map) restore 2e-16 — the target manifold is known, so drift is cheaply projectable |

The NC¹ statement then follows exactly as in the companion report (demo 2 +
Barrington): with transposition-generator tokens, one rotor step per token computes
the S₅ word problem with a linear readout, so the family's reachable function class
is NC¹-hard at fixed depth — under TC⁰ ≠ NC¹, strictly beyond softmax attention and
diagonal SSMs. The delta family reaches its NC¹ constructions only at the closed
boundary of its parameter space; the rotor family holds them in the interior — and,
per the no-go above, holds constructions (proper rotations) the delta family does not
contain at all.

## 4. What training actually shows — including two corrections to my own claims

Demo 5 was designed to prove "interior beats boundary." The runs said something more
precise, and two of my initial claims died on contact with the data. Both corpses are
kept on display, in this repository's tradition.

**Correction 1 — angle landscapes oscillate (A1).** Trained directly on length-16
parity, the rotor parks at θ ≈ 2.40 with a ~1e-16 gradient: a spurious local minimum.
Frequency-like parameters trade the delta family's boundary pathology for a
*multi-basin* pathology. A short length curriculum (T=2, whose landscape has a single
basin at π, then T=16) fixes it here; in the S₃ task it helps but does not eliminate
it (solve rate 1/5). This is a real cost of the family and the main open training
problem (§7).

**Correction 2 — no rate advantage on abelian tasks (A2).** I initially derived
gap ~ k⁻¹ for the sigmoid-boundary family; the measured exponent was 0.50, and
re-derivation shows why: the loss is *quadratic* in the boundary gap (residuals
≈ ±mg/2), so `dL/dw ∝ g²` through the saturating sigmoid, giving g ~ k^(−1/2). The
rotor's parity valley is *quartic* (cosine readout ⇒ residuals quadratic in the angle
error), also k^(−1/2). Measured: **0.50 and 0.50**. On a task both families can
approach, the interior chart buys no rate. The claims that survive are structural,
not kinetic.

**The phase/amplitude taxonomy (A2, and a genuinely useful by-product).** The two
families fail *differently in kind* at length. With angle error δ, rotor accuracy
oscillates with evaluation length — chance where the typical phase mδ hits π/2,
**anti-correlated (0.000)** at π, **perfect again (1.000)** where it aliases at 2π;
measured exactly at the three predicted lengths. The boundary family's gap is an
amplitude error: monotone decay to chance under a noise floor. Phase failures are
therefore *diagnosable* (periodic accuracy-vs-length curves) and *repairable*: both
families snap to exact parity post hoc (θ := π, β := 2) — on abelian tasks. The
asymmetry is that where the target is a proper rotation, the delta family has no snap
target to exist.

**The positive result (B).** On the S₃ word problem with generators {(01), (012)}
embedded in SO(4) — both proper rotations, so unreachable by any single-step delta
transition — the rotor layer trained end to end (backprop hand-derived through the
closed form; verified against finite differences at 2e-8) reaches **100% accuracy at
8× the training length** (1/5 seeds fully; delta baseline: 0/5, best 0.32 ≈ chance
1/6 + partial signal). And the solving run's transitions satisfy the S₃ presentation
relations **modulo a central sign**: it learned R₁ of order six with R₁³ central —
a *double cover*, a projective representation that the linear readout quotients out.
Gradient descent, given only sequence labels, discovered spin structure. (My first
relations check compared against one canonical embedding and failed on a perfect
solver — solutions are only defined up to conjugacy and central extension; the check
was wrong, not the network.)

## 5. Position–spectrum unification (problem 9.5), settled at mechanism level

One family now contains, as verified special cases: RoPE (constant bivector — demo
4e's exact equality), Mamba-3-style complex/rotational states (fixed planes,
data-dependent angles — the abelian torus of demo 4d), DeltaProduct's orthogonal
corner (two reflections = one rotor), and the new regime (data-dependent planes,
non-abelian). "Position" is the special case where the group action ignores content;
the rotor gate makes the action contextual. What remains open for 9.5 is choosing or
learning the *subgroup* per task — translation-like tori for positional structure,
permutation-generating sets for symbolic state — rather than leaving the whole of
SO(n) reachable.

## 6. Prior art, disclosed from memory

No search was run for this document (per the instruction). From memory, the nearest
neighbours and what they lack relative to this proposal: **RUM** (~2018) used a
data-dependent rotation inside a nonlinear RNN's vector update — the closest
mechanism I know — but not as the transition operator of a linear/matrix associative
memory, not chunk-parallelisable, and without the group-completeness / interior-reach
analysis. **expRNN / scoRNN / unitary-RNN** lines parameterise *weight* matrices on
the orthogonal group (input-independent — cannot do input-dependent group
composition, i.e. word problems). **RoPE** is the constant-rotor case;
**Mamba-3** the fixed-plane case; **DeltaNet / Gated DeltaNet / KDA / DeltaProduct /
RWKV-7** are the reflection-and-contraction lineage this composes with. **GATr**-style
geometric-algebra transformers use rotors as *data types* for equivariance, not as
memory transitions. **HRR** binding is commutative (abelian ceiling); **TPR** binding
is full-rank tensor product (no compression). To the best of my knowledge the specific
composition here — *data-dependent plane-and-angle bivector exponential, in closed
form, as the transition of a chunk-parallelisable gated delta memory, with the
completeness/no-go/quotient-relations analysis and the phase/amplitude failure
taxonomy* — is new. That belief must be checked against the literature before any
external novelty claim; this file's claims of *correctness* are machine-checked, its
claim of *novelty* is not.

## 7. Failure modes, open problems, and falsifiable predictions

What would kill it at scale, in descending expected likelihood:

1. **Trainability at depth.** Multi-basin angle landscapes may compound with depth
   and width; if curricula, multi-head angle-spread initialisation, and annealed
   λ-regularisation all fail to lift solve rates, the family is a construction, not
   an architecture. (Open problem: which objectives give non-degenerate valleys for
   angles? The quartic valley came from a cosine readout; CE with full readout in
   Part B behaved better.)
2. **The ceiling may not bind.** If frontier tasks rarely need non-abelian state
   tracking, the reachable-group advantage pays nothing (the companion report's §12
   already flags this debate).
3. **Kernel constants.** Rank-2 chunk corrections cost roughly 2× Gated DeltaNet's
   rank-1 machinery per step; if the throughput tax exceeds the eval gains, hybrids
   (a few rotor layers among GDN layers) are the only viable deployment. Not testable
   here — no GPU kernel was written.
4. **Numerics in fp16.** The fp32 simulation drifted 2e-6 per 5000 steps and repaired
   with two Newton–Schulz iterations; real fp16 kernels may drift faster than
   periodic repair tolerates.

Predictions someone with GPUs can score (all falsifiable; none verified here):

- **RD-1.** At matched parameters and steps, with a length curriculum, RotorDelta (or
  a hybrid replacing ¼ of GDN layers' erase with rotor gates) beats Gated DeltaNet on
  group word problems (S₃, S₅, A₅) and permutation-composition synthetic suites at
  ≥256 tokens, by a margin that grows with length.
- **RD-2.** Its length-generalisation failures, where they occur, are *periodic in
  evaluation length* (phase signature), unlike the monotone decay of gated-delta
  baselines — a distinguishing observable that requires no internals access.
- **RD-3.** Snapping trained rotors to the nearest exact finite-order element (a
  projection the delta family does not admit for proper rotations) converts
  near-solutions to exact length-invariant solutions on state-tracking tasks without
  degrading training loss.
- **RD-4.** On language modelling at ≤3B scale, adding rotor gates changes perplexity
  by <1% (most text needs little non-abelian tracking) while improving
  entity/state-tracking evals — i.e. the gains are localised, as the theory predicts.

## 8. Verdict

Claimed: a closed-form, unconditionally stable, chunk-parallelisable transition
family that holds the group elements hard state tracking needs in the interior of a
smooth parameter space; exact-construction completeness (with a two-line no-go for
the incumbent family); verified containment of RoPE, the Mamba-3 torus, and
DeltaProduct's corner; end-to-end learnability at toy scale with the discovered
double-cover phenomenon; and an honest account of what it costs — oscillatory
landscapes, no rate advantage on abelian tasks, unknown kernel constants.

Not claimed: novelty against the full literature (not searched, by instruction), or
any frontier-scale result. If the companion report's thesis is right — the layer is
an online learner, and the transition group is a design dial — then this is what
turning that dial to its natural next detent looks like when one person-session does
it honestly: mechanism proven, costs named, predictions on the record.
