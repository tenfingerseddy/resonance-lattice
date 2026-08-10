# Demo run record

Environment: CPython 3.11, no third-party packages. Seeds fixed in-source; IEEE-754
doubles make these outputs deterministic across platforms.

## `demo1_layers_are_online_learners.py`

```
1a. softmax attention == Nadaraya-Watson kernel regression
  [PASS] attention output == kernel-regression estimate   (max deviation 1.11e-16)
  [PASS] attention weights are convex (>=0, sum 1); output stays in conv(values)   (max deviation 0.00e+00)
1b. linear attention == Hebbian fast-weight program (one SGD step/token)
  [PASS] sum_i <q,k_i> v_i  ==  (sum_i v_i k_i^T) q   (max deviation 1.07e-14)
1c. DeltaNet recurrence == online SGD on 1/2||S k - v||^2
  [PASS] DeltaNet state == SGD-on-memory-loss state   (max deviation 4.44e-16)
    gated variant == multiplicative decay (weight decay) + delta step
  [PASS] Gated DeltaNet state == decay-then-SGD state   (max deviation 3.33e-16)
1d. crosstalk: Hebbian vs delta-rule storage, N=12 pairs in d=16
    orthonormal keys : Hebbian   0.0000 | delta x1 pass   0.0000 | delta x25 passes 1.61e-16
  [PASS] Hebbian storage exact when keys orthonormal   (max deviation 4.96e-16)
    correlated keys  : Hebbian   0.7931 | delta x1 pass   0.5595 | delta x25 passes 2.00e-02
  [PASS] delta rule (Kaczmarz) beats Hebbian by >10x under correlation

Summary: softmax attention = nonparametric kernel regression (keeps everything,
pays O(T) per token); linear attention = Hebbian memory (fast, but crosstalks);
DeltaNet/Gated DeltaNet = online least squares with forgetting (Widrow-Hoff 1960 +
Kaczmarz 1937), whose transition operators are Householder maps (1958).
The architecture question 'what should attention be?' has become the
statistics question 'which regression estimator should run in the forward pass?'

ALL PASS
```

## `demo2_transition_algebra.py`

```
2a. parity: eigenvalue -1 is the whole trick
  [PASS] a(x) in {+1,-1} computes parity exactly at every length
  [PASS] [0,1]-gate products never flip sign, never grow (no oscillation available)
2b. mod-3 counting via a unit-circle eigenvalue (rotation)
  [PASS] spectrum exp(2*pi*i/3) counts mod 3 exactly at every length
2c. the algebra: diagonal transitions commute, Householders don't
  [PASS] diag: A B == B A (abelian)
  [PASS] Householder: A B != B A (non-abelian)
    diagonal SSM == prefix-product closed form (the flattening that caps it at TC0)
  [PASS] recurrence == sum of per-token terms weighted by gate products
2d. streaming DeltaNet steps compute the S5 word problem exactly
  [PASS] every S5 element == product of its transposition micro-steps (all 120 checked)
  [PASS] 60 random streams (length <= 300): state == exact S5 group product

Summary: what a recurrent layer can track is the GROUP its transitions
generate. Nonneg diagonal gates -> abelian, TC0, no parity at any scale.
Eigenvalue -1 -> parity. Unit-circle spectrum -> modular counting (Mamba-3,
RoPE - same move). Householder products -> all of S5, i.e. NC1-complete
state tracking (Barrington), strictly beyond transformers/diagonal SSMs
unless TC0 == NC1. Expressivity is now a DESIGN DIAL, set by the spectrum
and commutativity of A(x) - not an emergent mystery.

ALL PASS
```

## `demo3_one_dual_map_two_timescales.py`

```
1. cubic Newton-Schulz iteration -> orthogonal matrix
  [PASS] O^T O == I   (deviation 2.22e-16)
2. O is the polar factor of G
  [PASS] O^T G symmetric   (deviation 1.33e-15)
  [PASS] O^T G positive semidefinite (min eigenvalue >= 0)   (deviation 0.00e+00)
3. O maximizes <G, A> over all orthogonal A (dual of the spectral norm)
  [PASS] <G, O> == nuclear norm of G (independent Jacobi computation)   (deviation 1.78e-15)
  [PASS] beats 500 random orthogonal contenders (best contender 7.2597 vs polar 14.1406)

Summary: 'orthogonalize the update' is not a heuristic — it is steepest
descent once you measure steps in the spectral norm (whose dual is the
nuclear norm). The identical map now trains trillion-parameter weights
(Muon/K2, outer loop) and writes test-time memories (ATLAS, inner loop).
Architecture and optimizer have become the same mathematical object at
two timescales.

ALL PASS
```

