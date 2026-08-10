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

## `demo4_rotor_gate.py`

```
4a. closed form == matrix exponential; exactly orthogonal for RAW a, b
  [PASS] R == expm(Z) (50 random raw a,b, mixed scales)   (deviation 1.04e-14)
  [PASS] R^T R == I always — no normalisation, no clamping, no boundary   (deviation 1.67e-15)
  [PASS] det R == +1 always (proper rotation)   (deviation 1.78e-15)
4b. stability: isometry at every parameter value vs boundary/unstable families
  [PASS] product of 400 random rotors: || . ||_F stays sqrt(n) (isometry)   (deviation 4.44e-15)
  [PASS] free-spectrum diag+rank-1 admits ||A^100||_F = 5.70e+14 (stability NOT structural; needs constraints/clamps)
      DeltaNet reflection at beta=1.80: stored signal after 512 swaps = 2.41e-50
      DeltaNet reflection at beta=1.98: stored signal after 512 swaps = 3.22e-05
      DeltaNet reflection at beta=2.00: stored signal after 512 swaps = 1.00e+00   <- exact only AT the boundary
4c. permutations as single rotors (the step DeltaNet needs beta=2 / cannot take)
  [PASS] transposition (i j) == rotor(plane span{e_i - e_j, e_aux}, angle pi)   (deviation 8.66e-17)
  [PASS] 3-cycle (0 1 2) == a SINGLE rotor at angle 2*pi/3   (deviation 1.11e-16)
  [PASS] ...whose det is +1 (so no single generalized-Householder step equals it)   (deviation 0.00e+00)
  [PASS] every one of the 120 S5 embeddings is a product of <= 2 plane rotations (max found: 2) — 2 rotor micro-steps per arbitrary token
4d. two reflections == one rotor; fixed planes are abelian, free planes are not
  [PASS] H(u) H(w) == rotor(span{u,w}, 2*angle(u,w))  [DeltaProduct(2) corner ⊂ rotors]   (deviation 2.78e-16)
  [PASS] same-plane rotors commute (the Mamba-3 / RoPE torus)   (deviation 1.67e-16)
  [PASS] free-plane rotors do NOT commute (escapes the abelian/TC0 ceiling of fixed planes)
4e. constant rotor == RoPE (state-side rotation == q/k rotation)
  [PASS] state-side constant rotor output == RoPE-rotated linear attention output   (deviation 4.00e-14)
4f. chunkwise closed form == sequential product (n x 2 matmuls only)
  [PASS] R == I + X N X^T with X = [a|b] (rank-2 factor form)   (deviation 4.44e-16)
  [PASS] chunk closed form (scalar cumprod x [I + rank-2C]) == dense product   (deviation 7.22e-16)
4g. fp32 drift over 5000 products, then Newton-Schulz self-repair
  [PASS] drift 1.96e-06 -> after 2 NS steps 2.22e-16

Summary: R(x) = exp(a(x)b(x)^T - b(x)a(x)^T) is a closed-form, exactly
orthogonal, everywhere-smooth transition at EVERY raw parameter value;
it reaches transpositions AND det=+1 elements (3-cycles) in ONE step,
absorbs DeltaProduct's orthogonal corner, RoPE, and the fixed-plane
torus as special cases; and it keeps the diag+rank-2 chunk algebra
that production kernels need. Demo 5 shows WHY the interior chart
matters: it is the difference between trainable and boundary-starved.

ALL PASS
```

## `demo5_interior_vs_boundary.py`

```
Part A. parity (abelian): curriculum, convergence rates, failure signatures
    A1 negative control (no curriculum): theta parks at 2.4024, |grad| = 7.7e-16 — spurious local minimum of an oscillatory angle landscape.
  [PASS] angle landscape trap is real (stuck far from pi with ~zero gradient)
    A2 curriculum finds the basin; convergence INTO it (10x budget: 3k -> 30k steps):
       rotor |theta - pi| : 3.83e-04 -> 1.21e-04   measured exponent 0.50  (quartic-valley derivation: 0.5)
       delta gap (2-beta) : 1.29e-03 -> 4.06e-04   measured exponent 0.50  (quadratic-loss-through-sigmoid derivation: 0.5)
       -> RATE TIE at k^(-1/2): on this abelian task the interior chart buys
          no optimisation-rate advantage. The advantage must be structural.
  [PASS] curriculum reaches the pi basin   (|theta-pi| = 3.8e-04)
  [PASS] rotor rate exponent ~ 0.5 (degenerate quartic valley, as derived)   (0.50)
  [PASS] delta rate exponent ~ 0.5 (quadratic loss through a saturating sigmoid, as derived)   (0.50)
    matched 3k-step budget, readout noise 0.02: PHASE vs AMPLITUDE failure
      length     | rotor   | delta   | gla     |
              16 |  1.0000 |  1.0000 |  0.5000 |  train
            1024 |  1.0000 |  1.0000 |  0.5000 |  
            8211 |  0.4997 |  0.5497 |  0.5000 |  quadrature
           16422 |  0.0000 |  0.5002 |  0.5000 |  anti-phase
           32843 |  1.0000 |  0.5000 |  0.5000 |  aliased
      -> rotor accuracy OSCILLATES with length (phase error: chance at quadrature,
         anti-correlated at half period, ~perfect again when the phase aliases);
         delta accuracy decays MONOTONICALLY to chance (amplitude error).
      snap-to-group repair: theta := pi and beta := 2 both give exact parity at
      every length — on ABELIAN tasks both families have a snap target;
      Part B is where the delta family has none.
  [PASS] phase signature: rotor ~chance at quadrature, <0.1 anti-phase, >0.9 aliased
  [PASS] amplitude signature: delta at chance at all three long lengths (monotone decay)
  [PASS] [0,1] gate is structural chance at long length

Part B. S3 word problem, generators {(01), (012)} in SO(4): representability trains
  [PASS] hand-derived rotor backprop matches finite differences   (max rel err 2.3e-08)
    curriculum T=4 then T=12, batch 48, Adam 0.05, 5 seeds each; chance = 1/6 = 0.167;
    'rel' = S3 relations modulo a central sign K = R1^3 (0 for a projective rep);
    'K-I' = distance of K from I (large + rel small => a DOUBLE COVER was learned):
      rotor seed 11:  acc@12 = 0.299   acc@96 = 0.168   rel = 2.820   K-I = 1.615
      rotor seed 22:  acc@12 = 1.000   acc@96 = 1.000   rel = 0.184   K-I = 2.828
      rotor seed 33:  acc@12 = 0.174   acc@96 = 0.152   rel = 3.249   K-I = 2.378
      rotor seed 44:  acc@12 = 0.342   acc@96 = 0.342   rel = 2.563   K-I = 1.069
      rotor seed 55:  acc@12 = 0.404   acc@96 = 0.164   rel = 3.232   K-I = 1.668
      delta seed 11:  acc@12 = 0.162   acc@96 = 0.154   rel = 1.029   K-I = 1.001
      delta seed 22:  acc@12 = 0.199   acc@96 = 0.168   rel = 1.693   K-I = 2.112
      delta seed 33:  acc@12 = 0.176   acc@96 = 0.168   rel = 1.000   K-I = 0.980
      delta seed 44:  acc@12 = 0.334   acc@96 = 0.316   rel = 1.438   K-I = 2.561
      delta seed 55:  acc@12 = 0.203   acc@96 = 0.145   rel = 0.883   K-I = 0.232
    rotor solve rate: 1/5 (multi-basin angle landscapes — the Part A
    trap in high dimension; curriculum helps but does not eliminate it).
  [PASS] rotor learns the S3 word problem end to end and holds it at 8x length   (1/5 seeds at >= 0.98 both lengths)
  [PASS] solving runs are (projective) S3 representations: relations-mod-center hold   (some are double covers: K far from I with rel ~ 0)
  [PASS] single-step delta baseline stays near chance (no proper rotation to snap to)   (best delta acc@96 = 0.316)

Summary (calibrated by the runs): the rotor gate's advantage is NOT an
optimisation-rate win — on abelian parity both families converge at
k^(-1/2) (A2), and the rotor's angle landscape has spurious minima that
need a curriculum (A1); its failures are phase-shaped, the boundary
family's amplitude-shaped. The advantage is structural: where hard state
tracking needs proper rotations, the delta family has nothing to reach OR
snap to, while the rotor family trains to exact group structure from
sequence labels alone — sometimes via a double cover (B).

ALL PASS
```

