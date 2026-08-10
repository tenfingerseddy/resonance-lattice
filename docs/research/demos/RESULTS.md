# Demo run record

Environment: CPython 3.11. Demos 1-5: stdlib only. Demos 6-8: numpy 2.4 (only
third-party package). Seeds fixed in-source. Demos 1-5 are bit-deterministic
across platforms (IEEE-754 + stdlib); demos 6-8 are deterministic per numpy/
BLAS build (tiny float variations possible across platforms, conclusions stable).

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

## `demo6_rotor_intent_ops.py`

```
6a. exactness: unit norm, endpoint, geodesic identity, invertibility
  [PASS] output is unit-norm at every t (scores stay calibrated)   (max dev 4.4e-16)
  [PASS] t=1 lands exactly on the anchor   (max dev 8.0e-16)
  [PASS] toward == SLERP (the sphere's geodesic — not an approximation of it)   (max dev 5.8e-16)
  [PASS] anti undoes toward exactly (auditable, reversible ops)   (max dev 6.2e-16)
6b. resolution: global lens rotor is an isometry; per-query pulls are not
  [PASS] lens rotor preserves ALL pairwise query similarities (exact isometry)   (max dev 2.0e-15)
    mean pairwise query cosine: raw -0.000 | toward(0.5) 0.501 (anchor-align 0.71) | additive(0.8) 0.391 (align 0.63)
    -> any per-query pull trades resolution for alignment (both do);
       only the GLOBAL lens rotor conditions retrieval at zero resolution cost.
  [PASS] per-query pulls collapse resolution; global rotor does not
    additive pre-renormalisation norm can reach 0.988 (degenerate direction noise near anchors); rotor never leaves the sphere.
6c. composition: exact products, near-commuting at small angles (BCH law)
  [PASS] halving both angles shrinks the order effect ~4x (first-order BCH, measured)   (ratio 4.01)
    -> multi-intent composition is well-behaved: order matters in principle
       (non-abelian), negligibly at small strengths, exactly invertibly always.
6d. disambiguation on real text (repo docs, LSA stand-in — mechanism scale)
    corpus: repo-docs LSA (1071 passages, 256d)
    250 ambiguous (internal-docs + benchmarks) midpoint queries;
    primary: group-A precision@10 (intent focus); secondary: median rank of
    the specific A-parent passage (specificity cost of aiming at a centroid):
      method                    | A-prec@10 | median parent rank
      raw query                  |    51.8   |    1.0
      rotor->centroid t=0.3 (X)  |    44.3   |    1.0
      rotor->centroid t=0.6 (X)  |    43.2   |   32.0
      rotor->contrast t=0.15     |    63.5   |    1.0
      rotor->contrast t=0.3      |    73.8   |    1.0
      rotor->contrast t=0.6      |    84.8   |    1.0
      additive contrast +0.25    |    64.0   |    1.0
      additive contrast +0.5     |    73.7   |    1.0
      trust x1.25 (needs glob)   |    88.5   |    1.0
  [PASS] aiming at a broad centroid is ANTI-discriminative (documented anti-pattern)   (51.8 -> 44.3)
  [PASS] aiming at the CONTRAST lifts group precision >= 15 points (toward+anti jointly)   (51.8 -> 84.8)
  [PASS] the anti-pattern also costs specificity (centroid t=0.6 parent rank collapses) while the contrast keeps the parent at rank ~1 even at t=0.6
    rotor vs additive on the contrast: 84.8 vs 73.7 A-prec@10 — rotor ahead.
    trust weights win tie-breaking outright WHEN the intent is expressible as
    a source glob; rotors condition toward regions no glob can name. The two
    are complementary dials (amplitude vs phase), not competitors.
    (Pre-registered in ROTOR_LENS.md: if additive matches rotor on the real
     encoder bench, the rotor claim shrinks to exactness/invertibility/audit.)

Summary: toward/anti/compose exist in closed form on the query side of
`band @ q` — unit-norm always, geodesic-exact, invertible, composable, and
as a GLOBAL lens rotor, resolution-lossless (an isometry). Per-query pulls
(rotor or additive) buy alignment with resolution; the lens form does not.

ALL PASS
```

## `demo7_latent_graph.py`

```
corpus: repo-docs LSA (1075 passages, 256d)
7a. navigability: kNN graph (k=8), long-range links, search discipline
    connectivity: largest component 1075/1075 passages (100.0%)
    greedy on DIRECTED kNN edges  :  72.7% reach target
    greedy on BIDIRECTIONAL edges : 100.0% reach target
    best-first with backtracking  :  99.7% reach target, touching a mean of 5/1075 nodes (1% of corpus)
    -> similarity says who your neighbours are; it does not say who counts
       YOU as a neighbour. Symmetrising the links is the load-bearing step
       (HNSW does exactly this); backtracking then buys efficiency. Measured
       in the open here — production search stays FAISS (field/ann.py).
  [PASS] kNN graph is essentially one connected component (>= 95%)
  [PASS] edge symmetry is load-bearing: directed greedy fails often, undirected doesn't   (72.7% -> 100.0%)
  [PASS] best-first navigation is reliable (>= 99%) while touching <= 15% of corpus   (99.7%, 1%)
7b. the regime diagnostic: are document chains PATHS or CLOUDS?
    976 chain steps in 49 files; median consecutive-passage angle 61 deg
    chain curvature mean cos(d_k, d_k+1) = -0.473 +- 0.005
    (i.i.d.-cloud signature: -0.500; persistent path: > 0)
  [PASS] regime detected: documents are clouds, not paths (curvature ~ -1/2)   (-0.473)
    next-passage prediction (find k+1 among all passages):
      method                           | hit@1 | hit@5 |  MRR
      greedy: similarity to current   |   8.5 |  24.6 | 0.163
      momentum t=1.0 (path tool)      |   3.7 |  12.9 | 0.085
      momentum t=0.35 (path tool)     |   6.5 |  21.1 | 0.138
      midpoint of last two            |   7.9 |  25.6 | 0.165
      doc centroid, ungated (cloud)   |   4.1 |  16.7 | 0.104
      typed gate (same doc) + greedy  |  14.9 |  46.1 | 0.300
      typed gate + doc centroid       |   5.8 |  23.8 | 0.163
  [PASS] momentum harm is monotone in strength (as the cloud regime demands)   (12.9 < 21.1 < 24.6)
  [PASS] cloud tool alone cannot rank within the cloud (centroid far below greedy)   (16.7 vs 24.6)
  [PASS] typed metadata edge + local similarity is where traversal wins (>= 1.5x greedy)   (greedy 24.6 -> gated 46.1)
  [PASS] the local signal is real beyond membership (gated greedy > gated centroid)   (46.1 vs 23.8)
    -> the original hypothesis (momentum beats greedy) FAILED on this band;
       the curvature diagnostic explains why (clouds, not paths) and would
       flip the recommendation wherever it measures > 0. What actually makes
       vectors 'easier to traverse' here: TYPED edges from metadata the .rlat
       already stores (same-document, adjacency), gating a local similarity
       ranking. Run against a production .rlat, this demo re-measures the
       regime and re-scores every method.
7c. relation types from raw pairs: k-planes clustering of displacements
    identifiability boundary: purity 84.3% on fully-random concepts (weak in-plane mass) vs 98.8% on feature-bearing concepts
  [PASS] k-planes separates planted relations on feature-bearing entities (>= 95%)   (purity 98.8%; selected by unsupervised objective over 10 restarts)
    plane recovery error (last relation) 0.127; recovered rotors match the
    true relation on 200/200 unseen nodes — typed edges generated, not stored
  [PASS] recovered relations APPLY correctly to unseen nodes (>= 90%)   (200/200)
    real corpus: 4300 kNN edges; adjacent-in-document base rate 10.1%; max k-planes cluster enrichment 1.67x
  [PASS] displacement geometry carries SOME unseen structural signal (>= 1.25x)   (1.67x — weak at LSA quality; production-band test pre-registered)
7d. hierarchy for free: single-linkage components across a threshold sweep
    threshold 0.780: 901 concepts (largest 10, singletons 801)
    threshold 0.700: 607 concepts (largest 81, singletons 497)
    threshold 0.645: 373 concepts (largest 570, singletons 325)
  [PASS] levels nest exactly (every fine concept sits inside one coarser concept)
    receipt — one fine concept and its passages:
      docs/internal/benchmarks/02_fabric_failure_analysis.md:10883+423
      docs/internal/benchmarks/02_fabric_failure_analysis.md:13255+387
      docs/internal/benchmarks/02_fabric_failure_analysis.md:13644+508
    -> broader/narrower (SKOS-style) levels from one threshold sweep; every
       concept is a set of passages with exact source receipts. (Production
       variant: complete-linkage, already in field/algebra.py, resists the
       single-linkage chaining visible in the largest coarse concept.)

Summary: yes — a traversable graph is latent in the band, but each layer
needs its OWN mathematics, and two hypotheses died honestly on the way:
connections alone don't give traversal (backtracking does, 7a); momentum
doesn't help where documents are clouds rather than paths — a one-number
curvature diagnostic selects the right traversal law per band (7b);
relations are 2-D planes in displacement space, recoverable by k-planes
(not k-means) and applicable as virtual typed edges (7c); and nested
concept levels with receipts come from a threshold sweep (7d).

ALL PASS
```

## `demo8_orbit_retrieval.py`

```
corpus: repo-docs LSA (1082 passages, 256d)
    displacement fields: 458 fit / 574 held-out reading-adjacency pairs; 2273 cross-file kNN pairs
8a. the quotient proposition: group-averaging == removing the mined planes
  [PASS] averaging over the cyclic group C_5 of a rotor == exact plane removal
  [PASS] ...and the result is an orthogonal projection (idempotent)
    -> group-invariant retrieval is flat retrieval in a projected band:
       the 'graph' collapses into the coordinates. Zero query-time cost.
8b. eval A — held-out reading-continuation (the hard task from demo 7)
    mined 4 adjacency rotors (angles [0.02, -0.04, 0.01, 0.11], support [155, 137, 79, 87])
    flat     : hit@5  27.0  MRR 0.173
    orbit-max: hit@5  27.0  MRR 0.173   (9 matvecs/query)
8c. eval B — cross-register retrieval (DESIGN -> VERDICT of the same bench)
    4 bench dirs with both DESIGN* and VERDICT* files
    mined 4 cross-file rotors (angles [-0.07, 0.01, -0.03, 0.01], support [478, 367, 874, 554])
    mode selector: median |theta| = 0.02 rad — near-zero angles mean
    the mined planes carry SYMMETRIC spread (nuisance axes), not directed
    moves: the group is ~self-inverse, so its quotient, not its orbit, is
    the right functor on this band.
    flat       : verdict-recall@10  75.9  MRR 0.317
    orbit      : verdict-recall@10  79.6  MRR 0.319
    quotient   : verdict-recall@10  87.0  MRR 0.321
    pca-removal: verdict-recall@10  83.3  MRR 0.335
    random-quot: verdict-recall@10  77.8  MRR 0.320
8d. regression guard: topical precision must not degrade
    same-top-level-dir precision@10: flat 84.0 | orbit 83.7 | quotient 87.5 | pca-removal 86.1 | random-quot 83.8

Measured verdicts (set after running, stated as findings, not hypotheses):
  [PASS] quotient proposition holds exactly (the graph can live in the coordinates)
  [PASS] orbit-max does NOT rescue reading-continuation (cloud regime, as demo 7 said)   (flat 27.0 vs orbit 27.0)
  [PASS] cross-register: at least one edge-free mode beats flat recall@10 by >= 5 points   (flat 75.9 -> orbit 79.6 / quotient 87.0)
  [PASS] targeting matters: removing random directions does not reproduce the gain   (random-quot 77.8 vs flat 75.9)
    displacement-mined quotient vs global-PCA removal ('all-but-the-top'): 87.0 vs 83.3 — the mining is load-bearing
  [PASS] guard: the winning mode costs <= 5 points of topical precision

Summary: no edges were stored and no query surface was added. The corpus's
recurring transformations, mined once, either transport the query (orbit-max,
2K+1 matvecs, per-hit receipts) or vanish into the coordinates (quotient
band, zero overhead). Where demo 7's diagnostic says there is no directional
signal (reading order), orbit honestly buys nothing — the win to check on a
production band is cross-register recall, pre-registered in
ORBIT_RETRIEVAL.md with kill criteria.

ALL PASS
```

