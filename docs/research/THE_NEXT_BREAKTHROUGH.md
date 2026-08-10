# The next breakthrough in transformer architecture will be mathematical — finding it

**Status:** research synthesis with runnable receipts · **Date:** 2026-08-10 ·
**Demos:** [`demos/`](demos/) (Python stdlib only, deterministic, all checks pass — see [`demos/RESULTS.md`](demos/RESULTS.md))

---

## The verdict

A breakthrough cannot be conjured on demand, but it can be *located*: find where the
mathematics is already cashing out at frontier scale, characterise the mechanism
precisely, and state falsifiable predictions about how it completes. That is what this
document does. The location is unambiguous:

> **The sequence-mixing layer is being re-derived as an online learner.** The forward
> pass of the next generation of "transformers" runs an explicit optimisation
> algorithm over an associative memory, one update per token. Attention, linear
> attention, state-space models, DeltaNet, and the test-time-training family are all
> special cases — points in a design space whose axes are now mathematically
> characterised: **(i)** the inner *objective*, **(ii)** the inner *optimiser and its
> geometry*, **(iii)** the *algebra of the transition operators* (which provably fixes
> what the layer can compute), and **(iv)** the *memory topology* (how state size
> scales with context), all under one hard constraint — the update must factor into
> large matrix multiplications.

This is not a forecast of an idea that might appear. Every pillar already exists in
theory *and* ships in production as of mid-2026 (§8): Qwen3.5 interleaves Gated
DeltaNet with full attention 3:1; Moonshot's Kimi lineage runs a channelwise-gated
delta rule in most layers; Mamba-3 moved its spectrum onto the complex unit circle
specifically because the algebra says real non-negative gates cannot count; Google's
Titans/ATLAS line puts momentum and second-order steps *inside* the forward pass; and
the same duality map that writes ATLAS's test-time memories trained a trillion-parameter
model as Muon. What remains open — the part that will be recognised in hindsight as
"the breakthrough" — is a short list of precisely stated mathematical problems (§9).
Solving any two of them likely produces the next step-change in capability per FLOP;
§10 pre-registers predictions so this call can be scored.

The one-line summary: **the architecture question "what should attention be?" has
become the mathematical question "which online learning algorithm should run in the
forward pass — with which objective, in which geometry, over which group, at which
resolution?"** That reframing, plus the theorems that make each axis a design dial, is
the breakthrough.

---

## 1. What counts as a breakthrough — and why "mathematical" is the safe bet

Define an architecture breakthrough operationally: *a change to the sequence-mixing or
parameter-organisation of the network that produces a capability-per-unit-compute jump
at frontier scale and is adopted by at least two independent frontier labs within
~24 months.* By that definition the transformer era has had perhaps seven, and each
was a piece of mathematics before it was a module:

| Breakthrough | The mathematics it is |
|---|---|
| Attention (2017) | Nadaraya–Watson kernel regression (1964) evaluated in an embedding space (verified exactly in [demo 1](demos/demo1_layers_are_online_learners.py)) |
| Mixture-of-Experts at scale | conditional computation: block-sparse structure decouples parameters from FLOPs |
| RoPE (2021) | a group representation: positions act as rotations, so relative offset becomes an inner-product invariant |
| FlashAttention (2022) | an algebraic identity — associativity of streaming softmax renormalisation — not an approximation |
| μP (2022) | the infinite-width scaling limit as a design constraint: one parameterisation transfers hyperparameters across scale |
| Multi-head Latent Attention (2024) | low-rank joint factorisation of the KV map, moved inside the architecture |
| Muon (2024–25) | steepest descent in the *right norm*: dualising the gradient under the spectral norm ([demo 3](demos/demo3_one_dual_map_two_timescales.py)) |

"The next breakthrough will be mathematical" is therefore not a bold premise — it is
base rates. The task is to say *which* mathematics. The distinctive fact about
2024–2026 is that four previously separate mathematical threads have converged on one
object, and frontier labs have started shipping it before the theory is finished. That
convergence-ahead-of-completion is the classic signature of a breakthrough in
progress.

## 2. Method and epistemic status

What was done: a synthesis of the primary literature to January 2026, verified and
updated against the public record to August 2026 (release notes, tech reports, ICLR/
NeurIPS 2025–26 proceedings); the load-bearing identities and separations were then
re-derived and machine-checked in three dependency-free Python scripts under
[`demos/`](demos/), in this repository's receipts-before-claims style.

What this is **not**: a frontier-scale experiment. Nothing here trains a large model.
The demos prove *mechanism* (exact identities, exact constructions), not *scaling*;
where scaling evidence is cited it is other people's published work, and where a claim
rests on a post-January-2026 report I could not independently verify, it is marked
*as reported*. §12 lists the ways this call could be wrong. The predictions in §10 are
written so that being wrong will be visible.

## 3. The finding: one design space behind every serious sequence layer

Three independent research programmes arrived at the same unification within roughly a
year of each other:

- **State-space duality** (Mamba-2, 2024): attention-style and recurrent-style
  computation are the same semiseparable matrix algebra read in two orders [8].
- **Test-time regression** (Stanford, 2025): every effective sequence layer is an
  associative memory fit by regression at inference time; layers differ only in
  regressor class, weighting, and optimisation algorithm [13].
- **MIRAS** (Google, 2025): sequence models are instances of online optimisation over
  a memory, specified by objective ("attentional bias"), retention (regulariser), and
  update algorithm [15].

Written as one table — each row is a shipped architecture, each column a mathematical
choice (demo 1 verifies the middle columns exactly; demo 2 the fourth):

| Layer | Estimator / inner objective | Inner optimiser step | Transition operator A(x) | State size |
|---|---|---|---|---|
| Softmax attention | nonparametric kernel regression | none — appends to the sample set | n/a | O(T) KV cache |
| Linear attention (2020) [5] | linear memory, correlation loss | Hebbian (fast weights, 1992 [6]) | I | O(1) |
| RetNet / GLA / Mamba-2 [7][8] | + exponential forgetting | Hebbian + decay | diagonal, entries in [0,1] | O(1) |
| DeltaNet (2024) [9] | least squares ½‖Sk−v‖² | delta rule = Widrow–Hoff (1960); cycled, it is Kaczmarz (1937) | I − βkkᵀ — a generalised Householder (1958) | O(1) |
| Gated DeltaNet (2025) [10] | + weight decay | decay, then delta step | α(I − βkkᵀ) | O(1) |
| Kimi Delta Attention (2025) [24] | + channelwise decay | diag decay, then delta step | diag(α)(I − βkkᵀ) | O(1) |
| RWKV-7 (2025) [11] | generalised delta rule | decoupled erase/write | diag(w) − rank-1, spectrum may leave [0,1] | O(1) |
| DeltaProduct (2025) [12] | least squares, several steps per token | n_h delta micro-steps | product of n_h Householders | O(1) |
| TTT (2024) [14] | self-supervised inner loss | SGD on an MLP's weights | nonlinear | O(1) |
| Titans (2025) [16] | ℓ₂ "surprise" + forgetting | SGD + momentum + decay | (deep memory) | O(1) |
| ATLAS (2025) [17] | sliding-window ℓ₂ ("Omega rule"), polynomial features | **Muon-style orthogonalised step** | (deep memory) | O(1) |
| Mamba-3 (2026) [18] | SSD algebra, better integrator | trapezoidal discretisation | diagonal **complex**, unit circle | O(1) |
| Log-linear attention (2026) [19] | the above, at every scale | per-scale delta/gated updates | Fenwick-tree hierarchy | **O(log T)** |
| NSA / MoBA / DSA sparse attention (2025) [20][21] | kernel regression on a selected support | none | n/a | O(T), pruned reads |

Two things make this table a *finding* rather than a taxonomy:

1. **The columns are load-bearing.** Each is governed by its own body of theorems
   (§§4–7), so moving along a column has predictable consequences. That is new.
   Architecture search used to be empirical alchemy; these axes have proofs.
2. **The rows at the bottom are the frontier's actual roadmap.** The historical
   sequence Hebbian → +decay → delta → +gating → multi-step → momentum →
   preconditioned is *precisely the history of optimisation algorithms, replayed
   inside the forward pass at a 60-year lag*: Hebb (1949) → leaky integration →
   Widrow–Hoff/Kaczmarz (1937–60) → weight decay → block methods → momentum →
   second-order/orthogonalised (2024–25, already inside ATLAS). The continuation of
   this sequence is not a mystery; it is the remaining chapters of the optimisation
   textbook, subject to the parallelisability constraint of §6.

[Demo 1](demos/demo1_layers_are_online_learners.py) machine-checks the identities
behind rows 1–6 (softmax attention ≡ Nadaraya–Watson; linear attention ≡ Hebbian fast
weights; DeltaNet ≡ online SGD on the memory loss; Gated DeltaNet ≡ decay-then-step)
and then measures *why* the field walked down this column: with correlated keys,
Hebbian read-back suffers crosstalk (mean error 0.79 in the demo's configuration)
while the delta rule — which is exactly Kaczmarz's 1937 projection method for linear
systems — drives it to ~10⁻² and, given passes, to zero.

## 4. Pillar A — algebra: expressivity became a design dial

The deepest change since 2023 is that *what a layer can compute* stopped being
folklore and became a set of sharp theorems keyed to one object: the spectrum and
commutativity of the transition operators A(x).

- Fixed-depth, log-precision transformers sit inside uniform **TC⁰** [1]; so do
  diagonal SSMs — Mamba included — despite "recurrence" (*the illusion of state*) [2].
  Neither can, at any scale, solve problems complete for **NC¹** (unless TC⁰ = NC¹),
  and the canonical NC¹-complete problem is composing permutations in S₅
  (Barrington's theorem [3]).
- Products of *commuting* operators flatten: a diagonal-gated recurrence is a sum of
  per-token terms weighted by cumulative products of scalars — iterated scalar
  multiplication is in TC⁰ [4]. The associative-scan trick that makes these models
  fast and their expressivity ceiling are *the same fact*.
- Gates in [0,1] cannot even represent parity: products of non-negative scalars never
  oscillate. Allowing eigenvalue **−1** buys parity and modular counting [22];
  allowing the **complex unit circle** buys counting mod k (k-th roots of unity) —
  this is exactly Mamba-3's complex-state move, which its authors show is equivalent
  to data-dependent RoPE [18]; positional rotations and state spectra are one
  mechanism.
- Householder transitions (DeltaNet's I − βkkᵀ at β → 2) generate reflections, hence
  the full orthogonal group, hence every permutation: the S₅ word problem becomes an
  *exact construction* — and with n_h Householders per token (DeltaProduct [12]) any
  S₅ token stream is tracked in one layer. RWKV-7's transition family is likewise
  provably beyond TC⁰ [11]. Under standard conjectures, **these recurrences are
  strictly more expressive than the transformers they aim to replace** — a first.
- Chain-of-thought is the same theorem family from the other side: serial token
  generation buys the depth that fixed-depth parallel computation lacks [23]. Serial
  compute can live in generated tokens (CoT) or in the transition algebra (above);
  architectures are choosing where to put it.

[Demo 2](demos/demo2_transition_algebra.py) makes every positive claim constructive
and exact, with no training: parity from a {−1,+1} gate at any length; mod-3 counting
from e^{2πi/3}; commuting vs non-commuting transition products; the TC⁰ flattening
identity for diagonal SSMs; and 60 random S₅ token streams (length ≤ 300) tracked to
machine precision by streamed DeltaNet steps — all 120 group elements verified as
Householder micro-step products.

Design consequence, already visible in shipped systems: the eigenvalue range of the
gate, the rank structure of the update, and the number of micro-steps per token are
now *specification-level choices* with known computational consequences, the way
numerical analysts choose an integrator's order. (Mamba-3 explicitly frames its
discretisation and complex spectrum this way [18].)

## 5. Pillar B — statistics: memory as an estimator with a capacity theory

The same table read as statistics: softmax attention is the nonparametric end
(keeps every sample, pays O(T) per query, never crosstalks, cannot extrapolate
outside the convex hull of its values — demo 1 checks that too); fixed-state layers
are parametric compressions (O(1) cost, bounded capacity, crosstalk under correlated
keys). Between them:

- The delta rule is the *correct* online least-squares step where Hebbian storage is
  merely the zeroth-order one — demo 1d quantifies the gap; QK-normalisation drops out
  of the same regression view as a conditioning fix rather than a trick [13].
- **Log-linear attention** [19] answers "O(1) or O(T)?" with "O(log T)": a
  Fenwick-tree hierarchy of fixed-size states, finer for recent context — a
  multi-resolution estimator whose bias grows with temporal distance. That is the
  first principled interior point on the memory-topology axis.
- Trainable sparse attention (NSA, MoBA, DeepSeek's DSA [20][21]) is the dual
  approach: keep the nonparametric memory, learn to prune the kernel's support.
- ATLAS's polynomial feature maps [17] raise the parametric memory's capacity the
  classical way — richer regressors — and its sliding-window "Omega rule" objective
  replaces per-token SGD with a windowed regression, trading recency bias for local
  optimality.

The open statistical question — optimal state allocation for a given
context-length distribution — is §9.3.

## 6. Pillar C — the constraint: it must factor into large matrix multiplications

The selection pressure that decides which mathematics ships is a cost model:
tensor-core FLOPs are ~cheap, memory bandwidth and sequential dependencies are not.
Every surviving design obeys it, and several exist *because* of it:

- DeltaNet became viable only when its Householder-product recurrence was rewritten
  chunkwise via the **WY representation** (Bischof & Van Loan, 1987) [9];
- Mamba-2's SSD is exactly the block decomposition of a **semiseparable matrix** into
  diagonal (attention-like, matmul) and low-rank (recurrent) blocks [8];
- FlashAttention is streaming-softmax associativity turned into tiles [25];
- Mamba-3's MIMO reformulation exists to raise decode-time arithmetic intensity [18];
- 2-simplicial attention was published *with its Triton kernels*, because a trilinear
  form only matters if it tiles [26];
- ATLAS is explicit that its inner optimiser was chosen among the ones that
  parallelise across the sequence [17].

This is mathematics under a chosen metric — hardware-contingent, and it would change
if the substrate changed. But within the current regime it yields a crisp survival
criterion, worth stating as the field's working meta-theorem: **an architectural idea
ships if and only if its algebra factors into large matrix multiplications with
chunk-level parallelism across the sequence.** (Full attention pays O(T²) and still
ships because it tiles perfectly; elegant ideas that serialise do not ship at any
asymptotic cost.) The ideas in §9 are stated so that this constraint is part of the
problem, not an afterthought.

## 7. Pillar D — geometry: one duality map, two timescales

The optimiser thread used to be separate from architecture. It no longer is:

- μP fixed *scale*: parameterise so the infinite-width limit does feature learning,
  and hyperparameters transfer [27].
- Muon fixed *geometry*: the steepest-descent step for a matrix parameter under the
  spectral norm is the **polar factor** of the gradient (dual norm = nuclear norm),
  computed by Newton–Schulz iteration; "modular duality" generalises this to
  per-layer norms [28][29][30]. Moonshot validated it at 1T parameters / 15.5T tokens
  (Kimi K2, MuonClip) [31].
- ATLAS then imported the *same map* into the forward pass as the inner memory
  optimiser [17] — and recent theory shows Muon's outer-loop advantage over Adam is
  itself an associative-memory effect (heavy-tailed feature frequencies) [32],
  closing the circle: **the outer optimiser's mathematics and the inner memory's
  mathematics are the same object at two timescales.**

[Demo 3](demos/demo3_one_dual_map_two_timescales.py) verifies the map itself from
first principles, stdlib only: cubic Newton–Schulz converges to an orthogonal O with
OᵀG symmetric PSD (the polar characterisation), ⟨G,O⟩ equals the nuclear norm computed
by an independent Jacobi eigensolver, and O dominates 500 random orthogonal
contenders.

Once the architecture *is* an optimiser (§3), this pillar stops being about training
efficiency and becomes architecture theory proper: the inner loop needs its μP (what
is the width-scaling limit of a test-time learner?) and has just been handed its
geometry (what norm should a *memory write* be steepest in?). Neither question was
even well-posed in 2023. Both are now concrete (§9.2).

## 8. Production evidence (as of 2026-08)

The unification is not a paper exercise; it is deployed, with real disagreement only
about *where on the axes* to sit:

| System | Position in the design space |
|---|---|
| Qwen3-Next (Sep 2025) → Qwen3.5-397B-A17B (Feb 2026) | Gated DeltaNet : full attention ≈ 3:1 — delta-rule inner loop at frontier scale (*as reported* for Qwen3.5 [33]) |
| Kimi Linear (Oct 2025) → Kimi K3 (2026) | channelwise-gated delta rule (KDA); K3 *as reported* runs 69 of 93 layers linear [24][34] |
| Mamba-3 (ICLR 2026) | complex unit-circle spectrum, trapezoidal integrator, MIMO decode [18] |
| RWKV-7 "Goose" (2025) | generalised delta rule, provably beyond TC⁰ [11] |
| DeepSeek V3.2 / NSA (2025) | trainable sparsity over the nonparametric memory [20] |
| Nemotron-H, Granite 4.0, Falcon-H1, Jamba (2024–25) | SSM/attention hybrids in production |
| Titans / ATLAS (Google, 2025) | momentum and orthogonalised second-order steps inside the forward pass [16][17] |
| Kimi K2 (Jul 2025) | the dual map at the outer timescale: Muon at 1T parameters [31] |
| GPT-oss (Aug 2025) | the conservative point: alternating full/sliding-window attention with sinks |

Honest counter-evidence: MiniMax shipped lightning (linear) attention at scale in
MiniMax-01 (Jan 2025) [35], then reverted its flagship M2 (late 2025) to full
attention, citing infrastructure maturity and evaluation blind spots. At least one
frontier lab tried the bet and stepped back — the disagreement is real, the axes are
not settled, and full attention's simplicity still wins ties. §10's predictions take
that risk on explicitly.

## 9. Where the jump completes: five precisely stated open problems

The breakthrough, stated as the mathematics still missing. Each problem is concrete
enough to be attacked now; each has a named payoff; solving any two at frontier scale
is, I judge, sufficient for a step-change adopted across labs.

**9.1 The stable-and-complete parameterisation problem.** Find a continuous
parameterisation of transition operators that (a) guarantees stable recurrences
(spectrum in the closed unit disc), (b) achieves NC¹-complete state tracking in a
fixed number of layers, and (c) admits chunk-parallel evaluation. DeltaProduct offers
stability with growing n_h; RWKV-7 offers expressivity without a stability guarantee;
the gap between them is an open question flagged in that literature [12]. Payoff:
recurrences strictly and *safely* more expressive than transformers — algorithmic
generalisation (code, maths, agentic state) without CoT overhead on every step.
*(Addendum, same date: an invented candidate satisfying (a)–(c) at mechanism level —
with machine-checked receipts and honestly reported training costs — is developed in
[`ROTOR_DELTA.md`](ROTOR_DELTA.md).)*

**9.2 The inner-geometry problem.** The inner loop just acquired second-order steps
(ATLAS); it has no μP. Derive the scaling limit and the correct norm for test-time
memory updates — a "μP of the inner loop" and a "modular duality of memory writes" —
so inner learning rates and objectives transfer across width, depth, and window
length instead of being retuned per model. All pieces exist (tensor programs for the
outer loop [27]; dualised steepest descent [29][30]; the inner loop is now literally
an optimiser). Payoff: test-time learning that scales predictably — the property that
made μP indispensable, applied where the field is moving.

**9.3 The memory-topology problem.** Characterise the optimal state-size schedule
s(T) and its allocation across scales for a given query/recall distribution — the
rate-distortion theory of context. Log-linear attention proves O(log T) is
implementable [19]; sparse attention proves pruned O(T) is trainable [20][21]; hybrid
ratios (3:1, 69/93) are currently folklore. Payoff: replace today's hand-tuned hybrid
ratios with an allocation that provably dominates them per byte of state.

**9.4 The higher-order problem.** Determine when interactions beyond pairwise change
the *exponent* of the scaling law rather than its constant, and find their
matmul-factorable forms. The 2-simplicial result (trilinear attention; exponent gains
of order 7–20% on reasoning-heavy evals, token-constrained regime) is the existence
proof [26]; the test-time-regression framework independently derives higher-order
softmax generalisations [13]. In a token-scarce era, exponent changes compound into
the largest available prize — and a trilinear form is a tensor contraction, i.e.
still matmuls (§6). Payoff: better capability *slope*, not offset.

**9.5 The position–spectrum unification.** RoPE is a group action on activations;
Mamba-3 shows data-dependent rotations are equivalent to a complex state spectrum
[18]. Complete this: one theory of position-as-group-action covering rotary embedding,
gating, and state evolution, with the group learned or chosen per task (translation,
scaling, permutation subgroups for structured data). Payoff: length generalisation
and structured-data handling by construction rather than by patch (every current
long-context extension — YaRN-style scaling, NTK tricks — is a manual perturbation of
exactly this object).

A note on what was *considered and ranked lower* as "the" locus: mean-field/PDE
descriptions of token dynamics (beautiful, so far descriptive [36]); energy-based /
modern-Hopfield reformulations (re-derivations of the same memory object; adoption
thin); category-theoretic unifications (taxonomy, not capability); numerics/precision
work and KV-cache compression (indispensable engineering, but they optimise the
existing object). Each could surprise; none currently shows the
theory-plus-production convergence documented above.

## 10. Pre-registered predictions (scoreable)

Written 2026-08-10, in the spirit of this repository's pre-registered benchmark
designs. Score P1–P3 on 2027-12-31, P4–P5 on 2028-12-31. "Frontier release" = a
flagship or near-flagship model from a major lab, publicly documented.

- **P1 — richer inner loops.** ≥1 frontier release ships a sequence layer whose
  in-context update goes beyond a first-order single-step gated delta rule (multi-step
  Householder products, momentum, sliding-window objective, or orthogonalised/
  preconditioned steps). *Falsified if* every frontier release to the score date uses
  only softmax attention, sparsity, or single-step delta/decay updates.
- **P2 — expressivity as a cited design constraint.** ≥1 frontier release's tech
  report justifies an architectural choice by state-tracking/expressivity class
  (TC⁰/NC¹, group word problems, eigenvalue range) — proofs moving from post-hoc
  analysis into design documents. *Falsified if* none does.
- **P3 — the hybrid bet holds.** Of the labs shipping linear/hybrid flagships in
  mid-2026 (Alibaba/Qwen, Moonshot, NVIDIA-Nemotron, IBM-Granite), at least two keep
  or increase their linear:full ratio in their next flagship; the MiniMax-style
  full-attention reversion stays the minority. *Falsified if* a majority revert.
- **P4 — exponent, replicated.** An independent group (different lab from the
  original) publishes a ≥8B-parameter, ≥1T-token study either confirming a
  scaling-law *exponent* improvement for a higher-order attention variant or cleanly
  refuting it. I predict confirmation in the token-constrained regime. *Falsified
  if* refuted (partial credit to the framework: an exponent claim tested at all).
- **P5 — a 9.x problem falls.** One of §9.1–9.5 is solved to publication standard
  with frontier-scale evidence — most likely 9.1 or 9.3 first. *Falsified if* all
  five remain open on 2028-12-31.

Confidence, calibrated coarsely: P1 ~85%, P2 ~60%, P3 ~75%, P4 ~55%, P5 ~65%. If
fewer than three resolve true, the thesis of this document was wrong in a way that
matters, and §12 says how.

## 11. Why this lands in a retrieval repository

`rlat` is the nonparametric end of §5 made external and governable: retrieval over a
corpus *is* kernel regression with receipts — every "memory" carries provenance,
drift status, and a hash, and is never silently compressed. The theory above says the
field is racing down the parametric direction: compressing context into opaque fast
weights that answer quickly and cite nothing. Both ends survive on the mathematics —
compression wins on latency and state size; nonparametric memory is the only end that
can *verify*. As frontier models internalise more memory, the value of external
memory concentrates exactly where this repository already sits: owned, inspectable,
citeable evidence, and explicit control over how a model may use it. (The grounding
modes `augment`/`knowledge`/`constrain` are, in the language of §3, a user-held prior
over how much weight the consumer model's own parametric memory gets versus the
verified nonparametric one.)

## 12. Honest limits

- **No experiment at scale.** The demos prove identities and constructions, not that
  enriched inner loops win at 10²⁵ FLOPs. Representability ≠ learnability; the
  learnability evidence cited is other people's and mostly ≤10B scale outside the
  production systems of §8.
- **Selection risk.** A synthesis can mistake a well-published direction for the
  important one. The strongest specific counter-signal on record is MiniMax M2's
  reversion to full attention (§8).
- **The constraint could move.** §6's cost model is hardware-contingent; a substrate
  shift (extreme-sparsity accelerators, processing-in-memory) would re-rank the axes,
  and §9.3's answer in particular is relative to the cost model.
- **The jump could come from elsewhere.** Data curation, RL post-training, and
  test-time-compute strategies are compounding fast; "architecture" may simply matter
  less this cycle. I weight against this because exponent-level effects (§9.4) and
  provable expressivity gaps (§4) are the two levers that survive arbitrary data and
  post-training improvements — but it is a judgement, recorded here so it can be
  wrong in public.
- **Post-cutoff facts.** Qwen3.5 and K3 details are *as reported* in the cited
  secondary sources; layer ratios and dates should be re-verified against the primary
  tech reports before being quoted onward.

---

## References

Primary sources verified against the public record 2026-08-10. arXiv identifiers
given where confirmed.

1. Merrill & Sabharwal, *The Parallelism Tradeoff: Limitations of Log-Precision Transformers*, TACL 2023. arXiv:2207.00729
2. Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models*, ICML 2024. arXiv:2404.08819
3. Barrington, *Bounded-width polynomial-size branching programs recognize exactly those languages in NC¹*, JCSS 1989
4. Hesse, Allender & Barrington, *Uniform constant-depth threshold circuits for division and iterated multiplication*, JCSS 2002
5. Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML 2020. arXiv:2006.16236
6. Schmidhuber, *Learning to control fast-weight memories*, Neural Computation 1992; Schlag, Irie & Schmidhuber, *Linear Transformers are Secretly Fast Weight Programmers*, ICML 2021. arXiv:2102.11174
7. Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*, ICML 2024. arXiv:2312.06635
8. Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2), ICML 2024. arXiv:2405.21060
9. Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (DeltaNet), NeurIPS 2024. arXiv:2406.06484
10. Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025. arXiv:2412.06464
11. Peng et al., *RWKV-7 "Goose" with Expressive Dynamic State Evolution*, 2025. arXiv:2503.14456
12. Siems et al., *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*, NeurIPS 2025. arXiv:2502.10297
13. Wang, Shi & Fox, *Test-time regression: a unifying framework for designing sequence models with associative memory*, 2025. arXiv:2501.12352
14. Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*, 2024. arXiv:2407.04620
15. Behrouz et al., *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization* (MIRAS), 2025. arXiv:2504.13173
16. Behrouz, Zhong & Mirrokni, *Titans: Learning to Memorize at Test Time*, 2025. arXiv:2501.00663
17. Behrouz et al., *ATLAS: Learning to Optimally Memorize the Context at Test Time*, 2025. arXiv:2505.23735
18. *Mamba-3: Improved Sequence Modeling using State Space Principles*, ICLR 2026 (OpenReview id HwCvaJOiCj)
19. Guo, Yang et al., *Log-Linear Attention*, ICLR 2026. arXiv:2506.04761
20. Yuan et al. (DeepSeek), *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, 2025. arXiv:2502.11089
21. Lu et al. (Moonshot), *MoBA: Mixture of Block Attention for Long-Context LLMs*, 2025. arXiv:2502.13189
22. Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, ICLR 2025. arXiv:2411.12537
23. Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought*, ICLR 2024. arXiv:2310.07923; Li et al., *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*, ICLR 2024. arXiv:2402.12875
24. Moonshot AI, *Kimi Linear: An Expressive, Efficient Attention Architecture* (Kimi Delta Attention), Oct 2025
25. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. arXiv:2205.14135
26. Roy et al. (Meta), *Fast and Simplex: 2-Simplicial Attention in Triton*, 2025. arXiv:2507.02754 (lineage: Clift et al., *Logic and the 2-Simplicial Transformer*, 2019. arXiv:1909.00668)
27. Yang, Hu et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (μP), 2022. arXiv:2203.03466
28. Jordan et al., *Muon: An optimizer for hidden layers in neural networks*, 2024 (kellerjordan.github.io)
29. Bernstein & Newhouse, *Old Optimizer, New Norm: An Anthology*, 2024. arXiv:2409.20325
30. Bernstein & Newhouse, *Modular Duality in Deep Learning*, 2024. arXiv:2410.21265
31. Moonshot AI, *Kimi K2 technical report* (MuonClip; Muon at 1T parameters / 15.5T tokens), 2025; Liu et al., *Muon is Scalable for LLM Training* (Moonlight), 2025. arXiv:2502.16982
32. *Muon Outperforms Adam in Tail-End Associative Memory Learning*, 2025. arXiv:2509.26030
33. Labonne, *Qwen3.5: Nobody Agrees on Attention Anymore*, HuggingFace blog, Feb 2026 (secondary; reports the 3:1 Gated DeltaNet interleave in Qwen3.5-397B-A17B)
34. *Linear Attention at Frontier Scale: Kimi K3's KDA Claim, Fact-Checked*, acingai.com, 2026 (secondary; reports 69/93 linear layers)
35. MiniMax, *MiniMax-01: Scaling Foundation Models with Lightning Attention*, 2025. arXiv:2501.08313
36. Geshkovski, Letrouit, Polyanskiy & Rigollet, *A mathematical perspective on Transformers*, 2023. arXiv:2312.10794

Classical mathematics resurfacing in the table of §3: Kaczmarz (1937), Hebb (1949),
Householder (1958), Widrow–Hoff (1960), Nadaraya & Watson (1964), Krohn–Rhodes
(1965), Bischof & Van Loan's WY representation (1987), Barrington (1989), polar
decomposition via Newton–Schulz (Higham's lineage). The engine is new; the parts are
older than the people building it.
