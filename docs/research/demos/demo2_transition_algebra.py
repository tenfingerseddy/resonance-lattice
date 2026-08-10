"""Demo 2 — The transition operator's algebra decides what a layer can compute.

Write any of the modern recurrent mixers as  S_t = S_{t-1} A(x_t) + B(x_t).
The homogeneous part A(x) is the layer's "transition operator", and the
GROUP it generates is a hard ceiling on state tracking:

  - Mamba-2 / GLA / RetNet:   A(x) diagonal, entries in [0, 1]
        -> abelian transitions, spectrum can't leave the nonneg reals.
        Iterated products of commuting nonneg scalars flatten into prefix
        products - TC0 territory (Hesse et al. 2002). Parity is already
        out of reach at any scale (Grazzi et al. 2024; Merrill et al. 2024;
        Mamba-3 paper 2026 reports the same experimentally).

  - eigenvalue -1 allowed (DeltaNet with beta in (0,2], RWKV-7):
        parity becomes a ONE-LINE construction.

  - complex spectrum on the unit circle (Mamba-3), equivalently 2x2
    rotation blocks: counting mod k = k-th roots of unity.

  - Householder transitions (DeltaNet family):  A = I - 2 k k^T is a
    reflection; reflections generate the full orthogonal group, which
    contains every permutation matrix. Word problems over S5 - Barrington's
    (1989) NC1-complete problem, believed strictly beyond both transformers
    and diagonal SSMs (both TC0, Merrill & Sabharwal 2023) - are computed
    EXACTLY by streaming one DeltaNet step per generator. Multiple
    Householders per token = DeltaProduct (Siems et al. 2025).

Nothing here is trained. These are exact constructions - upper bounds on
what the families can REPRESENT (the cited papers show trained models reach
them in practice, and the impossibility halves are theorems, not runs).

Python stdlib only. Run:  python3 demo2_transition_algebra.py
"""

import cmath
import math
import random

from linalg import eye, matmul, max_abs_diff, mscale, msub, outer

rng = random.Random(20260810)
PASSES = []


def check(name, ok):
    PASSES.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# --------------------------------------------------------------------------
# 2a. Parity needs a sign flip in the spectrum.
#     With a_t in {+1, -1}:   s_t = a(x_t) * s_{t-1}   computes parity
#     exactly, for ANY length. With gates confined to [0, 1] (the
#     Mamba-2/GLA range) a product of gates can never change sign and its
#     magnitude never grows: the 1-D state physically cannot oscillate with
#     token count. (The full finite-precision impossibility for parity is
#     Grazzi et al. 2024, Thm 1; this is the mechanism behind it.)
# --------------------------------------------------------------------------

print("2a. parity: eigenvalue -1 is the whole trick")
ok = True
for _ in range(200):
    T = rng.randrange(1, 2000)
    bits = [rng.randrange(2) for _ in range(T)]
    s = 1.0
    for b in bits:
        s *= (-1.0 if b else 1.0)          # transition a(x) in {-1, +1}
    ok &= (s == (1.0 if sum(bits) % 2 == 0 else -1.0))
check("a(x) in {+1,-1} computes parity exactly at every length", ok)

ok = True
for _ in range(200):
    prod, sign_flipped = 1.0, False
    for _ in range(rng.randrange(1, 500)):
        g = rng.random()                   # gate in [0, 1]
        prev = prod
        prod *= g
        sign_flipped |= (prod < 0) or (abs(prod) > abs(prev) + 1e-15)
    ok &= not sign_flipped
check("[0,1]-gate products never flip sign, never grow (no oscillation available)", ok)

# --------------------------------------------------------------------------
# 2b. Counting mod k needs k-th roots of unity (rotations).
#     s_t = exp(2*pi*i/3)^{x_t} * s_{t-1} counts 1-tokens mod 3 exactly.
#     A complex unit eigenvalue is the same object as a 2x2 rotation block -
#     this is Mamba-3's "complex-valued SSM == data-dependent RoPE" move.
# --------------------------------------------------------------------------

print("2b. mod-3 counting via a unit-circle eigenvalue (rotation)")
w3 = cmath.exp(2j * math.pi / 3)
ok = True
for _ in range(200):
    bits = [rng.randrange(2) for _ in range(rng.randrange(1, 1500))]
    s = 1 + 0j
    for b in bits:
        if b:
            s *= w3
    target = sum(bits) % 3
    decoded = min(range(3), key=lambda r: abs(s - w3 ** r))
    ok &= (decoded == target)
check("spectrum exp(2*pi*i/3) counts mod 3 exactly at every length", ok)

# --------------------------------------------------------------------------
# 2c. Abelian vs non-abelian transitions.
#     Diagonal transitions commute; Householder transitions do not.
#     Commuting products flatten into an order-free closed form (that is the
#     associative-scan trick AND the TC0 ceiling at once). Non-commuting
#     products are genuinely sequential: over S5, iterated product is
#     NC1-complete (Barrington 1989).
# --------------------------------------------------------------------------

print("2c. the algebra: diagonal transitions commute, Householders don't")
n = 5
Da = [[(rng.random() if i == j else 0.0) for j in range(n)] for i in range(n)]
Db = [[(rng.random() if i == j else 0.0) for j in range(n)] for i in range(n)]
check("diag: A B == B A (abelian)", max_abs_diff(matmul(Da, Db), matmul(Db, Da)) < 1e-15)


def householder_swap(i, j, n):
    """I - 2 k k^T with k = (e_i - e_j)/sqrt(2)  ==  permutation matrix (i j)."""
    k = [0.0] * n
    k[i], k[j] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    return msub(eye(n), mscale(2.0, outer(k, k)))


H1, H2 = householder_swap(0, 1, n), householder_swap(1, 2, n)
check("Householder: A B != B A (non-abelian)", max_abs_diff(matmul(H1, H2), matmul(H2, H1)) > 0.5)

# Diagonal SSM flattening: recurrent evaluation == order-free prefix form
#   s_T = sum_t (prod_{s>t} a_s) * u_t   (elementwise). Every token's
# contribution is its own independent term - iterated SCALAR products, which
# sit inside TC0. No such flattening exists for the Householder stream.
print("    diagonal SSM == prefix-product closed form (the flattening that caps it at TC0)")
T = 64
a_seq = [[rng.random() for _ in range(n)] for _ in range(T)]
u_seq = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(T)]
s = [0.0] * n
for a, u in zip(a_seq, u_seq):
    s = [ai * si + ui for ai, si, ui in zip(a, s, u)]
closed = [0.0] * n
for t in range(T):
    w = [1.0] * n
    for sfx in range(t + 1, T):
        w = [wi * ai for wi, ai in zip(w, a_seq[sfx])]
    closed = [ci + wi * ui for ci, wi, ui in zip(closed, w, u_seq[t])]
check("recurrence == sum of per-token terms weighted by gate products",
      max(abs(x - y) for x, y in zip(s, closed)) < 1e-9)

# --------------------------------------------------------------------------
# 2d. The S5 word problem, solved exactly by streaming DeltaNet steps.
#
#     Represent permutation g by P(g): P e_i = e_{g(i)}. A transposition's
#     matrix IS a Householder reflection, i.e. exactly DeltaNet's transition
#     S (I - beta k k^T) at the boundary beta = 2 (the value the
#     negative-eigenvalue extension unlocks), with nothing written (v = 0).
#
#     Feeding one transposition token per step, the state carries the exact
#     group product g_1 g_2 ... g_T. For ARBITRARY S5 tokens, expand each
#     into <= 4 transpositions and take that many micro-steps per token -
#     which is precisely DeltaProduct with n_h = 4 (identity micro-steps =
#     beta 0).
# --------------------------------------------------------------------------

print("2d. streaming DeltaNet steps compute the S5 word problem exactly")


def perm_matrix(p):
    n = len(p)
    P = [[0.0] * n for _ in range(n)]
    for i, pi in enumerate(p):
        P[pi][i] = 1.0
    return P


def compose(g, h):                      # (g o h)(x) = g(h(x))
    return tuple(g[h[i]] for i in range(len(g)))


def cycle_decomposition_transpositions(p):
    """p as an ordered list of transpositions [(i,j), ...] with
    p = t_1 o t_2 o ... o t_m (function composition, leftmost applied last)."""
    m = len(p)
    seen, out = [False] * m, []
    for start in range(n):
        if seen[start] or p[start] == start:
            seen[start] = True
            continue
        cyc, x = [], start
        while not seen[x]:
            seen[x] = True
            cyc.append(x)
            x = p[x]
        # (c0 c1 ... ck) = (c0 ck) o (c0 c(k-1)) o ... o (c0 c1)
        out.extend((cyc[0], cyc[m]) for m in range(len(cyc) - 1, 0, -1))
    return out


ident = tuple(range(5))
all_perms = []


def heap_perms(arr, k, acc):
    if k == 1:
        acc.append(tuple(arr))
        return
    for i in range(k):
        heap_perms(arr, k - 1, acc)
        arr[0 if k % 2 else i], arr[k - 1] = arr[k - 1], arr[0 if k % 2 else i]


heap_perms(list(range(5)), 5, all_perms)

# sanity: decomposition really reproduces every element of S5
ok = True
for p in all_perms:
    M = eye(5)
    for (i, j) in cycle_decomposition_transpositions(p):
        M = matmul(M, householder_swap(i, j, 5))   # right-multiply, stream order
    ok &= max_abs_diff(M, perm_matrix(p)) < 1e-12
check("every S5 element == product of its transposition micro-steps (all 120 checked)", ok)

# the streaming run: tokens are arbitrary S5 elements
ok = True
for trial in range(60):
    T = rng.randrange(1, 300)
    tokens = [all_perms[rng.randrange(120)] for _ in range(T)]
    S = eye(5)                          # DeltaNet state, S_0 = I
    for g in tokens:
        for (i, j) in cycle_decomposition_transpositions(g):
            # S <- S (I - 2 k k^T): one DeltaNet micro-step, beta=2, v=0
            S = matmul(S, householder_swap(i, j, 5))
    ref = ident                          # ground truth group product
    for g in tokens:
        ref = compose(ref, g)            # running product g_1 g_2 ... g_T
    ok &= max_abs_diff(S, perm_matrix(ref)) < 1e-9
    # readout is linear: where did element 0 go? row argmax of S = ref[...]
    got = max(range(5), key=lambda r: S[0][r])
    ok &= (ref[got] == 0)
check("60 random streams (length <= 300): state == exact S5 group product", ok)

print()
print("Summary: what a recurrent layer can track is the GROUP its transitions")
print("generate. Nonneg diagonal gates -> abelian, TC0, no parity at any scale.")
print("Eigenvalue -1 -> parity. Unit-circle spectrum -> modular counting (Mamba-3,")
print("RoPE - same move). Householder products -> all of S5, i.e. NC1-complete")
print("state tracking (Barrington), strictly beyond transformers/diagonal SSMs")
print("unless TC0 == NC1. Expressivity is now a DESIGN DIAL, set by the spectrum")
print("and commutativity of A(x) - not an emergent mystery.")
print()
print("ALL PASS" if all(PASSES) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(PASSES) else 1)
