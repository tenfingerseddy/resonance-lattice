"""Demo 4 — The rotor gate: derivation receipts for an INVENTED transition family.

Proposal (this repository, 2026-08): make the memory transition a
data-dependent PLANE ROTATION, parameterised on the group by the exponential
of a data-dependent bivector,

    R(x) = exp( a(x) b(x)^T - b(x) a(x)^T ),      a(x), b(x) = linear maps of x

with the closed form (no matrix exponential, no iteration, no normalisation):

    Z  = a b^T - b a^T                (rank-2 skew: the minimal nontrivial one)
    lam = sqrt(|a|^2 |b|^2 - (a.b)^2)  (the rotation angle — Gram area of a,b)
    R  = I + (sin lam / lam) Z + ((1 - cos lam) / lam^2) Z^2

Composed with decay and the delta-rule write, the layer ("RotorDelta") is

    S_t = gamma_t * S_{t-1} R_t (I - beta_t k_t k_t^T) + beta_t v_t k_t^T .

Why invent this: existing families reach the group elements that hard state
tracking needs either at a parameter BOUNDARY (DeltaNet's beta -> 2), inside
an abelian torus (fixed-plane rotations: Mamba-3 complex states, RoPE), or
without a stability guarantee (diag + rank-1 with free spectrum). The rotor
gate puts the full rotation group — pi-rotations included — in the INTERIOR
of a smooth unconstrained parameter space, is exactly norm-preserving at
every parameter value, and keeps the diag + rank-2 algebra that chunkwise
(WY-style) kernels need.

This script machine-checks each mathematical claim. Stdlib only.
Run:  python3 demo4_rotor_gate.py
"""

import math
import random
import struct

from linalg import (eye, frob, gauss_vector, jacobi_eigenvalues, lu_det,
                    matmul, matvec, max_abs_diff, madd, mscale, msub, outer,
                    normalize, transpose, vdot, vsub, vscale)

rng = random.Random(20260810)
PASSES = []


def check(name, err, tol=1e-9):
    ok = err <= tol
    PASSES.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   (deviation {err:.2e})")


# ---------------------------------------------------------------- the gate

def rotor(a, b):
    """R = exp(a b^T - b a^T), closed form. Exact orthogonal for ANY raw a, b."""
    n = len(a)
    p, q, c = vdot(a, a), vdot(b, b), vdot(a, b)
    lam2 = max(p * q - c * c, 0.0)
    lam = math.sqrt(lam2)
    if lam < 1e-8:                                    # series limits
        c1, c2 = 1.0 - lam2 / 6.0, 0.5 - lam2 / 24.0
    else:
        c1, c2 = math.sin(lam) / lam, (1.0 - math.cos(lam)) / lam2
    Z = msub(outer(a, b), outer(b, a))
    return madd(eye(n), madd(mscale(c1, Z), mscale(c2, matmul(Z, Z))))


def expm(Z, terms=30):
    """Reference matrix exponential: scaling-and-squaring + Taylor."""
    n = len(Z)
    s = max(0, int(math.ceil(math.log2(max(frob(Z), 1e-12)))) + 1)
    Zs = mscale(1.0 / 2 ** s, Z)
    T, term = eye(n), eye(n)
    for k in range(1, terms):
        term = mscale(1.0 / k, matmul(term, Zs))
        T = madd(T, term)
    for _ in range(s):
        T = matmul(T, T)
    return T


print("4a. closed form == matrix exponential; exactly orthogonal for RAW a, b")
n = 6
worst_form = worst_orth = worst_det = 0.0
for _ in range(50):
    a = gauss_vector(rng, n, sigma=rng.choice([0.1, 1.0, 3.0]))
    b = gauss_vector(rng, n, sigma=rng.choice([0.1, 1.0, 3.0]))
    R = rotor(a, b)
    worst_form = max(worst_form, max_abs_diff(R, expm(msub(outer(a, b), outer(b, a)))))
    worst_orth = max(worst_orth, max_abs_diff(matmul(transpose(R), R), eye(n)))
    worst_det = max(worst_det, abs(lu_det(R) - 1.0))
check("R == expm(Z) (50 random raw a,b, mixed scales)", worst_form, 1e-8)
check("R^T R == I always — no normalisation, no clamping, no boundary", worst_orth, 1e-11)
check("det R == +1 always (proper rotation)", worst_det, 1e-9)

# --------------------------------------------------------------------------
# 4b. Stability is structural, not enforced.
#     gamma * R has spectral norm exactly gamma <= 1 at EVERY parameter
#     value. Compare: (i) a diag+rank-1 transition with free spectrum can
#     silently exceed radius 1 and explode; (ii) DeltaNet reaches the
#     orthogonal reflection only AT the boundary beta = 2 — anywhere short
#     of it, the stored signal decays as |1-beta|^m.
# --------------------------------------------------------------------------

print("4b. stability: isometry at every parameter value vs boundary/unstable families")
prod = eye(n)
for _ in range(400):
    prod = matmul(prod, rotor(gauss_vector(rng, n), gauss_vector(rng, n)))
check("product of 400 random rotors: || . ||_F stays sqrt(n) (isometry)",
      abs(frob(prod) - math.sqrt(n)), 1e-10)

# a diag+rank-1 family with free parameters admits spectral radius > 1:
# sample until the rank-1 term feeds back into the diagonal and explodes.
grew = 0.0
while grew <= 10.0:
    D = [[(rng.uniform(0.9, 0.99) if i == j else 0.0) for j in range(4)] for i in range(4)]
    A = madd(D, mscale(0.5, outer(normalize(gauss_vector(rng, 4)),
                                  normalize(gauss_vector(rng, 4)))))
    Ak = eye(4)
    for _ in range(100):
        Ak = matmul(Ak, A)
    grew = frob(Ak)
PASSES.append(grew > 10.0)
print(f"  [{'PASS' if PASSES[-1] else 'FAIL'}] free-spectrum diag+rank-1 admits ||A^100||_F = "
      f"{grew:.2e} (stability NOT structural; needs constraints/clamps)")

for beta in (1.8, 1.98, 2.0):
    sig = abs(1.0 - beta) ** 512
    print(f"      DeltaNet reflection at beta={beta:4.2f}: stored signal after 512 swaps = {sig:.2e}"
          + ("   <- exact only AT the boundary" if beta == 2.0 else ""))

# --------------------------------------------------------------------------
# 4c. Group elements as SINGLE rotor steps.
#     Embed sigma in S_m as rho(sigma) = diag(P_sigma, det sigma) in SO(m+1).
#     - a transposition (i j) is the rotor with plane span{e_i - e_j, e_aux}
#       and angle pi;
#     - a 3-cycle is a single rotor with angle 2*pi/3 — which NO single
#       DeltaNet step can represent: I - beta k k^T is orthogonal only for
#       beta*|k|^2 in {0, 2}, giving det +1 (identity) or det -1
#       (reflection); a 3-cycle's embedding has det +1 and is not I.
# --------------------------------------------------------------------------

def perm_matrix(p):
    m = len(p)
    P = [[0.0] * m for _ in range(m)]
    for i, pi in enumerate(p):
        P[pi][i] = 1.0
    return P


def perm_sign(p):
    seen, sign = [False] * len(p), 1
    for s in range(len(p)):
        if not seen[s]:
            x, clen = s, 0
            while not seen[x]:
                seen[x] = True
                x, clen = p[x], clen + 1
            sign *= -1 if clen % 2 == 0 else 1
    return sign


def rho(p):
    """S_m -> SO(m+1): diag(P_p, det p). A homomorphism (checked below)."""
    m = len(p)
    M = [[0.0] * (m + 1) for _ in range(m + 1)]
    P = perm_matrix(p)
    for i in range(m):
        for j in range(m):
            M[i][j] = P[i][j]
    M[m][m] = float(perm_sign(p))
    return M


print("4c. permutations as single rotors (the step DeltaNet needs beta=2 / cannot take)")
m = 5
worst = 0.0
for (i, j) in [(0, 1), (1, 4), (2, 3)]:
    u = [0.0] * (m + 1)
    u[i], u[j] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    v = [0.0] * (m + 1)
    v[m] = 1.0
    # angle pi in plane span{u, v}: rotor with a = pi*u, b = v gives lam = pi
    R = rotor(vscale(math.pi, u), v)
    p = list(range(m))
    p[i], p[j] = j, i
    worst = max(worst, max_abs_diff(R, rho(p)))
check("transposition (i j) == rotor(plane span{e_i - e_j, e_aux}, angle pi)", worst, 1e-9)

cyc = [1, 2, 0, 3, 4]                     # the 3-cycle 0->1->2->0 in S5
target = rho(cyc)
u = normalize([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
v = normalize([1.0, 1.0, -2.0, 0.0, 0.0, 0.0])
th = 2 * math.pi / 3
best = min(max_abs_diff(rotor(vscale(+th, u), v), target),
           max_abs_diff(rotor(vscale(-th, u), v), target))
check("3-cycle (0 1 2) == a SINGLE rotor at angle 2*pi/3", best, 1e-9)
check("...whose det is +1 (so no single generalized-Householder step equals it)",
      abs(lu_det(target) - 1.0), 1e-9)

# every S5 element needs at most TWO rotor micro-steps: the symmetric part
# (R + R^T)/2 of rho(sigma) has eigenvalues 2cos(theta_i) (twice per rotation
# plane) and 1 (fixed directions) — count planes via eigenvalues < 2.
all_perms = []


def heap_perms(arr, k, acc):
    if k == 1:
        acc.append(tuple(arr))
        return
    for i in range(k):
        heap_perms(arr, k - 1, acc)
        arr[0 if k % 2 else i], arr[k - 1] = arr[k - 1], arr[0 if k % 2 else i]


heap_perms(list(range(5)), 5, all_perms)
max_planes = 0
for p in all_perms:
    Rp = rho(list(p))
    Sym = mscale(0.5, madd(Rp, transpose(Rp)))
    eigs = jacobi_eigenvalues(Sym)
    planes = sum(1 for e in eigs if e < 1.0 - 1e-9) / 2.0
    max_planes = max(max_planes, planes)
PASSES.append(max_planes <= 2.0)
print(f"  [{'PASS' if PASSES[-1] else 'FAIL'}] every one of the 120 S5 embeddings is a product of "
      f"<= 2 plane rotations (max found: {max_planes:.0f}) — 2 rotor micro-steps per arbitrary token")

# --------------------------------------------------------------------------
# 4d. Containments: the rotor family absorbs the known good points.
#     (i)  DeltaProduct's orthogonal corner: two Householder reflections
#          compose to ONE rotor (product of two reflections = rotation by
#          twice the angle between their normals, in their span).
#     (ii) Fixed-plane rotors commute — the abelian torus where Mamba-3's
#          complex states and RoPE live. Free planes do not commute.
# --------------------------------------------------------------------------

print("4d. two reflections == one rotor; fixed planes are abelian, free planes are not")


def householder(u):
    return msub(eye(len(u)), mscale(2.0, outer(u, u)))


u = normalize(gauss_vector(rng, n))
w0 = gauss_vector(rng, n)
w = normalize(vsub(w0, vscale(vdot(u, w0), u)))       # w perp u
phi = rng.uniform(0.2, 1.2)
wphi = normalize([math.cos(phi) * ui + math.sin(phi) * wi for ui, wi in zip(u, w)])
HH = matmul(householder(u), householder(wphi))
best = min(max_abs_diff(HH, rotor(vscale(+2 * phi, u), w)),
           max_abs_diff(HH, rotor(vscale(-2 * phi, u), w)))
check("H(u) H(w) == rotor(span{u,w}, 2*angle(u,w))  [DeltaProduct(2) corner ⊂ rotors]", best, 1e-9)

Ra, Rb = rotor(vscale(0.7, u), w), rotor(vscale(1.9, u), w)         # same plane
check("same-plane rotors commute (the Mamba-3 / RoPE torus)",
      max_abs_diff(matmul(Ra, Rb), matmul(Rb, Ra)), 1e-12)
a2, b2 = gauss_vector(rng, n), gauss_vector(rng, n)
Rc = rotor(a2, b2)                                                   # free plane
PASSES.append(max_abs_diff(matmul(Ra, Rc), matmul(Rc, Ra)) > 1e-3)
print(f"  [{'PASS' if PASSES[-1] else 'FAIL'}] free-plane rotors do NOT commute "
      f"(escapes the abelian/TC0 ceiling of fixed planes)")

# --------------------------------------------------------------------------
# 4e. RoPE is the constant-rotor special case.
#     With a CONSTANT rotor R, state-side accumulation
#         S_T = sum_s v_s k_s^T R^{T-s},  o = S_T q
#     equals linear attention with absolute-position-rotated keys/queries
#         o = sum_s <R^s k_s, R^T q> v_s        (relative property, exactly)
# --------------------------------------------------------------------------

print("4e. constant rotor == RoPE (state-side rotation == q/k rotation)")
dk, dv, T = 6, 3, 24
Rc = rotor(vscale(0.31, normalize(gauss_vector(rng, dk))), normalize(gauss_vector(rng, dk)))
ks = [gauss_vector(rng, dk) for _ in range(T)]
vs = [gauss_vector(rng, dv) for _ in range(T)]
q = gauss_vector(rng, dk)
S = [[0.0] * dk for _ in range(dv)]
for k, v in zip(ks, vs):
    S = madd(matmul(S, Rc), outer(v, k))          # S <- S R + v k^T
o1 = matvec(S, q)
# <R^s k_s, R^T q> with orthogonal R equals k_s^T R^{T-s} q; build it explicitly:
RT = eye(dk)
for _ in range(T):
    RT = matmul(RT, Rc)
o2 = [0.0] * dv
Rpow = eye(dk)
for s, (k, v) in enumerate(zip(ks, vs), start=1):
    Rpow = matmul(Rpow, Rc)
    w = vdot(matvec(Rpow, k), matvec(RT, q))       # <R^s k_s, R^T q>
    o2 = [oi + w * vi for oi, vi in zip(o2, v)]
check("state-side constant rotor output == RoPE-rotated linear attention output",
      max(abs(x - y) for x, y in zip(o1, o2)), 1e-9)

# --------------------------------------------------------------------------
# 4f. Chunk-parallel algebra (the WY-style form kernels need).
#     R = I + G with G = X N X^T, X = [a | b] (n x 2), N a 2x2 —
#     so gamma R is "scalar-diagonal + rank-2", the same shape Gated
#     DeltaNet kernels exploit at rank 1. The chunk product telescopes into
#     I + a sum of rank-2 corrections, assembled with n x 2 matmuls only.
# --------------------------------------------------------------------------

print("4f. chunkwise closed form == sequential product (n x 2 matmuls only)")


def rotor_factors(a, b):
    """Return (X, N) with R = I + X N X^T, X = [a|b]."""
    p, q, c = vdot(a, a), vdot(b, b), vdot(a, b)
    lam2 = max(p * q - c * c, 0.0)
    lam = math.sqrt(lam2)
    if lam < 1e-8:
        c1, c2 = 1.0 - lam2 / 6.0, 0.5 - lam2 / 24.0
    else:
        c1, c2 = math.sin(lam) / lam, (1.0 - math.cos(lam)) / lam2
    X = [[ai, bi] for ai, bi in zip(a, b)]
    N = [[-c2 * q, c1 + c2 * c], [-(c1) + c2 * c, -c2 * p]]
    return X, N


C = 16
gammas = [rng.uniform(0.9, 1.0) for _ in range(C)]
toks = [(gauss_vector(rng, n), gauss_vector(rng, n)) for _ in range(C)]
dense = eye(n)
for g, (a, b) in zip(gammas, toks):
    dense = matmul(dense, mscale(g, rotor(a, b)))

worst = 0.0
for (a, b) in toks:                                    # factor identity first
    X, N = rotor_factors(a, b)
    G = matmul(matmul(X, N), transpose(X))
    worst = max(worst, max_abs_diff(madd(eye(n), G), rotor(a, b)))
check("R == I + X N X^T with X = [a|b] (rank-2 factor form)", worst, 1e-9)

gprod = 1.0
corrections = []                                       # list of (W  n x 2, X  n x 2)
for g, (a, b) in zip(gammas, toks):
    X, N = rotor_factors(a, b)
    PX = [row[:] for row in X]                         # (I + sum W V^T) X, via n x 2 ops
    for (W, V) in corrections:
        VtX = matmul(transpose(V), X)                  # 2 x 2
        PX = madd(PX, matmul(W, VtX))
    corrections.append((matmul(PX, N), X))
    gprod *= g
closed = mscale(gprod, eye(n))
for (W, V) in corrections:
    closed = madd(closed, mscale(gprod, matmul(W, transpose(V))))
check("chunk closed form (scalar cumprod x [I + rank-2C]) == dense product",
      max_abs_diff(closed, dense), 1e-9)

# --------------------------------------------------------------------------
# 4g. Finite precision: drift is slow and SELF-REPAIRABLE with the demo-3 map.
#     Simulate fp32 storage of the running product of 5000 random rotors;
#     orthogonality drifts by rounding; two cubic Newton-Schulz steps
#     (exactly demo 3's polar map) restore it. Rotor states admit a cheap
#     periodic re-orthonormalisation because the TARGET manifold is known.
# --------------------------------------------------------------------------

print("4g. fp32 drift over 5000 products, then Newton-Schulz self-repair")
f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
P32 = eye(n)
for _ in range(5000):
    P32 = matmul(P32, rotor(gauss_vector(rng, n), gauss_vector(rng, n)))
    P32 = [[f32(x) for x in row] for row in P32]
drift = max_abs_diff(matmul(transpose(P32), P32), eye(n))
X = [row[:] for row in P32]
for _ in range(2):
    X = msub(mscale(1.5, X), mscale(0.5, matmul(matmul(X, transpose(X)), X)))
repaired = max_abs_diff(matmul(transpose(X), X), eye(n))
PASSES.append(drift < 1e-3 and repaired < 1e-6 and repaired < drift / 10)
print(f"  [{'PASS' if PASSES[-1] else 'FAIL'}] drift {drift:.2e} -> after 2 NS steps {repaired:.2e}")

print()
print("Summary: R(x) = exp(a(x)b(x)^T - b(x)a(x)^T) is a closed-form, exactly")
print("orthogonal, everywhere-smooth transition at EVERY raw parameter value;")
print("it reaches transpositions AND det=+1 elements (3-cycles) in ONE step,")
print("absorbs DeltaProduct's orthogonal corner, RoPE, and the fixed-plane")
print("torus as special cases; and it keeps the diag+rank-2 chunk algebra")
print("that production kernels need. Demo 5 shows WHY the interior chart")
print("matters: it is the difference between trainable and boundary-starved.")
print()
print("ALL PASS" if all(PASSES) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(PASSES) else 1)
