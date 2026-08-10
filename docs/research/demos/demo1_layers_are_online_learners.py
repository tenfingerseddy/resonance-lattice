"""Demo 1 — Every major sequence-mixing layer is a regression estimator.

The unifying identity behind the 2024-2026 architecture wave (linear
attention, DeltaNet, Gated DeltaNet, Mamba-2/GLA, TTT, Titans, ATLAS):

    a sequence layer = an associative memory, trained ONLINE, inside the
    forward pass, by an explicit optimization algorithm.

This script verifies the identities exactly (to float precision) on random
data, then measures WHY the field moved from Hebbian storage to the delta
rule: crosstalk under correlated keys.

Everything here is Python stdlib. Run:  python3 demo1_layers_are_online_learners.py
"""

import math
import random

from linalg import (eye, gauss_matrix, gauss_vector, gram_schmidt, matmul,
                    matvec, max_abs_diff, msub, mscale, madd, outer,
                    normalize, transpose, vdot, vnorm, vsub)

rng = random.Random(20260810)
PASSES = []


def check(name, err, tol=1e-9):
    ok = err <= tol
    PASSES.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   (max deviation {err:.2e})")


# --------------------------------------------------------------------------
# 1a. Softmax attention IS Nadaraya-Watson kernel regression (1964).
#
#     attn(q) = sum_i softmax(q.k_i/sqrt(d))_i v_i
#             = sum_i K(q,k_i) v_i / sum_i K(q,k_i),   K = exp(<q,k>/sqrt(d))
#
# i.e. a NONPARAMETRIC estimator of the map k -> v, evaluated at q.
# Nonparametric = it never compresses: it must keep every (k_i, v_i) pair
# (the KV cache) and pay O(T) per query. That is the transformer's defining
# cost, restated as a statistics fact.
# --------------------------------------------------------------------------

def softmax_attention(q, keys, values, scale):
    logits = [vdot(q, k) / scale for k in keys]
    m = max(logits)
    w = [math.exp(x - m) for x in logits]
    z = sum(w)
    w = [x / z for x in w]
    d_v = len(values[0])
    out = [0.0] * d_v
    for wi, v in zip(w, values):
        for j in range(d_v):
            out[j] += wi * v[j]
    return out, w


def nadaraya_watson(q, keys, values, scale):
    kern = [math.exp(vdot(q, k) / scale) for k in keys]
    z = sum(kern)
    d_v = len(values[0])
    out = [0.0] * d_v
    for ki, v in zip(kern, values):
        for j in range(d_v):
            out[j] += (ki / z) * v[j]
    return out


print("1a. softmax attention == Nadaraya-Watson kernel regression")
d, d_v, T = 8, 4, 32
keys = [gauss_vector(rng, d) for _ in range(T)]
values = [gauss_vector(rng, d_v) for _ in range(T)]
q = gauss_vector(rng, d)
a, w = softmax_attention(q, keys, values, math.sqrt(d))
b = nadaraya_watson(q, keys, values, math.sqrt(d))
check("attention output == kernel-regression estimate", max(abs(x - y) for x, y in zip(a, b)), 1e-12)

# Kernel smoothers interpolate: the output is a convex combination of the
# stored values, so it can NEVER leave their convex hull.
lo = [min(v[j] for v in values) for j in range(d_v)]
hi = [max(v[j] for v in values) for j in range(d_v)]
inside = all(lo[j] - 1e-12 <= a[j] <= hi[j] + 1e-12 for j in range(d_v))
check("attention weights are convex (>=0, sum 1); output stays in conv(values)",
      0.0 if (inside and abs(sum(w) - 1.0) < 1e-12 and min(w) >= 0.0) else 1.0, 0.5)

# --------------------------------------------------------------------------
# 1b. Linear attention (Katharopoulos et al. 2020) IS a fast-weight program
#     (Schmidhuber 1992): one HEBBIAN step per token on a linear memory.
#
#     attention form:   o = sum_i <q, k_i> v_i
#     online-learner:   W_t = W_{t-1} + v_t k_t^T          (pure Hebb, lr=1)
#                       o   = W_T q
#
# The Hebbian step is gradient DESCENT on the correlation loss
#     L_t(W) = -<v_t, W k_t>,   grad = -v_t k_t^T.
# --------------------------------------------------------------------------

print("1b. linear attention == Hebbian fast-weight program (one SGD step/token)")
attn_form = [0.0] * d_v
for k, v in zip(keys, values):
    c = vdot(q, k)
    attn_form = [o + c * vj for o, vj in zip(attn_form, v)]

W = [[0.0] * d for _ in range(d_v)]
for k, v in zip(keys, values):
    W = madd(W, outer(v, k))           # W += v k^T  ==  W -= grad(-<v, W k>)
fw_form = matvec(W, q)
check("sum_i <q,k_i> v_i  ==  (sum_i v_i k_i^T) q", max(abs(x - y) for x, y in zip(attn_form, fw_form)), 1e-10)

# --------------------------------------------------------------------------
# 1c. DeltaNet (Yang et al. 2024) IS online gradient descent on the
#     least-squares memory objective  L_t(S) = 1/2 ||S k_t - v_t||^2 :
#
#     grad_S L_t = (S k_t - v_t) k_t^T
#     SGD step:  S <- S - b (S k_t - v_t) k_t^T
#              = S (I - b k_t k_t^T) + b v_t k_t^T      <- DeltaNet recurrence
#
# The delta rule is Widrow-Hoff / LMS (1960). Its transition operator
# (I - b k k^T) is exactly a (generalized) Householder map (1958) - the
# fact demo 2 builds on.
#
# Note DeltaNet L2-normalizes keys: with ||k||=1 and b in (0,2), the
# transition's spectrum lies in [-1,1] and the recurrence is stable. (With
# raw Gaussian keys the eigenvalue 1 - b||k||^2 can be far outside [-1,1]
# and the state blows up - stability pins the spectrum to [-1,1], which is
# precisely the dial demo 2 turns.)
# --------------------------------------------------------------------------

print("1c. DeltaNet recurrence == online SGD on 1/2||S k - v||^2")
beta = 0.7
keys = [normalize(k) for k in keys]
S_rec = [[0.0] * d for _ in range(d_v)]
S_sgd = [[0.0] * d for _ in range(d_v)]
for k, v in zip(keys, values):
    # recurrence form: S (I - b k k^T) + b v k^T
    P = msub(eye(d), mscale(beta, outer(k, k)))
    S_rec = madd(matmul(S_rec, P), mscale(beta, outer(v, k)))
    # explicit SGD form: S - b * grad
    err = vsub(matvec(S_sgd, k), v)
    S_sgd = msub(S_sgd, mscale(beta, outer(err, k)))
check("DeltaNet state == SGD-on-memory-loss state", max_abs_diff(S_rec, S_sgd), 1e-10)

# Gated DeltaNet (Yang et al. 2025) = decoupled weight decay, then the step:
#     S <- a_t * S;  S <- S - b (S k - v) k^T
print("    gated variant == multiplicative decay (weight decay) + delta step")
alpha = 0.93
S_rec = [[0.0] * d for _ in range(d_v)]
S_sgd = [[0.0] * d for _ in range(d_v)]
for k, v in zip(keys, values):
    P = msub(eye(d), mscale(beta, outer(k, k)))
    S_rec = madd(matmul(mscale(alpha, S_rec), P), mscale(beta, outer(v, k)))
    S_dec = mscale(alpha, S_sgd)                       # decoupled decay
    err = vsub(matvec(S_dec, k), v)
    S_sgd = msub(S_dec, mscale(beta, outer(err, k)))   # then SGD step
check("Gated DeltaNet state == decay-then-SGD state", max_abs_diff(S_rec, S_sgd), 1e-10)

# --------------------------------------------------------------------------
# 1d. WHY the delta rule won: crosstalk under correlated keys.
#
# Store N (key, value) pairs, then read every key back.
#   Hebbian readback:  W k_j = v_j + sum_{i!=j} <k_i, k_j> v_i
#                      exact only if keys are orthonormal.
#   Delta readback:    online least squares; cycling over the pairs is
#                      exactly Kaczmarz's method (1937) for solving the
#                      linear system S K = V, which converges whenever the
#                      keys are linearly independent (N <= d suffices
#                      generically) - even when strongly correlated.
# --------------------------------------------------------------------------

print("1d. crosstalk: Hebbian vs delta-rule storage, N=12 pairs in d=16")


def store_hebbian(pairs, d):
    W = [[0.0] * d for _ in range(len(pairs[0][1]))]
    for k, v in pairs:
        W = madd(W, outer(v, k))
    return W


def store_delta(pairs, d, beta, n_passes):
    S = [[0.0] * d for _ in range(len(pairs[0][1]))]
    for _ in range(n_passes):
        for k, v in pairs:
            err = vsub(matvec(S, k), v)
            S = msub(S, mscale(beta, outer(err, k)))
    return S


def readback_error(W, pairs):
    return sum(vnorm(vsub(matvec(W, k), v)) for k, v in pairs) / len(pairs)


d, N = 16, 12
val = [normalize(gauss_vector(rng, d)) for _ in range(N)]

for corr, label in [(0.0, "orthonormal keys "), (0.85, "correlated keys  ")]:
    if corr == 0.0:
        K = gram_schmidt(gauss_matrix(rng, d, d))
        ks = transpose(K)[:N]
    else:
        g = normalize(gauss_vector(rng, d))
        ks = [normalize([corr * gi + math.sqrt(1 - corr ** 2) * ei
                         for gi, ei in zip(g, gauss_vector(rng, d))]) for _ in range(N)]
    pairs = list(zip(ks, val))
    e_hebb = readback_error(store_hebbian(pairs, d), pairs)
    e_d1 = readback_error(store_delta(pairs, d, 1.0, 1), pairs)
    e_d25 = readback_error(store_delta(pairs, d, 1.0, 25), pairs)
    print(f"    {label}: Hebbian {e_hebb:8.4f} | delta x1 pass {e_d1:8.4f} | delta x25 passes {e_d25:.2e}")
    if corr == 0.0:
        check("Hebbian storage exact when keys orthonormal", e_hebb, 1e-9)
    else:
        PASSES.append(e_hebb > 10 * e_d25)
        print(f"  [{'PASS' if PASSES[-1] else 'FAIL'}] delta rule (Kaczmarz) beats Hebbian by >10x under correlation")

print()
print("Summary: softmax attention = nonparametric kernel regression (keeps everything,")
print("pays O(T) per token); linear attention = Hebbian memory (fast, but crosstalks);")
print("DeltaNet/Gated DeltaNet = online least squares with forgetting (Widrow-Hoff 1960 +")
print("Kaczmarz 1937), whose transition operators are Householder maps (1958).")
print("The architecture question 'what should attention be?' has become the")
print("statistics question 'which regression estimator should run in the forward pass?'")
print()
print("ALL PASS" if all(PASSES) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(PASSES) else 1)
