"""Demo 3 — One duality map, two timescales.

The steepest-descent direction for a matrix parameter depends on the norm
you measure steps in. Under the spectral norm, the optimal unit step against
gradient G is not G/||G|| but the POLAR FACTOR of G — the nearest orthogonal
matrix, msign(G) = U V^T from the SVD  G = U S V^T. Formally:

    argmax_{||A||_2 <= 1} <G, A>  =  U V^T,   with value  <G, UV^T> = sum_i s_i
    (trace duality: the spectral norm's dual is the nuclear norm)

That single map is now doing production work at BOTH timescales of learning:

  - OUTER LOOP (gradient descent on weights): Muon replaces the raw momentum
    update with its polar factor via Newton-Schulz iterations (Jordan et al.
    2024; Bernstein & Newhouse's "modular duality" is the general theory).
    Moonshot's Kimi K2 pre-trained a 1-trillion-parameter model this way
    (MuonClip, 15.5T tokens) — the largest known validation.

  - INNER LOOP (test-time learning of the in-context memory): ATLAS
    (Behrouz et al. 2025) updates its memory module with Muon-style
    orthogonalized steps — the same second-order-ish dual map, now running
    INSIDE the forward pass, per context window.

When the architecture is an optimizer (demo 1), optimizer mathematics IS
architecture mathematics. This demo verifies the map itself, stdlib only:

  1. cubic Newton-Schulz  X <- 1.5 X - 0.5 X X^T X  converges to an
     orthogonal matrix O;
  2. O is the polar factor:  O^T G is symmetric positive semidefinite
     (the two properties that uniquely characterize it);
  3. variational optimality: <G, O> equals the nuclear norm of G (checked
     against singular values from an independent Jacobi eigensolver) and
     beats every random orthogonal contender.

Run:  python3 demo3_one_dual_map_two_timescales.py
"""

import math
import random

from linalg import (eye, frob, gauss_matrix, jacobi_eigenvalues, matmul,
                    max_abs_diff, mscale, msub, random_orthogonal,
                    trace_product, transpose)

rng = random.Random(20260810)
PASSES = []


def check(name, err, tol):
    ok = err <= tol
    PASSES.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   (deviation {err:.2e})")


n = 6
G = gauss_matrix(rng, n, n)

# 1. Newton-Schulz: X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k, X_0 = G/||G||_F.
#    (Muon uses a tuned quintic for speed; the classical cubic converges to
#    the same fixed point — all singular values driven to 1.)
print("1. cubic Newton-Schulz iteration -> orthogonal matrix")
X = mscale(1.0 / frob(G), G)
for _ in range(60):
    X = msub(mscale(1.5, X), mscale(0.5, matmul(matmul(X, transpose(X)), X)))
O = X
check("O^T O == I", max_abs_diff(matmul(transpose(O), O), eye(n)), 1e-10)

# 2. polar-factor characterization: O orthogonal AND O^T G symmetric PSD
#    together pin O = U V^T uniquely (for nonsingular G).
print("2. O is the polar factor of G")
P = matmul(transpose(O), G)
check("O^T G symmetric", max_abs_diff(P, transpose(P)), 1e-9)
eigs = jacobi_eigenvalues([[0.5 * (P[i][j] + P[j][i]) for j in range(n)] for i in range(n)])
check("O^T G positive semidefinite (min eigenvalue >= 0)", max(0.0, -eigs[0]), 1e-9)

# 3. variational optimality:  <G, O> == nuclear norm == sum of singular
#    values (computed independently as sqrt eigs of G^T G), and no random
#    orthogonal direction does better.
print("3. O maximizes <G, A> over all orthogonal A (dual of the spectral norm)")
nuclear = sum(math.sqrt(max(0.0, e)) for e in jacobi_eigenvalues(matmul(transpose(G), G)))
gain_O = trace_product(G, O)
check("<G, O> == nuclear norm of G (independent Jacobi computation)",
      abs(gain_O - nuclear), 1e-8)
worst = max(trace_product(G, random_orthogonal(rng, n)) for _ in range(500))
ok = worst <= gain_O + 1e-12
PASSES.append(ok)
print(f"  [{'PASS' if ok else 'FAIL'}] beats 500 random orthogonal contenders "
      f"(best contender {worst:.4f} vs polar {gain_O:.4f})")

print()
print("Summary: 'orthogonalize the update' is not a heuristic — it is steepest")
print("descent once you measure steps in the spectral norm (whose dual is the")
print("nuclear norm). The identical map now trains trillion-parameter weights")
print("(Muon/K2, outer loop) and writes test-time memories (ATLAS, inner loop).")
print("Architecture and optimizer have become the same mathematical object at")
print("two timescales.")
print()
print("ALL PASS" if all(PASSES) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(PASSES) else 1)
