"""Minimal dense linear algebra, Python stdlib only.

Every matrix is a list of rows; every vector is a list of floats.
Sizes in the demos never exceed ~16x16, so clarity beats speed throughout.
"""

import math
import random


# ---------------------------------------------------------------- vectors

def vdot(u, v):
    return sum(a * b for a, b in zip(u, v))


def vsub(u, v):
    return [a - b for a, b in zip(u, v)]


def vscale(c, u):
    return [c * a for a in u]


def vnorm(u):
    return math.sqrt(vdot(u, u))


def normalize(u):
    n = vnorm(u)
    return [a / n for a in u]


# ---------------------------------------------------------------- matrices

def zeros(m, n):
    return [[0.0] * n for _ in range(m)]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A):
    return [list(col) for col in zip(*A)]


def madd(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def msub(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mscale(c, A):
    return [[c * a for a in row] for row in A]


def matmul(A, B):
    Bt = transpose(B)
    return [[vdot(row, col) for col in Bt] for row in A]


def matvec(A, x):
    return [vdot(row, x) for row in A]


def outer(u, v):
    return [[a * b for b in v] for a in u]


def frob(A):
    return math.sqrt(sum(a * a for row in A for a in row))


def max_abs_diff(A, B):
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def trace_product(A, B):
    """<A, B> = trace(A^T B), the Frobenius inner product."""
    return sum(a * b for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def gauss_matrix(rng, m, n, sigma=1.0):
    return [[rng.gauss(0.0, sigma) for _ in range(n)] for _ in range(m)]


def gauss_vector(rng, n, sigma=1.0):
    return [rng.gauss(0.0, sigma) for _ in range(n)]


# ------------------------------------------------------- orthogonalization

def gram_schmidt(A):
    """Orthonormalize the columns of a square matrix (assumed full rank)."""
    cols = transpose(A)
    out = []
    for c in cols:
        for q in out:
            c = vsub(c, vscale(vdot(q, c), q))
        out.append(normalize(c))
    return transpose(out)


def random_orthogonal(rng, n):
    return gram_schmidt(gauss_matrix(rng, n, n))


# ------------------------------------------------- symmetric eigenvalues

def jacobi_eigenvalues(A, sweeps=100, tol=1e-13):
    """Eigenvalues of a symmetric matrix via cyclic Jacobi rotations."""
    n = len(A)
    M = [row[:] for row in A]
    for _ in range(sweeps):
        off = math.sqrt(sum(M[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(M[p][q]) < 1e-300:
                    continue
                theta = 0.5 * math.atan2(2.0 * M[p][q], M[q][q] - M[p][p])
                c, s = math.cos(theta), math.sin(theta)
                for k in range(n):
                    mkp, mkq = M[k][p], M[k][q]
                    M[k][p] = c * mkp - s * mkq
                    M[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = M[p][k], M[q][k]
                    M[p][k] = c * mpk - s * mqk
                    M[q][k] = s * mpk + c * mqk
    return sorted(M[i][i] for i in range(n))
