"""Demo 6 — Rotor intent operators for rlat retrieval (mechanism receipts).

rlat v2.0 ships the Intent Lattice as a kind tag only; the operators —
`anti`, `--toward`, multi-intent composition — are deferred to v2.1+
(src/resonance_lattice/intent/__init__.py). This demo prototypes those
operators as ROTOR operations on the query embedding, at the single plug-in
point `field/dense.py` exposes (`scores = band @ q` over L2-normalised
vectors), and checks the properties that make rotors the right substrate:

  toward(q, c, t): rotate q by t*angle(q,c) in plane span{q, c}
  anti           : the same rotation, negative angle (exact inverse)
  lens rotor     : ONE fixed rotation (plane from two anchors) applied to
                   every query — a portable, composable, auditable lens dial

Receipts:
  6a  exact unit norm at every t; t=1 lands exactly on the anchor;
      toward == SLERP (the geodesic); anti(toward(q)) == q to 1e-12
  6b  a global lens rotor is an exact ISOMETRY (zero resolution loss over
      the query distribution), while per-query pulls — rotor-toward AND
      additive — necessarily collapse resolution; additive additionally
      passes through degenerate norms (measured)
  6c  composition: rotor products compose exactly and near-commute at small
      angles — the commutator shrinks 4x when both angles halve (the
      Baker-Campbell-Hausdorff first-order law, measured)
  6d  a retrieval experiment on REAL text (this repository's own docs,
      LSA stand-in embeddings — see corpus.py caveat): ambiguous two-topic
      queries, disambiguated by `--toward <anchor>`; rotor vs raw vs
      additive vs trust-weight re-ranking. Mechanism-scale evidence only;
      the pre-registered production bench design lives in ROTOR_LENS.md.

Requires numpy (demos 1-5 remain stdlib-only).
Run:  python3 demo6_rotor_intent_ops.py
"""

from __future__ import annotations

import math

import numpy as np

from corpus import build_repo_corpus

rng = np.random.default_rng(20260810)
PASSES = []


def check(name, ok, detail=""):
    PASSES.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('   (' + detail + ')') if detail else ''}")


def unit(x):
    return x / np.linalg.norm(x)


def plane_rotate(x, u, v, theta):
    """Rotate x by theta in the oriented plane (u -> v), u,v orthonormal.
    R u = cos(theta) u + sin(theta) v. Exact rank-2 update, O(d)."""
    cu, cv = float(x @ u), float(x @ v)
    c, s = math.cos(theta), math.sin(theta)
    return x + cu * ((c - 1) * u + s * v) + cv * (-s * u + (c - 1) * v)


def toward(q, anchor, t):
    """Rotate unit q toward unit anchor by fraction t of their angle."""
    w = float(np.clip(q @ anchor, -1.0, 1.0))
    omega = math.acos(w)
    if omega < 1e-9:
        return q.copy()
    v = unit(anchor - w * q)                      # tangent at q, toward anchor
    return plane_rotate(q, q, v, t * omega)


def slerp(q, c, t):
    w = float(np.clip(q @ c, -1.0, 1.0))
    omega = math.acos(w)
    if omega < 1e-9:
        return q.copy()
    return (math.sin((1 - t) * omega) * q + math.sin(t * omega) * c) / math.sin(omega)


# ---------------------------------------------------------------- 6a ------

print("6a. exactness: unit norm, endpoint, geodesic identity, invertibility")
d = 256
worst_norm = worst_end = worst_slerp = worst_inv = 0.0
for _ in range(200):
    q, c = unit(rng.standard_normal(d)), unit(rng.standard_normal(d))
    t = float(rng.uniform(0, 1))
    out = toward(q, c, t)
    worst_norm = max(worst_norm, abs(float(np.linalg.norm(out)) - 1.0))
    worst_end = max(worst_end, float(np.linalg.norm(toward(q, c, 1.0) - c)))
    worst_slerp = max(worst_slerp, float(np.linalg.norm(out - slerp(q, c, t))))
    # exact inverse: rotate back by -t*omega in the same plane
    w = float(np.clip(q @ c, -1, 1))
    v = unit(c - w * q)
    back = plane_rotate(out, q, v, -t * math.acos(w))
    worst_inv = max(worst_inv, float(np.linalg.norm(back - q)))
check("output is unit-norm at every t (scores stay calibrated)", worst_norm < 1e-12,
      f"max dev {worst_norm:.1e}")
check("t=1 lands exactly on the anchor", worst_end < 1e-9, f"max dev {worst_end:.1e}")
check("toward == SLERP (the sphere's geodesic — not an approximation of it)",
      worst_slerp < 1e-9, f"max dev {worst_slerp:.1e}")
check("anti undoes toward exactly (auditable, reversible ops)", worst_inv < 1e-12,
      f"max dev {worst_inv:.1e}")

# ---------------------------------------------------------------- 6b ------

print("6b. resolution: global lens rotor is an isometry; per-query pulls are not")
Q = np.stack([unit(rng.standard_normal(d)) for _ in range(200)])
g1, g2 = unit(rng.standard_normal(d)), unit(rng.standard_normal(d))
u = g1
v = unit(g2 - float(g2 @ g1) * g1)
L = np.stack([plane_rotate(q, u, v, 0.6) for q in Q])          # one fixed rotor
G_before = Q @ Q.T
iso_dev = float(np.abs(L @ L.T - G_before).max())
check("lens rotor preserves ALL pairwise query similarities (exact isometry)",
      iso_dev < 1e-10, f"max dev {iso_dev:.1e}")

anchor = unit(rng.standard_normal(d))
T_rot = np.stack([toward(q, anchor, 0.5) for q in Q])
lam = 0.8
T_add = np.stack([unit(q + lam * anchor) for q in Q])
off = ~np.eye(200, dtype=bool)
m0 = float(G_before[off].mean())
m_rot = float((T_rot @ T_rot.T)[off].mean())
m_add = float((T_add @ T_add.T)[off].mean())
align_rot = float((T_rot @ anchor).mean())
align_add = float((T_add @ anchor).mean())
print(f"    mean pairwise query cosine: raw {m0:.3f} | toward(0.5) {m_rot:.3f} "
      f"(anchor-align {align_rot:.2f}) | additive(0.8) {m_add:.3f} (align {align_add:.2f})")
print("    -> any per-query pull trades resolution for alignment (both do);")
print("       only the GLOBAL lens rotor conditions retrieval at zero resolution cost.")
check("per-query pulls collapse resolution; global rotor does not",
      m_rot > m0 + 0.05 and m_add > m0 + 0.05 and iso_dev < 1e-10)

norms_mid = [float(np.linalg.norm(q - float(q @ anchor) * anchor)) for q in Q[:50]]
print(f"    additive pre-renormalisation norm can reach {min(norms_mid):.3f} "
      f"(degenerate direction noise near anchors); rotor never leaves the sphere.")

# ---------------------------------------------------------------- 6c ------

print("6c. composition: exact products, near-commuting at small angles (BCH law)")
u2 = unit(rng.standard_normal(d))
v2 = unit(rng.standard_normal(d) - 0.0 * u2)
v2 = unit(v2 - float(v2 @ u2) * u2)


def commutator_norm(th1, th2, probes=64):
    X = np.stack([unit(rng.standard_normal(d)) for _ in range(probes)])
    ab = np.stack([plane_rotate(plane_rotate(x, u, v, th1), u2, v2, th2) for x in X])
    ba = np.stack([plane_rotate(plane_rotate(x, u2, v2, th2), u, v, th1) for x in X])
    return float(np.linalg.norm(ab - ba) / math.sqrt(probes))


c_full = commutator_norm(0.4, 0.4)
c_half = commutator_norm(0.2, 0.2)
ratio = c_full / max(c_half, 1e-12)
check("halving both angles shrinks the order effect ~4x (first-order BCH, measured)",
      3.4 < ratio < 4.6, f"ratio {ratio:.2f}")
print("    -> multi-intent composition is well-behaved: order matters in principle")
print("       (non-abelian), negligibly at small strengths, exactly invertibly always.")

# ---------------------------------------------------------------- 6d ------

print("6d. disambiguation on real text (repo docs, LSA stand-in — mechanism scale)")
corpus = build_repo_corpus("/home/user/resonance-lattice")
band, coords = corpus["band"], corpus["coords"]
print(f"    corpus: {corpus['label']}")

grp_a = [i for i, (s, _, _) in enumerate(coords) if s.startswith("docs/internal/")]
grp_b = [i for i, (s, _, _) in enumerate(coords) if s.startswith("benchmarks/")]
rng.shuffle(grp_a)
rng.shuffle(grp_b)
fit_a, eval_a = grp_a[: len(grp_a) // 2], grp_a[len(grp_a) // 2:]
fit_b, eval_b = grp_b[: len(grp_b) // 2], grp_b[len(grp_b) // 2:]
anchor_a = unit(band[fit_a].mean(axis=0))
# The discriminative direction is the CONTRAST between intents, not a broad
# group's centroid (which, for a group covering half the corpus, nearly
# coincides with the global centroid — aiming there makes a query generic,
# not intent-flavoured). Operationally this is `--toward A` composed with
# `anti B` — the operator triple the intent stub lists is jointly necessary.
anchor_contrast = unit(unit(band[fit_a].mean(axis=0)) - unit(band[fit_b].mean(axis=0)))
is_a = np.array([s.startswith("docs/internal/") for (s, _, _) in coords])

N_PAIRS = 250
pairs = [(eval_a[int(rng.integers(len(eval_a)))], eval_b[int(rng.integers(len(eval_b)))])
         for _ in range(N_PAIRS)]

# `--toward` biases retrieval toward an intent REGION. The right primary
# metric is group precision in the top-k; the rank of the specific parent
# passage is the SECONDARY metric that exposes the strength dial's cost
# (aiming at a centroid trades specificity for group focus). A first version
# of this experiment scored only "find the parent passage" and was saturated
# at hit@5 = 100% — kept in git history as a design lesson: tie-breaking is
# an amplitude task (trust weights win it); re-aiming is for moving the
# query, and should be scored as such.


def evaluate(method, k=10):
    prec = 0.0
    ranks = []
    for a_idx, b_idx in pairs:
        q = unit(band[a_idx] + band[b_idx])        # ambiguous two-topic query
        q2, weights = method(q)
        scores = band @ q2
        if weights is not None:
            scores = scores * weights
        order = np.argsort(-scores)
        prec += float(is_a[order[:k]].mean())
        ranks.append(int(np.where(order == a_idx)[0][0]) + 1)
    return 100.0 * prec / N_PAIRS, float(np.median(ranks))


methods = {
    "raw query                 ": lambda q: (q, None),
    "rotor->centroid t=0.3 (X) ": lambda q: (toward(q, anchor_a, 0.3), None),
    "rotor->centroid t=0.6 (X) ": lambda q: (toward(q, anchor_a, 0.6), None),
    "rotor->contrast t=0.15    ": lambda q: (toward(q, anchor_contrast, 0.15), None),
    "rotor->contrast t=0.3     ": lambda q: (toward(q, anchor_contrast, 0.3), None),
    "rotor->contrast t=0.6     ": lambda q: (toward(q, anchor_contrast, 0.6), None),
    "additive contrast +0.25   ": lambda q: (unit(q + 0.25 * anchor_contrast), None),
    "additive contrast +0.5    ": lambda q: (unit(q + 0.5 * anchor_contrast), None),
    "trust x1.25 (needs glob)  ": lambda q: (q, np.where(is_a, 1.25, 1.0)),
}
results = {}
print(f"    {N_PAIRS} ambiguous (internal-docs + benchmarks) midpoint queries;")
print("    primary: group-A precision@10 (intent focus); secondary: median rank of")
print("    the specific A-parent passage (specificity cost of aiming at a centroid):")
print("      method                    | A-prec@10 | median parent rank")
for name, fn in methods.items():
    p10, mrank = evaluate(fn)
    results[name] = (p10, mrank)
    print(f"      {name} |   {p10:5.1f}   | {mrank:6.1f}")

raw_p = results["raw query                 "][0]
cen_p = results["rotor->centroid t=0.3 (X) "][0]
rot_p = max(results["rotor->contrast t=0.3     "][0], results["rotor->contrast t=0.6     "][0])
add_p = max(results["additive contrast +0.25   "][0], results["additive contrast +0.5    "][0])
check("aiming at a broad centroid is ANTI-discriminative (documented anti-pattern)",
      cen_p < raw_p, f"{raw_p:.1f} -> {cen_p:.1f}")
check("aiming at the CONTRAST lifts group precision >= 15 points (toward+anti jointly)",
      rot_p >= raw_p + 15.0, f"{raw_p:.1f} -> {rot_p:.1f}")
check("the anti-pattern also costs specificity (centroid t=0.6 parent rank collapses) "
      "while the contrast keeps the parent at rank ~1 even at t=0.6",
      results["rotor->centroid t=0.6 (X) "][1] > 5.0
      and results["rotor->contrast t=0.6     "][1] <= 2.0)
print(f"    rotor vs additive on the contrast: {rot_p:.1f} vs {add_p:.1f} A-prec@10 — "
      f"{'rotor ahead' if rot_p > add_p + 1 else 'additive ahead' if add_p > rot_p + 1 else 'tie'}.")
print("    trust weights win tie-breaking outright WHEN the intent is expressible as")
print("    a source glob; rotors condition toward regions no glob can name. The two")
print("    are complementary dials (amplitude vs phase), not competitors.")
print("    (Pre-registered in ROTOR_LENS.md: if additive matches rotor on the real")
print("     encoder bench, the rotor claim shrinks to exactness/invertibility/audit.)")

print()
print("Summary: toward/anti/compose exist in closed form on the query side of")
print("`band @ q` — unit-norm always, geodesic-exact, invertible, composable, and")
print("as a GLOBAL lens rotor, resolution-lossless (an isometry). Per-query pulls")
print("(rotor or additive) buy alignment with resolution; the lens form does not.")
print()
print("ALL PASS" if all(PASSES) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(PASSES) else 1)
