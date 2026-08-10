"""Demo 5 — Training the rotor gate: what an honest experiment actually shows.

Demo 4 established what the rotor family can REPRESENT. This demo asks what
gradient descent actually REACHES, against two incumbents:

    rotor:  transition = rotation by angle lambda(a, b), raw unconstrained params
    delta:  transition eigenvalue 1 - beta, beta = 2*sigmoid(w)  (boundary at 2)
    gla:    transition eigenvalue sigmoid(w) in (0, 1)           (sign-fixed)

Three results, including one AGAINST the invention's naive pitch — kept
because this repository publishes what the runs say, not what the author
hoped (cf. benchmarks/r4_continuous_credit):

  A1. Angle landscapes oscillate. Trained directly on length-16 parity, the
      rotor parks in a spurious local minimum (negative control). A short
      length curriculum (T=2 then T=16, same budget for every model) finds
      the right basin.
  A2. RATE TIE, DISTINCT FAILURE PHYSICS (both derivations corrected by the
      run itself): on parity — an ABELIAN task whose exact transition both
      families can approach — the rotor sits in a QUARTIC valley (cosine
      readout => loss ~ err^4 => err ~ k^{-1/2}) and the sigmoid-boundary
      family has a QUADRATIC loss squashed through a saturating
      parameterisation (dL/dw ~ gap^2 => gap ~ k^{-1/2}). Same measured
      exponent 0.5 for both; neither reaches exactness at finite budget; on
      abelian tasks the rotor earns NO optimisation advantage. But the
      failure signatures differ in kind: the rotor's angle error is a PHASE
      error — accuracy OSCILLATES with evaluation length (chance at
      quadrature, ~0 anti-correlated at half period, ~1 again when it
      aliases) — while the delta gap is an AMPLITUDE error — accuracy decays
      MONOTONICALLY to chance. Both are exactly repairable post hoc by
      snapping to the nearest exact group element (theta := pi, beta := 2);
      the asymmetry appears only where no snap target exists (Part B).
  B.  The separation is REPRESENTABILITY, and it trains. On the S3 word
      problem with generators {(01), (012)} embedded in SO(4), both
      generators are proper rotations (det +1), so no single
      generalized-Householder step can equal either — there is nothing for
      the delta family to snap TO. Trained end to end (backprop hand-derived
      through the rotor closed form, verified by finite differences), rotor
      runs reach 100% at 8x the training length; the delta baseline stays at
      ~chance. Solving runs satisfy the S3 relations UP TO A CENTRAL SIGN —
      gradient descent discovers a double cover (a projective, spin-like
      representation), which the readout quotients out. Multi-basin
      landscapes are real: only some seeds solve; rates reported.

Python stdlib only. Run:  python3 demo5_interior_vs_boundary.py
"""

import math
import random

from linalg import madd, matmul, matvec, mscale, msub, outer, transpose, vdot

PASSES = []


def check(name, ok, detail=""):
    PASSES.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('   (' + detail + ')') if detail else ''}")


# ============================================================ Part A ======

print("Part A. parity (abelian): curriculum, convergence rates, failure signatures")

SIGMA_EVAL = 0.02          # additive readout noise at eval: interference floor
LOG2 = math.log(2.0)


def binom_logpmf(T, m):
    return (math.lgamma(T + 1) - math.lgamma(m + 1) - math.lgamma(T - m + 1)) - T * LOG2


def train_ms(T):
    return [(m, math.exp(binom_logpmf(T, m))) for m in range(T + 1)]


# after m swap tokens: yhat = (1 - s_m)/2, target m mod 2, exact expected MSE.
#   rotor: s_m = cos(m*theta);  delta: s_m = (1-beta)^m;  gla: s_m = g^m.

def loss_grad_rotor(theta, ms):
    L = dL = 0.0
    for m, w in ms:
        r = (1.0 - math.cos(m * theta)) / 2.0 - (m % 2)
        L += w * r * r
        dL += w * r * m * math.sin(m * theta)
    return L, dL


def loss_grad_sig(wp, scale, ms):
    s = 1.0 / (1.0 + math.exp(-wp))
    e = 1.0 - scale * s
    L = dL = 0.0
    for m, w in ms:
        r = (1.0 - e ** m) / 2.0 - (m % 2)
        L += w * r * r
        if m > 0:
            dL += w * r * (-m * e ** (m - 1))
    return L, dL * (-scale * s * (1.0 - s))


def train_scalar(loss_grad, x0, lrs, stages):
    best = None
    for lr in lrs:
        x = x0
        for T, steps in stages:
            ms = train_ms(T)
            for _ in range(steps):
                _, g = loss_grad(x, ms)
                x -= lr * g
        L, g = loss_grad(x, train_ms(stages[-1][0]))
        if best is None or L < best[0]:
            best = (L, x, g, lr)
    return best


# A1: negative control — no curriculum, spurious minimum
_, th_stuck, g_stuck, _ = train_scalar(loss_grad_rotor, 2.6, (0.05, 0.2, 0.8), ((16, 3000),))
print(f"    A1 negative control (no curriculum): theta parks at {th_stuck:.4f}, |grad| = "
      f"{abs(g_stuck):.1e} — spurious local minimum of an oscillatory angle landscape.")
check("angle landscape trap is real (stuck far from pi with ~zero gradient)",
      abs(th_stuck - math.pi) > 0.3 and abs(g_stuck) < 1e-8)

# A2: with curriculum, at two budgets -> convergence-rate exponents
budgets = (3000, 30000)
rotor_fit, delta_fit = [], []
for B in budgets:
    stages = ((2, 1500), (16, B))
    _, th, _, _ = train_scalar(loss_grad_rotor, 2.6, (0.05, 0.2, 0.8), stages)
    _, wd, gd, _ = train_scalar(lambda w, ms: loss_grad_sig(w, 2.0, ms), 2.197,
                                (0.05, 0.2, 0.8, 3.0), stages)
    rotor_fit.append(abs(th - math.pi))
    delta_fit.append((2.0 - 2.0 / (1.0 + math.exp(-wd)), gd))
_, wg, _, _ = train_scalar(lambda w, ms: loss_grad_sig(w, 1.0, ms), 0.0,
                           (0.05, 0.2, 0.8, 3.0), ((2, 1500), (16, 3000)))
gate = 1.0 / (1.0 + math.exp(-wg))

r_ratio = rotor_fit[0] / rotor_fit[1]
d_ratio = delta_fit[0][0] / delta_fit[1][0]
p_rotor = math.log(r_ratio) / math.log(budgets[1] / budgets[0])
p_delta = math.log(d_ratio) / math.log(budgets[1] / budgets[0])
print(f"    A2 curriculum finds the basin; convergence INTO it (10x budget: 3k -> 30k steps):")
print(f"       rotor |theta - pi| : {rotor_fit[0]:.2e} -> {rotor_fit[1]:.2e}   "
      f"measured exponent {p_rotor:.2f}  (quartic-valley derivation: 0.5)")
print(f"       delta gap (2-beta) : {delta_fit[0][0]:.2e} -> {delta_fit[1][0]:.2e}   "
      f"measured exponent {p_delta:.2f}  (quadratic-loss-through-sigmoid derivation: 0.5)")
print(f"       -> RATE TIE at k^(-1/2): on this abelian task the interior chart buys")
print(f"          no optimisation-rate advantage. The advantage must be structural.")
check("curriculum reaches the pi basin", rotor_fit[0] < 1e-3, f"|theta-pi| = {rotor_fit[0]:.1e}")
check("rotor rate exponent ~ 0.5 (degenerate quartic valley, as derived)",
      0.3 <= p_rotor <= 0.7, f"{p_rotor:.2f}")
check("delta rate exponent ~ 0.5 (quadratic loss through a saturating sigmoid, as derived)",
      0.3 <= p_delta <= 0.7, f"{p_delta:.2f}")


def accuracy(sig_of_m, T):
    lo = max(0, T // 2 - 8 * int(math.sqrt(T)) - 4)
    hi = min(T, T // 2 + 8 * int(math.sqrt(T)) + 4)
    acc = tot = 0.0
    for m in range(lo, hi + 1):
        w = math.exp(binom_logpmf(T, m))
        margin = -sig_of_m(m) / 2.0                 # yhat - 0.5
        if m % 2 == 0:
            margin = -margin
        acc += w * 0.5 * (1.0 + math.erf(margin / (SIGMA_EVAL * math.sqrt(2))))
        tot += w
    return acc / tot


theta3k, gap3k = math.pi - rotor_fit[0], delta_fit[0][0]
delta_err = rotor_fit[0]
# lengths chosen so the TYPICAL phase error m*|theta-pi| (m ~ T/2) hits
# quadrature (pi/2), anti-phase (pi), and full aliasing (2*pi):
T_quad = int(round(math.pi / delta_err))
T_anti = int(round(2 * math.pi / delta_err))
T_alias = int(round(4 * math.pi / delta_err))
print(f"    matched 3k-step budget, readout noise {SIGMA_EVAL}: PHASE vs AMPLITUDE failure")
print("      length     | rotor   | delta   | gla     |")
sigs = {"rotor": lambda m: math.cos(m * theta3k),
        "delta": lambda m: (gap3k - 1.0) ** m,
        "gla": lambda m: gate ** m}
table = {}
for T, tag in ((16, "train"), (1024, ""), (T_quad, "quadrature"),
               (T_anti, "anti-phase"), (T_alias, "aliased")):
    row = {k: accuracy(f, T) for k, f in sigs.items()}
    table[tag or T] = row
    print(f"      {T:10d} | {row['rotor']:7.4f} | {row['delta']:7.4f} | {row['gla']:7.4f} |"
          f"  {tag}")
print("      -> rotor accuracy OSCILLATES with length (phase error: chance at quadrature,")
print("         anti-correlated at half period, ~perfect again when the phase aliases);")
print("         delta accuracy decays MONOTONICALLY to chance (amplitude error).")
print("      snap-to-group repair: theta := pi and beta := 2 both give exact parity at")
print("      every length — on ABELIAN tasks both families have a snap target;")
print("      Part B is where the delta family has none.")
check("phase signature: rotor ~chance at quadrature, <0.1 anti-phase, >0.9 aliased",
      abs(table["quadrature"]["rotor"] - 0.5) < 0.15
      and table["anti-phase"]["rotor"] < 0.1 and table["aliased"]["rotor"] > 0.9)
check("amplitude signature: delta at chance at all three long lengths (monotone decay)",
      all(0.4 < table[t]["delta"] < 0.65 for t in ("quadrature", "anti-phase", "aliased")))
check("[0,1] gate is structural chance at long length", table["anti-phase"]["gla"] < 0.55)

# ============================================================ Part B ======

print()
print("Part B. S3 word problem, generators {(01), (012)} in SO(4): representability trains")

S3 = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]
IDX = {p: i for i, p in enumerate(S3)}
GEN = [(1, 0, 2), (1, 2, 0)]                      # (01), (012)
D = 4


def compose(g, h):                                 # (g o h)(x) = g(h(x))
    return tuple(g[h[i]] for i in range(3))


S0 = [1.0 / math.sqrt(2), 0.0, 0.0, 1.0 / math.sqrt(2)]


def rotor_forward(a, b):
    p, q, c = vdot(a, a), vdot(b, b), vdot(a, b)
    lam2 = max(p * q - c * c, 0.0)
    lam = math.sqrt(lam2)
    if lam < 1e-6:
        c1, c2 = 1.0 - lam2 / 6.0, 0.5 - lam2 / 24.0
    else:
        c1, c2 = math.sin(lam) / lam, (1.0 - math.cos(lam)) / lam2
    Z = msub(outer(a, b), outer(b, a))
    Z2 = matmul(Z, Z)
    R = [[(1.0 if i == j else 0.0) + c1 * Z[i][j] + c2 * Z2[i][j] for j in range(D)]
         for i in range(D)]
    return R, (a, b, p, q, c, lam, c1, c2, Z, Z2)


def rotor_backward(G, cache):
    a, b, p, q, c, lam, c1, c2, Z, Z2 = cache
    gc1 = sum(G[i][j] * Z[i][j] for i in range(D) for j in range(D))
    gc2 = sum(G[i][j] * Z2[i][j] for i in range(D) for j in range(D))
    Zt = transpose(Z)
    M = madd(mscale(c1, G), mscale(c2, madd(matmul(G, Zt), matmul(Zt, G))))
    Mmt = msub(M, transpose(M))
    da = matvec(Mmt, b)
    db = [-x for x in matvec(Mmt, a)]
    if lam < 1e-4:
        t1 = -1.0 / 3.0 + lam * lam / 30.0
        t2 = -1.0 / 12.0 + lam * lam / 180.0
    else:
        t1 = (lam * math.cos(lam) - math.sin(lam)) / lam ** 3
        t2 = (lam * math.sin(lam) - 2.0 * (1.0 - math.cos(lam))) / lam ** 4
    coef = gc1 * t1 + gc2 * t2
    da = [dai + coef * (q * ai - c * bi) for dai, ai, bi in zip(da, a, b)]
    db = [dbi + coef * (p * bi - c * ai) for dbi, ai, bi in zip(db, a, b)]
    return da, db


def delta_forward(k, w):
    beta = 2.0 / (1.0 + math.exp(-w))
    R = [[(1.0 if i == j else 0.0) - beta * k[i] * k[j] for j in range(D)] for i in range(D)]
    return R, (k, w, beta)


def delta_backward(G, cache):
    k, w, beta = cache
    Gk = matvec(G, k)
    Gtk = matvec(transpose(G), k)
    dk = [-beta * (x + y) for x, y in zip(Gk, Gtk)]
    s = beta / 2.0
    dw = -vdot(k, Gk) * 2.0 * s * (1.0 - s)
    return dk, dw


def run_model(kind, seed, stages=((4, 250), (12, 500)), batch=48, lr=0.05):
    rng = random.Random(seed)
    if kind == "rotor":
        params = {f"{n}{t}": [rng.gauss(0, 0.7) for _ in range(D)]
                  for t in range(2) for n in ("a", "b")}
    else:
        params = {f"k{t}": [rng.gauss(0, 0.7) for _ in range(D)] for t in range(2)}
        params.update({f"w{t}": [0.0] for t in range(2)})
    params["W"] = [rng.gauss(0, 0.3) for _ in range(6 * D)]
    params["bias"] = [0.0] * 6
    mom = {k: [0.0] * len(v) for k, v in params.items()}
    var = {k: [0.0] * len(v) for k, v in params.items()}

    def transitions():
        if kind == "rotor":
            return [rotor_forward(params[f"a{t}"], params[f"b{t}"]) for t in range(2)]
        return [delta_forward(params[f"k{t}"], params[f"w{t}"][0]) for t in range(2)]

    def forward_backward(seqs):
        Rs = transitions()
        grads = {k: [0.0] * len(v) for k, v in params.items()}
        Gacc = [[[0.0] * D for _ in range(D)] for _ in range(2)]
        L = 0.0
        for seq in seqs:
            states = [S0]
            for tok in seq:
                states.append(matvec(Rs[tok][0], states[-1]))
            ref = (0, 1, 2)
            for tok in seq:
                ref = compose(GEN[tok], ref)       # matches left-multiplied state
            label = IDX[ref]
            sT = states[-1]
            logits = [sum(params["W"][c * D + i] * sT[i] for i in range(D)) + params["bias"][c]
                      for c in range(6)]
            mx = max(logits)
            ex = [math.exp(z - mx) for z in logits]
            Zs = sum(ex)
            probs = [e / Zs for e in ex]
            L += -math.log(max(probs[label], 1e-300))
            dz = [probs[c] - (1.0 if c == label else 0.0) for c in range(6)]
            for c in range(6):
                for i in range(D):
                    grads["W"][c * D + i] += dz[c] * sT[i]
                grads["bias"][c] += dz[c]
            g = [sum(dz[c] * params["W"][c * D + i] for c in range(6)) for i in range(D)]
            for t in range(len(seq) - 1, -1, -1):
                tok = seq[t]
                s_prev = states[t]
                for i in range(D):
                    for j in range(D):
                        Gacc[tok][i][j] += g[i] * s_prev[j]
                g = matvec(transpose(Rs[tok][0]), g)
        for t in range(2):
            if kind == "rotor":
                da, db = rotor_backward(Gacc[t], Rs[t][1])
                grads[f"a{t}"] = [x + y for x, y in zip(grads[f"a{t}"], da)]
                grads[f"b{t}"] = [x + y for x, y in zip(grads[f"b{t}"], db)]
            else:
                dk, dw = delta_backward(Gacc[t], Rs[t][1])
                grads[f"k{t}"] = [x + y for x, y in zip(grads[f"k{t}"], dk)]
                grads[f"w{t}"][0] += dw
        n = float(len(seqs))
        return L / n, {k: [x / n for x in v] for k, v in grads.items()}

    step = 0
    for T, n_steps in stages:                       # identical curriculum for all models
        for _ in range(n_steps):
            step += 1
            seqs = [[rng.randrange(2) for _ in range(T)] for _ in range(batch)]
            _, grads = forward_backward(seqs)
            b1, b2, eps = 0.9, 0.999, 1e-8
            for k in params:
                for i in range(len(params[k])):
                    mom[k][i] = b1 * mom[k][i] + (1 - b1) * grads[k][i]
                    var[k][i] = b2 * var[k][i] + (1 - b2) * grads[k][i] ** 2
                    mhat = mom[k][i] / (1 - b1 ** step)
                    vhat = var[k][i] / (1 - b2 ** step)
                    params[k][i] -= lr * mhat / (math.sqrt(vhat) + eps)

    def acc(T_eval, n_eval=512):
        Rs = transitions()
        good = 0
        erng = random.Random(seed + 777)
        for _ in range(n_eval):
            seq = [erng.randrange(2) for _ in range(T_eval)]
            s = S0
            for tok in seq:
                s = matvec(Rs[tok][0], s)
            ref = (0, 1, 2)
            for tok in seq:
                ref = compose(GEN[tok], ref)
            logits = [sum(params["W"][c * D + i] * s[i] for i in range(D)) + params["bias"][c]
                      for c in range(6)]
            good += (max(range(6), key=lambda c: logits[c]) == IDX[ref])
        return good / n_eval

    # Basis-free representation check. A solving network need not match any
    # canonical embedding — and need not even be a linear representation of
    # S3: a PROJECTIVE one (double cover) also computes the word problem,
    # since a central sign K commutes past everything and the readout can
    # quotient it out. So test the S3 relations MODULO a central element
    # K := R1^3 (for a genuine linear rep, K = I):
    #     K^2 = I,  K central,  R0^2 in {I, K},  (R0 R1)^2 in {I, K}.
    Rs = transitions()
    R0, R1 = Rs[0][0], Rs[1][0]
    I4 = [[1.0 if i == j else 0.0 for j in range(D)] for i in range(D)]

    def dist(A, B):
        return math.sqrt(sum((A[i][j] - B[i][j]) ** 2 for i in range(D) for j in range(D)))

    K = matmul(R1, matmul(R1, R1))
    rel = max(dist(matmul(K, K), I4),
              dist(matmul(K, R0), matmul(R0, K)),
              dist(matmul(K, R1), matmul(R1, K)),
              min(dist(matmul(R0, R0), I4), dist(matmul(R0, R0), K)),
              min(dist(matmul(matmul(R0, R1), matmul(R0, R1)), I4),
                  dist(matmul(matmul(R0, R1), matmul(R0, R1)), K)))
    kdist = dist(K, I4)
    return acc(12), acc(96), rel, kdist, params, forward_backward


def gradcheck():
    _, _, _, _, params, fb = run_model("rotor", seed=1, stages=((3, 1),), batch=2)
    rng = random.Random(5)
    seqs = [[rng.randrange(2) for _ in range(3)] for _ in range(2)]
    _, grads = fb(seqs)
    worst = 0.0
    h = 1e-5
    for key in ("a0", "b0", "a1", "b1", "W", "bias"):
        for i in range(len(params[key])):
            keep = params[key][i]
            params[key][i] = keep + h
            Lp, _ = fb(seqs)
            params[key][i] = keep - h
            Lm, _ = fb(seqs)
            params[key][i] = keep
            fd = (Lp - Lm) / (2 * h)
            an = grads[key][i]
            worst = max(worst, abs(fd - an) / max(1e-6, abs(fd) + abs(an)))
    return worst


gc = gradcheck()
check("hand-derived rotor backprop matches finite differences", gc < 1e-4, f"max rel err {gc:.1e}")

print("    curriculum T=4 then T=12, batch 48, Adam 0.05, 5 seeds each; chance = 1/6 = 0.167;")
print("    'rel' = S3 relations modulo a central sign K = R1^3 (0 for a projective rep);")
print("    'K-I' = distance of K from I (large + rel small => a DOUBLE COVER was learned):")
results = {"rotor": [], "delta": []}
for kind in ("rotor", "delta"):
    for seed in (11, 22, 33, 44, 55):
        a12, a96, rel, kdist, _, _ = run_model(kind, seed)
        results[kind].append((a12, a96, rel, kdist))
        print(f"      {kind:5s} seed {seed}:  acc@12 = {a12:.3f}   acc@96 = {a96:.3f}"
              f"   rel = {rel:.3f}   K-I = {kdist:.3f}")

solved = [r for r in results["rotor"] if r[0] >= 0.98 and r[1] >= 0.98]
best_delta = max(r[1] for r in results["delta"])
print(f"    rotor solve rate: {len(solved)}/5 (multi-basin angle landscapes — the Part A")
print(f"    trap in high dimension; curriculum helps but does not eliminate it).")
check("rotor learns the S3 word problem end to end and holds it at 8x length",
      len(solved) >= 1, f"{len(solved)}/5 seeds at >= 0.98 both lengths")
check("solving runs are (projective) S3 representations: relations-mod-center hold",
      all(r[2] < 0.2 for r in solved) and len(solved) >= 1,
      "some are double covers: K far from I with rel ~ 0" if any(r[3] > 1.0 for r in solved) else "")
check("single-step delta baseline stays near chance (no proper rotation to snap to)",
      best_delta <= 0.5, f"best delta acc@96 = {best_delta:.3f}")

print()
print("Summary (calibrated by the runs): the rotor gate's advantage is NOT an")
print("optimisation-rate win — on abelian parity both families converge at")
print("k^(-1/2) (A2), and the rotor's angle landscape has spurious minima that")
print("need a curriculum (A1); its failures are phase-shaped, the boundary")
print("family's amplitude-shaped. The advantage is structural: where hard state")
print("tracking needs proper rotations, the delta family has nothing to reach OR")
print("snap to, while the rotor family trains to exact group structure from")
print("sequence labels alone — sometimes via a double cover (B).")
print()
print("ALL PASS" if all(PASSES) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(PASSES) else 1)
