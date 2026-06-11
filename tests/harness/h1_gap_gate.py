"""h1_gap_gate — the H1 D-gate scoring core (arm (b) vs arm (c), the make-or-break math).

Pins `benchmarks/h1_gap_gate.py` (`horizon-1-capture.md` §D). The scorer trains
arm (c) leakage-free (out-of-fold) and applies the frozen PASS/KILL rule:
PASS = ΔF1 ≥ 0.10 AND exact-McNemar p<0.05; else KILL.

Contracts:

  (a) PASS — when the learned arm genuinely beats a mediocre closed-form arm on a
      separable signal (ΔF1 ≥ 0.10 and McNemar significant).
  (b) KILL — when the closed-form arm is already perfect, the learned arm can only
      tie or lose ⇒ ΔF1 < 0.10 ⇒ KILL. And the dangerous case: when arm (c) loses
      *significantly* (McNemar p<0.05 in the wrong direction), the ΔF1 floor still
      binds — a significant loss is never a PASS.
  (c) McNemar — exact binomial on per-question correctness: discordant counts and
      the two-sided p are correct (10-vs-2 significant; 6-vs-6 not).
  (d) Leakage-free CV — arm (c) probabilities are out-of-fold, in [0,1], aligned.
  (e) Deterministic — a fixed seed reproduces the whole GateResult.

Hermetic — seeded numpy/sklearn only, no disk, no network.
"""

from __future__ import annotations

import math
import sys

import numpy as np

# benchmarks/h1_gap_gate.py needs scikit-learn ([bench] extra) — on lean
# installs run() reports an honest SKIP instead of crashing the sweep.
try:
    from benchmarks.h1_gap_gate import (
        arm_c_oof,
        extract_features,
        score_gate,
        to_matrices,
        _mcnemar_exact,
    )
    _SKIP_REASON = None
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] != "sklearn":
        raise
    _SKIP_REASON = "scikit-learn not installed (pip install rlat[bench])"

_TS = "2026-06-01T12:00:{:02d}+00:00"


def _row(emb, top, *, session="s", ts=0):
    return {
        "ts": _TS.format(ts % 60),
        "session": session,
        "layer": "source",
        "is_user_query": True,
        "query_emb": [float(x) for x in emb],
        "ranked": [{"rank": 0, "idx": 0, "score": float(top)}],
    }


def _make_stream(rng, specs):
    """specs: list of (qid, centre, top_score, n_paraphrases, label). Each
    question's paraphrases sit tight around its (orthogonal) centre so they
    cluster together and separate across questions; spread over 2 sessions."""
    rows, rowqids, labels = [], [], {}
    ts = 0
    for qid, centre, top, k, lab in specs:
        labels[qid] = lab
        for j in range(k):
            emb = np.array(centre, dtype="float64") + rng.normal(0, 0.02, len(centre))
            rows.append(_row(emb, top, session="s%d" % (j % 2), ts=ts))
            rowqids.append(qid)
            ts += 1
    return rows, rowqids, labels


def _make_separable(n_pos, n_neg, seed):
    """A separable gap-detection toy: gaps carry a low signal, non-gaps a high one,
    plus a pure-noise feature. A logistic arm (c) recovers the signal; a corrupted
    arm (b) does not."""
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    labels = np.array([True] * n_pos + [False] * n_neg)
    sig = np.concatenate([rng.normal(-2.5, 1.0, n_pos), rng.normal(2.5, 1.0, n_neg)])
    noise = rng.normal(0, 1, n)
    X = np.column_stack([sig, noise])
    perm = rng.permutation(n)  # shuffle so folds aren't class-ordered
    return X[perm], labels[perm]


def _check_pass() -> int:
    X, labels = _make_separable(40, 60, seed=1)
    b = labels.copy()
    b[::3] = ~b[::3]  # deterministically corrupt ~1/3 → a mediocre arm (b)
    res = score_gate(X, b, labels, seed=1, n_boot=500)
    if not (res.f1_c > res.f1_b and res.delta_f1 >= 0.10 and res.mcnemar_p < 0.05):
        print(f"[h1_gap_gate] pass: learned arm did not clear the gate: {res}",
              file=sys.stderr)
        return 1
    if res.verdict != "PASS":
        print(f"[h1_gap_gate] pass: verdict not PASS: {res}", file=sys.stderr)
        return 1
    # The c-wins discordant count should dominate (arm (b)'s errors arm (c) fixes).
    if res.mcnemar_b01 <= res.mcnemar_b10:
        print(f"[h1_gap_gate] pass: c-wins not dominant: b01={res.mcnemar_b01} "
              f"b10={res.mcnemar_b10}", file=sys.stderr)
        return 1
    return 0


def _check_kill() -> int:
    X, labels = _make_separable(40, 60, seed=2)
    b = labels.copy()  # arm (b) already perfect
    res = score_gate(X, b, labels, seed=2, n_boot=500)
    if res.f1_b != 1.0:
        print(f"[h1_gap_gate] kill: arm (b) not perfect as set up: {res.f1_b}",
              file=sys.stderr)
        return 1
    # The learned arm cannot beat a perfect baseline → ΔF1 ≤ 0 → KILL, whatever
    # McNemar says (the floor is the binding criterion).
    if res.delta_f1 > 0 or res.verdict != "KILL":
        print(f"[h1_gap_gate] kill: a perfect baseline was not a KILL: {res}",
              file=sys.stderr)
        return 1
    return 0


def _check_kill_when_c_loses_significantly() -> int:
    # The dangerous case the floor must override: arm (c) gets PURE NOISE features
    # (cannot learn) while arm (b) is a decent predictor, so c loses AND McNemar is
    # significant in the WRONG direction. The verdict must still be KILL — the ΔF1
    # floor binds first, McNemar never flips a loss into a PASS.
    rng = np.random.default_rng(7)
    n_pos, n_neg = 45, 55
    n = n_pos + n_neg
    labels = np.array([True] * n_pos + [False] * n_neg)
    X = rng.normal(0, 1, size=(n, 2))  # noise only — no signal for c to learn
    b = labels.copy()
    b[::8] = ~b[::8]                    # arm (b) decent (~88% correct)
    perm = rng.permutation(n)
    X, labels, b = X[perm], labels[perm], b[perm]
    res = score_gate(X, b, labels, seed=7, n_boot=500)
    if res.delta_f1 >= 0:
        print(f"[h1_gap_gate] kill-sig: noise-c did not lose: ΔF1={res.delta_f1}",
              file=sys.stderr)
        return 1
    if res.mcnemar_p >= 0.05:
        print(f"[h1_gap_gate] kill-sig: McNemar not significant (wrong dir): {res}",
              file=sys.stderr)
        return 1
    if res.verdict != "KILL":
        print(f"[h1_gap_gate] kill-sig: a significant LOSS was not KILL — the floor "
              f"failed to bind: {res}", file=sys.stderr)
        return 1
    return 0


def _check_mcnemar() -> int:
    b_correct = np.array([False] * 10 + [True] * 2 + [True] * 8)
    c_correct = np.array([True] * 10 + [False] * 2 + [True] * 8)
    b01, b10, p = _mcnemar_exact(b_correct, c_correct)
    if b01 != 10 or b10 != 2 or p >= 0.05:
        print(f"[h1_gap_gate] mcnemar: 10-vs-2 wrong: b01={b01} b10={b10} p={p}",
              file=sys.stderr)
        return 1
    b01, b10, p = _mcnemar_exact(
        np.array([False] * 6 + [True] * 6), np.array([True] * 6 + [False] * 6))
    if b01 != 6 or b10 != 6 or p <= 0.5:
        print(f"[h1_gap_gate] mcnemar: 6-vs-6 should be n.s.: b01={b01} b10={b10} "
              f"p={p}", file=sys.stderr)
        return 1
    # No disagreement → p = 1.0, no crash.
    _, _, p = _mcnemar_exact(np.array([True, True]), np.array([True, True]))
    if p != 1.0:
        print(f"[h1_gap_gate] mcnemar: no-discordance p != 1.0: {p}", file=sys.stderr)
        return 1
    return 0


def _check_oof() -> int:
    for n_pos, n_neg, sd in [(30, 30, 5), (54, 77, 11)]:  # balanced + real 54/131
        X, labels = _make_separable(n_pos, n_neg, seed=sd)
        p = arm_c_oof(X, labels.astype(int), folds=5, seed=sd)
        if len(p) != n_pos + n_neg or p.min() < 0.0 or p.max() > 1.0:
            print(f"[h1_gap_gate] oof: probabilities malformed at {n_pos}/{n_neg}: "
                  f"len={len(p)} min={p.min()} max={p.max()}", file=sys.stderr)
            return 1
    return 0


def _check_deterministic() -> int:
    X, labels = _make_separable(40, 60, seed=3)
    b = labels.copy()
    b[::4] = ~b[::4]
    r1 = score_gate(X, b, labels, seed=3, n_boot=300)
    r2 = score_gate(X, b, labels, seed=3, n_boot=300)
    if r1 != r2:
        print(f"[h1_gap_gate] deterministic: two passes differ:\n{r1}\n{r2}",
              file=sys.stderr)
        return 1
    return 0


def _check_extract_endtoend() -> int:
    # 3 gap questions (weak, recurring) + 3 non-gap (strong), each 4 paraphrases on
    # an orthogonal centre. arm (b) should flag exactly the gaps, and the whole
    # pipeline must run into the scorer.
    rng = np.random.default_rng(20)
    eye = np.eye(6)
    specs = [(f"gap{i}", eye[i], 0.10, 4, True) for i in range(3)] \
        + [(f"ok{i}", eye[3 + i], 0.80, 4, False) for i in range(3)]
    rows, rowqids, labels = _make_stream(rng, specs)
    qfeats, names = extract_features(rows, rowqids, labels)
    if len(qfeats) != 6 or len(names) != 8:
        print(f"[h1_gap_gate] extract: shape wrong: {len(qfeats)} qs, {len(names)} "
              f"features", file=sys.stderr)
        return 1
    by_id = {qf.question_id: qf for qf in qfeats}
    for i in range(3):
        g = by_id[f"gap{i}"]
        # min ≈ mean ≈ 0.10 (all paraphrases weak), maha finite, 4 weak in cluster.
        if not (g.arm_b and g.label) or g.x[0] > 0.2 or abs(g.x[1] - 0.10) > 0.01 \
                or not math.isfinite(g.x[2]) or g.x[3] != 4.0:
            print(f"[h1_gap_gate] extract: gap question mis-aggregated: {g}",
                  file=sys.stderr)
            return 1
        ok = by_id[f"ok{i}"]
        if ok.arm_b or ok.label or ok.x[0] < 0.7 or abs(ok.x[1] - 0.80) > 0.01 \
                or ok.x[3] != 0.0:
            print(f"[h1_gap_gate] extract: non-gap question mis-aggregated: {ok}",
                  file=sys.stderr)
            return 1
    X, arm_b, lab = to_matrices(qfeats)
    if X.shape != (6, 8) or arm_b.shape != (6,) or lab.shape != (6,):
        print(f"[h1_gap_gate] extract: matrix shapes wrong: {X.shape} {arm_b.shape} "
              f"{lab.shape}", file=sys.stderr)
        return 1
    # arm (b) flags exactly the gaps here → F1_b perfect → end-to-end KILL. (folds=3
    # for this 3-per-class toy; the real gate uses the default 5 on 54/77.)
    res = score_gate(X, arm_b, lab, seed=20, n_boot=200, folds=3)
    if res.f1_b != 1.0 or res.verdict != "KILL":
        print(f"[h1_gap_gate] extract: perfect arm (b) not a clean KILL: {res}",
              file=sys.stderr)
        return 1
    return 0


def _check_extract_alignment() -> int:
    # Interleave non-user-source rows (internal + insight) carrying labelless qids.
    # row_question_ids spans the RAW stream; the extractor must filter with the
    # clause predicate so features land on the right question — and the junk qids
    # (which have NO label) must never be looked up (no KeyError) or mis-attributed.
    rng = np.random.default_rng(22)
    e = np.eye(3)

    def _u(centre, top, sess, ts):
        return _row(np.array(centre, "float64") + rng.normal(0, 0.02, 3), top,
                    session=sess, ts=ts)

    internal = _row([1, 0, 0], 0.1)
    internal["is_user_query"] = False
    insight = _row([0, 1, 0], 0.1)
    insight["layer"] = "insight"
    raw = [internal, _u(e[0], 0.1, "s0", 1), insight,
           _u(e[0], 0.1, "s1", 2), _u(e[1], 0.8, "s0", 3), _u(e[1], 0.8, "s1", 4)]
    rowqids = ["INTERNAL", "g", "INSIGHT", "g", "k", "k"]
    labels = {"g": True, "k": False}  # the junk qids deliberately have NO label
    qfeats, _ = extract_features(raw, rowqids, labels)
    by = {qf.question_id: qf for qf in qfeats}
    if set(by) != {"g", "k"}:
        print(f"[h1_gap_gate] align: wrong question set (junk leaked?): {set(by)}",
              file=sys.stderr)
        return 1
    if not (by["g"].arm_b and by["g"].label) or by["g"].x[3] != 2.0:
        print(f"[h1_gap_gate] align: gap question mis-aggregated under junk: "
              f"{by['g']}", file=sys.stderr)
        return 1
    if by["k"].arm_b or by["k"].label:
        print(f"[h1_gap_gate] align: non-gap flagged under junk: {by['k']}",
              file=sys.stderr)
        return 1
    return 0


def _check_extract_length_guard() -> int:
    # row_question_ids must align to the RAW stream — a length mismatch raises
    # rather than silently truncating into the gate.
    rng = np.random.default_rng(23)
    rows, rowqids, labels = _make_stream(rng, [("g", np.eye(3)[0], 0.1, 2, True)])
    try:
        extract_features(rows, rowqids[:-1], labels)
    except ValueError:
        return 0
    print("[h1_gap_gate] length-guard: mismatched row_question_ids did not raise",
          file=sys.stderr)
    return 1


def _check_extract_clip_and_single() -> int:
    # A lone single-paraphrase question: no same-session followup → gap_seconds inf
    # → clipped to 3600; cluster size 1 → reproduced 1 → arm (b) does NOT flag it,
    # even though it is weak.
    rng = np.random.default_rng(21)
    rows, rowqids, labels = _make_stream(rng, [("lone", np.eye(4)[0], 0.10, 1, True)])
    qfeats, names = extract_features(rows, rowqids, labels)
    qf = qfeats[0]
    gap_col = names.index("min_gap_seconds")
    if qf.x[gap_col] != 3600.0:
        print(f"[h1_gap_gate] clip: inf re-ask gap not clipped: {qf.x[gap_col]}",
              file=sys.stderr)
        return 1
    if qf.arm_b or qf.x[3] != 1.0:
        print(f"[h1_gap_gate] clip: lone weak query flagged or mis-counted: {qf}",
              file=sys.stderr)
        return 1
    return 0


def run() -> int:
    if _SKIP_REASON:
        print(f"[h1_gap_gate] SKIP — {_SKIP_REASON}", file=sys.stderr)
        return 2  # harness SKIP sentinel — runner reports as skipped, not passed
    for check in [
        _check_pass,
        _check_kill,
        _check_kill_when_c_loses_significantly,
        _check_mcnemar,
        _check_oof,
        _check_deterministic,
        _check_extract_endtoend,
        _check_extract_alignment,
        _check_extract_length_guard,
        _check_extract_clip_and_single,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[h1_gap_gate] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
