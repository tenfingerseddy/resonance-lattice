"""curator_gap — the arm (b) closed-form gap-candidate combiner.

Pins the C3-gap conjunction (`.claude/plans/insight-engine/the-curator-head.md` §6
"Build decision"; `horizon-1-capture.md` arm (b)). The combiner turns the
score-gap and intent-cluster *features* into a per-query *decision*:

    candidate := weak score-gap (top < 0.30) AND reproducible (≥2 weak queries in
                 the cluster) AND lexical does not veto.

Contracts:

  (a) Empty stream → empty.
  (b) Two-clause candidate — a cluster of ≥2 weak paraphrases is a candidate; a
      well-served query is not weak; a lone weak query is weak-but-not-reproduced.
  (c) Reproducibility is over the WEAK members — one weak query among strong
      cluster-mates is not reproduced (reproduced == 1, not a candidate).
  (d) Lexical veto — an injected falsy (lexical found it) vetoes a candidate; a
      truthy (lexical also failed) keeps it; None / past-list-end passes. The veto
      is truthiness-based (a numpy False vetoes); a non-list signal is ignored.
  (e) Cross-session reproducibility — two weak paraphrases in different sessions
      still reproduce (the weak count is session-agnostic).
  (f) Filter + alignment — internal / insight rows are dropped; decisions align to
      the user-source rows in stream order.
  (g) Degenerate — ragged embeddings → all-singleton clusters → reproduced 1 → no
      candidate, no crash.
  (h) Deterministic — no clock; the same stream reproduces.

Recurring-gap queue (C4) contracts:

  (i) Dedup — N candidate occurrences of one gap → ONE queue entry (occurrences N);
      separate gap intents get separate entries; served queries get none.
  (j) Recurrence gates — min_occurrences and min_sessions exclude under-recurring
      gaps; min_sessions=2 isolates the cross-session gap.
  (k) Empty / veto — empty stream, all-served, or all-vetoed → empty queue.
  (l) Deterministic — entries in cluster-id order; the same stream reproduces.

Hermetic — no model load, no disk; seeded numpy noise only.
"""

from __future__ import annotations

import sys

import numpy as np

from resonance_lattice.curator import gap

_TS = "2026-06-01T12:00:00+00:00"


def _row(emb, top, *, layer="source", is_user=True, session="s"):
    return {
        "ts": _TS,
        "session": session,
        "layer": layer,
        "is_user_query": is_user,
        "query_emb": [float(x) for x in emb],
        "ranked": [{"rank": 0, "idx": 0, "score": float(top)}],
    }


def _near(rng, centre, n, top, *, session="s"):
    """n rows tight around `centre` (so they cluster) each with top score `top`."""
    c = np.array(centre, dtype="float64")
    return [_row(c + rng.normal(0, 0.02, size=len(centre)), top, session=session)
            for _ in range(n)]


def _check_empty() -> int:
    if gap.gap_candidates([]) != []:
        print("[curator_gap] empty: non-empty for empty stream", file=sys.stderr)
        return 1
    return 0


def _check_two_clause_candidate() -> int:
    rng = np.random.default_rng(20)
    weak = _near(rng, [1, 0, 0, 0], 2, 0.10)    # two weak paraphrases → a gap
    served = _near(rng, [0, 1, 0, 0], 1, 0.80)  # well-served, different intent
    lone = _near(rng, [0, 0, 1, 0], 1, 0.10)    # a lone weak query, third intent
    decs = gap.gap_candidates(weak + served + lone)
    if len(decs) != 4:
        print(f"[curator_gap] two-clause: expected 4, got {len(decs)}",
              file=sys.stderr)
        return 1
    if not (decs[0].is_candidate and decs[1].is_candidate
            and decs[0].weak and decs[0].reproduced == 2):
        print(f"[curator_gap] two-clause: reproduced weak gap not flagged: "
              f"{decs[0]}, {decs[1]}", file=sys.stderr)
        return 1
    if decs[2].is_candidate or decs[2].weak:
        print(f"[curator_gap] two-clause: served query flagged weak/candidate: "
              f"{decs[2]}", file=sys.stderr)
        return 1
    if decs[3].is_candidate or not decs[3].weak or decs[3].reproduced != 1:
        print(f"[curator_gap] two-clause: lone weak query mishandled: {decs[3]}",
              file=sys.stderr)
        return 1
    return 0


def _check_reproducibility_over_weak() -> int:
    # One intent: 1 weak + 3 strong. The weak one is not reproduced (only 1 weak
    # member), so not a candidate — guards calcifying around a single near-miss.
    rng = np.random.default_rng(21)
    rows = _near(rng, [1, 0, 0, 0], 1, 0.10) + _near(rng, [1, 0, 0, 0], 3, 0.80)
    decs = gap.gap_candidates(rows)
    if not decs[0].weak or decs[0].reproduced != 1 or decs[0].is_candidate:
        print(f"[curator_gap] reproduce-weak: lone weak among strong was a "
              f"candidate: {decs[0]}", file=sys.stderr)
        return 1
    if any(d.weak for d in decs[1:]):
        print(f"[curator_gap] reproduce-weak: a strong cluster-mate read weak: "
              f"{[d.weak for d in decs]}", file=sys.stderr)
        return 1
    return 0


def _check_lexical_veto() -> int:
    rng = np.random.default_rng(22)
    rows = _near(rng, [1, 0, 0, 0], 2, 0.10)  # a reproduced weak gap
    base = gap.gap_candidates(rows)
    if not (base[0].is_candidate and base[1].is_candidate
            and base[0].lexical_miss is None):
        print(f"[curator_gap] lexical: two-clause base not a candidate: {base}",
              file=sys.stderr)
        return 1
    vetoed = gap.gap_candidates(rows, lexical_miss=[False, False])
    if vetoed[0].is_candidate or vetoed[1].is_candidate \
            or vetoed[0].lexical_miss is not False:
        print(f"[curator_gap] lexical: False (found it) did not veto: {vetoed}",
              file=sys.stderr)
        return 1
    confirmed = gap.gap_candidates(rows, lexical_miss=[True, True])
    if not confirmed[0].is_candidate or confirmed[0].lexical_miss is not True:
        print(f"[curator_gap] lexical: True (also failed) dropped the candidate: "
              f"{confirmed[0]}", file=sys.stderr)
        return 1
    # A short list degrades to None past its end (passes, two-clause).
    short = gap.gap_candidates(rows, lexical_miss=[False])
    if short[0].is_candidate or not short[1].is_candidate \
            or short[1].lexical_miss is not None:
        print(f"[curator_gap] lexical: short list not None-padded: {short}",
              file=sys.stderr)
        return 1
    return 0


def _check_lexical_veto_robust() -> int:
    rng = np.random.default_rng(25)
    rows = _near(rng, [1, 0, 0, 0], 2, 0.10)
    # numpy bools (a caller doing list(np_mask)) must veto like Python False — the
    # veto is truthiness-based, not identity.
    decs = gap.gap_candidates(rows, lexical_miss=list(np.array([False, False])))
    if any(d.is_candidate for d in decs):
        print(f"[curator_gap] lexical-robust: numpy False did not veto: {decs}",
              file=sys.stderr)
        return 1
    # A non-list lexical_miss is ignored (degrades to two-clause), never raises.
    for bad in ({0: False}, "nope", 5):
        decs = gap.gap_candidates(rows, lexical_miss=bad)
        if not all(d.is_candidate for d in decs) \
                or any(d.lexical_miss is not None for d in decs):
            print(f"[curator_gap] lexical-robust: non-list not ignored: {bad!r} -> "
                  f"{decs}", file=sys.stderr)
            return 1
    return 0


def _check_cross_session_reproduced() -> int:
    # Two weak paraphrases in DIFFERENT sessions still reproduce — the weak count
    # is session-agnostic, so the cluster has 2 weak members → a candidate.
    rng = np.random.default_rng(26)
    rows = (_near(rng, [1, 0, 0, 0], 1, 0.10, session="s1")
            + _near(rng, [1, 0, 0, 0], 1, 0.10, session="s2"))
    decs = gap.gap_candidates(rows)
    if not all(d.is_candidate and d.reproduced == 2 for d in decs):
        print(f"[curator_gap] cross-session: weak pair across sessions not "
              f"reproduced: {decs}", file=sys.stderr)
        return 1
    if any(d.distinct_sessions != 2 for d in decs):
        print(f"[curator_gap] cross-session: distinct_sessions not 2: "
              f"{[d.distinct_sessions for d in decs]}", file=sys.stderr)
        return 1
    return 0


def _check_filter_and_align() -> int:
    rng = np.random.default_rng(23)
    weak = _near(rng, [1, 0, 0, 0], 2, 0.10)
    rows = [
        weak[0],
        _row([1, 0, 0, 0], 0.10, is_user=False),    # machinery — ignored
        weak[1],
        _row([1, 0, 0, 0], 0.10, layer="insight"),  # insight band — not the unit
    ]
    decs = gap.gap_candidates(rows)
    if len(decs) != 2 or not (decs[0].is_candidate and decs[1].is_candidate):
        print(f"[curator_gap] filter: non-user/insight rows leaked or misaligned: "
              f"{decs}", file=sys.stderr)
        return 1
    return 0


def _check_degenerate() -> int:
    # Ragged embeddings → intent clusters are all singletons → reproduced 1 → no
    # candidate even though weak; must not crash.
    a = _row([1, 0, 0], 0.10)
    braw = {"ts": _TS, "session": "s", "layer": "source", "is_user_query": True,
            "query_emb": [1.0, 0.0], "ranked": [{"rank": 0, "idx": 0, "score": 0.10}]}
    decs = gap.gap_candidates([a, braw])
    if len(decs) != 2 or any(not d.weak for d in decs) \
            or any(d.is_candidate for d in decs) or any(d.reproduced != 1 for d in decs):
        print(f"[curator_gap] degenerate: ragged stream not all weak-singletons: "
              f"{decs}", file=sys.stderr)
        return 1
    return 0


def _check_deterministic() -> int:
    rng = np.random.default_rng(24)
    rows = _near(rng, [1, 0, 0, 0], 2, 0.10) + _near(rng, [0, 1, 0, 0], 2, 0.10)
    a = gap.gap_candidates(rows)
    b = gap.gap_candidates(rows)
    key = lambda ds: [(d.is_candidate, d.weak, d.reproduced, d.cluster_id) for d in ds]
    if key(a) != key(b):
        print("[curator_gap] deterministic: two passes differ", file=sys.stderr)
        return 1
    return 0


def _check_queue_dedup() -> int:
    # Gap A: 3 weak paraphrases (one intent). Gap B: 2 weak. Served: 1. The queue
    # dedups each gap to ONE entry — the runaway-compute guard.
    rng = np.random.default_rng(30)
    rows = (_near(rng, [1, 0, 0, 0], 3, 0.10)
            + _near(rng, [0, 1, 0, 0], 2, 0.10)
            + _near(rng, [0, 0, 1, 0], 1, 0.80))
    q = gap.recurring_gaps(rows)
    if len(q) != 2:
        print(f"[curator_gap] queue-dedup: expected 2 entries, got {len(q)}: {q}",
              file=sys.stderr)
        return 1
    if q[0].cluster_id != 0 or q[0].occurrences != 3 \
            or q[1].cluster_id != 1 or q[1].occurrences != 2:
        print(f"[curator_gap] queue-dedup: wrong dedup/occurrences: {q}",
              file=sys.stderr)
        return 1
    return 0


def _check_queue_recurrence_gates() -> int:
    rng = np.random.default_rng(31)
    single = _near(rng, [1, 0, 0, 0], 2, 0.10, session="s")        # 1 session
    cross = (_near(rng, [0, 1, 0, 0], 1, 0.10, session="s1")
             + _near(rng, [0, 1, 0, 0], 1, 0.10, session="s2"))    # 2 sessions
    rows = single + cross
    if len(gap.recurring_gaps(rows)) != 2:
        print("[curator_gap] queue-gates: default min_sessions=1 dropped a gap",
              file=sys.stderr)
        return 1
    strict = gap.recurring_gaps(rows, min_sessions=2)
    if len(strict) != 1 or strict[0].cluster_id != 1 \
            or strict[0].distinct_sessions != 2:
        print(f"[curator_gap] queue-gates: min_sessions=2 did not isolate the "
              f"cross-session gap: {strict}", file=sys.stderr)
        return 1
    # min_occurrences gates on the candidate count.
    rng2 = np.random.default_rng(32)
    rows2 = _near(rng2, [1, 0, 0, 0], 1, 0.10) + _near(rng2, [0, 1, 0, 0], 3, 0.10)
    q = gap.recurring_gaps(rows2)
    if len(q) != 1 or q[0].cluster_id != 1 or q[0].occurrences != 3:
        print(f"[curator_gap] queue-gates: lone weak query queued, or wrong count: "
              f"{q}", file=sys.stderr)
        return 1
    if gap.recurring_gaps(rows2, min_occurrences=4) != []:
        print("[curator_gap] queue-gates: min_occurrences=4 did not exclude the "
              "3-occurrence gap", file=sys.stderr)
        return 1
    return 0


def _check_queue_empty_and_veto() -> int:
    if gap.recurring_gaps([]) != []:
        print("[curator_gap] queue-empty: non-empty for empty stream",
              file=sys.stderr)
        return 1
    rng = np.random.default_rng(33)
    weak = _near(rng, [1, 0, 0, 0], 2, 0.10)
    if gap.recurring_gaps(weak, lexical_miss=[False, False]) != []:
        print("[curator_gap] queue-veto: lexically-vetoed candidates were queued",
              file=sys.stderr)
        return 1
    served = _near(rng, [1, 0, 0, 0], 3, 0.80)
    if gap.recurring_gaps(served) != []:
        print("[curator_gap] queue-veto: an all-served stream produced a queue",
              file=sys.stderr)
        return 1
    # A PARTIAL veto must not inflate occurrences: 3 weak, the middle one vetoed →
    # the entry counts the 2 surviving candidates, not all 3 weak members.
    rng2 = np.random.default_rng(35)
    trio = _near(rng2, [1, 0, 0, 0], 3, 0.10)
    q = gap.recurring_gaps(trio, lexical_miss=[True, False, True])
    if len(q) != 1 or q[0].occurrences != 2:
        print(f"[curator_gap] queue-veto: partial veto inflated occurrences: {q}",
              file=sys.stderr)
        return 1
    return 0


def _check_queue_deterministic() -> int:
    rng = np.random.default_rng(34)
    rows = _near(rng, [1, 0, 0, 0], 2, 0.10) + _near(rng, [0, 1, 0, 0], 2, 0.10)
    a = gap.recurring_gaps(rows)
    b = gap.recurring_gaps(rows)
    key = lambda es: [(e.cluster_id, e.occurrences, e.distinct_sessions) for e in es]
    if key(a) != key(b):
        print("[curator_gap] queue-deterministic: two passes differ",
              file=sys.stderr)
        return 1
    return 0


def run() -> int:
    for check in [
        _check_empty,
        _check_two_clause_candidate,
        _check_reproducibility_over_weak,
        _check_lexical_veto,
        _check_lexical_veto_robust,
        _check_cross_session_reproduced,
        _check_filter_and_align,
        _check_degenerate,
        _check_deterministic,
        _check_queue_dedup,
        _check_queue_recurrence_gates,
        _check_queue_empty_and_veto,
        _check_queue_deterministic,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[curator_gap] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
