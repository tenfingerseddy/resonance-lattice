"""curator_signals — the closed-form clauses (arm (b)): score-gap + intent.

Pins the C3a/C3b contracts (`.claude/plans/insight-engine/the-curator-head.md` §6
first clause; `capture.md` §4 intent; `horizon-1-capture.md` arm (b)). Each clause
is a pure feature extractor over the same user-source rows; neither bakes in a
gap decision of its own.

Score-gap contracts:

  (a) Empty stream → empty; only user-intent source rows count (internal /
      insight rows are ignored, fingerprints required).
  (b) Separation — a weakly-matched query has the lowest top_score; an
      out-of-distribution query (far embedding) has the highest maha; a
      well-served query has a high top_score and a low maha.
  (c) Degenerate inputs — an all-identical stream yields maha 0 (no spurious
      signal, no crash); a ragged embedding falls back to score-only features.
  (d) Deterministic / pure — no clock; the same stream reproduces.
  (e) The clause runs on a REAL stream — field.retrieve → capture → drain →
      score_gap_features.

Intent-cluster contracts:

  (f) Empty/filter — same row-selection unit as score-gap.
  (g) Grouping — paraphrases (high cosine) form one cluster; orthogonal centres
      stay separate; cluster ids follow stream order; sizes are the recurrence.
  (h) Session-spread — one intent across two sessions ⇒ distinct_sessions 2.
  (i) Degenerate — ragged → all-singletons; a zero/NaN embedding is its own
      singleton and never merges into a real intent.
  (j) Deterministic / pure — no clock; the same stream reproduces.

Reformulation contracts:

  (k) Empty/filter — same row-selection unit.
  (l) Fast re-ask — a same-session near-paraphrase N seconds later ⇒ has_followup,
      high next_cosine, gap_seconds N; the last query is (False, 0.0, inf).
  (m) Distinct intent — an orthogonal next query ⇒ followup but low cosine.
  (n) Sessions — cross-session is not a followup; interleaved s1/s2/s1 finds the
      next *same-session* query.
  (o) Degenerate — ragged → cosine 0 with followup+gap intact; unparseable ts →
      gap inf; out-of-order ts clamps the gap to 0.
  (p) Deterministic — gap from captured timestamps, not a wall clock; reproduces.

Hermetic — no model load, no disk; seeded numpy noise, a hand-built BandHandle.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from resonance_lattice import field
from resonance_lattice.field import capture
from resonance_lattice.curator import signals
from resonance_lattice.store.archive import BandHandle

_TS = "2026-06-01T12:00:00+00:00"


def _row(emb, top, *, layer="source", is_user=True, session="s", ts=_TS):
    return {
        "ts": ts,
        "session": session,
        "layer": layer,
        "is_user_query": is_user,
        "query_emb": [float(x) for x in emb],
        "ranked": [{"rank": 0, "idx": 0, "score": float(top)}],
    }


def _ts(sec: int) -> str:
    """An aware ISO timestamp `sec` seconds past 2026-06-01T12:00:00Z — for
    pinning reformulation time gaps without a wall clock."""
    return f"2026-06-01T12:00:{sec:02d}+00:00"


def _served(rng, n, centre, top_lo=0.88, top_hi=0.92):
    """n well-served rows: embeddings tight around `centre` (+ small noise so
    the maha column has spread), high top scores."""
    out = []
    for _ in range(n):
        emb = np.array(centre, dtype="float64") + rng.normal(0, 0.02, size=len(centre))
        out.append(_row(emb, rng.uniform(top_lo, top_hi)))
    return out


def _near(rng, centre, n, *, session="s", sigma=0.02):
    """n rows whose embeddings sit tight around `centre` (cosine ≈ 1 to each
    other, so they cluster; small noise so they are not byte-identical)."""
    c = np.array(centre, dtype="float64")
    return [_row(c + rng.normal(0, sigma, size=len(centre)), 0.9, session=session)
            for _ in range(n)]


def _check_empty_and_filter() -> int:
    if signals.score_gap_features([]) != []:
        print("[curator_signals] empty: non-empty for empty stream", file=sys.stderr)
        return 1
    rows = [
        _row([1, 0, 0], 0.9, is_user=False),       # machinery — ignored
        _row([1, 0, 0], 0.9, layer="insight"),      # insight layer — not the unit
        {"ts": _TS, "session": "s", "layer": "source",
         "is_user_query": True, "query_emb": None, "ranked": []},  # no fingerprint
    ]
    if signals.score_gap_features(rows) != []:
        print("[curator_signals] filter: counted a non-user/insight/embless row",
              file=sys.stderr)
        return 1
    return 0


def _check_separation() -> int:
    rng = np.random.default_rng(0)
    served = _served(rng, 8, [1, 0, 0])
    gap_low = _row([1, 0, 0], 0.05)            # near served, but weak match
    gap_far = _row([0, 1, 0], 0.9)             # strong score, far embedding
    feats = signals.score_gap_features(served + [gap_low, gap_far])
    if len(feats) != 10:
        print(f"[curator_signals] separation: expected 10, got {len(feats)}",
              file=sys.stderr)
        return 1
    tops = [f.top_score for f in feats]
    mahas = [f.maha for f in feats]
    if tops.index(min(tops)) != 8:
        print(f"[curator_signals] separation: low-top query is not the min: {tops}",
              file=sys.stderr)
        return 1
    if mahas.index(max(mahas)) != 9:
        print(f"[curator_signals] separation: far-emb query is not the max maha: "
              f"{mahas}", file=sys.stderr)
        return 1
    if not all(tops[i] > 0.8 for i in range(8)):
        print(f"[curator_signals] separation: a served top dipped: {tops[:8]}",
              file=sys.stderr)
        return 1
    if not all(mahas[i] < mahas[9] for i in range(8)):
        print("[curator_signals] separation: a served maha exceeded the OOD query",
              file=sys.stderr)
        return 1
    return 0


def _check_degenerate() -> int:
    # All identical → no served spread → maha 0 everywhere (no spurious signal).
    flat = signals.score_gap_features([_row([1, 0, 0], 0.9) for _ in range(5)])
    if len(flat) != 5 or any(f.maha != 0.0 for f in flat):
        print(f"[curator_signals] degenerate: identical stream produced maha: "
              f"{[f.maha for f in flat]}", file=sys.stderr)
        return 1
    # A ragged embedding → score-only fallback (maha 0), top_score preserved.
    ragged = signals.score_gap_features([
        _row([1, 0, 0], 0.9),
        {"ts": _TS, "session": "s", "layer": "source", "is_user_query": True,
         "query_emb": [1.0, 0.0], "ranked": [{"rank": 0, "idx": 0, "score": 0.3}]},
    ])
    if len(ragged) != 2 or any(f.maha != 0.0 for f in ragged) \
            or abs(ragged[1].top_score - 0.3) > 1e-9:
        print(f"[curator_signals] degenerate: ragged fallback wrong: {ragged}",
              file=sys.stderr)
        return 1
    # A non-finite embedding component (capture rounds float('nan') through) must
    # NOT poison the maha column for the rest of the stream.
    rng = np.random.default_rng(9)
    feats = signals.score_gap_features(
        _served(rng, 6, [1, 0, 0]) + [_row([float("nan"), 0.0, 0.0], 0.9)])
    if any(not math.isfinite(f.maha) for f in feats):
        print(f"[curator_signals] degenerate: a NaN embedding poisoned maha: "
              f"{[f.maha for f in feats]}", file=sys.stderr)
        return 1
    return 0


def _check_deterministic() -> int:
    rng = np.random.default_rng(3)
    rows = _served(rng, 6, [1, 0, 0]) + [_row([0, 0, 1], 0.1)]
    a = signals.score_gap_features(rows)
    b = signals.score_gap_features(rows)
    if [(f.top_score, f.maha) for f in a] != [(f.top_score, f.maha) for f in b]:
        print("[curator_signals] deterministic: two passes differ", file=sys.stderr)
        return 1
    return 0


def _handle(km_id: str) -> BandHandle:
    band = np.eye(3, dtype="float32")
    return BandHandle(name="base", band=band, ann_blob=None, km_id=km_id)


def _check_real_stream() -> int:
    km = "curator-real"
    capture.drain(km)
    q = np.array([1.0, 0.0, 0.0], dtype="float32")
    field.retrieve(q, _handle(km), None, None, top_k=2)
    field.retrieve(q, _handle(km), None, None, top_k=2)
    rows = capture.drain(km)
    feats = signals.score_gap_features(rows)
    if len(feats) != 2:
        print(f"[curator_signals] real-stream: expected 2 source rows, got "
              f"{len(feats)}", file=sys.stderr)
        return 1
    if not all(abs(f.top_score - 1.0) < 1e-6 for f in feats):
        print(f"[curator_signals] real-stream: top score not the eye-retrieval "
              f"1.0: {[f.top_score for f in feats]}", file=sys.stderr)
        return 1
    return 0


def _check_intent_empty_and_filter() -> int:
    if signals.intent_clusters([]) != []:
        print("[curator_signals] intent-empty: non-empty for empty stream",
              file=sys.stderr)
        return 1
    rows = [
        _row([1, 0, 0], 0.9, is_user=False),       # machinery — ignored
        _row([1, 0, 0], 0.9, layer="insight"),      # insight layer — not the unit
        {"ts": _TS, "session": "s", "layer": "source",
         "is_user_query": True, "query_emb": None, "ranked": []},  # no fingerprint
    ]
    if signals.intent_clusters(rows) != []:
        print("[curator_signals] intent-filter: counted a non-user/insight/embless "
              "row", file=sys.stderr)
        return 1
    return 0


def _check_intent_grouping() -> int:
    # Three paraphrase clusters around orthogonal centres: 3 + 2 + 1 rows. Within
    # a centre cosine ≈ 1 (> 0.7 ⇒ one cluster); across centres ≈ 0 (⇒ separate).
    rng = np.random.default_rng(1)
    rows = (_near(rng, [1, 0, 0, 0], 3)
            + _near(rng, [0, 1, 0, 0], 2)
            + _near(rng, [0, 0, 1, 0], 1))
    feats = signals.intent_clusters(rows)
    ids = [f.cluster_id for f in feats]
    sizes = [f.size for f in feats]
    if ids != [0, 0, 0, 1, 1, 2]:
        print(f"[curator_signals] intent-grouping: cluster ids not stream-ordered "
              f"3/2/1: {ids}", file=sys.stderr)
        return 1
    if sizes != [3, 3, 3, 2, 2, 1]:
        print(f"[curator_signals] intent-grouping: sizes wrong: {sizes}",
              file=sys.stderr)
        return 1
    if any(f.distinct_sessions != 1 for f in feats):
        print("[curator_signals] intent-grouping: single-session run not all 1",
              file=sys.stderr)
        return 1
    return 0


def _check_intent_sessions() -> int:
    # One intent (same centre) queried across two sessions ⇒ distinct_sessions 2.
    rng = np.random.default_rng(2)
    rows = _near(rng, [1, 0, 0, 0], 2, session="s1") \
        + _near(rng, [1, 0, 0, 0], 2, session="s2")
    feats = signals.intent_clusters(rows)
    if any(f.cluster_id != 0 or f.size != 4 for f in feats):
        print(f"[curator_signals] intent-sessions: cross-session paraphrases did "
              f"not form one cluster: {[(f.cluster_id, f.size) for f in feats]}",
              file=sys.stderr)
        return 1
    if any(f.distinct_sessions != 2 for f in feats):
        print(f"[curator_signals] intent-sessions: session-spread wrong: "
              f"{[f.distinct_sessions for f in feats]}", file=sys.stderr)
        return 1
    # A repeated session inside one cluster must dedup: s1, s1, s2 ⇒ 2, not 3.
    rng2 = np.random.default_rng(8)
    rep = (_near(rng2, [1, 0, 0, 0], 1, session="s1")
           + _near(rng2, [1, 0, 0, 0], 1, session="s1")
           + _near(rng2, [1, 0, 0, 0], 1, session="s2"))
    rfeats = signals.intent_clusters(rep)
    if any(f.size != 3 for f in rfeats) \
            or any(f.distinct_sessions != 2 for f in rfeats):
        print(f"[curator_signals] intent-sessions: repeated session not deduped: "
              f"{[(f.size, f.distinct_sessions) for f in rfeats]}", file=sys.stderr)
        return 1
    return 0


def _check_intent_degenerate() -> int:
    rng = np.random.default_rng(5)
    # Ragged embeddings → all-singletons fallback (no crash).
    ragged = signals.intent_clusters([
        _row([1, 0, 0], 0.9),
        {"ts": _TS, "session": "s", "layer": "source", "is_user_query": True,
         "query_emb": [1.0, 0.0], "ranked": [{"rank": 0, "idx": 0, "score": 0.3}]},
    ])
    if [f.cluster_id for f in ragged] != [0, 1] or any(f.size != 1 for f in ragged):
        print(f"[curator_signals] intent-degenerate: ragged not all-singletons: "
              f"{ragged}", file=sys.stderr)
        return 1
    # A zero vector clusters with nothing (cosine 0 < 0.7) — its own singleton,
    # never absorbed into the real intent.
    z = signals.intent_clusters(_near(rng, [1, 0, 0, 0], 2) + [_row([0, 0, 0, 0], 0.9)])
    if z[2].size != 1 or z[0].cluster_id != z[1].cluster_id \
            or z[2].cluster_id == z[0].cluster_id:
        print(f"[curator_signals] intent-degenerate: zero vector merged into an "
              f"intent: {[(f.cluster_id, f.size) for f in z]}", file=sys.stderr)
        return 1
    # A non-finite embedding (capture rounds float('nan') through) is sanitised to
    # a singleton and must NOT poison the real cluster's membership.
    nanf = signals.intent_clusters(
        _near(rng, [1, 0, 0, 0], 3) + [_row([float("nan"), 0, 0, 0], 0.9)])
    if nanf[0].size != 3 or nanf[3].size != 1:
        print(f"[curator_signals] intent-degenerate: NaN row poisoned the cluster: "
              f"{[(f.cluster_id, f.size) for f in nanf]}", file=sys.stderr)
        return 1
    # An ndarray embedding (a shape capture never emits) must not raise the shared
    # row filter — the "never raises on any row shape" contract, both clauses.
    arr_row = {"ts": _TS, "session": "s", "layer": "source", "is_user_query": True,
               "query_emb": np.array([1.0, 0.0, 0.0]),
               "ranked": [{"rank": 0, "idx": 0, "score": 0.9}]}
    if len(signals.intent_clusters([arr_row])) != 1 \
            or len(signals.score_gap_features([arr_row])) != 1:
        print("[curator_signals] intent-degenerate: ndarray embedding raised or was "
              "dropped in the shared filter", file=sys.stderr)
        return 1
    return 0


def _check_intent_stream_order() -> int:
    # The first-APPEARING cluster must get id 0 even when it is the SMALLER one —
    # this separates first-appearance ordering from size- or insertion-ordering,
    # which the contiguous-block tests can't.
    rng = np.random.default_rng(7)
    rows = (_near(rng, [0, 1, 0, 0], 1)     # idx 0   — intent B (ends up size 2)
            + _near(rng, [1, 0, 0, 0], 2)   # idx 1,2 — intent A (size 3)
            + _near(rng, [0, 1, 0, 0], 1)   # idx 3   — B
            + _near(rng, [1, 0, 0, 0], 1))  # idx 4   — A
    feats = signals.intent_clusters(rows)
    ids = [f.cluster_id for f in feats]
    sizes = [f.size for f in feats]
    if ids != [0, 1, 1, 0, 1]:
        print(f"[curator_signals] intent-order: id not first-appearance "
              f"(smaller intent appeared first): {ids}", file=sys.stderr)
        return 1
    if sizes != [2, 3, 3, 2, 3]:
        print(f"[curator_signals] intent-order: sizes wrong: {sizes}",
              file=sys.stderr)
        return 1
    return 0


def _check_intent_deterministic() -> int:
    rng = np.random.default_rng(3)
    rows = _near(rng, [1, 0, 0, 0], 4) + _near(rng, [0, 1, 0, 0], 2)
    a = signals.intent_clusters(rows)
    b = signals.intent_clusters(rows)
    key = lambda fs: [(f.cluster_id, f.size, f.distinct_sessions) for f in fs]
    if key(a) != key(b):
        print("[curator_signals] intent-deterministic: two passes differ",
              file=sys.stderr)
        return 1
    return 0


def _check_reform_empty_and_filter() -> int:
    if signals.reformulation_features([]) != []:
        print("[curator_signals] reform-empty: non-empty for empty stream",
              file=sys.stderr)
        return 1
    rows = [
        _row([1, 0, 0], 0.9, is_user=False),
        _row([1, 0, 0], 0.9, layer="insight"),
        {"ts": _TS, "session": "s", "layer": "source",
         "is_user_query": True, "query_emb": None, "ranked": []},
    ]
    if signals.reformulation_features(rows) != []:
        print("[curator_signals] reform-filter: counted a non-user/insight/embless "
              "row", file=sys.stderr)
        return 1
    return 0


def _check_reform_fast_reask() -> int:
    # A fast same-intent re-ask: near-identical embeddings, same session, 5s apart.
    rng = np.random.default_rng(11)
    e = np.array([1.0, 0, 0, 0])
    a = _row(e + rng.normal(0, 0.02, 4), 0.9, ts=_ts(0))
    b = _row(e + rng.normal(0, 0.02, 4), 0.9, ts=_ts(5))
    feats = signals.reformulation_features([a, b])
    if len(feats) != 2:
        print(f"[curator_signals] reform-fast: expected 2, got {len(feats)}",
              file=sys.stderr)
        return 1
    if not feats[0].has_followup or feats[0].next_cosine < 0.9 \
            or abs(feats[0].gap_seconds - 5.0) > 1e-6:
        print(f"[curator_signals] reform-fast: q0 not a 5s high-cosine re-ask: "
              f"{feats[0]}", file=sys.stderr)
        return 1
    # The last query has no followup → the no-reformulation sentinel.
    if feats[1].has_followup or feats[1].next_cosine != 0.0 \
            or feats[1].gap_seconds != float("inf"):
        print(f"[curator_signals] reform-fast: last query not (False,0,inf): "
              f"{feats[1]}", file=sys.stderr)
        return 1
    return 0


def _check_reform_distinct_intent() -> int:
    # Consecutive but DIFFERENT intent (orthogonal): followup exists, cosine low —
    # the combiner won't read this as a same-intent reformulation.
    a = _row([1, 0, 0, 0], 0.9, ts=_ts(0))
    b = _row([0, 1, 0, 0], 0.9, ts=_ts(3))
    feats = signals.reformulation_features([a, b])
    if not feats[0].has_followup or feats[0].next_cosine > 0.5:
        print(f"[curator_signals] reform-distinct: orthogonal next not low-cosine: "
              f"{feats[0]}", file=sys.stderr)
        return 1
    return 0


def _check_reform_sessions() -> int:
    # Different sessions → no same-session followup for either query.
    a = _row([1, 0, 0, 0], 0.9, session="s1", ts=_ts(0))
    b = _row([1, 0, 0, 0], 0.9, session="s2", ts=_ts(2))
    feats = signals.reformulation_features([a, b])
    if feats[0].has_followup or feats[1].has_followup:
        print(f"[curator_signals] reform-sessions: cross-session counted as a "
              f"followup: {[f.has_followup for f in feats]}", file=sys.stderr)
        return 1
    # Interleaved s1, s2, s1 → q0's followup is index 2 (the next s1), skipping s2.
    c = _row([1, 0, 0, 0], 0.9, session="s1", ts=_ts(0))
    d = _row([0, 1, 0, 0], 0.9, session="s2", ts=_ts(1))
    e = _row([1, 0, 0, 0], 0.9, session="s1", ts=_ts(4))
    feats = signals.reformulation_features([c, d, e])
    if not feats[0].has_followup or abs(feats[0].gap_seconds - 4.0) > 1e-6 \
            or feats[0].next_cosine < 0.9:
        print(f"[curator_signals] reform-sessions: interleaved next-same-session "
              f"wrong: {feats[0]}", file=sys.stderr)
        return 1
    if feats[1].has_followup or feats[2].has_followup:
        print(f"[curator_signals] reform-sessions: lone s2 / last s1 got a "
              f"followup: {[f.has_followup for f in feats]}", file=sys.stderr)
        return 1
    return 0


def _check_reform_degenerate() -> int:
    # Ragged embeddings → cosine unavailable (0.0) but followup + gap still hold.
    a = _row([1, 0, 0], 0.9, ts=_ts(0))
    braw = {"ts": _ts(2), "session": "s", "layer": "source", "is_user_query": True,
            "query_emb": [1.0, 0.0], "ranked": [{"rank": 0, "idx": 0, "score": 0.9}]}
    feats = signals.reformulation_features([a, braw])
    if not feats[0].has_followup or feats[0].next_cosine != 0.0 \
            or abs(feats[0].gap_seconds - 2.0) > 1e-6:
        print(f"[curator_signals] reform-degenerate: ragged did not degrade to "
              f"cosine-0 with followup+gap: {feats[0]}", file=sys.stderr)
        return 1
    # Unparseable ts → gap inf (timing unknown) but the followup still counts.
    x = _row([1, 0, 0, 0], 0.9, ts="not-a-date")
    y = _row([1, 0, 0, 0], 0.9, ts=_ts(3))
    feats = signals.reformulation_features([x, y])
    if not feats[0].has_followup or feats[0].gap_seconds != float("inf"):
        print(f"[curator_signals] reform-degenerate: unparseable ts not inf gap: "
              f"{feats[0]}", file=sys.stderr)
        return 1
    # Out-of-order timestamps (clock skew) clamp the gap to 0, never negative.
    p = _row([1, 0, 0, 0], 0.9, ts=_ts(9))
    q = _row([1, 0, 0, 0], 0.9, ts=_ts(4))
    feats = signals.reformulation_features([p, q])
    if feats[0].gap_seconds != 0.0:
        print(f"[curator_signals] reform-degenerate: negative gap not clamped: "
              f"{feats[0].gap_seconds}", file=sys.stderr)
        return 1
    # A zero-norm / NaN followup embedding yields a finite cosine 0.0 (never NaN),
    # with the followup + gap intact — the cosine path's fail-safe in this clause.
    for bad in ([0.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0, 0.0]):
        feats = signals.reformulation_features(
            [_row([1, 0, 0, 0], 0.9, ts=_ts(0)), _row(bad, 0.9, ts=_ts(1))])
        if not feats[0].has_followup or feats[0].next_cosine != 0.0:
            print(f"[curator_signals] reform-degenerate: bad followup embedding did "
                  f"not give finite cosine 0.0: {feats[0]}", file=sys.stderr)
            return 1
    return 0


def _check_reform_deterministic() -> int:
    rng = np.random.default_rng(13)
    e = np.array([1.0, 0, 0, 0])
    rows = [_row(e + rng.normal(0, 0.02, 4), 0.9, ts=_ts(s)) for s in (0, 2, 7)]
    a = signals.reformulation_features(rows)
    b = signals.reformulation_features(rows)
    key = lambda fs: [(f.has_followup, f.next_cosine, f.gap_seconds) for f in fs]
    if key(a) != key(b):
        print("[curator_signals] reform-deterministic: two passes differ",
              file=sys.stderr)
        return 1
    return 0


def run() -> int:
    for check in [
        _check_empty_and_filter,
        _check_separation,
        _check_degenerate,
        _check_deterministic,
        _check_real_stream,
        _check_intent_empty_and_filter,
        _check_intent_grouping,
        _check_intent_sessions,
        _check_intent_degenerate,
        _check_intent_stream_order,
        _check_intent_deterministic,
        _check_reform_empty_and_filter,
        _check_reform_fast_reask,
        _check_reform_distinct_intent,
        _check_reform_sessions,
        _check_reform_degenerate,
        _check_reform_deterministic,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[curator_signals] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
