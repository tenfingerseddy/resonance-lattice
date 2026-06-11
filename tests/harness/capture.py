"""capture — the self-aware retrieval heart.

Pins the C1-rebuild contracts (`.claude/plans/insight-engine/capture.md` §3,
resolved 2026-06-02):

  (a) Observe roundtrip — a recorded retrieval reads back with the fingerprint,
      the per-rank scores, the layer, and is_user_query (default True).
  (b) Raise-your-hand flag — inside `internal_retrieval()`, is_user_query=False.
  (c) No identity → not observable — `km_id=None` is skipped (a synthetic archive
      with no source path captures nothing), and never crashes.
  (d) drain clears — the fold reads via drain; a second drain is empty.
  (e) Never breaks retrieval — a hostile embedding / ranked is swallowed.
  (f) Sorted + idx preserved — out-of-order ranked sorts score-descending, idx
      and positional rank preserved.
  (g) Bounded — the per-corpus ring caps at `_MAX_BUFFERED`.
  (h) The heart actually observes — a real `field.retrieve` /
      `field.retrieve_insight` call lands an observation against the corpus, and
      `internal_retrieval()` flags it through the real call path.

Hermetic — no model load, no disk; the integration uses a hand-built BandHandle
(`registry=None` skips dedup, so no corpus fixture is needed).
"""

from __future__ import annotations

import sys

import numpy as np

from resonance_lattice import field
from resonance_lattice.field import capture
from resonance_lattice.store.archive import BandHandle


def _src(name: str = "src", ranked=((0, 0.9),)) -> None:
    capture.observe(name, [0.1, 0.2], list(ranked), "source")


def _check_observe_roundtrip() -> int:
    km = "rt-km"
    capture.drain(km)
    capture.observe(km, [0.1, 0.2, 0.3], [(2, 0.4), (0, 0.9)], "source")
    rows = capture.buffered(km)
    if len(rows) != 1:
        print(f"[capture] roundtrip: expected 1 row, got {len(rows)}", file=sys.stderr)
        return 1
    r = rows[0]
    ok = (
        r["query_emb"] == [0.1, 0.2, 0.3]
        and r["layer"] == "source"
        and r["is_user_query"] is True
        and [x["score"] for x in r["ranked"]] == [0.9, 0.4]
    )
    capture.drain(km)
    if not ok:
        print(f"[capture] roundtrip: fields wrong: {r}", file=sys.stderr)
        return 1
    return 0


def _check_internal_flag() -> int:
    km = "int-km"
    capture.drain(km)
    with capture.internal_retrieval():
        capture.observe(km, [0.0], [(0, 0.5)], "source")
    # Outside the block the default returns to user.
    capture.observe(km, [0.0], [(0, 0.5)], "source")
    rows = capture.buffered(km)
    capture.drain(km)
    if [r["is_user_query"] for r in rows] != [False, True]:
        print(f"[capture] internal-flag: {[r['is_user_query'] for r in rows]}", file=sys.stderr)
        return 1
    return 0


def _check_no_km_skipped() -> int:
    before = capture.buffered(None)
    capture.observe(None, [0.0], [(0, 0.5)], "source")
    capture.observe("", [0.0], [(0, 0.5)], "source")
    if capture.buffered(None) != before or capture.buffered("") != []:
        print("[capture] no-km: a row was buffered without identity", file=sys.stderr)
        return 1
    return 0


def _check_drain() -> int:
    km = "drain-km"
    capture.drain(km)
    capture.observe(km, [0.0], [(0, 0.5)], "source")
    capture.observe(km, [0.0], [(1, 0.4)], "insight")
    drained = capture.drain(km)
    if len(drained) != 2:
        print(f"[capture] drain: expected 2, got {len(drained)}", file=sys.stderr)
        return 1
    if capture.drain(km) != [] or capture.buffered(km) != []:
        print("[capture] drain: buffer not cleared", file=sys.stderr)
        return 1
    return 0


class _BoomEmb:
    def __iter__(self):
        raise RuntimeError("boom")


def _check_never_breaks() -> int:
    km = "boom-km"
    capture.drain(km)
    try:
        capture.observe(km, _BoomEmb(), [(0, 0.5)], "source")
        capture.observe(km, [0.0], [("not-a-pair",)], "source")  # bad ranked
        capture.observe(km, object(), None, "source")
    except Exception as e:  # the whole point: it must NOT propagate
        print(f"[capture] never-breaks: observe raised {e!r}", file=sys.stderr)
        return 1
    if capture.buffered(km):
        print("[capture] never-breaks: a hostile row got buffered", file=sys.stderr)
        return 1
    return 0


def _check_sorted_idx() -> int:
    km = "sort-km"
    capture.drain(km)
    capture.observe(km, [0.0], [(5, 0.6), (2, 0.9), (7, 0.7)], "source")
    r = capture.buffered(km)[0]
    capture.drain(km)
    if [x["score"] for x in r["ranked"]] != [0.9, 0.7, 0.6]:
        print("[capture] sorted: not score-descending", file=sys.stderr)
        return 1
    if [x["idx"] for x in r["ranked"]] != [2, 7, 5]:
        print("[capture] sorted: idx not preserved through the sort", file=sys.stderr)
        return 1
    if [x["rank"] for x in r["ranked"]] != [0, 1, 2]:
        print("[capture] sorted: rank not positional", file=sys.stderr)
        return 1
    return 0


def _check_bounded() -> int:
    km = "ring-km"
    capture.drain(km)
    for i in range(capture._MAX_BUFFERED + 25):
        capture.observe(km, [0.0], [(i, 0.5)], "source")
    n = len(capture.buffered(km))
    capture.drain(km)
    if n != capture._MAX_BUFFERED:
        print(f"[capture] bounded: ring is {n}, expected {capture._MAX_BUFFERED}", file=sys.stderr)
        return 1
    return 0


def _handle(km_id: str) -> BandHandle:
    # Three unit rows in 3-space; cosine == dot for L2-normalised vectors.
    band = np.eye(3, dtype="float32")
    return BandHandle(name="base", band=band, ann_blob=None, km_id=km_id)


def _check_heart_source() -> int:
    km = "heart-src"
    capture.drain(km)
    q = np.array([1.0, 0.0, 0.0], dtype="float32")
    result = field.retrieve(q, _handle(km), None, None, top_k=2)
    rows = capture.buffered(km)
    capture.drain(km)
    if len(rows) != 1 or rows[0]["layer"] != "source" or rows[0]["is_user_query"] is not True:
        print(f"[capture] heart-source: observation missing/wrong: {rows}", file=sys.stderr)
        return 1
    if [x["idx"] for x in rows[0]["ranked"]] != [i for i, _ in result]:
        print("[capture] heart-source: ranked idx mismatch vs result", file=sys.stderr)
        return 1
    return 0


def _check_heart_insight() -> int:
    km = "heart-ins"
    capture.drain(km)
    band = np.eye(3, dtype="float32")
    q = np.array([0.0, 1.0, 0.0], dtype="float32")
    field.retrieve_insight(q, band, None, top_k=2, km_id=km)
    rows = capture.buffered(km)
    capture.drain(km)
    if len(rows) != 1 or rows[0]["layer"] != "insight":
        print(f"[capture] heart-insight: observation missing/wrong: {rows}", file=sys.stderr)
        return 1
    # No km_id passed → not observable (the verified light-read path).
    field.retrieve_insight(q, band, None, top_k=2)
    if capture.buffered(km):
        print("[capture] heart-insight: km-less insight retrieval was observed", file=sys.stderr)
        return 1
    return 0


def _check_heart_internal() -> int:
    km = "heart-int"
    capture.drain(km)
    q = np.array([1.0, 0.0, 0.0], dtype="float32")
    with capture.internal_retrieval():
        field.retrieve(q, _handle(km), None, None, top_k=1)
    rows = capture.buffered(km)
    capture.drain(km)
    if len(rows) != 1 or rows[0]["is_user_query"] is not False:
        print(f"[capture] heart-internal: flag not applied through field.retrieve: {rows}", file=sys.stderr)
        return 1
    return 0


def run() -> int:
    for check in [
        _check_observe_roundtrip,
        _check_internal_flag,
        _check_no_km_skipped,
        _check_drain,
        _check_never_breaks,
        _check_sorted_idx,
        _check_bounded,
        _check_heart_source,
        _check_heart_insight,
        _check_heart_internal,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[capture] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
