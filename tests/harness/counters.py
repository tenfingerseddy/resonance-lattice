"""counters — the Tier-0 closed-form count layer (no model).

Pins the C2 contracts (`.claude/plans/insight-engine/horizon-1-capture.md`
stage C2: "the counters move on a real query stream"; mechanism in
`capture.md` §3 + `boundary.md` count tier):

  (a) Empty stream → empty map; only user-intent rows count (is_user_query).
  (b) Access count increments per hit; idx + layer keys are correct.
  (c) Layer separation + the `layer` filter.
  (d) Ebbinghaus decay — a recent hit reinforces more than an old one.
  (e) Log dampening — N hits reinforce LESS than N× a single hit.
  (f) Clock-free determinism — `now=None` defaults to the latest ts and
      reproduces; the newest hit is undecayed, an earlier one is faded.
  (g) Never raises — hostile rows (bad ranked, non-int idx, non-dict) are
      skipped, the pass survives.
  (h) The counters move on a REAL stream — `field.retrieve` through `capture`,
      drained, counted: the hit idx lands with the right access count.

Hermetic — no model load, no disk; the integration uses a hand-built BandHandle
(`registry=None` skips dedup, so no corpus fixture is needed).
"""

from __future__ import annotations

import math
import sys

import numpy as np

from resonance_lattice import field
from resonance_lattice.field import capture, counters
from resonance_lattice.store.archive import BandHandle

_NOW = "2026-06-01T12:00:00+00:00"
_OLD = "2026-05-02T12:00:00+00:00"   # 30 days before _NOW → decay factor e^-1


def _row(layer, idxs, ts=_NOW, *, is_user=True, session="s"):
    return {
        "ts": ts,
        "session": session,
        "layer": layer,
        "is_user_query": is_user,
        "query_emb": [0.0],
        "ranked": [{"rank": i, "idx": d, "score": 0.9 - i * 0.1}
                   for i, d in enumerate(idxs)],
    }


def _check_empty_and_internal() -> int:
    if counters.access_reinforcement([]) != {}:
        print("[counters] empty: non-empty result for empty stream", file=sys.stderr)
        return 1
    # An internal (machinery) row must not count.
    stats = counters.access_reinforcement([_row("source", [0], is_user=False)])
    if stats:
        print(f"[counters] internal: machinery row counted: {stats}", file=sys.stderr)
        return 1
    return 0


def _check_counts_and_keys() -> int:
    stats = counters.access_reinforcement([
        _row("source", [0, 1]),
        _row("source", [0]),
        _row("insight", [0]),
    ])
    src0 = stats.get(("source", 0))
    src1 = stats.get(("source", 1))
    ins0 = stats.get(("insight", 0))
    if not (src0 and src1 and ins0):
        print(f"[counters] keys: missing expected keys: {sorted(stats)}", file=sys.stderr)
        return 1
    # Layer separation: (source,0) and (insight,0) are distinct units — src0
    # saw 2 user hits, ins0 saw 1; a merge would have shown 3 on one key.
    if src0.access_count != 2 or src1.access_count != 1 or ins0.access_count != 1:
        print(f"[counters] counts: src0={src0.access_count} src1={src1.access_count} "
              f"ins0={ins0.access_count}", file=sys.stderr)
        return 1
    return 0


def _check_layer_filter() -> int:
    rows = [_row("source", [0]), _row("insight", [0])]
    only = counters.access_reinforcement(rows, layer="insight")
    if set(only) != {("insight", 0)}:
        print(f"[counters] filter: layer=insight leaked: {sorted(only)}", file=sys.stderr)
        return 1
    return 0


def _check_decay() -> int:
    # Same now; one recent hit, one 30-days-old hit, distinct units.
    stats = counters.access_reinforcement(
        [_row("source", [0], ts=_NOW), _row("source", [1], ts=_OLD)],
        now=_NOW,
    )
    recent = stats[("source", 0)]
    old = stats[("source", 1)]
    if not (recent.reinforcement > old.reinforcement):
        print(f"[counters] decay: recent {recent.reinforcement} !> old "
              f"{old.reinforcement}", file=sys.stderr)
        return 1
    # The old hit should be ~e^-1 of a fresh one (τ = 30 days).
    if not (abs(old.decayed_access - math.exp(-1.0)) < 1e-6):
        print(f"[counters] decay: old decayed_access {old.decayed_access} != e^-1",
              file=sys.stderr)
        return 1
    if not (abs(recent.decayed_access - 1.0) < 1e-9):
        print(f"[counters] decay: recent decayed_access {recent.decayed_access} != 1.0",
              file=sys.stderr)
        return 1
    return 0


def _check_log_dampening() -> int:
    one = counters.access_reinforcement([_row("source", [0])], now=_NOW)
    ten = counters.access_reinforcement(
        [_row("source", [0]) for _ in range(10)], now=_NOW
    )
    r1 = one[("source", 0)].reinforcement
    r10 = ten[("source", 0)].reinforcement
    if ten[("source", 0)].access_count != 10:
        print(f"[counters] dampening: count {ten[('source', 0)].access_count} != 10",
              file=sys.stderr)
        return 1
    # Diminishing returns: ten hits are worth more than one, but far less than 10×.
    if not (r1 < r10 < 10 * r1):
        print(f"[counters] dampening: expected r1 < r10 < 10*r1, got "
              f"r1={r1} r10={r10}", file=sys.stderr)
        return 1
    if not (abs(r10 - math.log1p(10.0)) < 1e-9):
        print(f"[counters] dampening: r10 {r10} != log1p(10)", file=sys.stderr)
        return 1
    return 0


def _check_now_default() -> int:
    rows = [_row("source", [0], ts=_OLD), _row("source", [1], ts=_NOW)]
    a = counters.access_reinforcement(rows)            # now=None → latest ts (_NOW)
    b = counters.access_reinforcement(rows, now=_NOW)
    if {k: v.reinforcement for k, v in a.items()} != {k: v.reinforcement for k, v in b.items()}:
        print("[counters] now-default: did not default to latest ts", file=sys.stderr)
        return 1
    # The newest hit is undecayed; the older one is faded.
    if not (abs(a[("source", 1)].decayed_access - 1.0) < 1e-9):
        print(f"[counters] now-default: newest not undecayed: "
              f"{a[('source', 1)].decayed_access}", file=sys.stderr)
        return 1
    if not (a[("source", 0)].decayed_access < 1.0):
        print(f"[counters] now-default: oldest not faded: "
              f"{a[('source', 0)].decayed_access}", file=sys.stderr)
        return 1
    if a[("source", 1)].last_ts != _NOW:
        print(f"[counters] now-default: last_ts wrong: {a[('source', 1)].last_ts}",
              file=sys.stderr)
        return 1
    return 0


def _check_never_raises() -> int:
    hostile = [
        "not-a-dict",
        {"is_user_query": True, "layer": "source", "ranked": "not-a-list", "ts": _NOW},
        {"is_user_query": True, "layer": "source",
         "ranked": [{"rank": 0, "idx": "nope"}], "ts": _NOW},   # non-int idx
        {"is_user_query": True, "layer": "source",
         "ranked": [{"rank": 0}], "ts": _NOW},                  # missing idx
        {"is_user_query": True, "layer": "source",
         "ranked": [{"idx": 0}], "ts": "garbage"},              # bad ts → undecayed
    ]
    try:
        stats = counters.access_reinforcement(hostile, now=_NOW)
    except Exception as e:
        print(f"[counters] never-raises: pass raised {e!r}", file=sys.stderr)
        return 1
    # Only the last row is well-formed enough to count (idx 0, bad ts → weight 1).
    if stats.get(("source", 0)) is None or stats[("source", 0)].access_count != 1:
        print(f"[counters] never-raises: good row not counted: {stats}", file=sys.stderr)
        return 1
    # Hostile PARAMETERS (caller mistakes) must degrade to defaults, not crash —
    # these run outside the per-row guard.
    try:
        s_now = counters.access_reinforcement([_row("source", [0])], now=12345)
        s_tau = counters.access_reinforcement([_row("source", [0])], tau_days="30")
    except Exception as e:
        print(f"[counters] never-raises: hostile now/tau raised {e!r}", file=sys.stderr)
        return 1
    if s_now.get(("source", 0)) is None or s_tau.get(("source", 0)) is None:
        print("[counters] never-raises: hostile now/tau dropped the good row",
              file=sys.stderr)
        return 1
    return 0


def _check_last_ts_chronological() -> int:
    # Mixed offsets: the lexically-larger string is chronologically EARLIER.
    early_utc = _row("source", [0], ts="2026-06-01T12:00:00+05:00")  # 07:00Z
    late_utc = _row("source", [0], ts="2026-06-01T10:00:00+00:00")   # 10:00Z
    stats = counters.access_reinforcement([early_utc, late_utc])
    got = stats[("source", 0)].last_ts
    if got != "2026-06-01T10:00:00+00:00":
        print(f"[counters] last_ts: picked {got}, not the chronologically latest",
              file=sys.stderr)
        return 1
    return 0


def _handle(km_id: str) -> BandHandle:
    band = np.eye(3, dtype="float32")
    return BandHandle(name="base", band=band, ann_blob=None, km_id=km_id)


def _check_real_stream() -> int:
    km = "counters-real"
    capture.drain(km)
    q = np.array([1.0, 0.0, 0.0], dtype="float32")  # top hit is idx 0
    field.retrieve(q, _handle(km), None, None, top_k=2)
    field.retrieve(q, _handle(km), None, None, top_k=2)
    rows = capture.drain(km)
    stats = counters.access_reinforcement(rows)
    top = stats.get(("source", 0))
    if top is None or top.access_count != 2:
        print(f"[counters] real-stream: idx 0 not counted twice: {stats}", file=sys.stderr)
        return 1
    if not (top.reinforcement > 0.0):
        print(f"[counters] real-stream: no reinforcement: {top}", file=sys.stderr)
        return 1
    return 0


def run() -> int:
    for check in [
        _check_empty_and_internal,
        _check_counts_and_keys,
        _check_layer_filter,
        _check_decay,
        _check_log_dampening,
        _check_now_default,
        _check_last_ts_chronological,
        _check_never_raises,
        _check_real_stream,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[counters] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
