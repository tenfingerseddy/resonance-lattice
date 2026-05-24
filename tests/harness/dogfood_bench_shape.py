"""dogfood_bench_shape — substrate validation for Claim 1 measurement.

We can't simulate the public Claim 1 result. What we CAN do is verify the
bench correctly distinguishes a compounding loop from a non-compounding one
when fed seeded data — that validates the measurement; the claim itself
stays gated on real usage.

The quality axis is mean answer faithfulness per session (not a user
accept/reject verdict — see docs/internal/GROUNDING_MODEL.md).

Guarantees:

  1. No compounding (constant duration, constant faithfulness) -> FAIL.
  2. Compounding on both axes (duration drops + faithfulness rises) -> PASS.
  3. Speed compounds but quality doesn't -> FAIL.
  4. Quality compounds but speed doesn't -> FAIL.
  5. With <20 sessions, claim_1_passes stays False even when both axes
     are moving correctly (the roadmap's 20-session bar is hard).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _make_event(day_offset: int, session_id: str, duration_ms: int,
                faithfulness: float) -> dict:
    ts = (datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
          + timedelta(days=day_offset, minutes=duration_ms // 60))
    return {
        "timestamp": ts.isoformat(timespec="seconds"),
        "session_id": session_id,
        "km_path": "synthetic.rlat",
        "query": f"synthetic query {duration_ms}",
        "duration_ms": duration_ms,
        "insight_hits": 1,
        "source_hits": 5,
        "faithfulness": faithfulness,
        "intent_context": None,
        "lens_id": None,
    }


def _bench_module():
    """Load the bench module by path. Inject into sys.modules first so
    dataclass introspection (which walks sys.modules to resolve type
    references in the class body) finds it during exec."""
    import importlib.util
    import sys as _sys
    if "bench_lensed_dogfood" in _sys.modules:
        return _sys.modules["bench_lensed_dogfood"]
    bench_path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_lensed_dogfood.py"
    spec = importlib.util.spec_from_file_location("bench_lensed_dogfood", bench_path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["bench_lensed_dogfood"] = mod
    spec.loader.exec_module(mod)
    return mod


def _scorecard_from_events(events_dicts: list[dict]):
    bench = _bench_module()
    events = [bench.DogfoodEvent(**d) for d in events_dicts]
    return bench.compute_scorecard(events)


def _flat_session(day: int, duration_ms: int, n_queries: int,
                  faithfulness: float) -> list[dict]:
    """Build one session's events, all at the given faithfulness level."""
    sess_id = f"2026-05-{day:02d}"
    return [
        _make_event(day, sess_id, duration_ms, faithfulness)
        for _ in range(n_queries)
    ]


def run() -> int:
    failures = 0

    # ---- Guarantee 1: no compounding → FAIL ----
    events: list[dict] = []
    for day in range(1, 21):
        events.extend(_flat_session(day, duration_ms=1000, n_queries=5,
                                     faithfulness=0.6))
    sc = _scorecard_from_events(events)
    if sc.claim_1_passes:
        print(f"[shape] FAIL g1: flat events claimed pass "
              f"(speed_delta={sc.speed_delta} quality_delta={sc.quality_delta})",
              file=sys.stderr)
        failures += 1
    elif sc.speed_delta != 0 or sc.quality_delta != 0:
        print(f"[shape] FAIL g1: flat events produced non-zero deltas "
              f"({sc.speed_delta}, {sc.quality_delta})", file=sys.stderr)
        failures += 1
    else:
        print("[shape] g1 (no compounding -> FAIL) OK", file=sys.stderr)

    # ---- Guarantee 2: both axes compound -> PASS ----
    events = []
    for day in range(1, 21):
        # Speed drops from 2000ms day 1 to 500ms day 20.
        dur = int(2000 - (day - 1) * (1500 / 19))
        # Faithfulness rises from 0.30 day 1 to 0.80 day 20.
        faith = 0.30 + (day - 1) * (0.50 / 19)
        events.extend(_flat_session(day, duration_ms=dur, n_queries=5,
                                     faithfulness=faith))
    sc = _scorecard_from_events(events)
    if not sc.claim_1_passes:
        print(f"[shape] FAIL g2: compounding loop did not pass "
              f"(speed_delta={sc.speed_delta} quality_delta={sc.quality_delta})",
              file=sys.stderr)
        failures += 1
    elif sc.speed_delta >= 0:
        print(f"[shape] FAIL g2: speed_delta should be negative "
              f"({sc.speed_delta})", file=sys.stderr)
        failures += 1
    elif sc.quality_delta <= 0:
        print(f"[shape] FAIL g2: quality_delta should be positive "
              f"({sc.quality_delta})", file=sys.stderr)
        failures += 1
    else:
        print(f"[shape] g2 (both axes compound -> PASS) OK "
              f"(speed_delta={sc.speed_delta:+.1f}ms, "
              f"quality_delta={sc.quality_delta:+.3f})", file=sys.stderr)

    # ---- Guarantee 3: speed-only compounding -> FAIL ----
    events = []
    for day in range(1, 21):
        dur = int(2000 - (day - 1) * (1500 / 19))
        events.extend(_flat_session(day, duration_ms=dur, n_queries=5,
                                     faithfulness=0.5))  # flat quality
    sc = _scorecard_from_events(events)
    if sc.claim_1_passes:
        print(f"[shape] FAIL g3: speed-only compounding passed "
              f"({sc.quality_delta})", file=sys.stderr)
        failures += 1
    else:
        print("[shape] g3 (speed-only compounding -> FAIL) OK", file=sys.stderr)

    # ---- Guarantee 4: quality-only compounding -> FAIL ----
    events = []
    for day in range(1, 21):
        faith = 0.30 + (day - 1) * (0.50 / 19)
        events.extend(_flat_session(day, duration_ms=1000, n_queries=5,
                                     faithfulness=faith))
    sc = _scorecard_from_events(events)
    if sc.claim_1_passes:
        print(f"[shape] FAIL g4: quality-only compounding passed "
              f"({sc.speed_delta})", file=sys.stderr)
        failures += 1
    else:
        print("[shape] g4 (quality-only compounding -> FAIL) OK", file=sys.stderr)

    # ---- Guarantee 5: <20 sessions stays False even if compounding ----
    events = []
    for day in range(1, 16):  # only 15 sessions
        dur = int(2000 - (day - 1) * 100)
        faith = 0.30 + (day - 1) * 0.04
        events.extend(_flat_session(day, duration_ms=dur, n_queries=5,
                                     faithfulness=faith))
    sc = _scorecard_from_events(events)
    if sc.claim_1_passes:
        print(f"[shape] FAIL g5: 15 sessions claimed pass "
              f"(n_sessions={sc.n_sessions})", file=sys.stderr)
        failures += 1
    else:
        print(f"[shape] g5 (15 sessions: claim_1 stays False) OK "
              f"(speed_delta={sc.speed_delta:+.0f}ms, "
              f"quality_delta={sc.quality_delta:+.3f}, n_sessions={sc.n_sessions})",
              file=sys.stderr)

    if failures:
        print(f"[shape] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[shape] all guarantees OK — Claim 1 measurement is trustworthy",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
