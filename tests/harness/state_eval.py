"""state_eval — longitudinal benchmark scorecard contracts.

Pins manifesto §"The longitudinal benchmark" — both axes (useful +
effortless) + secondary signals + cumulative-window comparison rules.
Eight contracts:

  (a) Empty window → useful=0.0 + effortless=inf + 0 touches.

  (b) Useful axis weights by intent_level — a satisfied direction
      counts more than a satisfied step.

  (c) Effortless axis = touches / satisfied; touches = recall fires +
      outcome resolutions.

  (d) Recall hit-rate = recalls-with-rows / total-recalls inside window.

  (e) Memory depth snapshot reads counts by level from the live store.

  (f) Verdict-confidence distribution counts {high, medium, low} from
      criterion checks resolved in the window.

  (g) WindowComparison pass condition — both axes must move correctly.

  (h) `daily_windows(n)` returns N consecutive day-bounded windows in
      chronological order ending today (UTC).

  (l) `aggregate_scorecards` surfaces `memory_depth` as the LAST
      scorecard's snapshot (point-in-time), not the sum across windows.

  (m) `secondary_recall_dropped_at` unifies gate-level `dropped_at` and
      hook-level `status` into one categorical that sums across windows
      in `aggregate_scorecards`.

Hermetic — temp dir + synthetic outcomes + recall entries + memory rows;
no encoder, no LLM.
"""

from __future__ import annotations

import datetime as _dt
import sys
import tempfile
from pathlib import Path

import numpy as np


def _check_empty_window() -> int:
    from resonance_lattice.memory.store import Memory
    from resonance_lattice.state import (
        WindowSpec, compute_session_scorecard,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        scorecard = compute_session_scorecard(
            state_root=Path(td),
            memory=memory,
            window=WindowSpec(
                since="2026-05-08T00:00:00Z",
                until="2026-05-09T00:00:00Z",
            ),
        )
    if scorecard.useful_axis != 0.0:
        print(f"[state_eval] FAIL (a): useful={scorecard.useful_axis!r}",
              file=sys.stderr)
        return 1
    if scorecard.effortless_axis != float("inf"):
        print(f"[state_eval] FAIL (a): effortless={scorecard.effortless_axis!r}",
              file=sys.stderr)
        return 1
    if scorecard.user_touches != 0:
        print(f"[state_eval] FAIL (a): touches={scorecard.user_touches!r}",
              file=sys.stderr)
        return 1
    print("[state_eval] (a) empty window OK", file=sys.stderr)
    return 0


def _check_useful_weights_by_level() -> int:
    from resonance_lattice.memory.store import Memory
    from resonance_lattice.state import (
        Attribution, CriterionCheck, OutcomeLedger, OutcomeRecord,
        WindowSpec, compute_session_scorecard,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        ledger = OutcomeLedger(Path(td))
        # One satisfied step (weight 1) + one satisfied direction
        # (weight 30); useful_axis must reflect the heavier weight.
        for level, intent_id in [
            ("step", "01HZ_S"),
            ("direction", "01HZ_D"),
        ]:
            ledger.write(OutcomeRecord(
                intent_id=intent_id,
                intent_level=level,
                criterion_checks=[CriterionCheck(
                    criterion_text="x", measure="user_confirms",
                    verdict="satisfied",
                )],
                roll_up_verdict="satisfied",
                attribution=[],
                resolved_at="2026-05-08T12:00:00Z",
            ))
        scorecard = compute_session_scorecard(
            state_root=Path(td),
            memory=memory,
            window=WindowSpec(
                since="2026-05-08T00:00:00Z",
                until="2026-05-09T00:00:00Z",
            ),
        )
    # Both intents satisfied — useful_axis = 31/31 = 1.0.
    if abs(scorecard.useful_axis - 1.0) > 1e-9:
        print(f"[state_eval] FAIL (b): useful={scorecard.useful_axis!r}",
              file=sys.stderr)
        return 1
    if scorecard.intents_satisfied_weight != 31.0:
        print(f"[state_eval] FAIL (b): satisfied_weight="
              f"{scorecard.intents_satisfied_weight!r}", file=sys.stderr)
        return 1
    print("[state_eval] (b) useful weights by level OK", file=sys.stderr)
    return 0


def _check_effortless_touches() -> int:
    from resonance_lattice.memory.store import Memory
    from resonance_lattice.state import (
        CriterionCheck, OutcomeLedger, OutcomeRecord,
        RecallCache, RecallEntry, RecallHitMetadata,
        WindowSpec, compute_session_scorecard,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        ledger = OutcomeLedger(Path(td))
        cache = RecallCache(Path(td))

        # 3 user-prompt touches in window + 1 outcome resolution = 4 touches.
        for i in range(3):
            cache.append(RecallEntry(
                turn_id=f"t{i}", timestamp=f"2026-05-08T12:00:0{i}Z",
                prompt_hash=f"h{i}", intent_kind="implement",
                row_metadata=[
                    RecallHitMetadata(row_id=f"r{i}", rank=0, cosine=0.9),
                ],
            ))
        ledger.write(OutcomeRecord(
            intent_id="01HZ_T", intent_level="task",
            criterion_checks=[CriterionCheck(
                criterion_text="x", measure="user_confirms",
                verdict="satisfied",
            )],
            roll_up_verdict="satisfied", attribution=[],
            resolved_at="2026-05-08T12:30:00Z",
        ))
        scorecard = compute_session_scorecard(
            state_root=Path(td),
            memory=memory,
            window=WindowSpec(
                since="2026-05-08T00:00:00Z",
                until="2026-05-09T00:00:00Z",
            ),
        )
    if scorecard.user_touches != 4:
        print(f"[state_eval] FAIL (c): touches={scorecard.user_touches!r}",
              file=sys.stderr)
        return 1
    if scorecard.intents_satisfied_count != 1:
        print(f"[state_eval] FAIL (c): satisfied="
              f"{scorecard.intents_satisfied_count!r}", file=sys.stderr)
        return 1
    if scorecard.effortless_axis != 4.0:
        print(f"[state_eval] FAIL (c): effortless="
              f"{scorecard.effortless_axis!r}", file=sys.stderr)
        return 1
    print("[state_eval] (c) effortless touches OK", file=sys.stderr)
    return 0


def _check_recall_hit_rate() -> int:
    from resonance_lattice.memory.store import Memory
    from resonance_lattice.state import (
        RecallCache, RecallEntry, RecallHitMetadata,
        WindowSpec, compute_session_scorecard,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        cache = RecallCache(Path(td))

        # 2 hits with rows + 1 hit with empty rows = 2/3 hit-rate.
        cache.append(RecallEntry(
            turn_id="t1", timestamp="2026-05-08T12:00:00Z",
            prompt_hash="h1", intent_kind="implement",
            row_metadata=[RecallHitMetadata(row_id="r", rank=0, cosine=0.9)],
        ))
        cache.append(RecallEntry(
            turn_id="t2", timestamp="2026-05-08T12:00:01Z",
            prompt_hash="h2", intent_kind="implement",
            row_metadata=[RecallHitMetadata(row_id="r", rank=0, cosine=0.9)],
        ))
        cache.append(RecallEntry(
            turn_id="t3", timestamp="2026-05-08T12:00:02Z",
            prompt_hash="h3", intent_kind="implement",
            row_metadata=[],
        ))
        scorecard = compute_session_scorecard(
            state_root=Path(td),
            memory=memory,
            window=WindowSpec(
                since="2026-05-08T00:00:00Z",
                until="2026-05-09T00:00:00Z",
            ),
        )
    if abs(scorecard.secondary_recall_hit_rate - 2/3) > 1e-9:
        print(f"[state_eval] FAIL (d): hit_rate="
              f"{scorecard.secondary_recall_hit_rate!r}", file=sys.stderr)
        return 1
    print("[state_eval] (d) recall hit-rate OK", file=sys.stderr)
    return 0


def _check_memory_depth() -> int:
    from resonance_lattice.memory.store import Memory
    from resonance_lattice.state import (
        WindowSpec, compute_session_scorecard,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        # 3 events + 2 patterns + 1 learning.
        for i in range(3):
            memory.add_row(
                text=f"event {i}",
                polarity=["factual", "workspace:abc123"],
                transcript_hash="manual",
                embedding=np.zeros(768, dtype=np.float32),
                level="event",
            )
        for i in range(2):
            memory.add_row(
                text=f"pattern {i}",
                polarity=["factual", "workspace:abc123"],
                transcript_hash="distilled:x",
                embedding=np.zeros(768, dtype=np.float32),
                level="pattern",
                origin="distilled",
            )
        memory.add_row(
            text="learning",
            polarity=["factual", "workspace:abc123"],
            transcript_hash="distilled:y",
            embedding=np.zeros(768, dtype=np.float32),
            level="learning",
            origin="distilled",
        )
        scorecard = compute_session_scorecard(
            state_root=Path(td),
            memory=memory,
            window=WindowSpec(
                since="2026-05-08T00:00:00Z",
                until="2026-05-09T00:00:00Z",
            ),
        )
    expected = {"event": 3, "pattern": 2, "learning": 1}
    if scorecard.secondary_memory_depth_by_level != expected:
        print(f"[state_eval] FAIL (e): depth="
              f"{scorecard.secondary_memory_depth_by_level!r}", file=sys.stderr)
        return 1
    print("[state_eval] (e) memory depth snapshot OK", file=sys.stderr)
    return 0


def _check_verdict_confidence_distribution() -> int:
    from resonance_lattice.memory.store import Memory
    from resonance_lattice.state import (
        CriterionCheck, OutcomeLedger, OutcomeRecord,
        WindowSpec, compute_session_scorecard,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        ledger = OutcomeLedger(Path(td))
        for conf in ["high", "high", "medium", "low"]:
            ledger.write(OutcomeRecord(
                intent_id="t",
                intent_level="task",
                criterion_checks=[CriterionCheck(
                    criterion_text="x", measure="user_confirms",
                    verdict="satisfied", verdict_confidence=conf,
                )],
                roll_up_verdict="satisfied", attribution=[],
                resolved_at="2026-05-08T12:00:00Z",
            ))
        scorecard = compute_session_scorecard(
            state_root=Path(td),
            memory=memory,
            window=WindowSpec(
                since="2026-05-08T00:00:00Z",
                until="2026-05-09T00:00:00Z",
            ),
        )
    expected = {"high": 2, "medium": 1, "low": 1}
    if scorecard.secondary_verdict_confidence != expected:
        print(f"[state_eval] FAIL (f): "
              f"{scorecard.secondary_verdict_confidence!r}", file=sys.stderr)
        return 1
    print("[state_eval] (f) verdict-confidence distribution OK",
          file=sys.stderr)
    return 0


def _check_window_comparison() -> int:
    from resonance_lattice.state import (
        SessionScorecard, WindowSpec, aggregate_windows,
    )

    def _card(satisfied, total, touches, label):
        c = SessionScorecard(window=WindowSpec(
            since="x", until="y", label=label,
        ))
        c.intents_satisfied_count = satisfied
        c.intents_total_count = total
        c.intents_satisfied_weight = float(satisfied)
        c.intents_total_weight = float(total)
        c.user_touches = touches
        return c

    # Early: 2/4 satisfied (useful 0.5), 12 touches (effortless 6.0)
    # Late:  3/4 satisfied (useful 0.75), 9 touches (effortless 3.0)
    # Both axes move correctly → benchmark passes.
    early = [_card(2, 4, 12, "e")]
    late = [_card(3, 4, 9, "l")]
    cards = early + late
    comparison = aggregate_windows(cards, early=(0, 1), late=(1, 2))
    if not comparison.benchmark_passed:
        print(f"[state_eval] FAIL (g.1): benchmark didn't pass",
              file=sys.stderr)
        return 1

    # Useful regresses: late 1/4 < early 2/4 → fail.
    cards2 = [_card(2, 4, 12, "e"), _card(1, 4, 8, "l")]
    comparison2 = aggregate_windows(cards2, early=(0, 1), late=(1, 2))
    if comparison2.useful_passed:
        print(f"[state_eval] FAIL (g.2): useful regression accepted",
              file=sys.stderr)
        return 1
    print("[state_eval] (g) window comparison + benchmark gate OK",
          file=sys.stderr)
    return 0


def _check_aggregate_memory_depth_is_snapshot() -> int:
    """`aggregate_scorecards` (used by `--compare`) must surface
    memory_depth as the LAST scorecard's snapshot, not the sum across
    windows. Memory_depth is a point-in-time count; summing 5 windows
    × 64 events inflates to 320 like run #3's --compare did before the
    fix.
    """
    from resonance_lattice.state import (
        SessionScorecard, WindowSpec, aggregate_windows,
    )

    def _card(depth, label):
        c = SessionScorecard(window=WindowSpec(
            since="x", until="y", label=label,
        ))
        c.secondary_memory_depth_by_level = depth
        return c

    cards = [
        _card({"event": 50, "pattern": 10}, "w0"),
        _card({"event": 55, "pattern": 15}, "w1"),
        _card({"event": 60, "pattern": 20}, "w2"),
    ]
    comparison = aggregate_windows(cards, early=(0, 2), late=(2, 3))
    if comparison.early.secondary_memory_depth_by_level != {
        "event": 55, "pattern": 15,
    }:
        print(f"[state_eval] FAIL (l): early window summed instead of "
              f"snapshot: {comparison.early.secondary_memory_depth_by_level!r}",
              file=sys.stderr)
        return 1
    if comparison.late.secondary_memory_depth_by_level != {
        "event": 60, "pattern": 20,
    }:
        print(f"[state_eval] FAIL (l): late window wrong: "
              f"{comparison.late.secondary_memory_depth_by_level!r}",
              file=sys.stderr)
        return 1
    print("[state_eval] (l) aggregate memory_depth is snapshot OK",
          file=sys.stderr)
    return 0


def _check_daily_windows() -> int:
    from resonance_lattice.state import daily_windows

    # `end_iso` mid-day so the last window must run from today-midnight
    # through end_iso (not floor to yesterday). Earlier N-1 windows are
    # full days walked back from today-midnight.
    end_iso = "2026-05-08T14:30:00Z"
    windows = daily_windows(n_sessions=5, end_iso=end_iso)
    if len(windows) != 5:
        print(f"[state_eval] FAIL (h): n={len(windows)}", file=sys.stderr)
        return 1
    if windows[0].since >= windows[-1].since:
        print(f"[state_eval] FAIL (h): not chronological", file=sys.stderr)
        return 1
    if windows[-1].until != end_iso:
        print(f"[state_eval] FAIL (h): last until={windows[-1].until!r} "
              f"(want {end_iso!r}; today must be in the window)",
              file=sys.stderr)
        return 1
    if windows[-1].since != "2026-05-08T00:00:00Z":
        print(f"[state_eval] FAIL (h): last since={windows[-1].since!r}",
              file=sys.stderr)
        return 1
    # Earlier windows are full days; the last window is partial (today
    # midnight → end_iso). Verify the shape of each.
    for w in windows[:-1]:
        since = _dt.datetime.fromisoformat(w.since.replace("Z", "+00:00"))
        until = _dt.datetime.fromisoformat(w.until.replace("Z", "+00:00"))
        if (until - since) != _dt.timedelta(days=1):
            print(f"[state_eval] FAIL (h): non-day earlier window {w!r}",
                  file=sys.stderr)
            return 1
    last_since = _dt.datetime.fromisoformat(
        windows[-1].since.replace("Z", "+00:00"),
    )
    last_until = _dt.datetime.fromisoformat(
        windows[-1].until.replace("Z", "+00:00"),
    )
    if last_until <= last_since or last_until - last_since > _dt.timedelta(days=1):
        print(f"[state_eval] FAIL (h): last window shape: {windows[-1]!r}",
              file=sys.stderr)
        return 1
    print(f"[state_eval] (h) daily_windows({len(windows)}) — last includes "
          f"today through end_iso OK", file=sys.stderr)
    return 0


def _check_session_marker_windows() -> int:
    """Markers in `sessions.jsonl` override calendar-day windows.

    Two markers + an `end_iso` after both → two windows: marker[0]→marker[1]
    and marker[1]→end_iso. Labels carry the session_id prefix.
    """
    from resonance_lattice.state import SessionMarkerLog, daily_windows

    with tempfile.TemporaryDirectory() as td:
        log = SessionMarkerLog(Path(td))
        log.write(session_id="01HZSESSION0AAAAAAAAAAAAAA",
                  timestamp="2026-05-08T08:00:00Z")
        log.write(session_id="01HZSESSION0BBBBBBBBBBBBBB",
                  timestamp="2026-05-08T16:00:00Z")
        windows = daily_windows(
            n_sessions=5,
            end_iso="2026-05-08T20:00:00Z",
            state_root=Path(td),
        )
    if len(windows) != 2:
        print(f"[state_eval] FAIL (i): n={len(windows)}", file=sys.stderr)
        return 1
    if windows[0].since != "2026-05-08T08:00:00Z":
        print(f"[state_eval] FAIL (i): w0.since={windows[0].since!r}",
              file=sys.stderr)
        return 1
    if windows[0].until != "2026-05-08T16:00:00Z":
        print(f"[state_eval] FAIL (i): w0.until={windows[0].until!r}",
              file=sys.stderr)
        return 1
    if windows[1].since != "2026-05-08T16:00:00Z":
        print(f"[state_eval] FAIL (i): w1.since={windows[1].since!r}",
              file=sys.stderr)
        return 1
    if windows[1].until != "2026-05-08T20:00:00Z":
        print(f"[state_eval] FAIL (i): w1.until={windows[1].until!r}",
              file=sys.stderr)
        return 1
    if not windows[0].label.startswith("session-01HZSESS"):
        print(f"[state_eval] FAIL (i): w0.label={windows[0].label!r}",
              file=sys.stderr)
        return 1
    print("[state_eval] (i) session marker windows OK", file=sys.stderr)
    return 0


def _check_session_marker_fallback_when_absent() -> int:
    """No markers → fall back to calendar-day windows even when state_root
    is supplied. Bridges the empty-state case so callers can always pass
    state_root without short-circuiting the daily-window default."""
    from resonance_lattice.state import daily_windows

    with tempfile.TemporaryDirectory() as td:
        windows_with_root = daily_windows(
            n_sessions=3,
            end_iso="2026-05-08T14:30:00Z",
            state_root=Path(td),
        )
    windows_without_root = daily_windows(
        n_sessions=3,
        end_iso="2026-05-08T14:30:00Z",
    )
    if len(windows_with_root) != 3:
        print(f"[state_eval] FAIL (j): n={len(windows_with_root)}",
              file=sys.stderr)
        return 1
    if [(w.since, w.until) for w in windows_with_root] != \
       [(w.since, w.until) for w in windows_without_root]:
        print("[state_eval] FAIL (j): empty-marker fallback diverges from "
              "default daily_windows", file=sys.stderr)
        return 1
    print("[state_eval] (j) marker-absent fallback OK", file=sys.stderr)
    return 0


def _check_weekly_windows() -> int:
    """(k) `weekly_windows(n_weeks=N, end_iso=T)` returns N consecutive
    7-day windows ending at T. The last window's `until` is exactly T
    (partial-week activity is captured); earlier windows are exact
    7-day blocks walked back. Independent of `daily_windows` — no
    session-marker dependency.
    """
    from resonance_lattice.state import weekly_windows

    end_iso = "2026-05-09T14:30:00Z"
    windows = weekly_windows(n_weeks=2, end_iso=end_iso)
    if len(windows) != 2:
        print(f"[state_eval] FAIL (k): n={len(windows)} (want 2)",
              file=sys.stderr)
        return 1
    if windows[-1].until != end_iso:
        print(f"[state_eval] FAIL (k): last until={windows[-1].until!r} "
              f"(want {end_iso!r})", file=sys.stderr)
        return 1
    # First window is exactly 7 days; second runs from end-7d to end_iso.
    early_since = _dt.datetime.fromisoformat(
        windows[0].since.replace("Z", "+00:00"),
    )
    early_until = _dt.datetime.fromisoformat(
        windows[0].until.replace("Z", "+00:00"),
    )
    if (early_until - early_since) != _dt.timedelta(days=7):
        print(f"[state_eval] FAIL (k): early window not 7 days: "
              f"{windows[0]!r}", file=sys.stderr)
        return 1
    if windows[1].since != windows[0].until:
        print(f"[state_eval] FAIL (k): windows not contiguous: "
              f"{windows[0].until!r} vs {windows[1].since!r}",
              file=sys.stderr)
        return 1

    # n_weeks=4 → 4 contiguous 7-day windows ending at end_iso.
    quad = weekly_windows(n_weeks=4, end_iso=end_iso)
    if len(quad) != 4:
        print(f"[state_eval] FAIL (k): n_weeks=4 returned {len(quad)}",
              file=sys.stderr)
        return 1
    for prev, nxt in zip(quad[:-1], quad[1:]):
        if prev.until != nxt.since:
            print(f"[state_eval] FAIL (k): non-contiguous in 4-window: "
                  f"{prev.until!r} → {nxt.since!r}", file=sys.stderr)
            return 1
    print("[state_eval] (k) weekly_windows: contiguous 7-day blocks ending "
          "at end_iso OK", file=sys.stderr)
    return 0


def _check_dropped_at_distribution() -> int:
    """`secondary_recall_dropped_at` unifies the gate-level `dropped_at`
    (when daemon answered) with the hook-level status (daemon failure).
    The longitudinal-v3 bench surfaced "16/20 sessions had no recall"
    with no way to attribute. This scorecard field gives the head-vs-late
    comparison a categorical it can summarise.
    """
    from resonance_lattice.memory.store import Memory
    from resonance_lattice.state import (
        RecallDiagnosticEntry, RecallDiagnosticLog,
        SessionScorecard, WindowSpec, aggregate_windows,
        compute_session_scorecard,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        log = RecallDiagnosticLog(Path(td))
        # Three daemon-answered + two daemon-failures inside window.
        log.append(RecallDiagnosticEntry(
            turn_id="t1", timestamp="2026-05-08T12:00:00Z",
            prompt_hash="h1", intent_kind="implement", intent_id=None,
            status="ok", n_hits=2,
            diagnostic={"dropped_at": "ok", "n_rows": 50},
        ))
        log.append(RecallDiagnosticEntry(
            turn_id="t2", timestamp="2026-05-08T12:00:01Z",
            prompt_hash="h2", intent_kind="implement", intent_id=None,
            status="no_hit", n_hits=0,
            diagnostic={"dropped_at": "below_recurrence", "n_rows": 50},
        ))
        log.append(RecallDiagnosticEntry(
            turn_id="t3", timestamp="2026-05-08T12:00:02Z",
            prompt_hash="h3", intent_kind="implement", intent_id=None,
            status="no_hit", n_hits=0,
            diagnostic={"dropped_at": "below_recurrence", "n_rows": 50},
        ))
        log.append(RecallDiagnosticEntry(
            turn_id="t4", timestamp="2026-05-08T12:00:03Z",
            prompt_hash="h4", intent_kind="implement", intent_id=None,
            status="daemon_unreachable", n_hits=0, diagnostic=None,
        ))
        log.append(RecallDiagnosticEntry(
            turn_id="t5", timestamp="2026-05-08T12:00:04Z",
            prompt_hash="h5", intent_kind="implement", intent_id=None,
            status="no_store", n_hits=0, diagnostic=None,
        ))
        # Plus one outside the window — must be ignored.
        log.append(RecallDiagnosticEntry(
            turn_id="t6", timestamp="2026-05-07T00:00:00Z",
            prompt_hash="h6", intent_kind="implement", intent_id=None,
            status="ok", n_hits=2,
            diagnostic={"dropped_at": "ok", "n_rows": 50},
        ))
        scorecard = compute_session_scorecard(
            state_root=Path(td),
            memory=memory,
            window=WindowSpec(
                since="2026-05-08T00:00:00Z",
                until="2026-05-09T00:00:00Z",
            ),
        )
    expected = {
        "ok": 1,
        "below_recurrence": 2,
        "daemon_unreachable": 1,
        "no_store": 1,
    }
    if scorecard.secondary_recall_dropped_at != expected:
        print(f"[state_eval] FAIL (m): dropped_at distribution mismatch — "
              f"got {scorecard.secondary_recall_dropped_at!r}, want "
              f"{expected!r}", file=sys.stderr)
        return 1

    # Aggregation across two windows must SUM (categorical counters, not
    # snapshots like memory_depth).
    def _card(dist, label):
        c = SessionScorecard(window=WindowSpec(
            since="x", until="y", label=label,
        ))
        c.secondary_recall_dropped_at = dist
        return c
    cards = [
        _card({"ok": 1, "below_recurrence": 3}, "w0"),
        _card({"ok": 5, "wrong_workspace": 2}, "w1"),
    ]
    comparison = aggregate_windows(cards, early=(0, 1), late=(1, 2))
    if comparison.early.secondary_recall_dropped_at != {
        "ok": 1, "below_recurrence": 3,
    }:
        print(f"[state_eval] FAIL (m): aggregate early — "
              f"{comparison.early.secondary_recall_dropped_at!r}",
              file=sys.stderr)
        return 1
    if comparison.late.secondary_recall_dropped_at != {
        "ok": 5, "wrong_workspace": 2,
    }:
        print(f"[state_eval] FAIL (m): aggregate late — "
              f"{comparison.late.secondary_recall_dropped_at!r}",
              file=sys.stderr)
        return 1
    print("[state_eval] (m) dropped_at distribution + aggregate OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_empty_window,
        _check_useful_weights_by_level,
        _check_effortless_touches,
        _check_recall_hit_rate,
        _check_memory_depth,
        _check_verdict_confidence_distribution,
        _check_window_comparison,
        _check_aggregate_memory_depth_is_snapshot,
        _check_daily_windows,
        _check_session_marker_windows,
        _check_session_marker_fallback_when_absent,
        _check_weekly_windows,
        _check_dropped_at_distribution,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[state_eval] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
