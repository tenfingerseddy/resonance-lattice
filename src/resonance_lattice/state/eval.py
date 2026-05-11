"""Longitudinal eval scaffolding — the measurement substrate the manifesto
demands before the features it measures.

Manifesto §"The longitudinal benchmark":

  20-session dogfood benchmark on the rlat repo itself. Sessions run in
  order; memory accumulates across them. Two scorecard axes captured
  automatically per session:

    Useful     — proportion of intents satisfied per session, weighted by
                 intent level
    Effortless — user touches per intent satisfied (count of user prompts,
                 slash commands, corrections, confirmations)

  Plus three secondary signals tracked but not pass/fail:

    Recall hit-rate — proportion of UserPromptSubmit recalls that produced
                      injected context (vs empty / refused)
    Memory depth    — count of memory rows by level over time; the slope
                      from event to pattern to learning to principle is
                      the development signal
    Verdict-conf    — distribution of high / medium / low verdict
                      confidences over time

  Pass condition (architecture):
    Useful     — `satisfied` proportion at sessions 16–20 strictly greater
                 than at sessions 1–5 (cumulative-window comparison)
    Effortless — touches-per-satisfied-intent at 16–20 strictly less than
                 at 1–5

This module ships the *scaffolding*: scorecard data shape + computation
from existing artifacts (recall_cache + outcomes ledger + memory store).
The actual longitudinal benchmark task set lives outside the substrate
— a separate run-script that invokes Claude Code on the curated tasks.

Sessions are window-bounded by ISO timestamps. v1 default is per-day
windows (one calendar day = one "session") so the timeline is splittable
without hook plumbing for explicit session_id capture. Operators can
supply explicit `(since, until)` pairs when they want richer slicing.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..memory.store import Memory, Row
from .ledger import OutcomeLedger, OutcomeRecord
from .recall_cache import RecallCache, RecallEntry
from .recall_diagnostic import RecallDiagnosticEntry, RecallDiagnosticLog

# Intent-level weights for the "useful" axis. Higher levels count more —
# a satisfied direction is worth more than a satisfied step. Engineering
# spec parameters; tunable without rewriting the manifesto.
INTENT_LEVEL_WEIGHTS: dict[str, float] = {
    "step": 1.0,
    "task": 3.0,
    "goal": 10.0,
    "direction": 30.0,
}


@dataclass(frozen=True)
class WindowSpec:
    """One scorecard window — half-open `[since, until)`."""

    since: str  # ISO 8601
    until: str
    label: str = ""  # optional label for human-readable output


@dataclass
class SessionScorecard:
    """One scorecard. The manifesto's two axes + three secondary signals.

    `useful` and `effortless` are the pass/fail axes; the three
    `secondary_*` fields are tracked but don't gate the pass condition.
    """

    window: WindowSpec
    intents_satisfied_count: int = 0
    intents_total_count: int = 0
    intents_satisfied_weight: float = 0.0
    intents_total_weight: float = 0.0
    user_touches: int = 0
    secondary_recall_hits_with_context: int = 0
    secondary_recall_hits_total: int = 0
    secondary_memory_depth_by_level: dict[str, int] = field(default_factory=dict)
    secondary_verdict_confidence: dict[str, int] = field(default_factory=dict)
    # Per-recall outcome distribution from `recall_diagnostic.jsonl`.
    # Categorical key is the gate-level `dropped_at` (ok / below_cosine_floor
    # / wrong_workspace / below_confidence_gap / below_recurrence / no_rows)
    # when the daemon answered, else the hook-level status (no_store /
    # daemon_unreachable / daemon_error). One unified categorical so the
    # head-vs-late comparison can say "early sessions missed for X, late
    # missed for Y" in one signal.
    secondary_recall_dropped_at: dict[str, int] = field(default_factory=dict)

    @property
    def useful_axis(self) -> float:
        """Weighted proportion of intents satisfied. 0.0 when no intents
        landed in the window — this is "no signal", not "0% useful"."""
        if self.intents_total_weight == 0:
            return 0.0
        return self.intents_satisfied_weight / self.intents_total_weight

    @property
    def effortless_axis(self) -> float:
        """User touches per satisfied intent (lower is better). Returns
        infinity when zero satisfied intents — caller should treat that
        as "no signal" not "infinite friction"."""
        if self.intents_satisfied_count == 0:
            return float("inf")
        return self.user_touches / self.intents_satisfied_count

    @property
    def secondary_recall_hit_rate(self) -> float:
        """Proportion of recalls that surfaced rows. 0.0 means no recall
        ever fired in the window."""
        if self.secondary_recall_hits_total == 0:
            return 0.0
        return (
            self.secondary_recall_hits_with_context
            / self.secondary_recall_hits_total
        )


@dataclass
class WindowComparison:
    """Cumulative-window comparison — early vs late.

    Manifesto pass condition: `useful_late > useful_early` AND
    `effortless_late < effortless_early`. Both must hold; one without the
    other is a regression.
    """

    early: SessionScorecard
    late: SessionScorecard

    @property
    def useful_passed(self) -> bool:
        return self.late.useful_axis > self.early.useful_axis

    @property
    def effortless_passed(self) -> bool:
        # `inf` from "no satisfied intents" never lets effortless pass —
        # which is the right behaviour: a window with no satisfied
        # intents can't claim friction reduction.
        return self.late.effortless_axis < self.early.effortless_axis

    @property
    def benchmark_passed(self) -> bool:
        return self.useful_passed and self.effortless_passed


def _within_window(ts: str, window: WindowSpec) -> bool:
    return window.since <= ts < window.until


def _row_count_by_level(rows: list[Row]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.level] = counts.get(row.level, 0) + 1
    return counts


def _dropped_at_distribution(
    diagnostics: list[RecallDiagnosticEntry], window: WindowSpec,
) -> dict[str, int]:
    """Count recall outcomes by unified `dropped_at` / `status` category.

    For entries where the daemon answered (`status` in {ok, no_hit}),
    the bucket is the gate-level `dropped_at` from the serialised
    RankDiagnostic. For entries where the daemon failed (status in
    {daemon_unreachable, daemon_error, no_store}), the bucket is the
    hook-level status itself. Surface a single categorical so the
    head-vs-late comparison can say "early sessions missed for X, late
    missed for Y."
    """
    counts: dict[str, int] = {}
    for entry in diagnostics:
        if not _within_window(entry.timestamp, window):
            continue
        if entry.diagnostic is not None and "dropped_at" in entry.diagnostic:
            key = str(entry.diagnostic["dropped_at"])
        else:
            key = entry.status
        counts[key] = counts.get(key, 0) + 1
    return counts


def _verdict_confidence_distribution(
    outcomes: list[OutcomeRecord], window: WindowSpec,
) -> dict[str, int]:
    """Count {high, medium, low} verdict confidences across criterion
    checks resolved in the window."""
    counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for record in outcomes:
        if not _within_window(record.resolved_at, window):
            continue
        for cc in record.criterion_checks:
            counts[cc.verdict_confidence] = (
                counts.get(cc.verdict_confidence, 0) + 1
            )
    return counts


def compute_session_scorecard(
    state_root: Path | str,
    *,
    memory: Memory,
    window: WindowSpec,
    outcomes: list[OutcomeRecord] | None = None,
    recalls: list[RecallEntry] | None = None,
    diagnostics: list[RecallDiagnosticEntry] | None = None,
    memory_depth: dict[str, int] | None = None,
) -> SessionScorecard:
    """Compute one scorecard from existing artifacts.

    Reads outcomes + recalls + diagnostics + memory state when not
    supplied. Callers aggregating across multiple windows pass the
    pre-loaded lists to avoid O(N) full-file reparse per window — the
    longitudinal eval is the only realistic multi-window caller, and
    that's the path the optimisation matters on.

    No hook plumbing required — pure aggregation over what the closed
    loop already writes during normal operation.
    """
    state_root = Path(state_root)
    if outcomes is None:
        outcomes = list(OutcomeLedger(state_root).iter_records())
    if recalls is None:
        recalls = RecallCache(state_root).read_recent(limit=None)
    if diagnostics is None:
        diagnostics = RecallDiagnosticLog(state_root).read_recent(limit=None)

    intents_satisfied_count = 0
    intents_total_count = 0
    intents_satisfied_weight = 0.0
    intents_total_weight = 0.0
    for record in outcomes:
        if not _within_window(record.resolved_at, window):
            continue
        weight = INTENT_LEVEL_WEIGHTS.get(record.intent_level, 1.0)
        intents_total_count += 1
        intents_total_weight += weight
        if record.roll_up_verdict == "satisfied":
            intents_satisfied_count += 1
            intents_satisfied_weight += weight

    # User touches v1 = UserPromptSubmit fires (one cache entry per prompt)
    # plus intent-resolution events (each accept/reject is one touch).
    # `corrections` (/correct) and inline confirmations land later.
    prompt_touches = sum(
        1 for r in recalls if _within_window(r.timestamp, window)
    )
    resolution_touches = sum(
        1 for r in outcomes if _within_window(r.resolved_at, window)
    )
    user_touches = prompt_touches + resolution_touches

    # Recall hit-rate: per cache entry, "with context" means at least one
    # row surfaced (row_metadata non-empty).
    recalls_in_window = [
        r for r in recalls if _within_window(r.timestamp, window)
    ]
    hits_total = len(recalls_in_window)
    hits_with_context = sum(
        1 for r in recalls_in_window if r.row_metadata
    )

    # Memory depth — point-in-time snapshot, not window-bounded.
    if memory_depth is None:
        rows, _ = memory.read_all()
        memory_depth = _row_count_by_level(rows)

    return SessionScorecard(
        window=window,
        intents_satisfied_count=intents_satisfied_count,
        intents_total_count=intents_total_count,
        intents_satisfied_weight=intents_satisfied_weight,
        intents_total_weight=intents_total_weight,
        user_touches=user_touches,
        secondary_recall_hits_with_context=hits_with_context,
        secondary_recall_hits_total=hits_total,
        secondary_memory_depth_by_level=memory_depth,
        secondary_verdict_confidence=_verdict_confidence_distribution(
            outcomes, window,
        ),
        secondary_recall_dropped_at=_dropped_at_distribution(
            diagnostics, window,
        ),
    )


def daily_windows(
    *,
    n_sessions: int,
    end_iso: str | None = None,
    state_root: Path | str | None = None,
) -> list[WindowSpec]:
    """Build N windows ending at `end_iso`.

    When `state_root` is supplied AND `<state_root>/ledger/sessions.jsonl`
    contains markers, windows are bounded by consecutive markers — the
    trailing N marker-to-marker (or marker-to-end) intervals. The window
    label is `session-<session_id-prefix>` so the scorecard slices per
    session. Sessions that span midnight stay coherent; multiple sessions
    on one day each get their own scorecard.

    Otherwise falls back to N consecutive day-bounded windows. The LAST
    window ends at `end_iso` itself (default: now, UTC) so activity
    earlier today appears in the eval; earlier windows are midnight-
    aligned and walked back from there.
    """
    if end_iso is None:
        end_dt = _dt.datetime.now(_dt.timezone.utc)
    else:
        end_dt = _dt.datetime.fromisoformat(
            end_iso.replace("Z", "+00:00") if end_iso.endswith("Z") else end_iso,
        )
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=_dt.timezone.utc)
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if state_root is not None:
        # Local import so eval.py doesn't depend on sessions at import time
        # (sessions.py imports from .ledger; .ledger imports from memory —
        # keeping this lazy avoids surprising import cycles for callers
        # that only want the calendar-day path).
        from .sessions import SessionMarkerLog

        markers = SessionMarkerLog(state_root).read_all()
        # Drop markers ≥ end so the last interval ends at end_iso, not at
        # a future marker. Take the trailing N — earlier sessions are
        # silently dropped because the scorecard surface is N-bounded.
        markers = [m for m in markers if m.timestamp < end_str]
        if markers:
            tail = markers[-n_sessions:]
            session_windows: list[WindowSpec] = []
            for i, m in enumerate(tail):
                until = (
                    tail[i + 1].timestamp if i + 1 < len(tail) else end_str
                )
                session_windows.append(WindowSpec(
                    since=m.timestamp,
                    until=until,
                    label=f"session-{m.session_id[:8]}",
                ))
            return session_windows

    midnight = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    windows: list[WindowSpec] = []
    # Earlier windows: full days from midnight to midnight, walked back.
    for i in range(n_sessions - 1, 0, -1):
        start = midnight - _dt.timedelta(days=i)
        stop = start + _dt.timedelta(days=1)
        windows.append(WindowSpec(
            since=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            until=stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
            label=start.strftime("session-%Y-%m-%d"),
        ))
    # Last window: today-midnight → end_iso (now). Captures partial-day
    # activity so the smoke surfaces immediately.
    windows.append(WindowSpec(
        since=midnight.strftime("%Y-%m-%dT%H:%M:%SZ"),
        until=end_str,
        label=midnight.strftime("session-%Y-%m-%d"),
    ))
    return windows


def weekly_windows(
    *,
    n_weeks: int = 2,
    end_iso: str | None = None,
) -> list[WindowSpec]:
    """Build N consecutive 7-day windows ending at `end_iso`.

    The default `n_weeks=2` returns `[prev_week, current_week]` — the
    shape `rlat memory rollup` aggregates into "did this week beat
    last week?" The last window ends at `end_iso` itself (default: now)
    so partial-week activity appears immediately; earlier windows are
    full 7-day blocks walked back from there.

    Independent of `daily_windows`: weekly rollups don't depend on
    session markers and don't try to slice by session_id — the unit is
    the calendar week, not the work session.
    """
    if end_iso is None:
        end_dt = _dt.datetime.now(_dt.timezone.utc)
    else:
        end_dt = _dt.datetime.fromisoformat(
            end_iso.replace("Z", "+00:00") if end_iso.endswith("Z") else end_iso,
        )
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=_dt.timezone.utc)
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    week = _dt.timedelta(days=7)
    windows: list[WindowSpec] = []
    for i in range(n_weeks - 1, 0, -1):
        start = end_dt - week * (i + 1)
        stop = end_dt - week * i
        windows.append(WindowSpec(
            since=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            until=stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
            label=start.strftime("week-of-%Y-%m-%d"),
        ))
    last_start = end_dt - week
    windows.append(WindowSpec(
        since=last_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        until=end_str,
        label=last_start.strftime("week-of-%Y-%m-%d"),
    ))
    return windows


def aggregate_windows(
    scorecards: list[SessionScorecard],
    *,
    early: tuple[int, int] = (0, 5),
    late: tuple[int, int] = (15, 20),
) -> WindowComparison:
    """Compute the cumulative-window comparison.

    `early` and `late` are half-open index ranges into `scorecards`.
    Each window aggregates by summing counts/weights and concatenating
    secondary distributions. The manifesto's pass condition is checked
    on the resulting scorecards' `useful_axis` and `effortless_axis`.
    """
    if not scorecards:
        empty_window = WindowSpec(since="", until="", label="empty")
        empty = SessionScorecard(window=empty_window)
        return WindowComparison(early=empty, late=empty)
    early_cards = scorecards[early[0]:early[1]]
    late_cards = scorecards[late[0]:late[1]]
    return WindowComparison(
        early=_aggregate(early_cards, label=f"early[{early[0]}:{early[1]}]"),
        late=_aggregate(late_cards, label=f"late[{late[0]}:{late[1]}]"),
    )


def _aggregate(
    scorecards: list[SessionScorecard], *, label: str,
) -> SessionScorecard:
    if not scorecards:
        return SessionScorecard(
            window=WindowSpec(since="", until="", label=label),
        )
    out = SessionScorecard(
        window=WindowSpec(
            since=scorecards[0].window.since,
            until=scorecards[-1].window.until,
            label=label,
        ),
    )
    verdict_acc: dict[str, int] = {}
    dropped_acc: dict[str, int] = {}
    for card in scorecards:
        out.intents_satisfied_count += card.intents_satisfied_count
        out.intents_total_count += card.intents_total_count
        out.intents_satisfied_weight += card.intents_satisfied_weight
        out.intents_total_weight += card.intents_total_weight
        out.user_touches += card.user_touches
        out.secondary_recall_hits_with_context += (
            card.secondary_recall_hits_with_context
        )
        out.secondary_recall_hits_total += card.secondary_recall_hits_total
        for k, v in card.secondary_verdict_confidence.items():
            verdict_acc[k] = verdict_acc.get(k, 0) + v
        for k, v in card.secondary_recall_dropped_at.items():
            dropped_acc[k] = dropped_acc.get(k, 0) + v
    # memory_depth is a point-in-time snapshot, not an additive counter
    # — summing across windows inflates the count by the number of
    # scorecards aggregated (5 windows × 64 events = 320 reported, where
    # actual store has 64). The last scorecard's snapshot reflects the
    # end-state of the aggregate window.
    out.secondary_memory_depth_by_level = dict(
        scorecards[-1].secondary_memory_depth_by_level
    )
    out.secondary_verdict_confidence = verdict_acc
    out.secondary_recall_dropped_at = dropped_acc
    return out


def render_summary(scorecard: SessionScorecard) -> str:
    """One-block human-readable summary of a single scorecard."""
    label = scorecard.window.label or scorecard.window.since
    lines = [
        f"Scorecard [{label}]",
        f"  useful_axis     {scorecard.useful_axis:.3f} "
        f"({scorecard.intents_satisfied_count}/{scorecard.intents_total_count} "
        f"intents satisfied, weighted)",
        f"  effortless_axis {scorecard.effortless_axis:.3f} "
        f"({scorecard.user_touches} touches / "
        f"{scorecard.intents_satisfied_count} satisfied)",
        f"  recall_hit_rate {scorecard.secondary_recall_hit_rate:.3f} "
        f"({scorecard.secondary_recall_hits_with_context}/"
        f"{scorecard.secondary_recall_hits_total} recalls surfaced rows)",
        f"  memory_depth    {scorecard.secondary_memory_depth_by_level}",
        f"  verdict_conf    {scorecard.secondary_verdict_confidence}",
        f"  dropped_at      {scorecard.secondary_recall_dropped_at}",
    ]
    return "\n".join(lines)


def render_comparison(comparison: WindowComparison) -> str:
    """Comparison view — early vs late + benchmark pass/fail."""
    lines = [
        render_summary(comparison.early),
        "",
        render_summary(comparison.late),
        "",
        f"useful  {comparison.early.useful_axis:.3f} → "
        f"{comparison.late.useful_axis:.3f} "
        f"({'PASS' if comparison.useful_passed else 'FAIL'})",
        f"effort  {comparison.early.effortless_axis:.3f} → "
        f"{comparison.late.effortless_axis:.3f} "
        f"({'PASS' if comparison.effortless_passed else 'FAIL'})",
        f"benchmark {'PASS' if comparison.benchmark_passed else 'FAIL'}",
    ]
    return "\n".join(lines)


def scorecard_to_dict(scorecard: SessionScorecard) -> dict:
    """JSON-serialisable snapshot — used by the eval CLI's --json output."""
    return asdict(scorecard) | {
        "useful_axis": scorecard.useful_axis,
        "effortless_axis": (
            scorecard.effortless_axis
            if scorecard.effortless_axis != float("inf")
            else None
        ),
        "secondary_recall_hit_rate": scorecard.secondary_recall_hit_rate,
    }
