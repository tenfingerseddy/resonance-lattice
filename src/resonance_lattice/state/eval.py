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
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .claim import Claim
from .claim_outcome import ClaimOutcomeLog, ClaimOutcomeRecord
from .recall_cache import RecallCache, RecallEntry
from .recall_diagnostic import RecallDiagnosticEntry, RecallDiagnosticLog

if TYPE_CHECKING:  # annotations only — `state` must not import `memory` at
    # module load (memory.claim_store imports state.claim → import cycle).
    from ..memory.claim_store import ExperienceClaimStore

# Intent-level weights for the "useful" axis. Higher levels count more —
# a satisfied direction is worth more than a satisfied step. Engineering
# spec parameters; tunable without rewriting the manifesto.
INTENT_LEVEL_WEIGHTS: dict[str, float] = {
    "step": 1.0,
    "task": 3.0,
    "goal": 10.0,
    "direction": 30.0,
}

# Minimum effect size for a paired-bench useful axis to count as
# passing. Conventional Cohen's-d thresholds: small=0.2, medium=0.5,
# large=0.8. Anything below 0.2 is noise even when the directional
# count is correct — v5_paired had +4/-3 wins with cohen_d_z=0.019
# (mean useful delta +0.006); nominally "passes" mean>0 + wins>losses
# but the effect is binomial coin-flip. v5_paired2 cleared this bar
# at d_z=0.318 (mean +0.094, 5/1 wins) — measurable signal above noise.
MIN_USEFUL_COHEN_D_Z = 0.2


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


@dataclass(frozen=True)
class PairedComparison:
    """Per-session paired comparison — arm_on vs arm_off on the same prompts.

    Single-arm comparisons (`WindowComparison`) are dominated by
    cross-run variance: repeated single-arm runs of the SAME prompt
    set produced different verdicts purely from sampling noise. Paired
    runs cancel that variance — the same Sonnet randomness, intent
    ordering, and live state appear in both arms, so the per-session
    delta is the substrate's measured contribution to that session.

    Pass condition: `mean_useful_delta > 0` AND `n_useful_positive >
    n_useful_negative` AND `cohen_d_z_useful >= MIN_USEFUL_COHEN_D_Z`.
    The mean is the magnitude; the directional count is the second-
    order check that the mean isn't driven by one or two outliers;
    the effect-size gate prevents nominal "passes" on +1 vs -0
    coin-flips with vanishing mean delta.

    `cohen_d_z` is the effect size for a paired design: `mean(deltas) /
    stdev(deltas)`. df-free so callers don't need a critical-value
    table; conventional thresholds are 0.2 / 0.5 / 0.8 (small / medium
    / large).

    Only the two raw delta lists are stored; everything else is derived
    on access so the dataclass can't drift out of sync with its inputs.
    """

    per_session_useful_deltas: list[float]
    per_session_effortless_deltas: list[float]

    @property
    def n_sessions(self) -> int:
        return len(self.per_session_useful_deltas)

    @property
    def mean_useful_delta(self) -> float:
        return (
            sum(self.per_session_useful_deltas) / self.n_sessions
            if self.n_sessions else 0.0
        )

    @property
    def mean_effortless_delta(self) -> float:
        finite = [d for d in self.per_session_effortless_deltas if d == d]
        return sum(finite) / len(finite) if finite else 0.0

    @property
    def n_useful_positive(self) -> int:
        return sum(1 for d in self.per_session_useful_deltas if d > 0)

    @property
    def n_useful_negative(self) -> int:
        return sum(1 for d in self.per_session_useful_deltas if d < 0)

    @property
    def n_useful_zero(self) -> int:
        return sum(1 for d in self.per_session_useful_deltas if d == 0)

    @property
    def cohen_d_z_useful(self) -> float | None:
        if self.n_sessions < 2:
            return None
        sd = statistics.pstdev(self.per_session_useful_deltas)
        return self.mean_useful_delta / sd if sd > 0 else None

    @property
    def useful_passed(self) -> bool:
        d_z = self.cohen_d_z_useful
        return (
            self.mean_useful_delta > 0
            and self.n_useful_positive > self.n_useful_negative
            and d_z is not None
            and d_z >= MIN_USEFUL_COHEN_D_Z
        )

    @property
    def effortless_passed(self) -> bool:
        # Lower effortless is better; positive delta = on arm had MORE
        # touches per satisfied intent than off (worse). So pass = mean
        # delta < 0.
        return self.mean_effortless_delta < 0

    @property
    def benchmark_passed(self) -> bool:
        return self.useful_passed and self.effortless_passed


def paired_comparison(
    arm_on: list[SessionScorecard], arm_off: list[SessionScorecard],
) -> PairedComparison:
    """Join two arms by index, compute per-session deltas.

    Both lists MUST be in the same session order (session 1 first, then
    2, …). Length mismatch raises ValueError — the paired design demands
    same-N. Sessions where either arm has `effortless_axis = inf` (zero
    satisfied intents) contribute NaN to the effortless delta list and
    are skipped in the effortless mean — they carry no friction signal.
    """
    if len(arm_on) != len(arm_off):
        raise ValueError(
            f"paired arms must have same length; "
            f"on={len(arm_on)} off={len(arm_off)}"
        )
    useful_deltas = [
        on.useful_axis - off.useful_axis
        for on, off in zip(arm_on, arm_off)
    ]
    effortless_deltas: list[float] = []
    for on, off in zip(arm_on, arm_off):
        if on.effortless_axis == float("inf") or off.effortless_axis == float("inf"):
            effortless_deltas.append(float("nan"))
        else:
            effortless_deltas.append(on.effortless_axis - off.effortless_axis)
    return PairedComparison(
        per_session_useful_deltas=useful_deltas,
        per_session_effortless_deltas=effortless_deltas,
    )


def scorecard_from_step_eval(
    *,
    n_satisfied_steps: int,
    n_total_steps: int,
    task_satisfied: bool,
    label: str = "",
) -> SessionScorecard:
    """Build a one-session SessionScorecard from a task+steps eval shape.

    The longitudinal bench's `s{N}_eval_steps.json` carries one task +
    N steps per session; this helper applies INTENT_LEVEL_WEIGHTS so
    bench callers don't reach into SessionScorecard internals and so
    the weighting stays colocated with the rest of the eval module.

    `user_touches` is approximated as `1 + n_total_steps` — one claude
    -p call + one accept/reject per step. Consistent across both arms
    so paired effortless deltas remain meaningful (the absolute value
    differs from the full closed-loop touches metric, but the delta
    cancels the offset).
    """
    task_weight = INTENT_LEVEL_WEIGHTS["task"]
    step_weight = INTENT_LEVEL_WEIGHTS["step"]
    total_weight = task_weight + step_weight * n_total_steps
    satisfied_weight = (
        (task_weight if task_satisfied else 0.0)
        + step_weight * n_satisfied_steps
    )
    return SessionScorecard(
        window=WindowSpec(since="", until="", label=label),
        intents_satisfied_count=(1 if task_satisfied else 0) + n_satisfied_steps,
        intents_total_count=1 + n_total_steps,
        intents_satisfied_weight=satisfied_weight,
        intents_total_weight=total_weight,
        user_touches=1 + n_total_steps,
    )


def render_paired_comparison(comparison: PairedComparison) -> str:
    """Human-readable paired comparison block."""
    d_z_str = (
        f"{comparison.cohen_d_z_useful:.3f}"
        if comparison.cohen_d_z_useful is not None
        else "n/a"
    )
    lines = [
        f"PairedComparison (n={comparison.n_sessions})",
        f"  useful_delta_mean     {comparison.mean_useful_delta:+.4f} "
        f"(cohen_d_z={d_z_str})",
        f"  useful_direction      "
        f"+{comparison.n_useful_positive} / "
        f"-{comparison.n_useful_negative} / "
        f"={comparison.n_useful_zero}",
        f"  effortless_delta_mean {comparison.mean_effortless_delta:+.4f} "
        f"(lower-is-better; <0 means on-arm easier)",
        f"  useful     {'PASS' if comparison.useful_passed else 'FAIL'}",
        f"  effortless {'PASS' if comparison.effortless_passed else 'FAIL'}",
        f"  benchmark  {'PASS' if comparison.benchmark_passed else 'FAIL'}",
    ]
    return "\n".join(lines)


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


def _claim_count_by_level(claims: list[Claim]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        counts[claim.kind] = counts.get(claim.kind, 0) + 1
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
    outcomes: list[ClaimOutcomeRecord], window: WindowSpec,
) -> dict[str, int]:
    """Count {high, medium, low} verdict confidences across criterion
    checks resolved in the window."""
    counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for record in outcomes:
        if not _within_window(record.resolved_at, window):
            continue
        for cc in record.details.criterion_checks:
            counts[cc.verdict_confidence] = (
                counts.get(cc.verdict_confidence, 0) + 1
            )
    return counts


def compute_session_scorecard(
    state_root: Path | str,
    *,
    memory: ExperienceClaimStore,
    window: WindowSpec,
    outcomes: list[ClaimOutcomeRecord] | None = None,
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
    # `intent`-kind only — the scorecard's useful/effortless axes count
    # intents; a `session`-kind record would double-count the session
    # alongside its constituent intents. Filter both the self-read and a
    # caller-supplied list so the contract holds whatever the source.
    if outcomes is None:
        outcomes = list(
            ClaimOutcomeLog(state_root).iter_records(kind="intent")
        )
    else:
        outcomes = [o for o in outcomes if o.kind == "intent"]
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
        weight = INTENT_LEVEL_WEIGHTS.get(record.details.intent_level, 1.0)
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
        memory_depth = _claim_count_by_level(memory.read_all())

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
        # (sessions.py imports from ..memory._common; keeping this lazy
        # avoids pulling memory into state at module load — the same
        # cycle the TYPE_CHECKING note above guards against).
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
