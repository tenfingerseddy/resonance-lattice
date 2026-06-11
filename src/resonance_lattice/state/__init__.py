"""Agent-state primitives — workspace identity, live intent, outcome ledger.

Per architecture §"Workspace" / §"Intent" / §"Outcomes". These are the new
state surfaces the Horizon 1 spike adds on top of the v2.2 memory store:

- `workspace.resolve_workspace(cwd)` — git-default identity with declaration override.
- `intent` — live intent graph at `<workspace-root>/.rlat-state/intent/`.
- `ledger` — append-only outcome records at `<workspace-root>/.rlat-state/ledger/`.

Durable memory + durable intent stay in the per-user store at
`~/.rlat/memory/<user-id>/`; only the *live* per-workspace state lives here.
"""

from .claim import (
    DISTILLED_PREFIX,
    EXPERIENCE_KINDS,
    MANUAL_TRANSCRIPT_HASH,
    MIGRATED_PREFIX,
    PRIMARY_POLARITY,
    Claim,
    ClaimKind,
    ClaimSource,
    ClaimState,
    CorpusFacts,
    ExperienceFacts,
    PrimaryPolarity,
    derive_origin,
    evolve,
)
from .intent import (
    DurableIntentStore,
    Intent,
    LiveIntentStore,
    durable_intent_dir,
    intent_dir,
    make_intent_id,
)
from .claim_outcome import (
    Attribution,
    ClaimOutcomeLog,
    ClaimOutcomeRecord,
    CriterionCheck,
    IntentOutcomeDetails,
    Signal,
    combine_signals,
    roll_up,
)
from .measure import (
    LLMJudge,
    MEASURE_KINDS,
    evaluate_criterion,
    parse_measure,
    pending_to_signal,
    synthesize_criterion_checks,
)
from ._jsonl_log import ledger_dir
from .attribution import attribution_from_entries
from .eval import (
    INTENT_LEVEL_WEIGHTS,
    PairedComparison,
    SessionScorecard,
    WindowComparison,
    WindowSpec,
    aggregate_windows,
    compute_session_scorecard,
    daily_windows,
    paired_comparison,
    render_comparison,
    render_paired_comparison,
    render_summary,
    scorecard_from_step_eval,
    scorecard_to_dict,
    weekly_windows,
)
from .recall_cache import (
    RecallCache,
    RecallEntry,
    RecallHitMetadata,
    hash_prompt,
    make_turn_id,
)
from .recall_diagnostic import (
    RecallDiagnosticEntry,
    RecallDiagnosticLog,
    STATUS_DAEMON_ERROR,
    STATUS_DAEMON_UNREACHABLE,
    STATUS_NO_HIT,
    STATUS_NO_STORE,
    STATUS_OK,
)
from .sessions import SessionMarker, SessionMarkerLog, sessions_path
from .signals import PendingSignal, PendingSignalLog
from .trajectory import render_trajectory_primer
from .workspace import (
    STATE_DIR,
    STATE_ROOT_ENV,
    WORKSPACE_DECLARATION_FILE,
    WorkspaceIdentity,
    declare_workspace,
    resolve_primary_km,
    resolve_state_root,
    resolve_workspace,
    state_root_for,
)

__all__ = [
    "Claim",
    "ClaimKind",
    "ClaimSource",
    "ClaimState",
    "CorpusFacts",
    "ExperienceFacts",
    "PrimaryPolarity",
    "PRIMARY_POLARITY",
    "EXPERIENCE_KINDS",
    "MANUAL_TRANSCRIPT_HASH",
    "DISTILLED_PREFIX",
    "MIGRATED_PREFIX",
    "derive_origin",
    "evolve",
    "STATE_DIR",
    "STATE_ROOT_ENV",
    "WORKSPACE_DECLARATION_FILE",
    "WorkspaceIdentity",
    "declare_workspace",
    "resolve_primary_km",
    "resolve_state_root",
    "resolve_workspace",
    "state_root_for",
    "DurableIntentStore",
    "Intent",
    "LiveIntentStore",
    "durable_intent_dir",
    "intent_dir",
    "make_intent_id",
    "Attribution",
    "CriterionCheck",
    "IntentOutcomeDetails",
    "ClaimOutcomeLog",
    "ClaimOutcomeRecord",
    "Signal",
    "combine_signals",
    "LLMJudge",
    "MEASURE_KINDS",
    "evaluate_criterion",
    "parse_measure",
    "pending_to_signal",
    "synthesize_criterion_checks",
    "ledger_dir",
    "roll_up",
    "PendingSignal",
    "PendingSignalLog",
    "RecallCache",
    "RecallEntry",
    "RecallHitMetadata",
    "RecallDiagnosticEntry",
    "RecallDiagnosticLog",
    "STATUS_DAEMON_ERROR",
    "STATUS_DAEMON_UNREACHABLE",
    "STATUS_NO_HIT",
    "STATUS_NO_STORE",
    "STATUS_OK",
    "SessionMarker",
    "SessionMarkerLog",
    "sessions_path",
    "INTENT_LEVEL_WEIGHTS",
    "PairedComparison",
    "SessionScorecard",
    "WindowComparison",
    "WindowSpec",
    "aggregate_windows",
    "attribution_from_entries",
    "compute_session_scorecard",
    "daily_windows",
    "hash_prompt",
    "make_turn_id",
    "paired_comparison",
    "render_comparison",
    "render_paired_comparison",
    "render_summary",
    "scorecard_from_step_eval",
    "scorecard_to_dict",
    "weekly_windows",
    "render_trajectory_primer",
]
