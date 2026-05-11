"""Agent-state primitives — workspace identity, live intent, outcome ledger.

Per architecture §"Workspace" / §"Intent" / §"Outcomes". These are the new
state surfaces the Horizon 1 spike adds on top of the v2.2 memory store:

- `workspace.resolve_workspace(cwd)` — git-default identity with declaration override.
- `intent` — live intent graph at `<workspace-root>/.rlat-state/intent/`.
- `ledger` — append-only outcome records at `<workspace-root>/.rlat-state/ledger/`.

Durable memory + durable intent stay in the per-user store at
`~/.rlat/memory/<user-id>/`; only the *live* per-workspace state lives here.
"""

from .intent import LiveIntent, LiveIntentStore, intent_dir, make_intent_id
from .ledger import (
    Attribution,
    CriterionCheck,
    OutcomeLedger,
    OutcomeRecord,
    Signal,
    combine_signals,
    ledger_dir,
    roll_up,
)
from .attribution import attribution_from_entries
from .eval import (
    INTENT_LEVEL_WEIGHTS,
    SessionScorecard,
    WindowComparison,
    WindowSpec,
    aggregate_windows,
    compute_session_scorecard,
    daily_windows,
    render_comparison,
    render_summary,
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
    WORKSPACE_DECLARATION_FILE,
    WorkspaceIdentity,
    declare_workspace,
    resolve_workspace,
    state_root_for,
    workspace_polarity_tag,
)

__all__ = [
    "STATE_DIR",
    "WORKSPACE_DECLARATION_FILE",
    "WorkspaceIdentity",
    "declare_workspace",
    "resolve_workspace",
    "state_root_for",
    "workspace_polarity_tag",
    "LiveIntent",
    "LiveIntentStore",
    "intent_dir",
    "make_intent_id",
    "Attribution",
    "CriterionCheck",
    "OutcomeLedger",
    "OutcomeRecord",
    "Signal",
    "combine_signals",
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
    "SessionScorecard",
    "WindowComparison",
    "WindowSpec",
    "aggregate_windows",
    "attribution_from_entries",
    "compute_session_scorecard",
    "daily_windows",
    "hash_prompt",
    "make_turn_id",
    "render_comparison",
    "render_summary",
    "scorecard_to_dict",
    "weekly_windows",
    "render_trajectory_primer",
]
