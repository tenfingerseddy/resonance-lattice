"""Intent records and stores — the live workspace graph + the durable store.

An *intent* is a goal the agent is pursuing. It is **not** earned knowledge —
it has no trust, no Beta tally. It has a fixed shape and a lifecycle
(`proposed → active → satisfied`). The claim system keeps intents out of the
claim table for exactly that reason — `docs/internal/claim-system-design.md`
§5. Before this split intents were `Row`s in the per-user memory store, gated
by `level`; that was expedient, not right.

Intents live in two stores, by lifespan, sharing one `Intent` record:

  LiveIntentStore     — the live working graph, per workspace, at
                        `<workspace-root>/.rlat-state/intent/`. Holds `step`
                        and `task` intents — the fast-moving plan.
  DurableIntentStore  — durable goals and directions, per user, at
                        `<user-root>/durable_intent/`. Holds `goal` and
                        `direction` intents — the standing ambitions a
                        workspace task ladders up into.

Both serialise to plain JSON: per-turn writes (status flips, sub-step adds)
happen faster than the embedded memory substrate is built for, and structural
traversal (find ready siblings; find blocked descendants) gains nothing from
similarity-based recall.

The live store additionally keeps two append-only logs — `decomposition.jsonl`
(parent → child splits) and `transitions.jsonl` (status changes).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypedDict

import portalocker

from ..memory._common import (
    atomic_write_json,
    make_ulid,
    utcnow_iso,
    validate_criterion,
    validate_enum,
)
from ..memory.store import INTENT_KIND_VALUES, IntentKind

# --- Intent-record enums ---------------------------------------------------

# The agent's stance toward an intent's goal.
Stance = Literal["do", "avoid", "know"]
STANCE_VALUES: frozenset[str] = frozenset({"do", "avoid", "know"})

# The agent's estimate of how reachable the goal is.
Achievability = Literal["low", "medium", "high"]
ACHIEVABILITY_VALUES: frozenset[str] = frozenset({"low", "medium", "high"})

# The lifecycle status. `proposed` is captured-but-not-yet-confirmed — used by
# the plan-mode-listening hook to record plan-mode output before the user
# accepts it; it transitions to `active` on first execution signal or
# `abandoned` on rejection.
Status = Literal[
    "proposed", "active", "blocked", "satisfied", "abandoned", "superseded",
]
STATUS_VALUES: frozenset[str] = frozenset(
    {"proposed", "active", "blocked", "satisfied", "abandoned", "superseded"}
)

# The intent ladder, split by lifespan. Live intents are the fast-moving plan;
# durable intents are the standing ambitions a live task ladders up into.
LIVE_LEVELS: frozenset[str] = frozenset({"step", "task"})
DURABLE_LEVELS: frozenset[str] = frozenset({"goal", "direction"})


class Criterion(TypedDict):
    """One success criterion. SMART minus time-bound; lifecycle bounds it.

    `text` is the specific, measurable statement. `measure` names the
    verification mechanism — one of `mechanical:<spec>`, `user_confirms`,
    `llm_judges:<rubric>` (architecture §"Success criteria").
    """
    text: str
    measure: str


# --- The record ------------------------------------------------------------


@dataclass
class Intent:
    """One intent — a goal the agent is pursuing.

    Shared by both stores: `level` is one of `LIVE_LEVELS` in the live graph,
    one of `DURABLE_LEVELS` in the durable store. Mutable — `set_status` flips
    `status` in place. An intent carries no trust; it is not earned knowledge.
    """

    intent_id: str
    level: str
    text: str
    parent_ids: list[str]
    blocks: list[str]
    stance: Stance
    achievability: Achievability
    status: Status
    success_criteria: list[Criterion]
    constraints: list[str]
    created_under_intent_kind: IntentKind
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- On-disk layout --------------------------------------------------------

INTENT_DIR = "intent"
ACTIVE_FILE = "active.json"
DECOMPOSITION_LOG = "decomposition.jsonl"
TRANSITIONS_LOG = "transitions.jsonl"

DURABLE_DIR = "durable_intent"
DURABLE_FILE = "intents.json"

_SCHEMA_VERSION = 1


@dataclass
class _IntentFile:
    """In-memory mirror of a store's JSON file — used for read-modify-write."""

    schema_version: int = _SCHEMA_VERSION
    intents: list[Intent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intents": [i.to_dict() for i in self.intents],
        }


def intent_dir(state_root: Path) -> Path:
    """`<state-root>/intent/` — the live intent graph directory."""
    return state_root / INTENT_DIR


def durable_intent_dir(user_root: Path) -> Path:
    """`<user-root>/durable_intent/` — the durable intent store directory."""
    return user_root / DURABLE_DIR


def _validate_intent_payload(
    *,
    level: str,
    allowed_levels: frozenset[str],
    stance: str,
    achievability: str,
    status: str,
    success_criteria: list[Any],
    constraints: list[Any],
    created_under_intent_kind: str,
) -> None:
    """Enum + shape validation for one intent. `level` is checked against the
    store's own subset of the ladder — a live store rejects `goal`, a durable
    store rejects `step`."""
    if level not in allowed_levels:
        raise ValueError(
            f"level {level!r} not valid for this store; "
            f"allowed: {sorted(allowed_levels)}"
        )
    validate_enum("stance", stance, STANCE_VALUES)
    validate_enum("achievability", achievability, ACHIEVABILITY_VALUES)
    validate_enum("status", status, STATUS_VALUES)
    validate_enum(
        "created_under_intent_kind", created_under_intent_kind,
        INTENT_KIND_VALUES,
    )
    if not isinstance(success_criteria, list):
        raise ValueError("success_criteria must be a list")
    for c in success_criteria:
        validate_criterion(c)
    if not isinstance(constraints, list) or any(
        not isinstance(s, str) for s in constraints
    ):
        raise ValueError("constraints must be a list[str]")


# --- Stores ----------------------------------------------------------------


class _IntentStore:
    """Shared JSON-backed intent store — lock, read-modify-write, add, read.

    Reads and writes are serialised under a portalocker advisory lock so
    concurrent hook invocations don't lose updates; the JSON file is rewritten
    atomically via tmp + os.replace.

    Subclasses set three class attributes — `_SUBDIR`, `_FILE`, and
    `_ALLOWED_LEVELS` — and the live store adds the decomposition/transition
    logs on top. They differ in nothing else.
    """

    _SUBDIR: str
    _FILE: str
    _ALLOWED_LEVELS: frozenset[str]

    def __init__(self, root: Path | str):
        self._root = Path(root) / self._SUBDIR
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._root / ".lock"
        self._lock_path.touch(exist_ok=True)

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def _read(self) -> _IntentFile:
        path = self._root / self._FILE
        if not path.exists():
            return _IntentFile()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt file — surface as empty so capture can rebuild rather
            # than crashing every hook (architecture §"Defences against
            # abrupt endings" — never fatal).
            return _IntentFile()
        intents = [
            Intent(**i) for i in payload.get("intents", [])
            if isinstance(i, dict)
        ]
        return _IntentFile(
            schema_version=payload.get("schema_version", _SCHEMA_VERSION),
            intents=intents,
        )

    def _write(self, state: _IntentFile) -> None:
        atomic_write_json(self._root / self._FILE, state.to_dict())

    def list_all(self) -> list[Intent]:
        """Snapshot of every intent in the store (any status)."""
        with self._lock():
            return list(self._read().intents)

    def read(self, intent_id: str) -> Intent | None:
        """One intent by id, or `None` if absent."""
        with self._lock():
            for intent in self._read().intents:
                if intent.intent_id == intent_id:
                    return intent
            return None

    def _mutate(self, intent_id: str, fn: Callable[[Intent], bool]) -> Intent:
        """Locked find-modify-write. `fn(intent)` mutates the matched intent
        in place and returns True to persist, or False for an idempotent
        no-op. Raises `KeyError` if `intent_id` is absent."""
        with self._lock():
            state = self._read()
            for i, intent in enumerate(state.intents):
                if intent.intent_id == intent_id:
                    if fn(intent):
                        state.intents[i] = intent
                        self._write(state)
                    return intent
            raise KeyError(f"intent_id {intent_id!r} not in {self._FILE}")

    def add_intent(
        self,
        *,
        level: str,
        text: str,
        stance: Stance,
        achievability: Achievability,
        success_criteria: list[Criterion],
        constraints: list[str],
        created_under_intent_kind: IntentKind = "none",
        parent_ids: list[str] | None = None,
        blocks: list[str] | None = None,
        status: Status = "active",
    ) -> Intent:
        """Add a new intent. Returns the persisted record."""
        _validate_intent_payload(
            level=level,
            allowed_levels=self._ALLOWED_LEVELS,
            stance=stance,
            achievability=achievability,
            status=status,
            success_criteria=success_criteria,
            constraints=constraints,
            created_under_intent_kind=created_under_intent_kind,
        )
        now = utcnow_iso()
        intent = Intent(
            intent_id=make_ulid(),
            level=level,
            text=text,
            parent_ids=list(parent_ids) if parent_ids else [],
            blocks=list(blocks) if blocks else [],
            stance=stance,
            achievability=achievability,
            status=status,
            success_criteria=[dict(c) for c in success_criteria],
            constraints=list(constraints),
            created_under_intent_kind=created_under_intent_kind,
            created_at=now,
            updated_at=now,
        )
        with self._lock():
            state = self._read()
            state.intents.append(intent)
            self._write(state)
        return intent


class LiveIntentStore(_IntentStore):
    """The live working intent graph at `<state-root>/.rlat-state/intent/`.

    Holds `step` and `task` intents plus two append-only logs:
    `decomposition.jsonl` and `transitions.jsonl`. A mid-append crash leaves
    at most one truncated trailing line; recovery drops it on next read.
    """

    _SUBDIR = INTENT_DIR
    _FILE = ACTIVE_FILE
    _ALLOWED_LEVELS = LIVE_LEVELS

    def _append_log(self, log_name: str, entry: dict[str, Any]) -> None:
        # Single write keeps the line atomic up to PIPE_BUF on POSIX even
        # before the lock; combined with the caller-side `with self._lock()`
        # contract, concurrent writers can't interleave log entries.
        line = json.dumps(entry, sort_keys=True) + "\n"
        with open(self._root / log_name, "a", encoding="utf-8") as f:
            f.write(line)

    def _read_log(self, log_name: str) -> list[dict[str, Any]]:
        path = self._root / log_name
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Mid-append crash leaves at most one truncated trailing
                # line — silently drop, never raise.
                continue
        return out

    def set_status(
        self,
        intent_id: str,
        new_status: Status,
        *,
        reason: str = "",
    ) -> Intent:
        """Flip an intent's status. Records the transition to the log."""
        validate_enum("status", new_status, STATUS_VALUES)
        now = utcnow_iso()

        def _apply(intent: Intent) -> bool:
            if intent.status == new_status:
                return False  # no-op; idempotent
            self._append_log(TRANSITIONS_LOG, {
                "intent_id": intent_id,
                "from": intent.status,
                "to": new_status,
                "reason": reason,
                "at": now,
            })
            intent.status = new_status
            intent.updated_at = now
            return True

        return self._mutate(intent_id, _apply)

    def record_decomposition(
        self,
        parent_intent_id: str,
        child_intent_ids: Iterable[str],
        *,
        rationale: str = "",
    ) -> None:
        """Log a decomposition decision. Children must already exist in the
        graph (decomposition is observed, not enacted, by this method)."""
        children = list(child_intent_ids)
        if not children:
            return  # empty decomposition is a no-op, not an error
        with self._lock():
            self._append_log(DECOMPOSITION_LOG, {
                "parent_intent_id": parent_intent_id,
                "child_intent_ids": children,
                "rationale": rationale,
                "at": utcnow_iso(),
            })

    def add_block(self, intent_id: str, blocks_intent_id: str) -> Intent:
        """Mark `intent_id` as blocking `blocks_intent_id` (sibling
        dependency, the only project-management primitive not derivable
        from the parent chain — architecture §"Project management as a
        projection")."""
        def _apply(intent: Intent) -> bool:
            if blocks_intent_id in intent.blocks:
                return False  # idempotent
            intent.blocks.append(blocks_intent_id)
            intent.updated_at = utcnow_iso()
            return True

        return self._mutate(intent_id, _apply)

    def read_transitions(self) -> list[dict[str, Any]]:
        """Read the transitions log (skips trailing truncated line)."""
        return self._read_log(TRANSITIONS_LOG)

    def read_decompositions(self) -> list[dict[str, Any]]:
        """Read the decomposition log (skips trailing truncated line)."""
        return self._read_log(DECOMPOSITION_LOG)


class DurableIntentStore(_IntentStore):
    """Durable goals and directions at `<user-root>/durable_intent/`.

    Per-user and cross-workspace — the standing ambitions a workspace's live
    task ladders up into. Populated by `rlat intent declare-durable`; read by
    `rlat intent durable` and the cross-store `rlat intent path` walk.
    """

    _SUBDIR = DURABLE_DIR
    _FILE = DURABLE_FILE
    _ALLOWED_LEVELS = DURABLE_LEVELS


# Public alias for hooks/CLI that pre-compute an intent id outside the store.
make_intent_id = make_ulid
