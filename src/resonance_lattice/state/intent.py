"""Live intent graph — `<workspace-root>/.rlat-state/intent/`.

Architecture §"Live intent — in agent-state" specifies plain JSON for the
working graph because per-turn writes (status flips, sub-step adds) happen
faster than the embedded memory substrate is designed for, and structural
traversal (find ready siblings; find blocked descendants) doesn't benefit
from similarity-based recall.

Three files:

  active.json         — the live working graph (tasks + steps, statuses).
                        Read-modify-write under portalocker; atomic via
                        tmp + os.replace.
  decomposition.jsonl — append-only log of decomposition decisions
                        (parent_intent → child_intent[s]).
  transitions.jsonl   — append-only log of status changes
                        (intent_id, from → to, reason, timestamp).

Durable intent (resolved goals/directions worth keeping) is *promoted* into
the per-user memory store via Capture's distil context — that path lives in
the memory store, not here. This module owns only the live graph; the
promotion bridge is wired in Horizon 2 of the harness roadmap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import portalocker

from ..memory._common import (
    atomic_write_json,
    make_ulid,
    utcnow_iso,
    validate_criterion,
    validate_enum,
)
from ..memory.store import (
    ACHIEVABILITY_VALUES,
    Achievability,
    Criterion,
    INTENT_KIND_VALUES,
    INTENT_LEVELS,
    IntentKind,
    STANCE_VALUES,
    STATUS_VALUES,
    Stance,
    Status,
)

INTENT_DIR = "intent"
ACTIVE_FILE = "active.json"
DECOMPOSITION_LOG = "decomposition.jsonl"
TRANSITIONS_LOG = "transitions.jsonl"

_ACTIVE_SCHEMA_VERSION = 1


@dataclass
class LiveIntent:
    """One live intent — either a `task` or a `step`.

    Goals and directions live in the memory store (durable), not here.
    Live-intent rows promote *up* to durable when they resolve and earned
    their place; that path is a Horizon 2 deliverable.
    """

    intent_id: str
    level: str  # 'task' or 'step'
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


@dataclass
class _ActiveFile:
    """In-memory mirror of `active.json` — used for read-modify-write."""

    schema_version: int = _ACTIVE_SCHEMA_VERSION
    intents: list[LiveIntent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intents": [i.to_dict() for i in self.intents],
        }


_LIVE_LEVELS: frozenset[str] = frozenset({"step", "task"})


def _validate_live_level(level: str) -> None:
    """Live intent rows must be `step` or `task` — `goal` and `direction`
    rows live in the durable memory store. The intent-ladder split (live vs
    durable) is the cutoff between fast structural traversal and similarity
    recall (architecture §"Where intent lives — two homes")."""
    if level not in _LIVE_LEVELS:
        if level in INTENT_LEVELS:
            raise ValueError(
                f"level {level!r} is durable — live intent owns {{step, task}} "
                f"only; {{goal, direction}} live in the per-user memory store"
            )
        raise ValueError(
            f"level must be one of {sorted(_LIVE_LEVELS)}; got {level!r}"
        )


def _validate_intent_payload(
    *,
    level: str,
    stance: str,
    achievability: str,
    status: str,
    success_criteria: list[Any],
    constraints: list[Any],
    created_under_intent_kind: str,
) -> None:
    """Enum + shape validation. Same rules as Row's intent fields, but
    `level` here is constrained to {step, task}."""
    _validate_live_level(level)
    validate_enum("stance", stance, STANCE_VALUES)
    validate_enum("achievability", achievability, ACHIEVABILITY_VALUES)
    validate_enum("status", status, STATUS_VALUES)
    validate_enum(
        "created_under_intent_kind",
        created_under_intent_kind,
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


def intent_dir(state_root: Path) -> Path:
    """`<state-root>/intent/`."""
    return state_root / INTENT_DIR


class LiveIntentStore:
    """Live intent graph at `<state-root>/intent/`.

    Reads + writes serialised under a portalocker advisory lock so concurrent
    hook invocations (Stop hook + UserPromptSubmit racing on a status flip)
    don't lose updates. Active graph rewrites are atomic via tmp+os.replace;
    `decomposition.jsonl` and `transitions.jsonl` are append-only with a
    line-buffered write so a kill mid-append leaves at most one truncated
    trailing line (recovery: drop it on next read).
    """

    def __init__(self, state_root: Path | str):
        self._root = intent_dir(Path(state_root))
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._root / ".lock"
        self._lock_path.touch(exist_ok=True)

    # -- I/O helpers ------------------------------------------------------

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def _read_active(self) -> _ActiveFile:
        path = self._root / ACTIVE_FILE
        if not path.exists():
            return _ActiveFile()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt active.json — surface as empty so capture can rebuild,
            # rather than crashing every hook. Architecture §"Defences
            # against abrupt endings" — never fatal.
            return _ActiveFile()
        intents = [
            LiveIntent(**i) for i in payload.get("intents", [])
            if isinstance(i, dict)
        ]
        return _ActiveFile(
            schema_version=payload.get(
                "schema_version", _ACTIVE_SCHEMA_VERSION
            ),
            intents=intents,
        )

    def _write_active(self, state: _ActiveFile) -> None:
        atomic_write_json(self._root / ACTIVE_FILE, state.to_dict())

    def _append_log(self, log_name: str, entry: dict[str, Any]) -> None:
        # Single write keeps the line atomic up to PIPE_BUF on POSIX even
        # before the lock; combined with the caller-side `with self._lock()`
        # contract, concurrent writers can't interleave decomposition or
        # transition entries.
        path = self._root / log_name
        line = json.dumps(entry, sort_keys=True) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    # -- Public API -------------------------------------------------------

    def list_active(self) -> list[LiveIntent]:
        """Snapshot of every live intent (any status)."""
        with self._lock():
            return list(self._read_active().intents)

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
    ) -> LiveIntent:
        """Add a new live intent. Returns the persisted row."""
        _validate_intent_payload(
            level=level,
            stance=stance,
            achievability=achievability,
            status=status,
            success_criteria=success_criteria,
            constraints=constraints,
            created_under_intent_kind=created_under_intent_kind,
        )
        now = utcnow_iso()
        intent = LiveIntent(
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
            state = self._read_active()
            state.intents.append(intent)
            self._write_active(state)
        return intent

    def set_status(
        self,
        intent_id: str,
        new_status: Status,
        *,
        reason: str = "",
    ) -> LiveIntent:
        """Flip an intent's status. Records the transition to the log."""
        validate_enum("status", new_status, STATUS_VALUES)
        now = utcnow_iso()
        with self._lock():
            state = self._read_active()
            for i, intent in enumerate(state.intents):
                if intent.intent_id == intent_id:
                    if intent.status == new_status:
                        return intent  # no-op; idempotent
                    self._append_log(TRANSITIONS_LOG, {
                        "intent_id": intent_id,
                        "from": intent.status,
                        "to": new_status,
                        "reason": reason,
                        "at": now,
                    })
                    intent.status = new_status
                    intent.updated_at = now
                    state.intents[i] = intent
                    self._write_active(state)
                    return intent
            raise KeyError(f"intent_id {intent_id!r} not in live graph")

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

    def add_block(self, intent_id: str, blocks_intent_id: str) -> LiveIntent:
        """Mark `intent_id` as blocking `blocks_intent_id` (sibling
        dependency, the only project-management primitive not derivable
        from the parent chain — architecture §"Project management as a
        projection")."""
        with self._lock():
            state = self._read_active()
            for i, intent in enumerate(state.intents):
                if intent.intent_id == intent_id:
                    if blocks_intent_id not in intent.blocks:
                        intent.blocks.append(blocks_intent_id)
                        intent.updated_at = utcnow_iso()
                        state.intents[i] = intent
                        self._write_active(state)
                    return intent
            raise KeyError(f"intent_id {intent_id!r} not in live graph")

    def read_transitions(self) -> list[dict[str, Any]]:
        """Read the transitions log (skips trailing truncated line)."""
        return self._read_log(TRANSITIONS_LOG)

    def read_decompositions(self) -> list[dict[str, Any]]:
        """Read the decomposition log (skips trailing truncated line)."""
        return self._read_log(DECOMPOSITION_LOG)

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
                # line — silently drop, never raise. The append-only
                # contract guarantees prior entries are intact.
                continue
        return out


# Public alias for hooks/CLI that pre-compute an intent id outside the store.
make_intent_id = make_ulid
