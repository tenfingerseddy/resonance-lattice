"""Flat memory store — v2.2 (Horizon 1 schema).

Per-user `memory.npz` + `sidecar.jsonl` pair under `~/.rlat/memory/<user-id>/`.
Atomic write via `portalocker` advisory lock + tmp + os.replace pair, mirroring
v2.0 archive write contracts.

v2.2 extends the v2.1 9-field row to 13 memory fields + 5 intent-only fields,
adding the strength × development × truth axes the agent-harness architecture
requires (`.claude/plans/agent-harness-architecture.md` §"Memory schema").
v1 sidecars load through `_apply_v1_defaults`; all additions are additive, no
breaking change. Mirrors the v4 → v4.1 archive bump pattern.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict

import numpy as np
import portalocker

from ..field._runtime_common import l2_normalize
from ..field.encoder import DIM, Encoder
from ._common import (
    make_ulid,
    utcnow_iso,
    validate_criterion as _validate_criterion,
    validate_enum as _validate_enum,
)

SCHEMA_VERSION = 2

PrimaryPolarity = Literal["prefer", "avoid", "factual"]
# Polarity is `[primary, *scope_tags]`. Scope tags include `workspace:<hash>`
# and `cross-workspace`; the closed primary is enforced at write time.
Polarity = list[str]
PRIMARY_POLARITY: frozenset[str] = frozenset({"prefer", "avoid", "factual"})

# `transcript_hash` discriminators. Manual rows (CLI `rlat memory add`) carry
# the literal string; distil-emitted rows carry `<DISTILLED_PREFIX><source_sha>`;
# migrated rows from v2.0 LayeredMemory carry `<MIGRATED_PREFIX><tier>`. All
# three are excluded from the distil-input filter so the distiller never
# reprocesses its own output OR a v2.0-migrated row as raw capture.
MANUAL_TRANSCRIPT_HASH = "manual"
DISTILLED_PREFIX = "distilled:"
MIGRATED_PREFIX = "migrated:"

# Memory ladder (event → pattern → learning → principle) and intent ladder
# (step → task → goal → direction) share the `level` field; the two enums are
# non-overlapping so `level` alone discriminates memory rows from intent rows.
Level = Literal[
    "event", "pattern", "learning", "principle",
    "step", "task", "goal", "direction",
]
MEMORY_LEVELS: frozenset[str] = frozenset(
    {"event", "pattern", "learning", "principle"}
)
INTENT_LEVELS: frozenset[str] = frozenset(
    {"step", "task", "goal", "direction"}
)
LEVEL_VALUES: frozenset[str] = MEMORY_LEVELS | INTENT_LEVELS

Criticality = Literal["low", "normal", "high", "severe"]
CRITICALITY_VALUES: frozenset[str] = frozenset(
    {"low", "normal", "high", "severe"}
)

Confidence = Literal["low", "medium", "high", "verified"]
CONFIDENCE_VALUES: frozenset[str] = frozenset(
    {"low", "medium", "high", "verified"}
)

Origin = Literal[
    "manual", "distilled", "migrated", "outcome_derived", "intent_resolution",
]
ORIGIN_VALUES: frozenset[str] = frozenset(
    {"manual", "distilled", "migrated", "outcome_derived", "intent_resolution"}
)

# Captured at row-write time — the agent's intent-kind context, used by the
# manifesto recall re-rank's `level_match` factor. `none` is the default for
# rows captured outside an active intent (e.g. v1-migrated rows).
IntentKind = Literal[
    "debug", "design", "implement", "review", "explain", "refactor", "none",
]
INTENT_KIND_VALUES: frozenset[str] = frozenset(
    {"debug", "design", "implement", "review", "explain", "refactor", "none"}
)

# Intent-row-only enums.
Stance = Literal["do", "avoid", "know"]
STANCE_VALUES: frozenset[str] = frozenset({"do", "avoid", "know"})

Achievability = Literal["low", "medium", "high"]
ACHIEVABILITY_VALUES: frozenset[str] = frozenset({"low", "medium", "high"})

Status = Literal[
    "proposed", "active", "blocked", "satisfied", "abandoned", "superseded",
]
STATUS_VALUES: frozenset[str] = frozenset(
    {"proposed", "active", "blocked", "satisfied", "abandoned", "superseded"}
)
# `proposed` is the captured-but-not-yet-confirmed state — used by the
# planning-mode-listening hook to record plan-mode output before the
# user explicitly accepts. Transitions to `active` on first execution
# signal or `abandoned` on rejection.


class Criterion(TypedDict):
    """One success criterion. SMART minus time-bound; lifecycle bounds it.

    `text` is the specific, measurable statement. `measure` names the
    verification mechanism — one of `mechanical:<spec>`, `user_confirms`,
    `llm_judges:<rubric>` (architecture §"Success criteria").
    """
    text: str
    measure: str


def _derive_origin(transcript_hash: str) -> Origin:
    """Categorical `origin` from the v1 `transcript_hash` prefix.

    Maps the prefix system to the v2.2 `origin` enum so v1 rows missing
    `origin` populate from the existing field on load (architecture
    §"Migration from v2.1").
    """
    if transcript_hash.startswith(DISTILLED_PREFIX):
        return "distilled"
    if transcript_hash.startswith(MIGRATED_PREFIX):
        return "migrated"
    return "manual"


_ulid = make_ulid  # Local alias kept for callers within this module.


@dataclass(frozen=True)
class Row:
    """One sidecar row.

    13 memory fields (the schema's strength × development × truth axes from
    architecture §"Memory schema") plus 5 intent-only fields populated when
    `level` is one of `INTENT_LEVELS`. Frozen because mutations always go
    through `Memory.update_row(row_id, **fields)` which rebuilds the row.
    """

    # --- v2.1 base (9 fields) — unchanged shape ---
    row_id: str
    text: str
    polarity: Polarity
    recurrence_count: int
    created_at: str
    last_corroborated_at: str
    transcript_hash: str
    is_bad: bool
    schema_version: int = SCHEMA_VERSION

    # --- v2.2 memory extension (4 net-new + origin lift + intent-kind ctx) ---
    level: Level = "event"
    criticality: Criticality = "normal"
    confidence: Confidence = "medium"
    parent_ids: list[str] = field(default_factory=list)
    cited_passages: list[str] = field(default_factory=list)
    origin: Origin = "manual"
    created_under_intent_kind: IntentKind = "none"

    # --- v2.2 intent-row extension (forbidden when level ∈ MEMORY_LEVELS) ---
    stance: Stance | None = None
    achievability: Achievability | None = None
    status: Status | None = None
    success_criteria: list[Criterion] | None = None
    constraints: list[str] | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        return asdict(self)

    def primary_polarity(self) -> str:
        """Extract the single primary tag. Exactly one is guaranteed by
        `_validate_polarity` at write time."""
        return next(p for p in self.polarity if p in PRIMARY_POLARITY)

    def is_manual(self) -> bool:
        return self.transcript_hash == MANUAL_TRANSCRIPT_HASH

    def is_distilled(self) -> bool:
        return self.transcript_hash.startswith(DISTILLED_PREFIX)

    def is_migrated(self) -> bool:
        return self.transcript_hash.startswith(MIGRATED_PREFIX)

    def is_intent(self) -> bool:
        """True if this row's `level` is in the intent ladder."""
        return self.level in INTENT_LEVELS

    def summary(self, *, max_text: int = 80) -> str:
        """Single-line tabular row for CLI / harness display."""
        text = self.text.replace("\n", " ").strip()
        if len(text) > max_text:
            text = text[: max_text - 1] + "…"
        bad = " [bad]" if self.is_bad else ""
        return (
            f"{self.row_id}  [{self.primary_polarity():<7}]  "
            f"rec={self.recurrence_count:<3}{bad}  {text}"
        )


def _validate_polarity(polarity: list[str]) -> None:
    if not polarity:
        raise ValueError("polarity must contain at least one tag")
    primaries = [p for p in polarity if p in PRIMARY_POLARITY]
    if len(primaries) != 1:
        raise ValueError(
            f"polarity must have exactly one primary tag from {sorted(PRIMARY_POLARITY)}; "
            f"got {primaries!r} in {polarity!r}"
        )


def _validate_row_fields(
    *,
    level: str,
    criticality: str,
    confidence: str,
    origin: str,
    created_under_intent_kind: str,
    stance: str | None,
    achievability: str | None,
    status: str | None,
    success_criteria: list[Any] | None,
    constraints: list[Any] | None,
) -> None:
    """Validate the v2.2 enum fields and intent-only required-vs-forbidden rule.

    Intent-only fields are *required* when `level` is in `INTENT_LEVELS` and
    *forbidden* when `level` is in `MEMORY_LEVELS`. Architecture §"Schema
    additions for intent" — clean validation rule, driven by the level field.
    """
    _validate_enum("level", level, LEVEL_VALUES)
    _validate_enum("criticality", criticality, CRITICALITY_VALUES)
    _validate_enum("confidence", confidence, CONFIDENCE_VALUES)
    _validate_enum("origin", origin, ORIGIN_VALUES)
    _validate_enum(
        "created_under_intent_kind",
        created_under_intent_kind,
        INTENT_KIND_VALUES,
    )
    intent_fields = {
        "stance": stance,
        "achievability": achievability,
        "status": status,
        "success_criteria": success_criteria,
        "constraints": constraints,
    }
    is_intent = level in INTENT_LEVELS
    if is_intent:
        missing = [k for k, v in intent_fields.items() if v is None]
        if missing:
            raise ValueError(
                f"intent row (level={level!r}) requires fields {missing}; "
                f"all of stance/achievability/status/success_criteria/"
                f"constraints must be set"
            )
        _validate_enum("stance", stance, STANCE_VALUES)
        _validate_enum("achievability", achievability, ACHIEVABILITY_VALUES)
        _validate_enum("status", status, STATUS_VALUES)
        if not isinstance(success_criteria, list):
            raise ValueError("success_criteria must be a list of {text, measure}")
        for c in success_criteria:
            _validate_criterion(c)
        if not isinstance(constraints, list) or any(
            not isinstance(s, str) for s in constraints
        ):
            raise ValueError("constraints must be a list[str]")
    else:
        present = [k for k, v in intent_fields.items() if v is not None]
        if present:
            raise ValueError(
                f"memory row (level={level!r}) must not set intent-only "
                f"fields {present}"
            )


def _apply_v1_defaults(obj: dict[str, Any]) -> dict[str, Any]:
    """Populate v2.2 fields that are missing from a v1 sidecar row.

    v1 rows carried `schema_version=1` and lacked the seven memory-extension
    fields and five intent-only fields. The migration table in architecture
    §"Migration from v2.1" specifies the per-field default; this helper
    realises it. Memory-row defaults are applied to all v1 rows (v1 had no
    intent rows by construction).
    """
    if obj.get("schema_version", SCHEMA_VERSION) >= SCHEMA_VERSION:
        return obj
    # Memory-extension defaults from the migration table.
    obj.setdefault("level", "event")
    obj.setdefault("criticality", "normal")
    # `is_bad` rows in v1 partially encoded low confidence; lift accordingly.
    obj.setdefault("confidence", "low" if obj.get("is_bad") else "medium")
    obj.setdefault("parent_ids", [])
    obj.setdefault("cited_passages", [])
    # Origin lifts from the existing transcript_hash prefix.
    obj.setdefault(
        "origin",
        _derive_origin(obj.get("transcript_hash", MANUAL_TRANSCRIPT_HASH)),
    )
    obj.setdefault("created_under_intent_kind", "none")
    return obj


def path_for_user(user_id: str | None = None, root: Path | None = None) -> Path:
    """Resolve `~/.rlat/memory/<user-id>/`. Falls back through
    `RLAT_MEMORY_USER` → `USER` → `USERNAME` per §0.1.
    """
    if user_id is None:
        user_id = (
            os.environ.get("RLAT_MEMORY_USER")
            or os.environ.get("USER")
            or os.environ.get("USERNAME")
        )
    if not user_id:
        raise RuntimeError(
            "could not derive user_id from RLAT_MEMORY_USER / USER / USERNAME — "
            "pass --user explicitly"
        )
    base = Path(root) if root is not None else Path.home() / ".rlat" / "memory"
    return base / user_id


# ---------------------------------------------------------------------------
# On-disk I/O
# ---------------------------------------------------------------------------


def _load_sidecar(root: Path) -> list[Row]:
    """Load sidecar rows. Drops unknown keys per §18.8 (additive schema
    growth); emits a single stderr warning on the first row whose
    `schema_version` exceeds the current writer version (Appendix D D.9
    (d) — never crash on a future schema).
    """
    p = root / "sidecar.jsonl"
    if not p.exists():
        return []
    rows: list[Row] = []
    known = {f.name for f in Row.__dataclass_fields__.values()}
    warned_future = False
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        sv = obj.get("schema_version", SCHEMA_VERSION)
        if isinstance(sv, int) and sv > SCHEMA_VERSION and not warned_future:
            print(
                f"[rlat memory] warning: sidecar at {p} carries "
                f"schema_version={sv} > writer version {SCHEMA_VERSION}; "
                f"unknown fields dropped, row loaded best-effort.",
                file=sys.stderr,
            )
            warned_future = True
        obj = _apply_v1_defaults(obj)
        rows.append(Row(**{k: v for k, v in obj.items() if k in known}))
    return rows


def _load_band(root: Path, n_expected: int) -> np.ndarray:
    p = root / "memory.npz"
    if not p.exists():
        if n_expected != 0:
            raise ValueError(
                f"sidecar has {n_expected} rows but {p} missing — store corrupt"
            )
        return np.zeros((0, DIM), dtype=np.float32)
    with np.load(p) as z:
        band = z["band"]
    if band.shape[0] != n_expected:
        raise ValueError(
            f"sidecar/band row mismatch: sidecar has {n_expected} rows, "
            f"band has {band.shape[0]}"
        )
    if band.shape[1] != DIM:
        raise ValueError(
            f"band dim {band.shape[1]} != expected {DIM} — encoder mismatch"
        )
    return band.astype(np.float32, copy=False)


def _atomic_write_sidecar(root: Path, rows: list[Row]) -> None:
    """Tmp + os.replace for the JSONL only. Caller must hold the lock."""
    sidecar = root / "sidecar.jsonl"
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r.to_jsonl_dict(), sort_keys=True) for r in rows),
        encoding="utf-8",
    )
    os.replace(tmp, sidecar)


def _atomic_write_band(root: Path, band: np.ndarray) -> None:
    """Tmp + os.replace for the NPZ only. Caller must hold the lock."""
    band_p = root / "memory.npz"
    tmp = band_p.with_suffix(band_p.suffix + ".tmp")
    # np.savez auto-appends `.npz` to a path that doesn't already have it,
    # then writes there — file-handle form sidesteps the suffix logic so the
    # tmp file lands exactly where os.replace expects.
    with open(tmp, "wb") as f:
        np.savez(f, band=np.ascontiguousarray(band, dtype=np.float32))
    os.replace(tmp, band_p)


def _atomic_write_pair(root: Path, rows: list[Row], band: np.ndarray) -> None:
    if band.shape[0] != len(rows):
        raise ValueError(
            f"row count mismatch: rows={len(rows)} band={band.shape[0]}"
        )
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_sidecar(root, rows)
    _atomic_write_band(root, band)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Memory:
    """Flat memory store at `~/.rlat/memory/<user-id>/`.

    Every mutating call acquires the portalocker lock, re-reads disk so
    concurrent writers are visible, applies the mutation, and writes
    atomically. Single Python process per session, but multiple processes
    may write concurrently (Stop hook + manual `rlat memory add` racing) —
    portalocker serialises them and the re-read-under-lock pattern prevents
    lost updates.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        user_id: str | None = None,
        encoder: Encoder | None = None,
    ):
        self.root = Path(root) if root is not None else path_for_user(user_id=user_id)
        self._encoder: Encoder | None = encoder
        self.root.mkdir(parents=True, exist_ok=True)
        # Pre-create the lock file so portalocker.Lock(mode="r+b") doesn't
        # have to stat-then-touch on every acquisition.
        self._lock_path = self.root / ".lock"
        self._lock_path.touch(exist_ok=True)

    def _ensure_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder()
        return self._encoder

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path),
            mode="r+b",
            flags=portalocker.LOCK_EX,
        )

    def _read_state(self) -> tuple[list[Row], np.ndarray]:
        rows = _load_sidecar(self.root)
        band = _load_band(self.root, n_expected=len(rows))
        return rows, band

    def read_all(self) -> tuple[list[Row], np.ndarray]:
        """Snapshot of (rows, band). Acquires the lock so readers see a
        consistent pair even when a writer is mid-flight.
        """
        with self._lock():
            rows, band = self._read_state()
            return list(rows), band.copy()

    def add_row(
        self,
        text: str,
        polarity: list[str],
        *,
        transcript_hash: str,
        intent: str = "",
        embedding: np.ndarray | None = None,
        level: Level = "event",
        criticality: Criticality = "normal",
        confidence: Confidence = "medium",
        parent_ids: list[str] | None = None,
        cited_passages: list[str] | None = None,
        origin: Origin | None = None,
        created_under_intent_kind: IntentKind = "none",
        stance: Stance | None = None,
        achievability: Achievability | None = None,
        status: Status | None = None,
        success_criteria: list[Criterion] | None = None,
        constraints: list[str] | None = None,
        recurrence_count: int = 1,
    ) -> str:
        """Append a row. Returns the new row_id (ULID).

        `embedding` is optional — if omitted, the encoder is loaded lazily
        and runs `text + " | intent: " + intent`. Callers with a pre-computed
        embedding (distil, migration) pass it directly.

        `origin` defaults to the value derived from `transcript_hash` so v2.1
        call-sites stay valid without specifying origin explicitly. Intent
        rows (level ∈ INTENT_LEVELS) must set stance/achievability/status/
        success_criteria/constraints; memory rows must not.
        """
        _validate_polarity(polarity)
        if origin is None:
            origin = _derive_origin(transcript_hash)
        _validate_row_fields(
            level=level,
            criticality=criticality,
            confidence=confidence,
            origin=origin,
            created_under_intent_kind=created_under_intent_kind,
            stance=stance,
            achievability=achievability,
            status=status,
            success_criteria=success_criteria,
            constraints=constraints,
        )
        if embedding is None:
            encoder = self._ensure_encoder()
            payload = f"{text} | intent: {intent}" if intent else text
            embedding = encoder.encode([payload])[0]
            l2_normalize(embedding)
        elif embedding.shape != (DIM,):
            raise ValueError(f"embedding shape {embedding.shape} != ({DIM},)")

        now = utcnow_iso()
        new_row = Row(
            row_id=_ulid(),
            text=text,
            polarity=list(polarity),
            recurrence_count=recurrence_count,
            created_at=now,
            last_corroborated_at=now,
            transcript_hash=transcript_hash,
            is_bad=False,
            schema_version=SCHEMA_VERSION,
            level=level,
            criticality=criticality,
            confidence=confidence,
            parent_ids=list(parent_ids) if parent_ids else [],
            cited_passages=list(cited_passages) if cited_passages else [],
            origin=origin,
            created_under_intent_kind=created_under_intent_kind,
            stance=stance,
            achievability=achievability,
            status=status,
            success_criteria=(
                [dict(c) for c in success_criteria]
                if success_criteria is not None
                else None
            ),
            constraints=list(constraints) if constraints is not None else None,
        )

        with self._lock():
            rows, band = self._read_state()
            rows.append(new_row)
            band = np.vstack([band, embedding[None, :]])
            _atomic_write_pair(self.root, rows, band)
        return new_row.row_id

    def add_rows_batch(
        self,
        rows: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> list[str]:
        """Bulk-append N rows under a single lock acquisition.

        Each entry in `rows` is a `{text, polarity, transcript_hash,
        intent?}` dict; `embeddings` is the (N, DIM) matrix in matching
        order. Returns the new row_ids in the same order.

        Avoids the O(N²) read-modify-write of N separate `add_row`
        calls — used by the v2.0 → v2.1 migration (§14.4) where N can
        run into the hundreds and the per-call sidecar+band re-read
        would dominate wall time.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != DIM:
            raise ValueError(
                f"embeddings must be (N, {DIM}); got {embeddings.shape}"
            )
        if embeddings.shape[0] != len(rows):
            raise ValueError(
                f"row/embedding count mismatch: rows={len(rows)} "
                f"embeddings={embeddings.shape[0]}"
            )
        if not rows:
            return []
        new_rows: list[Row] = []
        now = utcnow_iso()
        for r in rows:
            _validate_polarity(r["polarity"])
            level = r.get("level", "event")
            criticality = r.get("criticality", "normal")
            confidence = r.get("confidence", "medium")
            origin = r.get("origin") or _derive_origin(r["transcript_hash"])
            cuik = r.get("created_under_intent_kind", "none")
            stance = r.get("stance")
            achievability = r.get("achievability")
            status = r.get("status")
            success_criteria = r.get("success_criteria")
            constraints = r.get("constraints")
            _validate_row_fields(
                level=level,
                criticality=criticality,
                confidence=confidence,
                origin=origin,
                created_under_intent_kind=cuik,
                stance=stance,
                achievability=achievability,
                status=status,
                success_criteria=success_criteria,
                constraints=constraints,
            )
            new_rows.append(Row(
                row_id=_ulid(),
                text=r["text"],
                polarity=list(r["polarity"]),
                recurrence_count=r.get("recurrence_count", 1),
                created_at=now,
                last_corroborated_at=now,
                transcript_hash=r["transcript_hash"],
                is_bad=False,
                schema_version=SCHEMA_VERSION,
                level=level,
                criticality=criticality,
                confidence=confidence,
                parent_ids=list(r.get("parent_ids") or []),
                cited_passages=list(r.get("cited_passages") or []),
                origin=origin,
                created_under_intent_kind=cuik,
                stance=stance,
                achievability=achievability,
                status=status,
                success_criteria=(
                    [dict(c) for c in success_criteria]
                    if success_criteria is not None
                    else None
                ),
                constraints=(
                    list(constraints) if constraints is not None else None
                ),
            ))
        with self._lock():
            existing, band = self._read_state()
            existing.extend(new_rows)
            band = (
                np.vstack([band, embeddings.astype(np.float32, copy=False)])
                if band.size
                else embeddings.astype(np.float32, copy=False)
            )
            _atomic_write_pair(self.root, existing, band)
        return [r.row_id for r in new_rows]

    def update_row(self, row_id: str, **fields: Any) -> Row:
        """Update mutable fields on a row by id. Returns the updated Row.

        Mutable: `recurrence_count`, `last_corroborated_at`, `transcript_hash`,
        `is_bad`, `polarity`, plus the v2.2 axes (`level`, `criticality`,
        `confidence`, `parent_ids`, `cited_passages`, `origin`,
        `created_under_intent_kind`) and intent-only fields (`stance`,
        `achievability`, `status`, `success_criteria`, `constraints`).
        Immutable: `row_id`, `text`, `created_at`, `schema_version`.

        The band is never touched by this path — only the sidecar is rewritten.
        After merge the resulting row is revalidated so updates can't leave
        the row in an inconsistent state (e.g. switching `level` to an intent
        value without populating the intent-only fields).
        """
        immutable = {"row_id", "text", "created_at", "schema_version"}
        bad = set(fields) & immutable
        if bad:
            raise ValueError(f"cannot update immutable fields: {sorted(bad)}")
        if "polarity" in fields:
            _validate_polarity(fields["polarity"])

        with self._lock():
            rows, _ = self._read_state()
            for i, r in enumerate(rows):
                if r.row_id == row_id:
                    merged = {**asdict(r), **fields}
                    _validate_row_fields(
                        level=merged["level"],
                        criticality=merged["criticality"],
                        confidence=merged["confidence"],
                        origin=merged["origin"],
                        created_under_intent_kind=merged[
                            "created_under_intent_kind"
                        ],
                        stance=merged["stance"],
                        achievability=merged["achievability"],
                        status=merged["status"],
                        success_criteria=merged["success_criteria"],
                        constraints=merged["constraints"],
                    )
                    updated = Row(**merged)
                    rows[i] = updated
                    _atomic_write_sidecar(self.root, rows)
                    return updated
            raise KeyError(f"row_id {row_id!r} not in memory")

    def delete_rows(self, row_ids: Iterable[str]) -> int:
        """Delete rows by id. Compacts the band index.

        Returns the number of rows actually deleted (callers can detect
        partial misses without raising).
        """
        targets = set(row_ids)
        if not targets:
            return 0
        with self._lock():
            rows, band = self._read_state()
            keep_mask = np.array([r.row_id not in targets for r in rows], dtype=bool)
            kept = [r for r, k in zip(rows, keep_mask) if k]
            removed = len(rows) - len(kept)
            if removed == 0:
                return 0
            band = band[keep_mask] if band.size else band
            _atomic_write_pair(self.root, kept, band)
            return removed
