"""Flat memory store — v2.1.

Per-user `memory.npz` + `sidecar.jsonl` pair under `~/.rlat/memory/<user-id>/`.
Atomic write via `portalocker` advisory lock + tmp + os.replace pair, mirroring
v2.0 archive write contracts.

The 9-field row schema is locked in `.claude/plans/fabric-agent-flat-memory.md`
§0.2; this module is the single source of truth on disk for that schema.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import portalocker

from ..field._runtime_common import l2_normalize
from ..field.encoder import DIM, Encoder
from ._common import utcnow_iso

SCHEMA_VERSION = 1

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

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


def _ulid() -> str:
    """26-char Crockford base-32 ULID. Stdlib-only.

    Encodes 48-bit ms timestamp + 80-bit randomness. Lexicographically
    sortable, collision-safe across machines.
    """
    import datetime as _dt
    ts_ms = int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)
    rand_bits = secrets.randbits(80)
    value = (ts_ms << 80) | rand_bits
    return "".join(_CROCKFORD[(value >> (5 * (25 - i))) & 0b11111] for i in range(26))


@dataclass(frozen=True)
class Row:
    """One sidecar row. The 9-field schema from §0.2 of the v2.1 plan.

    Frozen because the v2.1 store always rebuilds rows on update — a
    mutation goes through `Memory.update_row(row_id, **fields)` which
    constructs a new Row from `asdict(old) | fields`.
    """

    row_id: str
    text: str
    polarity: Polarity
    recurrence_count: int
    created_at: str
    last_corroborated_at: str
    transcript_hash: str
    is_bad: bool
    schema_version: int = SCHEMA_VERSION

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
    ) -> str:
        """Append a row. Returns the new row_id (ULID).

        `embedding` is optional — if omitted, the encoder is loaded lazily
        and runs `text + " | intent: " + intent` (per §0.2 last paragraph).
        Callers with a pre-computed embedding (distil, migration) pass it
        directly to skip the encoder load.
        """
        _validate_polarity(polarity)
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
            recurrence_count=1,
            created_at=now,
            last_corroborated_at=now,
            transcript_hash=transcript_hash,
            is_bad=False,
            schema_version=SCHEMA_VERSION,
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
            new_rows.append(Row(
                row_id=_ulid(),
                text=r["text"],
                polarity=list(r["polarity"]),
                recurrence_count=1,
                created_at=now,
                last_corroborated_at=now,
                transcript_hash=r["transcript_hash"],
                is_bad=False,
                schema_version=SCHEMA_VERSION,
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

        Mutable per §0.2: `recurrence_count`, `last_corroborated_at`,
        `transcript_hash`, `is_bad`, `polarity`. Immutable: `row_id`,
        `text`, `created_at`, `schema_version`.

        The band is never touched by this path — only the sidecar is rewritten.
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
                    updated = Row(**{**asdict(r), **fields})
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
