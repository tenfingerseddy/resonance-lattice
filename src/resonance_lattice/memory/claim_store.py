"""ExperienceClaimStore — the per-user experience-claim backend.

Per-user `claims.jsonl` + `band.npz` pair under `~/.rlat/memory/<user-id>/`.

A claim's embedding is the store's parallel band array, keyed by row
position; `Claim` itself carries no vector (design note §3.1). Writes are
atomic — portalocker advisory lock + tmp + os.replace — mirroring the v2.0
archive write contracts.

Only `experience`-source claims live here; their `facts` is always an
`ExperienceFacts`. Each claim row is the core fields plus the `facts`
fields flattened alongside them (the two field-name sets are disjoint).
The core-field serialisation and the `write_many` / `delete` merge
plumbing live in `state/claim_io.py`.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import portalocker

from ..field.encoder import DIM, Encoder
from ..state.claim import EXPERIENCE_KINDS, Claim, ExperienceFacts, derive_origin
from ..state.claim_io import (
    delete_claims,
    experience_claim_from_row as _row_to_claim,
    experience_claim_to_row as _claim_to_row,
    merge_claims,
)
from ._common import make_ulid, utcnow_iso
from .store import path_for_user, seed_tallies_for_rung

CLAIMS_FILE = "claims.jsonl"
BAND_FILE = "band.npz"
LOCK_FILE = ".claims.lock"

# Legacy pre-Stage-4 store files, imported once by `_migrate_legacy_sidecar`.
_LEGACY_CLAIMS = "sidecar.jsonl"
_LEGACY_BAND = "memory.npz"

# Claim kinds an experience row may carry — the memory ladder. A pre-Stage-3
# legacy sidecar may still hold intent-level rows; the migration drops them.
_MEMORY_KINDS: frozenset[str] = frozenset(
    {"event", "pattern", "learning", "principle"}
)

def new_experience_claim(
    *,
    content: str,
    polarity: tuple[str, ...],
    transcript_hash: str,
    kind: str = "event",
    rung: str = "medium",
    parent_ids: tuple[str, ...] = (),
    recurrence_count: int = 1,
    criticality: str = "normal",
    created_under_intent_kind: str = "none",
    attribute_key: str = "",
) -> Claim:
    """Mint a fresh experience `Claim` — the one place the `make_ulid` /
    `source` / `created_at` / Beta-seed / `origin` plumbing lives.

    `rung` seeds the Beta tallies (`confidence` is derived from them, never
    stored); `origin` is implied by `transcript_hash`; `created_at` and
    `last_corroborated_at` are stamped now. For reconstructing a *historical*
    claim — distinct created/corroborated times, explicit tallies — build
    the `Claim` directly; this is for newly-minted claims only.

    Born `candidate`, not `active`: an asserted (uncited) claim earns
    retrievability through `state.claim_lifecycle.consolidate_experience`
    (recurrence + outcome trust), the experience analog of the corpus
    compression gate. The per-user `ExperienceClaimStore` recall is
    state-blind, so this is inert there; it gains teeth on the unified band,
    whose retrieval surfaces only `active` claims.
    """
    now = utcnow_iso()
    corroboration, falsification = seed_tallies_for_rung(rung)
    return Claim(
        claim_id=make_ulid(),
        source="experience",
        kind=kind,
        content=content,
        created_at=now,
        corroboration=corroboration,
        falsification=falsification,
        trust_as_of="",
        state="candidate",
        parent_ids=tuple(parent_ids),
        facts=ExperienceFacts(
            polarity=tuple(polarity),
            recurrence_count=recurrence_count,
            criticality=criticality,
            created_under_intent_kind=created_under_intent_kind,
            transcript_hash=transcript_hash,
            origin=derive_origin(transcript_hash),
            last_corroborated_at=now,
            is_bad=False,
            attribute_key=attribute_key,
        ),
    )


class ExperienceClaimStore:
    """Per-user experience-claim store at `~/.rlat/memory/<user-id>/`.

    Every mutating call acquires the portalocker lock, re-reads disk so
    concurrent writers are visible, applies the mutation, and writes
    atomically — the same re-read-under-lock contract `Memory` used.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        user_id: str | None = None,
        encoder: Encoder | None = None,
    ):
        self.root = (
            Path(root) if root is not None
            else path_for_user(user_id=user_id)
        )
        self._encoder = encoder
        # POSIX: create with mode=0o700 so a fresh install has no window
        # between mkdir and chmod where the dir is world-readable. The
        # follow-up `chmod` tightens existing-install dirs that were
        # created before this hardening landed (idempotent). Windows
        # ignores `mode=` and relies on inherited ACLs — both paths
        # produce the right end-state on each platform.
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            try:
                os.chmod(self.root, 0o700)
            except OSError:
                pass
        # Pre-create the lock file so portalocker.Lock(mode="r+b") doesn't
        # stat-then-touch on every acquisition.
        self._lock_path = self.root / LOCK_FILE
        self._lock_path.touch(exist_ok=True)

    # -- internals --------------------------------------------------------

    def _ensure_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder()
        return self._encoder

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def _migrate_legacy_sidecar(self) -> None:
        """One-shot import of a pre-Stage-4 `sidecar.jsonl` + `memory.npz`
        into the claim store. Runs once — the legacy files are renamed
        `.migrated` after a successful write so it never re-fires. The old
        rows are read as raw dicts (no `Row` class needed); intent-level
        rows a pre-Stage-3 sidecar may still carry are dropped.

        The legacy store's own `.lock` is held across the read+rename so a
        racing pre-cutover `Memory` writer can't be read mid-update. The
        caller already holds the claim-store lock."""
        sidecar = self.root / _LEGACY_CLAIMS
        legacy_band = self.root / _LEGACY_BAND
        legacy_lock = self.root / ".lock"
        legacy_lock.touch(exist_ok=True)
        with portalocker.Lock(
            str(legacy_lock), mode="r+b", flags=portalocker.LOCK_EX,
        ):
            self._import_legacy(sidecar, legacy_band)

    def _import_legacy(self, sidecar: Path, legacy_band: Path) -> None:
        """Read the legacy pair, write the claim store, archive the old
        files. Runs under both the claim-store and legacy locks."""
        raw = [
            json.loads(line)
            for line in sidecar.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if legacy_band.exists():
            with np.load(legacy_band) as z:
                band = z["band"]
        else:
            band = np.zeros((0, DIM), dtype=np.float32)
        if band.shape[0] != len(raw):
            raise ValueError(
                f"legacy store corrupt: {len(raw)} sidecar rows, "
                f"{band.shape[0]} band rows"
            )
        claims: list[Claim] = []
        keep: list[int] = []
        for i, d in enumerate(raw):
            kind = d.get("level", "event")
            if kind not in _MEMORY_KINDS:
                continue  # intent row — not a claim
            corr = d.get("corroboration", 0.0)
            fals = d.get("falsification", 0.0)
            if corr == 0.0 and fals == 0.0:
                # Pre-v2.2 rows lack tallies; an is_bad row without an
                # explicit rung seeds `low`, matching the old default-fill.
                default_rung = "low" if d.get("is_bad") else "medium"
                corr, fals = seed_tallies_for_rung(
                    d.get("confidence", default_rung))
            created = d.get("created_at", "")
            claims.append(Claim(
                claim_id=d["row_id"],
                source="experience",
                kind=kind,
                content=d["text"],
                created_at=created,
                corroboration=corr,
                falsification=fals,
                trust_as_of=d.get("trust_as_of", ""),
                state="active",
                parent_ids=tuple(d.get("parent_ids") or ()),
                facts=ExperienceFacts(
                    polarity=tuple(d.get("polarity") or ()),
                    recurrence_count=d.get("recurrence_count", 1),
                    criticality=d.get("criticality", "normal"),
                    created_under_intent_kind=d.get(
                        "created_under_intent_kind", "none"),
                    transcript_hash=d.get("transcript_hash", "manual"),
                    # Pre-v2.2 rows lack `origin`; derive it from the
                    # transcript-hash prefix, matching the old default-fill.
                    origin=d.get("origin")
                    or derive_origin(d.get("transcript_hash", "manual")),
                    last_corroborated_at=d.get("last_corroborated_at", created),
                    is_bad=d.get("is_bad", False),
                ),
            ))
            keep.append(i)
        new_band = (
            band[keep] if keep else np.zeros((0, DIM), dtype=np.float32)
        )
        self._atomic_write(claims, new_band)
        sidecar.rename(sidecar.with_suffix(".jsonl.migrated"))
        if legacy_band.exists():
            legacy_band.rename(legacy_band.with_suffix(".npz.migrated"))

    def _read_state(self) -> tuple[list[Claim], np.ndarray]:
        """Load `(claims, band)`. Caller must hold the lock."""
        claims_path = self.root / CLAIMS_FILE
        if not claims_path.exists() and (self.root / _LEGACY_CLAIMS).exists():
            self._migrate_legacy_sidecar()
        all_rows: list[dict] = []
        if claims_path.exists():
            for line in claims_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    all_rows.append(json.loads(line))
        band_path = self.root / BAND_FILE
        if not band_path.exists():
            if all_rows:
                raise ValueError(
                    f"{claims_path} has {len(all_rows)} claims but {band_path} "
                    f"is missing — store corrupt"
                )
            return [], np.zeros((0, DIM), dtype=np.float32)
        with np.load(band_path) as z:
            band = z["band"]
        if band.shape[0] != len(all_rows):
            raise ValueError(
                f"claims/band row mismatch: {len(all_rows)} claims, "
                f"{band.shape[0]} band rows"
            )
        if band.shape[1] != DIM:
            raise ValueError(
                f"band dim {band.shape[1]} != expected {DIM} — encoder mismatch"
            )
        # Drop distilled-ladder kinds (pattern/learning/principle) a prior
        # dogfood store may carry — they have no live consumer post-Phase
        # B/4. Events alone survive; position-keyed band rows go with them.
        # The store is rewritten in-shape on the next write_many.
        keep_idx = [
            i for i, r in enumerate(all_rows)
            if r.get("kind") in EXPERIENCE_KINDS
        ]
        dropped = len(all_rows) - len(keep_idx)
        if dropped:
            print(
                f"[memory] dropped {dropped} non-event claim(s) from "
                f"{claims_path} on load (Phase B/4 ladder retirement)",
                file=sys.stderr,
            )
        claims = [_row_to_claim(all_rows[i]) for i in keep_idx]
        if dropped:
            band = band[keep_idx] if keep_idx else np.zeros((0, DIM), dtype=np.float32)
        return claims, band.astype(np.float32, copy=False)

    def _atomic_write(self, claims: list[Claim], band: np.ndarray) -> None:
        """Tmp + os.replace for both files. Caller must hold the lock.

        Tmp filenames carry pid + random hex so two processes that
        bypass the portalocker lock (e.g. a debug script writing
        directly) don't collide on `claims.jsonl.tmp` / `band.npz.tmp`
        and silently truncate each other. The lock remains the primary
        guarantee against lost updates; the unique tmp filename is
        defence-in-depth on the corruption boundary, matching the
        contract in `store/archive.py::_unique_tmp_path`.
        """
        if band.shape[0] != len(claims):
            raise ValueError(
                f"row count mismatch: claims={len(claims)} band={band.shape[0]}"
            )
        suffix = f".{os.getpid()}.{secrets.token_hex(4)}.tmp"
        claims_path = self.root / CLAIMS_FILE
        tmp = claims_path.with_suffix(claims_path.suffix + suffix)
        tmp.write_text(
            "\n".join(
                json.dumps(_claim_to_row(c), sort_keys=True) for c in claims
            ),
            encoding="utf-8",
        )
        # Tighten the tmp file before `os.replace` so the mode survives
        # the rename (POSIX `rename` preserves mode of the source inode).
        # Skipped on Windows; ACLs inherit from the parent dir.
        if os.name != "nt":
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
        os.replace(tmp, claims_path)
        band_path = self.root / BAND_FILE
        band_tmp = band_path.with_suffix(band_path.suffix + suffix)
        # np.savez appends `.npz` to a bare path, then writes there — the
        # file-handle form sidesteps that so the tmp file lands exactly
        # where os.replace expects.
        with open(band_tmp, "wb") as f:
            np.savez(f, band=np.ascontiguousarray(band, dtype=np.float32))
        if os.name != "nt":
            try:
                os.chmod(band_tmp, 0o600)
            except OSError:
                pass
        os.replace(band_tmp, band_path)

    # -- public API ------------------------------------------------------

    def read_all(self) -> list[Claim]:
        """Every claim in the store."""
        with self._lock():
            claims, _ = self._read_state()
            return claims

    def read_all_with_band(self) -> tuple[list[Claim], np.ndarray]:
        """`(claims, band)` snapshot — the band is what recall ranks on."""
        with self._lock():
            claims, band = self._read_state()
            return claims, band.copy()

    def read(self, claim_id: str) -> Claim | None:
        """One claim by id, or `None` if absent."""
        with self._lock():
            claims, _ = self._read_state()
            for claim in claims:
                if claim.claim_id == claim_id:
                    return claim
            return None

    def write(self, claim: Claim, *, embedding: np.ndarray | None = None) -> None:
        """Insert `claim`, or replace the claim with the same `claim_id`.

        `embedding` is optional — omitted, the band vector is reused when
        the claim's `content` is unchanged, else `content` is re-encoded.
        """
        self.write_many(
            [claim],
            embeddings=None if embedding is None else embedding[None, :],
        )

    def write_many(
        self,
        claims: list[Claim],
        *,
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Insert-or-replace a batch of claims under one store update.

        `embeddings` is the optional (N, DIM) matrix in `claims` order; when
        omitted, each claim's band vector is reused if its `content` is
        unchanged and otherwise encoded from `content` in one batch.
        """
        if not claims:
            return
        with self._lock():
            existing, band = self._read_state()
            rows, new_band = merge_claims(
                existing, band, claims,
                embeddings=embeddings,
                encoder_provider=self._ensure_encoder,
            )
            self._atomic_write(rows, new_band)

    def delete(self, claim_ids: list[str]) -> int:
        """Delete claims by id. Returns the count actually removed."""
        if not claim_ids:
            return 0
        with self._lock():
            existing, band = self._read_state()
            kept, new_band, removed = delete_claims(existing, band, claim_ids)
            if removed == 0:
                return 0
            self._atomic_write(kept, new_band)
            return removed
