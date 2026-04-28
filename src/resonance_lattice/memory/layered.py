"""LayeredMemory — three-tier memory over the v2.0 encoder.

Each tier is a `(jsonl, npy)` pair on disk under `memory_root/`:

  memory_root/
    working.jsonl    # one MemoryEntry per line (no embedding column)
    working.npy      # (N, 768) float32 embeddings, L2-normalised
    episodic.jsonl   episodic.npy
    semantic.jsonl   semantic.npy

This is *not* the v0.11 LayeredMemory shape — v0.11 used per-tier `Lattice`
(.rlat) instances with the chunker + ANN machinery. v2.0 separates
"build-time content corpus" (.rlat) from "runtime append-only memory" because
a single `rlat memory add "..."` call shouldn't rebuild a chunker, write a
ZIP, and run an ANN constructor. JSONL+NPY is what fits the append shape.

Retrieval: cosine across all entries in active tiers, weighted by tier and
by per-entry salience. No ANN — at v2.0-typical memory sizes (≤ a few
thousand entries total) exact dense is sub-millisecond and avoids the
churn of rebuilding HNSW on every `add`.

Phase 5 deliverable. Base plan §Phase 5 + memory/__init__.py docstring.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import DIM, Encoder

TIER_NAMES = ("working", "episodic", "semantic")

# Defaults match v0.11: working dominates fresh recall, semantic is the
# stable knowledge base. Tier weights are an implicit UX detail; not a knob.
DEFAULT_TIER_WEIGHTS: dict[str, float] = {
    "working": 0.5,
    "episodic": 0.3,
    "semantic": 0.2,
}


@dataclass
class MemoryEntry:
    """One row in a tier. The `embedding` is stored in the `.npy` sibling
    file (one row per JSONL line, same order); `to_jsonl_dict` strips it."""
    text: str
    salience: float = 1.0
    source_id: str = ""
    session: str | None = None
    created_utc: str = ""  # ISO 8601, set on add
    tier: str = "working"
    recurrence_count: int = 1
    # Populated only when loaded — caller doesn't pass it on add.
    embedding: np.ndarray | None = None

    def to_jsonl_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tier_paths(root: Path, tier: str) -> tuple[Path, Path]:
    return root / f"{tier}.jsonl", root / f"{tier}.npy"


def _load_tier(root: Path, tier: str) -> tuple[list[MemoryEntry], np.ndarray]:
    """Load (entries, embeddings). Empty arrays / lists if files missing.

    Recovery: a crash between the two `os.replace` calls in `_save_tier`
    leaves the JSONL renamed (new state) but the NPY still old (and a
    `tmp_npy` orphan on disk). We detect the mismatch + complete the
    second rename before raising — turns silent corruption into a
    self-healing recovery on the common interrupted-pair-write path.
    """
    jsonl_path, npy_path = _tier_paths(root, tier)
    if not jsonl_path.exists():
        return [], np.zeros((0, DIM), dtype=np.float32)
    entries: list[MemoryEntry] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        entries.append(MemoryEntry(**obj))

    # Self-heal an interrupted pair-write: if a tmp_npy is still hanging
    # around AND the live npy is missing or row-count-stale relative to
    # the jsonl, complete the second os.replace.
    tmp_npy = npy_path.with_suffix(npy_path.suffix + ".tmp")
    if tmp_npy.exists():
        npy_stale = (not npy_path.exists()) or _safe_npy_rowcount(npy_path) != len(entries)
        if npy_stale and _safe_npy_rowcount(tmp_npy) == len(entries):
            os.replace(tmp_npy, npy_path)
        else:
            # Either the tmp doesn't match the jsonl either (write hadn't
            # finished even the tmp) or the live npy is already current
            # and the tmp is stale orphan cruft. Clean up the tmp; the
            # state checks below decide whether to raise.
            tmp_npy.unlink(missing_ok=True)

    if not npy_path.exists():
        raise ValueError(
            f"tier {tier!r} corrupt: {jsonl_path} has {len(entries)} rows but "
            f"{npy_path} is missing and no recoverable tmp was found. "
            f"Re-init the tier or restore from the prior state."
        )
    embeddings = np.load(npy_path)
    if embeddings.shape[0] != len(entries):
        raise ValueError(
            f"tier {tier!r} corrupt: {jsonl_path} has {len(entries)} rows but "
            f"{npy_path} has {embeddings.shape[0]}. Re-init the tier."
        )
    return entries, embeddings


def _safe_npy_rowcount(path: Path) -> int:
    """Return row 0 of the NPY at `path`, or -1 if the file is unreadable.

    Used by `_load_tier`'s self-heal — we need to peek at a tmp file's
    row count to decide whether it's the recovery target, but a partial
    write of the tmp itself shouldn't itself raise.
    """
    try:
        return int(np.load(path).shape[0])
    except Exception:
        return -1


def _save_tier(
    root: Path, tier: str, entries: Sequence[MemoryEntry], embeddings: np.ndarray,
) -> None:
    """Atomic save of a tier's JSONL + NPY. Tmp + os.replace pair so a crash
    mid-write leaves the prior state intact."""
    if embeddings.shape[0] != len(entries):
        raise ValueError(
            f"row mismatch: entries={len(entries)} embeddings={embeddings.shape[0]}"
        )
    jsonl_path, npy_path = _tier_paths(root, tier)
    root.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    tmp_npy = npy_path.with_suffix(npy_path.suffix + ".tmp")
    tmp_jsonl.write_text(
        "\n".join(json.dumps(e.to_jsonl_dict(), sort_keys=True) for e in entries),
        encoding="utf-8",
    )
    # np.save auto-appends `.npy` when the path doesn't end in `.npy`
    # (e.g. `working.npy.tmp` would become `working.npy.tmp.npy`). Passing
    # a file handle bypasses the suffix logic so the tmp file lands at the
    # path os.replace expects.
    with open(tmp_npy, "wb") as f:
        np.save(f, np.ascontiguousarray(embeddings, dtype=np.float32))
    os.replace(tmp_jsonl, jsonl_path)
    os.replace(tmp_npy, npy_path)


@dataclass
class _TierState:
    entries: list[MemoryEntry] = field(default_factory=list)
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, DIM), dtype=np.float32),
    )


class LayeredMemory:
    """Three-tier memory orchestrator. Tier state is held in RAM after load
    so successive add/recall calls don't re-read disk; flushed to JSONL+NPY
    explicitly via `save()` (called by `add` after each append).

    **Embedding-staleness contract on returned `MemoryEntry`s**: `recall`
    and `all_entries` mutate `entry.embedding` in place to attach the
    embedding from the tier's npy slice — so a single returned entry's
    `.embedding` is valid only between that call and the next call that
    re-touches the same tier. Callers that need to hold an embedding past
    a re-query should `entry.embedding.copy()` it. Single-shot CLI usage
    isn't affected; long-lived API consumers should be aware."""

    def __init__(self, memory_root: str | Path, encoder: Encoder | None = None):
        self.root = Path(memory_root)
        self._encoder = encoder
        self._tiers: dict[str, _TierState] = {}
        for name in TIER_NAMES:
            entries, embeddings = _load_tier(self.root, name)
            self._tiers[name] = _TierState(entries=entries, embeddings=embeddings)

    @classmethod
    def init(cls, memory_root: str | Path) -> "LayeredMemory":
        """Initialise an empty memory root. Idempotent — existing tiers are
        preserved, only missing tiers are created."""
        root = Path(memory_root)
        root.mkdir(parents=True, exist_ok=True)
        for name in TIER_NAMES:
            jsonl, npy = _tier_paths(root, name)
            if not jsonl.exists():
                jsonl.write_text("", encoding="utf-8")
            if not npy.exists():
                # File-handle form so np.save doesn't append a stray `.npy`
                # to a path that already has it.
                with open(npy, "wb") as f:
                    np.save(f, np.zeros((0, DIM), dtype=np.float32))
        return cls(root)

    def _ensure_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder()
        return self._encoder

    def _save_tier_state(self, tier: str) -> None:
        st = self._tiers[tier]
        _save_tier(self.root, tier, st.entries, st.embeddings)

    def add(
        self,
        text: str,
        tier: str = "working",
        salience: float = 1.0,
        source_id: str = "",
        session: str | None = None,
    ) -> MemoryEntry:
        """Encode + append to `tier`. Atomic on disk via `_save_tier`'s
        tmp + os.replace pair."""
        if tier not in TIER_NAMES:
            raise ValueError(f"unknown tier {tier!r}; valid: {TIER_NAMES}")
        encoder = self._ensure_encoder()
        emb = encoder.encode([text])[0]
        l2_normalize(emb)
        entry = MemoryEntry(
            text=text, salience=salience, source_id=source_id, session=session,
            created_utc=_utcnow_iso(), tier=tier, recurrence_count=1,
            embedding=emb,
        )
        st = self._tiers[tier]
        st.entries.append(entry)
        st.embeddings = np.vstack([st.embeddings, emb[None, :]])
        self._save_tier_state(tier)
        return entry

    def add_many(
        self,
        texts: Sequence[str],
        tier: str = "working",
        source_id: str = "",
        session: str | None = None,
    ) -> list[MemoryEntry]:
        """Batched add — one encoder call for the whole list, one disk write
        at the end. Use this when ingesting from a transcript or doc batch."""
        if not texts:
            return []
        if tier not in TIER_NAMES:
            raise ValueError(f"unknown tier {tier!r}; valid: {TIER_NAMES}")
        encoder = self._ensure_encoder()
        embs = encoder.encode(list(texts))
        l2_normalize(embs)
        now = _utcnow_iso()
        new_entries = [
            MemoryEntry(
                text=text, salience=1.0, source_id=source_id, session=session,
                created_utc=now, tier=tier, recurrence_count=1,
                embedding=embs[i],
            )
            for i, text in enumerate(texts)
        ]
        st = self._tiers[tier]
        st.entries.extend(new_entries)
        st.embeddings = np.vstack([st.embeddings, embs])
        self._save_tier_state(tier)
        return new_entries

    def recall(
        self,
        query: str,
        top_k: int = 10,
        tier_weights: dict[str, float] | None = None,
        tiers: Iterable[str] | None = None,
    ) -> list[tuple[float, MemoryEntry]]:
        """Cross-tier weighted recall.

        Score = `cosine(q, entry.embedding) * tier_weight[tier] * salience`.
        Returns `[(score, entry), ...]` sorted descending. Scores from
        different tiers are directly comparable because the per-tier weight
        is folded in.
        """
        weights = tier_weights or DEFAULT_TIER_WEIGHTS
        active = list(tiers) if tiers is not None else list(TIER_NAMES)
        encoder = self._ensure_encoder()
        q_emb = encoder.encode([query])[0]
        l2_normalize(q_emb)
        scored: list[tuple[float, MemoryEntry]] = []
        for tier in active:
            st = self._tiers.get(tier)
            if st is None or not st.entries:
                continue
            sims = st.embeddings @ q_emb
            tw = weights.get(tier, 0.1)
            for i, entry in enumerate(st.entries):
                # Re-attach the embedding from the npy slice so callers that
                # need it (consolidation, near-dup checks) don't re-encode.
                entry.embedding = st.embeddings[i]
                score = float(sims[i]) * tw * entry.salience
                scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def tier_size(self, tier: str) -> int:
        return len(self._tiers[tier].entries)

    def all_entries(self, tier: str) -> list[MemoryEntry]:
        """Return entries with embeddings re-attached. Used by retention
        and consolidation."""
        st = self._tiers[tier]
        for i, entry in enumerate(st.entries):
            entry.embedding = st.embeddings[i]
        return list(st.entries)

    def replace_tier(
        self, tier: str, entries: list[MemoryEntry], embeddings: np.ndarray,
    ) -> None:
        """Wholesale replace a tier's contents — used by retention.gc and
        consolidation when rewriting after promotion / decay."""
        if tier not in TIER_NAMES:
            raise ValueError(f"unknown tier {tier!r}")
        if embeddings.shape[0] != len(entries):
            raise ValueError(
                f"row mismatch: entries={len(entries)} embeddings={embeddings.shape[0]}"
            )
        self._tiers[tier] = _TierState(entries=entries, embeddings=embeddings)
        self._save_tier_state(tier)

    def append_to_tier(
        self, tier: str, entry: MemoryEntry, embedding: np.ndarray,
    ) -> None:
        """Append a single (entry, embedding) pair to `tier` and persist.

        Public API for consolidation's "promote to semantic" path so that
        module doesn't need to reach into `_tiers[tier]` directly. Bypasses
        the encoder (caller already has the embedding). For new text use
        `add()` / `add_many()` instead.
        """
        if tier not in TIER_NAMES:
            raise ValueError(f"unknown tier {tier!r}; valid: {TIER_NAMES}")
        if embedding.shape != (self._tiers[tier].embeddings.shape[1],):
            raise ValueError(
                f"embedding shape {embedding.shape} != tier dim "
                f"{self._tiers[tier].embeddings.shape[1]}"
            )
        st = self._tiers[tier]
        st.entries.append(entry)
        st.embeddings = np.vstack([st.embeddings, embedding[None, :]])
        self._save_tier_state(tier)
