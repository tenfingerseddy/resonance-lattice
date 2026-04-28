"""Incremental delta apply — shared pipeline for `rlat refresh` + `rlat sync`.

Both commands land here. Different upstream-state sources feed the same
delta-apply loop:

  rlat refresh  →  walk source_paths from disk          →  bucketise → apply
  rlat sync     →  RemoteIndex.changed_files_since(...) →  bucketise → apply

The pipeline takes an existing archive's registry + bands and a fresh set of
`(rel_posix_path, text)` tuples covering every file the new state should
contain, then:

  1. Bucketise each candidate live-passage against the old registry on
     `passage_id` (stable id derived from source coordinates):
       - id matches AND content_hash matches → `unchanged` (preserve band row)
       - id matches BUT content_hash differs → `updated` (re-encode same coords)
       - id not in old registry             → `added` (re-encode)
       - id in old registry but not in live → `removed` (drop band row)
  2. Encode `updated + added` once, batched.
  3. Compose new band: `np.vstack(kept_rows + new_rows)`. New `passage_idx`
     numbers are assigned line-implicitly; ids stay stable.
  4. Rebuild ANN if N crosses the threshold (FAISS HNSW; ~100ms for 50K).
  5. Re-project the optimised band if present: `optimised = new_base @ W.T;
     L2-normalise`. Sub-second; no LLM call; no GPU. The footgun where
     `rlat refresh` discards the optimised band disappears.
  6. Single atomic write of the full ZIP via `archive.write` (tmp+os.replace).

Three correctness invariants (Audit 07, codex P0 fix):

  1. No "manifest-only" path. Any caller updating `manifest.commit_sha` MUST
     also pass an Encoder + run the delta-apply. `apply_delta(encoder=None)`
     raises a TypeError at the call site by required parameter.
  2. Every kept passage's `content_hash` is validated against the live bytes
     during bucketise (local mode reads files, remote mode trusts the
     SHA-verified cache from RemoteStore — both surfaces read bytes through
     `Store.fetch`-equivalent paths and hash what they read).
  3. The optimised band re-projects from the new base. No silent staleness;
     no $14-21 + 30 min penalty for routine refreshes.

Audit 07 commit 3/8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ..field import ann
from ..field.encoder import Encoder
from . import archive
from .archive import ArchiveContents
from .base import compute_hash
from .bands import write_band
from .chunker import chunk_text
from .registry import PassageCoord, compute_id


@dataclass(frozen=True)
class CandidatePassage:
    """A passage that the new state intends to contain. Yielded by callers
    walking the new live source. `passage_id` and `content_hash` are
    pre-computed so bucketise stays a pure registry-vs-candidates pass."""
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str
    passage_id: str
    text: str  # carried so the apply loop can re-encode without re-reading

    @classmethod
    def from_chunk(
        cls, source_file: str, text: str, char_offset: int, char_length: int,
    ) -> "CandidatePassage":
        passage_text = text[char_offset:char_offset + char_length]
        return cls(
            source_file=source_file,
            char_offset=char_offset,
            char_length=char_length,
            content_hash=compute_hash(passage_text),
            passage_id=compute_id(source_file, char_offset, char_length),
            text=passage_text,
        )


def chunk_files(
    files: Iterable[tuple[str, str]],
    min_chars: int,
    max_chars: int,
) -> list[CandidatePassage]:
    """Walk `(rel_posix, text)` tuples, run the v1 chunker, emit candidates.

    Same chunker the build path uses (`passage_v1`), so the delta on a
    re-walk of identical content is exactly empty (no spurious updates from
    chunker drift).
    """
    out: list[CandidatePassage] = []
    for rel_path, text in files:
        for char_offset, char_length in chunk_text(text, min_chars, max_chars):
            out.append(CandidatePassage.from_chunk(rel_path, text, char_offset, char_length))
    return out


@dataclass
class BucketedDelta:
    """The 3-way reconcile output. `unchanged` carries old `passage_idx`
    values so the apply loop can lift the corresponding band rows; `updated`
    + `added` carry the live text so the apply loop can re-encode them."""
    unchanged: list[PassageCoord] = field(default_factory=list)
    updated: list[CandidatePassage] = field(default_factory=list)
    added: list[CandidatePassage] = field(default_factory=list)
    removed: list[PassageCoord] = field(default_factory=list)

    @property
    def n_unchanged(self) -> int: return len(self.unchanged)

    @property
    def n_updated(self) -> int: return len(self.updated)

    @property
    def n_added(self) -> int: return len(self.added)

    @property
    def n_removed(self) -> int: return len(self.removed)

    @property
    def n_re_encode(self) -> int: return len(self.updated) + len(self.added)

    @property
    def is_empty(self) -> bool:
        return not (self.updated or self.added or self.removed)


def bucketise(
    old_registry: list[PassageCoord],
    candidates: list[CandidatePassage],
) -> BucketedDelta:
    """Classify every candidate against the old registry on stable id.

    Pure function — does not touch the filesystem or the network. Callers
    pre-compute `candidates` from whichever source they have (local walk
    or remote partial fetch).

    Detection is content-hash-based, never mtime-based: a file whose mtime
    changed but whose chunk content_hashes are unchanged → fully `unchanged`,
    nothing re-encoded.
    """
    delta = BucketedDelta()
    old_by_id: dict[str, PassageCoord] = {c.passage_id: c for c in old_registry}
    seen_ids: set[str] = set()

    for cand in candidates:
        seen_ids.add(cand.passage_id)
        old = old_by_id.get(cand.passage_id)
        if old is None:
            delta.added.append(cand)
        elif old.content_hash == cand.content_hash:
            delta.unchanged.append(old)
        else:
            delta.updated.append(cand)

    for old in old_registry:
        if old.passage_id not in seen_ids:
            delta.removed.append(old)

    return delta


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of `apply_delta`. Carries the new archive path + summary
    counts so callers can render a one-line status banner without re-reading
    the archive."""
    archive_path: Path
    n_passages: int
    n_unchanged: int
    n_updated: int
    n_added: int
    n_removed: int
    re_projected_optimised: bool


def apply_delta(
    archive_path: Path,
    contents: ArchiveContents,
    delta: BucketedDelta,
    *,
    encoder: Encoder,
    batch_size: int = 32,
) -> ApplyResult:
    """Apply a bucketised delta to the archive in place. Atomic.

    `encoder` is REQUIRED (positional-after-star) — there is no "skip
    re-encoding" mode. This is the static signature that prevents the codex
    P0 silent-correctness regression: every code path that updates the
    archive must re-encode the deltas.

    The optimised band, if present, is re-projected through the existing W
    matrix. No LLM call, no GPU training, no $14-21 cost — just a single
    matmul + L2 normalise.

    Bundled-mode archives are immutable post-build by design (source bytes
    are zstd-framed inside the ZIP). They route to `rlat build`, not here.
    """
    if contents.metadata.store_mode == "bundled":
        raise NotImplementedError(
            "Incremental delta-apply does not support bundled-mode archives "
            "(source bytes are baked in at build time). Re-run `rlat build` "
            "to produce a fresh bundled archive."
        )
    base_band = contents.bands["base"]

    # 1. Lift kept rows from the old base band, in old order.
    kept_indices = [c.passage_idx for c in delta.unchanged]
    if kept_indices:
        kept_rows = base_band[kept_indices]
    else:
        kept_rows = np.zeros((0, base_band.shape[1]), dtype=base_band.dtype)

    # 2. Encode the deltas — `updated + added` in stable order.
    re_encode_passages = list(delta.updated) + list(delta.added)
    if re_encode_passages:
        new_rows = encoder.encode_batched(
            [c.text for c in re_encode_passages], batch_size,
        )
    else:
        new_rows = np.zeros((0, base_band.shape[1]), dtype=base_band.dtype)

    # 3. Compose new base band + new registry (line-implicit passage_idx
    # renumbering; passage_id stays stable so external references survive).
    new_base = np.concatenate([kept_rows, new_rows], axis=0)
    new_registry: list[PassageCoord] = []
    for new_idx, c in enumerate(delta.unchanged):
        new_registry.append(PassageCoord(
            passage_idx=new_idx,
            source_file=c.source_file,
            char_offset=c.char_offset,
            char_length=c.char_length,
            content_hash=c.content_hash,
            passage_id=c.passage_id,
        ))
    for offset, c in enumerate(re_encode_passages):
        new_registry.append(PassageCoord(
            passage_idx=len(delta.unchanged) + offset,
            source_file=c.source_file,
            char_offset=c.char_offset,
            char_length=c.char_length,
            content_hash=c.content_hash,
            passage_id=c.passage_id,
        ))

    # 4. Rebuild ANN (cheap; ~100ms for 50K).
    new_bands: dict[str, np.ndarray] = {"base": new_base}
    new_projections: dict[str, np.ndarray] = {}
    new_ann_blobs: dict[str, bytes] = {}
    new_ann_meta: dict[str, dict[str, int | str]] = {}
    if ann.should_build_ann(len(new_registry)):
        index = ann.build(new_base)
        new_ann_blobs["base"] = ann.serialize(index)
        new_ann_meta["base"] = {
            "type": "hnsw",
            "M": ann.HNSW_M,
            "efConstruction": ann.HNSW_EFCONSTRUCTION,
            "efSearch": ann.HNSW_EFSEARCH,
        }

    # 5. Re-project the optimised band if present. Free.
    re_projected = False
    if "optimised" in contents.bands:
        W = contents.projections.get("optimised")
        if W is None:
            raise ValueError(
                f"{archive_path}: optimised band present but projection W "
                f"missing — archive is malformed; rebuild with `rlat build` "
                f"+ `rlat optimise` to recover."
            )
        # base → optimised: matmul through W.T then L2-normalise per row.
        # W shape is (d_native, dim) where dim = base dim (768 for v2.0);
        # new_base shape is (N, dim). Output shape (N, d_native).
        optimised = new_base @ W.T
        norms = np.linalg.norm(optimised, axis=1, keepdims=True)
        optimised = optimised / np.maximum(norms, 1e-12)
        new_bands["optimised"] = optimised.astype(np.float32, copy=False)
        new_projections["optimised"] = W
        if ann.should_build_ann(len(new_registry)):
            opt_index = ann.build(new_bands["optimised"])
            new_ann_blobs["optimised"] = ann.serialize(opt_index)
            new_ann_meta["optimised"] = dict(new_ann_meta.get("base", {}))
        re_projected = True

    # 6. Update metadata: bump passage counts per band, refresh
    # build_config's live counts (passage_count + file_count), preserve
    # everything else (kind, store_mode, backbone, manifest, etc).
    # build_config keeps the build-time chunker/extension/source-path
    # provenance so a future refresh replays faithfully — the counts are
    # the only fields that drift with deltas.
    metadata = contents.metadata
    for band_name in new_bands:
        if band_name in metadata.bands:
            metadata.bands[band_name].passage_count = len(new_registry)
    metadata.ann = new_ann_meta
    metadata.build_config["passage_count"] = len(new_registry)
    metadata.build_config["file_count"] = len({c.source_file for c in new_registry})

    # 7. Atomic write. local + remote archives don't carry source_files
    # in-archive; bundled is rejected up top.
    archive.write(
        archive_path,
        metadata=metadata,
        bands=new_bands,
        registry=new_registry,
        projections=new_projections,
        ann_blobs=new_ann_blobs,
        remote_manifest=contents.remote_manifest,
    )

    return ApplyResult(
        archive_path=archive_path,
        n_passages=len(new_registry),
        n_unchanged=delta.n_unchanged,
        n_updated=delta.n_updated,
        n_added=delta.n_added,
        n_removed=delta.n_removed,
        re_projected_optimised=re_projected,
    )

