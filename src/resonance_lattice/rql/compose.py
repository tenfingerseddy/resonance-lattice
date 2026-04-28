"""Multi-knowledge-model composition ops.

`compose` — federated search over N knowledge models, attribution preserved.
`merge`   — physically combine two KMs into a new .rlat, semantic dedupe.

Both are CROSS-MODEL ops and use the base band per the cross-model rule.
A `compose` call against a mix of optimised + base KMs ignores the
optimised bands; the federated search runs entirely on the comparable
base bands.

`diff` from the original Phase 6 hypothesis was dropped: it's `unique` with
a different default threshold — same operation, different framing. The
single-op surface (`unique`) is the honest one.

Phase 6 deliverable.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.algebra import greedy_cluster
from ..field.dense import topk_indices
from ..store import archive
from ..store.archive import ArchiveContents
from ..store.metadata import BandInfo, Metadata
from ..store.registry import PassageCoord
from .compare import _require_compatible_pair
from .types import Citation, ComposedHit


class ComposedKnowledgeModel:
    """Read-only federated view over multiple knowledge models.

    Constructed by `compose(**named_kms)`. Each search runs against every
    member's base band and returns globally-ranked `ComposedHit`s with
    per-result corpus attribution. No new .rlat is written; this is a
    runtime view, not a persistent artefact (use `merge` for that).

    Backbone-revision check at construction: all members must share a
    revision OR the caller is warned-but-allowed (raises on dim mismatch
    because that's incompatibility; warns on revision mismatch because
    cosine ordering still holds).
    """

    def __init__(self, named_kms: dict[str, ArchiveContents]):
        if not named_kms:
            raise ValueError("compose requires at least one knowledge model")
        self._members: dict[str, tuple[np.ndarray, list[PassageCoord]]] = {}
        first_dim: int | None = None
        first_revision: str | None = None
        mismatched_revisions: list[tuple[str, str]] = []
        for label, contents in named_kms.items():
            if "base" not in contents.bands:
                raise ValueError(
                    f"compose member {label!r} has no base band — cross-model "
                    f"composition requires every KM to carry one"
                )
            handle = contents.select_band(prefer="base")
            if first_dim is None:
                first_dim = handle.band.shape[1]
                first_revision = contents.metadata.backbone.revision
            elif handle.band.shape[1] != first_dim:
                raise ValueError(
                    f"compose member {label!r} base-band dim "
                    f"{handle.band.shape[1]} != {first_dim}; cannot federate "
                    f"across mismatched dims"
                )
            elif contents.metadata.backbone.revision != first_revision:
                mismatched_revisions.append(
                    (label, contents.metadata.backbone.revision)
                )
            self._members[label] = (handle.band, contents.registry)
        if mismatched_revisions:
            # Honour the documented contract: cosine ordering still holds
            # across distinct embedding distributions, but magnitudes are
            # incomparable. Surface as a UserWarning so callers see it
            # without a hard error.
            details = ", ".join(f"{lbl}={rev}" for lbl, rev in mismatched_revisions)
            warnings.warn(
                f"compose: backbone revision mismatch (expected {first_revision}, "
                f"differs in: {details}). Cosine ordering still meaningful; "
                f"magnitude comparisons are across different embedding distributions.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def labels(self) -> list[str]:
        """Corpus labels in insertion order."""
        return list(self._members)

    def search(
        self, query_embedding: np.ndarray, *, top_k: int = 10,
    ) -> list[ComposedHit]:
        """Federated cosine top-k across all members. Returns `top_k` hits
        sorted by descending score, each tagged with its source corpus.

        `query_embedding` is expected L2-normalised in the same dim as the
        members' base bands (caller's responsibility — typically a 768d
        gte-mb-base output).
        """
        if top_k <= 0:
            return []
        # Per-member top-k; merge the candidate pool, take global top-k.
        # Per-member capped at top_k each: any single member's score below
        # the global #top_k is also below its own #top_k, so the union of
        # per-member top_k lists necessarily contains the global top_k.
        candidates: list[tuple[float, str, int]] = []
        for label, (band, registry) in self._members.items():
            if band.shape[0] == 0:
                continue
            scores = band @ query_embedding
            k = min(top_k, band.shape[0])
            ordered = topk_indices(scores, k)
            for idx in ordered:
                candidates.append((float(scores[idx]), label, int(idx)))
        candidates.sort(key=lambda t: -t[0])
        out: list[ComposedHit] = []
        for score, label, idx in candidates[:top_k]:
            registry = self._members[label][1]
            out.append(ComposedHit(
                corpus_label=label,
                citation=Citation.from_coord(registry[idx]),
                score=score,
            ))
        return out


def compose(**named_kms: ArchiveContents) -> ComposedKnowledgeModel:
    """Build a federated read-only view over multiple knowledge models.

    Usage:

        composed = compose(docs=archive.read("docs.rlat"), code=archive.read("code.rlat"))
        hits = composed.search(query_emb, top_k=10)
        # Each hit carries hit.corpus_label so you know which KM it came from.

    Cross-model rule: all members must carry a base band of identical dim.
    Optimised bands are ignored — federated search uses only the base.

    Backbone-revision mismatches are NOT raised: cosine ordering still holds
    across distinct embedding distributions (both bands are unit vectors),
    just with the magnitude caveat already documented in `cli/compare`.
    """
    return ComposedKnowledgeModel(named_kms)


def merge(
    contents_a: ArchiveContents,
    contents_b: ArchiveContents,
    output_path: str | Path,
    *,
    dedupe_threshold: float = 0.92,
) -> int:
    """Physically merge two knowledge models into a new .rlat. Returns the
    number of unique passages in the merged archive.

    Input contents must (a) carry a base band, (b) share a backbone revision
    (rejected on mismatch — different revisions yield non-comparable
    embeddings, so a merged corpus would be silently corrupt), (c) be in a
    mode whose source files don't live inside the .rlat (i.e. NOT
    `bundled`). For v2.0 we don't carry source/ across merge — that's a
    Phase 6+ enhancement; the merged KM uses local-mode resolution against
    the original `source_file` paths. Raise on bundled inputs rather than
    silently dropping the source/ section.

    Semantic dedupe: builds the union of bands, runs `greedy_cluster` at
    `dedupe_threshold`, and keeps the FIRST member of each cluster. Because
    A's rows are placed before B's in the union, A wins on collisions —
    "merge B into A" semantics, useful for "augment my main KM with new
    material from B."

    Output is written atomically via `archive.write`. Format version + ANN
    construction follow the same path as `rlat build`. Optimised bands
    are NOT carried; merge produces a base-only archive.

    Memory ceiling: the union band feeds `greedy_cluster`, which allocates
    an `(N_a + N_b, N_a + N_b)` float32 cosine matrix. At a union of 50K
    rows that's ~10 GB — the practical merge ceiling on a typical box.
    Above that, callers should pre-shard or reduce one input's size first.

    Future work: per-passage origin in the merged registry (which input
    each passage came from), an ANN attach point on `ComposedKnowledgeModel`
    for high-QPS federated search, and bundled-mode source/ concat.
    """
    if contents_a.metadata.store_mode == "bundled" or contents_b.metadata.store_mode == "bundled":
        raise NotImplementedError(
            "merge of bundled-mode knowledge models is not supported in v2.0; "
            "rebuild inputs in local mode (their source files stay outside "
            "the .rlat) before merging"
        )
    band_a, band_b = _require_compatible_pair(contents_a, contents_b)

    union_band = np.vstack([band_a, band_b]).astype(np.float32)
    union_registry = list(contents_a.registry) + list(contents_b.registry)
    n_a = band_a.shape[0]

    # Greedy-cluster the union; for each cluster, keep the first member
    # (= the A-row when both KMs contribute, since A is placed first).
    clusters = greedy_cluster(union_band, dedupe_threshold)
    keep_indices = sorted(c[0] for c in clusters)

    kept_band = union_band[keep_indices]
    kept_coords = [union_registry[i] for i in keep_indices]
    # Renumber passage_idx in the merged registry — registry idx is line-
    # implicit so we re-issue them sequentially, preserving everything else.
    merged_registry = [
        PassageCoord(
            passage_idx=new_idx,
            source_file=c.source_file,
            char_offset=c.char_offset,
            char_length=c.char_length,
            content_hash=c.content_hash,
            passage_id=c.passage_id,
        )
        for new_idx, c in enumerate(kept_coords)
    ]

    # Build merged metadata. Inherit backbone from A (already verified ==
    # to B's via _require_compatible_pair). New band info reflects merged
    # passage_count. `local` mode because we don't carry source/ across merge.
    base_info_a = contents_a.metadata.bands["base"]
    merged_metadata = Metadata(
        kind=contents_a.metadata.kind,
        backbone=contents_a.metadata.backbone,
        bands={
            "base": BandInfo(
                role=base_info_a.role,
                dim=base_info_a.dim,
                l2_norm=base_info_a.l2_norm,
                passage_count=len(kept_coords),
            ),
        },
        store_mode="local",
        build_config={
            "merged_from": [
                contents_a.metadata.build_config.get("source_root", "<unknown>"),
                contents_b.metadata.build_config.get("source_root", "<unknown>"),
            ],
            "dedupe_threshold": dedupe_threshold,
            "n_a": n_a,
            "n_b": band_b.shape[0],
            "n_merged": len(kept_coords),
        },
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )

    # Defensive: re-L2 the kept band (should already be unit-norm since
    # both inputs were L2 at build time, but np.vstack on slightly-stale
    # rows can drift from unit length).
    contig = np.ascontiguousarray(kept_band, dtype=np.float32)
    l2_normalize(contig)

    archive.write(
        output_path,
        metadata=merged_metadata,
        bands={"base": contig},
        registry=merged_registry,
    )
    return len(kept_coords)
