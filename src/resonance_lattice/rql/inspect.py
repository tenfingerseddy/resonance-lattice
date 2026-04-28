"""Within-corpus inspection ops.

`profile`         — typed corpus snapshot (source distribution, drift counts,
                    bands, encoder identity).
`locate`          — passage_idx → Citation (source coords + content_hash).
`near_duplicates` — within-corpus clusters of pairwise cosine ≥ threshold.

`contradictions` is reserved for the experimental commit (Phase 6 #5);
omitted here so the foundation surface stays auditable.

All three ops operate on `ArchiveContents` directly. Drift verification is
opt-in via the `store=` kwarg on `profile`; `locate` does not verify (it
returns the build-time content_hash; callers compute drift on demand via
`Store.verify(...)`).

Phase 6 deliverable.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..field.algebra import greedy_cluster
from ..store.archive import ArchiveContents
from ..store.base import DriftStatus, Store
from .types import Citation, DuplicateCluster, Profile


def locate(contents: ArchiveContents, passage_idx: int) -> Citation:
    """Resolve `passage_idx` to a `Citation` (source coords + build-time hash).

    O(1). Does NOT call `Store.verify` — drift is the caller's question, not
    `locate`'s. Pair with `store.verify(c.source_file, c.char_offset,
    c.char_length, c.content_hash)` when you need a live drift status.

    Raises `IndexError` if `passage_idx` is out of range — that means the
    caller passed an idx from a different knowledge model's registry, which
    is a programming error.
    """
    return Citation.from_coord(contents.registry[passage_idx])


def profile(
    contents: ArchiveContents,
    *,
    store: Store | None = None,
) -> Profile:
    """Typed snapshot of a knowledge model's structural + verified-retrieval
    statistics. `drift_counts` is empty unless `store` is provided — verifying
    every passage requires reading every source file once, so the verify pass
    is opt-in.

    Field meanings:
    - `n_passages`: total registry rows.
    - `n_source_files`: distinct `source_file` values.
    - `source_distribution`: `source_file -> passage count`.
    - `drift_counts`: `DriftStatus -> count` across the whole corpus when
      `store=` is provided; `{}` otherwise.
    - `bands`: `band_name -> dim`.
    - `encoder_id` / `encoder_revision`: from metadata.backbone.
    - `format_version`: from metadata.format_version (always 4 in v2.0).
    """
    registry = contents.registry
    source_distribution = dict(Counter(c.source_file for c in registry))

    drift_counts: dict[DriftStatus, int] = {}
    if store is not None:
        statuses = (
            store.verify(c.source_file, c.char_offset, c.char_length, c.content_hash)
            for c in registry
        )
        drift_counts = dict(Counter(statuses))

    bands = {name: int(arr.shape[1]) for name, arr in contents.bands.items()}

    backbone = contents.metadata.backbone
    return Profile(
        n_passages=len(registry),
        n_source_files=len(source_distribution),
        source_distribution=source_distribution,
        drift_counts=drift_counts,
        bands=bands,
        encoder_id=backbone.name,
        encoder_revision=backbone.revision,
        format_version=contents.metadata.format_version,
    )


def near_duplicates(
    contents: ArchiveContents,
    *,
    threshold: float = 0.95,
    prefer: str | None = None,
) -> list[DuplicateCluster]:
    """Within-corpus near-duplicate clustering.

    Greedy single-linkage at cosine ≥ `threshold` via
    `field.algebra.greedy_cluster`. Singletons are dropped — every returned
    cluster has ≥ 2 members. `mean_cosine` is the arithmetic mean of all
    distinct pair cosines within the cluster (NOT just the seed row).

    Band selection: in-corpus op, so by default `prefer=None` picks optimised
    when present (the in-corpus-trained band gives tighter dedup) else base.
    Pass `prefer="base"` to force base — only relevant when comparing dedup
    behaviour across optimised vs. unoptimised builds.

    `threshold=0.95` is the conservative default: real near-paraphrases
    typically clear 0.95 with the gte-mb encoder. Below 0.92 the algorithm
    starts merging legitimately-distinct topics.

    O(N²) memory: `greedy_cluster` allocates an (N, N) cosine matrix. Fine
    at v2.0-typical knowledge-model scales (≤ ~50K passages); callers at
    larger scale should pre-shard by source_file and run per-shard.
    """
    handle = contents.select_band(prefer)
    band = handle.band
    if band.shape[0] == 0:
        return []

    raw_clusters = greedy_cluster(band, threshold)
    clusters: list[DuplicateCluster] = []
    for members in raw_clusters:
        if len(members) < 2:
            continue
        # Mean over distinct pairs (k*(k-1)/2 of them) — re-derive sims for
        # this small slice rather than holding the full (N, N) matrix from
        # `greedy_cluster`'s scope.
        sub_band = band[members]
        sub_sims = sub_band @ sub_band.T
        k = len(members)
        triu_sum = float(np.triu(sub_sims, k=1).sum())
        mean_cosine = triu_sum / (k * (k - 1) / 2)
        citations = [
            Citation.from_coord(contents.registry[m]) for m in members
        ]
        clusters.append(DuplicateCluster(citations=citations, mean_cosine=mean_cosine))
    return clusters
