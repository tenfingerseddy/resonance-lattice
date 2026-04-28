"""Episodic → semantic promotion.

The promotion rule: an episodic entry is promoted to semantic when its
content is "stable" — measured here as having ≥ `recurrence_threshold`
near-duplicates within episodic. Near-duplicate = cosine ≥ `dup_threshold`
between two embeddings (default 0.92, conservative — prose paraphrases
rarely cross this).

When promoted, the cluster of near-duplicates collapses to a single semantic
entry whose `recurrence_count` carries the cluster size. The episodic
near-duplicates are then dropped (they've graduated).

Idempotent on stable input: running consolidate twice produces the same
state — the second run sees no clusters meeting the threshold, returns 0.

Phase 5 deliverable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..field.algebra import greedy_cluster

DEFAULT_RECURRENCE_THRESHOLD = 3
DEFAULT_DUP_THRESHOLD = 0.92


def consolidate(
    memory,
    recurrence_threshold: int = DEFAULT_RECURRENCE_THRESHOLD,
    dup_threshold: float = DEFAULT_DUP_THRESHOLD,
    session: str | None = None,
) -> int:
    """Scan episodic for clusters of ≥ `recurrence_threshold` near-duplicates,
    promote each cluster to semantic, drop the episodic copies. Optionally
    restrict to entries from `session`.

    Returns the number of episodic entries promoted (sum of cluster sizes).
    """
    episodic_entries = memory.all_entries("episodic")
    if not episodic_entries:
        return 0
    if session is not None:
        # Restrict the scan + drop set to entries matching session, keeping
        # cross-session episodic untouched.
        keep_idx = [
            i for i, e in enumerate(episodic_entries) if e.session == session
        ]
    else:
        keep_idx = list(range(len(episodic_entries)))
    if not keep_idx:
        return 0
    sub_entries = [episodic_entries[i] for i in keep_idx]
    sub_embs = np.vstack([e.embedding[None, :] for e in sub_entries])

    clusters = greedy_cluster(sub_embs, dup_threshold)
    promoted_count = 0
    promoted_global_idx: set[int] = set()
    for cluster in clusters:
        if len(cluster) < recurrence_threshold:
            continue
        # Use the cluster's representative (first member, deterministic) as
        # the promoted entry. recurrence_count carries the cluster size so a
        # downstream caller can prefer "well-represented" semantic entries.
        rep = sub_entries[cluster[0]]
        rep_emb = sub_embs[cluster[0]]
        promoted = replace(
            rep, tier="semantic", recurrence_count=len(cluster),
            embedding=rep_emb,
        )
        # Public API rather than touching `memory._tiers` directly — keeps
        # consolidation's contract with LayeredMemory minimal + testable.
        memory.append_to_tier("semantic", promoted, rep_emb)
        for local_i in cluster:
            promoted_global_idx.add(keep_idx[local_i])
        promoted_count += len(cluster)

    if not promoted_global_idx:
        return 0

    # Drop promoted episodic entries via replace_tier; survivors stay in
    # original order so passage_idx-like positional behaviour is preserved.
    survivors = [
        e for i, e in enumerate(episodic_entries)
        if i not in promoted_global_idx
    ]
    if survivors:
        survivor_embs = np.vstack([e.embedding[None, :] for e in survivors])
    else:
        # Empty survivor list — recover the tier dim from any pre-clustering
        # episodic embedding (sub_embs is non-empty here because keep_idx
        # had to be non-empty for clusters to exist). This module talks to
        # LayeredMemory only via its public surface — no internal-state
        # reach-in.
        survivor_embs = np.zeros((0, sub_embs.shape[1]), dtype=np.float32)
    memory.replace_tier("episodic", survivors, survivor_embs)
    return promoted_count
