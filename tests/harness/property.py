"""Property-based invariants (fixed-seed trials in lieu of Hypothesis).

Field algebra (Phase 1):
- merge(a, merge(b, c)) == merge(merge(a, b), c)   (associativity)
- merge(a, empty) == a                              (identity)
- merge(a, b) == merge(b, a)                        (commutativity)
- subtract(a, a) == empty
- intersect(a, b) == intersect(b, a)                (commutativity)
- diff(a, a) == empty
- diff(a, b) >= 0 elementwise

RQL ops (Phase 6):
- compare(a, b).overlap_score == compare(b, a).overlap_score (symmetric)
- len(unique(a, a)) == 0 (a passage near-matches itself)
- len(intersect(a, a, threshold=1.0)) >= n_passages // 2 (self-matches dominate)
- corpus_diff(a, a) → empty added/removed, all rows unchanged
- merge(a, b) returns at most n_a + n_b passages (dedupe never inflates)
- corpus_diff/merge raise on revision mismatch

Registry (Phase 2):
- every chunk written is recoverable by its source coord.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from resonance_lattice.field import algebra
from resonance_lattice.field._runtime_common import l2_normalize
from resonance_lattice.rql import compare, corpus_diff, intersect, merge, unique
from resonance_lattice.store import archive
from resonance_lattice.store.metadata import BackboneInfo, BandInfo, Metadata
from resonance_lattice.store.registry import PassageCoord, compute_id

_DIM = 768
_TRIALS = 32  # fixed-seed trials in lieu of Hypothesis (a [dev]-extra dep).
_SEED = 0
# Float32 has ~7 sig-figs; a 3-way sum of std-1 normals can lose the last few
# mantissa bits to cancellation. Pick atol to admit that without hiding any
# real algebra-contract bugs.
_FP_ATOL = 1e-5


def _write_km(
    tmp: Path, name: str, band: np.ndarray, *, revision: str = "abc",
) -> archive.ArchiveContents:
    n = band.shape[0]
    p = tmp / f"{name}.rlat"
    archive.write(
        p,
        metadata=Metadata(
            backbone=BackboneInfo(revision=revision),
            bands={"base": BandInfo(role="retrieval_default", dim=_DIM, l2_norm=True, passage_count=n)},
            store_mode="local",
        ),
        bands={"base": band},
        registry=[
            PassageCoord(
                i, f"{name}.md", i * 100, 80, f"sha256:{name}{i}",
                passage_id=compute_id(f"{name}.md", i * 100, 80),
            ) for i in range(n)
        ],
    )
    return archive.read(p)


def _check_field_algebra(rng: np.random.Generator, failures: list[str]) -> None:
    for trial in range(_TRIALS):
        a = rng.normal(size=_DIM).astype(np.float32)
        b = rng.normal(size=_DIM).astype(np.float32)
        c = rng.normal(size=_DIM).astype(np.float32)
        e = algebra.empty(_DIM)
        cases = [
            ("merge_associative",
             np.allclose(
                 algebra.merge(algebra.merge(a, b), c),
                 algebra.merge(a, algebra.merge(b, c)),
                 atol=_FP_ATOL)),
            ("merge_identity",
             np.allclose(algebra.merge(a, e), a, atol=_FP_ATOL)),
            ("merge_commutative",
             np.allclose(algebra.merge(a, b), algebra.merge(b, a), atol=_FP_ATOL)),
            ("subtract_self_empty",
             np.allclose(algebra.subtract(a, a), e, atol=_FP_ATOL)),
            ("intersect_commutative",
             np.allclose(algebra.intersect(a, b), algebra.intersect(b, a), atol=_FP_ATOL)),
            ("diff_self_empty",
             np.allclose(algebra.diff(a, a), e, atol=_FP_ATOL)),
            ("diff_nonnegative",
             (algebra.diff(a, b) >= 0).all()),
        ]
        for name, ok in cases:
            if not ok:
                failures.append(f"algebra trial={trial} {name}")


def _check_greedy_cluster_transitive(failures: list[str]) -> None:
    """A↔B above threshold AND B↔C above threshold should land all three in
    one cluster even if A↔C is below threshold (single-linkage semantics).
    Regression guard for the seed-greedy bug fixed in commit A.
    """
    v0 = np.array([1.0, 0.0, 0.0] + [0.0] * (_DIM - 3), dtype=np.float32)
    v1 = np.array([0.97, 0.243, 0.0] + [0.0] * (_DIM - 3), dtype=np.float32)
    v2 = np.array([0.883, 0.468, 0.0] + [0.0] * (_DIM - 3), dtype=np.float32)
    embs = np.stack([v0, v1, v2])
    l2_normalize(embs)
    sims = embs @ embs.T
    if not (sims[0, 1] >= 0.95 and sims[1, 2] >= 0.95 and sims[0, 2] < 0.95):
        # The fixture vectors didn't land in the transitive-chain regime
        # — skip rather than flake. (Shouldn't happen with the values above.)
        return
    clusters = algebra.greedy_cluster(embs, threshold=0.95)
    if not (len(clusters) == 1 and sorted(clusters[0]) == [0, 1, 2]):
        failures.append(f"greedy_cluster transitive: {clusters!r}")


def _check_rql_invariants(rng: np.random.Generator, failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        n = 8
        band_a = rng.normal(size=(n, _DIM)).astype(np.float32)
        l2_normalize(band_a)
        band_b = rng.normal(size=(n, _DIM)).astype(np.float32)
        l2_normalize(band_b)
        band_a_diffrev = band_a.copy()
        a = _write_km(tmp, "a", band_a)
        b = _write_km(tmp, "b", band_b)
        a_diffrev = _write_km(tmp, "a2", band_a_diffrev, revision="xyz")

        # compare overlap_score is symmetric
        cr_ab = compare(a, b)
        cr_ba = compare(b, a)
        if abs(cr_ab.overlap_score - cr_ba.overlap_score) > _FP_ATOL:
            failures.append(
                f"compare_overlap_symmetric: ab={cr_ab.overlap_score} ba={cr_ba.overlap_score}",
            )

        # unique(a, a) is empty (every passage near-matches itself)
        if unique(a, a, threshold=0.99) != []:
            failures.append("unique_self_empty")

        # intersect(a, a, threshold=1.0) returns at least one pair per row
        # (each passage is its own perfect match — but cross-pair off-diagonal
        # may also surface. We just check non-empty + count >= n.)
        pairs = intersect(a, a, threshold=0.999)
        if len(pairs) < n:
            failures.append(f"intersect_self len={len(pairs)} expected >={n}")

        # corpus_diff(a, a) should classify everything as unchanged
        cd_self = corpus_diff(a, a, threshold=0.99)
        if cd_self.added or cd_self.removed or cd_self.unchanged_count != n:
            failures.append(
                f"corpus_diff_self_unchanged: added={len(cd_self.added)} "
                f"removed={len(cd_self.removed)} unchanged={cd_self.unchanged_count}",
            )

        # corpus_diff raises on revision mismatch (regression guard for commit A's strict guard)
        try:
            corpus_diff(a, a_diffrev)
            failures.append("corpus_diff_revision_mismatch_should_raise")
        except ValueError:
            pass

        # merge raises on revision mismatch
        try:
            merge(a, a_diffrev, tmp / "should_fail.rlat")
            failures.append("merge_revision_mismatch_should_raise")
        except ValueError:
            pass

        # merge dedupe never inflates: |output| <= |a| + |b|
        merge_path = tmp / "merged.rlat"
        n_kept = merge(a, b, merge_path, dedupe_threshold=0.92)
        if n_kept > 2 * n:
            failures.append(f"merge_inflated: {n_kept} > {2 * n}")


def _check_memory_v21_invariants(rng: np.random.Generator, failures: list[str]) -> None:
    """P.1-P.6 from .claude/plans/fabric-agent-flat-memory.md §0.9 / §10.

    P.2 (insertion-order independence) is the only one that requires a
    fuzz harness — the other five are pinned as deterministic contracts
    in their respective `memory_v21_*` suites. P.2 is verified here by
    seeding the same row set in `_TRIALS` random orderings, confirming
    `recall()` returns the same cosine-sorted hit list each time.
    """
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.recall import recall
    from resonance_lattice.memory.store import Memory
    from ._testutil import FixedEncoder, patch_zero_encoder

    patch_zero_encoder()
    cwd_tag = workspace_tag_for_cwd("/proj")
    query_vec = np.zeros(_DIM, dtype=np.float32)
    query_vec[0] = 1.0

    for trial in range(_TRIALS):
        # 8 rows with distinct cosines spaced > 0.05 (clears confidence
        # gap); recurrence > 3 (clears recurrence gate); known cwd tag.
        n = 8
        cosines = np.linspace(0.95, 0.71, n).astype(np.float32)
        embs = np.zeros((n, _DIM), dtype=np.float32)
        embs[:, 0] = cosines
        embs[:, 1] = np.sqrt(np.maximum(0.0, 1.0 - cosines ** 2))
        rows_payload = [
            {
                "text": f"row {i}",
                "polarity": ["factual", cwd_tag],
                "transcript_hash": f"distilled:tx{i:04d}",
                "embedding": embs[i],
            }
            for i in range(n)
        ]

        # Two random orderings → identical recall output.
        order_a = list(range(n))
        order_b = list(range(n))
        rng.shuffle(order_a)
        rng.shuffle(order_b)

        cwd_hash = cwd_tag.removeprefix("workspace:")
        with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
            mem_a = Memory(root=Path(td_a) / "u", encoder=FixedEncoder(query_vec))
            mem_b = Memory(root=Path(td_b) / "u", encoder=FixedEncoder(query_vec))
            for idx in order_a:
                row_id = mem_a.add_row(**rows_payload[idx])
                mem_a.update_row(row_id, recurrence_count=5)
            for idx in order_b:
                row_id = mem_b.add_row(**rows_payload[idx])
                mem_b.update_row(row_id, recurrence_count=5)
            hits_a = recall("anything", store=mem_a, cwd_hash=cwd_hash, top_k=n)
            hits_b = recall("anything", store=mem_b, cwd_hash=cwd_hash, top_k=n)

        cosines_a = [round(h.cosine, 6) for h in hits_a]
        cosines_b = [round(h.cosine, 6) for h in hits_b]
        if cosines_a != cosines_b:
            failures.append(
                f"P.2_recall_order_dependent (trial={trial}): "
                f"{cosines_a} != {cosines_b}"
            )
            return  # one failure is enough


def run() -> int:
    rng = np.random.default_rng(_SEED)
    failures: list[str] = []
    _check_field_algebra(rng, failures)
    _check_greedy_cluster_transitive(failures)
    _check_rql_invariants(rng, failures)
    _check_memory_v21_invariants(rng, failures)
    if failures:
        for f in failures[:10]:
            print(f"[property] FAIL {f}")
        return 1
    return 0
