"""Corpus-side archive orchestrators over the lifecycle spine.

The state-transition spine lives at `state.claim_lifecycle`; this
module owns the corpus-specific archive I/O that wraps it — the drift
cascade applied to a `.rlat`, and the attribution writeback.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from ..state.claim import FINAL_STATES, Claim
from ..state.claim_lifecycle import (
    accumulate_outcome,
    consolidate_corpus,
    consolidate_experience,
    propagate_drift,
    rederive_outcome,
)
from .insight_attribution import InsightWeight

# The attribution apply is generic over the outcome type: the criterion path
# (S4 d3) feeds `CriterionOutcome`s through `criterion_weighted`. The function
# only calls `reducer(outcomes)` and applies the per-claim weights, so it is
# agnostic to the outcome shape — `_O` ties the outcome list to its reducer.
_O = TypeVar("_O")

__all__ = [
    "apply_attribution_to_archive",
    "apply_weights_to_archive",
    "apply_drift_cascade_to_archive",
]

# Above this archive size, an attribution pass that triggers the full
# insight-layer rewrite emits a stderr warning — the rewrite re-zips the
# entire .rlat, so a large archive paying that cost is worth surfacing.
_LARGE_ARCHIVE_WARN_BYTES = 25 * 1024 * 1024


def apply_drift_cascade_to_archive(km_path, contents=None) -> tuple[int, int]:
    """Run the drift cascade against the current source layer; rewrite the
    insight layer in place if anything flipped to stale.

    `contents` is an optional pre-loaded `ArchiveContents` — pass it when
    you've just written the archive and want to avoid re-reading the
    (potentially large) source band. The function falls back to a fresh
    `archive.read` when contents is None.

    Returns `(n_drifted, n_total_insights)`. A no-drift result rewrites
    nothing. Returns `(0, 0)` when the archive has no insight layer.
    """
    from pathlib import Path
    from . import archive

    p = Path(km_path)
    if contents is None:
        contents = archive.read(p)
    if not contents.insights:
        return 0, 0

    fresh_hashes = {c.passage_id: c.content_hash for c in contents.registry}

    updated, drifted_idx = propagate_drift(contents.insights, fresh_hashes)
    if not drifted_idx:
        return 0, len(contents.insights)

    insight_band = contents.bands[archive.INSIGHT_BAND_NAME]
    archive.write_insight_layer_in_place(p, updated, insight_band)
    return len(drifted_idx), len(contents.insights)


def apply_attribution_to_archive(
    km_path,
    outcomes: list[_O],
    *,
    reducer: Callable[[list[_O]], dict[str, InsightWeight]],
    contents=None,
) -> tuple[int, int]:
    """Fold ONE attribution reducer's verdict into the insight layer's Beta
    trust, then rewrite the layer in place.

    Thin wrapper over `apply_weights_to_archive` for a single `(outcomes,
    reducer)` pair (e.g. `CriterionOutcome` + `criterion_weighted`). Idempotent
    for that pair: a seeded corpus claim's tally is RE-DERIVED from its born seed
    + the reducer's full-ledger weight (see `apply_weights_to_archive`), so
    re-running with the same outcomes is a no-op.

    To fold MULTIPLE reducers in one pass, merge their weights first and call
    `apply_weights_to_archive` ONCE — two SET-from-seed applies in sequence
    would clobber, where one merged apply is correct.

    `contents` is an optional pre-loaded `ArchiveContents`. Returns
    `(n_updated, n_retired)`.
    """
    return apply_weights_to_archive(
        km_path, reducer(outcomes), contents=contents)


def _fold_weight(ins: Claim, w: InsightWeight) -> Claim:
    """Apply one claim's reducer weight to its Beta tally.

    A corpus claim with a recorded born seed RE-DERIVES the absolute tally
    (`seed + weight`) — idempotent when `w` is the full-ledger cumulative
    weight (§B BLOCKER). An experience claim, or a corpus claim minted before
    the seed field existed (sentinel < 0), falls back to additive
    `accumulate_outcome`. The source guard keeps `seed_*` (a `CorpusFacts`
    field) off experience claims.
    """
    if ins.source == "corpus":
        seed_corr = ins.facts.seed_corroboration
        seed_fals = ins.facts.seed_falsification
        if seed_corr >= 0.0 and seed_fals >= 0.0:
            return rederive_outcome(
                ins,
                seed_corroboration=seed_corr,
                seed_falsification=seed_fals,
                corroboration=w.corroboration,
                falsification=w.falsification,
            )
    return accumulate_outcome(
        ins, corroboration=w.corroboration, falsification=w.falsification)


def apply_weights_to_archive(
    km_path,
    weights: dict[str, InsightWeight],
    *,
    contents=None,
) -> tuple[int, int]:
    """Fold a PRE-COMPUTED per-claim weight map into the insight layer's Beta
    trust, then rewrite the layer in place.

    The shared core. Each claim's weight is folded via `_fold_weight` (corpus:
    re-derive from born seed; experience/unseeded: additive), then the spine
    re-evaluates the claim — a corpus claim runs the citation/verdict state
    machine, an experience claim the recurrence + trust earning gate; the
    source routing keeps an experience claim out of `consolidate_corpus`'s
    `CorpusFacts`-only reads. A claim a run of bad outcomes pushed below the
    retire floor transitions to `retired`.

    Callers must supply the CUMULATIVE weight over the full outcome ledger
    (the reducers do this) so the re-derivation is idempotent. Returns
    `(n_updated, n_retired)` — `n_updated` claims actually changed (a
    zero-weight credit is not a change), of which `n_retired` crossed the
    retire floor on this pass. A pass that changes nothing rewrites nothing.
    Returns `(0, 0)` when the archive has no insight layer or no weights.
    """
    import sys
    from pathlib import Path
    from . import archive

    p = Path(km_path)
    if contents is None:
        contents = archive.read(p)
    if not contents.insights:
        return 0, 0
    if not weights:
        return 0, 0

    updated: list[Claim] = []
    n_updated = 0
    n_retired = 0
    for ins in contents.insights:
        w = weights.get(ins.claim_id)
        if w is None or ins.state in FINAL_STATES:
            updated.append(ins)
            continue
        accumulated = _fold_weight(ins, w)
        evolved = (
            consolidate_corpus(accumulated) if ins.source == "corpus"
            else consolidate_experience(accumulated)
        )
        if evolved != ins:
            n_updated += 1
            if evolved.state == "retired" and ins.state != "retired":
                n_retired += 1
        updated.append(evolved)

    if n_updated == 0:
        return 0, 0

    size = p.stat().st_size
    if size > _LARGE_ARCHIVE_WARN_BYTES:
        print(
            f"[insight-attribution] rewriting large archive "
            f"({size / (1024 * 1024):.0f} MB) — the insight-layer write "
            f"re-zips the whole .rlat",
            file=sys.stderr,
        )
    insight_band = contents.bands[archive.INSIGHT_BAND_NAME]
    archive.write_insight_layer_in_place(p, updated, insight_band)
    return n_updated, n_retired
