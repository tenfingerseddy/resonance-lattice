"""decide — the closed-form (arm-b) decide tier over PERSISTED telemetry.

CRITICAL_PATH Step 2: the live call-site where the closed-form `signals` clauses
read the `.rlat`'s own `insight/telemetry.jsonl` (via `store.telemetry.read`, the
member Step 1 folds) and emit candidates — arm (b) on real persisted data, no
head, no network, no learned weights.

**What a candidate is.** Two signals, combined — **demand × relative-undercoverage**:

1. **Demand (reproducibility).** The intent-cluster clause: *which intents recur*. A confirmed-recurring intent is
   a high-demand topic the corpus is repeatedly asked about.
2. **Relative undercoverage (the re-aim of the killed gap signal).** The *absolute* weak-score gap signal was
   KILLED (`curator/gap.py`: a weak top-1 cosine does not indicate a gap on a densely-covered corpus — a sibling
   doc masks the removed answer). Its kill note invited a re-aim to a **relative** signal, and that is what this
   tier uses: rank recurring intents by the corpus coverage they actually receive — the mean captured top-1
   cosine — *relative to the stream*, and keep the ones the corpus answers RELATIVELY WORST. The
   `coverage_quantile` gate keeps that undercovered tail; `coverage_quantile=1.0` disables it (pure-demand
   behaviour — the override). Measured strength (`benchmarks/bench_relative_gap_detector.py`, a controlled 5×5
   excision): cluster-mean coverage separates excised-topic intents from retained-topic intents at **AUC 0.92**
   (gap mean 0.863 vs retained 0.906). Two honest caveats: (i) per-query the intents OVERLAP, so it is the cluster
   MEAN, not any single query, that carries the signal; (ii) end-to-end this depends on the upstream intent
   clustering — the shipped 0.70 paraphrase floor chains same-frame queries into one cluster, which would flatten
   the per-intent coverage, so a less chain-prone threshold is the gating follow-up. The killed *absolute* floor
   finds nothing on the same data (every score sits well above its 0.30 weak-floor). This tier is a
   **prioritiser**, not an answer-lift: it concentrates the separately-proven gap-fill on demand × emptiness; the
   lift itself is a different result.

A candidate is therefore **demand the corpus is comparatively weak on** — still never a truth claim: the fill it
triggers is born low-trust and earns by outcomes. The combination concentrates the proven gap-fill on demand ×
emptiness, and avoids spending fills on recurring topics the corpus already answers well (where a fill, being
gap-rate-limited, would not lift answers).

Closed-form, pure, deterministic, no model. Never raises — it reads a best-effort
telemetry log; an absent/corrupt log or a stream with no recurrence yields no
candidates, never an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..store import telemetry
from .signals import intent_clusters, score_gap_features, user_source_indices

# An intent "recurs" only once it has been asked at least this many times — one
# query is noise until it repeats (`capture.md` §6). Mirrors `gap._MIN_REPRODUCED`.
_MIN_OCCURRENCES = 2
# Distinct sessions the intent must span. 1 = recurs at all; raise to 2 for the
# stricter cross-session "confirmed recurring" gate (`gap.recurring_gaps`).
_MIN_SESSIONS = 1
# RELATIVE coverage gate (the re-aim of the killed absolute gap signal). Keep a
# recurring intent only if its corpus coverage — the mean captured top-1 cosine of
# its queries — is at or below this quantile of the whole stream's per-query
# coverage, i.e. the demand the corpus answers RELATIVELY WORST (gap × demand). 0.5
# keeps roughly the worse-covered tail relative to the stream; `1.0` disables the
# gate (keep every recurring intent — pure demand, the auto+override off-switch). An
# absolute floor is deliberately NOT used: it was killed because redundancy keeps
# top-1 high even where the answer is gone. Strength is modest (see
# bench_relative_gap_detector); the margin is thin, so this prioritises, not proves.
_COVERAGE_QUANTILE = 0.5

# Most-recent telemetry rows decide() reads. The persisted member is a pure
# byte-append (unbounded by design — history is cheap to keep), but the
# clustering below is O(N²) over the rows it's fed; a long-lived heavily-used
# archive would hit a real cliff. 8192 recent rows spans many sessions of
# recurrence signal while keeping the pair math bounded (~67M small-vector
# ops worst case). 2026-06 review.
_DECIDE_WINDOW = 8192


@dataclass(frozen=True)
class RecurringIntent:
    """One confirmed-recurring intent candidate, deduped to a single entry per
    intent cluster (a topic is authored once, not once per occurrence — the
    runaway-compute guard, `gap.py`).

    `occurrences` is the paraphrase-recurrence (cluster size); `distinct_sessions`
    is the cross-session recurrence. `mean_top_score` is the cluster's mean captured
    top-1 cosine — its corpus coverage (low ⇒ the corpus answers this recurring
    intent poorly ⇒ a real gap-fill target; the relative-undercoverage signal).
    `query_centroid` is the L2-normalised mean of the cluster's query fingerprints —
    the representative intent vector Step 3 retrieves corpus passages against to
    author a grounded fill for this intent."""

    cluster_id: int
    occurrences: int
    distinct_sessions: int
    query_centroid: list[float]
    # Coverage (mean captured top-1 cosine). Defaulted + last so existing keyword
    # constructors (benchmarks / probes) keep working; the product path always sets it.
    mean_top_score: float = 0.0


def _centroid(embs: list) -> list[float]:
    """L2-normalised mean of a cluster's query embeddings (rounded to 6 dp, the
    capture fingerprint precision). A ragged / non-numeric set degrades to the
    first readable member, or `[]` — never raises."""
    try:
        arr = np.asarray(embs, dtype="float64")
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError
    except (ValueError, TypeError):
        for e in embs:
            try:
                return [round(float(x), 6) for x in e]
            except (TypeError, ValueError):
                continue
        return []
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    unit = arr / np.where(norms == 0.0, 1.0, norms)
    c = unit.mean(axis=0)
    cn = float(np.linalg.norm(c))
    if cn > 0.0:
        c = c / cn
    return [round(float(x), 6) for x in c]


def recurring_intents(
    observations,
    *,
    min_occurrences: int = _MIN_OCCURRENCES,
    min_sessions: int = _MIN_SESSIONS,
    coverage_quantile: float = _COVERAGE_QUANTILE,
) -> list[RecurringIntent]:
    """Confirmed-recurring intent candidates over a capture stream — one entry per
    intent cluster that recurs AND the corpus answers relatively poorly, returned
    undercovered-first (largest coverage deficit leads).

    Runs the intent-cluster clause (`signals.intent_clusters`, NOT the killed
    absolute score-gap clause) and the aligned score clause (`score_gap_features`,
    for each query's captured top-1 cosine). It keeps a cluster that

      * recurs — `≥ min_occurrences` members spanning `≥ min_sessions` sessions, AND
      * is relatively undercovered — its mean captured top-1 cosine is at or below
        the `coverage_quantile` of the whole stream's per-query top-1 cosines (the
        demand the corpus answers worst). `coverage_quantile = 1.0` disables this
        gate (every recurring intent passes — pure demand).

    Each kept cluster carries its `mean_top_score` (coverage) and the centroid of
    its query fingerprints for the authoring step. Results are ordered by ascending
    coverage (the worst-covered, highest-value gap first), then cluster id.

    Pure, deterministic, never raises. `[]` for an empty stream, one with no
    recurrence, or one where no recurring intent clears the coverage gate."""
    obs = list(observations or [])  # materialise once — iterator-safe, single pass
    idx = user_source_indices(obs)
    if not idx:
        return []
    # `intent_clusters` assumes encoder-homogeneous fingerprints (every in-corpus
    # query_emb shares the gte-mb dim). A ragged, externally-corrupted telemetry set
    # degrades the clause to all-singletons → no candidates: fail-safe (never a
    # wrong candidate), and unreachable from the encoder's own fixed-dim output.
    ic = intent_clusters(obs)
    sg = score_gap_features(obs)  # aligned to the SAME user-source rows (same predicate)
    n = min(len(idx), len(ic), len(sg))  # same predicate ⇒ same length; defensive

    members: dict[int, list] = {}
    info: dict[int, tuple[int, int]] = {}  # cluster_id -> (size, distinct_sessions)
    scores: dict[int, list] = {}           # cluster_id -> captured top-1 cosines
    all_top: list[float] = []
    for i in range(n):
        cid = ic[i].cluster_id
        members.setdefault(cid, []).append(obs[idx[i]].get("query_emb"))
        info[cid] = (ic[i].size, ic[i].distinct_sessions)  # cluster-constant
        scores.setdefault(cid, []).append(sg[i].top_score)
        all_top.append(sg[i].top_score)

    # The RELATIVE coverage threshold (gap × demand): a recurring intent passes only
    # if the corpus answers it no better than the `coverage_quantile` of the stream.
    # `>= 1.0` (or an empty stream) ⇒ +inf ⇒ the gate is off and every recurring
    # intent passes (the auto+override off-switch). An ABSOLUTE floor is deliberately
    # avoided — it was the killed signal (redundancy keeps top-1 high past the answer).
    if coverage_quantile >= 1.0 or not all_top:
        threshold = float("inf")
    else:
        q = max(0.0, min(1.0, coverage_quantile))
        threshold = float(np.quantile(np.asarray(all_top, dtype="float64"), q))

    kept: list[tuple[int, int, int, float]] = []  # (cid, size, sessions, mean_top)
    for cid in members:
        size, sessions = info[cid]
        if size < min_occurrences or sessions < min_sessions:
            continue
        cs = scores.get(cid) or []
        mean_top = float(np.mean(cs)) if cs else 0.0
        # Keep ties at the threshold (the quantile is inclusive); the 1e-9 tolerance
        # stops a float ULP from dropping an at-threshold cluster — critical for the
        # all-equal-coverage case (uniform corpus / degenerate stream), which must
        # keep every recurring intent, not silently drop them all.
        if mean_top > threshold + 1e-9:  # the corpus answers this recurring intent relatively WELL → not a gap
            continue
        kept.append((cid, size, sessions, mean_top))

    # Worst-covered first (largest coverage deficit = highest-value gap), then
    # cluster id for a stable, deterministic order.
    kept.sort(key=lambda t: (t[3], t[0]))
    return [
        RecurringIntent(
            cluster_id=cid,
            occurrences=size,
            distinct_sessions=sessions,
            mean_top_score=round(mean_top, 6),
            query_centroid=_centroid(members[cid]),
        )
        for (cid, size, sessions, mean_top) in kept
    ]


def decide(
    km_id: str | None,
    *,
    min_occurrences: int = _MIN_OCCURRENCES,
    min_sessions: int = _MIN_SESSIONS,
    coverage_quantile: float = _COVERAGE_QUANTILE,
) -> list[RecurringIntent]:
    """The decide call-site (CRITICAL_PATH Step 2): read the `.rlat`'s persisted
    telemetry and emit confirmed-recurring, relatively-undercovered intent
    candidates — arm (b) on real persisted data, no LLM, no network.

    Reads `insight/telemetry.jsonl` from inside the `.rlat` at `km_id` (the member
    Step 1's fold writes), so the candidates derive from the corpus's own
    accumulated use across sessions and processes — not one process's live buffer.
    `coverage_quantile` gates by relative undercoverage (gap × demand); pass `1.0`
    for pure-demand behaviour. Never raises; no corpus path / no telemetry / no
    recurrence / nothing past the coverage gate → `[]`."""
    try:
        return recurring_intents(
            telemetry.read(km_id, tail=_DECIDE_WINDOW),
            min_occurrences=min_occurrences,
            min_sessions=min_sessions,
            coverage_quantile=coverage_quantile,
        )
    except Exception:
        return []
