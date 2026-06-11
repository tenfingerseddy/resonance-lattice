"""signals — the closed-form detection clauses (arm (b)) over the capture stream.

Each clause is a pure function from the `field.capture` observation list to a
per-query feature — no model, no learned weights, no network. They are arm (b)
in the H1 §D gate (`horizon-1-capture.md`): the baseline the curator head's
learned parameters must beat. The learnable combiner that weighs them is arm (c)
and lives elsewhere.

It owns three clauses today, all pure feature extractors over the same
user-source rows:

The **score-gap clause** (`the-curator-head.md` §6, first clause): is a query
weakly matched by the corpus? Two closed-form features the combiner reads, both
from the captured fingerprint + scores alone. (Note: as a *gap* signal this was
KILLED in H1 §D, 2026-06-02 — a weak score does not indicate a gap on a dense
corpus, where a sibling doc masks the removed answer; see `curator/gap.py` and
`review-log.md`. The features are still well-defined; they just don't detect gaps.)

- **top score** — the captured nearest-corpus cosine (`ranked[0].score`). A low
  top score means no passage matched well: a candidate gap.
- **Mahalanobis distance** — the query embedding's distance to the *well-served*
  query distribution (the rows whose top score cleared a quantile). Far from what
  the corpus answers well ⇒ out-of-distribution ⇒ a candidate gap. Diagonal
  covariance (per-dimension variance + ε) — robust with few samples and free of a
  768×768 inverse (`the-curator-head.md` §5: Mahalanobis OOD, no training).

The **intent-cluster clause** (`capture.md` §4, "Intent — cluster the query
embeddings"): group paraphrase-similar user queries into a coarse intent by
closed-form complete-linkage clustering of the embeddings. It emits the cluster id
(the same-intent key the reformulation clause reads), the cluster size, and the
session-spread — the reproducibility features the gap combiner reads ("a gap is
real if it reproduces across paraphrases / sessions"). The cluster→intent *label*
(debug / design / …) is arm (c)'s learned mapper, not this clause.

The **reformulation clause** (`capture.md` §4, "Unmet outcome"): did the user
quickly re-ask a similar query in the same session? A fast same-intent
reformulation is the cleanest store-side "unmet" signal (Hassan, CIKM'13). It
emits, per query, whether a same-session followup exists, the cosine to it, and
the seconds until it — the three raw inputs the combiner thresholds, never a
verdict (refinement is often progress, not struggle).

This module emits the **features** only. The gap *decision* combines them with
the reproducibility and lexical-probe clauses (later increments): arm (b)'s
closed-form combiner and arm (c)'s learned combiner read these numbers, so each
clause stays a pure feature extractor with no threshold of its own to bake in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from resonance_lattice.field.algebra import complete_linkage_cluster


@dataclass(frozen=True)
class ScoreGapFeatures:
    """The score-gap clause's per-query closed-form features.

    `top_score` is the captured nearest-corpus cosine (low ⇒ weak match);
    `maha` is the diagonal-Mahalanobis distance to the well-served query
    distribution (high ⇒ out-of-distribution). Both feed arm (c)'s combiner;
    `is_score_gap` is arm (b)'s closed-form call over them."""

    top_score: float
    maha: float


def user_source_indices(observations) -> list[int]:
    """The indices of the rows that are user-intent source-band retrievals carrying
    a fingerprint — the per-query unit every clause operates on, in stream order.

    Public so a caller (e.g. the H1 D-gate harness) can align external per-row data
    to the **same rows the clauses emit**, with no predicate drift: a clause's i-th
    output corresponds to the i-th index this returns.

    A fingerprint is a non-empty sequence; the emptiness test avoids truthiness on
    the value itself (`not ndarray` raises). capture only ever stores a list or
    None, but the predicate holds for any row shape."""
    out = []
    for i, o in enumerate(observations or []):
        if not isinstance(o, dict):
            continue
        if not o.get("is_user_query", True):
            continue
        if o.get("layer") != "source":
            continue
        emb = o.get("query_emb")
        if emb is None:
            continue
        try:
            if len(emb) == 0:
                continue
        except TypeError:  # a scalar — not a fingerprint
            continue
        out.append(i)
    return out


def _user_source_rows(observations) -> list[dict]:
    """The rows `user_source_indices` selects — one row per user query carrying a
    fingerprint. One user search emits a source and an insight observation sharing
    `query_emb`; the source row is the canonical per-query unit, and the insight
    row would double-count. Single predicate, so rows and indices never drift."""
    obs = list(observations or [])
    return [obs[i] for i in user_source_indices(obs)]


def _top_score(row: dict) -> float:
    """The row's best retrieval score — `ranked` is sorted score-descending by
    `capture.observe`, so element 0 is the top hit. 0.0 when nothing ranked
    (a retrieval that surfaced nothing is maximally weak)."""
    ranked = row.get("ranked") or []
    if not ranked:
        return 0.0
    try:
        s = float(ranked[0]["score"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return s if math.isfinite(s) else 0.0


def score_gap_features(
    observations,
    *,
    served_quantile: float = 0.5,
    eps: float = 1e-6,
) -> list[ScoreGapFeatures]:
    """Per-query score-gap features, aligned with the user-source rows (in
    stream order).

    The *well-served* reference is the rows whose top score is at or above the
    `served_quantile` of the stream's top scores — the queries the corpus answers
    well. The Mahalanobis distance of every query to that reference's mean, scaled
    by its per-dimension variance, is the OOD signal. With fewer than two
    well-served rows the reference is degenerate, so `maha` falls back to 0.0
    (no embedding signal) and only `top_score` carries the clause.

    Pure and deterministic: no clock, no model, no learned weights. Returns `[]`
    for an empty stream; never raises on a row with a malformed embedding (it is
    skipped upstream by `_user_source_rows`' `query_emb` guard)."""
    rows = _user_source_rows(observations)
    if not rows:
        return []

    top = np.array([_top_score(r) for r in rows], dtype="float64")
    try:
        embs = np.array([r["query_emb"] for r in rows], dtype="float64")
    except (ValueError, TypeError):
        # Ragged / non-numeric embeddings — fall back to score-only features.
        return [ScoreGapFeatures(top_score=float(t), maha=0.0) for t in top]
    # A non-finite component in even one served row would poison `mu`/`var` and
    # NaN the whole maha column — capture rounds `float('nan')` straight through,
    # so sanitise before the reference stats. A corrupt row degrades to itself.
    embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)

    served = top >= np.quantile(top, served_quantile)
    ref = embs[served]
    if ref.shape[0] >= 2 and embs.ndim == 2:
        mu = ref.mean(axis=0)
        var = ref.var(axis=0) + eps
        maha = np.sqrt((((embs - mu) ** 2) / var).sum(axis=1))
    else:
        maha = np.zeros(len(rows), dtype="float64")

    return [
        ScoreGapFeatures(top_score=float(top[i]), maha=float(maha[i]))
        for i in range(len(rows))
    ]


@dataclass(frozen=True)
class IntentCluster:
    """The intent-cluster clause's per-query closed-form features.

    `cluster_id` is the coarse intent a user query falls into (0-based; clusters
    ordered by first appearance in the stream, so the id is stable). `size` is
    the cluster's membership — the paraphrase-recurrence count. `distinct_sessions`
    is how many distinct sessions the intent spans — cross-session recurrence, the
    stronger reproducibility signal the gap combiner reads (`capture.md` §4,
    "reproducible across paraphrases / sessions"). The intent *label* is arm (c)'s
    job, not this clause."""

    cluster_id: int
    size: int
    distinct_sessions: int


# gte-mb paraphrase floor — paraphrases of one need typically clear cosine 0.7
# (`rql/compare.py:147`). The default; a keyword override, never a user knob.
_PARAPHRASE_COSINE = 0.7


def intent_clusters(
    observations,
    *,
    threshold: float = _PARAPHRASE_COSINE,
) -> list[IntentCluster]:
    """Per-query intent-cluster features, aligned with the user-source rows (in
    stream order).

    Closed-form COMPLETE-linkage clustering of the query embeddings by cosine ≥
    `threshold` (`field.algebra.complete_linkage_cluster`). Complete- not
    single-linkage on purpose: single-linkage chains same-frame queries on
    different topics into one mega-cluster (measured: single @ 0.70 → 1 cluster /
    pairwise-F1 0.13 vs complete @ 0.70 → the true clusters / F1 1.0,
    `bench_intent_clustering`), which would flatten the per-intent demand signal.
    Complete-linkage requires every cross pair to clear the threshold, so a cluster
    is a genuine clique. Paraphrases of one need land in one cluster; the cluster's
    size and session-spread are the reproducibility features the gap combiner reads,
    and the cluster id is the same-intent key the reformulation clause reads.

    Pure and deterministic: no clock, no model, no learned weights — the
    cluster→intent label is arm (c)'s mapper, not this clause. Returns `[]` for an
    empty stream; a ragged / non-numeric embedding set falls back to all-singletons
    (each query its own intent), so it never raises. A zero-norm or non-finite
    embedding is its own singleton too — junk never merges into a real intent.

    Cost / caveats: this is a batch read of the capture buffer (offline, never the
    hot path) — `complete_linkage_cluster` is O(N²) distances + O(N² log N) linkage
    in the buffered query count (capped at `capture._MAX_BUFFERED`, well under its
    ceiling). Complete-linkage is conservative (it splits rather than chains), so a
    loosely-worded paraphrase may fall into its own singleton instead of merging;
    that costs recurrence recall, never precision — a chained mega-cluster (the
    single-linkage failure) would corrupt the signal far worse."""
    rows = _user_source_rows(observations)
    if not rows:
        return []

    try:
        embs = np.array([r["query_emb"] for r in rows], dtype="float64")
    except (ValueError, TypeError):
        embs = None
    if embs is None or embs.ndim != 2:
        # Ragged / non-numeric — no cosine is computable; every query is its own
        # intent (size 1, its own session).
        return [
            IntentCluster(cluster_id=i, size=1, distinct_sessions=1)
            for i in range(len(rows))
        ]

    # A non-finite component would NaN its whole cosine row; sanitise to a zero
    # vector, which clusters with nothing (cosine 0 < threshold) — a junk row
    # becomes a singleton instead of poisoning a real cluster.
    embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    unit = embs / np.where(norms == 0.0, 1.0, norms)  # zero-norm stays zero
    clusters = complete_linkage_cluster(unit.astype("float32"), threshold)

    sessions = [r.get("session") for r in rows]
    cid_of = [0] * len(rows)
    size_of = [0] * len(rows)
    sess_of = [0] * len(rows)
    for cluster_id, members in enumerate(clusters):
        size = len(members)
        distinct = len({sessions[m] for m in members})
        for m in members:
            cid_of[m] = cluster_id
            size_of[m] = size
            sess_of[m] = distinct

    return [
        IntentCluster(
            cluster_id=cid_of[i], size=size_of[i], distinct_sessions=sess_of[i]
        )
        for i in range(len(rows))
    ]


@dataclass(frozen=True)
class ReformulationFeatures:
    """The reformulation clause's per-query 'unmet' (struggle) features.

    A *fast same-intent reformulation* — the user quickly re-asks a similar query
    in the same session — is the cleanest store-side dissatisfaction signal in the
    IR literature (Hassan, CIKM'13; `capture.md` §4). The signal is query-stream
    only (the heart sees no action or result-acceptance). The clause maps the
    three spec conditions to three raw features the combiner thresholds, never a
    decision of its own:

    - `has_followup` — a later query exists in the *same session* at all
      (condition 3: there was a re-ask).
    - `next_cosine` — cosine to that next same-session query (condition 1: high ⇒
      same intent). 0.0 when there is no followup or the embeddings are unreadable.
    - `gap_seconds` — seconds from this query to that one (condition 2: short ⇒ a
      *fast* re-ask). `inf` when there is no followup or the timing is unknown, so
      the "short" test fails safe.

    No followup ⇒ `(False, 0.0, inf)` — every threshold fails, not a
    reformulation. Refinement is often progress, not struggle (`capture.md` §6):
    these are a *probability*'s inputs, never a verdict."""

    has_followup: bool
    next_cosine: float
    gap_seconds: float


def _parse_ts(value) -> datetime | None:
    """Parse a capture row's ISO `ts` to an aware UTC datetime; None if
    unparseable. Mirrors `field.counters._parse_ts` — naïve values are read as
    UTC so a gap subtraction never raises on a tz mismatch."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _unit_embeddings(rows):
    """`(N, D)` L2-normalised query embeddings for the rows, or None if the set
    is ragged / non-numeric (then cosine is unavailable and degrades to 0.0).
    Non-finite components are sanitised to a zero vector (cosine 0 with all)."""
    try:
        embs = np.array([r["query_emb"] for r in rows], dtype="float64")
    except (ValueError, TypeError):
        return None
    if embs.ndim != 2:
        return None
    embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms == 0.0, 1.0, norms)


def reformulation_features(observations) -> list[ReformulationFeatures]:
    """Per-query reformulation features, aligned with the user-source rows (in
    stream order).

    For each query it finds the **next query in the same session** (the re-ask)
    and reports whether one exists, the cosine to it, and the seconds until it.
    The next-same-session index is resolved in one O(N) backward pass; the cosine
    is a dot of the L2-normalised embeddings.

    Pure and deterministic: the time gap is read from the captured timestamps, not
    a wall clock, so the §D replay reproduces. Returns `[]` for an empty stream;
    ragged embeddings degrade `next_cosine` to 0.0 (followup + gap still computed)
    and an unparseable timestamp degrades `gap_seconds` to `inf` — it never
    raises. A negative gap (clock skew / same-second re-ask) clamps to 0.0."""
    rows = _user_source_rows(observations)
    n = len(rows)
    if n == 0:
        return []

    unit = _unit_embeddings(rows)
    sessions = [r.get("session") for r in rows]
    times = [_parse_ts(r.get("ts")) for r in rows]

    # next[i] = smallest k > i with the same session, in one backward pass.
    next_of = [None] * n
    seen_next: dict = {}
    for i in range(n - 1, -1, -1):
        s = sessions[i]
        next_of[i] = seen_next.get(s)
        seen_next[s] = i

    out = []
    for i in range(n):
        j = next_of[i]
        if j is None:
            out.append(ReformulationFeatures(False, 0.0, float("inf")))
            continue
        if unit is not None:
            cos = max(-1.0, min(1.0, float(unit[i] @ unit[j])))
        else:
            cos = 0.0
        if times[i] is not None and times[j] is not None:
            gap = max((times[j] - times[i]).total_seconds(), 0.0)
        else:
            gap = float("inf")
        out.append(ReformulationFeatures(True, cos, gap))
    return out
