"""gap — the arm (b) closed-form gap-candidate decision (no learned weights).

KILLED as a shipping gap signal (H1 §D, 2026-06-02 — see
`.claude/plans/insight-engine/review-log.md`). The whole premise this module rests
on — "a weak retrieval score ⇒ a candidate gap" — is **empirically false on a
densely-covered corpus**: removing the answer doc barely moves the top-1 cosine
(gap 0.774 vs non-gap 0.813; 0/54 below the 0.30 floor) because a topically-
adjacent sibling still matches high, so neither a fixed-threshold nor a learned
detector finds the gaps from retrieval fingerprints (learned F1 0.591 vs the
always-say-gap baseline 0.584). The gap is real but lives in answer quality, which
fingerprint capture excludes. The code stays (it correctly computes its features,
and the dedup/recurrence logic is reusable if gap-detect is re-aimed at the
answer-quality channel), but **do not treat retrieval-score gap candidates as a
shipping signal**. The rest below describes the design as it was built.

The `signals` clauses emit *features*; this module makes the *decision*. It is
**arm (b)** in the H1 §D gate (`horizon-1-capture.md`): the fixed-threshold
conjunction the curator head's learned combiner (arm (c)) must beat. If the
learned combiner adds nothing over this conjunction, gap-detect collapses into
arm (b) and the head is cut from this task (§D KILL).

The gap recipe (`the-curator-head.md` §6) is a conjunction of three clauses. On
the **fingerprint heart** only two are computable (§6 "Build decision
2026-06-02"): capture stores fingerprints, not words, so the lexical/BM25
trap-guard has nothing to tokenise, and this branch has no runnable lexical path.
So the combiner runs:

    gap candidate(query) := weak score-gap            (top score < 0.30 — the
                                                        "retrieval genuinely
                                                        failed" floor on gte-mb)
                        AND reproducible               (≥ 2 *weak* queries share
                                                        its intent cluster)
                        AND lexical also failed        (optional, injected; absent
                                                        on the fingerprint heart →
                                                        vacuously satisfied)

Two design points the recipe insists on:

- **Reproducibility is over the WEAK members of the cluster, not all members.** A
  cluster of five queries with one weak member is *not* a reproduced gap — it is a
  single weak query against an intent the corpus mostly answers. Counting weak
  members guards the "telemetry calcifies around near-misses" risk
  (`capture.md` §6): a one-off low score is noise until it recurs.
- **The lexical clause vetoes, never creates.** Supplied as a per-query
  `lexical_miss` signal (a future retrieval-time hit/miss boolean): `None` ⇒
  unavailable ⇒ the clause passes (two-clause mode); truthy ⇒ lexical *also*
  failed ⇒ a genuine absence, passes; falsy ⇒ lexical *found* the content ⇒ the
  gap was an encoder reach failure, not a true gap (the Argus-Eyes trap), so the
  candidate is vetoed. Truthiness, not identity, so a numpy bool behaves.

A candidate is **never truth** — gap-detection is ~70% reliable at best
(`_FOUNDATION.md` §7); this decision feeds the candidate queue, the cloud authors
a fill, the fill is born low-trust and earns by outcomes. Pure, deterministic, no
learned weights, never raises. Emits the decision *plus* its supporting features
so a reviewer (and arm (c)) can see why.

This module also owns the **sleep-time recurring-gap queue** (`recurring_gaps`,
the C4 stage, `capture.md` §5): it dedups the per-query candidates to **one entry
per recurring gap intent**, gated by recurrence. That dedup *is* the
runaway-compute guard — N occurrences of one gap cost **one** cloud fill, not N
([`enrichment.md`](enrichment.md)'s only cloud touch). It is buffer-triggered and
offline; nothing here calls the cloud.
"""

from __future__ import annotations

from dataclasses import dataclass

from .signals import intent_clusters, score_gap_features

# The "retrieval genuinely failed" floor: below this absolute top-1 cosine, no
# passage topically matched on gte-modernbert-base 768d (cli/_grounding.py:98,
# the AUGMENT gate's empirical floor). A fixed closed-form constant, not a learned
# weight and not a user knob — the §D harness may pin it.
_WEAK_TOP_SCORE = 0.30
# A gap is "reproduced" once at least this many *weak* queries share an intent
# cluster — one weak query is noise until it recurs (`capture.md` §6).
_MIN_REPRODUCED = 2


@dataclass(frozen=True)
class GapDecision:
    """The arm (b) per-query gap-candidate decision, with its supporting features.

    `is_candidate` is the conjunction's verdict. The rest are why: `weak` (the
    top score cleared the failure floor downward), `reproduced` (how many weak
    queries share this query's intent cluster, including itself), `cluster_id` /
    `distinct_sessions` (the intent and its session-spread), `top_score` / `maha`
    (the score-gap features), and `lexical_miss` (the injected trap-guard signal,
    `None` when unavailable)."""

    is_candidate: bool
    weak: bool
    reproduced: int
    cluster_id: int
    distinct_sessions: int
    top_score: float
    maha: float
    lexical_miss: bool | None


def gap_candidates(
    observations,
    *,
    weak_top_score: float = _WEAK_TOP_SCORE,
    min_reproduced: int = _MIN_REPRODUCED,
    lexical_miss: list | None = None,
) -> list[GapDecision]:
    """The closed-form gap-candidate decision per query, aligned with the
    user-source rows (in stream order).

    Runs the score-gap and intent-cluster clauses over the same stream (both
    select the identical user-source rows, so they align one-to-one), marks the
    weak queries (`top_score < weak_top_score`), counts the weak queries per
    intent cluster, and returns a candidate where a weak query's cluster holds
    `≥ min_reproduced` weak queries and the optional lexical clause does not veto.

    `lexical_miss`, when given, is a per-query list aligned to the user-source
    rows: a truthy entry (lexical also failed) passes, a falsy entry (lexical
    found it) vetoes, `None` is unknown and passes. Omit it (the fingerprint-heart
    default) for two-clause mode; a non-list value is ignored (two-clause).

    Pure and deterministic: no clock, no model, no learned weights. Returns `[]`
    for an empty stream and never raises — the clauses it calls never raise, and a
    length-mismatched `lexical_miss` degrades to `None` past its end."""
    sg = score_gap_features(observations)
    ic = intent_clusters(observations)
    # Both clauses filter to the identical user-source rows, so they are the same
    # length and order; min() is a defensive guard that can't actually trip.
    n = min(len(sg), len(ic))
    if n == 0:
        return []

    weak = [sg[i].top_score < weak_top_score for i in range(n)]
    weak_in_cluster: dict[int, int] = {}
    for i in range(n):
        if weak[i]:
            cid = ic[i].cluster_id
            weak_in_cluster[cid] = weak_in_cluster.get(cid, 0) + 1

    # Only a real list is honoured; any other shape (dict, generator, scalar)
    # degrades to two-clause mode rather than raising — the never-raises contract.
    lex = lexical_miss if isinstance(lexical_miss, list) else []
    out = []
    for i in range(n):
        cid = ic[i].cluster_id
        reproduced = weak_in_cluster.get(cid, 0)
        lm = lex[i] if i < len(lex) else None
        # Tri-state veto, truthiness-based so a numpy bool behaves like a Python
        # one: `None` (unknown) or a truthy "lexical also failed" passes; a falsy
        # "lexical found it" vetoes (the Argus-Eyes trap-guard).
        lexical_ok = lm is None or bool(lm)
        is_candidate = (
            weak[i]
            and reproduced >= min_reproduced
            and lexical_ok
        )
        out.append(
            GapDecision(
                is_candidate=is_candidate,
                weak=weak[i],
                reproduced=reproduced,
                cluster_id=cid,
                distinct_sessions=ic[i].distinct_sessions,
                top_score=sg[i].top_score,
                maha=sg[i].maha,
                lexical_miss=lm,
            )
        )
    return out


@dataclass(frozen=True)
class GapQueueEntry:
    """One confirmed-recurring gap intent, queued for the sleep-time fill.

    Deduped to **one entry per intent cluster** — a gap is filled once, not once
    per occurrence (the runaway-compute guard). `occurrences` is how many
    candidate queries share the intent; `distinct_sessions` is how many sessions
    it recurred across (the confirmed-recurring strength a stricter gate reads)."""

    cluster_id: int
    occurrences: int
    distinct_sessions: int


def recurring_gaps(
    observations,
    *,
    min_occurrences: int = _MIN_REPRODUCED,
    min_sessions: int = 1,
    weak_top_score: float = _WEAK_TOP_SCORE,
    min_reproduced: int = _MIN_REPRODUCED,
    lexical_miss: list | None = None,
) -> list[GapQueueEntry]:
    """The sleep-time queue of confirmed-recurring gaps — one entry per intent
    cluster, gated by recurrence. The C4 stage (`capture.md` §5).

    Runs the combiner, keeps the candidate decisions, and **dedups them by intent
    cluster**: each recurring gap becomes a single queue entry, so N occurrences
    cost one cloud fill, not N (the runaway-compute guard). A cluster is queued
    only if it has `≥ min_occurrences` candidate queries **and** spans
    `≥ min_sessions` sessions — raise `min_sessions` to 2 for the stricter
    cross-session "confirmed recurring" gate. Entries are returned in cluster-id
    order.

    Pure, deterministic, offline, never calls the cloud and never raises — the
    combiner it reads never raises."""
    decs = gap_candidates(
        observations,
        weak_top_score=weak_top_score,
        min_reproduced=min_reproduced,
        lexical_miss=lexical_miss,
    )
    occ: dict[int, int] = {}
    sess: dict[int, int] = {}
    for d in decs:
        if not d.is_candidate:
            continue
        occ[d.cluster_id] = occ.get(d.cluster_id, 0) + 1
        sess[d.cluster_id] = d.distinct_sessions  # cluster-constant
    return [
        GapQueueEntry(
            cluster_id=cid, occurrences=occ[cid], distinct_sessions=sess[cid]
        )
        for cid in sorted(occ)
        if occ[cid] >= min_occurrences and sess[cid] >= min_sessions
    ]
