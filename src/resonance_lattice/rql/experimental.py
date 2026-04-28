"""Experimental ops — heuristic surfaces flagged as such.

`contradictions` — high-cosine + low-lexical pairs (paraphrased disagreement).
`audit`          — fact-check a claim: supporting + contradicting evidence.

Both are HEURISTIC and will produce noise. Ship under the experimental
banner because the underlying user story (cross-corpus contradiction
detection, claim auditing) is genuinely novel under verified retrieval —
no other RAG library returns "did N independent sources back this claim,
or 1 source repeated N times" with full provenance — but the lexical
heuristic isn't a precise oracle. Iterate post-launch on real corpora.

Phase 6 deliverable.
"""

from __future__ import annotations

import numpy as np

from ..field import dense
from ..store.archive import ArchiveContents
from ..store.base import Store
from ..store.verified import verify_hits
from .types import AuditReport, Citation, CitationHit, ContradictionPair


def _jaccard_3gram(text_a: str, text_b: str) -> float:
    """Jaccard similarity of token-3-gram sets. Cheap stand-in for "do these
    two passages share lexical material?" Whitespace tokenisation +
    lowercase. Returns 1.0 (treat as max-overlap) for inputs too short to
    form a 3-gram — that filter pushes them out of contradiction-pair
    consideration rather than false-positively flagging short stubs.
    """
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) < 3 or len(tokens_b) < 3:
        return 1.0
    grams_a = {tuple(tokens_a[i:i + 3]) for i in range(len(tokens_a) - 2)}
    grams_b = {tuple(tokens_b[i:i + 3]) for i in range(len(tokens_b) - 2)}
    if not grams_a or not grams_b:
        return 1.0
    intersection = grams_a & grams_b
    union = grams_a | grams_b
    return len(intersection) / len(union)


def contradictions(
    contents: ArchiveContents,
    store: Store,
    *,
    cosine_threshold: float = 0.85,
    lexical_threshold: float = 0.3,
    max_pairs: int = 1000,
    prefer: str | None = None,
) -> list[ContradictionPair]:
    """Within-corpus pairs with high semantic similarity AND low lexical overlap.

    EXPERIMENTAL. The heuristic: passages that "say the same thing in
    different words" are paraphrases. If they ALSO appear to disagree
    (which we can't detect from cosine + Jaccard alone), this is a
    candidate contradiction — but the heuristic will surface
    same-meaning paraphrases too. Triage required.

    Output: `max_pairs` ContradictionPairs sorted by descending cosine.

    O(N²) cosine matrix + per-pair text fetch (cached by Store per
    source_file). Above ~50K passages this gets expensive; pre-shard.

    Why ship: no other RAG library surfaces this question at all.
    Even an imperfect heuristic that returns triagable candidates is
    a non-trivial product surface, especially with verified citations.
    """
    handle = contents.select_band(prefer)
    band = handle.band
    n = band.shape[0]
    if n < 2:
        return []
    sims = band @ band.T
    mask = np.triu(sims >= cosine_threshold, k=1)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []

    pairs: list[tuple[float, float, int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        coord_i = contents.registry[i]
        coord_j = contents.registry[j]
        text_i = store.fetch(
            coord_i.source_file, coord_i.char_offset, coord_i.char_length,
        )
        text_j = store.fetch(
            coord_j.source_file, coord_j.char_offset, coord_j.char_length,
        )
        jaccard = _jaccard_3gram(text_i, text_j)
        if jaccard <= lexical_threshold:
            pairs.append((float(sims[i, j]), jaccard, i, j))

    pairs.sort(key=lambda t: -t[0])
    pairs = pairs[:max_pairs]
    return [
        ContradictionPair(
            citation_a=Citation.from_coord(contents.registry[i]),
            citation_b=Citation.from_coord(contents.registry[j]),
            cosine=cos,
            jaccard=jac,
        )
        for cos, jac, i, j in pairs
    ]


def audit(
    contents: ArchiveContents,
    store: Store,
    claim_embedding: np.ndarray,
    *,
    support_threshold: float = 0.7,
    top_k_support: int = 10,
    contradiction_cosine: float = 0.85,
    contradiction_lexical: float = 0.3,
) -> AuditReport:
    """Fact-check a claim against a corpus.

    EXPERIMENTAL. Composed surface: runs `evidence`-style retrieval to
    find supporting passages, then for each supporting passage looks for
    high-cosine + low-lexical neighbours (potential paraphrased
    disagreement). Returns supporting + contradicting evidence in one
    typed report.

    Why this isn't just `evidence` + `contradictions` separately: the
    user-facing question "did 5 independent sources back this claim, or
    1 source repeated 5 times?" requires a single op that surfaces both
    `source_count` (sources in `supporting`) and the contradiction
    candidates linked to those specific supporting passages. Composing
    `evidence` + `contradictions` doesn't tie the contradicting set to
    THIS claim's supporting set.

    Always uses base band (audit is intended to be reproducible across
    knowledge models — a claim audited against KM A and KM B should use
    comparable bands).

    `claim_embedding` is L2-normalised in the same dim as the base band.
    Caller does the encoding (matches `evidence`'s contract).
    """
    handle = contents.select_band(prefer="base")
    band = handle.band
    raw_hits = dense.search(
        claim_embedding,
        band,
        contents.registry,
        handle.projection,
        top_k=top_k_support,
    )
    supporting_raw = [(idx, score) for idx, score in raw_hits if score >= support_threshold]
    if not supporting_raw:
        return AuditReport(
            supporting=[], contradicting=[],
            source_count=0, drift_fraction=0.0,
        )
    supporting_verified = verify_hits(supporting_raw, store, contents.registry)
    supporting_hits = [
        CitationHit(
            citation=Citation.from_coord(contents.registry[v.passage_idx]),
            score=v.score,
            text=v.text,
        )
        for v in supporting_verified
    ]

    # Find contradicting passages: for each supporting passage, scan rows
    # with cosine ≥ contradiction_cosine, filter by low Jaccard.
    contradicting_idxs: set[int] = set()
    contradicting_payloads: list[tuple[int, float, str]] = []
    supporting_idxs = {v.passage_idx for v in supporting_verified}
    for v in supporting_verified:
        if not v.text:
            continue  # missing source — can't compute lexical
        passage_vec = band[v.passage_idx]
        sims = band @ passage_vec
        candidates = np.flatnonzero(sims >= contradiction_cosine)
        for c_idx in candidates:
            c_idx = int(c_idx)
            if c_idx in supporting_idxs or c_idx in contradicting_idxs:
                continue
            other_coord = contents.registry[c_idx]
            other_text = store.fetch(
                other_coord.source_file,
                other_coord.char_offset,
                other_coord.char_length,
            )
            jaccard = _jaccard_3gram(v.text, other_text)
            if jaccard <= contradiction_lexical:
                contradicting_idxs.add(c_idx)
                contradicting_payloads.append((c_idx, float(sims[c_idx]), other_text))

    contradicting_hits = [
        CitationHit(
            citation=Citation.from_coord(contents.registry[idx]),
            score=score,
            text=text,
        )
        for idx, score, text in contradicting_payloads
    ]

    sources = {h.citation.source_file for h in supporting_hits}
    drift_count = sum(
        1 for v in supporting_verified if v.drift_status != "verified"
    )
    drift_fraction = drift_count / len(supporting_verified)

    return AuditReport(
        supporting=supporting_hits,
        contradicting=contradicting_hits,
        source_count=len(sources),
        drift_fraction=drift_fraction,
    )
