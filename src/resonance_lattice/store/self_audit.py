"""self_audit — the corpus examining its OWN SHAPE (the rlat moat): contradictions, gaps, drift.

Retrieval reads the corpus to answer a question. The self-audit reads the *whole cloud* to find what's wrong with
it — properties invisible to any single top-k query because they are facts about the SET, not a point:

  - CONTRADICTION CANDIDATES (this module's core): HIGH-COSINE CROSS-DOCUMENT passage pairs. Geometry does the
    cheap part — surface pairs that are about the SAME thing — so an LLM judges STANCE on a bounded shortlist
    instead of every pair. HONEST: geometry only guarantees same-topic; MOST high-cosine pairs are redundant
    RESTATEMENTS, not contradictions, and on a redundant corpus there can be many. So the stored signal is
    "same-topic pairs" (a candidate set the stance judge filters), NOT "contradictions" — and the pass is hard-
    capped (`per_row_cap` + a top-`max_pairs` heap) so it stays bounded regardless of redundancy.
  - GAPS — empty/under-served regions (the demand×coverage path in `curator.decide`).

The two STRONG foundational signals are CONTRADICTIONS + GAPS. DRIFT of corpus passages (a content-hash mismatch)
is included but DEMOTED: it is transient — `rlat refresh` re-syncs the corpus to its sources and clears it, so a
stored "this passage drifted" mostly just means "run refresh" (which then makes it 0). It is kept only as a
between-refreshes hint. The genuinely useful staleness — a stored FACT going stale vs the live WORLD (especially
an external fetched fact whose URL has moved on) — is NOT corpus drift and is NOT fixed by refresh; detecting it
needs a re-fetch (the act layer, network), not this cheap-math pass.

This module owns the CONTRADICTION-CANDIDATE geometry primitive: pure, local, no LLM. The stance judgement (does
this pair actually contradict, and which side is authoritative) is done by the caller — an agent or a metered
client — on just the handful of candidates this returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import archive

# Above this passage count the O(n^2) contradiction matmul is skipped (the ANN index is the scale fix) — the
# default size-guard and the `pairs_skipped` flag share it.
_MAX_AUDIT_PASSAGES = 60000


@dataclass(frozen=True)
class ContradictionCandidate:
    """One high-cosine CROSS-DOCUMENT passage pair — a candidate for stance judgement.

    `cosine` is their similarity (high ⇒ same topic). `a`/`b` are `{passage_idx, source_file, text}`. A caller
    judges whether they genuinely CONTRADICT (vs merely paraphrase) and which is authoritative; geometry only
    guarantees they are close enough to be ABOUT the same thing."""

    cosine: float
    a: dict
    b: dict


def _normalise(band: np.ndarray) -> np.ndarray:
    band = np.asarray(band, dtype="float32")
    return band / (np.linalg.norm(band, axis=1, keepdims=True) + 1e-9)


def find_contradiction_candidates(
    km_path,
    *,
    min_cosine: float = 0.92,
    max_cosine: float = 0.99,
    max_pairs: int = 300,
    source_root=None,
    block: int = 1024,
    resolve_text: bool = True,
    max_passages: int | None = _MAX_AUDIT_PASSAGES,
    per_row_cap: int = 64,
    contents=None,
) -> list[ContradictionCandidate]:
    """High-cosine CROSS-DOCUMENT passage pairs — the geometry-narrowed contradiction candidates.

    For every passage, finds passages in OTHER source files whose cosine ≥ `min_cosine` (same-file pairs are
    skipped — a doc restating itself is not a contradiction to surface). Returns the top `max_pairs` by cosine,
    each with both passages' resolved text so a stance judge can rule on them. Pure + local — no LLM, no network.

    Cross-document is the point: a contradiction worth flagging is two SOURCES disagreeing, not one document's
    HONEST framing: geometry only guarantees the pair is about the SAME THING (high cosine). MOST high-cosine
    cross-doc pairs are redundant RESTATEMENTS, not contradictions — on a redundant corpus they can be many. So
    this is a CANDIDATE set the caller's stance judge filters; the stored signal is "same-topic pairs", not
    "contradictions". Two hard caps keep it cheap and bounded REGARDLESS of corpus redundancy: `per_row_cap` (only
    each passage's top-K above-threshold neighbours are considered — bounds the inner work to n·K, not n²) and a
    global top-`max_pairs` heap (bounds memory + output to `max_pairs`, never the millions a redundant corpus
    would otherwise yield). The BLAS matmul is the only O(n²) cost; above `max_passages` it's skipped (the ANN
    index is the scale fix) — callers detect the skip via `compute_self_audit`'s `pairs_skipped` flag.

    Pass pre-loaded `contents` to avoid re-reading the band (the build already has it in memory). Never raises on a
    missing source: a pair whose text won't resolve still returns (empty text)."""
    import heapq

    contents = contents if contents is not None else archive.read(Path(km_path))
    band = _normalise(contents.bands["base"])
    reg = contents.registry
    n = len(band)
    if n < 2:
        return []
    if max_passages is not None and n > max_passages:
        return []  # the BLAS matmul is O(n^2); skip above the guard (compute_self_audit marks pairs_skipped)
    sf = np.array([c.source_file for c in reg])

    # Bounded min-heap of (cosine, i, j), size <= max_pairs — memory is capped no matter how redundant the corpus.
    heap: list[tuple[float, int, int]] = []
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        sims = band[i0:i1] @ band.T  # (rows, n) — BLAS, fast
        for bi in range(i1 - i0):
            i = i0 + bi
            row = sims[bi]
            above = np.where(row >= min_cosine)[0]
            if above.size:
                # Cross-document + j>i (dedup) + BELOW max_cosine, vectorised — applied BEFORE the per-row cap so
                # the cap keeps the top genuine CROSS-DOC candidates. The max_cosine upper bound EXCLUDES near-exact
                # duplicates (cosine ~1.0 = the SAME text repeated as boilerplate/includes across docs): identical
                # passages can't CONTRADICT, they're redundancy (a different signal). A contradiction lives in the
                # band [min_cosine, max_cosine): same topic, but NOT identical.
                above = above[(above > i) & (sf[above] != sf[i]) & (row[above] < max_cosine)]
            if above.size == 0:
                continue
            # Per-row cap: keep only this passage's TOP-K above-threshold cross-doc neighbours, so a redundant row
            # (millions of near-dups) costs K, not its full width — bounds the inner loop to n·K total.
            if above.size > per_row_cap:
                above = above[np.argpartition(row[above], -per_row_cap)[-per_row_cap:]]
            for j in above.tolist():
                item = (float(row[j]), i, int(j))
                if len(heap) < max_pairs:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
    if not heap:
        return []
    found = sorted(heap, reverse=True)

    # Text resolution is optional — skip it for compact, fast storage (the stored audit keeps only
    # indices + source_files; the act-layer re-resolves text when it judges a pair).
    text = (_resolve_texts(km_path, contents, {i for _, i, _ in found} | {j for _, _, j in found}, source_root)
            if resolve_text else {})
    out = []
    for cos, i, j in found:
        a = {"passage_idx": i, "source_file": reg[i].source_file}
        b = {"passage_idx": j, "source_file": reg[j].source_file}
        if resolve_text:
            a["text"] = text.get(i, "")
            b["text"] = text.get(j, "")
        out.append(ContradictionCandidate(cosine=round(cos, 4), a=a, b=b))
    return out


def rank_contradictions_by_demand(candidates, contents, observations, *, near: float = 0.7):
    """Order contradiction candidates by QUERY DEMAND — conflicts on topics people actually ask about come first.

    The gap×demand insight applied to conflicts: a contradiction nobody queries is academic; one in the path of
    real query traffic is the coin-flip users keep hitting. Demand for a pair = how much captured user-query
    traffic lands near it: for each stored `query_emb`, its cosine to the NEARER of the pair's two passages,
    counted only when ≥ `near` (so distant queries don't dilute) and summed across queries. Pairs no query comes
    near keep demand 0 and fall to the back; the original cosine order is the tiebreak.

    Pure geometry over the stored telemetry — no LLM, one matmul. The score is used only to ORDER candidates
    against each other (a RELATIVE comparison — the lesson that absolute embedding distance is uninformative on the
    compact cone, but relative rank is not). Returns a NEW ordered list; on no usable query vectors or a dim
    mismatch (e.g. an MRL-optimised band vs base-dim query fingerprints), returns the candidates unchanged."""
    qs = [o["query_emb"] for o in (observations or [])
          if o.get("is_user_query", True) and o.get("query_emb")]
    if not qs or not candidates:
        return list(candidates)
    band = _normalise(contents.bands["base"])
    Q = np.asarray(qs, dtype="float32")
    if Q.ndim != 2 or Q.shape[1] != band.shape[1]:
        return list(candidates)  # fingerprints from a different band dim — can't compare; leave order untouched
    Q = _normalise(Q)

    n = band.shape[0]

    def _demand(c) -> float:
        ia, ib = c.a.get("passage_idx"), c.b.get("passage_idx")
        if ia is None or ib is None or not (0 <= ia < n and 0 <= ib < n):
            return 0.0
        m = np.maximum(Q @ band[ia], Q @ band[ib])
        return float(m[m >= near].sum())

    return sorted(candidates, key=lambda c: (-_demand(c), -c.cosine))


@dataclass(frozen=True)
class DriftedPassage:
    """One corpus passage whose source no longer matches its build-time content hash.

    `drift_status` is `"drifted"` (source changed) or `"missing"` (source gone). This is pure book-keeping — a
    re-hash of the slice vs `PassageCoord.content_hash` — no LLM. It's the DRIFT input to the self-audit: stale
    content the loop should re-fetch/update."""

    passage_idx: int
    source_file: str
    drift_status: str


def find_drifted_passages(km_path, *, source_root=None, contents=None) -> list[DriftedPassage]:
    """Corpus passages whose source drifted or went missing — pure math (a content-hash re-check per passage).

    Scans the registry, re-hashes each passage's slice via `store.verify`, and returns those that no longer match
    their recorded `content_hash`. No LLM, no network beyond the store's own source access. Pass pre-loaded
    `contents` to avoid re-reading the archive. Never raises: a passage that can't be checked is treated as
    not-drifted (the audit under-reports rather than crashing)."""
    contents = contents if contents is not None else archive.read(Path(km_path))
    try:
        from . import open_store
        store = open_store(Path(km_path), contents, source_root)
    except Exception:
        return []
    out: list[DriftedPassage] = []
    for idx, coord in enumerate(contents.registry):
        try:
            status = store.verify(coord.source_file, coord.char_offset, coord.char_length, coord.content_hash)
        except Exception:
            continue
        if status in ("drifted", "missing"):
            out.append(DriftedPassage(passage_idx=idx, source_file=coord.source_file, drift_status=str(status)))
    return out


_SELF_AUDIT_VERSION = 1


def compute_self_audit(
    km_path=None,
    *,
    contents=None,
    min_cosine: float = 0.92,
    max_pairs: int = 300,
    source_root=None,
    gaps: list | None = None,
    check_drift: bool = True,
) -> dict:
    """The corpus's shape-report — the foundational, LLM-FREE self-audit, ready to store in the `.rlat`.

    Assembles the cheap-math signals: HIGH-COSINE CROSS-DOC PAIRS (the contradiction CANDIDATES — geometry only
    guarantees same-topic, so most are restatements the act-layer's stance judge filters) + DRIFT (content-hash
    re-check). `gaps` (the demand×coverage signal — `curator.decide`) is telemetry-dependent and passed in (none
    at build, since a fresh corpus has no demand). Compact by design — pairs carry indices + source_files, not
    text. `pairs_skipped` is true when the corpus exceeded the contradiction size-guard (so a 0 count means
    "skipped", not "none found").

    Pass pre-loaded `contents` (band + registry) to compute IN-MEMORY before the archive exists — the build does
    this so the audit folds into the single write (no second ZIP rewrite). DRIFT needs the on-disk source store, so
    it is computed only when a `km_path` is given; at build (`km_path=None`) drift is empty anyway (a fresh corpus
    matches its just-built hashes).

    Pure + deterministic + no LLM. Never raises — a signal that can't be computed degrades to empty."""
    if contents is None and km_path is not None:
        try:
            contents = archive.read(Path(km_path))
        except Exception:
            contents = None
    n = len(contents.registry) if contents is not None else 0
    pairs_skipped = bool(contents is not None and n > _MAX_AUDIT_PASSAGES)
    try:
        pairs = find_contradiction_candidates(
            km_path, min_cosine=min_cosine, max_pairs=max_pairs, source_root=source_root,
            resolve_text=False, contents=contents)
    except Exception:
        pairs = []
    if km_path is not None and check_drift:
        try:
            drift = find_drifted_passages(km_path, source_root=source_root, contents=contents)
        except Exception:
            drift = []
    else:
        # In-memory build pass (no source store yet; a fresh corpus has no
        # drift) — or `check_drift=False`: remote-mode sync skips the
        # re-check because on a RemoteStore it HTTP-fetches every manifest
        # entry, and right after a sync the manifest is freshly re-pinned,
        # so drift is definitionally empty (pure network cost, no signal).
        drift = []
    gap_rows = list(gaps or [])
    return {
        "version": _SELF_AUDIT_VERSION,
        "high_cosine_pairs": [{"cosine": c.cosine, "a": c.a, "b": c.b} for c in pairs],
        "pairs_skipped": pairs_skipped,
        "drift": [{"passage_idx": d.passage_idx, "source_file": d.source_file, "drift_status": d.drift_status}
                  for d in drift],
        "gaps": gap_rows,
        "counts": {"high_cosine_pairs": len(pairs), "drift": len(drift), "gaps": len(gap_rows)},
    }


def attach_self_audit(km_path, *, gaps: list | None = None, source_root=None,
                      check_drift: bool = True) -> dict:
    """Compute the self-audit and STORE it in the `.rlat` — the foundational build/refresh hook.

    Best-effort: never raises. A failed audit (unreadable archive, etc.) must NEVER break a build or refresh, so
    any exception degrades to a no-op and returns `{}`. Returns the stored report on success.
    `check_drift=False` skips the source re-check — remote-mode sync passes it (see compute_self_audit)."""
    try:
        report = compute_self_audit(km_path, source_root=source_root, gaps=gaps,
                                    check_drift=check_drift)
        archive.write_self_audit_in_place(km_path, report)
        return report
    except Exception:
        return {}


def _resolve_texts(km_path, contents, idxs, source_root) -> dict:
    """Best-effort passage-text resolution for the candidate set; a source that won't open yields empty text
    (the audit still reports the pair). Opens the store once."""
    out: dict = {}
    try:
        from . import open_store
        store = open_store(Path(km_path), contents, source_root)
    except Exception:
        return out
    reg = contents.registry
    for i in idxs:
        try:
            out[i] = store.fetch(reg[i].source_file, reg[i].char_offset, reg[i].char_length)
        except Exception:
            out[i] = ""
    return out
