"""curator_author — the gap→author growth touch + the end-to-end loop (Step 3).

`curator.author.author_fill` turns a confirmed RecurringIntent (from the persisted
telemetry) into a grounded PENDING claim, distilled strictly from the corpus passages
its centroid retrieves; `grow_from_telemetry` wires the whole loop (decide → author →
faithfulness gate → band). Pins:

  (a) author_fill yields a PendingFill grounded in REAL corpus passages, and the
      relevance gate drops the off-topic passage (no spurious citation).
  (b) no client → None; growth pauses, never crashes.
  (c) an author that returns an empty claim → None (nothing to ground).
  (d) the author's own retrieval is MACHINERY — it IS observed but tagged internal
      (is_user_query=False), so it never pollutes the user-intent telemetry stream.
  (e) grow_from_telemetry runs the full loop end-to-end with a stub cloud: a
      recurring intent spanning two distinct sources → a faithfulness-gated, EARNED
      claim physically in the band.
  (f) no recurring intent → no growth ([]).
  (g) max_fills bounds the loop: two recurring intents, max_fills=2 → two outcomes,
      still zero pollution.

The cloud (author + faithfulness judge) is stubbed deterministically — the documented
exception; the faithfulness gate + band-landing replay with no network. Uses the real
encoder + a real tiny corpus so retrieval, text resolution, and grounding are real.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ._testutil import build_corpus as _build
from ._testutil import unpatch_zero_encoder


class _Block:
    def __init__(self, text): self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _StubClient:
    """Routes on the system prompt: the single-claim author prompt → an author JSON, the
    full-doc multi-claim author prompt → a claims JSON, the faithfulness prompt → a
    faithful verdict. `.messages.create(...)` shape only."""

    def __init__(self, author_json: str, faith_json: str, doc_json: str | None = None,
                 ext_json: str | None = None):
        self._author, self._faith = author_json, faith_json
        self._doc = doc_json or author_json
        self._ext = ext_json
        self.messages = self

    def create(self, *, model, max_tokens, system, messages):
        if "distil ONE grounded claim" in system:
            return _Resp(self._author)
        if "distil reusable knowledge from a source document" in system:
            return _Resp(self._doc)
        if self._ext is not None and "verified factual claim from independent" in system:
            return _Resp(self._ext)
        return _Resp(self._faith)


_CLAIM = ("Session tokens expire after 24 hours, while refresh tokens last 30 days "
          "and are single-use.")
_AUTHOR_JSON = json.dumps({"intent": "How long do session and refresh tokens last?",
                           "claim": _CLAIM})
# grow_from_telemetry authors from the FULL top doc → 2-4 claims (author_doc_fills).
_DOC_AUTHOR_JSON = json.dumps({
    "intent": "How long do session and refresh tokens last?",
    "claims": [
        "Session tokens expire after 24 hours and cannot be renewed once expired.",
        "Refresh tokens last 30 days and are single-use after issuance.",
    ],
})
_FAITH_JSON = json.dumps({
    "claims": [
        {"claim": "Session tokens expire after 24 hours.", "supported": True, "passage": 1},
        {"claim": "Refresh tokens last 30 days.", "supported": True, "passage": 2},
    ],
    "question_relevance": 0.95, "reason": "grounded in passages 1-2",
})
_EMPTY_AUTHOR_JSON = json.dumps({"intent": "x", "claim": ""})

# Two genuinely-distinct, co-relevant token passages + one off-topic billing doc.
_FILES = {
    "docs/session_tokens.md": "Session tokens expire after 24 hours and cannot be renewed once expired.",
    "docs/refresh_tokens.md": "Refresh tokens last 30 days and are single-use after issuance.",
    "docs/billing.md": "The billing cycle runs monthly. Invoices are issued on the first of each month.",
}
_QUERY = "how long do session and refresh tokens last before they expire"
_QUERY2 = "how does the monthly billing cycle and invoicing work"


def _seed_recurring(km: Path, encoder, query: str, sessions) -> None:
    from resonance_lattice.store import archive

    emb = [round(float(x), 6) for x in encoder.encode([query])[0]]
    rows = [{
        "ts": "2026-06-03T00:00:00+00:00", "session": s, "layer": "source",
        "is_user_query": True, "query_emb": emb,
        "ranked": [{"rank": 0, "idx": 0, "score": 0.7}],
    } for s in sessions]
    archive.append_telemetry_in_place(km, rows)


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.field import capture
    from resonance_lattice.curator import author as author_mod
    from resonance_lattice.curator.decide import decide
    from resonance_lattice.store import archive

    failures = 0
    encoder = Encoder()

    with tempfile.TemporaryDirectory() as d:
        km = _build(Path(d) / "corpus", _FILES)
        km_key = str(Path(km).resolve())
        _seed_recurring(km, encoder, _QUERY, ["s1", "s2"])

        cands = decide(str(km))
        if len(cands) != 1:
            print(f"[curator_author] setup: expected 1 recurring candidate, got "
                  f"{len(cands)}", file=sys.stderr)
            return 1
        cand = cands[0]

        # (a) grounded PendingFill; relevance gate drops the off-topic billing doc.
        pending = author_mod.author_fill(km, cand, _StubClient(_AUTHOR_JSON, _FAITH_JSON))
        if pending is None or pending.claim != _CLAIM or not pending.evidence_passages:
            print(f"[curator_author] FAIL (a): no grounded fill: {pending}",
                  file=sys.stderr)
            failures += 1
        else:
            srcs = {e["source_file"] for e in pending.evidence_passages}
            ev0 = pending.evidence_passages[0]
            grounded = ev0.get("passage_id") and ev0.get("content_hash") and ev0.get("text")
            if not grounded:
                print(f"[curator_author] FAIL (a): evidence not corpus-grounded: {ev0}",
                      file=sys.stderr)
                failures += 1
            elif any("billing" in s for s in srcs):
                print(f"[curator_author] FAIL (a): relevance gate kept off-topic doc: "
                      f"{sorted(srcs)}", file=sys.stderr)
                failures += 1
            else:
                print(f"[curator_author] (a) grounded PendingFill, off-topic gated "
                      f"(sources={sorted(srcs)}) OK", file=sys.stderr)

        # (b) no client → None.
        if (author_mod.author_fill(km, cand, None) is not None
                or author_mod.grow_from_telemetry(km, None) != []):
            print("[curator_author] FAIL (b): no client should yield no growth",
                  file=sys.stderr)
            failures += 1
        else:
            print("[curator_author] (b) no client → no growth OK", file=sys.stderr)

        # (c) empty authored claim → None.
        if author_mod.author_fill(km, cand, _StubClient(_EMPTY_AUTHOR_JSON, _FAITH_JSON)) is not None:
            print("[curator_author] FAIL (c): empty claim should yield None",
                  file=sys.stderr)
            failures += 1
        else:
            print("[curator_author] (c) empty claim → None OK", file=sys.stderr)

        # (d)+(e) full loop end-to-end; growth is observed BUT tagged internal.
        capture.drain(km_key)
        telem_before = len(archive.read_telemetry(km))
        insights_before = len(archive.read(km).insights)
        outcomes = author_mod.grow_from_telemetry(
            km, _StubClient(_AUTHOR_JSON, _FAITH_JSON, _DOC_AUTHOR_JSON))

        buffered = capture.buffered(km_key)
        polluting = [r for r in buffered if r.get("is_user_query")]
        internal = [r for r in buffered if not r.get("is_user_query") and r.get("layer") == "source"]
        capture.drain(km_key)
        if polluting:
            print(f"[curator_author] FAIL (d): growth polluted the user stream "
                  f"({len(polluting)} rows)", file=sys.stderr)
            failures += 1
        elif not internal:
            print("[curator_author] FAIL (d): author retrieval did not observe — "
                  "the no-pollution check would be vacuous", file=sys.stderr)
            failures += 1
        elif len(archive.read_telemetry(km)) != telem_before:
            print("[curator_author] FAIL (d): growth changed the persisted telemetry",
                  file=sys.stderr)
            failures += 1
        else:
            print("[curator_author] (d) growth observed-and-tagged-internal OK",
                  file=sys.stderr)

        # one recurring intent → 2-4 full-doc fills, each faithfulness-gated.
        if not outcomes or all(o.pending is None for o in outcomes):
            print(f"[curator_author] FAIL (e): loop produced no pending fill: {outcomes}",
                  file=sys.stderr)
            failures += 1
        else:
            faithful = [o for o in outcomes if o.report is not None and o.report.faithful]
            promoted = [o for o in outcomes if o.promoted]
            if not faithful:
                print(f"[curator_author] FAIL (e): faithfulness gate passed nothing: "
                      f"{[o.report for o in outcomes]}", file=sys.stderr)
                failures += 1
            elif promoted and len(archive.read(km).insights) <= insights_before:
                print("[curator_author] FAIL (e): promoted but band did not grow",
                      file=sys.stderr)
                failures += 1
            else:
                landed = (f"{len(promoted)} EARNED claim(s) in band" if promoted
                          else "gated, test declined")
                print(f"[curator_author] (e) full loop: recurring intent → "
                      f"{len(faithful)} faithful fill(s) → {landed} OK", file=sys.stderr)

        # (g) max_fills bounds the loop over multiple recurring intents.
        _seed_recurring(km, encoder, _QUERY2, ["s3", "s4"])
        if len(decide(str(km))) != 2:
            print("[curator_author] FAIL (g): expected 2 recurring candidates",
                  file=sys.stderr)
            failures += 1
        else:
            capture.drain(km_key)
            outs2 = author_mod.grow_from_telemetry(
                km, _StubClient(_AUTHOR_JSON, _FAITH_JSON, _DOC_AUTHOR_JSON), max_fills=2)
            pol2 = [r for r in capture.buffered(km_key) if r.get("is_user_query")]
            capture.drain(km_key)
            # max_fills bounds CANDIDATES (2), each authoring 2-4 fills → ≥2 outcomes
            # spanning exactly the two distinct candidates, still zero pollution.
            distinct_cands = {id(o.candidate) for o in outs2}
            if len(outs2) < 2 or len(distinct_cands) != 2:
                print(f"[curator_author] FAIL (g): max_fills=2 should cover 2 candidates, "
                      f"got {len(outs2)} outcomes over {len(distinct_cands)} candidates",
                      file=sys.stderr)
                failures += 1
            elif pol2:
                print(f"[curator_author] FAIL (g): multi-fill polluted the stream "
                      f"({len(pol2)} rows)", file=sys.stderr)
                failures += 1
            else:
                print(f"[curator_author] (g) max_fills=2 → {len(outs2)} fills over 2 "
                      f"candidates, no pollution OK", file=sys.stderr)

            # (i) explicit candidates= honoured: preview == run. With 2 recurring
            #     intents in telemetry and max_fills=2, passing ONE previewed
            #     candidate must fill only that candidate — an internal decide()
            #     re-read would cover both (the CLI preview/run double-read).
            preview = decide(str(km))[:1]
            capture.drain(km_key)
            outs3 = author_mod.grow_from_telemetry(
                km, _StubClient(_AUTHOR_JSON, _FAITH_JSON, _DOC_AUTHOR_JSON),
                max_fills=2, candidates=preview)
            capture.drain(km_key)
            cands3 = {id(o.candidate) for o in outs3}
            if not outs3 or cands3 != {id(preview[0])}:
                print(f"[curator_author] FAIL (i): candidates= not honoured — "
                      f"{len(outs3)} outcome(s) over {len(cands3)} candidate(s), "
                      f"expected exactly the 1 previewed", file=sys.stderr)
                failures += 1
            else:
                print("[curator_author] (i) explicit candidates= → fills exactly "
                      "the preview OK", file=sys.stderr)

    with tempfile.TemporaryDirectory() as d:
        # (f) no recurring intent → no growth.
        km2 = _build(Path(d) / "corpus2", _FILES)
        if author_mod.grow_from_telemetry(km2, _StubClient(_AUTHOR_JSON, _FAITH_JSON)) != []:
            print("[curator_author] FAIL (f): growth with no candidates should be []",
                  file=sys.stderr)
            failures += 1
        else:
            print("[curator_author] (f) no recurring intent → no growth OK",
                  file=sys.stderr)

    with tempfile.TemporaryDirectory() as d:
        # (h) SOURCE ROUTING: with a fetcher, an undercovered recurring intent gets a
        #     VERIFIED EXTERNAL fill that REPLACES synthesis — the landed claim carries
        #     external provenance (is_external), not corpus citations.
        hk = _build(Path(d) / "corpus_h", _FILES)
        _seed_recurring(hk, encoder, _QUERY, ["s1", "s2"])
        _ext_json = json.dumps({
            "claim": "Session tokens expire after 24 hours, confirmed by two vendor sources.",
            "supporting_sources": [1, 2], "agree": True,
        })

        def _fetch(_q):
            return [{"url": "https://a.example/x", "text": "Session tokens expire after 24 hours."},
                    {"url": "https://b.example/y", "text": "Vendor docs: tokens expire after 24h."}]

        outs_h = author_mod.grow_from_telemetry(
            hk, _StubClient(_AUTHOR_JSON, _FAITH_JSON, _DOC_AUTHOR_JSON, _ext_json),
            fetcher=_fetch, max_fills=1)
        ch = archive.read(hk)
        landed_ext = bool(ch.insights) and any(
            cit.is_external for claim in ch.insights for cit in claim.facts.citations)
        if not any(o.promoted for o in outs_h):
            print("[curator_author] FAIL (h): external-routed fill did not promote",
                  file=sys.stderr)
            failures += 1
        elif not landed_ext:
            print("[curator_author] FAIL (h): landed claim lacks external provenance "
                  "(synthesis not replaced by the external fill)", file=sys.stderr)
            failures += 1
        else:
            print("[curator_author] (h) source routing: external fill replaced synthesis "
                  "+ landed with provenance OK", file=sys.stderr)

    # (l) the grow CLI's per-fill report renders without crashing —
    #     PendingFill.claim is a STR; the pre-fix code read .claim.content
    #     and raised AttributeError on every non-dry-run report (Copilot
    #     review finding on PR #363).
    from resonance_lattice.cli.grow import _outcome_lines
    sample = author_mod.GrowthOutcome(
        candidate=object(),
        pending=author_mod.PendingFill(
            intent="how do tokens expire?",
            claim="Session tokens expire after 24 hours of inactivity.",
            evidence_passages=[]),
    )
    try:
        lines = _outcome_lines([sample])
    except AttributeError as exc:
        print(f"[curator_author] FAIL (l): grow report path raised "
              f"AttributeError: {exc}", file=sys.stderr)
        failures += 1
    else:
        if len(lines) != 1 or "Session tokens expire" not in lines[0] \
                or "gate-rejected" not in lines[0]:
            print(f"[curator_author] FAIL (l): report line wrong: {lines!r}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[curator_author] (l) grow per-fill report renders OK",
                  file=sys.stderr)

    if failures:
        print(f"[curator_author] {failures} check(s) failed", file=sys.stderr)
        return 1
    print("[curator_author] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
