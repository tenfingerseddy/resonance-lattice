"""faithful_promotion — deep-search answers earn the insight layer via the
faithfulness gate, not a user verdict.

Exercises `store.promotion.promote_if_faithful` end to end with a stubbed
faithfulness judge (the GROUNDING_MODEL confidence lifecycle entry point):

  1. A faithful answer is promoted — the insight layer grows by one.
  2. An unfaithful answer is not promoted — the archive is untouched.
  3. Re-promoting the same faithful answer is idempotent (no duplicate).
"""

from __future__ import annotations

import sys
import tempfile
from collections import namedtuple
from pathlib import Path

from ._testutil import build_corpus as _build
from ._testutil import unpatch_zero_encoder


_StubContent = namedtuple("_StubContent", "text")
_StubResponse = namedtuple("_StubResponse", "content")


class _StubClient:
    """Anthropic-shaped client replaying one scripted faithfulness verdict."""

    def __init__(self, response_text: str):
        self._text = response_text
        outer = self

        class _Messages:
            def create(self, **kwargs):
                return _StubResponse(content=[_StubContent(text=outer._text)])

        self.messages = _Messages()


_FAITHFUL = (
    '{"claims": ['
    '{"claim": "tokens expire after 24h", "supported": true, "passage": 1},'
    '{"claim": "refresh tokens rotate weekly", "supported": true, "passage": 2}'
    '], "question_relevance": 0.95, "reason": "grounded"}'
)
_UNFAITHFUL = (
    '{"claims": ['
    '{"claim": "c1", "supported": false, "passage": null},'
    '{"claim": "c2", "supported": false, "passage": null}'
    '], "question_relevance": 0.9, "reason": "claims not grounded"}'
)
_FAITHFUL_EXT = (
    '{"claims": ['
    '{"claim": "the widget ships in March 2027", "supported": true, "passage": 1}'
    '], "question_relevance": 0.95, "reason": "grounded in both external sources"}'
)


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.store import archive
    from resonance_lattice.store.promotion import promote_if_faithful

    failures = 0

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, {
            "a.md": "# Auth\n\nSession tokens expire after 24 hours.",
            "b.md": "# Tokens\n\nRefresh tokens rotate weekly.",
        })
        c0 = archive.read(km)
        evidence = [
            {"source_file": c.source_file, "char_offset": c.char_offset,
             "char_length": c.char_length, "score": 0.9,
             "text": "Session tokens and refresh tokens.",
             "passage_id": c.passage_id, "content_hash": c.content_hash}
            for c in c0.registry[:2]
        ]
        faithful_answer = (
            "Session tokens expire after 24 hours, and refresh tokens "
            "rotate on a weekly cadence."
        )

        # ---- Guarantee 1: faithful -> promoted ----
        report, outcomes = promote_if_faithful(
            km, question="how long do tokens last",
            answer=faithful_answer, evidence_passages=evidence,
            client=_StubClient(_FAITHFUL),
        )
        c1 = archive.read(km)
        if not report.faithful:
            print("[faithful_promotion] FAIL g1: report not faithful",
                  file=sys.stderr)
            failures += 1
        elif len(c1.insights) != 1 or not any(o.promoted for o in outcomes):
            print(f"[faithful_promotion] FAIL g1: insights={len(c1.insights)} "
                  f"outcomes={[o.promoted for o in outcomes]}", file=sys.stderr)
            failures += 1
        else:
            print("[faithful_promotion] g1 (faithful -> promoted) OK",
                  file=sys.stderr)

        # ---- Guarantee 2: unfaithful -> not promoted ----
        report, outcomes = promote_if_faithful(
            km, question="unrelated question",
            answer="An ungrounded claim about an unrelated topic entirely.",
            evidence_passages=evidence, client=_StubClient(_UNFAITHFUL),
        )
        c2 = archive.read(km)
        if report.faithful or outcomes:
            print(f"[faithful_promotion] FAIL g2: unfaithful answer promoted "
                  f"(faithful={report.faithful}, outcomes={outcomes})",
                  file=sys.stderr)
            failures += 1
        elif len(c2.insights) != 1:
            print(f"[faithful_promotion] FAIL g2: insight count changed "
                  f"({len(c2.insights)})", file=sys.stderr)
            failures += 1
        else:
            print("[faithful_promotion] g2 (unfaithful -> not promoted) OK",
                  file=sys.stderr)

        # ---- Guarantee 3: re-promoting the same answer is idempotent ----
        report, outcomes = promote_if_faithful(
            km, question="how long do tokens last",
            answer=faithful_answer, evidence_passages=evidence,
            client=_StubClient(_FAITHFUL),
        )
        c3 = archive.read(km)
        if len(c3.insights) != 1:
            print(f"[faithful_promotion] FAIL g3: re-promotion duplicated "
                  f"(insights={len(c3.insights)})", file=sys.stderr)
            failures += 1
        elif any(o.promoted for o in outcomes):
            print(f"[faithful_promotion] FAIL g3: re-promotion reported a "
                  f"new promotion", file=sys.stderr)
            failures += 1
        else:
            print("[faithful_promotion] g3 (re-promotion idempotent) OK",
                  file=sys.stderr)

        # ---- Guarantee 4: an EXTERNAL fill LANDS with external provenance ----
        # The full external-provenance path: external evidence (non-corpus URLs,
        # synthetic external: ids) → faithfulness gate → external-aware compression
        # gate (passed_external, no corpus-coverage requirement) → band, carrying the
        # source_url provenance. Proves Increments 1+2+3 end-to-end.
        ext_km = _build(Path(d) / "ext_corpus", {
            "x.md": "# Topic\n\nUnrelated local content about something else.",
            "y.md": "# More\n\nMore unrelated local content here.",
        })
        ext_evidence = [
            {"source_file": "https://a.example/x", "source_url": "https://a.example/x",
             "passage_id": "external:aaaaaaaaaaaaaaaa", "content_hash": "sha256:aa",
             "text": "The widget ships in March 2027, per the vendor announcement.",
             "score": 1.0, "drift_status": "external"},
            {"source_file": "https://b.example/y", "source_url": "https://b.example/y",
             "passage_id": "external:bbbbbbbbbbbbbbbb", "content_hash": "sha256:bb",
             "text": "The vendor confirms a March 2027 ship date for the widget.",
             "score": 1.0, "drift_status": "external"},
        ]
        report, outcomes = promote_if_faithful(
            ext_km, question="when does the widget ship",
            answer="The widget ships in March 2027, confirmed by two independent sources.",
            evidence_passages=ext_evidence, client=_StubClient(_FAITHFUL_EXT),
        )
        ce = archive.read(ext_km)
        landed_external = bool(ce.insights) and any(
            cit.is_external for claim in ce.insights for cit in claim.facts.citations)
        if not report.faithful:
            print("[faithful_promotion] FAIL g4: external answer not faithful", file=sys.stderr)
            failures += 1
        elif len(ce.insights) != 1 or not any(o.promoted for o in outcomes):
            print(f"[faithful_promotion] FAIL g4: external fill did not land "
                  f"(insights={len(ce.insights)}, reason={outcomes[0].test_result.reason if outcomes else 'none'})",
                  file=sys.stderr)
            failures += 1
        elif not landed_external:
            print("[faithful_promotion] FAIL g4: landed claim missing external provenance",
                  file=sys.stderr)
            failures += 1
        else:
            print("[faithful_promotion] g4 (external fill lands with provenance) OK",
                  file=sys.stderr)

        # ---- Guarantee 5: external fills are DURABLE across the drift cascade ----
        # detect_drift skips external citations (not corpus passages), so a refresh
        # against an UNCHANGED corpus must NOT stale/evict the landed external fill.
        from resonance_lattice.state.claim_lifecycle import propagate_drift
        fresh = {c.passage_id: c.content_hash for c in ce.registry}
        kept, drifted = propagate_drift(list(ce.insights), fresh)
        ext_states = [k.state for k in kept
                      if any(cit.is_external for cit in k.facts.citations)]
        if drifted or any(s != "active" for s in ext_states):
            print(f"[faithful_promotion] FAIL g5: external fill staled by drift cascade "
                  f"(drifted={drifted}, ext_states={ext_states})", file=sys.stderr)
            failures += 1
        else:
            print("[faithful_promotion] g5 (external fill durable across drift cascade) OK",
                  file=sys.stderr)

        # ---- Guarantee 6: the provenance="user" override lands at a HIGHER trust tier ----
        # The Claude-in-loop keystone: a user-vouched fact (still faithfulness-gated, so never unverified) seeds
        # the band at the USER tier — strictly higher trust than the same fact landed at the default (corpus) tier.
        from resonance_lattice.store.insight import beta_mean, confidence_band
        _RANK = {"low": 0, "medium": 1, "high": 2, "verified": 3}

        def _land(provenance):
            # Reuse the g1 setup (a two-passage synthesis that is PROVEN to land); only the provenance varies.
            with tempfile.TemporaryDirectory() as d6:
                k = _build(Path(d6) / "c", {
                    "a.md": "# Auth\n\nSession tokens expire after 24 hours.",
                    "b.md": "# Tokens\n\nRefresh tokens rotate weekly.",
                })
                cc = archive.read(k)
                ev = [{"source_file": r.source_file, "char_offset": r.char_offset, "char_length": r.char_length,
                       "score": 0.9, "text": "Session tokens and refresh tokens.",
                       "passage_id": r.passage_id, "content_hash": r.content_hash} for r in cc.registry[:2]]
                promote_if_faithful(
                    k, question="how long do tokens last",
                    answer="Session tokens expire after 24 hours, and refresh tokens rotate on a weekly cadence.",
                    evidence_passages=ev, client=_StubClient(_FAITHFUL), provenance=provenance)
                ins = archive.read(k).insights
                if not ins:
                    return None
                return beta_mean(ins[0].corroboration, ins[0].falsification)

        t_user, t_corpus = _land("user"), _land(None)
        if t_user is None or t_corpus is None:
            print(f"[faithful_promotion] FAIL g6: fill did not land (user={t_user}, corpus={t_corpus})",
                  file=sys.stderr)
            failures += 1
        elif not (t_user > t_corpus and _RANK[confidence_band(t_user)] >= _RANK[confidence_band(t_corpus)]):
            print(f"[faithful_promotion] FAIL g6: user tier not higher "
                  f"(user={t_user:.3f}/{confidence_band(t_user)} corpus={t_corpus:.3f}/{confidence_band(t_corpus)})",
                  file=sys.stderr)
            failures += 1
        else:
            print("[faithful_promotion] g6 (provenance=user lands higher trust) OK", file=sys.stderr)

        # ---- Guarantee 7: CALLER-VERIFIED landing (client=None) — the free agent/human path ----
        # No LLM judge: the caller asserts verification with an explicit faithfulness score. The compression test +
        # >=2-citation + trust gates still apply; a missing score REFUSES to land (the safety floor).
        with tempfile.TemporaryDirectory() as d7:
            k7 = _build(Path(d7) / "c", {
                "a.md": "# Auth\n\nSession tokens expire after 24 hours.",
                "b.md": "# Tokens\n\nRefresh tokens rotate weekly.",
            })
            c7 = archive.read(k7)
            ev7 = [{"source_file": r.source_file, "char_offset": r.char_offset, "char_length": r.char_length,
                    "score": 0.9, "text": "Session tokens and refresh tokens.",
                    "passage_id": r.passage_id, "content_hash": r.content_hash} for r in c7.registry[:2]]
            ans7 = "Session tokens expire after 24 hours, and refresh tokens rotate on a weekly cadence."
            rep7, outs7 = promote_if_faithful(
                k7, question="how long do tokens last", answer=ans7, evidence_passages=ev7,
                client=None, faithfulness=0.9, provenance="user")
            landed7 = archive.read(k7).insights
            if not (rep7.faithful and any(o.promoted for o in outs7) and len(landed7) == 1):
                print(f"[faithful_promotion] FAIL g7: caller-verified did not land "
                      f"(faithful={rep7.faithful}, insights={len(landed7)})", file=sys.stderr)
                failures += 1
            else:
                # the safety floor: client=None AND no faithfulness score -> refuse, archive untouched
                rep7b, outs7b = promote_if_faithful(
                    k7, question="another", answer="A different grounded fact about token rotation.",
                    evidence_passages=ev7, client=None, faithfulness=None)
                if rep7b.faithful or outs7b or len(archive.read(k7).insights) != 1:
                    print(f"[faithful_promotion] FAIL g7: missing-faithfulness must refuse "
                          f"(faithful={rep7b.faithful}, insights={len(archive.read(k7).insights)})", file=sys.stderr)
                    failures += 1
                else:
                    print("[faithful_promotion] g7 (caller-verified landing + safety floor) OK", file=sys.stderr)

        # ---- Guarantee 8: caller-verified does NOT escape the TRUST floor ----
        # client=None with a LOW asserted faithfulness seeds trust below the promote floor (0.5) -> not promoted.
        # Proves a caller cannot game the gate: skipping the LLM judge still respects every downstream gate.
        with tempfile.TemporaryDirectory() as d8:
            k8 = _build(Path(d8) / "c", {
                "a.md": "# Auth\n\nSession tokens expire after 24 hours.",
                "b.md": "# Tokens\n\nRefresh tokens rotate weekly.",
            })
            c8 = archive.read(k8)
            ev8 = [{"source_file": r.source_file, "char_offset": r.char_offset, "char_length": r.char_length,
                    "score": 0.9, "text": "Session tokens and refresh tokens.",
                    "passage_id": r.passage_id, "content_hash": r.content_hash} for r in c8.registry[:2]]
            rep8, outs8 = promote_if_faithful(
                k8, question="how long do tokens last",
                answer="Session tokens expire after 24 hours, and refresh tokens rotate on a weekly cadence.",
                evidence_passages=ev8, client=None, faithfulness=0.1)
            if any(o.promoted for o in outs8) or archive.read(k8).insights:
                print(f"[faithful_promotion] FAIL g8: low caller faithfulness must not clear the trust floor "
                      f"(insights={len(archive.read(k8).insights)})", file=sys.stderr)
                failures += 1
            else:
                print("[faithful_promotion] g8 (caller-verified respects trust floor) OK", file=sys.stderr)

    if failures:
        print(f"[faithful_promotion] {failures} guarantee(s) failed",
              file=sys.stderr)
        return 1
    print("[faithful_promotion] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
