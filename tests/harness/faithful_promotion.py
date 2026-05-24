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

    if failures:
        print(f"[faithful_promotion] {failures} guarantee(s) failed",
              file=sys.stderr)
        return 1
    print("[faithful_promotion] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
