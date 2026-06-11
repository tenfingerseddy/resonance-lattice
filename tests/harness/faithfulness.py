"""faithfulness — the grounding gate scores answers correctly.

The faithfulness gate is the entry point of the confidence lifecycle
(docs/internal/GROUNDING_MODEL.md): a deep-search answer earns a
provisional insight only if it is faithfully grounded. This suite checks
the scoring with a stubbed LLM, offline.

Guarantees:

  1. Fully-grounded answer, on-topic citations -> faithful.
  2. Unsupported claims drag claim_support below the floor -> not faithful.
  3. On-topic-but-WRONG-question: every claim supported (claim_support 1.0)
     but the passages answer an adjacent topic -> not faithful. This is the
     failure mode bare claim-support misses (the Session-4 RLS case).
  4. Empty answer, or no cited passages -> not faithful, with no LLM call.
  5. Judge output unparseable -> not faithful, reason records it.
  6. claim_support is the supported-fraction (2 of 4 -> 0.5).
"""

from __future__ import annotations

import sys

from ._testutil import StubJudgeClient as _StubClient


_PASSAGES = [
    {"source_file": "a.md", "text": "Session tokens expire after 24 hours."},
    {"source_file": "b.md", "text": "Refresh tokens rotate weekly."},
]


def run() -> int:
    from resonance_lattice.store.faithfulness import assess_faithfulness

    failures = 0

    # ---- Guarantee 1: fully grounded, on-topic -> faithful ----
    client = _StubClient(
        '{"claims": ['
        '{"claim": "tokens expire after 24h", "supported": true, "passage": 1},'
        '{"claim": "refresh tokens rotate weekly", "supported": true, "passage": 2}'
        '], "question_relevance": 0.95, "reason": "fully grounded"}'
    )
    r = assess_faithfulness(
        "How long do tokens last?",
        "Tokens expire after 24h; refresh tokens rotate weekly.",
        _PASSAGES, client,
    )
    if not r.faithful or r.claim_support != 1.0:
        print(f"[faithfulness] FAIL g1: {r}", file=sys.stderr)
        failures += 1
    else:
        print("[faithfulness] g1 (grounded -> faithful) OK", file=sys.stderr)

    # ---- Guarantee 2: unsupported claims -> not faithful ----
    client = _StubClient(
        '{"claims": ['
        '{"claim": "c1", "supported": true, "passage": 1},'
        '{"claim": "c2", "supported": false, "passage": null},'
        '{"claim": "c3", "supported": false, "passage": null}'
        '], "question_relevance": 0.9, "reason": "two claims ungrounded"}'
    )
    r = assess_faithfulness("q", "answer", _PASSAGES, client)
    if r.faithful:
        print(f"[faithfulness] FAIL g2: unsupported claims passed: {r}",
              file=sys.stderr)
        failures += 1
    elif abs(r.claim_support - 1 / 3) > 1e-9:
        print(f"[faithfulness] FAIL g2: claim_support={r.claim_support}",
              file=sys.stderr)
        failures += 1
    else:
        print("[faithfulness] g2 (unsupported -> not faithful) OK",
              file=sys.stderr)

    # ---- Guarantee 3: on-topic-but-wrong-question -> not faithful ----
    client = _StubClient(
        '{"claims": ['
        '{"claim": "c1", "supported": true, "passage": 1},'
        '{"claim": "c2", "supported": true, "passage": 2}'
        '], "question_relevance": 0.2, "reason": "passages cover an adjacent topic"}'
    )
    r = assess_faithfulness("q", "answer", _PASSAGES, client)
    if r.faithful:
        print(f"[faithfulness] FAIL g3: wrong-question answer passed: {r}",
              file=sys.stderr)
        failures += 1
    elif r.claim_support != 1.0:
        print(f"[faithfulness] FAIL g3: claim_support should be 1.0: {r}",
              file=sys.stderr)
        failures += 1
    else:
        print("[faithfulness] g3 (wrong-question -> not faithful) OK",
              file=sys.stderr)

    # ---- Guarantee 4: empty answer / no passages -> not faithful, no call ----
    client = _StubClient("{}")
    r = assess_faithfulness("q", "   ", _PASSAGES, client)
    g4_ok = not r.faithful and client.calls == 0
    client2 = _StubClient("{}")
    r = assess_faithfulness("q", "answer", [], client2)
    g4_ok = g4_ok and not r.faithful and client2.calls == 0
    if not g4_ok:
        print(f"[faithfulness] FAIL g4: empty/no-passage path made an LLM "
              f"call or passed", file=sys.stderr)
        failures += 1
    else:
        print("[faithfulness] g4 (empty/no-passages -> not faithful, no call) OK",
              file=sys.stderr)

    # ---- Guarantee 5: unparseable judge output -> not faithful ----
    client = _StubClient("this is not json at all")
    r = assess_faithfulness("q", "answer", _PASSAGES, client)
    if r.faithful or "parse" not in r.reason:
        print(f"[faithfulness] FAIL g5: {r}", file=sys.stderr)
        failures += 1
    else:
        print("[faithfulness] g5 (parse failure -> not faithful) OK",
              file=sys.stderr)

    # ---- Guarantee 6: claim_support is the supported fraction ----
    client = _StubClient(
        '{"claims": ['
        '{"claim": "c1", "supported": true, "passage": 1},'
        '{"claim": "c2", "supported": true, "passage": 2},'
        '{"claim": "c3", "supported": false, "passage": null},'
        '{"claim": "c4", "supported": false, "passage": null}'
        '], "question_relevance": 0.9, "reason": "half grounded"}'
    )
    r = assess_faithfulness("q", "answer", _PASSAGES, client)
    if r.claim_support != 0.5:
        print(f"[faithfulness] FAIL g6: claim_support={r.claim_support}, "
              f"expected 0.5", file=sys.stderr)
        failures += 1
    else:
        print("[faithfulness] g6 (claim_support = supported fraction) OK",
              file=sys.stderr)

    if failures:
        print(f"[faithfulness] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[faithfulness] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
