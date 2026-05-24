"""reverification — LLM re-verification of stale insights.

Guarantees:

  1. With no stale insights, reverify is a no-op (returns []).
  2. LLM says "supports=true" → insight flips back to accepted +
     source_passage_hashes refresh + verdict_signals append.
  3. LLM says "supports=false" → insight flips to retired.
  4. LLM parse failure → outcome.new_state = "skipped"; row stays stale.
  5. All citations orphaned (source removed) → retired without LLM call.
  6. --limit caps the number of stale rows processed per pass.

Stub LLM client: a small dataclass that mimics anthropic.Client.messages.create
shape. Lets the test run offline.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_insight_passage, unpatch_zero_encoder


@dataclass
class _StubMessage:
    text: str


@dataclass
class _StubResponse:
    content: list


class _StubMessages:
    def __init__(self, verdict_text: str):
        self._text = verdict_text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _StubResponse(content=[_StubMessage(text=self._text)])


class _StubClient:
    def __init__(self, verdict_text: str):
        self.messages = _StubMessages(verdict_text)


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store import archive
    from resonance_lattice.store.reverification import reverify_stale_insights

    failures = 0

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        root = d / "corpus"
        files = {
            "a.md": "# Auth\n\nSession tokens expire after 24 hours.",
            "b.md": "# Tokens\n\nRefresh tokens rotate weekly.",
        }
        km = _build(root, files)
        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry]
        src_hashes = [c.content_hash for c in c0.registry]
        encoder = Encoder()

        # ---- Guarantee 1: empty stale set is no-op ----
        accepted = [make_insight_passage(
            0, "Sessions use 24h tokens.", src_ids[:1], src_hashes[:1],
            state="accepted",
        )]
        band = encoder.encode([accepted[0].content]).astype("float32")
        archive.write_insight_layer_in_place(km, accepted, band)

        outcomes = reverify_stale_insights(
            km, _StubClient('{"supports": true, "reason": "ok"}'),
            model="haiku-test",
        )
        if outcomes:
            print(f"[reverify] FAIL g1: outcomes returned for non-stale set "
                  f"({len(outcomes)})", file=sys.stderr)
            failures += 1
        else:
            print("[reverify] g1 (empty stale set is no-op) OK", file=sys.stderr)

        # ---- Setup stale state for next guarantees ----
        stale = [replace(accepted[0], verdict_state="stale")]
        archive.write_insight_layer_in_place(km, stale, band)

        # ---- Guarantee 2: supports=true -> accepted ----
        client = _StubClient(
            '{"supports": true, "reason": "the updated source still covers tokens"}'
        )
        outcomes = reverify_stale_insights(km, client, model="haiku-test")
        if len(outcomes) != 1 or outcomes[0].new_state != "accepted":
            print(f"[reverify] FAIL g2: {outcomes}", file=sys.stderr)
            failures += 1
        else:
            c1 = archive.read(km)
            ins = c1.insights[0]
            if ins.verdict_state != "accepted":
                print(f"[reverify] FAIL g2: state={ins.verdict_state}",
                      file=sys.stderr)
                failures += 1
            elif not any(s.source == "llm" and s.polarity == "accept"
                         for s in ins.verdict_signals):
                print("[reverify] FAIL g2: no llm-accept signal appended",
                      file=sys.stderr)
                failures += 1
            else:
                print("[reverify] g2 (supports=true -> accepted) OK",
                      file=sys.stderr)

        # ---- Guarantee 3: supports=false -> retired ----
        # Set back to stale.
        archive.write_insight_layer_in_place(
            km, [replace(c1.insights[0], verdict_state="stale")], band,
        )
        client = _StubClient(
            '{"supports": false, "reason": "the new source talks about something else"}'
        )
        outcomes = reverify_stale_insights(km, client, model="haiku-test")
        if outcomes[0].new_state != "retired":
            print(f"[reverify] FAIL g3: {outcomes[0]}", file=sys.stderr)
            failures += 1
        else:
            c2 = archive.read(km)
            if c2.insights[0].verdict_state != "retired":
                print(f"[reverify] FAIL g3: state={c2.insights[0].verdict_state}",
                      file=sys.stderr)
                failures += 1
            else:
                print("[reverify] g3 (supports=false -> retired) OK",
                      file=sys.stderr)

        # ---- Guarantee 4: LLM parse failure -> skipped, row stays stale ----
        archive.write_insight_layer_in_place(
            km, [replace(c2.insights[0], verdict_state="stale")], band,
        )
        client = _StubClient("this is not json at all")
        outcomes = reverify_stale_insights(km, client, model="haiku-test")
        if outcomes[0].new_state != "skipped":
            print(f"[reverify] FAIL g4: {outcomes[0]}", file=sys.stderr)
            failures += 1
        else:
            c3 = archive.read(km)
            if c3.insights[0].verdict_state != "stale":
                print(f"[reverify] FAIL g4: row didn't stay stale "
                      f"(state={c3.insights[0].verdict_state})", file=sys.stderr)
                failures += 1
            else:
                print("[reverify] g4 (parse failure -> skipped, stays stale) OK",
                      file=sys.stderr)

        # ---- Guarantee 5: all citations orphan -> retired without LLM call ----
        # Build a fresh stale insight that cites a passage_id that doesn't exist.
        from resonance_lattice.store.insight import InsightCitation
        orphan = replace(
            stale[0],
            citations=(
                InsightCitation(passage_id="does-not-exist",
                                char_span=None, confidence=0.9),
            ),
            verdict_state="stale",
        )
        archive.write_insight_layer_in_place(km, [orphan], band)
        client = _StubClient(
            '{"supports": true, "reason": "should never be called"}'
        )
        outcomes = reverify_stale_insights(km, client, model="haiku-test")
        if outcomes[0].new_state != "retired":
            print(f"[reverify] FAIL g5: {outcomes[0]}", file=sys.stderr)
            failures += 1
        elif client.messages.calls != 0:
            print(f"[reverify] FAIL g5: LLM called {client.messages.calls} "
                  f"times on orphan", file=sys.stderr)
            failures += 1
        else:
            print("[reverify] g5 (orphan citations -> retired, no LLM call) OK",
                  file=sys.stderr)

        # ---- Guarantee 6: --limit caps processing ----
        # Re-create two stale rows.
        two_stale = [
            replace(stale[0], insight_idx=0, verdict_state="stale"),
            replace(stale[0], insight_idx=1, verdict_state="stale",
                    content="Different content",
                    insight_id="diff" + stale[0].insight_id[:12]),
        ]
        two_band = encoder.encode(
            [r.content for r in two_stale]
        ).astype("float32")
        archive.write_insight_layer_in_place(km, two_stale, two_band)
        client = _StubClient(
            '{"supports": true, "reason": "ok"}'
        )
        outcomes = reverify_stale_insights(
            km, client, model="haiku-test", limit=1,
        )
        if len(outcomes) != 1:
            print(f"[reverify] FAIL g6: limit=1 produced {len(outcomes)} outcomes",
                  file=sys.stderr)
            failures += 1
        else:
            print("[reverify] g6 (--limit caps stale rows processed) OK",
                  file=sys.stderr)

    if failures:
        print(f"[reverify] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[reverify] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
