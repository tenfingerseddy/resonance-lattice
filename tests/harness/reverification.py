"""reverification — LLM re-verification of stale insights.

Guarantees:

  1. With no stale insights, reverify is a no-op (returns []).
  2. LLM says "supports=true" → claim flips back to active +
     source_passage_hashes refresh + verdict_signals append.
  3. LLM says "supports=false" → claim flips to retired.
  4. LLM parse failure → outcome.new_state = "skipped"; row stays stale.
  5. All citations orphaned (source removed) → retired without LLM call.
  6. --limit caps the number of stale rows processed per pass.
  7. Successful reverify pass stamps the archive's
     `insight_layer_last_reverify_utc` heartbeat — the freshness signal
     surfaced by `rlat profile`. A parse-failure-only pass (only
     `skipped` outcomes) does NOT stamp it — no successful work
     happened, the heartbeat stays where it was.
  8. `cost_cap_usd` caps cumulative LLM spend. After the first call
     pushes observed spend past the cap, subsequent stale rows record
     as skipped with the cap reason and the LLM is not called again.
  9. Mid-pass exception flushes in-flight progress. A network outage
     (the stub client raises on the second call) propagates the
     exception to the caller, but the `finally`-block flush still
     writes the partial `updated_insights` (the first row's
     re-verification) to disk. The next `rlat reverify` pass only
     touches rows that stayed stale; already-committed work isn't
     re-billed against the cost cap.

Stub LLM client: a small dataclass that mimics anthropic.Client.messages.create
shape, including a `usage` attribute that the cost meter reads. Lets
the test run offline.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim, unpatch_zero_encoder


@dataclass
class _StubMessage:
    text: str


@dataclass
class _StubUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _StubResponse:
    content: list
    usage: _StubUsage | None = None


class _StubMessages:
    def __init__(self, verdict_text: str, usage: _StubUsage | None = None):
        self._text = verdict_text
        self._usage = usage
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _StubResponse(
            content=[_StubMessage(text=self._text)],
            usage=self._usage,
        )


class _StubClient:
    def __init__(self, verdict_text: str, usage: _StubUsage | None = None):
        self.messages = _StubMessages(verdict_text, usage)


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
        accepted = [make_corpus_claim(
            "Sessions use 24h tokens.", src_ids[:1], src_hashes[:1],
            state="active",
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
        stale = [replace(accepted[0], state="stale")]
        archive.write_insight_layer_in_place(km, stale, band)

        # ---- Guarantee 2: supports=true -> active ----
        client = _StubClient(
            '{"supports": true, "reason": "the updated source still covers tokens"}'
        )
        outcomes = reverify_stale_insights(km, client, model="haiku-test")
        if len(outcomes) != 1 or outcomes[0].new_state != "active":
            print(f"[reverify] FAIL g2: {outcomes}", file=sys.stderr)
            failures += 1
        else:
            c1 = archive.read(km)
            ins = c1.insights[0]
            if ins.state != "active":
                print(f"[reverify] FAIL g2: state={ins.state}",
                      file=sys.stderr)
                failures += 1
            elif not any(s.source == "llm" and s.polarity == "accept"
                         for s in ins.facts.verdict_signals):
                print("[reverify] FAIL g2: no llm-accept signal appended",
                      file=sys.stderr)
                failures += 1
            else:
                print("[reverify] g2 (supports=true -> active) OK",
                      file=sys.stderr)

        # ---- Guarantee 3: supports=false -> retired ----
        # Set back to stale.
        archive.write_insight_layer_in_place(
            km, [replace(c1.insights[0], state="stale")], band,
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
            if c2.insights[0].state != "retired":
                print(f"[reverify] FAIL g3: state={c2.insights[0].state}",
                      file=sys.stderr)
                failures += 1
            else:
                print("[reverify] g3 (supports=false -> retired) OK",
                      file=sys.stderr)

        # ---- Guarantee 4: LLM parse failure -> skipped, row stays stale ----
        archive.write_insight_layer_in_place(
            km, [replace(c2.insights[0], state="stale")], band,
        )
        client = _StubClient("this is not json at all")
        outcomes = reverify_stale_insights(km, client, model="haiku-test")
        if outcomes[0].new_state != "skipped":
            print(f"[reverify] FAIL g4: {outcomes[0]}", file=sys.stderr)
            failures += 1
        else:
            c3 = archive.read(km)
            if c3.insights[0].state != "stale":
                print(f"[reverify] FAIL g4: row didn't stay stale "
                      f"(state={c3.insights[0].state})", file=sys.stderr)
                failures += 1
            else:
                print("[reverify] g4 (parse failure -> skipped, stays stale) OK",
                      file=sys.stderr)

        # ---- Guarantee 5: all citations orphan -> retired without LLM call ----
        # Build a fresh stale insight that cites a passage_id that doesn't exist.
        from resonance_lattice.state.claim import evolve
        from resonance_lattice.store.insight import InsightCitation
        orphan = evolve(
            stale[0],
            citations=(
                InsightCitation(passage_id="does-not-exist",
                                char_span=None, confidence=0.9),
            ),
            state="stale",
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
        # Re-create two stale rows — distinct content gives distinct claim_ids.
        two_stale = [
            make_corpus_claim(stale[0].content, src_ids[:1], src_hashes[:1],
                              state="stale"),
            make_corpus_claim("Different content", src_ids[:1], src_hashes[:1],
                              state="stale"),
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

        # ---- Guarantee 7: successful reverify stamps the heartbeat ----
        # The last successful pass above (g6, supports=true) should have
        # set insight_layer_last_reverify_utc to a non-empty ISO timestamp.
        c_post = archive.read(km)
        heartbeat = c_post.metadata.insight_layer_last_reverify_utc
        if not heartbeat:
            print(f"[reverify] FAIL g7: heartbeat empty after successful pass",
                  file=sys.stderr)
            failures += 1
        else:
            # A parse-failure-only pass must NOT advance the heartbeat.
            # Reset to stale; run a parse-failure-only pass; confirm
            # heartbeat is unchanged.
            archive.write_insight_layer_in_place(
                km, [replace(c_post.insights[0], state="stale")],
                encoder.encode([c_post.insights[0].content]).astype("float32"),
            )
            c_pre_skip = archive.read(km)
            heartbeat_pre_skip = c_pre_skip.metadata.insight_layer_last_reverify_utc
            outcomes_skip = reverify_stale_insights(
                km, _StubClient("not json"), model="haiku-test",
            )
            c_post_skip = archive.read(km)
            if any(o.new_state != "skipped" for o in outcomes_skip):
                print(f"[reverify] FAIL g7: parse-failure pass produced "
                      f"non-skipped outcomes: {outcomes_skip}", file=sys.stderr)
                failures += 1
            elif c_post_skip.metadata.insight_layer_last_reverify_utc \
                    != heartbeat_pre_skip:
                print(f"[reverify] FAIL g7: skipped-only pass advanced heartbeat",
                      file=sys.stderr)
                failures += 1
            else:
                print("[reverify] g7 (heartbeat stamped on success, "
                      "stays on parse-fail) OK", file=sys.stderr)

        # ---- Guarantee 8: cost_cap_usd stops the loop early ----
        # Two fresh stale rows; stub reports ~$0.315 per call
        # (100k input * $3/M + 1k output * $15/M = $0.315). Cap at
        # $0.20 → first call's observed spend ($0.315) crosses the cap;
        # the second row short-circuits with the cap reason and the
        # LLM is not called again.
        archive.write_insight_layer_in_place(km, two_stale, two_band)
        capped_client = _StubClient(
            '{"supports": true, "reason": "ok"}',
            usage=_StubUsage(input_tokens=100_000, output_tokens=1_000),
        )
        outcomes_cap = reverify_stale_insights(
            km, capped_client, model="haiku-test", cost_cap_usd=0.20,
        )
        skipped = [o for o in outcomes_cap if o.new_state == "skipped"]
        active = [o for o in outcomes_cap if o.new_state == "active"]
        if len(active) != 1 or len(skipped) != 1:
            print(f"[reverify] FAIL g8: expected 1 active + 1 cap-skipped, got "
                  f"active={len(active)} skipped={len(skipped)}",
                  file=sys.stderr)
            failures += 1
        elif "cost cap" not in skipped[0].reason:
            print(f"[reverify] FAIL g8: skipped reason missing cap mention: "
                  f"{skipped[0].reason!r}", file=sys.stderr)
            failures += 1
        elif capped_client.messages.calls != 1:
            print(f"[reverify] FAIL g8: LLM called "
                  f"{capped_client.messages.calls} times under cap (want 1)",
                  file=sys.stderr)
            failures += 1
        else:
            print("[reverify] g8 (cost_cap_usd stops loop after cap crossed) OK",
                  file=sys.stderr)

        # ---- Guarantee 9: mid-pass exception flushes in-flight progress ----
        # Two stale insights; the stub client raises on the second LLM call.
        # The first insight's re-verification should land on disk; the
        # exception should propagate to the caller; the second insight
        # should stay stale for the next pass.
        c_post = archive.read(km)
        two_stale_again = [
            evolve(c_post.insights[0], state="stale"),
            evolve(c_post.insights[1], state="stale"),
        ]
        archive.write_insight_layer_in_place(km, two_stale_again, two_band)

        class _RaiseOnSecondCall:
            def __init__(self):
                self.calls = 0
                self.messages = self

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return _StubResponse(
                        content=[_StubMessage(
                            text='{"supports": true, "reason": "ok"}'
                        )],
                        usage=_StubUsage(input_tokens=10, output_tokens=10),
                    )
                raise RuntimeError("simulated network outage on call 2")

        raise_client = _RaiseOnSecondCall()
        raised = False
        try:
            reverify_stale_insights(km, raise_client, model="haiku-test")
        except RuntimeError as exc:
            raised = "network outage" in str(exc)

        c_final = archive.read(km)
        first_active = c_final.insights[0].state == "active"
        second_stale = c_final.insights[1].state == "stale"
        if not raised:
            print("[reverify] FAIL g9: exception didn't propagate "
                  "to the caller", file=sys.stderr)
            failures += 1
        elif not first_active:
            print(f"[reverify] FAIL g9: first row didn't flush "
                  f"(state={c_final.insights[0].state})", file=sys.stderr)
            failures += 1
        elif not second_stale:
            print(f"[reverify] FAIL g9: second row state="
                  f"{c_final.insights[1].state} (want stale for next pass)",
                  file=sys.stderr)
            failures += 1
        else:
            print("[reverify] g9 (mid-pass exception flushes in-flight "
                  "progress) OK", file=sys.stderr)

    if failures:
        print(f"[reverify] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[reverify] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
