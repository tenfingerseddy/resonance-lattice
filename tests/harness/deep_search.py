"""deep_search — the multi-hop research loop has the contracts the CLI depends on.

Seven guarantees, all measured here:

  1. **Plan → search → answer**: the simplest happy path returns the
     refiner's `answer` decision verbatim and stops the loop.

  2. **Search → search → answer (multi-hop)**: when the refiner picks
     `search`, the loop re-encodes + retrieves with the new query and
     continues; the second hop's `answer` decision wins.

  3. **Give-up path**: when the refiner picks `give_up`, the loop
     returns the canonical `GIVE_UP_ANSWER` (so the strict judge can
     score this as a principled refusal, not a hallucination).

  4. **Max-hops exhaustion → synth**: when the refiner keeps picking
     `search` past the budget, the loop synthesises a final answer
     from the union of accumulated evidence.

  5. **Name-check union semantics**: the namecheck runs over the union
     of all hops' verified passages, not just the last hop. A
     question whose distinctive token never appeared in any hop's
     evidence gets a refusal directive prepended (default) or the
     full name-mismatch refusal (`strict_names=True`).

  6. **Cost + token accounting**: every LLM call's tokens accumulate
     into `result.input_tokens` + `result.output_tokens` + `cost_usd`.
     Used by the CLI to print spend and by the bench harness for
     $/correct comparisons.

  7. **Insight-aware retrieval**: a corpus with a promoted insight
     layer surfaces accepted insights into each hop's evidence,
     labelled distinctly from source passages (architecture §10).

  8. **Parse-failure recovery**: when the refiner emits unparseable
     output, the loop recovers with a synth call over accumulated
     evidence rather than handing back raw model text.

Phase 7 deliverable. Ships v2.0 launch headline.
"""

from __future__ import annotations

import sys
import tempfile
from collections import namedtuple
from pathlib import Path

from ._testutil import Args as _Args, build_corpus


# A minimal stub of the anthropic SDK message shape the loop reads:
#   msg.content[0].text + msg.usage.input_tokens + msg.usage.output_tokens
_StubContent = namedtuple("_StubContent", "text")
_StubUsage = namedtuple("_StubUsage", "input_tokens output_tokens")


class _StubMessage:
    def __init__(self, text: str, in_tok: int = 100, out_tok: int = 50) -> None:
        self.content = [_StubContent(text=text)]
        self.usage = _StubUsage(input_tokens=in_tok, output_tokens=out_tok)


class _StubClient:
    """Anthropic-shaped client that replays a scripted list of responses.

    Per call to `messages.create` we pop the next scripted text off the
    queue. If we run out of scripted responses, raise so the test fails
    loudly rather than silently looping forever.
    """

    def __init__(self, scripted: list[str]):
        self._queue = list(scripted)
        self.calls: list[tuple[str, str]] = []  # (system, user)

        class _Messages:
            def __init__(self, outer: "_StubClient") -> None:
                self._outer = outer

            def create(self, *, model: str, max_tokens: int, system: str,
                       messages: list[dict]) -> _StubMessage:
                if not self._outer._queue:
                    raise RuntimeError(
                        "deep_search test: scripted response queue exhausted; "
                        "loop made more LLM calls than expected"
                    )
                user = messages[0]["content"]
                self._outer.calls.append((system, user))
                return _StubMessage(self._outer._queue.pop(0))

        self.messages = _Messages(self)


# ---------------------------------------------------------------------------
# Fixture corpus — a single doc about Materialized Lake View (MLV) so we can
# trigger the canonical fb37 name-aliasing case (question says MVE, corpus
# says MLV).
# ---------------------------------------------------------------------------

_MLV_CONTENT = (
    "# Materialized Lake View (MLV)\n\n"
    "Materialized Lake Views (MLV) are a Fabric feature that materialises a "
    "lake-table query result into a separate table for fast read access.\n\n"
    "## Default action\n\n"
    "When the target lake table already exists, the default action is to "
    "OVERWRITE its contents. Use the --append flag to add rows instead. "
    "MLV refresh is incremental when supported by the source.\n"
)


def _build_fixture(root: Path) -> Path:
    return build_corpus(
        root, {"mlv.md": _MLV_CONTENT}, mode="local",
    )


# ---------------------------------------------------------------------------
# Guarantees
# ---------------------------------------------------------------------------


def _check_plan_search_answer(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search

    client = _StubClient([
        "MLV default action",                                                   # planner
        '{"action": "answer", "answer": "The default is overwrite."}',         # refiner hop 2
    ])
    result = deep_search(
        km, "What is the default action of MLV?", client=client,
        max_hops=4, top_k=3,
    )
    if result.answer != "The default is overwrite.":
        print(f"[deep_search] FAIL guarantee 1: answer={result.answer!r}",
              file=sys.stderr)
        return 1
    kinds = [h.kind for h in result.hops]
    if kinds[0] != "plan" or "search" not in kinds or "decide_answer" not in kinds:
        print(f"[deep_search] FAIL guarantee 1: hop kinds={kinds}",
              file=sys.stderr)
        return 1
    if result.cost_usd <= 0:
        print(f"[deep_search] FAIL guarantee 1: cost not accumulated "
              f"(cost={result.cost_usd})", file=sys.stderr)
        return 1
    print("[deep_search] guarantee 1 (plan → search → answer) OK", file=sys.stderr)
    return 0


def _check_multi_hop(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search

    client = _StubClient([
        "MLV overview",                                                          # planner
        '{"action": "search", "query": "MLV default behaviour"}',                # refiner hop 2
        '{"action": "answer", "answer": "Overwrite when target exists."}',       # refiner hop 3
    ])
    result = deep_search(
        km, "What does MLV do by default?", client=client,
        max_hops=4, top_k=3,
    )
    if result.answer != "Overwrite when target exists.":
        print(f"[deep_search] FAIL guarantee 2: answer={result.answer!r}",
              file=sys.stderr)
        return 1
    n_search = sum(1 for h in result.hops if h.kind == "search")
    if n_search != 2:
        print(f"[deep_search] FAIL guarantee 2: expected 2 search hops, "
              f"got {n_search}; kinds={[h.kind for h in result.hops]}",
              file=sys.stderr)
        return 1
    print("[deep_search] guarantee 2 (multi-hop search) OK", file=sys.stderr)
    return 0


def _check_give_up(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search
    from resonance_lattice.deep_search.prompts import GIVE_UP_ANSWER

    client = _StubClient([
        "completely unrelated query",                                            # planner
        '{"action": "give_up"}',                                                 # refiner hop 2
    ])
    result = deep_search(
        km, "How do I configure F4096 SKU pricing?", client=client,
        max_hops=4, top_k=3,
    )
    # The give_up answer would have been GIVE_UP_ANSWER; namecheck may
    # then prepend a refusal directive because F4096 + SKU aren't in the
    # MLV corpus. Both shapes count as a successful give-up.
    if GIVE_UP_ANSWER not in result.answer:
        print(f"[deep_search] FAIL guarantee 3: GIVE_UP_ANSWER not in "
              f"answer={result.answer!r}", file=sys.stderr)
        return 1
    if not any(h.kind == "decide_give_up" for h in result.hops):
        print(f"[deep_search] FAIL guarantee 3: no decide_give_up hop",
              file=sys.stderr)
        return 1
    print("[deep_search] guarantee 3 (give-up path) OK", file=sys.stderr)
    return 0


def _check_max_hops_synth(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search

    # Refiner picks search every time → loop exhausts hops → synth is called.
    client = _StubClient([
        "MLV q1",                                                                # planner
        '{"action": "search", "query": "MLV q2"}',                               # hop 2
        '{"action": "search", "query": "MLV q3"}',                               # hop 3
        '{"action": "search", "query": "MLV q4"}',                               # hop 4
        "Synthesised answer about MLV from accumulated evidence.",               # synth
    ])
    result = deep_search(
        km, "Tell me everything about MLV.", client=client,
        max_hops=4, top_k=3,
    )
    if "Synthesised answer about MLV" not in result.answer:
        print(f"[deep_search] FAIL guarantee 4: answer={result.answer!r}",
              file=sys.stderr)
        return 1
    if not any(h.kind == "synth_after_max_hops" for h in result.hops):
        print(f"[deep_search] FAIL guarantee 4: synth_after_max_hops hop "
              f"missing; kinds={[h.kind for h in result.hops]}",
              file=sys.stderr)
        return 1
    print("[deep_search] guarantee 4 (max-hops synth) OK", file=sys.stderr)
    return 0


def _check_name_check_union(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search
    from resonance_lattice.deep_search.prompts import NAME_MISMATCH_ANSWER

    # MVE is not in the MLV corpus — every hop's evidence will mention
    # MLV but never MVE. After the loop, namecheck fails on MVE.
    client = _StubClient([
        "MVE default action",                                                    # planner
        '{"action": "answer", "answer": "The default action is overwrite."}',    # refiner
    ])
    result = deep_search(
        km, "What is the default action of MVE?", client=client,
        max_hops=4, top_k=3,
    )
    if "MVE" not in result.name_check_missing:
        print(f"[deep_search] FAIL guarantee 5: missing tokens should "
              f"include MVE; got {result.name_check_missing}",
              file=sys.stderr)
        return 1
    if "rlat-namecheck: missing" not in result.answer:
        print(f"[deep_search] FAIL guarantee 5: refusal directive not "
              f"prepended to answer:\n{result.answer}", file=sys.stderr)
        return 1
    if "Name verification failed" not in result.answer:
        print(f"[deep_search] FAIL guarantee 5: human-readable directive "
              f"missing; answer:\n{result.answer}", file=sys.stderr)
        return 1
    if result.strict_names_aborted:
        print(f"[deep_search] FAIL guarantee 5: strict_names_aborted true "
              f"under default (non-strict) mode", file=sys.stderr)
        return 1

    # Strict mode: same conditions, but answer is replaced entirely.
    client = _StubClient([
        "MVE default action",
        '{"action": "answer", "answer": "Should not appear in result."}',
    ])
    result = deep_search(
        km, "What is the default action of MVE?", client=client,
        max_hops=4, top_k=3, strict_names=True,
    )
    if not result.strict_names_aborted:
        print(f"[deep_search] FAIL guarantee 5: strict_names_aborted not set",
              file=sys.stderr)
        return 1
    if result.answer != NAME_MISMATCH_ANSWER:
        print(f"[deep_search] FAIL guarantee 5: strict answer should equal "
              f"NAME_MISMATCH_ANSWER; got {result.answer!r}", file=sys.stderr)
        return 1
    print("[deep_search] guarantee 5 (name-check union + strict) OK",
          file=sys.stderr)
    return 0


def _check_token_accounting(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search

    client = _StubClient([
        "MLV default",
        '{"action": "answer", "answer": "Overwrite."}',
    ])
    result = deep_search(
        km, "What does MLV default to?", client=client,
        max_hops=4, top_k=3,
    )
    # Two LLM calls scripted (planner + refiner). Each stub returns
    # in_tok=100, out_tok=50 → totals 200 / 100 / cost > 0.
    if result.input_tokens != 200 or result.output_tokens != 100:
        print(f"[deep_search] FAIL guarantee 6: tokens "
              f"in={result.input_tokens} out={result.output_tokens}, "
              f"expected 200/100", file=sys.stderr)
        return 1
    expected_cost = (200 / 1e6) * 3.0 + (100 / 1e6) * 15.0
    if abs(result.cost_usd - expected_cost) > 1e-9:
        print(f"[deep_search] FAIL guarantee 6: cost={result.cost_usd}, "
              f"expected {expected_cost}", file=sys.stderr)
        return 1
    print("[deep_search] guarantee 6 (token accounting) OK", file=sys.stderr)
    return 0


def _check_insight_aware_retrieval(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store import archive

    from ._testutil import make_corpus_claim

    # Isolated corpus — promoting an insight mutates the archive.
    km2 = _build_fixture(km.parent / "insight_aware")
    c0 = archive.read(km2)
    src_ids = [c.passage_id for c in c0.registry]
    src_hashes = [c.content_hash for c in c0.registry]
    insight = make_corpus_claim(
        "MLV refresh is incremental when the source supports it; otherwise "
        "the materialized lake view performs a full overwrite refresh.",
        src_ids[:1], src_hashes[:1], state="active",
    )
    band = Encoder().encode([insight.content]).astype("float32")
    archive.write_insight_layer_in_place(km2, [insight], band)

    client = _StubClient([
        "MLV refresh behaviour",
        '{"action": "answer", "answer": "Incremental when supported."}',
    ])
    result = deep_search(
        km2, "How does MLV refresh work?", client=client,
        max_hops=4, top_k=5,
    )
    search_hops = [h for h in result.hops if h.kind == "search"]
    if not any((h.n_insights or 0) > 0 for h in search_hops):
        print(f"[deep_search] FAIL guarantee 7: no insight retrieved "
              f"(n_insights per hop = {[h.n_insights for h in search_hops]})",
              file=sys.stderr)
        return 1
    # The refiner must have SEEN the insight, labelled as earned insight.
    refiner_prompts = [user for _system, user in client.calls]
    if not any("earned insight" in p for p in refiner_prompts):
        print("[deep_search] FAIL guarantee 7: insight evidence absent from "
              "the refiner prompt", file=sys.stderr)
        return 1
    # The result must carry the engaged insight id (attribution substrate).
    if insight.claim_id not in result.insight_ids:
        print(f"[deep_search] FAIL guarantee 7: insight_id not captured in "
              f"result.insight_ids ({result.insight_ids})", file=sys.stderr)
        return 1
    print("[deep_search] guarantee 7 (insight-aware retrieval) OK",
          file=sys.stderr)
    return 0


def _check_parse_fail_recovery(km: Path) -> int:
    from resonance_lattice.deep_search import deep_search

    # Refiner emits unparseable output → the loop must recover via a
    # synth call, not surface raw model text as the answer.
    client = _StubClient([
        "MLV default action",                            # planner
        "this is not valid json at all",                 # refiner hop 2 — unparseable
        "Synthesised recovery answer about MLV.",         # synth recovery
    ])
    result = deep_search(
        km, "What is the default action of MLV?", client=client,
        max_hops=4, top_k=3,
    )
    if "Synthesised recovery answer" not in result.answer:
        print(f"[deep_search] FAIL guarantee 8: answer not synth-recovered "
              f"({result.answer!r})", file=sys.stderr)
        return 1
    kinds = [h.kind for h in result.hops]
    if "parse_failed" not in kinds:
        print(f"[deep_search] FAIL guarantee 8: parse_failed hop not "
              f"recorded; kinds={kinds}", file=sys.stderr)
        return 1
    if "synth_after_parse_fail" not in kinds:
        print(f"[deep_search] FAIL guarantee 8: synth_after_parse_fail hop "
              f"missing; kinds={kinds}", file=sys.stderr)
        return 1
    print("[deep_search] guarantee 8 (parse-failure recovery) OK",
          file=sys.stderr)
    return 0


def _check_cost_cap_halts_loop(km: Path) -> int:
    """A `CostMeter` with a tight cap halts the loop before the refiner
    can run. The planner call accumulates ~$0.315 of metered spend
    (100K input + 1K output Sonnet tokens); with a $0.20 cap the
    pre-refiner check fires at the next hop and the loop returns a
    partial-answer-from-evidence (since hop 2 retrieved passages
    before the cap-check). Mirrors `reverification` g8."""
    from resonance_lattice.deep_search import deep_search
    from resonance_lattice._pricing import CostMeter

    class _ExpensiveMessage:
        def __init__(self, text: str) -> None:
            self.content = [_StubContent(text=text)]
            # ~$0.315 of Sonnet spend per call → first call alone crosses
            # the $0.20 cap.
            self.usage = _StubUsage(input_tokens=100_000, output_tokens=1_000)

    class _ExpensiveClient:
        def __init__(self, scripted: list[str]) -> None:
            self._queue = list(scripted)
            self.calls: list[tuple[str, str]] = []

            class _Messages:
                def __init__(self, outer):
                    self._outer = outer

                def create(self, *, model, max_tokens, system, messages):
                    if not self._outer._queue:
                        raise RuntimeError(
                            "deep_search test: scripted queue exhausted"
                        )
                    self._outer.calls.append((system, messages[0]["content"]))
                    return _ExpensiveMessage(self._outer._queue.pop(0))

            self.messages = _Messages(self)

    client = _ExpensiveClient(["MLV default action"])  # only the planner runs
    meter = CostMeter(cap_usd=0.20)

    result = deep_search(
        km, "What is the default action of MLV?", client=client,
        max_hops=4, top_k=3, meter=meter,
    )
    kinds = [h.kind for h in result.hops]
    ok = (
        len(client.calls) == 1  # only the planner ran; refiner was capped
        and "cost_cap_crossed" in kinds
        and meter.has_exceeded_cap()
    )
    if not ok:
        print(f"[deep_search] FAIL cost-cap: calls={len(client.calls)} "
              f"kinds={kinds} cost={meter.cost_so_far():.4f}", file=sys.stderr)
        return 1
    # Pre-flight cap on a second `deep_search` call sharing the same
    # meter — bails before the planner even runs (the `rlat probe`
    # session-wide pattern).
    client2 = _ExpensiveClient([])  # no calls expected
    result2 = deep_search(
        km, "Another question.", client=client2,
        max_hops=4, top_k=3, meter=meter,
    )
    ok = (
        len(client2.calls) == 0
        and "cost_cap_crossed" in [h.kind for h in result2.hops]
        and result2.answer.startswith("I cannot produce an answer")
    )
    if not ok:
        print(f"[deep_search] FAIL cost-cap pre-flight: calls={len(client2.calls)} "
              f"answer={result2.answer!r}", file=sys.stderr)
        return 1
    print("[deep_search] cost-cap halts loop + pre-flights subsequent calls OK",
          file=sys.stderr)
    return 0


def run() -> int:
    with tempfile.TemporaryDirectory() as d:
        km = _build_fixture(Path(d))
        for fn in (
            _check_plan_search_answer,
            _check_multi_hop,
            _check_give_up,
            _check_max_hops_synth,
            _check_name_check_union,
            _check_token_accounting,
            _check_insight_aware_retrieval,
            _check_parse_fail_recovery,
            _check_cost_cap_halts_loop,
        ):
            if fn(km) != 0:
                return 1
    print("[deep_search] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
