"""Anthropic Sonnet 4.6 pricing — one-place token-cost arithmetic.

Lifted out of duplications across `deep_search.loop` and the bench harness
(`benchmarks/user_bench/hallucination/{run,judge_only,rejudge_relaxed}.py`).
These sites hardcoded `$3 in / $15 out per million tokens` against
`claude-sonnet-4-6`; one place now.

`SONNET_MODEL` is the canonical model id. `cost_usd(in, out)` is the
arithmetic. If pricing changes, this is the only edit.
"""

from __future__ import annotations


SONNET_MODEL = "claude-sonnet-4-6"

# Anthropic published Sonnet 4.6 pricing (USD per million tokens). If you
# change these, also update any bench methodology docs that quote
# per-question $/q estimates.
_INPUT_USD_PER_MTOK = 3.0
_OUTPUT_USD_PER_MTOK = 15.0


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """USD spend for a Sonnet 4.6 call given its token usage."""
    return (
        input_tokens * _INPUT_USD_PER_MTOK / 1_000_000
        + output_tokens * _OUTPUT_USD_PER_MTOK / 1_000_000
    )


class CostMeter:
    """Per-session LLM cost tracker with an optional USD cap.

    A long-running command (`rlat reverify`, `rlat deep-search`) builds
    one meter per invocation and updates it after each LLM call.
    `has_exceeded_cap()` is the pre-flight check that lets the loop
    stop before issuing the next call once the cap has been crossed.
    The meter is honest about what it knows: it tracks *observed*
    token usage post-call, so a pre-flight check can't bound the next
    call's cost — but a cap-at-1.5x-budget contract reliably stops a
    runaway batch within one call of crossing the budget.

    Single-threaded: `add` is non-atomic. Callers must not share a
    meter across threads.
    """

    def __init__(self, cap_usd: float | None = None):
        self.cap_usd = cap_usd
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Record one LLM call's observed token usage."""
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)

    def cost_so_far(self) -> float:
        """Cumulative USD across recorded calls."""
        return cost_usd(self.input_tokens, self.output_tokens)

    def has_exceeded_cap(self) -> bool:
        """`True` once the recorded cost has crossed the cap, so the
        next call should not fire. No-cap meters always return False."""
        if self.cap_usd is None:
            return False
        return self.cost_so_far() >= self.cap_usd
