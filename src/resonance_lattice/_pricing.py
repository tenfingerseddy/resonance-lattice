"""Anthropic Sonnet 4.6 pricing — one-place token-cost arithmetic.

Lifted out of duplications across `optimise.synth_queries`,
`deep_search.loop`, and the bench harness (`benchmarks/user_bench/
hallucination/{run,judge_only,rejudge_relaxed}.py`). All five sites
hardcoded `$3 in / $15 out per million tokens` against
`claude-sonnet-4-6`; one place now.

`SONNET_MODEL` is the canonical model id. `cost_usd(in, out)` is the
arithmetic. If pricing changes, this is the only edit.
"""

from __future__ import annotations


SONNET_MODEL = "claude-sonnet-4-6"

# Anthropic published Sonnet 4.6 pricing (USD per million tokens). If you
# change these, also update docs/internal/OPTIMISE.md and any bench
# methodology docs that quote per-question $/q estimates.
_INPUT_USD_PER_MTOK = 3.0
_OUTPUT_USD_PER_MTOK = 15.0


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """USD spend for a Sonnet 4.6 call given its token usage."""
    return (
        input_tokens * _INPUT_USD_PER_MTOK / 1_000_000
        + output_tokens * _OUTPUT_USD_PER_MTOK / 1_000_000
    )
