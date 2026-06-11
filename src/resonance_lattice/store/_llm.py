"""Shared LLM-judge helper for store-layer modules.

`reverification` and `faithfulness` both make a single structured-output
LLM call and need the same tolerant JSON parse — Claude sometimes wraps the
object in a ```json fence or trails prose after it. One owner.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._pricing import CostMeter


def judge_json(
    client, model: str, system: str, user: str, *, max_tokens: int = 1000,
    meter: "CostMeter | None" = None, temperature: float | None = None,
) -> dict:
    """One round-trip to an Anthropic-shaped client. Returns parsed JSON,
    or `{"_parse_error": <snippet>}` when the response isn't parseable.

    `client` only needs `client.messages.create(...)` returning an object
    whose `.content[0].text` is the model's text — inject a stub for tests.

    `meter` — when supplied, records observed token usage from
    `resp.usage` so a per-session cap can be enforced upstream. A
    response without a `usage` attribute (a test stub, or a future SDK
    that names usage differently) is treated as zero — the meter never
    raises on a missing field.

    `temperature` — when supplied, forwarded to the model. A classifying
    judge (stance, faithfulness) wants `0.0` for a deterministic,
    reproducible verdict; left `None` it keeps the SDK default (back-compat
    for callers that never set it). A test stub's `create` may not accept
    the kwarg, so it is only passed through when not None.
    """
    extra = {} if temperature is None else {"temperature": temperature}
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        **extra,
    )
    if meter is not None:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meter.add(
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"_parse_error": text[:200]}
