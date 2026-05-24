"""Shared LLM-judge helper for store-layer modules.

`reverification` and `faithfulness` both make a single structured-output
LLM call and need the same tolerant JSON parse — Claude sometimes wraps the
object in a ```json fence or trails prose after it. One owner.
"""

from __future__ import annotations

import json


def judge_json(
    client, model: str, system: str, user: str, *, max_tokens: int = 1000,
) -> dict:
    """One round-trip to an Anthropic-shaped client. Returns parsed JSON,
    or `{"_parse_error": <snippet>}` when the response isn't parseable.

    `client` only needs `client.messages.create(...)` returning an object
    whose `.content[0].text` is the model's text — inject a stub for tests.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
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
