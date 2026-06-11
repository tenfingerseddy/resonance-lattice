"""Shared Anthropic client helpers — API-key discovery and a thin SDK wrapper.

Package-root (beside `_pricing.py`) so both `cli/*` and `memory/*` consumers
reach it without a cross-package import. Five commands depend on these helpers:
deep-search, intent, probe, reverify, and the SessionEnd capture hook
(assess-session was removed in 8ab08e13).
"""

from __future__ import annotations

import collections
import os
from typing import Callable

from ._pricing import SONNET_MODEL as DEFAULT_MODEL

LLMResponse = collections.namedtuple("LLMResponse", "text input_tokens output_tokens")
LLMClient = Callable[[str, list[dict], int], LLMResponse]


def discover_api_key() -> str | None:
    indirected = os.environ.get("RLAT_LLM_API_KEY_ENV")
    if indirected and (val := os.environ.get(indirected)):
        return val
    # CLAUDE_API_2 is Kane's active env-var slot for the harness LLM key;
    # CLAUDE_API + ANTHROPIC_API_KEY remain supported for legacy install
    # docs and for kaggle_secrets parity.
    for name in ("CLAUDE_API_2", "CLAUDE_API", "ANTHROPIC_API_KEY"):
        if (val := os.environ.get(name)):
            return val
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    client = UserSecretsClient()
    for name in ("CLAUDE_API_2", "CLAUDE_API", "ANTHROPIC_API_KEY"):
        try:
            return client.get_secret(name)
        except Exception:
            continue
    return None


def api_key_or_error(api_key: str | None = None) -> str:
    key = api_key or discover_api_key()
    if not key:
        raise RuntimeError(
            "An LLM API key is required for this command.\n\n"
            "Set an environment variable with your Anthropic API key:\n"
            "  export CLAUDE_API=sk-ant-...\n\n"
            "rlat checks CLAUDE_API_2, then CLAUDE_API, then ANTHROPIC_API_KEY.\n"
            "To keep your own variable name, point rlat at it:\n"
            "  export MY_KEY=sk-ant-...\n"
            "  export RLAT_LLM_API_KEY_ENV=MY_KEY"
        )
    return key


def default_client(
    api_key: str, model: str = DEFAULT_MODEL, *, timeout: float | None = None,
) -> LLMClient:
    """Wrap the `anthropic` SDK in our (system, messages, max_tokens) -> response shape.

    `timeout` bounds HTTP round-trip time (httpx-level). Hot-path callers
    (the SessionEnd capture hook) pass a tight value so a hung endpoint
    can't pin the user's prompt close; bulk callers leave it `None` for
    the SDK default.

    Concurrency-safe: anthropic.Anthropic is thread-safe per SDK docs (one
    HTTP client per process). A caller's ThreadPool may call this callable
    from N worker threads simultaneously.
    """
    import anthropic
    kwargs: dict = {"api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = timeout
    sdk = anthropic.Anthropic(**kwargs)

    def call(system: str, messages: list[dict], max_tokens: int) -> LLMResponse:
        resp = sdk.messages.create(
            model=model, system=system, messages=messages, max_tokens=max_tokens,
        )
        text = resp.content[0].text
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
    return call
