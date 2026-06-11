"""Shared LLM client seam for memory-side LLM callers — re-exported.

A `Callable[[str, list[dict], int], LLMResponse]`: the caller passes a
system prompt, the user-role messages, and a max-tokens cap; the client
returns the response text plus its in/out token counts.

The canonical definitions live in the package-root `_anthropic` module —
the shared Anthropic client home that also *produces* these types via
`default_client`. This module re-exports them so the memory-side callers
(the distillation / arrow modules, the agent-harness primitives
`what_next` / `decompose`, and the confidence raiser) import from one
place and share the **same** type object with the cli-side callers — one
type union, not a fanout.
"""

from __future__ import annotations

from .._anthropic import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse"]
