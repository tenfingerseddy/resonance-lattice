"""Text utilities — sentence splitting.

Regex-based splitter used by `store/chunker.py` (paragraph→sentence chunk
splits when a paragraph exceeds `max_chars`). Greedy on `[.!?]` followed
by whitespace; same heuristic the v2.0 chunker shipped with, lifted to
its own module so future text-handling code can reuse one regex instead
of cloning it inline.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def iter_sentence_spans(text: str) -> list[tuple[int, int]]:
    """`(offset, length)` per sentence, offsets indexing the input.

    Output covers the input non-overlappingly. A trailing sentence without
    terminal punctuation is emitted as the final span.
    """
    out: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END.finditer(text):
        out.append((pos, m.end() - pos))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text) - pos))
    return out


def split_sentences(text: str) -> list[str]:
    """Sentences in source order. Whitespace between sentences is dropped."""
    return [text[off:off + ln].strip() for off, ln in iter_sentence_spans(text)
            if text[off:off + ln].strip()]
