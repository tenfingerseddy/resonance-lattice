"""Passage chunker — `passage_v1`.

Default chunking strategy for v2.0 builds. Splits a text into
`(char_offset, char_length)` ranges sized between `min_chars` and `max_chars`.
Strategy:

  1. Split on paragraph boundaries (`\\n\\n+`).
  2. If a paragraph fits in `[min_chars, max_chars]`, take it whole.
  3. If it's too short, merge with the next paragraph until the merged size
     reaches `min_chars` (or paragraphs run out — final undersized chunk is
     emitted as-is rather than dropped, so no source range gets silently
     dropped from the registry).
  4. If it's too long, split at sentence boundaries (`. ` / `? ` / `! `,
     newline-terminated). If a sentence still exceeds `max_chars`, hard-split
     at `max_chars`.

Audit 05 (chunking benchmark) is the validator. Until it locks, the defaults
match what `BuildConfig` declares: `min_chars=200`, `max_chars=3200`.

Phase 3 deliverable. Replaces the v0.11 `passage_chunker` (which had the
same name but multiple knob flags). v2 collapses to one strategy + two
size constants.
"""

from __future__ import annotations

import re

# Sentence-end heuristic: punctuation followed by whitespace. Greedy enough
# for prose / docs / code-comments without burning a sentence segmenter dep.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Paragraph break: 2+ newlines (with optional intra-line whitespace).
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """Return `(offset, length)` per paragraph, paragraphs separated by `\\n\\n+`.

    The offsets index into the original text — paragraph splitting preserves
    char positions so downstream `PassageCoord` records the exact source
    range.
    """
    out: list[tuple[int, int]] = []
    pos = 0
    for m in _PARAGRAPH_BREAK.finditer(text):
        if m.start() > pos:
            out.append((pos, m.start() - pos))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text) - pos))
    return out


def _split_oversize(text: str, base_offset: int, max_chars: int) -> list[tuple[int, int]]:
    """Split a `(base_offset, len(text))` paragraph that's too long.

    First pass: sentence boundaries inside the paragraph; merge sentences into
    chunks ≤ `max_chars`. Any single sentence above `max_chars` is hard-split.
    Returns absolute-offset ranges.
    """
    sentences: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END.finditer(text):
        sentences.append((pos, m.end() - pos))
        pos = m.end()
    if pos < len(text):
        sentences.append((pos, len(text) - pos))

    out: list[tuple[int, int]] = []
    cur_start = -1
    cur_end = -1
    for s_off, s_len in sentences:
        if s_len > max_chars:
            # Flush any merged-so-far sentence then hard-split this one.
            if cur_start >= 0:
                out.append((base_offset + cur_start, cur_end - cur_start))
                cur_start = cur_end = -1
            for hard_off in range(s_off, s_off + s_len, max_chars):
                hard_len = min(max_chars, s_off + s_len - hard_off)
                out.append((base_offset + hard_off, hard_len))
            continue
        if cur_start < 0:
            cur_start, cur_end = s_off, s_off + s_len
            continue
        if (s_off + s_len) - cur_start <= max_chars:
            cur_end = s_off + s_len
        else:
            out.append((base_offset + cur_start, cur_end - cur_start))
            cur_start, cur_end = s_off, s_off + s_len
    if cur_start >= 0:
        out.append((base_offset + cur_start, cur_end - cur_start))
    return out


def chunk_text(
    text: str,
    min_chars: int = 200,
    max_chars: int = 3200,
) -> list[tuple[int, int]]:
    """Split `text` into `(char_offset, char_length)` chunks.

    Returns chunks in source order. Output covers the input non-overlappingly;
    short tails are emitted as-is rather than dropped (otherwise a small
    file's only paragraph would vanish from the registry).
    """
    # Whitespace-only files have no semantic content — emitting an
    # all-whitespace chunk wastes encoder compute on something that
    # produces a zero-information embedding.
    if not text or not text.strip():
        return []

    paragraphs = _paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[tuple[int, int]] = []
    pending_start = -1
    pending_end = -1

    def _flush_pending() -> None:
        nonlocal pending_start, pending_end
        if pending_start >= 0:
            chunks.append((pending_start, pending_end - pending_start))
            pending_start = pending_end = -1

    for off, ln in paragraphs:
        if ln > max_chars:
            _flush_pending()
            chunks.extend(_split_oversize(text[off:off + ln], off, max_chars))
            continue
        if ln >= min_chars:
            _flush_pending()
            chunks.append((off, ln))
            continue
        # Short paragraph: merge into pending until we reach min_chars.
        if pending_start < 0:
            pending_start, pending_end = off, off + ln
        elif (off + ln) - pending_start <= max_chars:
            pending_end = off + ln
        else:
            _flush_pending()
            pending_start, pending_end = off, off + ln
        if pending_end - pending_start >= min_chars:
            _flush_pending()

    _flush_pending()
    return chunks
