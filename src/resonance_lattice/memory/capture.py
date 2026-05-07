"""Stop-hook capture path.

Read a Claude Code session transcript, apply the heuristic gate, scrub
through Layer-1 redaction, write a row tagged
`["factual", "workspace:<cwd-hash>"]`. The downstream distil pipeline
later promotes captured rows to `prefer`/`avoid` polarity based on
semantic analysis; until then captured rows surface only when their
own cosine match clears the recall thresholds in §0.6.

Spec: `.claude/plans/fabric-agent-flat-memory.md` §5.2 + §13.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ._common import stable_hash, workspace_tag_for_cwd
from .redaction import RedactionEvent, Redactor
from .store import Memory

# Encoder max-seq is 8192 tokens (~32KB UTF-8); cap captured text well below
# so a runaway transcript can't silently truncate at encode time. Sessions
# that exceed this lose tail content but never lose retrieval correctness.
_MAX_CAPTURED_CHARS = 24_000

# Sub-MVP heuristic-gate threshold for "trivial single-exchange" sessions
# ("ok", "thanks", "continue"). Not in §5.2; deliberately conservative.
_TRIVIAL_USER_CHARS = 30


# ---------------------------------------------------------------------------
# Transcript shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation in an assistant turn. We don't model parameters
    in detail — only the path that the redactor needs to denylist-check."""

    name: str
    path: str | None = None
    content: str = ""


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Transcript:
    """Minimal Claude Code session transcript shape.

    Sub-MVP doesn't bind to any Claude Code-specific JSON schema; the Stop
    hook shim that wraps this module is responsible for parsing the live
    payload into this dataclass. That keeps the capture pipeline testable
    without a live Claude Code session.
    """

    session_id: str
    messages: Sequence[Message]
    cwd: str  # absolute working directory at session start


# ---------------------------------------------------------------------------
# Heuristic gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    min_assistant_chars: int = 200
    require_tool_use: bool = True
    trivial_user_max_chars: int = _TRIVIAL_USER_CHARS


@dataclass(frozen=True)
class GateResult:
    passed: bool
    skip_reason: str | None = None  # None when passed

    @classmethod
    def skip(cls, reason: str) -> "GateResult":
        return cls(passed=False, skip_reason=reason)


def evaluate_gate(transcript: Transcript, config: GateConfig | None = None) -> GateResult:
    """Cheap, no-LLM heuristic per §5.2: tools / volume / triviality."""
    cfg = config or GateConfig()
    assistant_msgs = [m for m in transcript.messages if m.role == "assistant"]
    user_msgs = [m for m in transcript.messages if m.role == "user"]

    total_assistant_chars = sum(len(m.content) for m in assistant_msgs)
    if total_assistant_chars < cfg.min_assistant_chars:
        return GateResult.skip(
            f"assistant content {total_assistant_chars} chars < {cfg.min_assistant_chars}"
        )

    if cfg.require_tool_use and not any(m.tool_calls for m in assistant_msgs):
        return GateResult.skip("no tool use in session")

    total_user_chars = sum(len(m.content) for m in user_msgs)
    if (
        len(user_msgs) <= 1
        and total_user_chars <= cfg.trivial_user_max_chars
    ):
        return GateResult.skip(
            f"trivial single-exchange user content ({total_user_chars} chars)"
        )

    return GateResult(passed=True)


# ---------------------------------------------------------------------------
# Capture pipeline
# ---------------------------------------------------------------------------


def transcript_hash(transcript: Transcript) -> str:
    """Stable SHA-256 over session_id + every message + tool call.

    Used as the row's `transcript_hash` for §18.6 same-transcript dedup at
    distil time. Implementation delegates to `_common.stable_hash` so the
    daemon recall path (#88) shares the same hashing convention.
    """
    parts: list[bytes | str] = [transcript.session_id]
    for m in transcript.messages:
        parts.append(m.role)
        parts.append(m.content)
        for tc in m.tool_calls:
            parts.append(tc.name)
            parts.append(tc.path or "")
            parts.append(tc.content)
    return stable_hash(parts)


@dataclass(frozen=True)
class CaptureResult:
    row_id: str | None
    skip_reason: str | None
    redactions: int  # count of pattern hits across text + tool calls


def _scrub_transcript(
    transcript: Transcript, redactor: Redactor
) -> tuple[str, list[RedactionEvent]]:
    """Apply Layer-1 redaction to every assistant message + every tool call.

    Returns `(scrubbed_text, events)` — events are *buffered*, not logged,
    so the caller can correlate them with the row_id once the row is
    written (§6.4 audit-log contract).
    """
    pieces: list[str] = []
    events_buffer: list[RedactionEvent] = []
    for msg in transcript.messages:
        if msg.role != "assistant":
            continue
        scrubbed_text, events = redactor.scrub(msg.content)
        events_buffer.extend(events)
        pieces.append(scrubbed_text)
        # Tool-call payloads land in the audit log but never in the
        # captured text — they don't generalise as lessons. We always run
        # the pattern scrub on the content (even when path is None — a
        # `bash` call with inline `export AWS_KEY=...` is the canonical
        # way Layer-1 leaks bypass a denylist-only filter); the denylist
        # branch only fires when there *is* a path to glob-match.
        for tc in msg.tool_calls:
            if tc.path is not None:
                _, tc_events = redactor.scrub_tool_call(tc.path, tc.content)
            else:
                _, tc_events = redactor.scrub(tc.content)
            events_buffer.extend(tc_events)
    return "\n\n".join(pieces).strip(), events_buffer


def capture(
    transcript: Transcript,
    *,
    store: Memory,
    redactor: Redactor,
    gate: GateConfig | None = None,
) -> CaptureResult:
    """Run the full Stop-hook pipeline.

    Always returns a `CaptureResult`; never raises. The Stop hook fires
    on every session end, so a memory failure (encoder load, lock
    timeout, disk full, schema corruption) must never block the user's
    prompt close. Failures land in `skip_reason` with a short error
    type prefix; the audit log captures the exception class for ops to
    triage.

    Polarity is `["factual", "workspace:<hash>"]` — a placeholder the
    distiller replaces with `["prefer"|"avoid", ...]` once it semantically
    analyses the captured content.
    """
    gate_result = evaluate_gate(transcript, gate)
    if not gate_result.passed:
        return CaptureResult(
            row_id=None, skip_reason=gate_result.skip_reason, redactions=0
        )

    try:
        text, events = _scrub_transcript(transcript, redactor)
        redactions = sum(e.matches for e in events)
        if not text:
            # Buffered events are still worth logging even on a no-write
            # path so a session of pure-secrets isn't a silent gap in the
            # audit trail; correlate against transcript_hash.
            if events:
                redactor.log_events(events, row_id=transcript_hash(transcript))
            return CaptureResult(
                row_id=None,
                skip_reason="empty assistant content after scrub",
                redactions=redactions,
            )
        if len(text) > _MAX_CAPTURED_CHARS:
            # Tail-truncate so the encoder never sees more than its max-seq;
            # silent truncation at encode would cost retrieval correctness on
            # whatever falls past 8192 tokens.
            text = text[:_MAX_CAPTURED_CHARS]

        polarity = ["factual", workspace_tag_for_cwd(transcript.cwd)]
        row_id = store.add_row(
            text=text,
            polarity=polarity,
            transcript_hash=transcript_hash(transcript),
        )
        # Log the buffered events with the now-known row_id for audit
        # correlation (§6.4).
        if events:
            redactor.log_events(events, row_id=row_id)
        return CaptureResult(row_id=row_id, skip_reason=None, redactions=redactions)
    except Exception as exc:
        # Fail-open. Skip-reason carries the exception *type* only — never
        # the message, since exceptions can attach paths, polarity strings,
        # or row text that was the very thing the redactor was trying to
        # protect (a ValueError from store.add_row may quote the offending
        # row text verbatim).
        return CaptureResult(
            row_id=None,
            skip_reason=f"capture failed: {type(exc).__name__}",
            redactions=0,
        )
