"""Stop-hook capture path.

Read a Claude Code session transcript, apply the heuristic gate, scrub
through Layer-1 redaction, write one or more `event` claims tagged
`("factual", "workspace:<cwd-hash>")`. With `client=None` the pipeline
writes a single claim from the full scrubbed text. With `client` set,
the LLM extracts atomic facts and one claim is written per fact;
extractor failure falls back to the single-claim path so a flaky LLM
never degrades the `client=None` behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from ..state.claim import Claim, evolve
from ._common import stable_hash, utcnow_iso, workspace_tag_for_cwd
from ._llm import LLMClient
from .claim_store import ExperienceClaimStore, new_experience_claim
from .extract import extract_events
from .redaction import RedactionEvent, Redactor

# Encoder max-seq is 8192 tokens (~32KB UTF-8); cap captured text well below
# so a runaway transcript can't silently truncate at encode time. Sessions
# that exceed this keep their tail (recent work) and drop the head — see
# the truncation site in `capture()`.
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

    Used as the claim's `transcript_hash` for §18.6 same-transcript dedup
    at distil time. Implementation delegates to `_common.stable_hash` so the
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
    claim_ids: tuple[str, ...]  # empty tuple ⇔ no write (skip or no-op)
    skip_reason: str | None
    redactions: int  # pattern hits across assistant text + tool calls + user turns
    # World-attribute claims the passive miner landed in (or deduped
    # against) the knowledge model's insight band — empty when no client,
    # no km, or none passed the extractor's four gates.
    attribute_claim_ids: tuple[str, ...] = ()


def _scrub_transcript(
    transcript: Transcript, redactor: Redactor
) -> tuple[str, list[RedactionEvent]]:
    """Apply Layer-1 redaction to every assistant message + every tool call.

    Returns `(scrubbed_text, events)` — events are *buffered*, not logged,
    so the caller can correlate them with the claim_id once the claim is
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


def _scrub_user_turns(
    transcript: Transcript, redactor: Redactor
) -> tuple[str, list[RedactionEvent]]:
    """Scrubbed USER-turn channel — the attribute miner's source.

    World facts live in what the user states in passing ("our tenant is
    EU-only"); the extractor's GATE 2 trusts only user statements, so the
    miner reads this channel while the assistant narrative stays the
    event-capture source. Same Layer-1 scrub, same buffered-events
    contract as `_scrub_transcript`.
    """
    pieces: list[str] = []
    events_buffer: list[RedactionEvent] = []
    for msg in transcript.messages:
        if msg.role != "user":
            continue
        scrubbed, events = redactor.scrub(msg.content)
        events_buffer.extend(events)
        pieces.append(scrubbed)
    return "\n\n".join(pieces).strip(), events_buffer


def _find_dup_in(
    claims: list[Claim], *, text: str, workspace_tag: str
) -> Claim | None:
    """Return the matching event claim in `claims` or None.

    Same-workspace check matters: identical text from two different repos
    (e.g. two checkouts of the same project) is genuinely two events; the
    workspace tag is what tells them apart. Only event-level claims are
    eligible — a captured claim is always written at kind=event, and we
    don't want to bump recurrence on a promoted pattern from below.
    """
    for c in claims:
        if c.kind != "event":
            continue
        if c.content != text:
            continue
        if workspace_tag in c.facts.polarity:
            return c
    return None


def capture(
    transcript: Transcript,
    *,
    store: ExperienceClaimStore,
    redactor: Redactor,
    gate: GateConfig | None = None,
    client: LLMClient | None = None,
    km_path: str | Path | None = None,
) -> CaptureResult:
    """Run the full Stop-hook pipeline.

    Always returns a `CaptureResult`; never raises. The Stop hook fires
    on every session end, so a memory failure (encoder load, lock
    timeout, disk full, schema corruption) must never block the user's
    prompt close. Failures land in `skip_reason` with a short error
    type prefix; the audit log captures the exception class for ops to
    triage.

    Atomic facts land with the polarity the extractor classified
    ("factual" / "prefer" / "avoid"); whole-session fallback rows are
    "factual".

    `client=None` runs the single-claim path. `client` set runs the
    atomic-extraction path; on extractor failure (`None`) the single-claim
    path resumes, preserving the `client=None` behaviour.

    With `client` set AND `km_path` passed explicitly, the PASSIVE
    attribute miner also runs (capture-frontier charter, 2nd source):
    `extract_attributes` gates durable WORLD facts out of the scrubbed
    USER turns and lands them in that knowledge model's insight band —
    facts about the world the model covers, true for anyone it is shared
    with; person-facts are dropped by the extractor's scope gate (privacy
    contract, Kane 2026-06-10). DORMANT by default — see
    `_mine_attributes` for the open findings gating it. Best-effort —
    attribute failures never block the event capture.
    """
    gate_result = evaluate_gate(transcript, gate)
    if not gate_result.passed:
        return CaptureResult(
            claim_ids=(), skip_reason=gate_result.skip_reason, redactions=0
        )

    try:
        text, events = _scrub_transcript(transcript, redactor)
        user_text, user_events = _scrub_user_turns(transcript, redactor)
        events.extend(user_events)
        redactions = sum(e.matches for e in events)
        if not text:
            # Buffered events are still worth logging even on a no-write
            # path so a session of pure-secrets isn't a silent gap in the
            # audit trail; correlate against transcript_hash.
            if events:
                redactor.log_events(events, row_id=transcript_hash(transcript))
            # The user channel is independent of the assistant channel —
            # a session whose assistant text scrubs to empty can still
            # carry user-stated world facts (review finding).
            attribute_ids = _mine_attributes(
                user_text[-_MAX_CAPTURED_CHARS:], client=client,
                km_path=km_path, store=store,
            )
            return CaptureResult(
                claim_ids=(),
                skip_reason="empty assistant content after scrub",
                redactions=redactions,
                attribute_claim_ids=attribute_ids,
            )
        if len(text) > _MAX_CAPTURED_CHARS:
            # Keep the tail — the session's recent work — and drop the
            # head. The capped text is `distil()`'s input; the head of a
            # session is its stale opening (recall warm-up, re-orientation),
            # while what the session actually did lands at the end.
            text = text[-_MAX_CAPTURED_CHARS:]
        if len(user_text) > _MAX_CAPTURED_CHARS:
            # Same tail rule: a correction late in the session ("actually
            # we're on F16 now") is likelier the current truth.
            user_text = user_text[-_MAX_CAPTURED_CHARS:]

        workspace_tag = workspace_tag_for_cwd(transcript.cwd)
        # Whole-session fallback rows stay "factual"; atomic facts carry the
        # polarity the extractor classified (factual / prefer / avoid).
        polarity = ("factual", workspace_tag)
        t_hash = transcript_hash(transcript)

        atomic = extract_events(text, client=client)
        if atomic is None:
            # `client is None` or extractor failed — fall back to the
            # single-claim path so a flaky LLM never degrades it.
            claim_id = _write_single(
                store, text=text, polarity=polarity, t_hash=t_hash,
                workspace_tag=workspace_tag,
            )
            claim_ids: tuple[str, ...] = (claim_id,)
            log_row = claim_id
            skip_reason: str | None = None
        elif not atomic:
            # Extractor ran, found nothing durable — planning/navigation
            # sessions land here.
            claim_ids = ()
            log_row = t_hash
            skip_reason = "no extractable events"
        else:
            claim_ids = _write_atomic(
                store, facts=atomic, t_hash=t_hash,
                workspace_tag=workspace_tag,
            )
            # Multiple claim_ids share the same scrubbed text, so the
            # redaction events correlate at session grain (transcript_hash)
            # rather than to any single claim.
            log_row = t_hash
            skip_reason = None
        if events:
            redactor.log_events(events, row_id=log_row)
        attribute_ids = _mine_attributes(
            user_text, client=client, km_path=km_path, store=store,
        )
        return CaptureResult(
            claim_ids=claim_ids, skip_reason=skip_reason, redactions=redactions,
            attribute_claim_ids=attribute_ids,
        )
    except Exception as exc:
        # Fail-open. Skip-reason carries the exception *type* only — never
        # the message, since exceptions can attach paths, polarity strings,
        # or claim text that was the very thing the redactor was trying to
        # protect (a ValueError from store.write may quote the offending
        # text verbatim).
        return CaptureResult(
            claim_ids=(),
            skip_reason=f"capture failed: {type(exc).__name__}",
            redactions=0,
        )


def _mine_attributes(
    text: str,
    *,
    client: LLMClient | None,
    km_path: str | Path | None,
    store: ExperienceClaimStore,
) -> tuple[str, ...]:
    """The PASSIVE attribute source (capture-frontier charter): mine durable
    WORLD attributes from the already-scrubbed session text and land them in
    the knowledge model's insight band — the artifact learns its own world.

    Scope is the privacy contract (Kane's 2026-06-10 direction): the
    extractor's GATE 4 emits only facts about the world the knowledge model
    covers — true for anyone the artifact is shared with — and DROPS facts
    about the individual speaker; the miner stores nothing personal. The
    gate is a validated LLM prompt (E2c run 1: 0 leaks / 7 person traps),
    not a structural guarantee — the inspectable band + lens review is the
    backstop. Band-level exact-content dedup: a re-stated attribute returns
    the existing claim's id instead of appending a duplicate row (semantic
    near-duplicates are the key-normalisation follow-up).

    `text` is the SCRUBBED USER-TURN channel (`_scrub_user_turns`) — the
    one place world facts are actually stated; GATE 2 trusts only the
    user's own words.

    DORMANT pending one open finding: the 4-gate domain-neutral prompt
    needs an E2b-style re-validation (the measured precision numbers
    belong to its 3-gate ancestor). Until then the miner runs ONLY when a
    caller passes `km_path` explicitly — no auto-discovery. Best-effort
    everywhere: any failure returns `()` and never disturbs the event
    capture.
    """
    try:
        if client is None or km_path is None:
            return ()
        target = Path(km_path)
        if not target.is_file():
            return ()
        from ..store import archive
        from .attribute_capture import capture_attributes
        from .attribute_extract import extract_attributes

        attrs = extract_attributes(text, client=client)
        if not attrs:
            return ()
        existing = archive.read_insight_layer(target)
        prior = {c.content: c.claim_id for c in existing[0]} if existing else {}
        out: list[str] = []
        fresh: list[str] = []
        for attr in dict.fromkeys(a.strip() for a in attrs if a.strip()):
            if attr in prior:
                out.append(prior[attr])
            else:
                fresh.append(attr)
        if fresh:
            claims = capture_attributes(
                target, fresh, criticality="high",
                encoder=store._ensure_encoder(),
            )
            out.extend(c.claim_id for c in claims)
        return tuple(out)
    except Exception:  # noqa: BLE001 — attributes must never block capture
        return ()


def _write_single(
    store: ExperienceClaimStore,
    *,
    text: str,
    polarity: tuple[str, ...],
    t_hash: str,
    workspace_tag: str,
) -> str:
    """One claim per session with same-text dedup.

    A fresh capture that's textually identical to an existing claim in
    the same workspace bumps that claim's `recurrence_count` and
    `last_corroborated_at` instead of writing a new one. Without this,
    every re-run of a stable prompt+response accumulates duplicate event
    claims; arrow1 then sees a "cluster" of identical copies and the LLM
    correctly refuses to promote noise. (Recurrence is the architecture's
    signal that an event is *recurring*, not re-captured 8 times.)
    """
    existing = _find_dup_in(
        store.read_all(), text=text, workspace_tag=workspace_tag,
    )
    if existing is not None:
        store.write(evolve(
            existing,
            recurrence_count=existing.facts.recurrence_count + 1,
            last_corroborated_at=utcnow_iso(),
        ))
        return existing.claim_id
    claim = new_experience_claim(
        content=text, polarity=polarity, transcript_hash=t_hash,
    )
    store.write(claim)
    return claim.claim_id


def _write_atomic(
    store: ExperienceClaimStore,
    *,
    facts: list[tuple[str, str]],
    t_hash: str,
    workspace_tag: str,
) -> tuple[str, ...]:
    """One claim per `(fact, polarity)`, batched into one store update.

    Each fact lands with the polarity the extractor classified
    ("factual" / "prefer" / "avoid") so the recall valence rerank has
    real signal to weigh (2026-06 review: a uniform "factual" tuple made
    the valence term a constant — inert for ranking).

    Within-capture duplicates (the extractor emitting the same fact text
    twice) collapse to the FIRST occurrence's polarity. Cross-session
    duplicates (an existing claim with the same `(content, workspace)`)
    bump recurrence_count instead of writing a sibling. Snapshot is read
    once so dedup stays O(N×snapshot) lookups against a single disk read.
    """
    snapshot = store.read_all()
    to_write: list[Claim] = []
    seen: set[str] = set()
    for fact, fact_polarity in facts:
        if fact in seen:
            continue
        seen.add(fact)
        existing = _find_dup_in(
            snapshot, text=fact, workspace_tag=workspace_tag,
        )
        if existing is not None:
            to_write.append(evolve(
                existing,
                recurrence_count=existing.facts.recurrence_count + 1,
                last_corroborated_at=utcnow_iso(),
            ))
        else:
            to_write.append(new_experience_claim(
                content=fact, polarity=(fact_polarity, workspace_tag),
                transcript_hash=t_hash,
            ))
    if to_write:
        store.write_many(to_write)
    return tuple(c.claim_id for c in to_write)
