"""Distil — §7 LLM-driven lesson extraction over recently captured rows.

Reads the captured-side of the per-user store (rows tagged
`["factual", workspace:<hash>]` by the Stop-hook capture path), invokes
an injectable `LLMClient` against the §22 Appendix C distil prompt,
and writes the resulting lessons back as `prefer`/`avoid`/`factual`
polarity rows. The §0.6 retrieval pipeline gates at recurrence ≥ 3,
so distil is what eventually promotes captured noise into hookable
expertise.

The LLM seam mirrors `optimise/synth_queries.py` — the same callable
shape `(system, messages, max_tokens) -> LLMResponse` so the harness
suite can inject a canned-response stub without touching the network.

Spec: `.claude/plans/fabric-agent-flat-memory.md` §7.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import Encoder
from ._common import utcnow_iso, workspace_tag_for_cwd
from .redaction import Redactor
from .store import DISTILLED_PREFIX, PRIMARY_POLARITY, Memory, Row

LLMResponse = collections.namedtuple("LLMResponse", "text input_tokens output_tokens")
LLMClient = Callable[[str, list[dict], int], LLMResponse]

# §4.5 dedupe threshold — same value the v0.11/v2.0 consolidation step
# used. Any candidate lesson whose nearest existing-row cosine is at or
# above this corroborates instead of writing a new row.
DEDUPE_COSINE = 0.92

# §7.6 cap — at most this many lessons survive per invocation.
DEFAULT_MAX_LESSONS = 2

# Watermark file — lives next to sidecar.jsonl in the per-user root.
_WATERMARK_FILENAME = ".distil_watermark.json"

# §18.6 idempotency journal — append-only `(row_id, transcript_hash)` pairs
# recording every distil corroboration / new-row write. Lets a re-run skip
# rows that this transcript has already touched, even when the row's
# canonical text drifts away from the candidate text after a prior cosine
# corroboration. Replaces the earlier text-based heuristic that broke
# whenever distil-time corroborate updated `transcript_hash` but left the
# row text unchanged (codex P1.2, 2026-05-02).
_JOURNAL_FILENAME = ".distil_journal.jsonl"

# Distil prompt template path (relative to the package).
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "distil.md"

def _distilled_hash(source_transcript_hash: str) -> str:
    """Map a capture-time transcript hash to the distilled-row form.

    Idempotent — distilled-form hashes are returned unchanged. The
    guard prevents `distilled:distilled:<sha>` corruption when a
    candidate is re-pinned to a row that was itself distilled (e.g.
    `--session <distilled-hash>` slips through `_select_capture_rows`).
    """
    if source_transcript_hash.startswith(DISTILLED_PREFIX):
        return source_transcript_hash
    return f"{DISTILLED_PREFIX}{source_transcript_hash}"


# ---------------------------------------------------------------------------
# Result + payload types
# ---------------------------------------------------------------------------


@dataclass
class DistilResult:
    """Outcome of one `distil(...)` invocation."""

    written_row_ids: list[str] = field(default_factory=list)
    corroborated_row_ids: list[str] = field(default_factory=list)
    skipped_count: int = 0
    processed_count: int = 0
    new_watermark_utc: str | None = None  # None when a failure preserves the watermark
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Candidate:
    """One JSON record from the distiller's output, post-validation."""

    text: str
    intent: str
    polarity: list[str]
    rationale: str
    transcript_hash: str  # the captured row this candidate derives from


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------


def _load_watermark(root: Path) -> dict:
    p = root / _WATERMARK_FILENAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_watermark(root: Path, watermark_utc: str, processed: int) -> None:
    payload = {
        "last_processed_utc": watermark_utc,
        "last_run_utc": utcnow_iso(),
        "rows_processed_total": processed,
    }
    p = root / _WATERMARK_FILENAME
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(p)


def _load_journal(root: Path) -> set[tuple[str, str]]:
    """Read the append-only journal into a `{(row_id, transcript_hash)}` set.

    Empty / missing journal returns an empty set — first-run distil
    behaves the same as if no rows had been seen.

    `row_id == ""` is a legitimate sentinel from the new-row crash-recovery
    path (see `distil()`): the journal is appended *before* `add_row`
    returns, so a crash in between leaves a transcript-only entry that
    blocks reruns via the per-transcript guard. Real `(row_id, th)` pairs
    only come from the corroborate path.
    """
    p = root / _JOURNAL_FILENAME
    if not p.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = obj.get("row_id")
        th = obj.get("transcript_hash")
        if isinstance(rid, str) and isinstance(th, str):
            seen.add((rid, th))
    return seen


def _journal_append(root: Path, row_id: str, transcript_hash: str) -> None:
    """Append one `(row_id, transcript_hash, ts)` entry to the journal.

    Append-only and never compacted — for typical use volumes (50
    corroborations per distil run × 100 runs ≈ 500 KB lifetime), the
    file stays small. v2.1.1 follow-up: prune entries for rows that
    `gc` has deleted.
    """
    p = root / _JOURNAL_FILENAME
    payload = {
        "row_id": row_id,
        "transcript_hash": transcript_hash,
        "ts": utcnow_iso(),
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Filter + prompt construction
# ---------------------------------------------------------------------------


def _is_capture_row(row: Row) -> bool:
    """A capture-time row carries `factual` primary + a workspace tag,
    and its `transcript_hash` is the SHA over the original transcript.

    Manual rows, previously-distilled rows, and v2.0-migrated rows are
    all excluded so distil never reprocesses its own output (or a
    cross-version artefact) as if it were raw capture.
    """
    if "factual" not in row.polarity:
        return False
    if row.is_manual() or row.is_distilled() or row.is_migrated():
        return False
    return True


def _select_capture_rows(
    rows: list[Row],
    *,
    watermark_utc: str | None,
    since: str | None,
    all_rows: bool,
    session: str | None,
) -> list[Row]:
    """Filter the sidecar to the capture-side rows the distiller should
    process this run. Per §7.3 override flags."""
    candidates = [r for r in rows if _is_capture_row(r)]
    if session is not None:
        # `session` is interpreted as a transcript_hash — Sub-MVP doesn't
        # have a separate session_id field; transcript_hash is the only
        # session-stable identifier.
        return [r for r in candidates if r.transcript_hash == session]
    if all_rows:
        return candidates
    if since is not None:
        return [r for r in candidates if r.created_at > since]
    if watermark_utc:
        return [r for r in candidates if r.created_at > watermark_utc]
    return candidates


def _build_prompt(
    capture_rows: list[Row],
    *,
    user_id: str,
    workspace_path: str,
    workspace_hash: str,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    transcript_lines = [
        f"row_id={r.row_id} created_at={r.created_at} transcript_hash={r.transcript_hash}\n"
        f"text:\n{r.text}\n"
        for r in capture_rows
    ]
    return (
        template.replace("{user_id}", user_id)
        .replace("{workspace_path}", workspace_path)
        .replace("{workspace_hash}", workspace_hash)
        .replace("{transcript}", "\n---\n".join(transcript_lines))
    )


# ---------------------------------------------------------------------------
# JSON parsing + validation
# ---------------------------------------------------------------------------


# Match a top-level JSON array even when the model wraps it in prose.
_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", re.DOTALL)


def _extract_json_array(text: str) -> list | None:
    """Best-effort extraction of the lesson array from a free-form LLM
    response. Returns None when no parseable array exists.
    """
    text = text.strip()
    candidates: list[str] = []
    if text.startswith("["):
        candidates.append(text)
    candidates.extend(m.group(0) for m in _JSON_ARRAY_RE.finditer(text))
    for cand in candidates:
        try:
            value = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return None


def _validate_lesson(
    obj: object, workspace_hash: str
) -> tuple[str, str, list[str], str] | None:
    """Validate one lesson dict + canonicalise polarity tags.

    Returns `(text, intent, polarity, rationale)` or None on rejection.
    """
    if not isinstance(obj, dict):
        return None
    text = obj.get("text")
    intent = obj.get("intent", "")
    polarity = obj.get("polarity")
    rationale = obj.get("rationale", "")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(intent, str):
        return None
    if not isinstance(polarity, list) or not polarity:
        return None
    if not isinstance(rationale, str):
        return None

    # Interpolate any literal `{workspace_hash}` placeholder the LLM
    # echoed from the prompt.
    polarity = [
        tag.replace("{workspace_hash}", workspace_hash) if isinstance(tag, str) else tag
        for tag in polarity
    ]
    if not all(isinstance(tag, str) for tag in polarity):
        return None
    primaries = [t for t in polarity if t in PRIMARY_POLARITY]
    if len(primaries) != 1:
        return None
    return text.strip(), intent.strip(), list(polarity), rationale.strip()


# ---------------------------------------------------------------------------
# Embedding + dedupe (against the live store)
# ---------------------------------------------------------------------------


def _embed_lesson(text: str, intent: str, encoder: Encoder) -> np.ndarray:
    """Same `text + " | intent: " + intent` convention the store uses."""
    payload = f"{text} | intent: {intent}" if intent else text
    embedding = encoder.encode([payload])[0]
    l2_normalize(embedding)
    return embedding


def _find_dedupe_match(
    candidate_emb: np.ndarray,
    rows: list[Row],
    band: np.ndarray,
    candidate_primary: str,
) -> tuple[int, float] | None:
    """Return `(index, cosine)` for the highest-cosine row that meets
    the §4.5 dedupe threshold AND shares the candidate's primary
    polarity. None if no match.
    """
    if band.size == 0:
        return None
    cosines = band @ candidate_emb
    # Mask to rows that match the candidate's primary polarity.
    same_primary = np.array(
        [candidate_primary in r.polarity for r in rows], dtype=bool
    )
    if not same_primary.any():
        return None
    masked = np.where(same_primary, cosines, -np.inf)
    idx = int(np.argmax(masked))
    if masked[idx] >= DEDUPE_COSINE:
        return idx, float(masked[idx])
    return None


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def distil(
    *,
    store: Memory,
    redactor: Redactor,
    client: LLMClient,
    encoder: Encoder | None = None,
    user_id: str = "user",
    workspace_path: str | None = None,
    since: str | None = None,
    all_rows: bool = False,
    session: str | None = None,
    dry_run: bool = False,
    max_lessons: int = DEFAULT_MAX_LESSONS,
    max_tokens: int = 2000,
) -> DistilResult:
    """Run §7 distil on the recently-captured rows.

    Always returns a DistilResult; never raises on distiller misbehaviour
    (malformed JSON, empty response, dedupe-only outcome). Re-raises only
    on caller-side errors (Memory I/O, encoder load, redactor failure).
    """
    workspace_path = workspace_path or str(Path.cwd())
    cwd_tag = workspace_tag_for_cwd(workspace_path)
    cwd_hash = cwd_tag.removeprefix("workspace:")

    rows, band = store.read_all()
    watermark = _load_watermark(store.root).get("last_processed_utc")
    capture_rows = _select_capture_rows(
        rows,
        watermark_utc=watermark,
        since=since,
        all_rows=all_rows,
        session=session,
    )
    result = DistilResult(processed_count=len(capture_rows))
    if not capture_rows:
        return result

    if encoder is None:
        encoder = store._ensure_encoder()  # type: ignore[attr-defined]

    prompt = _build_prompt(
        capture_rows,
        user_id=user_id,
        workspace_path=workspace_path,
        workspace_hash=cwd_hash,
    )
    response: LLMResponse = client(
        "You distil durable lessons from coding-session transcripts.",
        [{"role": "user", "content": prompt}],
        max_tokens,
    )
    parsed = _extract_json_array(response.text)
    if parsed is None:
        # §7.9 row 1: malformed JSON — no rows written, watermark does
        # NOT advance.
        result.notes.append("malformed JSON; watermark preserved")
        return result

    # Validate each candidate; cap at max_lessons.
    candidates: list[_Candidate] = []
    for raw in parsed[:max_lessons]:
        validated = _validate_lesson(raw, cwd_hash)
        if validated is None:
            result.skipped_count += 1
            continue
        text, intent, polarity, rationale = validated
        # §6.2 Layer 1 redaction — drop the candidate entirely if any
        # built-in pattern fires. Doubles as the §6.2 Layer 2 safety net.
        scrub_text, text_events = redactor.scrub(text)
        scrub_intent, intent_events = redactor.scrub(intent)
        if text_events or intent_events:
            redactor.log_events(text_events + intent_events)
            result.skipped_count += 1
            continue
        # Pin the candidate to the most recent capture row by
        # transcript_hash so dedupe (c) can detect re-runs against the
        # same source content.
        latest_capture = max(capture_rows, key=lambda r: r.created_at)
        candidates.append(_Candidate(
            text=scrub_text,
            intent=scrub_intent,
            polarity=polarity,
            rationale=rationale,
            transcript_hash=latest_capture.transcript_hash,
        ))

    if dry_run:
        result.notes.append(f"dry-run: {len(candidates)} candidates would write")
        return result

    # §18.6 idempotency journal — `(row_id, transcript_hash)` set of every
    # corroboration / new-row write across all prior distil runs. Codex
    # P1.2 (2026-05-02): the earlier text-based heuristic broke whenever
    # cosine corroborate updated a row's `transcript_hash` but left the
    # text unchanged (next run's normalize-equality check missed). The
    # journal is the canonical record per the §18.6 spec wording: "a row
    # + transcript_hash pair never increments recurrence twice."
    seen_pairs = _load_journal(store.root)
    # Per-transcript guard: any prior journal entry for `distilled_th`
    # means this source transcript was already processed by an earlier
    # distil run — skip every candidate sourced from it. Re-running
    # distil over the same transcript is a diagnostic operation, not a
    # productive one (LLM non-determinism would otherwise produce
    # uncontrolled row drift). Built from the journal so it survives
    # process restart.
    seen_source_transcripts: set[str] = {th for (_, th) in seen_pairs}

    # Mutate the in-memory snapshot so successive candidates dedupe
    # against earlier writes from the same batch.
    for cand in candidates:
        primary = next(t for t in cand.polarity if t in PRIMARY_POLARITY)
        distilled_th = _distilled_hash(cand.transcript_hash)

        if distilled_th in seen_source_transcripts:
            continue

        emb = _embed_lesson(cand.text, cand.intent, encoder)

        # (c) cosine-based corroboration: a different-transcript lesson
        # that semantically duplicates an existing row bumps recurrence
        # instead of writing a new row.
        dedupe = _find_dedupe_match(emb, rows, band, primary)
        if dedupe is not None:
            idx, _cos = dedupe
            existing = rows[idx]
            # Per-pair guard inside one run: a second candidate from the
            # same transcript that cosine-matches the same row should
            # not double-bump.
            if (existing.row_id, distilled_th) in seen_pairs:
                continue
            # Journal first, then mutate (codex P2.1, 2026-05-02). If we
            # crash between the journal append and the row update, on
            # rerun the per-transcript guard above blocks the whole
            # transcript — we lose the corroboration but never
            # double-bump. The opposite order risks duplicate increments
            # on rerun, which silently inflates recurrence counts.
            _journal_append(store.root, existing.row_id, distilled_th)
            seen_pairs.add((existing.row_id, distilled_th))
            seen_source_transcripts.add(distilled_th)
            updated = store.update_row(
                existing.row_id,
                recurrence_count=existing.recurrence_count + 1,
                last_corroborated_at=utcnow_iso(),
                transcript_hash=distilled_th,
            )
            rows[idx] = updated
            result.corroborated_row_ids.append(existing.row_id)
            continue

        # New row. Journal-stub first (empty row_id placeholder) so a
        # crash between journal-append and add_row produces "skip this
        # transcript on rerun" not "duplicate row on rerun" — the
        # per-transcript guard above keys on transcript_hash only.
        _journal_append(store.root, "", distilled_th)
        seen_source_transcripts.add(distilled_th)
        new_id = store.add_row(
            text=cand.text,
            polarity=cand.polarity,
            transcript_hash=distilled_th,
            intent=cand.intent,
            embedding=emb,
        )
        rows, band = store.read_all()
        seen_pairs.add((new_id, distilled_th))
        result.written_row_ids.append(new_id)

    # Watermark advances on success per §7.9 row 2 + on a successful but
    # zero-write outcome.
    if since is None and session is None and not dry_run:
        latest = max(capture_rows, key=lambda r: r.created_at).created_at
        _save_watermark(
            store.root, latest, len(capture_rows)
        )
        result.new_watermark_utc = latest
    return result
