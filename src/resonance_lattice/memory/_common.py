"""Helpers shared across the v2.1 flat-memory modules.

These live here (not in `store.py` / `capture.py`) because the future
MVP migrate + daemon recall paths derive workspace + transcript hashes
the same way as capture, and share the timestamp shape with every row
write.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Iterable


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with second precision and trailing Z.

    Locked to the v2.1 sidecar `created_at` / `last_corroborated_at` shape.
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_cwd(cwd: str) -> str:
    """Canonicalise a cwd string before hashing.

    Windows paths are case-insensitive at the filesystem level, but Claude
    Code's UserPromptSubmit envelope sometimes passes a lowercased drive
    letter (`c:\\Users\\...`) while `os.getcwd()` returns uppercase
    (`C:\\Users\\...`) — the same workspace would hash to two different
    `workspace:<hash>` tags and the §0.6 workspace gate would drop every
    hit. `os.path.normcase` lowercases on Windows and is a no-op on POSIX,
    so we route every workspace_hash input through it.
    """
    return os.path.normcase(cwd)


def workspace_hash(cwd: str) -> str:
    """sha256[:6] of normalised cwd, used as the `workspace:<hash>` scope.

    Six hex chars = 24 bits ≈ 16M-way collision space; collisions matter
    only for cross-workspace bleed risk and §18.3 mitigations are wired at
    the retrieval layer (D.8 harness suite checks intentional collisions).
    Path normalisation is via `os.path.normcase` — case-folds on Windows,
    no-op on POSIX. See `_normalise_cwd` for the rationale.
    """
    return hashlib.sha256(_normalise_cwd(cwd).encode("utf-8")).hexdigest()[:6]


def workspace_tag_for_cwd(cwd: str | Path | None = None) -> str:
    """Build the `workspace:<hash>` scope-tag string for `cwd`.

    Defaults to `Path.cwd()`. Callers (manual CLI add, Stop-hook capture,
    future MVP migrate) share this single derivation so the harness has
    one mock point for the §18.3 cwd-collision contract test.
    """
    target = str(cwd) if cwd is not None else str(Path.cwd())
    return f"workspace:{workspace_hash(target)}"


def stable_hash(parts: Iterable[bytes | str]) -> str:
    """Stable SHA-256 hex over a sequence of byte/string parts.

    Uses NUL separators between parts so concatenation can't collide
    (`"ab" + "c"` and `"a" + "bc"` produce different hashes). Fed by the
    Stop-hook capture path to derive `transcript_hash`, and by the future
    daemon recall path to dedup query bodies.
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode("utf-8")
        h.update(part)
        h.update(b"\x00")
    return h.hexdigest()


_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def make_ulid() -> str:
    """26-char Crockford base-32 ULID. Stdlib-only.

    Encodes 48-bit ms timestamp + 80-bit randomness. Lexicographically
    sortable, collision-safe across machines. Used for both `Row.row_id`
    in the memory store and `LiveIntent.intent_id` in the live intent
    graph — same shape so the two id-spaces are visually distinguishable
    only by their containers, not their format.
    """
    ts_ms = int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)
    rand_bits = secrets.randbits(80)
    value = (ts_ms << 80) | rand_bits
    return "".join(_CROCKFORD[(value >> (5 * (25 - i))) & 0b11111] for i in range(26))


def validate_enum(name: str, value: Any, allowed: frozenset[str]) -> None:
    """Reject `value` if not a member of `allowed`. Shared across `Row`
    write paths, the live intent store, and the outcome ledger so all
    enum-shaped fields surface the same error message format."""
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)}; got {value!r}"
        )


def validate_criterion(c: Any) -> None:
    """Reject anything that isn't `{text: str, measure: str}`. The success
    criterion shape (architecture §"Success criteria") is the same in
    intent rows and outcome records — one validator covers both."""
    if not isinstance(c, dict) or set(c.keys()) != {"text", "measure"}:
        raise ValueError(
            f"success_criteria entry must be {{text, measure}}; got {c!r}"
        )
    if not isinstance(c["text"], str) or not isinstance(c["measure"], str):
        raise ValueError(
            f"criterion text and measure must be strings; got {c!r}"
        )


def parse_iso_utc(ts: str) -> _dt.datetime:
    """Tolerant ISO-8601 parser → tz-aware UTC datetime.

    Accepts the trailing-`Z` shape rlat writes plus the standard ISO
    offset shape `datetime.fromisoformat` produces. Falls back to "now"
    on unparseable input so a single corrupted timestamp can't tank
    rerank, forget, or eval — the architecture's fail-open posture.
    """
    cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        when = _dt.datetime.fromisoformat(cleaned)
    except ValueError:
        return _dt.datetime.now(_dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return when


CONFIDENCE_DILUTION: dict[str, str] = {
    "verified": "high",
    "high": "medium",
    "medium": "low",
    "low": "low",
}
"""Distillation confidence-dilution map shared across the three arrows.

Architecture §"Field interactions worth knowing": confidence dilutes by one
step per promotion unless separately verified. Forces the system to verify
higher-level beliefs rather than blindly trust the chain. `low → low` is the
floor — additional dilution can't push below it."""


def parse_llm_json(text: str) -> Any:
    """Parse a model response that may be wrapped in ``` / ```json fences,
    or returned without the opening `{` because the assistant turn was
    prefilled.

    Three known failure modes this recovers from:
      1. ```json...``` fence — strip one outer fence with language tag.
      2. Prefilled with `{` — model's response text starts with the JSON
         body (e.g. `"promote": false, "reason": "..."}`); prepend `{`.
      3. Plain JSON — pass through.

    Raises `json.JSONDecodeError` on actual parse failure.
    """
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Prefill recovery: if the response neither opens with `{` / `[` nor
    # is a JSON literal, the caller likely prefilled `{` and we got the
    # body. Prepending the opening brace turns it back into valid JSON.
    if s and s[0] not in "{[\"" and not (s[0].isdigit() or s in {"true", "false", "null"}):
        s = "{" + s
    return json.loads(s)


_HEDGE_PATTERN = re.compile(
    r"\b("
    r"in some cases"
    r"|sometimes"
    r"|might"
    r"|may (?:be|need|want|require|provide|cause|result|lead|help|fail)"
    r"|could (?:be|potentially)"
    r"|tends to"
    r"|often"
    r"|usually"
    r")\b",
    re.IGNORECASE,
)
"""Hedge phrases that defeat falsifiability.

The architecture's distil prompts (arrows 1/2/3) already tell the LLM the
output must be falsifiable — a future event must be able to contradict it
— and explicitly forbid "in some cases". These are the patterns that, in
practice, signal the LLM hedged anyway. `_validate_promotion` rejects on
match so unfalsifiable rows can't reach the store.

Conservative by design: matches obvious hedges only. Won't catch every
unfalsifiable claim (e.g. tautologies, empty predicates like "is
important"), but the false-positive rate stays near zero on legitimate
prescriptive rules. The post-LLM cosine + word-count gates already cover
other failure modes."""


def falsifiability_violation(text: str) -> str | None:
    """Return a rejection reason if `text` contains a hedge phrase, else
    None. Shared across `distil_arrow{1,2,3}._validate_promotion`."""
    match = _HEDGE_PATTERN.search(text)
    if match is None:
        return None
    return f"unfalsifiable hedge phrase: {match.group(0)!r}"


def reject_text_quality(
    text: str,
    encoded_text: Any,
    anchor_embedding: Any,
    *,
    max_words: int,
    post_validation_cosine: float,
) -> str | None:
    """Run the post-LLM text-quality gates shared across the three arrows:
    length cap → cosine alignment with the parent anchor → falsifiability.

    Arrow-specific gates (e.g. arrow3's "shorter than parent learning")
    layer on top of this in the caller's `_validate_promotion`.
    """
    word_count = len(text.split())
    if word_count > max_words:
        return f"text too long ({word_count} words > {max_words})"
    cos = float(anchor_embedding @ encoded_text)
    if cos < post_validation_cosine:
        return f"post-LLM alignment {cos:.3f} < {post_validation_cosine}"
    hedge = falsifiability_violation(text)
    if hedge is not None:
        return hedge
    return None


def atomic_write_json(target: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomic JSON write: tmp + os.replace, sorted keys.

    Used by the live intent graph + workspace declarations + any future
    state file that wants the same single-tmp-file guarantee. Caller is
    responsible for any locking — this helper is lock-agnostic so it can
    sit inside or outside a portalocker section."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, indent=indent),
        encoding="utf-8",
    )
    os.replace(tmp, target)
