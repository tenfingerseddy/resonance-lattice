"""Layer-1 redaction — pattern-based credential scrub at capture time.

Per `.claude/plans/fabric-agent-flat-memory.md` §6.2 Layer 1: every captured
text and every distil candidate passes through `Redactor.scrub` before
landing in the sidecar. Mechanical patterns only — semantic / private-context
redaction lives in the distiller's prompt (Layer 2), and per-project tuning
lives in `.rlat/capture.toml` (Layer 3).
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ._common import utcnow_iso

# ---------------------------------------------------------------------------
# Built-in patterns — §6.2 Layer 1
# ---------------------------------------------------------------------------

# Tuple-of-tuples, not list, so callers can't mutate the built-ins. Order
# matters: `anthropic_key` and `openai_key` both start with `sk-`; matching
# `anthropic_key` first attributes correctly without changing the redaction
# (both shapes get replaced).
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{40,}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("github_fine_pat", re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    ),
    (
        "pem_key",
        re.compile(
            r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----"
        ),
    ),
    # `long_hex` is conservative on the false-positive side — matches MD5 /
    # SHA-1 / SHA-256 hex but also any other 32+ char hex sequence. Tunable
    # per-project via `redact_extra` (Layer 3) if it over-matches.
    ("long_hex", re.compile(r"\b[A-Fa-f0-9]{32,}\b")),
)

# Glob patterns whose tool-call payloads get redacted *entire* (not just
# pattern-scanned). Path is preserved so distil context isn't lost.
DENYLIST_PATHS: tuple[str, ...] = (
    ".env*",
    ".envrc",
    "*.secret",
    "*.key",
    "*.pem",
    "id_rsa",
    "*/.aws/credentials*",
    "*\\.aws\\credentials*",  # Windows path separators in the glob
)

REDACTED_TOKEN = "<REDACTED>"
REDACTED_FILE = "<REDACTED file contents>"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedactionEvent:
    """One pattern hit. Audit-log line is built from this."""

    pattern: str
    matches: int
    layer: int = 1


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------


class Redactor:
    """Pattern-based scrubber.

    Constructor accepts:
    - `extra_patterns`: per-project regexes appended to (not replacing) the
      built-ins. Layer-3 surface.
    - `denylist_paths`: per-project globs appended to the built-in denylist.
    - `audit_log_path`: where redaction events are appended. None disables
      the audit log (test fixtures opt out; production callers always pass
      the live path).

    Note: extra patterns can only *add* coverage — they cannot weaken the
    built-ins. Callers wanting to skip a built-in pattern (e.g., `long_hex`
    over-matching on a hex-heavy corpus) need a code change, not config.
    That's deliberate per §6.2.
    """

    def __init__(
        self,
        *,
        extra_patterns: Iterable[str] | None = None,
        denylist_paths: Iterable[str] | None = None,
        audit_log_path: Path | None = None,
    ):
        self._patterns: list[tuple[str, re.Pattern[str]]] = list(SECRET_PATTERNS)
        if extra_patterns:
            for i, raw in enumerate(extra_patterns):
                self._patterns.append((f"extra_{i}", re.compile(raw)))
        self._denylist: tuple[str, ...] = DENYLIST_PATHS + tuple(denylist_paths or ())
        self._audit_log_path = audit_log_path

    # -- core scrubbers ----------------------------------------------------

    def scrub(self, text: str) -> tuple[str, list[RedactionEvent]]:
        """Replace every match with `<REDACTED>` and record events.

        Iteration order is the pattern order in `_patterns`; built-ins
        come first so attribution is stable for the audit log.
        """
        events: list[RedactionEvent] = []
        out = text
        for name, pat in self._patterns:
            new_out, n = pat.subn(REDACTED_TOKEN, out)
            if n > 0:
                events.append(RedactionEvent(pattern=name, matches=n))
                out = new_out
        return out, events

    def scrub_tool_call(
        self, path: str, content: str
    ) -> tuple[str, list[RedactionEvent]]:
        """If `path` matches the denylist, replace `content` entirely.

        Returns `(content, events)`. The path itself is never redacted —
        callers preserving context need to know *which* file was read,
        just not what was in it.
        """
        if self._matches_denylist(path):
            return REDACTED_FILE, [RedactionEvent(pattern="denylist_path", matches=1)]
        return self.scrub(content)

    def _matches_denylist(self, path: str) -> bool:
        # fnmatch matches the *whole* string against the glob, so `.env*`
        # won't hit `/home/kane/.env`. Test against the full path AND the
        # basename, in both POSIX and Windows separator styles.
        normalised = path.replace("\\", "/")
        basename = normalised.rsplit("/", 1)[-1]
        candidates = {path, normalised, basename}
        return any(
            fnmatch.fnmatch(cand, pat)
            for pat in self._denylist
            for cand in candidates
        )

    # -- audit log ---------------------------------------------------------

    def log_events(
        self, events: Iterable[RedactionEvent], *, row_id: str | None = None
    ) -> None:
        """Append one line per event to the audit log.

        Append-only by contract — the file is never read by retrieval and
        never overwritten in place. Rotation is deferred to a later phase
        per §6.4.
        """
        if self._audit_log_path is None:
            return
        ts = utcnow_iso()
        row_part = f"  row_id={row_id}" if row_id else ""
        lines = [
            f"{ts}  layer={ev.layer}  pattern={ev.pattern}  "
            f"matches={ev.matches}{row_part}"
            for ev in events
        ]
        if not lines:
            return
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
