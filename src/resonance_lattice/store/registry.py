"""Source coordinates — passage_idx → (source_file, char_offset, char_length).

Stored as `passages.jsonl` inside the .rlat ZIP. One JSON object per passage,
in passage_idx order:

  {"id": "a3f1c2d4...", "source_file": "src/foo.py", "char_offset": 1024,
   "char_length": 800, "content_hash": "sha256:..."}

`passage_idx` is implied by line ordering — line 0 is passage 0, etc. — so
it is not stored in the JSON. The registry is the bridge between the field
band (a (N, D) matrix indexed by passage_idx) and the store layer (which
resolves source_file:char_offset back to text).

`id` (sha256_short of `(source_file, char_offset, char_length)`) is the
**stable identity** that survives `rlat refresh` / `rlat sync` deltas. A
passage that changes content but keeps its source-file slice keeps the
same id; a passage that moves within a file gets a new id (correct: the
citation contract pins to the slice, not the text). Audit 07 §"Identity
decision" for the quality rationale; without stable ids, deletes shift
every later passage_idx and silently break `corpus_diff`, evidence chains,
and any consumer that bookmarked `passage_idx`.

Phase 2 deliverable. v4.1 schema bump (additive `id` field) — Phase 7
incremental-sync work; legacy v4 archives load through `compute_id`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable


def compute_id(source_file: str, char_offset: int, char_length: int) -> str:
    """Stable passage id derived from source coordinates.

    16-char hex slice of SHA-256 over `source_file \\x1f char_offset \\x1f
    char_length`. 64-bit space; safe under birthday-paradox up to ~10⁹
    passages (v2.0 corpora bounded at ~1M). Identical inputs → identical
    id, so the same passage in two snapshots of the same archive resolves
    to the same id, and refresh/sync can pivot the band-row reorder on it.
    """
    h = hashlib.sha256()
    h.update(source_file.encode("utf-8", errors="replace"))
    h.update(b"\x1f")
    h.update(str(char_offset).encode("ascii"))
    h.update(b"\x1f")
    h.update(str(char_length).encode("ascii"))
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class PassageCoord:
    passage_idx: int
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str
    passage_id: str


def load_jsonl(text_lines: Iterable[str]) -> list[PassageCoord]:
    """Parse passages.jsonl lines into a list[PassageCoord].

    `passage_idx` is assigned from line position — every input line must
    correspond to exactly one passage. Blank lines are NOT silently skipped
    because that would renumber all downstream passages and break the
    `(passage_idx ↔ band row)` join the rest of the system depends on. Raises
    `json.JSONDecodeError` on malformed or empty lines (which is correct: a
    `passages.jsonl` with mid-file blank lines is malformed). Standard
    line-iteration over a file or `splitlines()` does not produce trailing
    empty entries, so well-formed archives parse cleanly.

    `id` is read from the line if present (v4.1+ archives) or computed
    from `(source_file, char_offset, char_length)` if absent (legacy v4
    archives). Either way the resulting `PassageCoord.passage_id` is
    deterministic from the source coordinates.
    """
    coords: list[PassageCoord] = []
    for line in text_lines:
        obj = json.loads(line)
        source_file = obj["source_file"]
        char_offset = int(obj["char_offset"])
        char_length = int(obj["char_length"])
        passage_id = obj.get("id") or compute_id(source_file, char_offset, char_length)
        coords.append(PassageCoord(
            passage_idx=len(coords),
            source_file=source_file,
            char_offset=char_offset,
            char_length=char_length,
            content_hash=obj["content_hash"],
            passage_id=passage_id,
        ))
    return coords


def write_jsonl(coords: list[PassageCoord]) -> str:
    """Serialise to JSONL, one line per coord, in `passage_idx` order.

    `passage_idx` is omitted from the line — it's recoverable from line
    position at load time. `passage_id` IS emitted (under key `id`) so
    consumers and refresh/sync deltas can pivot on it without recomputing.
    Raises if the list isn't in contiguous 0..N order.
    """
    for i, c in enumerate(coords):
        if c.passage_idx != i:
            raise ValueError(
                f"PassageCoord list must be in contiguous passage_idx order; "
                f"position {i} has passage_idx={c.passage_idx}"
            )
    parts = []
    for c in coords:
        d = asdict(c)
        d.pop("passage_idx")
        # Re-key passage_id → "id" for compactness on disk; round-trips
        # through load_jsonl's `obj.get("id")` lookup.
        d["id"] = d.pop("passage_id")
        parts.append(json.dumps(d, sort_keys=True))
    return "\n".join(parts)
