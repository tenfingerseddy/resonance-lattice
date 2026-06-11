"""Claim-kind-aware serve framing — the measured active ingredient.

R1 (constraint band) and R2 (falsification ledger) measured that HOW a
world claim is framed at serve time is what changes answers: R2's
ledger arm beat a topically-identical control by 86pp — the verdict
framing, not the topic, does the work. These are the framings the
benchmarks served; every serve surface (daemon prompt injection,
`rlat search --format context`, future skill-context) renders through
this module so the proven wording stays in one place.

Pure string rendering. Callers sanitise their own lines (delimiter
neutralising, newline flattening) before handing them over — this
module never sees transport concerns.
"""
from __future__ import annotations

from typing import Iterable

# The R1-proven framing (benchmarks/constraint_band: blind 62% violation
# -> served 7%): a standing constraint is a hard rule of the world the
# knowledge model covers, stated up front, not retrieved-by-relevance.
CONSTRAINTS_HEADING = "Standing constraints for this environment:"

# The R2-proven framing (benchmarks/falsification_ledger: 0/7 vs 6/7 on a
# topical control): the VERDICT is the active ingredient, so the heading
# carries it. Each line should keep its evidence pointer in the text
# ("Tried X; falsified by Y") — the capture surface documents that
# convention.
FALSIFIED_HEADING = "Tried and falsified in this environment:"

# kind -> section heading, in render order (hard rules first).
_SECTION_FOR_KIND: tuple[tuple[str, str], ...] = (
    ("constraint", CONSTRAINTS_HEADING),
    ("negation", FALSIFIED_HEADING),
)


def frame_claim_lines(rows: Iterable[tuple[str, str]]) -> str:
    """Render (kind, line) rows into the proven kind-framed sections.

    `rows` pairs each pre-sanitised line with its claim kind
    (`"constraint"` or `"negation"`); unknown kinds are skipped (the
    caller decides what to serve, this only frames). Returns the section
    blocks joined by a blank line — constraints first — or `""` when
    nothing renders.
    """
    by_kind: dict[str, list[str]] = {}
    for kind, line in rows:
        text = line.strip()
        if text:
            by_kind.setdefault(kind, []).append(f"- {text}")
    sections = [
        heading + "\n" + "\n".join(by_kind[kind])
        for kind, heading in _SECTION_FOR_KIND
        if by_kind.get(kind)
    ]
    return "\n\n".join(sections)
