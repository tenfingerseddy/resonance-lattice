"""Memory primer generation — the `.claude/memory-primer.md` artefact.

Pairs with `cli/summary.py` (which generates the *code* primer
`.claude/resonance-context.md` from a knowledge model). This file is the
*memory* counterpart: a markdown context block for an AI assistant booting
into a session, summarising what's in the layered memory tree.

The primer pulls top hits from the semantic and episodic tiers (working is
session-local — too noisy for a primer) using a deterministic synthetic
"corpus centroid" query: the mean of all semantic + episodic embeddings,
L2-renormalised. That gives the assistant a stable, knob-free "what does
this user remember about" digest.

Phase 5 deliverable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import MaterialiserConfig
from ..field.algebra import centroid as _centroid
from .layered import LayeredMemory, MemoryEntry


def _sections(scored: list[tuple[float, MemoryEntry]], char_budget: int) -> str:
    """Render scored entries until the per-tier budget is met. The first
    entry is always included even if oversized — better one truncated
    representative than an empty section."""
    if not scored:
        return "_(empty)_"
    parts: list[str] = []
    used = 0
    for score, entry in scored:
        text = entry.text.strip().replace("\n", " ")
        if len(text) > 280:
            text = text[:279] + "…"
        line = (
            f"- ({score:+.3f}, recurrence={entry.recurrence_count}) "
            f"{entry.source_id or '·'}: {text}"
        )
        if used + len(line) > char_budget and parts:
            break
        parts.append(line)
        used += len(line)
    return "\n".join(parts)


def generate_memory_primer(
    memory_root: str | Path,
    output: str | Path,
    config: MaterialiserConfig | None = None,
    novelty_threshold: float = 0.3,
) -> int:
    """Write a markdown memory primer to `output`. Returns char count.

    `novelty_threshold` is the minimum cosine-to-centroid an entry needs to
    survive — gates against trivially-shared boilerplate dominating the
    primer. Defaults to 0.3 (matches v0.11 behaviour).

    Returns the size of the written primer in characters.
    """
    config = config or MaterialiserConfig()
    memory = LayeredMemory(memory_root)
    cpt = config.chars_per_token

    # Build the synthetic centroid query from semantic + episodic. Skip
    # working — it's session-local and biases the primer toward the most
    # recent topic, which isn't what the assistant wants on boot.
    semantic_embs = memory._tiers["semantic"].embeddings
    episodic_embs = memory._tiers["episodic"].embeddings
    pool = np.vstack([semantic_embs, episodic_embs])
    if pool.shape[0] == 0:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = (
            f"# Memory primer\n\n"
            f"_(memory tree at `{memory_root}` is empty — nothing to prime)_\n"
        )
        out.write_text(body, encoding="utf-8")
        return len(body)

    # `field.algebra.centroid` does mean+L2-normalise with the empty-band
    # zero-vector guard. Single source of truth for "corpus centroid"
    # across cli/compare, cli/summary, and here.
    centroid = _centroid(pool)

    # Rather than re-encode a query string, search via the centroid directly.
    # Bypass `recall`'s encoder + use the embeddings already on disk —
    # primer generation should be free of LLM/encoder cost beyond what's
    # already cached.
    def _score_tier(tier: str, weight: float) -> list[tuple[float, MemoryEntry]]:
        st = memory._tiers[tier]
        if not st.entries:
            return []
        sims = st.embeddings @ centroid
        out = []
        for i, entry in enumerate(st.entries):
            entry.embedding = st.embeddings[i]
            cos = float(sims[i])
            if cos < novelty_threshold:
                continue
            out.append((cos * weight * entry.salience, entry))
        out.sort(key=lambda pair: pair[0], reverse=True)
        return out

    semantic_hits = _score_tier("semantic", 1.0)
    episodic_hits = _score_tier("episodic", 0.7)

    n_semantic = memory.tier_size("semantic")
    n_episodic = memory.tier_size("episodic")
    n_working = memory.tier_size("working")

    md_parts: list[str] = []
    md_parts.append("# Memory primer")
    md_parts.append("")
    md_parts.append(
        f"Layered memory at `{memory_root}` — "
        f"working={n_working}, episodic={n_episodic}, semantic={n_semantic}."
    )
    md_parts.append("")
    md_parts.append("## Semantic (consolidated knowledge)")
    md_parts.append("")
    md_parts.append(_sections(semantic_hits, config.sections_landscape * cpt))
    md_parts.append("")
    md_parts.append("## Episodic (per-session context)")
    md_parts.append("")
    md_parts.append(_sections(episodic_hits, config.sections_evidence * cpt))
    md_parts.append("")
    body = "\n".join(md_parts)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return len(body)
