"""reconcile — the contradiction ACT layer: judge the geometry candidates, surface the REAL contradictions.

The self-audit stores `high_cosine_pairs` — same-topic CANDIDATES, mostly restatements. This is where the
cheap-filter → LLM-judge moat is realised: a stance judge rules on ONLY those candidates (a bounded shortlist),
confirming which pairs actually CONTRADICT (vs restate/agree) and which side looks more authoritative. The bench
(`bench_haiku_vs_sonnet_*`) found Haiku matches Sonnet on this kind of judgement at ~2× less cost, so the act
layer is cheap.

`judge_contradictions` SURFACES the confirmed contradictions as a structured finding. `reconcile_contradiction`
is the policy-gated ACT that RESOLVES one — and it is deliberately NON-DESTRUCTIVE: it never edits the corpus
source files. Instead it records a high-trust RESOLUTION claim in the band's earned layer (the authoritative value
+ that it supersedes the losing side), citing BOTH conflicting passages. The conflicting passages stay; serve just
ranks the resolution first (it lands at a high provenance tier). The CALLER (human / agent) decides the resolution
and which side wins — this is not an autonomous edit.

Injected `client` (Anthropic-shaped; inject a stub for tests). Best-effort, never raises — a failed judge degrades
to "not confirmed", never a crash.
"""
from __future__ import annotations

STANCE_SYSTEM = """\
You judge whether two passages, already known to be about the SAME topic, CONTRADICT each other.

A CONTRADICTION is: both passages state a value or fact for the SAME quantity, measured the SAME way, and those
two values are INCOMPATIBLE — they cannot both be true.

NOT a contradiction (answer "paraphrase" or "agree"):
- A restatement, paraphrase, or summary of the same fact.
- Two DIFFERENT metrics, measurements, conditions, columns, or time periods that merely share a subject or a
  number — e.g. one reports a ceiling/oracle value and the other a net-of-echo value; one is "version 1" and the
  other "version 2". Different quantities cannot contradict, even when they mention the same entity.
- The SAME value reported twice — that is agreement, not contradiction.
- Dense tables of many metrics: do NOT call a contradiction just because two tables share a row label or a number.
  Identify the ONE specific quantity in question and check only whether ITS value is incompatible.

Only answer "contradict" if you can name the single shared quantity and state the two incompatible values for it.

If they contradict, say which passage looks MORE AUTHORITATIVE or RECENT if discernible from the text (an
official/primary source, a newer date); otherwise "unclear".

Reply with a single JSON object, nothing else:
{"stance": "contradict"|"paraphrase"|"agree"|"unrelated", "more_authoritative": "a"|"b"|"unclear", "reason": "<one line; if contradict, name the shared quantity and the two values>"}
"""


def _pair_message(a: dict, b: dict) -> str:
    return (f"PASSAGE A (source: {a.get('source_file', '?')}):\n{a.get('text', '')}\n\n"
            f"PASSAGE B (source: {b.get('source_file', '?')}):\n{b.get('text', '')}\n")


def judge_contradictions(
    km_path,
    client,
    *,
    model: str | None = None,
    max_pairs: int = 50,
    min_cosine: float = 0.92,
    source_root=None,
    demand_rank: bool = True,
) -> list[dict]:
    """Stance-judge the corpus's high-cosine same-topic pairs; return the CONFIRMED contradictions.

    Surfaces the geometry candidates (`store.self_audit.find_contradiction_candidates`, with text resolved) to a
    stance judge and keeps only the pairs it rules `contradict`. Each confirmed row carries both passages, the
    cosine, the authority hint, and the judge's one-line reason — a structured finding for a human/agent to act on.

    `demand_rank` (auto-on): the GEOMETRY does the prioritisation, not the judge. It fetches a wider candidate POOL
    by cosine, orders it by QUERY DEMAND (`rank_contradictions_by_demand` over the stored telemetry — conflicts in
    the path of real query traffic first), and spends the (costly) judge on only the top `max_pairs`. So the
    expensive step lands on the conflicts users actually hit; with no telemetry it degrades to plain cosine order
    (unchanged). Returns `[]` on no client / no candidates / any failure (best-effort, never raises)."""
    if client is None:
        return []
    try:
        from pathlib import Path

        from .._pricing import SONNET_MODEL
        from ..store import archive
        from ..store._llm import judge_json
        from ..store.self_audit import find_contradiction_candidates, rank_contradictions_by_demand

        pool = max_pairs * 4 if demand_rank else max_pairs
        candidates = find_contradiction_candidates(
            km_path, max_pairs=pool, min_cosine=min_cosine, source_root=source_root, resolve_text=True)
        if demand_rank and candidates:
            try:
                contents = archive.read(Path(km_path))
                candidates = rank_contradictions_by_demand(
                    candidates, contents, archive.read_telemetry(km_path))
            except Exception:
                pass  # ranking is best-effort — fall back to cosine order
        candidates = candidates[:max_pairs]
        confirmed: list[dict] = []
        for c in candidates:
            if not (c.a.get("text") and c.b.get("text")):
                continue  # can't judge a pair whose text didn't resolve
            out = judge_json(client, model or SONNET_MODEL, STANCE_SYSTEM,
                             _pair_message(c.a, c.b), max_tokens=300, temperature=0.0)
            if not isinstance(out, dict) or out.get("_parse_error"):
                continue
            if str(out.get("stance", "")).strip().lower() != "contradict":
                continue
            confirmed.append({
                "cosine": c.cosine,
                "a": {"passage_idx": c.a.get("passage_idx"), "source_file": c.a.get("source_file"),
                      "text": c.a.get("text", "")[:300]},
                "b": {"passage_idx": c.b.get("passage_idx"), "source_file": c.b.get("source_file"),
                      "text": c.b.get("text", "")[:300]},
                "more_authoritative": str(out.get("more_authoritative", "unclear")).strip().lower(),
                "reason": str(out.get("reason", "")).strip()[:200],
            })
        return confirmed
    except Exception:
        return []


def reconcile_contradiction(
    km_path,
    passage_idx_a: int,
    passage_idx_b: int,
    resolution: str,
    *,
    source_root=None,
    client=None,
    faithfulness: float = 0.9,
    provenance: str = "user",
    question: str = "contradiction resolution",
):
    """RESOLVE a confirmed contradiction between two corpus passages — NON-DESTRUCTIVE, policy-gated.

    Records `resolution` (the authoritative synthesis: the winning value + that it supersedes the losing side) as a
    high-trust claim in the band's earned layer, CITING BOTH conflicting passages as provenance. It does NOT edit
    the corpus source files — the conflicting passages remain; serve simply ranks this resolution first because it
    lands at a high provenance tier (`provenance` default "user" — the human/agent who decided).

    The CALLER decides the resolution + which side wins (this is the gated ACT, not an autonomous edit).
    `resolution` should be a genuine synthesis (the winning value + the supersede note). A verbatim copy of — or a
    phrase wholly contained in — a cited passage is rejected HERE as adding nothing; beyond that, synthesis quality
    is the caller's responsibility. Grounding: the free (`client=None`) path TRUSTS the caller's resolution (no
    structural check that it matches the passages — the human/agent decided it), so no API key is needed; pass a
    judge `client` to gate the resolution with the metered faithfulness check against the two passages instead.

    Returns `(landed: bool, outcomes)`. Never raises — any failure degrades to `(False, [])`."""
    try:
        from pathlib import Path

        from ..store import archive, open_store
        from ..store.promotion import promote_if_faithful

        if not str(resolution).strip():
            return False, []
        contents = archive.read(Path(km_path))
        store = open_store(Path(km_path), contents, source_root)
        reg = contents.registry
        evidence = []
        for idx in (passage_idx_a, passage_idx_b):
            if idx < 0 or idx >= len(reg):
                return False, []  # a stale/invalid index — refuse rather than cite the wrong passage
            coord = reg[idx]
            try:
                text = store.fetch(coord.source_file, coord.char_offset, coord.char_length)
            except Exception:
                text = ""
            evidence.append({
                "source_file": coord.source_file, "char_offset": coord.char_offset,
                "char_length": coord.char_length, "score": 0.9, "text": text,
                "passage_id": coord.passage_id, "content_hash": coord.content_hash,
            })

        # A verbatim copy of (or a phrase wholly contained in) a cited passage is not a resolution — it just
        # re-states one side at high trust and adds no supersede decision. The compression test only counts
        # citations, so enforce the synthesis floor here. (Whitespace/case-insensitive substring; cheap, no model.)
        _norm = " ".join(str(resolution).lower().split())
        for ev in evidence:
            pt = " ".join(str(ev["text"]).lower().split())
            if pt and (_norm == pt or _norm in pt):
                return False, []
        report, outcomes = promote_if_faithful(
            km_path, question=question, answer=str(resolution), evidence_passages=evidence,
            client=client, faithfulness=faithfulness, provenance=provenance, contents=contents)
        return bool(any(o.promoted for o in outcomes)), outcomes
    except Exception:
        return False, []
