"""curator_reconcile — the contradiction ACT layer (judge the geometry candidates, surface real contradictions).

`curator.reconcile.judge_contradictions` stance-judges the corpus's high-cosine same-topic pairs and returns only
the pairs the judge rules `contradict`. Pins:

  (a) a planted contradiction (24h vs 48h, cross-doc) + a judge that says "contradict" -> 1 confirmed finding with
      the cosine, both source_files, the authority hint, and the reason.
  (b) the SAME high-cosine pair + a judge that says "paraphrase" -> 0 confirmed (geometry alone never confirms).
  (c) no client -> []; a parse-error verdict -> not confirmed (best-effort).

Real encoder + tiny corpus (the geometry candidate must be real); the JUDGE is a deterministic stub (no API).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ._testutil import build_corpus as _build
from ._testutil import unpatch_zero_encoder


class _Block:
    def __init__(self, text): self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _StubJudge:
    """Returns a canned stance verdict for the stance-judge system prompt; `.messages.create` shape only."""

    def __init__(self, verdict_json: str):
        self._json = verdict_json
        self.messages = self

    def create(self, *, model, max_tokens, system, messages, **kwargs):
        return _Resp(self._json)


_CONTRADICT = json.dumps({"stance": "contradict", "more_authoritative": "a", "reason": "24h vs 48h disagree"})
_PARAPHRASE = json.dumps({"stance": "paraphrase", "more_authoritative": "unclear", "reason": "same meaning"})

_CONTRA_FILES = {
    "a.md": "Session tokens expire after 24 hours and cannot be renewed.",
    "b.md": "Session tokens expire after 48 hours and cannot be renewed.",
    "c.md": "The billing cycle runs monthly and invoices issue on the first.",
}


def _check_confirmed() -> int:
    from resonance_lattice.curator.reconcile import judge_contradictions
    with tempfile.TemporaryDirectory() as d:
        km = _build(Path(d) / "corpus", _CONTRA_FILES)
        confirmed = judge_contradictions(km, _StubJudge(_CONTRADICT), min_cosine=0.85)
        if len(confirmed) != 1:
            print(f"[curator_reconcile] (a) expected 1 confirmed contradiction, got {len(confirmed)}",
                  file=sys.stderr); return 1
        f = confirmed[0]
        files = {f["a"]["source_file"], f["b"]["source_file"]}
        if files != {"a.md", "b.md"}:
            print(f"[curator_reconcile] (a) wrong pair confirmed: {files}", file=sys.stderr); return 1
        if f["more_authoritative"] != "a" or not f["reason"] or not f.get("cosine"):
            print(f"[curator_reconcile] (a) finding missing fields: {f}", file=sys.stderr); return 1
    return 0


def _check_paraphrase_not_confirmed() -> int:
    from resonance_lattice.curator.reconcile import judge_contradictions
    with tempfile.TemporaryDirectory() as d:
        km = _build(Path(d) / "corpus", _CONTRA_FILES)
        if judge_contradictions(km, _StubJudge(_PARAPHRASE), min_cosine=0.85):
            print("[curator_reconcile] (b) a paraphrase verdict must not confirm a contradiction", file=sys.stderr)
            return 1
    return 0


def _check_degrade() -> int:
    from resonance_lattice.curator.reconcile import judge_contradictions
    with tempfile.TemporaryDirectory() as d:
        km = _build(Path(d) / "corpus", _CONTRA_FILES)
        if judge_contradictions(km, None, min_cosine=0.85) != []:
            print("[curator_reconcile] (c) no client must yield []", file=sys.stderr); return 1
        if judge_contradictions(km, _StubJudge("not json at all"), min_cosine=0.85) != []:
            print("[curator_reconcile] (c) a parse-error verdict must not confirm", file=sys.stderr); return 1
    return 0


def _check_reconcile_write() -> int:
    """reconcile_contradiction records a high-trust RESOLUTION claim citing BOTH passages, NON-destructively."""
    from resonance_lattice.curator.reconcile import reconcile_contradiction
    from resonance_lattice.store import archive
    from resonance_lattice.store.insight import beta_mean, confidence_band
    from resonance_lattice.store.self_audit import find_contradiction_candidates
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, _CONTRA_FILES)
        cands = find_contradiction_candidates(km, min_cosine=0.85, source_root=str(root))
        pair = next((c for c in cands if {c.a["source_file"], c.b["source_file"]} == {"a.md", "b.md"}), None)
        if pair is None:
            print("[curator_reconcile] (recon) no a/b candidate to reconcile", file=sys.stderr); return 1
        ia, ib = pair.a["passage_idx"], pair.b["passage_idx"]
        before = {c.passage_id: c.content_hash for c in archive.read(km).registry}
        src_before = {f.name: f.read_bytes() for f in sorted(root.glob("*.md"))}
        # a VERBATIM copy of a cited passage is not a resolution synthesis -> must NOT land
        if reconcile_contradiction(km, ia, ib, pair.a["text"], source_root=str(root),
                                   client=None, faithfulness=0.92)[0]:
            print("[curator_reconcile] (recon) a verbatim copy of a passage must NOT land", file=sys.stderr)
            return 1
        resolution = ("Authoritative: session tokens expire after 24 hours per a.md, the primary source; the "
                      "48-hour figure stated in b.md is superseded and should not be relied on.")
        landed, outcomes = reconcile_contradiction(
            km, ia, ib, resolution, source_root=str(root), client=None, faithfulness=0.92, provenance="user")
        if not landed:
            print(f"[curator_reconcile] (recon) resolution did not land: {outcomes}", file=sys.stderr); return 1
        contents = archive.read(km)
        # NON-DESTRUCTIVE on disk: the corpus source files are byte-identical before/after
        src_after = {f.name: f.read_bytes() for f in sorted(root.glob("*.md"))}
        if src_after != src_before:
            print("[curator_reconcile] (recon) corpus SOURCE FILES must be byte-identical (non-destructive)",
                  file=sys.stderr); return 1
        res = next((c for c in contents.insights if c.content == resolution), None)
        if res is None:
            print("[curator_reconcile] (recon) resolution claim not in band", file=sys.stderr); return 1
        cited = {cit.passage_id for cit in res.facts.citations}
        expected = {contents.registry[ia].passage_id, contents.registry[ib].passage_id}
        if not expected <= cited:
            print(f"[curator_reconcile] (recon) resolution must cite both passages: cited={cited} "
                  f"expected={expected}", file=sys.stderr); return 1
        after = {c.passage_id: c.content_hash for c in contents.registry}
        if before != after:
            print("[curator_reconcile] (recon) corpus passages must be UNCHANGED (non-destructive)",
                  file=sys.stderr); return 1
        if confidence_band(beta_mean(res.corroboration, res.falsification)) not in ("high", "verified"):
            print("[curator_reconcile] (recon) resolution should seed high trust (user tier)", file=sys.stderr)
            return 1
        # an empty resolution must refuse
        if reconcile_contradiction(km, ia, ib, "  ", source_root=str(root), client=None, faithfulness=0.9)[0]:
            print("[curator_reconcile] (recon) empty resolution must NOT land", file=sys.stderr); return 1
    return 0


def run() -> int:
    unpatch_zero_encoder()
    for check in (_check_confirmed, _check_paraphrase_not_confirmed, _check_degrade, _check_reconcile_write):
        rc = check()
        if rc != 0:
            return rc
    print("[curator_reconcile] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
