"""self_audit — the corpus self-audit geometry primitives (contradiction candidates).

`store.self_audit.find_contradiction_candidates` narrows the corpus to high-cosine CROSS-DOCUMENT passage pairs —
the handful a stance judge then rules on. Pins:

  (a) a planted contradiction (two docs, same topic, opposing value) surfaces as a cross-file candidate; the
      unrelated doc never pairs with it.
  (b) a corpus with no close cross-doc pair yields no candidates.
  (c) same-file pairs are never returned (a doc restating itself is not a contradiction to surface).

Real encoder + real tiny corpora (the geometry must be real cosines).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ._testutil import build_corpus as _build
from ._testutil import unpatch_zero_encoder


def _check_planted() -> int:
    from resonance_lattice.store.self_audit import find_contradiction_candidates
    with tempfile.TemporaryDirectory() as d:
        km = _build(Path(d) / "corpus", {
            "a.md": "Session tokens expire after 24 hours and cannot be renewed.",
            "b.md": "Session tokens expire after 48 hours and cannot be renewed.",
            "c.md": "The billing cycle runs monthly and invoices issue on the first of the month.",
        })
        cands = find_contradiction_candidates(km, min_cosine=0.85)
        if not cands:
            print("[self_audit] (a) expected the planted contradiction, got none", file=sys.stderr)
            return 1
        top = cands[0]
        files = {top.a["source_file"], top.b["source_file"]}
        if files != {"a.md", "b.md"}:
            print(f"[self_audit] (a) top candidate is not the a/b contradiction: {files}", file=sys.stderr)
            return 1
        if top.a["source_file"] == top.b["source_file"]:
            print("[self_audit] (c) returned a same-file pair", file=sys.stderr)
            return 1
        # the unrelated billing doc must not be paired with anything
        if any("c.md" in (c.a["source_file"], c.b["source_file"]) for c in cands):
            print("[self_audit] (a) the unrelated doc was wrongly paired", file=sys.stderr)
            return 1
        if not (top.a.get("text") and top.b.get("text")):
            print("[self_audit] (a) candidate text did not resolve", file=sys.stderr)
            return 1
    return 0


def _check_no_pairs() -> int:
    from resonance_lattice.store.self_audit import find_contradiction_candidates
    with tempfile.TemporaryDirectory() as d:
        km = _build(Path(d) / "corpus", {
            "x.md": "Photosynthesis converts sunlight into chemical energy in plants.",
            "y.md": "The quarterly tax filing deadline falls on the fifteenth.",
        })
        if find_contradiction_candidates(km, min_cosine=0.9):
            print("[self_audit] (b) unrelated docs should yield no candidates", file=sys.stderr)
            return 1
    return 0


def _check_drift() -> int:
    from resonance_lattice.store.self_audit import find_drifted_passages
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, {
            "a.md": "Session tokens expire after 24 hours and cannot be renewed.",
            "b.md": "The billing cycle runs monthly and invoices issue on the first.",
        })
        if find_drifted_passages(km, source_root=str(root)):
            print("[self_audit] (drift) fresh corpus should have no drift", file=sys.stderr)
            return 1
        # mutate one source so its content hash no longer matches the build-time record
        (root / "a.md").write_text("Session tokens now expire after 72 hours instead.", encoding="utf-8")
        drifted = find_drifted_passages(km, source_root=str(root))
        if not any(dp.source_file == "a.md" and dp.drift_status == "drifted" for dp in drifted):
            print(f"[self_audit] (drift) mutated source not flagged: {drifted}", file=sys.stderr)
            return 1
        if any(dp.source_file == "b.md" for dp in drifted):
            print("[self_audit] (drift) unchanged source wrongly flagged", file=sys.stderr)
            return 1
    return 0


def _check_store_roundtrip() -> int:
    from resonance_lattice.store import archive
    from resonance_lattice.store.self_audit import compute_self_audit
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, {
            "a.md": "Session tokens expire after 24 hours and cannot be renewed.",
            "b.md": "Session tokens expire after 48 hours and cannot be renewed.",
            "c.md": "The billing cycle runs monthly and invoices issue on the first.",
        })
        report = compute_self_audit(km, min_cosine=0.85, source_root=str(root),
                                    gaps=[{"intent_cluster": 0, "occurrences": 3}])
        archive.write_self_audit_in_place(km, report)
        back = archive.read_self_audit(km)
        if back.get("version") != 1:
            print(f"[self_audit] (store) version missing: {back.get('version')}", file=sys.stderr); return 1
        if back["counts"]["high_cosine_pairs"] < 1:
            print("[self_audit] (store) high-cosine pair not in stored report", file=sys.stderr); return 1
        if back["counts"]["gaps"] != 1:
            print(f"[self_audit] (store) passed-in gap not stored: {back['counts']}", file=sys.stderr); return 1
        if back.get("pairs_skipped") is not False:
            print(f"[self_audit] (store) pairs_skipped should be False on a tiny corpus: {back.get('pairs_skipped')}",
                  file=sys.stderr); return 1
        # the stored candidate is COMPACT — indices + source_file, no text
        c0 = back["high_cosine_pairs"][0]
        if "text" in c0["a"] or "passage_idx" not in c0["a"]:
            print(f"[self_audit] (store) candidate not compact: {c0}", file=sys.stderr); return 1
        # every other member survived the in-place write — the corpus still reads + retrieves
        contents = archive.read(km)
        if not contents.registry or contents.bands["base"].shape[0] < 1:
            print("[self_audit] (store) in-place write damaged the corpus", file=sys.stderr); return 1
    return 0


def _check_demand_rank() -> int:
    """Demand-ranking (geometry × telemetry): the conflict in the path of real query traffic ranks FIRST."""
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store import archive
    from resonance_lattice.store.self_audit import (
        find_contradiction_candidates,
        rank_contradictions_by_demand,
    )
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, {
            "tok_a.md": "Session tokens expire after 24 hours and cannot be renewed.",
            "tok_b.md": "Session tokens expire after 48 hours and cannot be renewed.",
            "col_a.md": "The primary widget casing colour is red on all shipped units.",
            "col_b.md": "The primary widget casing colour is blue on all shipped units.",
        })
        cands = find_contradiction_candidates(km, min_cosine=0.85, source_root=str(root))
        files_seen = [{c.a["source_file"], c.b["source_file"]} for c in cands]
        if {"tok_a.md", "tok_b.md"} not in files_seen or {"col_a.md", "col_b.md"} not in files_seen:
            print(f"[self_audit] (demand) expected both conflicts as candidates: {files_seen}", file=sys.stderr)
            return 1
        # capture query traffic about TOKENS only
        qv = Encoder().encode([
            "how long until a session token expires",
            "session token expiry duration",
            "when do my tokens expire",
        ])
        rows = [{"ts": "t", "session": "s", "layer": "source", "is_user_query": True,
                 "query_emb": [round(float(x), 6) for x in v], "ranked": []} for v in qv]
        archive.append_telemetry_in_place(km, rows)
        contents = archive.read(km)
        ranked = rank_contradictions_by_demand(cands, contents, archive.read_telemetry(km))
        top = {ranked[0].a["source_file"], ranked[0].b["source_file"]}
        if top != {"tok_a.md", "tok_b.md"}:
            print(f"[self_audit] (demand) queried conflict must rank first, got {top}", file=sys.stderr)
            return 1
        # no telemetry -> order unchanged (cosine), and the function is non-mutating
        if [id(c) for c in rank_contradictions_by_demand(cands, contents, [])] != [id(c) for c in cands]:
            print("[self_audit] (demand) empty telemetry must leave the order unchanged", file=sys.stderr)
            return 1
    return 0


def run() -> int:
    unpatch_zero_encoder()
    for check in (_check_planted, _check_no_pairs, _check_drift, _check_store_roundtrip, _check_demand_rank):
        rc = check()
        if rc != 0:
            return rc
    print("[self_audit] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
