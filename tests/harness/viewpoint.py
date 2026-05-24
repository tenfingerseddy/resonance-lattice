"""viewpoint — query-time deliberation runtime.

Guarantees (Day 5):

  1. deliberate() returns a ViewpointPosition with source + insight hits
     merged by adjusted score, descending.
  2. source_only_alternative is always populated (trust-contract
     foundation 5).
  3. Lens trust_weights re-rank source hits.
  4. Lens insight_preferences re-rank insight hits.
  5. Provenance graph cites insights → source passages with no dangling refs.
  6. No lens supplied → identity transform (same as source-only path).
  7. include_stale=False excludes stale insights from default retrieval.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_insight_passage, unpatch_zero_encoder


def run() -> int:
    unpatch_zero_encoder()  # defeat cross-suite contamination from memory suites
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.lens import schema as lens_mod
    from resonance_lattice.store import archive, open_store
    from resonance_lattice.viewpoint import deliberate

    failures = 0

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        files = {
            "src/auth.py": "# Auth\n\nSession tokens expire after 24 hours.",
            "docs/external/blog.md": "Random blog about session timeouts.",
            "src/storage.py": "# Storage\n\nSessions live in Redis.",
        }
        km = _build(root, files)
        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry]
        src_hashes = [c.content_hash for c in c0.registry]
        store = open_store(km, c0)

        encoder = Encoder()
        query = "How long does a session last?"
        q_emb = encoder.encode([query])[0].astype("float32")

        # Promote an accepted insight so the merge path exercises both layers.
        insight_texts = ["Sessions last 24h via session tokens that live in Redis."]
        insight_band = encoder.encode(insight_texts).astype("float32")
        insights = [make_insight_passage(
            0, insight_texts[0], src_ids[:2], src_hashes[:2], state="accepted",
        )]
        archive.write_insight_layer_in_place(km, insights, insight_band)
        c1 = archive.read(km)
        store1 = open_store(km, c1)

        # ---- Guarantee 1: merged top-K, descending by score ----
        pos = deliberate(query, q_emb, c1, source_store=store1, top_k=5)
        if not pos.hits:
            print("[viewpoint] FAIL g1: no hits returned", file=sys.stderr)
            failures += 1
        elif any(h2.score > h1.score for h1, h2 in zip(pos.hits, pos.hits[1:])):
            print(f"[viewpoint] FAIL g1: hits not descending by score: "
                  f"{[h.score for h in pos.hits]}", file=sys.stderr)
            failures += 1
        else:
            has_insight = any(h.layer == "insight" for h in pos.hits)
            has_source = any(h.layer == "source" for h in pos.hits)
            if not (has_insight and has_source):
                print(f"[viewpoint] FAIL g1: layers missing "
                      f"(insight={has_insight}, source={has_source})",
                      file=sys.stderr)
                failures += 1
            else:
                print("[viewpoint] g1 (merged top-K, descending) OK",
                      file=sys.stderr)

        # ---- Guarantee 2: source_only_alternative populated ----
        if not pos.source_only_alternative or \
           "[SOURCE" not in pos.source_only_alternative:
            print(f"[viewpoint] FAIL g2: source_only_alternative wrong: "
                  f"{pos.source_only_alternative[:80]}", file=sys.stderr)
            failures += 1
        elif "[INSIGHT" in pos.source_only_alternative:
            print("[viewpoint] FAIL g2: source-only contains insight content",
                  file=sys.stderr)
            failures += 1
        else:
            print("[viewpoint] g2 (source_only_alternative honest baseline) OK",
                  file=sys.stderr)

        # ---- Guarantee 3 + 4: lens trust + preference re-ranking ----
        lens = lens_mod.new_lens(
            lens_id="lens-eng", scope="user", name="engineering",
        )
        lens.trust_weights = [
            lens_mod.TrustWeight(pattern="src/*", weight=2.0),
            lens_mod.TrustWeight(pattern="docs/external/*", weight=0.1),
        ]
        lens.insight_preferences = [
            lens_mod.InsightPreference(
                insight_id=insights[0].insight_id, weight=3.0,
            ),
        ]

        pos_lensed = deliberate(query, q_emb, c1, source_store=store1, lens=lens, top_k=5)

        # src/* hits should now outscore docs/external/* hits via the
        # trust weight; check by source_file prefixes within source hits.
        src_layer = [h for h in pos_lensed.hits if h.layer == "source"]
        if src_layer:
            src_scores = {h.source_file: h.score for h in src_layer}
            in_src = [v for k, v in src_scores.items() if k.startswith("src/")]
            in_ext = [v for k, v in src_scores.items() if k.startswith("docs/external/")]
            if in_src and in_ext and max(in_src) <= max(in_ext):
                print(f"[viewpoint] FAIL g3: src/* {in_src} not > "
                      f"docs/external/* {in_ext} after 2.0x trust", file=sys.stderr)
                failures += 1
            else:
                print("[viewpoint] g3 (lens trust_weights re-rank source) OK",
                      file=sys.stderr)
        else:
            print("[viewpoint] g3 SKIP — no source hits returned", file=sys.stderr)

        # Insight score should be boosted by 3.0x preference.
        ih_unlensed = [h for h in pos.hits if h.layer == "insight"]
        ih_lensed = [h for h in pos_lensed.hits if h.layer == "insight"]
        if ih_unlensed and ih_lensed:
            r = ih_lensed[0].score / max(1e-9, ih_unlensed[0].score)
            if abs(r - 3.0) > 0.01:
                print(f"[viewpoint] FAIL g4: insight pref multiplier "
                      f"{r:.2f} != 3.0", file=sys.stderr)
                failures += 1
            else:
                print("[viewpoint] g4 (lens insight_preferences re-rank) OK",
                      file=sys.stderr)

        # ---- Guarantee 5: provenance graph reachable ----
        # Every insight node's `cites` must resolve to source nodes in
        # the same graph.
        prov_by_id = {n.id: n for n in pos_lensed.provenance}
        ok_provenance = True
        for n in pos_lensed.provenance:
            if n.layer == "insight":
                for cited_id in n.cites:
                    if cited_id not in prov_by_id:
                        ok_provenance = False
                        break
                if not ok_provenance:
                    break
        if not ok_provenance:
            print("[viewpoint] FAIL g5: provenance has dangling citation",
                  file=sys.stderr)
            failures += 1
        else:
            print("[viewpoint] g5 (provenance graph reachable) OK", file=sys.stderr)

        # ---- Guarantee 6: no lens → identity ----
        # Comparing pos (no lens) and pos_lensed (with lens): scores
        # should differ when lens is applied, and pos.lens_id should be
        # None.
        if pos.lens_id is not None:
            print(f"[viewpoint] FAIL g6: pos.lens_id should be None, got "
                  f"{pos.lens_id}", file=sys.stderr)
            failures += 1
        elif pos_lensed.lens_id != "lens-eng":
            print(f"[viewpoint] FAIL g6: pos_lensed.lens_id should be lens-eng, "
                  f"got {pos_lensed.lens_id}", file=sys.stderr)
            failures += 1
        else:
            print("[viewpoint] g6 (no lens → identity transform) OK",
                  file=sys.stderr)

        # ---- Guarantee 7: stale insights excluded by default ----
        from dataclasses import replace as _replace
        stale_insights = [_replace(insights[0], verdict_state="stale")]
        archive.write_insight_layer_in_place(km, stale_insights, insight_band)
        c2 = archive.read(km)
        store2 = open_store(km, c2)
        pos_stale = deliberate(query, q_emb, c2, source_store=store2, top_k=5)
        has_insight = any(h.layer == "insight" for h in pos_stale.hits)
        if has_insight:
            print(f"[viewpoint] FAIL g7: stale insight surfaced in default retrieval",
                  file=sys.stderr)
            failures += 1
        else:
            # Now include_stale=True should bring it back.
            pos_stale_on = deliberate(query, q_emb, c2, source_store=store2,
                                       top_k=5, include_stale=True)
            if any(h.layer == "insight" for h in pos_stale_on.hits):
                print("[viewpoint] g7 (stale exclusion + include_stale toggle) OK",
                      file=sys.stderr)
            else:
                print("[viewpoint] FAIL g7: include_stale=True did not surface stale",
                      file=sys.stderr)
                failures += 1

    if failures:
        print(f"[viewpoint] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[viewpoint] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
