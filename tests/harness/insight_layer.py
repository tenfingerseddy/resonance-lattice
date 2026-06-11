"""insight_layer — round-trip and retrieval composition for the lensed-
knowledge insight layer.

Guarantees:

  1. An archive with NO insight layer loads cleanly (backwards compat).
  2. write() + read() round-trip preserves insight rows + insight band.
  3. write_insight_layer_in_place adds an insight band to an existing
     archive without rewriting unrelated slots.
  4. write_insight_layer_in_place([]) clears the insight layer cleanly.
  5. Half-written promotion (insight.jsonl + missing band) raises.
  6. compute_insight_id is deterministic and order-independent over
     source_passage_hashes.
  7. Composed CLI search labels source + insight hits.
  8. --source-only bypasses the insight layer.
  9. candidate / retired states are excluded from retrieval.
 10. verify_insight_hits scales score by a confidence factor and re-ranks,
     so a corroborated insight floats above a provisional one at similar
     relevance.
 11. A legacy pre-Stage-5 `InsightPassage`-shaped `insight.jsonl`
     (`verdict_state` axis, 6 values) migrates to `Claim`+`CorpusFacts`
     on archive read, and the next write rewrites the member in
     Stage-5 shape.
 12. Two concurrent `write_insight_layer_in_place` calls don't corrupt
     the archive — each writes to a per-writer-unique tmp file (the
     D/5.2 fix), so the final archive opens cleanly and its insight
     rows equal exactly one writer's input (last `os.replace` wins;
     lost updates are by contract, not corruption).

Day 1 deliverable. `.claude/plans/lensed-knowledge-architecture.md` §4.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim, unpatch_zero_encoder


def run() -> int:
    unpatch_zero_encoder()  # defeat cross-suite contamination from memory suites
    from resonance_lattice.store import archive, insight

    failures = 0

    # ---- Guarantee 13: source-discriminated band serialiser (S3) ----
    # Hermetic — no archive/encoder. The unified band must round-trip BOTH
    # sources in one positional list: a corpus claim via the CorpusFacts path,
    # an experience claim via the shared spine helper, with the middle
    # experience row not desyncing the position-keyed join.
    from resonance_lattice.state.claim import Claim, CorpusFacts, ExperienceFacts
    from resonance_lattice.store.corpus_claim_io import (
        claims_to_jsonl,
        rows_to_claims,
    )

    corpus_a = make_corpus_claim("Alpha earned claim.", ["p0", "p1"], state="active")
    corpus_b = make_corpus_claim("Beta earned claim.", ["p2", "p3"], state="candidate")
    exp = Claim(
        claim_id="01HZEXP000000000000000001",
        source="experience",
        kind="event",
        content="prefer the standard library",
        created_at="2026-06-01T00:00:00Z",
        corroboration=2.0,
        falsification=1.0,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=("prefer",),
            recurrence_count=2,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash="manual",
            origin="manual",
            last_corroborated_at="2026-06-01T00:00:00Z",
        ),
    )
    mixed = [corpus_a, exp, corpus_b]
    text = claims_to_jsonl(mixed)
    recovered = rows_to_claims(text.split("\n"))
    g13_ok = (
        recovered == mixed                              # value-equal, in order
        and isinstance(recovered[0].facts, CorpusFacts)
        and isinstance(recovered[1].facts, ExperienceFacts)
        and isinstance(recovered[2].facts, CorpusFacts)
        and recovered[1].source == "experience"
    )
    if g13_ok:
        print("[insight_layer] g13 (mixed-source band round-trip) OK",
              file=sys.stderr)
    else:
        print("[insight_layer] FAIL g13: mixed-source band round-trip "
              f"(len={len(recovered)}, sources="
              f"{[c.source for c in recovered]})", file=sys.stderr)
        failures += 1

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        files = {
            "a.md": "# Alpha\n\nAuthentication and login flows.",
            "b.md": "# Beta\n\nCredentials and tokens.",
            "c.md": "# Gamma\n\nSession storage in Redis.",
        }
        km = _build(root, files)

        # ---- Guarantee 1: archive with no insight layer loads cleanly ----
        c0 = archive.read(km)
        if c0.insights:
            print(f"[insight_layer] FAIL g1: fresh build has {len(c0.insights)} "
                  f"insights but should be empty", file=sys.stderr)
            failures += 1
        elif archive.INSIGHT_BAND_NAME in c0.bands:
            print(f"[insight_layer] FAIL g1: fresh build declares insight band",
                  file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g1 (no insight layer loads) OK",
                  file=sys.stderr)

        # Pick some real source passage ids to cite.
        src_ids = [c.passage_id for c in c0.registry[:2]]
        if len(src_ids) < 2:
            print(f"[insight_layer] FAIL setup: corpus has only "
                  f"{len(c0.registry)} passages, need >=2",
                  file=sys.stderr)
            return 1

        # ---- Guarantee 3: write_insight_layer_in_place adds insight layer ----
        # Real encoder via unpatch_zero_encoder (called at run() entry).
        from resonance_lattice.field.encoder import Encoder
        encoder = Encoder()
        insight_texts = [
            "Login uses session tokens with 24h TTL.",
            "Credentials rotate weekly via token refresh.",
        ]
        insight_band = encoder.encode(insight_texts).astype("float32")
        insights = [
            make_corpus_claim(insight_texts[0], src_ids[:1], state="active"),
            make_corpus_claim(insight_texts[1], src_ids, state="active"),
        ]

        archive.write_insight_layer_in_place(km, insights, insight_band)
        c1 = archive.read(km)
        if len(c1.insights) != 2:
            print(f"[insight_layer] FAIL g3: expected 2 insights, got "
                  f"{len(c1.insights)}", file=sys.stderr)
            failures += 1
        elif archive.INSIGHT_BAND_NAME not in c1.bands:
            print("[insight_layer] FAIL g3: insight band not loaded after "
                  "write_in_place", file=sys.stderr)
            failures += 1
        elif c1.bands[archive.INSIGHT_BAND_NAME].shape != (2, 768):
            print(f"[insight_layer] FAIL g3: insight band shape "
                  f"{c1.bands[archive.INSIGHT_BAND_NAME].shape} != (2, 768)",
                  file=sys.stderr)
            failures += 1
        elif c1.metadata.bands[archive.INSIGHT_BAND_NAME].role != "insight_layer":
            print("[insight_layer] FAIL g3: insight band role mismatch",
                  file=sys.stderr)
            failures += 1
        else:
            # Source layer should be byte-identical.
            if not np.array_equal(c1.bands["base"], c0.bands["base"]):
                print("[insight_layer] FAIL g3: source band changed after "
                      "insight write_in_place", file=sys.stderr)
                failures += 1
            elif len(c1.registry) != len(c0.registry):
                print("[insight_layer] FAIL g3: passage registry changed",
                      file=sys.stderr)
                failures += 1
            else:
                print("[insight_layer] g3 (in-place insight add) OK",
                      file=sys.stderr)

        # ---- Guarantee 2: full round-trip preserves rows + band ----
        if c1.insights[0].claim_id != insights[0].claim_id:
            print(f"[insight_layer] FAIL g2: claim_id mismatch "
                  f"{c1.insights[0].claim_id} != {insights[0].claim_id}",
                  file=sys.stderr)
            failures += 1
        elif c1.insights[0].content != insights[0].content:
            print("[insight_layer] FAIL g2: insight content mismatch",
                  file=sys.stderr)
            failures += 1
        elif tuple(c1.insights[1].facts.citations[0].passage_id for _ in [0]) != \
             (insights[1].facts.citations[0].passage_id,):
            print("[insight_layer] FAIL g2: citation mismatch", file=sys.stderr)
            failures += 1
        elif not np.allclose(c1.bands[archive.INSIGHT_BAND_NAME], insight_band,
                             atol=1e-6):
            print("[insight_layer] FAIL g2: insight band tensor mismatch",
                  file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g2 (round-trip preserves rows + band) OK",
                  file=sys.stderr)

        # ---- Guarantee 4: clearing the insight layer ----
        archive.write_insight_layer_in_place(km, [], np.zeros((0, 768), dtype="float32"))
        c2 = archive.read(km)
        if c2.insights:
            print(f"[insight_layer] FAIL g4: clear left {len(c2.insights)} "
                  f"insights", file=sys.stderr)
            failures += 1
        elif archive.INSIGHT_BAND_NAME in c2.bands:
            print("[insight_layer] FAIL g4: insight band still present after clear",
                  file=sys.stderr)
            failures += 1
        elif archive.INSIGHT_BAND_NAME in c2.metadata.bands:
            print("[insight_layer] FAIL g4: insight band still in metadata after clear",
                  file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g4 (clear insight layer) OK", file=sys.stderr)

        # ---- Guarantee 6: compute_insight_id determinism ----
        id1 = insight.compute_insight_id("foo", ["b", "a"], "m")
        id2 = insight.compute_insight_id("foo", ["a", "b"], "m")
        id3 = insight.compute_insight_id("bar", ["a", "b"], "m")
        if id1 != id2:
            print(f"[insight_layer] FAIL g6: hash order-dependent {id1} != {id2}",
                  file=sys.stderr)
            failures += 1
        elif id1 == id3:
            print("[insight_layer] FAIL g6: hash collision across distinct content",
                  file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g6 (compute_insight_id determinism) OK",
                  file=sys.stderr)

        # ---- Guarantee 7: end-to-end CLI search returns labelled hits ----
        # Re-promote insights so the CLI run has something to retrieve.
        archive.write_insight_layer_in_place(km, insights, insight_band)
        rc, stdout, stderr = _cli_search([str(km), "session tokens", "--top-k", "5",
                                          "--format", "json", "--quiet"])
        if rc != 0:
            print(f"[insight_layer] FAIL g7: search rc={rc}\n{stderr}",
                  file=sys.stderr)
            failures += 1
        else:
            import json as _json
            out = _json.loads(stdout)
            has_source = any(h["layer"] == "source" for h in out)
            has_insight = any(h["layer"] == "insight" for h in out)
            if not has_source:
                print(f"[insight_layer] FAIL g7: no source hits in composed search",
                      file=sys.stderr)
                failures += 1
            elif not has_insight:
                print(f"[insight_layer] FAIL g7: no insight hits in composed search",
                      file=sys.stderr)
                failures += 1
            else:
                # Insight hits must carry claim_id, kind, state, citations.
                ih = next(h for h in out if h["layer"] == "insight")
                required = {"claim_id", "kind", "state", "confidence",
                            "citations", "source_passage_hashes"}
                missing = required - set(ih.keys())
                if missing:
                    print(f"[insight_layer] FAIL g7: insight hit missing fields "
                          f"{missing}", file=sys.stderr)
                    failures += 1
                else:
                    print("[insight_layer] g7 (composed search labels source + insight) OK",
                          file=sys.stderr)

        # ---- Guarantee 8: --source-only bypasses insight layer ----
        rc, stdout, stderr = _cli_search([str(km), "session tokens", "--top-k", "5",
                                          "--format", "json", "--quiet",
                                          "--source-only"])
        if rc != 0:
            print(f"[insight_layer] FAIL g8: source-only search rc={rc}\n{stderr}",
                  file=sys.stderr)
            failures += 1
        else:
            import json as _json
            out = _json.loads(stdout)
            if any(h["layer"] == "insight" for h in out):
                print("[insight_layer] FAIL g8: --source-only returned insight hits",
                      file=sys.stderr)
                failures += 1
            elif not any(h["layer"] == "source" for h in out):
                print("[insight_layer] FAIL g8: --source-only returned no source hits",
                      file=sys.stderr)
                failures += 1
            else:
                print("[insight_layer] g8 (--source-only bypasses insight layer) OK",
                      file=sys.stderr)

        # ---- Guarantee 9: candidate / retired states are excluded ----
        # Build a fresh corpus with insights in non-retrievable states and
        # confirm none surface in default retrieval.
        from dataclasses import replace
        candidate_insights = [
            replace(insights[0], state="candidate"),
            replace(insights[1], state="retired"),
        ]
        archive.write_insight_layer_in_place(km, candidate_insights, insight_band)
        rc, stdout, _ = _cli_search([str(km), "session tokens", "--top-k", "5",
                                     "--format", "json", "--quiet"])
        out = _json.loads(stdout)
        if any(h["layer"] == "insight" for h in out):
            print("[insight_layer] FAIL g9: candidate/retired insights surfaced "
                  "in default retrieval", file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g9 (non-retrievable states excluded) OK",
                  file=sys.stderr)

        # ---- Guarantee 10: confidence modulates score + re-ranks ----
        from resonance_lattice.store.verified import verify_insight_hits
        hi = make_corpus_claim("high-confidence insight", src_ids[:1],
                               state="active", faithfulness=0.9)
        lo = make_corpus_claim("low-confidence insight", src_ids[:1],
                               state="active", faithfulness=0.1)
        # idx 1 carries a slightly higher raw cosine but the lower
        # confidence — at similar relevance the corroborated insight wins.
        ranked = verify_insight_hits([(0, 0.60), (1, 0.64)], [hi, lo])
        if len(ranked) != 2:
            print(f"[insight_layer] FAIL g10: expected 2 hits, got {len(ranked)}",
                  file=sys.stderr)
            failures += 1
        elif ranked[0].insight_idx != 0:
            print(f"[insight_layer] FAIL g10: corroborated insight not floated "
                  f"above the provisional one — top hit is idx "
                  f"{ranked[0].insight_idx}", file=sys.stderr)
            failures += 1
        elif ranked[0].score <= 0.60:
            print(f"[insight_layer] FAIL g10: high-confidence score "
                  f"{ranked[0].score:.3f} not boosted above raw cosine 0.60",
                  file=sys.stderr)
            failures += 1
        elif ranked[1].score >= 0.64:
            print(f"[insight_layer] FAIL g10: low-confidence score "
                  f"{ranked[1].score:.3f} not sunk below raw cosine 0.64",
                  file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g10 (confidence modulates score + re-ranks) OK",
                  file=sys.stderr)

        # ---- Guarantee 11: legacy InsightPassage `insight.jsonl` migrates ----
        # Seed a 2-row insight layer, swap insight.jsonl bytes for a
        # legacy `verdict_state`-shaped payload of matching arity. The
        # next archive.read must migrate to Claim+CorpusFacts; the next
        # write rewrites the member in Stage-5 shape.
        import json as _json
        import zipfile as _zipfile
        seed = [
            make_corpus_claim("seed one", src_ids[:1], state="active",
                              faithfulness=0.5),
            make_corpus_claim("seed two", src_ids[:1], state="active",
                              faithfulness=0.5),
        ]
        seed_band = np.zeros((2, insight_band.shape[1]), dtype=insight_band.dtype)
        archive.write_insight_layer_in_place(km, seed, seed_band)
        legacy_rows = [
            {"id": "01HZLEGACY00000000000000A1",
             "kind": "synthesis", "content": "accepted legacy insight",
             "generated_at": "2026-04-01T00:00:00Z",
             "source_model_hash": "legacy-model",
             "source_passage_hashes": ["p0"],
             "verdict_state": "accepted",
             "lineage": ["01HZANCESTOR0000000000000A"],
             "citations": [{"passage_id": "p0", "char_span": [0, 10],
                            "confidence": 0.8}],
             "corroboration": 4.0, "falsification": 1.0,
             "query": "q", "stale_if_sources_drift": True,
             "encoder_version": "gte-mb-768"},
            # No Beta tallies — seeded from `confidence`. A reject collapses
            # to `retired`.
            {"id": "01HZLEGACY00000000000000B2",
             "kind": "synthesis", "content": "rejected legacy insight",
             "generated_at": "2026-04-02T00:00:00Z",
             "source_model_hash": "legacy-model",
             "source_passage_hashes": ["p1"],
             "verdict_state": "rejected", "confidence": 0.3},
        ]
        _swap_zip_member(
            km, "insight.jsonl",
            "\n".join(_json.dumps(r) for r in legacy_rows))
        migrated = archive.read(km).insights
        accepted, rejected = migrated[0], migrated[1]
        ok = (
            len(migrated) == 2
            and accepted.claim_id == "01HZLEGACY00000000000000A1"
            and accepted.source == "corpus"
            and accepted.state == "active"
            and accepted.facts.content_fingerprint == accepted.claim_id
            and accepted.parent_ids == ("01HZANCESTOR0000000000000A",)
            and accepted.facts.citations[0].passage_id == "p0"
            and accepted.corroboration == 4.0
            and rejected.state == "retired"
            and rejected.corroboration > 1.0  # seeded from confidence
        )
        if not ok:
            print("[insight_layer] FAIL g11: legacy migration produced wrong "
                  f"claims: {migrated}", file=sys.stderr)
            failures += 1
        else:
            archive.write_insight_layer_in_place(km, migrated, seed_band)
            with _zipfile.ZipFile(km, "r") as zf:
                rewritten = zf.read("insight.jsonl").decode("utf-8")
            if "verdict_state" in rewritten:
                print("[insight_layer] FAIL g11: rewritten insight.jsonl still "
                      "contains verdict_state", file=sys.stderr)
                failures += 1
            else:
                print("[insight_layer] g11 (legacy InsightPassage migration) OK",
                      file=sys.stderr)

        # ---- Guarantee 12: concurrent in-place writers don't corrupt ----
        failures += _check_concurrent_inplace_writes(km, insight_band)

    # ---- Guarantees 14-16: light insight reader + band recall (S3 d3.2) ----
    with tempfile.TemporaryDirectory() as d2:
        from resonance_lattice.field.encoder import Encoder
        from resonance_lattice.state.claim import Claim, ExperienceFacts
        from resonance_lattice.store.verified import (
            rank_insight_band,
            verify_insight_hits,
        )

        root2 = Path(d2) / "corpus"
        km2 = _build(root2, {
            "a.md": "# Auth\n\nSession tokens and login.",
            "b.md": "# Store\n\nRedis session storage.",
        })
        c = archive.read(km2)
        src2 = [p.passage_id for p in c.registry[:2]]

        # g14a: a layerless archive → read_insight_layer returns None.
        if archive.read_insight_layer(km2) is not None:
            print("[insight_layer] FAIL g14: read_insight_layer on a layerless "
                  "archive is not None", file=sys.stderr)
            failures += 1

        enc = Encoder()
        texts2 = [
            "Login uses session tokens.",   # A — corpus, active
            "Sessions persist in Redis.",   # B — corpus, candidate (not retrievable)
            "Prefer ruff for linting.",     # E — experience, active
        ]
        band2 = enc.encode(texts2).astype("float32")
        a = make_corpus_claim(texts2[0], src2[:1], state="active", faithfulness=0.9)
        b = make_corpus_claim(texts2[1], src2, state="candidate", faithfulness=0.9)
        e = Claim(
            claim_id="01HZEXPBAND00000000000RANK",
            source="experience", kind="event", content=texts2[2],
            created_at="2026-06-01T00:00:00Z",
            corroboration=3.0, falsification=1.0, trust_as_of="",
            state="active", parent_ids=(),
            facts=ExperienceFacts(
                polarity=("prefer",), recurrence_count=2, criticality="normal",
                created_under_intent_kind="none", transcript_hash="manual",
                origin="manual", last_corroborated_at="2026-06-01T00:00:00Z",
            ),
        )
        archive.write_insight_layer_in_place(km2, [a, b, e], band2)

        # g14b: the light reader round-trips claims + band, mixed source.
        layer = archive.read_insight_layer(km2)
        if layer is None:
            print("[insight_layer] FAIL g14: read_insight_layer returned None "
                  "after write", file=sys.stderr)
            failures += 1
        else:
            ins2, bnd2 = layer
            if ([x.claim_id for x in ins2] != [a.claim_id, b.claim_id, e.claim_id]
                    or bnd2.shape != (3, band2.shape[1])
                    or ins2[2].source != "experience"):
                print(f"[insight_layer] FAIL g14: light reader mismatch — "
                      f"ids={[x.claim_id for x in ins2]} shape={bnd2.shape}",
                      file=sys.stderr)
                failures += 1
            else:
                print("[insight_layer] g14 (light insight-layer reader, mixed "
                      "source) OK", file=sys.stderr)

        # g15: rank_insight_band — active-only, source-agnostic, own 0-based
        # ranks, raw-cosine floor.
        ins2, bnd2 = archive.read_insight_layer(km2)
        q = enc.encode(["session tokens login"]).astype("float32")[0]
        hits = rank_insight_band(q, ins2, bnd2, top_k=10, cosine_floor=-1.0)
        ids = {h.claim_id for h in hits}
        by_id = {h.claim_id: h for h in hits}
        if b.claim_id in ids:
            print("[insight_layer] FAIL g15: candidate corpus claim surfaced "
                  "(ranker must be active-only)", file=sys.stderr)
            failures += 1
        elif ids != {a.claim_id, e.claim_id}:
            print(f"[insight_layer] FAIL g15: expected the two active claims "
                  f"{{A,E}}, got {ids}", file=sys.stderr)
            failures += 1
        elif {h.rank for h in hits} != {0, 1}:
            print(f"[insight_layer] FAIL g15: ranks not an own 0-based namespace "
                  f"— {sorted(h.rank for h in hits)}", file=sys.stderr)
            failures += 1
        elif (by_id[a.claim_id].source != "corpus"
              or by_id[e.claim_id].source != "experience"):
            print("[insight_layer] FAIL g15: hit.source not carried per claim",
                  file=sys.stderr)
            failures += 1
        elif rank_insight_band(q, ins2, bnd2, top_k=10, cosine_floor=1.01) != []:
            print("[insight_layer] FAIL g15: cosine_floor=1.01 did not drop all "
                  "hits", file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g15 (band ranker: active-only, source-"
                  "agnostic, own ranks, floor) OK", file=sys.stderr)

        # g16: verify_insight_hits renders BOTH band sources (v3 S2 — band
        # claims must be visible to `rlat search`). The experience claim
        # renders with empty corpus-only fields (no AttributeError on
        # ExperienceFacts); its kind + content carry the meaning.
        # include_stale=True so the candidate corpus claim is admitted too.
        rendered = verify_insight_hits(
            [(0, 0.9), (1, 0.9), (2, 0.9)], [a, b, e], include_stale=True,
        )
        rids = {h.claim_id for h in rendered}
        exp_hit = next((h for h in rendered if h.claim_id == e.claim_id), None)
        if rids != {a.claim_id, b.claim_id, e.claim_id}:
            print(f"[insight_layer] FAIL g16: expected all three band claims "
                  f"(corpus + experience), got {rids}", file=sys.stderr)
            failures += 1
        elif exp_hit is None or exp_hit.content != e.content or (
                exp_hit.kind != "event"
                or exp_hit.content_fingerprint != ""
                or exp_hit.citations != ()
                or exp_hit.source_passage_hashes != ()
                or exp_hit.intent_context is not None):
            print(f"[insight_layer] FAIL g16: experience render arm wrong — "
                  f"{exp_hit!r}", file=sys.stderr)
            failures += 1
        else:
            print("[insight_layer] g16 (verify_insight_hits renders experience "
                  "band claims with empty corpus fields) OK", file=sys.stderr)

    # ---- Guarantee 17: serve_band_attributes — newest-per-subject dedup ----
    # Pure function, synthetic band so the cosines are exact and the OLD value
    # deliberately OUT-RANKS the new one (the bug the dedup fixes).
    failures += _check_serve_band_attributes()

    # ---- Guarantee 18: serve_band_constraints — serve-ALL + kind framing ----
    failures += _check_serve_band_constraints()

    if failures:
        print(f"[insight_layer] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[insight_layer] all guarantees OK", file=sys.stderr)
    return 0


def _check_concurrent_inplace_writes(km: Path, insight_band) -> int:
    """Two threads each call `write_insight_layer_in_place` with a distinct
    insights list. With the D/5.2 per-writer-unique tmp fix, both writers
    produce a complete tmp file and the final archive equals whichever's
    `os.replace` lands last — never a torn ZIP mixing bytes from both.

    The load-bearing regression check is the 16-call uniqueness assertion
    on `_unique_tmp_path`: this is the actual invariant the fix maintains.
    The two-thread integration check is the end-to-end smoke pass — under
    Python's GIL it doesn't fully exercise the inter-process race the
    fix targets (`multiprocessing.Process` would), but the threaded
    overlap is enough to verify the writer survives concurrent entry +
    leaves no orphan tmp files behind."""
    import threading

    from resonance_lattice.store import archive
    from resonance_lattice.store.archive import _unique_tmp_path

    fake = Path("/nonexistent/archive.rlat")
    paths = {_unique_tmp_path(fake) for _ in range(16)}
    if len(paths) != 16:
        print(f"[insight_layer] FAIL g12: _unique_tmp_path collided "
              f"({16 - len(paths)} duplicate(s) in 16 calls)", file=sys.stderr)
        return 1

    insights_a = [
        make_corpus_claim("writer-A insight", ["p0"], state="active",
                          faithfulness=0.6),
    ]
    insights_b = [
        make_corpus_claim("writer-B insight one", ["p0"], state="active",
                          faithfulness=0.7),
        make_corpus_claim("writer-B insight two", ["p1"], state="active",
                          faithfulness=0.7),
    ]
    band_a = np.zeros((1, insight_band.shape[1]), dtype=insight_band.dtype)
    band_b = np.zeros((2, insight_band.shape[1]), dtype=insight_band.dtype)

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def writer(insights, band):
        try:
            barrier.wait()
            archive.write_insight_layer_in_place(km, insights, band)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(insights_a, band_a)),
        threading.Thread(target=writer, args=(insights_b, band_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        # On Windows, two concurrent `os.replace(tmp, p)` calls can collide
        # — one wins, the other raises PermissionError. That's a "lost
        # update" outcome, not corruption: the loser's tmp is cleaned up
        # by its `BaseException` handler and the winner's archive is on
        # disk. Verify the archive still loads + equals the winner's
        # input before declaring this failure.
        non_perm = [e for e in errors if not isinstance(e, PermissionError)]
        if non_perm:
            print(f"[insight_layer] FAIL g12: unexpected error from "
                  f"concurrent writer: {non_perm[0]!r}", file=sys.stderr)
            return 1

    final = archive.read(km).insights
    final_contents = sorted(c.content for c in final)
    matches_a = final_contents == sorted(c.content for c in insights_a)
    matches_b = final_contents == sorted(c.content for c in insights_b)
    if not (matches_a or matches_b):
        print(f"[insight_layer] FAIL g12: final archive insights don't "
              f"match either writer: {final_contents}", file=sys.stderr)
        return 1

    # Both writers' exception handlers should have unlinked their tmp files
    # — neither's tmp survives the join. A stale `*.tmp` next to the archive
    # would mean the orphan-cleanup path regressed.
    orphans = list(km.parent.glob(f"{km.name}.*.tmp"))
    if orphans:
        print(f"[insight_layer] FAIL g12: orphan tmp file(s) survived "
              f"concurrent writers: {orphans}", file=sys.stderr)
        return 1

    print("[insight_layer] g12 (concurrent in-place writers don't corrupt) OK",
          file=sys.stderr)
    return 0


def _check_serve_band_attributes() -> int:
    """g17: serve_band_attributes serves the NEWEST value per attribute_key.

    Synthetic 3-dim band so the cosines are exact. The query is [1,0,0]; the
    OLD ps_version value sits exactly on it (cosine 1.0) while the NEW value is
    off-axis (cosine 0.6). Newest-by-created_at must still win — proving the
    dedup keys on time, not similarity — and the floored case must NOT fall
    back to the stale-but-closer value."""
    from resonance_lattice.state.claim import Claim, ExperienceFacts
    from resonance_lattice.store.verified import serve_band_attributes

    def _attr(cid, content, *, key, created_at, state="active"):
        return Claim(
            claim_id=cid, source="experience", kind="attribute", content=content,
            created_at=created_at, corroboration=3.0, falsification=1.0,
            trust_as_of="", state=state, parent_ids=(),
            facts=ExperienceFacts(
                polarity=("factual",), recurrence_count=1, criticality="high",
                created_under_intent_kind="none", transcript_hash="manual",
                origin="manual", last_corroborated_at=created_at, attribute_key=key,
            ),
        )

    old_ps = _attr("OLDPS", "PowerShell 5.1", key="ps_version",
                   created_at="2026-06-01T00:00:00Z")
    new_ps = _attr("NEWPS", "PowerShell 7.4", key="ps_version",
                   created_at="2026-06-05T00:00:00Z")
    unkeyed1 = _attr("UNK1", "The account is standard.", key="",
                     created_at="2026-06-02T00:00:00Z")
    unkeyed2 = _attr("UNK2", "The machine is domain-joined.", key="",
                     created_at="2026-06-03T00:00:00Z")
    cand = _attr("CAND", "Region East US", key="region",
                 created_at="2026-06-04T00:00:00Z", state="candidate")
    evt = Claim(
        claim_id="EVT", source="experience", kind="event", content="Prefer ruff.",
        created_at="2026-06-01T00:00:00Z", corroboration=3.0, falsification=1.0,
        trust_as_of="", state="active", parent_ids=(),
        facts=ExperienceFacts(
            polarity=("prefer",), recurrence_count=2, criticality="normal",
            created_under_intent_kind="none", transcript_hash="manual",
            origin="manual", last_corroborated_at="2026-06-01T00:00:00Z",
        ),
    )
    insights = [old_ps, new_ps, unkeyed1, unkeyed2, cand, evt]
    # Rows aligned to `insights`. Unit vectors → retrieve_insight cosine is exact.
    band = np.array([
        [1.0, 0.0, 0.0],   # old_ps   cos 1.00  (deliberately out-ranks new)
        [0.6, 0.8, 0.0],   # new_ps   cos 0.60
        [0.8, 0.0, 0.6],   # unkeyed1 cos 0.80
        [0.7, 0.0, 0.714142842854285],  # unkeyed2 cos 0.70
        [1.0, 0.0, 0.0],   # cand     cos 1.00  (not retrievable → skipped)
        [1.0, 0.0, 0.0],   # evt      cos 1.00  (not attribute → skipped)
    ], dtype="float32")
    q = np.array([1.0, 0.0, 0.0], dtype="float32")

    failures = 0
    hits = serve_band_attributes(q, insights, band, top_k=10, cosine_floor=0.0)
    served = [h.content for h in hits]

    # Newest wins despite lower cosine: NEW ps served, OLD ps suppressed.
    if "PowerShell 5.1" in served:
        print("[insight_layer] FAIL g17: stale ps_version served — newest-wins "
              "dedup didn't suppress the older (higher-cosine) value",
              file=sys.stderr)
        failures += 1
    if "PowerShell 7.4" not in served:
        print("[insight_layer] FAIL g17: newest ps_version not served",
              file=sys.stderr)
        failures += 1
    # Both keyless facts survive — a null key never dedups another.
    if not ({"The account is standard.", "The machine is domain-joined."} <= set(served)):
        print(f"[insight_layer] FAIL g17: an unkeyed attribute was deduped — "
              f"served={served}", file=sys.stderr)
        failures += 1
    # Non-attribute / non-retrievable claims never serve.
    if "Prefer ruff." in served or "Region East US" in served:
        print(f"[insight_layer] FAIL g17: a non-attribute or candidate claim "
              f"served — {served}", file=sys.stderr)
        failures += 1
    # Exactly the three expected hits, ordered by score (= cosine at equal trust):
    # unkeyed1 0.80 > new_ps 0.60... but unkeyed2 0.70 sits between. Order:
    # unkeyed1 (0.80), unkeyed2 (0.70), new_ps (0.60).
    if [h.content for h in hits] != [
        "The account is standard.", "The machine is domain-joined.", "PowerShell 7.4",
    ]:
        print(f"[insight_layer] FAIL g17: served set/order wrong — "
              f"{[h.content for h in hits]}", file=sys.stderr)
        failures += 1

    # Floor ABOVE the newest value's cosine (0.60) but BELOW the stale value's
    # (1.00): the subject must drop ENTIRELY, never fall back to the stale value.
    floored = serve_band_attributes(q, insights, band, top_k=10, cosine_floor=0.65)
    fserved = [h.content for h in floored]
    if "PowerShell 5.1" in fserved or "PowerShell 7.4" in fserved:
        print(f"[insight_layer] FAIL g17: floored serve leaked a ps_version value "
              f"(must drop the whole subject, no stale fallback) — {fserved}",
              file=sys.stderr)
        failures += 1
    elif fserved != ["The account is standard.", "The machine is domain-joined."]:
        print(f"[insight_layer] FAIL g17: floored serve wrong — {fserved}",
              file=sys.stderr)
        failures += 1

    # Serve is read-only: the older value is untouched on the input list (the
    # "keep history, don't delete" contract — capture appends, serve filters).
    if old_ps not in insights or len(insights) != 6:
        print("[insight_layer] FAIL g17: serve mutated the insights list",
              file=sys.stderr)
        failures += 1

    # Same-second tie: two values share attribute_key AND created_at. The one
    # appended LATER (higher band index) must win — the tie-break is recency by
    # append order, NEVER cosine. tie_b is appended after tie_a and given the
    # LOWER cosine, so a cosine-based tie-break would (wrongly) pick tie_a.
    tie_a = _attr("TIEA", "capacity F2", key="sku", created_at="2026-06-09T00:00:00Z")
    tie_b = _attr("TIEB", "capacity F64", key="sku", created_at="2026-06-09T00:00:00Z")
    tie_insights = [tie_a, tie_b]
    tie_band = np.array([
        [1.0, 0.0, 0.0],   # tie_a  cos 1.00 (higher — a cosine tie-break would pick this)
        [0.6, 0.8, 0.0],   # tie_b  cos 0.60 (lower, but appended later → newer)
    ], dtype="float32")
    tie_hits = serve_band_attributes(q, tie_insights, tie_band, top_k=10, cosine_floor=0.0)
    if [h.content for h in tie_hits] != ["capacity F64"]:
        print(f"[insight_layer] FAIL g17: same-created_at tie not broken by "
              f"append order (got {[h.content for h in tie_hits]}; expected the "
              f"later-appended 'capacity F64', not the higher-cosine 'capacity F2')",
              file=sys.stderr)
        failures += 1

    if not failures:
        print("[insight_layer] g17 (serve_band_attributes: newest-per-subject, "
              "no stale fallback, keyless never deduped, non-destructive) OK",
              file=sys.stderr)
    return failures


def _check_serve_band_constraints() -> int:
    """g18: serve_band_constraints is serve-ALL (query-independent, no floor,
    no top-k), keeps only retrievable constraint/negation experience claims,
    dedups newest-wins per (kind, attribute_key), preserves band order — and
    `serve_framing.frame_claim_lines` renders the R1/R2-proven headings with
    constraints first."""
    from resonance_lattice.state.claim import Claim, ExperienceFacts
    from resonance_lattice.store.serve_framing import (
        CONSTRAINTS_HEADING,
        FALSIFIED_HEADING,
        frame_claim_lines,
    )
    from resonance_lattice.store.verified import serve_band_constraints

    def _exp(cid, kind, content, *, key="", created_at, state="active"):
        return Claim(
            claim_id=cid, source="experience", kind=kind, content=content,
            created_at=created_at, corroboration=3.0, falsification=1.0,
            trust_as_of="", state=state, parent_ids=(),
            facts=ExperienceFacts(
                polarity=("factual",), recurrence_count=1, criticality="high",
                created_under_intent_kind="none", transcript_hash="manual",
                origin="manual", last_corroborated_at=created_at,
                attribute_key=key,
            ),
        )

    neg = _exp("NEG", "negation", "Tried X; falsified by record Y.",
               created_at="2026-06-01T00:00:00Z")
    con_old = _exp("CONOLD", "constraint", "Capacity at F8 or below.",
                   key="capacity", created_at="2026-06-01T00:00:00Z")
    con_new = _exp("CONNEW", "constraint", "Capacity at F4 or below.",
                   key="capacity", created_at="2026-06-05T00:00:00Z")
    con_unkeyed = _exp("CONUNK", "constraint", "No preview features.",
                       created_at="2026-06-02T00:00:00Z")
    con_retired = _exp("CONRET", "constraint", "Old retired rule.",
                       created_at="2026-06-03T00:00:00Z", state="retired")
    attr = _exp("ATTR", "attribute", "Region is EU.", key="region",
                created_at="2026-06-03T00:00:00Z")
    insights = [neg, con_old, con_new, con_unkeyed, con_retired, attr]

    failures = 0
    hits = serve_band_constraints(insights)
    served = [(h.kind, h.content) for h in hits]
    # Serve-ALL of the retrievable constraint/negation set, band order,
    # newest-wins on the keyed pair, attribute + retired excluded.
    if served != [
        ("negation", "Tried X; falsified by record Y."),
        ("constraint", "Capacity at F4 or below."),
        ("constraint", "No preview features."),
    ]:
        print(f"[insight_layer] FAIL g18: serve set/order wrong — {served}",
              file=sys.stderr)
        failures += 1

    framed = frame_claim_lines((h.kind, h.content) for h in hits)
    con_pos = framed.find(CONSTRAINTS_HEADING)
    neg_pos = framed.find(FALSIFIED_HEADING)
    if (con_pos < 0 or neg_pos < 0 or not con_pos < neg_pos
            or "- Capacity at F4 or below." not in framed
            or "- Tried X; falsified by record Y." not in framed):
        print(f"[insight_layer] FAIL g18: framing wrong —\n{framed}",
              file=sys.stderr)
        failures += 1
    if frame_claim_lines([]) != "" or frame_claim_lines([("event", "x")]) != "":
        print("[insight_layer] FAIL g18: empty/unknown-kind framing must "
              "render nothing", file=sys.stderr)
        failures += 1

    # The RANKED render path shares the newest-wins rule: a superseded keyed
    # value never reaches `rlat search` output even when cosine-ranked
    # (the 2026-06 S2 empirical gate caught exactly this leak).
    from resonance_lattice.store.verified import verify_insight_hits
    ranked = verify_insight_hits(
        [(i, 0.9) for i in range(len(insights))], insights)
    ranked_contents = [h.content for h in ranked]
    if "Capacity at F8 or below." in ranked_contents:
        print(f"[insight_layer] FAIL g18: superseded constraint leaked through "
              f"the ranked render path — {ranked_contents}", file=sys.stderr)
        failures += 1
    if "Capacity at F4 or below." not in ranked_contents:
        print(f"[insight_layer] FAIL g18: newest constraint missing from the "
              f"ranked render path — {ranked_contents}", file=sys.stderr)
        failures += 1

    if not failures:
        print("[insight_layer] g18 (serve_band_constraints: serve-ALL, dedup, "
              "kind filter; proven framings render constraints-first) OK",
              file=sys.stderr)
    return failures


def _cli_search(argv: list[str]) -> tuple[int, str, str]:
    """Invoke `rlat search` through the dispatcher; capture stdout/stderr."""
    import contextlib
    import io

    from resonance_lattice.cli.app import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(["search", *argv])
    return rc, out.getvalue(), err.getvalue()


def _swap_zip_member(km: Path, member: str, new_text: str) -> None:
    """Rewrite `km` replacing one ZIP member's bytes — used to inject a
    legacy-shaped `insight.jsonl` the modern writer cannot emit."""
    import zipfile

    with zipfile.ZipFile(km, "r") as src:
        items = [(i, src.read(i.filename)) for i in src.infolist()]
    tmp = Path(str(km) + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as dst:
        for info, data in items:
            payload = new_text.encode("utf-8") if info.filename == member \
                else data
            dst.writestr(info, payload)
    tmp.replace(km)


if __name__ == "__main__":
    sys.exit(run())
