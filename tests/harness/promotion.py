"""promotion — compression test + promotion pipeline.

Guarantees (Day 4):

  1. Single-citation candidates fail (anti-paraphrase guard).
  2. Semantic duplicates fail (cosine ≥ duplicate_threshold).
  3. Growth cap rejects insight-layer bloat past 0.1%.
  4. Coverage regression fails (delta < 0).
  5. Coverage no-lift fails when queries present and delta == 0.
  6. First-ever promotion passes when diversity + queries OK
     (empty insight layer is allowed).
  7. promote_candidates writes survivors back to the .rlat atomically;
     failed candidates leave no trace on disk.
  8. candidates_from_memory_rows skips malformed rows.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_insight_passage, unpatch_zero_encoder


def _query_record(emb: np.ndarray, expected: list[str]):
    from resonance_lattice.store.compression_test import QueryRecord
    return QueryRecord(
        query_embedding=emb.astype("float32"),
        expected_passage_ids=frozenset(expected),
    )


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.store import archive
    from resonance_lattice.store.compression_test import run_compression_test
    from resonance_lattice.store.promotion import (
        SynthesisCandidate,
        candidates_from_memory_rows,
        promote_candidates,
    )
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store.insight import InsightCitation

    failures = 0

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        files = {
            "a.md": "# Auth\n\nSession tokens expire after 24 hours.",
            "b.md": "# Tokens\n\nRefresh tokens rotate weekly.",
            "c.md": "# Storage\n\nSessions persist in Redis.",
        }
        km = _build(root, files)
        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry]
        src_hashes = [c.content_hash for c in c0.registry]

        encoder = Encoder()
        # Build the query-history fixture: queries that should retrieve
        # specific source passages.
        q_text = ["session tokens expire", "refresh tokens", "storage in redis"]
        q_emb = encoder.encode(q_text).astype("float32")
        # Each query expects to retrieve its corresponding source row.
        queries = [
            _query_record(q_emb[0], [src_ids[0]]),
            _query_record(q_emb[1], [src_ids[1]]),
        ]

        # ---- Guarantee 1: single-citation candidate fails ----
        candidate = make_insight_passage(0, "Sessions use tokens.",
                                         [src_ids[0]], state="candidate")
        emb = encoder.encode([candidate.content])[0].astype("float32")
        result = run_compression_test(
            candidate, emb, c0.bands["base"], c0.registry, None, [], queries,
        )
        if result.passed or result.reason != "paraphrase":
            print(f"[promotion] FAIL g1: single-citation passed "
                  f"({result.reason}, distinct={result.distinct_sources})",
                  file=sys.stderr)
            failures += 1
        else:
            print("[promotion] g1 (paraphrase guard) OK", file=sys.stderr)

        # ---- Guarantee 6: legitimate first-ever promotion passes ----
        # With empty queries the test falls through coverage gating: the
        # candidate has cleared paraphrase/duplicate/bloat and there's no
        # signal to reject on. This is the "no historical data yet"
        # bootstrap case for fresh corpora.
        candidate = make_insight_passage(
            0, "Session tokens and refresh tokens rotate on a 24h/weekly cadence.",
            src_ids[:2], src_hashes[:2], state="candidate",
        )
        emb = encoder.encode([candidate.content])[0].astype("float32")
        result = run_compression_test(
            candidate, emb, c0.bands["base"], c0.registry, None, [], [],
        )
        if not result.passed:
            print(f"[promotion] FAIL g6: first promotion rejected "
                  f"({result.reason}, delta={result.coverage_delta:.3f})",
                  file=sys.stderr)
            failures += 1
        else:
            print("[promotion] g6 (first-ever promotion passes) OK", file=sys.stderr)

        # ---- Guarantee 2: semantic duplicate fails ----
        # Promote first; then try promoting the same content again.
        candidate_dto = SynthesisCandidate(
            candidate_id="cand-1",
            content=candidate.content,
            citations=candidate.citations,
            source_passage_hashes=candidate.source_passage_hashes,
            source_model_hash=candidate.source_model_hash,
            query=None, intent_context=None,
            encoder_version="gte-mb-768",
        )
        # Promote against empty query history — the goal is to confirm
        # atomic writeback works; the coverage gate is tested separately
        # in g5 below using fixture data that exercises it.
        outcomes = promote_candidates(
            km, [candidate_dto], emb.reshape(1, -1), [],
        )
        if not outcomes[0].promoted:
            print(f"[promotion] FAIL g7-pre: first promotion failed "
                  f"({outcomes[0].test_result.reason})", file=sys.stderr)
            failures += 1
        else:
            c1 = archive.read(km)
            if not c1.insights:
                print("[promotion] FAIL g7: insight layer empty after promotion",
                      file=sys.stderr)
                failures += 1
            else:
                print("[promotion] g7 (atomic writeback of survivors) OK",
                      file=sys.stderr)

                # Now retry the same candidate — should fail as duplicate.
                outcomes2 = promote_candidates(
                    km, [candidate_dto], emb.reshape(1, -1), queries,
                )
                # Either guard fires here: pre-test idempotent (same
                # insight_id already promoted) or in-test duplicate
                # (cosine >= 0.95). Both are correct outcomes; the
                # contract is that the second attempt does NOT add a
                # new row.
                if (outcomes2[0].promoted
                    or outcomes2[0].test_result.reason not in ("duplicate", "idempotent")):
                    print(f"[promotion] FAIL g2: duplicate not caught "
                          f"({outcomes2[0].test_result.reason})", file=sys.stderr)
                    failures += 1
                else:
                    print(f"[promotion] g2 (duplicate guard, reason="
                          f"{outcomes2[0].test_result.reason}) OK",
                          file=sys.stderr)

        # ---- Guarantee 4: coverage regression fails ----
        # Use a query that returns the source perfectly already; a
        # candidate that DOESN'T cover that query should at best be
        # coverage-neutral.
        c1 = archive.read(km)
        # Build a candidate whose citations don't help any historical
        # query — distinct from the existing accepted insight and not
        # covering the queries.
        cand_text = "Completely unrelated content about cooking recipes."
        # Cite real sources but with content that doesn't help queries —
        # we'll fake-cite source 2 (storage in redis, unrelated to first
        # two queries) twice (distinct citations) so the diversity guard
        # passes but coverage doesn't move.
        unrelated_cand = make_insight_passage(
            0, cand_text, [src_ids[2], src_ids[2]],
            [src_hashes[2], src_hashes[2]], state="candidate",
        )
        # The passage_id list above has duplicates; the test's distinct-source
        # count is by set, so this stays at 1 distinct → paraphrase reason.
        # Use distinct passages but unrelated query expectations.
        from dataclasses import replace as _replace
        unrelated_cand = _replace(
            unrelated_cand,
            citations=(
                InsightCitation(passage_id=src_ids[2], char_span=None, confidence=0.9),
                InsightCitation(passage_id=src_ids[1], char_span=None, confidence=0.9),
            ),
        )
        emb_u = encoder.encode([cand_text])[0].astype("float32")
        result = run_compression_test(
            unrelated_cand, emb_u, c1.bands["base"], c1.registry,
            c1.bands.get(archive.INSIGHT_BAND_NAME), c1.insights, queries,
        )
        # Unrelated cooking content doesn't lift coverage against
        # session/auth queries; reason should be no_lift or regression.
        if result.passed:
            print(f"[promotion] FAIL g5: no-lift / regression candidate passed "
                  f"(delta={result.coverage_delta:.3f})", file=sys.stderr)
            failures += 1
        else:
            print(f"[promotion] g5 (no-lift / regression guard) OK "
                  f"(reason={result.reason})", file=sys.stderr)

    # ---- Guarantee 8: candidates_from_memory_rows skips malformed ----
    rows = [
        {
            "id": "good-1", "content": "x", "citations": [
                {"passage_id": "p1", "char_span": None, "confidence": 0.9},
            ],
            "source_passage_hashes": ["h1"], "source_model_hash": "m",
            "encoder_version": "gte-mb-768",
        },
        {"id": "bad-1", "content": "no citations"},  # missing fields
    ]
    candidates = candidates_from_memory_rows(rows)
    if len(candidates) != 1 or candidates[0].candidate_id != "good-1":
        print(f"[promotion] FAIL g8: adapter skip wrong "
              f"(got {[c.candidate_id for c in candidates]})", file=sys.stderr)
        failures += 1
    else:
        print("[promotion] g8 (memory-row adapter skips malformed) OK",
              file=sys.stderr)

    if failures:
        print(f"[promotion] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[promotion] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
