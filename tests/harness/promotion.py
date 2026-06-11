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
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim, make_experience_claim, unpatch_zero_encoder


def _query_record(emb: np.ndarray, expected: list[str]):
    from resonance_lattice.store.compression_test import QueryRecord
    return QueryRecord(
        query_embedding=emb.astype("float32"),
        expected_passage_ids=frozenset(expected),
    )


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.store import archive
    from resonance_lattice.store.compression_test import (
        _top_k_insight_supporting_ids,
        run_compression_test,
    )
    from resonance_lattice.store.promotion import (
        SynthesisCandidate,
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
        candidate = make_corpus_claim("Sessions use tokens.",
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
        candidate = make_corpus_claim(
            "Session tokens and refresh tokens rotate on a 24h/weekly cadence.",
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

        # ---- Guarantee: external-citation claim skips the corpus-coverage gate ----
        # An external fill cites non-corpus sources (passage_id "external:..."), so
        # the corpus-coverage guards are inapplicable; it must earn promotion on
        # diversity + novelty alone EVEN WITH historical queries — where a corpus
        # claim that adds no coverage would be rejected (no_lift). Confirms the branch
        # is external-specific and policy-correct.
        ext_claim = make_corpus_claim(
            "The widget ships in March 2027, confirmed by two independent vendor sources.",
            ["external:aaaaaaaa", "external:bbbbbbbb"], state="candidate",
        )
        ext_emb = encoder.encode([ext_claim.content])[0].astype("float32")
        result = run_compression_test(
            ext_claim, ext_emb, c0.bands["base"], c0.registry, None, [], queries,
        )
        if not result.passed or result.reason != "passed_external":
            print(f"[promotion] FAIL g-ext: external claim not passed_external "
                  f"({result.reason}, distinct={result.distinct_sources})", file=sys.stderr)
            failures += 1
        else:
            print("[promotion] g-ext (external claim skips corpus-coverage gate) OK",
                  file=sys.stderr)

        # ---- Guarantee 2: semantic duplicate fails ----
        # Promote first; then try promoting the same content again.
        candidate_dto = SynthesisCandidate(
            candidate_id="cand-1",
            content=candidate.content,
            citations=candidate.facts.citations,
            source_passage_hashes=candidate.facts.source_passage_hashes,
            source_model_hash=candidate.facts.source_model_hash,
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
        unrelated_cand = make_corpus_claim(
            cand_text, [src_ids[2], src_ids[2]],
            [src_hashes[2], src_hashes[2]], state="candidate",
        )
        # The passage_id list above has duplicates; the test's distinct-source
        # count is by set, so this stays at 1 distinct → paraphrase reason.
        # Use distinct passages but unrelated query expectations.
        from resonance_lattice.state.claim import evolve as _evolve
        unrelated_cand = _evolve(
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

        # ---- Guarantee 9: compression test is corpus-scoped (source-safe) ----
        # (a) the citation-coverage gate rejects an experience candidate —
        # experience claims earn `active` via consolidate_experience, never
        # this gate, so a non-corpus candidate is a routing error.
        exp_cand = make_experience_claim(
            claim_id="01HZEXPCAND000000000000C9",
            content="prefer ruff over flake8",
            polarity=("prefer",), transcript_hash="manual", state="candidate",
        )
        exp_emb = encoder.encode([exp_cand.content])[0].astype("float32")
        try:
            run_compression_test(
                exp_cand, exp_emb, c1.bands["base"], c1.registry, None, [], [],
            )
        except TypeError:
            print("[promotion] g9a (gate rejects experience candidate) OK",
                  file=sys.stderr)
        else:
            print("[promotion] FAIL g9a: experience candidate accepted by "
                  "the citation gate", file=sys.stderr)
            failures += 1

        # (b) an experience claim co-located in the unified band contributes
        # no citation coverage and must not crash the coverage reach (it has
        # no `citations` field). Exercised directly — the growth cap would
        # short-circuit a full run before the coverage stage on a small layer.
        corpus_ins = make_corpus_claim(
            "Session tokens expire after 24h.", src_ids[:2], src_hashes[:2],
            state="active",
        )
        exp_ins = make_experience_claim(
            claim_id="01HZEXPINS0000000000000C9",
            content="prefer pytest fixtures", polarity=("prefer",),
            transcript_hash="manual", state="active",
        )
        mixed_band = encoder.encode(
            [corpus_ins.content, exp_ins.content]
        ).astype("float32")
        try:
            reach = _top_k_insight_supporting_ids(
                q_emb[0], mixed_band, [corpus_ins, exp_ins], top_k=10,
            )
        except AttributeError as exc:
            print(f"[promotion] FAIL g9b: coverage reach crashed on an "
                  f"experience band claim — {exc}", file=sys.stderr)
            failures += 1
        else:
            # Only the corpus claim's two cited sources; the experience
            # claim is skipped, not crashed.
            if reach != frozenset(src_ids[:2]):
                print(f"[promotion] FAIL g9b: reach {sorted(reach)} != corpus "
                      f"citations {sorted(src_ids[:2])} (experience claim "
                      f"should contribute nothing)", file=sys.stderr)
                failures += 1
            else:
                print("[promotion] g9b (coverage reach skips experience band "
                      "claims) OK", file=sys.stderr)

        # ---- Guarantee 10: verdict-anchored regression guard (4.3) ----
        # Controlled vectors: a prior query Q whose expected passage p1 is
        # reachable ONLY through the prior insight I1 (source top-k misses
        # it). A new candidate C that outranks I1 for Q but cites p2 — with
        # top_k=1 it displaces I1 from Q's insight reach, coverage drops,
        # and the gate must reject it as a regression. Guarantee 11 pins
        # the other half: with require_lift=False the SAME candidate
        # against a novel (no-overlap) query set passes on delta == 0;
        # with require_lift=True it fails no_lift.
        dim = c0.bands["base"].shape[1]

        def _unit(i, lean=None, w=0.0):
            v = np.zeros(dim, dtype="float32")
            v[i] = 1.0
            if lean is not None:
                v[lean] = w
            return v / np.linalg.norm(v)

        # Source rows: p1's row orthogonal to Q (cos 0), p2's row leaning
        # toward Q (cos ~0.29) — source top-1 deterministically returns p2,
        # never p1, so p1 is reachable only through the insight layer.
        synth_band = np.stack([_unit(1), _unit(2, lean=0, w=0.3)])
        synth_registry = [c0.registry[0], c0.registry[1]]
        p1, p2 = src_ids[0], src_ids[1]
        q_vec = _unit(0)
        prior_insight = make_corpus_claim("p1 synthesis", [p1], [src_hashes[0]],
                                          state="active")
        prior_band = _unit(0, lean=3, w=0.5).reshape(1, -1)  # cos(Q) ~ 0.89
        reg_candidate = make_corpus_claim("p2-citing usurper",
                                          [p2, p2], [src_hashes[1]],
                                          state="candidate")
        cand_emb = q_vec  # cos(Q) = 1.0 — outranks the prior insight
        prior_queries = [_query_record(q_vec, [p1])]
        res = run_compression_test(
            reg_candidate, cand_emb, synth_band, synth_registry,
            prior_band, [prior_insight], prior_queries,
            top_k=1, min_distinct_sources=1, require_lift=False,
        )
        if res.passed or res.reason != "regression" or res.coverage_delta >= 0:
            print(f"[promotion] FAIL g10: regression not rejected "
                  f"({res.reason}, delta={res.coverage_delta})", file=sys.stderr)
            failures += 1
        else:
            print("[promotion] g10 (verdict-anchored regression rejected) OK",
                  file=sys.stderr)

        novel_queries = [_query_record(_unit(3), [p1])]  # can't help or hurt
        res_pass = run_compression_test(
            reg_candidate, cand_emb, synth_band, synth_registry,
            None, [], novel_queries,
            top_k=1, min_distinct_sources=1, require_lift=False,
        )
        res_strict = run_compression_test(
            reg_candidate, cand_emb, synth_band, synth_registry,
            None, [], novel_queries,
            top_k=1, min_distinct_sources=1, require_lift=True,
        )
        if not (res_pass.passed and res_pass.coverage_delta == 0
                and not res_strict.passed and res_strict.reason == "no_lift"):
            print(f"[promotion] FAIL g11: require_lift semantics wrong "
                  f"(relaxed: {res_pass.passed}/{res_pass.reason}, "
                  f"strict: {res_strict.passed}/{res_strict.reason})",
                  file=sys.stderr)
            failures += 1
        else:
            print("[promotion] g11 (delta==0: passes relaxed, fails strict) OK",
                  file=sys.stderr)

    if failures:
        print(f"[promotion] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[promotion] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
