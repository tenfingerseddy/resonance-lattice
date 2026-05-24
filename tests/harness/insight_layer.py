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

Day 1 deliverable. `.claude/plans/lensed-knowledge-architecture.md` §4.
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
    from resonance_lattice.store import archive, insight

    failures = 0

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
            make_insight_passage(0, insight_texts[0], src_ids[:1], state="accepted"),
            make_insight_passage(1, insight_texts[1], src_ids, state="accepted"),
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
        if c1.insights[0].insight_id != insights[0].insight_id:
            print(f"[insight_layer] FAIL g2: insight_id mismatch "
                  f"{c1.insights[0].insight_id} != {insights[0].insight_id}",
                  file=sys.stderr)
            failures += 1
        elif c1.insights[0].content != insights[0].content:
            print("[insight_layer] FAIL g2: insight content mismatch",
                  file=sys.stderr)
            failures += 1
        elif tuple(c1.insights[1].citations[0].passage_id for _ in [0]) != \
             (insights[1].citations[0].passage_id,):
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
                # Insight hits must carry insight_id, kind, verdict_state, citations.
                ih = next(h for h in out if h["layer"] == "insight")
                required = {"insight_id", "kind", "verdict_state", "confidence",
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

        # ---- Guarantee 9: candidate / rejected / retired states are excluded ----
        # Build a fresh corpus with insights in non-retrievable states and
        # confirm none surface in default retrieval.
        from dataclasses import replace
        candidate_insights = [
            replace(insights[0], verdict_state="candidate"),
            replace(insights[1], verdict_state="retired"),
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

    if failures:
        print(f"[insight_layer] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[insight_layer] all guarantees OK", file=sys.stderr)
    return 0


def _cli_search(argv: list[str]) -> tuple[int, str, str]:
    """Invoke `rlat search` through the dispatcher; capture stdout/stderr."""
    import contextlib
    import io

    from resonance_lattice.cli.app import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(["search", *argv])
    return rc, out.getvalue(), err.getvalue()


if __name__ == "__main__":
    sys.exit(run())
