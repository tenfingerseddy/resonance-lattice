"""search_lens_cli — `rlat search --lens` end-to-end + auto-recorded dogfood
events.

Guarantees:

  1. `rlat search --lens X.lens` re-ranks source hits by lens.trust_weights.
  2. `rlat search --lens X.lens` re-ranks insight hits by lens.insight_preferences.
  3. `rlat search --source-only` ignores the lens even when --lens is passed.
  4. With `.rlat-state/ledger/` present, every search appends one dogfood
     event with the lens_id stamped.
  5. Without `.rlat-state/ledger/`, no event is recorded (opt-in by presence).
  6. `bench_lensed_dogfood scorecard` aggregates the recorded events.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_insight_passage, run_cli, unpatch_zero_encoder


def _make_lens(d: Path, lens_id: str, name: str,
               trust_weights=None, insight_prefs=None):
    from resonance_lattice.lens import schema as lens_mod
    lens = lens_mod.new_lens(lens_id=lens_id, scope="user", name=name)
    if trust_weights:
        lens.trust_weights = [
            lens_mod.TrustWeight(pattern=p, weight=w) for p, w in trust_weights
        ]
    if insight_prefs:
        lens.insight_preferences = [
            lens_mod.InsightPreference(insight_id=k, weight=v) for k, v in insight_prefs
        ]
    out = d / f"{lens_id}.lens"
    lens_mod.save(lens, out)
    return out


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store import archive

    failures = 0

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        files = {
            "src/auth.py": "# Auth\n\nSession tokens expire after 24 hours.",
            "docs/external/blog.md": "Random blog about session timeouts.",
        }
        km = _build(root, files)
        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry]
        src_hashes = [c.content_hash for c in c0.registry]

        encoder = Encoder()
        insights = [make_insight_passage(
            0, "Sessions last 24h via session tokens.",
            src_ids[:1], src_hashes[:1], state="accepted",
        )]
        band = encoder.encode([insights[0].content]).astype("float32")
        archive.write_insight_layer_in_place(km, insights, band)

        # ---- Guarantee 1+2: --lens re-ranks ----
        lens_path = _make_lens(
            Path(d), "lens-eng", "engineering",
            trust_weights=[("src/*", 2.0), ("docs/external/*", 0.1)],
            insight_prefs=[(insights[0].insight_id, 3.0)],
        )
        rc, out, _ = run_cli([
            "search", str(km), "session tokens",
            "--top-k", "5", "--format", "json", "--quiet",
            "--lens", str(lens_path),
        ])
        if rc != 0:
            print(f"[search_lens_cli] FAIL g1+g2: rc={rc}", file=sys.stderr)
            failures += 1
        else:
            parsed = json.loads(out)
            src_hits = [h for h in parsed if h["layer"] == "source"]
            ins_hits = [h for h in parsed if h["layer"] == "insight"]

            # Baseline (no lens) for comparison.
            rc2, out2, _ = run_cli([
                "search", str(km), "session tokens",
                "--top-k", "5", "--format", "json", "--quiet",
            ])
            base = json.loads(out2)
            base_src = [h for h in base if h["layer"] == "source"]
            base_ins = [h for h in base if h["layer"] == "insight"]

            # src/* hits should have higher score than baseline (2x trust)
            base_src_scores = {h["source_file"]: h["score"] for h in base_src}
            lensed_src_scores = {h["source_file"]: h["score"] for h in src_hits}
            src_lensed_lift_ok = any(
                lensed_src_scores.get(f, 0) > base_src_scores.get(f, 0) * 1.5
                for f in lensed_src_scores
                if f.startswith("src/")
            )
            if not src_lensed_lift_ok and base_src and src_hits:
                print(f"[search_lens_cli] FAIL g1: src trust didn't lift "
                      f"(baseline={base_src_scores}, lensed={lensed_src_scores})",
                      file=sys.stderr)
                failures += 1
            elif ins_hits and base_ins:
                ratio = ins_hits[0]["score"] / max(1e-9, base_ins[0]["score"])
                if abs(ratio - 3.0) > 0.05:
                    print(f"[search_lens_cli] FAIL g2: insight preference "
                          f"multiplier {ratio:.2f} != 3.0", file=sys.stderr)
                    failures += 1
                else:
                    print("[search_lens_cli] g1+g2 (lens trust + preference re-rank) OK",
                          file=sys.stderr)
            else:
                print("[search_lens_cli] g1+g2 OK (no insight to test g2)",
                      file=sys.stderr)

        # ---- Guarantee 3: --source-only ignores --lens ----
        rc, out, _ = run_cli([
            "search", str(km), "session tokens",
            "--top-k", "5", "--format", "json", "--quiet",
            "--source-only", "--lens", str(lens_path),
        ])
        if rc != 0:
            print(f"[search_lens_cli] FAIL g3: rc={rc}", file=sys.stderr)
            failures += 1
        else:
            parsed = json.loads(out)
            if any(h["layer"] == "insight" for h in parsed):
                print("[search_lens_cli] FAIL g3: --source-only with --lens "
                      "returned insight hits", file=sys.stderr)
                failures += 1
            else:
                # Source hits in source-only mode must match the source-only
                # baseline scores for the same passage_idx. The test against
                # the no-lens (but insight-included) baseline is unreliable
                # because top-K dedup may surface different chunks of the
                # same source file when insight crowds the merge.
                rc3, out3, _ = run_cli([
                    "search", str(km), "session tokens",
                    "--top-k", "5", "--format", "json", "--quiet",
                    "--source-only",
                ])
                no_lens_so = {
                    h["passage_idx"]: h["score"]
                    for h in json.loads(out3) if h["layer"] == "source"
                }
                lensed_so = {
                    h["passage_idx"]: h["score"]
                    for h in parsed if h["layer"] == "source"
                }
                drift = [
                    (pi, no_lens_so[pi], lensed_so[pi])
                    for pi in lensed_so
                    if pi in no_lens_so and abs(no_lens_so[pi] - lensed_so[pi]) > 1e-4
                ]
                if drift:
                    print(f"[search_lens_cli] FAIL g3: --source-only applied "
                          f"lens (drifted passages: {drift})", file=sys.stderr)
                    failures += 1
                else:
                    print("[search_lens_cli] g3 (--source-only ignores --lens) OK",
                          file=sys.stderr)

        # ---- Guarantee 4+5: auto-record dogfood event ----
        ledger_dir = Path(d) / ".rlat-state" / "ledger"

        # G5: no ledger dir → no event
        rc, _, _ = run_cli([
            "search", str(km), "any query",
            "--top-k", "1", "--format", "json", "--quiet",
        ])
        if (ledger_dir / "dogfood_events.jsonl").exists():
            print("[search_lens_cli] FAIL g5: event recorded without ledger dir",
                  file=sys.stderr)
            failures += 1

        # G4: create ledger dir, run search, expect event appended.
        # The cwd-based check means we cd into the tmp dir for this part.
        import os
        old_cwd = Path.cwd()
        try:
            ledger_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(d)
            rc, _, _ = run_cli([
                "search", str(km), "auto-record test",
                "--top-k", "3", "--format", "json", "--quiet",
                "--lens", str(lens_path),
            ])
            events_file = ledger_dir / "dogfood_events.jsonl"
            if not events_file.exists():
                print("[search_lens_cli] FAIL g4: no event file created",
                      file=sys.stderr)
                failures += 1
            else:
                lines = events_file.read_text(encoding="utf-8").splitlines()
                if not lines:
                    print("[search_lens_cli] FAIL g4: event file empty",
                          file=sys.stderr)
                    failures += 1
                else:
                    event = json.loads(lines[-1])
                    if event["query"] != "auto-record test":
                        print(f"[search_lens_cli] FAIL g4: event query wrong: "
                              f"{event['query']}", file=sys.stderr)
                        failures += 1
                    elif event["lens_id"] != "lens-eng":
                        print(f"[search_lens_cli] FAIL g4: event lens_id wrong: "
                              f"{event['lens_id']}", file=sys.stderr)
                        failures += 1
                    elif event["duration_ms"] <= 0:
                        print(f"[search_lens_cli] FAIL g4: duration_ms not positive: "
                              f"{event['duration_ms']}", file=sys.stderr)
                        failures += 1
                    else:
                        print("[search_lens_cli] g4 (auto-record dogfood event) OK",
                              file=sys.stderr)
                        print("[search_lens_cli] g5 (no record without ledger dir) OK",
                              file=sys.stderr)

                        # ---- Guarantee 6: scorecard aggregates events ----
                        # Import the bench module and aggregate.
                        sys.path.insert(0, str(old_cwd / "benchmarks"))
                        try:
                            import bench_lensed_dogfood as bld
                            events = bld.read_events(events_file)
                            sc = bld.compute_scorecard(events)
                            if sc.total_events < 1:
                                print(f"[search_lens_cli] FAIL g6: scorecard "
                                      f"total_events={sc.total_events}",
                                      file=sys.stderr)
                                failures += 1
                            else:
                                print("[search_lens_cli] g6 (scorecard aggregates events) OK",
                                      file=sys.stderr)
                        finally:
                            sys.path.pop(0)
        finally:
            os.chdir(old_cwd)

    if failures:
        print(f"[search_lens_cli] {failures} guarantee(s) failed",
              file=sys.stderr)
        return 1
    print("[search_lens_cli] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
