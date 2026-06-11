"""search_lens_cli — `rlat search --lens` end-to-end.

Guarantees:

  1. `rlat search --lens X.lens` re-ranks source hits by lens.trust_weights.
  2. `rlat search --lens X.lens` re-ranks insight hits by lens.insight_preferences.
  3. `rlat search --source-only` ignores the lens even when --lens is passed.
  4. Search spills NO sidecar beside the corpus, and a bare search (persistence
     default-off) is read-only — capture lives in-memory at the heart
     (field.retrieve / retrieve_insight) and folds INSIDE the .rlat only at a
     session boundary (tests/harness/telemetry.py), never into a sidecar.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim, run_cli, unpatch_zero_encoder


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
        insights = [make_corpus_claim(
            "Sessions last 24h via session tokens.",
            src_ids[:1], src_hashes[:1], state="active",
        )]
        band = encoder.encode([insights[0].content]).astype("float32")
        archive.write_insight_layer_in_place(km, insights, band)

        # ---- Guarantee 1+2: --lens re-ranks ----
        lens_path = _make_lens(
            Path(d), "lens-eng", "engineering",
            trust_weights=[("src/*", 2.0), ("docs/external/*", 0.1)],
            insight_prefs=[(insights[0].facts.content_fingerprint, 3.0)],
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

        # ---- Guarantee 4: no sidecar; persistence is in-file + opt-in ----
        # Capture lives at the heart (field.retrieve / retrieve_insight, in
        # memory — pinned by tests/harness/capture.py) and folds INSIDE the
        # .rlat at a session boundary (store.telemetry — pinned by
        # tests/harness/telemetry.py), never into a sidecar next to the corpus.
        #   4a: a bare search (persistence default-off) writes nothing at all.
        #   4b: with persistence ON, the SAME shipped cmd_search path folds a
        #       row INTO the .rlat — locks the cmd_search->telemetry.flush wiring
        #       positively, not just the negative read-only check.
        import os
        from resonance_lattice.field import capture as _capture
        km_key = str(Path(km).resolve())
        persist_env = {k: os.environ.pop(k, None)
                       for k in ("RLAT_CAPTURE_PERSIST", "RLAT_DOGFOOD_SESSION")}
        try:
            before = {p for p in Path(d).rglob("*") if p.is_file()}
            run_cli([
                "search", str(km), "auto-record test",
                "--top-k", "3", "--format", "json", "--quiet", "--lens", str(lens_path),
            ])
            spilled = sorted(
                str(p) for p in ({q for q in Path(d).rglob("*") if q.is_file()} - before)
            )
            if spilled:
                print(f"[search_lens_cli] FAIL g4a: search spilled file(s) beside "
                      f"the corpus: {spilled}", file=sys.stderr)
                failures += 1
            elif archive.read_telemetry(km):
                print("[search_lens_cli] FAIL g4a: bare search mutated the .rlat "
                      "(persistence is opt-in)", file=sys.stderr)
                failures += 1
            else:
                print("[search_lens_cli] g4a (no sidecar; bare search is "
                      "read-only) OK", file=sys.stderr)

            os.environ["RLAT_CAPTURE_PERSIST"] = "1"
            run_cli([
                "search", str(km), "auto-record test",
                "--top-k", "3", "--format", "json", "--quiet", "--lens", str(lens_path),
            ])
            telem = archive.read_telemetry(km)
            spilled2 = sorted(
                str(p) for p in ({q for q in Path(d).rglob("*") if q.is_file()} - before)
            )
            if not telem:
                print("[search_lens_cli] FAIL g4b: persisted search wrote NO "
                      "telemetry row into the .rlat (cmd_search->flush wiring "
                      "broken)", file=sys.stderr)
                failures += 1
            elif spilled2:
                print(f"[search_lens_cli] FAIL g4b: persisted search spilled a "
                      f"sidecar beside the corpus: {spilled2}", file=sys.stderr)
                failures += 1
            else:
                print(f"[search_lens_cli] g4b (persist-on: CLI search folds "
                      f"{len(telem)} row(s) INTO the .rlat, no sidecar) OK",
                      file=sys.stderr)
        finally:
            for k, v in persist_env.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)
            _capture.drain(km_key)  # don't leave a dead-path buffer entry

    failures += _check_set_trust_round_trip()

    if failures:
        print(f"[search_lens_cli] {failures} guarantee(s) failed",
              file=sys.stderr)
        return 1
    print("[search_lens_cli] all guarantees OK", file=sys.stderr)
    return 0


def _check_set_trust_round_trip() -> int:
    """Guarantee: `rlat lens set-trust` — the write surface the 2026-06
    review added (the lens layer was write-blind) — adds, updates, and
    removes a pattern, surviving the save/load round trip."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        lens_path = str(Path(d) / "t.lens")
        rc, _, _ = run_cli(["lens", "create", "--id", "t1", "--name", "T",
                            "-o", lens_path])
        rc1, out1, _ = run_cli(["lens", "set-trust", lens_path,
                                "docs/external/*", "0.2"])
        rc2, out2, _ = run_cli(["lens", "set-trust", lens_path,
                                "docs/external/*", "0.5"])
        rc3, show_out, _ = run_cli(["lens", "show", lens_path,
                                    "--format", "json"])
        try:
            weights = {tw["pattern"]: tw["weight"]
                       for tw in json.loads(show_out)["trust_weights"]}
        except (ValueError, KeyError):   # non-JSON / failed show -> guarantee fails, not a crash
            weights = None
        rc4, _, _ = run_cli(["lens", "set-trust", lens_path,
                             "docs/external/*", "--remove"])
        rc5, show2, _ = run_cli(["lens", "show", lens_path, "--format", "json"])
        try:
            gone = json.loads(show2)["trust_weights"] == []
        except (ValueError, KeyError):
            gone = False
        rc6, _, err6 = run_cli(["lens", "set-trust", lens_path, "x/*", "-1"])

    ok = (rc == 0 and rc1 == 0 and rc2 == 0 and rc3 == 0
          and weights == {"docs/external/*": 0.5}
          and rc4 == 0 and rc5 == 0 and gone and rc6 != 0)
    if not ok:
        print(f"[search_lens_cli] FAIL set-trust round trip: rcs="
              f"{(rc, rc1, rc2, rc3, rc4, rc5, rc6)} weights={weights!r} "
              f"gone={gone}", file=sys.stderr)
        return 1
    print("[search_lens_cli] set-trust add/update/remove round trip OK",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
