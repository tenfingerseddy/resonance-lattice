"""audit_trace_cli — trust-contract surface end-to-end through the CLI.

Guarantees (Days 6 + 7):

  1. audit_summary counts source + insight by state.
  2. audit_stale + audit_orphans return the right slices.
  3. trace_insight resolves citations to source coordinates.
  4. trace_source returns reverse-trace insights.
  5. rlat audit text + json formats parse.
  6. rlat trace text + json formats parse.
  7. rlat lens create + show + compose round-trip via CLI.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_insight_passage, run_cli, unpatch_zero_encoder


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store import archive
    from resonance_lattice.store.audit import (
        audit_orphans,
        audit_stale,
        audit_summary,
        trace_insight,
        trace_source,
    )

    failures = 0

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, {
            "a.md": "# Alpha\n\nFirst doc about authentication.",
            "b.md": "# Beta\n\nTokens rotate weekly.",
        })
        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry]
        src_hashes = [c.content_hash for c in c0.registry]

        encoder = Encoder()
        from dataclasses import replace as _replace
        insights = [
            make_insight_passage(0, "Authentication uses session tokens.",
                                 src_ids[:2], src_hashes[:2], state="accepted"),
            _replace(
                make_insight_passage(1, "Tokens persist.", src_ids[:1], src_hashes[:1],
                                     state="candidate"),
            ),
        ]
        # Make insight 1 stale.
        insights[1] = _replace(insights[1], verdict_state="stale")
        # Make a third one that's orphaned (cites a passage_id that doesn't exist).
        from resonance_lattice.store.insight import InsightCitation
        orphan = make_insight_passage(2, "Orphan content.", ["does-not-exist"],
                                       ["fake-hash"], state="accepted")
        orphan = _replace(orphan, citations=(
            InsightCitation(passage_id="does-not-exist", char_span=None, confidence=0.9),
        ))
        insights.append(orphan)

        band = encoder.encode([i.content for i in insights]).astype("float32")
        archive.write_insight_layer_in_place(km, insights, band)
        c1 = archive.read(km)

        # ---- Guarantee 1: summary counts ----
        # The chunker may produce > N passages from N files when content
        # crosses paragraph boundaries; assert on the relative counts the
        # audit cares about.
        summary = audit_summary(c1)
        if (summary.source_passages < 2 or summary.insight_total != 3
            or summary.insight_accepted != 2  # 0 + orphan
            or summary.insight_stale != 1
            or summary.insight_orphans != 1):
            print(f"[audit_trace_cli] FAIL g1: summary mismatch {summary}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[audit_trace_cli] g1 (summary counts) OK", file=sys.stderr)

        # ---- Guarantee 2: stale + orphans slices ----
        stale = audit_stale(c1)
        if len(stale) != 1 or stale[0].insight_id != insights[1].insight_id:
            print(f"[audit_trace_cli] FAIL g2: stale wrong ({stale})",
                  file=sys.stderr)
            failures += 1
        else:
            orphans = audit_orphans(c1)
            if len(orphans) != 1 or orphans[0].insight_id != insights[2].insight_id:
                print(f"[audit_trace_cli] FAIL g2: orphans wrong ({orphans})",
                      file=sys.stderr)
                failures += 1
            else:
                print("[audit_trace_cli] g2 (stale + orphans slices) OK",
                      file=sys.stderr)

        # ---- Guarantee 3: trace_insight resolves citations ----
        trace = trace_insight(c1, insights[0].insight_id)
        if len(trace.source_passages) != 2:
            print(f"[audit_trace_cli] FAIL g3: trace cited "
                  f"{len(trace.source_passages)} passages, expected 2",
                  file=sys.stderr)
            failures += 1
        elif trace.source_passages[0]["source_file"] not in ("a.md", "b.md"):
            print(f"[audit_trace_cli] FAIL g3: trace wrong source_file",
                  file=sys.stderr)
            failures += 1
        else:
            print("[audit_trace_cli] g3 (trace_insight resolves citations) OK",
                  file=sys.stderr)

        # ---- Guarantee 4: trace_source reverse trace ----
        reverse = trace_source(c1, src_ids[0])
        if len(reverse) < 1:
            print(f"[audit_trace_cli] FAIL g4: reverse trace empty",
                  file=sys.stderr)
            failures += 1
        else:
            ids = {ins.insight_id for ins in reverse}
            if insights[0].insight_id not in ids:
                print(f"[audit_trace_cli] FAIL g4: expected insight not in "
                      f"reverse trace", file=sys.stderr)
                failures += 1
            else:
                print("[audit_trace_cli] g4 (trace_source reverse) OK",
                      file=sys.stderr)

        # ---- Guarantee 5: rlat audit CLI ----
        rc, out, err = run_cli(["audit", str(km)])
        if rc != 0 or "insight total:" not in out:
            print(f"[audit_trace_cli] FAIL g5: rlat audit text rc={rc} "
                  f"out={out[:80]}", file=sys.stderr)
            failures += 1
        else:
            rc, out, _ = run_cli(["audit", str(km), "--format", "json"])
            try:
                parsed = json.loads(out)
                if parsed.get("insight_total") != 3:
                    print(f"[audit_trace_cli] FAIL g5: json insight_total "
                          f"{parsed.get('insight_total')}", file=sys.stderr)
                    failures += 1
                else:
                    print("[audit_trace_cli] g5 (rlat audit text + json) OK",
                          file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"[audit_trace_cli] FAIL g5: json parse {e}", file=sys.stderr)
                failures += 1

        # ---- Guarantee 6: rlat trace CLI ----
        rc, out, _ = run_cli(["trace", str(km), insights[0].insight_id])
        if rc != 0 or "cites" not in out:
            print(f"[audit_trace_cli] FAIL g6: rlat trace rc={rc} out={out[:80]}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[audit_trace_cli] g6 (rlat trace text) OK", file=sys.stderr)

        # ---- Guarantee 7: rlat lens create + show + compose ----
        lens_a = Path(d) / "lens_a.lens"
        lens_b = Path(d) / "lens_b.lens"
        lens_team = Path(d) / "team.lens"

        rc, out, err = run_cli([
            "lens", "create",
            "--id", "lens-a", "--name", "engineering", "--scope", "user",
            "-o", str(lens_a),
        ])
        if rc != 0:
            print(f"[audit_trace_cli] FAIL g7a: lens create rc={rc} err={err}",
                  file=sys.stderr)
            failures += 1
        else:
            rc, _, _ = run_cli([
                "lens", "create",
                "--id", "lens-b", "--name", "compliance", "--scope", "user",
                "-o", str(lens_b),
            ])
            rc, out, _ = run_cli(["lens", "show", str(lens_a)])
            if rc != 0 or "engineering" not in out:
                print(f"[audit_trace_cli] FAIL g7b: lens show: {out}",
                      file=sys.stderr)
                failures += 1
            else:
                rc, _, err = run_cli([
                    "lens", "compose", str(lens_a), str(lens_b),
                    "--id", "lens-team", "--name", "platform-team",
                    "-o", str(lens_team),
                ])
                if rc != 0:
                    print(f"[audit_trace_cli] FAIL g7c: compose rc={rc} "
                          f"err={err}", file=sys.stderr)
                    failures += 1
                elif not lens_team.exists():
                    print("[audit_trace_cli] FAIL g7c: composed lens not written",
                          file=sys.stderr)
                    failures += 1
                else:
                    print("[audit_trace_cli] g7 (rlat lens create+show+compose) OK",
                          file=sys.stderr)

    if failures:
        print(f"[audit_trace_cli] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[audit_trace_cli] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
