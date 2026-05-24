"""insight_lifecycle — verdict state machine + drift cascade.

Guarantees (Day 2 of lensed-knowledge build):

  1. record_verdict appends a verdict signal without mutating the input.
  2. consolidate_state transitions per the architecture §4.4 table:
     - candidate + compression_pass + verdict>0 + diversity → accepted
     - candidate + compression_fail → rejected
     - any state + user reject → rejected (user authority wins)
     - candidate + /correct → rejected_corrected
     - stale + re-verify pass → accepted
     - stale + re-verify fail → retired
     - rejected/retired are absorbing
  3. InsightPassage.confidence is the derived Beta mean (corroboration /
     total); accumulate_outcome moves it slowly and signed by the outcome.
  4. propagate_drift flips accepted insights to stale when any cited
     source content_hash changes; preserves indices; stale rows take
     a falsification hit so confidence visibly drops.
  5. apply_drift_cascade_to_archive runs end-to-end against a real .rlat,
     rewrites the insight band in place, and reflects in next read.
  6. Drift cascade is no-op when no insight layer is present.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_insight_passage as _make_insight
from ._testutil import unpatch_zero_encoder


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.store import insight_lifecycle as lc

    failures = 0

    # ---- Guarantee 1: record_verdict is pure + appends signal ----
    ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"])
    after = lc.record_verdict(ins, source="user", polarity="accept")
    if len(after.verdict_signals) != 1:
        print(f"[insight_lifecycle] FAIL g1: signals={len(after.verdict_signals)}",
              file=sys.stderr)
        failures += 1
    elif len(ins.verdict_signals) != 0:
        print("[insight_lifecycle] FAIL g1: input mutated", file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g1 (record_verdict pure + appends signal) OK",
              file=sys.stderr)

    # ---- Guarantee 2a: candidate + pass + accept + 2 distinct cites → accepted ----
    ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], state="candidate")
    ins = lc.record_verdict(ins, source="user", polarity="accept")
    ins = lc.consolidate_state(ins, compression_test_pass=True)
    if ins.verdict_state != "accepted":
        print(f"[insight_lifecycle] FAIL g2a: state={ins.verdict_state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2a (candidate→accepted on test+verdict+diversity) OK",
              file=sys.stderr)

    # ---- Guarantee 2b: candidate + 1 citation → stays candidate ----
    ins = _make_insight(0, "x", ["p1"], ["h1"], state="candidate")
    ins = lc.record_verdict(ins, source="user", polarity="accept")
    ins = lc.consolidate_state(ins, compression_test_pass=True)
    if ins.verdict_state != "candidate":
        print(f"[insight_lifecycle] FAIL g2b: single-citation got "
              f"state={ins.verdict_state}, expected candidate (anti-paraphrase)",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2b (anti-paraphrase guard holds candidate) OK",
              file=sys.stderr)

    # ---- Guarantee 2c: compression test fail → rejected ----
    ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], state="candidate")
    ins = lc.record_verdict(ins, source="user", polarity="accept")
    ins = lc.consolidate_state(ins, compression_test_pass=False)
    if ins.verdict_state != "rejected":
        print(f"[insight_lifecycle] FAIL g2c: state={ins.verdict_state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2c (test fail → rejected) OK", file=sys.stderr)

    # ---- Guarantee 2d: user reject overrides accepted state ----
    ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], state="accepted")
    ins = lc.record_verdict(ins, source="user", polarity="reject")
    ins = lc.consolidate_state(ins)
    if ins.verdict_state != "rejected":
        print(f"[insight_lifecycle] FAIL g2d: state={ins.verdict_state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2d (user reject overrides accepted) OK",
              file=sys.stderr)

    # ---- Guarantee 2e: /correct → rejected_corrected ----
    ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], state="candidate")
    replacement = _make_insight(1, "y", ["p1", "p2"], ["h1", "h2"], state="candidate")
    ins = lc.consolidate_state(ins, correction_replacement=replacement)
    if ins.verdict_state != "rejected_corrected":
        print(f"[insight_lifecycle] FAIL g2e: state={ins.verdict_state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2e (/correct → rejected_corrected) OK",
              file=sys.stderr)

    # ---- Guarantee 2f: stale + re-verify pass → accepted ----
    ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], state="stale")
    ins = lc.record_verdict(ins, source="user", polarity="accept")
    ins = lc.consolidate_state(ins, compression_test_pass=True)
    if ins.verdict_state != "accepted":
        print(f"[insight_lifecycle] FAIL g2f: state={ins.verdict_state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2f (stale + reverify pass → accepted) OK",
              file=sys.stderr)

    # ---- Guarantee 2g: stale + re-verify fail → retired ----
    ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], state="stale")
    ins = lc.consolidate_state(ins, compression_test_pass=False)
    if ins.verdict_state != "retired":
        print(f"[insight_lifecycle] FAIL g2g: state={ins.verdict_state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2g (stale + reverify fail → retired) OK",
              file=sys.stderr)

    # ---- Guarantee 2h: rejected/retired absorbing ----
    for absorbing in ("rejected", "retired", "rejected_corrected"):
        ins = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], state=absorbing)
        ins = lc.record_verdict(ins, source="user", polarity="accept")
        out = lc.consolidate_state(ins, compression_test_pass=True)
        if out.verdict_state != absorbing:
            print(f"[insight_lifecycle] FAIL g2h: {absorbing} → {out.verdict_state}",
                  file=sys.stderr)
            failures += 1
            break
    else:
        print("[insight_lifecycle] g2h (absorbing states stay) OK", file=sys.stderr)

    # ---- Guarantee 3: Beta confidence — seeded, bounded, signed, slow ----
    seeded = _make_insight(0, "x", ["p1", "p2"], ["h1", "h2"], faithfulness=0.9)
    base = seeded.confidence
    corrob = lc.accumulate_outcome(seeded, corroboration=1.0)
    falsf = lc.accumulate_outcome(seeded, falsification=1.0)
    if not (0.0 < base < 1.0):
        print(f"[insight_lifecycle] FAIL g3: seeded conf out of range {base}",
              file=sys.stderr)
        failures += 1
    elif not (corrob.confidence > base > falsf.confidence):
        print(f"[insight_lifecycle] FAIL g3: not signed — corroborate="
              f"{corrob.confidence:.3f} base={base:.3f} falsify="
              f"{falsf.confidence:.3f}", file=sys.stderr)
        failures += 1
    elif (corrob.confidence - base) > 0.2:
        print(f"[insight_lifecycle] FAIL g3: one outcome moved confidence "
              f"{corrob.confidence - base:.3f} — Beta should be slow",
              file=sys.stderr)
        failures += 1
    else:
        print(f"[insight_lifecycle] g3 (Beta confidence: base={base:.2f} "
              f"+1→{corrob.confidence:.2f} -1→{falsf.confidence:.2f}) OK",
              file=sys.stderr)

    # ---- Guarantee 4: propagate_drift flips accepted → stale ----
    insights = [
        _make_insight(0, "x", ["p1", "p2"], ["h1_old", "h2"], state="accepted"),
        _make_insight(1, "y", ["p3"], ["h3"], state="accepted"),
        _make_insight(2, "z", ["p1"], ["h1_old"], state="candidate"),  # not accepted
    ]
    fresh_hashes = {"p1": "h1_new", "p2": "h2", "p3": "h3"}  # p1 drifted
    updated, drifted_idx = lc.propagate_drift(insights, fresh_hashes)
    if drifted_idx != [0, 2]:
        print(f"[insight_lifecycle] FAIL g4: drifted_idx={drifted_idx}, expected [0, 2]",
              file=sys.stderr)
        failures += 1
    elif updated[0].verdict_state != "stale":
        print(f"[insight_lifecycle] FAIL g4: row 0 state={updated[0].verdict_state}",
              file=sys.stderr)
        failures += 1
    elif updated[1].verdict_state != "accepted":
        print(f"[insight_lifecycle] FAIL g4: row 1 state={updated[1].verdict_state} "
              "(should be untouched, p3 not drifted)", file=sys.stderr)
        failures += 1
    elif updated[2].verdict_state != "candidate":
        # candidate state stays candidate even though cited source drifted —
        # the cascade only flips accepted → stale.
        print(f"[insight_lifecycle] FAIL g4: candidate row state changed to "
              f"{updated[2].verdict_state}", file=sys.stderr)
        failures += 1
    elif updated[0].confidence >= insights[0].confidence:
        print(f"[insight_lifecycle] FAIL g4: drifted row confidence "
              f"{updated[0].confidence:.3f} not below pre-drift "
              f"{insights[0].confidence:.3f} (falsification hit missing)",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g4 (drift cascade flips accepted + "
              "falsifies) OK", file=sys.stderr)

    # ---- Guarantee 5 & 6: end-to-end drift via apply_drift_cascade_to_archive ----
    with tempfile.TemporaryDirectory() as d:
        from resonance_lattice.field.encoder import Encoder
        from resonance_lattice.store import archive

        root = Path(d) / "corpus"
        files = {
            "a.md": "# Alpha\n\nAuthentication flows.",
            "b.md": "# Beta\n\nToken management.",
        }
        km = _build(root, files)
        c0 = archive.read(km)

        # G6: no insight layer → no-op
        n_drifted, n_total = lc.apply_drift_cascade_to_archive(km)
        if n_drifted != 0 or n_total != 0:
            print(f"[insight_lifecycle] FAIL g6: empty corpus cascade "
                  f"({n_drifted}, {n_total})", file=sys.stderr)
            failures += 1
        else:
            print("[insight_lifecycle] g6 (drift cascade is no-op without insights) OK",
                  file=sys.stderr)

        # Promote insights citing real source passages
        src_ids = [c.passage_id for c in c0.registry[:2]]
        src_hashes = [c.content_hash for c in c0.registry[:2]]
        encoder = Encoder()
        texts = ["Auth uses session tokens.", "Tokens refresh weekly."]
        band = encoder.encode(texts).astype("float32")
        insights = [
            _make_insight(0, texts[0], src_ids[:1], src_hashes[:1], state="accepted"),
            _make_insight(1, texts[1], src_ids, src_hashes, state="accepted"),
        ]
        archive.write_insight_layer_in_place(km, insights, band)

        # Modify source a.md to force a content_hash change, then refresh
        (root / "a.md").write_text(
            "# Alpha\n\nAuthentication flows with new mechanism details.",
            encoding="utf-8",
        )
        from resonance_lattice.cli.maintain import cmd_refresh
        from ._testutil import Args as _Args
        _rc = cmd_refresh(_Args(
            knowledge_model=str(km),
            source=None, source_root=None, batch_size=4, ext=None,
            discard_optimised=False, dry_run=False,
        ))

        # Now read back and confirm at least one insight flipped to stale.
        c1 = archive.read(km)
        if not c1.insights:
            print("[insight_lifecycle] FAIL g5: insights lost after refresh",
                  file=sys.stderr)
            failures += 1
        elif not any(i.verdict_state == "stale" for i in c1.insights):
            print(f"[insight_lifecycle] FAIL g5: no insight flipped to stale; "
                  f"states={[i.verdict_state for i in c1.insights]}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[insight_lifecycle] g5 (end-to-end drift cascade via refresh) OK",
                  file=sys.stderr)

    if failures:
        print(f"[insight_lifecycle] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[insight_lifecycle] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
