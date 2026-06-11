"""insight_lifecycle — verdict state machine + drift cascade.

Guarantees (Day 2 of lensed-knowledge build):

  1. record_verdict appends a verdict signal without mutating the input.
  2. consolidate transitions per the architecture §4.4 table:
     - candidate + compression_pass + verdict not net-negative + diversity
       → active
     - candidate + compression_fail → retired
     - any state + user reject → retired (user authority wins)
     - candidate + /correct → retired
     - stale + re-verify pass → active
     - stale + re-verify fail → retired
     - retired is absorbing
  3. Claim.trust is the derived Beta mean (corroboration /
     total); accumulate_outcome moves it slowly and signed by the outcome.
  4. propagate_drift flips active claims to stale when any cited
     source content_hash changes; preserves indices; stale rows take
     a falsification hit so confidence visibly drops.
  5. apply_drift_cascade_to_archive runs end-to-end against a real .rlat,
     rewrites the insight band in place, and reflects in next read.
  6. Drift cascade is no-op when no insight layer is present.
  7. consolidate retires an insight whose confidence falls below
     RETIRE_FLOOR; a healthy insight is left untouched.
  8. apply_attribution_to_archive skips the .rlat rewrite entirely when a
     reducer credits an insight with zero net weight — nothing moved,
     nothing is re-zipped.
  9. The poison-guarded criterion reducer moves corpus-claim trust on a real
     .rlat: a satisfied load-bearing intent corroborates; a low-confidence
     not_satisfied attenuates but does not retire a healthy claim.
  10. apply_attribution_to_archive routes a mixed band by source — corpus via
     consolidate_corpus, experience via the recurrence + trust earning gate.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim as _make_insight
from ._testutil import make_experience_claim as _make_exp
from ._testutil import unpatch_zero_encoder


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.state.claim_lifecycle import (
        GateSignals,
        accumulate_outcome,
        consolidate_corpus,
        propagate_drift,
        record_verdict,
    )
    from resonance_lattice.store.insight_lifecycle import (
        apply_attribution_to_archive,
        apply_drift_cascade_to_archive,
    )

    failures = 0

    # ---- Guarantee 1: record_verdict is pure + appends signal ----
    ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"])
    after = record_verdict(ins, source="user", polarity="accept")
    if len(after.facts.verdict_signals) != 1:
        print(f"[insight_lifecycle] FAIL g1: signals={len(after.facts.verdict_signals)}",
              file=sys.stderr)
        failures += 1
    elif len(ins.facts.verdict_signals) != 0:
        print("[insight_lifecycle] FAIL g1: input mutated", file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g1 (record_verdict pure + appends signal) OK",
              file=sys.stderr)

    # ---- Guarantee 2a: candidate + pass + accept + 2 distinct cites → active ----
    ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"], state="candidate")
    ins = record_verdict(ins, source="user", polarity="accept")
    ins = consolidate_corpus(ins, signals=GateSignals(compression_test_pass=True))
    if ins.state != "active":
        print(f"[insight_lifecycle] FAIL g2a: state={ins.state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2a (candidate→active on test+verdict+diversity) OK",
              file=sys.stderr)

    # ---- Guarantee 2b: candidate + 1 citation → stays candidate ----
    ins = _make_insight("x", ["p1"], ["h1"], state="candidate")
    ins = record_verdict(ins, source="user", polarity="accept")
    ins = consolidate_corpus(ins, signals=GateSignals(compression_test_pass=True))
    if ins.state != "candidate":
        print(f"[insight_lifecycle] FAIL g2b: single-citation got "
              f"state={ins.state}, expected candidate (anti-paraphrase)",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2b (anti-paraphrase guard holds candidate) OK",
              file=sys.stderr)

    # ---- Guarantee 2c: compression test fail → retired ----
    ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"], state="candidate")
    ins = record_verdict(ins, source="user", polarity="accept")
    ins = consolidate_corpus(ins, signals=GateSignals(compression_test_pass=False))
    if ins.state != "retired":
        print(f"[insight_lifecycle] FAIL g2c: state={ins.state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2c (test fail → retired) OK", file=sys.stderr)

    # ---- Guarantee 2d: user reject overrides active state ----
    ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"], state="active")
    ins = record_verdict(ins, source="user", polarity="reject")
    ins = consolidate_corpus(ins)
    if ins.state != "retired":
        print(f"[insight_lifecycle] FAIL g2d: state={ins.state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2d (user reject overrides active) OK",
              file=sys.stderr)

    # ---- Guarantee 2e: /correct → retired ----
    ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"], state="candidate")
    replacement = _make_insight("y", ["p1", "p2"], ["h1", "h2"], state="candidate")
    ins = consolidate_corpus(ins, signals=GateSignals(correction_replacement=replacement))
    if ins.state != "retired":
        print(f"[insight_lifecycle] FAIL g2e: state={ins.state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2e (/correct → retired) OK",
              file=sys.stderr)

    # ---- Guarantee 2f: stale + re-verify pass → active ----
    ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"], state="stale")
    ins = record_verdict(ins, source="user", polarity="accept")
    ins = consolidate_corpus(ins, signals=GateSignals(compression_test_pass=True))
    if ins.state != "active":
        print(f"[insight_lifecycle] FAIL g2f: state={ins.state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2f (stale + reverify pass → active) OK",
              file=sys.stderr)

    # ---- Guarantee 2g: stale + re-verify fail → retired ----
    ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"], state="stale")
    ins = consolidate_corpus(ins, signals=GateSignals(compression_test_pass=False))
    if ins.state != "retired":
        print(f"[insight_lifecycle] FAIL g2g: state={ins.state}",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g2g (stale + reverify fail → retired) OK",
              file=sys.stderr)

    # ---- Guarantee 2h: retired absorbing ----
    for absorbing in ("retired",):
        ins = _make_insight("x", ["p1", "p2"], ["h1", "h2"], state=absorbing)
        ins = record_verdict(ins, source="user", polarity="accept")
        out = consolidate_corpus(ins, signals=GateSignals(compression_test_pass=True))
        if out.state != absorbing:
            print(f"[insight_lifecycle] FAIL g2h: {absorbing} → {out.state}",
                  file=sys.stderr)
            failures += 1
            break
    else:
        print("[insight_lifecycle] g2h (absorbing states stay) OK", file=sys.stderr)

    # ---- Guarantee 3: Beta trust — seeded, bounded, signed, slow ----
    seeded = _make_insight("x", ["p1", "p2"], ["h1", "h2"], faithfulness=0.9)
    base = seeded.trust
    corrob = accumulate_outcome(seeded, corroboration=1.0)
    falsf = accumulate_outcome(seeded, falsification=1.0)
    if not (0.0 < base < 1.0):
        print(f"[insight_lifecycle] FAIL g3: seeded trust out of range {base}",
              file=sys.stderr)
        failures += 1
    elif not (corrob.trust > base > falsf.trust):
        print(f"[insight_lifecycle] FAIL g3: not signed — corroborate="
              f"{corrob.trust:.3f} base={base:.3f} falsify="
              f"{falsf.trust:.3f}", file=sys.stderr)
        failures += 1
    elif (corrob.trust - base) > 0.2:
        print(f"[insight_lifecycle] FAIL g3: one outcome moved trust "
              f"{corrob.trust - base:.3f} — Beta should be slow",
              file=sys.stderr)
        failures += 1
    else:
        print(f"[insight_lifecycle] g3 (Beta trust: base={base:.2f} "
              f"+1→{corrob.trust:.2f} -1→{falsf.trust:.2f}) OK",
              file=sys.stderr)

    # ---- Guarantee 4: propagate_drift flips active → stale ----
    insights = [
        _make_insight("x", ["p1", "p2"], ["h1_old", "h2"], state="active"),
        _make_insight("y", ["p3"], ["h3"], state="active"),
        _make_insight("z", ["p1"], ["h1_old"], state="candidate"),  # not active
    ]
    fresh_hashes = {"p1": "h1_new", "p2": "h2", "p3": "h3"}  # p1 drifted
    updated, drifted_idx = propagate_drift(insights, fresh_hashes)
    if drifted_idx != [0, 2]:
        print(f"[insight_lifecycle] FAIL g4: drifted_idx={drifted_idx}, expected [0, 2]",
              file=sys.stderr)
        failures += 1
    elif updated[0].state != "stale":
        print(f"[insight_lifecycle] FAIL g4: row 0 state={updated[0].state}",
              file=sys.stderr)
        failures += 1
    elif updated[1].state != "active":
        print(f"[insight_lifecycle] FAIL g4: row 1 state={updated[1].state} "
              "(should be untouched, p3 not drifted)", file=sys.stderr)
        failures += 1
    elif updated[2].state != "candidate":
        # candidate state stays candidate even though cited source drifted —
        # the cascade only flips active → stale.
        print(f"[insight_lifecycle] FAIL g4: candidate row state changed to "
              f"{updated[2].state}", file=sys.stderr)
        failures += 1
    elif updated[0].trust >= insights[0].trust:
        print(f"[insight_lifecycle] FAIL g4: drifted row trust "
              f"{updated[0].trust:.3f} not below pre-drift "
              f"{insights[0].trust:.3f} (falsification hit missing)",
              file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g4 (drift cascade flips active + "
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
        n_drifted, n_total = apply_drift_cascade_to_archive(km)
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
            _make_insight(texts[0], src_ids[:1], src_hashes[:1], state="active"),
            _make_insight(texts[1], src_ids, src_hashes, state="active"),
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
            dry_run=False,
        ))

        # Now read back and confirm at least one insight flipped to stale.
        c1 = archive.read(km)
        if not c1.insights:
            print("[insight_lifecycle] FAIL g5: insights lost after refresh",
                  file=sys.stderr)
            failures += 1
        elif not any(i.state == "stale" for i in c1.insights):
            print(f"[insight_lifecycle] FAIL g5: no insight flipped to stale; "
                  f"states={[i.state for i in c1.insights]}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[insight_lifecycle] g5 (end-to-end drift cascade via refresh) OK",
                  file=sys.stderr)

    # ---- Guarantee 7: retire floor — sub-floor trust → retired ----
    # faithfulness=0.0 seeds trust at 0.25, below RETIRE_FLOOR (0.3).
    sunk = _make_insight("x", ["p1", "p2"], ["h1", "h2"], faithfulness=0.0)
    retired = consolidate_corpus(sunk)
    healthy = _make_insight("y", ["p1", "p2"], ["h1", "h2"], faithfulness=0.9)
    kept = consolidate_corpus(healthy)
    if retired.state != "retired":
        print(f"[insight_lifecycle] FAIL g7: sub-floor insight (trust "
              f"{sunk.trust:.2f}) not retired — state={retired.state}",
              file=sys.stderr)
        failures += 1
    elif kept.state == "retired":
        print(f"[insight_lifecycle] FAIL g7: healthy insight (trust "
              f"{healthy.trust:.2f}) wrongly retired", file=sys.stderr)
        failures += 1
    else:
        print("[insight_lifecycle] g7 (retire floor drops sub-floor insights) OK",
              file=sys.stderr)

    # ---- Guarantee 8: zero-weight attribution skips the .rlat rewrite ----
    with tempfile.TemporaryDirectory() as d:
        from resonance_lattice.field.encoder import Encoder
        from resonance_lattice.store import archive
        from resonance_lattice.store.insight_attribution import InsightWeight

        root = Path(d) / "corpus"
        km = _build(root, {"a.md": "# Alpha\n\nAuthentication flows."})
        c0 = archive.read(km)
        src = c0.registry[0]
        band = Encoder().encode(["Auth uses session tokens."]).astype("float32")
        seeded = [_make_insight("Auth uses session tokens.",
                                [src.passage_id], [src.content_hash],
                                state="active")]
        archive.write_insight_layer_in_place(km, seeded, band)
        ins_id = seeded[0].claim_id

        before = km.read_bytes()
        # A reducer crediting a real insight with zero net weight: the row
        # is left identical, so the whole-ZIP rewrite must be skipped.
        zero_reducer = lambda outcomes: {  # noqa: E731
            ins_id: InsightWeight(corroboration=0.0, falsification=0.0)
        }
        n_updated, n_retired = apply_attribution_to_archive(
            km, [], reducer=zero_reducer,
        )
        after = km.read_bytes()
        if (n_updated, n_retired) != (0, 0):
            print(f"[insight_lifecycle] FAIL g8: zero-weight pass reported "
                  f"({n_updated}, {n_retired}) != (0, 0)", file=sys.stderr)
            failures += 1
        elif after != before:
            print("[insight_lifecycle] FAIL g8: archive was rewritten "
                  "despite zero tally change", file=sys.stderr)
            failures += 1
        else:
            print("[insight_lifecycle] g8 (zero-weight attribution skips "
                  "the rewrite) OK", file=sys.stderr)

    # ---- Guarantee 9: criterion reducer + apply, §D (trust math) ----
    # The poison-guarded criterion reducer moves corpus-claim trust on a real
    # .rlat WHEN a CriterionOutcome's attributed ids match band claims: a
    # satisfied+load-bearing intent corroborates; a LOW-confidence not_satisfied
    # attenuates (falsifies a little) but does NOT retire a healthy claim.
    # SCOPE (honest, per S4 §B): this drives apply_attribution_to_archive with a
    # hand-built CriterionOutcome carrying matching band ids — it proves the
    # reducer + apply + poison-guard math, NOT the full production path. In real
    # operation an intent's attribution ids come from the RecallCache, which
    # surfaces EXPERIENCE-store ids today (not insight-band corpus ids), so the
    # criterion path moves zero *corpus* trust until S3's unified recall stamps
    # band ids into attribution. The corpus loop CLOSES at S3; this gate proves
    # the keystone machinery is correct in isolation.
    with tempfile.TemporaryDirectory() as d:
        from resonance_lattice.field.encoder import Encoder
        from resonance_lattice.store import archive
        from resonance_lattice.store.insight_attribution import (
            CriterionOutcome,
            criterion_weighted,
        )

        root = Path(d) / "corpus"
        km = _build(root, {
            "a.md": "# Alpha\n\nAuthentication flows.",
            "b.md": "# Beta\n\nToken storage.",
        })
        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry[:2]]
        src_hashes = [c.content_hash for c in c0.registry[:2]]
        texts = ["Auth uses session tokens.", "Tokens persist in Redis."]
        band = Encoder().encode(texts).astype("float32")
        # Both seeded healthy (faithfulness 0.9 → trust well above RETIRE_FLOOR).
        seeded = [
            _make_insight(texts[0], src_ids[:1], src_hashes[:1],
                          state="active", faithfulness=0.9),
            _make_insight(texts[1], src_ids, src_hashes,
                          state="active", faithfulness=0.9),
        ]
        archive.write_insight_layer_in_place(km, seeded, band)
        a_id, b_id = seeded[0].claim_id, seeded[1].claim_id
        a_seed, b_seed = seeded[0].trust, seeded[1].trust

        # A: a resolved intent whose criteria were satisfied, load-bearing on A
        # (primary tier), high confidence, measured. B: a low-confidence
        # not_satisfied attributing to B (primary) — the poison-guard case.
        outcomes = [
            CriterionOutcome(((a_id, "primary"),), "satisfied", "high",
                             "signal", "user"),
            CriterionOutcome(((b_id, "primary"),), "not_satisfied", "low",
                             "signal", "user"),
        ]
        n_updated, n_retired = apply_attribution_to_archive(
            km, outcomes, reducer=criterion_weighted,
        )
        by_id = {i.claim_id: i for i in archive.read(km).insights}
        if (n_updated, n_retired) != (2, 0):
            print(f"[insight_lifecycle] FAIL g9: counts ({n_updated}, "
                  f"{n_retired}) != (2, 0)", file=sys.stderr)
            failures += 1
        elif not (by_id[a_id].trust > a_seed and by_id[a_id].state == "active"):
            print(f"[insight_lifecycle] FAIL g9: satisfied did not corroborate "
                  f"A — trust {by_id[a_id].trust:.3f} vs seed {a_seed:.3f}, "
                  f"state {by_id[a_id].state}", file=sys.stderr)
            failures += 1
        elif not (by_id[b_id].trust < b_seed):
            print(f"[insight_lifecycle] FAIL g9: low-conf not_satisfied did "
                  f"not attenuate-falsify B — trust {by_id[b_id].trust:.3f} "
                  f"vs seed {b_seed:.3f}", file=sys.stderr)
            failures += 1
        elif by_id[b_id].state != "active":
            print(f"[insight_lifecycle] FAIL g9: a single low-confidence "
                  f"not_satisfied RETIRED a healthy claim — state "
                  f"{by_id[b_id].state}, trust {by_id[b_id].trust:.3f} "
                  f"(must stay active, poison guard)", file=sys.stderr)
            failures += 1
        else:
            print(f"[insight_lifecycle] g9 (criterion path §D: A corroborated "
                  f"{a_seed:.2f}→{by_id[a_id].trust:.2f} active; B attenuated "
                  f"{b_seed:.2f}→{by_id[b_id].trust:.2f} still active) OK",
                  file=sys.stderr)

    # ---- Guarantee 10: attribution routes by source over a mixed band ----
    # §D for deliverable 2's source-aware lifecycle. A unified band holding a
    # corpus claim + two experience claims: apply_attribution_to_archive must
    # route each to its own consolidator — corpus through consolidate_corpus,
    # experience through the recurrence+trust earning gate — without crashing
    # on the experience claims' missing CorpusFacts fields. A corroborated
    # candidate experience claim (recurrence ≥ 2) EARNS active; a hard-
    # falsified one retires; the corpus claim still corroborates.
    # SCOPE: the experience claims are hand-placed in the band here (as g9
    # hand-builds its CriterionOutcome); the production path that recalls
    # experience claims into the band is deliverable 3.
    with tempfile.TemporaryDirectory() as d:
        from resonance_lattice.field.encoder import Encoder
        from resonance_lattice.store import archive
        from resonance_lattice.store.insight_attribution import InsightWeight

        root = Path(d) / "corpus"
        km = _build(root, {
            "a.md": "# Alpha\n\nAuthentication flows.",
            "b.md": "# Beta\n\nToken storage.",
        })
        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry[:2]]
        src_hashes = [c.content_hash for c in c0.registry[:2]]

        texts = [
            "Auth uses session tokens.",      # corpus — corroborated
            "Prefer pytest fixtures here.",   # experience candidate — earns active
            "The repo uses poetry.",          # experience active — falsified → retired
        ]
        band = Encoder().encode(texts).astype("float32")
        corpus_a = _make_insight(texts[0], src_ids[:1], src_hashes[:1],
                                 state="active", faithfulness=0.9)
        # Born candidate, recurrence 2 (recurred once), medium seed (trust 0.50).
        exp_earn = _make_exp(
            claim_id="01HZEXPEARN0000000000000A1", content=texts[1],
            polarity=("factual",), transcript_hash="manual",
            confidence="medium", recurrence_count=2, state="candidate",
        )
        exp_falsify = _make_exp(
            claim_id="01HZEXPFALS0000000000000B2", content=texts[2],
            polarity=("factual",), transcript_hash="manual",
            confidence="medium", recurrence_count=4, state="active",
        )
        archive.write_insight_layer_in_place(
            km, [corpus_a, exp_earn, exp_falsify], band,
        )
        a_id, e_earn, e_fals = (
            corpus_a.claim_id, exp_earn.claim_id, exp_falsify.claim_id,
        )
        a_seed = corpus_a.trust

        # A hand-built weights dict isolates the source routing from any one
        # reducer (as g8 does). Corroborate A + the candidate experience
        # claim; hard-falsify the active experience claim below RETIRE_FLOOR.
        weights = {
            a_id: InsightWeight(corroboration=2.0, falsification=0.0),
            e_earn: InsightWeight(corroboration=3.0, falsification=0.0),
            e_fals: InsightWeight(corroboration=0.0, falsification=8.0),
        }
        n_updated, n_retired = apply_attribution_to_archive(
            km, [], reducer=lambda _o: weights,
        )
        by_id = {i.claim_id: i for i in archive.read(km).insights}
        if (n_updated, n_retired) != (3, 1):
            print(f"[insight_lifecycle] FAIL g10: counts ({n_updated}, "
                  f"{n_retired}) != (3, 1)", file=sys.stderr)
            failures += 1
        elif not (by_id[a_id].state == "active"
                  and by_id[a_id].trust > a_seed):
            print(f"[insight_lifecycle] FAIL g10: corpus claim mis-handled — "
                  f"state {by_id[a_id].state}, trust {by_id[a_id].trust:.3f} "
                  f"vs seed {a_seed:.3f}", file=sys.stderr)
            failures += 1
        elif by_id[e_earn].state != "active":
            print(f"[insight_lifecycle] FAIL g10: corroborated candidate "
                  f"experience claim did not EARN active — state "
                  f"{by_id[e_earn].state}, trust {by_id[e_earn].trust:.3f}",
                  file=sys.stderr)
            failures += 1
        elif by_id[e_fals].state != "retired":
            print(f"[insight_lifecycle] FAIL g10: falsified experience claim "
                  f"not retired — state {by_id[e_fals].state}, trust "
                  f"{by_id[e_fals].trust:.3f}", file=sys.stderr)
            failures += 1
        else:
            print(f"[insight_lifecycle] g10 (attribution routes by source: "
                  f"corpus corroborated active; experience earned active "
                  f"{by_id[e_earn].trust:.2f}; experience falsified→retired) OK",
                  file=sys.stderr)

    if failures:
        print(f"[insight_lifecycle] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[insight_lifecycle] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
