"""insight_attribution — criterion-level reducer + poison guard (S4 d3 + d4).

The reducer folds a resolved intent's per-criterion verdicts into per-claim Beta
weight, attenuated by the poison guard (verdict_confidence × source ×
provenance) so a weak or machine-asserted verdict cannot move a corpus claim's
trust at full weight (GROUNDING_MODEL §"Confidence & attribution").

Guarantees:

  1. Poison guard scales by confidence × source × provenance; unknown keys
     attenuate (fall to the most conservative factor).
  2. Criterion reducer — satisfied corroborates, scaled; unknown is inert;
     tier weights load-bearing (primary) claims more.
  3. not_satisfied falsifies, and a LOW-confidence failure is strictly weaker
     than a HIGH one (a weak verdict cannot retire a good claim).
  4. decisive_verdict — satisfied → weakest confidence + override-if-any;
     not_satisfied → strongest failing check, always source "signal".
  5. A user_override satisfied moves trust strictly less than a measured one.
  6. The poison guard on a REAL low-confidence not_satisfied through the
     production seam (evaluate_criterion → decisive_verdict → criterion_weighted).
"""

from __future__ import annotations

import sys


def run() -> int:
    from resonance_lattice.state.claim_outcome import CriterionCheck
    from resonance_lattice.store.insight_attribution import (
        CriterionOutcome,
        criterion_weighted,
        decisive_verdict,
        poison_guard_scale,
    )

    failures = 0

    def _cc(verdict, confidence, source="signal"):
        return CriterionCheck(
            criterion_text="c", measure="m", verdict=verdict,
            verdict_confidence=confidence, verdict_source=source,
        )

    # ---- Guarantee 1: poison guard scales by confidence × source × provenance
    full = poison_guard_scale("high", "signal", "user")
    low = poison_guard_scale("low", "signal", "user")
    override = poison_guard_scale("high", "user_override", "user")
    harvested = poison_guard_scale("high", "signal", "harvested")
    unknown_key = poison_guard_scale("bogus", "bogus", "bogus")
    if not (full == 1.0 and 0 < low < full and 0 < override < full
            and 0 < harvested < full and unknown_key <= low):
        print(f"[insight_attribution] FAIL g1: poison guard — full={full} "
              f"low={low} override={override} harvested={harvested} "
              f"unknown={unknown_key}", file=sys.stderr)
        failures += 1
    else:
        print("[insight_attribution] g1 (poison guard scaling) OK",
              file=sys.stderr)

    # ---- Guarantee 2: criterion reducer — satisfied corroborates, scaled;
    #      unknown is inert; tier weights load-bearing more ----
    w = criterion_weighted([CriterionOutcome(
        attributed=(("a", "primary"), ("b", "incidental")),
        roll_up="satisfied", verdict_confidence="high",
        verdict_source="signal", provenance="user")])
    inert = criterion_weighted([CriterionOutcome(
        attributed=(("a", "primary"),), roll_up="unknown",
        verdict_confidence="high", verdict_source="signal", provenance="user")])
    if not (w["a"].corroboration > w["b"].corroboration > 0.0
            and w["a"].falsification == 0.0):
        print(f"[insight_attribution] FAIL g2: satisfied corroboration/tier — "
              f"a={w['a']} b={w['b']}", file=sys.stderr)
        failures += 1
    elif inert:
        print(f"[insight_attribution] FAIL g2: unknown roll-up produced "
              f"weights {inert}", file=sys.stderr)
        failures += 1
    else:
        print("[insight_attribution] g2 (criterion reducer: satisfied + tier "
              "+ unknown inert) OK", file=sys.stderr)

    # ---- Guarantee 3: not_satisfied falsifies, and a LOW-confidence
    #      falsification is strictly weaker than a HIGH one (poison guard —
    #      a weak verdict cannot retire a good claim) ----
    high_fail = criterion_weighted([CriterionOutcome(
        (("a", "primary"),), "not_satisfied", "high", "signal", "user")])
    low_fail = criterion_weighted([CriterionOutcome(
        (("a", "primary"),), "not_satisfied", "low", "signal", "user")])
    if not (high_fail["a"].falsification > 0.0
            and high_fail["a"].corroboration == 0.0
            and 0.0 < low_fail["a"].falsification < high_fail["a"].falsification):
        print(f"[insight_attribution] FAIL g3: falsification scaling — "
              f"high={high_fail['a'].falsification:.3f} "
              f"low={low_fail['a'].falsification:.3f}", file=sys.stderr)
        failures += 1
    else:
        print("[insight_attribution] g3 (not_satisfied falsifies, low-conf "
              "attenuated) OK", file=sys.stderr)

    # ---- Guarantee 4: decisive_verdict — satisfied → weakest confidence +
    #      override-if-any; not_satisfied → strongest failing check ----
    sat_conf, sat_src = decisive_verdict(
        [_cc("satisfied", "high", "signal"),
         _cc("satisfied", "low", "user_override")], "satisfied")
    fail_conf, fail_src = decisive_verdict(
        [_cc("satisfied", "low", "signal"),
         _cc("not_satisfied", "high", "signal")], "not_satisfied")
    # A failure is never a vouch: even a not_satisfied check tagged
    # user_override scales as "signal" (no 0.7 discount on a falsification).
    fail_override = decisive_verdict(
        [_cc("not_satisfied", "high", "user_override")], "not_satisfied")
    empty_conf, empty_src = decisive_verdict([], "satisfied")
    if not (sat_conf == "low" and sat_src == "user_override"
            and fail_conf == "high" and fail_src == "signal"
            and fail_override == ("high", "signal")
            and empty_conf == "low" and empty_src == "signal"):
        print(f"[insight_attribution] FAIL g4: decisive — sat=({sat_conf},"
              f"{sat_src}) fail=({fail_conf},{fail_src}) "
              f"fail_override={fail_override} "
              f"empty=({empty_conf},{empty_src})", file=sys.stderr)
        failures += 1
    else:
        print("[insight_attribution] g4 (decisive_verdict reduction) OK",
              file=sys.stderr)

    # ---- Guarantee 5: a user_override satisfied moves trust strictly less
    #      than a measured (signal) satisfied — the override discount ----
    measured = criterion_weighted([CriterionOutcome(
        (("a", "primary"),), "satisfied", "high", "signal", "user")])
    vouched = criterion_weighted([CriterionOutcome(
        (("a", "primary"),), "satisfied", "high", "user_override", "user")])
    if not (0.0 < vouched["a"].corroboration < measured["a"].corroboration):
        print(f"[insight_attribution] FAIL g5: override discount — "
              f"measured={measured['a'].corroboration:.3f} "
              f"vouched={vouched['a'].corroboration:.3f}", file=sys.stderr)
        failures += 1
    else:
        print("[insight_attribution] g5 (user_override discounted vs "
              "measured) OK", file=sys.stderr)

    # ---- Guarantee 6 (S3-close §B finding 5): the poison guard on a REAL
    #      low-confidence not_satisfied through the PRODUCTION seam. g4
    #      hand-builds the CriterionCheck; here it is produced by the real
    #      `evaluate_criterion` (an llm_judges criterion with a judge returning a
    #      low-confidence not_satisfied), then run through decisive_verdict →
    #      criterion_weighted. The guard must ATTENUATE — a low-confidence
    #      failure falsifies strictly less than a high-confidence one — so a
    #      single weak (e.g. S5 auto-harvested / llm-only) reject cannot retire a
    #      healthy corpus claim. This is the seam g4 left unasserted for the low
    #      case (it only tested a HIGH failing check). ----
    from resonance_lattice.state.measure import evaluate_criterion

    real_check = evaluate_criterion(
        {"text": "answer is correct", "measure": "llm_judges:answer is correct"},
        [],  # no user signal → the judge runs and supplies the native verdict
        judge=lambda spec, evidence: ("not_satisfied", "low"),
        evidence="the agent's completed work",
    )
    seam_conf, seam_src = decisive_verdict([real_check], "not_satisfied")
    low_w = criterion_weighted([CriterionOutcome(
        (("c", "primary"),), "not_satisfied", seam_conf, seam_src, "user")])
    high_w = criterion_weighted([CriterionOutcome(
        (("c", "primary"),), "not_satisfied", "high", "signal", "user")])
    if not (real_check.verdict == "not_satisfied"
            and real_check.verdict_confidence == "low"
            and (seam_conf, seam_src) == ("low", "signal")
            and 0.0 < low_w["c"].falsification < high_w["c"].falsification
            and low_w["c"].corroboration == 0.0):
        print(f"[insight_attribution] FAIL g6: real-seam poison guard — "
              f"check=({real_check.verdict},{real_check.verdict_confidence}) "
              f"decisive=({seam_conf},{seam_src}) "
              f"low_fals={low_w['c'].falsification:.3f} "
              f"high_fals={high_w['c'].falsification:.3f}", file=sys.stderr)
        failures += 1
    else:
        print("[insight_attribution] g6 (real low-conf not_satisfied attenuates "
              "via decisive_verdict → criterion_weighted) OK", file=sys.stderr)

    if failures:
        print(f"[insight_attribution] {failures} guarantee(s) failed",
              file=sys.stderr)
        return 1
    print("[insight_attribution] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
