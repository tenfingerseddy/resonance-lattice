"""measure — the three success-criterion measure evaluators (H1/S4 d1).

Pins `state/measure.py` (architecture §"Success criteria"). Contracts:

  (a) parse_measure splits `<kind>` / `<kind>:<spec>`; an unknown kind →
      `("", measure)`.

  (b) user_confirms: a `user` satisfied/not_satisfied signal decides the
      verdict at high confidence; absent any user signal → `unknown`.

  (c) mechanical:<spec>: a `mechanical` signal decides; absent one (and no
      user signal) → `unknown` — a stray off-source signal never decides it.

  (d) llm_judges:<rubric>: an injected judge decides at *low* confidence
      (llm is low authority by construction); no judge → `unknown`.

  (e) user override: a `user` signal outranks a mechanical or llm verdict,
      with the conflict flag set on a loud disagreement.

  (f) uninterpretable measure → `unknown`, no signals seen.

Hermetic — pure functions, no temp dir, no LLM, no network.
"""

from __future__ import annotations

import sys

_P = "measure"


def _crit(text: str, measure: str) -> dict:
    return {"text": text, "measure": measure}


def _sig(source: str, verdict: str):
    from resonance_lattice.state import Signal
    return Signal(source=source, value={"verdict": verdict},
                  timestamp="2026-06-01T00:00:00Z")


def _check_parse() -> int:
    from resonance_lattice.state import parse_measure

    cases = {
        "user_confirms": ("user_confirms", ""),
        "mechanical:exit_code==0": ("mechanical", "exit_code==0"),
        "llm_judges:answer is correct": ("llm_judges", "answer is correct"),
        "bogus:whatever": ("", "bogus:whatever"),
        "": ("", ""),
    }
    for measure, expected in cases.items():
        got = parse_measure(measure)
        if got != expected:
            print(f"[{_P}] FAIL (a): parse {measure!r} → {got!r} != "
                  f"{expected!r}", file=sys.stderr)
            return 1
    print(f"[{_P}] (a) parse_measure OK", file=sys.stderr)
    return 0


def _check_user_confirms() -> int:
    from resonance_lattice.state import evaluate_criterion

    crit = _crit("the user is happy", "user_confirms")
    sat = evaluate_criterion(crit, [_sig("user", "satisfied")])
    nsat = evaluate_criterion(crit, [_sig("user", "not_satisfied")])
    none = evaluate_criterion(crit, [])
    # An off-source (mechanical) signal must not decide a user_confirms crit.
    off = evaluate_criterion(crit, [_sig("mechanical", "satisfied")])
    ok = (
        sat.verdict == "satisfied" and sat.verdict_confidence == "high"
        and nsat.verdict == "not_satisfied"
        and none.verdict == "unknown"
        and off.verdict == "unknown"
        and sat.measure == "user_confirms"
        and len(sat.signals_seen) == 1 and sat.signals_seen[0].source == "user"
    )
    if not ok:
        print(f"[{_P}] FAIL (b): user_confirms: sat={sat.verdict} "
              f"nsat={nsat.verdict} none={none.verdict} off={off.verdict}",
              file=sys.stderr)
        return 1
    print(f"[{_P}] (b) user_confirms OK", file=sys.stderr)
    return 0


def _check_mechanical() -> int:
    from resonance_lattice.state import evaluate_criterion

    crit = _crit("tests pass", "mechanical:exit_code==0")
    sat = evaluate_criterion(crit, [_sig("mechanical", "satisfied")])
    nsat = evaluate_criterion(crit, [_sig("mechanical", "not_satisfied")])
    none = evaluate_criterion(crit, [])
    # A stray llm signal is NOT authoritative for a mechanical criterion.
    stray = evaluate_criterion(crit, [_sig("llm", "satisfied")])
    # A user accept with NO mechanical evidence is honoured — but marked
    # user_override, not a measured pass (it must not masquerade as a run test).
    user_only = evaluate_criterion(crit, [_sig("user", "satisfied")])
    # A user accept does NOT whitewash a genuinely failed test — conflict.
    whitewash = evaluate_criterion(
        crit, [_sig("mechanical", "not_satisfied"), _sig("user", "satisfied")])
    ok = (
        sat.verdict == "satisfied" and sat.verdict_confidence == "high"
        and sat.verdict_source == "signal"
        and nsat.verdict == "not_satisfied"
        and none.verdict == "unknown"
        and stray.verdict == "unknown"
        and user_only.verdict == "satisfied"
        and user_only.verdict_source == "user_override"
        and whitewash.verdict == "not_satisfied"
        and whitewash.conflict_flag is True
    )
    if not ok:
        print(f"[{_P}] FAIL (c): mechanical: sat={sat.verdict}/{sat.verdict_source} "
              f"nsat={nsat.verdict} none={none.verdict} stray={stray.verdict} "
              f"user_only={user_only.verdict}/{user_only.verdict_source} "
              f"whitewash={whitewash.verdict}/{whitewash.conflict_flag}",
              file=sys.stderr)
        return 1
    print(f"[{_P}] (c) mechanical OK", file=sys.stderr)
    return 0


def _check_llm_judges() -> int:
    from resonance_lattice.state import evaluate_criterion

    crit = _crit("the answer addresses the question", "llm_judges:is it right")

    def judge_sat(rubric, evidence):
        return "satisfied", "high"

    def judge_nsat(rubric, evidence):
        return "not_satisfied", "medium"

    sat = evaluate_criterion(crit, [], judge=judge_sat, evidence="some work")
    nsat = evaluate_criterion(crit, [], judge=judge_nsat, evidence="some work")
    no_judge = evaluate_criterion(crit, [])
    # A user signal short-circuits the judge (it would override it anyway) and
    # resolves the criterion as a user_override — the user's verdict is kept.
    user_no_judge = evaluate_criterion(crit, [_sig("user", "satisfied")])
    user_over_judge = evaluate_criterion(
        crit, [_sig("user", "not_satisfied")], judge=judge_sat, evidence="x")
    # llm verdict is low authority by construction → low confidence, so the
    # poison guard attenuates it.
    ok = (
        sat.verdict == "satisfied" and sat.verdict_confidence == "low"
        and sat.verdict_source == "signal"
        and nsat.verdict == "not_satisfied"
        and no_judge.verdict == "unknown"
        and user_no_judge.verdict == "satisfied"
        and user_no_judge.verdict_source == "user_override"
        # the judge is short-circuited, so no llm signal was synthesised
        and not any(s.source == "llm" for s in user_no_judge.signals_seen)
        and user_over_judge.verdict == "not_satisfied"
        and any(s.source == "llm" for s in sat.signals_seen)
    )
    if not ok:
        print(f"[{_P}] FAIL (d): llm_judges: sat={sat.verdict}/"
              f"{sat.verdict_confidence} nsat={nsat.verdict} "
              f"no_judge={no_judge.verdict} "
              f"user_no_judge={user_no_judge.verdict}/{user_no_judge.verdict_source} "
              f"user_over_judge={user_over_judge.verdict}", file=sys.stderr)
        return 1
    print(f"[{_P}] (d) llm_judges OK", file=sys.stderr)
    return 0


def _check_user_override() -> int:
    from resonance_lattice.state import evaluate_criterion

    # mechanical says satisfied, user says not — user wins, conflict flagged.
    mech_crit = _crit("tests pass", "mechanical:exit_code==0")
    override = evaluate_criterion(
        mech_crit,
        [_sig("mechanical", "satisfied"), _sig("user", "not_satisfied")],
    )
    # llm says satisfied, user says not — user wins.
    llm_crit = _crit("good answer", "llm_judges:is it right")

    def judge_sat(rubric, evidence):
        return "satisfied", "high"

    llm_override = evaluate_criterion(
        llm_crit, [_sig("user", "not_satisfied")],
        judge=judge_sat, evidence="x",
    )
    ok = (
        override.verdict == "not_satisfied" and override.conflict_flag is True
        and llm_override.verdict == "not_satisfied"
    )
    if not ok:
        print(f"[{_P}] FAIL (e): override: mech={override.verdict}/"
              f"conflict={override.conflict_flag} llm={llm_override.verdict}",
              file=sys.stderr)
        return 1
    print(f"[{_P}] (e) user override OK", file=sys.stderr)
    return 0


def _check_uninterpretable() -> int:
    from resonance_lattice.state import evaluate_criterion

    crit = _crit("does a thing", "made_up_measure")
    res = evaluate_criterion(crit, [_sig("user", "satisfied")])
    ok = res.verdict == "unknown" and res.signals_seen == []
    if not ok:
        print(f"[{_P}] FAIL (f): uninterpretable: {res.verdict} "
              f"signals={res.signals_seen!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (f) uninterpretable measure OK", file=sys.stderr)
    return 0


def _pending(source: str, verdict: str, intent_id: str | None = "i1"):
    from resonance_lattice.state import PendingSignal
    return PendingSignal(
        source=source, tool_name="t", tool_payload={},
        value={"verdict": verdict}, intent_id=intent_id,
        captured_at="2026-06-01T00:00:00Z",
    )


def _check_pending_to_signal() -> int:
    from resonance_lattice.state import pending_to_signal

    sig = pending_to_signal(_pending("user", "satisfied"))
    ok = (
        sig.source == "user"
        and sig.value == {"verdict": "satisfied"}
        and sig.timestamp == "2026-06-01T00:00:00Z"  # captured_at → timestamp
    )
    if not ok:
        print(f"[{_P}] FAIL (g): pending_to_signal: {sig!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (g) pending_to_signal OK", file=sys.stderr)
    return 0


def _check_synthesize_declared() -> int:
    """An intent with two declared criteria → two per-criterion checks
    evaluated against the pending signals; roll-up is the AND across them."""
    from resonance_lattice.state import roll_up, synthesize_criterion_checks

    criteria = [
        _crit("the user is happy", "user_confirms"),
        _crit("tests pass", "mechanical:exit_code==0"),
    ]
    pending = [_pending("user", "satisfied"), _pending("mechanical", "satisfied")]
    checks = synthesize_criterion_checks(criteria, pending)
    ok = (
        len(checks) == 2
        and checks[0].criterion_text == "the user is happy"
        and checks[0].verdict == "satisfied"
        and checks[1].measure == "mechanical:exit_code==0"
        and checks[1].verdict == "satisfied"
        and roll_up(checks) == "satisfied"
    )
    # One criterion unmet → roll-up not_satisfied.
    pending_fail = [_pending("user", "satisfied"),
                    _pending("mechanical", "not_satisfied")]
    checks_fail = synthesize_criterion_checks(criteria, pending_fail)
    ok = ok and roll_up(checks_fail) == "not_satisfied"
    if not ok:
        print(f"[{_P}] FAIL (h): synthesize declared: "
              f"{[(c.verdict) for c in checks]} roll={roll_up(checks)} "
              f"fail_roll={roll_up(checks_fail)}", file=sys.stderr)
        return 1
    print(f"[{_P}] (h) synthesize declared criteria OK", file=sys.stderr)
    return 0


def _check_synthesize_fallback() -> int:
    """No declared criteria → the single user_confirms fallback is evaluated
    from the user signal; neither criteria nor fallback → empty (unknown)."""
    from resonance_lattice.state import roll_up, synthesize_criterion_checks

    pending = [_pending("user", "not_satisfied")]
    fallback = {"text": "user not_satisfied: x", "measure": "user_confirms"}
    checks = synthesize_criterion_checks([], pending, fallback=fallback)
    empty = synthesize_criterion_checks([], pending)
    ok = (
        len(checks) == 1
        and checks[0].measure == "user_confirms"
        and checks[0].verdict == "not_satisfied"
        and roll_up(checks) == "not_satisfied"
        and empty == []
        and roll_up(empty) == "unknown"
    )
    if not ok:
        print(f"[{_P}] FAIL (i): synthesize fallback: checks={checks!r} "
              f"empty={empty!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (i) synthesize fallback + empty OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_parse,
        _check_user_confirms,
        _check_mechanical,
        _check_llm_judges,
        _check_user_override,
        _check_uninterpretable,
        _check_pending_to_signal,
        _check_synthesize_declared,
        _check_synthesize_fallback,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print(f"[{_P}] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
