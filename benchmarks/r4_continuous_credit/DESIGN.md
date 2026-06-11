# R4 — Continuous-Credit Suppression (pre-registered design)

**Status**: design locked before the instrumented run and before any
stream is observed. **Date**: 2026-06-11. Funded by Kane (R4 budget
added 2026-06-11). Roadmap Stage 6, bet R4.

## The bet, in plain language

The self-cleaning rule the closed-loop program needs must be SAFE (never
retires a true fact) and EFFECTIVE (retires wrong facts fast enough to
matter). Every rule tried so far fails one side, and the diagnosis is
information loss: each serve's measured effect (a continuous number) is
crushed to a ±0.05 yes/no before any rule sees it. R4 keeps the
continuous per-serve credit stream and runs an anytime-valid sequential
test over it — the first approach that adds evidence rather than tuning
thresholds on the crushed signal.

## Phase 1 — the instrumented baseline run (one-time LLM spend)

`bench_closed_loop_v2.py` (now persisting `serve_log`) on the locked
n=31 PowerShell fixture (`powershell_locked_v1_ctrl2.json`), Sonnet
answerer + judges (same model family as every locked prior run):

    --compare-policies --seeds 3 --rounds 8
    --suppress-mode helpfulness --suppress-stat wilson2 --extra-wrongs

- `learning` arm = wilson2 (the best prior rule) — the live baseline.
- `nolearn` arm = suppression off — every fact keeps being observed, so
  its serve log is the unbiased stream for off-policy rule replay.
- Output: `benchmarks/results/outcome/closed_loop_v2_r4_instrumented.json`
  including the full per-serve stream (round, item, fact, gold/wrong,
  cov, floor, delta) per seed per mode.

## Phase 2 — the pre-registered R4 rule (offline replay, $0)

**Rule (locked now, before any stream exists):** per fact, maintain an
anytime-valid confidence sequence on the mean per-serve delta — Hoeffding
boundary, deltas clipped to [-1, 1], alpha = 0.05:

    radius(n) = sqrt( ln(2/alpha) * 2 / n ) * 1.7 / 2     # anytime-valid via
    # the stitched bound approximation; conservative constant pre-chosen

    mean_n ± radius(n)  after every serve of that fact

- **PROTECT** a fact whose lower bound > 0 (confidently helpful).
- **CUT** a fact whose upper bound < 0.05 (confidently providing less
  than the corroboration deadband) AND which has ≥ 3 serves.
- Otherwise: keep observing.

No parameter may be tuned after seeing the stream. If the constant or
alpha is changed post-hoc, the run is exploratory and says so.

## Pre-registered bars (the prior program's LOCK bar: safe + sig + multi-seed)

Replayed on the `nolearn` streams (unbiased observation), per seed,
judged against wilson2 replayed on the identical streams:

1. **SAFE**: R4 cuts **zero golds** in every seed (wilson2's recorded
   failure mode was 2/7 golds under its effective sibling).
2. **EFFECTIVE**: by the final round, R4 has cut **at least as many
   wrongs as wilson2** in every seed, and strictly more in at least 2/3
   seeds (else it's a tie, stated as such).
3. **MULTI-SEED**: bars 1–2 hold across all 3 seeds individually — no
   averaging away a bad seed.
4. Secondary (reported, not gated): rounds-to-first-correct-cut;
   protect-coverage of golds; the same comparison vs `point` and
   `neverhelped`.

PASS = all of 1–3. A PASS earns the live confirmation run (the winning
rule as `--suppress-stat r4` in a fresh A/B) before any product claim —
replay is on-policy-valid only.

## What a PASS buys / what it doesn't

Buys: the missing safe+effective suppression rule — the last unsolved
segment of the "gets better with use" loop — plus a permanent free test
bench (the stream) for all future rule research. Doesn't buy: the live
end-to-end closed-loop claim (that's the confirmation run), or anything
about capture.

## Artifacts

- This file (committed before the run).
- `replay_rules.py` — the offline analyzer: replays R4 + the three prior
  rules over the recorded streams, applies the bars.
- `run_results/` — instrumented-run pointer + replay outputs + VERDICT.
