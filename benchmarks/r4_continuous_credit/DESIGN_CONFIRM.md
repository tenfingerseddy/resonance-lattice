# R4 — r4c live confirmation (pre-registered before the run)

**Date**: 2026-06-11. The replay candidate r4c (DESIGN_V2.md: judge a
fact by its best per-item record) now drives suppression LIVE in a fresh
instrumented run — the off-policy replay's prediction meets reality.
Funded from the remaining R4 budget (~$8/run measured, ~$38 available).

## Run

`bench_closed_loop_v2.py --compare-policies --seeds 3 --rounds 8
--suppress-mode helpfulness --suppress-stat r4c --extra-wrongs`
on the locked n=31 PowerShell fixture, Sonnet (same as the baseline).
Constants identical to DESIGN_V2 (locked): protect cell-mean ≥ +0.30 at
n≥2, sticky; cut at ≥8 total serves with no cell ≥ +0.15 at n≥2.

## Pre-registered bars (the honest "gets better with use" claim)

1. **SAFE**: zero golds suppressed at the final round in EVERY seed.
2. **EFFECTIVE (transfer)**: ≥2 wrongs suppressed at the final round in
   every seed — the replay's prediction carried live.
3. **VALUE**: learning(r4c) − nolearn paired over the 31 items,
   seed-averaged: mean > 0 with p_wilcoxon < 0.05 (the wilson2 baseline
   measured +0.074, p=0.031, while killing a gold every seed; r4c must
   deliver value without the casualty).
4. All seeds individually for bars 1–2.

PASS all four = the loop's safe self-cleaning claim is live-confirmed —
LOCK per the program's bar (safe + sig + multi-seed). Any FAIL is
recorded as-is; no post-hoc re-scoring.

## Live verdict (2026-06-11, run_results/live_confirm_stream.json) — FAIL

| Bar | Result |
|---|---|
| SAFE (0 golds every seed) | **FAIL** — one seed suppressed a gold (seed-avg 0.33) |
| TRANSFER (>=2 wrongs every seed) | final seed-avg 3.33 (moot given SAFE fail) |
| VALUE (learning - nolearn > 0, p<0.05) | **FAIL** — mean -0.028, p_wilcoxon 0.76 |

The off-policy replay was optimistic on BOTH axes: live suppression
changes the serving distribution (cut facts free serve slots; per-cell
evidence accrues differently), and a gold was cut before accumulating
its protecting on-item record. This is precisely why the live
confirmation was pre-registered as mandatory - replay-only evidence
would have shipped a false safety claim.

R4 cycle disposition: the "gets better with use" suppression claim
remains NOT earned. Net assets: the committed streams (now two: nolearn
baseline + r4c live), the structural diagnosis (pooled credit), and a
sharpened constraint - off-policy replay on this fixture overestimates
safety because selection feedback matters. Any v3 must model the
feedback (e.g. protect-before-cut ordering, minimum on-item exposure
before any cut is allowed) and pre-register cost-weighted bars. ~$8 =
this live run; ~$10.50 = the full R4 cycle incl. the instrumented
baseline and Haiku confirms. Cycle closed per fail-fast.
