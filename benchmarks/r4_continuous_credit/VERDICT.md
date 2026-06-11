# R4 — run 1 verdict (2026-06-11)

**Pre-registered rule: FAIL.** The confidence-sequence rule is the only
rule that never cuts a gold (0/3 seeds — every prior rule kills one) but
it cuts zero wrongs: with the pre-chosen conservative constant the
interval is still ±~0.5 at ~20 serves/fact — powerless against wrongs
whose true effect is ≈0. Safe-but-useless, now from the continuous side.

## Replay numbers (corrected analyzer*; nolearn streams, 3 seeds)

| Rule | golds cut (per seed) | wrongs cut (per seed) |
|---|---|---|
| **R4 (pre-registered)** | **0, 0, 0** | 0, 0, 0 |
| point | 1, 1, 1 | 4, 4, 4 |
| wilson2 | 1, 1, 1 | 3, 3, 3 |
| neverhelped | 1, 0, 0 | 2, 3, 1 |

\*Analyzer correction made before the verdict: the stream's `is_gold` is
per-serve (right answer for the served item); fact identity uses the
band-level `is_wrong`. Verdict shape unchanged by the fix.

## The diagnostic finding (what the $-funded stream actually bought)

Per-fact pooled credit on the unbiased streams:

| fact (abridged) | serves | mean Δ | gold? |
|---|---|---|---|
| ConstrainedLanguage mode | 72 | **+0.810** | yes |
| auth-proxy internet | 24 | **+0.639** | yes |
| FullLanguage mode | 71 | +0.201 | no |
| PowerShell 7.4 | 143 | −0.036 | no |
| **Windows PowerShell 5.1** | 122 | **−0.140** | **yes** |
| RemoteSigned policy | 48 | −0.167 | no |
| NoLanguage mode | 46 | −0.184 | no |

The third gold's pooled average is WORSE than two wrong facts', because
the selector serves it heavily off-item, where even a true fact doesn't
help. **Pooled per-fact helpfulness is the wrong sufficient statistic:**
on it, this gold and the neutral wrongs are not just hard to separate —
they are ordered the wrong way. No threshold, binarised or continuous,
Wilson or confidence-sequence, can be simultaneously safe and effective
on this statistic. Four rule families have now failed here for the same
structural reason, not four tuning accidents.

**v2 direction (pre-register before any further run): context-conditioned
credit** — score each fact's effect per serve-context (per (fact, item)
or relevance-weighted), so a gold's on-item lift isn't drowned by
off-item dilution. On the recorded stream, PowerShell 5.1's on-item
serves are exactly where its value lives; pooled, it looks like noise.

## Also recorded

- The instrumented A/B itself replicates the loop's value-add: learning
  (wilson2) − nolearn = **+0.074 mean, p_wilcoxon 0.031** (n=31, 3
  seeds, paired) — consistent with the prior point/8r result, now with
  the full per-serve stream persisted
  (`benchmarks/results/outcome/closed_loop_v2_r4_instrumented.json`).
- All future rule iteration on this fixture is now offline and free;
  this verdict's exploratory analysis cost $0 beyond the one funded run.
