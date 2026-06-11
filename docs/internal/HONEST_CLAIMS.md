# Honest Claims

What `rlat` claims, what it doesn't, and what's measured vs. projected. Calibration discipline — every public claim either has evidence behind it or is explicitly flagged as forward-looking. The retrieval-substrate entries date from the v2.0 cycle; the user-world entries from the v3 cycle.

## What we claim, with evidence

### Retrieval quality (base band, no optimised, no rerank)

> **gte-modernbert-base 768d CLS+L2 + FAISS HNSW** scores BEIR-5 mean **0.5144 nDCG@10** / **0.5666 R@10** at zero training cost.

Evidence: [`benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json`](../../benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json).

Per-corpus locked floor (matches [`BENCHMARK_GATE.md`](BENCHMARK_GATE.md) — single source of truth):

| Corpus | passages | nDCG@10 | R@10 |
|---|---:|---:|---:|
| nfcorpus | 3,633 | 0.3431 | 0.1640 |
| scifact | 5,183 | 0.7672 | 0.8926 |
| arguana | 8,674 | 0.7430 | 0.9637 |
| scidocs | 25,657 | 0.1946 | 0.2014 |
| fiqa | 57,638 | 0.5239 | 0.6114 |

Comparators on the v2.0 stack (same chunker / ANN / scoring — apples-to-apples):
- BGE-large-v1.5 (1024d): 0.4888 — `rlat` base wins by **+0.026 nDCG@10**.
- E5-large-v2 (1024d): 0.4331 — `rlat` base wins by **+0.081 nDCG@10**.
- Qwen3-Embedding-4B: loader-incompat in this run; retry pending v2.0.1.

Evidence: [`benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json`](../../benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json) (encoder comparison, run on Kaggle T4). Plus an off-stack reference: Qwen3-Embedding-8B (last-token pooling) measured 0.500 in earlier work — rlat base wins by **+0.014 nDCG@10** mean, but that's not apples-to-apples against rlat's chunker/ANN, so the v2.0-stack comparators above are the load-bearing numbers.

Honest framing: scidocs scores are intentionally low across the field because the corpus is short-passage retrieval where dense bi-encoders without domain adaptation generally floor at the 0.18–0.22 nDCG@10 band; nfcorpus is ~3K medical passages with long synonym chains. These are corpus characteristics, not method failings.

### Encoder reproducibility

> Determinism: bit-exact across CPU runs of the same revision; L2-norm error < 1.2e-7.

Evidence: Phase 1 close-out empirical runs against `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. Verified on Intel CPU + OpenVINO runtime.

### Latency

> Warm encode 12.1ms / cold 942ms (single query, T4 not warmed).

Evidence: Phase 1 close-out measurement on the standard T4 instance.

ANN search adds ~sub-millisecond. Cold start dominated by encoder load + tokenizer init.

### Storage modes

> All three storage modes — bundled, local, remote — ship in v2.0 and round-trip through the v4 `.rlat` ZIP format.

Evidence: round-trip + parity tests in `tests/harness/`; remote-mode end-to-end confirmed against a fixture HTTP server (the `incremental_sync` harness exercises build → sync-no-op → modify-upstream → sync-delta → catalog-mode-add → catalog-mode-remove → manifest-pin-advance, 5 guarantees, all hermetic). Remote manifest persists as `manifest.json` at the top of the ZIP. v2.0 reconciliation is read-only `rlat freshness` (CI gate) + `rlat sync` (incremental delta-apply on the same `store/incremental.py` pipeline as `rlat refresh`). The codex P0 manifest-only-sync mode is statically impossible — `apply_delta` requires the encoder, the only manifest-write path is `apply_delta`. Audit 07 is the design source of truth.

### Live freshness (`rlat watch`)

> **`rlat watch`** keeps a local-mode archive current as you edit, on the same `store/incremental.py` pipeline as `rlat refresh`. Default UX is silent. `--once` is a synchronous one-shot for CI / pre-commit. Concurrent FS events can't race the atomic write path; transient read failures don't become silent deletes.

Evidence: `tests/harness/watch_loop.py` — 8 hermetic guarantees, all green (the optimised-band re-projection guarantee left with the retired optimise feature). Specifically: (1) zero-arg auto-discovery of `*.rlat` in cwd, (2) single-event refresh updates the archive, (3) add+remove cycle reflects both, (4) per-archive `threading.Lock` serialises concurrent refreshes (closes the `<archive>.tmp` race), (5) bundled-mode preflight rejection with `rlat convert` hint, (6) `--once` reconciles synchronously without waiting for events (the CI/pre-commit shape that the original event-waiting `--once` got wrong), (7) `force=True` dispatch on rename/delete/dir events bypasses the suffix pre-filter so `foo.md → foo.bak` doesn't leave stale passages indexed, (8) `_filter_skipped_removals` defends against the silent-delete hazard where Windows file locks during atomic save would otherwise make `bucketise` emit destructive removals for transiently unreadable files. Mental model in the implementation: events are hints to reconcile, not the unit of correctness — `bucketise` against the live source tree is the source of truth.

### Deep-search loop accuracy

> **`rlat deep-search`** scores **92.2% answerable accuracy at 0% hallucination, $0.009/q** on the Microsoft Fabric corpus 11-lane v4 bench (63 questions, Sonnet 4.6, relaxed rubric).

Evidence: [`benchmarks/results/user_bench/hallucination_v4.json`](../../benchmarks/results/user_bench/hallucination_v4.json). Same 11-lane matrix produced these comparators on the same test set:

- LLM+grep/glob/read_file (8 tool calls): 94.1% accuracy at $0.060/q — `rlat deep-search` within 2 pp at **6.5× lower spend** and faster wall-time.
- Single-shot `rlat search --mode augment`: 76.5% accuracy.
- LLM-only (no retrieval): 56.9% accuracy at 19.6% hallucination.
- `rlat search --mode constrain`: 91.7% distractor refusal — the compliance floor for wrong-but-confident-is-worse-than-no-answer workloads.

The bench is built around a real-world corpus the LLM partially knows: Microsoft Fabric documentation has been in market since 2023 (Sonnet has substantial training data) but contains 559 files dated post Sonnet's January 2026 cutoff. This is the right shape to measure whether grounding actually moves the needle vs the LLM's training prior.

### Namecheck distinctive-token verification

> **`--strict-names`** catches name-aliasing distractor failures where the encoder surfaces a similarly-named real entity for a fake-product-name question.

Evidence: 12 of the 63 hallucination-bench questions are distractors — fake F SKU codes, made-up product names, Power-BI-only features asked as if they were Fabric features. Without namecheck, a single-shot retrieval surfaces a similarly-named real entity and the LLM answers about that adjacent entity (a hallucination by user intent, even when the fact is correct about the wrong product). Wired into `rlat skill-context`, `rlat search --format context`, and `_grounding.py` distinctive-token verification on the grounding-emit boundary. Harness suite at [`tests/harness/name_check.py`](../../tests/harness/name_check.py) (16 guarantees).

### Session-start primer effectiveness

> **Code primer** (`rlat summary` → `.claude/resonance-context.md`) **and memory primer** (`rlat memory primer` → `.claude/memory-primer.md`) **have measurable but tier-specific value** — they shine on the tier their content was designed for and degrade to roughly cold elsewhere.

Evidence: [`benchmarks/results/user_bench/primer_effectiveness.json`](../../benchmarks/results/user_bench/primer_effectiveness.json). 25-scenario × 5-lane bench on `resonance-lattice.rlat` (3,506 passages / 126 files), Sonnet 4.6 + 4-state relaxed rubric. Per-tier turn-1 accuracy:

| Tier | cold | code primer | memory primer | both primers | rlat search |
|---|---:|---:|---:|---:|---:|
| 1 — orientation | 0/5 | **3/5** | 0/5 | **3/5** | 0/5 |
| 2 — specific factual | 0/10 | 2/10 | 2/10 | 3/10 | **8/10** |
| 3 — cross-reference | 0/5 | 0/5 | 1/5 | 1/5 | **2/5** |
| 4 — memory recall | 0/5 | 0/5 | **5/5** | **5/5** | 4/5 |

Aggregate turn-1 correct: cold 0% / code primer 20% / memory primer 32% / both primers 48% / rlat search 56%.

Token cost of each surface (Sonnet 4.6 input pricing):
- Code primer: ~1,708 tokens/call (~3 KB markdown).
- Memory primer: ~746 tokens/call (~2 KB markdown).
- Both concatenated: ~2,454 tokens/call.
- `rlat search` top-5: ~704 tokens of passages/call (dynamic per turn).

Combined-stack reading: load both primers at session start (free, ~5 KB) **and** keep `rlat search` available. Primers carry orientation + memory recall; per-turn search picks up specific facts. For session-start questions that turn out to need synthesis, escalate to `rlat deep-search`. Honest framing on the 25-scenario MVP sample: tier-level n=5 is small; per-tier numbers are directional, not precise.

### Source provenance

> Every passage carries `(source_file, char_offset, char_length, content_hash, drift_status)`. Cite-back is free; drift detection is free.

Evidence: ported from v0.11 WS3 #292; `Store.verify` walks the registry against live source bytes. Tested in `tests/harness/property.py:_check_rql_invariants` (drift-aware ops surface drift fraction in evidence reports).

### Environment / world-premise serving (v3 user-world band)

> Serving the user's true environment premise lifts answers on premise-decisive items, replicated across two corpus types under paired oracle−placebo contrasts.

Scope — and the scope of every user-world claim below (standing constraints, falsification ledger): **gate-conditional** — measured on decisive-item subsets, items where the blind answer actually fails, not unconditional across all questions.

Evidence: the locked env-premise proof (paired within-item oracle−placebo, build-gate-selected decisive items, two corpus types). R1's design isolates serve value "oracle-style, exactly like the locked env-premise proof did" ([`benchmarks/constraint_band/DESIGN.md`](../../benchmarks/constraint_band/DESIGN.md)); it is the first of the three serve-proven content classes in the v3 earned-claim list (`.claude/plans/v3-ship-plan.md`).

### Standing constraints (R1 + R1-X)

> Serving a handful of standing hard rules ("never preview features", "organic only", "NSW matters only") collapses rule-violating answers at zero collateral cost — in a Fabric tenant, a home garden, and a NSW law practice.

Design (pre-registered before any arm ran, both benches): paired within-item arms — blind / served / placebo (R1-X adds a blind-2 resample as the noise-aware placebo reference) — judged by a binary violation verdict with quoted-span evidence; a violation-decisive item gate (STEP 0); two guards: the placebo arm (an irrelevant served rule must not change answers) and a collateral set (questions untouched by every constraint must stay substantively answered). Serve-all, no retrieval, no selection — ~3–10 rules fit in context every time.

R1 numbers (Fabric; [`benchmarks/constraint_band/VERDICT.md`](../../benchmarks/constraint_band/VERDICT.md)):

| Measure | Subscription primary | API (Haiku) confirm |
|---|---|---|
| Blind gate yield | 15/24 (62%) | 10/15 on the decisive subset |
| Served violation rate | **1/15 (7%)** | **0/15 (0%)** |
| Placebo violation rate | 14/15 (93%) | 9/15 (60%) |
| Collateral substantive | 10/10 → 10/10 | 10/10 → 10/10 |

R1-X numbers (cross-domain; [`benchmarks/constraint_band_xdomain/VERDICT.md`](../../benchmarks/constraint_band_xdomain/VERDICT.md)):

| Measure | Garden | Practice |
|---|---|---|
| Blind gate (violation-decisive) | 12/12 | 11/12 |
| Served violation rate | **2/12 (17%)** | **1/11 (9%)** |
| Placebo flips (bar: ≤ blind-2 flips) | 0 (≤1) | 0 (≤0) |
| Collateral substantive | 6/6 → 6/6 | 6/6 → 6/6 |

API-judge confirm on R1-X: served 0/8 (garden) and 0/4 (practice) within the API judge's own decisive subsets; collateral 12/12 → 12/12.

Scope limits:

- Conditional on **constraint-decisive items** (the STEP 0 gate), stated as such. Gate yield is itself a finding: 62% of natural Fabric answers and 23/24 garden/legal answers violate a standing rule blind.
- **Serve-all design** — proven for small rule sets served whole; says nothing about retrieval-selected constraints.
- Cross-domain generality is earned on **garden + NSW practice** (plus Fabric); other domains are extrapolation.
- Serve value only, oracle-style — nothing about *capturing* constraints (that's E2c's track).

DISCLAIMER — **the placebo guard is judge-sensitive**, stated as the R1-X verdict states it: "under Haiku the guard FAILS in both domains (garden 2 placebo flips vs 0 blind-2; practice 2 vs 1), while under the subscription judge it passes with zero placebo flips. Haiku's known leniency on hedged answers compresses its gate (8/12, 4/12) and shifts borderline calls; the divergence concentrates exactly on the items the two judges already disagree about at the blind gate. Honest statement: the *constraint-specific* effect (the claim) is decisive under both judges; whether an *irrelevant* served rule also nudges answers a little is judge-dependent and unresolved at this n." Same picture on R1 (post-merge correction): computed within the API judge's own decisive subset, placebo reads 7/10 vs a 100% blind reference — a 30pp breach of the ±10pp form. The served-collapse and zero-over-blocking bars are judge-robust everywhere; only the placebo guard moves with the judge. Claims state the constraint-specific effect only.

### Falsification ledger (R2)

> Serving tried-and-falsified claims as first-class atoms ("Tried and falsified in this project (evidence: …): <finding>") stops the assistant re-recommending dead ends — and the falsification **verdict** is the active ingredient, not topical mention of the approach.

Design (pre-registered): paired within-item arms — blind / ledger / topical-mention control (a verdict-free description of the same approach in identical framing) / irrelevant-ledger placebo — binary `recommends` judge with quoted-span evidence, a recommendation-decisive gate, and a collateral set answered with the full 10-atom ledger served.

Run-2 numbers (fictional "Lumera" project; [`benchmarks/falsification_ledger/VERDICT.md`](../../benchmarks/falsification_ledger/VERDICT.md)):

| Measure | Result |
|---|---|
| Gate yield (blind recommends) | 7/20 (35%) |
| Ledger arm recommend rate | **0/7 (0%)** |
| Topical-mention control | 6/7 (86%) — active-ingredient gap **86pp** |
| Placebo, raw | 5/7 (71%) — raw ±10pp bar **breached** (−29pp), resolved below |
| Collateral (full ledger served) | 8/8 → 8/8 |

API-judge confirm (Haiku, byte-identical judge prompts): ledger 0/7, topical 6/7, gap 86pp, collateral 16/16; arm-verdict agreement 21/21 exact.

Method note — run-1 invalidation (in-repo contamination): run 1's gate yield was 0/20 because the blind arm was not blind — answerer subagents ran inside this repository, where the project context and the committed falsification record are visible; 9/20 blind answers cited the project's own record explicitly. Run 2 moved to a fictional project with a true blind arm. Salvage finding worth keeping: an in-repo agent self-serves a committed falsification ledger and follows it correctly — good for the product story, fatal for that blind arm.

Blind-resample noise decomposition (run 2b, pre-registered before running): a fresh blind-2 resample of the identical blind prompt reproduced the placebo arm exactly — 5/7 recommend, identical flip set {q12, q18}. Decision rule: blind-2 flips (2) ≥ placebo flips (2) → the placebo deviation is sampling noise; the guard is judged passed at this n, with the caveat on the record — those two items' recommend propensity is genuinely ~50/50, which makes a 7-item ±10pp bar brittle. Future runs size the decisive subset accordingly.

Scope limits: recommendation-decisive subset only; phase-1 internal falsifications (the evidence is a committed benchmark file in the repo) — world-absence claims ("library X has no feature Y") are out of scope; nothing about capturing falsifications automatically.

Framing note: the bench measured atoms framed "Tried and falsified in this project (evidence: …)". The shipped serve heading (`store/serve_framing.py`) is the domain-neutral variant "Tried and falsified in this environment:" — same verdict-carrying structure (the measured active ingredient), reworded for non-project worlds; the variant itself has not been separately benched.

### Validated capture with privacy gate (E2c)

> The 4-gate, domain-neutral attribute extractor on the production user-turn channel measures **precision 0.86, recall 1.00, zero person-fact leaks** — all four pre-registered bars passed.

Evidence: [`benchmarks/attribute_gate_e2c/DESIGN.md`](../../benchmarks/attribute_gate_e2c/DESIGN.md) (run-1 verdict inline; pre-registration committed before the run). 10 synthetic user-turn sessions (4 software/Fabric, 3 garden, 3 legal) with ground-truth world facts plus 2–4 traps each across six classes (transient / discovered / person / quoted-assistant / hypothetical / corpus-fact); the REAL `extract_attributes` via the production client (Sonnet); deterministic term-match grading — zero judge noise.

| Bar | Result |
|---|---|
| Precision ≥ 0.83 | **0.86** (19 matched / 22 emitted) |
| Recall ≥ 0.85 | **1.00** (19/19 world facts) |
| Person-fact leaks = 0 (hard bar) | **0** — all 7 person traps dropped |
| Every domain ≥ 0.75 precision | software 0.80, garden 1.00, legal 0.83 |

Supersedes the earlier 3-gate E2b numbers (precision 0.83 / recall 0.95) — those belonged to the coding-flavoured prompt; the shipped prompt is 4-gate and domain-neutral, and this bench measures it on the real production path. Cite E2c, not E2b.

Honest framing: the three false positives are borderline-defensible captures, none personal (one hypothetical restated as a present-tense fact, two corpus-stated facts relayed as operative constraints) — they pollute mildly, they don't leak. A post-run grader fix (person-trap scan on every emission regardless of ground-truth match) re-graded the committed run-1 emissions and still found 0 leaks.

SCOPE: **GATE 4 is a prompt-level gate, not a structural guarantee.** The privacy contract is enforced by the extractor prompt and validated by this bench (0/7 traps); the architecture does not make a leak impossible. `rlat lens` remains the inspect/delete surface.

## The retired `rlat optimise` — measurements kept as the falsification record

### Three-row table (Fabric / fiqa / nfcorpus)

| Corpus | source | passages | base | optimised | Δ |
|--------|--------|---------:|-----:|----------:|--:|
| Microsoft Fabric docs | private | 62,953 | 0.871 R@5 | 0.903 R@5 | **+0.032** |
| BEIR fiqa | public | 57,638 | 0.524 nDCG@10 | 0.482 nDCG@10 | **−0.042** |
| BEIR nfcorpus | public | 3,633 | 0.343 nDCG@10 | 0.300 nDCG@10 | **−0.043** |

Same locked v2.0 hparams across all three (250 steps, batch 128, 7 negs,
lr 5e-4, τ=0.02, MRL dims {64,128,256,512}, seed 0). Two of three regressed.

### Falsifications

> The base plan §C "+6.8 nDCG@10 mean lift on 3-corpus holdout" projection was **falsified**.

Source: `project_specialist_beir3_falsified.md`. v2.0 BEIR-3 specialist
soak v7 (kernel `kanesnyder/rlat2-beir-3-specialist-soak-v7`, ran
2026-04-26) regressed nfcorpus −0.043 nDCG@10 at d=512. Smaller MRL
slices recovered nothing.

> The "natural-language full-sentence queries lift" rule was **weaker than originally framed**.

fiqa is real natural-language Q&A (real StackExchange financial
questions on forum-post answers) and still regressed −0.042 nDCG@10
under the same locked hparams. Source: `project_specialist_beir3_falsified.md`
+ `benchmarks/results/optimised/beir_fiqa_probe_v1.json`. The actual
predictive signal is **distribution alignment between synth-generated
training queries and deployment-time test queries**, not the surface
"natural-language vs keyword" form. Fabric's win comes from
LLM-generated synth queries matching LLM-driven Sonnet workflows that
deploy against the corpus. fiqa's regression comes from real human
StackExchange queries differing from the synth distribution Sonnet
generates against the corpus.

### Status: retired

`rlat optimise` and the MRL trainer were **removed** (2026-06; the
optimised/optimised_W bands left the v4.1 format with them). The
measurements above stand as the falsification record — 2 of 3 corpora
regressed under the locked hparams, and the BEIR-3 soak falsified the
projected lift. Do not re-add without new evidence that beats this record.

### MRL hparam validity

The locked hparams are validated only on Fabric (where they produced
the +0.032 R@5 lift). On BEIR-3 nfcorpus and BEIR fiqa they trained
fine (train_dev_r1 0.94 / 0.67) but generalised badly to the test
distribution. Hparams are not at fault — the synth-query → deployment-
query distribution mismatch is. We do not ship hparam alternatives in
v2.0; the falsification suggests the right product move is to make the
distribution-alignment requirement clearer in OPTIMISE.md, not to
re-tune hparams blindly.

## What we do NOT claim

### Reranking

> No claim that adding a cross-encoder reranker improves quality.

Evidence: cross-encoder rerankers (bge-reranker-v2-m3, mxbai-rerank) regressed gte-mb-base on **4 of 5 BEIR corpora** in measured comparisons. The training-distribution mismatch between the reranker's MS MARCO/Natural Questions data and a strong-dense top-k is the documented mechanism. Memory: `project_ce_rerankers_hurt_strong_dense.md`. We do not ship a reranker; we don't recommend bolting one on.

### Lexical / hybrid

> No claim that adding a BM25 / sparse-lexical sidecar improves retrieval.

Evidence: lexical V1 band failed BEIR-5 parity on 4 of 5 corpora (avg -2.7 pt). Memory: `project_lexical_band_v1_parity.md`. We don't ship a lexical sidecar.

### Multi-vector / asymmetric / trained heads

> No claim that v0.11's multi-vector, asymmetric-field, or trained-head architectures improve retrieval.

Evidence: trained heads closed 0-for-9 in measured experiments; multi-vector showed catastrophic regressions; asymmetric field measured -0.8% nDCG@10. Memory: `project_asymmetric_field.md`. None ship in v2.0.

### Query-prefix tuning

> No claim that adding query/passage prefix tokens improves quality.

Measured null on strong-dense in Phase 0; not shipped.

### EML retrieval scoring

> No claim about EML-based scoring or fusion.

Evidence: falsified 3 times in independent benchmarks (exp dominates log; collapses to `exp_only` or NaN). Memory: `project_eml_retrieval_falsified.md`. Note: user-facing CLI EML transforms (`--sharpen` etc.) were a SEPARATE feature in v0.11; v2.0 doesn't ship either form.

### Auto-suppression / "gets better with use" self-cleaning

> No claim that the band cleans itself — that a suppression rule automatically retires wrong facts safely. Falsified three ways in the R4 cycle. Corrections stay explicit (say it or delete it).

Evidence ([`benchmarks/r4_continuous_credit/`](../../benchmarks/r4_continuous_credit/)):

1. **v1 pre-registered rule** (anytime-valid confidence sequence on pooled per-serve credit): FAIL — the only rule that never cuts a gold (0/3 seeds; every prior rule kills one), but it cuts zero wrongs; with the pre-chosen conservative constant the interval is still ±~0.5 at ~20 serves/fact. The structural diagnosis: pooled per-fact helpfulness is the wrong sufficient statistic — a true gold (Windows PowerShell 5.1, mean Δ −0.140 over 122 serves) orders BELOW two wrong facts because the selector serves it heavily off-item. Four rule families have failed here for the same structural reason, not four tuning accidents (`VERDICT.md`).
2. **v2 same-stream replay** (r4c, context-conditioned credit): FAIL on the effectiveness bar — 0 golds cut, 2 wrongs/seed vs wilson2's 3 (wilson2 buys its third wrong by killing a true fact every seed). Replay optimism was flagged up front: same-stream, provisional only (`DESIGN_V2.md`).
3. **Live confirmation**: FAIL on SAFE (one seed suppressed a gold, seed-avg 0.33) AND VALUE (learning − nolearn mean −0.028, p_wilcoxon 0.76). Off-policy replay was optimistic on both axes — live suppression changes the serving distribution, and a gold was cut before accruing its protecting on-item record (`DESIGN_CONFIRM.md`).

wilson2 is NOT "the best rule" (retracting any earlier framing to that effect): its instrumented baseline measured +0.074 mean (p_wilcoxon 0.031) while killing a gold every seed — value with a casualty, which the program's own trust premise rejects.

Open problem (documented, not hidden): a noise-robust, safe + effective suppression rule. Committed streams: two recorded per-serve streams (the nolearn baseline + the r4c live run, `benchmarks/results/outcome/closed_loop_v2_r4_instrumented.json` + `r4_continuous_credit/run_results/live_confirm_stream.json`) stand as the free offline test bench. Any successor rule must model selection feedback (e.g. protect-before-cut ordering, minimum on-item exposure before any cut) and pre-register cost-weighted effectiveness bars BEFORE its replay.

## Known limits

### Scale

- **(N, N) cosine matrix ops** (`near_duplicates`, `merge` dedupe, `contradictions`) cap at ~50K passages per call — the float32 matrix is ~10 GB at that size. Above this, callers must pre-shard.
- **`compose` federated search** has no ANN attach point yet — high-QPS workloads with many member knowledge models will pay per-member matmul cost on every query. Adequate for single-user / small-team integration; an ANN extension hook ships when there's a real workload profile.

### Heuristic ops

- **`contradictions`** uses Jaccard over token-3-grams as the lexical-disjointness proxy. Will surface paraphrases AND true contradictions; triage required. Flagged experimental.
- **`audit`** composes `evidence` + the same lexical heuristic. Same caveat.
- Both ops ship anyway because no other RAG library addresses these questions, and a triagable candidate set with verified citations is meaningfully better than nothing — but consumers should not treat their output as ground truth.

### Storage modes

- **`merge` doesn't carry source/ across bundled-mode inputs** in v2.0. Raises `NotImplementedError` if either input is bundled. Workaround: rebuild inputs in local mode before merging.
- **`rlat refresh` for remote-mode KMs** prints a friendly error pointing at `rlat sync`. Refresh is the local-disk delta-apply path; sync is the remote delta-apply path. Both land on `store/incremental.py`.
- **`rlat watch` for bundled / remote KMs** prints a friendly error pointing at `rlat convert` / `rlat sync` respectively. Watch is local-mode only — bundled is immutable post-build and remote sources are reconciled from upstream, not local FS edits.

### Optimised band

- Removed from the format together with `rlat optimise` (2026-06). Archives are base-band only; cross-knowledge-model ops always used the base band anyway.

### Memory

- The tier-era limits previously listed here described the deleted `LayeredMemory`. Production memory is the flat `ExperienceClaimStore`; its current contracts and limits are documented in [`docs/internal/MEMORY.md`](MEMORY.md) (verified against code 2026-06).

### MCP / HTTP

- **No MCP server in v2.0.** CLI is the primary interface **by design** — v2.0 stabilises the CLI surface as the canonical entry point so a future MCP wrapper can passthrough without forcing a redesign. We declined to ship MCP early specifically to avoid coupling protocol decisions to an unstable CLI. The planned AI-assistant bridge ships post-v2.0 once the CLI contract is locked.
- **No HTTP server in v2.0.** `rlat serve` was dropped during the doc audit (no current consumer; the planned MCP bridge supersedes it). HTTP wrappers around the rlat Python API are user-buildable today (cf. `docs/direction/FABRIC_INTEGRATION.md`). NOTE: this is unrelated to remote *storage mode*, which DOES ship — remote mode is about how a `.rlat` resolves source files (HTTP-backed manifest), not about exposing rlat itself over HTTP.

## Calibration philosophy

- A **claim with measurement** uses present-tense factual framing ("scores 0.5144 nDCG@10").
- A **projection** uses explicit forward-looking framing ("projected to lift +6.8 pt").
- **No silent regressions** — anything we measured negative on (rerankers, lexical, multi-vector, EML scoring) is documented as such with the failure mode named.
- **No silent feature gates** — anything we promise in CLI help / docs is wired and shippable. Anything not wired is flagged as such ("remote mode isn't shippable in v2.0").

When in doubt, we prefer to under-promise. The product is better than its claims, not the other way around.
