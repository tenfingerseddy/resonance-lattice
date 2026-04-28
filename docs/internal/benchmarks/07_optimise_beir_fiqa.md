# Benchmark 7 — BEIR fiqa optimise probe

## What it measures

Single-corpus probe of `rlat optimise` against BEIR fiqa (financial Q&A
from forum threads, 57,638 passages, 648 queries). Public-corpus positive
case for the optimise three-row table — matches the OPTIMISE.md "natural-
language full-sentence questions about content" rule:

- **base nDCG@10 / R@10** (vs locked BEIR-5 floor)
- **optimised nDCG@10 / R@10** (per-MRL-dim {64, 128, 256, 512})
- **best-d** (which MRL slice wins)
- **Δ vs base** (lift or regression at best-d)

## Hypothesis

fiqa is the BEIR-5 corpus most aligned with the optimise sweet-spot:

- Queries are real natural-language full-sentence user questions
  ("How does diversification reduce risk?", "What are the tax
  implications of...")
- Corpus answers are forum-post prose — same conversational register
  as the synth queries Sonnet generates during optimise
- 57,638 passages — comparable scale to Fabric (62,953), validates
  hparams on a comparable size

Predicted: +2-5 nDCG@10 lift at d=512 vs base.

**Falsification condition**: if fiqa regresses (similar to nfcorpus
Δ −0.043), the OPTIMISE.md "when to use" rule needs tightening before
launch + HONEST_CLAIMS.md surfaces. The result ships in the optimise
three-row table (Fabric / fiqa / nfcorpus) either way.

## Methodology

Direct adapter from `bench_beir3_optimised_soak.py`. Same locked v2.0
hparams:

- 250 training steps
- Batch 128
- 7 hard negatives per positive
- Learning rate 5e-4
- Temperature τ=0.02
- MRL nested dims {64, 128, 256, 512}
- Seed 0
- Encoder revision pinned at
  `e7f32e3c00f91d699e8c43b53106206bcc72bb22`
- Synth queries: per-query LLM call with stratify-cap 3 per source
  doc, target ~6K queries

The corpus_description anchor for the synth pipeline is fiqa-specific:

> financial Q&A from forum threads; users ask natural-language questions
> about investing, taxation, retirement planning, and personal finance,
> and seek answers grounded in forum discussion

(per `feedback_synth_queries_corpus_scoped.md`: corpus-aware anchors are
critical; the wrong description biases Claude wrong)

## Run venue

Kaggle T4 (~30-50 min for 57K passages; matches Fabric probe wall time).

Kernel slug retains pre-rename `specialist` form per the Audit 0 finding
that Kaggle slugs are immutable: `kanesnyder/rlat2-beir-fiqa-optimise-
probe-v1` is a NEW kernel pushed under post-rename naming.

Source dataset version-bumped to v3 to carry post-rename `optimise/`
module: `kanesnyder/rlat2-beir3-soak-source` v3 (re-uploaded
2026-04-27).

## Locked controls

- All hparams identical to BEIR-3 nfcorpus probe (the regression case).
  The variable under test is corpus-distribution alignment, not
  hparams.
- Encoder revision pinned via
  `install_encoder.get_pinned_revision()` — fails fast if anyone forks
  off-revision.
- Eval methodology mirrors `bench_beir5_floor.py` (same BEIR loader,
  same EvaluateRetrieval wrapper). Numbers compose with the locked
  floor.

## Reproducibility

Local (with CLAUDE_API + CUDA):

```bash
pip install rlat[optimise,bench]
export CLAUDE_API=sk-ant-...
RLAT_SOAK_CORPORA=fiqa python benchmarks/bench_beir3_optimised_soak.py \
  --output benchmarks/results/optimised/beir_fiqa_probe_v1.json
```

Kaggle T4 (recommended):

```bash
PYTHONUTF8=1 kaggle kernels push -p kaggle/rlat2-beir-fiqa-optimise-probe \
    --accelerator NvidiaTeslaT4
PYTHONUTF8=1 kaggle kernels output kanesnyder/rlat2-beir-fiqa-optimise-probe-v1 \
    -p ./kaggle_outputs/fiqa_v1/
```

The Kaggle kernel reuses the existing `kanesnyder/rlat-api-key` private
dataset for API-key delivery (per
`src/resonance_lattice/optimise/synth_queries.py:_resolve_api_key`
fallback chain).

## Honest framing

- One probe is not a definitive ruling. fiqa being one of 5 BEIR
  corpora means the result is one data point. Combined with Fabric
  (independent positive) + nfcorpus (independent negative), it forms
  a three-point hypothesis check on the corpus-profile rule.
- If fiqa positives, it means the rule "natural-language full-sentence
  user questions on natural-prose corpus → optimise lifts" is well-
  supported on a public-reproducible benchmark. Strongest possible
  evidence absent more probes.
- If fiqa negatives, it doesn't mean optimise is dead — Fabric is real
  and verified. It means the rule "BEIR-style benchmarks tend to
  regress under locked v2.0 hparams" is the more reliable framing for
  users, and OPTIMISE.md tightens accordingly.

## Output schema

```json
{
  "config": {
    "encoder_model": "Alibaba-NLP/gte-modernbert-base",
    "encoder_revision": "...",
    "runtime": "torch",
    "top_k": 10,
    "synth_mode": "per-query+stratify-cap-3-per-doc",
    "phase": "all"
  },
  "aggregate": {
    "base_ndcg_at_10_mean": ...,
    "optimised_ndcg_at_10_mean": ...,
    "delta_ndcg_mean": ...,
    "total_cost_usd": ...,
    "corpora_count": 1
  },
  "trials": [
    {
      "corpus": "fiqa",
      "n_passages": 57638,
      "n_queries": 648,
      "base_ndcg_at_10": 0.5239,
      "base_recall_at_10": 0.6114,
      "optimised_ndcg_at_10": ...,
      "optimised_recall_at_10": ...,
      "optimised_best_dim": ...,
      "optimised_per_dim": { "64": {...}, "128": {...}, "256": {...}, "512": {...} },
      "delta_ndcg": ...,
      "delta_recall": ...,
      "n_synth_queries": ...,
      "n_llm_calls": ...,
      "cost_usd": ...,
      ...
    }
  ]
}
```

## Related work surfaced from prior measurement runs

- `project_specialist_beir3_falsified.md`: locked v2.0 hparams
  regressed BEIR-3 mean nDCG@10 by −0.040. Falsified the +6.8 pt
  projection. Drives this probe — does fiqa pattern with
  Fabric (positive) or nfcorpus (negative)?
- `project_mrl_specialist_encoder.md`: best-d is corpus-dependent
  (Fabric d=256 / arXiv d=128 / PowerShell d=512). This probe also
  reports per-dim numbers so the table includes the best-d for fiqa.
- BENCHMARK_GATE.md: locked fiqa base 0.5239 nDCG@10 / 0.6114 R@10.
  This is the column-1 reference for the optimise comparison.
