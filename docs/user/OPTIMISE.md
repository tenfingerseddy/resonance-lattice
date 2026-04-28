# Optimise

`rlat optimise` adds an **MRL optimised band** to a knowledge model — a corpus-specific projection that lifts retrieval quality (typically a few R@5 points) AND compresses the embedding dim (768 → 512) at the cost of one LLM bill per knowledge model. **Opt-in, idempotent, in-place**: run it once per corpus, re-run if you regenerate the base.

The optimised band is **corpus-specific, not user-specific.** Synth queries are LLM-generated natural-language questions about your corpus content; real users querying that same corpus also ask natural-language questions about that content. Both distributions land in similar territory, so the optimised band lifts retrieval for ANY natural-language querier of the corpus — not just you. The lift doesn't transfer to OTHER corpora, but it does transfer across users.

## Verified results — three-row table

| Corpus | source | passages | base | optimised | Δ | best d |
|--------|--------|---------:|-----:|----------:|--:|-------:|
| **Microsoft Fabric docs** (canonical) | private | 62,953 | 0.871 R@5 | **0.903 R@5** | **+0.032** | 256 / 512 (tied) |
| BEIR fiqa (financial Q&A, real StackExchange) | public | 57,638 | 0.524 nDCG@10 | 0.482 nDCG@10 | **−0.042** | 512 (best of bad) |
| BEIR nfcorpus (medical, keyword) | public | 3,633 | 0.343 nDCG@10 | 0.300 nDCG@10 | **−0.043** | 512 (best of bad) |

Result JSONs: [`fabric_probe_v1.json`](../../benchmarks/results/optimised/fabric_probe_v1.json) ·
[`beir_fiqa_probe_v1.json`](../../benchmarks/results/optimised/beir_fiqa_probe_v1.json) ·
[`v2_specialist_soak_3beir.json`](../../benchmarks/results/beir/new_arch/v2_specialist_soak_3beir.json)
(nfcorpus + scifact + arguana). All under the same locked v2.0 hparams
(250 steps, batch 128, 7 negs, lr 5e-4, τ=0.02, MRL dims {64,128,256,512}, seed 0).

**Two of three regressed.** The optimised band is opt-in-with-real-caveat, not "always lifts".

### What the three rows actually tell us

The original framing — "optimise lifts on natural-language full-sentence
queries" — was too loose. fiqa is real natural-language Q&A from
StackExchange (forum users asking financial questions) and **still
regressed** at every MRL dim. The actual predictive signal is narrower:

**Optimise lifts when the synth-query distribution Sonnet generates
matches the deployment-time query distribution.** Fabric's win came
because Microsoft's documentation queries (real and synth) live in the
same conceptual register; Anthropic's product, not real users, generates
the synth queries that match what other Sonnet-driven workflows ask.
fiqa's regression came because real StackExchange users use
colloquial / specific phrasing, while Sonnet generates more formal /
conceptual synth queries against the same corpus — the trained
projection fits the synth distribution (train_dev_r1 0.67 on fiqa, 0.94
on Fabric) but doesn't transfer to the real test queries.

This is exactly the register-shift hypothesis from
`project_specialist_beir3_falsified.md` — except it now applies to
fiqa too, despite fiqa LOOKING natural-language. Be cautious about
inferring from query *appearance*; what matters is whose distribution
generates the deployment queries.

## When to optimise

The strongest predictor of lift is **distribution alignment between
synth queries and deployment queries**, not "natural-language vs
keyword". Optimise is most likely to help when:

- Your **deployment-time queries come from an LLM-driven workflow** —
  Sonnet (or another assistant) is generating questions about your
  corpus to answer end-user requests. Synth queries match deployment
  queries because both come from LLMs in similar conceptual registers.
- Your corpus is **stable** — re-optimising costs another LLM bill, so
  pair it with corpus regenerations, not weekly edits.
- You have **≥1,000 passages**. Smaller corpora produce too few synth
  queries for InfoNCE to converge (training early-kills at step 100 if
  dev R@1 < 0.2).
- Your corpus has **coherent topical identity** — Fabric documentation,
  PowerShell reference, a single product's API guide. Coherent corpora
  let the LLM derive corpus-aware style anchors that match the real
  query distribution.

## When **not** to optimise

- **Real human users issue colloquial / specific queries** that don't
  match what Sonnet generates against the same corpus. Three out of
  three measured BEIR corpora regressed under locked hparams. fiqa is
  the cautionary tale — it *looks* natural-language but regressed
  −0.042 nDCG@10.
- **Corpus is too small** (< ~1,000 passages). Locked hparams
  (batch=128, 250 steps, 7 negatives) don't have enough signal; you'll
  spend ~$10 + 30min GPU and get a band no better than base.
- **Heterogeneous corpus** (mixed unrelated topics, mixed languages,
  mixed registers). The trained projection averages across topics and
  helps none of them as much as separate per-topic `.rlat`s +
  cross-corpus ops would.
- **Keyword-style queries** ("auth bug", "deploy"). Synth queries are
  full-sentence; keyword queries land out-of-distribution and the lift
  drops to ~0 or goes negative.
- **Base band already saturated** on your corpus (e.g. base nDCG@10
  ≥ 0.85 on a stable benchmark). Headroom for the optimised band is
  small; cost rarely justifies the marginal lift.
- **Cost / time isn't worth it.** Base band is free; ~$10 + ~50min on
  T4 is real money for a marginal product win.

**Recommended workflow**: run `rlat optimise --estimate` for a cost
preview, then optimise on a small sample (e.g. 5K passages) and measure
on a held-out test set before optimising the full corpus.

Cross-knowledge-model `rlat compare` always uses the base band, so optimising both sides of a comparison doesn't change comparison numbers.

## Cost & duration

| Corpus size | LLM cost (Sonnet 4.6) | Wall time on CUDA (T4) | Wall time on CPU |
|---|---|---|---|
| 5K passages | ~$3 | ~10 min | ~1 hour |
| 40K passages | ~$8–12 | ~50 min | ~5 hours |
| 100K passages | ~$15–25 | ~90 min | ~12 hours |

Cost scales with `target_queries` (default 6,000 — capped 3 queries per source file with seed-stratified sampling), not raw passage count. Wall time is dominated by the synth-query LLM calls, parallelized across 8 worker threads.

Get a real number for your corpus before committing:

```bash
rlat optimise my-corpus.rlat --estimate
```

Output:

```
[optimise] pre-flight estimate
  knowledge model     39235 passages, 2412 files
  device              cpu
  LLM calls           ≈ 6001 (1 anchor + 6000 per-query, cap 3/file)
  cost                ≈ $8.50 USD (Sonnet 4.6)
  training queries    ≈ 5,100 (after retention)
  wall time           ≈ 280 min (plus LLM call time, 8-worker concurrent)
```

## Always pass `--corpus-description`

This is the single most important flag. The synth-query pipeline conditions both the style-anchor LLM call AND every per-query LLM call on a one-line corpus description. Omitting it falls back to a generic placeholder — biased synth queries that don't match real-user query style on your corpus.

```bash
rlat optimise my-corpus.rlat \
  --corpus-description "Microsoft Fabric documentation"

rlat optimise powershell.rlat \
  --corpus-description "PowerShell documentation"

rlat optimise medical-papers.rlat \
  --corpus-description "biomedical research papers; users ask medical and nutrition information-need queries"
```

Match the register and topic of real users. Verified examples that produced lift:

| Corpus | corpus_description |
|--------|---------------------|
| Microsoft Fabric docs | `"Microsoft Fabric documentation"` |
| PowerShell docs | `"PowerShell documentation"` |
| arXiv abstracts | `"arXiv research abstracts"` |
| BEIR nfcorpus (regressed) | `"biomedical research papers; users ask medical and nutrition information-need queries"` |

The `--corpus-description` argument has a CLI default that's a generic fallback so the command runs without it, but **every non-trivial corpus should pass an explicit string**.

## API key

`rlat optimise` is the only command that needs an LLM key. Discovery order:

1. If `RLAT_LLM_API_KEY_ENV` is set, read the env var named by its value.
2. Else `CLAUDE_API`.
3. Else `ANTHROPIC_API_KEY`.

```bash
# Common case
export ANTHROPIC_API_KEY=sk-ant-...
rlat optimise my-corpus.rlat --corpus-description "..."

# Indirection if your shell already has the key under a different name
export MY_KEY=sk-ant-...
export RLAT_LLM_API_KEY_ENV=MY_KEY
rlat optimise my-corpus.rlat --corpus-description "..."
```

Without a key, `rlat optimise` errors with the same message verbatim.

## Optimising a remote-mode knowledge model

`rlat optimise` reads passage texts via the Store. Against a remote-mode archive that means an HTTP fetch per file during the synth-query pipeline — a network blip mid-run wastes the LLM bill. The right shape is to materialise locally first using `rlat convert`, then optimise:

```bash
rlat convert upstream.rlat --to local --source-root ./local-mirror/ -o working.rlat
rlat optimise working.rlat --corpus-description "Microsoft Fabric documentation"
# Optionally back to remote afterwards:
rlat convert working.rlat --to remote --remote-url-base https://upstream.example.com/v1/
```

The conversion is fast (no re-encoding) and the optimise pass then runs entirely against the local copy. The optimised W projection, once trained, survives the round-trip back to remote unchanged — `convert` is band-preserving by design.

## Run

```bash
rlat optimise my-corpus.rlat --corpus-description "..."             # interactive — confirms cost
rlat optimise my-corpus.rlat --corpus-description "..." --yes       # skip confirmation
rlat optimise my-corpus.rlat --corpus-description "..." \
                              --cache-dir ./cache                    # custom synth-query cache
rlat optimise my-corpus.rlat --estimate                              # dry run
```

The pipeline:

1. **Style anchors** (LLM): one call. 15 sampled passages + your `corpus_description` → 5 corpus-aware natural-language query-style anchors as a JSON array. Cached per-corpus.
2. **Synth queries** (LLM): one call per query, 8-worker `ThreadPoolExecutor`. Stratification caps queries-per-source-file at 3 with seed=0 shuffle, then targets ~6000 total queries. Length + meta filters drop polluted queries.
3. **Encode queries** (frozen gte-modernbert-base): one local pass.
4. **Hard-negative mining**: top-64 cosine candidates per query against the base band; drop the positive, np.shuffle(seed=0), keep 7.
5. **Train MRL InfoNCE**: 250 steps on `(64, 128, 256, 512)` nested slices with uniform 1/4 weighting. fp32 throughout. Locked hyperparameters; no `--lr` / `--steps` knobs.
6. **Project + atomic write**: passages projected through `W` → `(N, 512)` L2-normalised, written back to the `.rlat` alongside the W matrix and a fresh FAISS HNSW index for the optimised band.

The replacement is atomic: the original archive stays intact until the new optimised band + W + ANN are fully written, then `os.replace` swaps in the new ZIP. A crash mid-run leaves the original `.rlat` unchanged.

## Idempotent

Re-running `rlat optimise` retrains the same slot. Useful when you've changed the synth-query cache (deleted a corrupted file's cache entry) or want to re-roll the InfoNCE seed (currently locked at 0).

## Early-kill

If dev R@1 at d=512 is below 0.2 at training step 100, the run aborts with exit code **4** and the optimised band is **not** written. Cause: synth queries too off-distribution from the corpus, or the corpus is too small. The base archive is untouched.

## Querying an optimised corpus

After `rlat optimise` succeeds, every subsequent `rlat search` automatically uses the optimised band — no flag needed. The base band is still inside the `.rlat` (cross-model `rlat compare` always uses it).

```bash
rlat search my-corpus.rlat "..."
# [search] band=optimised ann=yes hits=10 (39235 passages)
```

If you want to force base-band search for any reason (debugging, comparison), there's no CLI flag in v2.0 — extract via the Python API:

```python
from resonance_lattice.store import archive
contents = archive.read("my-corpus.rlat")
handle = contents.select_band(prefer="base")  # optimised override
```

## Caching

Synth-query cache layout (default `~/.cache/rlat/optimise/`):

```
~/.cache/rlat/optimise/
└── <corpus_fingerprint>/
    ├── _anchors.json
    └── queries.jsonl        # one line per kept query, append-only, lock-guarded
```

The cache lives outside any project directory by default so it doesn't get accidentally `git add .`-ed. Mid-run crashes resume by skipping passage_idx values already in `queries.jsonl`. Override with `--cache-dir <dir>` if you want it sibling to a specific knowledge model.

Delete `<corpus_fingerprint>/queries.jsonl` to force a clean re-run with fresh synth queries (preserves the anchor cache so you don't pay for the anchor call again). Delete the whole `<corpus_fingerprint>/` directory to re-roll anchors + queries.

## What doesn't work

- **Optimising a remote-mode .rlat**: the synth pipeline reads passage texts via the Store, which works for `bundled` and `local` modes; `remote` works in principle but requires network for every passage fetch and isn't recommended.
- **Re-rolling the seed without changing synth queries**: training is deterministic on `(synth queries, seed=0)`. To get a different W, delete the cache and re-run (which re-pays the API).
- **Cross-corpus W transfer**: the trained W is corpus-specific. Loading it into a different `.rlat` is not supported and would produce a non-meaningful band.

## Related

- [docs/internal/OPTIMISE.md](../internal/OPTIMISE.md) — implementation reference (training pipeline, locked hparams, Phase A/B/C bisect history).
- [docs/internal/BENCHMARK_GATE.md](../internal/BENCHMARK_GATE.md) — published-vs-measured numbers including the verified Fabric run.
