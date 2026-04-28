# MRL Optimise — Internal Reference

Implementation companion to [docs/user/OPTIMISE.md](../user/OPTIMISE.md). User-facing material lives there; this doc captures the locked configuration that produced the verified Fabric +3.2 R@5 lift on the v2.0 stack, plus the Phase A/B/C spec-compliance bisect that arrived there.

> Source-of-truth code: `src/resonance_lattice/optimise/` + `src/resonance_lattice/cli/optimise.py`.
> Authoritative spec: kane's annotated dump of `mrl_fabric_remote_train.py:407-572` (the original-positive-results recipe).
> Verified result: Fabric corpus, gte-mb backbone, base R@5 = 0.871, optimised R@5 = 0.903 at d=256/512 (+3.2 pt). Matches the spec's recorded MRL d=512 R@5 = 0.9032 verbatim. Run JSON: `benchmarks/results/optimised/fabric_probe_v1.json` (commit `c4727899` v6 v7, 2026-04-26).

## Pipeline shape

```
.rlat (base band, registry, source/) ──┐
                                       ▼
                     synth_queries.generate(..., corpus_description="...")
                       │   1 LLM call: 5 corpus-aware style anchors (JSON array)
                       │   ~6000 calls: ONE per query, 8-worker ThreadPool
                       │   stratify cap-3-queries-per-source-file, seed-shuffle
                       │   filters: len<10, len>400, "passage" in lower()[:40]
                       │   per-corpus disk cache (append-only, lock-guarded)
                       ▼
                  list[SynthQuery]
                       │
              Encoder.encode(queries) (gte-mb 768d, frozen)
                       ▼
                query_embeddings (Q, 768)
                       │
            mine_negatives.mine(q, base_band, q_pos)
                       │   top-64 candidates per query, drop positive,
                       │   np.shuffle(seed=0), take 7
                       ▼
                negatives (Q, 7)
                       │
             train_mrl.train(base, q, q_pos, negs, device)
                       │   250 InfoNCE steps over MRL_DIMS={64,128,256,512}
                       │   AdamW(weight_decay=0.0), randn*1/sqrt(768) init
                       │   fp32 throughout (no autocast — temp=0.02 saturates fp16)
                       │   early-kill at step 100 (dev R@1 < 0.2)
                       ▼
                W (512, 768) float32
                       │
            write_slot.project_and_write(km, W)
                       │   project: base @ W.T → (N, 512), L2
                       │   ANN: FAISS HNSW if N > 5000
                       │   atomic in-place via archive.write_band_in_place
                       ▼
.rlat (base + optimised + W + optimised_ann)
```

## Phase A/B/C — the spec-compliance bisect

**Background.** The first v2.0 implementation regressed on every probe corpus tested (nfcorpus v3/v4 −0.05 nDCG@10 across all 4 MRL slices; Fabric v5 −0.10 R@5). All three of the spec's "subtle traps that cause replication failure" were present in the v2.0 port. Patches landed 2026-04-26 across three commits:

### Phase A — `train_mrl.py` (commit `6a80d0f0`)

Three single-line divergences from the original-positive-results spec, all called out explicitly in the spec's "subtle traps" section:

| Bug | v2.0 (broken) | spec (fixed) | Why it matters |
|------|---------------|---------------|-----------------|
| `weight_decay` | PyTorch AdamW default = `0.01` | `0.0` (explicit) | On a tiny (512,768) projection across 250 steps, 0.01 decay continuously pulls W toward zero |
| `W` init | `xavier_uniform_(W)` (≈0.0685 spread) | `randn(*) * (1/sqrt(backbone_dim))` (≈0.0361 spread) | xavier ≈2× larger spread → softmax saturates differently with temp=0.02 |
| Train loop dtype | `fp16` on CUDA | `fp32` always | temp=0.02 → logits in [−50,+50]; exp(50) ≈ 5.2e21 vs fp16 max ≈ 65504 → silent overflow |

### Phase B — `mine_negatives.py` (commit `dfcf0c0c`)

| Bug | v2.0 (broken) | spec (fixed) | Why it matters |
|------|---------------|---------------|-----------------|
| Mining strategy | top-8 candidates → take 7 hardest in cosine order | top-64 candidates → drop positive → np.shuffle(seed=0) → take 7 | "7 hardest possible" biases against likely false-negatives (high-cos paraphrases that ARE relevant). Wider hard pool + shuffle dilutes systemic false-negative contamination |

### Phase C — `synth_queries.py` (commit `7d678f5f`)

Five structural changes from the per-file-batched v2.0 implementation:

| Aspect | v2.0 (broken) | spec (fixed) |
|--------|---------------|---------------|
| Granularity | ONE LLM call per file (all passages batched + JSONL output) | ONE LLM call per query |
| Concurrency | sequential | 8-worker `ThreadPoolExecutor` |
| Stratification | none — all-passages-from-file in one call | cap 3 queries per source_file → seed-shuffle → trim to target ~6000 |
| Filters | none in `_parse_synth_jsonl` | `len<10`, `len>400`, `"passage" in q.lower()[:40]` → reject |
| Corpus identity | no `corpus_description` parameter | required `corpus_description` propagated through anchor + per-query system prompts |

**Cache schema** changed from per-file JSONL to per-corpus single JSONL (append-only, lock-guarded for concurrent worker safety). Mid-run crashes resume by skipping passage_idx values already in the cache.

### Why all three at once

The bugs compound. fp16 saturation (A3) masks weight_decay over-decay (A1) by also corrupting gradients. Wrong init (A2) saturates the first step's softmax → temp=0.02 gradients are stale before anything else gets a chance. Per-file batched synth queries (C1) feed a different distribution than the per-passage stratified set the original recipe trained against — even with all of A+B fixed, low-quality training data still regresses. v2.0 v6 confirmed: Phase A+B alone is insufficient on its own (untested directly, but the bugs are mathematically interdependent enough that bisecting individual fixes is wasted budget).

### Bisect option for v2.1+

If a future change regresses again, individual phases are revertible:
- `git revert 7d678f5f` → drops Phase C (synth_queries spec rewrite)
- `git revert dfcf0c0c` → drops Phase B (mine_negatives top-64+shuffle)
- `git revert 6a80d0f0` → drops Phase A (train_mrl spec hparams)

## `optimise/device.py`

`select() -> "cuda" | "cpu"`: tries `torch.cuda.is_available()` via `importlib.util.find_spec` (no torch import on miss). Returns `"cpu"` unconditionally if torch isn't installed.

`estimate_wall_time(device, n_passages, n_queries) -> float`: minutes. Calibrated against measured runs:

- CUDA T4 (Kaggle free tier): ~50 min for 62K passages × 6K queries (Fabric v6: 6.4 min encode + 38 min synth + 3 sec train + 1 min eval).
- CUDA A100 (RunPod): ~25 min for the same — spec's calibration corpus.
- CPU: ~5h for the same. Linear in `N × Q`.

## `optimise/mine_negatives.py`

`mine(query_embeddings, passage_embeddings, query_passage_idx, top_k=64, seed=0) -> (Q, 7) int32`. Single batched matmul `Q @ P.T`, `argpartition` for top-64 indices, drop the positive `passage_idx` from each query's row, `np.random.default_rng(seed).shuffle(non_positive)`, take first 7.

Edge case: if positive isn't in the top-64 (off-distribution synth query), all 64 candidates are eligible; shuffle + take 7. If after dropping positive there are fewer than 7 distinct candidates (degenerate small-corpus case), cycle-pad. If zero non-positive candidates remain (only happens at N==1), raise — silent NaN downstream is worse than loud error.

## `optimise/train_mrl.py`

### Hyperparameters (locked, no CLI override)

| Parameter | Value | Rationale |
|---|---:|---|
| `STEPS` | 250 | Spec — full convergence on 40K-passage corpora. |
| `BATCH` | 128 | Spec. |
| `NEGATIVES_PER_QUERY` | 7 | Spec. |
| `LR` | 5e-4 | Spec. AdamW. |
| `WEIGHT_DECAY` | **0.0** | **Spec — must be explicit; PyTorch default is 0.01 (different recipe).** |
| `TEMPERATURE` | 0.02 | InfoNCE temperature. Critical with W init scale: any larger spread saturates softmax on step 1. |
| `MRL_DIMS` | `(64, 128, 256, 512)` | Spec. Nested — d=256 is the first 256 columns of d=512. |
| `MRL_WEIGHT` | `1/4` per slice | Uniform. |
| `EARLY_KILL_R1_THRESHOLD` | 0.2 | Spec — calibrated against truly broken runs. |
| `EARLY_KILL_STEP` | 100 | Spec. |
| `NO_PROGRESS_WINDOW` | 60 | Suppressed if `best_loss < SATURATION_FLOOR`. |
| `SATURATION_FLOOR` | 0.01 | Clean convergence shouldn't trip either kill. |
| `SEED` | 0 | Spec. |
| `DEV_FRACTION` | 0.1 | 10% holdout for the early-kill check. |

Any change to these pairs with a `BENCHMARK_GATE.md` update — the verified Fabric +3.2 R@5 was measured at exactly these values.

### W initialization

```python
W = torch.empty(d_native, backbone_dim, dtype=torch.float32, requires_grad=True)
W.normal_(mean=0.0, std=1.0 / (backbone_dim ** 0.5))
```

For gte-mb: std ≈ 0.0361. **Critical for temp=0.02 stability.** Using `xavier_uniform_(W)` (std≈0.069) saturates the first softmax and the recipe collapses on step 1 — this was the v2.0 bug fixed in Phase A.

### Loss

```python
for d in MRL_DIMS:
    q_d   = L2(q_proj[:, :d])           # (B, d)
    pos_d = L2(pos_proj[:, :d])         # (B, d)
    neg_d = L2(neg_proj[:, :, :d])      # (B, K, d)

    pos_sim = (q_d * pos_d).sum(-1) / TAU            # (B,)
    neg_sim = (q_d.unsqueeze(1) * neg_d).sum(-1) / TAU   # (B, K)
    logits = concat([pos_sim, neg_sim], dim=1)        # (B, K+1)
    loss += MRL_WEIGHT * cross_entropy(logits, target=0)
```

The target=0 reduces `cross_entropy` to `-log(softmax_0)`. Uniform 0.25 weighting across slices means smaller dims pull as hard as larger dims, which is what makes the nested representation hold up under truncation.

**No in-batch negatives beyond the 7 mined.** Only `(1 + n_neg) = 8` columns per row. Several MRL/Matryoshka reference implementations use full in-batch all-pairs — that's a different recipe with different temperature requirements.

### Precision

fp32 throughout the train loop on every device. Spec calls for "loss precision = fp32 (no autocast inside train loop)". With `temp=0.02` the logits = sim/0.02 land in [−50,+50]; `exp(50) ≈ 5.2e21` vs fp16 max ≈ 65504 → silent overflow that produces stale gradients indistinguishable from convergence. Phase A3 fix.

Embedding precision (passage encode + query encode at corpus-build time) IS the encoder's concern; it can run bf16/fp16 with autocast there because no division-by-temp happens during encode.

### Returns

```python
@dataclass
class TrainResult:
    W: np.ndarray              # (d_native=512, backbone_dim=768) float32
    final_loss: float
    final_dev_r1: float
    steps_completed: int
    early_killed: bool
    kill_reason: KillReason
```

`progress(step, loss, best_loss)` callback fires every step; cli/optimise throttles to once per second for stderr output.

## `optimise/synth_queries.py`

### LLM-call seam

```python
LLMResponse = NamedTuple("LLMResponse", "text input_tokens output_tokens")
LLMClient = Callable[[str, list[dict], int], LLMResponse]
```

Production wraps `anthropic.Anthropic.messages.create` via `default_client(api_key)`. The third argument is `max_tokens` so anchor calls (1024 tokens) and per-query calls (200 tokens) can use different budgets without per-client config. Tests substitute a fake callable returning canned `LLMResponse` namedtuples — no SDK monkey-patching.

### Two LLM phases

**1. Style anchors** (1 call per corpus). 15 sampled passages + `corpus_description` → 5 corpus-aware natural-language query anchors as a JSON array:

```
You are analyzing a corpus described as: {corpus_description}.
Below are 15 random passages from this corpus. Based on the content and style
of these passages, produce exactly 5 plausible natural-language questions that
a real user of this corpus would ask. Match the register and topic-distribution
you see in the passages. Do NOT quote the passages verbatim. Return ONLY a JSON
array of 5 strings, no preamble or explanation.
```

Cached at `<cache_dir>/<fp>/_anchors.json`. Reused on re-run.

**2. Synth queries** (one call per query, 8-worker concurrent). System prompt carries `corpus_description` + the 5 anchors:

```
You are a user of the following corpus: {corpus_description}. Given a passage
from this corpus, produce ONE natural-language question that a real user would
ask whose answer is contained in the passage. The question should NOT verbatim
copy sentences from the passage; it should reflect how a real user of this
corpus would phrase the query. Avoid meta-phrasings like 'according to the
passage' or 'what does the article say'. Return ONLY the question, no preamble.

Style anchors (these are example real-user questions matching the natural
style of this corpus, not your target passage):
- {anchor 1}
...
```

User content is just `PASSAGE:\n\n{text[:4000]}\n\nWrite one question.`

### Stratification

```python
def _stratified_passage_sample(passage_idxs, source_files, queries_per_file_cap=3, target_queries=6000, seed=0):
    rng = random.Random(seed)
    by_file = {}
    for pos, src in enumerate(source_files):
        by_file.setdefault(src, []).append(pos)
    candidates = []
    for src, positions in by_file.items():
        rng.shuffle(positions)
        candidates.extend(positions[:queries_per_file_cap])
    rng.shuffle(candidates)
    return candidates[:target_queries]
```

Without stratification, big files dominate the synth distribution. Spec §4c precise.

### corpus_description — most common cross-corpus failure

**Always pass it.** Spec §9 trap #3: "Empty `--corpus-description` falls back to a generic placeholder AND falls back to Fabric-style anchors → biased synth queries on non-Fabric corpora." Examples that match the verified-positive corpora:

| Corpus | corpus_description |
|--------|---------------------|
| Fabric docs | `"Microsoft Fabric documentation"` |
| PowerShell docs | `"PowerShell documentation"` |
| arXiv abstracts | `"arXiv research abstracts"` |
| BEIR nfcorpus | `"biomedical research papers; users ask medical and nutrition information-need queries"` |
| BEIR scifact | `"scientific paper abstracts; users issue claim-verification queries"` |
| BEIR arguana | `"argumentative essays; users issue a claim and seek counter-arguments"` |

CLI: `rlat optimise <km> --corpus-description "..."`. The `cli/optimise.py` argparse default is a generic fallback so the command runs without it, but the flag is the difference between "the spec replicates" and "the spec regresses."

### File-level cache

```
<cache_dir>/
└── <corpus_fingerprint>/
    ├── _anchors.json          # 1 LLM call, 5 strings
    └── queries.jsonl          # 1 line per kept query, append-only
```

`corpus_fingerprint` is `sha256` over `(passage_idx, first 100 chars)` tuples — different corpora get different cache dirs, identical re-runs hit cache. Mid-run crashes resume by skipping `passage_idx` values already in `queries.jsonl`. Lock-guarded for concurrent worker safety.

### Cost (verified)

`estimate_cost(target_queries=6000) -> float (USD)`:
- Per-query: ~400 input tokens (system + anchors + passage) × ~30 output tokens
- Anchor: ~3000 input + 200 output, one-time

Sonnet 4.6 at $3/M input + $15/M output. Verified Fabric run: **$8.16** for 6000 queries on 62K-passage corpus.

## `optimise/write_slot.py`

`project_and_write(km_path, W, nested_mrl_dims=(64, 128, 256, 512))`:

1. Load archive, fetch base band.
2. Check `W.shape[1] == base.shape[1]` (W is `(d_native, backbone_dim)`).
3. Project: `optimised = (base @ W.T)`; L2-normalise.
4. Build FAISS HNSW index if `N > field.ann.ANN_THRESHOLD_N`.
5. Construct `BandInfo(role="in_corpus_retrieval", dim_native, w_shape, nested_mrl_dims, trained_from="bands/base.npz")`.
6. `archive.write_band_in_place(km_path, band_name="optimised", band_info, band_data, projection=W, ann_blob)`.

Atomic-write contract from `archive.write_band_in_place` (Phase 2 #17): tmp ZIP + `os.replace`, exception-safe tmp cleanup, all other archive members streamed through unchanged via `shutil.copyfileobj`.

## `cli/optimise.py`

The orchestrator. Pre-flight `--estimate` prints cost + duration without touching the API. Without `--estimate`, prints the same estimate then prompts via stdin (`y` / `yes`); `--yes` / `-y` skips the prompt.

`--corpus-description` flag added 2026-04-26 with help text quoting spec §9 trap #3. Generic fallback so the command runs without it, but every non-trivial corpus should pass it.

Reads passage texts via `open_store_or_exit` (any storage mode), so a `bundled` archive optimises from inside the .rlat without needing a `--source-root`.

Exit codes:
- `0` — success.
- `1` — runtime error (no base band, store unreachable, write failure).
- `4` — `train_mrl.EXIT_EARLY_KILLED` (dev R@1 below threshold at step 100).

After a successful run, the next `rlat search` automatically uses the optimised band via `ArchiveContents.select_band(prefer=None)` (optimised-if-present-else-base).

## Verified results

### Fabric (canonical positive-results corpus)

| Dim | R@5 | Hit@1 | ΔR@5 vs base 0.871 |
|---:|---:|---:|---:|
| 64 | 0.806 | 0.484 | −0.065 |
| 128 | 0.839 | 0.419 | −0.032 |
| **256** | **0.903** | 0.484 | **+0.032** |
| **512** | **0.903** | 0.516 | **+0.032** |

Best d = 256/512 tied. Replicates the spec's recorded `MRL d=512 R@5 = 0.9032` exactly.
Run: `benchmarks/results/optimised/fabric_probe_v1.json`. Backbone: gte-modernbert-base 768d, pinned revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. Cost: $8.16. Wall: ~50 min on Kaggle T4.

### BEIR-3 status (closed)

Two runs on Kaggle confirmed the locked v2.0 hparams **regress public BEIR corpora**:

- **nfcorpus** (`rlat2-beir-3-specialist-soak-v7`, 2026-04-26): base 0.343 → optimised 0.300 nDCG@10 (Δ −0.043) at best d=512. Per-dim {64: 0.207, 128: 0.244, 256: 0.283, 512: 0.300} — every slice below base. Run JSON: `benchmarks/results/beir/new_arch/v2_specialist_soak_3beir.json`. Cost: $9.39.
- **fiqa** (`rlat2-beir-fiqa-optimise-probe-v1`, 2026-04-27): base 0.524 → optimised 0.482 nDCG@10 (Δ −0.042) at best d=512. Per-dim {64: 0.309, 128: 0.403, 256: 0.454, 512: 0.482}. Run JSON: `benchmarks/results/optimised/beir_fiqa_probe_v1.json`. Cost: $2.47.

Both corpora trained cleanly (nfcorpus train_dev_r1 0.94; fiqa 0.67) — no early kill, no loss anomalies. Generalisation to test queries failed.

**Falsification of base plan §C "+6.8 nDCG@10 mean lift" projection.** The actual predictive signal is distribution alignment between synth-generated training queries and deployment-time test queries, not corpus surface form. Fabric (Sonnet-against-MS-docs ≈ Sonnet-against-MS-docs deployment) wins; BEIR (Sonnet-generated synth queries ≠ real-user StackExchange / medical-search queries) loses.

Three-row table (canonical position in `docs/user/OPTIMISE.md` and `docs/user/BENCHMARKS.md`):

| Corpus | source | passages | base | optimised | Δ |
|--------|--------|---------:|-----:|----------:|--:|
| Microsoft Fabric docs | private | 62,953 | 0.871 R@5 | 0.903 R@5 | +0.032 |
| BEIR fiqa | public | 57,638 | 0.524 nDCG@10 | 0.482 nDCG@10 | −0.042 |
| BEIR nfcorpus | public | 3,633 | 0.343 nDCG@10 | 0.300 nDCG@10 | −0.043 |

Two of three regressed. `rlat optimise` ships **opt-in-with-real-caveat**.

## Known limits

- **Heterogeneous corpora trade hit@1 for R@5.** Fabric optimised gains +3.2 R@5 but loses 6.5pt hit@1 at d=256. Consumer code should use top-K retrieval, not top-1, to benefit.
- **Dim selection is corpus-dependent.** Spec memory `project_mrl_optimised_encoder.md` recorded best-d as Fabric=256, arXiv=128, PowerShell=512. v2.0 ships locked d=512 with d=256 nested-equivalent; both work as the "headline" lift dim. v2.1 may add `auto-d` that picks at write_slot time via dev R@1 across slices.
- **Synth distribution must approximate test queries.** Corpora where real-world queries diverge significantly from LLM-generated synth (e.g. BEIR nfcorpus medical info-need queries) may show smaller lift than coherent project-internal corpora. `corpus_description` is the primary lever.

## Testing

End-to-end smoke (`tests/harness/optimise_roundtrip.py`): build a tiny corpus, monkey-patch `synth_queries.default_client` to return canned `LLMResponse` namedtuples, run the full pipeline, assert optimised band exists with expected dim and `W.shape == (d_native, backbone_dim)`. Mock LLM matches the new 3-arg `(system, messages, max_tokens)` contract; anchor returns JSON array; per-query returns plain text question.

The fake-client smoke is the only path that doesn't need an API key. Real-client validation runs through `bench_fabric_optimised_probe.py` + `bench_beir3_optimised_soak.py` on Kaggle.

`band_parity` suite: confirms optimised-present + base-only archives both retrieve correctly via `select_band(prefer="optimised")` and `select_band(prefer="base")` — the cross-knowledge-model rule depends on this.

## Cross-references

- Spec source: kane's annotated `mrl_fabric_remote_train.py` dump (2026-04-26).
- User-facing: [OPTIMISE.md](../user/OPTIMISE.md).
- Format: [KNOWLEDGE_MODEL_FORMAT.md](./KNOWLEDGE_MODEL_FORMAT.md) §"bands/optimised".
- Floor numbers: [BENCHMARK_GATE.md](./BENCHMARK_GATE.md).
- Spec-compliance commits: `6a80d0f0` (Phase A train_mrl), `dfcf0c0c` (Phase B mine_negatives), `7d678f5f` (Phase C synth_queries).
- Verified result: `c4727899` v6 v7 Fabric run.
