# Benchmark 6 — Build & query speed

## What it measures

Developer-experience metrics for `rlat` and named alternatives at a fixed
grid of corpus sizes. **No LLM calls.** Fully reproducible — deterministic
synthetic corpus, locked seed, locked encoder revision.

Per (approach × N):

- **build_seconds** — corpus → searchable index
- **warm_query_p50_ms / p95_ms** — median + 95th-percentile of N=50
  warm queries
- **on_disk_mb** — final index size on disk
- **peak_rss_mb** — peak RSS during build (cross-platform via psutil
  on Windows / `resource.getrusage` on Unix)

## Approaches

| Key | Approach |
|---|---|
| `rlat` | gte-modernbert-base 768d via auto-runtime (ONNX/OpenVINO), FAISS HNSW (M=32, efC=200, efS=128) above N=5000 |
| `sentence-transformers+faiss` | same backbone via PyTorch, FAISS HNSW (same params) |
| `chromadb` | Chroma local with default all-MiniLM-L6-v2 ONNX embedder + Chroma's HNSW |
| `pinecone` | (opt-in via `--include-pinecone`; network-bound, requires `PINECONE_API_KEY`) |

## Locked corpus + queries

Synthetic corpus: random walks of a fixed 100-word vocabulary at lognormal-
distributed lengths (mean ~80 words, clipped 30–250). Locked seed `0`.
Same corpus across every approach. Same 50 queries across every approach.

Reproducibility comes from seed + encoder-revision pin: re-running on the
same hardware should produce bit-exact warm-query times within OS noise.

## Why this measures what it claims

- **Build time** is dominated by encoder-encode time. The same encoder
  via different runtimes (ONNX vs PyTorch) genuinely produces different
  build times — this is an honest comparison of the runtime stack.
- **Warm query latency** is the user-facing metric. P50 and P95 capture
  the typical and worst-case interactive query.
- **On-disk size** matters for distribution: a smaller `.rlat` ships
  faster, fits on more devices, and decompresses in less RAM.
- **Peak RSS** matters for cheap CI / edge deployment: rlat's CPU-only
  query path should fit in <500 MB peak for reasonable corpora.

## Hardware variance

Numbers are hardware-specific. The committed run is on Windows-11 +
Intel CPU (no CUDA). On a fresh clone the user will see different
absolute numbers but the **ratios** between approaches should hold:

- rlat warm-query latency competitive with sentence-transformers+faiss
- rlat on-disk size smaller than chroma (verified retrieval metadata
  is leaner than chroma's sqlite + WAL)
- chroma cold-start adds non-trivial overhead from the embedded
  database

## Reproducibility

```bash
pip install rlat[bench]
rlat install-encoder
python -m benchmarks.user_bench.build_query_speed.run \
  --output benchmarks/results/user_bench/build_query_speed.json
# Default grid: N ∈ {1000, 10000}
# Add --include-50k for the full 50K corpus tier (~30 min)
```

## Locked controls

- `--n-passages 1000 10000` — default grid for v1; opt-in 50K via
  `--include-50k`.
- `--n-queries 50` — 50 warm queries per approach. P95 is informative
  at N≥40; less at smaller N.
- Sentence-transformers loads `Alibaba-NLP/gte-modernbert-base` when
  available (matched-encoder comparison); falls back to
  `all-MiniLM-L6-v2` otherwise (different-encoder comparison; noted
  in `notes` field).
- Chroma uses its default ONNX embedder (`all-MiniLM-L6-v2`); we don't
  override because the point is "what does the user get from
  `pip install chromadb`".

## Honest framing

- **rlat is not always faster than sentence-transformers** at build
  time. On the test machine, ST built 1000 passages in 58 s vs
  rlat's 174 s — ST is ~3× faster on encode because PyTorch handles
  CPU batching more aggressively than OpenVINO on this CPU. This is
  a real finding; we publish it.
- **rlat wins on warm query latency** (19 ms p50 vs ST's 31 ms) and
  **on disk size** (2.7 MB vs 5.7 MB). Both matter for production —
  build is one-shot.
- **Chroma is harder to compare fairly** — different encoder
  (all-MiniLM-L6-v2 has 384d output vs rlat's 768d), different
  similarity calibration. Reported alongside as "the alternative
  bundle a user would `pip install`".
- **Pinecone is excluded by default** because it's network-bound and
  pricing-bound; users who want that comparison can opt in.

## Related work surfaced from prior measurement runs

- BENCHMARK_GATE.md: warm encode 12.1 ms (single query against the
  encoder), L2 norm error 1.19e-7 across runtimes (Phase 1 close).
  This bench's warm-query timing is encode + retrieve, so 19 ms p50
  ≈ 12 ms encode + 7 ms ANN search, consistent with the gate.
- Audit 04 (FAISS HNSW lock at M=32 efC=200 efS=128): rlat's
  efSearch=128 is sufficient for N≤50K with R@10 ≥0.95 vs exact.
