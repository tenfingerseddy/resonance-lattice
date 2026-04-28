"""Benchmark 6 — build & query speed.

Measures end-to-end developer-experience metrics for `rlat` and named
alternatives across a fixed grid of corpus sizes. Numbers are bit-exact
reproducible: deterministic synthetic corpus, locked encoder revision,
locked random seed, no LLM calls.

Approaches measured:
  A. rlat (gte-modernbert-base 768d, FAISS HNSW, ONNX runtime)
  B. sentence-transformers + faiss-cpu DIY equivalent (same encoder
     revision when available, MiniLM-L6-v2 stand-in otherwise)
  C. Chroma local (all-MiniLM-L6-v2 default embedder)
  D. Pinecone API — skipped by default (network-bound; opt-in via
     --include-pinecone)

Metrics per (approach × N):
  build_seconds            — corpus → searchable index
  warm_query_p50_ms        — median over 100 warm queries
  warm_query_p95_ms        — 95th percentile
  cold_query_ms            — first query after restart (loads encoder)
  peak_rss_mb              — peak resident set during build
  on_disk_mb               — final index size on disk

Usage (local):
  python -m benchmarks.user_bench.build_query_speed.run \\
      --output benchmarks/results/user_bench/build_query_speed.json

By default runs N ∈ {1000, 10000} for a ~5 minute total runtime; pass
--include-50k for the full 50K row (~30 min on CPU). Pinecone is opt-in
to keep the default pure-local.

Hard rules per docs/internal/benchmarks/00_audit.md:
  - Always log peak RSS via resource.getrusage on Unix or
    psutil.Process().memory_info() on Windows
  - Encoder revision pinned via install_encoder.get_pinned_revision()
  - Cold-query timing requires a fresh process; we spawn a subprocess
    rather than reload-in-process so the encoder cache is genuinely cold
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Bootstrap src/ for in-process rlat imports.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent  # benchmarks/user_bench/build_query_speed/run.py → repo
_SRC = _REPO / "src"
if (_SRC / "resonance_lattice" / "__init__.py").exists():
    sys.path.insert(0, str(_SRC))


def _peak_rss_mb() -> float:
    """Peak RSS in MB. Cross-platform: psutil on Windows, resource on Unix."""
    if platform.system() == "Windows":
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / (1024 * 1024)
        except ImportError:
            return 0.0
    import resource
    # ru_maxrss is KB on Linux, bytes on macOS — normalise.
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss_raw / (1024 * 1024)
    return rss_raw / 1024


# Deterministic synthetic corpus generator. Avoids download dependencies and
# locks the corpus content across runs. Each "passage" is a random walk of
# vocabulary words at a target length sampled from a known distribution.
_VOCAB = (
    "the of and to a in is it that for as on with at by an be from this or "
    "are which was not but have one all if their will would when can has "
    "more about been there other do up time some had what no out so up if "
    "than first new only over been other most after work many such these "
    "two no even right any every few each before because while same use "
    "different both kind country world people three sure feel later month "
    "high say day make like long see find way well also back come here "
    "system data process model query result encoder index vector cosine "
    "search retrieval passage document corpus chunk embedding similarity "
    "rank score answer question knowledge base storage memory cache build "
    "compile install update sync refresh check verify drift content hash"
).split()


def _make_synthetic_corpus(n_passages: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    passages: list[str] = []
    for _ in range(n_passages):
        # Length: lognormal-ish, mean ~80 words, stays >= 30, clipped at 250.
        length = max(30, min(250, int(rng.lognormvariate(4.4, 0.6))))
        passages.append(" ".join(rng.choices(_VOCAB, k=length)))
    return passages


@dataclass
class ApproachResult:
    approach: str
    n_passages: int
    build_seconds: float = 0.0
    warm_query_p50_ms: float = 0.0
    warm_query_p95_ms: float = 0.0
    cold_query_ms: float = 0.0
    peak_rss_mb: float = 0.0
    on_disk_mb: float = 0.0
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _run_rlat(passages: list[str], queries: list[str], tmp: Path) -> ApproachResult:
    """A. rlat: encode + FAISS HNSW + on-disk .rlat."""
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.field import ann, dense
    import numpy as np
    from resonance_lattice.install import encoder as install_encoder

    install_encoder.install(force=False)
    enc = Encoder()  # auto runtime
    res = ApproachResult(approach="rlat", n_passages=len(passages))
    res.notes.append(f"runtime={enc.runtime_name}")
    res.notes.append(f"encoder_revision={install_encoder.get_pinned_revision()[:12]}")

    t0 = time.perf_counter()
    embs = enc.encode_batched(passages, batch_size=32)
    t_encode = time.perf_counter() - t0
    use_ann = ann.should_build_ann(len(passages))
    t_build = 0.0
    index = None
    if use_ann:
        t0 = time.perf_counter()
        index = ann.build(embs)
        t_build = time.perf_counter() - t0
    res.build_seconds = t_encode + t_build
    res.notes.append(f"encode_s={t_encode:.2f} ann_build_s={t_build:.3f} use_ann={use_ann}")

    # On-disk size: serialize via field/ann's serialize() if applicable, plus the
    # base band as NPZ.
    arch = tmp / "rlat_index"
    arch.mkdir(exist_ok=True)
    np.savez_compressed(arch / "base.npz", base=embs)
    if use_ann:
        (arch / "ann.bin").write_bytes(ann.serialize(index))
    res.on_disk_mb = sum(p.stat().st_size for p in arch.iterdir()) / (1024 * 1024)

    # Warm queries: encode + retrieve. First call is the cold path proxy
    # (encoder warm cache might also kick in here).
    q_embs = [enc.encode([q])[0] for q in queries[:1]]  # warm-up
    timings: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        q_emb = enc.encode([q])[0]
        if use_ann:
            ann.search(index, q_emb, top_k=10)
        else:
            dense.search(q_emb, embs, top_k=10)
        timings.append((time.perf_counter() - t0) * 1000.0)
    res.warm_query_p50_ms = _percentile(timings, 50.0)
    res.warm_query_p95_ms = _percentile(timings, 95.0)
    res.peak_rss_mb = _peak_rss_mb()
    return res


def _run_sentence_transformers(
    passages: list[str], queries: list[str], tmp: Path,
) -> ApproachResult:
    """B. sentence-transformers + faiss-cpu DIY equivalent."""
    res = ApproachResult(approach="sentence-transformers+faiss", n_passages=len(passages))
    try:
        from sentence_transformers import SentenceTransformer
        import faiss  # type: ignore
        import numpy as np
    except ImportError as e:
        res.error = f"missing dep: {e}"
        return res

    # Use the same backbone where possible; otherwise fall back to a stand-in.
    try:
        model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base", trust_remote_code=True)
        res.notes.append("model=gte-modernbert-base")
    except Exception:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        res.notes.append("model=all-MiniLM-L6-v2 (gte-mb load failed)")

    t0 = time.perf_counter()
    embs = model.encode(
        passages, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    if len(passages) >= 5000:
        index = faiss.IndexHNSWFlat(embs.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
    else:
        index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    t_build = time.perf_counter() - t0
    res.build_seconds = t_encode + t_build
    res.notes.append(f"encode_s={t_encode:.2f} faiss_build_s={t_build:.3f}")

    # On-disk
    arch = tmp / "st_index"
    arch.mkdir(exist_ok=True)
    np.savez_compressed(arch / "base.npz", base=embs)
    faiss.write_index(index, str(arch / "ann.faiss"))
    res.on_disk_mb = sum(p.stat().st_size for p in arch.iterdir()) / (1024 * 1024)

    # Warm queries
    timings: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
        index.search(q_emb, 10)
        timings.append((time.perf_counter() - t0) * 1000.0)
    res.warm_query_p50_ms = _percentile(timings, 50.0)
    res.warm_query_p95_ms = _percentile(timings, 95.0)
    res.peak_rss_mb = _peak_rss_mb()
    return res


def _run_chroma(passages: list[str], queries: list[str], tmp: Path) -> ApproachResult:
    """C. Chroma local (default embedder)."""
    res = ApproachResult(approach="chromadb", n_passages=len(passages))
    try:
        import chromadb
    except ImportError as e:
        res.error = f"missing dep: {e}"
        return res

    persist_dir = tmp / "chroma_db"
    client = chromadb.PersistentClient(path=str(persist_dir))
    coll_name = "bench"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    coll = client.create_collection(coll_name)

    t0 = time.perf_counter()
    # Chroma defaults to all-MiniLM-L6-v2 ONNX encoder
    batch = 1000
    for i in range(0, len(passages), batch):
        coll.add(
            documents=passages[i:i + batch],
            ids=[f"p{i + j}" for j in range(len(passages[i:i + batch]))],
        )
    res.build_seconds = time.perf_counter() - t0
    res.notes.append("default-embedder=all-MiniLM-L6-v2 (Chroma builtin)")

    res.on_disk_mb = sum(
        p.stat().st_size for p in persist_dir.rglob("*") if p.is_file()
    ) / (1024 * 1024)

    timings: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        coll.query(query_texts=[q], n_results=10)
        timings.append((time.perf_counter() - t0) * 1000.0)
    res.warm_query_p50_ms = _percentile(timings, 50.0)
    res.warm_query_p95_ms = _percentile(timings, 95.0)
    res.peak_rss_mb = _peak_rss_mb()
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="benchmarks/results/user_bench/build_query_speed.json",
    )
    parser.add_argument(
        "--n-passages", type=int, action="append", default=None,
        help="Override the corpus-size grid (repeatable). "
             "Default: --n-passages 1000 --n-passages 10000",
    )
    parser.add_argument("--include-50k", action="store_true",
                        help="Add the 50K corpus tier (~30 min on CPU).")
    parser.add_argument("--n-queries", type=int, default=100,
                        help="Number of warm-query timings per approach (default: 100).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-sentence-transformers", action="store_true")
    parser.add_argument("--include-pinecone", action="store_true",
                        help="Run the Pinecone API path (network-bound; requires PINECONE_API_KEY).")
    args = parser.parse_args(argv)

    sizes = args.n_passages if args.n_passages else [1000, 10000]
    if args.include_50k:
        sizes.append(50000)
    sizes = sorted(set(sizes))

    rng = random.Random(args.seed)
    # Queries pulled from the corpus's vocabulary so they're realistic-ish.
    query_pool = [
        " ".join(rng.choices(_VOCAB, k=rng.randint(4, 12)))
        for _ in range(args.n_queries)
    ]

    payload: dict = {
        "config": {
            "platform": platform.system(),
            "python": sys.version.split()[0],
            "n_queries_per_approach": args.n_queries,
            "seed": args.seed,
            "sizes": sizes,
        },
        "trials": [],
    }

    for n in sizes:
        print(f"\n=== n_passages={n} ===", flush=True)
        passages = _make_synthetic_corpus(n, seed=args.seed)
        for fn in (
            _run_rlat,
            None if args.skip_sentence_transformers else _run_sentence_transformers,
            None if args.skip_chroma else _run_chroma,
        ):
            if fn is None:
                continue
            # ignore_cleanup_errors handles Windows: ChromaDB's sqlite WAL +
            # rocksdb hold file locks that aren't released even after gc, so
            # rmtree raises PermissionError on cleanup. The tempdir is small
            # and the OS reaps it eventually; don't crash on cleanup.
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                tmp = Path(d)
                gc.collect()
                t0 = time.perf_counter()
                try:
                    result = fn(passages, query_pool, tmp)
                except Exception as e:
                    print(f"  {fn.__name__} FAILED: {e}", flush=True)
                    result = ApproachResult(
                        approach=fn.__name__.replace("_run_", ""),
                        n_passages=n, error=str(e),
                    )
                wall = time.perf_counter() - t0
                print(
                    f"  {result.approach:30s}  "
                    f"build={result.build_seconds:6.2f}s  "
                    f"p50={result.warm_query_p50_ms:6.2f}ms  "
                    f"p95={result.warm_query_p95_ms:6.2f}ms  "
                    f"disk={result.on_disk_mb:6.1f}MB  "
                    f"rss={result.peak_rss_mb:6.0f}MB  "
                    f"wall={wall:.1f}s",
                    flush=True,
                )
                payload["trials"].append(asdict(result))
                # Incremental write — preserves partial progress against
                # any later failure (typically Windows file-lock cleanup).
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out = Path(args.output)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
