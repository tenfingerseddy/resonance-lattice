"""streaming — single-file `stream_topk` over a `.rlat` base band.

The OOM-safe slicer serve (`store.streaming`): stream the band one row-chunk at a
time straight out of the `.rlat`, never materialising the (N, D) matrix. These
guarantees pin the two things that can break in a streaming reader — chunk-boundary
correctness and row→key alignment — and the keyless (chunked-corpus) no-op.

  S1. Exact top-k vs a full-band in-RAM scan under the shared `(cosine, key)`
      descending ranking, with distinct controlled vectors, across a chunk boundary
      (`chunk_rows` < N): same keys, same order. (Pins the streaming MECHANICS —
      chunk reads + row→key alignment.)
  S4. The fast mmap serve (`materialize_band` → `topk_over_band`) is tie-exact with
      the full scan AND `stream_topk`, including an all-tie (constant-band) case.
  S2. `top_k >= N` returns every keyed row, still correctly ranked.
  S5. `SourceSnippets` reads a row's bundled description back by business key (the
      slice receipt), truncates over the cap, and returns None for a missing key.
  S3. A chunked (non-row-mode) corpus has no business keys → `stream_topk` yields [].
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus, check_guarantee, patch_zero_encoder


def _check(ok: bool, label: str) -> bool:
    return check_guarantee(ok, label, "streaming")


class DistinctEncoder:
    """Deterministic encoder emitting a distinct unit direction per row.

    The angle is read from an `ANGLE=<float>` token in the text, so the corpus
    and the query share one space and the cosine ranking is fully controlled:
    cosine(row, query@0) = cos(angle_row), descending in angle.
    """

    revision = "distinct-encoder-test"
    runtime_name = "distinct-encoder-test"

    def __init__(self, *a, **k) -> None:
        pass

    def _vec(self, text: str) -> np.ndarray:
        angle = 0.0
        for tok in text.split():
            if tok.startswith("ANGLE="):
                angle = float(tok[len("ANGLE="):])
                break
        v = np.zeros(768, dtype="float32")
        v[0] = math.cos(angle)
        v[1] = math.sin(angle)
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype("float32")

    def encode_batched(self, texts: list[str], batch_size: int = 0) -> np.ndarray:
        return self.encode(texts)


_ROWS = [
    ("L000", "ANGLE=0.00 alpha quiet"),
    ("L001", "ANGLE=0.20 bravo bright"),
    ("L002", "ANGLE=0.40 charlie central"),
    ("L003", "ANGLE=0.60 delta dim"),
    ("L004", "ANGLE=0.80 echo edge"),
]


def _build_rows(out: Path, enc: DistinctEncoder):
    from resonance_lattice.build.pipeline import build_rlat
    from resonance_lattice.build.walker import RowSourceWalker
    from resonance_lattice.config import Kind, StoreMode

    return build_rlat(
        RowSourceWalker(_ROWS, source_name="listings"), out,
        store_mode=StoreMode.BUNDLED, kind=Kind.CORPUS, encoder=enc,
        row_mode=True, batch_size=2,
    )


def _full_topk(km: Path, q: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Reference: full in-RAM scan with the same (cosine, key)-descending ranking."""
    from resonance_lattice.store import archive
    c = archive.read(km)
    band = np.ascontiguousarray(c.bands["base"], dtype="float32")
    keys = [str(c.registry[i].key) for i in range(len(c.registry))]
    sims = band @ q
    order = sorted(range(len(keys)), key=lambda i: (sims[i], keys[i]), reverse=True)[:k]
    return [(keys[i], float(sims[i])) for i in order]


def _s1_exact_across_chunks() -> bool:
    from resonance_lattice.store.streaming import stream_topk
    enc = DistinctEncoder()
    q = enc.encode(["ANGLE=0.00"])[0].astype("float32")
    with tempfile.TemporaryDirectory() as d:
        km = Path(d) / "km.rlat"
        _build_rows(km, enc)
        stream = stream_topk(km, q, 3, chunk_rows=2)  # 5 rows / 2 -> 3 chunks
        full = _full_topk(km, q, 3)
    ok = stream == full and [k for k, _ in stream] == ["L000", "L001", "L002"]
    if not ok:
        print(f"[streaming] FAIL S1: stream={stream} full={full}", file=sys.stderr)
    return _check(ok, "S1 (exact top-k across chunk boundary, distinct vectors)")


def _s2_topk_ge_n() -> bool:
    from resonance_lattice.store.streaming import stream_topk
    enc = DistinctEncoder()
    q = enc.encode(["ANGLE=0.00"])[0].astype("float32")
    with tempfile.TemporaryDirectory() as d:
        km = Path(d) / "km.rlat"
        _build_rows(km, enc)
        stream = stream_topk(km, q, 99, chunk_rows=2)
        full = _full_topk(km, q, 99)
    ok = stream == full and len(stream) == len(_ROWS)
    if not ok:
        print(f"[streaming] FAIL S2: stream={stream}", file=sys.stderr)
    return _check(ok, "S2 (top_k >= N returns all keyed rows, ranked)")


def _s4_mmap_serve_tie_exact() -> bool:
    """S4: the fast serve (materialize_band -> mmap -> topk_over_band) is TIE-EXACT
    with the full-scan reference (and thus with stream_topk). Includes an all-tie
    case (a constant band → every sim equal) so the cutoff-expansion tie handling
    is exercised, not just the distinct-vector path."""
    import numpy as _np
    from resonance_lattice.store.streaming import (
        materialize_band,
        read_keys,
        stream_topk,
        topk_over_band,
    )
    enc = DistinctEncoder()
    q = enc.encode(["ANGLE=0.00"])[0].astype("float32")
    with tempfile.TemporaryDirectory() as d:
        km = Path(d) / "km.rlat"
        _build_rows(km, enc)
        npy = Path(d) / "band.npy"
        materialize_band(km, npy)
        # load into RAM (not mmap) — the test only needs materialize_band's bytes to
        # be a valid .npy; mmap'ing would hold a handle that blocks Windows cleanup.
        band = _np.load(npy)
        keys = read_keys(km)
        mm = topk_over_band(band, keys, q, 3)
        full = _full_topk(km, q, 3)
        st = stream_topk(km, q, 3)
        # all-tie: zero query → every sim 0 → selection decided purely by key order
        qz = _np.zeros(768, dtype="float32")
        mm_tie = [k for k, _ in topk_over_band(band, keys, qz, 3)]
        full_tie = [k for k, _ in _full_topk(km, qz, 3)]
    # C1: keyless rows must NOT consume top-k slots — the slot backfills from the
    # next-best KEYED row (parity with stream_topk's skip-during-scan). Top-2 sims
    # here are keyless; a correct topk_over_band returns the keyed A, B (not []).
    kb_band = _np.array([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.7, 0.0]], dtype="float32")
    kb = [k for k, _ in topk_over_band(kb_band, [None, None, "A", "B"],
                                       _np.array([1.0, 0.0], dtype="float32"), 2)]
    ok = (mm == full and [k for k, _ in st] == [k for k, _ in full]
          and mm_tie == full_tie and kb == ["A", "B"])
    if not ok:
        print(f"[streaming] FAIL S4: mm={mm} full={full} tie(mm={mm_tie},full={full_tie}) "
              f"keyless_backfill={kb}", file=sys.stderr)
    return _check(ok, "S4 (mmap serve tie-exact + keyless backfill vs full/stream)")


def _s5_source_snippets() -> bool:
    """S5: SourceSnippets reads a row's bundled description back by its business key,
    truncates over the cap, and returns None for a missing key (so a non-bundled km
    or dropped row degrades to a keys-only hit instead of raising)."""
    from resonance_lattice.store.streaming import SourceSnippets
    enc = DistinctEncoder()
    with tempfile.TemporaryDirectory() as d:
        km = Path(d) / "km.rlat"
        _build_rows(km, enc)
        snip = SourceSnippets(km)
        try:
            full = snip.text("L000")
            last = snip.text("L004")
            missing = snip.text("NOPE-NOT-A-KEY")
            trunc = snip.text("L000", max_chars=5)
        finally:
            snip.close()  # release the handle so Windows can delete the tempdir
    ok = (full == "ANGLE=0.00 alpha quiet" and last == "ANGLE=0.80 echo edge"
          and missing is None
          and trunc is not None and trunc.endswith("…") and len(trunc) <= 6)
    if not ok:
        print(f"[streaming] FAIL S5: full={full!r} last={last!r} missing={missing!r} "
              f"trunc={trunc!r}", file=sys.stderr)
    return _check(ok, "S5 (SourceSnippets: key->text, truncation, missing->None)")


def _s3_keyless_returns_empty() -> bool:
    from resonance_lattice.store.streaming import stream_topk
    q = np.zeros(768, dtype="float32")
    q[0] = 1.0
    with tempfile.TemporaryDirectory() as d:
        km = build_corpus(Path(d) / "corpus",
                          {"a.md": "# A\n\nsome text here.", "b.md": "# B\n\nmore text here."})
        out = stream_topk(km, q, 5)
    ok = out == []
    if not ok:
        print(f"[streaming] FAIL S3: chunked corpus returned {out}", file=sys.stderr)
    return _check(ok, "S3 (chunked/keyless corpus yields no keys)")


def run() -> int:
    patch_zero_encoder()  # S3's build_corpus uses the global encoder; keep it cheap
    ok = True
    for chk in (_s1_exact_across_chunks, _s2_topk_ge_n, _s4_mmap_serve_tie_exact,
                _s5_source_snippets, _s3_keyless_returns_empty):
        ok = chk() and ok
    if ok:
        print("[streaming] all guarantees OK", file=sys.stderr)
        return 0
    print("[streaming] FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(run())
