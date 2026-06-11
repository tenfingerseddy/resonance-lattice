"""Streaming top-k over a `.rlat`'s base band — never materialises the (N, D) matrix.

The OOM-safe serve path for a large row-mode corpus in a memory-bounded host (the
Fabric UDF worker). Loading the band OR a FAISS index peaks at ~1.5 GB on the 94k
slicer corpus (the encoder floor ~542 MB + a full vector structure ~1 GB), which
exceeds the UDF worker ceiling — see `.claude/plans/slicer-reset-2026-06-07.md`.

This reads the band one row-chunk at a time **straight out of the `.rlat` ZIP** and
keeps only a top-k heap, so peak resident is the encoder floor + one chunk + the
heap (~a few MB above the encoder). It is an exact full scan — every row is scored,
no approximation — ranked by `(cosine, key)` descending (deterministic; ties broken
by key). **Single-file**: vectors come from `bands/base.npz` and keys from
`passages.jsonl`, both inside the one `.rlat` — no sidecar, no FAISS, no SQL.

`stream_topk` is the low-memory primitive (no `/tmp`, used by the harness + as a
fallback). `topk_over_band` + `materialize_band` are the FAST serve: decompress the
band once to an uncompressed `/tmp` `.npy`, `mmap` it, then every warm query is one
GEMV (~ms). Both share the `(cosine, key)`-descending ranking, so they agree.

Mechanics: the outer `.rlat` ZIP is `ZIP_STORED`, so the `bands/base.npz` member is
a seekable stream — an inner `ZipFile` can read its central directory. The band npy
inside is `np.savez_compressed` (DEFLATE), so its bytes are read sequentially and
sliced into row-chunks by the `.npy` row stride.
"""
from __future__ import annotations

import heapq
import json
import os
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as _npf
import zstandard as zstd

_BAND_MEMBER = "bands/base.npz"
_PASSAGES = "passages.jsonl"
_SOURCE_PREFIX = "source/"
# One ranking contract, shared by `stream_topk` (low-mem) and `topk_over_band`
# (fast mmap serve): raw cosine descending, ties broken by key ascending. Raw —
# NOT rounded — so the two agree exactly (rounding would merge a densely-packed
# tail into artificial ties that the two paths then resolve differently).
_DEFAULT_CHUNK_ROWS = 2048


_WRITE_CHUNK = 8 * 1024 * 1024


class SourceSnippets:
    """Row-mode source descriptions (the slicer's snippet receipt) read straight
    from a bundled `.rlat`, with the `ZipFile` held OPEN.

    Why held open: parsing the 94k-member central directory is ~200 ms, but reading
    + zstd-decompressing a member from an already-open handle is ~50 µs. So the
    serve layer opens this ONCE per km (cold) and `.text(key)` is ~ms for the
    handful of displayed hits — never a per-call re-open (~236 ms each, measured).

    Each `source/<key>` member is an individually zstd-framed UTF-8 blob (built by
    `store.bundled.pack_source_files`); for a row-mode km `source_file == key`, so
    the business key indexes the text directly. `.text(missing_key)` → None (also
    when the km isn't bundled, i.e. has no `source/`), so the caller degrades to a
    keys-only hit rather than raising. Close with `.close()` on eviction.

    Thread-safe: one instance is shared across concurrent calls (the serve layer
    caches it), but `ZipFile.read` + the zstd decompressor share a file cursor /
    internal state, so a lock serialises `.text()`. The reads are ~µs, so the lock
    is uncontended in practice (the band GEMV, which dominates, runs lock-free on the
    read-only mmap).
    """

    def __init__(self, rlat_path: str | Path) -> None:
        # Construct the cheap, ~never-throwing members BEFORE acquiring the file
        # handle, so a ZipFile open failure can't leave a leaked fd (the caller drops
        # the reference on any __init__ error). The ZipFile is the last step.
        self._dctx = zstd.ZstdDecompressor()
        self._lock = threading.Lock()
        self._zf: zipfile.ZipFile | None = None
        zf = zipfile.ZipFile(rlat_path)
        # Only HOLD the handle for a bundled km; a non-bundled km has no source/ and
        # would pin the ~60 MB central-directory parse for nothing (every .text()
        # would just return None). Close it now and short-circuit instead.
        if any(n.startswith(_SOURCE_PREFIX) for n in zf.namelist()):
            self._zf = zf
        else:
            zf.close()

    def text(self, key: str, *, max_chars: int = 240) -> str | None:
        if self._zf is None:
            return None  # non-bundled km — no source/ to read
        with self._lock:
            try:
                blob = self._zf.read(_SOURCE_PREFIX + str(key))
            except KeyError:
                return None  # this row's source was dropped
            except Exception:
                return None  # closed/corrupt handle — degrade to keys-only
            try:
                raw = self._dctx.decompress(blob).decode("utf-8", "replace").strip()
            except Exception:
                return None
        if len(raw) <= max_chars:
            return raw
        return raw[:max_chars].rstrip() + "…"

    def close(self) -> None:
        if self._zf is None:
            return
        try:
            self._zf.close()
        except Exception:
            pass


def read_keys(rlat_path: str | Path) -> list[str | None]:
    """Row-aligned business keys from a `.rlat`'s `passages.jsonl` (public wrapper).

    Cached once per worker by the serve layer — parsing 94k JSON lines is ~0.5 s,
    far too slow to repeat per query.
    """
    with zipfile.ZipFile(rlat_path) as zf:
        return _read_keys(zf)


def materialize_band(rlat_path: str | Path, dst_npy_path: str | Path) -> tuple[int, int]:
    """Stream-decompress the base band from the `.rlat` into an UNCOMPRESSED `.npy`
    at `dst_npy_path`, so the serve layer can `mmap` it (DEFLATE can't be mmap'd).

    Decompress-ONCE, on cold start: re-DEFLATE-ing the 290 MB band per query is the
    ~1.5 s (local) / ~6 s (worker) cost that makes the streaming serve too slow warm.
    Written here once; `np.load(..., mmap_mode="r")` then serves every warm query in
    a few ms. Streams the DEFLATE bytes chunk-by-chunk straight to the file, so the
    cold decompress never holds the full band in RAM. Returns (n_rows, dim).
    """
    dst = Path(dst_npy_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(rlat_path) as outer:
        with outer.open(_BAND_MEMBER) as npz_stream:
            with zipfile.ZipFile(npz_stream) as npz:
                with _open_band_npy(npz) as arr:
                    n_rows, dim, dtype = _read_npy_header(arr)
                    row_bytes = dim * dtype.itemsize
                    remaining = n_rows * row_bytes
                    # Per-writer-unique tmp so two concurrent cold materializes of
                    # the same km never share (and corrupt) one temp file.
                    tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
                    with open(tmp, "wb") as out:
                        _npf.write_array_header_1_0(out, {
                            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                            "fortran_order": False,
                            "shape": (int(n_rows), int(dim)),
                        })
                        while remaining > 0:
                            block = arr.read(min(_WRITE_CHUNK, remaining))
                            if not block:
                                break
                            out.write(block)
                            remaining -= len(block)
                    if remaining != 0:
                        tmp.unlink(missing_ok=True)
                        raise ValueError(f"base band truncated: {remaining} bytes short")
    tmp.replace(dst)  # atomic — a partial decompress never leaves a usable file
    return int(n_rows), int(dim)


def topk_over_band(
    band: np.ndarray,
    keys: list[str | None],
    query_vec: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """Cosine top-k over an in-memory or `mmap`'d band — vectorised, no Python row loop.

    The warm serve: one GEMV (`band @ q`, pages an mmap in from the OS cache) +
    `argpartition` for top-k, ~a few ms on 94k. Returns `[(key, score)]` descending,
    keyed rows only (keyless rows are dropped after selection). Deterministic: ties
    break by (score desc, key asc). `query_vec` must be L2-normalised (encoder output
    is); `top_k <= 0` → [].
    """
    if top_k <= 0 or band is None or len(band) == 0:
        return []
    q = np.ascontiguousarray(query_vec, dtype=np.float32).ravel()
    sims = np.asarray(band @ q, dtype=np.float32)
    n = sims.shape[0]
    # Exclude keyless rows from contention BEFORE selecting top-k, so the slot
    # backfills from the next-best KEYED row — parity with stream_topk, which skips
    # keyless rows during its scan. Masking after selection would shrink the result.
    if None in keys[:n]:
        sims = np.array(sims, dtype=np.float32, copy=True)
        for i in range(n):
            if keys[i] is None:
                sims[i] = -np.inf
    k = min(top_k, n)
    if k == n:
        cand = np.arange(n)
    else:
        # argpartition picks the top-k by sim alone, choosing arbitrarily among rows
        # tied at the cutoff (duplicate/templated listings share a vector → an exact
        # float tie). Expand the pool to ALL rows at-or-above the cutoff sim so the
        # (sim, key) sort below can't be robbed of a tied row, making this tie-exact
        # with stream_topk's heap + the full-scan reference.
        part = np.argpartition(-sims, k - 1)[:k]
        cutoff = float(sims[part].min())
        cand = np.flatnonzero(sims >= cutoff)
    # (sim, key) DESCENDING — same tie direction as stream_topk's heap (keeps the
    # largest (sim, key) tuples), so the two paths agree row-for-row.
    order = sorted(
        cand.tolist(),
        key=lambda i: (float(sims[i]), "" if keys[i] is None else str(keys[i])),
        reverse=True,
    )[:k]
    out: list[tuple[str, float]] = []
    for i in order:
        key = keys[i] if i < len(keys) else None
        if key is not None:
            out.append((str(key), float(sims[i])))
    return out


def _read_keys(zf: zipfile.ZipFile) -> list[str | None]:
    """Row-aligned business keys from `passages.jsonl` (line idx == passage idx).

    Streams line-by-line and retains only the small key strings — never holds the
    passage text. `key` is None for a chunked (non-row-mode) corpus.
    """
    keys: list[str | None] = []
    with zf.open(_PASSAGES) as f:
        for raw in f:  # ZipExtFile yields bytes lines, streaming
            line = raw.strip()
            if not line:
                continue
            keys.append(json.loads(line).get("key"))
    return keys


def _open_band_npy(npz: zipfile.ZipFile):
    """Return the streaming reader for the band's `.npy` member inside `base.npz`.

    `np.savez_compressed(f, embeddings=...)` stores the array as `embeddings.npy`;
    fall back to the first `.npy` member so a key rename can't silently break this.
    """
    names = npz.namelist()
    member = "embeddings.npy" if "embeddings.npy" in names else next(
        (n for n in names if n.endswith(".npy")), None)
    if member is None:
        raise ValueError(f"no .npy member in base.npz (members: {names})")
    return npz.open(member)


def _read_npy_header(arr) -> tuple[int, int, np.dtype]:
    """Parse the `.npy` header from a sequential stream; return (n_rows, dim, dtype).

    Uses numpy's version-dispatched header readers, then leaves `arr` positioned at
    the first array byte so the caller can read row-chunks directly.
    """
    major, minor = _npf.read_magic(arr)
    if (major, minor) == (1, 0):
        shape, fortran, dtype = _npf.read_array_header_1_0(arr)
    elif (major, minor) == (2, 0):
        shape, fortran, dtype = _npf.read_array_header_2_0(arr)
    else:  # numpy only emits 1.0/2.0 for a C-order float32 2-D band
        raise ValueError(f"unsupported .npy header version {(major, minor)} in base band")
    if fortran or len(shape) != 2:
        raise ValueError(f"base band must be C-order 2-D; got shape={shape} fortran={fortran}")
    return shape[0], shape[1], dtype


def stream_topk(
    rlat_path: str | Path,
    query_vec: np.ndarray,
    top_k: int,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
) -> list[tuple[str, float]]:
    """Cosine top-k over the base band, streamed from inside the `.rlat`.

    Returns `[(key, score), ...]` by `(cosine, key)` descending — keyed rows only (a
    row whose passage has no business `key` is skipped; chunked corpora yield
    nothing). `query_vec` must be L2-normalised in the band's space (encoder output
    already is); score is the raw cosine. Peak resident ≈ one `chunk_rows × dim`
    block + the heap. `top_k <= 0` returns [] (mirrors `field.dense.search`).
    """
    if top_k <= 0:
        return []
    q = np.ascontiguousarray(query_vec, dtype=np.float32).ravel()
    heap: list[tuple[float, str]] = []  # min-heap of (cosine, key); keeps top-k
    with zipfile.ZipFile(rlat_path) as outer:
        keys = _read_keys(outer)
        with outer.open(_BAND_MEMBER) as npz_stream:
            with zipfile.ZipFile(npz_stream) as npz:
                with _open_band_npy(npz) as arr:
                    n_rows, dim, dtype = _read_npy_header(arr)
                    if q.shape[0] != dim:
                        raise ValueError(f"query dim {q.shape[0]} != band dim {dim}")
                    row_bytes = dim * dtype.itemsize
                    idx = 0
                    while idx < n_rows:
                        n = min(chunk_rows, n_rows - idx)
                        raw = arr.read(n * row_bytes)
                        got = len(raw) // row_bytes
                        if got == 0:
                            break  # truncated band — stop rather than misread
                        block = np.frombuffer(raw, dtype=dtype, count=got * dim).reshape(got, dim)
                        sims = block @ q  # raw cosine; (sim, key) heap orders + ties
                        for j in range(got):
                            key = keys[idx + j] if (idx + j) < len(keys) else None
                            if key is None:
                                continue
                            item = (float(sims[j]), str(key))
                            if len(heap) < top_k:
                                heapq.heappush(heap, item)
                            elif item > heap[0]:
                                heapq.heapreplace(heap, item)
                        idx += got
    return [(key, score) for score, key in sorted(heap, reverse=True)]
