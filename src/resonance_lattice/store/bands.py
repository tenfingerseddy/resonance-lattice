"""NPZ band I/O — the base band.

Each band stored as a single NPZ entry under the key `"embeddings"` inside
the .rlat ZIP at `bands/<name>.npz`.

NPZ keys (for external readers): `embeddings` for bands. See
docs/internal/KNOWLEDGE_MODEL_FORMAT.md for the full layout.

Base band: (N, 768) L2-normalised float32.

Write paths apply `_runtime_common.l2_normalize` defensively so a caller
that produced a slightly-off-norm tensor doesn't silently store unnormalised
vectors. Reads do NOT re-check the norm to keep the load path fast.

Phase 2 deliverable. Base plan §2.2 + §4.6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..field._runtime_common import l2_normalize

if TYPE_CHECKING:
    from zipfile import ZipFile

_BAND_KEY = "embeddings"


def _load_array(zf: "ZipFile", path: str, key: str) -> np.ndarray:
    """Stream an NPZ entry from the ZIP and return a contiguous float32 view.

    Streams via `zf.open(path)` directly into `np.load` rather than reading
    the whole NPZ blob into a `bytes` buffer first — saves a memcpy of the
    band size (~150 MB for a 50K-passage corpus at 768d float32).
    """
    with zf.open(path) as f:
        npz = np.load(f)
        return np.ascontiguousarray(npz[key], dtype=np.float32)


def _write_npz(zf: "ZipFile", path: str, **arrays: np.ndarray) -> None:
    """Stream a compressed NPZ payload directly into a ZIP entry. Mirror of
    `_load_array` on the write side — avoids a `BytesIO` round-trip of the
    band size. The format spec (`KNOWLEDGE_MODEL_FORMAT.md`) calls for
    NPZ-internal deflate while the outer ZIP stays uncompressed; this is
    where that contract is enforced.
    """
    with zf.open(path, "w") as f:
        np.savez_compressed(f, **arrays)


def load_base(zf: "ZipFile", band_path: str = "bands/base.npz") -> np.ndarray:
    """Load (N, 768) base band from an open ZipFile. Returns float32."""
    arr = _load_array(zf, band_path, _BAND_KEY)
    if arr.ndim != 2:
        raise ValueError(f"base band at {band_path} has shape {arr.shape}; expected (N, D)")
    return arr


def write_band(zf: "ZipFile", band_path: str, embeddings: np.ndarray) -> None:
    """Write (N, D) L2-normalised embeddings to a band slot.

    `l2_normalize` is in-place; we copy so this function never mutates the
    caller's array. Encoder output is already normalised, so the renorm is
    typically a no-op — kept defensive because mid-training snapshots and
    hand-constructed bands aren't guaranteed unit-norm. The NPZ payload is
    deflate-compressed (`np.savez_compressed`) per the format spec while
    the outer ZIP entry stays uncompressed.
    """
    arr = np.ascontiguousarray(embeddings, dtype=np.float32).copy()
    l2_normalize(arr)
    _write_npz(zf, band_path, **{_BAND_KEY: arr})
