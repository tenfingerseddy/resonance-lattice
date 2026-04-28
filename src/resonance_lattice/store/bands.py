"""NPZ band I/O — base + optional optimised slots.

Each band stored as a single NPZ entry under the key `"embeddings"` inside
the .rlat ZIP at `bands/<name>.npz`. The optimised band is paired with its
MRL projection matrix at `bands/optimised_W.npz` (single key
`"projection"`); both are loaded atomically by `load_optimised`.

NPZ keys (for external readers): `embeddings` for bands,
`projection` for the optimised W matrix. See
docs/internal/KNOWLEDGE_MODEL_FORMAT.md for the full layout.

Base band: (N, 768) L2-normalised float32.
Optimised band: (N, 512) L2-normalised float32. Smaller MRL dims (64 / 128 /
256) are zero-copy views via `embeddings[:, :k]` and `W[:k]`.

Write paths apply `_runtime_common.l2_normalize` defensively so a caller
that produced a slightly-off-norm tensor (e.g. mid-training snapshots)
doesn't silently store unnormalised vectors. Reads do NOT re-check the norm
to keep the load path fast.

Phase 2 deliverable. Base plan §2.2 + §4.6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..field._runtime_common import l2_normalize

if TYPE_CHECKING:
    from zipfile import ZipFile

_BAND_KEY = "embeddings"
_W_KEY = "projection"


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


def load_optimised(
    zf: "ZipFile",
    band_path: str = "bands/optimised.npz",
    w_path: str = "bands/optimised_W.npz",
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load (band, W) atomically.

    Returns (None, None) if BOTH slots are absent — the canonical
    "optimised not present" case. Raises `ValueError` if exactly one of the
    two is present (the archive is half-written and a silent (None, None)
    would mask genuine corruption). Callers branch on `band is None` to
    detect "optimised not present" without separate existence probes.
    """
    names = set(zf.namelist())
    has_band = band_path in names
    has_w = w_path in names
    if not has_band and not has_w:
        return None, None
    if has_band ^ has_w:
        raise ValueError(
            f"optimised slot is half-written: {band_path} present={has_band}, "
            f"{w_path} present={has_w}. Re-run `rlat optimise --force` to repair."
        )
    band = _load_array(zf, band_path, _BAND_KEY)
    w = _load_array(zf, w_path, _W_KEY)
    if band.ndim != 2:
        raise ValueError(f"optimised band at {band_path} has shape {band.shape}; expected (N, D)")
    if w.ndim != 2:
        raise ValueError(f"projection at {w_path} has shape {w.shape}; expected (d_native, d_backbone)")
    return band, w


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


def write_projection(zf: "ZipFile", w_path: str, w: np.ndarray) -> None:
    """Write the MRL optimised projection matrix `(d_native, d_backbone)`
    to `bands/optimised_W.npz`. Paired with `write_band(... bands/
    optimised.npz)`; readers expect both or neither."""
    arr = np.ascontiguousarray(w, dtype=np.float32)
    _write_npz(zf, w_path, **{_W_KEY: arr})
