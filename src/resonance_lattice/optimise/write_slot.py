"""In-place optimised band write to a knowledge model.

After `train_mrl.train` returns the trained `W`:

  1. Project all base-band passages through `W` → `(N, d_native)`.
  2. L2-normalise the projected passages.
  3. Build a FAISS HNSW index over the optimised band (if N > threshold).
  4. Update `metadata.bands["optimised"]` with role, dim, w_shape, etc.
  5. Atomically write optimised + W + ANN bytes back to the .rlat via
     `archive.write_band_in_place`.

The whole step runs through the existing `archive.write_band_in_place`
atomic-write contract — a crash mid-write leaves the original archive
intact, the tmp ZIP is unlinked on exception. Other slots (base band,
source/, registry, build_config) round-trip unchanged.

Phase 4 deliverable. Base plan §4.3 steps 8-10, §4.6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..field import ann
from ..field._runtime_common import l2_normalize
from ..store import archive
from ..store.metadata import BandInfo


def project_and_write(
    km_path: str | Path,
    W: np.ndarray,
    nested_mrl_dims: tuple[int, ...] = (64, 128, 256, 512),
) -> None:
    """Project the base band through `W`, write the optimised slot atomically.

    `W` is shape `(d_native, backbone_dim)` per spec §4.6. The default
    `nested_mrl_dims` matches the values train_mrl.MRL_DIMS uses; pass
    explicitly if a future training run uses different nested slices.
    """
    km_path = Path(km_path)
    contents = archive.read(km_path)
    if "base" not in contents.bands:
        raise ValueError(
            f"{km_path} has no base band — optimised requires a base band "
            f"to project from. Did the build pipeline complete?"
        )
    base = contents.bands["base"]
    d_native, backbone_dim = W.shape
    if base.shape[1] != backbone_dim:
        raise ValueError(
            f"W shape {W.shape} doesn't match base band dim {base.shape[1]}. "
            f"Expected (d_native, {base.shape[1]})."
        )

    # Project passages through W and L2-normalise. The result is the
    # optimised band at d_native; smaller MRL dims are zero-copy slices
    # at query time per spec §4.6.
    optimised = np.ascontiguousarray(base @ W.T, dtype=np.float32)
    l2_normalize(optimised)

    # Optimised HNSW index built with the same threshold + params as the
    # base band. Threshold matches `field.ann.should_build_ann`.
    ann_blob: bytes | None = None
    if ann.should_build_ann(optimised.shape[0]):
        index = ann.build(optimised)
        ann_blob = ann.serialize(index)

    band_info = BandInfo(
        role="in_corpus_retrieval",
        dim=d_native,
        l2_norm=True,
        passage_count=optimised.shape[0],
        dim_native=d_native,
        w_shape=(d_native, backbone_dim),
        nested_mrl_dims=list(nested_mrl_dims),
        trained_from="bands/base.npz",
    )
    archive.write_band_in_place(
        km_path,
        band_name="optimised",
        band_info=band_info,
        band_data=optimised,
        projection=W.astype(np.float32, copy=False),
        ann_blob=ann_blob,
    )
