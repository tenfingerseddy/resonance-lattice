"""Round-trip — archive.read(defer_base_band) skips the base band ONLY when an
ANN index serves the query, and ANN retrieval still works without it.

Guards the lazy-band-skip OOM lever: the retrieval path (Fabric UDF) must avoid
holding the full (N,768) base band when FAISS answers the query, must still load
it for dense (no-ANN) corpora, and the default read must be unchanged.

Hermetic: synthetic band + a REAL (small) FAISS index over it — exercises the
deferred-read → ANN retrieve → hits path end to end, no encoder.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from resonance_lattice import field
from resonance_lattice.field import ann
from resonance_lattice.store import archive
from resonance_lattice.store import registry as registry_io
from resonance_lattice.store.metadata import BandInfo, Metadata


def _write(path: Path, *, with_ann: bool, n: int = 8, dim: int = 8) -> np.ndarray:
    band = np.random.default_rng(0).standard_normal((n, dim)).astype("float32")
    band /= np.linalg.norm(band, axis=1, keepdims=True)
    reg = [
        registry_io.PassageCoord(
            passage_idx=i, source_file=f"d{i}.txt", char_offset=0, char_length=5,
            content_hash="sha256:0", passage_id=registry_io.compute_id(f"d{i}.txt", 0, 5),
        )
        for i in range(n)
    ]
    meta = Metadata(bands={"base": BandInfo(role="retrieval_default", dim=dim, passage_count=n)})
    ann_blobs = {"base": ann.serialize(ann.build(band))} if with_ann else None
    archive.write(path, meta, {"base": band}, reg, ann_blobs=ann_blobs)
    return band


def run() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ann.rlat"
        band = _write(p, with_ann=True)

        # Default read → base band loaded (unchanged contract); also our query.
        c2 = archive.read(p)
        if "base" not in c2.bands or c2.select_band().band is None:
            print("[roundtrip] FAIL: default read must load the base band", file=sys.stderr)
            return 1
        query = band[0].copy()

        # Deferred read → base band skipped, ann_blob kept, handle.band None.
        c = archive.read(p, defer_base_band=True)
        if "base" in c.bands:
            print("[roundtrip] FAIL: defer did not skip base band (ANN present)", file=sys.stderr)
            return 1
        if "base" not in c.ann_blobs:
            print("[roundtrip] FAIL: ann_blob dropped on deferred read", file=sys.stderr)
            return 1
        h = c.select_band()
        if h.band is not None or h.ann_blob is None:
            print(f"[roundtrip] FAIL: deferred handle band={h.band is not None} "
                  f"ann={h.ann_blob is not None} (want band=None, ann present)", file=sys.stderr)
            return 1

        # The whole point: ANN retrieve works WITHOUT the band materialised.
        idx = ann.deserialize(h.ann_blob)
        hits = field.retrieve(query, h, idx, c.registry, top_k=3)
        if not hits:
            print("[roundtrip] FAIL: ANN retrieve on a deferred (band-less) read "
                  "returned no hits", file=sys.stderr)
            return 1

        # No ANN + defer → base band STILL loaded (dense retrieval needs it).
        p2 = Path(d) / "noann.rlat"
        _write(p2, with_ann=False)
        c3 = archive.read(p2, defer_base_band=True)
        if "base" not in c3.bands or c3.select_band().band is None:
            print("[roundtrip] FAIL: defer must NOT skip base when no ANN (dense needs it)",
                  file=sys.stderr)
            return 1

    print("[roundtrip] PASS — defer skips base only when ANN serves it; ANN retrieve OK band-less",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
