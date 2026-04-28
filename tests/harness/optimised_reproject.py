"""optimised_reproject — refresh + sync re-project the optimised band correctly.

Three guarantees:

  1. After a refresh that re-encodes some passages, the optimised band
     is re-projected through the same W: `optimised = (new_base @ W.T)`
     row-wise L2-normalised.

  2. Re-projection preserves W (no retraining, no LLM, no GPU). The
     archive's `optimised_W.npz` is byte-identical pre/post refresh.

  3. `--discard-optimised` opt-out drops the optimised band from the
     archive (back to base-only). The W matrix is also dropped.

Audit 07 commit 7/8.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


from ._testutil import Args as _Args
from ._testutil import build_corpus as _build


def _attach_optimised(km: Path, d_native: int = 256) -> np.ndarray:
    """Synthesise an optimised band + W projection on a freshly built
    archive. Avoids invoking `rlat optimise` (which calls an LLM) — for
    the re-projection contract test we just need any consistent
    `(W, optimised_band)` pair where optimised = base @ W.T then L2.
    Returns the W matrix."""
    from resonance_lattice.store import archive
    from resonance_lattice.store.metadata import BandInfo

    contents = archive.read(km)
    base = contents.bands["base"]
    n, dim = base.shape
    rng = np.random.default_rng(0)
    # Cast to float32 AFTER the divide — `arr_f32 / np.sqrt(dim)` re-promotes
    # to float64 because np.sqrt returns float64; the archive serialises as
    # float32 so a fixture that built W as float64 would silently round-trip
    # to a different array on the read side and break byte-identical W
    # preservation tests.
    W = (rng.normal(size=(d_native, dim)) / np.sqrt(dim)).astype(np.float32)
    optimised = base @ W.T
    optimised = optimised / np.maximum(
        np.linalg.norm(optimised, axis=1, keepdims=True), 1e-12,
    )
    optimised = optimised.astype(np.float32, copy=False)

    contents.bands["optimised"] = optimised
    contents.projections["optimised"] = W
    contents.metadata.bands["optimised"] = BandInfo(
        role="in_corpus_retrieval", dim=d_native, l2_norm=True,
        passage_count=n, dim_native=d_native, w_shape=(d_native, dim),
        nested_mrl_dims=[64, 128, 256], trained_from="base",
    )
    archive.write(
        km,
        metadata=contents.metadata,
        bands=contents.bands,
        registry=contents.registry,
        projections=contents.projections,
    )
    return W


def _refresh(km: Path, *, discard_optimised: bool = False) -> int:
    from resonance_lattice.cli.maintain import cmd_refresh
    return cmd_refresh(_Args(
        knowledge_model=str(km),
        source=None, source_root=None, batch_size=4, ext=None,
        discard_optimised=discard_optimised, dry_run=False,
    ))


def _read(km: Path):
    from resonance_lattice.store import archive
    return archive.read(km)


def run() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        files = {
            "a.md": "# Alpha\n\nFirst doc about authentication and login flows. "
                    "Sessions persist for 24 hours by default.",
            "b.md": "# Beta\n\nSecond doc about credentials and tokens. "
                    "Tokens rotate weekly. Logout clears the session.",
            "c.md": "# Gamma\n\nThird doc about session storage in Redis. "
                    "Each session has a TTL.",
        }
        km = _build(root, files)
        W_initial = _attach_optimised(km, d_native=256)

        c0 = _read(km)
        if "optimised" not in c0.bands:
            print("[optimised_reproject] FAIL setup: optimised band not present",
                  file=sys.stderr)
            return 1

        # Edit one file → triggers re-encode + re-projection.
        (root / "b.md").write_text(
            "# Beta v2\n\nRewritten content about API keys, secret rotation, "
            "and revocation. The new flow forces a logout on every revoke.",
            encoding="utf-8",
        )

        # ---- Guarantee 1: optimised band re-projected from new base ----
        rc = _refresh(km)
        if rc != 0:
            print(f"[optimised_reproject] FAIL guarantee 1: refresh rc={rc}",
                  file=sys.stderr)
            return 1
        c1 = _read(km)
        if "optimised" not in c1.bands:
            print(f"[optimised_reproject] FAIL guarantee 1: optimised band "
                  f"missing after refresh", file=sys.stderr)
            return 1
        new_base = c1.bands["base"]
        new_optimised = c1.bands["optimised"]
        # Compute the expected re-projection independently and compare.
        W_loaded = c1.projections["optimised"]
        expected = new_base @ W_loaded.T
        expected = expected / np.maximum(
            np.linalg.norm(expected, axis=1, keepdims=True), 1e-12,
        )
        if not np.allclose(new_optimised, expected, atol=1e-6):
            max_diff = float(np.max(np.abs(new_optimised - expected)))
            print(f"[optimised_reproject] FAIL guarantee 1: optimised band "
                  f"is not (new_base @ W.T) L2-normalised; max_diff={max_diff}",
                  file=sys.stderr)
            return 1
        print("[optimised_reproject] guarantee 1 (re-projection correct) OK",
              file=sys.stderr)

        # ---- Guarantee 2: W is preserved byte-identically ----
        if not np.array_equal(W_loaded, W_initial):
            print(f"[optimised_reproject] FAIL guarantee 2: W changed during "
                  f"refresh — re-projection should never retrain.",
                  file=sys.stderr)
            return 1
        print("[optimised_reproject] guarantee 2 (W preserved) OK",
              file=sys.stderr)

        # ---- Guarantee 3: --discard-optimised drops the band ----
        # Edit again so refresh has work to do.
        (root / "a.md").write_text(
            "# Alpha v2\n\nRewritten alpha doc. Authentication moved to OAuth.",
            encoding="utf-8",
        )
        rc = _refresh(km, discard_optimised=True)
        if rc != 0:
            print(f"[optimised_reproject] FAIL guarantee 3: refresh rc={rc}",
                  file=sys.stderr)
            return 1
        c2 = _read(km)
        if "optimised" in c2.bands:
            print(f"[optimised_reproject] FAIL guarantee 3: optimised band "
                  f"still present after --discard-optimised", file=sys.stderr)
            return 1
        if "optimised" in c2.projections:
            print(f"[optimised_reproject] FAIL guarantee 3: W still present "
                  f"after --discard-optimised", file=sys.stderr)
            return 1
        if "optimised" in c2.metadata.bands:
            print(f"[optimised_reproject] FAIL guarantee 3: optimised "
                  f"BandInfo still in metadata", file=sys.stderr)
            return 1
        print("[optimised_reproject] guarantee 3 (--discard-optimised) OK",
              file=sys.stderr)

    print("[optimised_reproject] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
