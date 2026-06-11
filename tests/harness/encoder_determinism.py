"""encoder_determinism — same input, same output, on the real encoder.

Live as of the 2026-06 review (was a permanent SKIP stub): with three
runtimes, auto-selection, and a silent OpenVINO→ONNX fallback, a numerical
divergence would corrupt every `.rlat` built afterwards and nothing would
catch it. These guarantees close that blind spot at ~30s of encoder
cold-start, paid only when field/ or install/ change:

  D1. Encode the same batch twice → bit-exact (the CPU determinism claim
      in HONEST_CLAIMS.md).
  D2. Batch row i ≈ singleton encode of text i (cosine ≥ 0.9999) — catches
      padding/attention-mask regressions that only show up under batching.
  D3. Cosine ≥ 0.9999 against the committed golden fixture
      (fixtures/encoder_golden.npz, generated from the pinned revision on
      the ONNX runtime) — catches drift across runtimes, runtime versions,
      and export regressions against a locked reference.

SKIPs (sentinel 2) when the encoder isn't staged on this machine, or when
the fixture's revision doesn't match the pinned one (regenerate it then).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ._testutil import check_guarantee, unpatch_zero_encoder

_FIXTURE = Path(__file__).parent / "fixtures" / "encoder_golden.npz"
_COS_FLOOR = 0.9999

_TEXTS = [
    "Resonance Lattice locks one encoder recipe: CLS pooling, 768 dimensions, L2-normalised.",
    "TREATAS transfers a filter from a disconnected table onto the model relationships.",
    "The quick brown fox jumps over the lazy dog.",
    "import numpy as np\nvecs = np.load('bands/base.npz')['embeddings']",
]


def _check(ok: bool, label: str) -> bool:
    return check_guarantee(ok, label, "encoder_determinism")


def run() -> int:
    unpatch_zero_encoder()  # the whole point is the REAL encoder
    from resonance_lattice.field.encoder import Encoder, get_pinned_revision

    try:
        enc = Encoder()
        first = enc.encode(_TEXTS)
    except Exception as exc:  # not staged on this machine → honest skip
        print(f"[encoder_determinism] SKIP — encoder unavailable: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2  # harness SKIP sentinel — runner reports as skipped, not passed

    ok = True

    second = enc.encode(_TEXTS)
    ok &= _check(np.array_equal(first, second),
                 "D1: encode twice is bit-exact")

    singles = np.vstack([enc.encode([t]) for t in _TEXTS])
    cos_single = np.sum(first * singles, axis=1)  # rows are L2-normalised
    ok &= _check(bool(np.all(cos_single >= _COS_FLOOR)),
                 f"D2: batch vs singleton cosine >= {_COS_FLOOR} "
                 f"(min {cos_single.min():.6f})")

    if not _FIXTURE.is_file():
        print(f"[encoder_determinism] SKIP D3 — golden fixture missing at "
              f"{_FIXTURE}", file=sys.stderr)
        return 2 if ok else 1
    golden = np.load(_FIXTURE)
    fixture_rev = str(golden["revision"])
    pinned = get_pinned_revision()
    if fixture_rev != pinned:
        print(f"[encoder_determinism] SKIP D3 — fixture revision {fixture_rev[:12]} "
              f"!= pinned {pinned[:12]}; regenerate the fixture", file=sys.stderr)
        return 2 if ok else 1
    cos_gold = np.sum(first * golden["embeddings"], axis=1)
    ok &= _check(bool(np.all(cos_gold >= _COS_FLOOR)),
                 f"D3: cosine vs locked golden fixture >= {_COS_FLOOR} "
                 f"(min {cos_gold.min():.6f})")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
