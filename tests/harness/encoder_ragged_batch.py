"""Encoder ragged-batch — `encode_batched` survives a non-divisible total.

The ONNX runtime's memory-pattern allocator keys on the first call's input
shape. With dynamic batch and seq_len axes, a non-divisible total produces
a smaller final batch whose shape no longer matches the pre-planned
attention buffer, crashing the allocator on some Linux builds. The guard
in `field/onnx_runtime.py:load` disables `enable_mem_pattern`; this suite
exercises the ragged path so a future re-enable fails loudly.

SKIP if the encoder isn't staged locally — the harness runs before
`rlat install-encoder` in fresh clones.
"""

from __future__ import annotations

import sys

from ._testutil import check_guarantee

_PREFIX = "encoder_ragged_batch"

# Total is one-and-a-bit batches so the last call is smaller than the first —
# the shape transition that crashed the allocator under mem_pattern reuse.
BATCH = 32
TOTAL = BATCH + 2


def run() -> int:
    try:
        from resonance_lattice.field.encoder import Encoder, DIM
        from resonance_lattice.install import encoder as install_encoder
    except Exception as exc:  # noqa: BLE001
        print(f"[{_PREFIX}] SKIP — import: {exc}", file=sys.stderr)
        return 0

    rev_dir = install_encoder.cache_dir(install_encoder.get_pinned_revision())
    if not (rev_dir / "model.onnx").exists():
        print(f"[{_PREFIX}] SKIP — encoder not staged "
              "(run `rlat install-encoder` to enable)", file=sys.stderr)
        return 0

    enc = Encoder(runtime="onnx")
    # `i % 17` spreads seq_len across rows so the attention-buffer shape varies
    # alongside the batch dim — the second mem_pattern key.
    texts = [
        f"passage {i} " + "lorem ipsum dolor sit amet " * ((i % 17) + 1)
        for i in range(TOTAL)
    ]

    cases = [
        (texts, BATCH, "ragged tail at default batch"),
        (texts[:5], 1, "bs=1 floor"),
    ]
    failures = 0
    for batch_texts, batch_size, label in cases:
        out = enc.encode_batched(batch_texts, batch_size=batch_size)
        expected = (len(batch_texts), DIM)
        ok = out.shape == expected
        failures += not check_guarantee(
            ok, f"{label}: shape {out.shape} == {expected}", _PREFIX,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
