"""ONNX Runtime inference for gte-modernbert-base.

Default query-time path on non-Intel CPUs. 2-4× faster than PyTorch CPU.
Models are converted to .onnx by `rlat install-encoder`.

Phase 1 deliverable. Base plan §1.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _runtime_common as common

_INSTALL_HINT = "It is a base dependency of rlat — try `pip install --force-reinstall rlat`."

# Names match the standard HF→ONNX export for AutoModel and are validated
# against the loaded session's input list at load() time, so a future export
# tweak fails loudly at install/load rather than silently on a per-batch run.
_INPUT_IDS = "input_ids"
_ATTENTION_MASK = "attention_mask"


@dataclass
class _OnnxHandle:
    session: Any  # onnxruntime.InferenceSession


def load(model_path: Path) -> _OnnxHandle:
    common.require_asset(model_path, "ONNX model")
    ort = common.require_module("onnxruntime", _INSTALL_HINT)
    # Auto-discover providers — `[gpu]` extra installs onnxruntime-gpu, which
    # exposes CUDAExecutionProvider; the build path benefits from CUDA when
    # available. Query path on CPU-only installs gets only CPUExecutionProvider.
    session = ort.InferenceSession(str(model_path), providers=ort.get_available_providers())
    declared = {i.name for i in session.get_inputs()}
    expected = {_INPUT_IDS, _ATTENTION_MASK}
    missing = expected - declared
    if missing:
        raise RuntimeError(
            f"ONNX export at {model_path} is missing inputs {sorted(missing)}; "
            f"declared inputs are {sorted(declared)}. Re-run `rlat install-encoder`."
        )
    return _OnnxHandle(session=session)


def encode_batch(handle: _OnnxHandle, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Run the forward pass and return CLS embeddings (N, 768).

    L2 normalisation is applied by the caller (`Encoder.encode`).
    """
    outputs = handle.session.run(
        None,
        {_INPUT_IDS: input_ids, _ATTENTION_MASK: attention_mask},
    )
    return common.cls_pool(outputs[0])
