"""OpenVINO Runtime inference for gte-modernbert-base.

Auto-detected query-time path on Intel CPUs. 1.5-2× faster than ONNX on Intel
hardware via AVX-512 + OpenMP. Models are converted to .xml/.bin by
`rlat install-encoder` after the ONNX export.

Phase 1 deliverable. Base plan §1.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _runtime_common as common

_INSTALL_HINT = "Install with `pip install rlat[build]`, or use the ONNX runtime path."


def is_available() -> bool:
    """True if OpenVINO Runtime is importable AND an Intel CPU is detected."""
    try:
        import openvino  # noqa: F401
    except ImportError:
        return False
    return common.is_intel_cpu()


@dataclass
class _OvHandle:
    compiled: Any  # openvino.CompiledModel
    request: Any  # openvino.InferRequest — reused per call (saves ~100-300us / query)


# Optimum's `optimum-cli export openvino` writes `openvino_model.xml`; an
# in-tree converter may emit `model.xml`. Accept either; encoder.py checks the
# same names when deciding whether to fall back to ONNX.
_XML_NAMES = ("openvino_model.xml", "model.xml")


def find_xml(model_dir: Path) -> Path | None:
    """Return the IR XML path inside `model_dir` if present, else None."""
    for name in _XML_NAMES:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def load(model_dir: Path) -> _OvHandle:
    """Compile the OpenVINO IR for CPU inference.

    `model_dir` contains either `openvino_model.xml`/.bin (Optimum convention)
    or `model.xml`/.bin. The single-thread infer request is created here and
    reused across `encode_batch` calls.
    """
    xml = find_xml(model_dir)
    if xml is None:
        raise RuntimeError(
            f"OpenVINO IR not found in {model_dir} "
            f"(looked for {' / '.join(_XML_NAMES)}). Run `rlat install-encoder`."
        )
    ov = common.require_module("openvino", _INSTALL_HINT)
    compiled = ov.Core().compile_model(str(xml), "CPU")
    return _OvHandle(compiled=compiled, request=compiled.create_infer_request())


def encode_batch(handle: _OvHandle, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Run the forward pass and return CLS embeddings (N, 768).

    L2 normalisation is applied by the caller (`Encoder.encode`).
    """
    result = handle.request.infer({"input_ids": input_ids, "attention_mask": attention_mask})
    out = next(iter(result.values()))
    return common.cls_pool(out)
