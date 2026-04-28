"""PyTorch inference for gte-modernbert-base.

Used at build time (encoding the corpus once) and at optimise time
(encoding synth queries + training MRL W). NOT the default query-time path
— ONNX/OpenVINO are 2-4× faster on CPU.

Requires the [build] or [optimise] install extras.

Phase 1 deliverable. Base plan §1.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import _runtime_common as common

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

_INSTALL_HINT = "Install with `pip install rlat[build]` (or `[optimise]`)."


@dataclass
class _TorchHandle:
    model: "PreTrainedModel"
    device: "torch.device"


def load(model_dir: Path) -> _TorchHandle:
    """Load the gte-mb model into eval-mode on the best available device.

    `model_dir` is the cache subdir holding the safetensors export.
    """
    common.require_asset(model_dir, "Torch model dir")
    torch = common.require_module("torch", _INSTALL_HINT)
    transformers = common.require_module("transformers", _INSTALL_HINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModel.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)
    return _TorchHandle(model=model, device=device)


def encode_batch(handle: _TorchHandle, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Run a forward pass and return CLS embeddings (N, 768) before L2-norm.

    Signature matches `onnx_runtime.encode_batch` and
    `openvino_runtime.encode_batch`.
    """
    import torch

    with torch.inference_mode():
        ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=handle.device)
        mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=handle.device)
        out = handle.model(input_ids=ids_t, attention_mask=mask_t)
        cls = out.last_hidden_state[:, 0, :]
    # On CPU `.cpu()` is a no-op but goes through dispatch; on CUDA it's the
    # required host copy. Skip the dispatch hop when already host-resident.
    return cls.numpy() if handle.device.type == "cpu" else cls.cpu().numpy()
