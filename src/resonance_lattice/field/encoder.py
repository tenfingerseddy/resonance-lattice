"""gte-modernbert-base encoder — single recipe.

Pinned at the HF revision recorded in `metadata.json`. CLS-pooled, L2-normalised,
output dim 768. Max sequence length 8192 tokens.

Inference runtime is auto-selected:
- Intel CPU detected → openvino_runtime
- Other CPU → onnx_runtime
- Build / optimise → torch_runtime (always)

Phase 1 deliverable. Base plan §1.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from ..install import encoder as install_encoder
from . import _runtime_common as common

# Locked constants — not configuration. MODEL_ID lives in `install.encoder`
# (the writer side); re-exported here as the canonical lookup symbol for
# anything that wants to read the encoder identity.
MODEL_ID = install_encoder.MODEL_ID
POOLING: Literal["cls"] = "cls"
DIM = 768
MAX_SEQ_LENGTH = 8192
L2_NORM = True

RuntimeName = Literal["auto", "onnx", "openvino", "torch"]
ConcreteRuntime = Literal["onnx", "openvino", "torch"]

# Each runtime module exposes `load(path) -> handle` and
# `encode_batch(handle, input_ids, attention_mask) -> CLS embeddings (N, DIM)`.
_RUNTIME_MODULES: dict[ConcreteRuntime, str] = {
    "onnx": "resonance_lattice.field.onnx_runtime",
    "openvino": "resonance_lattice.field.openvino_runtime",
    "torch": "resonance_lattice.field.torch_runtime",
}

EncodeBatchFn = Callable[[Any, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class _Runtime:
    handle: Any
    encode_batch: EncodeBatchFn


def get_pinned_revision() -> str:
    """Return the HF commit hash this install pinned the encoder at."""
    return install_encoder.get_pinned_revision()


def _select_runtime(requested: RuntimeName) -> ConcreteRuntime:
    """Resolve "auto" to a concrete runtime. "torch" is never auto-picked —
    build/optimise paths request it explicitly because it pulls in torch."""
    if requested != "auto":
        return requested
    # find_spec is faster than try-import (no side-effect import on miss) and
    # the cached spec is reused across calls within a process.
    if importlib.util.find_spec("openvino") is None:
        return "onnx"
    from . import openvino_runtime
    return "openvino" if openvino_runtime.is_available() else "onnx"


def _load_tokenizer(revision_dir: Path):
    tokenizers = common.require_module(
        "tokenizers",
        "It is a base dependency of rlat — try `pip install --force-reinstall rlat`.",
    )
    tok_path = revision_dir / "tokenizer.json"
    common.require_asset(tok_path, "Tokenizer")
    tok = tokenizers.Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=MAX_SEQ_LENGTH)
    tok.enable_padding()
    return tok


def _runtime_asset_path(runtime: ConcreteRuntime, revision_dir: Path) -> Path:
    return {
        "onnx": revision_dir / "model.onnx",
        "openvino": revision_dir / "openvino",
        "torch": revision_dir / "torch",
    }[runtime]


class Encoder:
    """Single-recipe gte-modernbert-base encoder.

    Construction is cheap; the runtime + tokenizer are lazy-loaded on first
    `encode()` call. Reuse one instance across calls — it caches state.
    """

    def __init__(self, runtime: RuntimeName = "auto", revision: str | None = None):
        self.revision = revision or get_pinned_revision()
        self.runtime_name: ConcreteRuntime = _select_runtime(runtime)
        self._tokenizer = None
        self._runtime: _Runtime | None = None

    def _ensure_loaded(self) -> None:
        if self._runtime is not None:
            return
        rev_dir = install_encoder.cache_dir(self.revision)
        # If auto resolved to openvino but the IR isn't actually staged in the
        # cache (e.g. offline pre-stage from a non-Intel host shipped only
        # model.onnx), fall back to onnx rather than crashing on load. We
        # check the XML files directly via the OV runtime helper since the
        # `openvino/` directory may exist while the IR pair does not.
        if self.runtime_name == "openvino":
            from . import openvino_runtime
            ov_dir = _runtime_asset_path("openvino", rev_dir)
            if not (ov_dir.exists() and openvino_runtime.find_xml(ov_dir) is not None):
                self.runtime_name = "onnx"
        self._tokenizer = _load_tokenizer(rev_dir)
        rt_module = importlib.import_module(_RUNTIME_MODULES[self.runtime_name])
        handle = rt_module.load(_runtime_asset_path(self.runtime_name, rev_dir))
        self._runtime = _Runtime(handle=handle, encode_batch=rt_module.encode_batch)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode passages or queries to (N, 768) L2-normalised float32.

        CLS pooling is performed inside the runtime export.
        """
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        self._ensure_loaded()
        assert self._runtime is not None and self._tokenizer is not None
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        cls = self._runtime.encode_batch(self._runtime.handle, input_ids, attention_mask)
        cls = np.ascontiguousarray(cls, dtype=np.float32)
        if L2_NORM:
            common.l2_normalize(cls)
        return cls

    def encode_batched(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode in fixed-size batches. Concatenates the per-batch outputs.

        Used by `cli/build.py` and `store/incremental.apply_delta` so a
        large encode (e.g. 50K passages) doesn't blow runtime peak memory
        in the tokenizer or onnx/openvino session — both of which allocate
        an `(input_ids, attention_mask)` pair sized to the input list.
        """
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            out.append(self.encode(texts[start:start + batch_size]))
        return np.concatenate(out, axis=0)


_default_encoder: Encoder | None = None


def encode(texts: list[str], runtime: RuntimeName = "auto") -> np.ndarray:
    """Module-level convenience that reuses a singleton Encoder.

    Pass `runtime` only when you need to pin it explicitly (e.g. build paths
    that always want torch). Query paths should use the default.
    """
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = Encoder(runtime=runtime)
        return _default_encoder.encode(texts)
    # Cached encoder reused if the requested runtime resolves to the same
    # concrete runtime. Re-resolving "auto" does no work after first call —
    # find_spec / is_available are cached.
    if _select_runtime(runtime) != _default_encoder.runtime_name:
        _default_encoder = Encoder(runtime=runtime)
    return _default_encoder.encode(texts)
