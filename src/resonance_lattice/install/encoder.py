"""HF download → ONNX export → optional OpenVINO conversion.

Triggered automatically on first `rlat build` or `rlat optimise` if cache
is empty. Can also be invoked explicitly via `rlat install-encoder`.

Phase 1 deliverable.
"""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

from .._paths import xdg_cache_root as _xdg_cache_root
from ..field._runtime_common import is_intel_cpu, require_module

# Pinned at the gte-modernbert-base commit every v2.0 benchmark + smoke
# was validated against. Resolved from HF on 2026-04-26 (model
# last_modified 2025-07-04). The pin makes installs reproducible: a
# fresh box that runs `rlat install-encoder` gets the SAME embedding
# distribution that produced the locked BEIR-5 floor, regardless of
# whether HF's `main` has moved on.
#
# Bumping the pin is a deliberate act: validate the new revision against
# `benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json`, then
# update this constant + the tracking entry in REBUILD_PLAN.md.
PINNED_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"

MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
_DEFAULT_REVISION_SYMBOLIC = "main"

# Layout under cache_root():
#   <revision>/
#     ├── revision.txt           -- pinned HF commit hash (authoritative)
#     ├── tokenizer.json
#     ├── model.onnx
#     ├── openvino/  (Intel CPU + OV available)
#     └── torch/     (HF snapshot — used by torch_runtime + as ONNX export source)
_REVISION_FILE = "revision.txt"
_TORCH_SUBDIR = "torch"
_OV_SUBDIR = "openvino"
_OV_XML_FILENAME = "openvino_model.xml"
_ONNX_FILENAME = "model.onnx"

_HF_HUB_HINT = "It is a base dependency of rlat — try `pip install --force-reinstall rlat`."
_BUILD_HINT = "Install with `pip install rlat[build]`."


def cache_root() -> Path:
    """Resolve `<XDG_CACHE_HOME or ~/.cache>/rlat/encoders/`."""
    return _xdg_cache_root() / "encoders"


def cache_dir(revision: str) -> Path:
    return cache_root() / revision


_OV_BIN_FILENAME = _OV_XML_FILENAME.removesuffix(".xml") + ".bin"


def _required_artifacts(target: Path) -> list[Path]:
    """Files that must be present for an install to count as complete on this
    host. OpenVINO IR is conditional — only required if Intel + openvino are
    both available, mirroring the install path's conversion gate. The torch
    snapshot is always required because torch_runtime reads from it directly
    (build / optimise paths) and ONNX export reads from it too."""
    base = [
        target / _REVISION_FILE,
        target / "tokenizer.json",
        target / _ONNX_FILENAME,
        target / _TORCH_SUBDIR / "config.json",
    ]
    if is_intel_cpu() and importlib.util.find_spec("openvino") is not None:
        ov_dir = target / _OV_SUBDIR
        base.append(ov_dir / _OV_XML_FILENAME)
        base.append(ov_dir / _OV_BIN_FILENAME)
    return base


def _has_weights(target: Path) -> bool:
    """At least one safetensors shard present in the torch snapshot. Filename
    isn't a fixed string (single-file `model.safetensors` vs sharded
    `model-00001-of-00002.safetensors`), so we glob."""
    torch_dir = target / _TORCH_SUBDIR
    return torch_dir.exists() and any(torch_dir.glob("*.safetensors"))


def is_installed(revision: str | None = None) -> bool:
    """True if the cache holds every artefact this host needs for `revision`."""
    root = cache_root()
    if not root.exists():
        return False

    def complete(target: Path) -> bool:
        return all(p.exists() for p in _required_artifacts(target)) and _has_weights(target)

    if revision is not None:
        return complete(root / revision)
    return any(complete(d) for d in root.iterdir() if d.is_dir())


def get_pinned_revision() -> str:
    """Return the HF commit hash this install pinned the encoder at.

    Resolution order:
    1. `PINNED_REVISION` constant if set AND that revision is cached locally
       (the package-declared pin is the source of truth — same constant goes
        into knowledge-model metadata.json, so retrieval and storage agree).
    2. Most-recent-mtime among cached revisions (dev / preview installs only).

    Build paths that need a specific revision should pass it to
    `Encoder(revision=...)`.
    """
    root = cache_root()
    if not root.exists():
        raise RuntimeError(
            f"No encoder cache at {root}. Run `rlat install-encoder` first."
        )
    if PINNED_REVISION and (root / PINNED_REVISION / _REVISION_FILE).exists():
        return PINNED_REVISION
    candidates = [d for d in root.iterdir() if d.is_dir() and (d / _REVISION_FILE).exists()]
    if not candidates:
        raise RuntimeError(
            f"Encoder cache at {root} has no revision pinned. "
            "Run `rlat install-encoder` to populate it."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return (candidates[0] / _REVISION_FILE).read_text(encoding="utf-8").strip()


# ---------- install pipeline ----------

def _resolve_revision(symbolic: str) -> str:
    """Resolve a symbolic ref ("main" / branch / tag) to its concrete HF commit
    hash. Already-concrete 40-char hashes are returned unchanged. Short hex
    prefixes are not assumed to be hashes (a tag like `0123456789` would
    collide); they go through the API resolver."""
    if len(symbolic) == 40 and all(c in "0123456789abcdef" for c in symbolic.lower()):
        return symbolic
    hf_hub = require_module("huggingface_hub", _HF_HUB_HINT)
    info = hf_hub.HfApi().model_info(repo_id=MODEL_ID, revision=symbolic)
    return info.sha


def _download_snapshot(revision: str, target: Path) -> Path:
    hf_hub = require_module("huggingface_hub", _HF_HUB_HINT)
    return Path(hf_hub.snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        local_dir=str(target),
        allow_patterns=[
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "*.safetensors",
        ],
    ))


def _export_onnx(snapshot_dir: Path, onnx_out: Path) -> None:
    """Wrap an HF AutoModel so its forward returns last_hidden_state (a plain
    tensor that torch.onnx.export can trace), then export to ONNX."""
    torch = require_module("torch", _BUILD_HINT)
    transformers = require_module("transformers", _BUILD_HINT)

    base = transformers.AutoModel.from_pretrained(str(snapshot_dir))
    base.eval()

    class _HiddenStateWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state

    wrapper = _HiddenStateWrapper(base)
    dummy_ids = torch.zeros((1, 16), dtype=torch.long)
    dummy_mask = torch.ones((1, 16), dtype=torch.long)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(onnx_out),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
        # opset 18 (not 17): ModernBERT export emits aten_split which is a
        # domain-18 function in onnxscript's torch_lib; opset 17 fails to
        # inline it ("Opset mismatch when inlining function ... has version 17
        # in the model but version 18 in the function").
        opset_version=18,
        do_constant_folding=True,
    )


def _export_openvino(onnx_path: Path, ov_dir: Path) -> None:
    ov = require_module("openvino", _BUILD_HINT)
    model = ov.convert_model(str(onnx_path))
    ov_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(ov_dir / _OV_XML_FILENAME))


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` via tmp + os.replace so partial writes can't
    leave a half-written sentinel."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def install(revision: str | None = None, force: bool = False) -> Path:
    """Download HF weights at revision, convert ONNX, convert OpenVINO if Intel.

    Returns the cache directory path.

    revision: explicit HF commit hash, or None to use PINNED_REVISION (or
              "main" if PINNED_REVISION is unset). Symbolic refs are resolved
              to a concrete hash before caching.
    force: re-run download + conversions even if all artefacts are present.
    """
    symbolic = revision or PINNED_REVISION or _DEFAULT_REVISION_SYMBOLIC
    concrete = _resolve_revision(symbolic)
    target = cache_dir(concrete)
    if is_installed(concrete) and not force:
        return target

    target.mkdir(parents=True, exist_ok=True)
    # If revision.txt is missing the previous install attempt was interrupted —
    # treat existing conversion outputs as suspect and regenerate them all
    # rather than blessing a possibly-truncated model.onnx on the retry.
    interrupted = not (target / _REVISION_FILE).exists()
    snapshot = _download_snapshot(concrete, target / _TORCH_SUBDIR)

    # tokenizer.json at the revision root makes runtime tokenizer load O(1)
    # without re-walking the snapshot tree.
    tokenizer_dst = target / "tokenizer.json"
    if force or interrupted or not tokenizer_dst.exists():
        shutil.copy(snapshot / "tokenizer.json", tokenizer_dst)

    onnx_dst = target / _ONNX_FILENAME
    if force or interrupted or not onnx_dst.exists():
        _export_onnx(snapshot, onnx_dst)

    # OpenVINO IR is best-effort: only when an Intel CPU is detected AND the
    # openvino package is importable. Non-Intel hosts and CPU-only installs
    # silently skip it; encoder.py falls back to ONNX at load.
    ov_xml = target / _OV_SUBDIR / _OV_XML_FILENAME
    if is_intel_cpu() and importlib.util.find_spec("openvino") is not None:
        if force or interrupted or not ov_xml.exists():
            _export_openvino(onnx_dst, target / _OV_SUBDIR)

    _atomic_write_text(target / _REVISION_FILE, concrete)
    return target
