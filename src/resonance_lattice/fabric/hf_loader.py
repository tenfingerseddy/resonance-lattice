"""HuggingFace-backed encoder cache fetch (revision-pinned).

The encoder revision is read from `metadata.backbone.revision` on the
.rlat at bootstrap time, and the matching ONNX cache is pulled from a
public HF model repo. Maintainer publishes via
`scripts/publish_fabric_encoder_to_hf.py`.
"""

from __future__ import annotations

from pathlib import Path

from ..install.encoder import cache_dir as encoder_cache_dir

ENCODER_HF_REPO = "tenfingers/rlat-gte-modernbert-base-onnx"

# HF rejects 40-hex tag names (SHA URL collision); the `enc-` prefix
# keeps tags parseable as plain refs.
_ENCODER_TAG_PREFIX = "enc-"


def encoder_tag_for(revision: str) -> str:
    return f"{_ENCODER_TAG_PREFIX}{revision}"


def fetch_encoder_from_hf(revision: str) -> Path:
    """Download the ONNX encoder cache for `revision` if not present.

    No-op when `cache_root()/encoders/<revision>/model.onnx` already
    exists. `huggingface_hub` is imported lazily so this module can
    sit on the rlat-core import path without the optional dep.
    """
    dst = encoder_cache_dir(revision)
    if (dst / "model.onnx").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=ENCODER_HF_REPO,
        revision=encoder_tag_for(revision),
        local_dir=str(dst),
        local_dir_use_symlinks=False,
    )
    return dst
