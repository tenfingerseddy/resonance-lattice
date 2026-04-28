"""Encoder installation — `rlat install-encoder`.

Downloads gte-modernbert-base at the pinned HF revision, converts to ONNX,
and (if Intel CPU detected) further to OpenVINO IR. Cached under
`~/.cache/rlat/encoders/<revision>/` (respects XDG_CACHE_HOME).

Phase 1 deliverable. Base plan §1.3.
"""
