"""rlat install-encoder

HF download → ONNX export → optional OpenVINO conversion.
Most users never invoke this directly; first `rlat build` triggers it.
Useful for offline environments — pre-stage the cache.

Phase 1 deliverable.
"""

from __future__ import annotations

import argparse


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the install-encoder subparser onto the top-level dispatcher."""
    p = sub.add_parser("install-encoder", help="Download + convert gte-modernbert-base")
    p.add_argument(
        "--revision",
        default=None,
        help="HF commit hash, branch, or tag. Defaults to the package-pinned revision.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run download + conversion even if the cache is already populated.",
    )
    p.set_defaults(func=cmd_install_encoder)


def cmd_install_encoder(args: argparse.Namespace) -> int:
    from ..install import encoder as install_encoder

    print(f"Installing {install_encoder.MODEL_ID}...")
    target = install_encoder.install(
        revision=getattr(args, "revision", None),
        force=getattr(args, "force", False),
    )
    revision = (target / "revision.txt").read_text(encoding="utf-8").strip()
    print(f"Installed at {target}")
    print(f"  revision: {revision}")
    print(f"  onnx:     {target / 'model.onnx'}")
    ov_xml = target / "openvino" / "openvino_model.xml"
    if ov_xml.exists():
        print(f"  openvino: {ov_xml}")
    return 0
