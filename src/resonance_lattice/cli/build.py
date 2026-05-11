"""`rlat build <source...> -o knowledge_model.rlat [--store-mode bundled|local|remote] [--kind corpus|intent]`

CLI wrapper around `resonance_lattice.build.build_rlat`. Owns argparse +
the stdout banner; the pipeline lives under `resonance_lattice.build` so
the Fabric UDF, notebook automation, and embedded library callers can
reuse it.

Single recipe — no encoder/precision/sparsify/field-type knobs. The
`--kind` flag tags the model as `corpus` (default) or `intent`; v2.0
ships the tag only, intent operators are deferred.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..build.pipeline import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    BuildError,
    build_rlat,
)
from ..build.walker import FilesystemSourceWalker, SourceWalker, common_root
from ..config import Kind, StoreMode


def _print_skipped(walker: SourceWalker, label: str) -> None:
    """Reasons → counts summary so the line stays short even when many
    files were skipped (binary blobs in a mixed corpus, etc.)."""
    skipped = walker.skipped
    if not skipped:
        return
    reasons: dict[str, int] = {}
    for _, reason in skipped:
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"[{label}] skipped {len(skipped)} files: "
          + ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())))


def cmd_build(args: argparse.Namespace) -> int:
    sources = [Path(p) for p in args.sources]
    if not sources:
        print("error: rlat build requires at least one source path")
        return 2

    try:
        store_mode = StoreMode(args.store_mode)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if store_mode is StoreMode.REMOTE and not args.remote_url_base:
        print(
            "error: --store-mode remote requires --remote-url-base <url-prefix>; "
            "e.g. --remote-url-base https://example.com/corpus/v1",
            file=sys.stderr,
        )
        return 2
    if args.remote_url_base and store_mode is not StoreMode.REMOTE:
        print(
            f"error: --remote-url-base only applies to --store-mode remote "
            f"(current: {store_mode.value})",
            file=sys.stderr,
        )
        return 2

    source_root = Path(args.source_root) if args.source_root else common_root(sources)
    extensions = (
        None
        if not args.ext
        else frozenset(("." + e.lstrip(".")).lower() for e in args.ext)
    )
    walker = FilesystemSourceWalker(sources, source_root, extensions)

    print(f"[build] walking sources rooted at {source_root}")
    try:
        result = build_rlat(
            walker,
            Path(args.output),
            store_mode=store_mode,
            kind=Kind(args.kind),
            runtime=getattr(args, "runtime", "auto") or "auto",
            batch_size=args.batch_size,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            remote_url_base=args.remote_url_base,
        )
    except BuildError as exc:
        _print_skipped(walker, "build")
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_skipped(walker, "build")
    out_size_mb = result.output_path.stat().st_size / (1024 * 1024)
    print(f"[build] wrote {result.output_path} ({out_size_mb:.2f} MB, "
          f"{result.n_passages} passages from {result.n_files} files)")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Register `rlat build` flags on the top-level dispatcher."""
    p = sub.add_parser("build", help="Build a knowledge model from source dirs")
    p.add_argument("sources", nargs="+", help="One or more source files or directories")
    p.add_argument("-o", "--output", required=True, help="Output .rlat path")
    p.add_argument("--store-mode", default=StoreMode.LOCAL.value,
                   choices=[m.value for m in StoreMode],
                   help="How to resolve source files at query time (default: local). "
                        "remote requires --remote-url-base.")
    p.add_argument("--remote-url-base", default=None,
                   help="URL prefix joined with each source_file relative path "
                        "to produce the upstream URL (required when "
                        "--store-mode remote). Example: "
                        "https://example.com/corpus/v1")
    p.add_argument("--kind", default=Kind.CORPUS.value,
                   choices=[k.value for k in Kind],
                   help="Knowledge-model kind tag (default: corpus)")
    p.add_argument("--source-root", default=None,
                   help="Root for source_file paths (default: common ancestor of sources)")
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                   help=f"Chunker min size (default: {DEFAULT_MIN_CHARS})")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                   help=f"Chunker max size (default: {DEFAULT_MAX_CHARS})")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Encoder batch size (default: 32)")
    p.add_argument("--runtime", default="auto",
                   choices=["auto", "openvino", "onnx", "torch"],
                   help="Encoder runtime (default: auto). auto = OpenVINO on "
                        "Intel CPUs, ONNX otherwise; torch is the slowest "
                        "fallback. Pass --runtime torch only when you need "
                        "the canonical build path on a non-Intel CPU.")
    p.add_argument("--ext", action="append", default=None,
                   help="Source file extension to include (repeatable; "
                        "default: built-in text-file allowlist)")
    p.set_defaults(func=cmd_build)
