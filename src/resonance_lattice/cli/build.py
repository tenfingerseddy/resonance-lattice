"""`rlat build <source...> -o knowledge_model.rlat [--store-mode bundled|local] [--kind corpus|intent]`

Builds a knowledge model end-to-end:

  1. Walk source paths (files or directories), collect text files.
  2. Chunk each file with `store.chunker.chunk_text`.
  3. Encode passages with the gte-modernbert-base encoder (torch runtime —
     build path always pulls torch via `[build]` extras).
  4. Build registry (`PassageCoord` per passage with sha256 content hash).
  5. Build FAISS HNSW ANN index over the base band when N > threshold.
  6. Pack `source/` zstd frames if `--store-mode bundled`.
  7. Atomically write the v4 `.rlat` ZIP via `store.archive.write`.

Single recipe — no encoder/precision/sparsify/field-type knobs. The
`--kind` flag tags the model as `corpus` (default) or `intent`; v2.0 ships
the tag only, intent operators are deferred.

Phase 3 deliverable. Base plan §3.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import Kind, StoreMode
from ..field import ann
from ..field.encoder import DIM, MAX_SEQ_LENGTH, MODEL_ID, POOLING, Encoder
from ..store import archive
from ..store.base import compute_hash
from ..store.bundled import pack_source_files
from ..store.chunker import chunk_text
from ..store.metadata import BackboneInfo, BandInfo, Metadata
from ..store.registry import PassageCoord, compute_id
from ..store.remote import compose_manifest

# Chunker bounds — recorded into `build_config` at build time and replayed
# verbatim by `rlat refresh` / `rlat watch`. Centralised here so the four
# call sites (CLI argparse defaults, refresh fallback, sync fallback,
# watch fallback) read from one source of truth.
_DEFAULT_MIN_CHARS = 200
_DEFAULT_MAX_CHARS = 3200

# Allowlist of source-text extensions. Conservative on purpose — a build
# that silently ingests a 50 MB binary file would produce garbage embeddings
# and waste compute. Users with non-listed extensions add explicit
# `--ext .foo` rather than getting surprised by mass-ingestion.
_DEFAULT_TEXT_EXTS = frozenset({
    ".py", ".pyi",
    ".md", ".rst", ".txt",
    ".js", ".jsx", ".ts", ".tsx",
    ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".java", ".kt", ".swift",
    ".rb", ".php", ".cs",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".html", ".css", ".scss",
    ".tex", ".org",
})


def _common_root(sources: list[Path]) -> Path:
    """Pick a sensible default `source_root`: the closest common ancestor of
    `sources`. Single source → its parent (file) or itself (dir).
    """
    resolved = [s.resolve() for s in sources]
    if len(resolved) == 1:
        return resolved[0] if resolved[0].is_dir() else resolved[0].parent
    return Path(os.path.commonpath([str(p) for p in resolved]))


def _walk_sources(
    sources: list[Path],
    source_root: Path,
    extensions: frozenset[str],
) -> tuple[list[tuple[str, str]], list[tuple[Path, str]]]:
    """Walk source paths and return `([(rel_path, text)], [(path, reason)])`.

    The second list reports files that were skipped — typically utf-8 decode
    failures or filesystem errors (broken symlinks, network share timeouts,
    Windows file locks). Silent skips are a footgun on first-time builds, so
    `cmd_build` prints a one-line summary.

    Paths are returned relative to `source_root` with forward slashes — what
    `PassageCoord.source_file` records and what the Store layer expects when
    resolving the coordinate back.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    skipped: list[tuple[Path, str]] = []
    for src in sources:
        files: Iterable[Path]
        if src.is_file():
            files = [src]
        else:
            files = src.rglob("*")
        # Sorted enumeration — cross-platform stable so dogfood / kaggle
        # builds produce identical passage_idx → coord mappings (matters for
        # cache reuse and goldens).
        for path in sorted(files):
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                skipped.append((path, "decode"))
                continue
            except OSError as exc:
                # Catches PermissionError, FileNotFoundError (broken symlink),
                # and Windows-specific lock errors — all reasons to skip
                # without aborting the whole build.
                skipped.append((path, type(exc).__name__))
                continue
            try:
                rel = path.resolve().relative_to(source_root.resolve())
            except ValueError:
                # File outside source_root — accept the absolute-path string
                # so the user sees a hint that source_root is wrong.
                rel = path.resolve()
            rel_posix = rel.as_posix()
            if rel_posix in seen:
                continue
            seen.add(rel_posix)
            out.append((rel_posix, text))
    return out, skipped


def _build_passages(
    files: list[tuple[str, str]],
    min_chars: int,
    max_chars: int,
) -> tuple[list[PassageCoord], list[str]]:
    """Chunk every file and emit `(registry, passage_texts)` in passage_idx order."""
    registry: list[PassageCoord] = []
    texts: list[str] = []
    for rel_path, text in files:
        for char_offset, char_length in chunk_text(text, min_chars, max_chars):
            passage_text = text[char_offset:char_offset + char_length]
            registry.append(PassageCoord(
                passage_idx=len(registry),
                source_file=rel_path,
                char_offset=char_offset,
                char_length=char_length,
                content_hash=compute_hash(passage_text),
                passage_id=compute_id(rel_path, char_offset, char_length),
            ))
            texts.append(passage_text)
    return registry, texts


def cmd_build(args: argparse.Namespace) -> int:
    sources = [Path(p) for p in args.sources]
    if not sources:
        print("error: rlat build requires at least one source path")
        return 2

    source_root = Path(args.source_root) if args.source_root else _common_root(sources)
    output_path = Path(args.output)
    store_mode = StoreMode(args.store_mode)
    kind = Kind(args.kind)
    extensions = (
        _DEFAULT_TEXT_EXTS
        if not args.ext
        else frozenset(("." + e.lstrip(".")).lower() for e in args.ext)
    )
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

    print(f"[build] walking sources rooted at {source_root}")
    files, skipped = _walk_sources(sources, source_root, extensions)
    if skipped:
        # Compress reasons → counts so the summary stays one line even when
        # many files were skipped (e.g. binary blobs in a mixed corpus).
        reasons: dict[str, int] = {}
        for _, reason in skipped:
            reasons[reason] = reasons.get(reason, 0) + 1
        print(f"[build] skipped {len(skipped)} files: "
              + ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())))
    if not files:
        print(f"error: no text files found under {sources} (extensions: "
              f"{sorted(extensions)})")
        return 1
    print(f"[build] {len(files)} files; chunking …")
    registry, passage_texts = _build_passages(files, args.min_chars, args.max_chars)
    if not registry:
        print("error: chunker produced no passages — every file may be empty")
        return 1
    runtime = getattr(args, "runtime", "auto") or "auto"
    print(f"[build] {len(registry)} passages; encoding (runtime={runtime}, batch={args.batch_size}) …")

    # Build runtime auto-selects: OpenVINO on Intel CPUs (bit-exact vs torch
    # per Phase 1 lock; ~10x faster than torch CPU), ONNX on non-Intel CPUs,
    # torch only when explicitly requested or as the fallback. CUDA torch
    # is always faster than any CPU runtime when available — pass
    # `--runtime torch` on a CUDA box.
    encoder = Encoder(runtime=runtime)
    # Encoder.encode L2-normalises per batch, so the concatenation is already
    # row-unit-norm — no second pass needed.
    base_band = encoder.encode_batched(passage_texts, args.batch_size)

    print(f"[build] encoded {len(passage_texts)} passages → ({base_band.shape}); "
          f"building metadata …")

    ann_meta: dict[str, dict[str, int | str]] = {}
    ann_blobs: dict[str, bytes] = {}
    if ann.should_build_ann(len(registry)):
        print(f"[build] building FAISS HNSW index (N={len(registry)} > "
              f"{ann.ANN_THRESHOLD_N}) …")
        index = ann.build(base_band)
        ann_blobs["base"] = ann.serialize(index)
        ann_meta["base"] = {
            "type": "hnsw",
            "M": ann.HNSW_M,
            "efConstruction": ann.HNSW_EFCONSTRUCTION,
            "efSearch": ann.HNSW_EFSEARCH,
        }

    metadata = Metadata(
        kind=kind.value,
        backbone=BackboneInfo(
            name=MODEL_ID,
            revision=encoder.revision,
            dim=DIM,
            pool=POOLING,
            max_seq_length=MAX_SEQ_LENGTH,
        ),
        bands={
            "base": BandInfo(
                role="retrieval_default",
                dim=DIM,
                l2_norm=True,
                passage_count=len(registry),
            ),
        },
        store_mode=store_mode.value,
        ann=ann_meta,
        build_config={
            "chunker": "passage_v1",
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "passage_count": len(registry),
            "file_count": len(files),
            "source_root": str(source_root),
            # Provenance for `rlat refresh` — replay-faithful rebuild
            # requires every input that affected the build. Without these,
            # refresh defaults to walking source_root and silently
            # adds/drops files vs. the original. `extensions: null` =
            # default allowlist; an explicit list overrides it.
            "source_paths": [str(p) for p in sources],
            "extensions": (
                None if not args.ext
                else sorted(("." + e.lstrip(".")).lower() for e in args.ext)
            ),
            "batch_size": args.batch_size,
        },
        created_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    source_files_payload: dict[str, bytes] | None = None
    if store_mode is StoreMode.BUNDLED:
        print(f"[build] zstd-framing source/ for bundled mode …")
        source_files_payload = pack_source_files(dict(files))

    remote_manifest_payload: dict[str, dict[str, str]] | None = None
    if store_mode is StoreMode.REMOTE:
        # Single source of truth for url+sha256 manifest composition; same
        # helper feeds `rlat convert --to remote`.
        remote_manifest_payload = compose_manifest(dict(files), args.remote_url_base)
        # Record the prefix in build_config so freshness / sync know
        # what base URL the manifest was built against (useful for
        # diagnostic + future-bump replay).
        prefix = args.remote_url_base.rstrip("/")
        metadata.build_config["upstream_url_base"] = prefix
        print(f"[build] emitted remote manifest with {len(remote_manifest_payload)} "
              f"entries (prefix: {prefix})")

    print(f"[build] writing {output_path} …")
    archive.write(
        output_path,
        metadata=metadata,
        bands={"base": base_band},
        registry=registry,
        ann_blobs=ann_blobs,
        source_files=source_files_payload,
        remote_manifest=remote_manifest_payload,
    )
    out_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[build] wrote {output_path} ({out_size_mb:.2f} MB, "
          f"{len(registry)} passages from {len(files)} files)")
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
    p.add_argument("--min-chars", type=int, default=_DEFAULT_MIN_CHARS,
                   help=f"Chunker min size (default: {_DEFAULT_MIN_CHARS})")
    p.add_argument("--max-chars", type=int, default=_DEFAULT_MAX_CHARS,
                   help=f"Chunker max size (default: {_DEFAULT_MAX_CHARS})")
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
