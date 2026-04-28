"""Bundled store — source files inside the .rlat ZIP under `source/`.

Each source file is zstd-framed individually so a single `fetch` decompresses
just one file, not the whole tree. The outer ZIP is `ZIP_STORED` (set by
`store.archive`); zstd handles content compression. Self-contained artefact —
ship on HF Hub, embed in CI, query offline.

Build-side pipeline (Phase 3 #25) calls `pack_source_files` to zstd-frame the
text content, then hands the resulting `dict[str, bytes]` to
`archive.write(source_files=...)`. Query-side, `BundledStore` (a `Store`
subclass) reverses the round-trip — `fetch` and `verify` are inherited from
`store.base.Store`; this file only owns the per-mode read primitive.

Phase 2 deliverable. Base plan §5.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import zstandard as zstd

from .archive import SOURCE_DIR
from .base import Store

# Module-level decompressor: zstd contexts are reusable for one-shot
# `.decompress(blob)` calls (the streaming API is the per-call one), so
# there's no reason to allocate a fresh context every fetch.
_DCTX = zstd.ZstdDecompressor()


def pack_source_files(files: dict[str, str]) -> dict[str, bytes]:
    """zstd-frame source files for `archive.write(source_files=...)`.

    Each file is compressed independently with a fresh `ZstdCompressor` —
    no streaming dict, no shared context. That's deliberate: random-access
    `fetch` of one file shouldn't depend on having decompressed any other
    file first. The space cost vs a shared dict is small for source-code
    corpora because files compress well in isolation.

    Returns a dict suitable to pass through to `archive.write` — keys are
    the relative source paths (e.g. `"src/foo.py"`); `archive.write` adds
    the `source/` ZIP prefix.
    """
    cctx = zstd.ZstdCompressor()
    return {path: cctx.compress(text.encode("utf-8")) for path, text in files.items()}


class BundledStore(Store):
    """Read source from inside the .rlat ZIP under `source/`.

    The ZipFile is opened per call rather than held open. Single-shot CLI
    queries don't pay for that, and avoiding a long-lived handle means the
    `.rlat` can be moved/replaced safely while the process is alive.
    `Store`'s text cache amortises across multiple hits in the same source
    file so the per-call ZIP open + zstd decompress only fires once per
    unique file per query.
    """

    def __init__(self, zip_path: str | Path):
        super().__init__()
        self.zip_path = Path(zip_path)

    def _read_full_text_uncached(self, source_file: str) -> str:
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            try:
                blob = zf.read(SOURCE_DIR + source_file)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"{SOURCE_DIR}{source_file} not in {self.zip_path} — "
                    f"either store_mode != 'bundled' or build dropped this file"
                ) from exc
        return _DCTX.decompress(blob).decode("utf-8")
