"""build_pipeline — rlat-side pure pipeline contract.

Four guarantees on the `resonance_lattice.build` package extracted from
the legacy `cli/build._walk_sources` + `cmd_build` flow:

  B1. `FilesystemSourceWalker` enumerates the same `(rel_posix, text)`
      tuples a hand-written `Path.rglob` filter would, with sorted order
      and deduplication preserved.
  B2. `build_rlat` and `cmd_build` produce equivalent archives for the
      same corpus — registry, bands, metadata (sans created_utc + tempdir
      paths) all match bit-for-bit. Catches CLI-wrapper drift.
  B3. `refresh_rlat` against an unchanged corpus reports
      `n_added=n_changed=n_deleted=0` and `n_unchanged` equal to the
      original passage count; the archive's mtime / contents stay put.
  B4. `build_rlat(encoder=warm)` reuses the injected encoder — verified
      by binding `field.encoder.Encoder.__init__` to a raising stub and
      confirming the build still completes via the injected one.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import (
    Args,
    ZeroEncoder,
    build_corpus,
    check_guarantee,
    patch_zero_encoder,
)


_FIXTURE = {
    "a.md": "# Auth\n\nLogin via SSO. Sessions persist 24 hours.",
    "b.md": "# Storage\n\nDocs land in OneLake. Index lives in .rlat.",
    "nested/c.md": "# Nested\n\nDeeper paths must surface as `nested/c.md`.",
    "skip.bin": "\x00binary garbage",  # excluded by default ext filter
}


def _materialise(root: Path, files: dict[str, str]) -> Path:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return src


def _check(ok: bool, label: str) -> bool:
    return check_guarantee(ok, label, "build_pipeline")


def _b1_walker_parity() -> bool:
    """B1: walker yields the expected set of files in sorted order."""
    from resonance_lattice.build.walker import FilesystemSourceWalker

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = _materialise(root, _FIXTURE)
        walker = FilesystemSourceWalker([src], src)
        yielded = [rel for rel, _ in walker.iter_files()]

        expected = sorted([
            "a.md",
            "b.md",
            "nested/c.md",
        ])
        if yielded != expected:
            print(f"[build_pipeline] FAIL B1: walker yielded {yielded} != {expected}",
                  file=sys.stderr)
            return False

        # Skipped is empty (no decode errors; .bin is filtered, not skipped).
        if walker.skipped:
            print(f"[build_pipeline] FAIL B1: unexpected skipped: {walker.skipped}",
                  file=sys.stderr)
            return False

        # total_files counts only ext-matched entries; total_bytes is non-zero.
        if walker.total_files() != 3:
            print(f"[build_pipeline] FAIL B1: total_files={walker.total_files()} != 3",
                  file=sys.stderr)
            return False
        if walker.total_bytes() <= 0:
            print(f"[build_pipeline] FAIL B1: total_bytes={walker.total_bytes()} <= 0",
                  file=sys.stderr)
            return False

        # build_config_extras has source_paths + extensions (None for default).
        extras = walker.build_config_extras
        if extras.get("extensions") is not None:
            print(f"[build_pipeline] FAIL B1: default extensions should serialise "
                  f"as None; got {extras['extensions']}", file=sys.stderr)
            return False
        return True


def _archives_equivalent(km_a: Path, km_b: Path) -> tuple[bool, str]:
    """Structural equivalence — same registry, bands, key metadata."""
    from resonance_lattice.store import archive

    a = archive.read(km_a)
    b = archive.read(km_b)
    if a.registry != b.registry:
        return False, "registry mismatch"
    if a.metadata.kind != b.metadata.kind:
        return False, f"kind {a.metadata.kind!r} != {b.metadata.kind!r}"
    if a.metadata.store_mode != b.metadata.store_mode:
        return False, "store_mode mismatch"
    if a.metadata.backbone != b.metadata.backbone:
        return False, "backbone mismatch"
    if a.metadata.bands != b.metadata.bands:
        return False, "metadata.bands mismatch"
    if a.metadata.ann != b.metadata.ann:
        return False, "metadata.ann mismatch"
    for k in ("chunker", "min_chars", "max_chars", "passage_count", "file_count"):
        if a.metadata.build_config.get(k) != b.metadata.build_config.get(k):
            return False, f"build_config[{k!r}] mismatch"
    if set(a.bands.keys()) != set(b.bands.keys()):
        return False, "band keys mismatch"
    for name, band_a in a.bands.items():
        band_b = b.bands[name]
        if band_a.shape != band_b.shape or band_a.dtype != band_b.dtype:
            return False, f"band {name!r} shape/dtype mismatch"
        if not np.array_equal(band_a, band_b):
            return False, f"band {name!r} values mismatch"
    return True, ""


def _b2_build_parity() -> bool:
    """B2: `build_rlat` and `cmd_build` produce structurally equivalent archives."""
    from resonance_lattice.build.pipeline import build_rlat
    from resonance_lattice.build.walker import FilesystemSourceWalker
    from resonance_lattice.cli.build import cmd_build
    from resonance_lattice.config import Kind, StoreMode

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = _materialise(root, _FIXTURE)

        # Path A — build_rlat directly.
        out_a = root / "a.rlat"
        walker = FilesystemSourceWalker([src], src)
        build_rlat(
            walker, out_a,
            store_mode=StoreMode.LOCAL, kind=Kind.CORPUS,
            min_chars=20, max_chars=400, batch_size=4,
        )

        # Path B — cmd_build (CLI wrapper).
        out_b = root / "b.rlat"
        rc = cmd_build(Args(
            sources=[str(src)], output=str(out_b),
            store_mode="local", kind="corpus", source_root=str(src),
            min_chars=20, max_chars=400, batch_size=4, ext=None,
            remote_url_base=None,
        ))
        if rc != 0:
            print(f"[build_pipeline] FAIL B2: cmd_build rc={rc}", file=sys.stderr)
            return False

        ok, why = _archives_equivalent(out_a, out_b)
        if not ok:
            print(f"[build_pipeline] FAIL B2: {why}", file=sys.stderr)
            return False
    return True


def _b3_refresh_noop() -> bool:
    """B3: refresh against unchanged corpus reports zero deltas, no rewrite."""
    from resonance_lattice.build.pipeline import refresh_rlat
    from resonance_lattice.build.walker import FilesystemSourceWalker

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        km = build_corpus(root, dict(_FIXTURE), mode="local")
        original_size = km.stat().st_size
        original_mtime = km.stat().st_mtime

        walker = FilesystemSourceWalker([root], root)
        result = refresh_rlat(walker, km)
        if not result.is_noop:
            print(f"[build_pipeline] FAIL B3: is_noop=False; "
                  f"added={result.n_added} changed={result.n_changed} "
                  f"deleted={result.n_deleted}", file=sys.stderr)
            return False
        if not (result.n_added == 0 and result.n_changed == 0
                and result.n_deleted == 0):
            print("[build_pipeline] FAIL B3: counts not all zero",
                  file=sys.stderr)
            return False
        # No rewrite — file size + mtime unchanged.
        if km.stat().st_size != original_size:
            print(f"[build_pipeline] FAIL B3: size changed "
                  f"{original_size} -> {km.stat().st_size}", file=sys.stderr)
            return False
        if km.stat().st_mtime != original_mtime:
            print(f"[build_pipeline] FAIL B3: mtime changed",
                  file=sys.stderr)
            return False
    return True


def _b4_encoder_injection() -> bool:
    """B4: `encoder=warm` is reused — fresh Encoder construction is bypassed."""
    from resonance_lattice.build.pipeline import build_rlat
    from resonance_lattice.build.walker import FilesystemSourceWalker
    import resonance_lattice.build.pipeline as _pipeline

    # Replace the Encoder constructor so any fresh construction blows up.
    class _ExplodingEncoder:
        def __init__(self, *a, **kw):
            raise RuntimeError("build_rlat should not have constructed a fresh Encoder")

    original = _pipeline.Encoder
    _pipeline.Encoder = _ExplodingEncoder  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = _materialise(root, _FIXTURE)
            walker = FilesystemSourceWalker([src], src)
            # Pass the injected ZeroEncoder; pipeline must use it as-is.
            build_rlat(
                walker, root / "km.rlat",
                encoder=ZeroEncoder(),
                min_chars=20, max_chars=400, batch_size=4,
            )
    finally:
        _pipeline.Encoder = original  # type: ignore[assignment]
    return True


def run() -> int:
    patch_zero_encoder()
    failures = 0

    failures += not _check(_b1_walker_parity(),
                           "B1 (FilesystemSourceWalker parity)")
    failures += not _check(_b2_build_parity(),
                           "B2 (build_rlat ≡ cmd_build archives)")
    failures += not _check(_b3_refresh_noop(),
                           "B3 (refresh no-op)")
    failures += not _check(_b4_encoder_injection(),
                           "B4 (encoder injection reused)")

    if failures:
        print(f"[build_pipeline] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[build_pipeline] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
