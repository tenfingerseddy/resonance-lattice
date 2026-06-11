"""Band parity — a base-only knowledge model loads and retrieves correctly.

With the MRL optimised band retired, `base` is the only band. This guards
invariant 4 (functional opt-out equivalence): a fresh build carries exactly a
`base` band, `select_band()` returns it with no projection, and the explicit
`prefer="base"` path (used by `cli/compare.py` for the cross-knowledge-model
rule) still resolves to base.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ._testutil import Args as _Args


def _build_basic_corpus(root: Path) -> Path:
    """Build a small bundled, base-only corpus."""
    from resonance_lattice.cli.build import cmd_build
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "src" / "a.py").write_text(
        "def hello():\n    return 42\n\ndef world():\n    return 99\n",
        encoding="utf-8",
    )
    (root / "docs" / "intro.md").write_text(
        "# Intro\n\nThis project does dense retrieval.\n\n"
        "## Notes\n\nMore content here.\n",
        encoding="utf-8",
    )
    out = root / "km.rlat"
    rc = cmd_build(_Args(
        sources=[str(root)], output=str(out),
        store_mode="bundled", kind="corpus", source_root=str(root),
        min_chars=20, max_chars=300, batch_size=4, ext=None,
        remote_url_base=None,
    ))
    if rc != 0:
        raise RuntimeError(f"build failed rc={rc}")
    return out


def run() -> int:
    from resonance_lattice.store import archive

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        km = _build_basic_corpus(root / "a")
        contents = archive.read(km)

        if "base" not in contents.bands:
            print("[band_parity] FAIL: base band missing on fresh build",
                  file=sys.stderr)
            return 1
        if "optimised" in contents.bands:
            print("[band_parity] FAIL: optimised band present (it is retired)",
                  file=sys.stderr)
            return 1

        # `select_band()` returns base — the only band.
        handle = contents.select_band()
        if handle.name != "base":
            print(f"[band_parity] FAIL: select_band picked {handle.name!r}, "
                  f"expected 'base'", file=sys.stderr)
            return 1

        # Explicit `prefer="base"` (the cross-knowledge-model compare rule).
        h_base = contents.select_band(prefer="base")
        if h_base.name != "base":
            print("[band_parity] FAIL: prefer='base' did not resolve to base",
                  file=sys.stderr)
            return 1

        # cli/compare wiring: a (base-only) compare resolves to the base band
        # dim. With optimise retired this is trivially base, but it keeps the
        # compare path — otherwise untested after the slim — from regressing.
        from resonance_lattice.cli.compare import _build_compare
        result = _build_compare(km, contents, km, contents, sample_size=4)
        base_dim = contents.bands["base"].shape[1]
        if result["b"]["band_dim"] != base_dim:
            print(f"[band_parity] FAIL: compare band_dim "
                  f"{result['b']['band_dim']} != base dim {base_dim}",
                  file=sys.stderr)
            return 1
        print("[band_parity] guarantee (base-only selection + compare) OK",
              file=sys.stderr)

    print("[band_parity] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
