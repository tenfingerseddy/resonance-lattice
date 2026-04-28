"""Shared helpers for harness suites.

Lifted out of the per-suite duplication that crept in across
`incremental_refresh`, `incremental_sync`, `optimised_reproject`,
`skill_context`, `band_parity`, and `conversion` — each had its own
`_Args` micro-class + a `_build` helper that mirrored argparse's
`Namespace` shape and `cmd_build` invocation pattern. One owner now.

Test-only module — production code never imports from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


class Args:
    """Argparse-Namespace stand-in for harness suites.

    cmd_* functions in `cli/` accept `argparse.Namespace`-like objects via
    duck typing (`args.foo`, `args.bar` attribute access). Constructing a
    real `Namespace` works but doesn't carry intent — the suites want a
    fixture-style "build me an args bag with these fields" call. `Args(**kw)`
    sets every kwarg as an attribute and is equivalent to `Namespace(**kw)`
    for downstream consumers.
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_StoreMode = Literal["bundled", "local", "remote"]


def build_corpus(
    root: Path,
    files: dict[str, str],
    *,
    mode: _StoreMode = "local",
    remote_url_base: str | None = None,
    min_chars: int = 20,
    max_chars: int = 400,
    batch_size: int = 4,
) -> Path:
    """Materialise `files` under `root` and run `cmd_build` against them.

    Returns the resulting `.rlat` path. Used by every harness suite that
    needs a small corpus to exercise refresh/sync/convert/optimise; lifted
    out of per-suite copies that all looked the same modulo the
    store_mode + source-dir layout.

    For `bundled` mode the files are written under `root/src/` and the
    build sources point at that subdir (matches what the per-suite
    `_build_remote` and `_build_bundled` did before this lift).
    """
    from resonance_lattice.cli.build import cmd_build

    if mode == "bundled" or mode == "remote":
        src_dir = root / "src"
    else:
        src_dir = root
    src_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = src_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    out = root / "km.rlat"
    rc = cmd_build(Args(
        sources=[str(src_dir)], output=str(out),
        store_mode=mode, kind="corpus", source_root=str(src_dir),
        min_chars=min_chars, max_chars=max_chars, batch_size=batch_size,
        ext=None,
        remote_url_base=remote_url_base,
    ))
    if rc != 0:
        raise RuntimeError(f"build rc={rc} (mode={mode}, root={root})")
    return out
