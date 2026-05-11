"""`SourceWalker` Protocol + filesystem implementation.

The build/refresh pipelines (`build_rlat`, `refresh_rlat`) iterate sources
through a walker. The filesystem walker preserves the legacy
`_walk_sources` behaviour verbatim; the Fabric package supplies a OneLake
walker (`fabric.onelake_walker`) that swaps discovery + body-read without
touching chunker/encoder code downstream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Protocol, runtime_checkable


DEFAULT_TEXT_EXTENSIONS: frozenset[str] = frozenset({
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


def common_root(sources: list[Path]) -> Path:
    """Closest common ancestor of `sources`. Single source → its parent
    (file) or itself (dir)."""
    resolved = [s.resolve() for s in sources]
    if len(resolved) == 1:
        return resolved[0] if resolved[0].is_dir() else resolved[0].parent
    return Path(os.path.commonpath([str(p) for p in resolved]))


@runtime_checkable
class SourceWalker(Protocol):
    """Yields `(rel_posix_path, utf8_text)` per source file.

    `source_root_for_metadata` is recorded into `metadata.build_config.source_root`.
    Filesystem walker returns the absolute path on disk; OneLake walker
    returns `/lakehouse/default/Files/<source_dir>` so OneLakeStore parses
    the same shape back at query time.

    `build_config_extras` supplies walker-specific provenance merged into
    `metadata.build_config` — `source_paths` + `extensions` for filesystem,
    `source_dir` for OneLake.

    `skipped` is populated as `iter_files` runs. Each entry is
    `(rel_posix_path, reason)`; reasons mirror the legacy `_walk_sources`
    set (`"decode"` for UTF-8 failures, exception class names for OS errors).

    `total_files` / `total_bytes` are pre-flight cap inputs — cheap to
    compute (filesystem: stat per entry; OneLake: list-paths response
    fields). They must not download or read file bodies.
    """

    @property
    def source_root_for_metadata(self) -> str: ...

    @property
    def build_config_extras(self) -> dict[str, object]: ...

    @property
    def skipped(self) -> list[tuple[str, str]]: ...

    def total_files(self) -> int: ...

    def total_bytes(self) -> int: ...

    def iter_files(self) -> Iterator[tuple[str, str]]: ...


class FilesystemSourceWalker:
    """Walk sources from the local filesystem — semantics preserved
    verbatim from the legacy `cli.build._walk_sources`.

    Sorted enumeration is cross-platform stable so dogfood / kaggle builds
    produce identical `passage_idx → coord` mappings (matters for cache
    reuse and goldens). Files outside `source_root` yield an absolute-path
    rel string — a hint that `source_root` is wrong, rather than a silent
    `ValueError`.
    """

    def __init__(
        self,
        sources: list[Path],
        source_root: Path,
        extensions: frozenset[str] | set[str] | None = None,
    ) -> None:
        self._sources = sources
        self._source_root = source_root
        self._extensions: frozenset[str] = (
            DEFAULT_TEXT_EXTENSIONS if extensions is None else frozenset(extensions)
        )
        self._skipped: list[tuple[str, str]] = []
        self._entries: list[tuple[str, Path]] = self._scan()

    def _scan(self) -> list[tuple[str, Path]]:
        """Discover `(rel_posix, abs_path)` without reading bodies."""
        root_resolved = self._source_root.resolve()
        seen: set[str] = set()
        out: list[tuple[str, Path]] = []
        for src in self._sources:
            files: Iterable[Path]
            if src.is_file():
                files = [src]
            else:
                files = src.rglob("*")
            for path in sorted(files):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in self._extensions:
                    continue
                resolved = path.resolve()
                try:
                    rel = resolved.relative_to(root_resolved)
                except ValueError:
                    rel = resolved
                rel_posix = rel.as_posix()
                if rel_posix in seen:
                    continue
                seen.add(rel_posix)
                out.append((rel_posix, path))
        return out

    @property
    def source_root_for_metadata(self) -> str:
        return str(self._source_root)

    @property
    def build_config_extras(self) -> dict[str, object]:
        return {
            "source_paths": [str(p) for p in self._sources],
            "extensions": (
                None
                if self._extensions == DEFAULT_TEXT_EXTENSIONS
                else sorted(self._extensions)
            ),
        }

    @property
    def skipped(self) -> list[tuple[str, str]]:
        return self._skipped

    def total_files(self) -> int:
        return len(self._entries)

    def total_bytes(self) -> int:
        total = 0
        for _, path in self._entries:
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def iter_files(self) -> Iterator[tuple[str, str]]:
        self._skipped.clear()
        for rel_posix, path in self._entries:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                self._skipped.append((rel_posix, "decode"))
                continue
            except OSError as exc:
                # PermissionError, FileNotFoundError (broken symlink), Windows
                # lock errors — all reasons to skip without aborting the build.
                self._skipped.append((rel_posix, type(exc).__name__))
                continue
            yield rel_posix, text
