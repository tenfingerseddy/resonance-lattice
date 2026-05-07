"""rlat watch — live, silent, self-discovering refresh loop.

Watches the source roots recorded in each archive's `build_config` and
runs `incremental.apply_delta` on every save, after a debounce window.
Default output is silent — errors are loud, success is invisible. Pre-loads
the encoder at startup so the first event refreshes immediately.

Multi-archive: zero-arg invocation discovers `*.rlat` in cwd and watches
all of their source roots concurrently. FS events fan out to every archive
whose source_paths cover the changed path.

Concurrency: each archive holds a `threading.Lock` that serialises
refreshes. This is non-negotiable — `incremental.apply_delta` writes via
`<archive>.tmp` + `os.replace`; two concurrent calls share that tmp path
and the second silently corrupts the first.

Bundled-mode and remote-mode archives are rejected at startup with
actionable error messages (use `rlat convert` or `rlat sync` respectively).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import StoreMode
from ..field.encoder import Encoder
from ..store import incremental
from .build import _walk_sources
from ._load import load_build_spec, load_or_exit


# Directory names whose churn would generate spurious events even though
# user-visible content didn't change. Checked as a path-component match
# against the path relative to the archive's source_root.
_BLOCKED_DIR_NAMES = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".ruff_cache",
})


@dataclass
class _ArchiveState:
    """Per-archive state owned by a watch session: source-of-truth for the
    refresh-input arguments + a per-archive lock that serialises
    `apply_delta` calls (closing the tmp-file race).

    `source_root_resolved` and `source_paths_resolved` are pre-computed at
    construction so per-event filtering doesn't pay `Path.resolve()` cost
    on every notification — symlink expansion + absolute-path normalisation
    is a non-trivial syscall (`stat` on every component) that fires
    thousands of times during a `git checkout`."""
    archive_path: Path
    source_root: Path
    source_paths: list[Path]
    source_root_resolved: Path
    source_paths_resolved: list[Path]
    extensions: frozenset[str]
    min_chars: int
    max_chars: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    refresh_count: int = 0
    error_count: int = 0


def _discover_archives(cwd: Path) -> list[Path]:
    """Return `*.rlat` files in `cwd` (non-recursive), sorted."""
    return sorted(cwd.glob("*.rlat"))


def _path_blocked(resolved_path: Path, resolved_root: Path) -> bool:
    """True if `resolved_path` falls under a known-noisy subdirectory of
    `resolved_root`. Both arguments are already resolved by the caller."""
    try:
        rel = resolved_path.relative_to(resolved_root)
    except ValueError:
        return True
    for part in rel.parts:
        if part in _BLOCKED_DIR_NAMES:
            return True
    return False


def _path_under_any(resolved_path: Path, resolved_roots: list[Path]) -> bool:
    """True if `resolved_path` is under any of `resolved_roots`. Both
    arguments are already resolved by the caller."""
    for root in resolved_roots:
        try:
            resolved_path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _preflight_archive(km_path: Path) -> _ArchiveState | None:
    """Load + validate an archive for watching. Returns the state to use,
    or `None` after printing an actionable error to stderr."""
    contents = load_or_exit(km_path)
    mode = StoreMode(contents.metadata.store_mode)
    if mode is StoreMode.BUNDLED:
        print(
            f"error: {km_path} is bundled-mode (immutable post-build). "
            f"Convert with: rlat convert {km_path} --to local",
            file=sys.stderr,
        )
        return None
    if mode is StoreMode.REMOTE:
        print(
            f"error: {km_path} is remote-mode. Use `rlat sync {km_path}` "
            f"for upstream reconciliation.",
            file=sys.stderr,
        )
        return None
    spec = load_build_spec(contents)
    if spec is None:
        print(
            f"error: {km_path} has no recorded source_root "
            f"(rebuild with v2.0+ to enable watch).",
            file=sys.stderr,
        )
        return None
    sources = [s for s in spec.source_paths if s.exists()]
    if not sources:
        print(
            f"error: {km_path} — no recorded source paths exist on disk; "
            f"nothing to watch.",
            file=sys.stderr,
        )
        return None
    return _ArchiveState(
        archive_path=km_path,
        source_root=spec.source_root,
        source_paths=sources,
        source_root_resolved=spec.source_root.resolve(),
        source_paths_resolved=[s.resolve() for s in sources],
        extensions=spec.extensions,
        min_chars=spec.min_chars,
        max_chars=spec.max_chars,
    )


def _log_refresh_error(state: _ArchiveState, exc: BaseException) -> None:
    """Bump the per-archive error counter and surface the failure to
    stderr in the standard one-line format. Used by both the debounced
    fire path and the synchronous --once flush."""
    state.error_count += 1
    print(
        f"[rlat watch] {state.archive_path.name}: error — {exc}",
        file=sys.stderr,
    )


def _skipped_rel_set(
    skipped: list[tuple[Path, str]], source_root: Path,
) -> set[str]:
    """Convert `_walk_sources`'s `skipped` paths to the rel-posix names
    that match registry `source_file` keys. Mirrors the rel-path
    computation in `_walk_sources` so a file walked as `docs/intro.md`
    that becomes unreadable also keys as `docs/intro.md`."""
    if not skipped:
        return set()
    root_resolved = source_root.resolve()
    out: set[str] = set()
    for path, _reason in skipped:
        try:
            rel = path.resolve().relative_to(root_resolved)
        except ValueError:
            rel = path.resolve()
        out.add(rel.as_posix())
    return out


def _filter_skipped_removals(
    delta: incremental.BucketedDelta, skipped_rel: set[str],
) -> int:
    """Demote removals whose `source_file` is in `skipped_rel` to
    `unchanged`. Returns the number of passages preserved.

    Defends against the silent-delete hazard where a transient read
    failure (Windows file lock during save, mid-write UTF-8 decode
    error) makes a real source file disappear from `_walk_sources`'s
    output. Bucketise then marks every passage from that file as
    `removed`, and apply_delta writes the deletion. With this filter,
    a "we couldn't read it on this pass" outcome preserves the
    existing rows — the next refresh will reconcile properly once the
    transient error clears. A genuinely deleted file isn't in
    `skipped` (it's silently absent from the walk), so the filter is
    a no-op there.
    """
    if not skipped_rel:
        return 0
    preserved = 0
    still_removed: list = []
    for coord in delta.removed:
        if coord.source_file in skipped_rel:
            delta.unchanged.append(coord)
            preserved += 1
        else:
            still_removed.append(coord)
    delta.removed = still_removed
    return preserved


def _refresh_one(
    state: _ArchiveState, encoder: Encoder, batch_size: int,
) -> tuple[int, int, int] | None:
    """Run a single incremental refresh. Returns `(updated, added, removed)`
    or `None` if nothing changed. Holds the per-archive lock for the entire
    call to serialise concurrent refreshes."""
    with state.lock:
        contents = load_or_exit(state.archive_path)
        files, skipped = _walk_sources(
            state.source_paths, state.source_root, state.extensions,
        )
        candidates = incremental.chunk_files(files, state.min_chars, state.max_chars)
        delta = incremental.bucketise(contents.registry, candidates)
        skipped_rel = _skipped_rel_set(skipped, state.source_root)
        preserved = _filter_skipped_removals(delta, skipped_rel)
        if preserved:
            print(
                f"[rlat watch] {state.archive_path.name}: deferred "
                f"{preserved} removal(s) — {len(skipped_rel)} source "
                f"file(s) unreadable (likely editor lock or transient "
                f"OS error); next refresh will reconcile",
                file=sys.stderr,
            )
        if delta.is_empty:
            return None
        result = incremental.apply_delta(
            state.archive_path, contents, delta,
            encoder=encoder, batch_size=batch_size,
        )
        state.refresh_count += 1
        return result.n_updated, result.n_added, result.n_removed


class _DebouncedRefresher:
    """Per-archive debouncer. Each `schedule()` call (re-)arms a timer that
    fires `debounce_s` later; bursts collapse into a single refresh."""

    def __init__(
        self,
        state: _ArchiveState,
        encoder: Encoder,
        debounce_s: float,
        batch_size: int,
        verbose: bool,
    ):
        self.state = state
        self.encoder = encoder
        self.debounce_s = debounce_s
        self.batch_size = batch_size
        self.verbose = verbose
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_s, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            self._timer = None
        try:
            counts = _refresh_one(self.state, self.encoder, self.batch_size)
        except Exception as exc:  # noqa: BLE001  — defer to operator
            _log_refresh_error(self.state, exc)
            return
        if counts is None:
            return
        if self.verbose:
            updated, added, removed = counts
            ts = time.strftime("%H:%M:%S")
            print(
                f"[rlat watch] {ts} {self.state.archive_path.name}: "
                f"refreshed: {updated} modified, {added} added, "
                f"{removed} removed",
                file=sys.stderr,
            )

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def flush(self) -> None:
        """Cancel any pending timer and fire one synchronous refresh.
        No-op if nothing was scheduled."""
        with self._lock:
            pending = self._timer is not None
            if pending:
                self._timer.cancel()
                self._timer = None
        if pending:
            self._fire()


class _WatchSession:
    def __init__(
        self,
        states: list[_ArchiveState],
        encoder: Encoder,
        debounce_s: float,
        batch_size: int,
        verbose: bool,
    ):
        self.states = states
        self.encoder = encoder
        self.debounce_s = debounce_s
        self.batch_size = batch_size
        self.verbose = verbose
        self.refreshers = {
            s.archive_path: _DebouncedRefresher(
                s, encoder, debounce_s, batch_size, verbose,
            )
            for s in states
        }
        self._touched: set[Path] = set()
        self._stop_event = threading.Event()
        self._observer = None  # late-bound to the lazy watchdog import

    def _on_event(self, event_path: str, *, force: bool = False) -> None:
        """Filter and dispatch one FS event to per-archive refreshers.

        `force=False` (the hot-path default for modify/create) applies the
        cheap suffix pre-filter so non-text saves don't trigger a re-walk.

        `force=True` bypasses the suffix filter — used for renames, moves,
        deletes, and directory events. Renaming `foo.md` to `foo.bak`
        leaves the *source* path with a watched suffix and the *dest*
        without one; either end matching means we still need to
        reconcile (otherwise stale `foo.md` passages stay indexed).
        Directory deletes likewise can't pre-filter on suffix because
        the path is the directory itself.

        bucketise is the source of truth — any event that lands a
        refresh scheduling will produce the canonical delta from a
        full source-tree walk. Over-triggering is bounded (empty
        delta = no archive write); under-triggering is correctness."""
        p = Path(event_path)
        suffix = p.suffix.lower()
        if not force:
            # Cheapest pre-filter: extension allowlist on the raw path.
            # If no archive subscribes to this suffix, skip the resolve()
            # syscall (stat-per-component) entirely.
            if not any(suffix in s.extensions for s in self.states):
                return
        resolved = p.resolve()
        for state in self.states:
            if not force and suffix not in state.extensions:
                continue
            if not _path_under_any(resolved, state.source_paths_resolved):
                continue
            if _path_blocked(resolved, state.source_root_resolved):
                continue
            self._touched.add(state.archive_path)
            self.refreshers[state.archive_path].schedule()

    def start(self) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        session = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory:
                    session._on_event(event.src_path)

            def on_created(self, event):
                if not event.is_directory:
                    session._on_event(event.src_path)

            def on_deleted(self, event):
                # Force=True so directory deletes (and files whose suffix
                # was the watched one but is now mid-rename) still trigger
                # a refresh — bucketise reconciles whatever's actually on
                # disk.
                session._on_event(event.src_path, force=True)

            def on_moved(self, event):
                # Both ends matter. `foo.md` → `foo.bak` keeps the watched
                # suffix only on src; `foo.bak` → `foo.md` only on dest;
                # `foo.md` → `../outside/foo.md` leaves dest outside any
                # watched root. Force=True + dispatch on both paths means
                # bucketise sees the full delta regardless of how the
                # rename intersects the suffix / root filters.
                session._on_event(event.src_path, force=True)
                session._on_event(event.dest_path, force=True)

        # Schedule one handler per unique directory root across all
        # archives — overlapping roots get a single observer subscription
        # rather than N copies.
        roots: set[Path] = set()
        for state in self.states:
            for src in state.source_paths:
                resolved = src.resolve()
                roots.add(resolved if resolved.is_dir() else resolved.parent)

        handler = _Handler()
        self._observer = Observer()
        for root in roots:
            self._observer.schedule(handler, str(root), recursive=True)
        self._observer.start()

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1.0)

    def run_once(self) -> None:
        """Synchronous one-shot reconciliation of every preflighted archive.

        Treats FS events as a stale notion: when --once runs, the source
        tree is the source of truth, and we reconcile the archive
        registry against current disk state. CI/pre-commit shape — files
        are typically already changed before the command starts (formatter
        pass, applied patch, git checkout), so waiting for further events
        is the wrong default. The watcher's value is in the steady-state
        `rlat watch` loop; --once short-circuits that into a single
        bucketise+apply pass.

        Does NOT start the watchdog observer (caller arranges this);
        events that race the synchronous pass are ignored. Each refresh
        is logged on `--verbose`; failures bump per-archive error_count
        and surface to stderr."""
        for state in self.states:
            try:
                counts = _refresh_one(state, self.encoder, self.batch_size)
            except Exception as exc:  # noqa: BLE001
                _log_refresh_error(state, exc)
                continue
            if counts is None:
                continue
            self._touched.add(state.archive_path)
            if self.verbose:
                updated, added, removed = counts
                ts = time.strftime("%H:%M:%S")
                print(
                    f"[rlat watch] {ts} {state.archive_path.name}: "
                    f"refreshed: {updated} modified, {added} added, "
                    f"{removed} removed",
                    file=sys.stderr,
                )

    def stop(self) -> None:
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        # Cancel every pending debouncer + drain in-flight refreshes by
        # acquiring each archive's lock (apply_delta runs under it, so the
        # acquire blocks until the rewrite has committed atomically).
        for refresher in self.refreshers.values():
            refresher.cancel()
        for state in self.states:
            with state.lock:
                pass


def _format_elapsed(seconds: float) -> str:
    """Render an elapsed-seconds count as `HHh MMm SSs` / `MMm SSs` / `SSs`."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _try_import_watchdog() -> bool:
    try:
        import watchdog  # noqa: F401
    except ImportError:
        print(
            "error: rlat watch requires the watchdog package. Install with: "
            "pip install rlat[watch]",
            file=sys.stderr,
        )
        return False
    return True


def cmd_watch(args: argparse.Namespace) -> int:
    if not _try_import_watchdog():
        return 1

    if args.knowledge_model:
        archive_paths = [Path(args.knowledge_model)]
    else:
        archive_paths = _discover_archives(Path.cwd())
        if not archive_paths:
            print(
                "error: no .rlat files found in cwd. Pass an explicit path "
                "or build one first with `rlat build`.",
                file=sys.stderr,
            )
            return 1

    states: list[_ArchiveState] = []
    for km_path in archive_paths:
        state = _preflight_archive(km_path)
        if state is None:
            return 1
        states.append(state)

    name_list = ", ".join(s.archive_path.name for s in states)
    n_roots = len({s.source_root for s in states})
    n_exts = len(set().union(*(s.extensions for s in states)))
    print(
        f"[rlat watch] {name_list} — watching {n_roots} source root(s), "
        f"{n_exts} extension(s)",
        file=sys.stderr,
    )

    print("[rlat watch] loading encoder…", file=sys.stderr, flush=True)
    encoder = Encoder(runtime="torch")
    print("[rlat watch] encoder ready", file=sys.stderr)

    debounce_ms = int(os.environ.get("RLAT_WATCH_DEBOUNCE_MS", "1000"))
    debounce_s = max(0.05, debounce_ms / 1000.0)

    session = _WatchSession(
        states=states,
        encoder=encoder,
        debounce_s=debounce_s,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    start_time = time.monotonic()
    if args.once:
        # Synchronous reconciliation — no observer, no debouncer. Files
        # are already changed before the command starts (pre-commit hook,
        # CI step, formatter pass). Waiting for further events here would
        # hang indefinitely.
        session.run_once()
    else:
        session.start()
        try:
            session.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            session.stop()

    elapsed = time.monotonic() - start_time
    n_refresh = sum(s.refresh_count for s in states)
    n_error = sum(s.error_count for s in states)

    if args.once:
        if n_refresh > 0:
            plural = "" if n_refresh == 1 else "es"
            print(
                f"[rlat watch] --once: {n_refresh} refresh{plural}, "
                f"{n_error} error(s).",
                file=sys.stderr,
            )
        else:
            print(
                "[rlat watch] --once: archive(s) already up to date.",
                file=sys.stderr,
            )
    else:
        plural_r = "" if n_refresh == 1 else "es"
        plural_e = "" if n_error == 1 else "s"
        print(
            f"[rlat watch] stopped. {_format_elapsed(elapsed)}, "
            f"{n_refresh} refresh{plural_r}, {n_error} error{plural_e}.",
            file=sys.stderr,
        )
    return 0 if n_error == 0 else 1


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "watch",
        help="Live, silent, self-refreshing watch loop "
             "(auto-discovers *.rlat in cwd)",
    )
    p.add_argument(
        "knowledge_model", nargs="?", default=None,
        help="Path to a .rlat (default: discover *.rlat in cwd)",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Synchronous one-shot: reconcile every preflighted archive "
             "against current disk state and exit. No observer, no event "
             "wait — CI / pre-commit shape (files are already changed "
             "before the command starts).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print one line per refresh (default: silent on success).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Encoder batch size (default: 32)",
    )
    p.set_defaults(func=cmd_watch)
