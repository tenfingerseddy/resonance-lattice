"""watch_loop — `rlat watch` is correct under live edits.

Nine guarantees (all hermetic, no actual filesystem-event delivery — we
exercise the deterministic seams the watchdog observer feeds into):

  1. Discovery: `_discover_archives` finds all `*.rlat` files in the cwd
     and ignores subdirectories.
  2. Single-event refresh: `_refresh_one` after a file modification
     correctly updates the archive (the debouncer's payload).
  3. Add/remove: `_refresh_one` after a create + delete cycle reflects
     both deltas in the registry.
  4. Concurrency safety: the per-archive lock serialises concurrent
     `_refresh_one` calls so two FS events firing milliseconds apart
     never race the `<archive>.tmp` write path.
  5. Bundled-mode pre-flight: `_preflight_archive` rejects bundled
     archives with an actionable `rlat convert` hint.
  6. Optimised band reprojection: a refresh through the watch path
     re-projects the optimised band correctly (no LLM call).
  7. --once with prior drift: `_WatchSession.run_once()` runs a single
     synchronous reconciliation against current disk state and exits.
     CI / pre-commit shape — the "wait for first event" semantic was
     a hang in pre-commit hooks where files are already changed.
  8. force-dispatch bypasses the suffix pre-filter: rename-out scenarios
     (foo.md → foo.bak) and directory deletes still trigger refreshes
     even though the post-rename suffix isn't in the watched allowlist.
  9. Skip preservation: `_filter_skipped_removals` defends against the
     silent-delete hazard where a transient read failure (Windows file
     lock, mid-write decode error) makes a real file disappear from
     `_walk_sources` and bucketise emits a destructive removal for it.
"""

from __future__ import annotations

import io
import sys
import tempfile
import threading
import time
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np


from ._testutil import build_corpus as _build
from .optimised_reproject import _attach_optimised


_FILES = {
    "a.md": "# Alpha\n\nFirst doc about authentication and login flows. "
            "Sessions persist for 24 hours by default.",
    "b.md": "# Beta\n\nSecond doc about credentials and tokens. "
            "Tokens rotate weekly. Logout clears the session.",
    "c.md": "# Gamma\n\nThird doc about session storage in Redis. "
            "Each session has a TTL.",
}


def _read(km: Path):
    from resonance_lattice.store import archive
    return archive.read(km)


def _make_state(km: Path):
    from resonance_lattice.cli.watch import _preflight_archive
    state = _preflight_archive(km)
    if state is None:
        raise RuntimeError(f"preflight returned None for {km}")
    return state


def run() -> int:
    from resonance_lattice.cli.watch import (
        _DebouncedRefresher,
        _discover_archives,
        _preflight_archive,
        _refresh_one,
    )
    from resonance_lattice.field.encoder import Encoder

    encoder = Encoder(runtime="torch")

    # ---- Guarantee 1: discovery ----
    with tempfile.TemporaryDirectory() as d:
        cwd = Path(d)
        # Build two archives in cwd + one in a subdir (must be ignored).
        km_a = cwd / "alpha.rlat"
        km_b = cwd / "beta.rlat"
        sub = cwd / "sub"
        sub.mkdir()
        km_c = sub / "gamma.rlat"
        for p in (km_a, km_b, km_c):
            p.write_bytes(b"")  # placeholder; discovery only globs by name
        found = _discover_archives(cwd)
        if found != sorted([km_a, km_b]):
            print(f"[watch_loop] FAIL guarantee 1: discovery returned "
                  f"{[p.name for p in found]}", file=sys.stderr)
            return 1
    print("[watch_loop] guarantee 1 (discovery) OK", file=sys.stderr)

    # ---- Guarantee 2: single-event refresh updates the archive ----
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        state = _make_state(km)

        a_ids_before = {c.passage_id for c in _read(km).registry
                        if c.source_file == "a.md"}
        (root / "b.md").write_text(
            "# Beta v2\n\nRewritten content about API keys, secret rotation, "
            "and revocation. The new flow forces a logout on every revoke.",
            encoding="utf-8",
        )
        counts = _refresh_one(state, encoder, batch_size=4)
        if counts is None:
            print("[watch_loop] FAIL guarantee 2: refresh returned None "
                  "(no delta detected)", file=sys.stderr)
            return 1
        c_after = _read(km)
        a_ids_after = {c.passage_id for c in c_after.registry
                       if c.source_file == "a.md"}
        # a.md was untouched — its passage_ids must survive verbatim.
        if a_ids_before != a_ids_after:
            print("[watch_loop] FAIL guarantee 2: a.md passage_ids changed "
                  "across an edit to b.md (selective re-encode broken)",
                  file=sys.stderr)
            return 1
        # b.md must have at least one passage in the registry.
        b_count = sum(1 for c in c_after.registry if c.source_file == "b.md")
        if b_count == 0:
            print("[watch_loop] FAIL guarantee 2: b.md has no passages after "
                  "refresh", file=sys.stderr)
            return 1
        if state.refresh_count != 1:
            print(f"[watch_loop] FAIL guarantee 2: refresh_count="
                  f"{state.refresh_count}, expected 1", file=sys.stderr)
            return 1
    print("[watch_loop] guarantee 2 (single-event refresh) OK", file=sys.stderr)

    # ---- Guarantee 3: add + delete reflected ----
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        state = _make_state(km)

        (root / "d.md").write_text(
            "# Delta\n\nFourth doc about audit logs and tamper-evident hashes.",
            encoding="utf-8",
        )
        (root / "c.md").unlink()
        counts = _refresh_one(state, encoder, batch_size=4)
        if counts is None:
            print("[watch_loop] FAIL guarantee 3: no delta detected after "
                  "add+delete", file=sys.stderr)
            return 1
        sources = {c.source_file for c in _read(km).registry}
        if "d.md" not in sources:
            print("[watch_loop] FAIL guarantee 3: d.md missing post-refresh",
                  file=sys.stderr)
            return 1
        if "c.md" in sources:
            print("[watch_loop] FAIL guarantee 3: c.md still present "
                  "post-delete", file=sys.stderr)
            return 1
    print("[watch_loop] guarantee 3 (add+delete) OK", file=sys.stderr)

    # ---- Guarantee 4: per-archive lock serialises concurrent refreshes ----
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        state = _make_state(km)

        # Force a non-empty delta for both threads to operate on.
        (root / "b.md").write_text(
            "# Beta concurrent\n\nRewritten doc with new content for the lock "
            "test. Tokens, secrets, rotation policies.",
            encoding="utf-8",
        )

        # Sanity: the lock must be a real Lock (acquired-blocks-other-acquire),
        # not a no-op sentinel.
        if not (hasattr(state.lock, "acquire") and hasattr(state.lock, "release")):
            print("[watch_loop] FAIL guarantee 4: state.lock is not a real "
                  "lock primitive", file=sys.stderr)
            return 1

        results: list[Exception | None] = [None, None]

        def _worker(i: int) -> None:
            try:
                _refresh_one(state, encoder, batch_size=4)
            except Exception as exc:  # noqa: BLE001
                results[i] = exc

        t0 = threading.Thread(target=_worker, args=(0,))
        t1 = threading.Thread(target=_worker, args=(1,))
        t0.start()
        t1.start()
        t0.join(timeout=120)
        t1.join(timeout=120)
        if t0.is_alive() or t1.is_alive():
            print("[watch_loop] FAIL guarantee 4: refresh threads hung "
                  "(possible deadlock)", file=sys.stderr)
            return 1
        for i, exc in enumerate(results):
            if exc is not None:
                print(f"[watch_loop] FAIL guarantee 4: thread {i} raised "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
        # After both threads, the archive must still be readable + consistent.
        c_final = _read(km)
        if not c_final.bands["base"].shape[0] == len(c_final.registry):
            print("[watch_loop] FAIL guarantee 4: archive inconsistent "
                  "post-concurrent-refresh (band rows != registry size)",
                  file=sys.stderr)
            return 1
        # The lock acquires once per refresh; refresh_count is incremented
        # inside the lock, so it must equal the number of threads that saw
        # a delta. The first refresh consumes the delta; the second sees an
        # empty delta and short-circuits — so refresh_count is exactly 1.
        if state.refresh_count != 1:
            print(f"[watch_loop] FAIL guarantee 4: refresh_count="
                  f"{state.refresh_count} (expected 1: second thread sees "
                  "the post-first-refresh empty delta)", file=sys.stderr)
            return 1
        # Final acquire/release must succeed — nothing leaked the lock.
        acquired = state.lock.acquire(timeout=1.0)
        if not acquired:
            print("[watch_loop] FAIL guarantee 4: lock leaked after both "
                  "threads completed", file=sys.stderr)
            return 1
        state.lock.release()
    print("[watch_loop] guarantee 4 (concurrency) OK", file=sys.stderr)

    # ---- Guarantee 5: bundled-mode pre-flight rejected with hint ----
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES), mode="bundled")
        buf = io.StringIO()
        with redirect_stderr(buf):
            state = _preflight_archive(km)
        if state is not None:
            print("[watch_loop] FAIL guarantee 5: preflight accepted "
                  "bundled-mode archive", file=sys.stderr)
            return 1
        msg = buf.getvalue()
        if "bundled-mode" not in msg or "rlat convert" not in msg:
            print(f"[watch_loop] FAIL guarantee 5: bundled-rejection message "
                  f"missing conversion hint. Got: {msg!r}", file=sys.stderr)
            return 1
    print("[watch_loop] guarantee 5 (bundled pre-flight) OK", file=sys.stderr)

    # ---- Guarantee 6: optimised band reprojected via watch path ----
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        W_initial = _attach_optimised(km, d_native=128)
        state = _make_state(km)

        (root / "b.md").write_text(
            "# Beta v3\n\nDifferent content again — for the optimised "
            "re-projection contract under the watch refresh path.",
            encoding="utf-8",
        )
        counts = _refresh_one(state, encoder, batch_size=4)
        if counts is None:
            print("[watch_loop] FAIL guarantee 6: no delta detected",
                  file=sys.stderr)
            return 1
        c1 = _read(km)
        if "optimised" not in c1.bands:
            print("[watch_loop] FAIL guarantee 6: optimised band missing "
                  "post-refresh", file=sys.stderr)
            return 1
        new_base = c1.bands["base"]
        new_optimised = c1.bands["optimised"]
        W_loaded = c1.projections["optimised"]
        if not np.array_equal(W_loaded, W_initial):
            print("[watch_loop] FAIL guarantee 6: W changed across refresh",
                  file=sys.stderr)
            return 1
        expected = new_base @ W_loaded.T
        expected = expected / np.maximum(
            np.linalg.norm(expected, axis=1, keepdims=True), 1e-12,
        )
        if not np.allclose(new_optimised, expected, atol=1e-6):
            max_diff = float(np.max(np.abs(new_optimised - expected)))
            print(f"[watch_loop] FAIL guarantee 6: optimised band is not "
                  f"new_base @ W.T L2-normalised; max_diff={max_diff}",
                  file=sys.stderr)
            return 1
    print("[watch_loop] guarantee 6 (optimised reprojection) OK",
          file=sys.stderr)

    # ---- Guarantee 7: --once runs a synchronous reconciliation ----
    # CI / pre-commit shape: edits land BEFORE `rlat watch --once` runs,
    # so the command must NOT hang waiting for further events. It walks
    # the tree, runs bucketise+apply, exits.
    from resonance_lattice.cli.watch import _WatchSession
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        state = _make_state(km)

        # Pre-modify before the --once run, exactly as a pre-commit hook
        # would do (formatter pass, applied patch, etc.).
        (root / "b.md").write_text(
            "# Beta once\n\nReconciled by --once after a pre-existing "
            "modification. No FS event delivery in this code path.",
            encoding="utf-8",
        )

        session = _WatchSession(
            states=[state], encoder=encoder,
            debounce_s=0.1, batch_size=4, verbose=False,
        )
        # Crucially: do NOT call session.start() (no observer).
        session.run_once()

        if state.refresh_count != 1:
            print(f"[watch_loop] FAIL guarantee 7: --once refresh_count="
                  f"{state.refresh_count}, expected 1", file=sys.stderr)
            return 1
        if state.archive_path not in session._touched:
            print("[watch_loop] FAIL guarantee 7: archive missing from "
                  "_touched after --once refresh", file=sys.stderr)
            return 1

        # Second --once on the same already-up-to-date archive is a no-op.
        session2 = _WatchSession(
            states=[state], encoder=encoder,
            debounce_s=0.1, batch_size=4, verbose=False,
        )
        # Reset refresh_count so we can detect the no-op.
        rc_before = state.refresh_count
        session2.run_once()
        if state.refresh_count != rc_before:
            print(f"[watch_loop] FAIL guarantee 7: second --once refreshed "
                  f"despite no drift (count went {rc_before} → "
                  f"{state.refresh_count})", file=sys.stderr)
            return 1
    print("[watch_loop] guarantee 7 (--once synchronous reconciliation) OK",
          file=sys.stderr)

    # ---- Guarantee 8: force=True bypasses the suffix pre-filter ----
    # Rename-out (foo.md → foo.bak) and directory deletes deliver paths
    # whose suffix isn't in the archive's extensions allowlist. Without
    # force=True dispatch, those events would be filtered out and stale
    # passages would remain in the archive.
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        state = _make_state(km)
        session = _WatchSession(
            states=[state], encoder=encoder,
            debounce_s=10.0, batch_size=4, verbose=False,
        )
        # A path under the watched root with a NON-watched suffix.
        non_watched = str(root / "foo.bak")

        # force=False (the hot-path default) must NOT touch.
        session._on_event(non_watched, force=False)
        if state.archive_path in session._touched:
            print("[watch_loop] FAIL guarantee 8a: force=False on "
                  ".bak path triggered the archive (suffix filter "
                  "should have dropped it)", file=sys.stderr)
            return 1

        # force=True (move/delete/dir path) MUST touch.
        session._on_event(non_watched, force=True)
        if state.archive_path not in session._touched:
            print("[watch_loop] FAIL guarantee 8b: force=True on "
                  ".bak path did NOT trigger the archive (rename-out / "
                  "directory-delete reconciliation broken)", file=sys.stderr)
            return 1

        # Cancel any debounced work the dispatch scheduled so the rest
        # of the suite runs cleanly.
        for refresher in session.refreshers.values():
            refresher.cancel()
    print("[watch_loop] guarantee 8 (force-dispatch bypass) OK", file=sys.stderr)

    # ---- Guarantee 9: skipped files don't become silent deletes ----
    # _walk_sources records files it couldn't read (Windows lock, mid-
    # write decode error) into `skipped`. Without _filter_skipped_removals,
    # bucketise sees no candidate for that file and emits a destructive
    # removal for every passage of it. Test the helper directly so we can
    # construct a `skipped` set without simulating a real FS lock.
    from resonance_lattice.cli.watch import (
        _filter_skipped_removals,
        _skipped_rel_set,
    )
    from resonance_lattice.store.incremental import BucketedDelta
    from resonance_lattice.store.registry import PassageCoord
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        contents = _read(km)
        # Pick the first passage of "b.md" — pretend b.md was unreadable
        # this pass, so its removal must be demoted to unchanged.
        b_coords = [c for c in contents.registry if c.source_file == "b.md"]
        a_coords = [c for c in contents.registry if c.source_file == "a.md"]
        if not b_coords or not a_coords:
            print("[watch_loop] FAIL guarantee 9 setup: missing passages",
                  file=sys.stderr)
            return 1

        # Construct a delta where b.md's passages are in `removed`
        # (simulating "couldn't read so bucketise dropped them") and
        # a.md's are in `unchanged` (untouched).
        delta = BucketedDelta()
        delta.unchanged = list(a_coords)
        delta.removed = list(b_coords)

        # Skip-set claims b.md was unreadable.
        skipped_rel = {"b.md"}
        preserved = _filter_skipped_removals(delta, skipped_rel)
        if preserved != len(b_coords):
            print(f"[watch_loop] FAIL guarantee 9: preserved {preserved} "
                  f"!= expected {len(b_coords)}", file=sys.stderr)
            return 1
        if delta.removed:
            print(f"[watch_loop] FAIL guarantee 9: {len(delta.removed)} "
                  "removals survived the skip filter", file=sys.stderr)
            return 1
        # Preserved passages must be in `unchanged`.
        b_ids = {c.passage_id for c in b_coords}
        unchanged_ids = {c.passage_id for c in delta.unchanged}
        if not b_ids.issubset(unchanged_ids):
            print("[watch_loop] FAIL guarantee 9: b.md passages not "
                  "promoted to unchanged after skip filter",
                  file=sys.stderr)
            return 1

        # _skipped_rel_set sanity: rel-posix conversion mirrors registry.
        synthetic_skipped = [(root / "b.md", "OSError")]
        rel_set = _skipped_rel_set(synthetic_skipped, root)
        if "b.md" not in rel_set:
            print(f"[watch_loop] FAIL guarantee 9: _skipped_rel_set "
                  f"didn't produce 'b.md' from absolute path; got {rel_set}",
                  file=sys.stderr)
            return 1
    print("[watch_loop] guarantee 9 (skip preservation) OK", file=sys.stderr)

    # ---- Bonus: the debouncer fires after schedule() ----
    # Quick sanity that the timer plumbing works; we don't call this a
    # "guarantee" because it's a structural property rather than a
    # contract on the watch surface.
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        km = _build(root, dict(_FILES))
        state = _make_state(km)
        (root / "b.md").write_text(
            "# Beta debounce\n\nDebounce-window content for the timer test. "
            "Should fire after 100ms of idleness.",
            encoding="utf-8",
        )
        deb = _DebouncedRefresher(
            state, encoder, debounce_s=0.1, batch_size=4, verbose=False,
        )
        deb.schedule()
        deb.schedule()  # rearm — coalescing
        deb.schedule()
        # Wait for the timer to fire + the refresh to complete.
        deadline = time.monotonic() + 30.0
        while state.refresh_count == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        deb.cancel()
        if state.refresh_count != 1:
            print(f"[watch_loop] FAIL debounce sanity: refresh_count="
                  f"{state.refresh_count} (expected 1 — three rapid "
                  "schedules should coalesce into one fire)", file=sys.stderr)
            return 1
    print("[watch_loop] debounce coalescing sanity OK", file=sys.stderr)

    # Smoke: cli/app imports cleanly with the new subparser wired in.
    from resonance_lattice.cli import app as _app  # noqa: F401

    print("[watch_loop] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
