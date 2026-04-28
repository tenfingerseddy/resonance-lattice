"""Maintenance commands — refresh / sync / freshness.

  rlat refresh   <km.rlat>   local mode   → incremental delta-apply
  rlat sync      <km.rlat>   remote mode  → incremental delta-apply
  rlat freshness <km.rlat>   remote mode  → read-only drift check

Both `refresh` and `sync` land on `store/incremental.apply_delta` — the
only difference is the upstream-state source. Refresh walks the local
filesystem (using `cli/build._walk_sources`); sync polls a `RemoteIndex`
(see `store/remote_index.py`).

The pipeline:

  1. Discover changed files (filesystem walk OR upstream poll).
  2. Bucketise candidates against the existing registry on stable
     `passage_id`. Unchanged passages keep their band rows untouched.
  3. Re-encode updated + added passages once, batched.
  4. Re-project the optimised band (if present) from the new base —
     free, no LLM call, no GPU.
  5. Atomic in-place write of the full archive.

Bundled archives are immutable post-build (source bytes baked in); `rlat
refresh` rejects them and points at `rlat build`. Audit 07 is the
design source of truth.

`freshness` walks the remote manifest, downloads each entry, hashes it,
and reports per-entry status (verified / drifted / missing) — read-only.
Useful as a CI gate before deciding whether to run sync.

Phase 3 deliverable. Refresh + sync rewritten as incremental delta-apply
in Audit 07 commits 4/8 + 6/8.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import StoreMode
from ..field.encoder import Encoder
from ..store import incremental, open_store
from ..store.archive import read as archive_read
from ..store.remote_index import HttpManifestIndex, RemoteDelta
from .build import _DEFAULT_TEXT_EXTS, _walk_sources
from ._load import load_or_exit


def cmd_refresh(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    mode = StoreMode(contents.metadata.store_mode)
    if mode is StoreMode.BUNDLED:
        print(
            "error: bundled-mode knowledge models are immutable post-build. "
            "Re-run `rlat build` to produce a fresh archive.",
            file=sys.stderr,
        )
        return 1
    if mode is StoreMode.REMOTE:
        print(
            "error: `rlat refresh` is for local-mode archives. Use "
            "`rlat sync` for remote-mode reconciliation.",
            file=sys.stderr,
        )
        return 1

    bc = contents.metadata.build_config
    source_root_str = args.source_root or bc.get("source_root")
    if not source_root_str:
        print(
            f"error: {km_path} has no recorded source_root (older build?). "
            "Pass --source-root <dir> to override.",
            file=sys.stderr,
        )
        return 1
    source_root = Path(source_root_str)

    # Source paths: prefer the recorded `source_paths` (provenance — every
    # input the original build saw); fall back to source_root for older
    # archives that didn't record them. CLI override wins.
    if args.source:
        sources = [Path(s) for s in args.source]
    elif bc.get("source_paths"):
        sources = [Path(p) for p in bc["source_paths"]]
    else:
        sources = [source_root]
    # Extension allowlist: prefer the recorded list; CLI override wins.
    if args.ext is not None:
        ext_arg = list(args.ext)
    elif bc.get("extensions") is not None:
        ext_arg = list(bc["extensions"])
    else:
        ext_arg = None
    extensions = (
        _DEFAULT_TEXT_EXTS
        if not ext_arg
        else frozenset(("." + e.lstrip(".")).lower() for e in ext_arg)
    )

    min_chars = int(bc.get("min_chars", 200))
    max_chars = int(bc.get("max_chars", 3200))

    print(f"[refresh] walking sources rooted at {source_root}")
    files, skipped = _walk_sources(sources, source_root, extensions)
    if skipped:
        reasons: dict[str, int] = {}
        for _, reason in skipped:
            reasons[reason] = reasons.get(reason, 0) + 1
        print(f"[refresh] skipped {len(skipped)} files: "
              + ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())))

    candidates = incremental.chunk_files(files, min_chars, max_chars)
    delta = incremental.bucketise(contents.registry, candidates)

    print(f"[refresh] delta: unchanged={delta.n_unchanged} "
          f"updated={delta.n_updated} added={delta.n_added} "
          f"removed={delta.n_removed}")

    if args.dry_run:
        print(f"[refresh] --dry-run: no changes written")
        return 0

    # Optimised-band warning (not abort): refresh now re-projects the
    # optimised band from the new base for free, so there's no $14-21 +
    # 30 min penalty. The --discard-optimised opt-out is preserved for
    # users who specifically want a base-only archive afterwards.
    re_project_optimised = "optimised" in contents.bands and not args.discard_optimised

    if delta.is_empty:
        if re_project_optimised:
            # Even with an empty delta, the optimised band stays valid against
            # the unchanged base. Skip the rewrite entirely — fast no-op.
            print(f"[refresh] no changes; archive already up to date")
            return 0
        print(f"[refresh] no source changes; nothing to do")
        return 0

    if "optimised" in contents.bands and args.discard_optimised:
        # Strip the optimised band from contents in-place before apply_delta
        # re-projects, so apply_delta sees a base-only archive.
        contents.bands.pop("optimised", None)
        contents.projections.pop("optimised", None)
        contents.metadata.bands.pop("optimised", None)
        contents.metadata.ann.pop("optimised", None)

    n_re_encode = delta.n_re_encode
    encoder = Encoder(runtime="torch")
    if n_re_encode:
        print(f"[refresh] re-encoding {n_re_encode} passage(s) "
              f"(runtime={encoder.runtime_name}, batch={args.batch_size})")
    result = incremental.apply_delta(
        km_path, contents, delta,
        encoder=encoder, batch_size=args.batch_size,
    )

    print(f"[refresh] wrote {result.archive_path} "
          f"({result.n_passages} passages)")
    if result.re_projected_optimised:
        print(f"[refresh] optimised band re-projected from new base "
              f"(no LLM cost, no GPU)")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    rc = _require_remote(km_path, contents, "sync")
    if rc is not None:
        return rc

    if not contents.remote_manifest:
        print(
            f"error: {km_path} has no remote manifest. Was it built with "
            "`--store-mode remote --remote-url-base <url>`?",
            file=sys.stderr,
        )
        return 1

    # Construct the upstream oracle. v2.0 launch ships HttpManifestIndex
    # only; --upstream-manifest URL puts it in catalog mode (detects added
    # files), default is poll mode (detects modified + removed only).
    if args.upstream_manifest:
        index = HttpManifestIndex.from_url(
            existing_manifest=contents.remote_manifest,
            manifest_url=args.upstream_manifest,
        )
        print(f"[sync] upstream manifest fetched from {args.upstream_manifest} "
              f"(head_ref={index.head_ref()[:12]})")
    else:
        index = HttpManifestIndex.from_existing(
            existing_manifest=contents.remote_manifest,
        )
        print(f"[sync] poll mode (no upstream catalog) — modified + removed "
              f"detection only")

    pinned_ref = contents.metadata.build_config.get("pinned_ref", "")
    print(f"[sync] discovering deltas vs pinned_ref={pinned_ref[:12] or '(unset)'} …")
    file_delta: RemoteDelta = index.changed_files_since(pinned_ref)
    print(f"[sync] file delta: added={len(file_delta.added)} "
          f"modified={len(file_delta.modified)} removed={len(file_delta.removed)} "
          f"unavailable={len(file_delta.unavailable)}")

    if args.dry_run:
        print(f"[sync] --dry-run: no changes written")
        for path in file_delta.added:
            print(f"  + {path}")
        for path in file_delta.modified:
            print(f"  ~ {path}")
        for path in file_delta.removed:
            print(f"  - {path}")
        for path in file_delta.unavailable:
            print(f"  ? {path}  (network error)")
        return 0

    # Network errors during poll-mode delta detection are ambiguous — the
    # file might be removed upstream OR the request might have transient-
    # failed. Refuse to silently delete corpus content; require an explicit
    # opt-in to treat unavailable paths as removals.
    if file_delta.unavailable:
        if args.treat_unreachable_as_removed:
            print(f"[sync] --treat-unreachable-as-removed: migrating "
                  f"{len(file_delta.unavailable)} unavailable path(s) into "
                  f"removed bucket", file=sys.stderr)
            # Mutate via reconstruction since RemoteDelta is frozen.
            file_delta = RemoteDelta(
                added=file_delta.added,
                modified=file_delta.modified,
                removed=sorted(set(file_delta.removed) | set(file_delta.unavailable)),
                unavailable=[],
                head_ref=file_delta.head_ref,
            )
        else:
            print(
                f"error: {len(file_delta.unavailable)} upstream path(s) "
                f"unreachable (network error). Refusing to sync — a "
                f"transient outage shouldn't delete corpus content. "
                f"Re-run when upstream is healthy, or pass "
                f"--treat-unreachable-as-removed to drop these paths.",
                file=sys.stderr,
            )
            for path in file_delta.unavailable[:10]:
                print(f"  ? {path}", file=sys.stderr)
            if len(file_delta.unavailable) > 10:
                print(f"  ? … and {len(file_delta.unavailable) - 10} more",
                      file=sys.stderr)
            return 2

    if file_delta.is_empty:
        print(f"[sync] no upstream changes; archive already up to date")
        # Still bump the pinned_ref so future polls compare against the
        # latest checkpoint — cheap, no archive rewrite needed.
        return 0

    # Fetch added + modified file bodies (unchanged files are not re-fetched
    # — their existing band rows stay verbatim, pivoting on stable passage_id).
    bc = contents.metadata.build_config
    min_chars = int(bc.get("min_chars", 200))
    max_chars = int(bc.get("max_chars", 3200))
    changed_paths = list(file_delta.added) + list(file_delta.modified)
    if changed_paths:
        print(f"[sync] fetching {len(changed_paths)} changed file(s) …")
    changed_files: list[tuple[str, str]] = []
    for path in changed_paths:
        data = index.fetch(path)
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        changed_files.append((path, text))

    # Build a BucketedDelta directly: unchanged-files' passages are lifted
    # from the old registry as-is; only changed-files' passages run through
    # bucketise.
    changed_set = set(file_delta.added) | set(file_delta.modified)
    removed_set = set(file_delta.removed)
    delta = incremental.BucketedDelta()
    for c in contents.registry:
        if c.source_file in removed_set:
            delta.removed.append(c)
        elif c.source_file in changed_set:
            # Will be re-bucketised against fresh candidates below.
            continue
        else:
            delta.unchanged.append(c)

    if changed_files:
        candidates = incremental.chunk_files(changed_files, min_chars, max_chars)
        old_changed_subset = [c for c in contents.registry if c.source_file in changed_set]
        sub_delta = incremental.bucketise(old_changed_subset, candidates)
        delta.unchanged.extend(sub_delta.unchanged)
        delta.updated.extend(sub_delta.updated)
        delta.added.extend(sub_delta.added)
        delta.removed.extend(sub_delta.removed)

    print(f"[sync] passage delta: unchanged={delta.n_unchanged} "
          f"updated={delta.n_updated} added={delta.n_added} "
          f"removed={delta.n_removed}")

    # Update remote_manifest in-place: drop removed, re-pin modified +
    # added against the just-fetched bytes' sha256. ALWAYS re-hash from
    # the live text — `index.upstream_spec` returns the OLD entry in poll
    # mode (existing manifest), and a `spec.get("sha256") or hash` fallback
    # silently keeps the stale pin because the old SHA is truthy. The
    # archive is internally consistent only if the manifest entry's SHA
    # matches the bytes we actually encoded.
    from ..store.base import sha256_hex
    new_manifest = dict(contents.remote_manifest)
    for path in file_delta.removed:
        new_manifest.pop(path, None)
    for path, text in changed_files:
        spec = index.upstream_spec(path)
        if spec is None:
            raise RuntimeError(
                f"sync: no upstream spec for changed path {path!r}"
            )
        live_sha = sha256_hex(text)
        # In catalog mode, validate the fetched bytes match what the
        # catalog asserted — surface upstream inconsistencies loudly
        # rather than silently writing whichever SHA the catalog
        # advertised.
        catalog_sha = spec.get("sha256")
        if catalog_sha and catalog_sha != live_sha:
            raise RuntimeError(
                f"sync: catalog SHA mismatch for {path!r}: "
                f"catalog={catalog_sha[:12]} live_bytes={live_sha[:12]}; "
                f"upstream catalog is inconsistent with what fetch returned"
            )
        new_manifest[path] = {"url": spec["url"], "sha256": live_sha}
    contents.remote_manifest.clear()
    contents.remote_manifest.update(new_manifest)
    contents.metadata.build_config["pinned_ref"] = file_delta.head_ref

    if "optimised" in contents.bands and args.discard_optimised:
        contents.bands.pop("optimised", None)
        contents.projections.pop("optimised", None)
        contents.metadata.bands.pop("optimised", None)
        contents.metadata.ann.pop("optimised", None)

    encoder = Encoder(runtime="torch")
    if delta.n_re_encode:
        print(f"[sync] re-encoding {delta.n_re_encode} passage(s) "
              f"(runtime={encoder.runtime_name}, batch={args.batch_size})")
    result = incremental.apply_delta(
        km_path, contents, delta,
        encoder=encoder, batch_size=args.batch_size,
    )

    print(f"[sync] wrote {result.archive_path} "
          f"({result.n_passages} passages, manifest pinned at "
          f"{file_delta.head_ref[:12]})")
    if result.re_projected_optimised:
        print(f"[sync] optimised band re-projected from new base "
              f"(no LLM cost, no GPU)")
    return 0


def _require_remote(km_path: Path, contents, op: str) -> int | None:
    """Return non-None exit code if `contents` isn't a remote archive."""
    mode = StoreMode(contents.metadata.store_mode)
    if mode is not StoreMode.REMOTE:
        print(
            f"error: `rlat {op}` is for remote-mode knowledge models only. "
            f"{km_path} is `{mode.value}`.",
            file=sys.stderr,
        )
        return 1
    return None


def cmd_freshness(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    rc = _require_remote(km_path, contents, "freshness")
    if rc is not None:
        return rc

    store = open_store(km_path, contents)
    print(f"[freshness] checking {len(contents.remote_manifest)} entries against upstream …")
    statuses = store.freshness()
    counts = {"verified": 0, "drifted": 0, "missing": 0}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1

    if args.format == "json":
        print(json.dumps({"counts": counts, "per_entry": statuses}, indent=2))
    else:
        print(f"[freshness] verified={counts['verified']}  "
              f"drifted={counts['drifted']}  missing={counts['missing']}")
        # Surface the non-verified ones; a clean run prints just the summary.
        for source_file, status in sorted(statuses.items()):
            if status != "verified":
                print(f"  {status:10}  {source_file}")
    # Non-zero exit on any drift/missing so CI / scripts can gate.
    return 0 if counts["drifted"] == 0 and counts["missing"] == 0 else 2


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p_refresh = sub.add_parser("refresh", help="Incremental delta-apply (local mode)")
    p_refresh.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p_refresh.add_argument(
        "--source", action="append", default=None,
        help="Override source paths (default: recorded source_paths)",
    )
    p_refresh.add_argument(
        "--source-root", default=None,
        help="Override recorded source_root",
    )
    p_refresh.add_argument(
        "--batch-size", type=int, default=32,
        help="Encoder batch size (default: 32)",
    )
    p_refresh.add_argument(
        "--ext", action="append", default=None,
        help="Override recorded source-file extensions (repeatable; "
             "default: use recorded `extensions` from build_config)",
    )
    p_refresh.add_argument(
        "--discard-optimised", action="store_true",
        help="Drop the optimised band on refresh instead of re-projecting "
             "it from the new base. Rare — re-projection is free (no LLM, "
             "no GPU) and preserves the trained corpus-specific projection.",
    )
    p_refresh.add_argument(
        "--dry-run", action="store_true",
        help="Walk sources + bucketise + report counts; do not write.",
    )
    p_refresh.set_defaults(func=cmd_refresh)

    p_sync = sub.add_parser("sync", help="Incremental delta-apply (remote mode)")
    p_sync.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p_sync.add_argument(
        "--upstream-manifest", default=None,
        help="URL of an upstream manifest endpoint serving the current "
             "{source_file: {url, sha256}} catalog. With this, sync "
             "discovers added + modified + removed paths in O(1) network "
             "calls. Without it, sync polls every URL in the existing "
             "manifest (modified + removed only — no added-file discovery).",
    )
    p_sync.add_argument(
        "--batch-size", type=int, default=32,
        help="Encoder batch size (default: 32)",
    )
    p_sync.add_argument(
        "--discard-optimised", action="store_true",
        help="Drop the optimised band on sync instead of re-projecting "
             "it from the new base. Rare — re-projection is free.",
    )
    p_sync.add_argument(
        "--dry-run", action="store_true",
        help="Discover deltas + report file paths; do not fetch or write.",
    )
    p_sync.add_argument(
        "--treat-unreachable-as-removed", action="store_true",
        help="Treat upstream paths that returned a network error as removed "
             "from the corpus. By default sync aborts on unreachable paths "
             "to avoid deleting corpus content during a transient upstream "
             "outage. Use only when you've confirmed upstream is healthy "
             "and the unreachable paths are genuinely gone.",
    )
    p_sync.set_defaults(func=cmd_sync)

    p_fresh = sub.add_parser("freshness", help="Read-only drift check (remote mode)")
    p_fresh.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p_fresh.add_argument(
        "--format", default="text", choices=["text", "json"],
        help="Output format (default: text)",
    )
    p_fresh.set_defaults(func=cmd_freshness)
