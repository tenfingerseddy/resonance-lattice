"""`rlat profile <knowledge_model.rlat> [--format text|json] [--source-root DIR]`

Knowledge-model diagnostics — what's in the .rlat without running a query.
Reports:

- Backbone: model id, pinned HF revision, dim, pooling, max-seq-length.
- Bands: name → role, dim, passage_count, MRL projection shape if present.
- Storage mode + source root (or manifest pointer for remote).
- Build config: chunker, min/max chars, file count, build timestamp.
- ANN: per-band index params if present.
- Drift summary: counts of verified / drifted / missing across the registry
  (skipped with `--no-drift` or for remote-mode where drift checking
  requires Phase 3 #30 manifest extraction).

Output formats:
  text  human-readable two-column tabulation
  json  structured object — feeds dashboards / scripts

Phase 3 deliverable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..store import archive, open_store
from ..store.base import Store
from ._load import load_or_exit


def _drift_summary(
    store: Store,
    registry: list,
) -> dict[str, int]:
    """Walk the registry and tally drift status.

    The Store's per-instance text cache turns the file-read cost into
    O(distinct files) instead of O(passages). The per-passage hash + slice
    work stays O(passages) — for a 50K-passage corpus expect a few seconds
    of compute on a warm disk. Use `--no-drift` to skip on huge corpora.
    """
    counts = {"verified": 0, "drifted": 0, "missing": 0}
    for coord in registry:
        status = store.verify(
            coord.source_file,
            coord.char_offset,
            coord.char_length,
            coord.content_hash,
        )
        counts[status] += 1
    return counts


def _build_profile(
    km_path: Path,
    contents: archive.ArchiveContents,
    source_root: str | None,
    skip_drift: bool,
) -> dict:
    """Assemble the profile object common to text and JSON formats."""
    meta = contents.metadata
    profile: dict = {
        "knowledge_model": str(km_path),
        "format_version": meta.format_version,
        "kind": meta.kind,
        "rlat_version": meta.rlat_version,
        "created_utc": meta.created_utc,
        "backbone": {
            "name": meta.backbone.name,
            "revision": meta.backbone.revision,
            "dim": meta.backbone.dim,
            "pool": meta.backbone.pool,
            "max_seq_length": meta.backbone.max_seq_length,
        },
        "bands": {
            name: {
                "role": info.role,
                "dim": info.dim,
                "l2_norm": info.l2_norm,
                "passage_count": info.passage_count,
                "w_shape": list(info.w_shape) if info.w_shape else None,
                "trained_from": info.trained_from,
            }
            for name, info in meta.bands.items()
        },
        "store_mode": meta.store_mode,
        "ann": dict(meta.ann),
        "build_config": dict(meta.build_config),
        "registry_size": len(contents.registry),
    }
    if skip_drift:
        profile["drift"] = {"status": "skipped", "reason": "--no-drift"}
        return profile
    try:
        store = open_store(km_path, contents, source_root)
    except (ValueError, NotImplementedError) as exc:
        # Profile is informational — record why drift was skipped rather
        # than failing the whole subcommand. The "status" tag lets JSON
        # consumers discriminate without sniffing key presence.
        profile["drift"] = {"status": "skipped", "reason": str(exc)}
        return profile
    counts = _drift_summary(store, contents.registry)
    profile["drift"] = {"status": "computed", **counts}
    return profile


def _format_text(profile: dict) -> str:
    out: list[str] = []

    def kv(key: str, value: object) -> None:
        out.append(f"  {key:<24} {value}")

    out.append(f"knowledge_model  {profile['knowledge_model']}")
    out.append(f"format_version   {profile['format_version']}  "
               f"kind={profile['kind']}  rlat={profile['rlat_version']}")
    out.append(f"created          {profile['created_utc']}")
    out.append("")
    out.append("backbone")
    bb = profile["backbone"]
    kv("name", bb["name"])
    # Full revision — `rlat profile` is the diagnostic command, an encoder
    # mismatch debug session needs the whole SHA, not a 12-char preview.
    kv("revision", bb["revision"] or "(unset)")
    kv("dim / pool / max_seq", f"{bb['dim']} / {bb['pool']} / {bb['max_seq_length']}")
    out.append("")
    out.append(f"bands  ({len(profile['bands'])})")
    for name, b in profile["bands"].items():
        kv(name, f"role={b['role']}  dim={b['dim']}  N={b['passage_count']}"
                 f"  l2={b['l2_norm']}"
                 + (f"  W={tuple(b['w_shape'])}" if b["w_shape"] else "")
                 + (f"  trained_from={b['trained_from']}" if b["trained_from"] else ""))
    out.append("")
    out.append(f"store_mode       {profile['store_mode']}")
    if profile["ann"]:
        ann_summary = ", ".join(
            f"{name}=hnsw(M={p.get('M')},efC={p.get('efConstruction')},"
            f"efS={p.get('efSearch')})"
            for name, p in profile["ann"].items()
        )
        out.append(f"ann              {ann_summary}")
    bc = profile["build_config"]
    if bc:
        out.append("build_config")
        for k, v in sorted(bc.items()):
            kv(k, v)
    out.append("")
    out.append(f"registry_size    {profile['registry_size']} passages")
    drift = profile["drift"]
    if drift["status"] == "skipped":
        out.append(f"drift            (skipped — {drift['reason']})")
    else:
        total = drift["verified"] + drift["drifted"] + drift["missing"] or 1
        out.append(f"drift            verified={drift['verified']} "
                   f"({100 * drift['verified'] / total:.1f}%)  "
                   f"drifted={drift['drifted']}  missing={drift['missing']}")
    return "\n".join(out)


def cmd_profile(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    profile = _build_profile(km_path, contents, args.source_root, args.no_drift)

    if args.format == "json":
        print(json.dumps(profile, indent=2))
    else:
        print(_format_text(profile))
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("profile", help="Knowledge-model semantic profile")
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument("--format", default="text", choices=["text", "json"])
    p.add_argument(
        "--source-root", default=None,
        help="Override recorded source_root (local mode only)",
    )
    p.add_argument(
        "--no-drift", action="store_true",
        help="Skip the drift walk (instant — useful on huge corpora or remote mode)",
    )
    p.set_defaults(func=cmd_profile)
