"""`rlat compare <a.rlat> <b.rlat> [--format text|json] [--sample N]`

Cross-knowledge-model comparison — always uses the base band per base plan
§3.4. A optimised + non-optimised pair compares on bases; optimised
bands are not interoperable across knowledge models by design.

Reports:

- Side-by-side profile: backbone revision (warn on mismatch), passage_count,
  band dim. Mismatch on revision means the cosine numbers below are still
  ordinally meaningful (both bands are unit vectors) but the magnitude
  comparison is across different embedding distributions — flagged honestly.
- Centroid cosine: cos(mean(A), mean(B)) — single thematic-alignment number.
- Mutual coverage: for a sample of `--sample` passages from each corpus,
  how strongly is each represented in the other? Mean of best-match cosines.
  Asymmetric (A→B and B→A) because corpora are different sizes and one
  may be a strict superset of the other.

Output:

  text  side-by-side text report with the three numbers
  json  structured object — feeds dashboards / scripts

Phase 3 deliverable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..field.algebra import centroid
from ..field.dense import sampled_mean_max_cosine
from ..store import archive
from ._load import load_or_exit


def _build_compare(
    a_path: Path, contents_a: archive.ArchiveContents,
    b_path: Path, contents_b: archive.ArchiveContents,
    sample_size: int,
) -> dict:
    handle_a = contents_a.select_band(prefer="base")
    handle_b = contents_b.select_band(prefer="base")
    rev_a = contents_a.metadata.backbone.revision
    rev_b = contents_b.metadata.backbone.revision
    centroid_a = centroid(handle_a.band)
    centroid_b = centroid(handle_b.band)
    return {
        "a": {
            "path": str(a_path),
            "backbone_revision": rev_a,
            "passage_count": handle_a.band.shape[0],
            "band_dim": handle_a.band.shape[1],
        },
        "b": {
            "path": str(b_path),
            "backbone_revision": rev_b,
            "passage_count": handle_b.band.shape[0],
            "band_dim": handle_b.band.shape[1],
        },
        "revision_match": rev_a == rev_b,
        "centroid_cosine": float(centroid_a @ centroid_b),
        "coverage_a_in_b": sampled_mean_max_cosine(handle_a.band, handle_b.band, sample_size=sample_size),
        "coverage_b_in_a": sampled_mean_max_cosine(handle_b.band, handle_a.band, sample_size=sample_size),
        "sample_size": min(sample_size, handle_a.band.shape[0], handle_b.band.shape[0]),
    }


def _truncate(s: str, width: int) -> str:
    """Cap `s` at `width` characters; trail with `…` if shortened. Keeps the
    last `width-1` chars (path tails are more identifying than path heads
    when the prefix is a long temp dir)."""
    if len(s) <= width:
        return s
    return "…" + s[-(width - 1):]


def _format_text(c: dict, col_width: int = 40) -> str:
    out: list[str] = []
    a_path = _truncate(c["a"]["path"], col_width)
    b_path = _truncate(c["b"]["path"], col_width)
    out.append(f"  {'':<20} {'a':<{col_width}}  {'b':<{col_width}}")
    out.append(f"  {'path':<20} {a_path:<{col_width}}  {b_path:<{col_width}}")
    out.append(f"  {'backbone revision':<20} "
               f"{c['a']['backbone_revision'][:col_width]:<{col_width}}  "
               f"{c['b']['backbone_revision'][:col_width]:<{col_width}}")
    out.append(f"  {'passages':<20} {c['a']['passage_count']:<{col_width}}  "
               f"{c['b']['passage_count']:<{col_width}}")
    out.append(f"  {'band dim':<20} {c['a']['band_dim']:<{col_width}}  "
               f"{c['b']['band_dim']:<{col_width}}")
    out.append("")
    out.append(f"centroid_cosine    {c['centroid_cosine']:+.4f}   "
               f"(thematic alignment of corpus means)")
    out.append(f"coverage A → B     {c['coverage_a_in_b']:+.4f}   "
               f"(mean max-cos of {c['sample_size']} A-samples vs B)")
    out.append(f"coverage B → A     {c['coverage_b_in_a']:+.4f}   "
               f"(mean max-cos of {c['sample_size']} B-samples vs A)")
    return "\n".join(out)


def cmd_compare(args: argparse.Namespace) -> int:
    a_path = Path(args.a)
    b_path = Path(args.b)
    contents_a = load_or_exit(a_path)
    contents_b = load_or_exit(b_path)

    if "base" not in contents_a.bands:
        print(f"error: {a_path} has no base band — cross-model compare requires it",
              file=sys.stderr)
        return 1
    if "base" not in contents_b.bands:
        print(f"error: {b_path} has no base band — cross-model compare requires it",
              file=sys.stderr)
        return 1
    if contents_a.bands["base"].shape[1] != contents_b.bands["base"].shape[1]:
        print(f"error: base-band dim mismatch ({contents_a.bands['base'].shape[1]} "
              f"vs {contents_b.bands['base'].shape[1]}) — incompatible knowledge models",
              file=sys.stderr)
        return 1

    result = _build_compare(a_path, contents_a, b_path, contents_b, args.sample)

    if not result["revision_match"]:
        # Stderr so JSON consumers see the warning too without polluting
        # stdout; both formats trip on the same condition.
        print(
            "WARNING: backbone revisions differ. Cosine numbers are "
            "ordinally meaningful (both bands are unit vectors) but the "
            "magnitudes are across different embedding distributions.",
            file=sys.stderr,
        )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(_format_text(result))
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("compare", help="Cross-knowledge-model comparison (base band)")
    p.add_argument("a", help="First .rlat knowledge model")
    p.add_argument("b", help="Second .rlat knowledge model")
    p.add_argument("--format", default="text", choices=["text", "json"])
    p.add_argument(
        "--sample", type=int, default=512,
        help="Sample size for mutual-coverage estimates (default: 512)",
    )
    p.set_defaults(func=cmd_compare)
