"""consolidate_insights — `rlat consolidate-insights` CLI guard.

The end-to-end intent-path consolidation — a real `intent accept`/`reject`
moving corpus trust UP/DOWN through the criterion reducer, plus idempotency —
is proven through real entry points in `insight_loop_e2e`. This suite pins the
command's standalone guard, the one thing that test (which always has outcomes)
does not exercise:

  1. No outcomes recorded → exits 0 with "nothing to consolidate", the corpus
     insight layer untouched.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim, run_cli, unpatch_zero_encoder


def _promote_two(km: Path) -> None:
    """Write two active insights into `km` (the layer the no-op must not move)."""
    from resonance_lattice.store import archive

    c0 = archive.read(km)
    src_ids = [c.passage_id for c in c0.registry[:2]]
    src_hashes = [c.content_hash for c in c0.registry[:2]]
    insights = [
        make_corpus_claim("Auth uses session tokens.",
                          src_ids[:1], src_hashes[:1], state="active"),
        make_corpus_claim("Tokens persist in Redis.",
                          src_ids, src_hashes, state="active"),
    ]
    band = np.zeros((2, 768), dtype="float32")
    archive.write_insight_layer_in_place(km, insights, band)


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.store import archive

    failures = 0
    files = {"a.md": "# Alpha\n\nAuthentication.", "b.md": "# Beta\n\nTokens."}

    # ---- Guarantee 1: no outcomes -> nothing to consolidate ----
    with tempfile.TemporaryDirectory() as d, contextlib.chdir(d):
        km = _build(Path(d) / "corpus", files)
        _promote_two(km)
        before = [i.trust for i in archive.read(km).insights]
        rc, _out, err = run_cli(["consolidate-insights", str(km)])
        after = [i.trust for i in archive.read(km).insights]
        if rc != 0:
            print(f"[consolidate_insights] FAIL g1: rc={rc}\n{err}",
                  file=sys.stderr)
            failures += 1
        elif "nothing to consolidate" not in err:
            print(f"[consolidate_insights] FAIL g1: missing notice\n{err}",
                  file=sys.stderr)
            failures += 1
        elif before != after:
            print("[consolidate_insights] FAIL g1: archive changed despite "
                  "no outcomes", file=sys.stderr)
            failures += 1
        else:
            print("[consolidate_insights] g1 (no outcomes -> no-op) OK",
                  file=sys.stderr)

    if failures:
        print(f"[consolidate_insights] {failures} guarantee(s) failed",
              file=sys.stderr)
        return 1
    print("[consolidate_insights] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
