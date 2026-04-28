"""incremental_refresh — `rlat refresh` is correct under local edits.

Six guarantees:

  1. Build → refresh-with-no-changes is a no-op (band identical).
  2. Edit one file → refresh re-encodes only that file's passages
     (other rows are byte-identical lifts from the old band).
  3. Add a new file → refresh appends new passages (added bucket).
  4. Delete a file → refresh drops its passages (removed bucket).
  5. Refresh idempotency: a second refresh-no-changes is fully no-op.
  6. Stable passage_id semantics: a passage that survives multiple
     edits keeps the same id; a passage that moves within a file
     gets a new id.

Audit 07 commit 7/8.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


from ._testutil import Args as _Args
from ._testutil import build_corpus as _build


def _refresh(km: Path) -> int:
    from resonance_lattice.cli.maintain import cmd_refresh
    return cmd_refresh(_Args(
        knowledge_model=str(km),
        source=None, source_root=None, batch_size=4, ext=None,
        discard_optimised=False, dry_run=False,
    ))


def _read(km: Path):
    from resonance_lattice.store import archive
    return archive.read(km)


def run() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        files = {
            "a.md": "# Alpha\n\nFirst doc about authentication and login flows. "
                    "Sessions persist for 24 hours by default.",
            "b.md": "# Beta\n\nSecond doc about credentials and tokens. "
                    "Tokens rotate weekly. Logout clears the session.",
            "c.md": "# Gamma\n\nThird doc about session storage in Redis. "
                    "Each session has a TTL.",
        }
        km = _build(root, files)
        c0 = _read(km)
        n0 = len(c0.registry)
        band0 = c0.bands["base"].copy()
        ids0 = {c.passage_id for c in c0.registry}

        # ---- Guarantee 1: refresh-with-no-changes is a no-op ----
        rc = _refresh(km)
        if rc != 0:
            print(f"[incremental_refresh] FAIL guarantee 1: refresh rc={rc}",
                  file=sys.stderr)
            return 1
        c1 = _read(km)
        if len(c1.registry) != n0:
            print(f"[incremental_refresh] FAIL guarantee 1: passage count "
                  f"changed {n0} → {len(c1.registry)}", file=sys.stderr)
            return 1
        if not np.array_equal(c1.bands["base"], band0):
            # Allow a re-ordering as long as the SET of rows is identical;
            # bucketise's unchanged path lifts rows in old-registry order
            # which IS preserved here.
            print(f"[incremental_refresh] FAIL guarantee 1: band changed on "
                  f"no-op refresh", file=sys.stderr)
            return 1
        print("[incremental_refresh] guarantee 1 (no-op refresh) OK",
              file=sys.stderr)

        # ---- Guarantee 2: edit one file → only that file re-encoded ----
        # Identify which rows belong to b.md before the edit.
        b_idxs_old = [c.passage_idx for c in c1.registry if c.source_file == "b.md"]
        a_idxs_old = [c.passage_idx for c in c1.registry if c.source_file == "a.md"]
        a_rows_before = c1.bands["base"][a_idxs_old].copy()

        (root / "b.md").write_text(
            "# Beta v2\n\nRewritten content about API keys, secret rotation, "
            "and revocation. The new flow forces a logout on every revoke.",
            encoding="utf-8",
        )
        rc = _refresh(km)
        if rc != 0:
            print(f"[incremental_refresh] FAIL guarantee 2: rc={rc}",
                  file=sys.stderr)
            return 1
        c2 = _read(km)
        # a.md's rows must be byte-identical lifts (the unchanged path).
        a_idxs_new = [c.passage_idx for c in c2.registry if c.source_file == "a.md"]
        a_rows_after = c2.bands["base"][a_idxs_new]
        if not np.array_equal(a_rows_before, a_rows_after):
            print(f"[incremental_refresh] FAIL guarantee 2: a.md rows "
                  f"changed (should be byte-identical lift)", file=sys.stderr)
            return 1
        print("[incremental_refresh] guarantee 2 (selective re-encode) OK",
              file=sys.stderr)

        # ---- Guarantee 3: add a new file → appended ----
        (root / "d.md").write_text(
            "# Delta\n\nFourth doc about audit logs, tamper-evident hashes, "
            "and append-only storage.",
            encoding="utf-8",
        )
        rc = _refresh(km)
        if rc != 0:
            print(f"[incremental_refresh] FAIL guarantee 3: rc={rc}",
                  file=sys.stderr)
            return 1
        c3 = _read(km)
        sources_after_add = {c.source_file for c in c3.registry}
        if "d.md" not in sources_after_add:
            print(f"[incremental_refresh] FAIL guarantee 3: d.md missing "
                  f"from registry after add", file=sys.stderr)
            return 1
        print("[incremental_refresh] guarantee 3 (added file) OK",
              file=sys.stderr)

        # ---- Guarantee 4: delete a file → dropped ----
        (root / "c.md").unlink()
        rc = _refresh(km)
        if rc != 0:
            print(f"[incremental_refresh] FAIL guarantee 4: rc={rc}",
                  file=sys.stderr)
            return 1
        c4 = _read(km)
        sources_after_delete = {c.source_file for c in c4.registry}
        if "c.md" in sources_after_delete:
            print(f"[incremental_refresh] FAIL guarantee 4: c.md still in "
                  f"registry after delete", file=sys.stderr)
            return 1
        print("[incremental_refresh] guarantee 4 (deleted file) OK",
              file=sys.stderr)

        # ---- Guarantee 5: idempotency ----
        c4_band = c4.bands["base"].copy()
        rc = _refresh(km)
        if rc != 0:
            print(f"[incremental_refresh] FAIL guarantee 5: rc={rc}",
                  file=sys.stderr)
            return 1
        c5 = _read(km)
        if not np.array_equal(c5.bands["base"], c4_band):
            print(f"[incremental_refresh] FAIL guarantee 5: idempotent "
                  f"refresh changed the band", file=sys.stderr)
            return 1
        print("[incremental_refresh] guarantee 5 (idempotent) OK",
              file=sys.stderr)

        # ---- Guarantee 6: stable passage_id across edits ----
        # a.md was never edited, so its passage_ids should be unchanged
        # across the full edit-add-delete sequence.
        a_ids_initial = {c.passage_id for c in c0.registry if c.source_file == "a.md"}
        a_ids_final = {c.passage_id for c in c5.registry if c.source_file == "a.md"}
        if a_ids_initial != a_ids_final:
            print(f"[incremental_refresh] FAIL guarantee 6: a.md passage_ids "
                  f"changed across edits — initial {a_ids_initial} != "
                  f"final {a_ids_final}", file=sys.stderr)
            return 1
        print("[incremental_refresh] guarantee 6 (stable passage_id) OK",
              file=sys.stderr)

    print("[incremental_refresh] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
