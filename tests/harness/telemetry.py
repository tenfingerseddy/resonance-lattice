"""telemetry — the capture→`.rlat` persistence fold (CRITICAL_PATH step 1).

The fold is the FRONT half of the self-improvement loop the back half already
had: `field.capture` observes into an in-memory ring, and `store.telemetry.flush`
drains + redacts + appends those rows to the `insight/telemetry.jsonl` member
INSIDE the `.rlat`. Pins (capture.md §3, architecture.md §7):

  (a) read on a fresh archive → [] (no telemetry member yet, no error).
  (b) append + read round-trips rows in order.
  (c) append ACCUMULATES — a second fold keeps the first fold's rows.
  (d) the fold preserves every other slot (base band, metadata, insights).
  (e) redaction (invariant §8): a secret in a string leaf is scrubbed; the
      numeric fingerprint/scores are untouched.
  (f) persistence gating — off for a bare search, on under a session / force env.
  (g) never raises — flush(None) / a bad path / disabled persistence all yield 0,
      and disabled persistence leaves the buffer intact for a later fold.

  (GATE) THE PRODUCT-LOCUS GATE — run the SHIPPED observe path (`field.retrieve`)
      against a REAL on-disk `.rlat`, fold, then OPEN THE ZIP and prove the
      redacted fingerprint rows physically live in `insight/telemetry.jsonl`.
      No encoder, no network — the artifact is inside the file.

Hermetic: a tiny real archive is written with synthetic bands; no encoder load.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from resonance_lattice import field
from resonance_lattice.field import capture
from resonance_lattice.store import archive, telemetry
from resonance_lattice.store import registry as registry_io
from resonance_lattice.store.metadata import BandInfo, Metadata

_TELEMETRY_MEMBER = "insight/telemetry.jsonl"


def _write_real_rlat(path: Path, n: int = 4, dim: int = 8) -> None:
    """Materialise a minimal but real v4 `.rlat` ZIP (synthetic L2 band, no
    encoder) so the fold writes into a genuine archive."""
    rng = np.random.default_rng(0)
    band = rng.standard_normal((n, dim)).astype("float32")
    band /= np.linalg.norm(band, axis=1, keepdims=True)
    registry = [
        registry_io.PassageCoord(
            passage_idx=i,
            source_file=f"doc{i}.txt",
            char_offset=0,
            char_length=10,
            content_hash="sha256:0",
            passage_id=registry_io.compute_id(f"doc{i}.txt", 0, 10),
        )
        for i in range(n)
    ]
    meta = Metadata(
        bands={"base": BandInfo(role="retrieval_default", dim=dim, passage_count=n)}
    )
    archive.write(path, meta, {"base": band}, registry)


def _sample_row(layer: str = "source", idx: int = 0, session: str = "s") -> dict:
    return {
        "ts": "2026-06-03T00:00:00+00:00",
        "session": session,
        "layer": layer,
        "is_user_query": True,
        "query_emb": [0.1, 0.2, 0.3],
        "ranked": [{"rank": 0, "idx": idx, "score": 0.9}],
    }


def _check_read_empty() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        if archive.read_telemetry(p) != []:
            print("[telemetry] read_empty: fresh archive should have no telemetry",
                  file=sys.stderr)
            return 1
        if telemetry.read(str(p)) != []:
            print("[telemetry] read_empty: telemetry.read should be []", file=sys.stderr)
            return 1
    return 0


def _check_append_read_roundtrip() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        rows = [_sample_row("source", 0), _sample_row("insight", 1)]
        n = archive.append_telemetry_in_place(p, rows)
        back = archive.read_telemetry(p)
        if n != 2 or back != rows:
            print(f"[telemetry] roundtrip: n={n} back={back}", file=sys.stderr)
            return 1
        # Empty append is a no-op, no rewrite.
        if archive.append_telemetry_in_place(p, []) != 0:
            print("[telemetry] roundtrip: empty append should be 0", file=sys.stderr)
            return 1
    return 0


def _check_append_accumulates() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        batch1 = [_sample_row("source", 0), _sample_row("source", 1)]
        batch2 = [_sample_row("insight", 2)]
        archive.append_telemetry_in_place(p, batch1)
        archive.append_telemetry_in_place(p, batch2)
        back = archive.read_telemetry(p)
        if back != batch1 + batch2:
            print(f"[telemetry] accumulate: expected 3 ordered rows, got {back}",
                  file=sys.stderr)
            return 1
    return 0


def _check_read_tail_window() -> int:
    """`telemetry.read(tail=N)` keeps only the most recent N rows — the
    bound the decide tier passes so its O(N²) clustering can't hit a cliff
    on a long-lived archive (2026-06 review)."""
    from resonance_lattice.store import telemetry

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        rows = [_sample_row("source", i) for i in range(7)]
        archive.append_telemetry_in_place(p, rows)
        tail3 = telemetry.read(str(p), tail=3)
        full = telemetry.read(str(p))
        if tail3 != rows[-3:] or full != rows:
            print(f"[telemetry] tail window: tail3={len(tail3)} full={len(full)}",
                  file=sys.stderr)
            return 1
    return 0


def _check_other_members_preserved() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p, n=5, dim=8)
        archive.append_telemetry_in_place(p, [_sample_row()])
        contents = archive.read(p)  # full read must still succeed
        if contents.bands["base"].shape != (5, 8):
            print(f"[telemetry] preserve: base band shape lost: "
                  f"{contents.bands['base'].shape}", file=sys.stderr)
            return 1
        if "base" not in contents.metadata.bands or contents.insights != []:
            print("[telemetry] preserve: metadata/insights disturbed", file=sys.stderr)
            return 1
    return 0


def _check_redaction() -> int:
    # The redactor catches CREDENTIALS (not PII like emails — that's its
    # documented scope). Capture rows carry no query text by construction, so
    # this is defense-in-depth: a credential that leaked into a string leaf
    # (e.g. a user-set session id) must be scrubbed. Use a shape it recognises.
    secret = "session-AKIAABCDEFGHIJKLMNOP-tag"  # AWS access key shape
    row = _sample_row(session=secret)
    out = telemetry._redact_row(row)
    if "AKIAABCDEFGHIJKLMNOP" in out["session"] or "REDACT" not in out["session"]:
        print(f"[telemetry] redaction: credential not scrubbed: {out['session']}",
              file=sys.stderr)
        return 1
    # Numeric fingerprint + scores untouched, structural fields intact.
    if out["query_emb"] != row["query_emb"] or out["ranked"] != row["ranked"]:
        print(f"[telemetry] redaction: numeric fields mutated: {out}", file=sys.stderr)
        return 1
    if out["layer"] != "source" or out["is_user_query"] is not True:
        print(f"[telemetry] redaction: structural fields mutated: {out}", file=sys.stderr)
        return 1
    return 0


def _with_env(**kv):
    """Set env vars, returning a restore dict of prior values (None = unset)."""
    prior = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return prior


def _restore_env(prior: dict):
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _check_persistence_gating() -> int:
    prior = _with_env(RLAT_CAPTURE_PERSIST=None, RLAT_DOGFOOD_SESSION=None)
    try:
        cases = [
            (dict(RLAT_CAPTURE_PERSIST=None, RLAT_DOGFOOD_SESSION=None), False),
            (dict(RLAT_CAPTURE_PERSIST="1", RLAT_DOGFOOD_SESSION=None), True),
            (dict(RLAT_CAPTURE_PERSIST=None, RLAT_DOGFOOD_SESSION="run-7"), True),
            (dict(RLAT_CAPTURE_PERSIST="0", RLAT_DOGFOOD_SESSION="run-7"), False),
        ]
        for env, expected in cases:
            _with_env(**env)
            if telemetry.persistence_enabled() != expected:
                print(f"[telemetry] gating: env={env} expected {expected}",
                      file=sys.stderr)
                return 1
    finally:
        _restore_env(prior)
    return 0


def _check_never_raises() -> int:
    prior = _with_env(RLAT_CAPTURE_PERSIST="1")
    try:
        if telemetry.flush(None) != 0:
            print("[telemetry] never_raises: flush(None) should be 0", file=sys.stderr)
            return 1
        if telemetry.flush("/no/such/path.rlat") != 0:
            print("[telemetry] never_raises: flush(bad path) should be 0",
                  file=sys.stderr)
            return 1
        if telemetry.read("/no/such/path.rlat") != []:
            print("[telemetry] never_raises: read(bad path) should be []",
                  file=sys.stderr)
            return 1
    finally:
        _restore_env(prior)
    return 0


def _check_disabled_keeps_buffer() -> int:
    km = "disabled-km"
    capture.drain(km)
    prior = _with_env(RLAT_CAPTURE_PERSIST="0", RLAT_DOGFOOD_SESSION=None)
    try:
        capture.observe(km, [0.1, 0.2], [(0, 0.9)], "source")
        if telemetry.flush(km) != 0:
            print("[telemetry] disabled: flush should be 0 when off", file=sys.stderr)
            return 1
        if len(capture.buffered(km)) != 1:
            print("[telemetry] disabled: buffer must be intact for a later fold",
                  file=sys.stderr)
            return 1
    finally:
        capture.drain(km)
        _restore_env(prior)
    return 0


def _check_gate_end_to_end() -> int:
    """THE PRODUCT-LOCUS GATE: shipped observe path → real .rlat → open the ZIP."""
    prior = _with_env(RLAT_CAPTURE_PERSIST="1", RLAT_DOGFOOD_SESSION="gate")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "real.rlat"
        _write_real_rlat(p, n=4, dim=8)
        contents = archive.read(p)
        handle = contents.select_band()
        km = handle.km_id
        capture.drain(km)  # defeat any cross-suite residue
        try:
            q = contents.bands["base"][0].copy()  # a real query vector
            result = field.retrieve(q, handle, None, contents.registry, top_k=3)
            folded = telemetry.flush(km)
            if folded != 1:
                print(f"[telemetry] GATE: expected 1 folded row, got {folded}",
                      file=sys.stderr)
                return 1

            # Discharge the gate: OPEN the .rlat ZIP and point at the artifact.
            with zipfile.ZipFile(p, "r") as zf:
                if _TELEMETRY_MEMBER not in zf.namelist():
                    print(f"[telemetry] GATE: {_TELEMETRY_MEMBER} NOT inside the .rlat — "
                          f"members={zf.namelist()}", file=sys.stderr)
                    return 1
                lines = zf.read(_TELEMETRY_MEMBER).decode("utf-8").splitlines()
            disk_rows = [json.loads(x) for x in lines if x.strip()]
            if len(disk_rows) != 1:
                print(f"[telemetry] GATE: expected 1 row in member, got {len(disk_rows)}",
                      file=sys.stderr)
                return 1
            r = disk_rows[0]
            shape_ok = (
                r["layer"] == "source"
                and r["is_user_query"] is True
                and isinstance(r.get("query_emb"), list) and r["query_emb"]
                and r.get("ranked")
                and [h["idx"] for h in r["ranked"]] == [i for i, _ in result]
            )
            if not shape_ok:
                print(f"[telemetry] GATE: persisted row shape wrong: {r}", file=sys.stderr)
                return 1
            # The decide-tier read path sees exactly the on-disk rows.
            if telemetry.read(km) != disk_rows:
                print("[telemetry] GATE: telemetry.read != on-disk rows", file=sys.stderr)
                return 1
            # The fold preserved the corpus: base band still loads.
            if archive.read(p).bands["base"].shape != (4, 8):
                print("[telemetry] GATE: base band corrupted by the fold", file=sys.stderr)
                return 1
        finally:
            capture.drain(km)
            _restore_env(prior)
    return 0


def run() -> int:
    for check in [
        _check_read_empty,
        _check_append_read_roundtrip,
        _check_append_accumulates,
        _check_read_tail_window,
        _check_other_members_preserved,
        _check_redaction,
        _check_persistence_gating,
        _check_never_raises,
        _check_disabled_keeps_buffer,
        _check_gate_end_to_end,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[telemetry] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
