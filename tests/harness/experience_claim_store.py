"""experience_claim_store — the per-user experience-claim backend.

Pins `ExperienceClaimStore` — the `claims.jsonl` + `band.npz` backend:

  (a) write + read_all round-trip — a claim (core + ExperienceFacts)
      reads back equal, and survives a store reopen.
  (b) read(claim_id) returns the claim; an absent id returns None.
  (c) write replaces the claim with the same claim_id — no duplicate.
  (d) write_many inserts a batch under one update, and rejects a batch
      with a duplicate claim_id.
  (e) write with unchanged content + no embedding reuses the band row —
      the encoder is never constructed (proven: encoder=None, no crash).
  (f) delete removes by id, returns the count, compacts the band.
  (g) claims and band stay parallel — read_all_with_band agrees.
  (h) a legacy pre-Stage-4 `sidecar.jsonl` + `memory.npz` is migrated to
      `claims.jsonl` on first open; intent-level rows are dropped.

Hermetic — every write passes an explicit embedding, so the encoder is
never loaded; no I/O beyond the temp dir.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import check_guarantee

_P = "experience_claim_store"


def _vec(fill: float):
    from resonance_lattice.field.encoder import DIM
    return np.full(DIM, fill, dtype=np.float32)


def _claim(claim_id: str, *, content: str = "prefer the standard library",
           kind: str = "event"):
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    return Claim(
        claim_id=claim_id,
        source="experience",
        kind=kind,
        content=content,
        created_at="2026-05-18T00:00:00Z",
        corroboration=3.0,
        falsification=1.0,
        trust_as_of="",
        state="active",
        parent_ids=("01HZEVENT0000000000000000A",),
        facts=ExperienceFacts(
            polarity=("prefer", "workspace:abc"),
            recurrence_count=4,
            criticality="normal",
            created_under_intent_kind="implement",
            transcript_hash="distilled:arrow1:x",
            origin="distilled",
            last_corroborated_at="2026-05-18T00:00:00Z",
        ),
    )


def _store(root: Path):
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    # encoder=None — a hermetic test must never construct the real encoder;
    # every write below supplies an explicit embedding.
    return ExperienceClaimStore(root=root, encoder=None)


def _check_roundtrip() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        c = _claim("01HZCLAIM000000000000000001")
        _store(root).write(c, embedding=_vec(0.1))
        reread = _store(root).read_all()  # fresh instance — survives reopen
    ok = (
        len(reread) == 1
        and reread[0] == c
        and reread[0].facts == c.facts
        and reread[0].trust == c.trust
    )
    return 0 if check_guarantee(ok, "(a) write + read_all round-trip", _P) else 1


def _check_read_by_id() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        store = _store(root)
        store.write(_claim("01HZCLAIM000000000000000002"), embedding=_vec(0.2))
        hit = store.read("01HZCLAIM000000000000000002")
        miss = store.read("01HZCLAIM00000000000000ZZZZ")
    ok = hit is not None and hit.claim_id == "01HZCLAIM000000000000000002" and miss is None
    return 0 if check_guarantee(ok, "(b) read by id + absent → None", _P) else 1


def _check_write_replaces() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        store = _store(root)
        store.write(_claim("01HZCLAIM000000000000000003",
                           content="original"), embedding=_vec(0.3))
        store.write(_claim("01HZCLAIM000000000000000003",
                           content="revised"), embedding=_vec(0.4))
        claims, band = store.read_all_with_band()
    ok = (
        len(claims) == 1
        and claims[0].content == "revised"
        and band.shape[0] == 1
        and np.allclose(band[0], _vec(0.4))
    )
    return 0 if check_guarantee(ok, "(c) write replaces same claim_id", _P) else 1


def _check_write_many() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        store = _store(root)
        batch = [
            _claim("01HZCLAIM000000000000000010", content="a"),
            _claim("01HZCLAIM000000000000000011", content="b"),
            _claim("01HZCLAIM000000000000000012", content="c"),
        ]
        store.write_many(batch, embeddings=np.vstack(
            [_vec(0.1), _vec(0.2), _vec(0.3)]))
        claims = store.read_all()
    ok = [c.claim_id for c in claims] == [b.claim_id for b in batch]

    with tempfile.TemporaryDirectory() as td:
        store = _store(Path(td) / "u")
        dup = [
            _claim("01HZCLAIM000000000000000099", content="x"),
            _claim("01HZCLAIM000000000000000099", content="y"),
        ]
        rejected = False
        try:
            store.write_many(dup, embeddings=np.vstack([_vec(0.1), _vec(0.2)]))
        except ValueError:
            rejected = True
    ok = ok and rejected
    return 0 if check_guarantee(
        ok, "(d) write_many batch insert + duplicate-id reject", _P) else 1


def _check_band_reuse_no_encoder() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        store = _store(root)
        c = _claim("01HZCLAIM000000000000000020", content="stable text")
        store.write(c, embedding=_vec(0.5))
        # Re-write the same claim, same content, NO embedding. The store
        # must reuse the band row — encoder=None would crash if it tried
        # to encode.
        store.write(_claim("01HZCLAIM000000000000000020",
                           content="stable text", kind="event"))
        claims, band = store.read_all_with_band()
    ok = (
        len(claims) == 1
        and claims[0].kind == "event"             # the rewrite landed
        and np.allclose(band[0], _vec(0.5))        # band row reused
    )
    return 0 if check_guarantee(
        ok, "(e) unchanged content reuses band — no encoder", _P) else 1


def _check_delete() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        store = _store(root)
        store.write_many(
            [_claim(f"01HZCLAIM00000000000000003{i}", content=str(i))
             for i in range(3)],
            embeddings=np.vstack([_vec(0.1), _vec(0.2), _vec(0.3)]),
        )
        removed = store.delete(
            ["01HZCLAIM000000000000000030", "01HZCLAIM00000000000000ABSENT"])
        claims, band = store.read_all_with_band()
        removed_none = store.delete([])
    ok = (
        removed == 1
        and removed_none == 0
        and [c.content for c in claims] == ["1", "2"]
        and band.shape[0] == 2
        and np.allclose(band[0], _vec(0.2))
    )
    return 0 if check_guarantee(ok, "(f) delete by id compacts band", _P) else 1


def _check_band_parallel() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        store = _store(root)
        store.write(_claim("01HZCLAIM000000000000000040"), embedding=_vec(0.7))
        store.write(_claim("01HZCLAIM000000000000000041"), embedding=_vec(0.8))
        claims, band = store.read_all_with_band()
    ok = (
        len(claims) == 2 == band.shape[0]
        and np.allclose(band[0], _vec(0.7))
        and np.allclose(band[1], _vec(0.8))
    )
    return 0 if check_guarantee(ok, "(g) claims + band stay parallel", _P) else 1


def _check_legacy_migration() -> int:
    import json

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        root.mkdir(parents=True)
        rows = [
            {"row_id": "01HZLEGACY0000000000000001",
             "text": "legacy event",
             "polarity": ["prefer", "workspace:a"], "recurrence_count": 3,
             "created_at": "2026-04-01T00:00:00Z",
             "last_corroborated_at": "2026-04-02T00:00:00Z",
             "transcript_hash": "manual", "is_bad": False,
             "level": "event", "criticality": "high", "confidence": "high",
             "parent_ids": ["01HZEVENT00000000000000000A"],
             "origin": "manual", "created_under_intent_kind": "implement",
             "corroboration": 3.0, "falsification": 1.0, "trust_as_of": ""},
            {"row_id": "01HZLEGACY0000000000000002", "text": "stale intent",
             "polarity": ["factual"], "recurrence_count": 1,
             "created_at": "2026-04-01T00:00:00Z",
             "last_corroborated_at": "2026-04-01T00:00:00Z",
             "transcript_hash": "manual", "is_bad": False, "level": "goal",
             "criticality": "normal", "confidence": "medium",
             "parent_ids": [], "origin": "manual",
             "created_under_intent_kind": "none",
             "corroboration": 2.0, "falsification": 2.0, "trust_as_of": ""},
        ]
        (root / "sidecar.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        np.savez(root / "memory.npz",
                 band=np.vstack([_vec(0.3), _vec(0.4)]))
        claims, band = _store(root).read_all_with_band()
        legacy_archived = (root / "sidecar.jsonl.migrated").exists()
        claims_written = (root / "claims.jsonl").exists()
    c = claims[0] if claims else None
    ok = (
        c is not None
        and [x.claim_id for x in claims] == ["01HZLEGACY0000000000000001"]
        and c.source == "experience"
        and c.kind == "event"
        and c.content == "legacy event"
        and c.facts.polarity == ("prefer", "workspace:a")
        and c.facts.criticality == "high"
        and c.parent_ids == ("01HZEVENT00000000000000000A",)
        and band.shape[0] == 1
        and np.allclose(band[0], _vec(0.3))
        and legacy_archived
        and claims_written
    )
    return 0 if check_guarantee(
        ok, "(h) legacy sidecar migration", _P) else 1


def run() -> int:
    failures = 0
    for check in (
        _check_roundtrip,
        _check_read_by_id,
        _check_write_replaces,
        _check_write_many,
        _check_band_reuse_no_encoder,
        _check_delete,
        _check_band_parallel,
        _check_legacy_migration,
    ):
        failures += check()
    if failures:
        print(f"[{_P}] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print(f"[{_P}] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
