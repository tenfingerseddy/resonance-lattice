"""memory_v21_recall — §0.6 retrieval pipeline contracts (Appendix D D.1).

Pins six guarantees on `memory.recall.recall` / `memory.recall.rank`:

  (a) Cosine-descending sort: result rows are sorted by cosine
      descending. Any reordering is a §0.6 step-5 violation.

  (b) Confidence gate suppresses below threshold: empty result when
      top1_cosine < cosine_floor OR (top1 - top2) < gap.

  (c) `min_recurrence` filter active by default: rows with
      recurrence_count < M are dropped at gate 4.

  (d) `top_k=N` truncates exactly: never returns more than N rows.

  (e) `is_bad: true` rows excluded: gate 1 drops them regardless of
      cosine.

  (f) Workspace filter scopes correctly to current `cwd_hash`. (D.8
      covers the collision-detection edge cases; this suite asserts
      the integration shape.)

Hermetic — uses FixedEncoder, deterministic seeded fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ._testutil import FixedEncoder, patch_zero_encoder


def _build_fixture_band(rows_meta: list[dict], dim: int = 768, seed: int = 42) -> np.ndarray:
    """Materialise an L2-normalised band where the i'th row's vector is
    chosen so the cosines we want for tests are exact.

    The fixtures use a "target cosine" field per row; we build the band
    as a small linear combination of a query vector + an orthogonal
    deviation vector so each row has a controlled cosine against a
    known query.
    """
    rng = np.random.default_rng(seed)
    query = rng.standard_normal(dim).astype("float32")
    query /= np.linalg.norm(query)
    band = np.zeros((len(rows_meta), dim), dtype="float32")
    for i, meta in enumerate(rows_meta):
        target_cos = meta.get("cosine", 0.0)
        ortho = rng.standard_normal(dim).astype("float32")
        ortho -= (ortho @ query) * query  # remove query component
        ortho /= np.linalg.norm(ortho)
        # Vector with cos(angle, query) = target_cos and unit length.
        band[i] = target_cos * query + (1.0 - target_cos**2) ** 0.5 * ortho
        band[i] /= np.linalg.norm(band[i])
    return query, band


def _add_rows_with_band(memory, rows_meta: list[dict], dim: int = 768) -> np.ndarray:
    """Add rows directly via `Memory.add_row(..., embedding=...)`.

    We pre-compute embeddings rather than relying on the encoder so
    cosines are deterministic and don't depend on encoder behaviour.
    Returns the query vector the cosines were planted against.
    """
    query, band = _build_fixture_band(rows_meta, dim=dim)
    for i, meta in enumerate(rows_meta):
        rid = memory.add_row(
            text=meta["text"],
            polarity=meta["polarity"],
            transcript_hash=meta.get("transcript_hash", f"hash{i}"),
            embedding=band[i],
        )
        # Apply mutable fields after creation (recurrence + is_bad
        # default to add_row's `1` and `False`).
        updates: dict[str, object] = {}
        if "recurrence_count" in meta:
            updates["recurrence_count"] = meta["recurrence_count"]
        if meta.get("is_bad"):
            updates["is_bad"] = True
        if updates:
            memory.update_row(rid, **updates)
    return query


def _make_memory(td: Path):
    from resonance_lattice.memory.store import Memory

    return Memory(root=td / "u")


# ---------------------------------------------------------------------------
# (a) Cosine-descending sort
# ---------------------------------------------------------------------------


def _check_descending_sort() -> int:
    import tempfile

    from resonance_lattice.memory.recall import rank

    with tempfile.TemporaryDirectory() as td:
        memory = _make_memory(Path(td))
        # Five rows in descending cosine but inserted in ascending order
        # (so a working sort must flip the storage order back to
        # cosine-descending). 0.05+ gaps so the confidence gate passes.
        cosines = [0.72, 0.78, 0.84, 0.90, 0.96]
        rows_meta = [
            {"text": f"row {i}", "polarity": ["prefer", "workspace:test00"],
             "cosine": cosines[i], "recurrence_count": 5}
            for i in range(5)
        ]
        query_vec = _add_rows_with_band(memory, rows_meta)
        rows, band = memory.read_all()

        # Inject the planted query vector into rank() via a stub encoder
        # whose encode() returns it.
        hits = rank(
            "anything", rows=rows, band=band, encoder=FixedEncoder(query_vec),
            cwd_hash="test00", top_k=10,
        )
        if len(hits) != 5:
            print(f"[memory_v21_recall] FAIL (a): expected 5 hits got "
                  f"{len(hits)}", file=sys.stderr)
            return 1
        for i in range(len(hits) - 1):
            if hits[i].cosine < hits[i + 1].cosine:
                print(f"[memory_v21_recall] FAIL (a): hits not "
                      f"cosine-descending at {i}: {[h.cosine for h in hits]}",
                      file=sys.stderr)
                return 1
    print("[memory_v21_recall] (a) cosine-descending sort OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) Confidence gate suppresses below threshold
# ---------------------------------------------------------------------------


def _check_confidence_gate() -> int:
    import tempfile

    from resonance_lattice.memory.recall import rank

    with tempfile.TemporaryDirectory() as td:
        memory = _make_memory(Path(td))
        rows_meta = [
            {"text": "near-tie A", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.85, "recurrence_count": 5},
            {"text": "near-tie B", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.82, "recurrence_count": 5},
        ]
        query_vec = _add_rows_with_band(memory, rows_meta)
        rows, band = memory.read_all()
        hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query_vec),
                    cwd_hash="test00", top1_top2_gap=0.05)
        if len(hits) != 0:
            print(f"[memory_v21_recall] FAIL (b): gap=0.03 should suppress "
                  f"both hits; got {len(hits)}", file=sys.stderr)
            return 1

        # Top1 below floor: lower cosine_floor relaxation reproduces the
        # gate logic — we want the original 0.7 floor to kill a 0.65 top1.
    with tempfile.TemporaryDirectory() as td:
        memory = _make_memory(Path(td))
        rows_meta = [
            {"text": "weak", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.65, "recurrence_count": 5},
        ]
        query_vec = _add_rows_with_band(memory, rows_meta)
        rows, band = memory.read_all()

        hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query_vec),
                    cwd_hash="test00")
        if hits:
            print(f"[memory_v21_recall] FAIL (b): top1=0.65 below 0.7 floor "
                  f"should be suppressed; got {len(hits)} hits", file=sys.stderr)
            return 1
    print("[memory_v21_recall] (b) confidence gate suppresses below threshold OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) Recurrence gate
# ---------------------------------------------------------------------------


def _check_recurrence_gate() -> int:
    import tempfile

    from resonance_lattice.memory.recall import rank

    with tempfile.TemporaryDirectory() as td:
        memory = _make_memory(Path(td))
        rows_meta = [
            {"text": "low-recurrence", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.95, "recurrence_count": 1},
            {"text": "high-recurrence", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.85, "recurrence_count": 5},
        ]
        query_vec = _add_rows_with_band(memory, rows_meta)
        rows, band = memory.read_all()
        hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query_vec),
                    cwd_hash="test00", min_recurrence=3)
        if len(hits) != 1:
            print(f"[memory_v21_recall] FAIL (c): expected 1 hit (high-rec), "
                  f"got {len(hits)}", file=sys.stderr)
            return 1
        if hits[0].row.text != "high-recurrence":
            print(f"[memory_v21_recall] FAIL (c): wrong row survived "
                  f"recurrence gate: {hits[0].row.text}", file=sys.stderr)
            return 1
    print("[memory_v21_recall] (c) recurrence gate filters rows below M OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) top_k truncation
# ---------------------------------------------------------------------------


def _check_top_k_truncation() -> int:
    import tempfile

    from resonance_lattice.memory.recall import rank

    with tempfile.TemporaryDirectory() as td:
        memory = _make_memory(Path(td))
        rows_meta = [
            {"text": f"row {i}", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.95 - 0.01 * i, "recurrence_count": 5}
            for i in range(10)
        ]
        query_vec = _add_rows_with_band(memory, rows_meta)
        rows, band = memory.read_all()
        hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query_vec),
                    cwd_hash="test00", top_k=3, top1_top2_gap=0.0)
        if len(hits) != 3:
            print(f"[memory_v21_recall] FAIL (d): top_k=3 returned "
                  f"{len(hits)} hits", file=sys.stderr)
            return 1
    print("[memory_v21_recall] (d) top_k truncation exact OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (e) is_bad rows excluded
# ---------------------------------------------------------------------------


def _check_is_bad_excluded() -> int:
    import tempfile

    from resonance_lattice.memory.recall import rank

    with tempfile.TemporaryDirectory() as td:
        memory = _make_memory(Path(td))
        rows_meta = [
            {"text": "high cosine but bad", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.99, "recurrence_count": 10, "is_bad": True},
            {"text": "modest cosine but good",
             "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.80, "recurrence_count": 5},
        ]
        query_vec = _add_rows_with_band(memory, rows_meta)
        rows, band = memory.read_all()
        hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query_vec), cwd_hash="test00")
        if any(h.row.is_bad for h in hits):
            print(f"[memory_v21_recall] FAIL (e): is_bad row in result: "
                  f"{[(h.row.text, h.row.is_bad) for h in hits]}",
                  file=sys.stderr)
            return 1
        if not any(h.row.text == "modest cosine but good" for h in hits):
            print(f"[memory_v21_recall] FAIL (e): good row missing from "
                  f"result; got {[h.row.text for h in hits]}", file=sys.stderr)
            return 1
    print("[memory_v21_recall] (e) is_bad rows excluded OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (f) Workspace filter
# ---------------------------------------------------------------------------


def _check_workspace_filter() -> int:
    import tempfile

    from resonance_lattice.memory.recall import rank

    with tempfile.TemporaryDirectory() as td:
        memory = _make_memory(Path(td))
        rows_meta = [
            {"text": "in scope", "polarity": ["prefer", "workspace:test00"],
             "cosine": 0.85, "recurrence_count": 5},
            {"text": "different workspace", "polarity": ["prefer", "workspace:other1"],
             "cosine": 0.95, "recurrence_count": 5},
            {"text": "cross-workspace", "polarity": ["prefer", "cross-workspace"],
             "cosine": 0.80, "recurrence_count": 5},
        ]
        query_vec = _add_rows_with_band(memory, rows_meta)
        rows, band = memory.read_all()
        hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query_vec),
                    cwd_hash="test00", top1_top2_gap=0.0)
        text_set = {h.row.text for h in hits}
        if "different workspace" in text_set:
            print(f"[memory_v21_recall] FAIL (f): workspace bleed — "
                  f"different-workspace row in result", file=sys.stderr)
            return 1
        if "in scope" not in text_set:
            print(f"[memory_v21_recall] FAIL (f): same-workspace row missing",
                  file=sys.stderr)
            return 1
        if "cross-workspace" not in text_set:
            print(f"[memory_v21_recall] FAIL (f): cross-workspace row missing",
                  file=sys.stderr)
            return 1
    print("[memory_v21_recall] (f) workspace filter scopes correctly OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_descending_sort,
        _check_confidence_gate,
        _check_recurrence_gate,
        _check_top_k_truncation,
        _check_is_bad_excluded,
        _check_workspace_filter,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_recall] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
