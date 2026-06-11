"""curator_decide — the closed-form (arm-b) decide tier over PERSISTED telemetry.

CRITICAL_PATH Step 2. `curator.decide.decide(km_id)` reads the `.rlat`'s own
`insight/telemetry.jsonl` (the member Step 1 folds) and emits confirmed-recurring
INTENT candidates via the non-killed intent-cluster clause — never the killed
weak-score gap signal. Pins:

  (a) empty / no telemetry → no candidates.
  (b) no recurrence (a single query) → no candidates.
  (c) a recurring intent (asked ≥2×) → one candidate with the right occurrences
      and a representative L2-normalised centroid.
  (d) distinct intents → only the recurring cluster surfaces; a singleton is dropped.
  (e) min_sessions gate — a within-one-session recurrence is dropped at min_sessions=2.
  (f) insight-layer rows and non-user (machinery) rows never count toward recurrence.
  (g) never raises — None / bad path / ragged embeddings yield [].

  (GATE) the decide reads candidates from telemetry PHYSICALLY inside a real .rlat
      (appended via the Step 1 writer), emits the recurring-intent candidate, and
      its centroid points at the recurring query — arm (b) on real persisted data,
      no LLM, no network.

Hermetic: tiny real archives with synthetic bands; no encoder.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from resonance_lattice.curator import decide as decide_mod
from resonance_lattice.store import archive
from resonance_lattice.store import registry as registry_io
from resonance_lattice.store.metadata import BandInfo, Metadata


def _write_real_rlat(path: Path, n: int = 4, dim: int = 8) -> None:
    rng = np.random.default_rng(0)
    band = rng.standard_normal((n, dim)).astype("float32")
    band /= np.linalg.norm(band, axis=1, keepdims=True)
    registry = [
        registry_io.PassageCoord(
            passage_idx=i, source_file=f"d{i}.txt", char_offset=0,
            char_length=10, content_hash="sha256:0",
            passage_id=registry_io.compute_id(f"d{i}.txt", 0, 10),
        )
        for i in range(n)
    ]
    meta = Metadata(
        bands={"base": BandInfo(role="retrieval_default", dim=dim, passage_count=n)}
    )
    archive.write(path, meta, {"base": band}, registry)


def _row(emb, session="s1", layer="source", user=True, idx=0, score=0.8):
    return {
        "ts": "2026-06-03T00:00:00+00:00",
        "session": session,
        "layer": layer,
        "is_user_query": user,
        "query_emb": list(emb),
        "ranked": [{"rank": 0, "idx": idx, "score": score}],
    }


# Three intents, mutually orthogonal (cosine 0 < the 0.7 paraphrase floor → separate clusters).
_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_C = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _persisted(path: Path, rows) -> None:
    archive.append_telemetry_in_place(path, rows)


def _check_empty() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        if decide_mod.decide(str(p)) != []:
            print("[curator_decide] empty: expected [] on a corpus with no telemetry",
                  file=sys.stderr)
            return 1
    return 0


def _check_no_recurrence() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        _persisted(p, [_row(_A)])  # one query — not recurring
        if decide_mod.decide(str(p)) != []:
            print("[curator_decide] no_recurrence: a single query is not a candidate",
                  file=sys.stderr)
            return 1
    return 0


def _check_recurring() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        _persisted(p, [_row(_A, "s1"), _row(_A, "s2"), _row(_A, "s3")])
        cands = decide_mod.decide(str(p))
        if len(cands) != 1 or cands[0].occurrences != 3 or cands[0].distinct_sessions != 3:
            print(f"[curator_decide] recurring: expected 1 candidate occ=3 sess=3, "
                  f"got {cands}", file=sys.stderr)
            return 1
        c = cands[0].query_centroid
        if len(c) != len(_A) or abs(float(np.linalg.norm(c)) - 1.0) > 1e-3:
            print(f"[curator_decide] recurring: centroid not unit-norm: {c}",
                  file=sys.stderr)
            return 1
        # Centroid of three identical _A vectors is _A itself.
        if float(np.array(c) @ np.array(_A)) < 0.99:
            print(f"[curator_decide] recurring: centroid doesn't point at the intent: {c}",
                  file=sys.stderr)
            return 1
    return 0


def _check_distinct_intents() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        # _A recurs (×2 across 2 sessions); _B asked once.
        _persisted(p, [_row(_A, "s1"), _row(_A, "s2"), _row(_B, "s1")])
        cands = decide_mod.decide(str(p))
        if len(cands) != 1 or cands[0].occurrences != 2:
            print(f"[curator_decide] distinct: only the recurring intent should "
                  f"surface, got {cands}", file=sys.stderr)
            return 1
        if float(np.array(cands[0].query_centroid) @ np.array(_A)) < 0.99:
            print("[curator_decide] distinct: surfaced the wrong intent", file=sys.stderr)
            return 1
    return 0


def _check_min_sessions() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        _persisted(p, [_row(_A, "s1"), _row(_A, "s1")])  # recurs, but one session
        if decide_mod.decide(str(p), min_sessions=2) != []:
            print("[curator_decide] min_sessions: one-session recurrence must drop at "
                  "min_sessions=2", file=sys.stderr)
            return 1
        if len(decide_mod.decide(str(p), min_sessions=1)) != 1:
            print("[curator_decide] min_sessions: should surface at min_sessions=1",
                  file=sys.stderr)
            return 1
    return 0


def _check_excludes_noise() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        # One real user-source query for intent _A, plus its paired insight row and
        # an internal (machinery) row — neither must count toward recurrence.
        _persisted(p, [
            _row(_A, "s1", layer="source", user=True),
            _row(_A, "s1", layer="insight", user=True),   # paired insight obs
            _row(_A, "s2", layer="source", user=False),   # machinery raised its hand
        ])
        if decide_mod.decide(str(p)) != []:
            print("[curator_decide] noise: insight/non-user rows must not make an "
                  "intent look recurring", file=sys.stderr)
            return 1
    return 0


def _check_never_raises() -> int:
    if decide_mod.decide(None) != []:
        print("[curator_decide] never_raises: decide(None) should be []", file=sys.stderr)
        return 1
    if decide_mod.decide("/no/such/path.rlat") != []:
        print("[curator_decide] never_raises: decide(bad path) should be []",
              file=sys.stderr)
        return 1
    # Ragged embeddings must not crash the clause.
    ragged = [_row([1.0, 0.0], "s1"), _row([1.0, 0.0, 0.0], "s2")]
    try:
        decide_mod.recurring_intents(ragged)
    except Exception as e:  # noqa: BLE001
        print(f"[curator_decide] never_raises: ragged embeddings raised {e!r}",
              file=sys.stderr)
        return 1

    # Iterator-safe: a one-shot generator must not be consumed by the first clause
    # and silently drop a recurring intent.
    def _gen():
        yield _row(_A, "s1")
        yield _row(_A, "s2")
    if len(decide_mod.recurring_intents(_gen())) != 1:
        print("[curator_decide] never_raises: generator input dropped a recurring intent",
              file=sys.stderr)
        return 1
    return 0


def _check_coverage_gate() -> int:
    """The relative-undercoverage gate (gap × demand, the re-aim of the killed
    absolute signal): of two equally-recurring intents, only the one the corpus
    answers RELATIVELY WORSE (lower mean top-1 cosine) surfaces by default."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        # _A is UNDERcovered (top-1 0.40), _B is WELL covered (0.90); both recur ×2.
        _persisted(p, [
            _row(_A, "s1", score=0.40), _row(_A, "s2", score=0.40),
            _row(_B, "s1", score=0.90), _row(_B, "s2", score=0.90),
        ])
        cands = decide_mod.decide(str(p))  # default coverage_quantile=0.5
        if len(cands) != 1:
            print(f"[curator_decide] coverage_gate: expected only the undercovered "
                  f"intent, got {len(cands)}: {cands}", file=sys.stderr)
            return 1
        if float(np.array(cands[0].query_centroid) @ np.array(_A)) < 0.99:
            print("[curator_decide] coverage_gate: surfaced the well-covered intent, "
                  "not the undercovered one", file=sys.stderr)
            return 1
        if abs(cands[0].mean_top_score - 0.40) > 1e-3:
            print(f"[curator_decide] coverage_gate: mean_top_score wrong: "
                  f"{cands[0].mean_top_score}", file=sys.stderr)
            return 1
    return 0


def _check_coverage_gate_off() -> int:
    """`coverage_quantile=1.0` disables the gate — both recurring intents surface
    (pure-demand behaviour, the auto+override off-switch)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        _persisted(p, [
            _row(_A, "s1", score=0.40), _row(_A, "s2", score=0.40),
            _row(_B, "s1", score=0.90), _row(_B, "s2", score=0.90),
        ])
        cands = decide_mod.decide(str(p), coverage_quantile=1.0)
        if len(cands) != 2:
            print(f"[curator_decide] coverage_gate_off: quantile=1.0 must keep both "
                  f"recurring intents, got {len(cands)}", file=sys.stderr)
            return 1
    return 0


def _check_coverage_ordering() -> int:
    """Candidates are returned worst-covered first (largest gap deficit leads), so a
    budget-limited consumer fills the highest-value gaps."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "km.rlat"
        _write_real_rlat(p)
        _persisted(p, [
            _row(_A, "s1", score=0.30), _row(_A, "s2", score=0.30),  # worst covered
            _row(_C, "s1", score=0.50), _row(_C, "s2", score=0.50),  # middle
            _row(_B, "s1", score=0.70), _row(_B, "s2", score=0.70),  # best covered
        ])
        cands = decide_mod.decide(str(p), coverage_quantile=1.0)  # gate off → all three, ordered
        order = [round(c.mean_top_score, 2) for c in cands]
        if order != [0.30, 0.50, 0.70]:
            print(f"[curator_decide] coverage_ordering: expected worst-covered first "
                  f"[0.30,0.50,0.70], got {order}", file=sys.stderr)
            return 1
    return 0


def _check_gate() -> int:
    """Step 2 gate: candidate derived from telemetry physically inside a real .rlat."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "real.rlat"
        _write_real_rlat(p)
        rows = [_row(_A, "s1"), _row(_A, "s2"), _row(_B, "s3")]
        archive.append_telemetry_in_place(p, rows)
        # Confirm the rows are really in the file (not memory).
        if len(archive.read_telemetry(p)) != 3:
            print("[curator_decide] GATE: telemetry not persisted in the .rlat",
                  file=sys.stderr)
            return 1
        cands = decide_mod.decide(str(p))  # reads insight/telemetry.jsonl from the file
        if len(cands) != 1:
            print(f"[curator_decide] GATE: expected 1 recurring-intent candidate from "
                  f"in-.rlat telemetry, got {len(cands)}", file=sys.stderr)
            return 1
        cand = cands[0]
        ok = (
            cand.occurrences == 2
            and cand.distinct_sessions == 2
            and float(np.array(cand.query_centroid) @ np.array(_A)) > 0.99
        )
        if not ok:
            print(f"[curator_decide] GATE: candidate shape/centroid wrong: {cand}",
                  file=sys.stderr)
            return 1
    return 0


def run() -> int:
    for check in [
        _check_empty,
        _check_no_recurrence,
        _check_recurring,
        _check_distinct_intents,
        _check_min_sessions,
        _check_excludes_noise,
        _check_never_raises,
        _check_coverage_gate,
        _check_coverage_gate_off,
        _check_coverage_ordering,
        _check_gate,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[curator_decide] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
