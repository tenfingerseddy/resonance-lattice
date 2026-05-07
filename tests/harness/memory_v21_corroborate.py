"""memory_v21_corroborate — §4.5 dedupe contracts (Appendix D D.3).

Pins four guarantees on `memory.distil.distil` corroboration:

  (a) Cosine ≥ 0.92 + matching primary polarity → recurrence++
      (existing row's recurrence_count goes up by exactly 1; no new row).

  (b) `last_corroborated_at` updated on the existing row.

  (c) Cosine < 0.92 → new row written; existing rows untouched.

  (d) Polarity mismatch (primary differs) → new row written even on
      text/embedding similarity ≥ 0.92.

Hermetic — uses a planted-embedding `FixedEncoder` so cosines are
exact.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import (
    FixedEncoder,
    make_stub_llm_client,
    patch_zero_encoder,
    seed_capture_memory,
)


def _planted_embeddings(target_cosine: float, dim: int = 768, seed: int = 7):
    """Return `(existing_emb, candidate_emb)` such that
    `existing_emb @ candidate_emb == target_cosine`. Both unit length.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim).astype("float32")
    base /= np.linalg.norm(base)
    ortho = rng.standard_normal(dim).astype("float32")
    ortho -= (ortho @ base) * base
    ortho /= np.linalg.norm(ortho)
    candidate = target_cosine * base + (1 - target_cosine ** 2) ** 0.5 * ortho
    candidate /= np.linalg.norm(candidate)
    return base, candidate


def _seed_capture(memory, *, transcript_hash: str = "newhash"):
    seed_capture_memory(memory, [{
        "text": "user said something noteworthy in this session",
        "transcript_hash": transcript_hash,
    }])


# ---------------------------------------------------------------------------
# (a) + (b) Cosine ≥ 0.92 + matching primary → recurrence++ + last_corroborated
# ---------------------------------------------------------------------------


def _check_corroboration_path() -> int:
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        cwd_tag = workspace_tag_for_cwd("/proj")
        existing_emb, candidate_emb = _planted_embeddings(target_cosine=0.95)

        existing_id = memory.add_row(
            text="the user prefers pytest -xvs for debugging",
            polarity=["prefer", cwd_tag],
            transcript_hash="prevhash",
            embedding=existing_emb,
        )
        memory.update_row(existing_id, last_corroborated_at="2025-01-01T00:00:00Z")
        rows_before, _ = memory.read_all()
        existing = next(r for r in rows_before if r.row_id == existing_id)
        recurrence_before = existing.recurrence_count
        last_before = existing.last_corroborated_at

        _seed_capture(memory, transcript_hash="newhash")

        canned = json.dumps([{
            "text": "the user prefers pytest -xvs for debugging tests",
            "intent": "debug a failing test",
            "polarity": ["prefer", "cross-workspace"],
            "rationale": "habit",
        }])

        result = distil(
            store=memory,
            redactor=Redactor(),
            client=make_stub_llm_client(canned),
            encoder=FixedEncoder(candidate_emb),
            workspace_path="/proj",
            all_rows=True,
        )

        if result.written_row_ids:
            print(f"[memory_v21_corroborate] FAIL (a): expected no new row "
                  f"on cosine≥0.92 match; got {result.written_row_ids}",
                  file=sys.stderr)
            return 1
        if existing_id not in result.corroborated_row_ids:
            print(f"[memory_v21_corroborate] FAIL (a): existing row "
                  f"{existing_id} not in corroborated set "
                  f"{result.corroborated_row_ids}", file=sys.stderr)
            return 1

        rows_after, _ = memory.read_all()
        bumped = next(r for r in rows_after if r.row_id == existing_id)
        if bumped.recurrence_count != recurrence_before + 1:
            print(f"[memory_v21_corroborate] FAIL (a): recurrence "
                  f"{recurrence_before} → {bumped.recurrence_count}, "
                  f"expected +1", file=sys.stderr)
            return 1
        if bumped.last_corroborated_at == last_before:
            print(f"[memory_v21_corroborate] FAIL (b): "
                  f"last_corroborated_at not updated "
                  f"({last_before!r} unchanged)", file=sys.stderr)
            return 1
    print("[memory_v21_corroborate] (a) cosine≥0.92 + matching primary "
          "→ recurrence++ OK", file=sys.stderr)
    print("[memory_v21_corroborate] (b) last_corroborated_at updated OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) Cosine < 0.92 → new row, existing untouched
# ---------------------------------------------------------------------------


def _check_below_threshold_writes_new() -> int:
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        cwd_tag = workspace_tag_for_cwd("/proj")
        # Cosine just below the §4.5 threshold.
        existing_emb, candidate_emb = _planted_embeddings(target_cosine=0.91)

        existing_id = memory.add_row(
            text="the user prefers pytest -xvs for debugging",
            polarity=["prefer", cwd_tag],
            transcript_hash="prevhash",
            embedding=existing_emb,
        )
        rows_before, _ = memory.read_all()
        existing = next(r for r in rows_before if r.row_id == existing_id)
        recurrence_before = existing.recurrence_count

        _seed_capture(memory, transcript_hash="newhash")

        canned = json.dumps([{
            "text": "user uses python -c for one-off snippets",
            "intent": "snippet quick check",
            "polarity": ["prefer", "cross-workspace"],
            "rationale": "different habit",
        }])
        result = distil(
            store=memory,
            redactor=Redactor(),
            client=make_stub_llm_client(canned),
            encoder=FixedEncoder(candidate_emb),
            workspace_path="/proj",
            all_rows=True,
        )

        if not result.written_row_ids:
            print(f"[memory_v21_corroborate] FAIL (c): cosine 0.91 < 0.92 "
                  f"should write new; got nothing", file=sys.stderr)
            return 1
        if existing_id in result.corroborated_row_ids:
            print(f"[memory_v21_corroborate] FAIL (c): existing row "
                  f"corroborated despite cosine 0.91", file=sys.stderr)
            return 1
        rows_after, _ = memory.read_all()
        unchanged = next(r for r in rows_after if r.row_id == existing_id)
        if unchanged.recurrence_count != recurrence_before:
            print(f"[memory_v21_corroborate] FAIL (c): existing "
                  f"recurrence inflated from {recurrence_before} to "
                  f"{unchanged.recurrence_count}", file=sys.stderr)
            return 1
    print("[memory_v21_corroborate] (c) cosine<0.92 → new row, existing "
          "untouched OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) Polarity mismatch → new row even on cosine ≥ 0.92
# ---------------------------------------------------------------------------


def _check_polarity_mismatch_writes_new() -> int:
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        cwd_tag = workspace_tag_for_cwd("/proj")
        existing_emb, candidate_emb = _planted_embeddings(target_cosine=0.97)

        existing_id = memory.add_row(
            text="the user prefers pytest -xvs",
            polarity=["avoid", cwd_tag],
            transcript_hash="prevhash",
            embedding=existing_emb,
        )
        rows_before, _ = memory.read_all()
        existing = next(r for r in rows_before if r.row_id == existing_id)
        recurrence_before = existing.recurrence_count

        _seed_capture(memory, transcript_hash="newhash")

        # Same-text candidate but `prefer` polarity (vs existing
        # `avoid`). High cosine should NOT corroborate — primary
        # polarity mismatch makes the rows semantically independent.
        canned = json.dumps([{
            "text": "the user prefers pytest -xvs",
            "intent": "preference",
            "polarity": ["prefer", "cross-workspace"],
            "rationale": "matches text but flipped polarity",
        }])
        result = distil(
            store=memory,
            redactor=Redactor(),
            client=make_stub_llm_client(canned),
            encoder=FixedEncoder(candidate_emb),
            workspace_path="/proj",
            all_rows=True,
        )

        if not result.written_row_ids:
            print(f"[memory_v21_corroborate] FAIL (d): polarity-mismatch "
                  f"candidate should write new row; got nothing",
                  file=sys.stderr)
            return 1
        if existing_id in result.corroborated_row_ids:
            print(f"[memory_v21_corroborate] FAIL (d): existing avoid row "
                  f"corroborated by prefer candidate", file=sys.stderr)
            return 1
        rows_after, _ = memory.read_all()
        unchanged = next(r for r in rows_after if r.row_id == existing_id)
        if unchanged.recurrence_count != recurrence_before:
            print(f"[memory_v21_corroborate] FAIL (d): avoid row recurrence "
                  f"inflated to {unchanged.recurrence_count}", file=sys.stderr)
            return 1
        # Confirm a fresh prefer row exists.
        primaries = [
            next(t for t in r.polarity if t in {"prefer", "avoid", "factual"})
            for r in rows_after
        ]
        if "prefer" not in primaries:
            print(f"[memory_v21_corroborate] FAIL (d): no prefer row "
                  f"after polarity-mismatch write; primaries={primaries}",
                  file=sys.stderr)
            return 1
    print("[memory_v21_corroborate] (d) polarity mismatch → new row even "
          "on cosine≥0.92 OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_corroboration_path,
        _check_below_threshold_writes_new,
        _check_polarity_mismatch_writes_new,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_corroborate] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
