"""memory_v22_consolidation — the session-end orchestrator.

`consolidation_pass` sequences confidence raising → forget. Each stage
has its own focused suite (`memory_v22_confidence`, `memory_v22_forget`);
this suite pins the orchestrator wiring.

  (a) `state_root=None` skips confidence raising; forget still runs.

  (b) `dry_run=True` runs both stages but writes nothing to the store.

Hermetic — `seed_capture_memory` + ZeroEncoder.

Guarantee (c): consolidation_pass forwards state_root into
raise_confidence_pass so implicit corroboration (M3) is reachable in
production (2026-06 review regression guard).
"""

from __future__ import annotations

import datetime as _dt
import sys
import tempfile
from pathlib import Path

from ._testutil import patch_zero_encoder, seed_capture_memory

# `seed_capture_memory` stamps captures at 2026-05-08; pin `now` one day
# later so forget's wall-clock age math is deterministic.
_FIXED_NOW = _dt.datetime(2026, 5, 9, tzinfo=_dt.timezone.utc)

_CAPTURES = [
    {"text": "user invoked pytest -xvs three times in a row",
     "transcript_hash": "cap-a-0001"},
    {"text": "user said avoid wildcard imports in main.py",
     "transcript_hash": "cap-b-0002"},
]


def _build_capture_store(td: Path):
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    memory = ExperienceClaimStore(root=td / "u", encoder=None)
    seed_capture_memory(memory, _CAPTURES, workspace_path="/proj")
    return memory


def _check_no_state_root_skips_confidence() -> int:
    from resonance_lattice.memory.session_end_pass import consolidation_pass

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        result = consolidation_pass(
            memory, state_root=None, now=_FIXED_NOW,
        )
        if result.confidence_changes:
            print(f"[memory_v22_consolidation] FAIL (a): confidence pass ran "
                  f"without state_root: {result.confidence_changes}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_consolidation] (a) state_root=None skips confidence "
          "raising OK", file=sys.stderr)
    return 0


def _check_dry_run() -> int:
    from resonance_lattice.memory.session_end_pass import consolidation_pass

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        before = [c.claim_id for c in memory.read_all()]
        consolidation_pass(
            memory, state_root=None, dry_run=True, now=_FIXED_NOW,
        )
        after = [c.claim_id for c in memory.read_all()]
        if after != before:
            print(f"[memory_v22_consolidation] FAIL (b): dry_run mutated "
                  f"store; before={before} after={after}", file=sys.stderr)
            return 1
    print("[memory_v22_consolidation] (b) dry_run writes nothing OK",
          file=sys.stderr)
    return 0


def _check_state_root_forwarded() -> int:
    """(c) consolidation_pass must forward state_root into
    raise_confidence_pass — without it the recall-cache read inside is
    gated off and implicit corroboration (M3) silently never fires
    (2026-06 review: every production consolidate had run with M3 off).
    Captures the kwargs via monkeypatch; no behavioural simulation needed.
    """
    from resonance_lattice.memory import session_end_pass

    seen: dict = {}
    real = session_end_pass.raise_confidence_pass

    def _spy(memory, **kwargs):
        seen.update(kwargs)
        return real(memory, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        state_root = Path(td) / "state"
        state_root.mkdir()
        session_end_pass.raise_confidence_pass = _spy
        try:
            session_end_pass.consolidation_pass(
                memory, state_root=state_root, now=_FIXED_NOW,
            )
        finally:
            session_end_pass.raise_confidence_pass = real
    if seen.get("state_root") != state_root:
        print(f"[memory_v22_consolidation] FAIL (c): state_root not forwarded "
              f"to raise_confidence_pass (got {seen.get('state_root')!r})",
              file=sys.stderr)
        return 1
    print("[memory_v22_consolidation] (c) state_root forwarded to confidence "
          "pass (M3 reachable) OK", file=sys.stderr)
    return 0


def run() -> int:
    patch_zero_encoder()
    for check in (_check_no_state_root_skips_confidence, _check_dry_run,
                  _check_state_root_forwarded):
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_consolidation] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
