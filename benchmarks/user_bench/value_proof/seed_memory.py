"""Seed the value-proof scratch memory store with real, accumulated lessons.

The value-proof needs LOAD-BEARING, REAL personal state (value-proof.md rule 1).
The organically-accumulated product store holds only raw recurrence-1 events the
recall gates drop (see stage1-status.md §C premise gate), so we import a curated
set of genuine, distilled rlat-dev lessons (lessons.jsonl — drawn from the repo's
own cross-session auto-memory) into a SCRATCH user store, leaving the real store
untouched. These are the user's real earned lessons, not invented-to-match text;
the fixture is authored independently and the specificity control is the anti-rig.

Claims are minted with recurrence_count >= 2 and a `cross-workspace` polarity tag
so the real recall path surfaces them from any cwd.

Usage:
    PYTHONUTF8=1 python -m benchmarks.user_bench.value_proof.seed_memory --reset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent
_SRC = _REPO / "src"
if (_SRC / "resonance_lattice" / "__init__.py").exists():
    sys.path.insert(0, str(_SRC))

SCRATCH_USER = "value_proof_scratch"
_LESSONS = _HERE.parent / "lessons.jsonl"


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed(*, reset: bool, recurrence: int = 3, user_id: str = SCRATCH_USER,
         lesson_ids: list[str] | None = None) -> int:
    from resonance_lattice.memory.claim_store import (
        ExperienceClaimStore,
        new_experience_claim,
    )
    from resonance_lattice.memory.store import path_for_user

    store_dir = path_for_user(user_id)
    if reset and store_dir.exists():
        shutil.rmtree(store_dir)
        print(f"[seed] reset {store_dir}")

    lessons = [
        json.loads(line)
        for line in _LESSONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if lesson_ids is not None:
        wanted = set(lesson_ids)
        lessons = [lz for lz in lessons if lz["id"] in wanted]

    store = ExperienceClaimStore(user_id=user_id)
    claims = [
        new_experience_claim(
            content=lz["content"],
            polarity=(lz["polarity"], "cross-workspace"),
            transcript_hash=_h(lz["id"] + lz["content"]),
            kind="event",
            rung="high",  # seed Beta high — these are established lessons
            recurrence_count=recurrence,
            criticality=lz.get("criticality", "normal"),
        )
        for lz in lessons
    ]
    store.write_many(claims)  # auto-encodes content -> band
    print(f"[seed] wrote {len(claims)} lessons to {store_dir} (user={user_id})")

    # Self-test: recall must surface lessons for a load-bearing query.
    from resonance_lattice.memory.recall import recall

    probe = "how should I run a large encoding benchmark"
    hits = recall(probe, store=store, top_k=5, auto_tune_cold_start=True)
    print(f"[seed] recall self-test '{probe}': {len(hits)} hits")
    for hclaim in hits[:3]:
        print(f"   cos={hclaim.cosine:.3f}  {hclaim.claim.content[:70]}")
    if not hits:
        print("[seed] WARNING: recall returned 0 hits — lessons will not surface",
              file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reset", action="store_true",
                   help="wipe the scratch store before seeding")
    p.add_argument("--recurrence", type=int, default=3)
    p.add_argument("--user-id", default=SCRATCH_USER)
    args = p.parse_args(argv)
    return seed(reset=args.reset, recurrence=args.recurrence, user_id=args.user_id)


if __name__ == "__main__":
    sys.exit(main())
