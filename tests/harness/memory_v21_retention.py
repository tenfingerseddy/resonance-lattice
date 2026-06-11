"""memory_v21_retention — §0.5 + Appendix D D.4 contracts.

Pins four invariants on the retention surface (gc as the only deletion
path; recurrence-driven retention; bad-vote rows preserved for
re-distil suppression):

  (a) §0.6 recurrence gate honoured. A row with `recurrence_count <
      M` (default M=3) is dropped from `recall()` even if it clears
      every other gate. This is the same invariant memory_v21_recall
      (c) tests, framed here from the retention angle: low-recurrence
      rows naturally stay below the injection gate, no automatic
      deletion required.

  (b) `gc --max-age-days N` removes rows whose `last_corroborated_at`
      is older than N days. Per §15.2 the age clock indexes
      `last_corroborated_at` (not `created_at`) so a row that
      corroborates again resets its eligibility window — the rule is
      "things with recurrence_count == 1 that haven't seen a new
      corroboration in a month."

  (c) gc never removes is_bad rows by default (kept for re-distil
      suppression per §0.5). With `--is-bad`, gc targets only
      is_bad rows. This is the §0.5 manual-escape-hatch contract:
      every other filter (`--polarity`, `--min-recurrence`,
      `--max-age-days`) skips is_bad rows unless `--is-bad` is also
      passed.

  (d) gc with `--max-age-days` skips a row whose `last_corroborated_at`
      is recent — even if `created_at` is older than the horizon.
      Corroboration resets the clock per §15.2.

Hermetic: time-injected sidecar fixture (we hand-write `created_at` /
`last_corroborated_at` ISO strings to fast-forward / rewind the clock
without `time.sleep`).
"""

from __future__ import annotations

import datetime
import sys
import tempfile
from pathlib import Path


def _days_ago_iso(days: int) -> str:
    """An ISO `last_corroborated_at` `days` before now — relative so the
    fixture never crosses the gc age horizon as wall-clock advances (a
    hardcoded date here is a time-bomb)."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

import numpy as np

from ._testutil import patch_zero_encoder, run_cli


_SEED_COUNTER = [0]


def _seed_row(memory, *, text: str, primary: str = "factual",
              recurrence: int = 5,
              created_at: str = "2026-04-01T00:00:00Z",
              last_corroborated_at: str | None = None,
              is_bad: bool = False,
              transcript_hash: str = "manual") -> str:
    """Build one experience `Claim` at a hand-picked age + recurrence +
    is_bad configuration and write it. Used by every D.4 case.

    `created_at` is set directly at construction — it's immutable on
    `claim.evolve`, so there is no post-write mutation to fast-forward
    the clock (the old sidecar-rewrite hack is gone).
    """
    from resonance_lattice.memory.store import seed_tallies_for_rung
    from resonance_lattice.state.claim import Claim, ExperienceFacts, derive_origin

    _SEED_COUNTER[0] += 1
    corr, fals = seed_tallies_for_rung("low" if is_bad else "medium")
    claim = Claim(
        claim_id=f"01HZRETENTIONFIXTURE{_SEED_COUNTER[0]:06d}",
        source="experience",
        kind="event",
        content=text,
        created_at=created_at,
        corroboration=corr,
        falsification=fals,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=(primary, "workspace:abc123"),
            recurrence_count=recurrence,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash=transcript_hash,
            origin=derive_origin(transcript_hash),
            last_corroborated_at=last_corroborated_at or created_at,
            is_bad=is_bad,
        ),
    )
    memory.write(claim, embedding=np.zeros(768, dtype=np.float32))
    return claim.claim_id


# ---------------------------------------------------------------------------
# (a) §0.6 recurrence gate honoured
# ---------------------------------------------------------------------------


def _check_recurrence_gate() -> int:
    from resonance_lattice.memory._common import workspace_hash
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.recall import recall
    from ._testutil import FixedEncoder

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        query_vec = np.zeros(768, dtype=np.float32)
        query_vec[0] = 1.0
        memory = ExperienceClaimStore(root=root, encoder=FixedEncoder(query_vec))

        # Two claims with cosine well above floor + adequate gap; one with
        # recurrence below M, one above. Only the high-recurrence claim
        # should survive the §0.6 gates.
        for i, recurrence in enumerate([1, 5]):
            emb = np.zeros(768, dtype=np.float32)
            emb[0] = 0.9 - 0.1 * i  # cosines: 0.9, 0.8 → gap ≥ 0.05
            emb[1] = float(np.sqrt(max(0.0, 1.0 - emb[0] * emb[0])))
            _seed_row(
                memory, text=f"row {i} recurrence={recurrence}",
                recurrence=recurrence, transcript_hash=f"manual-{i}",
            )
        # Re-write the band vectors so the planted cosines hold — the
        # _seed_row default embedding is a zero vector.
        claims = memory.read_all()
        embs = np.zeros((2, 768), dtype=np.float32)
        for i in range(2):
            embs[i, 0] = 0.9 - 0.1 * i
            embs[i, 1] = float(np.sqrt(max(0.0, 1.0 - embs[i, 0] ** 2)))
        memory.write_many(claims, embeddings=embs)

        hits = recall("anything", store=memory, cwd_hash="abc123", top_k=10)
    if len(hits) != 1 or hits[0].claim.facts.recurrence_count != 5:
        print(f"[memory_v21_retention] FAIL (a): expected 1 hit with "
              f"recurrence=5; got "
              f"{[(h.claim.facts.recurrence_count, h.cosine) for h in hits]}",
              file=sys.stderr)
        return 1
    print("[memory_v21_retention] (a) §0.6 recurrence gate drops below-M "
          "row OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) gc --max-age-days removes by last_corroborated_at
# ---------------------------------------------------------------------------


def _check_gc_max_age() -> int:
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        memory = ExperienceClaimStore(root=base / "u", encoder=None)
        old_id = _seed_row(
            memory, text="stale row",
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at="2025-01-01T00:00:00Z",
        )
        recent = _days_ago_iso(5)
        recent_id = _seed_row(
            memory, text="recent row",
            created_at=recent,
            last_corroborated_at=recent,
        )

        rc, out, err = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--max-age-days", "30",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (b): gc rc={rc}\n"
                  f"out:{out}\nerr:{err}", file=sys.stderr)
            return 1
        claims = memory.read_all()
        ids_left = {c.claim_id for c in claims}
    if old_id in ids_left:
        print(f"[memory_v21_retention] FAIL (b): old row {old_id} should "
              f"have been deleted by --max-age-days 30", file=sys.stderr)
        return 1
    if recent_id not in ids_left:
        print(f"[memory_v21_retention] FAIL (b): recent row {recent_id} "
              f"should have been preserved", file=sys.stderr)
        return 1
    print("[memory_v21_retention] (b) gc --max-age-days removes by "
          "last_corroborated_at OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) gc skips is_bad rows by default; --is-bad targets only is_bad rows
# ---------------------------------------------------------------------------


def _check_gc_isbad_preservation() -> int:
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        memory = ExperienceClaimStore(root=base / "u", encoder=None)
        # Both rows are old + low-recurrence + same polarity, but one
        # is_bad. A `--max-age-days 30 --polarity factual` sweep should
        # delete the not-bad row and PRESERVE the is_bad row.
        bad_id = _seed_row(
            memory, text="bad-voted row", recurrence=1, is_bad=True,
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at="2025-01-01T00:00:00Z",
        )
        normal_id = _seed_row(
            memory, text="normal stale row", recurrence=1, is_bad=False,
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at="2025-01-01T00:00:00Z",
        )

        rc, out, err = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--max-age-days", "30",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (c): default gc rc={rc}\n"
                  f"out:{out}\nerr:{err}", file=sys.stderr)
            return 1
        claims = memory.read_all()
        ids_after_default = {c.claim_id for c in claims}
        if bad_id not in ids_after_default:
            print(f"[memory_v21_retention] FAIL (c): default gc removed "
                  f"is_bad row {bad_id}; bad rows must be preserved unless "
                  f"--is-bad is passed", file=sys.stderr)
            return 1
        if normal_id in ids_after_default:
            print(f"[memory_v21_retention] FAIL (c): default gc failed to "
                  f"remove normal stale row {normal_id}", file=sys.stderr)
            return 1

        # --is-bad now targets ONLY the is_bad row.
        rc, out, err = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--is-bad",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (c): --is-bad gc rc={rc}\n"
                  f"out:{out}\nerr:{err}", file=sys.stderr)
            return 1
        claims = memory.read_all()
        ids_after_isbad = {c.claim_id for c in claims}
        if bad_id in ids_after_isbad:
            print(f"[memory_v21_retention] FAIL (c): --is-bad gc failed to "
                  f"remove is_bad row {bad_id}", file=sys.stderr)
            return 1
    print("[memory_v21_retention] (c) gc skips is_bad by default + "
          "--is-bad targets only is_bad OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) gc --max-age-days skips rows recently corroborated
# ---------------------------------------------------------------------------


def _check_gc_corroboration_resets_clock() -> int:
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        memory = ExperienceClaimStore(root=base / "u", encoder=None)
        # Created long ago BUT corroborated yesterday — gc must not
        # remove this row even though `created_at` is well past the
        # horizon. The clock is `last_corroborated_at` per §15.2.
        kept_id = _seed_row(
            memory, text="old but recently corroborated",
            recurrence=4,
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at=_days_ago_iso(5),
        )

        rc, _, _ = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--max-age-days", "30",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (d): gc rc={rc}", file=sys.stderr)
            return 1
        claims = memory.read_all()
        if kept_id not in {c.claim_id for c in claims}:
            print(f"[memory_v21_retention] FAIL (d): row {kept_id} with "
                  f"recent last_corroborated_at was deleted by --max-age-days; "
                  f"corroboration must reset the clock", file=sys.stderr)
            return 1
    print("[memory_v21_retention] (d) corroboration resets the gc age clock "
          "OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_recurrence_gate,
        _check_gc_max_age,
        _check_gc_isbad_preservation,
        _check_gc_corroboration_resets_clock,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_retention] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
