"""memory_v21_distil — §7 distil pipeline contracts (Appendix D D.2).

Pins five guarantees on `memory.distil.distil`:

  (a) Malformed JSON output → run logged + skipped, no partial write,
      watermark does NOT advance (§7.9 row 1).

  (b) Credential pattern in candidate text → row rejected at write time
      (Layer-1 redactor fires; candidate dropped before the
      `Memory.add_row` path).

  (c) Duplicate `(text_normalised, transcript_hash)` does not increment
      recurrence twice — re-running with the same canned response on
      the same captured rows is idempotent.

  (d) `workspace_hash` interpolated correctly into polarity tags —
      orchestrator threads the cwd hash through, not the raw
      `{workspace_hash}` placeholder.

  (e) Empty-output `[]` is a valid no-op — distiller returning `[]`
      writes nothing, advances the watermark, returns DistilResult
      with empty written/corroborated lists.

Hermetic — uses `LLMResponseStub` for the LLM seam, ZeroEncoder for
the embedding seam.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ._testutil import make_stub_llm_client, patch_zero_encoder, seed_capture_memory


def _build_capture_store(td: Path):
    from resonance_lattice.memory.store import Memory

    memory = Memory(root=td / "u")
    seed_capture_memory(memory, [
        {"text": "user invoked pytest -xvs three times in a row",
         "transcript_hash": "abc123"},
        {"text": "user said avoid wildcard imports in main.py",
         "transcript_hash": "def456"},
    ])
    return memory


# ---------------------------------------------------------------------------
# (a) Malformed JSON
# ---------------------------------------------------------------------------


def _check_malformed_json() -> int:
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        client = make_stub_llm_client("this is plain text, not a JSON array")
        before_rows, _ = memory.read_all()

        result = distil(store=memory, redactor=Redactor(), client=client,
                        workspace_path="/proj", all_rows=True)
        if result.written_row_ids or result.corroborated_row_ids:
            print(f"[memory_v21_distil] FAIL (a): malformed JSON produced "
                  f"writes: {result}", file=sys.stderr)
            return 1
        if result.new_watermark_utc is not None:
            print(f"[memory_v21_distil] FAIL (a): watermark advanced on "
                  f"malformed JSON: {result.new_watermark_utc}", file=sys.stderr)
            return 1
        after_rows, _ = memory.read_all()
        if len(after_rows) != len(before_rows):
            print(f"[memory_v21_distil] FAIL (a): row count changed "
                  f"({len(before_rows)} → {len(after_rows)})", file=sys.stderr)
            return 1
    print("[memory_v21_distil] (a) malformed JSON skipped + watermark preserved OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) Credential pattern rejected
# ---------------------------------------------------------------------------


def _check_credential_rejected() -> int:
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        # Candidate carries an Anthropic key inside the text — the
        # distiller's privacy contract says the LLM should refuse, but
        # we test the orchestrator's belt-and-suspenders Layer-1 scrub
        # by feeding it through anyway.
        leaky = json.dumps([{
            "text": "the user keeps sk-ant-" + "A" * 50 + " in main.py",
            "intent": "audit credentials",
            "polarity": ["factual", "cross-workspace"],
            "rationale": "credential leak",
        }])
        result = distil(store=memory, redactor=Redactor(), client=make_stub_llm_client(leaky),
                        workspace_path="/proj", all_rows=True)
        if result.written_row_ids:
            print(f"[memory_v21_distil] FAIL (b): credential-shaped row "
                  f"written: {result}", file=sys.stderr)
            return 1
        if result.skipped_count != 1:
            print(f"[memory_v21_distil] FAIL (b): expected skipped=1 from "
                  f"redactor; got {result.skipped_count}", file=sys.stderr)
            return 1
        # Verify the secret didn't land on disk anywhere.
        rows, _ = memory.read_all()
        for r in rows:
            if "sk-ant-" in r.text:
                print(f"[memory_v21_distil] FAIL (b): row text contains "
                      f"sk-ant-: {r.text!r}", file=sys.stderr)
                return 1
    print("[memory_v21_distil] (b) credential candidate rejected before write OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) Re-run idempotency on same transcript
# ---------------------------------------------------------------------------


def _check_idempotent_rerun() -> int:
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        canned = json.dumps([{
            "text": "the user prefers pytest -xvs for debugging",
            "intent": "debug pytest",
            "polarity": ["prefer", "cross-workspace"],
            "rationale": "habit",
        }])
        client = make_stub_llm_client(canned)

        first = distil(store=memory, redactor=Redactor(), client=client,
                       workspace_path="/proj", all_rows=True)
        if len(first.written_row_ids) != 1:
            print(f"[memory_v21_distil] FAIL (c): expected 1 write on "
                  f"first run; got {first.written_row_ids}", file=sys.stderr)
            return 1
        new_row_id = first.written_row_ids[0]
        rows1, _ = memory.read_all()
        recurrence_before = next(r.recurrence_count for r in rows1
                                  if r.row_id == new_row_id)

        # Second run with the same client + same captured rows. Per (c),
        # the candidate's transcript_hash matches the existing row's
        # transcript_hash → silent no-op (no recurrence bump, no new
        # row).
        second = distil(store=memory, redactor=Redactor(), client=client,
                        workspace_path="/proj", all_rows=True)
        if second.written_row_ids:
            print(f"[memory_v21_distil] FAIL (c): second run wrote a "
                  f"duplicate row: {second.written_row_ids}", file=sys.stderr)
            return 1
        rows2, _ = memory.read_all()
        recurrence_after = next(r.recurrence_count for r in rows2
                                 if r.row_id == new_row_id)
        if recurrence_after != recurrence_before:
            print(f"[memory_v21_distil] FAIL (c): recurrence bumped on "
                  f"same-transcript re-run ({recurrence_before} → "
                  f"{recurrence_after})", file=sys.stderr)
            return 1
    print("[memory_v21_distil] (c) same-transcript re-run is idempotent OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) workspace_hash interpolation
# ---------------------------------------------------------------------------


def _check_workspace_interpolation() -> int:
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        # Candidate uses the literal `{workspace_hash}` placeholder the
        # prompt teaches the model to emit. The orchestrator must
        # interpolate it before write.
        canned = json.dumps([{
            "text": "the project's commit cadence is simplify-codex-harness",
            "intent": "ship a clean commit",
            "polarity": ["prefer", "workspace:{workspace_hash}"],
            "rationale": "workspace-scoped cadence rule",
        }])
        result = distil(store=memory, redactor=Redactor(),
                        client=make_stub_llm_client(canned),
                        workspace_path="/proj", all_rows=True)
        if not result.written_row_ids:
            print(f"[memory_v21_distil] FAIL (d): expected 1 write; got "
                  f"{result}", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        new_row = next(r for r in rows if r.row_id == result.written_row_ids[0])
        expected_tag = workspace_tag_for_cwd("/proj")
        if expected_tag not in new_row.polarity:
            print(f"[memory_v21_distil] FAIL (d): expected {expected_tag} "
                  f"in polarity; got {new_row.polarity}", file=sys.stderr)
            return 1
        if any("{workspace_hash}" in tag for tag in new_row.polarity):
            print(f"[memory_v21_distil] FAIL (d): unresolved placeholder "
                  f"in polarity: {new_row.polarity}", file=sys.stderr)
            return 1
    print("[memory_v21_distil] (d) workspace_hash interpolated into "
          "polarity OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (e) Empty `[]` is a valid no-op
# ---------------------------------------------------------------------------


def _check_empty_array_noop() -> int:
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor

    with tempfile.TemporaryDirectory() as td:
        memory = _build_capture_store(Path(td))
        before, _ = memory.read_all()
        result = distil(store=memory, redactor=Redactor(),
                        client=make_stub_llm_client("[]"),
                        workspace_path="/proj", all_rows=True)
        if result.written_row_ids or result.corroborated_row_ids:
            print(f"[memory_v21_distil] FAIL (e): empty [] produced "
                  f"writes: {result}", file=sys.stderr)
            return 1
        if result.new_watermark_utc is None:
            print(f"[memory_v21_distil] FAIL (e): empty [] should advance "
                  f"watermark (success no-op)", file=sys.stderr)
            return 1
        after, _ = memory.read_all()
        if len(after) != len(before):
            print(f"[memory_v21_distil] FAIL (e): row count changed on "
                  f"empty [] response", file=sys.stderr)
            return 1
    print("[memory_v21_distil] (e) empty [] is a valid no-op + watermark "
          "advances OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (f) journal idempotency on cosine-corroborate path (codex P1.2)
# ---------------------------------------------------------------------------


def _check_journal_idempotent_after_corroborate() -> int:
    """Codex P1.2 regression: when distil run 1 cosine-corroborates an
    existing row (different text) and overwrites its `transcript_hash`,
    run 2 over the same source transcript must not bump recurrence
    again. The pre-fix text-equality heuristic missed this case
    because the corroborated row's text never matched the candidate
    text."""
    import numpy as np
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory
    from ._testutil import FixedEncoder

    with tempfile.TemporaryDirectory() as td:
        # FixedEncoder so candidate + seed row both end up at the same
        # unit vector → cosine 1.0, well above the 0.92 dedupe gate.
        # ZeroEncoder produces all-zero embeddings whose dot product is
        # 0, which would fall below the gate and never corroborate.
        planted = np.zeros(768, dtype="float32")
        planted[0] = 1.0
        memory = Memory(root=Path(td) / "u", encoder=FixedEncoder(planted))

        cwd_tag = workspace_tag_for_cwd("/proj")
        seed_text = "always run pytest -xvs over plain pytest"
        existing_row_id = memory.add_row(
            text=seed_text,
            polarity=["prefer", cwd_tag],
            transcript_hash="manual",
            embedding=planted.copy(),
        )

        seed_capture_memory(memory, [
            {"text": "transcript content for run 1",
             "transcript_hash": "src-T1"},
        ])
        canned = json.dumps([{
            "text": "the user prefers pytest -xvs for debugging",
            "intent": "debug pytest",
            "polarity": ["prefer", "cross-workspace"],
            "rationale": "habit",
        }])
        client = make_stub_llm_client(canned)

        # Run 1: corroborates the existing row (cosine path), bumps
        # recurrence, overwrites transcript_hash.
        first = distil(store=memory, redactor=Redactor(), client=client,
                       workspace_path="/proj", all_rows=True)
        if existing_row_id not in first.corroborated_row_ids:
            print(f"[memory_v21_distil] FAIL (f.1): expected first run to "
                  f"corroborate seed row {existing_row_id}; got {first}",
                  file=sys.stderr)
            return 1
        rows1, _ = memory.read_all()
        rec_after_run1 = next(
            r.recurrence_count for r in rows1 if r.row_id == existing_row_id
        )
        if rec_after_run1 != 2:
            print(f"[memory_v21_distil] FAIL (f.1): expected recurrence=2 "
                  f"after corroboration; got {rec_after_run1}", file=sys.stderr)
            return 1

        # Run 2: same canned response over the same source transcript.
        # Pre-fix, the text-equality check missed (seed_text was
        # untouched, candidate text was different) → re-bumped to 3.
        # With the journal, `(existing_row_id, distilled:src-T1)` is in
        # seen_pairs → no-op.
        second = distil(store=memory, redactor=Redactor(), client=client,
                        workspace_path="/proj", all_rows=True)
        if second.written_row_ids or second.corroborated_row_ids:
            print(f"[memory_v21_distil] FAIL (f.2): re-run produced "
                  f"writes/corroborations: {second}", file=sys.stderr)
            return 1
        rows2, _ = memory.read_all()
        rec_after_run2 = next(
            r.recurrence_count for r in rows2 if r.row_id == existing_row_id
        )
        if rec_after_run2 != rec_after_run1:
            print(f"[memory_v21_distil] FAIL (f.2): recurrence drifted on "
                  f"re-run ({rec_after_run1} → {rec_after_run2}); journal "
                  f"per-pair guard failed", file=sys.stderr)
            return 1
    print("[memory_v21_distil] (f) journal blocks double-bump on cosine-"
          "corroborate re-run OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_malformed_json,
        _check_credential_rejected,
        _check_idempotent_rerun,
        _check_workspace_interpolation,
        _check_empty_array_noop,
        _check_journal_idempotent_after_corroborate,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_distil] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
