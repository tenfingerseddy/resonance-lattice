"""insight_loop_e2e — the corpus-trust loop, end-to-end (S3 d3.4, the §D gate).

The REAL path, NO hand-seeded outcome. One contract, both directions:

  a real `.rlat` with an ACTIVE corpus insight claim
    → the live recall daemon ranks its insight band and the UserPromptSubmit
      hook stamps the corpus claim id (+ the active intent id) into the
      RecallCache  [recall → cache]
    → `rlat intent accept|reject` reads the cache, attributes the corpus claim,
      and writes a satisfied/not_satisfied intent outcome
      [cache → attribution → ledger]
    → `rlat consolidate-insights` folds it through the criterion reducer and the
      corpus claim's trust MOVES on disk — UP on accept (corroboration), DOWN on
      reject (falsification)  [ledger → criterion_weighted → apply]

This proves the loop the S4 keystone was inert for: an intent's attribution now
carries CORPUS ids (not just experience-store ids), so the criterion reducer
actually credits corpus trust. No `CriterionOutcome` is constructed by hand —
every stage is driven through its real entry point, both signs.

Driven from a `chdir`'d workspace because `consolidate-insights` resolves the
state root from the process cwd while the intent CLI resolves it from `--cwd`;
the two coincide only when the process cwd IS the workspace — exactly how a user
runs `rlat` from their project root. Hermetic: an in-process daemon (no
subprocess spawn) + a `FixedEncoder` planted to match the band vector, so
cosines are real with no model load; no LLM, no network.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# `accept` → satisfied roll-up (corroborates); `reject` → not_satisfied
# (falsifies). Asserted via the ledger record, not stdout — "satisfied" is a
# substring of "not_satisfied", so a stdout grep would mis-pass.
_EXPECTED_ROLLUP = {"accept": "satisfied", "reject": "not_satisfied"}


def _run(argv: list[str]) -> tuple[int, str, str]:
    from resonance_lattice.cli.app import main

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def _drive_corpus_loop(verdict: str):
    """Drive the full corpus-trust loop for a user `verdict` ("accept"/"reject")
    through real entry points, asserting every direction-AGNOSTIC invariant:
    the corpus id is stamped into the cache under the active intent, the
    intent-kind ledger record carries the expected roll-up + attributes the
    corpus claim, consolidate folds it, and a SECOND consolidate is idempotent
    (re-derivation — §B). Returns `(baseline_corr, baseline_fals, final_corr,
    final_fals)` on success, or None on a failed invariant (already logged).
    The caller asserts the direction (corroboration up / falsification up)."""
    import numpy as np

    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.daemon import daemon_socket_address
    from resonance_lattice.memory.store import path_for_user
    from resonance_lattice.memory.user_prompt import run_hook
    from resonance_lattice.state import (
        ClaimOutcomeLog,
        LiveIntentStore,
        RecallCache,
        resolve_state_root,
    )
    from resonance_lattice.store import archive

    from ._testutil import (
        FixedEncoder,
        ZeroEncoder,
        booted_daemon,
        build_corpus,
        make_corpus_claim,
    )

    expected_rollup = _EXPECTED_ROLLUP[verdict]
    original_cwd = os.getcwd()
    # The intent writer and consolidate reader must both resolve the state root
    # to <cwd>/.rlat-state; a stray RLAT_STATE_ROOT would split them. Neutralise it.
    prior_state_env = os.environ.pop("RLAT_STATE_ROOT", None)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            ws = Path(td).resolve()  # workspace == cwd == km location
            # A real .rlat, then overwrite its insight layer with one ACTIVE
            # corpus claim whose planted band vector (e0) gives cosine 1.0
            # against a FixedEncoder(e0) query.
            km = build_corpus(ws, {"a.md": "# A\n\nstar schema fact-table body."})
            src_pid = archive.read(km).registry[0].passage_id
            e0 = np.zeros(768, dtype="float32")
            e0[0] = 1.0
            claim = make_corpus_claim(
                "prefer a star schema for the sales fact table",
                [src_pid], state="active",
            )
            archive.write_insight_layer_in_place(km, [claim], e0.reshape(1, 768))
            target_id = claim.claim_id
            baseline_corr = claim.corroboration
            baseline_fals = claim.falsification

            rlats = sorted(p.name for p in ws.glob("*.rlat"))
            if rlats != [km.name]:
                print(f"[insight_loop_e2e] FAIL ({verdict}): workspace must hold "
                      f"exactly the one km; found {rlats}", file=sys.stderr)
                return None

            base = ws / "mem"
            memory_root = path_for_user(user_id="u", root=base)
            memory = ExperienceClaimStore(root=memory_root, encoder=ZeroEncoder())
            addr = daemon_socket_address(memory_root)

            os.chdir(ws)
            try:
                state_root = resolve_state_root(Path.cwd())
                # An ACTIVE intent must exist BEFORE recall so the hook stamps
                # its id onto the cache entry (deterministic attribution).
                intent = LiveIntentStore(state_root).add_intent(
                    level="task", text="design the sales star schema",
                    stance="do", achievability="medium",
                    success_criteria=[], constraints=[], status="active",
                )

                # --- [recall → cache] real hook recall against a live daemon
                with booted_daemon(
                    memory, address=addr, encoder=FixedEncoder(e0),
                ) as (server, _):
                    if server._listener is None:
                        print(f"[insight_loop_e2e] FAIL ({verdict}): daemon boot "
                              f"failed", file=sys.stderr)
                        return None
                    stdin = io.StringIO(json.dumps({
                        "prompt": "how should I model the sales fact table?",
                        "cwd": str(Path.cwd()),
                    }))
                    out, err = io.StringIO(), io.StringIO()
                    rc = run_hook(
                        stdin=stdin, stdout=out, stderr=err,
                        user_id="u", memory_root_base=base,
                    )
                if rc != 0:
                    print(f"[insight_loop_e2e] FAIL ({verdict}): hook rc={rc}",
                          file=sys.stderr)
                    return None

                # The recall leg: the corpus claim is in the cache, stamped with
                # the live intent id, source=corpus, primary-tier rank.
                stamped = [
                    m for e in RecallCache(state_root).read_for_intent(
                        intent.intent_id, since_iso=intent.created_at)
                    for m in e.row_metadata if m.claim_id == target_id
                ]
                if not stamped or stamped[0].source != "corpus" \
                        or stamped[0].rank != 0:
                    print(f"[insight_loop_e2e] FAIL ({verdict}): corpus claim not "
                          f"stamped at rank 0 / source corpus: {stamped!r}",
                          file=sys.stderr)
                    return None

                # --- [cache → attribution → ledger] real verdict ---
                rc, out, err = _run(
                    ["intent", "--cwd", str(Path.cwd()), verdict,
                     intent.intent_id])
                if rc != 0 or "(0 attributed)" in out:
                    print(f"[insight_loop_e2e] FAIL ({verdict}): rc={rc} "
                          f"out={out!r} err={err!r}", file=sys.stderr)
                    return None

                # Robust roll-up check via the ledger record (NOT stdout — see
                # _EXPECTED_ROLLUP). Proves synthesize_criterion_checks + roll_up
                # wired the user_confirms fallback to the right sign AND the
                # corpus id reached attribution.
                records = ClaimOutcomeLog(state_root).read(
                    intent_id=intent.intent_id)
                attributed = (
                    {a.claim_id for a in records[0].attribution}
                    if records else set()
                )
                if (len(records) != 1
                        or records[0].roll_up_verdict != expected_rollup
                        or target_id not in attributed):
                    shape = [(r.roll_up_verdict,
                              [a.claim_id for a in r.attribution])
                             for r in records]
                    print(f"[insight_loop_e2e] FAIL ({verdict}): outcome record "
                          f"wrong — {shape!r} (want roll_up {expected_rollup!r} "
                          f"attributing {target_id})", file=sys.stderr)
                    return None

                # --- [ledger → reducer → archive] real consolidate ---
                rc, out, err = _run(["consolidate-insights", str(km)])
                if rc != 0 or "intent outcome" not in out:
                    print(f"[insight_loop_e2e] FAIL ({verdict}): consolidate "
                          f"rc={rc} out={out!r} err={err!r}", file=sys.stderr)
                    return None

                # --- idempotency (§B BLOCKER fix): a SECOND consolidate
                # re-derives each claim's tally from its born seed + the same
                # full ledger, so it changes nothing. ---
                before = {c.claim_id: c
                          for c in archive.read(km).insights}[target_id]
                rc2, out2, err2 = _run(["consolidate-insights", str(km)])
                if rc2 != 0 or "0 insight(s) updated" not in out2:
                    print(f"[insight_loop_e2e] FAIL ({verdict}): second "
                          f"consolidate should change nothing; rc={rc2} "
                          f"out={out2!r} err={err2!r}", file=sys.stderr)
                    return None
                after = {c.claim_id: c
                         for c in archive.read(km).insights}[target_id]
                if (after.corroboration, after.falsification) != (
                        before.corroboration, before.falsification):
                    print(f"[insight_loop_e2e] FAIL ({verdict}): re-consolidation "
                          f"moved trust (non-idempotent)", file=sys.stderr)
                    return None
            finally:
                os.chdir(original_cwd)

            final = {c.claim_id: c
                     for c in archive.read(km).insights}.get(target_id)
            if final is None:
                print(f"[insight_loop_e2e] FAIL ({verdict}): target claim "
                      f"vanished after consolidate", file=sys.stderr)
                return None
            return (baseline_corr, baseline_fals,
                    final.corroboration, final.falsification)
    finally:
        if prior_state_env is not None:
            os.environ["RLAT_STATE_ROOT"] = prior_state_env


def _check_corpus_trust_loop_accept() -> int:
    """Accept → corpus trust UP (corroboration rises) end-to-end."""
    r = _drive_corpus_loop("accept")
    if r is None:
        return 1
    b_corr, _b_fals, f_corr, _f_fals = r
    if not (f_corr > b_corr):
        print(f"[insight_loop_e2e] FAIL: accept did not corroborate — "
              f"corroboration {b_corr} → {f_corr} (want strictly greater)",
              file=sys.stderr)
        return 1
    print(f"[insight_loop_e2e] (accept) corpus trust UP end-to-end: recall → "
          f"cache → accept → consolidate (corroboration {b_corr:.3f} → "
          f"{f_corr:.3f}) OK", file=sys.stderr)
    return 0


def _check_corpus_trust_loop_reject() -> int:
    """Reject → corpus trust DOWN (falsification rises) end-to-end — the
    symmetric DOWN half the §B review found proven only in pieces."""
    r = _drive_corpus_loop("reject")
    if r is None:
        return 1
    _b_corr, b_fals, _f_corr, f_fals = r
    if not (f_fals > b_fals):
        print(f"[insight_loop_e2e] FAIL: reject did not falsify — "
              f"falsification {b_fals} → {f_fals} (want strictly greater)",
              file=sys.stderr)
        return 1
    print(f"[insight_loop_e2e] (reject) corpus trust DOWN end-to-end: recall → "
          f"cache → reject → consolidate (falsification {b_fals:.3f} → "
          f"{f_fals:.3f}) OK", file=sys.stderr)
    return 0


def run() -> int:
    from ._testutil import patch_zero_encoder

    # build_corpus runs the real build pipeline; the stub encoder keeps the
    # base band cheap (we use only the explicitly-planted insight band).
    patch_zero_encoder()
    for check in [
        _check_corpus_trust_loop_accept,
        _check_corpus_trust_loop_reject,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[insight_loop_e2e] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
