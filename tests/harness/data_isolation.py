"""data_isolation — per-user / per-workspace stores cannot leak.

The data-isolation contract for Memory + Lens:

  (a) Two `ExperienceClaimStore` instances at different roots do not
      see each other's claims, even after independent writes — the
      stores are physically separate (no shared file under the
      memory_root tree).
  (b) `path_for_user` resolves distinct roots for distinct user_ids
      against the same base directory — the per-user dimension is
      driven by the path, not by process state.
  (c) Two daemon servers at different roots advertise different IPC
      addresses (POSIX socket path or Windows named pipe) — a client
      probing one root never connects to the other root's daemon by
      accident.
  (d) Per-user daemon authkeys are per-root and unique — a client
      forging connect-time auth from root B's authkey can't drive
      root A's daemon.
  (e) On POSIX, the per-user dir and its claim files are mode 0o700
      and 0o600 respectively, so a multi-tenant host's other users
      can't read another user's captured transcripts even with
      filesystem access to the home tree. Skipped on Windows (ACLs
      inherit from the parent dir).

Hermetic — temp dirs only; no real encoder, no real LLM, no network.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import check_guarantee


_P = "data_isolation"


def _vec(fill: float):
    from resonance_lattice.field.encoder import DIM
    return np.full(DIM, fill, dtype=np.float32)


def _claim(claim_id: str, content: str = "isolation test"):
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    return Claim(
        claim_id=claim_id,
        source="experience",
        kind="event",
        content=content,
        created_at="2026-05-23T00:00:00Z",
        corroboration=2.0,
        falsification=2.0,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=("factual", "workspace:isolation"),
            recurrence_count=1,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash="isolation",
            origin="manual",
            last_corroborated_at="2026-05-23T00:00:00Z",
            is_bad=False,
        ),
    )


def _check_two_roots_dont_share() -> int:
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root_a = base / "alice"
        root_b = base / "bob"

        store_a = ExperienceClaimStore(root=root_a, encoder=None)
        store_b = ExperienceClaimStore(root=root_b, encoder=None)

        store_a.write(_claim("01HZISO00000000000ALICE0001", content="alice's note"),
                      embedding=_vec(0.1))
        store_b.write(_claim("01HZISO0000000000000BOB0001", content="bob's note"),
                      embedding=_vec(0.2))

        # Fresh reopens — confirm disk shape rather than relying on
        # in-process state.
        a_claims = ExperienceClaimStore(root=root_a, encoder=None).read_all()
        b_claims = ExperienceClaimStore(root=root_b, encoder=None).read_all()

    ok = (
        len(a_claims) == 1
        and len(b_claims) == 1
        and a_claims[0].content == "alice's note"
        and b_claims[0].content == "bob's note"
        and a_claims[0].claim_id != b_claims[0].claim_id
    )
    return 0 if check_guarantee(
        ok, "(a) two roots don't share claims", _P) else 1


def _check_path_for_user_distinct_users() -> int:
    from resonance_lattice.memory.store import path_for_user

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        alice = path_for_user(user_id="alice", root=base)
        bob = path_for_user(user_id="bob", root=base)

    ok = alice != bob and alice.parent == bob.parent == base
    return 0 if check_guarantee(
        ok, "(b) path_for_user resolves distinct user roots", _P) else 1


def _check_daemon_addresses_distinct() -> int:
    from resonance_lattice.memory.daemon import daemon_socket_address

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        addr_a = daemon_socket_address(base / "alice")
        addr_b = daemon_socket_address(base / "bob")

    # Both addresses must be strings; both must be distinct.
    ok = (
        isinstance(addr_a, str)
        and isinstance(addr_b, str)
        and addr_a != addr_b
    )
    return 0 if check_guarantee(
        ok, "(c) daemon addresses are per-root", _P) else 1


def _check_authkeys_distinct() -> int:
    from resonance_lattice.memory.daemon import load_or_create_authkey

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root_a = base / "alice"
        root_b = base / "bob"
        root_a.mkdir(parents=True)
        root_b.mkdir(parents=True)
        key_a = load_or_create_authkey(root_a)
        key_b = load_or_create_authkey(root_b)

    # Distinct, both 32 bytes from `secrets.token_bytes`. The probability
    # of collision is ~2^-256 — a sound-bit gate, not a power-of-2 dice
    # roll like the unique-tmp suffix.
    ok = (
        isinstance(key_a, bytes)
        and isinstance(key_b, bytes)
        and len(key_a) == 32
        and len(key_b) == 32
        and key_a != key_b
    )
    return 0 if check_guarantee(
        ok, "(d) per-root authkeys are distinct", _P) else 1


def _check_posix_file_modes() -> int:
    if os.name == "nt":
        print(f"[{_P}] (e) POSIX file modes — SKIPPED on Windows",
              file=sys.stderr)
        return 0

    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor, RedactionEvent

    # Force the standard umask (0o022) so the test pins *chmod-as-action*
    # rather than passing accidentally under a tight umask. Without this,
    # a future regression that drops the chmod calls would still pass on
    # a developer box that happened to run with `umask 077`.
    prev_umask = os.umask(0o022)
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "user"
            store = ExperienceClaimStore(root=root, encoder=None)
            store.write(_claim("01HZISO00000000000FILEMODE1"),
                        embedding=_vec(0.3))

            root_mode = root.stat().st_mode & 0o777
            claims_mode = (root / "claims.jsonl").stat().st_mode & 0o777
            band_mode = (root / "band.npz").stat().st_mode & 0o777

            # Redaction log mode is tightened on first write.
            redactor = Redactor.for_memory_root(root)
            redactor.log_events(
                [RedactionEvent(pattern="test", matches=1)],
                row_id="01HZISO00000000000FILEMODE1",
            )
            redaction_mode = (root / "redaction.log").stat().st_mode & 0o777
    finally:
        os.umask(prev_umask)

    ok = (
        root_mode == 0o700
        and claims_mode == 0o600
        and band_mode == 0o600
        and redaction_mode == 0o600
    )
    if not ok:
        print(f"[{_P}] FAIL (e) modes: root={oct(root_mode)} "
              f"claims={oct(claims_mode)} band={oct(band_mode)} "
              f"redaction={oct(redaction_mode)}", file=sys.stderr)
    return 0 if check_guarantee(
        ok, "(e) per-user files have tight POSIX modes", _P) else 1


def run() -> int:
    failures = 0
    for check in (
        _check_two_roots_dont_share,
        _check_path_for_user_distinct_users,
        _check_daemon_addresses_distinct,
        _check_authkeys_distinct,
        _check_posix_file_modes,
    ):
        failures += check()
    if failures:
        print(f"[{_P}] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print(f"[{_P}] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
