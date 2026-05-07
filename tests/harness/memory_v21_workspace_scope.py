"""memory_v21_workspace_scope — D.8 workspace tag + scope contracts.

Pins four guarantees on the workspace-scoping pipeline (`§0.6` step 2
+ `§6.2` workspace tag + `§18.3` collision mitigation):

  (a) Workspace tag stamping at capture time uses sha256[:6] of cwd —
      the polarity tag stamped on every captured row is exactly
      `workspace:<sha256(cwd)[:6]>`. Verifies that capture() uses
      `_common.workspace_tag_for_cwd` consistently.

  (b) Recall scopes correctly: rows tagged `workspace:<hashA>` are
      dropped when the recall caller passes `cwd_hash=<hashB>`; rows
      tagged `workspace:<hashA>` survive when the caller passes
      `cwd_hash=<hashA>`.

  (c) `cross-workspace` polarity bypasses the filter: rows carrying
      `cross-workspace` survive any cwd_hash, with or without an
      accompanying `workspace:<hash>` tag.

  (d) Hash collision detection: a fixture with two distinct cwd paths
      whose `workspace_hash` collides (forced by brute-force hash
      search) verifies the recall filter doesn't bleed beyond the
      cwd_hash equality check. The test asserts the cwd-equality
      contract holds — both paths see the same row set under their
      shared hash. (No silent-bleed-detection diagnostic ships in
      MVP; if it lands later, this contract extends to assert the
      diagnostic fires.)

Hermetic — no live encoder, no LLM. Collision search is the slowest
step (~5-30s on first run), cached on disk across runs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import FixedEncoder, ZeroEncoder, patch_zero_encoder


_COLLISION_CACHE = Path(__file__).resolve().parent / "_fixtures" / "workspace_collision.json"


# ---------------------------------------------------------------------------
# (a) Capture stamps workspace:<sha256[:6](cwd)>
# ---------------------------------------------------------------------------


def _check_capture_stamp() -> int:
    from resonance_lattice.memory.capture import (
        Message, ToolCall, Transcript, capture,
    )
    from resonance_lattice.memory._common import workspace_hash
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    cwd = "/home/test/proj-alpha"
    expected_tag = f"workspace:{workspace_hash(cwd)}"

    transcript = Transcript(
        session_id="stamp",
        messages=[
            Message("user", "diagnose the failing build please look at recent commits"),
            Message("assistant", "x" * 300,
                    tool_calls=(ToolCall("bash", "/tmp", "ls"),)),
        ],
        cwd=cwd,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u", encoder=ZeroEncoder())
        redactor = Redactor()
        result = capture(transcript, store=memory, redactor=redactor)
        if not result.row_id:
            print(f"[memory_v21_workspace_scope] FAIL (a): capture skipped: "
                  f"{result.skip_reason}", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        captured = next(r for r in rows if r.row_id == result.row_id)
        if expected_tag not in captured.polarity:
            print(f"[memory_v21_workspace_scope] FAIL (a): expected tag "
                  f"{expected_tag} in polarity; got {captured.polarity}",
                  file=sys.stderr)
            return 1
        # Tag must use the canonical 6-char hex shape — falsifies
        # accidental collisions of `workspace:<full-hash>` vs `[:6]`.
        for tag in captured.polarity:
            if tag.startswith("workspace:") and len(tag) != len("workspace:") + 6:
                print(f"[memory_v21_workspace_scope] FAIL (a): tag has "
                      f"non-canonical length: {tag!r} (expected "
                      f"workspace:6-hex)", file=sys.stderr)
                return 1
    print(f"[memory_v21_workspace_scope] (a) capture stamps "
          f"{expected_tag} OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) Recall scopes correctly
# ---------------------------------------------------------------------------


def _check_recall_scope() -> int:
    from resonance_lattice.memory.recall import rank
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        # Two rows in different workspaces, both above all gates.
        rng = np.random.default_rng(0)
        query = rng.standard_normal(768).astype("float32")
        query /= np.linalg.norm(query)

        for tag, text in [
            ("workspace:aaaaaa", "alpha workspace row"),
            ("workspace:bbbbbb", "bravo workspace row"),
        ]:
            ortho = rng.standard_normal(768).astype("float32")
            ortho -= (ortho @ query) * query
            ortho /= np.linalg.norm(ortho)
            emb = 0.85 * query + (1 - 0.85**2)**0.5 * ortho
            emb /= np.linalg.norm(emb)
            rid = memory.add_row(text=text, polarity=["prefer", tag],
                                  transcript_hash=text, embedding=emb)
            memory.update_row(rid, recurrence_count=5)

        rows, band = memory.read_all()
        # Caller scoped to alpha — only alpha row should survive.
        alpha_hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query),
                           cwd_hash="aaaaaa")
        if len(alpha_hits) != 1 or alpha_hits[0].row.text != "alpha workspace row":
            print(f"[memory_v21_workspace_scope] FAIL (b): cwd=aaaaaa "
                  f"expected only alpha; got "
                  f"{[h.row.text for h in alpha_hits]}", file=sys.stderr)
            return 1

        # Caller scoped to bravo — only bravo row should survive.
        bravo_hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query),
                           cwd_hash="bbbbbb")
        if len(bravo_hits) != 1 or bravo_hits[0].row.text != "bravo workspace row":
            print(f"[memory_v21_workspace_scope] FAIL (b): cwd=bbbbbb "
                  f"expected only bravo; got "
                  f"{[h.row.text for h in bravo_hits]}", file=sys.stderr)
            return 1

        # Caller scoped to a third workspace — no rows survive.
        empty_hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query),
                           cwd_hash="cccccc")
        if empty_hits:
            print(f"[memory_v21_workspace_scope] FAIL (b): cwd=cccccc "
                  f"should be empty; got {len(empty_hits)} hits",
                  file=sys.stderr)
            return 1
    print("[memory_v21_workspace_scope] (b) recall scopes correctly to "
          "matching cwd_hash OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) cross-workspace bypass
# ---------------------------------------------------------------------------


def _check_cross_workspace_bypass() -> int:
    from resonance_lattice.memory.recall import rank
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        rng = np.random.default_rng(1)
        query = rng.standard_normal(768).astype("float32")
        query /= np.linalg.norm(query)

        cases = [
            (["prefer", "cross-workspace"], "cross-only"),
            (["prefer", "workspace:aaaaaa", "cross-workspace"],
             "cross plus workspace"),
            (["prefer", "workspace:aaaaaa"], "workspace-only"),
        ]
        for polarity, text in cases:
            ortho = rng.standard_normal(768).astype("float32")
            ortho -= (ortho @ query) * query
            ortho /= np.linalg.norm(ortho)
            emb = 0.90 * query + (1 - 0.90**2)**0.5 * ortho
            emb /= np.linalg.norm(emb)
            rid = memory.add_row(text=text, polarity=polarity,
                                  transcript_hash=text, embedding=emb)
            memory.update_row(rid, recurrence_count=5)

        rows, band = memory.read_all()
        # Caller in unrelated workspace — only cross-workspace rows
        # should bypass.
        unrelated = rank("q", rows=rows, band=band, encoder=FixedEncoder(query),
                          cwd_hash="zzzzzz", top1_top2_gap=0.0)
        text_set = {h.row.text for h in unrelated}
        if "cross-only" not in text_set:
            print(f"[memory_v21_workspace_scope] FAIL (c): cross-only row "
                  f"didn't bypass workspace filter; got {text_set}",
                  file=sys.stderr)
            return 1
        if "cross plus workspace" not in text_set:
            print(f"[memory_v21_workspace_scope] FAIL (c): mixed "
                  f"workspace+cross row didn't bypass; got {text_set}",
                  file=sys.stderr)
            return 1
        if "workspace-only" in text_set:
            print(f"[memory_v21_workspace_scope] FAIL (c): workspace-only "
                  f"row leaked into unrelated cwd; got {text_set}",
                  file=sys.stderr)
            return 1
    print("[memory_v21_workspace_scope] (c) cross-workspace bypasses filter OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) Hash collision behaviour
# ---------------------------------------------------------------------------


def _find_collision_pair() -> tuple[str, str, str]:
    """Brute-force search two distinct strings with the same sha256[:6].

    24-bit space → birthday-attack expected work ~2^12 ≈ 4K trials.
    Cached on disk so repeated runs don't re-search.
    """
    if _COLLISION_CACHE.exists():
        cached = json.loads(_COLLISION_CACHE.read_text(encoding="utf-8"))
        return cached["a"], cached["b"], cached["hash"]

    seen: dict[str, str] = {}
    i = 0
    while True:
        s = f"path-{i}"
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:6]
        if h in seen and seen[h] != s:
            other = seen[h]
            _COLLISION_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _COLLISION_CACHE.write_text(
                json.dumps({"a": other, "b": s, "hash": h}),
                encoding="utf-8",
            )
            return other, s, h
        seen[h] = s
        i += 1
        if i > 50_000_000:
            raise RuntimeError("collision search exceeded 50M iters")


def _check_collision_behaviour() -> int:
    from resonance_lattice.memory._common import workspace_hash
    from resonance_lattice.memory.recall import rank
    from resonance_lattice.memory.store import Memory

    path_a, path_b, shared_hash = _find_collision_pair()
    # Sanity-check the collision still holds.
    if workspace_hash(path_a) != shared_hash or workspace_hash(path_b) != shared_hash:
        print(f"[memory_v21_workspace_scope] FAIL (d): cached collision "
              f"pair drifted: {path_a!r}/{path_b!r}/{shared_hash!r}",
              file=sys.stderr)
        return 1
    if path_a == path_b:
        print(f"[memory_v21_workspace_scope] FAIL (d): collision search "
              f"returned identical path", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        rng = np.random.default_rng(7)
        query = rng.standard_normal(768).astype("float32")
        query /= np.linalg.norm(query)

        # Two rows with the colliding workspace tag — both stamped from
        # different cwd paths but landing on the same hash.
        for label in ("from-path-a", "from-path-b"):
            ortho = rng.standard_normal(768).astype("float32")
            ortho -= (ortho @ query) * query
            ortho /= np.linalg.norm(ortho)
            emb = 0.85 * query + (1 - 0.85**2)**0.5 * ortho
            emb /= np.linalg.norm(emb)
            rid = memory.add_row(
                text=label, polarity=["prefer", f"workspace:{shared_hash}"],
                transcript_hash=label, embedding=emb,
            )
            memory.update_row(rid, recurrence_count=5)

        rows, band = memory.read_all()
        # Caller from path_a sees both rows (since the hash collides).
        # The §18.3 contract for MVP is "no silent bleed beyond the
        # cwd_hash equality check" — under collision, hash equality
        # *does* match, so both rows surface. Diagnostic detection of
        # the underlying-cwd mismatch is post-MVP.
        hits = rank("q", rows=rows, band=band, encoder=FixedEncoder(query),
                     cwd_hash=shared_hash, top1_top2_gap=0.0)
        text_set = {h.row.text for h in hits}
        if text_set != {"from-path-a", "from-path-b"}:
            print(f"[memory_v21_workspace_scope] FAIL (d): expected both "
                  f"colliding rows to surface under shared hash; got "
                  f"{text_set}", file=sys.stderr)
            return 1
    print(f"[memory_v21_workspace_scope] (d) collision pair "
          f"({path_a!r}, {path_b!r}) → {shared_hash!r} surfaces both rows "
          f"under cwd-equality contract OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_capture_stamp,
        _check_recall_scope,
        _check_cross_workspace_bypass,
        _check_collision_behaviour,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_workspace_scope] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
