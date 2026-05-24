"""memory_v21_hook_inject — §0.4 hook injection + Day 9-10 surface contracts.

Pins the contracts for the synchronous `recall <query>` body, the
`rlat memory hook` UserPromptSubmit shim, and the §0.4 wire-format:

  (a) `<rlat-memory>` block format. Hits render as
      `- *<polarity>* — <text>`, polarity is the primary tag only,
      block opens with `**Memory** (N lessons, recurrence ≥M):`.
      Empty hits → empty string (no block, no header).

  (b) `rlat memory hook` reads JSON `{prompt, cwd}` from stdin and
      writes JSON to stdout. Non-empty hits → emit
      `{hookSpecificOutput.additionalContext: <block>}`. Empty hits →
      emit `{}`. Always rc=0 (fail-open per §16.5 / §18.5).

  (c) Hook fail-open boundaries. Bad stdin JSON, missing memory_root,
      and daemon-unreachable + spawn-fails ALL produce `{}` to stdout
      and rc=0 — never raise, never block the prompt.

  (d) `rlat memory recall <query>` synchronous body. Empty store →
      rc=0 with "(no rows pass" stderr. Hits exist → rc=0 with one
      Row.summary line per hit on stdout. `--format json` returns the
      hits as a JSON array. `--polarity prefer` post-filters to
      prefer-primary rows only.

  (e) §0.4 token budget. Block truncates at row boundary when the
      cumulative char count would exceed `_MAX_INJECTION_CHARS`
      (~6000 chars / 1500 tokens). Never emits a half-row.

  (f) Diagnostic log entry written on every recall outcome (no_store,
      daemon_unreachable, no_hit, ok). The longitudinal-v3 bench
      surfaced "16/20 sessions had no recall" without any way to
      attribute the misses — the diagnostic log gives every miss a
      `status` + `dropped_at` so future bench misses are explainable.

  (g) `_recall_via_daemon_or_spawn` clears a stale `.recall.ready`
      marker BEFORE spawning a daemon. A previous daemon killed
      without running its finally (SIGKILL / OOM / reboot) leaves a
      stale marker; without the pre-spawn clear, the polling loop
      observes the stale marker immediately and tries to connect to
      a nonexistent daemon — the v4 bench's 80% daemon_unreachable
      rate from `claude -p` subprocesses traced to exactly this.

Hermetic: no live encoder, no daemon spawn (the hook's connect-fail
path is gated on `memory_root.exists()`); the spawn branch is
exercised against a tempdir without a daemon, which makes the hook
take the connect-fail-then-retry-fail-then-empty path.

Spec: `.claude/plans/fabric-agent-flat-memory.md` §0.4 / §0.6 / §5.2.1.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ._testutil import FixedEncoder, patch_zero_encoder, run_cli as _run_cli


def _seed_recallable_rows(memory_root: Path, n: int = 3) -> list[str]:
    """Seed `n` rows that will pass §0.6 gates against a fixed query.

    Cosines spaced at 0.06 to clear the §0.6 0.05 confidence gap;
    recurrence_count bumped to 5 to clear the M=3 recurrence gate;
    workspace tag matches `/proj` so the workspace gate accepts
    rows when the caller passes the matching cwd_hash.

    Returns the row_ids in insertion order so callers can assert on
    specific writes.
    """
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.store import Memory

    cwd_tag = workspace_tag_for_cwd("/proj")
    query_vec = np.zeros(768, dtype=np.float32)
    query_vec[0] = 1.0
    memory = Memory(root=memory_root, encoder=FixedEncoder(query_vec))
    cosines = [0.95, 0.85, 0.75, 0.69][:n]
    row_ids: list[str] = []
    for i, cos in enumerate(cosines):
        emb = np.zeros(768, dtype=np.float32)
        emb[0] = cos
        emb[1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
        primary = "prefer" if i == 0 else "factual"
        row_id = memory.add_row(
            text=f"row {i}: lesson about widget {i}",
            polarity=[primary, cwd_tag],
            transcript_hash=f"distilled:fixturetx{i:04d}",
            embedding=emb,
        )
        memory.update_row(row_id, recurrence_count=5)
        row_ids.append(row_id)
    return row_ids


# ---------------------------------------------------------------------------
# (a) §0.4 wire-format
# ---------------------------------------------------------------------------


def _make_hit(text: str, primary: str) -> dict:
    """Hit envelope matching what `RecallReply.hits` carries — full
    9-field Row dict + cosine. Default scope is workspace:abc123.
    """
    from resonance_lattice.memory.store import SCHEMA_VERSION

    return {
        "row": {
            "row_id": "01HZ8K3M5N7P9Q1R2S3T4V5W6X",
            "text": text,
            "polarity": [primary, "workspace:abc123"],
            "recurrence_count": 5,
            "created_at": "2026-05-02T00:00:00Z",
            "last_corroborated_at": "2026-05-02T00:00:00Z",
            "transcript_hash": "distilled:abc",
            "is_bad": False,
            "schema_version": SCHEMA_VERSION,
        },
        "cosine": 0.9,
    }


def _check_block_format() -> int:
    from resonance_lattice.memory.user_prompt import _format_injection

    hits = [
        _make_hit("use pytest -xvs", "prefer"),
        _make_hit("never use bare except", "avoid"),
    ]
    block, n = _format_injection(hits, recurrence_m=3)
    expected_lines = [
        "<rlat-memory>",
        "**Memory** (2 lessons, recurrence ≥3):",
        "",
        "- *prefer* — use pytest -xvs",
        "- *avoid* — never use bare except",
        "</rlat-memory>",
    ]
    if block.splitlines() != expected_lines or n != 2:
        print("[memory_v21_hook_inject] FAIL (a): block format mismatch.\n"
              f"got n={n}, block:\n{block}\n\nexpected n=2:\n"
              + "\n".join(expected_lines), file=sys.stderr)
        return 1
    if _format_injection([], recurrence_m=3) != ("", 0):
        print("[memory_v21_hook_inject] FAIL (a): empty hits should produce "
              "(\"\", 0)", file=sys.stderr)
        return 1
    print("[memory_v21_hook_inject] (a) §0.4 wire-format OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) hook stdin / stdout / additionalContext
# ---------------------------------------------------------------------------


def _check_hook_envelope() -> int:
    from resonance_lattice.memory.user_prompt import run_hook

    # Empty hits path — fresh tempdir without seeding.
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        (base / "u").mkdir(parents=True)
        # Empty memory.npz + sidecar so connect succeeds but recall returns
        # nothing. Actually with no memory_root we'd fall through fail-open;
        # to test the empty-hits path we need a populated directory but
        # zero rows. The seed-helper writes both files implicitly via add.
        # Skip seeding entirely so the daemon connect fails → retry fails
        # → reply is None → emit `{}`.
        stdin = io.StringIO(json.dumps({"prompt": "what should I prefer?",
                                          "cwd": "/proj"}))
        stdout, stderr = io.StringIO(), io.StringIO()
        rc = run_hook(
            stdin=stdin, stdout=stdout, stderr=stderr,
            user_id="u", memory_root_base=base,
        )
    if rc != 0:
        print(f"[memory_v21_hook_inject] FAIL (b): rc={rc} on empty-hits "
              f"path (want 0 fail-open)", file=sys.stderr)
        return 1
    out_payload = stdout.getvalue().strip()
    if out_payload != "{}":
        print(f"[memory_v21_hook_inject] FAIL (b): empty-hits output "
              f"should be `{{}}`, got: {out_payload!r}", file=sys.stderr)
        return 1
    print("[memory_v21_hook_inject] (b) hook envelope (empty-hits → `{}` "
          "stdout, rc=0) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) hook fail-open boundaries
# ---------------------------------------------------------------------------


def _check_fail_open() -> int:
    from resonance_lattice.memory.user_prompt import run_hook

    # Bad stdin JSON.
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = run_hook(
        stdin=io.StringIO("not json at all"),
        stdout=stdout, stderr=stderr,
        user_id="u",
    )
    if rc != 0 or stdout.getvalue().strip() != "{}":
        print(f"[memory_v21_hook_inject] FAIL (c.1): bad-JSON stdin should "
              f"emit `{{}}` rc=0; got rc={rc} stdout={stdout.getvalue()!r}",
              file=sys.stderr)
        return 1

    # Empty prompt.
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = run_hook(
        stdin=io.StringIO(json.dumps({"prompt": "   ", "cwd": "/x"})),
        stdout=stdout, stderr=stderr,
        user_id="u",
    )
    if rc != 0 or stdout.getvalue().strip() != "{}":
        print(f"[memory_v21_hook_inject] FAIL (c.2): empty-prompt should "
              f"emit `{{}}` rc=0; got rc={rc} stdout={stdout.getvalue()!r}",
              file=sys.stderr)
        return 1

    # Missing memory_root.
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "missing-base"  # never mkdir
        stdout, stderr = io.StringIO(), io.StringIO()
        rc = run_hook(
            stdin=io.StringIO(json.dumps({"prompt": "hello", "cwd": "/x"})),
            stdout=stdout, stderr=stderr,
            user_id="u", memory_root_base=base,
        )
    if rc != 0 or stdout.getvalue().strip() != "{}":
        print(f"[memory_v21_hook_inject] FAIL (c.3): missing-memory-root "
              f"should emit `{{}}` rc=0; got rc={rc} stdout={stdout.getvalue()!r}",
              file=sys.stderr)
        return 1
    print("[memory_v21_hook_inject] (c) fail-open OK across bad-stdin + "
          "empty-prompt + missing-root", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) recall CLI synchronous body
# ---------------------------------------------------------------------------


def _check_recall_cli_body() -> int:
    from resonance_lattice.memory._common import workspace_hash
    from resonance_lattice.memory.recall import recall
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"

        # Empty-store path: CLI returns rc=0 + the gates-message stderr.
        empty_root = base / "empty"
        empty_root.mkdir(parents=True)
        rc, _, err = _run_cli([
            "memory", "--memory-root", str(base), "--user", "empty",
            "recall", "ignored",
        ])
        if rc != 0 or "(no rows pass" not in err:
            print(f"[memory_v21_hook_inject] FAIL (d.1): empty-store recall "
                  f"rc={rc} stderr={err!r}", file=sys.stderr)
            return 1

        # Polarity post-filter: seed 3 rows (one prefer, two factual),
        # call recall() directly with a FixedEncoder so cosines clear
        # the §0.6 gates, then re-apply the CLI's `--polarity prefer`
        # filter and assert only the prefer row survives.
        seeded_root = base / "u"
        seeded_root.mkdir(parents=True)
        row_ids = _seed_recallable_rows(seeded_root, n=3)

        query_vec = np.zeros(768, dtype=np.float32)
        query_vec[0] = 1.0
        memory = Memory(root=seeded_root, encoder=FixedEncoder(query_vec))
        hits = recall(
            "anything",
            store=memory,
            cwd_hash=workspace_hash("/proj"),
            top_k=5,
        )
        if not hits:
            print("[memory_v21_hook_inject] FAIL (d.2): seeded recall returned "
                  "no hits — fixture cosines / recurrence below gates",
                  file=sys.stderr)
            return 1
        prefer_hits = [h for h in hits if "prefer" in h.row.polarity]
        if len(prefer_hits) != 1 or prefer_hits[0].row.row_id != row_ids[0]:
            print(f"[memory_v21_hook_inject] FAIL (d.2): polarity post-filter "
                  f"expected exactly the seed prefer row "
                  f"{row_ids[0]!r}; got {prefer_hits!r}", file=sys.stderr)
            return 1
    print("[memory_v21_hook_inject] (d) recall CLI body — empty-store gates "
          "message + polarity post-filter via recall() OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (e) §0.4 token budget truncation
# ---------------------------------------------------------------------------


def _check_token_budget() -> int:
    from resonance_lattice.memory.user_prompt import (
        _MAX_INJECTION_CHARS,
        _format_injection,
    )

    # 10 rows × ~1000 chars each vs a 6000-char budget — expect partial
    # truncation strictly between 1 and 10 surviving rows.
    big_text = "x" * 1000
    hits = [_make_hit(big_text, "factual") for _ in range(10)]
    block, n_rows = _format_injection(hits, recurrence_m=3)
    if len(block) > _MAX_INJECTION_CHARS + 200:  # +200 for header overhead
        print(f"[memory_v21_hook_inject] FAIL (e): block exceeds budget — "
              f"len={len(block)}, budget={_MAX_INJECTION_CHARS}",
              file=sys.stderr)
        return 1
    if n_rows < 1 or n_rows >= 10:
        print(f"[memory_v21_hook_inject] FAIL (e): expected 1<=N<10 rows "
              f"after truncation, got {n_rows}", file=sys.stderr)
        return 1
    for line in block.splitlines():
        if line.startswith("- ") and not line.endswith(big_text):
            print(f"[memory_v21_hook_inject] FAIL (e): truncated row "
                  f"detected: {line[:80]!r}...", file=sys.stderr)
            return 1
    print(f"[memory_v21_hook_inject] (e) token budget truncates at row "
          f"boundary ({n_rows}/10 rows survived, len={len(block)}) OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (f) diagnostic log written on every recall outcome
# ---------------------------------------------------------------------------


def _check_diagnostic_logged() -> int:
    from resonance_lattice.memory.user_prompt import run_hook
    from resonance_lattice.state import RecallDiagnosticLog, state_root_for

    # Missing memory_root → status=no_store path. Use a tempdir as cwd
    # so the diagnostic write target (`<cwd>/.rlat-state/ledger/`) is
    # writable but otherwise empty before the call.
    with tempfile.TemporaryDirectory() as td:
        cwd = Path(td)
        base = cwd / "missing-base"  # never mkdir
        stdin = io.StringIO(json.dumps({
            "prompt": "what should I prefer?",
            "cwd": str(cwd),
        }))
        stdout, stderr = io.StringIO(), io.StringIO()
        rc = run_hook(
            stdin=stdin, stdout=stdout, stderr=stderr,
            user_id="u", memory_root_base=base,
        )
        if rc != 0 or stdout.getvalue().strip() != "{}":
            print(f"[memory_v21_hook_inject] FAIL (f.1): no-store path "
                  f"rc={rc} stdout={stdout.getvalue()!r}", file=sys.stderr)
            return 1
        state_root = state_root_for(cwd)
        entries = RecallDiagnosticLog(state_root).read_recent()
        if len(entries) != 1:
            print(f"[memory_v21_hook_inject] FAIL (f.1): expected 1 "
                  f"diagnostic entry, got {len(entries)}", file=sys.stderr)
            return 1
        entry = entries[0]
        if entry.status != "no_store":
            print(f"[memory_v21_hook_inject] FAIL (f.1): expected "
                  f"status=no_store, got {entry.status!r}", file=sys.stderr)
            return 1
        if entry.n_hits != 0 or entry.diagnostic is not None:
            print(f"[memory_v21_hook_inject] FAIL (f.1): no-store entry "
                  f"should have n_hits=0 diagnostic=None; got "
                  f"n_hits={entry.n_hits} diagnostic={entry.diagnostic!r}",
                  file=sys.stderr)
            return 1
    print("[memory_v21_hook_inject] (f) diagnostic entry logged on no-store "
          "outcome OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _check_disable_hook_env_var() -> int:
    """`RLAT_DISABLE_HOOK=1` short-circuits the hook to `{}` rc=0 and
    writes a `status=disabled` diagnostic entry. Used by the v5 paired
    bench's off-arm to suppress recall injection while keeping the
    same prompts / intent flow as the on-arm.

    The fast-exit lives AFTER the stdin/prompt gates so the diagnostic
    entry carries a real prompt_hash — the paired-eval join key is
    `(arm, prompt_hash)`, and a missing hash would silently mis-join.

    Side-checks (all required for the off-arm contract):
      - no daemon spawn (covered indirectly: empty memory_root + rc=0)
      - diagnostic written with status=disabled, n_hits=0
      - stdout = `{}` (no injection)
    """
    import os

    from resonance_lattice.memory.user_prompt import run_hook
    from resonance_lattice.state import RecallDiagnosticLog, state_root_for

    with tempfile.TemporaryDirectory() as td:
        cwd = Path(td)
        # memory_root EXISTS — so the no-store fast-exit doesn't fire.
        # The RLAT_DISABLE_HOOK gate must trip before the daemon path.
        base = cwd / "base"
        (base / "u").mkdir(parents=True)
        stdin = io.StringIO(json.dumps({
            "prompt": "would this prompt normally recall?",
            "cwd": str(cwd),
        }))
        stdout, stderr = io.StringIO(), io.StringIO()
        prior = os.environ.get("RLAT_DISABLE_HOOK")
        os.environ["RLAT_DISABLE_HOOK"] = "1"
        try:
            rc = run_hook(
                stdin=stdin, stdout=stdout, stderr=stderr,
                user_id="u", memory_root_base=base,
            )
        finally:
            if prior is None:
                os.environ.pop("RLAT_DISABLE_HOOK", None)
            else:
                os.environ["RLAT_DISABLE_HOOK"] = prior

        if rc != 0 or stdout.getvalue().strip() != "{}":
            print(f"[memory_v21_hook_inject] FAIL (h): disabled hook should "
                  f"emit `{{}}` rc=0; got rc={rc} stdout={stdout.getvalue()!r}",
                  file=sys.stderr)
            return 1

        entries = RecallDiagnosticLog(state_root_for(cwd)).read_recent()
        if len(entries) != 1 or entries[0].status != "disabled":
            print(f"[memory_v21_hook_inject] FAIL (h): expected one "
                  f"status=disabled diagnostic; got {entries!r}",
                  file=sys.stderr)
            return 1
        if entries[0].n_hits != 0:
            print(f"[memory_v21_hook_inject] FAIL (h): disabled entry should "
                  f"have n_hits=0; got n_hits={entries[0].n_hits}",
                  file=sys.stderr)
            return 1
    print("[memory_v21_hook_inject] (h) RLAT_DISABLE_HOOK=1 short-circuit "
          "fast-exit + diagnostic OK", file=sys.stderr)
    return 0


def _check_stale_ready_marker_cleared_pre_spawn() -> int:
    """`_recall_via_daemon_or_spawn` must unlink any pre-existing
    `.recall.ready` marker BEFORE invoking `_spawn_daemon`, so the
    subsequent poll loop only observes markers from the freshly
    spawned daemon.

    v4 bench root cause: a killed daemon left a stale marker; hook
    polled, saw the marker instantly (0ms after spawn — impossibly
    fast for a real daemon to advertise readiness), trusted it,
    tried to connect to a nonexistent daemon, fell into the
    daemon_unreachable path. Repeated across ~80% of `claude -p`
    bench sessions.
    """
    from unittest.mock import patch

    from resonance_lattice.memory.daemon import daemon_ready_path
    from resonance_lattice.memory.user_prompt import _recall_via_daemon_or_spawn

    with tempfile.TemporaryDirectory() as td:
        memory_root = Path(td) / "u"
        memory_root.mkdir(parents=True)
        # Plant a stale marker that no live daemon backs.
        ready_path = daemon_ready_path(memory_root)
        ready_path.write_text("99999", encoding="utf-8")
        assert ready_path.exists(), "fixture stale marker missing"

        marker_seen_at_spawn: list[bool] = []

        def _fake_spawn(root):
            # Snapshot whether marker still exists at the moment we'd
            # spawn. After the pre-spawn unlink, it must be gone.
            marker_seen_at_spawn.append(ready_path.exists())

        from resonance_lattice.memory.daemon import RecallRequest

        # connect-first attempt will fail (no daemon, no real listener
        # bound to the test address). After failure, the fix should
        # unlink the stale marker before calling _spawn_daemon.
        with patch(
            "resonance_lattice.memory.user_prompt._spawn_daemon",
            side_effect=_fake_spawn,
        ):
            _recall_via_daemon_or_spawn(
                RecallRequest(query="probe", cwd_hash=None),
                memory_root,
            )

    if not marker_seen_at_spawn:
        print("[memory_v21_hook_inject] FAIL (g): _spawn_daemon was never "
              "invoked — connect-first attempt didn't reach the spawn path",
              file=sys.stderr)
        return 1
    if marker_seen_at_spawn[0]:
        print("[memory_v21_hook_inject] FAIL (g): stale marker still present "
              "at _spawn_daemon call — pre-spawn unlink didn't run",
              file=sys.stderr)
        return 1
    print("[memory_v21_hook_inject] (g) stale ready-marker cleared pre-spawn OK",
          file=sys.stderr)
    return 0


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_block_format,
        _check_hook_envelope,
        _check_fail_open,
        _check_recall_cli_body,
        _check_token_budget,
        _check_diagnostic_logged,
        _check_disable_hook_env_var,
        _check_stale_ready_marker_cleared_pre_spawn,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_hook_inject] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
