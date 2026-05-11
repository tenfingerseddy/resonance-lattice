"""memory_v21_daemon — §0.8 daemon protocol contracts (Appendix D D.10).

Pins four guarantees on `memory.daemon`:

  (a) Boot-time encoder load is instantaneous when the encoder is
      already cached. Suite uses ZeroEncoder so the contract reduces
      to "the IPC handshake completes inside the boot budget"; the
      live gte-modernbert-base cold-load (~3s) is verified
      out-of-band via the install-encoder path.

  (b) p99 recall latency < 100 ms warm against a 50-row fixture.
      After the daemon is booted and the snapshot is loaded, 200
      sequential `request_recall` calls show p99 ≤ 100 ms.

  (c) Daemon crash → next hook call fails-open. Terminating the
      server thread mid-flight makes the next `request_recall`
      return `None` within the timeout — never an exception. Mirrors
      `memory_v21_hook (c)` at the IPC layer.

  (d) Doctor recovery lines — for each known failure mode (band
      missing, daemon unreachable, encoder mismatch), the doctor
      output contains the canonical recovery instruction. Always
      returns rc=0.

  (e) Ready marker — daemon writes `.recall.ready` after encoder +
      snapshot load, clears any stale marker first, removes the
      marker on clean exit. The hook polls for the marker on cold
      spawn instead of guessing at a boot-wait timeout.

Hermetic — daemon servers run as in-process threads bound to
per-test tempfile-derived addresses; no subprocesses, no real
network.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from ._testutil import (
    ZeroEncoder,
    booted_daemon,
    isolated_daemon_address,
    patch_zero_encoder,
    seed_capture_memory,
)


_BOOT_BUDGET_S = 2.0  # D.10 (a): boot-time on warm encoder cache
_P99_GATE_MS = 100.0  # D.10 (b): per-request warm latency
_FAIL_OPEN_TIMEOUT_MS = 200  # client-side budget for D.10 (c)


def _seed_50_rows(memory) -> None:
    """50 deterministic capture rows under the same workspace tag."""
    seed_capture_memory(memory, [
        {"text": f"capture row {i} for daemon harness fixture",
         "transcript_hash": f"daemonfixturetx{i:04d}"}
        for i in range(50)
    ])


# ---------------------------------------------------------------------------
# (a) Boot-time encoder load < budget on warm cache
# ---------------------------------------------------------------------------


def _check_boot_budget() -> int:
    from resonance_lattice.memory.daemon import (
        load_or_create_authkey,
        request_recall,
        RecallRequest,
    )
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root, encoder=ZeroEncoder())
        _seed_50_rows(memory)

        address = isolated_daemon_address(root)
        t0 = time.perf_counter()
        with booted_daemon(memory, address=address) as (server, _):
            boot_s = time.perf_counter() - t0
            if server._listener is None or boot_s >= _BOOT_BUDGET_S:
                print(f"[memory_v21_daemon] FAIL (a): boot took {boot_s:.3f}s "
                      f"(budget {_BOOT_BUDGET_S}s)", file=sys.stderr)
                return 1
            reply = request_recall(
                RecallRequest(query="probe", cosine_floor=0.0,
                              top1_top2_gap=0.0, min_recurrence=1),
                address=address, authkey=load_or_create_authkey(root),
                timeout_ms=2000,
            )
            if reply is None:
                print("[memory_v21_daemon] FAIL (a): probe request returned None",
                      file=sys.stderr)
                return 1
    print(f"[memory_v21_daemon] (a) boot {boot_s*1000:.1f}ms "
          f"(budget {_BOOT_BUDGET_S*1000:.0f}ms) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) p99 warm-recall latency < 100ms
# ---------------------------------------------------------------------------


def _check_p99_latency() -> int:
    from resonance_lattice.memory.daemon import (
        load_or_create_authkey,
        request_recall,
        RecallRequest,
    )
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root, encoder=ZeroEncoder())
        _seed_50_rows(memory)
        address = isolated_daemon_address(root)
        with booted_daemon(memory, address=address) as (server, _):
            if server._listener is None:
                print("[memory_v21_daemon] FAIL (b): server boot failed",
                      file=sys.stderr)
                return 1
            authkey = load_or_create_authkey(root)
            for _ in range(20):  # warmup
                request_recall(
                    RecallRequest(query="warmup", cosine_floor=0.0,
                                  top1_top2_gap=0.0, min_recurrence=1),
                    address=address, authkey=authkey, timeout_ms=2000,
                )
            samples = []
            for _ in range(200):
                t0 = time.perf_counter()
                reply = request_recall(
                    RecallRequest(query="benchmark query",
                                  cosine_floor=0.0, top1_top2_gap=0.0,
                                  min_recurrence=1),
                    address=address, authkey=authkey, timeout_ms=2000,
                )
                samples.append((time.perf_counter() - t0) * 1000)
                if reply is None:
                    print("[memory_v21_daemon] FAIL (b): request returned None",
                          file=sys.stderr)
                    return 1
            samples.sort()
            p50 = samples[100]
            p99 = samples[198]

    if p99 >= _P99_GATE_MS:
        print(f"[memory_v21_daemon] FAIL (b): p99 {p99:.2f}ms >= "
              f"gate {_P99_GATE_MS}ms (p50 {p50:.2f}ms)", file=sys.stderr)
        return 1
    print(f"[memory_v21_daemon] (b) p99 {p99:.2f}ms / p50 {p50:.2f}ms "
          f"(gate {_P99_GATE_MS}ms) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) Daemon crash → fail-open
# ---------------------------------------------------------------------------


def _check_fail_open_on_crash() -> int:
    from resonance_lattice.memory.daemon import (
        load_or_create_authkey,
        request_recall,
        RecallRequest,
    )
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root, encoder=ZeroEncoder())
        _seed_50_rows(memory)
        address = isolated_daemon_address(root)
        authkey = load_or_create_authkey(root)

        # Boot, confirm reachable, then exit the context (simulates a
        # graceful crash — the listener is gone but the address may
        # linger as a stale socket file).
        with booted_daemon(memory, address=address) as (server, _):
            if server._listener is None:
                print("[memory_v21_daemon] FAIL (c): server boot failed",
                      file=sys.stderr)
                return 1
            reply = request_recall(
                RecallRequest(query="probe", cosine_floor=0.0,
                              top1_top2_gap=0.0, min_recurrence=1),
                address=address, authkey=authkey, timeout_ms=2000,
            )
            if reply is None:
                print("[memory_v21_daemon] FAIL (c): live daemon refused probe",
                      file=sys.stderr)
                return 1

        # Wait long enough for the OS to fully release the socket.
        time.sleep(0.2)

        t0 = time.perf_counter()
        reply2 = request_recall(
            RecallRequest(query="post-crash", cosine_floor=0.0,
                          top1_top2_gap=0.0, min_recurrence=1),
            address=address, authkey=authkey,
            timeout_ms=_FAIL_OPEN_TIMEOUT_MS,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if reply2 is not None:
            print(f"[memory_v21_daemon] FAIL (c): post-crash request "
                  f"returned non-None: {reply2}", file=sys.stderr)
            return 1
        if elapsed_ms > _FAIL_OPEN_TIMEOUT_MS * 5:
            print(f"[memory_v21_daemon] FAIL (c): post-crash hang "
                  f"({elapsed_ms:.1f}ms > {_FAIL_OPEN_TIMEOUT_MS*5}ms)",
                  file=sys.stderr)
            return 1
    print(f"[memory_v21_daemon] (c) post-crash request returns None in "
          f"{elapsed_ms:.1f}ms (fail-open) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) Doctor recovery lines for each known failure mode
# ---------------------------------------------------------------------------


def _check_doctor_recovery_lines() -> int:
    from resonance_lattice.memory.daemon import (
        daemon_socket_address,
        diagnose,
    )
    from resonance_lattice.memory.store import Memory

    # Failure mode 1: per-user root present but band missing.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        root.mkdir(parents=True)
        report = diagnose(root)
        store_check = next(c for c in report.checks if c["name"] == "store")
        if store_check["ok"] or "recreate via" not in store_check["message"]:
            print(f"[memory_v21_daemon] FAIL (d.1): missing-band recovery "
                  f"line absent: {store_check}", file=sys.stderr)
            return 1

    # Failure mode 2: clean store, no daemon — should report
    # "not running (expected)".
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root, encoder=ZeroEncoder())
        _seed_50_rows(memory)
        report = diagnose(root)
        daemon_check = next(c for c in report.checks if c["name"] == "daemon")
        if "not running" not in daemon_check["message"]:
            print(f"[memory_v21_daemon] FAIL (d.2): clean-store daemon "
                  f"check missing 'not running': "
                  f"{daemon_check['message']!r}", file=sys.stderr)
            return 1

    # Failure mode 3: encoder revision mismatch. Bind the test daemon
    # to the *canonical* socket address so `diagnose` actually probes
    # it — earlier suites used `_isolated_address` here, which the
    # production diagnose path never sees, so the encoder-mismatch
    # branch was dead code.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root, encoder=ZeroEncoder())
        _seed_50_rows(memory)
        canonical = daemon_socket_address(root)
        with booted_daemon(memory, address=canonical,
                           encoder_revision="rev-A") as (server, _):
            if server._listener is None:
                print("[memory_v21_daemon] FAIL (d.3): server boot failed",
                      file=sys.stderr)
                return 1
            report = diagnose(root, encoder_revision="rev-B")
            daemon_check = next(c for c in report.checks
                                 if c["name"] == "daemon")
            if "revision" not in daemon_check["message"]:
                print(f"[memory_v21_daemon] FAIL (d.3): mismatch-revision "
                      f"recovery line absent: "
                      f"{daemon_check['message']!r}", file=sys.stderr)
                return 1
    print("[memory_v21_daemon] (d) doctor recovery lines for "
          "missing-band + not-running + encoder-mismatch OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _check_ready_marker() -> int:
    """The daemon writes `.recall.ready` after encoder + snapshot are
    loaded so hook callers can poll for it instead of guessing at a
    boot-wait timeout. Marker is removed on clean exit.
    """
    from resonance_lattice.memory.daemon import daemon_ready_path
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root, encoder=ZeroEncoder())
        _seed_50_rows(memory)

        # Plant a stale marker so we can verify the daemon clears it
        # before declaring readiness — a stale marker from a crashed
        # previous daemon must not fake-out a fresh hook poll.
        ready_path = daemon_ready_path(root)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text("stale-pid", encoding="utf-8")

        address = isolated_daemon_address(root)
        with booted_daemon(memory, address=address) as (server, _):
            # Poll briefly — marker is written immediately after
            # listener bind in serve_forever; booted_daemon returns
            # when listener is up, so the marker should appear within
            # a few hundred ms.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not ready_path.exists():
                time.sleep(0.02)
            if not ready_path.exists():
                print("[memory_v21_daemon] FAIL (e): marker not written after "
                      "daemon boot", file=sys.stderr)
                return 1
            content = ready_path.read_text(encoding="utf-8").strip()
            if content == "stale-pid":
                print("[memory_v21_daemon] FAIL (e): stale marker not cleared "
                      "before fresh daemon advertised", file=sys.stderr)
                return 1
        # Server stopped — marker should be cleaned up.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and ready_path.exists():
            time.sleep(0.02)
        if ready_path.exists():
            print("[memory_v21_daemon] FAIL (e): marker not removed on clean "
                  "exit", file=sys.stderr)
            return 1
    print("[memory_v21_daemon] (e) ready marker write-on-boot + clear-on-exit OK",
          file=sys.stderr)
    return 0


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_boot_budget,
        _check_p99_latency,
        _check_fail_open_on_crash,
        _check_doctor_recovery_lines,
        _check_ready_marker,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_daemon] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
