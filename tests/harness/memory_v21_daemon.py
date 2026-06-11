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
)


_BOOT_BUDGET_S = 2.0  # D.10 (a): boot-time on warm encoder cache
_P99_GATE_MS = 100.0  # D.10 (b): per-request warm latency
_FAIL_OPEN_TIMEOUT_MS = 200  # client-side budget for D.10 (c)


def _new_store(root):
    """A fresh `ExperienceClaimStore` over `root` — the daemon backend
    that succeeds `Memory`."""
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    return ExperienceClaimStore(root=root, encoder=ZeroEncoder())


def _seed_50_rows(memory) -> None:
    """50 deterministic capture `event` claims under one workspace tag."""
    import numpy as np
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.store import seed_tallies_for_rung
    from resonance_lattice.state.claim import Claim, ExperienceFacts, derive_origin

    cwd_tag = workspace_tag_for_cwd("/proj")
    corr, fals = seed_tallies_for_rung("medium")
    claims = []
    for i in range(50):
        th = f"daemonfixturetx{i:04d}"
        claims.append(Claim(
            claim_id=f"01HZDAEMONFIXTURE{i:011d}",
            source="experience",
            kind="event",
            content=f"capture row {i} for daemon harness fixture",
            created_at="2026-05-18T00:00:00Z",
            corroboration=corr,
            falsification=fals,
            trust_as_of="",
            state="active",
            parent_ids=(),
            facts=ExperienceFacts(
                polarity=("factual", cwd_tag),
                recurrence_count=1,
                criticality="normal",
                created_under_intent_kind="none",
                transcript_hash=th,
                origin=derive_origin(th),
                last_corroborated_at="2026-05-18T00:00:00Z",
                is_bad=False,
            ),
        ))
    memory.write_many(
        claims, embeddings=np.zeros((50, 768), dtype="float32"))


# ---------------------------------------------------------------------------
# (a) Boot-time encoder load < budget on warm cache
# ---------------------------------------------------------------------------


def _check_boot_budget() -> int:
    from resonance_lattice.memory.daemon import (
        load_or_create_authkey,
        request_recall,
        RecallRequest,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = _new_store(root)
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

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = _new_store(root)
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

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = _new_store(root)
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
        memory = _new_store(root)
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
        memory = _new_store(root)
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

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = _new_store(root)
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


def _check_corpus_band_recall() -> int:
    """(f) daemon corpus-binding (S3 d3): a request carrying `km_path` returns
    the workspace corpus band's matching claim in `reply.band_hits` — ranked by
    the same query embedding, source-tagged, with its own 0-based rank — even
    with an EMPTY experience store (the loop must not depend on experience
    memory). Hermetic: a `FixedEncoder` plants the query vector + the band rows
    are planted to match, so cosines are real with no model load.
    """
    import numpy as np

    from resonance_lattice.memory.daemon import (
        RecallRequest,
        load_or_create_authkey,
        request_recall,
    )
    from resonance_lattice.store import archive

    from ._testutil import FixedEncoder
    from ._testutil import build_corpus as _build
    from ._testutil import make_corpus_claim

    with tempfile.TemporaryDirectory() as td:
        # A real .rlat (base band is zero under the patched encoder — we use
        # only its insight band, written explicitly with planted vectors).
        km = _build(Path(td) / "corpus", {"a.md": "# A\n\nbody text here."})
        src = archive.read(km).registry[0].passage_id
        e0 = np.zeros(768, dtype="float32"); e0[0] = 1.0
        e1 = np.zeros(768, dtype="float32"); e1[1] = 1.0  # orthogonal → cos 0
        band = np.vstack([e0, e1])
        claims = [
            make_corpus_claim("matches the query", [src], state="active"),
            make_corpus_claim("orthogonal claim", [src], state="active"),
        ]
        archive.write_insight_layer_in_place(km, claims, band)
        target_id = claims[0].claim_id

        root = Path(td) / "u"
        memory = _new_store(root)  # EMPTY experience store — decoupled path
        address = isolated_daemon_address(root)
        with booted_daemon(
            memory, address=address, encoder=FixedEncoder(e0),
        ) as (server, _):
            if server._listener is None:
                print("[memory_v21_daemon] FAIL (f): boot failed", file=sys.stderr)
                return 1
            reply = request_recall(
                RecallRequest(query="q", km_path=str(km), cosine_floor=0.0,
                              top1_top2_gap=0.0, min_recurrence=1),
                address=address, authkey=load_or_create_authkey(root),
                timeout_ms=2000,
            )
    if reply is None:
        print("[memory_v21_daemon] FAIL (f): request returned None", file=sys.stderr)
        return 1
    if reply.hits:
        print(f"[memory_v21_daemon] FAIL (f): empty experience store but "
              f"{len(reply.hits)} experience hits", file=sys.stderr)
        return 1
    if len(reply.band_hits) != 1:
        print(f"[memory_v21_daemon] FAIL (f): expected 1 band hit (the floor "
              f"drops the orthogonal claim), got {len(reply.band_hits)}",
              file=sys.stderr)
        return 1
    top = reply.band_hits[0]
    if (top["claim_id"] != target_id or top["source"] != "corpus"
            or top["rank"] != 0):
        print(f"[memory_v21_daemon] FAIL (f): wrong top band hit: {top}",
              file=sys.stderr)
        return 1
    print("[memory_v21_daemon] (f) corpus band recall via daemon (km_path → "
          "band_hits, empty experience store) OK", file=sys.stderr)
    return 0


def _check_corpus_band_cache() -> int:
    """(g) `_corpus_band` caches by km mtime: a 2nd call on unchanged mtime
    returns the same loaded object (no re-read); an mtime bump forces a reload;
    a missing path / a .rlat with no insight layer returns None (fail-soft)."""
    import os

    import numpy as np

    from resonance_lattice.memory.daemon import DaemonServer
    from resonance_lattice.store import archive

    from ._testutil import build_corpus as _build
    from ._testutil import make_corpus_claim

    with tempfile.TemporaryDirectory() as td:
        km = _build(Path(td) / "corpus", {"a.md": "# A\n\nbody text here."})
        src = archive.read(km).registry[0].passage_id
        band = np.zeros((1, 768), dtype="float32"); band[0, 0] = 1.0
        archive.write_insight_layer_in_place(
            km, [make_corpus_claim("c", [src], state="active")], band)

        root = Path(td) / "u"
        server = DaemonServer(
            store=_new_store(root), encoder=ZeroEncoder(),
            address=isolated_daemon_address(root),
        )
        a = server._corpus_band(str(km))
        b = server._corpus_band(str(km))
        if a is None or b is None or a[0] is not b[0]:
            print("[memory_v21_daemon] FAIL (g): not cached on unchanged mtime",
                  file=sys.stderr)
            return 1
        t = Path(km).stat().st_mtime
        os.utime(km, (t + 100, t + 100))
        c = server._corpus_band(str(km))
        if c is None or c[0] is a[0]:
            print("[memory_v21_daemon] FAIL (g): did not reload on mtime change",
                  file=sys.stderr)
            return 1
        if server._corpus_band(str(Path(td) / "nope.rlat")) is not None:
            print("[memory_v21_daemon] FAIL (g): missing km not None",
                  file=sys.stderr)
            return 1
        km2 = _build(Path(td) / "c2", {"b.md": "# B\n\nno insight layer."})
        if server._corpus_band(str(km2)) is not None:
            print("[memory_v21_daemon] FAIL (g): no-insight-layer km not None",
                  file=sys.stderr)
            return 1
    print("[memory_v21_daemon] (g) _corpus_band cache + reload + fail-soft OK",
          file=sys.stderr)
    return 0


def _check_request_field_tolerance() -> int:
    """(h) `_handle_one` tolerates a request dict carrying UNKNOWN keys (a
    newer client with extra wire fields) by dropping them, and never crashes
    the serve loop on a malformed request — a missing required `query` yields
    a clean error reply, not an exception. Guards the §0.8 fail-open contract
    at the request-construction boundary: a warm daemon must outlive a client
    that learns a new field (the S3 `km_path` addition was exactly this case)."""
    from resonance_lattice.memory.daemon import DaemonServer

    class _FakeConn:
        def __init__(self, payload):
            self._payload = payload
            self.sent = None

        def recv(self):
            return self._payload

        def send(self, obj):
            self.sent = obj

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        server = DaemonServer(
            store=_new_store(root), encoder=ZeroEncoder(),
            address=isolated_daemon_address(root),
        )
        server.reload_snapshot()  # empty snapshot — no IPC, direct call

        # Unknown key alongside a valid query → dropped, reply is clean.
        conn = _FakeConn({"request": {
            "query": "hi", "cosine_floor": 0.0, "top1_top2_gap": 0.0,
            "min_recurrence": 1, "a_field_from_the_future": 123,
        }})
        server._handle_one(conn)
        if not isinstance(conn.sent, dict) or conn.sent.get("error"):
            print(f"[memory_v21_daemon] FAIL (h): unknown request key not "
                  f"tolerated: {conn.sent}", file=sys.stderr)
            return 1

        # Missing required `query` → clean error reply, never an exception.
        conn2 = _FakeConn({"request": {"cosine_floor": 0.0}})
        try:
            server._handle_one(conn2)
        except Exception as exc:
            print(f"[memory_v21_daemon] FAIL (h): malformed request raised "
                  f"{type(exc).__name__}", file=sys.stderr)
            return 1
        if not isinstance(conn2.sent, dict) or not conn2.sent.get("error"):
            print(f"[memory_v21_daemon] FAIL (h): missing-query should yield an "
                  f"error reply; got {conn2.sent}", file=sys.stderr)
            return 1
    print("[memory_v21_daemon] (h) request-field tolerance + malformed-request "
          "fail-soft OK", file=sys.stderr)
    return 0


def _check_drift_excludes_from_band_recall() -> int:
    """(i) drift→loop interplay (S3-close §B): a corpus claim flipped
    active→stale by the drift cascade DISAPPEARS from the daemon's band_hits —
    `rank_insight_band` is active-only (`is_retrievable`), so a stale claim can
    no longer be recalled and thus can no longer earn trust until reverification
    reactivates it. Composes the two halves the S3 suites otherwise test only in
    isolation (drift cascade vs band recall), pinning the loop's central safety
    property: drift removes a claim from the trust loop's input."""
    import os

    import numpy as np

    from resonance_lattice.memory.daemon import (
        RecallRequest,
        load_or_create_authkey,
        request_recall,
    )
    from resonance_lattice.state.claim_lifecycle import propagate_drift
    from resonance_lattice.store import archive

    from ._testutil import FixedEncoder
    from ._testutil import build_corpus as _build
    from ._testutil import make_corpus_claim

    with tempfile.TemporaryDirectory() as td:
        km = _build(Path(td) / "corpus", {"a.md": "# A\n\nbody text here."})
        src = archive.read(km).registry[0].passage_id
        e0 = np.zeros(768, dtype="float32"); e0[0] = 1.0
        # ACTIVE claim whose stored source hash matches the (simulated) source,
        # so it does NOT drift initially and IS recallable.
        claim = make_corpus_claim(
            "matches the query", [src], source_hashes=["h_orig"], state="active")
        archive.write_insight_layer_in_place(km, [claim], e0.reshape(1, 768))

        root = Path(td) / "u"
        memory = _new_store(root)
        address = isolated_daemon_address(root)
        with booted_daemon(
            memory, address=address, encoder=FixedEncoder(e0),
        ) as (server, _):
            if server._listener is None:
                print("[memory_v21_daemon] FAIL (i): boot failed", file=sys.stderr)
                return 1
            authkey = load_or_create_authkey(root)

            def _band_ids():
                reply = request_recall(
                    RecallRequest(query="q", km_path=str(km), cosine_floor=0.0,
                                  top1_top2_gap=0.0, min_recurrence=1),
                    address=address, authkey=authkey, timeout_ms=2000)
                if reply is None:
                    return None
                return {h["claim_id"] for h in reply.band_hits}

            # Baseline: the active claim is in band_hits.
            ids = _band_ids()
            if ids is None or claim.claim_id not in ids:
                print(f"[memory_v21_daemon] FAIL (i): active claim absent from "
                      f"baseline band_hits: {ids!r}", file=sys.stderr)
                return 1

            # Drift the cited source (its hash no longer matches the stored
            # one) → the cascade flips active→stale with a falsification.
            updated, drifted = propagate_drift([claim], {src: "h_changed"})
            if drifted != [0] or updated[0].state != "stale":
                print(f"[memory_v21_daemon] FAIL (i): drift cascade did not "
                      f"stale the claim: drifted={drifted} "
                      f"state={updated[0].state!r}", file=sys.stderr)
                return 1
            archive.write_insight_layer_in_place(km, updated, e0.reshape(1, 768))
            # Guarantee the daemon's mtime-keyed corpus cache reloads.
            t = Path(km).stat().st_mtime
            os.utime(km, (t + 100, t + 100))

            # Now the stale claim must be GONE from band_hits — frozen out of
            # the trust loop until reverification flips it back to active.
            ids2 = _band_ids()
            if ids2 is None or claim.claim_id in ids2:
                print(f"[memory_v21_daemon] FAIL (i): stale claim still in "
                      f"band_hits (drift did not remove it from recall): "
                      f"{ids2!r}", file=sys.stderr)
                return 1
    print("[memory_v21_daemon] (i) drift→loop: a drifted-to-stale claim drops "
          "from band_hits OK", file=sys.stderr)
    return 0


def _check_attribute_serve_dedup() -> int:
    """(j) daemon attribute serve: the corpus band's user-world ATTRIBUTE claims
    reach the hook via `reply.attribute_hits`, content-bearing and DEDUPED to the
    newest value per subject. Two `ps_version` attributes share a key; the OLDER
    one's vector sits ON the query (cos 1.0) while the NEWER one is off-axis
    (cos 0.6). Newest-by-created_at must win — the served fact is the new value,
    proving the dedup keys on time, not similarity, end-to-end through IPC.
    """
    import numpy as np

    from resonance_lattice.memory.daemon import (
        RecallRequest,
        load_or_create_authkey,
        request_recall,
    )
    from resonance_lattice.state.claim import Claim, ExperienceFacts
    from resonance_lattice.store import archive

    from ._testutil import FixedEncoder
    from ._testutil import build_corpus as _build

    def _attr(cid, content, *, created_at):
        return Claim(
            claim_id=cid, source="experience", kind="attribute", content=content,
            created_at=created_at, corroboration=3.0, falsification=1.0,
            trust_as_of="", state="active", parent_ids=(),
            facts=ExperienceFacts(
                polarity=("factual",), recurrence_count=1, criticality="high",
                created_under_intent_kind="none", transcript_hash="manual",
                origin="manual", last_corroborated_at=created_at,
                attribute_key="ps_version",
            ),
        )

    with tempfile.TemporaryDirectory() as td:
        km = _build(Path(td) / "corpus", {"a.md": "# A\n\nbody text here."})
        e0 = np.zeros(768, dtype="float32"); e0[0] = 1.0          # query + old
        vnew = np.zeros(768, dtype="float32"); vnew[0] = 0.6; vnew[1] = 0.8  # cos 0.6
        band = np.vstack([e0, vnew])
        claims = [
            _attr("01HZATTRPSOLD0000000000000", "PowerShell 5.1",
                  created_at="2026-06-01T00:00:00Z"),
            _attr("01HZATTRPSNEW0000000000000", "PowerShell 7.4",
                  created_at="2026-06-05T00:00:00Z"),
        ]
        archive.write_insight_layer_in_place(km, claims, band)

        root = Path(td) / "u"
        memory = _new_store(root)  # empty experience store — band-only path
        address = isolated_daemon_address(root)
        with booted_daemon(
            memory, address=address, encoder=FixedEncoder(e0),
        ) as (server, _):
            if server._listener is None:
                print("[memory_v21_daemon] FAIL (j): boot failed", file=sys.stderr)
                return 1
            reply = request_recall(
                RecallRequest(query="q", km_path=str(km), cosine_floor=0.0,
                              top1_top2_gap=0.0, min_recurrence=1),
                address=address, authkey=load_or_create_authkey(root),
                timeout_ms=2000,
            )
    if reply is None:
        print("[memory_v21_daemon] FAIL (j): request returned None", file=sys.stderr)
        return 1
    attrs = getattr(reply, "attribute_hits", [])
    contents = [a.get("content") for a in attrs]
    if "PowerShell 5.1" in contents:
        print(f"[memory_v21_daemon] FAIL (j): stale attribute served — newest-"
              f"wins dedup failed end-to-end: {contents}", file=sys.stderr)
        return 1
    if contents != ["PowerShell 7.4"]:
        print(f"[memory_v21_daemon] FAIL (j): expected only the newest value, "
              f"got {contents}", file=sys.stderr)
        return 1
    if attrs[0].get("attribute_key") != "ps_version":
        print(f"[memory_v21_daemon] FAIL (j): attribute_key not carried on the "
              f"wire: {attrs[0]}", file=sys.stderr)
        return 1
    print("[memory_v21_daemon] (j) daemon attribute serve: newest-per-subject "
          "dedup over IPC (content-bearing attribute_hits) OK", file=sys.stderr)
    return 0


def _check_constraint_serve_all() -> int:
    """(l) daemon constraint serve is ALL-always: standing-constraint and
    tried-and-falsified (negation) claims reach the hook via
    `reply.constraint_hits` even when their vectors are ORTHOGONAL to the
    query (cos 0.0 — under the attribute channel's 0.3 floor, which must
    drop the attribute claim in the same band). Newest-wins dedup applies
    per (kind, attribute_key): the older constraint value must not serve.
    """
    import numpy as np

    from resonance_lattice.memory.daemon import (
        RecallRequest,
        load_or_create_authkey,
        request_recall,
    )
    from resonance_lattice.state.claim import Claim, ExperienceFacts
    from resonance_lattice.store import archive

    from ._testutil import FixedEncoder
    from ._testutil import build_corpus as _build

    def _claim(cid, kind, content, *, created_at, key=""):
        return Claim(
            claim_id=cid, source="experience", kind=kind, content=content,
            created_at=created_at, corroboration=3.0, falsification=1.0,
            trust_as_of="", state="active", parent_ids=(),
            facts=ExperienceFacts(
                polarity=("factual",), recurrence_count=1, criticality="high",
                created_under_intent_kind="none", transcript_hash="manual",
                origin="manual", last_corroborated_at=created_at,
                attribute_key=key,
            ),
        )

    with tempfile.TemporaryDirectory() as td:
        km = _build(Path(td) / "corpus", {"a.md": "# A\n\nbody text here."})
        e0 = np.zeros(768, dtype="float32"); e0[0] = 1.0   # query axis
        def _axis(i):
            v = np.zeros(768, dtype="float32"); v[i] = 1.0
            return v
        band = np.vstack([_axis(2), _axis(3), _axis(4), _axis(5)])  # all cos 0 to e0
        claims = [
            _claim("01HZCONSOLD000000000000000", "constraint",
                   "Capacity must stay at F8 or below.",
                   created_at="2026-06-01T00:00:00Z", key="capacity"),
            _claim("01HZCONSNEW000000000000000", "constraint",
                   "Capacity must stay at F4 or below.",
                   created_at="2026-06-05T00:00:00Z", key="capacity"),
            _claim("01HZNEGATION00000000000000", "negation",
                   "Tried nightly full rebuilds; falsified by the 2026-05 cost record.",
                   created_at="2026-06-02T00:00:00Z"),
            _claim("01HZATTRIBUTE0000000000000", "attribute",
                   "Region is Australia East.",
                   created_at="2026-06-03T00:00:00Z", key="region"),
        ]
        archive.write_insight_layer_in_place(km, claims, band)

        root = Path(td) / "u"
        memory = _new_store(root)
        address = isolated_daemon_address(root)
        with booted_daemon(
            memory, address=address, encoder=FixedEncoder(e0),
        ) as (server, _):
            if server._listener is None:
                print("[memory_v21_daemon] FAIL (l): boot failed", file=sys.stderr)
                return 1
            reply = request_recall(
                RecallRequest(query="q", km_path=str(km), cosine_floor=0.0,
                              top1_top2_gap=0.0, min_recurrence=1),
                address=address, authkey=load_or_create_authkey(root),
                timeout_ms=2000,
            )
    if reply is None:
        print("[memory_v21_daemon] FAIL (l): request returned None", file=sys.stderr)
        return 1
    cons = getattr(reply, "constraint_hits", [])
    served = [(c.get("kind"), c.get("content")) for c in cons]
    if ("constraint", "Capacity must stay at F8 or below.") in served:
        print(f"[memory_v21_daemon] FAIL (l): stale constraint served — "
              f"newest-wins dedup failed: {served}", file=sys.stderr)
        return 1
    if served != [
        ("constraint", "Capacity must stay at F4 or below."),
        ("negation",
         "Tried nightly full rebuilds; falsified by the 2026-05 cost record."),
    ]:
        print(f"[memory_v21_daemon] FAIL (l): serve-all constraint channel "
              f"wrong (orthogonal vectors must still serve): {served}",
              file=sys.stderr)
        return 1
    if getattr(reply, "attribute_hits", []):
        print(f"[memory_v21_daemon] FAIL (l): attribute under the cosine floor "
              f"leaked into attribute_hits: {reply.attribute_hits}",
              file=sys.stderr)
        return 1
    if cons[0].get("claim_id") != "01HZCONSNEW000000000000000":
        print(f"[memory_v21_daemon] FAIL (l): claim_id not carried on the "
              f"wire: {cons[0]}", file=sys.stderr)
        return 1
    print("[memory_v21_daemon] (l) daemon constraint serve: ALL-always over "
          "IPC (no cosine gate, dedup, kinds carried) OK", file=sys.stderr)
    return 0


def _check_reply_forward_compat() -> int:
    """(k) reply decoding tolerates version skew: a NEWER daemon's extra
    reply fields are ignored, and a malformed reply (missing required keys)
    returns None instead of raising through the hook's fail-open contract
    (2026-06 review: RecallReply(**payload) TypeError'd on both)."""
    from resonance_lattice.memory.daemon import RecallReply, _decode_reply

    newer = _decode_reply({
        "hits": [], "encoder_revision": "rev", "server_version": 99,
        "a_field_from_the_future": {"shape": "unknown"},
    })
    if not isinstance(newer, RecallReply) or newer.server_version != 99:
        print(f"[memory_v21_daemon] FAIL (k): newer-server reply rejected "
              f"({newer!r})", file=sys.stderr)
        return 1
    malformed = _decode_reply({"unexpected": True})
    if malformed is not None:
        print(f"[memory_v21_daemon] FAIL (k): malformed reply should be None "
              f"({malformed!r})", file=sys.stderr)
        return 1
    print("[memory_v21_daemon] (k) reply forward-compat (skew-tolerant decode) OK",
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
        _check_corpus_band_recall,
        _check_corpus_band_cache,
        _check_request_field_tolerance,
        _check_drift_excludes_from_band_recall,
        _check_attribute_serve_dedup,
        _check_constraint_serve_all,
        _check_reply_forward_compat,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_daemon] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
