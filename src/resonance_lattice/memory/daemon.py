"""Recall daemon — long-lived process serving §0.6 retrieval over IPC.

The §12.4 retrieval gate (p50 ≤ 30ms / p95 ≤ 80ms) measures the warm
path: encoder + (rows, band) snapshot loaded once, queries served per
request. The daemon is the process that holds that cache; the
UserPromptSubmit hook subprocess connects via `multiprocessing.connection`,
sends a `RecallRequest`, receives `RecallReply` in sub-100ms.

IPC: stdlib `multiprocessing.connection` Listener / Client. AF_UNIX
socket on POSIX (`<root>/.recall.sock`); named pipe on Windows
(`\\\\.\\pipe\\rlat-memory-<user-id>`). Authkey is random bytes
written 0o600 on first daemon boot — prevents cross-tenant connect
on shared hosts.

Contract: §0.8 + §5.2.1.
"""

from __future__ import annotations

import os
import secrets
import socket as _socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import connection as _conn
from pathlib import Path

from ..field.encoder import Encoder
from ._common import workspace_hash
from .recall import rank
from .store import Memory, Row

DEFAULT_IDLE_EXIT_SECONDS = 1800  # 30 min per §0.8
# Client-side timeout: §5.2.1 specs 200ms targeting POSIX (AF_UNIX connect
# is sub-millisecond). On Windows, `multiprocessing.connection`'s named-pipe
# `Client(...)` handshake (CreateFileW + WaitNamedPipe + answer_challenge)
# costs ~440-900ms from a fresh subprocess — every Claude Code hook fire
# spawns a fresh subprocess, so each fire pays this cost. Empirically
# verified on Windows 11; bumping to 2000ms gives connect headroom and
# leaves enough budget for send/poll/recv. Total worst-case hook latency
# ~1.5s on Windows cold path; warm steady-state (e.g. Anthropic-gated
# probe loops in tests) still hits the §0.6 30/80ms gate.
DEFAULT_TIMEOUT_MS = 2000
DEFAULT_RELOAD_POLL_SECONDS = 1.0
SERVER_VERSION = 1

# Fraction of the request's wall-clock budget reserved for the IPC
# connect handshake; the remainder pays for send + poll + recv. Heuristic
# — connect is the most likely place to wedge (stale socket / closed
# pipe), but a hung daemon mid-reply can also eat the budget, so leaving
# headroom for the read side matters.
_CONNECT_BUDGET_FRACTION = 0.6

_AUTHKEY_FILENAME = ".daemon_authkey"
_SOCKET_FILENAME = ".recall.sock"


@dataclass(frozen=True)
class RecallRequest:
    query: str
    cwd_hash: str | None = None
    top_k: int = 5
    cosine_floor: float = 0.7
    top1_top2_gap: float = 0.05
    min_recurrence: int = 3


@dataclass(frozen=True)
class RecallReply:
    hits: list[dict]
    encoder_revision: str
    server_version: int = SERVER_VERSION
    error: str | None = None


# ---------------------------------------------------------------------------
# Address + authkey management
# ---------------------------------------------------------------------------


def daemon_socket_address(root: Path):
    """Return the IPC address bound to a per-user `root` directory.

    POSIX: AF_UNIX path under `<root>/.recall.sock`.
    Windows: named pipe `\\\\.\\pipe\\rlat-memory-<sha256[:6](root)>`.

    Named pipes have a global namespace on Windows; the cwd-hash
    suffix keeps two users' daemons from colliding on a shared host.
    """
    if os.name == "nt":
        suffix = workspace_hash(str(root))
        return r"\\.\pipe\rlat-memory-" + suffix
    return str(root / _SOCKET_FILENAME)


def load_or_create_authkey(root: Path) -> bytes:
    """Read or randomly initialise the per-root authkey.

    32 bytes from `secrets.token_bytes`; written with restrictive mode
    on POSIX (0o600). Both the daemon and its clients read this file —
    the same root holds them, so the authkey establishes "same user"
    without needing a network handshake.
    """
    p = root / _AUTHKEY_FILENAME
    if p.exists():
        return p.read_bytes()
    root.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    p.write_bytes(key)
    if os.name != "nt":
        os.chmod(p, 0o600)
    return key


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class DaemonServer:
    """Long-lived recall server bound to one per-user store.

    The server holds a cached `(rows, band)` snapshot and reloads it
    when `<root>/memory.npz` mtime changes. Per-request cost is just
    `rank(...)` over the cached snapshot — sub-millisecond at typical
    band sizes.
    """

    def __init__(
        self,
        *,
        store: Memory,
        encoder: Encoder,
        encoder_revision: str = "unknown",
        address=None,
        authkey: bytes | None = None,
        idle_exit_seconds: int = DEFAULT_IDLE_EXIT_SECONDS,
        reload_poll_seconds: float = DEFAULT_RELOAD_POLL_SECONDS,
    ):
        self.store = store
        self.encoder = encoder
        self.encoder_revision = encoder_revision
        self.address = address or daemon_socket_address(store.root)
        self.authkey = authkey or load_or_create_authkey(store.root)
        self.idle_exit_seconds = idle_exit_seconds
        self.reload_poll_seconds = reload_poll_seconds

        self._rows: list[Row] = []
        self._band = None
        self._band_path = store.root / "memory.npz"
        self._band_mtime: float = 0.0
        self._last_request_at: float = time.monotonic()
        self._stop = threading.Event()
        self._listener: _conn.Listener | None = None

    # -- snapshot management ----------------------------------------------

    def reload_snapshot(self) -> None:
        rows, band = self.store.read_all()
        self._rows = rows
        self._band = band
        if self._band_path.exists():
            self._band_mtime = self._band_path.stat().st_mtime

    def _maybe_reload(self) -> None:
        if not self._band_path.exists():
            return
        mtime = self._band_path.stat().st_mtime
        if mtime > self._band_mtime:
            self.reload_snapshot()

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass

    def serve_forever(self) -> None:
        self.reload_snapshot()
        self._listener = _conn.Listener(self.address, authkey=self.authkey)

        # Watchdog thread polls `_stop` + idle deadline, closes the
        # listener on either trigger so `accept()` raises and the main
        # loop exits. This is the cross-platform replacement for poking
        # `_listener._listener._socket.settimeout(...)` — that attribute
        # path doesn't exist on Windows named-pipe listeners, so the
        # POSIX-only timeout hack left Windows daemons unable to honour
        # the §0.8 30-min idle exit. The watchdog approach closes the
        # listener uniformly on both platforms.
        watchdog = threading.Thread(
            target=self._watchdog_loop, daemon=True
        )
        watchdog.start()
        try:
            while not self._stop.is_set():
                try:
                    conn = self._listener.accept()
                except (OSError, _socket.timeout):
                    # Watchdog closed the listener (idle or stop), or
                    # the OS interrupted the accept. Re-check the loop
                    # condition.
                    continue
                with conn:
                    # Reload BEFORE handling so the first request after a
                    # `memory.npz` write sees fresh rows. Reloading after
                    # was a one-request-stale window every time a writer
                    # mutated the snapshot.
                    self._maybe_reload()
                    self._handle_one(conn)
        finally:
            self._stop.set()
            try:
                self._listener.close()
            except Exception:
                pass
            self._listener = None
            watchdog.join(timeout=1.0)
            # Best-effort socket-file cleanup on POSIX so a fresh
            # daemon boot isn't blocked by a stale socket inode.
            if os.name != "nt" and isinstance(self.address, str):
                try:
                    os.unlink(self.address)
                except FileNotFoundError:
                    pass

    def _watchdog_loop(self) -> None:
        """Idle/stop poller; unwedges `accept()` so the main loop exits.

        Sets `_stop` *before* unwedging — otherwise the main loop's
        OSError handler would `continue` straight back into accept().

        Two unwedge paths:
          - POSIX: closing the listener socket aborts `accept()`
            immediately. Cheap, no handshake.
          - Windows: closing the named-pipe handle is *not* guaranteed
            to wake a pending `WaitForMultipleObjects` (MSDN says
            behavior is undefined for cross-thread `CloseHandle` on a
            handle in pending overlapped I/O). The reliable wake is to
            self-connect: a client connect targets the listener, which
            satisfies `ConnectNamedPipe`, accept() returns, the main
            loop's `_handle_one` recv()s an EOF or a challenge mismatch,
            then the loop iterates, sees `_stop`, and exits cleanly.
        """
        while not self._stop.is_set():
            if self._idle_expired():
                self._stop.set()
                break
            self._stop.wait(self.reload_poll_seconds)
        if os.name == "nt":
            self._wake_accept_via_self_connect()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass

    def _wake_accept_via_self_connect(self) -> None:
        """Connect to ourselves with a tight budget. The connect either
        completes the handshake (accept() returns a valid Connection,
        which `_handle_one` then drains and returns) or fails on the
        challenge — both wake accept().
        """
        try:
            client = _connect_with_timeout(
                self.address, self.authkey, timeout_s=0.5
            )
        except Exception:
            return
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass

    def _idle_expired(self) -> bool:
        idle = time.monotonic() - self._last_request_at
        return idle >= self.idle_exit_seconds

    # -- request handling --------------------------------------------------

    def _handle_one(self, conn: _conn.Connection) -> None:
        self._last_request_at = time.monotonic()
        try:
            payload = conn.recv()
        except (EOFError, OSError):
            return
        if not isinstance(payload, dict) or "request" not in payload:
            conn.send(asdict(RecallReply(
                hits=[], encoder_revision=self.encoder_revision,
                error="invalid request envelope",
            )))
            return
        req = RecallRequest(**payload["request"])
        try:
            hits = rank(
                req.query,
                rows=self._rows,
                band=self._band,
                encoder=self.encoder,
                cwd_hash=req.cwd_hash,
                top_k=req.top_k,
                cosine_floor=req.cosine_floor,
                top1_top2_gap=req.top1_top2_gap,
                min_recurrence=req.min_recurrence,
            )
        except Exception as exc:
            conn.send(asdict(RecallReply(
                hits=[], encoder_revision=self.encoder_revision,
                error=f"{type(exc).__name__}: {exc}",
            )))
            return
        serialised = [
            {"row": asdict(h.row), "cosine": h.cosine} for h in hits
        ]
        conn.send(asdict(RecallReply(
            hits=serialised, encoder_revision=self.encoder_revision,
        )))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _connect_with_timeout(address, authkey: bytes, timeout_s: float):
    """`_conn.Client(...)` with a wall-clock budget.

    `multiprocessing.connection.Client` has no native timeout: POSIX
    `SocketClient` calls `socket.connect()` without a deadline; Windows
    `PipeClient` retries `WaitNamedPipe` for a hardcoded ~1s. A wedged
    socket / stale pipe could otherwise hang the UserPromptSubmit hook
    for a full second-plus, defeating the §16.5 / §18.5 fail-open
    contract.

    Runs the connect on a daemon worker thread; if it doesn't return
    inside the budget, returns None (the worker is detached and dies on
    process exit). Cross-platform — same code-path POSIX + Windows.
    """
    box: dict[str, object] = {}

    def _attempt() -> None:
        try:
            box["conn"] = _conn.Client(address, authkey=authkey)
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            box["error"] = exc

    worker = threading.Thread(target=_attempt, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        return None
    return box.get("conn")


def request_recall(
    request: RecallRequest,
    *,
    address,
    authkey: bytes,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> RecallReply | None:
    """Connect, send a single request, read reply, close.

    Returns None on connect-fail, timeout, or any IPC error — the hook
    callers treat None as "no injection" per the §16.5 / §18.5
    fail-open contract. Never raises (modulo programming errors like
    invalid request types).
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    connect_budget = max(0.001, (timeout_ms / 1000.0) * _CONNECT_BUDGET_FRACTION)
    conn = _connect_with_timeout(address, authkey, connect_budget)
    if conn is None:
        return None
    try:
        conn.send({"request": asdict(request)})
        remaining = max(0.001, deadline - time.monotonic())
        if not conn.poll(remaining):
            return None
        payload = conn.recv()
        if not isinstance(payload, dict):
            return None
        return RecallReply(**payload)
    except (EOFError, OSError, ConnectionResetError):
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------


@dataclass
class DoctorReport:
    """One probe outcome — name + status + actionable recovery line."""
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, ok: bool, message: str) -> None:
        self.checks.append({"name": name, "ok": ok, "message": message})


def diagnose(root: Path, *, encoder_revision: str | None = None) -> DoctorReport:
    """Probe the per-user root. Returns a `DoctorReport` even on
    everything-OK — the operator wants to see what was checked.
    """
    report = DoctorReport()

    if not root.exists():
        report.add("root", False,
                   f"per-user root missing at {root}; create it via "
                   f"`rlat memory add <text>` to bootstrap")
        return report
    report.add("root", True, f"per-user root present at {root}")

    sidecar = root / "sidecar.jsonl"
    band = root / "memory.npz"
    if not sidecar.exists() or not band.exists():
        report.add("store", False,
                   f"memory.npz or sidecar.jsonl missing under {root}; "
                   f"recreate via `rlat memory add <text>`")
    else:
        report.add("store", True,
                   f"sidecar + band present ({sidecar.stat().st_size} + "
                   f"{band.stat().st_size} bytes)")

    address = daemon_socket_address(root)
    is_posix_path = isinstance(address, str) and not address.startswith(r"\\.\pipe")

    # POSIX: cheap pre-check via filesystem; Windows named pipes have
    # no static existence check, so we attempt a connect with a tight
    # timeout (50ms — interactive `doctor` shouldn't hang for half a
    # second when no daemon is running). On success we fall through to
    # the encoder-revision probe with a longer budget.
    if is_posix_path and not Path(address).exists():
        report.add("daemon", True,
                   "daemon not running (expected — lazy-started on first "
                   "hook fire)")
    else:
        authkey = load_or_create_authkey(root)
        # Windows named-pipe first-connect from a fresh subprocess takes
        # ~440-900ms (see daemon.py DEFAULT_TIMEOUT_MS comment); the
        # earlier 50ms misclassified live-but-cold daemons as "not running".
        probe_timeout = 500 if is_posix_path else 1500
        reply = request_recall(
            RecallRequest(query="__doctor_probe__", cwd_hash=None,
                           top_k=1, cosine_floor=0.0, top1_top2_gap=0.0,
                           min_recurrence=1),
            address=address, authkey=authkey, timeout_ms=probe_timeout,
        )
        if reply is None and not is_posix_path:
            # Windows + no daemon: indistinguishable from POSIX
            # "socket file missing" — report the same way.
            report.add("daemon", True,
                       "daemon not running (expected — lazy-started on first "
                       "hook fire)")
            return report
        if reply is None:
            report.add("daemon", False,
                       f"socket exists at {address} but daemon unreachable; "
                       f"remove socket file and let the next hook fire "
                       f"restart the daemon")
        elif (
            encoder_revision is not None
            and reply.encoder_revision != "unknown"
            and reply.encoder_revision != encoder_revision
        ):
            report.add("daemon", False,
                       f"daemon encoder revision {reply.encoder_revision!r} "
                       f"!= installed {encoder_revision!r}; remove socket at "
                       f"{address} and let the next hook fire restart the "
                       f"daemon under the current encoder")
        else:
            report.add("daemon", True,
                       f"daemon reachable at {address} "
                       f"(encoder {reply.encoder_revision})")
    return report
