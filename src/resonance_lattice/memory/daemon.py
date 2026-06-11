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
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from multiprocessing import AuthenticationError as _mp_AuthenticationError
from multiprocessing import connection as _conn
from pathlib import Path

from ..field.encoder import Encoder
from ..state.claim import Claim
from ._common import workspace_hash
from .claim_store import (
    BAND_FILE,
    CLAIMS_FILE,
    ExperienceClaimStore,
    _claim_to_row,
)
from .recall import _encode_query, rank_with_diagnostic

DEFAULT_IDLE_EXIT_SECONDS = 1800  # 30 min per §0.8

# Corpus band-recall (S3 d3) — how many corpus insight claims to attribute
# per recall, and the cosine floor below which a corpus claim is too far from
# the prompt to credit. A lower floor than experience recall's 0.7 default:
# attribution tiers by rank (top-2 = primary), so only the closest few corpus
# claims earn weight even at a permissive floor, and the poison guard
# (verdict_confidence × source × provenance, ≤1.0) + the rank-tier decay + the
# Beta model's slowness bound a loose match's impact. Note: a single
# high-confidence user accept still moves a primary corpus claim at full weight
# — there is no multi-outcome threshold on this path (that bound belongs to the
# experience confidence pass). The benchmark tunes these; kept conservative
# (small K) until then.
_BAND_RECALL_TOP_K = 5
_BAND_RECALL_COSINE_FLOOR = 0.3
# Bounded warm cache of corpus insight bands keyed by km path. A user works in
# a handful of corpora; the cap defends a long-lived per-user daemon that roams
# many workspaces from unbounded growth (evicts oldest on overflow).
_CORPUS_CACHE_MAX = 8
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
# Synchronisation marker: the daemon writes this file after encoder load
# + initial snapshot are complete. The hook polls for it on cold-spawn
# instead of guessing at a boot-wait timeout — the v3 bench surfaced
# first-call-after-spawn misses because the boot-wait (100ms) plus
# retry-timeout (500ms) couldn't cover a >1s cold encoder load. Marker
# is removed on daemon exit so a stale file from a crashed daemon
# doesn't fake-out a fresh hook fire.
_READY_FILENAME = ".recall.ready"


@dataclass(frozen=True)
class RecallRequest:
    query: str
    cwd_hash: str | None = None
    top_k: int = 5
    cosine_floor: float = 0.7
    top1_top2_gap: float = 0.05
    min_recurrence: int = 3
    # `intent_kind` opts the post-gate hits into the manifesto re-rank
    # (architecture §"Layer manifesto scoring factors"). None / "none"
    # preserves cosine ordering — the v2.1 wire shape every existing
    # daemon contract pins.
    intent_kind: str | None = None
    # `auto_tune_cold_start` lets the daemon override `cosine_floor` and
    # `min_recurrence` with relaxed values when the row count is below
    # `recall.COLD_START_ROW_THRESHOLD`. Default False preserves the
    # exact wire shape every test pins; the UserPromptSubmit hook opts
    # in so a fresh workspace surfaces something rather than nothing.
    auto_tune_cold_start: bool = False
    # `km_path` is the workspace's primary `.rlat` corpus, resolved by the
    # hook via `state.resolve_primary_km(cwd)`. When set, the daemon ALSO
    # ranks that corpus's insight band (source-agnostic, cosine × trust) with
    # the same query embedding and returns the top corpus claims in
    # `RecallReply.band_hits` — the unified-recall link that lets a resolved
    # intent's attribution carry corpus claim ids (S3 d3). None → the daemon
    # ranks experience claims only, the pre-d3 behaviour. Default None keeps
    # the wire shape every existing test pins.
    km_path: str | None = None


@dataclass(frozen=True)
class RecallReply:
    hits: list[dict]
    encoder_revision: str
    server_version: int = SERVER_VERSION
    error: str | None = None
    # The recurrence floor the daemon actually applied. Defaults to 3
    # (the v1 RecallRequest default) so older clients that ignore the
    # field get the same behaviour they had. The hook reads it instead
    # of hardcoding `_RECURRENCE_M = 3` so the injection-time gate stays
    # in sync with the daemon's actual filter.
    effective_min_recurrence: int = 3
    # Per-recall query-shape diagnostic — serialised `RankDiagnostic`.
    # The hook writes it to `recall_diagnostic.jsonl` so future bench
    # runs can attribute misses ("16/20 sessions had no recall" turns
    # into per-gate `dropped_at` counts).
    diagnostic: dict | None = None
    # Corpus insight-band hits for the request's `km_path` — each a flat
    # `{claim_id, source, rank, cosine}` (the shape the hook stamps into
    # `RecallCache.row_metadata`). Source-agnostic + cache-only: these are
    # NOT injected into the prompt (read-back is H2); they exist so a resolved
    # intent's attribution carries corpus claim ids and the criterion reducer
    # moves corpus trust (S3 d3). `rank` is the corpus ranking's OWN 0-based
    # index (not continued from the experience hits), so the attribution tier
    # is computed per source. Empty when no `km_path` / no corpus layer.
    band_hits: list[dict] = field(default_factory=list)
    # User-world ATTRIBUTE claims from the same insight band, content-bearing
    # and DEDUPED to the newest value per subject (`serve_band_attributes`).
    # Unlike `band_hits` (cache-only) these ARE injected at prompt time: a
    # user-world fact (SKU/role/version/corpus size) is something the agent
    # needs in hand to answer correctly. Each: `{content, attribute_key,
    # created_at, score}`. Empty when no `km_path` / no corpus layer / no
    # attribute claims. Defaulted + last so older clients ignore it.
    attribute_hits: list[dict] = field(default_factory=list)
    # Standing-constraint + tried-and-falsified claims from the same band,
    # served ALL-always (`serve_band_constraints` — the R1-proven design: no
    # cosine floor, no top-k, zero over-blocking measured). Injected at prompt
    # time with the kind-framed headings (`store.serve_framing`). Each:
    # `{claim_id, content, kind, attribute_key, created_at}`. Defaulted +
    # last so older clients ignore it.
    constraint_hits: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Address + authkey management
# ---------------------------------------------------------------------------


def daemon_ready_path(root: Path) -> Path:
    """File the daemon touches after encoder load to signal readiness.

    Hook callers poll for this file's existence on cold-spawn before
    re-attempting connect; sidesteps the brittle fixed-timeout heuristic.
    """
    return root / _READY_FILENAME


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

    The server holds a cached `(claims, band)` snapshot and reloads it
    when `<root>/band.npz` mtime changes. Per-request cost is just
    `rank(...)` over the cached snapshot — sub-millisecond at typical
    band sizes.
    """

    def __init__(
        self,
        *,
        store: ExperienceClaimStore,
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

        self._claims: list[Claim] = []
        self._band = None
        self._band_path = store.root / BAND_FILE
        self._band_mtime: float = 0.0
        self._last_request_at: float = time.monotonic()
        self._stop = threading.Event()
        self._listener: _conn.Listener | None = None
        # Warm cache of per-corpus insight bands (S3 d3): km path → (mtime,
        # insights, band). Reloaded on the .rlat's mtime change so a `rlat
        # consolidate`/`refresh` that rewrites the band is picked up; bounded
        # by `_CORPUS_CACHE_MAX`. Distinct from the per-user experience
        # snapshot above — the daemon is per-user but ranks whichever corpus
        # the request's cwd resolved to, never bound to one.
        self._corpus_cache: dict = {}

    # -- snapshot management ----------------------------------------------

    def reload_snapshot(self) -> None:
        claims, band = self.store.read_all_with_band()
        self._claims = claims
        self._band = band
        if self._band_path.exists():
            self._band_mtime = self._band_path.stat().st_mtime

    def _corpus_band(self, km_path: str):
        """Cached `(insights, band)` for a workspace corpus `.rlat`, reloaded
        on the archive's mtime change. Returns `None` when the path is
        unreadable or has no insight layer — corpus recall is best-effort, so
        a missing/half-built corpus simply yields no band hits, never an
        error. Reads only the (small) insight band via
        `archive.read_insight_layer`, not the full archive.
        """
        from ..store import archive

        try:
            mtime = Path(km_path).stat().st_mtime
        except OSError:
            return None
        cached = self._corpus_cache.get(km_path)
        if cached is not None and cached[0] == mtime:
            return cached[1], cached[2]
        try:
            layer = archive.read_insight_layer(km_path)
        except Exception:
            return None
        if layer is None:
            return None
        insights, band = layer
        if (len(self._corpus_cache) >= _CORPUS_CACHE_MAX
                and km_path not in self._corpus_cache):
            self._corpus_cache.pop(next(iter(self._corpus_cache)))
        self._corpus_cache[km_path] = (mtime, insights, band)
        return insights, band

    def _maybe_reload(self) -> None:
        if not self._band_path.exists():
            return
        mtime = self._band_path.stat().st_mtime
        if mtime > self._band_mtime:
            self.reload_snapshot()

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()
        # Snapshot — serve_forever's finally sets self._listener = None, and
        # the self-connect below gives that race a ~0.5s window (CI Windows
        # hit None.close() when idle-exit overlapped an explicit stop()).
        listener = self._listener
        if listener is not None:
            # Wake a pending accept() BEFORE closing: on Linux, close()
            # from another thread does NOT interrupt a blocked accept()
            # (observed wedged on ubuntu CI 2026-06-10 — serve_forever's
            # finally never ran, ready-marker never cleared). The
            # self-connect handshake is the reliable wake on every
            # platform; close() after is just resource cleanup.
            self._wake_accept_via_self_connect()
            try:
                listener.close()
            except Exception:
                pass

    def serve_forever(self) -> None:
        # Clear any stale marker from a previous crashed daemon before
        # loading the encoder, so a hook that's polling can't observe
        # the marker while we're still cold.
        ready_path = daemon_ready_path(self.store.root)
        try:
            ready_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        self.reload_snapshot()
        self._listener = _conn.Listener(self.address, authkey=self.authkey)
        # Encoder + snapshot loaded, listener bound — safe to advertise.
        try:
            ready_path.write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            pass

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
            # Remove the ready marker on clean exit so the next hook
            # spawn doesn't see a stale "ready" before the new daemon
            # has finished loading.
            try:
                daemon_ready_path(self.store.root).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def _watchdog_loop(self) -> None:
        """Idle/stop poller; unwedges `accept()` so the main loop exits.

        Sets `_stop` *before* unwedging — otherwise the main loop's
        OSError handler would `continue` straight back into accept().

        The unwedge is a self-connect on EVERY platform:
          - Windows: closing the named-pipe handle is *not* guaranteed
            to wake a pending `WaitForMultipleObjects` (MSDN: undefined
            for cross-thread `CloseHandle` on a handle in pending
            overlapped I/O).
          - Linux: cross-thread `close()` on the listening socket does
            NOT interrupt a blocked `accept()` either — the thread stays
            wedged until the next inbound connection (observed on ubuntu
            CI 2026-06-10: serve_forever's finally never ran). An earlier
            revision assumed close-aborts-accept on POSIX; that was false.
        The self-connect completes the handshake (or fails the challenge);
        either way accept() returns, the loop re-checks `_stop`, exits.
        """
        while not self._stop.is_set():
            if self._idle_expired():
                self._stop.set()
                break
            self._stop.wait(self.reload_poll_seconds)
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
        raw = payload["request"] if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            conn.send(asdict(RecallReply(
                hits=[], encoder_revision=self.encoder_revision,
                error="invalid request envelope",
            )))
            return
        # Tolerate unknown request keys (drop them) and never let a malformed
        # request crash the serve loop — a NEWER client carrying wire fields
        # this build doesn't know about must degrade to a clean error reply,
        # not a daemon-wide blackout. Construction is INSIDE the guard so a
        # missing required field (`query`) also fails soft.
        known = {f.name for f in fields(RecallRequest)}
        try:
            req = RecallRequest(**{k: v for k, v in raw.items() if k in known})
        except (TypeError, ValueError) as exc:
            conn.send(asdict(RecallReply(
                hits=[], encoder_revision=self.encoder_revision,
                error=f"bad request: {type(exc).__name__}",
            )))
            return
        cosine_floor = req.cosine_floor
        top1_top2_gap = req.top1_top2_gap
        min_recurrence = req.min_recurrence
        if req.auto_tune_cold_start:
            from .recall import cold_start_gates
            relaxed = cold_start_gates(len(self._claims))
            if relaxed is not None:
                cosine_floor, top1_top2_gap, min_recurrence = relaxed
        try:
            # Encode the query ONCE and reuse it for both the experience rank
            # and the corpus-band rank — the hook has no warm encoder, so the
            # daemon is the single embed point.
            query_emb = _encode_query(req.query, self.encoder)
            hits, diagnostic = rank_with_diagnostic(
                req.query,
                claims=self._claims,
                band=self._band,
                encoder=self.encoder,
                cwd_hash=req.cwd_hash,
                top_k=req.top_k,
                cosine_floor=cosine_floor,
                top1_top2_gap=top1_top2_gap,
                min_recurrence=min_recurrence,
                intent_kind=req.intent_kind,  # type: ignore[arg-type]
                query_emb=query_emb,
            )
        except Exception as exc:
            conn.send(asdict(RecallReply(
                hits=[], encoder_revision=self.encoder_revision,
                error=f"{type(exc).__name__}: {exc}",
            )))
            return
        # Flattened claim rows — the shape `user_prompt._format_injection`
        # rebuilds via `_row_to_claim`. `asdict` would nest `facts`.
        serialised = [
            {"claim": _claim_to_row(h.claim), "cosine": h.cosine}
            for h in hits
        ]
        # Corpus band recall (S3 d3): rank the workspace's insight band with
        # the same query embedding. Best-effort + isolated — a corpus failure
        # must never break experience recall, so it never raises out of here.
        band_serialised: list[dict] = []
        attribute_serialised: list[dict] = []
        constraint_serialised: list[dict] = []
        if req.km_path:
            try:
                from ..store.verified import (
                    rank_insight_band,
                    serve_band_attributes,
                    serve_band_constraints,
                )

                loaded = self._corpus_band(req.km_path)
                if loaded is not None:
                    binsights, bband = loaded
                    band_serialised = [
                        {"claim_id": h.claim_id, "source": h.source,
                         "rank": h.rank, "cosine": h.cosine}
                        for h in rank_insight_band(
                            query_emb, binsights, bband,
                            top_k=_BAND_RECALL_TOP_K,
                            cosine_floor=_BAND_RECALL_COSINE_FLOOR,
                        )
                    ]
                    # Same band, content-bearing channel: serve the user-world
                    # attribute claims, newest value per subject. The dedup is
                    # here (not in `rank_insight_band`) because it reads `facts`.
                    attribute_serialised = [
                        {"content": a.content, "attribute_key": a.attribute_key,
                         "created_at": a.created_at, "score": a.score}
                        for a in serve_band_attributes(
                            query_emb, binsights, bband,
                            top_k=_BAND_RECALL_TOP_K,
                            cosine_floor=_BAND_RECALL_COSINE_FLOOR,
                        )
                    ]
                    # Serve-ALL channel: standing constraints + falsified
                    # findings, query-independent (R1's proven no-selection
                    # design — a hard rule applies whether or not the query
                    # is about it).
                    constraint_serialised = [
                        {"claim_id": c.claim_id, "content": c.content,
                         "kind": c.kind, "attribute_key": c.attribute_key,
                         "created_at": c.created_at}
                        for c in serve_band_constraints(binsights)
                    ]
            except Exception:
                band_serialised = []
                attribute_serialised = []
                constraint_serialised = []
        conn.send(asdict(RecallReply(
            hits=serialised, encoder_revision=self.encoder_revision,
            effective_min_recurrence=min_recurrence,
            diagnostic=asdict(diagnostic),
            band_hits=band_serialised,
            attribute_hits=attribute_serialised,
            constraint_hits=constraint_serialised,
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
        except (FileNotFoundError, ConnectionRefusedError, OSError,
                EOFError, _mp_AuthenticationError) as exc:
            # EOFError / AuthenticationError: the listener closed mid-
            # handshake — the self-connect wake racing shutdown lands here
            # (BrokenPipeError is an OSError; EOFError is NOT). Without
            # these, every clean shutdown printed a thread traceback.
            box["error"] = exc

    worker = threading.Thread(target=_attempt, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        return None
    return box.get("conn")


def _decode_reply(payload: dict) -> RecallReply | None:
    """Decode a reply dict into `RecallReply`, forward-compatibly.

    A NEWER daemon may reply with fields this client doesn't know —
    `RecallReply(**payload)` raised TypeError, which escaped request_recall's
    catch tuple and broke the hook's fail-open contract on every
    client/server version skew (2026-06 review). Unknown keys are ignored;
    a payload missing required keys returns None (fail-open), never raises.
    """
    known = {f.name for f in fields(RecallReply)}
    try:
        return RecallReply(**{k: v for k, v in payload.items() if k in known})
    except TypeError:
        return None


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
        return _decode_reply(payload)
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

    claims = root / CLAIMS_FILE
    band = root / BAND_FILE
    if not claims.exists() or not band.exists():
        report.add("store", False,
                   f"{CLAIMS_FILE} or {BAND_FILE} missing under {root}; "
                   f"recreate via `rlat memory add <text>`")
    else:
        report.add("store", True,
                   f"claims + band present ({claims.stat().st_size} + "
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
