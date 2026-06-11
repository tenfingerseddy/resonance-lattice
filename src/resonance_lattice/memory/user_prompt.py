"""UserPromptSubmit hook shim — `rlat memory hook` entry point.

Claude Code fires this on every user message. Pipeline (per §5.2.1):

1. Read JSON `{prompt, cwd, session_id, ...}` from stdin.
2. Try to connect to the per-user recall daemon socket.
3. On connect-fail: lazy-spawn `rlat memory recall --daemon` as a
   background subprocess, wait 100 ms, retry once.
4. Send `RecallRequest(query=prompt, cwd_hash=...)`; receive
   `RecallReply`.
5. If hits non-empty: emit `{hookSpecificOutput.additionalContext: <§0.4 block>}`
   to stdout so Claude Code prepends the `<rlat-memory>` block to the
   user's prompt context. Empty hits → emit `{}` (no injection, no
   stderr).
6. Fail-open per §16.5 / §18.5 — any uncaught exception, daemon
   timeout, or one-shot subprocess failure ends with `{}` to stdout
   and a single stderr line for operator visibility. The user's
   prompt is never blocked.

Token budget per §0.4: 1500 tokens (~6000 chars by 4-char/token
proxy) PER BLOCK (memory / environment / constraints — three blocks
since v3 S2), truncate at row boundary so we never emit a half-row.

Latency profile: warm-recall is well within the §0.6 p95 80ms budget
(daemon p99 ~1.5ms + IPC ~5ms + encoder embed ~12ms). Cold-spawn
first-of-session is ~800ms (200ms initial connect-fail + 100ms boot
wait + 500ms retry) — outside the warm-recall budget, but amortised
to N=1 per session and absorbed by the user's prompt-typing latency.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Heavy imports (.daemon pulls multiprocessing.connection + the encoder
# import chain, .store pulls portalocker + numpy) are deferred into
# `run_hook` after the fast-exit gates so bad-stdin / empty-prompt /
# missing-root paths don't pay the import cost.

# §0.4 token budget: 1500 tokens per injection block; conservative 4
# chars/token proxy. Truncate at row boundary so callers never see a
# half-row.
_MAX_INJECTION_CHARS = 6000

# §5.2.1 hook → daemon retry window after lazy-spawn. The legacy 0.1s
# boot-wait + 0.5s retry assumes the daemon is listening within ~600ms
# of spawn. Cold encoder loads can take >1s, so the v3 bench surfaced
# first-call-after-spawn misses. The fix is the ready-marker poll loop
# below: after spawn we poll `daemon_ready_path` until either the
# marker appears (encoder + snapshot loaded → safe to connect) or
# `_DAEMON_READY_POLL_BUDGET_S` elapses (give up, fail-open).
#
# Cold-spawn retry budget is matched to the first-attempt budget — a
# fresh hook subprocess on Windows pays ~440-900ms for the named-pipe
# handshake (see `daemon.DEFAULT_TIMEOUT_MS` comment), and the older
# 500ms retry budget (=> 300ms connect budget after the 60% fraction)
# was too tight. v4.1 bench reproduced this: attempt-2 connected to
# a freshly-bound daemon but timed out before the handshake completed.
_DAEMON_BOOT_WAIT_S = 0.1
_DAEMON_RETRY_TIMEOUT_MS = 2000
_DAEMON_READY_POLL_INTERVAL_S = 0.1
_DAEMON_READY_POLL_BUDGET_S = 5.0

# §0.6 default M (recurrence threshold) — must match recall.py default.
_RECURRENCE_M = 3


def _iso_now() -> str:
    """Same-shape ISO timestamp `state` modules use. Inline here so the
    fast-exit gates above this line don't pay the `_common` import cost."""
    from ._common import utcnow_iso
    return utcnow_iso()


def _trace(event: str) -> None:
    """Append a single line to `~/.rlat/memory/.hook_trace.log` for "did
    Claude Code invoke this command at all" diagnosis. Best-effort: any
    failure is swallowed so the trace itself can never break the hook.

    **Default off** per codex P2.3 (memory/privacy posture). Enable only
    when actively diagnosing a hook misfire by setting
    `RLAT_HOOK_TRACE=1`. Trace lines include transcript paths and
    session ids — fine for one-shot diagnosis on the operator's own
    box, but not desirable as a long-running default.
    """
    if os.environ.get("RLAT_HOOK_TRACE") != "1":
        return
    try:
        from datetime import datetime, timezone
        log_dir = Path.home() / ".rlat" / "memory"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / ".hook_trace.log").open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}  {event}\n")
    except Exception:
        pass


_ATOMIC_CAPTURE_ENV = "RLAT_ATOMIC_CAPTURE"
_MINE_ATTRIBUTES_ENV = "RLAT_MINE_ATTRIBUTES"

# Hot-path bound: the SessionEnd hook runs synchronously on prompt close.
# A hung Anthropic endpoint must not pin the user's terminal — 15 s is
# generous for one ~1K-token round-trip and tight enough to keep failures
# observable. Bounds the SDK's default ~600 s timeout.
_HOOK_LLM_TIMEOUT_S = 15.0


def _atomic_capture_enabled() -> bool:
    """`RLAT_ATOMIC_CAPTURE=1` opts the capture hook into the LLM
    event-extraction path. Off by default."""
    return os.environ.get(_ATOMIC_CAPTURE_ENV, "0").strip().lower() not in (
        "", "0", "false", "off", "no",
    )


def _capture_hook_client():
    """Resolve a timeout-bounded LLM client for the capture hook, or
    `None` if no key is set or SDK init fails. Failure here MUST fall
    through to the single-claim path — never propagate, never block."""
    try:
        from .._anthropic import default_client, discover_api_key
        key = discover_api_key()
        if not key:
            return None
        return default_client(key.strip(), timeout=_HOOK_LLM_TIMEOUT_S)
    except Exception:
        return None


def _mine_attributes_enabled() -> bool:
    """`RLAT_MINE_ATTRIBUTES=1` opts the capture hook into mining durable
    WORLD attributes from the session's user turns into the workspace's
    primary knowledge model (the E2c-validated 4-gate extractor;
    person-facts are dropped by its scope gate). Implies the LLM capture
    path (atomic EVENT extraction also runs - one client, both extractors) — mining needs the same hook client. Off by default; v3 S1."""
    return os.environ.get(_MINE_ATTRIBUTES_ENV, "0").strip().lower() not in (
        "", "0", "false", "off", "no",
    )


def _capture_llm_context(cwd):
    """`(client, km_path)` the SessionEnd hook hands `capture()`.

    - client: created when atomic capture OR mining is enabled and a key
      resolves (`_capture_hook_client`); `None` otherwise.
    - km_path: the workspace's primary `.rlat` (`resolve_primary_km`,
      the same convention the recall hook uses), only when mining is
      enabled AND a client exists — no client means no extractor, so the
      wake stays cleanly off. Fails open on any resolution error.
    """
    mine = _mine_attributes_enabled()
    client = (_capture_hook_client()
              if (_atomic_capture_enabled() or mine) else None)
    km_path = None
    if mine and client is not None:
        try:
            from ..state.workspace import resolve_primary_km
            km_path = resolve_primary_km(cwd)
        except Exception:
            km_path = None
    return client, km_path


def _neutralise_boundary_tags(text: str) -> str:
    """Strip / disarm any rlat injection-block boundary tag in row text so a
    malicious or accidentally-formatted row can't break out of a block
    delimiter or spoof a closing tag. Covers every `<rlat-…>` block the hook
    emits (memory + context), open and close.

    Replacement uses the unicode "less-than" / "greater-than" full-width
    forms — visually similar so the row stays readable, but the literal
    `<` and `>` characters are gone so the block boundary is unforgeable.
    """
    for tag in ("</rlat-memory>", "<rlat-memory>",
                "</rlat-context>", "<rlat-context>"):
        text = text.replace(tag, tag.replace("<", "＜").replace(">", "＞"))
    return text


def _format_injection(hits: list[dict], recurrence_m: int) -> tuple[str, int]:
    """Render the §0.4 `<rlat-memory>` block from RecallReply hits.

    Truncates at row boundary once the cumulative char count would
    exceed `_MAX_INJECTION_CHARS`. Returns `(block, n_rows)` so the
    caller doesn't have to re-derive the count from the string. Claim
    content is run through `_neutralise_boundary_tags` so a claim
    containing `</rlat-memory>` (e.g. a captured session that quoted
    the spec) can't break the delimiter and inject downstream prompt
    content.
    """
    from .claim_store import _row_to_claim

    body_lines: list[str] = []
    char_budget = _MAX_INJECTION_CHARS
    for hit in hits:
        claim = _row_to_claim(hit["claim"])
        text = _neutralise_boundary_tags(
            claim.content.replace("\n", " ").strip())
        line = f"- *{claim.facts.primary_polarity()}* — {text}"
        if char_budget - len(line) - 1 < 0:
            break
        body_lines.append(line)
        char_budget -= len(line) + 1
    if not body_lines:
        return "", 0
    block = (
        "<rlat-memory>\n"
        f"**Memory** ({len(body_lines)} lessons, recurrence ≥{recurrence_m}):\n\n"
        + "\n".join(body_lines)
        + "\n</rlat-memory>"
    )
    return block, len(body_lines)


def _format_attribute_injection(attribute_hits: list[dict]) -> tuple[str, int]:
    """Render the user-world `<rlat-context>` block from RecallReply.attribute_hits.

    The content-bearing counterpart to `_format_injection`: unlike experience
    lessons (which carry a polarity) and unlike the cache-only corpus
    `band_hits`, these attribute facts ARE injected — a user-world fact (the
    user's own SKU/role/version/corpus size) is something the agent needs in
    hand to answer correctly. The daemon has already deduped to the NEWEST
    value per subject (`serve_band_attributes`), so this only renders.

    Same delimiter-safety + row-boundary char budget as `_format_injection`;
    tolerant of a missing/blank `content` key (best-effort wire shape).
    """
    body_lines: list[str] = []
    char_budget = _MAX_INJECTION_CHARS
    for hit in attribute_hits:
        text = _neutralise_boundary_tags(
            str(hit.get("content", "")).replace("\n", " ").strip())
        if not text:
            continue
        line = f"- {text}"
        if char_budget - len(line) - 1 < 0:
            break
        body_lines.append(line)
        char_budget -= len(line) + 1
    if not body_lines:
        return "", 0
    block = (
        "<rlat-context>\n"
        f"**Your environment** ({len(body_lines)} fact(s)):\n\n"
        + "\n".join(body_lines)
        + "\n</rlat-context>"
    )
    return block, len(body_lines)


def _format_constraint_injection(constraint_hits: list[dict]) -> tuple[str, int]:
    """Render standing constraints + falsified findings from
    RecallReply.constraint_hits as their own `<rlat-context>` block.

    The daemon serves these ALL-always (`serve_band_constraints` — R1's
    proven no-selection design), and the section headings come from
    `store.serve_framing` (the framing is the measured active ingredient,
    R2). Same delimiter-safety + row-boundary char budget as the sibling
    formatters; the budget is a transport safety valve, not a selection —
    a band would need hundreds of constraints to reach it.
    """
    from ..store.serve_framing import frame_claim_lines

    rows: list[tuple[str, str]] = []
    char_budget = _MAX_INJECTION_CHARS
    for hit in constraint_hits:
        kind = str(hit.get("kind", "constraint"))
        if kind not in ("constraint", "negation"):
            continue  # newer-daemon skew — don't count rows framing drops
        text = _neutralise_boundary_tags(
            str(hit.get("content", "")).replace("\n", " ").strip())
        if not text:
            continue
        if char_budget - len(text) - 3 < 0:
            break
        rows.append((kind, text))
        char_budget -= len(text) + 3
    body = frame_claim_lines(rows)
    if not body:
        return "", 0
    return f"<rlat-context>\n{body}\n</rlat-context>", len(rows)


def _resolve_active_intent_id(state_root: Path) -> str | None:
    """Look up the most-recently-created active/blocked/proposed intent.

    Returns None on any failure or empty store. Fail-open: a diagnostic
    write must never break the hook. Same selection rule as the
    success-path recall_cache stamp, hoisted so every diagnostic entry
    can include the intent_id regardless of whether the recall hit.
    """
    try:
        from ..state import LiveIntentStore
        store = LiveIntentStore(state_root)
        live = [
            i for i in store.list_all()
            if i.status in ("active", "blocked", "proposed")
        ]
        if not live:
            return None
        return max(live, key=lambda i: i.created_at).intent_id
    except Exception:
        return None


_UNSET: object = object()


def _log_diagnostic(
    *,
    cwd: str,
    prompt: str,
    intent_kind: str,
    status: str,
    n_hits: int,
    diagnostic: dict | None,
    intent_id: object = _UNSET,
) -> None:
    """Write one `recall_diagnostic.jsonl` entry. Best-effort — any
    exception is swallowed so the diagnostic surface can never break
    the hook (same fail-open contract as the rest of the path).

    `intent_id` accepts a sentinel default so the success path can pass
    the already-resolved value and avoid a duplicate LiveIntentStore
    read; miss / unreachable paths pass nothing and we resolve here.
    """
    try:
        from ..state import (
            RecallDiagnosticEntry,
            RecallDiagnosticLog,
            hash_prompt,
            make_turn_id,
            resolve_state_root,
        )
        state_root = resolve_state_root(cwd)
        if intent_id is _UNSET:
            resolved_intent_id = _resolve_active_intent_id(state_root)
        else:
            resolved_intent_id = intent_id  # type: ignore[assignment]
        RecallDiagnosticLog(state_root).append(RecallDiagnosticEntry(
            turn_id=make_turn_id(prompt),
            timestamp=_iso_now(),
            prompt_hash=hash_prompt(prompt),
            intent_kind=intent_kind,
            intent_id=resolved_intent_id,
            status=status,
            n_hits=n_hits,
            diagnostic=diagnostic,
        ))
    except Exception:
        pass


def _spawn_daemon(memory_root: Path) -> None:
    """Lazy-spawn the recall daemon as a detached background subprocess.

    Uses `sys.executable -m` so the hook works even when `rlat` isn't
    on PATH (common with Windows hook configs).
    """
    import subprocess

    cmd = [
        sys.executable, "-m", "resonance_lattice.cli.app",
        "memory",
        "--memory-root", str(memory_root.parent),
        "--user", memory_root.name,
        "recall", "--daemon",
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        )
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    except (OSError, FileNotFoundError):
        return


def _debug_daemon_log(event: str) -> None:
    """RLAT_DEBUG_DAEMON=1 gates a verbose connect-attempt trace.

    Investigation context (V1 in `_probe/longitudinal_v4/V1_INVESTIGATION.md`):
    `claude -p` subprocesses fire the hook but hit `daemon_unreachable` 80%
    of the time. Three candidate causes (per-user resolution drift, named
    pipe ACL, spawn-retry shape) need address + exception attribution to
    distinguish. This trace lets a probe capture pid/ppid + env-derived
    user_id + pipe address + exception type per attempt, all in one file.

    Always off when the env var isn't `1`. Will be retired once V1 is fixed.
    """
    if os.environ.get("RLAT_DEBUG_DAEMON") != "1":
        return
    try:
        from datetime import datetime, timezone
        log_dir = Path.home() / ".rlat" / "memory"
        log_dir.mkdir(parents=True, exist_ok=True)
        ppid = os.getppid() if hasattr(os, "getppid") else -1
        line = (
            f"{datetime.now(timezone.utc).isoformat()}  "
            f"pid={os.getpid()} ppid={ppid}  "
            f"USER={os.environ.get('USER')!r} "
            f"USERNAME={os.environ.get('USERNAME')!r} "
            f"RLAT_MEMORY_USER={os.environ.get('RLAT_MEMORY_USER')!r}  "
            f"{event}\n"
        )
        with (log_dir / ".daemon_debug.log").open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _recall_via_daemon_or_spawn(request, memory_root: Path):
    """Try the daemon; on connect-fail, lazy-spawn and retry once.

    Returns None on any failure — the caller treats None as "no
    injection" per the §16.5 / §18.5 fail-open contract.
    """
    import time

    from .daemon import (
        DEFAULT_TIMEOUT_MS,
        daemon_ready_path,
        daemon_socket_address,
        load_or_create_authkey,
        request_recall,
    )

    address = daemon_socket_address(memory_root)
    authkey = load_or_create_authkey(memory_root)
    _debug_daemon_log(
        f"entry memory_root={memory_root!s} address={address!r} "
        f"authkey_len={len(authkey)}"
    )
    reply = request_recall(
        request, address=address, authkey=authkey,
        timeout_ms=DEFAULT_TIMEOUT_MS,
    )
    _debug_daemon_log(
        f"attempt-1 reply_is_none={reply is None} "
        f"reply_error={getattr(reply, 'error', None)!r}"
    )
    if reply is not None:
        return reply

    # Clear any stale marker BEFORE spawn. A previous daemon killed
    # without running its finally block (SIGKILL, OOM, system reboot)
    # leaves a `.recall.ready` behind. The polling loop below would
    # then observe that stale marker immediately, conclude the daemon
    # is ready, and try connecting — failing because no daemon is
    # actually listening. Clearing here means any marker we observe
    # after spawn was written by the freshly-spawned daemon. Best-
    # effort: a concurrent spawn racing this hook would re-create
    # the marker, and the connect attempt below tolerates either way.
    ready_path = daemon_ready_path(memory_root)
    try:
        ready_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass

    _spawn_daemon(memory_root)
    _debug_daemon_log(f"spawned daemon for {memory_root!s}")
    deadline = time.monotonic() + _DAEMON_READY_POLL_BUDGET_S
    saw_marker = False
    while time.monotonic() < deadline:
        if ready_path.exists():
            saw_marker = True
            break
        time.sleep(_DAEMON_READY_POLL_INTERVAL_S)
    _debug_daemon_log(
        f"ready_marker saw={saw_marker} ready_path={ready_path!s}"
    )
    if not saw_marker:
        time.sleep(_DAEMON_BOOT_WAIT_S)
    reply2 = request_recall(
        request, address=address, authkey=authkey,
        timeout_ms=_DAEMON_RETRY_TIMEOUT_MS,
    )
    _debug_daemon_log(
        f"attempt-2 reply_is_none={reply2 is None} "
        f"reply_error={getattr(reply2, 'error', None)!r}"
    )
    return reply2


def run_hook(
    *,
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
    user_id: str | None = None,
    memory_root_base: Path | None = None,
) -> int:
    """Read the UserPromptSubmit envelope from stdin, recall, and emit
    the hook output JSON to stdout. Returns the process exit code.

    Always exits 0 — fail-open per §16.5 / §18.5. Errors surface as a
    single stderr line so operators can `tail -f ~/.rlat/memory/...`
    to debug, but they never block the prompt.
    """
    _trace("UserPromptSubmit:fired")
    _debug_daemon_log("run_hook:fired")
    try:
        payload = json.loads(stdin.read())
    except (json.JSONDecodeError, OSError) as exc:
        _debug_daemon_log(f"BAIL: bad stdin {type(exc).__name__}: {exc}")
        json.dump({}, stdout)
        return 0

    prompt = payload.get("prompt", "")
    cwd = payload.get("cwd") or os.getcwd()
    if not isinstance(prompt, str) or not prompt.strip():
        _debug_daemon_log(f"BAIL: empty/invalid prompt prompt_type={type(prompt).__name__} stripped_len={len(prompt.strip()) if isinstance(prompt, str) else -1}")
        json.dump({}, stdout)
        return 0

    # Paired-bench arm B: RLAT_DISABLE_HOOK=1 fast-exits with `{}` and
    # records a `status=disabled` diagnostic so the off-arm can still
    # account for "how many recalls would have fired here." Narrowly
    # scoped — this flag suppresses recall INJECTION only; SessionEnd
    # capture still runs if invoked. Callers wanting an end-to-end
    # memory-off arm (e.g. Option 1 clean comparison) must also skip
    # the capture invocation at their layer. Placed AFTER the stdin/
    # prompt gates so disabled-arm diagnostics carry a real prompt_hash
    # (lets per-session paired join key off `(arm, prompt_hash)`).
    if os.environ.get("RLAT_DISABLE_HOOK") == "1":
        # Pass intent_id=None explicitly so the disabled-arm path skips
        # `_resolve_active_intent_id`'s LiveIntentStore scan — the
        # paired-bench off arm doesn't need intent attribution and the
        # disabled-arm fast-exit should stay cheap.
        _log_diagnostic(
            cwd=cwd, prompt=prompt, intent_kind="none",
            status="disabled", n_hits=0, diagnostic=None,
            intent_id=None,
        )
        json.dump({}, stdout)
        return 0

    # Heavy imports gated behind the fast-exit checks above so bad-stdin
    # / empty-prompt fail-open paths skip the encoder + multiprocessing
    # + portalocker import chain.
    from ._common import workspace_hash
    from .daemon import RecallRequest
    from .intent_classify import classify_intent_kind
    from .store import path_for_user

    try:
        memory_root = path_for_user(user_id=user_id, root=memory_root_base)
    except RuntimeError as exc:
        _debug_daemon_log(f"BAIL: path_for_user RuntimeError: {exc}")
        json.dump({}, stdout)
        return 0
    _debug_daemon_log(f"memory_root resolved: {memory_root!s} exists={memory_root.exists()}")

    # S3 d3 — resolve the workspace's primary corpus so the daemon can ALSO
    # rank its insight band (the corpus-trust loop). Fail-open: a resolution
    # failure just means no corpus band-recall this turn, never a hook break.
    km_path: str | None = None
    try:
        from ..state import resolve_primary_km
        km = resolve_primary_km(cwd)
        km_path = str(km) if km is not None else None
    except Exception:
        km_path = None

    # First hook fire on a fresh install: don't spawn the daemon for an
    # empty store. Skip silently — but record the no-store status in
    # the diagnostic log so a fresh-workspace blackout is attributable
    # post-hoc. EXCEPTION (S3 d3, full decouple): when a corpus is resolvable
    # the daemon must still run — it ranks zero experience claims plus the
    # corpus insight band, so the corpus-trust loop fires even on a workspace
    # with no experience memory. The spawn path creates the (empty) store.
    if not memory_root.exists() and km_path is None:
        _log_diagnostic(
            cwd=cwd, prompt=prompt, intent_kind="none",
            status="no_store", n_hits=0, diagnostic=None,
        )
        json.dump({}, stdout)
        return 0

    # Cheap-path intent classifier — sub-millisecond regex scan; conditions
    # the daemon's manifesto re-rank without touching the 200ms hot-path
    # budget. "none" preserves the v2.1 cosine-only ordering.
    intent_kind = classify_intent_kind(prompt)

    request = RecallRequest(
        query=prompt,
        cwd_hash=workspace_hash(str(cwd)),
        intent_kind=intent_kind,
        # Cold-start auto-relax: when the per-user store has fewer than
        # `recall.COLD_START_ROW_THRESHOLD` rows, the daemon overrides
        # `cosine_floor` and `min_recurrence` to relaxed values so a
        # fresh workspace surfaces something rather than nothing. The
        # longitudinal benchmark caught the v1 defaults as a 20-session
        # blackout on diverse-task workloads. Reply carries
        # `effective_min_recurrence` so the injection-time gate stays
        # in sync with the daemon's actual filter.
        auto_tune_cold_start=True,
        # S3 d3 — when set, the daemon ALSO ranks this corpus's insight band
        # and returns the top corpus claims in `reply.band_hits` (cache-only,
        # never injected). None → experience-only recall (the pre-d3 path).
        km_path=km_path,
    )
    try:
        reply = _recall_via_daemon_or_spawn(request, memory_root)
    except Exception as exc:
        print(f"[rlat] hook recall failed: {type(exc).__name__}", file=stderr)
        _log_diagnostic(
            cwd=cwd, prompt=prompt, intent_kind=intent_kind,
            status="daemon_error", n_hits=0, diagnostic=None,
        )
        json.dump({}, stdout)
        return 0

    if reply is None:
        _log_diagnostic(
            cwd=cwd, prompt=prompt, intent_kind=intent_kind,
            status="daemon_unreachable", n_hits=0, diagnostic=None,
        )
        json.dump({}, stdout)
        return 0
    if reply.error:
        _log_diagnostic(
            cwd=cwd, prompt=prompt, intent_kind=intent_kind,
            status="daemon_error", n_hits=0, diagnostic=reply.diagnostic,
        )
        json.dump({}, stdout)
        return 0
    # Decouple the corpus-trust loop from experience state (S3 d3): the daemon
    # ALSO ranks the workspace's corpus insight band and returns those hits in
    # `reply.band_hits`. We must cache them — so a resolved intent's
    # attribution can carry corpus claim ids — even when no experience claim
    # surfaced. Bail only when NOTHING surfaced — experience hits, corpus band
    # hits, or content-bearing user-world attribute hits.
    attribute_hits = getattr(reply, "attribute_hits", []) or []
    constraint_hits = getattr(reply, "constraint_hits", []) or []
    if (not reply.hits and not reply.band_hits and not attribute_hits
            and not constraint_hits):
        _log_diagnostic(
            cwd=cwd, prompt=prompt, intent_kind=intent_kind,
            status="no_hit", n_hits=0, diagnostic=reply.diagnostic,
        )
        json.dump({}, stdout)
        return 0

    # Experience injection block — experience hits ONLY. Corpus claims are
    # never injected at prompt time (read-back is H2) and `_format_injection`
    # is `ExperienceFacts`-only (`primary_polarity()`), so corpus rows would
    # crash it. Empty string when there are no experience hits.
    # Use the daemon's effective_min_recurrence (defaults to _RECURRENCE_M for
    # back-compat with replies that pre-date the field) so the injection gate
    # matches the filter the daemon actually applied.
    injection_recurrence = getattr(
        reply, "effective_min_recurrence", _RECURRENCE_M,
    )
    block, n_rows = (
        _format_injection(reply.hits, injection_recurrence)
        if reply.hits else ("", 0)
    )
    # User-world attribute context — content-bearing, newest-per-subject (the
    # daemon already deduped). Injected alongside the experience block as its
    # own `<rlat-context>` delimiter.
    context_block, n_attrs = _format_attribute_injection(attribute_hits)
    # Standing constraints + falsified findings — served ALL-always by the
    # daemon, rendered with the proven kind framings.
    constraint_block, n_constraints = _format_constraint_injection(constraint_hits)

    # Persist this recall to the per-workspace cache so PostToolUse and
    # intent resolution can attribute outcomes back to the rows that
    # surfaced — experience hits AND corpus band hits. Best-effort — any
    # failure here must not break the hook.
    active_intent_id: object = _UNSET
    try:
        from ..state import (
            RecallCache,
            RecallEntry,
            RecallHitMetadata,
            hash_prompt,
            make_turn_id,
            resolve_state_root,
        )
        state_root = resolve_state_root(cwd)
        # Outcome-attributed retrieval (Horizon 4): stamp the recall with
        # the live intent_id when one is active so accept/reject can
        # attribute outcomes to recalls *deterministically* — no timestamp
        # window heuristic.
        active_intent_id = _resolve_active_intent_id(state_root)
        # Experience hits keep their enumerate() rank.
        row_metadata = [
            RecallHitMetadata(
                claim_id=hit["claim"]["claim_id"],
                rank=idx,
                cosine=float(hit.get("cosine", 0.0)),
                source=hit["claim"].get("source", "experience"),
            )
            for idx, hit in enumerate(reply.hits)
        ]
        # Corpus band hits — cache-only, each keeping its OWN 0-based rank
        # from the daemon's corpus ranking (NOT continued from the experience
        # hits; attribution tiers per source). Built all-or-nothing in its own
        # try so a malformed band row can never drop the experience stamp.
        try:
            band_rows = [
                RecallHitMetadata(
                    claim_id=bh["claim_id"],
                    rank=int(bh["rank"]),
                    cosine=float(bh.get("cosine", 0.0)),
                    source=bh.get("source", "corpus"),
                )
                for bh in reply.band_hits
            ]
        except Exception:
            band_rows = []
        row_metadata.extend(band_rows)
        cache = RecallCache(state_root)
        cache.append(RecallEntry(
            turn_id=make_turn_id(prompt),
            timestamp=_iso_now(),
            prompt_hash=hash_prompt(prompt),
            intent_kind=intent_kind,
            intent_id=active_intent_id,  # type: ignore[arg-type]
            row_metadata=row_metadata,
        ))
    except Exception:
        pass

    # Diagnostic status tracks EXPERIENCE recall (its consumers count
    # experience hits): "ok" when experience hits surfaced, else "no_hit" —
    # corpus-only stamping doesn't make experience recall a hit.
    _log_diagnostic(
        cwd=cwd, prompt=prompt, intent_kind=intent_kind,
        status="ok" if reply.hits else "no_hit",
        n_hits=len(reply.hits),
        diagnostic=reply.diagnostic, intent_id=active_intent_id,
    )

    # Inject whatever surfaced: the experience block, the user-world context
    # block, the constraint block — each its own delimiter, joined by blank
    # lines.
    injected = "\n\n".join(
        b for b in (block, context_block, constraint_block) if b)
    if injected:
        summary = []
        if n_rows:
            summary.append(f"{n_rows} claim(s)")
        if n_attrs:
            summary.append(f"{n_attrs} env fact(s)")
        if n_constraints:
            summary.append(f"{n_constraints} constraint(s)")
        print(f"[rlat] Recalled {', '.join(summary)}", file=stderr)
        json.dump({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": injected,
            }
        }, stdout)
    else:
        json.dump({}, stdout)
    return 0


def _parse_claude_code_transcript(transcript_path: Path, session_id: str, cwd: str):
    """Best-effort parser for Claude Code's JSONL transcript shape.

    Each line is a typed envelope; we keep only `type=user` / `type=assistant`
    entries. User messages carry `message.content[].text`; assistant messages
    carry `message.content[].text` + `message.content[].input.path` /
    `.content` for `tool_use` blocks. Anything we don't recognise is dropped
    silently — fail-open is the contract per §16.5.
    """
    from .capture import Message, ToolCall, Transcript

    messages: list[Message] = []
    if not transcript_path.exists():
        return Transcript(session_id=session_id, messages=tuple(), cwd=cwd)
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content_blocks = msg.get("content")
        if isinstance(content_blocks, str):
            # Claude Code emits plain-string content for normal user turns
            # (the block-list shape is only required when tool_use / thinking
            # blocks are interleaved). Treat string content as a single text
            # block so user-only sessions aren't dropped as 0-char.
            content_blocks = [{"type": "text", "text": content_blocks}]
        elif not isinstance(content_blocks, list):
            content_blocks = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
            elif btype == "tool_use":
                name = block.get("name", "")
                bin_ = block.get("input") or {}
                path = bin_.get("path") if isinstance(bin_, dict) else None
                content = (
                    bin_.get("content") if isinstance(bin_, dict) else ""
                ) or bin_.get("command", "") if isinstance(bin_, dict) else ""
                tool_calls.append(ToolCall(
                    name=str(name),
                    path=str(path) if isinstance(path, str) else None,
                    content=str(content) if content else "",
                ))
        if text_parts or tool_calls:
            messages.append(Message(
                role=t,  # type: ignore[arg-type]
                content="\n".join(text_parts),
                tool_calls=tuple(tool_calls),
            ))
    return Transcript(session_id=session_id, messages=tuple(messages), cwd=cwd)


def run_capture_hook(
    *,
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
    user_id: str | None = None,
    memory_root_base: Path | None = None,
) -> int:
    """SessionEnd-hook entry point. Reads Claude Code's SessionEnd envelope
    from stdin (`{session_id, transcript_path, cwd, ...}`), parses the JSONL
    transcript at `transcript_path`, runs the §5.2 capture pipeline, and
    emits `{}` to stdout.

    Always exits 0 — fail-open per §16.5 / §18.5. The SessionEnd hook fires
    when the session terminates and a memory failure must never block the
    user's session close. (The plan §5.2 calls this the "Stop hook" — but
    Claude Code's `Stop` event is per-turn, not per-session; SessionEnd
    matches the spec's once-per-session intent.)
    """
    _trace("SessionEnd:fired")
    try:
        payload = json.loads(stdin.read())
    except (json.JSONDecodeError, OSError):
        _trace("SessionEnd:bad-stdin")
        json.dump({}, stdout)
        return 0

    transcript_path_raw = payload.get("transcript_path")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd") or os.getcwd()
    _trace(f"SessionEnd:transcript_path={transcript_path_raw!r} session={session_id!r}")
    if not transcript_path_raw:
        _trace("SessionEnd:no-transcript-path")
        json.dump({}, stdout)
        return 0

    from .capture import GateConfig, capture
    from .claim_store import ExperienceClaimStore
    from .redaction import Redactor
    from .store import path_for_user

    try:
        memory_root = path_for_user(user_id=user_id, root=memory_root_base)
    except RuntimeError:
        json.dump({}, stdout)
        return 0
    memory_root.mkdir(parents=True, exist_ok=True)

    try:
        tp = Path(transcript_path_raw)
        if not tp.exists() and tp.parent.exists():
            # Claude Code re-mints session_id on /compact resume but keeps
            # appending to the original transcript file. The payload path
            # then references a UUID that was never written to disk. Fall
            # back to the most-recently-modified `.jsonl` in the project
            # transcripts dir (the live session's actual file).
            siblings = sorted(
                tp.parent.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if siblings:
                _trace(f"SessionEnd:fallback transcript_path={siblings[0].name}")
                tp = siblings[0]
        size = tp.stat().st_size if tp.exists() else -1
        _trace(f"SessionEnd:transcript exists={tp.exists()} size={size}")
        transcript = _parse_claude_code_transcript(tp, session_id, cwd)
        _trace(f"SessionEnd:parsed messages={len(transcript.messages)}")
    except Exception as exc:
        _trace(f"SessionEnd:parse-failed {type(exc).__name__}: {exc}")
        print(f"[rlat] capture parse failed: {type(exc).__name__}", file=stderr)
        json.dump({}, stdout)
        return 0

    try:
        store = ExperienceClaimStore(root=memory_root)
        redactor = Redactor.for_memory_root(memory_root)
        # Atomic event extraction + world-attribute mining are opt-in
        # (default off). Missing key or SDK init failure resolves to
        # `None`, which `capture()` takes as the single-claim path.
        client, km_path = _capture_llm_context(cwd)
        result = capture(transcript, store=store, redactor=redactor,
                          gate=GateConfig(), client=client, km_path=km_path)
    except Exception as exc:
        print(f"[rlat] capture failed: {type(exc).__name__}", file=stderr)
        json.dump({}, stdout)
        return 0

    if result.claim_ids:
        n = len(result.claim_ids)
        _trace(f"SessionEnd:captured count={n} redactions={result.redactions}")
        label = "rows" if n != 1 else "row"
        print(f"[rlat] Captured {n} {label} ({result.redactions} "
              f"redactions)", file=stderr)
    elif result.skip_reason:
        _trace(f"SessionEnd:skipped reason={result.skip_reason}")
        print(f"[rlat] Capture skipped: {result.skip_reason}", file=stderr)
    if result.attribute_claim_ids and km_path is not None:
        n_attr = len(result.attribute_claim_ids)
        _trace(f"SessionEnd:mined attributes={n_attr}")
        print(f"[rlat] Learned {n_attr} world fact(s) into {km_path.name} "
              f"(review: rlat lens / rlat profile)", file=stderr)
    json.dump({}, stdout)
    return 0


if __name__ == "__main__":
    sys.exit(run_hook())
