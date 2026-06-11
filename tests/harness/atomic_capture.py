"""atomic_capture — integration of `extract_events()` into `capture()`.

With `client` set the LLM decomposes the scrubbed session text into
atomic facts; one claim is written per fact. With `client` unset or on
extractor failure the single-claim path resumes, so a flaky LLM never
degrades the `client=None` behaviour.

Seven contracts:

  (a) Atomic success: extractor returns N facts → N claims written,
      one per fact; `result.claim_ids` lists their ids in extraction
      order; each claim's content is the fact verbatim.
  (b) Extractor-None fallback: a client that raises (and so
      `extract_events` swallows to None) drops to the single-claim path —
      one claim with the full scrubbed text, identical observable
      behaviour to `client=None`.
  (c) Empty-facts no-op: extractor returns `[]` → no claim written,
      `skip_reason="no extractable events"`, redactions still logged
      against `transcript_hash` for the session-level audit trail.
  (d) Within-capture duplicate facts collapse via `dict.fromkeys` —
      the same fact emitted twice writes one claim.
  (e) Cross-session dedup: a fact textually identical to an existing
      `(content, workspace)` claim bumps that claim's recurrence_count
      instead of writing a sibling.
  (f) Audit-log correlation on the atomic path uses `transcript_hash`
      (session grain), not any single `claim_id`, because the same
      scrubbed text drove all N writes.
  (g) `client=None` runs the single-claim path — no extractor is called,
      one claim is written, and the audit log correlates against
      `claim_id`.
  (h) `_capture_hook_client()` returns `None` and never raises when SDK
      init fails (no API key, anthropic missing, network error). The hook
      thus falls through to `capture(client=None)` — single-claim path —
      and the user's prompt close is never blocked by a flaky LLM seam.

Hermetic — fake LLM client, mocked encoder via `ZeroEncoder`, no
network, no real model load. Pairs with `extract_events` (which pins
the extractor's own contracts independently).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ._testutil import (
    ZeroEncoder, make_stub_llm_facts, make_stub_llm_raising,
    patch_zero_encoder,
)


def _transcript(session_id: str, *, text: str, cwd: str = "/proj"):
    from resonance_lattice.memory.capture import Message, ToolCall, Transcript
    return Transcript(
        session_id=session_id,
        messages=[
            Message("user", "diagnose the failing build please look at recent commits"),
            Message("assistant", text,
                    tool_calls=(ToolCall("bash", "/tmp", "ls"),)),
        ],
        cwd=cwd,
    )


def _check_atomic_success() -> int:
    from resonance_lattice.memory.capture import capture
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    facts = [
        "The project's Bronze layer uses Spark Structured Streaming for ingest.",
        "events_fact_hour handles the current 500 GB/day; events_fact_day did not.",
        "User prefers compact tables over verbose prose for tabular data.",
    ]
    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=ZeroEncoder())
        result = capture(
            _transcript("sess-A", text="x" * 400),
            store=memory, redactor=Redactor(), client=make_stub_llm_facts(facts),
        )
        if len(result.claim_ids) != len(facts):
            print(f"[atomic_capture] FAIL (a): expected {len(facts)} claim_ids, "
                  f"got {len(result.claim_ids)}: {result.claim_ids!r}",
                  file=sys.stderr)
            return 1
        by_id = {c.claim_id: c for c in memory.read_all()}
        contents = [by_id[cid].content for cid in result.claim_ids]
        if contents != facts:
            print(f"[atomic_capture] FAIL (a): content order/contents mismatch.\n"
                  f"  expected: {facts!r}\n  got: {contents!r}", file=sys.stderr)
            return 1
    print("[atomic_capture] (a) atomic success → N claims, content matches OK",
          file=sys.stderr)
    return 0


def _check_polarity_lands_per_fact() -> int:
    """(i) an extractor-classified polarity reaches the stored claim — the
    end-to-end wire the 2026-06 review added (capture previously hardcoded
    every claim "factual", leaving the recall valence rerank inert)."""
    import json

    from ._testutil import make_stub_llm_client

    from resonance_lattice.memory.capture import capture
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    reply = json.dumps({"facts": [
        {"text": "Bronze layer ingests Event Hubs JSON.", "polarity": "factual"},
        {"text": "Do not use wildcard imports in main.py.", "polarity": "avoid"},
        {"text": "User prefers compact tables.", "polarity": "prefer"},
    ]})
    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=ZeroEncoder())
        result = capture(
            _transcript("sess-pol", text="y" * 400),
            store=memory, redactor=Redactor(),
            client=make_stub_llm_client(reply),
        )
        by_id = {c.claim_id: c for c in memory.read_all()}
        got = [by_id[cid].facts.primary_polarity() for cid in result.claim_ids]
    if got != ["factual", "avoid", "prefer"]:
        print(f"[atomic_capture] FAIL (i): polarities {got!r} "
              f"(want ['factual', 'avoid', 'prefer'])", file=sys.stderr)
        return 1
    print("[atomic_capture] (i) extractor polarity lands per fact OK",
          file=sys.stderr)
    return 0


def _check_attribute_mining_lands_in_km() -> int:
    """(j) the passive attribute miner (capture-frontier 2nd source): with a
    client set and an EXPLICIT km_path, gated WORLD attributes land in that
    KM's insight band (the artifact learns its own world — person-facts are
    dropped upstream by the extractor's prompt-level scope gate, validated
    in benchmarks/attribute_gate_e2c) and their claim_ids
    surface on CaptureResult.attribute_claim_ids. A re-stated attribute
    dedups against the band (same claim id back, no duplicate row) and no
    attribute-kind claim ever lands in the per-user event store. Without
    km_path the miner is DORMANT and event capture is unaffected."""
    import json

    from ._testutil import build_corpus, make_stub_llm_client

    from resonance_lattice.memory.capture import capture
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.store import archive

    # One reply serves both extractors: extract_events reads "facts",
    # extract_attributes reads "attributes".
    reply = json.dumps({
        "facts": ["Bronze layer ingests Event Hubs JSON."],
        "attributes": ["The tenant is EU-only for data residency."],
    })
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        km = build_corpus(root / "ws", {"a.md": "# A\n\nWorkspace doc body."})
        memory = ExperienceClaimStore(root=root / "u", encoder=ZeroEncoder())
        result = capture(
            _transcript("sess-attr", text="z" * 400, cwd=str(root / "ws")),
            store=memory, redactor=Redactor(),
            client=make_stub_llm_client(reply),
            km_path=km,
        )
        if len(result.attribute_claim_ids) != 1:
            print(f"[atomic_capture] FAIL (j): attribute_claim_ids="
                  f"{result.attribute_claim_ids!r}", file=sys.stderr)
            return 1
        insights = archive.read(km).insights
        landed = [c for c in insights if c.claim_id in result.attribute_claim_ids]
        if len(landed) != 1 or "EU-only" not in landed[0].content:
            print(f"[atomic_capture] FAIL (j): insight band landing wrong "
                  f"({[c.content for c in insights]!r})", file=sys.stderr)
            return 1
        if len(result.claim_ids) != 1:
            print(f"[atomic_capture] FAIL (j): event capture disturbed "
                  f"({result.claim_ids!r})", file=sys.stderr)
            return 1
        if any(c.kind == "attribute" for c in memory.read_all()):
            print("[atomic_capture] FAIL (j): attribute leaked into the "
                  "per-user event store", file=sys.stderr)
            return 1

        # Re-statement in a later session dedups against the band: same
        # claim id back, band row count unchanged.
        again = capture(
            _transcript("sess-attr-2", text="y" * 400, cwd=str(root / "ws")),
            store=memory, redactor=Redactor(),
            client=make_stub_llm_client(reply),
            km_path=km,
        )
        if again.attribute_claim_ids != result.attribute_claim_ids:
            print(f"[atomic_capture] FAIL (j): re-statement minted a new "
                  f"claim ({again.attribute_claim_ids!r})", file=sys.stderr)
            return 1
        if len(archive.read(km).insights) != len(insights):
            print("[atomic_capture] FAIL (j): re-statement duplicated a "
                  "band row", file=sys.stderr)
            return 1

        # Negative: no explicit km_path → miner dormant, capture unaffected
        # (even with a .rlat sitting in cwd — auto-discovery is OFF).
        no_km = capture(
            _transcript("sess-attr-3", text="w" * 400, cwd=str(root / "ws")),
            store=memory, redactor=Redactor(),
            client=make_stub_llm_client(reply),
        )
        if no_km.attribute_claim_ids != () or len(no_km.claim_ids) != 1:
            print(f"[atomic_capture] FAIL (j): dormant path wrong "
                  f"(attrs={no_km.attribute_claim_ids!r} "
                  f"claims={no_km.claim_ids!r})", file=sys.stderr)
            return 1
    print("[atomic_capture] (j) world attributes land in the band, dedup "
          "on re-statement, nothing personal stored OK", file=sys.stderr)
    return 0


def _check_extractor_failure_fallback() -> int:
    from resonance_lattice.memory.capture import capture
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    body = "x" * 400 + "TAIL-MARKER"
    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=ZeroEncoder())
        result = capture(
            _transcript("sess-A", text=body),
            store=memory, redactor=Redactor(),
            client=make_stub_llm_raising(RuntimeError("simulated outage")),
        )
        if len(result.claim_ids) != 1:
            print(f"[atomic_capture] FAIL (b): expected baseline single-claim "
                  f"fallback, got {result.claim_ids!r}", file=sys.stderr)
            return 1
        claims = memory.read_all()
        if len(claims) != 1 or "TAIL-MARKER" not in claims[0].content:
            print(f"[atomic_capture] FAIL (b): fallback content lost full text "
                  f"(claims={len(claims)}, body in content="
                  f"{'TAIL-MARKER' in (claims[0].content if claims else '')})",
                  file=sys.stderr)
            return 1
    print("[atomic_capture] (b) extractor-None falls back to baseline OK",
          file=sys.stderr)
    return 0


def _check_empty_facts_noop() -> int:
    from resonance_lattice.memory.capture import capture
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=ZeroEncoder())
        result = capture(
            _transcript("sess-A", text="x" * 400),
            store=memory, redactor=Redactor(), client=make_stub_llm_facts([]),
        )
        if result.claim_ids:
            print(f"[atomic_capture] FAIL (c): expected empty claim_ids, "
                  f"got {result.claim_ids!r}", file=sys.stderr)
            return 1
        if result.skip_reason != "no extractable events":
            print(f"[atomic_capture] FAIL (c): expected skip_reason="
                  f"'no extractable events', got {result.skip_reason!r}",
                  file=sys.stderr)
            return 1
        if memory.read_all():
            print(f"[atomic_capture] FAIL (c): store has rows on no-op path",
                  file=sys.stderr)
            return 1
    print("[atomic_capture] (c) empty facts → no-op with explicit skip OK",
          file=sys.stderr)
    return 0


def _check_intra_capture_dedup() -> int:
    from resonance_lattice.memory.capture import capture
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    # Extractor emits the same fact twice; capture should collapse to one
    # claim — the architecture's recurrence_count tracks repeats across
    # sessions, not duplicates within a single extractor output.
    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=ZeroEncoder())
        result = capture(
            _transcript("sess-A", text="x" * 400),
            store=memory, redactor=Redactor(),
            client=make_stub_llm_facts(["fact A", "fact B", "fact A"]),
        )
        if len(result.claim_ids) != 2:
            print(f"[atomic_capture] FAIL (d): expected 2 unique claims, "
                  f"got {result.claim_ids!r}", file=sys.stderr)
            return 1
        contents = {c.content for c in memory.read_all()}
        if contents != {"fact A", "fact B"}:
            print(f"[atomic_capture] FAIL (d): unexpected content set {contents!r}",
                  file=sys.stderr)
            return 1
    print("[atomic_capture] (d) intra-capture dedup collapses identical facts "
          "OK", file=sys.stderr)
    return 0


def _check_cross_session_dedup() -> int:
    from resonance_lattice.memory.capture import capture
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    # Session A writes two facts. Session B re-extracts one of them — the
    # existing claim's recurrence_count bumps and no sibling is written.
    facts_a = ["fact A", "fact B"]
    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=ZeroEncoder())
        redactor = Redactor()
        r1 = capture(
            _transcript("sess-A", text="x" * 400),
            store=memory, redactor=redactor, client=make_stub_llm_facts(facts_a),
        )
        r2 = capture(
            _transcript("sess-B", text="y" * 400),
            store=memory, redactor=redactor,
            client=make_stub_llm_facts(["fact A", "fact C"]),
        )
        claims = memory.read_all()
        if len(claims) != 3:
            print(f"[atomic_capture] FAIL (e): expected 3 claims (A, B, C), "
                  f"got {len(claims)}: {[c.content for c in claims]!r}",
                  file=sys.stderr)
            return 1
        a_claim = next(c for c in claims if c.content == "fact A")
        if a_claim.facts.recurrence_count != 2:
            print(f"[atomic_capture] FAIL (e): expected fact A recurrence=2 "
                  f"after cross-session dedup, got "
                  f"{a_claim.facts.recurrence_count}", file=sys.stderr)
            return 1
        # `result.claim_ids[0]` from r2 must be the *same* id as r1's A — the
        # bump path reuses the claim_id rather than minting a new one.
        a_id_r1 = r1.claim_ids[facts_a.index("fact A")]
        a_id_r2 = r2.claim_ids[0]
        if a_id_r1 != a_id_r2:
            print(f"[atomic_capture] FAIL (e): expected reused id on bump, "
                  f"r1[A]={a_id_r1} r2[A]={a_id_r2}", file=sys.stderr)
            return 1
    print("[atomic_capture] (e) cross-session same-text bumps recurrence OK",
          file=sys.stderr)
    return 0


def _check_audit_log_correlation() -> int:
    from resonance_lattice.memory.capture import capture, transcript_hash
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    # Atomic path with redactions: audit lines correlate against
    # transcript_hash (session grain), never against any single claim_id.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        log = root / "redaction.log"
        memory = ExperienceClaimStore(root=root, encoder=ZeroEncoder())
        redactor = Redactor(audit_log_path=log)
        body = (
            "Init line uses sk-ant-" + "A" * 50
            + " for prod calls. We swapped to a fresh key after rotation. "
            + "Padding so the heuristic gate's 200-char floor is cleared. "
            * 4
        )
        transcript = _transcript("sess-leak", text=body)
        t_hash = transcript_hash(transcript)
        result = capture(
            transcript, store=memory, redactor=redactor,
            client=make_stub_llm_facts(["the prod API key was rotated", "rotation worked"]),
        )
        if len(result.claim_ids) != 2 or result.redactions < 1:
            print(f"[atomic_capture] FAIL (f): expected 2 claims + ≥1 "
                  f"redaction, got {result!r}", file=sys.stderr)
            return 1
        log_text = log.read_text(encoding="utf-8")
        if f"row_id={t_hash}" not in log_text:
            print(f"[atomic_capture] FAIL (f): atomic-path audit log missing "
                  f"session row_id={t_hash}; log:\n{log_text}",
                  file=sys.stderr)
            return 1
        for cid in result.claim_ids:
            if f"row_id={cid}" in log_text:
                print(f"[atomic_capture] FAIL (f): atomic-path audit log "
                      f"leaked per-claim row_id={cid} (should correlate at "
                      f"session grain only)", file=sys.stderr)
                return 1
        if "sk-ant-" in log_text:
            print(f"[atomic_capture] FAIL (f): audit log leaked secret",
                  file=sys.stderr)
            return 1
    print("[atomic_capture] (f) atomic-path audit log correlates at "
          "transcript_hash OK", file=sys.stderr)
    return 0


def _check_baseline_path_unchanged() -> int:
    from resonance_lattice.memory.capture import capture, transcript_hash
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    body = (
        "Init line uses sk-ant-" + "A" * 50
        + " for prod calls. We swapped to a fresh key after rotation. "
        + "Padding so the heuristic gate's 200-char floor is cleared. " * 4
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        log = root / "redaction.log"
        memory = ExperienceClaimStore(root=root, encoder=ZeroEncoder())
        redactor = Redactor(audit_log_path=log)
        transcript = _transcript("sess-base", text=body)
        result = capture(transcript, store=memory, redactor=redactor, client=None)
        t_hash = transcript_hash(transcript)
        if len(result.claim_ids) != 1:
            print(f"[atomic_capture] FAIL (g): expected baseline single "
                  f"claim_id, got {result.claim_ids!r}", file=sys.stderr)
            return 1
        claim_id = result.claim_ids[0]
        log_text = log.read_text(encoding="utf-8")
        if f"row_id={claim_id}" not in log_text:
            print(f"[atomic_capture] FAIL (g): baseline audit log missing "
                  f"row_id={claim_id} (should correlate per-claim)",
                  file=sys.stderr)
            return 1
        if f"row_id={t_hash}" in log_text:
            print(f"[atomic_capture] FAIL (g): baseline audit log used "
                  f"transcript_hash (should be claim_id)", file=sys.stderr)
            return 1
    print("[atomic_capture] (g) client=None preserves baseline correlation OK",
          file=sys.stderr)
    return 0


def _check_hook_client_init_failure() -> int:
    # If the LLM SDK construction itself fails (no key, ImportError on
    # `anthropic`, httpx pool exhaustion) the hook helper must return
    # `None` and never propagate — otherwise the outer fail-open catch
    # in `run_capture_hook` swallows even the single-claim baseline,
    # losing the whole session.
    import os
    from resonance_lattice.memory.user_prompt import _capture_hook_client

    # No key set anywhere → `discover_api_key()` returns None → helper
    # returns None without touching the SDK.
    saved = {k: os.environ.pop(k, None) for k in (
        "CLAUDE_API_2", "CLAUDE_API", "ANTHROPIC_API_KEY",
        "RLAT_LLM_API_KEY_ENV",
    )}
    try:
        out = _capture_hook_client()
        if out is not None:
            print(f"[atomic_capture] FAIL (h): expected None with no key, "
                  f"got {out!r}", file=sys.stderr)
            return 1
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    print("[atomic_capture] (h) hook client-init failure → None (no raise) OK",
          file=sys.stderr)
    return 0


def _check_user_turn_channel() -> int:
    """(k) the attribute miner reads SCRUBBED USER turns — the channel where
    world facts actually live (GATE 2 trusts only user statements; the
    assistant narrative stays the event-capture source). Bites three ways:
    the extractor's LLM pass sees the user's statement; it does NOT see the
    assistant narrative; and a secret in a user turn is redacted before any
    LLM call and counted in result.redactions."""
    import json

    from ._testutil import build_corpus

    from resonance_lattice.memory._llm import LLMResponse
    from resonance_lattice.memory.capture import (
        Message, ToolCall, Transcript, capture,
    )
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.redaction import Redactor

    secret = "AKIA" + "A" * 16
    user_text = ("Quick context before we start: our tenant is EU-only for "
                 f"data residency. Also ignore this old key {secret}.")
    assistant_marker = "ASSISTANT-NARRATIVE-" + "x" * 380
    reply = json.dumps({
        "facts": ["Bronze layer ingests Event Hubs JSON."],
        "attributes": ["The tenant is EU-only for data residency."],
    })
    seen: list[str] = []

    def recording_client(system, messages, max_tokens):
        seen.append(messages[0]["content"])
        return LLMResponse(text=reply, input_tokens=10, output_tokens=10)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        km = build_corpus(root / "ws", {"a.md": "# A\n\nWorkspace doc body."})
        memory = ExperienceClaimStore(root=root / "u", encoder=ZeroEncoder())
        transcript = Transcript(
            session_id="sess-user-turn",
            messages=[
                Message("user", user_text),
                Message("assistant", assistant_marker,
                        tool_calls=(ToolCall("bash", "/tmp", "ls"),)),
            ],
            cwd=str(root / "ws"),
        )
        result = capture(
            transcript, store=memory, redactor=Redactor(),
            client=recording_client, km_path=km,
        )
        attr_calls = [c for c in seen if "tenant is EU-only" in c]
        if not attr_calls:
            print("[atomic_capture] FAIL (k): no LLM pass saw the user's "
                  "statement — the miner is not reading user turns",
                  file=sys.stderr)
            return 1
        if any(assistant_marker in c for c in attr_calls):
            print("[atomic_capture] FAIL (k): the attribute pass mixed in "
                  "the assistant narrative", file=sys.stderr)
            return 1
        if any(secret in c for c in seen):
            print("[atomic_capture] FAIL (k): user-turn secret reached an "
                  "LLM call un-redacted", file=sys.stderr)
            return 1
        if result.redactions < 1:
            print(f"[atomic_capture] FAIL (k): user-turn redaction not "
                  f"counted (redactions={result.redactions})", file=sys.stderr)
            return 1
        if len(result.attribute_claim_ids) != 1:
            print(f"[atomic_capture] FAIL (k): expected 1 attribute claim, "
                  f"got {result.attribute_claim_ids!r}", file=sys.stderr)
            return 1
    print("[atomic_capture] (k) miner reads scrubbed user turns; assistant "
          "narrative + secrets never reach it OK", file=sys.stderr)
    return 0


def _check_mining_optin_context() -> int:
    """(m) the capture hook's LLM/km context follows the opt-in flags
    (v3 S1 wake): mining off -> no km ever; RLAT_MINE_ATTRIBUTES=1 with a
    resolvable client -> the workspace's primary .rlat is passed through;
    no key or no archive fails open to None. The seam is
    _capture_llm_context(cwd) - what run_capture_hook hands capture()."""
    import os

    from resonance_lattice.memory import user_prompt as up

    sentinel = object()
    real_client = up._capture_hook_client
    cases_failed = []
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "proj"
        (ws / ".git").mkdir(parents=True)          # pin the workspace root
        km = ws / "proj.rlat"
        km.write_bytes(b"placeholder")             # resolver checks is_file only

        def run_case(name, env, client_resolves, want_client, want_km):
            old_env = {k: os.environ.get(k) for k in
                       ("RLAT_ATOMIC_CAPTURE", "RLAT_MINE_ATTRIBUTES")}
            for k in old_env:
                os.environ.pop(k, None)
            os.environ.update(env)
            up._capture_hook_client = (
                (lambda: sentinel) if client_resolves else (lambda: None))
            try:
                client, km_path = up._capture_llm_context(ws)
            finally:
                up._capture_hook_client = real_client
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            got_client = client is sentinel
            got_km = (km_path is not None
                      and Path(km_path).resolve() == km.resolve())
            if got_client != want_client or got_km != want_km:
                cases_failed.append(
                    f"{name}: client={got_client} (want {want_client}) "
                    f"km={km_path} (want match={want_km})")

        run_case("flags-off", {}, True, False, False)
        run_case("atomic-only", {"RLAT_ATOMIC_CAPTURE": "1"}, True, True, False)
        run_case("mine-on", {"RLAT_MINE_ATTRIBUTES": "1"}, True, True, True)
        run_case("mine-no-key", {"RLAT_MINE_ATTRIBUTES": "1"}, False, False, False)
        km.unlink()
        run_case("mine-no-km", {"RLAT_MINE_ATTRIBUTES": "1"}, True, True, False)

    if cases_failed:
        for line in cases_failed:
            print(f"[atomic_capture] FAIL (m): {line}", file=sys.stderr)
        return 1
    print("[atomic_capture] (m) mining opt-in context: flags/key/km matrix OK",
          file=sys.stderr)
    return 0


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_atomic_success,
        _check_extractor_failure_fallback,
        _check_empty_facts_noop,
        _check_intra_capture_dedup,
        _check_cross_session_dedup,
        _check_audit_log_correlation,
        _check_baseline_path_unchanged,
        _check_hook_client_init_failure,
        _check_polarity_lands_per_fact,
        _check_attribute_mining_lands_in_km,
        _check_user_turn_channel,
        _check_mining_optin_context,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[atomic_capture] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
