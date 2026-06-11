"""Shared helpers for harness suites.

Lifted out of the per-suite duplication that crept in across
`incremental_refresh`, `incremental_sync`, `skill_context`,
`band_parity`, and `conversion` — each had its own
`_Args` micro-class + a `_build` helper that mirrored argparse's
`Namespace` shape and `cmd_build` invocation pattern. One owner now.

Test-only module — production code never imports from here.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Literal

import numpy as np


class Args:
    """Argparse-Namespace stand-in for harness suites.

    cmd_* functions in `cli/` accept `argparse.Namespace`-like objects via
    duck typing (`args.foo`, `args.bar` attribute access). Constructing a
    real `Namespace` works but doesn't carry intent — the suites want a
    fixture-style "build me an args bag with these fields" call. `Args(**kw)`
    sets every kwarg as an attribute and is equivalent to `Namespace(**kw)`
    for downstream consumers.
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_StoreMode = Literal["bundled", "local", "remote"]


def build_corpus(
    root: Path,
    files: dict[str, str],
    *,
    mode: _StoreMode = "local",
    remote_url_base: str | None = None,
    min_chars: int = 20,
    max_chars: int = 400,
    batch_size: int = 4,
) -> Path:
    """Materialise `files` under `root` and run `cmd_build` against them.

    Returns the resulting `.rlat` path. Used by every harness suite that
    needs a small corpus to exercise refresh/sync/convert; lifted
    out of per-suite copies that all looked the same modulo the
    store_mode + source-dir layout.

    For `bundled` mode the files are written under `root/src/` and the
    build sources point at that subdir (matches what the per-suite
    `_build_remote` and `_build_bundled` did before this lift).
    """
    from resonance_lattice.cli.build import cmd_build

    if mode == "bundled" or mode == "remote":
        src_dir = root / "src"
    else:
        src_dir = root
    src_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = src_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    out = root / "km.rlat"
    rc = cmd_build(Args(
        sources=[str(src_dir)], output=str(out),
        store_mode=mode, kind="corpus", source_root=str(src_dir),
        min_chars=min_chars, max_chars=max_chars, batch_size=batch_size,
        ext=None,
        remote_url_base=remote_url_base,
    ))
    if rc != 0:
        raise RuntimeError(f"build rc={rc} (mode={mode}, root={root})")
    return out


class ZeroEncoder:
    """Mock encoder — zero-vectors, deterministic, no model load.

    Used by suites that don't test recall scoring; cosines collapse
    to zero so the four §0.6 gates fire on the metadata path only.

    `__init__` swallows arbitrary args so this class can transparently
    stand in for `field.encoder.Encoder(runtime=...)` when
    `patch_zero_encoder` rebinds the symbol globally — without that
    swallow, downstream suites that build via `cmd_build` (which calls
    `Encoder(runtime=runtime)`) would TypeError after a memory_v21
    suite has run in the same `--all` sweep.
    """

    # `cmd_build` reads `encoder.revision` + `encoder.runtime_name` when
    # stamping build metadata — must be present so `--all` sweeps (where a
    # memory_v21 suite has rebound the symbol globally before an
    # incremental_* / arrow suite calls `cmd_build`) don't
    # AttributeError.
    revision = "zero-encoder-test"
    runtime_name = "zero-encoder-test"

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768), dtype="float32")

    def encode_batched(self, texts: list[str], batch_size: int = 0) -> np.ndarray:
        # Real `Encoder.encode_batched` was lifted in simplify-3 commit
        # 38d18d64; the fake fixture must mirror the surface for the same
        # cross-suite reason as `revision` above.
        return self.encode(texts)


class FixedEncoder:
    """Mock encoder that returns a pre-planted query vector.

    Recall + workspace-scope contract suites construct band rows with
    controlled cosines against a known query; passing this encoder to
    `rank()` lets the suite assert exact post-gate ordering without
    relying on the live encoder's behaviour.
    """

    def __init__(self, query_vec: np.ndarray):
        self.query_vec = query_vec.astype("float32", copy=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.query_vec], dtype="float32")


def make_stub_llm_client(canned: str):
    """Build a callable matching the `LLMClient` shape that always
    answers with `canned`. Used by suites that stand in for live
    Anthropic SDK calls."""
    from resonance_lattice.memory._llm import LLMResponse

    def client(system, messages, max_tokens):
        return LLMResponse(text=canned, input_tokens=10, output_tokens=10)

    return client


def make_stub_llm_facts(facts: list[str]):
    """`LLMClient` stub answering with a `{"facts": [...]}` JSON blob —
    matches `extract_events`' output contract."""
    import json
    return make_stub_llm_client(json.dumps({"facts": facts}))


def make_stub_llm_raising(exc: Exception):
    """`LLMClient` stub that raises on call — exercises the seam's
    "never propagate" contract."""
    def _raise(system, messages, max_tokens):
        raise exc
    return _raise


_StubJudgeContent = namedtuple("_StubJudgeContent", "text")
_StubJudgeResponse = namedtuple("_StubJudgeResponse", "content")


class StubJudgeClient:
    """Anthropic-shaped client replaying one scripted JSON response.

    Mimics the `client.messages.create(...)` surface that
    `store._llm.judge_json` consumes — the response exposes
    `.content[0].text`. `.calls` counts invocations so a suite can assert
    the LLM was NOT hit on a short-circuit path. Shared by the judge
    suites (faithfulness, load_bearing, …).
    """

    def __init__(self, response_text: str):
        self._text = response_text
        self.calls = 0
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls += 1
                return _StubJudgeResponse(
                    content=[_StubJudgeContent(text=outer._text)]
                )

        self.messages = _Messages()


def make_experience_claim(
    *,
    claim_id: str,
    content: str,
    polarity,
    transcript_hash: str,
    kind: str = "event",
    confidence: str = "medium",
    recurrence_count: int = 1,
    criticality: str = "normal",
    created_under_intent_kind: str = "none",
    origin: str | None = None,
    created_at: str = "2026-05-08T00:00:00Z",
    last_corroborated_at: str | None = None,
    is_bad: bool = False,
    state: str = "active",
):
    """Build an experience `state.claim.Claim` for harness fixtures.

    Successor to the old `Memory.add_row(...)` kwargs bag. `confidence`
    seeds the Beta tallies via `seed_tallies_for_rung` (confidence is a
    derived read-only property — never set directly). `origin` defaults
    to whatever the `transcript_hash` prefix implies. `state` defaults to
    `active` (a settled fixture claim); pass `candidate` to exercise the
    experience earning gate.
    """
    from resonance_lattice.state.claim import Claim, ExperienceFacts, derive_origin
    from resonance_lattice.memory.store import seed_tallies_for_rung

    corroboration, falsification = seed_tallies_for_rung(confidence)
    return Claim(
        claim_id=claim_id,
        source="experience",
        kind=kind,
        content=content,
        created_at=created_at,
        corroboration=corroboration,
        falsification=falsification,
        trust_as_of="",
        state=state,
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=tuple(polarity),
            recurrence_count=recurrence_count,
            criticality=criticality,
            created_under_intent_kind=created_under_intent_kind,
            transcript_hash=transcript_hash,
            origin=origin or derive_origin(transcript_hash),
            last_corroborated_at=last_corroborated_at or created_at,
            is_bad=is_bad,
        ),
    )


def seed_capture_memory(memory, captures: list[dict], *, workspace_path: str = "/proj") -> None:
    """Pre-populate an ExperienceClaimStore with capture-time claims.

    Each `captures` entry is a `{"text", "transcript_hash"}` dict; the
    helper stamps the §0.1 capture-time polarity (`factual` + the cwd
    workspace tag) and writes a zero-vector embedding so the dedupe
    path doesn't accidentally match by cosine.
    """
    from resonance_lattice.memory._common import workspace_tag_for_cwd

    cwd_tag = workspace_tag_for_cwd(workspace_path)
    for i, cap in enumerate(captures):
        memory.write(
            make_experience_claim(
                claim_id=f"01HZCAP{i:017d}",
                content=cap["text"],
                polarity=["factual", cwd_tag],
                transcript_hash=cap["transcript_hash"],
            ),
            embedding=np.zeros(768, dtype="float32"),
        )


def isolated_daemon_address(root: Path) -> str:
    """Per-test daemon IPC address that won't collide with a live user
    daemon on the same host.

    POSIX: a unique `<root>/.test-<salt>.sock` path.
    Windows: `\\\\.\\pipe\\rlat-test-<salt>` (named pipes have a global
    namespace — the salt is what keeps parallel test runs apart).
    """
    import os

    from resonance_lattice.memory._common import workspace_hash

    salt = os.urandom(3).hex()
    suffix = workspace_hash(str(root))[:6] + salt
    if os.name == "nt":
        return r"\\.\pipe\rlat-test-" + suffix
    return str(root / f".test-{suffix}.sock")


def booted_daemon(memory, *, address: str, encoder_revision: str = "test-rev",
                  idle_exit_seconds: int = 600,
                  reload_poll_seconds: float = 0.05,
                  boot_timeout_s: float = 2.0,
                  encoder=None):
    """Context manager that boots a `DaemonServer` in a daemon thread,
    waits for the listener to come up, and tears it down on exit.

    Yields `(server, thread)`. Used by the daemon harness suite to
    keep the boot/teardown plumbing out of every contract test body.
    `encoder` defaults to `ZeroEncoder()` (zero cosines — metadata-path
    contracts); pass a `FixedEncoder` to drive real cosines (e.g. corpus
    band-recall ordering) without a model load.
    """
    import contextlib
    import threading
    import time

    from resonance_lattice.memory.daemon import DaemonServer

    @contextlib.contextmanager
    def _ctx():
        server = DaemonServer(
            store=memory,
            encoder=encoder or ZeroEncoder(),
            encoder_revision=encoder_revision,
            address=address,
            idle_exit_seconds=idle_exit_seconds,
            reload_poll_seconds=reload_poll_seconds,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        deadline = time.monotonic() + boot_timeout_s
        while time.monotonic() < deadline and server._listener is None:
            time.sleep(0.01)
        try:
            yield server, thread
        finally:
            server.stop()
            thread.join(timeout=1.0)

    return _ctx()


def run_cli(argv: list[str], *, stdin_text: str | None = None) -> tuple[int, str, str]:
    """Invoke `rlat <argv>` through the CLI dispatch entry point.

    Captures stdout + stderr, returns (rc, stdout, stderr). Optional
    `stdin_text` stubs `sys.stdin` for the duration via `mock.patch`
    (contextlib has no `redirect_stdin`).
    """
    import contextlib
    import io
    from unittest.mock import patch

    from resonance_lattice.cli.app import main

    out, err = io.StringIO(), io.StringIO()
    stdin_ctx = (
        patch("sys.stdin", io.StringIO(stdin_text))
        if stdin_text is not None
        else contextlib.nullcontext()
    )
    with stdin_ctx, \
         contextlib.redirect_stdout(out), \
         contextlib.redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def check_guarantee(ok: bool, label: str, prefix: str) -> bool:
    """Print PASS/FAIL banner for a single harness guarantee.

    Returns the boolean unchanged so the caller can fold it into a
    failure counter (`failures += not check_guarantee(...)`). `prefix`
    is the suite name (e.g. "fabric_bootstrap") so a multi-suite log
    stays attributable.
    """
    import sys
    if not ok:
        print(f"[{prefix}] FAIL {label}", file=sys.stderr)
        return False
    print(f"[{prefix}] {label} OK", file=sys.stderr)
    return True


def make_corpus_claim(
    content: str,
    source_ids: list[str],
    source_hashes: list[str] | None = None,
    *,
    state: str = "candidate",
    faithfulness: float = 0.8,
):
    """Construct a corpus `Claim` fixture for harness suites.

    `source_hashes` defaults to source_ids (one-to-one binding by id);
    pass distinct values when testing the drift cascade. `state` lets
    suites build candidate / active / stale claims for state-machine
    coverage. `faithfulness` seeds the Beta trust prior so the fixture
    carries a valid `(corroboration, falsification)` state.

    The `Claim` is built directly (not via `new_corpus_claim`) so
    `created_at` is a fixed timestamp — harness assertions stay
    deterministic.
    """
    from resonance_lattice.state.claim import Claim, CorpusFacts
    from resonance_lattice.store.insight import (
        InsightCitation,
        compute_insight_id,
        seed_confidence,
    )

    if source_hashes is None:
        source_hashes = list(source_ids)
    citations = tuple(
        InsightCitation(passage_id=pid, char_span=None, confidence=0.9)
        for pid in source_ids
    )
    fingerprint = compute_insight_id(
        content, tuple(source_hashes), "model-x"
    )
    corroboration, falsification = seed_confidence(faithfulness)
    return Claim(
        claim_id=fingerprint,
        source="corpus",
        kind="synthesis",
        content=content,
        created_at="2026-05-13T10:00:00Z",
        corroboration=corroboration,
        falsification=falsification,
        trust_as_of="",
        state=state,
        parent_ids=(),
        facts=CorpusFacts(
            citations=citations,
            content_fingerprint=fingerprint,
            source_model_hash="model-x",
            source_passage_hashes=tuple(source_hashes),
            verdict_signals=(),
            query="test",
            intent_context=None,
            stale_if_sources_drift=True,
            encoder_version="gte-mb-768",
            seed_corroboration=corroboration,
            seed_falsification=falsification,
        ),
    )


def patch_zero_encoder() -> None:
    """Patch `field.encoder.Encoder` to the `ZeroEncoder` stub everywhere.

    Idempotent — a second call is a no-op. Memory suites import this at
    `run()` entry so any subsequent lazy `Encoder()` construction in the
    capture / store / CLI paths lands on the stub.

    Importer modules that did `from ..field.encoder import Encoder` have
    a local binding to the original class — patching only the source
    module misses them. We patch every known consumer explicitly.
    """
    import resonance_lattice.field.encoder as _enc
    import resonance_lattice.memory.claim_store as _store

    _enc.Encoder = ZeroEncoder  # type: ignore[assignment,misc]
    _store.Encoder = ZeroEncoder  # type: ignore[assignment,misc]

    # Mirror unpatch_zero_encoder's consumer list: modules that captured
    # Encoder via `from ..field.encoder import Encoder` at import time hold
    # the REAL class if they were imported before this call (e.g. an earlier
    # suite in the same --all sweep ran unpatch at entry). Rebinding only the
    # source modules would leave those consumers building with the real
    # encoder — non-hermetic and encoder-dependent. Only already-imported
    # modules are touched; nothing new is imported here.
    import sys as _sys
    for mod_name in (
        "resonance_lattice.cli.search",
        "resonance_lattice.cli.deep_search",
        "resonance_lattice.cli.skill_context",
        "resonance_lattice.build.pipeline",
        "resonance_lattice.store.incremental",
    ):
        mod = _sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "Encoder"):
            mod.Encoder = ZeroEncoder  # type: ignore[assignment,misc]


def unpatch_zero_encoder() -> None:
    """Restore the real `Encoder` class across every patched module.

    Inverse of `patch_zero_encoder`. Reloads `field.encoder` to recover
    the original class definition, then re-binds the symbol in every
    known consumer module that captured `ZeroEncoder` via `from ... import`.
    Idempotent — safe to call before any test even if no patch was
    installed.

    Suites that depend on real cosine scoring (insight_layer,
    audit_trace_cli, llm_free_retrieval) must call this at run() entry
    to defeat cross-suite contamination from earlier memory suites in
    an --all / --changed sweep.
    """
    import importlib

    import resonance_lattice.field.encoder as _enc
    importlib.reload(_enc)
    real_encoder = _enc.Encoder

    # The experience claim store captures Encoder at import time.
    import resonance_lattice.memory.claim_store as _store
    _store.Encoder = real_encoder  # type: ignore[assignment,misc]

    # CLI surfaces and core paths that do `from ..field.encoder import Encoder`.
    # Adding a new consumer requires extending this list — the harness
    # contract is documented in this module.
    for mod_name in (
        "resonance_lattice.cli.search",
        "resonance_lattice.cli.deep_search",
        "resonance_lattice.cli.skill_context",
        "resonance_lattice.build.pipeline",
        "resonance_lattice.store.incremental",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "Encoder"):
            setattr(mod, "Encoder", real_encoder)
