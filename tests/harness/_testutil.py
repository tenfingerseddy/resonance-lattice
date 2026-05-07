"""Shared helpers for harness suites.

Lifted out of the per-suite duplication that crept in across
`incremental_refresh`, `incremental_sync`, `optimised_reproject`,
`skill_context`, `band_parity`, and `conversion` — each had its own
`_Args` micro-class + a `_build` helper that mirrored argparse's
`Namespace` shape and `cmd_build` invocation pattern. One owner now.

Test-only module — production code never imports from here.
"""

from __future__ import annotations

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
    needs a small corpus to exercise refresh/sync/convert/optimise; lifted
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

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768), dtype="float32")


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
    answers with `canned`. Used by every distil/train suite to
    stand in for live Anthropic SDK calls.
    """
    from resonance_lattice.memory.distil import LLMResponse

    def client(system, messages, max_tokens):
        return LLMResponse(text=canned, input_tokens=10, output_tokens=10)

    return client


def seed_capture_memory(memory, captures: list[dict], *, workspace_path: str = "/proj") -> None:
    """Pre-populate a Memory with capture-time rows.

    Each `captures` entry is a `{"text", "transcript_hash"}` dict; the
    helper stamps the §0.1 capture-time polarity (`factual` + the cwd
    workspace tag) and writes a zero-vector embedding so the dedupe
    path doesn't accidentally match by cosine.
    """
    from resonance_lattice.memory._common import workspace_tag_for_cwd

    cwd_tag = workspace_tag_for_cwd(workspace_path)
    for cap in captures:
        memory.add_row(
            text=cap["text"],
            polarity=["factual", cwd_tag],
            transcript_hash=cap["transcript_hash"],
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
                  boot_timeout_s: float = 2.0):
    """Context manager that boots a `DaemonServer` in a daemon thread,
    waits for the listener to come up, and tears it down on exit.

    Yields `(server, thread)`. Used by the daemon harness suite to
    keep the boot/teardown plumbing out of every contract test body.
    """
    import contextlib
    import threading
    import time

    from resonance_lattice.memory.daemon import DaemonServer

    @contextlib.contextmanager
    def _ctx():
        server = DaemonServer(
            store=memory,
            encoder=ZeroEncoder(),
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
    import resonance_lattice.memory.store as _store

    _enc.Encoder = ZeroEncoder  # type: ignore[assignment,misc]
    _store.Encoder = ZeroEncoder  # type: ignore[assignment,misc]
