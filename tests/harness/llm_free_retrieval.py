"""llm_free_retrieval — the entire user-facing retrieval surface must work
WITHOUT any LLM call. The trust contract is that retrieval is mechanical,
fast, and offline; LLM use is opt-in via `rlat deep-search`.

This suite blocks the Anthropic client from being constructible during
the test — if any code path under test tries to instantiate it, the test
fails. Then it exercises:

  - rlat build (no LLM)
  - manual insight promotion via the store API (no LLM)
  - rlat search (default + --source-only + --include-stale)
  - rlat audit (text + json)
  - rlat trace
  - rlat lens create + show
  - drift cascade via rlat refresh

If any of these raise during the test, the LLM-free guarantee has been
broken and we need to investigate.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim, run_cli, unpatch_zero_encoder


class _NoAnthropicSentinel:
    """Replacement for the Anthropic SDK client class — raises on use.

    Mounted into `sys.modules` for the duration of the test so any
    attempt to instantiate `anthropic.Anthropic()` mid-test raises a
    distinctive AssertionError. The test framework prints this as the
    failure cause; the suite's assertion is "this never fires."
    """
    def __init__(self, *args, **kwargs):
        raise AssertionError(
            "LLM_FREE_GUARANTEE_BROKEN: anthropic.Anthropic() was "
            "instantiated during what should have been an LLM-free retrieval "
            "test. Investigate the call site and either route the LLM "
            "usage to rlat deep-search (the opt-in path) or fix the regression."
        )


import contextlib


@contextlib.contextmanager
def _llm_guard():
    """Context-manager wrap: install the sentinel for the duration of the
    test, then restore the real class so subsequent suites in an --all
    sweep that legitimately need anthropic aren't broken.
    """
    import anthropic
    original = anthropic.Anthropic  # type: ignore[attr-defined]
    anthropic.Anthropic = _NoAnthropicSentinel  # type: ignore[attr-defined]
    try:
        yield
    finally:
        anthropic.Anthropic = original  # type: ignore[attr-defined]


def run() -> int:
    unpatch_zero_encoder()
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store import archive

    failures = 0
    with _llm_guard(), tempfile.TemporaryDirectory() as d:
        root = Path(d) / "corpus"
        # ---- Guarantee 1: build succeeds with no LLM ----
        try:
            km = _build(root, {
                "a.md": "# Auth\n\nSession tokens expire after 24h.",
                "b.md": "# Tokens\n\nRefresh tokens rotate weekly.",
            })
            print("[llm_free_retrieval] g1 (rlat build, no LLM) OK", file=sys.stderr)
        except AssertionError as e:
            print(f"[llm_free_retrieval] FAIL g1: {e}", file=sys.stderr)
            return 1

        c0 = archive.read(km)
        src_ids = [c.passage_id for c in c0.registry]
        src_hashes = [c.content_hash for c in c0.registry]

        # ---- Guarantee 2: manual insight promotion, no LLM ----
        try:
            encoder = Encoder()                       # encoder is local, not LLM
            ins = [make_corpus_claim(
                "Sessions use 24h tokens; refresh weekly.",
                src_ids[:2], src_hashes[:2], state="active",
            )]
            band = encoder.encode([ins[0].content]).astype("float32")
            archive.write_insight_layer_in_place(km, ins, band)
            print("[llm_free_retrieval] g2 (manual promotion, no LLM) OK",
                  file=sys.stderr)
        except AssertionError as e:
            print(f"[llm_free_retrieval] FAIL g2: {e}", file=sys.stderr)
            failures += 1

        # ---- Guarantee 3: rlat search default + --source-only + --include-stale ----
        for flags in ([], ["--source-only"], ["--include-stale"]):
            try:
                rc, out, err = run_cli([
                    "search", str(km), "session tokens",
                    "--top-k", "5", "--format", "json", "--quiet",
                ] + flags)
                if rc != 0:
                    print(f"[llm_free_retrieval] FAIL g3 ({flags}): rc={rc}\n{err}",
                          file=sys.stderr)
                    failures += 1
                    break
                _ = json.loads(out)  # parses cleanly
            except AssertionError as e:
                print(f"[llm_free_retrieval] FAIL g3 ({flags}): {e}", file=sys.stderr)
                failures += 1
                break
        else:
            print("[llm_free_retrieval] g3 (rlat search × 3 flag combos) OK",
                  file=sys.stderr)

        # ---- Guarantee 4: rlat audit (text + json) ----
        try:
            rc1, out1, _ = run_cli(["audit", str(km)])
            rc2, out2, _ = run_cli(["audit", str(km), "--format", "json"])
            if rc1 != 0 or rc2 != 0:
                print(f"[llm_free_retrieval] FAIL g4: rc={rc1},{rc2}",
                      file=sys.stderr)
                failures += 1
            else:
                print("[llm_free_retrieval] g4 (rlat audit, no LLM) OK",
                      file=sys.stderr)
        except AssertionError as e:
            print(f"[llm_free_retrieval] FAIL g4: {e}", file=sys.stderr)
            failures += 1

        # ---- Guarantee 5: rlat trace, no LLM ----
        try:
            rc, out, _ = run_cli(["trace", str(km), ins[0].claim_id])
            if rc != 0:
                print(f"[llm_free_retrieval] FAIL g5: rc={rc}", file=sys.stderr)
                failures += 1
            else:
                print("[llm_free_retrieval] g5 (rlat trace, no LLM) OK",
                      file=sys.stderr)
        except AssertionError as e:
            print(f"[llm_free_retrieval] FAIL g5: {e}", file=sys.stderr)
            failures += 1

        # ---- Guarantee 6: rlat lens create + show, no LLM ----
        try:
            lens_path = Path(d) / "demo.lens"
            rc, _, err = run_cli([
                "lens", "create",
                "--id", "lens-demo", "--name", "demo", "--scope", "user",
                "-o", str(lens_path),
            ])
            if rc != 0:
                print(f"[llm_free_retrieval] FAIL g6: create rc={rc} err={err}",
                      file=sys.stderr)
                failures += 1
            else:
                rc, out, _ = run_cli(["lens", "show", str(lens_path)])
                if rc != 0:
                    print(f"[llm_free_retrieval] FAIL g6: show rc={rc}",
                          file=sys.stderr)
                    failures += 1
                else:
                    print("[llm_free_retrieval] g6 (rlat lens, no LLM) OK",
                          file=sys.stderr)
        except AssertionError as e:
            print(f"[llm_free_retrieval] FAIL g6: {e}", file=sys.stderr)
            failures += 1

        # ---- Guarantee 7: rlat refresh + drift cascade, no LLM ----
        try:
            (root / "a.md").write_text(
                "# Auth (revised)\n\nSession tokens expire after 12h now.",
                encoding="utf-8",
            )
            rc, _, _ = run_cli([
                "refresh", str(km), "--source-root", str(root),
            ])
            if rc != 0:
                print(f"[llm_free_retrieval] FAIL g8: refresh rc={rc}",
                      file=sys.stderr)
                failures += 1
            else:
                c_after = archive.read(km)
                if not any(i.state == "stale" for i in c_after.insights):
                    print("[llm_free_retrieval] FAIL g8: drift didn't cascade",
                          file=sys.stderr)
                    failures += 1
                else:
                    print("[llm_free_retrieval] g8 (refresh + drift cascade, no LLM) OK",
                          file=sys.stderr)
        except AssertionError as e:
            print(f"[llm_free_retrieval] FAIL g8: {e}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"[llm_free_retrieval] {failures} guarantee(s) failed",
              file=sys.stderr)
        return 1
    print("[llm_free_retrieval] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
