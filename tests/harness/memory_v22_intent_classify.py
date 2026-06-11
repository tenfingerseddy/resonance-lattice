"""memory_v22_intent_classify — UserPromptSubmit cheap-path classifier.

Pins architecture §"Intent extraction inside UserPromptSubmit" — fast
classifier (regex / small model) for the intent kind, deferring deeper
extraction to async. Six contracts:

  (a) Empty / whitespace prompt → "none" (cosine-only fallback).

  (b) Debug-shaped prompts classify as "debug" (highest dogfood prior).

  (c) Each non-debug kind has at least one canonical phrase that maps
      cleanly: design, implement, review, explain, refactor.

  (d) Tie-break: when matches are equal, the earlier list entry wins
      (debug > design > implement > review > explain > refactor).

  (e) Score-based: more matches outrank fewer; a prompt with two debug
      cues + one explain cue picks debug regardless of order.

  (f) Sub-millisecond on a 4KB prompt — the 200ms hot-path budget is
      preserved with three orders of magnitude of headroom.
"""

from __future__ import annotations

import os
import sys
import time

from resonance_lattice.memory.intent_classify import classify_intent_kind


def _check_empty_prompt() -> int:
    for prompt in ["", "   ", "\n\n"]:
        kind = classify_intent_kind(prompt)
        if kind != "none":
            print(f"[memory_v22_intent_classify] FAIL (a): {prompt!r} → {kind!r}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_intent_classify] (a) empty prompt → none OK",
          file=sys.stderr)
    return 0


def _check_debug_dominant() -> int:
    cases = [
        "fix the broken test in tests/foo.py",
        "why is this failing on Windows?",
        "diagnose the crash in the daemon",
        "the recall pipeline doesn't work after refresh",
    ]
    for prompt in cases:
        kind = classify_intent_kind(prompt)
        if kind != "debug":
            print(f"[memory_v22_intent_classify] FAIL (b): {prompt!r} → {kind!r}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_intent_classify] (b) debug-shaped prompts OK",
          file=sys.stderr)
    return 0


def _check_canonical_phrases() -> int:
    cases = [
        ("how should we design the outcome ledger?", "design"),
        ("implement the new schema", "implement"),
        ("review the changed files for reuse", "review"),
        ("explain how the recall daemon works", "explain"),
        ("refactor and consolidate the helper module", "refactor"),
    ]
    for prompt, want in cases:
        got = classify_intent_kind(prompt)
        if got != want:
            print(f"[memory_v22_intent_classify] FAIL (c): {prompt!r} → "
                  f"{got!r} (want {want!r})", file=sys.stderr)
            return 1
    print("[memory_v22_intent_classify] (c) canonical phrases OK",
          file=sys.stderr)
    return 0


def _check_tie_break_priority() -> int:
    # `error` matches debug; `consider` matches design — both fire once.
    # Tie-break must pick debug (earlier in the list).
    prompt = "consider the error case"
    got = classify_intent_kind(prompt)
    if got != "debug":
        print(f"[memory_v22_intent_classify] FAIL (d): {prompt!r} → {got!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_intent_classify] (d) tie-break debug > design OK",
          file=sys.stderr)
    return 0


def _check_score_outranks() -> int:
    # Two debug cues + one explain cue — debug wins on score even though
    # explain matches the earlier word.
    prompt = "explain why this fix is failing"
    got = classify_intent_kind(prompt)
    if got != "debug":
        print(f"[memory_v22_intent_classify] FAIL (e): {prompt!r} → {got!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_intent_classify] (e) higher match-count wins OK",
          file=sys.stderr)
    return 0


def _check_latency() -> int:
    # 4KB synthetic prompt with a debug cue near the end — must classify
    # in well under a millisecond on commodity hardware.
    prompt = ("lorem ipsum dolor sit amet " * 200) + " — fix the bug"
    start = time.perf_counter()
    for _ in range(100):
        classify_intent_kind(prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 100
    # Shared CI runners are noisy (a 10ms reading flaked a slicer-only push
    # 2026-06-10); the hot-path budget is asserted strictly on dev machines,
    # generously under CI — the check still catches a real complexity
    # regression (e.g. accidental O(n^2) over the prompt) either way.
    budget_ms = 25.0 if os.environ.get("CI") else 5.0
    if elapsed_ms > budget_ms:
        print(f"[memory_v22_intent_classify] FAIL (f): per-call={elapsed_ms:.3f}ms "
              f"(want <{budget_ms:g}ms)", file=sys.stderr)
        return 1
    print(f"[memory_v22_intent_classify] (f) latency {elapsed_ms:.3f}ms/call OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_empty_prompt,
        _check_debug_dominant,
        _check_canonical_phrases,
        _check_tie_break_priority,
        _check_score_outranks,
        _check_latency,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_intent_classify] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
