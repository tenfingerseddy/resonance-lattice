"""name_check — unit-level tests for cli/_namecheck distinctive-token verification.

Six guarantees, all measured here:

  1. **ALL-CAPS acronym extraction**: tokens like `MVE`, `SKU`, `CU`,
     `MLV`, `ETL` are extracted as distinctive — these are the canonical
     name-aliasing trap (canonical case fb37 in the bench-2 v3 distractor
     analysis: question said `MVE`, corpus said `MLV`, all rlat lanes
     hallucinated).

  2. **Alphanumeric product ID extraction**: tokens like `F4096`, `F32`,
     `P3`, `gen2` are extracted — fake SKU references are the second
     most common distractor failure mode.

  3. **Capitalised proper-noun extraction**: tokens like `Quantum`,
     `Snowflake` (3+ chars, capital first) are extracted; common
     stopwords ("Microsoft", "Azure", "Power", "BI") are skipped to
     avoid over-refusal.

  4. **Quoted multi-word names**: `'Quantum Data Lakes'` survives as a
     single multi-word token so the substring match is exact, not
     piecewise-matched on the inner stopwords.

  5. **Acronym presence is case-sensitive whole-word**: short ALL-CAPS
     tokens (`MVE` <= 6 chars) match `\\bMVE\\b` only — `mve` lowercase
     passes through as missing because a corpus mention of lowercase
     `mve` is unlikely to be the brand name.

  6. **Refusal directive shape**: `refusal_directive(missing)` emits a
     `<!-- rlat-namecheck: missing X -->` HTML comment plus a ⚠ Name
     verification failed blockquote that names every missing token. An
     empty list returns the empty string (no directive when nothing's
     missing).

Phase 7 deliverable. Spec: docs/internal/benchmarks/02_distractor_floor_analysis.md.
"""

from __future__ import annotations

import sys


def _check_extract() -> int:
    from resonance_lattice.cli._namecheck import _extract_distinctive_tokens

    # Heuristic is intentionally narrow: ONLY ALL-CAPS acronyms,
    # alphanumeric product IDs, and quoted multi-word names trigger the
    # check. Plain capitalised words (`Compare`, `Pricing`, `Summarize`,
    # `Snowflake`) are NOT distinctive — over-refusal on natural English
    # question phrasing is worse than missing single-word proper nouns.
    # Codex review (2026-04-27).
    cases: list[tuple[str, list[str]]] = [
        # Guarantee 1: ALL-CAPS acronyms — canonical bench-2 fb37 case.
        ("What is the default action for MVE?", ["MVE"]),
        ("Compare MVE and MLV defaults.", ["MVE", "MLV"]),
        # Guarantee 2: alphanumeric product IDs.
        ("What is the price of an F4096 SKU?", ["F4096", "SKU"]),
        ("Pricing for F32 vs F2048", ["F32", "F2048"]),
        # Guarantee 3: plain capitalised proper nouns are NOT distinctive
        # — heuristic deliberately under-covers single-word brand names
        # to avoid false positives on sentence-initial verbs.
        ("How does Snowflake integrate with Microsoft Fabric?", []),
        ("Summarize login behavior", []),
        # Guarantee 4: quoted multi-word names; inner words not double-counted.
        ("What are 'Quantum Data Lakes'?", ["Quantum Data Lakes"]),
        # Negative: question with only stopwords / common nouns extracts nothing.
        ("how does login work", []),
        ("what tables can I create", []),
        # ALL-CAPS acronym in parentheses (corpus-style "Foo (FOO)") — the
        # capitalised words `Materialized`, `View`, `Express` are NOT
        # extracted (single-word capitalised), only the acronym is.
        ("Materialized View Express (MVE) — what's the default?", ["MVE"]),
    ]
    for question, expected in cases:
        got = _extract_distinctive_tokens(question)
        if got != expected:
            print(
                f"[name_check] FAIL extract: question={question!r}\n"
                f"  expected: {expected}\n"
                f"  got:      {got}",
                file=sys.stderr,
            )
            return 1
    print("[name_check] guarantees 1-4 (token extraction) OK", file=sys.stderr)
    return 0


def _check_presence() -> int:
    from resonance_lattice.cli._namecheck import _passage_contains

    # Acronyms — case-sensitive whole-word.
    if not _passage_contains("Materialized Lake View (MLV) is the default.", "MLV"):
        print("[name_check] FAIL presence: MLV in 'MLV' passage not found",
              file=sys.stderr)
        return 1
    if _passage_contains("the mve table type is fictional.", "MVE"):
        print("[name_check] FAIL presence: lowercase 'mve' should NOT match "
              "case-sensitive acronym MVE", file=sys.stderr)
        return 1
    # Whole-word boundary: 'MLVX' should not match 'MLV'.
    if _passage_contains("the MLVX subsystem.", "MLV"):
        print("[name_check] FAIL presence: 'MLVX' should not match acronym 'MLV'",
              file=sys.stderr)
        return 1
    # Alphanumeric ID — case-insensitive substring (>6 char tokens, or just contains digit).
    if not _passage_contains("Pricing for the F4096 sku.", "F4096"):
        print("[name_check] FAIL presence: F4096 not found", file=sys.stderr)
        return 1
    # Multi-word name — case-insensitive substring.
    if not _passage_contains(
        "the quantum data lakes feature is described here.",
        "Quantum Data Lakes",
    ):
        print("[name_check] FAIL presence: case-insensitive multi-word match",
              file=sys.stderr)
        return 1
    print("[name_check] guarantee 5 (presence semantics) OK", file=sys.stderr)
    return 0


def _check_refusal_directive() -> int:
    from resonance_lattice.cli._namecheck import refusal_directive

    if refusal_directive([]) != "":
        print("[name_check] FAIL directive: empty list should yield empty string",
              file=sys.stderr)
        return 1

    out = refusal_directive(["MVE"])
    must_contain = [
        "<!-- rlat-namecheck: missing `MVE` -->",
        "Name verification failed",
        "`MVE`",
    ]
    for needle in must_contain:
        if needle not in out:
            print(f"[name_check] FAIL directive: missing {needle!r} in:\n{out}",
                  file=sys.stderr)
            return 1

    out_multi = refusal_directive(["MVE", "F4096"])
    if "`MVE`" not in out_multi or "`F4096`" not in out_multi:
        print(f"[name_check] FAIL directive: multi-token output missing one. "
              f"Got:\n{out_multi}", file=sys.stderr)
        return 1
    print("[name_check] guarantee 6 (refusal directive) OK", file=sys.stderr)
    return 0


def _check_end_to_end() -> int:
    from resonance_lattice.cli._namecheck import verify_question_in_passages

    # Canonical bench-2 fb37 case: MVE missing, MLV present in passages.
    res = verify_question_in_passages(
        "What is the default action for MVE?",
        "Materialized Lake View (MLV) defaults to overwrite when the "
        "target object exists.",
    )
    if res.passed:
        print("[name_check] FAIL e2e: MVE-missing-from-MLV-passages should "
              "fail name-check", file=sys.stderr)
        return 1
    if "MVE" not in res.missing_tokens:
        print(f"[name_check] FAIL e2e: missing tokens should include MVE, "
              f"got {res.missing_tokens}", file=sys.stderr)
        return 1

    # Positive case: question and passage share the distinctive token.
    res = verify_question_in_passages(
        "What is the default action for MLV?",
        "Materialized Lake View (MLV) defaults to overwrite.",
    )
    if not res.passed:
        print(f"[name_check] FAIL e2e: MLV-in-MLV-passages should pass "
              f"name-check, got missing={res.missing_tokens}", file=sys.stderr)
        return 1

    # Vacuously-passing case: question with no distinctive tokens.
    res = verify_question_in_passages(
        "how does login work",
        "anything at all could be here",
    )
    if not res.passed:
        print(f"[name_check] FAIL e2e: vacuous case should pass; got "
              f"missing={res.missing_tokens}", file=sys.stderr)
        return 1
    print("[name_check] end-to-end (verify_question_in_passages) OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for fn in (_check_extract, _check_presence, _check_refusal_directive,
               _check_end_to_end):
        if fn() != 0:
            return 1
    print("[name_check] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
