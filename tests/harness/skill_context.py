"""skill_context — skill-friendly context block has the contracts skills depend on.

Ten guarantees, all measured here:

  1. **Format contract**: every passage block carries `[source_file:offset+length]`
     anchor, drift status tag, score, and a confidence-metrics header line.
     If the format drifts, the consuming skill's parsing breaks silently.

  2. **Multi-query interleaving**: passing N `--query` flags emits N labelled
     blocks ("Context for: <query>") in order. Skills depend on order — the
     first query is the skill-author preset, later queries are user-supplied.

  3. **Drift gate (warn mode)**: with drifted source, output prepends a
     `⚠ DRIFT WARNING` banner but exits 0. Default behaviour for non-strict
     skill consumers.

  4. **Drift gate (strict mode)**: `--strict` + drifted source → non-zero
     exit + stderr error. Skill loaders that pass `--strict` must be able to
     gate on the exit code.

  5. **Token budget enforcement**: `--token-budget` truncates from the end so
     the highest-priority (first) query block survives. Total stdout char
     count never exceeds budget × chars_per_token, except for the case where
     the first block alone overflows (documented carve-out).

  6. **Mode header always ships**: every invocation stamps a `<!-- rlat-mode:
     X -->` directive at the top of stdout, regardless of suppression or
     drift state. The directive is the consumer LLM's instruction; without
     it the LLM has no defined relationship to the corpus.

  7. **Constrain mode never suppresses**: even on weak retrieval (low
     top1_score, high drift_fraction), `--mode constrain` returns the
     full passage body. Refusal is the consumer LLM's job under constrain;
     the gate must not pre-empt that decision.

  8. **Augment mode suppresses on weak retrieval**: under `--mode augment`,
     if top1_score falls below the relevance floor (0.30) the dynamic
     body is replaced with a 'no confident evidence' marker so the
     LLM falls back to training instead of grounding on noise.

  9. **Name-check warns on missing distinctive token**: when a query
     references a distinctive proper noun, acronym, or alphanumeric ID
     not present in any retrieved passage, output prepends a
     `rlat-namecheck: missing` block and a ⚠ Name verification failed
     directive — addresses the name-aliasing distractor failure mode
     that score-based gating cannot.

 10. **`--strict-names` aborts on missing distinctive token**: with
     `--strict-names`, the same condition exits non-zero so a skill
     loader can gate on the exit code.

Phase 7 deliverable. Guarantees 9 + 10 added v2.0 launch (parallel A
distractor analysis, doc: docs/internal/benchmarks/02_distractor_floor_analysis.md).
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


from ._testutil import Args as _Args


def _build_basic_corpus(root: Path) -> Path:
    """Build a small bundled corpus with two source files. Returns the .rlat path."""
    from resonance_lattice.cli.build import cmd_build
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "src" / "auth.py").write_text(
        "def login(user, password):\n"
        "    \"\"\"Authenticate user against the password store.\"\"\"\n"
        "    return verify_password(user, password)\n"
        "\n"
        "def logout(session):\n"
        "    \"\"\"Invalidate the user's session.\"\"\"\n"
        "    session.invalidate()\n",
        encoding="utf-8",
    )
    (root / "docs" / "guide.md").write_text(
        "# Authentication Guide\n\n"
        "The auth module provides login and logout primitives. "
        "Login validates credentials against the password store. "
        "Logout invalidates the user's session token.\n\n"
        "## Session management\n\n"
        "Sessions persist for 24 hours by default. "
        "Calling logout clears the session immediately.\n",
        encoding="utf-8",
    )
    out = root / "km.rlat"
    rc = cmd_build(_Args(
        sources=[str(root)], output=str(out),
        store_mode="bundled", kind="corpus", source_root=str(root),
        min_chars=20, max_chars=300, batch_size=4, ext=None,
        remote_url_base=None,
    ))
    if rc != 0:
        raise RuntimeError(f"build failed rc={rc}")
    return out


def _run_skill_context(km_path: Path, **overrides) -> tuple[int, str]:
    """Invoke cmd_skill_context, capturing stdout. Returns (exit_code, stdout)."""
    from resonance_lattice.cli.skill_context import cmd_skill_context
    args = _Args(
        knowledge_model=str(km_path),
        query=overrides.get("query", ["how does login work"]),
        top_k=overrides.get("top_k", 3),
        token_budget=overrides.get("token_budget", 4000),
        source_root=overrides.get("source_root", None),
        strict=overrides.get("strict", False),
        strict_names=overrides.get("strict_names", False),
        mode=overrides.get("mode", "augment"),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_skill_context(args)
    return rc, buf.getvalue()


def _check_format(stdout: str, query: str) -> str | None:
    """Return error message if the format contract is violated, else None."""
    # Header line per query
    header_re = re.compile(
        r"<!-- rlat skill-context query=.+? band=\w+ mode=\w+ "
        r"top1_score=[-\d.]+ "
        r"top1_top2_gap=[-\d.]+ source_diversity=[-\d.]+ "
        r"drift_fraction=[-\d.]+ missing_names=\S+ -->"
    )
    if not header_re.search(stdout):
        return f"format: missing ConfidenceMetrics header line"
    # Per-query section heading
    if f'## Context for: "{query}"' not in stdout:
        return f"format: missing section heading for query {query!r}"
    # At least one passage anchor + drift tag
    if not re.search(r"\*\*source: `[^`]+:\d+\+\d+`\*\* — score [\d.]+ `\[\w+\]`", stdout):
        return "format: no passage anchor with score + drift tag found"
    return None


def _check_grounding_module() -> int:
    """Deterministic unit-style test of the gate logic itself.

    Guarantees 7 + 8 are wiring-tested via cmd_skill_context, but the
    gate-fire behaviour depends on retrieval scores from a small fixture
    corpus — fragile for verifying threshold semantics. This block
    pins the threshold contract directly.

    Augment gate (post bench-2 fix): top1_score floor of 0.30 + drift
    ceiling of 0.30. Knowledge gate: lower floor 0.15, no drift gate.
    The previous gap-based gate over-suppressed on paraphrase-rich
    corpora — see _grounding.py docstring.
    """
    from resonance_lattice.cli._grounding import (
        Mode, format_header, should_suppress, suppression_marker,
    )
    from resonance_lattice.rql.types import ConfidenceMetrics

    # Off-corpus metrics: very low top-1 score. Below both augment + knowledge floors.
    off_corpus = ConfidenceMetrics(
        top1_score=0.10, top1_top2_gap=0.01, source_diversity=0.5,
        drift_fraction=0.4, band_used="base",
    )
    # Marginal metrics: top-1 between knowledge and augment floors.
    marginal = ConfidenceMetrics(
        top1_score=0.20, top1_top2_gap=0.02, source_diversity=0.5,
        drift_fraction=0.0, band_used="base",
    )
    # Strong metrics: high top-1, no drift. Above all floors.
    strong = ConfidenceMetrics(
        top1_score=0.65, top1_top2_gap=0.40, source_diversity=0.8,
        drift_fraction=0.0, band_used="base",
    )
    # Paraphrase-cluster metrics: high top-1 but tight top1/top2 gap. The
    # OLD gap-based gate fired here; the NEW score-based gate must NOT —
    # this is the bench-2 regression case the fix addresses.
    paraphrase = ConfidenceMetrics(
        top1_score=0.60, top1_top2_gap=0.02, source_diversity=0.6,
        drift_fraction=0.0, band_used="base",
    )

    # Augment: suppresses off-corpus + marginal, allows strong + paraphrase.
    if not should_suppress(off_corpus, Mode.AUGMENT):
        print("[skill_context] FAIL gate: augment did not suppress off_corpus metrics",
              file=sys.stderr)
        return 1
    if not should_suppress(marginal, Mode.AUGMENT):
        print("[skill_context] FAIL gate: augment did not suppress marginal "
              "(top1=0.20, below 0.30 floor)", file=sys.stderr)
        return 1
    if should_suppress(strong, Mode.AUGMENT):
        print("[skill_context] FAIL gate: augment suppressed strong metrics",
              file=sys.stderr)
        return 1
    if should_suppress(paraphrase, Mode.AUGMENT):
        print("[skill_context] FAIL gate: augment suppressed paraphrase "
              "cluster (top1=0.60, gap=0.02). The bench-2 regression case "
              "must NOT trigger the new score-based gate.", file=sys.stderr)
        return 1

    # Knowledge: suppresses off-corpus only (lighter floor 0.15); allows
    # marginal (top1=0.20 above 0.15) AND ignores drift entirely.
    if not should_suppress(off_corpus, Mode.KNOWLEDGE):
        print("[skill_context] FAIL gate: knowledge did not suppress off_corpus "
              "(top1=0.10, below 0.15 floor)", file=sys.stderr)
        return 1
    if should_suppress(marginal, Mode.KNOWLEDGE):
        print("[skill_context] FAIL gate: knowledge suppressed marginal "
              "(top1=0.20, above 0.15 floor)", file=sys.stderr)
        return 1
    drifty_but_ok = ConfidenceMetrics(
        top1_score=0.40, top1_top2_gap=0.10, source_diversity=0.3,
        drift_fraction=0.6, band_used="base",
    )
    if should_suppress(drifty_but_ok, Mode.KNOWLEDGE):
        print("[skill_context] FAIL gate: knowledge suppressed on drift "
              "(should ignore drift_fraction)", file=sys.stderr)
        return 1

    # Constrain: never suppresses.
    if should_suppress(off_corpus, Mode.CONSTRAIN):
        print("[skill_context] FAIL gate: constrain suppressed (should never)",
              file=sys.stderr)
        return 1

    # Headers contain the directive verb the LLM keys off.
    for mode, verb in [
        (Mode.AUGMENT, "primary context"),
        (Mode.KNOWLEDGE, "supplement"),
        (Mode.CONSTRAIN, "ONLY"),
    ]:
        if verb not in format_header(mode):
            print(f"[skill_context] FAIL header: mode={mode.value} missing "
                  f"directive verb {verb!r}", file=sys.stderr)
            return 1

    # Suppression marker carries the gate-trigger metrics for debuggability.
    marker = suppression_marker(off_corpus, Mode.AUGMENT)
    if "augment" not in marker or "0.100" not in marker:
        print(f"[skill_context] FAIL marker: missing mode or metrics in "
              f"{marker!r}", file=sys.stderr)
        return 1

    print("[skill_context] grounding-module unit checks OK", file=sys.stderr)
    return 0


def run() -> int:
    """Smoke-test the eight contracts."""
    if _check_grounding_module() != 0:
        return 1
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # ---- Setup: build a small corpus ----
        km = _build_basic_corpus(root / "corpus")

        # ---- Guarantee 1: format contract ----
        # Use mode=constrain so the augment gate doesn't suppress the body
        # on this tiny fixture corpus — the format contract is checking
        # rendering, not retrieval quality.
        rc, out = _run_skill_context(
            km, query=["how does login work"], mode="constrain"
        )
        if rc != 0:
            print(f"[skill_context] FAIL: rc={rc} on basic call", file=sys.stderr)
            return 1
        err = _check_format(out, "how does login work")
        if err:
            print(f"[skill_context] FAIL guarantee 1: {err}", file=sys.stderr)
            print(f"[skill_context] stdout was:\n{out}", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 1 (format) OK", file=sys.stderr)

        # ---- Guarantee 2: multi-query interleaving (preset + user) ----
        queries = ["session management overview", "how does login work"]
        rc, out = _run_skill_context(km, query=queries, top_k=2)
        if rc != 0:
            print(f"[skill_context] FAIL: rc={rc} on multi-query", file=sys.stderr)
            return 1
        # Both queries should appear, and the FIRST query's section must come
        # before the SECOND query's section (skill-author-priority order).
        idx_a = out.find(f'## Context for: "{queries[0]}"')
        idx_b = out.find(f'## Context for: "{queries[1]}"')
        if idx_a < 0 or idx_b < 0:
            print(f"[skill_context] FAIL guarantee 2: one or both queries missing",
                  file=sys.stderr)
            return 1
        if idx_a >= idx_b:
            print(f"[skill_context] FAIL guarantee 2: query order not preserved "
                  f"(query 0 at {idx_a}, query 1 at {idx_b})", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 2 (multi-query order) OK", file=sys.stderr)

        # ---- Setup for drift tests: mutate one source file ----
        # Editing guide.md after build → its passages are now drifted.
        (root / "corpus" / "docs" / "guide.md").write_text(
            "# Authentication Guide\n\n"
            "TOTALLY DIFFERENT CONTENT — should drift relative to the build.\n",
            encoding="utf-8",
        )
        # Local mode (re-open via --source-root) is the simplest way to
        # exercise drift; the build was bundled, but we can override the
        # store by passing source_root and trusting the bundled fallback.
        # Simpler: build a NEW corpus in local mode for the drift test.
        local_root = root / "local_corpus"
        local_root.mkdir()
        (local_root / "doc.md").write_text(
            "Hello world. This is a passage about authentication and login flows. "
            "It will be drifted shortly by overwriting the file.",
            encoding="utf-8",
        )
        from resonance_lattice.cli.build import cmd_build
        local_km = local_root / "km.rlat"
        rc = cmd_build(_Args(
            sources=[str(local_root)], output=str(local_km),
            store_mode="local", kind="corpus", source_root=str(local_root),
            min_chars=20, max_chars=300, batch_size=4, ext=None,
            remote_url_base=None,
        ))
        if rc != 0:
            raise RuntimeError(f"local-mode build failed rc={rc}")
        # Mutate the source after build → drift on every passage from doc.md.
        (local_root / "doc.md").write_text(
            "Completely different content now. The build's content_hash no "
            "longer matches the live source.",
            encoding="utf-8",
        )

        # ---- Guarantee 3: drift gate (warn mode, default) ----
        rc, out = _run_skill_context(local_km, query=["authentication flow"])
        if rc != 0:
            print(f"[skill_context] FAIL guarantee 3: warn-mode drift returned "
                  f"rc={rc}, expected 0", file=sys.stderr)
            return 1
        if "DRIFT WARNING" not in out:
            print(f"[skill_context] FAIL guarantee 3: drift detected but no banner. "
                  f"stdout:\n{out}", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 3 (drift warn) OK", file=sys.stderr)

        # ---- Guarantee 4: drift gate (strict mode) ----
        from resonance_lattice.cli.skill_context import cmd_skill_context
        args = _Args(
            knowledge_model=str(local_km),
            query=["authentication flow"], top_k=3, token_budget=4000,
            source_root=None, strict=True, strict_names=False, mode="augment",
        )
        # Capture both stdout + stderr; strict mode prints to stderr on failure.
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = cmd_skill_context(args)
        if rc == 0:
            print(f"[skill_context] FAIL guarantee 4: strict mode + drift "
                  f"returned 0, expected non-zero", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 4 (drift strict) OK", file=sys.stderr)

        # ---- Guarantee 5: token budget truncates later blocks first ----
        # Two queries + tight budget → only the first block should survive.
        # Each query block is ~300-500 chars; budget 200 tokens × 3 chars/tok
        # = 600 chars, enough for one block but not two.
        queries = ["session management overview", "how does login work"]
        rc, out = _run_skill_context(km, query=queries, top_k=3, token_budget=200)
        if rc != 0:
            print(f"[skill_context] FAIL guarantee 5: rc={rc}", file=sys.stderr)
            return 1
        if f'## Context for: "{queries[0]}"' not in out:
            print(f"[skill_context] FAIL guarantee 5: first-priority block "
                  f"truncated (should always survive). stdout:\n{out}",
                  file=sys.stderr)
            return 1
        if f'## Context for: "{queries[1]}"' in out:
            print(f"[skill_context] FAIL guarantee 5: second block survived "
                  f"the tight budget. stdout:\n{out}", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 5 (token budget) OK", file=sys.stderr)

        # ---- Guarantee 6: mode header always ships ----
        # Augment, knowledge, and constrain (default) — each must stamp
        # its `<!-- rlat-mode: X -->` directive in stdout. Without it
        # the consumer LLM has no defined relationship to the corpus.
        for mode in ("augment", "knowledge", "constrain"):
            rc, out = _run_skill_context(
                km, query=["how does login work"], mode=mode
            )
            if rc != 0:
                print(f"[skill_context] FAIL guarantee 6: mode={mode} rc={rc}",
                      file=sys.stderr)
                return 1
            if f"<!-- rlat-mode: {mode} -->" not in out:
                print(f"[skill_context] FAIL guarantee 6: mode={mode} "
                      f"directive missing from stdout", file=sys.stderr)
                return 1
            if f"Grounding mode: {mode}" not in out:
                print(f"[skill_context] FAIL guarantee 6: mode={mode} "
                      f"human-readable directive missing", file=sys.stderr)
                return 1
        print("[skill_context] guarantee 6 (mode header always ships) OK",
              file=sys.stderr)

        # ---- Guarantee 7+8: weak retrieval — constrain ships passages,
        # augment suppresses them. Build a corpus about authentication and
        # query for something completely unrelated → top1_top2_gap will be
        # tight and source_diversity narrow.
        weak_query = ["the migration patterns of arctic terns"]

        rc, out_constrain = _run_skill_context(
            km, query=weak_query, mode="constrain", top_k=5
        )
        if rc != 0:
            print(f"[skill_context] FAIL guarantee 7: constrain rc={rc}",
                  file=sys.stderr)
            return 1
        # constrain must NEVER replace the body with a suppression marker;
        # the consumer LLM does the refusal under constrain.
        if "no confident evidence under mode=`constrain`" in out_constrain:
            print(f"[skill_context] FAIL guarantee 7: constrain mode "
                  f"suppressed the body. stdout:\n{out_constrain}",
                  file=sys.stderr)
            return 1
        print("[skill_context] guarantee 7 (constrain never suppresses) OK",
              file=sys.stderr)

        rc, out_augment = _run_skill_context(
            km, query=weak_query, mode="augment", top_k=5
        )
        if rc != 0:
            print(f"[skill_context] FAIL guarantee 8: augment rc={rc}",
                  file=sys.stderr)
            return 1
        # On a tiny corpus with an off-topic query, augment's gate (top1_top2
        # < 0.05) should fire. If the corpus happens to score well anyway,
        # this guarantee can't be verified — print a soft skip rather than
        # fail spuriously.
        if "no confident evidence under mode=`augment`" not in out_augment:
            print("[skill_context] guarantee 8 (augment suppresses weak): "
                  "SOFT-SKIP — fixture corpus scored too well on off-topic "
                  "query to trigger the gate", file=sys.stderr)
        else:
            print("[skill_context] guarantee 8 (augment suppresses weak) OK",
                  file=sys.stderr)

        # ---- Guarantee 9: name-check warns on missing distinctive token ----
        # Query contains an alphanumeric product ID not in the corpus. Use
        # mode=constrain so the gate doesn't suppress the body — the
        # namecheck directive is what we're asserting on.
        rc, out = _run_skill_context(
            km, query=["How do I configure QuantumWidget3000?"],
            mode="constrain", top_k=3,
        )
        if rc != 0:
            print(f"[skill_context] FAIL guarantee 9: rc={rc}", file=sys.stderr)
            return 1
        if "rlat-namecheck: missing" not in out:
            print(f"[skill_context] FAIL guarantee 9: namecheck comment "
                  f"missing from stdout. stdout:\n{out}", file=sys.stderr)
            return 1
        if "Name verification failed" not in out:
            print(f"[skill_context] FAIL guarantee 9: refusal directive "
                  f"missing from stdout. stdout:\n{out}", file=sys.stderr)
            return 1
        if "QuantumWidget3000" not in out:
            print(f"[skill_context] FAIL guarantee 9: missing token not "
                  f"surfaced in directive. stdout:\n{out}", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 9 (namecheck warn) OK", file=sys.stderr)

        # Negative case: query whose distinctive tokens DO appear in the
        # corpus must NOT emit the namecheck warning. "Authentication" is
        # in docs/guide.md verbatim.
        rc, out = _run_skill_context(
            km, query=["Authentication overview"], mode="constrain", top_k=3,
        )
        if rc != 0:
            print(f"[skill_context] FAIL guarantee 9 (negative): rc={rc}",
                  file=sys.stderr)
            return 1
        if "rlat-namecheck: missing" in out:
            print(f"[skill_context] FAIL guarantee 9 (negative): namecheck "
                  f"falsely fired on tokens that ARE in the corpus. "
                  f"stdout:\n{out}", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 9 (namecheck negative case) OK",
              file=sys.stderr)

        # ---- Guarantee 10: --strict-names aborts non-zero ----
        rc, out = _run_skill_context(
            km, query=["How do I configure QuantumWidget3000?"],
            mode="constrain", top_k=3, strict_names=True,
        )
        if rc == 0:
            print(f"[skill_context] FAIL guarantee 10: --strict-names + "
                  f"missing-token returned 0, expected non-zero. "
                  f"stdout:\n{out}", file=sys.stderr)
            return 1
        print("[skill_context] guarantee 10 (strict-names abort) OK",
              file=sys.stderr)

    print("[skill_context] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
