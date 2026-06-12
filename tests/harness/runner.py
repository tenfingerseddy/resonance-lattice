"""Harness orchestrator.

Selects which suites to run for a given commit. Invoked from a pre-commit
hook (or `.claude/hooks/`):

  python -m tests.harness.runner --changed $(git diff --cached --name-only)
  python -m tests.harness.runner --all          # full sweep
  python -m tests.harness.runner --phase-gate   # adds benchmark_gate (a stub —
                                                # BEIR-5 floor reproduced
                                                # manually, see BENCHMARK_GATE.md)

Stub suites (no coverage yet) return SKIP (2) and are reported separately —
a green sweep lists exactly which suites actually ran vs skipped.

Selection rules:
  field layer + install pipeline (src/resonance_lattice/{field,install}/**) →
      parity + encoder_determinism + runtime_parity + property
  store layer (src/resonance_lattice/store/**) →
      parity + roundtrip + drift
  _anthropic (src/resonance_lattice/_anthropic.py) →
      memory_v22_api_key
  cli + docs/user/**.md →
      doc_examples + integration
  memory (src/resonance_lattice/memory/**) →
      memory_v21_* + memory_v22_* + doc_examples(memory)
  --phase-gate or --all →
      everything including benchmark_gate (BEIR-5)

"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

# Bootstrap src-layout onto sys.path so suites can `from resonance_lattice ...`
# without depending on the contributor having `pip install -e .` already.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Suite name → module path under tests.harness.*
SUITES: dict[str, str] = {
    "parity": "tests.harness.parity",
    "golden": "tests.harness.golden",
    "roundtrip": "tests.harness.roundtrip",
    "drift": "tests.harness.drift",
    "property": "tests.harness.property",
    "doc_examples": "tests.harness.doc_examples",
    "docs_truth": "tests.harness.docs_truth",
    "encoder_determinism": "tests.harness.encoder_determinism",
    "encoder_ragged_batch": "tests.harness.encoder_ragged_batch",
    "band_parity": "tests.harness.band_parity",
    "runtime_parity": "tests.harness.runtime_parity",
    "benchmark_gate": "tests.harness.benchmark_gate",
    "skill_context": "tests.harness.skill_context",
    "incremental_refresh": "tests.harness.incremental_refresh",
    "incremental_sync": "tests.harness.incremental_sync",
    "conversion": "tests.harness.conversion",
    "name_check": "tests.harness.name_check",
    "deep_search": "tests.harness.deep_search",
    "watch_loop": "tests.harness.watch_loop",
    "memory_v21_hook": "tests.harness.memory_v21_hook",
    "memory_v21_privacy": "tests.harness.memory_v21_privacy",
    "memory_v21_recall": "tests.harness.memory_v21_recall",
    "memory_v21_workspace_scope": "tests.harness.memory_v21_workspace_scope",
    "memory_v21_daemon": "tests.harness.memory_v21_daemon",
    "memory_v21_hook_inject": "tests.harness.memory_v21_hook_inject",
    "memory_v21_retention": "tests.harness.memory_v21_retention",
    "memory_v22_rerank": "tests.harness.memory_v22_rerank",
    "memory_v22_intent_classify": "tests.harness.memory_v22_intent_classify",
    "memory_v22_forget": "tests.harness.memory_v22_forget",
    "memory_v22_confidence": "tests.harness.memory_v22_confidence",
    "memory_v22_what_next": "tests.harness.memory_v22_what_next",
    "memory_v22_decompose": "tests.harness.memory_v22_decompose",
    "memory_v22_consolidation": "tests.harness.memory_v22_consolidation",
    "extract_events": "tests.harness.extract_events",
    "atomic_capture": "tests.harness.atomic_capture",
    "memory_v22_api_key": "tests.harness.memory_v22_api_key",
    "state_workspace": "tests.harness.state_workspace",
    "state_intent": "tests.harness.state_intent",
    "claim_outcome_log": "tests.harness.claim_outcome_log",
    "state_hooks": "tests.harness.state_hooks",
    "state_attribution": "tests.harness.state_attribution",
    "state_eval": "tests.harness.state_eval",
    "cli_intent_workspace": "tests.harness.cli_intent_workspace",
    "cli_intent_durable": "tests.harness.cli_intent_durable",
    "fabric_bootstrap": "tests.harness.fabric_bootstrap",
    "fabric_client": "tests.harness.fabric_client",
    "expertise_primer": "tests.harness.expertise_primer",
    "memory_dedup": "tests.harness.memory_dedup",
    "build_pipeline": "tests.harness.build_pipeline",
    "insight_layer": "tests.harness.insight_layer",
    "insight_lifecycle": "tests.harness.insight_lifecycle",
    "lens_io": "tests.harness.lens_io",
    "promotion": "tests.harness.promotion",
    "audit_trace_cli": "tests.harness.audit_trace_cli",
    "llm_free_retrieval": "tests.harness.llm_free_retrieval",
    "search_lens_cli": "tests.harness.search_lens_cli",
    "reverification": "tests.harness.reverification",
    "probe_weak_zone": "tests.harness.probe_weak_zone",
    "faithfulness": "tests.harness.faithfulness",
    "faithful_promotion": "tests.harness.faithful_promotion",
    "insight_attribution": "tests.harness.insight_attribution",
    "consolidate_insights": "tests.harness.consolidate_insights",
    "insight_loop_e2e": "tests.harness.insight_loop_e2e",
    "claim": "tests.harness.claim",
    "claim_io": "tests.harness.claim_io",
    "experience_claim_store": "tests.harness.experience_claim_store",
    "claim_lifecycle": "tests.harness.claim_lifecycle",
    "data_isolation": "tests.harness.data_isolation",
    "measure": "tests.harness.measure",
    "capture": "tests.harness.capture",
    "telemetry": "tests.harness.telemetry",
    "self_audit": "tests.harness.self_audit",
    "external_freshness": "tests.harness.external_freshness",
    "provenance_seed": "tests.harness.provenance_seed",
    "counters": "tests.harness.counters",
    "curator_signals": "tests.harness.curator_signals",
    "curator_gap": "tests.harness.curator_gap",
    "curator_decide": "tests.harness.curator_decide",
    "curator_author": "tests.harness.curator_author",
    "curator_external_fill": "tests.harness.curator_external_fill",
    "agent_fill": "tests.harness.agent_fill",
    "curator_reconcile": "tests.harness.curator_reconcile",
    "h1_gap_gate": "tests.harness.h1_gap_gate",
    "row_mode": "tests.harness.row_mode",
    "streaming": "tests.harness.streaming",
    "outcome_bench": "tests.harness.outcome_bench",
    "cli_smoke": "tests.harness.cli_smoke",
}


def select(changed: Iterable[str]) -> set[str]:
    """Return the set of suites that should run given the changed file set."""
    suites: set[str] = set()
    for path in changed:
        p = path.replace("\\", "/")
        if (p.startswith("src/resonance_lattice/field/")
                and not p.startswith("src/resonance_lattice/field/capture")
                and not p.startswith("src/resonance_lattice/field/counters")) \
                or p.startswith("src/resonance_lattice/install/"):
            # Encoder / runtime / dense / ann changes — the heavy parity suites.
            # The pure-python count + capture leaves don't touch the encoder, so
            # they're excluded here and select their own suites below.
            suites |= {"parity", "encoder_determinism", "encoder_ragged_batch",
                       "runtime_parity", "property"}
        if p.startswith("src/resonance_lattice/field/capture") or \
           p.startswith("src/resonance_lattice/field/__init__"):
            # The capture heart — observation fused into the retrieval primitive.
            # The persistence fold drains the same buffer, so guard it too.
            suites |= {"capture", "telemetry"}
        if p.startswith("src/resonance_lattice/field/counters"):
            # The Tier-0 closed-form count layer (no model) — its own suite only.
            suites |= {"counters"}
        if p.startswith("src/resonance_lattice/curator/"):
            # The curator head: its closed-form clauses (arm (b) signals), the
            # closed-form gap-candidate combiner, the decide tier over persisted
            # telemetry (CRITICAL_PATH §2), and the gap→author growth touch (§3).
            suites |= {"curator_signals", "curator_gap", "curator_decide",
                       "curator_author", "curator_external_fill", "curator_reconcile",
                       "agent_fill"}
        if p.startswith("src/resonance_lattice/store/self_audit"):
            # The corpus self-audit geometry primitives (contradiction candidates).
            suites |= {"self_audit"}
        if p.startswith("src/resonance_lattice/store/external_freshness"):
            # Re-fetch + re-judge an external fill vs the live world (the useful staleness).
            suites |= {"external_freshness"}
        if p.startswith("src/resonance_lattice/store/telemetry"):
            # The decide tier reads the telemetry member's contract, so a change
            # to the fold/seam must exercise it too.
            suites |= {"curator_decide"}
        if p.startswith("benchmarks/h1_gap_gate"):
            # The H1 D-gate scoring core (arm (b) vs arm (c) statistics).
            suites |= {"h1_gap_gate"}
        if p.startswith("src/resonance_lattice/store/streaming"):
            # The OOM-safe slicer serve: stream_topk / materialize_band /
            # SourceSnippets. Registered 2026-06-10 — the suite existed but
            # was never wired, so slicer-branch commits ran zero serve tests.
            suites |= {"streaming"}
        if p.startswith("benchmarks/outcome_bench"):
            # STEP-0 decompose/arm logic of the outcome harness (stubbed LLM).
            suites |= {"outcome_bench"}
        if p.startswith("src/resonance_lattice/store/"):
            # store/incremental.py is the delta-apply home for refresh + sync,
            # so any store/* change must exercise the incremental + conversion
            # suites. band_parity guards base-band selection in archive.py.
            # Insight layer (lensed knowledge) also lives under store/.
            suites |= {"parity", "roundtrip", "drift",
                       "incremental_refresh", "incremental_sync",
                       "band_parity", "conversion",
                       "insight_layer", "telemetry", "row_mode"}
        if p.startswith("src/resonance_lattice/store/insight") or \
           p.startswith("src/resonance_lattice/store/verified") or \
           p.startswith("src/resonance_lattice/store/corpus_claim_io"):
            # corpus_claim_io owns new_corpus_claim (lifecycle/promotion
            # touch) and the legacy migration (covered in insight_layer).
            # insight.py owns the provenance-tier trust seed.
            suites |= {"insight_layer", "insight_lifecycle", "provenance_seed"}
        if p.startswith("src/resonance_lattice/store/promotion"):
            # promotion threads the provenance tier + the caller-verified (client=None) landing path.
            suites |= {"provenance_seed", "faithful_promotion", "promotion", "agent_fill"}
        if p.startswith("src/resonance_lattice/store/insight_lifecycle"):
            suites |= {"insight_lifecycle", "consolidate_insights",
                       "claim_lifecycle", "insight_loop_e2e"}
        if p.startswith("src/resonance_lattice/state/claim_lifecycle"):
            suites |= {"claim_lifecycle", "insight_lifecycle",
                       "consolidate_insights", "promotion",
                       "faithful_promotion", "reverification",
                       "memory_v22_forget"}
        if p.startswith("src/resonance_lattice/store/insight_attribution"):
            suites |= {"insight_attribution", "consolidate_insights",
                       "insight_loop_e2e"}
        if p.startswith("src/resonance_lattice/cli/consolidate_insights") or \
           p.startswith("src/resonance_lattice/cli/_outcomes"):
            suites |= {"consolidate_insights", "claim_outcome_log",
                       "insight_loop_e2e"}
        if p.startswith("src/resonance_lattice/state/claim_io"):
            suites |= {"claim_io", "experience_claim_store"}
        elif p.startswith("src/resonance_lattice/state/claim"):
            suites |= {"claim"}
        if p.startswith("src/resonance_lattice/memory/claim_store"):
            suites |= {"experience_claim_store", "data_isolation"}
        if p.startswith("src/resonance_lattice/memory/store") or \
           p.startswith("src/resonance_lattice/memory/daemon") or \
           p.startswith("src/resonance_lattice/memory/redaction"):
            suites |= {"data_isolation"}
        if p.startswith("src/resonance_lattice/cli/maintain"):
            # maintain.cmd_refresh wires the drift cascade post-pass
            suites |= {"insight_lifecycle"}
        if p.startswith("src/resonance_lattice/lens/"):
            suites |= {"lens_io"}
        if p.startswith("src/resonance_lattice/store/compression_test") or \
           p.startswith("src/resonance_lattice/store/promotion"):
            suites |= {"promotion", "faithful_promotion"}
        if p.startswith("src/resonance_lattice/store/faithfulness") or \
           p.startswith("src/resonance_lattice/store/reverification") or \
           p.startswith("src/resonance_lattice/store/_llm"):
            suites |= {"faithfulness", "reverification", "faithful_promotion"}
        if p.startswith("src/resonance_lattice/store/audit") or \
           p.startswith("src/resonance_lattice/cli/audit") or \
           p.startswith("src/resonance_lattice/cli/trace") or \
           p.startswith("src/resonance_lattice/cli/lens"):
            suites |= {"audit_trace_cli", "llm_free_retrieval"}
        if p.startswith("src/resonance_lattice/cli/lens"):
            # The lens CLI write surface (set-trust) is covered by the
            # lens round-trip + search overlay suites.
            suites |= {"lens_io", "search_lens_cli"}
        if p.startswith("src/resonance_lattice/_anthropic"):
            suites |= {"memory_v22_api_key"}
        if p.startswith("src/resonance_lattice/build/"):
            # Pure-Python build/refresh pipeline; CLI + UDF both wrap it.
            # walker.py + pipeline.py carry the row-mode (slicer) build.
            suites |= {"build_pipeline", "incremental_refresh", "watch_loop",
                       "fabric_bootstrap", "row_mode"}
        if p.startswith("src/resonance_lattice/fabric/"):
            # UDF runtime helpers: bootstrap/search/embed + the slicer key-set
            # surface (slice_with_state) and its row-mode build dependency.
            suites |= {"fabric_bootstrap", "fabric_client", "row_mode"}
        if p.startswith("src/resonance_lattice/cli/build"):
            suites |= {"build_pipeline", "incremental_refresh"}
        if p.startswith("src/resonance_lattice/cli/maintain"):
            suites |= {"incremental_refresh", "incremental_sync"}
        if p.startswith("src/resonance_lattice/cli/convert"):
            suites |= {"conversion"}
        if p.startswith("src/resonance_lattice/cli/") or p.startswith("docs/user/"):
            suites |= {"doc_examples"}
        if p.startswith("src/resonance_lattice/cli/") or p.startswith("docs/site/"):
            # Behavioural docs claims (command surface, key requirements)
            # validated against the parser — drifted three audit cycles.
            suites |= {"docs_truth"}
        if p.startswith("src/resonance_lattice/cli/"):
            # Cheap whole-surface dispatch check: every command's --help +
            # init-project end-to-end (catches hand-built-Namespace drift).
            suites |= {"cli_smoke"}
        if p.startswith("src/resonance_lattice/cli/skill_context"):
            suites |= {"skill_context", "name_check"}
        if p.startswith("src/resonance_lattice/cli/_grounding"):
            suites |= {"skill_context"}
        if p.startswith("src/resonance_lattice/cli/_namecheck"):
            suites |= {"name_check", "skill_context"}
        if p.startswith("src/resonance_lattice/cli/search"):
            suites |= {"name_check", "doc_examples", "insight_layer",
                       "search_lens_cli", "telemetry"}
        if p.startswith("src/resonance_lattice/lens/"):
            suites |= {"lens_io", "search_lens_cli"}
        if p.startswith("src/resonance_lattice/deep_search/"):
            suites |= {"deep_search", "name_check"}
        if p.startswith("src/resonance_lattice/cli/deep_search"):
            suites |= {"deep_search", "doc_examples", "faithful_promotion"}
        if p.startswith("src/resonance_lattice/cli/watch"):
            suites |= {"watch_loop", "doc_examples"}
        if p.startswith("src/resonance_lattice/memory/"):
            # Flat-memory store + redactor + capture + recall layer is
            # gated by the memory_v21_* / memory_v22_* suites.
            suites |= {
                "doc_examples",
                "memory_v21_hook", "memory_v21_privacy",
                "memory_v21_recall", "memory_v21_workspace_scope",
                "memory_v21_daemon", "memory_v21_hook_inject",
                "memory_v21_retention", "memory_v22_rerank",
                "memory_v22_intent_classify",
                "memory_v22_forget",
                "memory_v22_confidence", "memory_v22_what_next",
                "memory_v22_decompose", "memory_v22_consolidation",
                "memory_v22_api_key",
                "extract_events",
                "atomic_capture",
                "insight_loop_e2e",
            }
        if p.startswith("src/resonance_lattice/cli/memory"):
            suites |= {"memory_v21_hook", "doc_examples"}
        if p.startswith("src/resonance_lattice/state/"):
            suites |= {
                "state_workspace", "state_intent", "claim_outcome_log",
                "state_hooks", "state_attribution", "state_eval",
                "cli_intent_workspace",
            }
        if p.startswith("src/resonance_lattice/state/measure"):
            suites |= {"measure"}
        if p.startswith("src/resonance_lattice/cli/intent") or \
           p.startswith("src/resonance_lattice/cli/workspace"):
            suites |= {"cli_intent_workspace", "cli_intent_durable",
                       "insight_loop_e2e"}
        if p.startswith("src/resonance_lattice/cli/probe"):
            suites |= {"probe_weak_zone"}
        if p.startswith("src/resonance_lattice/rql/"):
            suites |= {"property"}
        if p.startswith("src/resonance_lattice/fabric/"):
            suites |= {"fabric_bootstrap"}
        if p.startswith("src/resonance_lattice/expertise/") or \
           p.startswith("src/resonance_lattice/cli/expertise"):
            suites |= {"expertise_primer"}
        if p.startswith("src/resonance_lattice/memory/dedup") or \
           p.startswith("src/resonance_lattice/memory/capture"):
            suites |= {"memory_dedup"}
        if p.startswith("src/resonance_lattice/cli/_fabric") or \
           p.startswith("src/resonance_lattice/cli/fabric"):
            suites |= {"fabric_client"}
        if p.startswith("src/resonance_lattice/field/text"):
            # Sentence splitter is consumed by the chunker.
            suites |= {"roundtrip"}
    return suites


# Suite run() sentinel: a stub with no real coverage returns SKIP. The runner
# reports skips loudly and separately so a green sweep can't silently imply
# coverage that never ran (2026-06 review: 7 stubs were indistinguishable
# from passes).
SKIP = 2


def run(suites: Iterable[str]) -> int:
    """Run the named suites. Returns 0 on success, non-zero on failure."""
    import importlib

    failures: list[str] = []
    skipped: list[str] = []
    for suite in suites:
        module_path = SUITES.get(suite)
        if not module_path:
            print(f"[harness] unknown suite: {suite}", file=sys.stderr)
            failures.append(suite)
            continue
        mod = importlib.import_module(module_path)
        result = mod.run()  # each suite exposes run() -> int (SKIP == stub)
        if result == SKIP:
            skipped.append(suite)
        elif result != 0:
            failures.append(suite)
    if skipped:
        print(f"[harness] SKIPPED (stubs, no coverage): {', '.join(sorted(skipped))}",
              file=sys.stderr)
    if failures:
        print(f"[harness] FAILURES in: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tests.harness.runner")
    parser.add_argument("--changed", nargs="+", default=[],
                        help="changed file paths (typically from git diff --cached --name-only)")
    parser.add_argument("--all", action="store_true",
                        help="run every suite except benchmark_gate")
    parser.add_argument("--phase-gate", action="store_true",
                        help="run every suite including benchmark_gate "
                             "(NOTE: benchmark_gate is a stub — the BEIR-5 "
                             "floor is reproduced manually per "
                             "docs/internal/BENCHMARK_GATE.md, not here)")
    args = parser.parse_args(argv)

    if args.phase_gate:
        suites = set(SUITES.keys())
    elif args.all:
        suites = set(SUITES.keys()) - {"benchmark_gate"}
    elif args.changed:
        suites = select(args.changed)
    else:
        parser.error("must pass --changed, --all, or --phase-gate")

    if not suites:
        print("[harness] no suites selected for this change set", file=sys.stderr)
        return 0

    print(f"[harness] running suites: {sorted(suites)}", file=sys.stderr)
    return run(suites)


if __name__ == "__main__":
    sys.exit(main())
