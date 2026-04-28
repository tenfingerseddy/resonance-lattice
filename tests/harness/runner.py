"""Harness orchestrator.

Selects which suites to run for a given commit. Invoked from a pre-commit
hook (or `.claude/hooks/`):

  python -m tests.harness.runner --changed $(git diff --cached --name-only)
  python -m tests.harness.runner --all          # full sweep
  python -m tests.harness.runner --phase-gate   # exit-gate including benchmark_gate

Selection rules:
  field layer + install pipeline (src/resonance_lattice/{field,install}/**) →
      parity + encoder_determinism + runtime_parity + property
  store layer (src/resonance_lattice/store/**) →
      parity + roundtrip + drift
  optimise (src/resonance_lattice/optimise/**) →
      optimise_roundtrip + band_parity
  cli + docs/user/**.md →
      doc_examples + integration
  memory (src/resonance_lattice/memory/**) →
      memory_cycle + doc_examples(memory)
  --phase-gate or --all →
      everything including benchmark_gate (BEIR-5)

Phase 0 scaffold — selection logic in place, suites empty until phase fills.
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
    "memory_cycle": "tests.harness.memory_cycle",
    "encoder_determinism": "tests.harness.encoder_determinism",
    "optimise_roundtrip": "tests.harness.optimise_roundtrip",
    "band_parity": "tests.harness.band_parity",
    "runtime_parity": "tests.harness.runtime_parity",
    "benchmark_gate": "tests.harness.benchmark_gate",
    "skill_context": "tests.harness.skill_context",
    "incremental_refresh": "tests.harness.incremental_refresh",
    "incremental_sync": "tests.harness.incremental_sync",
    "optimised_reproject": "tests.harness.optimised_reproject",
    "conversion": "tests.harness.conversion",
    "name_check": "tests.harness.name_check",
    "deep_search": "tests.harness.deep_search",
}


def select(changed: Iterable[str]) -> set[str]:
    """Return the set of suites that should run given the changed file set."""
    suites: set[str] = set()
    for path in changed:
        p = path.replace("\\", "/")
        if p.startswith("src/resonance_lattice/field/") or p.startswith("src/resonance_lattice/install/"):
            suites |= {"parity", "encoder_determinism", "runtime_parity", "property"}
        if p.startswith("src/resonance_lattice/store/"):
            # store/incremental.py is the re-projection home for refresh +
            # sync, so any store/* change must exercise optimised_reproject
            # alongside the basic delta-apply + conversion suites.
            suites |= {"parity", "roundtrip", "drift",
                       "incremental_refresh", "incremental_sync",
                       "optimised_reproject", "conversion"}
        if p.startswith("src/resonance_lattice/optimise/"):
            suites |= {"optimise_roundtrip", "band_parity", "optimised_reproject"}
        if p.startswith("src/resonance_lattice/cli/maintain"):
            suites |= {"incremental_refresh", "incremental_sync", "optimised_reproject"}
        if p.startswith("src/resonance_lattice/cli/convert"):
            suites |= {"conversion"}
        if p.startswith("src/resonance_lattice/cli/") or p.startswith("docs/user/"):
            suites |= {"doc_examples"}
        if p.startswith("src/resonance_lattice/cli/skill_context"):
            suites |= {"skill_context", "name_check"}
        if p.startswith("src/resonance_lattice/cli/_grounding"):
            suites |= {"skill_context"}
        if p.startswith("src/resonance_lattice/cli/_namecheck"):
            suites |= {"name_check", "skill_context"}
        if p.startswith("src/resonance_lattice/cli/search"):
            suites |= {"name_check", "doc_examples"}
        if p.startswith("src/resonance_lattice/deep_search/"):
            suites |= {"deep_search", "name_check"}
        if p.startswith("src/resonance_lattice/cli/deep_search"):
            suites |= {"deep_search", "doc_examples"}
        if p.startswith("src/resonance_lattice/memory/"):
            suites |= {"memory_cycle", "doc_examples"}
        if p.startswith("src/resonance_lattice/rql/"):
            suites |= {"property"}
    return suites


def run(suites: Iterable[str]) -> int:
    """Run the named suites. Returns 0 on success, non-zero on failure."""
    import importlib

    failures: list[str] = []
    for suite in suites:
        module_path = SUITES.get(suite)
        if not module_path:
            print(f"[harness] unknown suite: {suite}", file=sys.stderr)
            failures.append(suite)
            continue
        mod = importlib.import_module(module_path)
        result = mod.run()  # each suite exposes run() -> int
        if result != 0:
            failures.append(suite)
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
                        help="run every suite including benchmark_gate (slow)")
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
