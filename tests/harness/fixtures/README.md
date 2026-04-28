# Harness fixtures (small, deterministic)

Small corpora (≤100 docs) used by the per-commit harness suites. Must build
fast (<5 s) so the harness runs cheaply.

Larger BEIR-style fixtures live under `../fixtures_large/` and are opt-in
(skipped by default in the per-commit selection).

Populated as suites land, starting Phase 1.
