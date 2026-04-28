# Contributing to Resonance Lattice

`rlat` is a knowledge-model retrieval CLI. The project is BSL-1.1
licensed (source-available, commercial-use restricted during the
change-licence window — see [LICENSE.md](LICENSE.md) and
[`docs/user/FAQ.md`](docs/user/FAQ.md) for the licence FAQ).

This document covers how to set up a dev environment and what the
review cadence looks like.

## Dev setup

Python 3.12+ required.

```bash
git clone https://github.com/tenfingerseddy/resonance-lattice.git
cd resonance-lattice
python -m venv .venv && source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate                              # Windows
pip install -e ".[dev,bench]"
rlat install-encoder
```

The `[dev]` extra pulls `pytest`, `ruff`, and `pyright`. The `[bench]`
extra pulls the user-bench reproducibility harness. The `[build]` and
`[optimise]` extras pull `transformers + torch` (only needed for
`rlat build` / `rlat optimise`, not for read-side workflows).

Re-run `pip install -e .` after changing CLI argparse wiring or extras
— editable installs cache console-script entry points.

## Cadence

Every commit on `main` follows a four-step gate (per
[CLAUDE.md](CLAUDE.md)):

1. **simplify** — review the change for reuse, quality, and efficiency
   (the `simplify` skill bundles the three sub-reviews).
2. **codex-review-cycle** — independent code review by Codex CLI.
3. **harness** — `python -m tests.harness.runner --changed $(git diff --cached --name-only)`
   must be green.
4. **board** — every commit that ships a board-tracked deliverable
   marks its issue Done on the GitHub Project. If the commit ships
   nothing tracked, state `Board: no item` in the commit message.

## Pull requests

- Open a PR against `main`. Small focused PRs review faster than big
  bundled ones.
- Include the harness gate output in the PR description.
- Sign off using the `Co-Authored-By:` trailer if the change was
  pair-written with an LLM assistant.
- Doc edits land alongside code edits when the public surface or a
  measured number changes — `docs/internal/HONEST_CLAIMS.md` and
  `docs/user/BENCHMARKS.md` should never trail the code that produced
  the numbers they cite.

## Filing issues

Use the public issues page for bug reports, feature requests, and
general discussion. For security-related reports see
[SECURITY.md](SECURITY.md).

A good bug report includes:

- `rlat --version` output.
- The exact command(s) you ran.
- The full stack trace or unexpected output.
- A minimal reproduction (a small sample corpus or a one-line
  `rlat search` invocation that shows the issue).

## Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
Report conduct issues via the same channel as security advisories.
