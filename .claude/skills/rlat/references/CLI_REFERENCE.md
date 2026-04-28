# CLI reference

The canonical reference is [docs/user/CLI.md](../../../../docs/user/CLI.md) — every command, every flag, current to v2.0.

The 16 SHIP commands at v2.0:

| Command | One-line |
|---|---|
| `rlat install-encoder` | One-time HF download → ONNX export → optional OpenVINO conversion |
| `rlat init-project` | Auto-detect sources + build + write primer in one command |
| `rlat build` | Build a knowledge model from source dirs |
| `rlat search` | Top-k retrieval (text / json / context formats) |
| `rlat deep-search` | Multi-hop research loop (plan → retrieve → refine → synth); requires API key |
| `rlat profile` | Knowledge-model semantic profile + drift report |
| `rlat compare` | Cross-knowledge-model comparison (always base band) |
| `rlat summary` | Generate `.claude/resonance-context.md` primer |
| `rlat refresh` | Incremental delta-apply (local mode) |
| `rlat sync` | Incremental delta-apply (remote mode) |
| `rlat freshness` | Read-only drift check (remote mode) |
| `rlat optimise` | Add MRL optimised band to a knowledge model (opt-in; requires API key) |
| `rlat memory` | Layered memory: add / recall / consolidate / primer / gc |
| `rlat skill-context` | Markdown context block for Anthropic-skill `!command` injection |
| `rlat convert` | Switch a knowledge model between storage modes (no rebuild) |

Memory subcommand syntax: `rlat memory --memory-root <PATH> {add|recall|consolidate|primer|gc} [args]`. The `--memory-root` flag goes on the parent `memory` command, not on the subcommands.

Three modes apply to `rlat search --format context` and `rlat skill-context`:

- `--mode augment` (default) — passages as primary context, blended with training.
- `--mode constrain` — passages are the only source of truth; refuse on thin evidence.
- `--mode knowledge` — passages supplement training.

Add `--strict-names` when proper-noun fidelity matters (the namecheck distinctive-token gate refuses on name-aliasing distractor failures).

For full per-flag behaviour and recipes, read [docs/user/CLI.md](../../../../docs/user/CLI.md).
