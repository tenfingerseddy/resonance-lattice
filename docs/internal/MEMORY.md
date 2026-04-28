# Memory Subsystem — Internal Reference

The `memory/` package is the v2.0 LayeredMemory implementation. It pairs with the Store layer (build-time content corpora) but uses a different on-disk shape because memory is **append-only at runtime**, not built once and queried.

> Source-of-truth code: `src/resonance_lattice/memory/` + `src/resonance_lattice/cli/memory.py`.
> User-facing surface: the `rlat memory <sub>` section in [CLI.md](../user/CLI.md). (`docs/user/CORE_FEATURES.md` is a Phase 7 deliverable — when it lands, it will pull from this reference.)
> v0.11 reference: `legacy/v0.11.0:src/resonance_lattice/layered_memory.py` (built on the v0.11 `Lattice` class which doesn't exist in v2.0 — v2.0 ports the *idea*, not the implementation).

## Three tiers, three policies

| Tier | Purpose | Half-life | Cap | Promotion target |
|---|---|---|---|---|
| **working** | Session-local scratch | 1 day | 200 | (none — use `gc` to drop expired) |
| **episodic** | Per-session records | 14 days | 2,000 | semantic (via `consolidate`) |
| **semantic** | Consolidated knowledge | ∞ | 20,000 | (none — relevance-pruned only) |

Default tier weights at recall: `working=0.5, episodic=0.3, semantic=0.2`. Working dominates fresh recall (you usually want the just-said context); semantic is the stable reference. Override via `--tier-weights` (CLI) or `tier_weights=` (Python).

## On-disk shape

```
memory_root/
├── working.jsonl   # one MemoryEntry per line, embedding stripped
├── working.npy     # (N, 768) float32 embeddings, L2-normalised
├── episodic.jsonl  episodic.npy
└── semantic.jsonl  semantic.npy
```

JSONL+NPY pair, *not* a `.rlat` per tier. Reasoning:

- A `rlat memory add "..."` call shouldn't rebuild a chunker, write a ZIP, and run an ANN constructor. Append + atomic-replace is the right shape.
- At v2.0-typical memory sizes (≤ a few thousand entries total) exact dense cosine is sub-millisecond — ANN would be churn-cost without a query-latency win.
- `MemoryEntry` carries `recurrence_count` and `salience`, neither of which fit cleanly in the `PassageCoord` shape that the .rlat registry uses.

Atomic write: tmp + `os.replace` for both `.jsonl` and `.npy` siblings. A crash mid-write leaves the previous tier state intact. The `np.save(file_handle, ...)` form is used instead of `np.save(path, ...)` because the latter auto-appends `.npy` to paths that don't end in `.npy` — bit Phase 5 smoke once on `working.npy.tmp` becoming `working.npy.tmp.npy`.

## API surface

```python
class LayeredMemory:
    # `init` is a constructor-shaped classmethod that creates missing tier
    # files; `__init__` is the loader. Accepts `encoder=` for test injection.
    @classmethod
    def init(cls, memory_root) -> "LayeredMemory": ...
    def __init__(self, memory_root, encoder: Encoder | None = None): ...

    def add(self, text, tier="working", salience=1.0, source_id="", session=None) -> MemoryEntry: ...
    def add_many(self, texts, tier="working", source_id="", session=None) -> list[MemoryEntry]: ...
    def recall(self, query, top_k=10, tier_weights=None, tiers=None) -> list[(score, MemoryEntry)]: ...
    def tier_size(self, tier) -> int: ...
    def all_entries(self, tier) -> list[MemoryEntry]: ...
    def replace_tier(self, tier, entries, embeddings) -> None: ...
    def append_to_tier(self, tier, entry, embedding) -> None: ...   # consolidation's promote path

# free functions
retention.decay(score, age_seconds, half_life) -> float
retention.gc(memory, tier) -> int   # returns count removed
consolidation.consolidate(memory, recurrence_threshold=3, dup_threshold=0.92, session=None) -> int
primer.generate_memory_primer(memory_root, output, novelty_threshold=0.3) -> int   # char count
```

The encoder is lazy: not loaded until the first `add` or `recall`. Tests pass `encoder=` to `__init__` to inject a fake.

**Embedding staleness contract**: `recall` and `all_entries` mutate `entry.embedding` in place to re-attach the embedding from the tier's npy slice. The `.embedding` reference is valid only between that call and the next call that re-touches the same tier — long-lived API consumers should `entry.embedding.copy()` if they want to hold it across re-queries. Single-shot CLI usage is unaffected.

## Recall scoring

```
score(entry) = cosine(q_emb, entry.embedding) * tier_weight[entry.tier] * entry.salience
```

Cosine because both `q_emb` and stored embeddings are L2-normalised (encoder convention). Tier weight folds in so scores from different tiers are directly comparable when sorting. Salience is a per-entry user knob (`--salience` on `rlat memory add`) — the default is 1.0 and most users never touch it.

## Consolidation

Promotion rule: an episodic entry is promoted to semantic when its content is "stable" — measured as having ≥ `recurrence_threshold` (default 3) near-duplicates within episodic. Near-duplicate = `cosine ≥ dup_threshold` (default 0.92, conservative — prose paraphrases rarely cross this).

The promoted cluster collapses to a single semantic entry whose `recurrence_count` carries the cluster size. The episodic near-duplicates are dropped (they've graduated). Idempotent on stable input — re-running produces no further promotions because the episodic survivors don't form a cluster of size ≥ threshold.

`--session <id>` (CLI) / `session=` (Python) restricts both the **scan** and the **drop** to entries whose `session` field matches. Cross-session episodic entries stay untouched. Use this to consolidate one debug session's notes without affecting another's.

`source_id` is a free-form caller-supplied label (e.g. filename, audit reference, conversation tag). It rides along on every entry and survives consolidation (the cluster representative carries the source_id of the first cluster member). Used by the primer renderer to disambiguate entries with similar text.

## Retention / gc

```
effective_score = entry.salience * 0.5 ** (age_seconds / half_life)
```

Drops:
1. Entries with `effective_score < floor_score` (default `1e-3`).
2. Anything beyond the tier's `cap_count` (sorted by effective_score descending).

`semantic` has `half_life = inf` — pure cap-by-relevance, never decays. The cap (20K) is high enough that a long-running user shouldn't hit it for years.

## Memory primer

`generate_memory_primer(memory_root, output, novelty_threshold=0.3)` produces a markdown context block at `output` summarising the semantic + episodic tiers. Skips working (session-local — too noisy for a primer that's loaded at session start).

Strategy: synthetic centroid query (mean of semantic+episodic embeddings, L2-renorm), score each entry against it, drop anything below `novelty_threshold`, render top-N per tier into a markdown block. Free of LLM/encoder cost beyond the encoding already done at `add` time.

## CLI

Five subcommands under `rlat memory`:

```bash
rlat memory add "text"                       # append to working (default tier)
rlat memory add "text" --tier episodic --salience 1.5
rlat memory recall "query" --top-k 10
rlat memory recall "query" --format json
rlat memory consolidate                      # episodic → semantic
rlat memory consolidate --session sess-12 --recurrence-threshold 5
rlat memory primer -o .claude/memory-primer.md
rlat memory gc                               # all tiers
rlat memory gc --tier working --tier episodic
```

`--memory-root <dir>` overrides the default `./memory/` location (top-level argparse, before the subcommand).

## Cross-references

- [docs/user/CLI.md](../user/CLI.md) §"rlat memory" — command reference.
- v0.11 source: `legacy/v0.11.0:src/resonance_lattice/layered_memory.py` (single file, 492 lines).
