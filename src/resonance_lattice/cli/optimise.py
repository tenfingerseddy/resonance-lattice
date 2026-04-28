"""`rlat optimise <knowledge_model.rlat> [--estimate] [--yes]`

Adds the MRL optimised band in-place. Idempotent — running twice retrains
the same slot atomically.

Pre-flight `--estimate` (no LLM calls): computes USD + wall-time based on
passage count + unique source files. Without `--estimate`, prints the same
estimate and prompts before actually running unless `--yes` is passed.

Pipeline (base plan §4.3, orchestrated end-to-end):

  1. Load archive + base band + registry + passage texts (via Store).
  2. Synth queries via Claude Haiku 4.5 (synth_queries.generate).
  3. Encode synth queries with frozen gte-mb (Encoder).
  4. Hard-negative mining (mine_negatives.mine).
  5. Train MRL InfoNCE (train_mrl.train).
  6. Project + atomic in-place band write (write_slot.project_and_write).

Locked hyperparameters; no `--rerank` / `--cascade` / `--no-mrl` knobs.

Phase 4 deliverable. Base plan §4.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..field.encoder import Encoder
from ..optimise import device as device_mod
from ..optimise import mine_negatives, synth_queries, train_mrl, write_slot
from ..store import archive, open_store
from ._load import load_or_exit, open_store_or_exit


def _print_estimate(n_passages: int, n_files: int, dev: device_mod.Device) -> None:
    target = min(synth_queries.DEFAULT_TARGET_QUERIES,
                 n_files * synth_queries.DEFAULT_QUERIES_PER_FILE_CAP)
    cost = synth_queries.estimate_cost(target)
    n_queries = int(target * synth_queries.EXPECTED_RETENTION)
    minutes = device_mod.estimate_wall_time(dev, n_passages, n_queries)
    print("[optimise] pre-flight estimate", file=sys.stderr)
    print(f"  knowledge model     {n_passages} passages, {n_files} files", file=sys.stderr)
    print(f"  device              {dev}", file=sys.stderr)
    print(f"  LLM calls           ≈ {target + 1} (1 anchor + {target} per-query, "
          f"cap {synth_queries.DEFAULT_QUERIES_PER_FILE_CAP}/file)", file=sys.stderr)
    print(f"  cost                ≈ ${cost:.2f} USD (Sonnet 4.6)", file=sys.stderr)
    print(f"  training queries    ≈ {n_queries:,} (after retention)", file=sys.stderr)
    print(f"  wall time           ≈ {minutes:.0f} min "
          f"(plus LLM call time, 8-worker concurrent)", file=sys.stderr)


def _confirm_or_exit(prompt: str) -> bool:
    """Read a yes/no from stdin. Defaults to no on EOF / empty input —
    we'd rather a non-interactive caller pass `--yes` explicitly than
    silently spend $20 because they piped /dev/null at us."""
    print(prompt, end="", file=sys.stderr, flush=True)
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return False
    return line.strip().lower() in {"y", "yes"}


def cmd_optimise(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    if "base" not in contents.bands:
        print(f"error: {km_path} has no base band — `rlat build` first",
              file=sys.stderr)
        return 1

    n_passages = len(contents.registry)
    n_files = len(set(c.source_file for c in contents.registry))
    dev = device_mod.select()

    if args.estimate:
        _print_estimate(n_passages, n_files, dev)
        return 0

    _print_estimate(n_passages, n_files, dev)
    if not args.yes:
        if not _confirm_or_exit("[optimise] proceed? [y/N] "):
            print("[optimise] aborted (rerun with --yes to skip the prompt)",
                  file=sys.stderr)
            return 0

    # Load passage texts via the appropriate Store. `local` and `bundled`
    # both work — `bundled` is faster (no FS round-trip).
    store = open_store_or_exit(km_path, contents, args.source_root)
    print(f"[optimise] loading {n_passages} passage texts via "
          f"{contents.metadata.store_mode} store …", file=sys.stderr)
    passages_text: list[str] = []
    for c in contents.registry:
        passages_text.append(store.fetch(c.source_file, c.char_offset, c.char_length))
    passage_idxs = [c.passage_idx for c in contents.registry]
    source_files = [c.source_file for c in contents.registry]

    # Step 1 — synth queries (LLM). Cache lives under `~/.cache/rlat/optimise/`
    # by default rather than as a sibling to the .rlat — sibling-mode caches
    # have a habit of getting `git add .`-ed accidentally and the directory
    # can grow to hundreds of MB on a 40K-passage corpus.
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir
        else Path.home() / ".cache" / "rlat" / "optimise"
    )
    print(f"[optimise] synth queries → cache_dir={cache_dir}", file=sys.stderr)

    def _synth_progress(stage: str, done: int, total: int) -> None:
        if total:
            pct = 100.0 * done / total
            print(f"\r[optimise] {stage} {done}/{total} ({pct:.0f}%)",
                  end="", file=sys.stderr, flush=True)
        if done == total:
            print(file=sys.stderr)

    synth_result = synth_queries.generate(
        passage_idxs=passage_idxs,
        passages=passages_text,
        source_files=source_files,
        cache_dir=cache_dir,
        corpus_description=args.corpus_description,
        progress=_synth_progress,
    )
    print(f"[optimise] {len(synth_result.queries)} queries  "
          f"{synth_result.n_llm_calls} LLM calls  "
          f"${synth_result.cost_usd:.2f}", file=sys.stderr)
    # Surface low retention so the operator sees LLM format drift before
    # training also fails downstream. 50% is the alarm threshold —
    # well-behaved Haiku 4.5 retains >90% in dogfood runs.
    if synth_result.n_lines_emitted > 0:
        retention = synth_result.n_lines_kept / synth_result.n_lines_emitted
        if retention < 0.5:
            print(f"warning: low query retention ({retention:.0%} of LLM-emitted "
                  f"lines parsed cleanly). The model may be drifting on output "
                  f"format; inspect the cache or retry.", file=sys.stderr)
    if len(synth_result.queries) < train_mrl.BATCH:
        print(f"error: only {len(synth_result.queries)} synth queries produced; "
              f"need ≥{train_mrl.BATCH}. Re-run with a larger corpus or check "
              f"the cache for parse failures.", file=sys.stderr)
        return 1

    # Step 2 — encode synth queries with frozen gte-mb. Encoder.encode
    # already L2-normalises per-batch and returns one combined array; no
    # post-normalisation needed (Phase 4 simplify-3 found this redundancy
    # in build.py too — same fix).
    encoder = Encoder()
    print(f"[optimise] encoding {len(synth_result.queries)} synth queries "
          f"(runtime={encoder.runtime_name}) …", file=sys.stderr)
    queries_text = [q.query for q in synth_result.queries]
    query_pos_idx = np.array([q.passage_idx for q in synth_result.queries], dtype=np.int32)
    query_emb = encoder.encode(queries_text)

    # Step 3 — hard-negative mining
    base_band = contents.bands["base"]
    print(f"[optimise] mining hard negatives …", file=sys.stderr)
    negs = mine_negatives.mine(query_emb, base_band, query_pos_idx)

    # Step 4 — train MRL W
    print(f"[optimise] training MRL on {dev} ({train_mrl.STEPS} steps, "
          f"batch={train_mrl.BATCH}) …", file=sys.stderr)
    t0 = time.perf_counter()
    last_print = [0.0]

    def _train_progress(step: int, loss: float, best_loss: float) -> None:
        # Throttle to once per second so the progress doesn't spam stderr
        # when steps are fast.
        now = time.perf_counter()
        if now - last_print[0] < 1.0 and step != train_mrl.STEPS:
            return
        last_print[0] = now
        pct = 100.0 * step / train_mrl.STEPS
        print(f"\r[optimise] step {step}/{train_mrl.STEPS} ({pct:.0f}%)  "
              f"loss={loss:.4f}  best={best_loss:.4f}",
              end="", file=sys.stderr, flush=True)

    result = train_mrl.train(
        base_band=base_band,
        query_embeddings=query_emb,
        query_passage_idx=query_pos_idx,
        negatives=negs,
        device=dev,
        progress=_train_progress,
    )
    print(file=sys.stderr)
    train_minutes = (time.perf_counter() - t0) / 60.0
    print(f"[optimise] trained {result.steps_completed} steps in "
          f"{train_minutes:.1f} min  "
          f"final_loss={result.final_loss:.4f}  "
          f"dev_r1={result.final_dev_r1:.3f}", file=sys.stderr)
    if result.early_killed:
        if result.kill_reason == "dev_r1_below_threshold":
            print(f"error: training early-killed at step "
                  f"{train_mrl.EARLY_KILL_STEP} — dev R@1 was "
                  f"{result.final_dev_r1:.3f} < "
                  f"{train_mrl.EARLY_KILL_R1_THRESHOLD}. The synth queries are "
                  f"likely too off-distribution for this corpus, or the corpus "
                  f"is too small for MRL specialisation. Optimised band NOT "
                  f"written.", file=sys.stderr)
        else:  # "no_progress_saturation"
            print(f"error: training stopped — loss plateaued for "
                  f"{train_mrl.NO_PROGRESS_WINDOW} steps without crossing the "
                  f"saturation floor ({train_mrl.SATURATION_FLOOR}). The "
                  f"corpus may not benefit from MRL specialisation, or the "
                  f"synth queries lack sufficient variety. Optimised band "
                  f"NOT written.", file=sys.stderr)
        return train_mrl.EXIT_EARLY_KILLED

    # Step 5 — project + atomic in-place write
    print(f"[optimise] writing optimised band + W + ANN index to "
          f"{km_path} …", file=sys.stderr)
    write_slot.project_and_write(km_path, result.W, train_mrl.MRL_DIMS)

    # Verify by reloading
    contents2 = archive.read(km_path)
    sp_info = contents2.metadata.bands.get("optimised")
    if sp_info is None:
        print("error: optimised band missing after write", file=sys.stderr)
        return 1
    print(f"[optimise] done. optimised band: dim={sp_info.dim}  "
          f"N={sp_info.passage_count}  W={tuple(sp_info.w_shape)}",
          file=sys.stderr)
    print(f"[optimise] queries: {synth_result.cost_usd:.2f} USD  "
          f"training: {train_minutes:.1f} min  "
          f"total: ${synth_result.cost_usd:.2f}", file=sys.stderr)
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("optimise",
                       help="Add MRL optimised band to a knowledge model (opt-in)")
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument("--estimate", action="store_true",
                   help="Print pre-flight cost + duration estimate; don't run.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the cost-confirmation prompt.")
    p.add_argument("--source-root", default=None,
                   help="Override recorded source_root (local mode only).")
    p.add_argument("--cache-dir", default=None,
                   help="Synth-query cache directory "
                        "(default: ~/.cache/rlat/optimise/<corpus_fingerprint>/).")
    p.add_argument("--corpus-description", default="a knowledge model",
                   help="Short corpus description (e.g. 'Microsoft Fabric "
                        "documentation'). Drives the LLM's anchor + per-query "
                        "system prompt — the most common cross-corpus "
                        "replication failure mode is omitting this. Always pass it.")
    p.set_defaults(func=cmd_optimise)
