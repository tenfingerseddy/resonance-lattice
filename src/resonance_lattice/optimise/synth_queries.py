"""LLM-driven synth query generation for MRL optimised training.

Spec (mrl_fabric_remote_train.py:266-369 — original-positive-results path):

  1. **Style anchors**: ONE LLM call. 15 randomly-sampled passages + a
     `corpus_description` string ("Microsoft Fabric documentation", etc.)
     → 5 corpus-aware natural-language query anchors. Cached per-corpus.

  2. **Synth queries**: ONE LLM call PER QUERY (not per file, not per
     passage). 8 concurrent workers via ThreadPoolExecutor. Stratification
     caps queries-per-source-file at 3 with seed=0 shuffle, then targets
     ~6000 total queries.

  3. **Filters**: drop queries with len<10, len>400, or "passage" in the
     first 40 chars (LLM meta-phrasing).

API key discovery (base plan §6.1):
  RLAT_LLM_API_KEY_ENV → CLAUDE_API → ANTHROPIC_API_KEY → kaggle_secrets

File-level disk cache: each kept query is appended to a per-corpus
`<cache_dir>/<corpus_fingerprint>/queries.jsonl`. Mid-run crashes resume
from the cache without re-charging the API.

Phase 4 deliverable. Phase C spec-compliance fixed 2026-04-26 after v3/v4/v5
probes showed full regression caused by per-file (vs per-query) batching,
no stratification, no filters, and no corpus_description on the anchor call.
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

# Sonnet 4.6 — memory `project_mrl_optimised_encoder.md` warns: "Do NOT
# use claude-haiku for query gen without re-measuring — Sonnet 4.6 is what
# produced the viable result." Pinned for reproducibility.
from .._pricing import SONNET_MODEL as DEFAULT_MODEL, cost_usd as _cost_usd  # noqa: F401

DEFAULT_N_ANCHORS = 5
DEFAULT_ANCHOR_SAMPLE = 15
# Spec §4c: stratify cap-3-queries-per-source-file. Big files don't dominate
# the synth distribution; small files still get representation.
DEFAULT_QUERIES_PER_FILE_CAP = 3
DEFAULT_TARGET_QUERIES = 6000
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_TOKENS_ANCHOR = 1024
DEFAULT_MAX_TOKENS_SYNTH = 200

# Spec §4c filters: reject too-short, too-long, and meta-phrased queries.
MIN_QUERY_LEN = 10
MAX_QUERY_LEN = 400
META_REJECT_WORD = "passage"
META_REJECT_PREFIX_LEN = 40

EXPECTED_RETENTION = 0.85

# Bumped whenever the cache-key inputs or the on-disk JSONL schema change.
# A bump invalidates every existing cache directory (different fingerprint)
# without needing per-user manual rm — old caches simply become orphaned.
# Bump on: prompt template edit, model swap, fingerprint-input change,
# JSONL schema change.
CACHE_SCHEMA_VERSION = 2


@dataclass
class SynthQuery:
    query: str
    passage_idx: int


@dataclass
class SynthQueryResult:
    queries: list[SynthQuery]
    style_anchors: list[str]
    n_llm_calls: int
    n_input_tokens: int
    n_output_tokens: int
    cost_usd: float
    n_lines_emitted: int = 0
    n_lines_kept: int = 0


LLMResponse = collections.namedtuple("LLMResponse", "text input_tokens output_tokens")
LLMClient = Callable[[str, list[dict], int], LLMResponse]


def discover_api_key() -> str | None:
    indirected = os.environ.get("RLAT_LLM_API_KEY_ENV")
    if indirected and (val := os.environ.get(indirected)):
        return val
    if (val := os.environ.get("CLAUDE_API")):
        return val
    if (val := os.environ.get("ANTHROPIC_API_KEY")):
        return val
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    client = UserSecretsClient()
    for name in ("CLAUDE_API", "ANTHROPIC_API_KEY"):
        try:
            return client.get_secret(name)
        except Exception:
            continue
    return None


def api_key_or_error(api_key: str | None = None) -> str:
    key = api_key or discover_api_key()
    if not key:
        raise RuntimeError(
            "`rlat optimise` requires an LLM API key to generate synthetic "
            "training queries.\n\n"
            "Set an environment variable with your Anthropic API key, then either:\n"
            "  1. Name it CLAUDE_API or ANTHROPIC_API_KEY:\n"
            "       export CLAUDE_API=sk-ant-...\n"
            "       rlat optimise my-corpus.rlat\n\n"
            "  2. Or keep your own variable name and point rlat at it:\n"
            "       export MY_KEY=sk-ant-...\n"
            "       export RLAT_LLM_API_KEY_ENV=MY_KEY\n"
            "       rlat optimise my-corpus.rlat\n\n"
            "Cost: ~$10-20 per corpus of 6K target queries (Sonnet 4.6, one-time)."
        )
    return key


def default_client(api_key: str, model: str = DEFAULT_MODEL) -> LLMClient:
    """Wrap the `anthropic` SDK in our (system, messages, max_tokens) -> response shape.

    Concurrency-safe: anthropic.Anthropic is thread-safe per SDK docs (one
    HTTP client per process). The ThreadPool in `generate()` calls this
    callable from N worker threads simultaneously.
    """
    import anthropic
    sdk = anthropic.Anthropic(api_key=api_key)

    def call(system: str, messages: list[dict], max_tokens: int) -> LLMResponse:
        resp = sdk.messages.create(
            model=model, system=system, messages=messages, max_tokens=max_tokens,
        )
        text = resp.content[0].text
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
    return call


def estimate_cost(n_queries: int = DEFAULT_TARGET_QUERIES) -> float:
    """USD projection for `n_queries` per-query Sonnet calls + 1 anchor call.

    Per-query: ~400 tok input (system + anchors + passage) × ~30 tok output.
    Anchor: ~3000 input + 200 output, one-time.
    """
    synth_input = n_queries * 400
    synth_output = n_queries * 30
    anchor_input = 3000
    anchor_output = 200
    return _cost_usd(synth_input + anchor_input, synth_output + anchor_output)


_DEFAULT_CORPUS_DESCRIPTION_FALLBACK = "a knowledge model"

_ANCHOR_SYSTEM_TEMPLATE = """\
You are analyzing a corpus described as: {corpus_description}.
Below are 15 random passages from this corpus. Based on the content and style
of these passages, produce exactly 5 plausible natural-language questions that
a real user of this corpus would ask. Match the register and topic-distribution
you see in the passages. Do NOT quote the passages verbatim. Return ONLY a JSON
array of 5 strings, no preamble or explanation.\
"""

_SYNTH_SYSTEM_TEMPLATE = """\
You are a user of the following corpus: {corpus_description}. Given a passage
from this corpus, produce ONE natural-language question that a real user would
ask whose answer is contained in the passage. The question should NOT verbatim
copy sentences from the passage; it should reflect how a real user of this
corpus would phrase the query. Avoid meta-phrasings like 'according to the
passage' or 'what does the article say'. Return ONLY the question, no preamble.

Style anchors (these are example real-user questions matching the natural
style of this corpus, not your target passage):
{anchors_block}\
"""


def derive_style_anchors(
    sample_passages: Sequence[str],
    client: LLMClient,
    corpus_description: str,
) -> tuple[list[str], LLMResponse]:
    """One LLM call → 5 corpus-aware query-style anchors. JSON-array output."""
    formatted = "\n\n".join(
        f"[{i + 1}] {p.strip()[:1000]}" for i, p in enumerate(sample_passages)
    )
    system = _ANCHOR_SYSTEM_TEMPLATE.format(corpus_description=corpus_description)
    user = formatted
    response = client(system, [{"role": "user", "content": user}], DEFAULT_MAX_TOKENS_ANCHOR)
    # Spec § 4b parses as JSON array. Strip markdown fences if present.
    raw = response.text.strip()
    if raw.startswith("```"):
        # Drop leading fence + optional language tag, drop trailing fence.
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        parsed = json.loads(raw)
        anchors = [str(a).strip() for a in parsed if isinstance(a, str)]
    except (json.JSONDecodeError, TypeError):
        # Fall back to line-parse if the LLM didn't honour JSON output.
        anchors = [
            line.strip().lstrip("-•*0123456789. )").strip(' "')
            for line in response.text.splitlines()
            if line.strip()
        ]
    anchors = [a for a in anchors if 3 <= len(a.split()) <= 30][:DEFAULT_N_ANCHORS]
    return anchors, response


def _filter_query(query: str) -> str | None:
    """Apply spec §4c filters. Returns the cleaned query or None to drop."""
    q = query.strip()
    if len(q) < MIN_QUERY_LEN or len(q) > MAX_QUERY_LEN:
        return None
    if META_REJECT_WORD in q.lower()[:META_REJECT_PREFIX_LEN]:
        return None
    return q


def _generate_one_query(
    passage_idx: int,
    passage_text: str,
    system: str,
    client: LLMClient,
) -> tuple[SynthQuery | None, int, LLMResponse | None]:
    """One LLM call → one query for one passage. Returns (kept_or_None, emitted_count, response).

    `emitted_count` is 1 if the LLM responded with text (regardless of filter),
    0 if the call failed. Filter rejections still count as emitted so callers
    can compute retention.
    """
    user = f"PASSAGE:\n\n{passage_text.strip()[:4000]}\n\nWrite one question."
    try:
        response = client(system, [{"role": "user", "content": user}], DEFAULT_MAX_TOKENS_SYNTH)
    except Exception:
        return None, 0, None
    cleaned = _filter_query(response.text)
    if cleaned is None:
        return None, 1, response
    return SynthQuery(query=cleaned, passage_idx=passage_idx), 1, response


def _stratified_passage_sample(
    passage_idxs: Sequence[int],
    source_files: Sequence[str],
    queries_per_file_cap: int,
    target_queries: int,
    seed: int,
) -> list[int]:
    """Pick a list of passage_idxs to call the LLM about, applying spec §4c
    stratification: at most `queries_per_file_cap` per source_file, then
    seed-shuffle and trim to `target_queries`.

    Returns positional indices into `passage_idxs` (not the values themselves)
    so the caller can fetch the corresponding texts.
    """
    rng = random.Random(seed)
    by_file: dict[str, list[int]] = {}
    for pos, src in enumerate(source_files):
        by_file.setdefault(src, []).append(pos)
    candidates: list[int] = []
    for src, positions in by_file.items():
        rng.shuffle(positions)
        candidates.extend(positions[:queries_per_file_cap])
    rng.shuffle(candidates)
    return candidates[:target_queries]


def _file_cache_path(cache_dir: Path, corpus_fingerprint: str) -> Path:
    return cache_dir / corpus_fingerprint / "queries.jsonl"


def _save_one_to_cache(path: Path, q: SynthQuery, lock: threading.Lock) -> None:
    """Append one query to the JSONL cache atomically (lock-guarded)."""
    line = json.dumps({"q": q.query, "i": q.passage_idx}) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def _load_cache(path: Path) -> list[SynthQuery]:
    if not path.exists():
        return []
    out: list[SynthQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(SynthQuery(query=obj["q"], passage_idx=int(obj["i"])))
    return out


def _corpus_fingerprint(
    passage_idxs: Sequence[int],
    passages: Sequence[str],
    source_files: Sequence[str],
    corpus_description: str,
    model: str = DEFAULT_MODEL,
    schema_version: int = CACHE_SCHEMA_VERSION,
) -> str:
    """Cache key for the synth-query JSONL.

    Hashes EVERY input that changes the cache contents:
      - schema_version  — invalidates the whole cache when prompt/format
                          changes ship.
      - model           — swapping models would mix two query distributions.
      - corpus_description — drives the system prompt; same passages with
                          a different description produce different queries.
      - source_files    — same passage text but a different source file
                          (e.g. doc moved between repos) is a different
                          provenance and shouldn't reuse cached queries.
      - passage_idx     — registry-position discriminator.
      - passages (full) — earlier versions hashed only `text[:100]`, which
                          let edits past char 100 reuse stale queries.
                          Full-text hash is the correct cache key.

    Returns a 16-char hex slice — same length as before so existing
    cache directory names retain their visual shape.
    """
    h = hashlib.sha256()
    h.update(b"rlat-optimise-cache-v")
    h.update(str(schema_version).encode("ascii"))
    h.update(b"\x1e")
    h.update(model.encode("utf-8", errors="replace"))
    h.update(b"\x1e")
    h.update(corpus_description.encode("utf-8", errors="replace"))
    h.update(b"\x1e")
    for idx, source, text in sorted(zip(passage_idxs, source_files, passages)):
        h.update(str(idx).encode("ascii"))
        h.update(b"\x1f")
        h.update(source.encode("utf-8", errors="replace"))
        h.update(b"\x1f")
        h.update(text.encode("utf-8", errors="replace"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def generate(
    passage_idxs: Sequence[int],
    passages: Sequence[str],
    source_files: Sequence[str],
    client: LLMClient | None = None,
    cache_dir: Path | None = None,
    corpus_description: str = _DEFAULT_CORPUS_DESCRIPTION_FALLBACK,
    n_anchor_sample: int = DEFAULT_ANCHOR_SAMPLE,
    queries_per_file_cap: int = DEFAULT_QUERIES_PER_FILE_CAP,
    target_queries: int = DEFAULT_TARGET_QUERIES,
    concurrency: int = DEFAULT_CONCURRENCY,
    seed: int = 0,
    progress: Callable[[str, int, int], None] | None = None,
) -> SynthQueryResult:
    """Generate synth queries per spec §4 (per-query LLM call + stratification).

    `corpus_description` MUST be passed for non-trivial corpora. Anchors
    derived without it default to the fallback string and bias query style
    against the actual corpus. Per spec §9 trap #3, this is the most common
    cross-corpus replication failure mode.

    `cache_dir` enables resumable runs: the per-query JSONL is append-only
    so a mid-corpus crash resumes from the next un-tried passage. The cache
    is keyed on (corpus_fingerprint), not (corpus + run-id), so multiple
    runs on the same corpus share the cache.
    """
    if not (len(passage_idxs) == len(passages) == len(source_files)):
        raise ValueError(
            f"length mismatch: passage_idxs={len(passage_idxs)}, "
            f"passages={len(passages)}, source_files={len(source_files)}"
        )
    if len(passages) == 0:
        return SynthQueryResult([], [], 0, 0, 0, 0.0)

    if client is None:
        client = default_client(api_key_or_error())

    fp = _corpus_fingerprint(
        passage_idxs, passages, source_files, corpus_description,
    )

    # Step 1 — style anchors. Cached per corpus.
    rng = random.Random(seed)
    sample_idx = rng.sample(range(len(passages)), min(n_anchor_sample, len(passages)))
    sample = [passages[i] for i in sample_idx]
    n_calls = 0
    n_input = 0
    n_output = 0

    anchor_cache_path = (
        cache_dir / fp / "_anchors.json" if cache_dir is not None else None
    )
    if anchor_cache_path is not None and anchor_cache_path.exists():
        anchors = json.loads(anchor_cache_path.read_text(encoding="utf-8"))
    else:
        anchors, anchor_resp = derive_style_anchors(sample, client, corpus_description)
        n_calls += 1
        n_input += anchor_resp.input_tokens
        n_output += anchor_resp.output_tokens
        if anchor_cache_path is not None:
            anchor_cache_path.parent.mkdir(parents=True, exist_ok=True)
            anchor_cache_path.write_text(json.dumps(anchors), encoding="utf-8")

    anchors_block = "\n".join(f"- {a}" for a in anchors)
    synth_system = _SYNTH_SYSTEM_TEMPLATE.format(
        corpus_description=corpus_description, anchors_block=anchors_block,
    )

    # Step 2 — stratified passage sample (cap-3-per-file, target ~6K queries).
    sampled_positions = _stratified_passage_sample(
        passage_idxs, source_files, queries_per_file_cap, target_queries, seed,
    )

    # Resume from cache: skip any passage_idx that already has a kept query.
    cache_path = _file_cache_path(cache_dir, fp) if cache_dir is not None else None
    already: dict[int, SynthQuery] = {}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for q in _load_cache(cache_path):
            already.setdefault(q.passage_idx, q)
    todo = [
        pos for pos in sampled_positions if passage_idxs[pos] not in already
    ]

    # Step 3 — concurrent per-query LLM calls. ThreadPool because the SDK is
    # I/O bound (HTTP) — a process pool would just add serialization overhead.
    cache_lock = threading.Lock()
    aggregate_lock = threading.Lock()
    n_lines_emitted = 0
    n_lines_kept = 0
    new_queries: list[SynthQuery] = []

    def _worker(pos: int) -> None:
        nonlocal n_calls, n_input, n_output, n_lines_emitted, n_lines_kept
        q, emitted, resp = _generate_one_query(
            int(passage_idxs[pos]), passages[pos], synth_system, client,
        )
        with aggregate_lock:
            if resp is not None:
                n_calls += 1
                n_input += resp.input_tokens
                n_output += resp.output_tokens
            n_lines_emitted += emitted
            if q is not None:
                n_lines_kept += 1
                new_queries.append(q)
        if q is not None and cache_path is not None:
            _save_one_to_cache(cache_path, q, cache_lock)

    if todo:
        total = len(todo)
        done_counter = [0]
        if progress is not None:
            progress("synth", 0, total)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_worker, pos) for pos in todo]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                done_counter[0] += 1
                if progress is not None and (
                    done_counter[0] % 50 == 0 or done_counter[0] == total
                ):
                    progress("synth", done_counter[0], total)

    # Sort by passage_idx so the final list is deterministic. Worker results
    # arrive via `as_completed` (completion order, not submission order), so
    # `new_queries` is run-to-run nondeterministic without this sort —
    # downstream training would see different mini-batches across reruns of
    # the same corpus + cache. Cached entries are merged in too so a partial
    # cache + fresh-fill produces identical output to a fresh full run.
    all_queries = sorted(
        list(already.values()) + new_queries, key=lambda q: q.passage_idx,
    )

    return SynthQueryResult(
        queries=all_queries,
        style_anchors=anchors,
        n_llm_calls=n_calls,
        n_input_tokens=n_input,
        n_output_tokens=n_output,
        cost_usd=_cost_usd(n_input, n_output),
        n_lines_emitted=n_lines_emitted + len(already),  # cache entries count as kept
        n_lines_kept=n_lines_kept + len(already),
    )
