"""UDF cold-start init + warm-call cache lookup.

Mtime is checked on every call so a re-uploaded .rlat propagates within
one warm call. LRU(8) bounds the in-memory cache; the on-disk .rlat is
unlinked alongside any evicted entry.
"""

from __future__ import annotations

import json
import os
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .._paths import xdg_cache_root
from ..config import StoreMode
from ..field import ann, retrieve
from ..field.encoder import Encoder
from ..install.encoder import cache_dir as encoder_cache_dir
from ..store import archive
from ..store.bundled import BundledStore
from ..store.verified import filter_verified, verify_hits
from .hf_loader import fetch_encoder_from_hf
from .lakehouse_loader import _km_dir, fetch_rlat, get_rlat_mtime, list_km_paths
from .onelake_store import OneLakeStore

_STATE_LRU_MAX = 8
_ENCODER_CACHE_MAX = 2
_KM_CACHE_DIR = xdg_cache_root() / "fabric" / "km"

# OneLake-side path (under the bound lakehouse's Files/) where the encoder
# cache is mirrored. Same layout the build notebook writes via
# XDG_CACHE_HOME=/lakehouse/default/Files/.rlat-cache → xdg_cache_root() ==
# .rlat-cache/rlat, so the cache_dir(revision) resolves to
# .rlat-cache/rlat/encoders/<revision>/. The UDF mirrors that path
# bidirectionally via the SDK (no POSIX mount available in UDF runtime).
_ONELAKE_ENCODER_PREFIX = ".rlat-cache/rlat/encoders"

# (km_name, mtime) -> (contents, store, encoder, ann_index).
_STATE: "OrderedDict[tuple[str, str], tuple[Any, Any, Encoder, Any]]" = OrderedDict()

# Encoder cache keyed by HF revision. Single source of truth — every
# Encoder() construction goes through _load_encoder(), so _STATE's
# encoders and embed()'s encoders share these instances. Capped at
# _ENCODER_CACHE_MAX because encoder weights stay resident (~hundreds of
# MB) and revisions roll forward on rebuilds.
_ENCODER_CACHE: "OrderedDict[str, Encoder]" = OrderedDict()

# Slicer serve cache: (km_name, upstream_mtime) -> (band_mmap, keys). The band is
# DEFLATE-decompressed ONCE to an uncompressed /tmp .npy and mmap'd, so a warm
# slice_stream query is encode + one GEMV (~ms) instead of re-decompressing the
# 290 MB band + re-parsing 94k keys per call (the ~6 s + ~2 s the perf decomposition
# found — see .claude/plans/slicer-reset-2026-06-07.md). mmap pages are file-backed
# and reclaimable, so the band never enters the OOM-prone process heap. Capped small
# — each slot pins one big uncompressed band on /tmp.
_SLICER_CACHE: "OrderedDict[tuple[str, str], tuple[Any, list, str, Any]]" = OrderedDict()
_SLICER_CACHE_MAX = 2
# Serializes the cold band build (double-checked below) so concurrent invocations
# for the same km don't double-materialise / clobber the cache. Warm hits skip it.
_SLICER_LOCK = threading.Lock()


class _LakehouseClient(Protocol):
    """Structural shape of `fabric.functions.FabricLakehouseClient`.

    Defined here so the runtime helpers stay importable without the
    `fabric.functions` package — the harness substitutes a duck-typed mock.
    """
    def connectToFiles(self) -> Any: ...


def _km_path_on_disk(km_name: str) -> Path:
    return _KM_CACHE_DIR / f"{km_name}.rlat"


def _evict_disk(km_name: str) -> None:
    p = _km_path_on_disk(km_name)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass  # a still-open handle (Windows) shouldn't abort an eviction loop


def _onelake_has_encoder(lakehouse: Any, revision: str) -> bool:
    """One get_file_properties() call (~50ms) to check whether OneLake
    has the encoder cache for this revision. Used to skip the 600 MB
    upload when the cache is already populated."""
    onelake_prefix = f"{_ONELAKE_ENCODER_PREFIX}/{revision}"
    try:
        fc = lakehouse.connectToFiles().get_file_client(f"{onelake_prefix}/model.onnx")
        fc.get_file_properties()
        return True
    except Exception:
        return False


def _seed_encoder_cache_from_onelake(lakehouse: Any, revision: str) -> bool:
    """Mirror Files/.rlat-cache/rlat/encoders/<revision>/* into the local
    cache directory `encoder_cache_dir(revision)`. Returns True if the
    local cache ends with `model.onnx` present (either it was already
    there or this call populated it). Returns False on any failure mode
    or when OneLake is empty — caller falls through to HF and uploads
    afterwards.
    """
    local_dir = encoder_cache_dir(revision)
    if (local_dir / "model.onnx").exists():
        return True  # already populated locally (prior call in this container)

    from .lakehouse_loader import _list_via_onelake_rest
    onelake_prefix = f"{_ONELAKE_ENCODER_PREFIX}/{revision}"
    try:
        sub = lakehouse.connectToFiles().get_sub_directory_client(onelake_prefix)
        full_names = _list_via_onelake_rest(sub)
    except Exception:
        return False

    if not full_names:
        return False

    local_dir.mkdir(parents=True, exist_ok=True)
    files_client = lakehouse.connectToFiles()
    rev_marker = f"/{revision}/"
    for full in full_names:
        # full looks like "<lh>/Files/.rlat-cache/rlat/encoders/<rev>/<file>"
        if rev_marker not in full:
            continue
        rel = full.split(rev_marker, 1)[1]
        if not rel or rel.endswith("/"):
            continue
        try:
            data = files_client.get_file_client(f"{onelake_prefix}/{rel}").download_file().readall()
        except FileNotFoundError:
            continue
        out = local_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)

    return (local_dir / "model.onnx").exists()


def _upload_encoder_cache_to_onelake(lakehouse: Any, revision: str) -> None:
    """Walk `encoder_cache_dir(revision)/` and upload each file under
    Files/.rlat-cache/rlat/encoders/<revision>/. `upload_data(overwrite=True)`
    makes concurrent writes safe — the encoder is content-addressed by HF
    revision so racing writers produce identical bytes."""
    local_dir = encoder_cache_dir(revision)
    if not local_dir.exists():
        return
    onelake_prefix = f"{_ONELAKE_ENCODER_PREFIX}/{revision}"
    files_client = lakehouse.connectToFiles()
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        with open(path, "rb") as f:
            files_client.get_file_client(f"{onelake_prefix}/{rel}").upload_data(f.read(), overwrite=True)


def _load_encoder(revision: str, lakehouse: Any = None) -> Encoder:
    """Return an Encoder for `revision`, loaded once and cached.

    Single point where encoders enter memory — used by both bootstrap()
    and embed_query() so the same encoder object serves search and embed
    traffic alike. Eager warm-up amortises the first inference call.

    If `lakehouse` is supplied, the cache is seeded from OneLake first and
    populated back to OneLake on cache miss — self-healing across cold-
    starts AND across consumer surfaces (build notebook + UDF share the
    same OneLake path, but neither depends on the other running).
    """
    enc = _ENCODER_CACHE.get(revision)
    if enc is not None:
        _ENCODER_CACHE.move_to_end(revision)
        return enc

    if lakehouse is not None:
        try:
            _seed_encoder_cache_from_onelake(lakehouse, revision)
        except Exception:
            pass  # fall through to HF; self-heal via upload below

    fetch_encoder_from_hf(revision)  # no-op when local cache is populated
    enc = Encoder()
    enc.encode(["warm"])
    _ENCODER_CACHE[revision] = enc
    while len(_ENCODER_CACHE) > _ENCODER_CACHE_MAX:
        _ENCODER_CACHE.popitem(last=False)

    # Self-heal OneLake: upload only when OneLake doesn't already have the
    # encoder cache. The cheap get_file_properties() check (~50ms) saves
    # the ~10s upload on every cold-start where OneLake is already populated.
    if lakehouse is not None:
        try:
            if not _onelake_has_encoder(lakehouse, revision):
                _upload_encoder_cache_to_onelake(lakehouse, revision)
        except Exception:
            pass  # non-fatal: encoder is loaded; next cold-start retries

    return enc


def _peek_revision(path: Path) -> str:
    """Read just `metadata.json` from a `.rlat` ZIP and return its
    `backbone.revision` (or "" if absent).

    Bypasses `archive.read()` — that eagerly loads every band + the
    passage registry, hundreds of MB on real corpora. The cold-path
    revision peek only needs ~1 KB.
    """
    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("metadata.json"))
    return meta.get("backbone", {}).get("revision", "") or ""


def _open_fabric_store(lakehouse: Any, km_path: Path, contents: Any) -> Any:
    """Construct the right Store for a UDF context.

    `bundled` reads source from the .rlat archive (works as-is). `local`
    reads source from OneLake via the bound lakehouse client. `remote`
    isn't supported here — remote-mode .rlats expect HTTP URLs that the
    UDF runtime has no business proxying.
    """
    mode = StoreMode(contents.metadata.store_mode)
    if mode is StoreMode.BUNDLED:
        return BundledStore(km_path)
    if mode is StoreMode.LOCAL:
        source_root = contents.metadata.build_config.get("source_root", "")
        return OneLakeStore(lakehouse, source_root)
    raise RuntimeError(
        f"Fabric UDF does not support store_mode={mode.value!r}; "
        f"build the .rlat with --store-mode bundled or local"
    )


def bootstrap(
    lakehouse: _LakehouseClient, km_name: str
) -> tuple[tuple[Any, Any, Encoder, Any], bool]:
    """Return `(state, cold)` where state = (contents, store, encoder, ann_index).

    `ann_index` is the base ANN index, deserialised once here; the serialised
    bytes are freed from `contents.ann_blobs` so the worker holds only the live
    index (paired with lazy-band-skip dropping the base band).

    `cold` is True when this call had to download .rlat + load encoder,
    False on a cache hit. Consumers can split warm/cold latency.
    """
    upstream_mtime = get_rlat_mtime(lakehouse, km_name)
    cache_key = (km_name, upstream_mtime)

    cached = _STATE.get(cache_key)
    if cached is not None:
        _STATE.move_to_end(cache_key)
        return cached, False

    for stale_key in [k for k in _STATE if k[0] == km_name]:
        _STATE.pop(stale_key, None)
    _evict_disk(km_name)

    path = fetch_rlat(lakehouse, km_name, _KM_CACHE_DIR)
    # Lazy-band-skip: the UDF retrieval path (search/slice) runs ANN when the km
    # has an index, which never reads the base band — so don't hold the full
    # (N,768) matrix in the worker. Falls back to loading it for dense (no-ANN)
    # corpora. Pairs with the no-FAISS slice_stream sidecar for the largest km.
    contents = archive.read(path, defer_base_band=True)
    revision = contents.metadata.backbone.revision
    if not revision:
        raise RuntimeError(
            f"km {km_name!r} has empty backbone.revision in metadata — rebuild "
            "with rlat>=2.0 (older builds didn't pin the encoder revision)."
        )
    enc = _load_encoder(revision, lakehouse=lakehouse)
    store = _open_fabric_store(lakehouse, path, contents)
    # Deserialise the base ANN index ONCE and FREE the serialised bytes — the
    # worker otherwise holds both the ~316 MB blob (in contents.ann_blobs) AND
    # the live index, and re-deserialises per call. With lazy-band-skip already
    # dropping the base band, the resident set is then ~encoder + live index
    # (both redundant vector copies gone).
    base_blob = contents.ann_blobs.pop("base", None)
    ann_index = ann.deserialize(base_blob) if base_blob is not None else None
    state = (contents, store, enc, ann_index)

    _STATE[cache_key] = state
    while len(_STATE) > _STATE_LRU_MAX:
        evicted_key, _ = _STATE.popitem(last=False)
        # Keep the on-disk file when another live entry still references it
        # (rapid rebuilds can leave multiple mtime entries for the same km).
        if not any(k[0] == evicted_key[0] for k in _STATE):
            _evict_disk(evicted_key[0])
    return state, True


def search_with_state(
    state: tuple[Any, Any, Encoder, Any],
    query: str,
    top_k: int,
    verified_only: bool,
    *,
    cold: bool = False,
) -> dict[str, Any]:
    """Run retrieval against the bootstrapped state. Returns wire-format dict."""
    contents, store, enc, *rest = state  # 4-tuple from bootstrap (cached index) or legacy 3-tuple
    ann_index = rest[0] if rest else None
    handle = contents.select_band()
    if ann_index is None and handle.ann_blob:  # legacy 3-tuple state: deserialise on demand
        ann_index = ann.deserialize(handle.ann_blob)
    qv = enc.encode([query])[0]
    hits = retrieve(qv, handle, ann_index, contents.registry, top_k)
    verified = verify_hits(hits, store, contents.registry)
    if verified_only:
        verified = filter_verified(verified)
    return {
        "band": handle.name,
        "cold": cold,
        "hits": [
            {
                "passage_idx":  h.passage_idx,
                "source_file":  h.source_file,
                "char_offset":  h.char_offset,
                "char_length":  h.char_length,
                "content_hash": h.content_hash,
                "drift_status": h.drift_status,
                "score":        float(h.score),
                "text":         h.text,
                "key":          h.key,
            }
            for h in verified
        ],
    }


def slice_with_state(
    state: tuple[Any, Any, Encoder, Any],
    query: str,
    top_k: int,
    *,
    verified_only: bool = True,
    cold: bool = False,
) -> dict[str, Any]:
    """Semantic-slicer surface: a plain-language query → a business-KEY set.

    The Data App's slicer flow (semantic-slicer-handoff.md §4): type meaning
    → rlat returns the matched keys → the app builds
    `TREATAS({keys}, 'Dim'[Key])` for the Execute DAX Queries API. Returns:

        {band, cold, keys: [<key>, ...], hits: [{key, score, text,
         drift_status}, ...]}

    `keys` is the deduped, score-ordered key set (the TREATAS payload);
    `hits` keeps the per-key score + matched snippet (the "receipt" the
    plan's ranked-confirm UX shows). Only row-mode (keyed) knowledge models
    yield keys — a hit whose `key` is None (a chunked corpus) is dropped
    from `keys` (its snippet still appears in `hits`), so calling this on a
    non-keyed km returns an empty key set rather than erroring.
    """
    contents, store, enc, *rest = state  # 4-tuple from bootstrap (cached index) or legacy 3-tuple
    ann_index = rest[0] if rest else None
    handle = contents.select_band()
    if ann_index is None and handle.ann_blob:  # legacy 3-tuple state: deserialise on demand
        ann_index = ann.deserialize(handle.ann_blob)
    qv = enc.encode([query])[0]
    hits = retrieve(qv, handle, ann_index, contents.registry, top_k)
    verified = verify_hits(hits, store, contents.registry)
    if verified_only:
        verified = filter_verified(verified)
    keys: list[str] = []
    seen: set[str] = set()
    for h in verified:
        if h.key is not None and h.key not in seen:
            seen.add(h.key)
            keys.append(h.key)
    return {
        "band": handle.name,
        "cold": cold,
        "keys": keys,
        "hits": [
            {
                "key":          h.key,
                "score":        float(h.score),
                "text":         h.text,
                "drift_status": h.drift_status,
            }
            for h in verified
        ],
    }


def embed_query(lakehouse: _LakehouseClient, query: str) -> list[float]:
    """Return the L2-normalised CLS embedding for `query`.

    Invariant the warm path relies on: every `.rlat` in a given lakehouse
    is built with the same encoder revision (the deploy notebook pins it).
    Otherwise embed-ed query vectors would not be comparable to the
    vectors stored in `dbo.passages`. If a tenant ever mixes revisions,
    embed() picks an arbitrary one and the cosine ranking degrades.
    """
    if _STATE:
        _, _, enc, _ = next(iter(_STATE.values()))
        return enc.encode([query])[0].tolist()

    km_names = list_km_paths(lakehouse)
    if not km_names:
        raise RuntimeError(
            f"embed() needs at least one .rlat in the lakehouse to pin the "
            f"encoder revision; none found under Files/{_km_dir()}/"
        )
    revision = ""
    for km in km_names:
        path = fetch_rlat(lakehouse, km, _KM_CACHE_DIR)
        rev = _peek_revision(path)
        if rev:
            revision = rev
            break
    if not revision:
        raise RuntimeError(
            f"no .rlat under Files/{_km_dir()}/ has a pinned "
            f"backbone.revision (checked: {km_names}). Rebuild with "
            "rlat>=2.0 — older builds didn't pin the encoder revision."
        )
    enc = _load_encoder(revision, lakehouse=lakehouse)
    return enc.encode([query])[0].tolist()


def _slicer_band_path(km_name: str) -> Path:
    return _KM_CACHE_DIR / f"{km_name}.band.npy"


def _close_slicer_entry(entry: "tuple[Any, list, str, Any] | None") -> None:
    """Release a cached entry's OS handles (band mmap + source-snippet ZipFile) —
    but ONLY on Windows.

    The warm serve hands out the cached entry WITHOUT the lock, so a concurrent
    eviction can run while a reader is still mid-`band @ q` on the band mmap. Force-
    closing that mmap (`mm.close()`) under it is a use-after-free → a hard, uncatchable
    worker segfault. On Linux (the UDF) we must NOT force-close: dropping the cache
    reference is enough — CPython refcounting unmaps the band and closes the ZipFile
    once the last in-flight reader releases them, and unlink-while-open is safe on
    Linux, so the `/tmp` `.npy` / `.rlat` still delete cleanly. Windows (the test path)
    can't unlink an open file and runs the cache single-threaded (no live reader during
    eviction), so there we DO close to let the tempdir clean up."""
    if not entry or os.name != "nt":
        return
    mm = getattr(entry[0], "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except Exception:
            pass
    snippets = entry[3] if len(entry) > 3 else None
    if snippets is not None:
        snippets.close()


def _clear_slicer_cache() -> None:
    """Drop every cached band, closing mmaps + unlinking the /tmp .npy files."""
    while _SLICER_CACHE:
        key, entry = _SLICER_CACHE.popitem()
        _close_slicer_entry(entry)
        _evict_slicer_band(key[0])


def _evict_slicer_band(km_name: str) -> None:
    p = _slicer_band_path(km_name)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def _slicer_band(lakehouse: _LakehouseClient, km_name: str) -> tuple[Any, list, str, Any]:
    """Return `(band_mmap, keys, revision, snippets)` for the slicer, busting on mtime.

    Cold: fetch the `.rlat`, DEFLATE-decompress its band ONCE to an uncompressed
    `/tmp` `.npy`, `mmap` it, parse the keys, peek the encoder revision, and open a
    held-handle `SourceSnippets` reader over the bundled `source/` (its 94k-member
    central directory parsed once) — then cache all four. Warm: reuse them (no
    decompress, no re-parse, no central-dir re-parse, and crucially no touch of the
    on-disk `.rlat`, which a concurrent `bootstrap` LRU could have evicted). The mmap
    is reclaimable page cache, so the band never lands in the process heap. The cold
    build is serialized by `_SLICER_LOCK` (double-checked) so concurrent first-calls
    for one km don't double-materialise / clobber the cache.
    """
    from ..store.streaming import SourceSnippets, materialize_band, read_keys
    from .lakehouse_loader import fetch_rlat, get_rlat_mtime

    cache_key = (km_name, get_rlat_mtime(lakehouse, km_name))
    cached = _SLICER_CACHE.get(cache_key)
    if cached is not None:
        _SLICER_CACHE.move_to_end(cache_key)
        return cached

    with _SLICER_LOCK:
        # Re-check under the lock — another thread may have built it while we waited.
        cached = _SLICER_CACHE.get(cache_key)
        if cached is not None:
            _SLICER_CACHE.move_to_end(cache_key)
            return cached

        # A changed upstream mtime invalidates any prior entry for this km — drop it
        # (closing its mmap), re-fetch the .rlat, and re-materialise the band.
        for stale in [k for k in _SLICER_CACHE if k[0] == km_name]:
            _close_slicer_entry(_SLICER_CACHE.pop(stale, None))
        _evict_disk(km_name)
        _evict_slicer_band(km_name)

        path = fetch_rlat(lakehouse, km_name, _KM_CACHE_DIR)
        revision = _peek_revision(path)
        if not revision:
            raise RuntimeError(
                f"km {km_name!r} has empty backbone.revision in metadata — rebuild "
                "with rlat>=2.0 (older builds didn't pin the encoder revision)."
            )
        materialize_band(path, _slicer_band_path(km_name))
        band = np.load(_slicer_band_path(km_name), mmap_mode="r")
        # Held-handle source reader for the snippet receipt; pays the ~200 ms
        # central-dir parse here (cold) so warm snippet reads are ~µs. A non-bundled
        # km has no source/, so `.text()` returns None and hits degrade to keys-only.
        try:
            snippets = SourceSnippets(path)
        except Exception:
            snippets = None
        entry = (band, read_keys(path), revision, snippets)
        _SLICER_CACHE[cache_key] = entry
        while len(_SLICER_CACHE) > _SLICER_CACHE_MAX:
            old_key, old_entry = _SLICER_CACHE.popitem(last=False)
            _close_slicer_entry(old_entry)
            if not any(k[0] == old_key[0] for k in _SLICER_CACHE):
                _evict_slicer_band(old_key[0])
                _evict_disk(old_key[0])
        return entry


def slice_stream_native(
    lakehouse: _LakehouseClient, km_name: str, query: str, top_k: int,
    snippet_top_n: int = 50,
) -> dict[str, Any]:
    """OOM-safe + fast semantic slicer — mmap the band out of the `.rlat`, GEMV top-k.

    On the FIRST call for a km the band is DEFLATE-decompressed once to an
    uncompressed `/tmp` `.npy` and mmap'd (`_slicer_band`); every warm query is then
    just `encode(query)` + one vectorised GEMV over the mmap (~ms) — no per-call
    decompress, no key re-parse, no FAISS, no band in the process heap. Single-file:
    vectors + keys both from the one `.rlat`; `bootstrap()` is deliberately NOT
    called (it would `archive.read` the band + deserialize the index — the ~1.5 GB
    OOM, see `.claude/plans/slicer-reset-2026-06-07.md`).

    Returns `{keys, hits: [{key, score, text}]}` — `keys` is the deduped,
    score-ordered TREATAS payload; `hits` keeps each ranked row's score, and the top
    `snippet_top_n` carry `text` (the matched description, the slice receipt — read
    from the bundled `source/` via the held-handle `SourceSnippets`, ~µs warm). Hits
    past `snippet_top_n` carry no `text` key (the UI shows only the top few), and a
    non-bundled km yields no `text` at all. `hits` is NOT deduped, so a corpus with
    repeated business keys gives `len(hits) >= len(keys)` — consumers read `keys` for
    the filter and must not zip the two positionally.
    """
    from ..store.streaming import topk_over_band

    band, keys, revision, snippets = _slicer_band(lakehouse, km_name)
    enc = _load_encoder(revision, lakehouse=lakehouse)
    qv = enc.encode([query])[0]
    ranked = topk_over_band(band, keys, qv, top_k)
    keys_out: list[str] = []
    seen: set[str] = set()
    for key, _score in ranked:
        if key not in seen:
            seen.add(key)
            keys_out.append(key)
    # Clamp the snippet count to [0, top_k]: a negative caller value must not silently
    # disable the receipt by accident, and there's no point reading past the ranked set
    # (ranked is already capped at top_k). The UI only renders the top ~50 regardless.
    n_snip = max(0, min(snippet_top_n, top_k))
    hits: list[dict[str, Any]] = []
    for i, (key, score) in enumerate(ranked):
        hit: dict[str, Any] = {"key": key, "score": float(score)}
        if snippets is not None and i < n_snip:
            text = snippets.text(key)
            if text:
                hit["text"] = text
        hits.append(hit)
    return {"keys": keys_out, "hits": hits}


def list_kms_for(lakehouse: _LakehouseClient) -> list[dict[str, Any]]:
    """Return one row per KM listed in the manifest, with header metadata.

    Reads ONLY `metadata.json` per archive (same posture as
    `_peek_revision`) — `archive.read()` eagerly loads every band + the
    ANN blob, which on the 94k-row corpus is the documented ~1.5 GB
    worker-killer the slicer serve was redesigned to avoid. A discovery
    endpoint must never be able to OOM the worker.
    """
    out: list[dict[str, Any]] = []
    for km_name in list_km_paths(lakehouse):
        path = fetch_rlat(lakehouse, km_name, _KM_CACHE_DIR)
        with zipfile.ZipFile(path, "r") as zf:
            meta = json.loads(zf.read("metadata.json"))
        bands = meta.get("bands", {}) or {}
        base = bands.get("base", {}) or {}
        out.append({
            "kmName":           km_name,
            "n_passages":       int(base.get("passage_count", 0)),
            "created_utc":      meta.get("created_utc", ""),
            "encoder_revision": (meta.get("backbone", {}) or {}).get("revision", "") or "",
        })
    return out
