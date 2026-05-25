"""UDF cold-start init + warm-call cache lookup.

Mtime is checked on every call so a re-uploaded .rlat propagates within
one warm call. LRU(8) bounds the in-memory cache; the on-disk .rlat is
unlinked alongside any evicted entry.
"""

from __future__ import annotations

import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

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

# (km_name, mtime) -> (contents, store, encoder).
_STATE: "OrderedDict[tuple[str, str], tuple[Any, Any, Encoder]]" = OrderedDict()

# Encoder cache keyed by HF revision. Single source of truth — every
# Encoder() construction goes through _load_encoder(), so _STATE's
# encoders and embed()'s encoders share these instances. Capped at
# _ENCODER_CACHE_MAX because encoder weights stay resident (~hundreds of
# MB) and revisions roll forward on rebuilds.
_ENCODER_CACHE: "OrderedDict[str, Encoder]" = OrderedDict()


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
        p.unlink()


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
) -> tuple[tuple[Any, Any, Encoder], bool]:
    """Return `(state, cold)` where state = (contents, store, encoder).

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
    contents = archive.read(path)
    revision = contents.metadata.backbone.revision
    if not revision:
        raise RuntimeError(
            f"km {km_name!r} has empty backbone.revision in metadata — rebuild "
            "with rlat>=2.0 (older builds didn't pin the encoder revision)."
        )
    enc = _load_encoder(revision, lakehouse=lakehouse)
    store = _open_fabric_store(lakehouse, path, contents)
    state = (contents, store, enc)

    _STATE[cache_key] = state
    while len(_STATE) > _STATE_LRU_MAX:
        evicted_key, _ = _STATE.popitem(last=False)
        # Keep the on-disk file when another live entry still references it
        # (rapid rebuilds can leave multiple mtime entries for the same km).
        if not any(k[0] == evicted_key[0] for k in _STATE):
            _evict_disk(evicted_key[0])
    return state, True


def search_with_state(
    state: tuple[Any, Any, Encoder],
    query: str,
    top_k: int,
    verified_only: bool,
    *,
    cold: bool = False,
) -> dict[str, Any]:
    """Run retrieval against the bootstrapped state. Returns wire-format dict."""
    contents, store, enc = state
    handle = contents.select_band()
    idx = ann.deserialize(handle.ann_blob) if handle.ann_blob else None
    qv = enc.encode([query])[0]
    hits = retrieve(qv, handle, idx, contents.registry, top_k)
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
        _, _, enc = next(iter(_STATE.values()))
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


def list_kms_for(lakehouse: _LakehouseClient) -> list[dict[str, Any]]:
    """Return one row per KM listed in the manifest, with header metadata."""
    out: list[dict[str, Any]] = []
    for km_name in list_km_paths(lakehouse):
        path = fetch_rlat(lakehouse, km_name, _KM_CACHE_DIR)
        contents = archive.read(path)
        out.append({
            "kmName":           km_name,
            "n_passages":       len(contents.registry),
            "created_utc":      contents.metadata.created_utc,
            "encoder_revision": contents.metadata.backbone.revision,
        })
    return out
