"""UDF cold-start init + warm-call cache lookup.

Mtime is checked on every call so a re-uploaded .rlat propagates within
one warm call. LRU(8) bounds the in-memory cache; the on-disk .rlat is
unlinked alongside any evicted entry.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

from .._paths import xdg_cache_root
from ..config import StoreMode
from ..field import ann, retrieve
from ..field.encoder import Encoder
from ..store import archive
from ..store.bundled import BundledStore
from ..store.verified import filter_verified, verify_hits
from .hf_loader import fetch_encoder_from_hf
from .lakehouse_loader import fetch_rlat, get_rlat_mtime, list_km_paths
from .onelake_store import OneLakeStore

_STATE_LRU_MAX = 8
_KM_CACHE_DIR = xdg_cache_root() / "fabric" / "km"

# (km_name, mtime) -> (contents, store, encoder).
_STATE: "OrderedDict[tuple[str, str], tuple[Any, Any, Encoder]]" = OrderedDict()


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
    fetch_encoder_from_hf(revision)
    store = _open_fabric_store(lakehouse, path, contents)
    enc = Encoder()
    enc.encode(["warm"])  # load weights eagerly so the first real call doesn't pay it
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
