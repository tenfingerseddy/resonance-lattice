# Resonance Lattice — Microsoft Fabric UDF entry point.
#
# Functions exposed to callers:
#   - slice_stream(kmName, query, ...) → deduped business-KEY set (the slicer).
#         Streams the band out of the .rlat one row-chunk at a time — never
#         loads the (N,768) band or a FAISS index, so it serves the full 94k
#         corpus at ~encoder + one chunk (~650 MB) instead of OOMing at ~1.5 GB.
#         Single file: vectors + keys both from the one .rlat, no sidecar/SQL.
#   - embed(query)                 → 768-d L2-normalised CLS embedding
#   - list_kms()                   → discovery endpoint, lists Files/rlat/*.rlat
#   - search / slice               → bundled-.rlat retrieval (band + FAISS) for
#         SMALL/medium corpora; OOM-prone on large ones — use slice_stream there.
#   - probe_runtime()              → dependency-free worker filesystem diagnostic.
#
# They read the bound Lakehouse via the @udf.connection decorator. The
# kmLakehouse alias must be configured in the Fabric portal under Manage
# Connections; bind it to the Lakehouse holding Files/rlat/<kmName>.rlat.
#
# embed() is called by Fabric SQL DB's dbo.rlat_search stored procedure
# via sp_invoke_external_rest_endpoint — same UDF process, same warm
# Encoder. Vector parity between SQL DB and external `rlat search fabric://`
# is mechanical (one encoder serves both surfaces).
#
# Build / refresh live in the Fabric notebook (`fabric_build.ipynb`),
# not here — the encoder weights resident in memory plus the build path's
# transient peak exceeds the Fabric Python-worker memory ceiling on real
# corpora. The notebook kernel has more headroom and we've validated it
# end-to-end.
#
# The encoder revision is read off each .rlat at bootstrap time and the
# matching ONNX cache is pulled from HuggingFace (revision-pinned). No
# encoder tarball needs to live in the Lakehouse.

import os
import re

# Force the rlat encoder cache onto /tmp before any rlat import — the Fabric
# UDF runtime's $HOME may be read-only, but /tmp is always writable.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rlat")

# Where .rlat files live under the bound Lakehouse's Files/ root.
# Default is `rlat` (matches the published convention in docs/user/FABRIC.md).
# Set this to `knowledge`, `data`, or any other subdir name if the workspace
# already uses a different layout.
os.environ.setdefault("RLAT_FABRIC_KM_DIR", "rlat")

import fabric.functions as fn
from resonance_lattice.fabric import (
    bootstrap,
    embed_query,
    list_kms_for,
    search_with_state,
    slice_stream_native,
    slice_with_state,
)

udf = fn.UserDataFunctions()

# kmName flows into OneLake + /tmp file paths; constrain it to a strict
# allowlist (no path separators or dots) so a caller can't traverse out of
# Files/<km-dir>/ or /tmp.
_KM_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _check_km(km_name: str) -> str:
    if not _KM_NAME_RE.fullmatch(km_name or ""):
        raise ValueError("invalid kmName (allowed: letters, digits, _ -, 1-64 chars)")
    return km_name


@udf.connection(alias="kmLakehouse", argName="lakehouse")
@udf.function()
def search(
    lakehouse: fn.FabricLakehouseClient,
    kmName: str,
    query: str,
    topK: int = 10,
    verifiedOnly: bool = True,
) -> dict:
    """Single-shot retrieval against the named knowledge model.

    Returns {band: str, cold: bool, hits: [{passage_idx, source_file,
    char_offset, char_length, content_hash, drift_status, score, text,
    key}, ...]}. `cold=True` means this call paid the .rlat download +
    encoder load cost; warm calls are sub-100ms. `key` is the per-row
    business key for row-mode (slicer) knowledge models, else null. Hit
    field names are rlat-internal snake_case; parameter names above are
    camelCase per Fabric SDK rules.
    """
    _check_km(kmName)
    state, cold = bootstrap(lakehouse, kmName)
    return search_with_state(state, query, topK, verifiedOnly, cold=cold)


@udf.connection(alias="kmLakehouse", argName="lakehouse")
@udf.function()
def slice(
    lakehouse: fn.FabricLakehouseClient,
    kmName: str,
    query: str,
    topK: int = 200,
    verifiedOnly: bool = True,
) -> dict:
    """Semantic-slicer surface — a plain-language query → a business-KEY set.

    The Data App calls this, gets `keys`, and builds
    `TREATAS({keys}, 'Dim'[Key])` for the Execute DAX Queries API so the
    semantic model recomputes its measures over the matched rows.

    Returns {band, cold, keys: [<key>, ...], hits: [{key, score, text,
    drift_status}, ...]}. `keys` is the deduped, score-ordered TREATAS
    payload; `hits` keeps each key's score + matched snippet (the
    ranked-confirm receipt). A larger `topK` default than `search` because
    a slicer wants the whole matched set, not a top-10 read.
    """
    _check_km(kmName)
    state, cold = bootstrap(lakehouse, kmName)
    return slice_with_state(state, query, topK, verified_only=verifiedOnly, cold=cold)


@udf.connection(alias="kmLakehouse", argName="lakehouse")
@udf.function()
def slice_stream(
    lakehouse: fn.FabricLakehouseClient,
    kmName: str,
    query: str,
    topK: int = 200,
    snippetTopN: int = 50,
) -> dict:
    """Semantic slicer over the FULL corpus — single-file, rlat-native, OOM-safe.

    `slice()` opens the bundled .rlat through `bootstrap()`, which loads the
    whole vector band AND deserializes the FAISS index — peaking at ~1.5 GB on
    the 94k London corpus and OOMing the worker (see
    `.claude/plans/slicer-reset-2026-06-07.md`). This streams the base band one
    row-chunk at a time straight out of the `.rlat` (`store.streaming` via
    `slice_stream_native`). On the FIRST call the band is decompressed once to an
    uncompressed `/tmp` `.npy` and mmap'd; every warm query is then encode + one
    GEMV over the mmap (~ms, measured) — no per-call decompress, no key re-parse,
    no band in the process heap. Exact full scan, ranked by `(cosine, key)`
    descending. **Single file**: vectors + keys both come from the one `.rlat` — no
    parquet sidecar, no FAISS, no SQL.

    Returns {keys: [...], hits: [{key, score, text}, ...]} — `keys` is the deduped,
    score-ordered TREATAS payload the Data App passes to the Execute DAX Queries API;
    the top `snippetTopN` hits also carry `text` (the matched description, the slice
    receipt, read from the bundled source/ in the same .rlat). The matched aggregates
    come from the semantic model via DAX; the snippet shows WHY each row matched.
    """
    _check_km(kmName)
    return slice_stream_native(lakehouse, kmName, query, topK, snippetTopN)


@udf.connection(alias="kmLakehouse", argName="lakehouse")
@udf.function()
def embed(lakehouse: fn.FabricLakehouseClient, query: str) -> list:
    """Return the 768-d L2-normalised CLS embedding for `query`.

    Called by Fabric SQL DB's dbo.rlat_search stored procedure to embed
    the user query before VECTOR_DISTANCE ranking.

    Wire envelope (Fabric UDF wraps every return):
        {functionName, status, output, errors}
    Direct HTTPS callers (e.g. rlat search fabric://) read `output`.
    Stored procedure callers go through sp_invoke_external_rest_endpoint
    which wraps the body under `$.result`, so the SQL-side path is
    `JSON_QUERY(@response, '$.result.output')`. Don't change the return
    type without updating both consumer envelopes.
    """
    return embed_query(lakehouse, query)


@udf.connection(alias="kmLakehouse", argName="lakehouse")
@udf.function()
def list_kms(lakehouse: fn.FabricLakehouseClient) -> list:
    """Discovery endpoint — enumerate the .rlats present in Files/rlat/.

    Returns [{kmName, n_passages, created_utc, encoder_revision}, ...].
    """
    return list_kms_for(lakehouse)


@udf.function()
def probe_runtime() -> dict:
    """Diagnostic — what does the UDF Python worker filesystem look like?

    Asks: is `/lakehouse/default/Files/` available as a POSIX mount the
    way it is in Fabric notebooks? If yes, the encoder cache can live
    there (persistent, workspace-scoped) instead of `/tmp/rlat`
    (ephemeral, lost on container recycle).

    Returns enough information to decide whether the cheap one-line
    cold-start fix (XDG_CACHE_HOME = OneLake path) works, vs needing
    the explicit `lakehouse.connectToFiles()` download fallback.

    Pure stdlib — no rlat imports — so this stays callable even on
    early cold-start before any encoder work runs.
    """
    import os
    import platform
    import socket
    import sys
    import tempfile
    import time

    paths_to_check = [
        "/lakehouse/default",
        "/lakehouse/default/Files",
        "/lakehouse/default/Files/.rlat-cache",
        "/lakehouse/default/Files/rlat",
        "/tmp",
        "/tmp/rlat",
        os.environ.get("XDG_CACHE_HOME", ""),
        os.environ.get("HOME", ""),
    ]

    fs_findings: list[dict] = []
    for p in paths_to_check:
        if not p:
            continue
        entry: dict = {"path": p, "exists": os.path.exists(p)}
        if entry["exists"]:
            try:
                entry["is_dir"] = os.path.isdir(p)
                entry["readable"] = os.access(p, os.R_OK)
                entry["writable"] = os.access(p, os.W_OK)
                if entry["is_dir"]:
                    try:
                        entry["listdir_sample"] = sorted(os.listdir(p))[:8]
                    except Exception as e:
                        entry["listdir_error"] = f"{type(e).__name__}: {str(e)[:100]}"
            except Exception as e:
                entry["stat_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        fs_findings.append(entry)

    # Try to write + read a file under /lakehouse/default/Files/ — the
    # load-bearing question. If this works, B1 (XDG_CACHE_HOME -> OneLake)
    # is the fix. If it errors, we need B2 (explicit lakehouse client).
    write_probe: dict = {"target": "/lakehouse/default/Files/_rlat_probe.txt"}
    try:
        os.makedirs("/lakehouse/default/Files", exist_ok=True)
        token = f"probe-{int(time.time())}-{os.getpid()}"
        with open(write_probe["target"], "w", encoding="utf-8") as f:
            f.write(token)
        with open(write_probe["target"], "r", encoding="utf-8") as f:
            read_back = f.read()
        write_probe["round_trip_ok"] = (read_back == token)
        write_probe["token"] = token
        try:
            os.unlink(write_probe["target"])
            write_probe["cleanup_ok"] = True
        except Exception as e:
            write_probe["cleanup_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    except Exception as e:
        write_probe["error"] = f"{type(e).__name__}: {str(e)[:150]}"

    return {
        "python":         sys.version,
        "platform":       platform.platform(),
        "hostname":       socket.gethostname(),
        "pid":            os.getpid(),
        "cwd":            os.getcwd(),
        "tmp_dir":        tempfile.gettempdir(),
        "env": {
            "XDG_CACHE_HOME":          os.environ.get("XDG_CACHE_HOME"),
            "HOME":                    os.environ.get("HOME"),
            "RLAT_FABRIC_KM_DIR":      os.environ.get("RLAT_FABRIC_KM_DIR"),
        },
        "fs":             fs_findings,
        "onelake_write":  write_probe,
    }
