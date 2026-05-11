# Resonance Lattice — Microsoft Fabric UDF entry point.
#
# Two functions exposed to callers:
#   - search(kmName, query, ...)   → top-K verified hits
#   - list_kms()                   → discovery endpoint, lists Files/rlat/*.rlat
#
# Both read the bound Lakehouse via the @udf.connection decorator. The
# kmLakehouse alias must be configured in the Fabric portal under Manage
# Connections; bind it to the Lakehouse holding Files/rlat/<kmName>.rlat.
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

# Force the rlat encoder cache onto /tmp before any rlat import — the Fabric
# UDF runtime's $HOME may be read-only, but /tmp is always writable.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rlat")

# Where .rlat files live under the bound Lakehouse's Files/ root.
# Default is `rlat` (matches the published convention in docs/user/FABRIC.md).
# Set this to `knowledge`, `data`, or any other subdir name if the workspace
# already uses a different layout.
os.environ.setdefault("RLAT_FABRIC_KM_DIR", "rlat")

import fabric.functions as fn
from resonance_lattice.fabric import bootstrap, list_kms_for, search_with_state

udf = fn.UserDataFunctions()


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
    char_offset, char_length, content_hash, drift_status, score, text},
    ...]}. `cold=True` means this call paid the .rlat download + encoder
    load cost; warm calls are sub-100ms. Hit field names are rlat-internal
    snake_case; parameter names above are camelCase per Fabric SDK rules.
    """
    state, cold = bootstrap(lakehouse, kmName)
    return search_with_state(state, query, topK, verifiedOnly, cold=cold)


@udf.connection(alias="kmLakehouse", argName="lakehouse")
@udf.function()
def list_kms(lakehouse: fn.FabricLakehouseClient) -> list:
    """Discovery endpoint — enumerate the .rlats present in Files/rlat/.

    Returns [{kmName, n_passages, created_utc, encoder_revision}, ...].
    """
    return list_kms_for(lakehouse)
