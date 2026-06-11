"""Microsoft Fabric UDF integration — server-side runtime helpers.

Imported by the `fabric/udf/function_app.py` template. Stays import-clean
without `fabric.functions` installed (Protocol shim in bootstrap.py) so the
helpers are unit-testable outside the Fabric runtime.

Public surface:
- `bootstrap(lakehouse, km_name)` — cold-start init or warm cache lookup.
  Returns `((contents, store, encoder), cold)` where `cold=True` means this
  call paid the download + load cost. Cache-busts on OneLake mtime change;
  LRU(8) over (km_name, mtime).
- `search_with_state(state, query, top_k, verified_only, *, cold=False)` —
  runs retrieval using a state tuple from `bootstrap()`. Returns the
  JSON-serialisable `{band, cold, hits}` shape.
- `embed_query(lakehouse, query)` — returns the 768-d L2-normalised CLS
  embedding as a Python list of floats. Called by Fabric SQL DB's
  `dbo.rlat_search` stored procedure via `sp_invoke_external_rest_endpoint`.
  Reuses any warm `Encoder` from `_STATE`; falls back to a revision-keyed
  encoder cache populated by peeking the first deployed `.rlat`'s
  `metadata.json` (no full archive read).
- `list_kms_for(lakehouse)` — globs `Files/rlat/*.rlat` and returns
  `[{kmName, n_passages, created_utc, encoder_revision}, ...]`. Powers the
  KM-discovery endpoint.
- `OneLakeStore` — `Store` subclass for local-mode `.rlat`s whose
  source bytes live under `Files/`. Read-only.
- `errors.FabricSetupError` — raised when the bound Lakehouse layout doesn't
  match what the UDF expects (`Files/rlat/<km>.rlat`).

Build / refresh live in the Fabric notebook (`fabric_build.ipynb`), not
in the UDF — the encoder's resident memory exceeds the Python-worker
ceiling on real corpora. Notebook builds use the same `build_rlat` /
`refresh_rlat` Python API exposed at the package root.

The encoder is fetched from HuggingFace at the revision pinned by the
`.rlat`'s `metadata.backbone.revision` — no per-workspace tarball staging.
"""

from ._runtime import (
    bootstrap,
    embed_query,
    list_kms_for,
    search_with_state,
    slice_stream_native,
    slice_with_state,
)
from .errors import FabricSetupError
from .onelake_store import OneLakeStore

__all__ = [
    "FabricSetupError",
    "OneLakeStore",
    "bootstrap",
    "embed_query",
    "list_kms_for",
    "search_with_state",
    "slice_stream_native",
    "slice_with_state",
]
