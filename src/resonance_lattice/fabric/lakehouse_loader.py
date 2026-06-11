"""OneLake `.rlat` reads via the Fabric-injected lakehouse client.

`FabricLakehouseClient.connectToFiles()` returns a
`DataLakeDirectoryClient` rooted at the Lakehouse's `Files/`. .rlats
live under `Files/<RLAT_FABRIC_KM_DIR>/` (default `rlat`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .errors import FabricSetupError

_DEFAULT_KM_DIR = "rlat"
_KM_DIR_ENV = "RLAT_FABRIC_KM_DIR"
_RLAT_SUFFIX = ".rlat"


def _km_dir() -> str:
    return os.environ.get(_KM_DIR_ENV, _DEFAULT_KM_DIR).strip("/")


def _onelake_path(km_name: str) -> str:
    return f"{_km_dir()}/{km_name}{_RLAT_SUFFIX}"


def _file_client(lakehouse: Any, km_name: str) -> Any:
    return lakehouse.connectToFiles().get_file_client(_onelake_path(km_name))


def get_rlat_mtime(lakehouse: Any, km_name: str) -> str:
    """Stat the OneLake .rlat and return `last_modified` as ISO 8601."""
    try:
        props = _file_client(lakehouse, km_name).get_file_properties()
    except Exception as e:
        raise FabricSetupError(
            f"could not stat Files/{_onelake_path(km_name)} on the bound "
            f"Lakehouse — verify the .rlat is uploaded ({type(e).__name__}: {e})"
        ) from e
    return props.last_modified.isoformat()


def fetch_rlat(lakehouse: Any, km_name: str, dst_root: Path) -> Path:
    """Download `Files/<km-dir>/<km>.rlat` to `<dst_root>/<km>.rlat` if missing.

    Streams the download to disk in chunks rather than `readall()` — a 617 MB
    `.rlat` `readall()`'d into a `bytes` is a ~617 MB transient that, stacked on
    a warm encoder (~542 MB), spikes the worker toward the OOM the streaming
    serve exists to avoid. Chunked download keeps the cold-start peak at one
    block. Disk-bound either way; only the RAM transient changes.
    """
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / f"{km_name}{_RLAT_SUFFIX}"
    if dst.exists():
        return dst
    fc = _file_client(lakehouse, km_name)
    downloader = fc.download_file()
    with open(dst, "wb") as f:
        for chunk in downloader.chunks():  # StorageStreamDownloader.chunks() — block at a time
            f.write(chunk)
    return dst


def list_km_paths(lakehouse: Any) -> list[str]:
    """List `.rlat` files under the configured KM directory and return
    their stems (no `.rlat` suffix).

    Fabric's wrapped `DataLakeDirectoryClient` is stripped: no `get_paths`,
    no `get_file_system_client`. The probe confirmed only the ADLS Gen2
    List Paths REST endpoint works. The SDK client supplies the URL
    parts and a `CustomTokenCredential` that mints storage-scope tokens.
    """
    sub = lakehouse.connectToFiles().get_sub_directory_client(_km_dir())
    names = _list_via_onelake_rest(sub)
    out: list[str] = []
    for full in names:
        basename = full.rsplit("/", 1)[-1]
        if basename.endswith(_RLAT_SUFFIX):
            out.append(basename[: -len(_RLAT_SUFFIX)])
    return sorted(set(out))


def _list_via_onelake_rest(sub: Any) -> list[str]:
    """Return file `name` strings under `sub`'s directory via the ADLS
    Gen2 List Paths REST endpoint.

    Factored out so tests can patch the REST call without standing up
    an HTTP server. Directories (`isDirectory == "true"`) are skipped.

    Uses `requests.Session(trust_env=False)` to mirror the Azure SDK's
    transport: Fabric Python-worker containers set HTTPS_PROXY for the
    process, and that proxy returns `404` on `CONNECT` to OneLake hosts.
    The SDK bypasses it; we do too.
    """
    import urllib.parse

    import requests

    host = "/".join(sub.url.split("/")[:3])  # https://<account-host>
    list_url = (
        f"{host}/{sub.file_system_name}?resource=filesystem"
        f"&directory={urllib.parse.quote(sub.path_name, safe='/')}"
        f"&recursive=false"
    )
    token = sub.credential.get_token("https://storage.azure.com/.default").token
    session = requests.Session()
    session.trust_env = False  # ignore HTTPS_PROXY/NO_PROXY
    response = session.get(
        list_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    return [
        p["name"]
        for p in body.get("paths", [])
        if p.get("isDirectory") != "true"
    ]
