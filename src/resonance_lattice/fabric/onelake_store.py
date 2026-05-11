"""Store subclass that reads source bytes from OneLake.

Lets a local-mode `.rlat` work inside the Fabric UDF runtime: the archive
records source paths (not bytes), and this store resolves them through
the bound `FabricLakehouseClient` instead of the local filesystem.

The build notebook always passes `--source-root /lakehouse/default/Files/<SUBDIR>`
(the Fabric notebook mount). At UDF time we extract the `<SUBDIR>` portion
from that recorded mount path and prepend it to each `source_file` lookup.
"""

from __future__ import annotations

from typing import Any

from ..store.base import Store
from .errors import FabricSetupError

_FILES_MARKER = "/Files/"


class OneLakeStore(Store):
    """Reads source files from OneLake under `Files/<files_prefix>/`."""

    def __init__(self, lakehouse: Any, source_root: str) -> None:
        super().__init__()
        if _FILES_MARKER not in source_root:
            raise FabricSetupError(
                f"OneLake store cannot resolve source_root {source_root!r}: "
                f"expected a Fabric mount path like "
                f"'/lakehouse/default/Files/<subdir>'"
            )
        self._lakehouse = lakehouse
        self._files_prefix = source_root.split(_FILES_MARKER, 1)[1].strip("/")

    def _read_full_text_uncached(self, source_file: str) -> str:
        path = (
            f"{self._files_prefix}/{source_file}"
            if self._files_prefix
            else source_file
        )
        fc = self._lakehouse.connectToFiles().get_file_client(path)
        try:
            data = fc.download_file().readall()
        except Exception as e:
            # Surface as FileNotFoundError so Store.verify maps to "missing"
            # per the drift-status contract.
            if type(e).__name__ in ("ResourceNotFoundError", "FileNotFoundError"):
                raise FileNotFoundError(path) from e
            raise
        return data.decode("utf-8")
