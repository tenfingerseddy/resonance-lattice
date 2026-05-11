"""fabric_bootstrap — UDF runtime helper contract.

Nine guarantees:

  1. Cold bootstrap downloads the .rlat, fetches the encoder by the
     revision pinned in metadata.backbone.revision, opens the store,
     and returns ((contents, store, encoder), cold=True).
  2. Warm bootstrap on unchanged mtime is a cache hit (no re-download,
     no re-encode); returns cold=False.
  3. Upstream mtime change evicts the cached state, deletes the local
     .rlat, and re-bootstraps from the new bytes (cold=True again).
  4. LRU(8) cap: a 9th distinct (km, mtime) evicts the oldest entry.
  5. list_kms_for returns one row per Files/rlat/*.rlat with kmName,
     n_passages, created_utc, encoder_revision.
  6. Missing .rlat raises FabricSetupError with a clear message.
  7. .rlat with empty metadata.backbone.revision raises RuntimeError
     (catches old builds that didn't pin the encoder).
  8. cold flag toggles deterministically across cache states.
  9. OneLakeStore reads source bytes via the bound lakehouse client,
     reports FileNotFoundError on missing files, and rejects non-Fabric
     source_root paths with FabricSetupError.

The pre-2.1.0a13 build path (OneLakeSourceWalker + udf_build/udf_refresh)
moved out of the UDF entirely — the encoder weights resident in memory
push past the Fabric Python-worker ceiling. Build / refresh now live in
the Fabric notebook (`fabric_build.ipynb`) and use the rlat Python API
directly against the lakehouse mount. The harness coverage for that
flow is `build_pipeline` (which exercises `build_rlat` / `refresh_rlat`
end-to-end via `FilesystemSourceWalker`).

Mocks the FabricLakehouseClient via duck typing — the real Fabric SDK
class isn't importable here, but the helpers only need .connectToFiles()
to return something with .get_file_client() and .get_sub_directory_client().
"""

from __future__ import annotations

import datetime
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import build_corpus as _build
from ._testutil import check_guarantee, patch_zero_encoder


class _MockFileProperties:
    def __init__(self, last_modified: datetime.datetime) -> None:
        self.last_modified = last_modified


class _MockFileClient:
    """Quacks like azure.storage.filedatalake.DataLakeFileClient.

    Holds a back-reference to the parent directory's files dict so
    `upload_data` (the path the build/refresh handlers + telemetry writer
    take) mutates the source of truth. `download_file` and
    `get_file_properties` raise `FileNotFoundError` on missing paths;
    `get_file_client` itself is local (no I/O) and never raises.
    """

    def __init__(
        self,
        files: dict[str, tuple[bytes, datetime.datetime]],
        full_path: str,
    ) -> None:
        self._files = files
        self._path = full_path

    def get_file_properties(self) -> _MockFileProperties:
        if self._path not in self._files:
            raise FileNotFoundError(self._path)
        _, mtime = self._files[self._path]
        return _MockFileProperties(mtime)

    def download_file(self) -> "_MockDownload":
        if self._path not in self._files:
            raise FileNotFoundError(self._path)
        payload, _ = self._files[self._path]
        return _MockDownload(payload)

    def upload_data(self, data: bytes, overwrite: bool = True) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)
        self._files[self._path] = (bytes(data), now)


class _MockDownload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def readall(self) -> bytes:
        return self._payload


class _MockDirectoryClient:
    """Quacks like the stripped DataLakeDirectoryClient the Fabric UDF
    runtime injects: get_file_client + get_sub_directory_client only.
    Listing isn't on this surface — production goes via OneLake REST."""

    def __init__(self, files: dict[str, tuple[bytes, datetime.datetime]],
                 prefix: str = "", path_name: str = "") -> None:
        self._files = files
        self._prefix = prefix
        self.path_name = path_name

    def get_file_client(self, path: str) -> _MockFileClient:
        full = f"{self._prefix}{path}" if self._prefix else path
        return _MockFileClient(self._files, full)

    def get_sub_directory_client(self, name: str) -> "_MockDirectoryClient":
        return _MockDirectoryClient(
            self._files, prefix=f"{self._prefix}{name.strip('/')}/",
            path_name=f"{self.path_name}/{name.strip('/')}".strip("/"),
        )


class _MockLakehouse:
    def __init__(self, files: dict[str, tuple[bytes, datetime.datetime]]) -> None:
        self._dir = _MockDirectoryClient(files)

    def connectToFiles(self) -> _MockDirectoryClient:
        return self._dir


def _km_bytes(km_path: Path) -> bytes:
    return km_path.read_bytes()


def _build_km(root: Path, name: str = "team-docs") -> Path:
    """Build a tiny bundled `.rlat` under `root` and return its path.

    Bundled because the UDF runtime resolves local-mode `.rlat`s through
    OneLakeStore, which requires a `/Files/` mount path that tempdirs
    don't have. Bundled `.rlat`s read source bytes from the archive
    itself — same shape Kane's python-stdlib.rlat uses in production.
    """
    files = {
        "a.md": "# Auth\n\nLogin via SSO. Sessions persist 24 hours.",
        "b.md": "# Storage\n\nDocs land in OneLake. Index lives in .rlat.",
    }
    return _build(root / name, files, mode="bundled")


def _patch_bootstrap_module(km_cache: Path):
    """Bind the runtime module to a temp KM cache + a no-op HF fetch."""
    import resonance_lattice.fabric._runtime as bs
    from ._testutil import ZeroEncoder

    bs._KM_CACHE_DIR = km_cache
    bs._STATE.clear()
    bs.Encoder = ZeroEncoder  # type: ignore[assignment]
    bs.fetch_encoder_from_hf = lambda revision: km_cache  # type: ignore[assignment]
    return bs


def _check(ok: bool, label: str) -> bool:
    return check_guarantee(ok, label, "fabric_bootstrap")


def run() -> int:
    patch_zero_encoder()
    failures = 0

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        km_cache = root / "km_cache"
        bs = _patch_bootstrap_module(km_cache)

        km_path = _build_km(root)
        # Stamp a synthetic backbone revision into the just-built .rlat so
        # the bootstrap revision-pin path is exercised. cmd_build leaves it
        # blank when the encoder is the ZeroEncoder mock.
        from resonance_lattice.store import archive
        c = archive.read(km_path)
        c.metadata.backbone.revision = "test-rev-aaaaaaaa"
        archive.write(
            path=km_path,
            metadata=c.metadata,
            registry=c.registry,
            bands=c.bands,
            ann_blobs=c.ann_blobs,
            projections=c.projections,
        )

        payload = _km_bytes(km_path)
        t1 = datetime.datetime(2026, 5, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)
        files = {f"rlat/team-docs.rlat": (payload, t1)}
        lakehouse = _MockLakehouse(files)

        # ---- Guarantee 1: cold bootstrap returns a state tuple ----
        state, cold = bs.bootstrap(lakehouse, "team-docs")
        contents, store, enc = state
        passed_g1 = (
            cold is True
            and len(contents.registry) > 0
            and contents.metadata.backbone.revision == "test-rev-aaaaaaaa"
            and (km_cache / "team-docs.rlat").exists()
        )
        failures += not _check(passed_g1, "guarantee 1 (cold bootstrap)")

        # ---- Guarantee 2: warm bootstrap is a cache hit ----
        prev_state_id = id(state)
        state2, cold2 = bs.bootstrap(lakehouse, "team-docs")
        passed_g2 = (
            cold2 is False
            and id(state2) == prev_state_id
            and len(bs._STATE) == 1
        )
        failures += not _check(passed_g2, "guarantee 2 (warm cache hit)")

        # ---- Guarantee 3: mtime drift evicts and re-bootstraps ----
        # New mtime → new bytes (rebuild the corpus, give it different content).
        files2_root = root / "v2"
        km2 = _build_km(files2_root, "team-docs-v2")
        c2 = archive.read(km2)
        c2.metadata.backbone.revision = "test-rev-bbbbbbbb"
        archive.write(
            path=km2,
            metadata=c2.metadata,
            registry=c2.registry,
            bands=c2.bands,
            ann_blobs=c2.ann_blobs,
            projections=c2.projections,
        )
        t2 = t1 + datetime.timedelta(hours=1)
        lakehouse._dir._files["rlat/team-docs.rlat"] = (km2.read_bytes(), t2)

        state3, cold3 = bs.bootstrap(lakehouse, "team-docs")
        contents3, _, _ = state3
        passed_g3 = (
            cold3 is True
            and id(state3) != prev_state_id
            and contents3.metadata.backbone.revision == "test-rev-bbbbbbbb"
            and len(bs._STATE) == 1  # old entry evicted
        )
        failures += not _check(passed_g3, "guarantee 3 (mtime drift evicts)")

        # ---- Guarantee 4: LRU(8) cap ----
        # Bootstrap 9 distinct kms; the first one should be evicted.
        bs._STATE.clear()
        ts = []
        for i in range(9):
            km_i = _build_km(root / f"lru_{i}", f"km_{i}")
            ci = archive.read(km_i)
            ci.metadata.backbone.revision = f"rev-{i:08x}"
            archive.write(
                path=km_i,
                metadata=ci.metadata,
                registry=ci.registry,
                bands=ci.bands,
                ann_blobs=ci.ann_blobs,
                projections=ci.projections,
            )
            t_i = t1 + datetime.timedelta(minutes=i)
            ts.append(t_i)
            lakehouse._dir._files[f"rlat/km_{i}.rlat"] = (km_i.read_bytes(), t_i)
            bs.bootstrap(lakehouse, f"km_{i}")
        passed_g4 = (
            len(bs._STATE) == 8
            and ("km_0", ts[0].isoformat()) not in bs._STATE  # oldest evicted
            and ("km_8", ts[8].isoformat()) in bs._STATE      # newest present
        )
        failures += not _check(passed_g4, "guarantee 4 (LRU(8) eviction)")

        # ---- Guarantee 5: list_kms_for returns metadata rows ----
        # Production discovery hits OneLake REST. Patch _list_via_onelake_rest
        # to read the mock filesystem instead of doing real HTTP.
        import resonance_lattice.fabric.lakehouse_loader as _ll
        _orig_rest = _ll._list_via_onelake_rest

        def _fake_rest(sub):  # noqa: ARG001
            prefix = "rlat/"
            return [
                f"<lh>/Files/{k}"
                for k in lakehouse._dir._files
                if k.startswith(prefix)
            ]

        _ll._list_via_onelake_rest = _fake_rest
        try:
            rows = bs.list_kms_for(lakehouse)
        finally:
            _ll._list_via_onelake_rest = _orig_rest

        names = {r["kmName"] for r in rows}
        passed_g5 = (
            "team-docs" in names
            and "km_0" in names
            and all({"kmName", "n_passages", "created_utc", "encoder_revision"} <= r.keys()
                    for r in rows)
        )
        failures += not _check(passed_g5, "guarantee 5 (list_kms_for shape)")

        # ---- Guarantee 6: missing .rlat raises FabricSetupError ----
        from resonance_lattice.fabric.errors import FabricSetupError
        try:
            bs.bootstrap(lakehouse, "does-not-exist")
        except FabricSetupError:
            passed_g6 = True
        except Exception as e:
            print(f"[fabric_bootstrap] FAIL guarantee 6: wrong exception {type(e).__name__}: {e}",
                  file=sys.stderr)
            passed_g6 = False
        else:
            passed_g6 = False
        failures += not _check(passed_g6, "guarantee 6 (missing .rlat -> FabricSetupError)")

        # ---- Guarantee 7: empty backbone.revision raises RuntimeError ----
        km_blank = _build_km(root / "blank", "blank")
        # Don't stamp revision — leave the cmd_build default ("" with mocks).
        cb = archive.read(km_blank)
        passed_g7_setup = cb.metadata.backbone.revision == ""
        if not passed_g7_setup:
            cb.metadata.backbone.revision = ""
            archive.write(
                path=km_blank,
                metadata=cb.metadata,
                registry=cb.registry,
                bands=cb.bands,
                ann_blobs=cb.ann_blobs,
                projections=cb.projections,
            )
        t_blank = t1 + datetime.timedelta(hours=2)
        lakehouse._dir._files["rlat/blank.rlat"] = (km_blank.read_bytes(), t_blank)
        bs._STATE.clear()
        try:
            bs.bootstrap(lakehouse, "blank")
        except RuntimeError as e:
            passed_g7 = "backbone.revision" in str(e)
        except Exception as e:
            print(f"[fabric_bootstrap] FAIL guarantee 7: wrong exception {type(e).__name__}: {e}",
                  file=sys.stderr)
            passed_g7 = False
        else:
            passed_g7 = False
        failures += not _check(passed_g7, "guarantee 7 (empty revision -> RuntimeError)")

        # ---- Guarantee 8: cold flag toggles across cache states ----
        bs._STATE.clear()
        km_g8 = _build_km(root / "g8", "g8")
        c8 = archive.read(km_g8)
        c8.metadata.backbone.revision = "rev-g8000000"
        archive.write(
            path=km_g8,
            metadata=c8.metadata,
            registry=c8.registry,
            bands=c8.bands,
            ann_blobs=c8.ann_blobs,
            projections=c8.projections,
        )
        t_g8 = t1 + datetime.timedelta(hours=3)
        lakehouse._dir._files["rlat/g8.rlat"] = (km_g8.read_bytes(), t_g8)
        _, cold_first = bs.bootstrap(lakehouse, "g8")
        _, cold_second = bs.bootstrap(lakehouse, "g8")
        passed_g8 = cold_first is True and cold_second is False
        failures += not _check(passed_g8, "guarantee 8 (cold flag toggles)")

        # ---- Guarantee 9: OneLakeStore reads source bytes from the lakehouse ----
        from resonance_lattice.fabric.onelake_store import OneLakeStore
        from resonance_lattice.fabric.errors import FabricSetupError

        # Drop a source file into the mock OneLake under Files/docs/.
        text = "# Hello\n\nWorld."
        lakehouse._dir._files["docs/hello.md"] = (
            text.encode("utf-8"), t1 + datetime.timedelta(hours=4),
        )
        olstore = OneLakeStore(lakehouse, "/lakehouse/default/Files/docs")

        read_ok = olstore._read_full_text_uncached("hello.md") == text

        missing_ok = False
        try:
            olstore._read_full_text_uncached("does-not-exist.md")
        except FileNotFoundError:
            missing_ok = True
        except Exception as e:
            print(f"[fabric_bootstrap] FAIL guarantee 9 missing: wrong exception "
                  f"{type(e).__name__}: {e}", file=sys.stderr)

        bad_root_ok = False
        try:
            OneLakeStore(lakehouse, "/not/a/fabric/mount")
        except FabricSetupError:
            bad_root_ok = True

        passed_g9 = read_ok and missing_ok and bad_root_ok
        failures += not _check(passed_g9, "guarantee 9 (OneLakeStore reads)")

    if failures:
        print(f"[fabric_bootstrap] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[fabric_bootstrap] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
