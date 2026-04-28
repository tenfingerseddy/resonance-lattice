"""ZIP archive read/write — knowledge-model file format v4.

Renamed from `store/knowledge_model.py` to avoid collision with the top-level
`resonance_lattice.knowledge_model` orchestrator module. This file owns the
on-disk format; the top-level module owns the runtime handle.

A .rlat file is a ZIP archive (ZIP_STORED — no internal compression: NPZ files
are already deflate-compressed and ZIP-on-ZIP wastes CPU). Layout per
KNOWLEDGE_MODEL_FORMAT.md:

  my-corpus.rlat (ZIP archive)
  ├── metadata.json          -- backbone + bands registry + build_config
  ├── passages.jsonl         -- one JSON object per passage, line-implicit idx
  ├── bands/
  │   ├── base.npz           -- (N, 768) L2-normalised
  │   ├── optimised.npz     -- (N, 512) optional, after rlat optimise
  │   └── optimised_W.npz   -- (512, 768) MRL projection, paired w/ optimised
  ├── ann/
  │   ├── base.faiss         -- FAISS HNSW index for base band (when N > 5000)
  │   └── optimised.faiss   -- FAISS HNSW index for optimised band (when present)
  └── source/                -- only if metadata.store_mode=bundled
      └── ...                -- zstd-compressed source files, flat layout

Atomic write: tmp file in the same directory + `os.replace`. A crash mid-write
leaves the original (or absence) untouched. ANN blobs are passed as raw bytes
(serialised by field/ann.py via `faiss.serialize_index`) so this module stays
library-agnostic — a Phase 7+ ANN swap doesn't touch this file.

Phase 2 deliverable. Base plan §2.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bands as bands_io
from . import registry as registry_io
from .metadata import FORMAT_VERSION, BandInfo, Metadata, from_json, to_json

# Public layout constants — `store.bundled` reads from `SOURCE_DIR`,
# build pipelines write under it. Single source of truth for the v4 ZIP layout.
SOURCE_DIR = "source/"

_METADATA_PATH = "metadata.json"
_PASSAGES_PATH = "passages.jsonl"
_BAND_DIR = "bands/"
_ANN_DIR = "ann/"
_ANN_SUFFIX = ".faiss"
# Remote-mode manifest: `{source_file: {"url": str, "sha256": str}}` mapping.
# Present iff metadata.store_mode == "remote". Lives at the top of the ZIP
# rather than embedded in metadata.json so a 50K-source-file manifest
# (~7 MB) doesn't bloat every metadata read.
_MANIFEST_PATH = "manifest.json"


@dataclass
class BandHandle:
    """Resolved band — what `ArchiveContents.select_band` returns.

    Carries the band tensor + (optional) MRL projection + (optional) raw
    ANN bytes. Callers deserialise the ANN blob via `field/ann.deserialize`
    on demand — keeps store/ library-agnostic so a future ANN swap doesn't
    touch this module.
    """
    name: str
    band: np.ndarray
    projection: np.ndarray | None
    ann_blob: bytes | None


@dataclass
class ArchiveContents:
    """Eagerly-loaded snapshot of a v4 .rlat archive.

    `bands` and `projections` are float32 arrays in memory. `ann_blobs` are
    raw bytes (deserialised on demand by `field/ann.py`). Source files
    (bundled mode) are NOT loaded here — Store classes open the ZIP again
    for lazy resolution. `remote_manifest` is the parsed `manifest.json`
    when `metadata.store_mode == "remote"`; empty dict otherwise.
    """
    metadata: Metadata
    registry: list[registry_io.PassageCoord]
    bands: dict[str, np.ndarray]
    projections: dict[str, np.ndarray] = field(default_factory=dict)
    ann_blobs: dict[str, bytes] = field(default_factory=dict)
    remote_manifest: dict[str, dict[str, str]] = field(default_factory=dict)

    def select_band(self, prefer: str | None = None) -> BandHandle:
        """Return the band that retrieval should run against.

        - `prefer="base"` enforces base-band-only (cross-knowledge-model ops
          per CLAUDE.md "compare always uses base band").
        - `prefer=None` picks optimised if present, else base — the default
          for in-corpus search where the in-corpus-trained band is preferred.
        - `prefer=<name>` picks an explicit band; raises `KeyError` if absent.
        """
        if prefer is not None:
            if prefer not in self.bands:
                raise KeyError(
                    f"band {prefer!r} not in this knowledge model "
                    f"(available: {sorted(self.bands)})"
                )
            name = prefer
        else:
            name = "optimised" if "optimised" in self.bands else "base"
        return BandHandle(
            name=name,
            band=self.bands[name],
            projection=self.projections.get(name),
            ann_blob=self.ann_blobs.get(name),
        )


def _band_path(name: str) -> str:
    return f"{_BAND_DIR}{name}.npz"


def _projection_path(name: str) -> str:
    return f"{_BAND_DIR}{name}_W.npz"


def _ann_path(name: str) -> str:
    return f"{_ANN_DIR}{name}{_ANN_SUFFIX}"


def read(path: str | Path) -> ArchiveContents:
    """Open a v4 .rlat ZIP and load metadata + registry + all bands eagerly.

    ANN blobs are returned as raw bytes (not deserialised) so this module
    stays library-agnostic. Source files (bundled mode) are not loaded —
    Store classes resolve them via a separate ZipFile open. Raises
    `ValueError` on format-version mismatch or missing required slots.
    """
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        meta_text = zf.read(_METADATA_PATH).decode("utf-8")
        metadata = from_json(meta_text)
        if metadata.format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version {metadata.format_version} in {p} "
                f"(this build expects v{FORMAT_VERSION}); "
                f"see docs/internal/KNOWLEDGE_MODEL_FORMAT.md for migration policy"
            )

        passages_text = zf.read(_PASSAGES_PATH).decode("utf-8")
        registry = registry_io.load_jsonl(passages_text.splitlines())

        bands: dict[str, np.ndarray] = {}
        projections: dict[str, np.ndarray] = {}
        for band_name, band_info in metadata.bands.items():
            if band_info.w_shape is not None:
                # Paired band+projection (optimised). load_optimised enforces
                # both-or-neither at the file level; we additionally require
                # them present here because metadata declared the band.
                band, w = bands_io.load_optimised(
                    zf, _band_path(band_name), _projection_path(band_name),
                )
                if band is None or w is None:
                    raise ValueError(
                        f"metadata declares band '{band_name}' with w_shape but "
                        f"archive is missing the band/projection NPZ files"
                    )
                bands[band_name] = band
                projections[band_name] = w
            else:
                bands[band_name] = bands_io.load_base(zf, _band_path(band_name))

        ann_blobs: dict[str, bytes] = {}
        for name in zf.namelist():
            if name.startswith(_ANN_DIR) and name.endswith(_ANN_SUFFIX):
                ann_band_name = name[len(_ANN_DIR):-len(_ANN_SUFFIX)]
                ann_blobs[ann_band_name] = zf.read(name)

        remote_manifest: dict[str, dict[str, str]] = {}
        if metadata.store_mode == "remote":
            if _MANIFEST_PATH not in zf.namelist():
                raise ValueError(
                    f"{p} declares store_mode='remote' but is missing "
                    f"{_MANIFEST_PATH} — archive is corrupt or built by an "
                    f"older tool that didn't emit the manifest"
                )
            remote_manifest = json.loads(zf.read(_MANIFEST_PATH).decode("utf-8"))

    return ArchiveContents(
        metadata=metadata,
        registry=registry,
        bands=bands,
        projections=projections,
        ann_blobs=ann_blobs,
        remote_manifest=remote_manifest,
    )


def write(
    path: str | Path,
    metadata: Metadata,
    bands: dict[str, np.ndarray],
    registry: list[registry_io.PassageCoord],
    projections: dict[str, np.ndarray] | None = None,
    ann_blobs: dict[str, bytes] | None = None,
    source_files: dict[str, bytes] | None = None,
    remote_manifest: dict[str, dict[str, str]] | None = None,
) -> None:
    """Write a fresh v4 .rlat ZIP atomically.

    Atomic via temp file in the same directory + `os.replace`. A crash mid-write
    leaves the original (or absence) untouched. ZIP_STORED outer compression —
    NPZ files are already deflate-compressed.

    Inputs:
      - `metadata.format_version` must be v4 (caller's responsibility).
      - `bands` keys must match `metadata.bands` keys (same name in both).
      - `projections` is required for any band with `w_shape is not None`.
      - `source_files` keys are POSIX-style relative paths (e.g. "src/foo.py").

    Raises `ValueError` if metadata.bands and bands disagree, or if a paired
    band is missing its projection.
    """
    if metadata.format_version != FORMAT_VERSION:
        raise ValueError(
            f"metadata.format_version is {metadata.format_version}; "
            f"writer only emits v{FORMAT_VERSION}"
        )
    projections = projections or {}
    ann_blobs = ann_blobs or {}
    source_files = source_files or {}
    remote_manifest = remote_manifest or {}

    declared = set(metadata.bands.keys())
    provided = set(bands.keys())
    if declared != provided:
        raise ValueError(
            f"metadata.bands {sorted(declared)} disagrees with bands payload "
            f"{sorted(provided)}; declare every band in metadata before write"
        )
    for band_name, info in metadata.bands.items():
        if info.w_shape is not None and band_name not in projections:
            raise ValueError(
                f"band '{band_name}' declares w_shape={info.w_shape} but "
                f"projections payload is missing it"
            )
    if metadata.store_mode == "remote" and not remote_manifest:
        raise ValueError(
            "store_mode='remote' requires a non-empty remote_manifest "
            "({source_file: {url, sha256}} mapping); none provided"
        )
    if remote_manifest and metadata.store_mode != "remote":
        raise ValueError(
            f"remote_manifest provided but metadata.store_mode is "
            f"{metadata.store_mode!r}; the manifest only ships in remote mode"
        )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(p) + ".tmp")

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(_METADATA_PATH, to_json(metadata))
            zf.writestr(_PASSAGES_PATH, registry_io.write_jsonl(registry))

            for band_name, band_data in bands.items():
                bands_io.write_band(zf, _band_path(band_name), band_data)
            for band_name, w in projections.items():
                bands_io.write_projection(zf, _projection_path(band_name), w)

            for band_name, blob in ann_blobs.items():
                zf.writestr(_ann_path(band_name), blob)

            for src_rel_path, content in source_files.items():
                zf.writestr(f"{SOURCE_DIR}{src_rel_path}", content)

            if remote_manifest:
                # sort_keys for diff-stable output across rebuilds.
                zf.writestr(
                    _MANIFEST_PATH,
                    json.dumps(remote_manifest, sort_keys=True, indent=2),
                )
    except BaseException:
        # Original (or absence) is already untouched; just clean up the tmp
        # so we don't accumulate orphaned `.tmp` files on disk-full / kill.
        tmp_path.unlink(missing_ok=True)
        raise

    os.replace(tmp_path, p)


def write_band_in_place(
    path: str | Path,
    band_name: str,
    band_info: BandInfo,
    band_data: np.ndarray,
    projection: np.ndarray | None = None,
    ann_blob: bytes | None = None,
) -> None:
    """Add or replace a band slot in an existing v4 archive without rewriting
    unrelated slots. Used by `rlat optimise` to fill the optimised band +
    its projection (and optional ANN index) on a knowledge model whose base
    band was built earlier.

    Implementation: stream the existing archive into a tmp file, dropping any
    members that the new band displaces (band NPZ, projection NPZ, ANN blob,
    metadata) and writing fresh entries in their place. Atomically replaces
    the original via `os.replace`. Other slots (other bands, source/, registry,
    other ANN blobs, build_config) are copied unchanged.

    Phase 4 deliverable; the implementation is here so Phase 2 round-trip
    tests can exercise it before the optimise CLI lands.
    """
    p = Path(path)
    tmp_path = Path(str(p) + ".tmp")

    skipped = {
        _METADATA_PATH,
        _band_path(band_name),
        _projection_path(band_name),  # always skip; may not exist on the source
        _ann_path(band_name),         # ditto
    }

    with zipfile.ZipFile(p, "r") as src:
        meta_text = src.read(_METADATA_PATH).decode("utf-8")
        metadata = from_json(meta_text)
        if metadata.format_version != FORMAT_VERSION:
            raise ValueError(
                f"refuse to mutate v{metadata.format_version} archive; "
                f"in-place writer only handles v{FORMAT_VERSION}"
            )
        if band_info.w_shape is not None and projection is None:
            raise ValueError(
                f"band '{band_name}' declares w_shape={band_info.w_shape} but "
                f"no projection passed; pair with the projection or unset w_shape"
            )
        metadata.bands[band_name] = band_info

        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as dst:
                dst.writestr(_METADATA_PATH, to_json(metadata))
                # Stream preserved members chunk-by-chunk so peak memory stays
                # bounded by the read buffer, not the size of the largest
                # source file. Critical for bundled-mode archives where
                # `source/` may aggregate >1 GB; without streaming, optimise
                # would briefly materialise each member fully into RSS.
                for info in src.infolist():
                    if info.filename in skipped:
                        continue
                    with src.open(info.filename, "r") as fsrc, \
                         dst.open(info, "w", force_zip64=True) as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
                bands_io.write_band(dst, _band_path(band_name), band_data)
                if projection is not None:
                    bands_io.write_projection(dst, _projection_path(band_name), projection)
                if ann_blob is not None:
                    dst.writestr(_ann_path(band_name), ann_blob)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    os.replace(tmp_path, p)
