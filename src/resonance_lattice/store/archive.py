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
  │   └── base.npz           -- (N, 768) L2-normalised
  ├── ann/
  │   └── base.faiss         -- FAISS HNSW index for base band (when N > 5000)
  └── source/                -- only if metadata.store_mode=bundled
      └── ...                -- zstd-compressed source files, flat layout

Atomic write: per-writer-unique tmp file in the same directory + `os.replace`.
A crash mid-write leaves the original (or absence) untouched. The tmp filename
carries the writer's pid + random suffix so two processes mutating the same
archive don't collide on `{path}.tmp` and silently corrupt each other's
half-written ZIP. Cross-process lost-update risk (two writers' deltas
applied against the same pre-state) is *not* prevented — callers running a
long-lived mutator (e.g. `rlat watch`) alongside a one-shot mutator must
serialise themselves. ANN blobs are passed as raw bytes (serialised by
field/ann.py via `faiss.serialize_index`) so this module stays
library-agnostic — a Phase 7+ ANN swap doesn't touch this file.

Phase 2 deliverable. Base plan §2.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bands as bands_io
from . import corpus_claim_io
from . import registry as registry_io
from ..state.claim import Claim
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
# Insight layer (lensed knowledge, additive; absent from pre-v2.1 archives).
# JSONL file alongside passages.jsonl; band lives under `bands/insight.npz`
# and is registered as a band named "insight" in metadata.bands so the
# existing band-loading machinery picks it up without special-casing.
_INSIGHT_PATH = "insight.jsonl"
INSIGHT_BAND_NAME = "insight"
# Telemetry log (Insight Engine self-improvement loop; capture.md §3,
# architecture.md §7). Append-only JSONL: one redacted `field.capture`
# observation row per line ({ts, session, layer, is_user_query, query_emb,
# ranked} — a query *fingerprint* + per-rank scores, NEVER the query text).
# Lives under `insight/` so the engine's runtime telemetry groups apart from
# the promoted-claims member `insight.jsonl`. This member is the ZIP binding of
# a format-agnostic contract: the planned SQLite format re-home (capture.md §3)
# swaps `read_telemetry`/`append_telemetry_in_place` for table ops; the row
# format, the fold caller, and every reader stay unchanged.
_TELEMETRY_PATH = "insight/telemetry.jsonl"

# The corpus self-audit — the foundational, LLM-free shape-report (contradiction
# candidates + drift + gaps) computed at build/refresh by `store.self_audit`. A
# single JSON member (not append-only like telemetry — it is recomputed whole), so
# every `.rlat` carries its own map of where it's empty / contradicts / is stale.
_SELF_AUDIT_PATH = "insight/self_audit.json"


@dataclass
class BandHandle:
    """Resolved band — what `ArchiveContents.select_band` returns.

    Carries the band tensor + (optional) raw ANN bytes. Callers deserialise
    the ANN blob via `field/ann.deserialize` on demand — keeps store/
    library-agnostic so a future ANN swap doesn't touch this module.
    """
    name: str
    band: np.ndarray | None  # None on a defer_base_band read (ANN serves the query)
    ann_blob: bytes | None
    # The resolved path of the .rlat this band came from, as a string —
    # the corpus identity the capture heart keys telemetry by + folds into
    # (`field.retrieve`, capture.md §3). None when the handle was built
    # without a source path (a synthetic/in-memory archive).
    km_id: str | None = None


@dataclass
class ArchiveContents:
    """Loaded snapshot of a v4 .rlat archive (eager by default).

    `bands` are float32 arrays in memory — except the `base` band is omitted
    from the dict when `read(defer_base_band=True)` and an ANN index serves
    retrieval (the dict is never None-valued; only `BandHandle.band` is None).
    `ann_blobs` are
    raw bytes (deserialised on demand by `field/ann.py`). Source files
    (bundled mode) are NOT loaded here — Store classes open the ZIP again
    for lazy resolution. `remote_manifest` is the parsed `manifest.json`
    when `metadata.store_mode == "remote"`; empty dict otherwise.

    `insights` is the loaded insight layer (lensed knowledge); empty list
    when the archive has no insight.jsonl entry (pre-v2.1 or no insight
    yet promoted). The insight band, when present, lives under the same
    `bands[INSIGHT_BAND_NAME]` slot as any other band.
    """
    metadata: Metadata
    registry: list[registry_io.PassageCoord]
    bands: dict[str, np.ndarray]
    ann_blobs: dict[str, bytes] = field(default_factory=dict)
    remote_manifest: dict[str, dict[str, str]] = field(default_factory=dict)
    insights: list[Claim] = field(default_factory=list)
    # The resolved path this archive was read from, as the corpus identity
    # the capture heart keys telemetry by (stamped by `read`; None for a
    # synthetic/in-memory archive). Propagated onto every BandHandle.
    source_path: Path | None = None

    def insight_band(self) -> BandHandle | None:
        """Return the insight band's BandHandle, or None if no insight layer.

        Parallel to `select_band` for the source layer. Encapsulates the
        `bands[INSIGHT_BAND_NAME]` / `ann_blobs.get(INSIGHT_BAND_NAME)`
        lookup pattern so callers don't reach into the raw dicts.
        """
        if INSIGHT_BAND_NAME not in self.bands:
            return None
        return BandHandle(
            name=INSIGHT_BAND_NAME,
            band=self.bands[INSIGHT_BAND_NAME],
            ann_blob=self.ann_blobs.get(INSIGHT_BAND_NAME),
            km_id=str(self.source_path) if self.source_path else None,
        )

    def select_band(self, prefer: str | None = None) -> BandHandle:
        """Return the band that retrieval should run against.

        `base` is the only retrieval band, so `prefer=None` resolves to it.
        `prefer=<name>` picks an explicit band and raises `KeyError` if
        absent — `cli/compare.py` passes `prefer="base"` to make the
        cross-knowledge-model "compare always uses base band" rule explicit.
        """
        if prefer is not None:
            if prefer not in self.bands:
                raise KeyError(
                    f"band {prefer!r} not in this knowledge model "
                    f"(available: {sorted(self.bands)})"
                )
            name = prefer
        else:
            name = "base"
        return BandHandle(
            name=name,
            band=self.bands.get(name),  # None when deferred (prefer= path still strict above)
            ann_blob=self.ann_blobs.get(name),
            km_id=str(self.source_path) if self.source_path else None,
        )


def _unique_tmp_path(p: Path) -> Path:
    """`{p}.{pid}.{rand}.tmp` — unique per writer.

    Two processes mutating the same archive no longer share a tmp
    filename: neither's `ZipFile(tmp, "w")` truncates the other's
    in-progress bytes, neither's exception-handler `unlink` deletes
    the other's tmp, and neither's `os.replace(tmp, ...)` races on the
    same source path. Filenames keep the literal `.tmp` suffix so a
    future cleanup glob (none ships today) can still find orphans
    from SIGKILLed writers.
    """
    return Path(f"{p}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def _band_path(name: str) -> str:
    return f"{_BAND_DIR}{name}.npz"


def _ann_path(name: str) -> str:
    return f"{_ANN_DIR}{name}{_ANN_SUFFIX}"


def read(path: str | Path, *, defer_base_band: bool = False) -> ArchiveContents:
    """Open a v4 .rlat ZIP and load metadata + registry + bands.

    `defer_base_band=True` skips materialising the `base` band when it carries an
    ANN index — ANN-mode `field.retrieve` never dereferences `handle.band`, so
    the retrieval path (e.g. the Fabric UDF) avoids holding the full (N,768)
    matrix in RAM. The base band is still loaded when there is no ANN (dense
    retrieval needs it); compare/RQL use the default eager read.

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

        # Which bands carry an ANN index? Computed first so a deferred read can
        # skip the base band when ANN will serve the query (see defer_base_band).
        ann_band_names = {
            name[len(_ANN_DIR):-len(_ANN_SUFFIX)]
            for name in zf.namelist()
            if name.startswith(_ANN_DIR) and name.endswith(_ANN_SUFFIX)
        }

        bands: dict[str, np.ndarray] = {}
        for band_name in metadata.bands:
            # Lazy-band-skip: on a retrieval-path read, don't materialise the
            # base band when an ANN index serves it (ANN-mode field.retrieve
            # never reads handle.band) — avoids the full (N,768) matrix in RAM.
            if defer_base_band and band_name == "base" and "base" in ann_band_names:
                continue
            bands[band_name] = bands_io.load_base(zf, _band_path(band_name))

        ann_blobs: dict[str, bytes] = {
            n: zf.read(_ann_path(n)) for n in ann_band_names
        }

        remote_manifest: dict[str, dict[str, str]] = {}
        if metadata.store_mode == "remote":
            if _MANIFEST_PATH not in zf.namelist():
                raise ValueError(
                    f"{p} declares store_mode='remote' but is missing "
                    f"{_MANIFEST_PATH} — archive is corrupt or built by an "
                    f"older tool that didn't emit the manifest"
                )
            remote_manifest = json.loads(zf.read(_MANIFEST_PATH).decode("utf-8"))

        # Insight layer (lensed knowledge). Absent in pre-v2.1 archives and
        # in fresh post-build archives that haven't accumulated any promoted
        # insight yet. Both cases load as an empty list — no error.
        insights: list[Claim] = []
        if _INSIGHT_PATH in zf.namelist():
            insight_text = zf.read(_INSIGHT_PATH).decode("utf-8")
            if insight_text.strip():
                insights = corpus_claim_io.rows_to_claims(
                    insight_text.splitlines()
                )
        if insights and INSIGHT_BAND_NAME not in bands:
            raise ValueError(
                f"{p} has {_INSIGHT_PATH} with {len(insights)} insight rows "
                f"but no '{INSIGHT_BAND_NAME}' band declared in metadata.bands "
                f"— archive is half-promoted. Re-run consolidation or remove "
                f"{_INSIGHT_PATH}."
            )
        if insights and len(bands.get(INSIGHT_BAND_NAME, [])) != len(insights):
            raise ValueError(
                f"{p} insight band has {len(bands[INSIGHT_BAND_NAME])} rows "
                f"but insight.jsonl has {len(insights)} — half-written promotion"
            )

    return ArchiveContents(
        metadata=metadata,
        registry=registry,
        bands=bands,
        ann_blobs=ann_blobs,
        remote_manifest=remote_manifest,
        insights=insights,
        source_path=p.resolve(),
    )


def read_insight_layer(path: str | Path) -> tuple[list[Claim], np.ndarray] | None:
    """Load ONLY the insight layer (claims + band) from a v4 .rlat.

    `read` loads every band (the base band can be large) plus the registry and
    ANN blobs eagerly. The prompt-time band-recall path needs only the insight
    band + its claims, so this opens the ZIP and reads just `insight.jsonl` +
    `bands/insight.npz`, skipping the base band, registry, and ANN.

    Returns `(insights, insight_band)` with a positional row↔band join, or
    `None` when the archive has no insight layer (pre-v2.1, or nothing promoted
    yet). Raises `ValueError` on a half-written layer (`insight.jsonl` present
    but the band entry absent, or a row/band count mismatch).

    Keys off the actual ZIP entries (`insight.jsonl` + `bands/insight.npz`),
    not `metadata.bands` like `read` — so it joins the band file it actually
    read (the ground truth the recall path needs) and is fractionally more
    permissive than `read` on the pathological case of a present band blob that
    `metadata.bands` forgot to declare. A missing/corrupt file raises the
    underlying `zipfile`/OS error — the recall caller wraps this in its
    fail-open guard.
    """
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        names = zf.namelist()
        if _INSIGHT_PATH not in names:
            return None
        insight_text = zf.read(_INSIGHT_PATH).decode("utf-8")
        if not insight_text.strip():
            return None
        insights = corpus_claim_io.rows_to_claims(insight_text.splitlines())
        band_path = _band_path(INSIGHT_BAND_NAME)
        if band_path not in names:
            raise ValueError(
                f"{p} has {_INSIGHT_PATH} with {len(insights)} rows but no "
                f"'{INSIGHT_BAND_NAME}' band — archive is half-promoted"
            )
        band = bands_io.load_base(zf, band_path)
    if band.shape[0] != len(insights):
        raise ValueError(
            f"{p} insight band has {band.shape[0]} rows but insight.jsonl has "
            f"{len(insights)} — half-written promotion"
        )
    return insights, band


def _telemetry_to_bytes(rows: list[dict]) -> bytes:
    """Serialise telemetry rows to the `insight/telemetry.jsonl` member encoding.

    Single source of truth shared by `write` (full rewrite) and
    `append_telemetry_in_place` (incremental append) so the two paths can't drift.
    """
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def write(
    path: str | Path,
    metadata: Metadata,
    bands: dict[str, np.ndarray],
    registry: list[registry_io.PassageCoord],
    ann_blobs: dict[str, bytes] | None = None,
    source_files: dict[str, bytes] | None = None,
    remote_manifest: dict[str, dict[str, str]] | None = None,
    insights: list[Claim] | None = None,
    self_audit: dict | None = None,
    telemetry: list[dict] | None = None,
) -> None:
    """Write a fresh v4 .rlat ZIP atomically.

    Atomic via per-writer-unique tmp file + `os.replace` (see
    `_unique_tmp_path`). A crash mid-write leaves the original (or absence)
    untouched. ZIP_STORED outer compression — NPZ files are already
    deflate-compressed.

    Inputs:
      - `metadata.format_version` must be v4 (caller's responsibility).
      - `bands` keys must match `metadata.bands` keys (same name in both).
      - `source_files` keys are POSIX-style relative paths (e.g. "src/foo.py").

    Raises `ValueError` if metadata.bands and bands disagree.
    """
    if metadata.format_version != FORMAT_VERSION:
        raise ValueError(
            f"metadata.format_version is {metadata.format_version}; "
            f"writer only emits v{FORMAT_VERSION}"
        )
    ann_blobs = ann_blobs or {}
    source_files = source_files or {}
    remote_manifest = remote_manifest or {}
    insights = insights or []

    declared = set(metadata.bands.keys())
    provided = set(bands.keys())
    if declared != provided:
        raise ValueError(
            f"metadata.bands {sorted(declared)} disagrees with bands payload "
            f"{sorted(provided)}; declare every band in metadata before write"
        )
    if insights and INSIGHT_BAND_NAME not in bands:
        raise ValueError(
            f"insights provided ({len(insights)} rows) but no "
            f"'{INSIGHT_BAND_NAME}' band in bands payload — declare the band "
            f"in metadata.bands and supply its (M, D) array"
        )
    if insights and len(bands.get(INSIGHT_BAND_NAME, [])) != len(insights):
        raise ValueError(
            f"insight band has {len(bands[INSIGHT_BAND_NAME])} rows but "
            f"insights list has {len(insights)} — band-row join would break"
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
    tmp_path = _unique_tmp_path(p)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(_METADATA_PATH, to_json(metadata))
            zf.writestr(_PASSAGES_PATH, registry_io.write_jsonl(registry))

            for band_name, band_data in bands.items():
                bands_io.write_band(zf, _band_path(band_name), band_data)

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

            if insights:
                zf.writestr(
                    _INSIGHT_PATH, corpus_claim_io.claims_to_jsonl(insights)
                )

            # Preserve the append-only telemetry member across a full rewrite —
            # apply_delta (refresh/sync) and convert pass the existing rows so the
            # capture log is not silently dropped on rebuild.
            if telemetry:
                zf.writestr(_TELEMETRY_PATH, _telemetry_to_bytes(telemetry))

            if self_audit:
                # The corpus self-audit, folded into the SINGLE build write — same member + byte format as
                # `write_self_audit_in_place` (sort_keys), so build no longer needs a second ZIP rewrite to add it.
                zf.writestr(
                    _SELF_AUDIT_PATH, json.dumps(self_audit, sort_keys=True)
                )
        os.replace(tmp_path, p)
    except BaseException:
        # Original (or absence) is already untouched. Clean up the tmp
        # on any failure (write, close, or `os.replace`) so we don't
        # accumulate orphaned `.tmp` files on disk-full / kill / a
        # Windows `os.replace` collision with another concurrent writer.
        tmp_path.unlink(missing_ok=True)
        raise


def write_insight_layer_in_place(
    path: str | Path,
    insights: list[Claim],
    insight_band: np.ndarray,
    ann_blob: bytes | None = None,
    *,
    mark_reverified_utc: str | None = None,
) -> None:
    """Replace the insight layer (insight.jsonl + bands/insight.npz + optional
    ann/insight.faiss + metadata band registration) in an existing archive
    without rewriting unrelated slots.

    Used by the compression-test promotion pipeline (Day 4) every time
    consolidation graduates synthesis candidates to the corpus insight layer.
    Atomic via per-writer-unique tmp file + `os.replace` (see
    `_unique_tmp_path`). Source band, source registry, bundled source
    files, remote manifest, and any other bands are copied unchanged.

    Empty `insights` clears the insight layer entirely (insight.jsonl
    dropped; `INSIGHT_BAND_NAME` removed from metadata.bands; insight band
    NPZ and ANN files dropped).

    `mark_reverified_utc` — when non-empty, stamps the metadata's
    `insight_layer_last_reverify_utc` field. The reverification pass
    passes its own completion timestamp through; other callers leave
    it `None` (or pass an empty string) to preserve the existing
    heartbeat. No code path today clears the heartbeat by design.
    """
    if insights and len(insights) != insight_band.shape[0]:
        raise ValueError(
            f"insights list has {len(insights)} rows but band has "
            f"{insight_band.shape[0]} — band-row join would break"
        )

    p = Path(path)
    tmp_path = _unique_tmp_path(p)

    skipped = {
        _METADATA_PATH,
        _INSIGHT_PATH,
        _band_path(INSIGHT_BAND_NAME),
        _ann_path(INSIGHT_BAND_NAME),
    }

    with zipfile.ZipFile(p, "r") as src:
        meta_text = src.read(_METADATA_PATH).decode("utf-8")
        metadata = from_json(meta_text)
        if metadata.format_version != FORMAT_VERSION:
            raise ValueError(
                f"refuse to mutate v{metadata.format_version} archive; "
                f"in-place writer only handles v{FORMAT_VERSION}"
            )

        if insights:
            metadata.bands[INSIGHT_BAND_NAME] = BandInfo(
                role="insight_layer",
                dim=int(insight_band.shape[1]),
                l2_norm=True,
                passage_count=int(insight_band.shape[0]),
            )
        else:
            metadata.bands.pop(INSIGHT_BAND_NAME, None)

        if mark_reverified_utc:
            metadata.insight_layer_last_reverify_utc = mark_reverified_utc

        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as dst:
                dst.writestr(_METADATA_PATH, to_json(metadata))
                for info in src.infolist():
                    if info.filename in skipped:
                        continue
                    with src.open(info.filename, "r") as fsrc, \
                         dst.open(info, "w", force_zip64=True) as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
                if insights:
                    dst.writestr(
                        _INSIGHT_PATH,
                        corpus_claim_io.claims_to_jsonl(insights),
                    )
                    bands_io.write_band(
                        dst, _band_path(INSIGHT_BAND_NAME), insight_band,
                    )
                    if ann_blob is not None:
                        dst.writestr(_ann_path(INSIGHT_BAND_NAME), ann_blob)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        os.replace(tmp_path, p)
    except BaseException:
        # Windows: a concurrent writer's `os.replace` landing first can
        # cause this one to raise PermissionError. Orphan-cleanup keeps
        # the on-disk surface tidy under contention.
        tmp_path.unlink(missing_ok=True)
        raise


def read_telemetry(path: str | Path) -> list[dict]:
    """Load the append-only telemetry rows from a v4 .rlat (the Insight Engine
    capture log; the ZIP side of the `store.telemetry` seam).

    Returns `[]` when the archive has no telemetry member yet (a pre-Engine
    corpus, or one that has never folded a session). Each row is one
    `field.capture`-shaped dict. Decoded tolerantly (`errors="replace"`) and
    parsed per-line, so a byte-corrupt or malformed line is skipped rather than
    raising — a torn or foreign-written member never makes the whole log
    unreadable (the reader degrades to the good rows). The ZIP itself opening is
    the caller's risk (a missing/corrupt archive raises the underlying
    `zipfile`/OS error; `store.telemetry.read` wraps it).
    """
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        if _TELEMETRY_PATH not in zf.namelist():
            return []
        text = zf.read(_TELEMETRY_PATH).decode("utf-8", errors="replace")
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_self_audit(path: str | Path) -> dict:
    """The stored self-audit report (`insight/self_audit.json`) of a `.rlat`, or `{}` when absent/unreadable.

    Never raises: a missing member, a corrupt archive, or unparseable JSON yields `{}` so a reader degrades to
    "no audit yet" rather than crashing."""
    p = Path(path)
    try:
        with zipfile.ZipFile(p, "r") as zf:
            if _SELF_AUDIT_PATH not in zf.namelist():
                return {}
            return json.loads(zf.read(_SELF_AUDIT_PATH).decode("utf-8", errors="replace"))
    except Exception:
        return {}


def write_self_audit_in_place(path: str | Path, report: dict) -> None:
    """Write (replace) the `insight/self_audit.json` member of an existing v4 `.rlat`, preserving every other slot.

    Mirrors `append_telemetry_in_place`'s atomic per-writer-tmp + `os.replace` so a crash mid-write leaves the
    original archive untouched — the audit write NEVER risks the corpus. Unlike telemetry this REPLACES the member
    (the audit is recomputed whole at build/refresh), so any prior report is dropped."""
    p = Path(path)
    payload = json.dumps(report, sort_keys=True).encode("utf-8")
    tmp_path = _unique_tmp_path(p)
    with zipfile.ZipFile(p, "r") as src:
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as dst:
                for info in src.infolist():
                    if info.filename == _SELF_AUDIT_PATH:
                        continue
                    with src.open(info.filename, "r") as fsrc, \
                         dst.open(info, "w", force_zip64=True) as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
                dst.writestr(_SELF_AUDIT_PATH, payload)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp_path, p)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def append_telemetry_in_place(path: str | Path, rows: list[dict]) -> int:
    """Append telemetry `rows` to the `insight/telemetry.jsonl` member of an
    existing v4 .rlat, preserving every other slot. Returns the count appended
    (0 on empty input — a true no-op, no rewrite).

    **True byte-append.** The existing member's bytes are carried through
    verbatim and the new rows are concatenated after them (a missing trailing
    newline is healed first so a torn final line is never merged onto a new
    row). The existing bytes are NEVER decoded or re-parsed, so a line this
    build can't parse — a foreign/future writer's row, or bit-rot — is preserved
    rather than silently dropped, and a byte-corrupt member can never wedge the
    fold. The existing member is read inside the single archive open below (no
    separate `read_telemetry` round-trip).

    Atomic via per-writer unique tmp + `os.replace`, mirroring
    `write_band_in_place`: a crash mid-fold leaves the original archive
    untouched, so telemetry persistence NEVER risks the corpus — the prime
    directive (capture never breaks retrieval) extended to the disk fold. The
    whole-archive copy per fold is exactly why the fold fires at SESSION
    boundaries, not per query (architecture.md §7); the SQLite re-home replaces
    this body with a cheap row insert.

    Rows must already be redacted (invariant §8) — this is pure I/O and does
    not inspect contents; `store.telemetry.flush` owns redaction.
    """
    if not rows:
        return 0
    p = Path(path)
    addition = _telemetry_to_bytes(rows)

    tmp_path = _unique_tmp_path(p)
    skipped = {_TELEMETRY_PATH}
    with zipfile.ZipFile(p, "r") as src:
        existing = (
            src.read(_TELEMETRY_PATH) if _TELEMETRY_PATH in src.namelist() else b""
        )
        if existing and not existing.endswith(b"\n"):
            existing += b"\n"  # isolate a torn final line from the new rows
        payload = existing + addition
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as dst:
                for info in src.infolist():
                    if info.filename in skipped:
                        continue
                    with src.open(info.filename, "r") as fsrc, \
                         dst.open(info, "w", force_zip64=True) as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
                dst.writestr(_TELEMETRY_PATH, payload)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        os.replace(tmp_path, p)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return len(rows)


def write_band_in_place(
    path: str | Path,
    band_name: str,
    band_info: BandInfo,
    band_data: np.ndarray,
    ann_blob: bytes | None = None,
) -> None:
    """Add or replace a band slot in an existing v4 archive without rewriting
    unrelated slots. Used to fill the insight band (and optional ANN index)
    on a knowledge model whose base band was built earlier.

    Implementation: stream the existing archive into a per-writer-unique
    tmp file (see `_unique_tmp_path`), dropping any members that the new
    band displaces (band NPZ, ANN blob, metadata) and writing fresh entries
    in their place. Atomically replaces the original via `os.replace`. Other
    slots (other bands, source/, registry, other ANN blobs, build_config)
    are copied unchanged.
    """
    p = Path(path)
    tmp_path = _unique_tmp_path(p)

    skipped = {
        _METADATA_PATH,
        _band_path(band_name),
        _ann_path(band_name),
    }

    with zipfile.ZipFile(p, "r") as src:
        meta_text = src.read(_METADATA_PATH).decode("utf-8")
        metadata = from_json(meta_text)
        if metadata.format_version != FORMAT_VERSION:
            raise ValueError(
                f"refuse to mutate v{metadata.format_version} archive; "
                f"in-place writer only handles v{FORMAT_VERSION}"
            )
        metadata.bands[band_name] = band_info

        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as dst:
                dst.writestr(_METADATA_PATH, to_json(metadata))
                # Stream preserved members chunk-by-chunk so peak memory stays
                # bounded by the read buffer, not the size of the largest
                # source file. Critical for bundled-mode archives where
                # `source/` may aggregate >1 GB; without streaming, the rewrite
                # would briefly materialise each member fully into RSS.
                for info in src.infolist():
                    if info.filename in skipped:
                        continue
                    with src.open(info.filename, "r") as fsrc, \
                         dst.open(info, "w", force_zip64=True) as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
                bands_io.write_band(dst, _band_path(band_name), band_data)
                if ann_blob is not None:
                    dst.writestr(_ann_path(band_name), ann_blob)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        os.replace(tmp_path, p)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
