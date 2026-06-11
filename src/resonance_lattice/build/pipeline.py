"""`build_rlat` + `refresh_rlat` — pure Python build/refresh pipelines.

`cli/build.py` and `cli/maintain.py` are argparse wrappers over these
functions. External callers (the Fabric `fabric_build.ipynb` notebook,
embedded library use) import from here directly. No argparse, no
stdio, no `sys.exit` — exceptions surface `BuildError` / `RefreshError`.
"""

from __future__ import annotations

import datetime as _dt
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import Kind, StoreMode
from ..field import ann
from ..field.encoder import DIM, MAX_SEQ_LENGTH, MODEL_ID, POOLING, Encoder
from ..store import archive, incremental
from ..store.base import compute_hash
from ..store.bundled import pack_source_files
from ..store.chunker import chunk_text
from ..store.metadata import BackboneInfo, BandInfo, Metadata
from ..store.registry import PassageCoord, compute_id, compute_key_id
from ..store.remote import compose_manifest
from .walker import SourceWalker


DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 3200
DEFAULT_BATCH_SIZE = 32

# Row-mode emits one passage per row with NO upper bound on length (the
# chunker, which caps chunked passages at max_chars, is bypassed). The
# encoder still truncates at MAX_SEQ_LENGTH tokens, so a row longer than the
# window is only PARTIALLY represented by its embedding while fetch/verify
# still see the whole text. We can't cap without breaking "one passage per
# row", but we can WARN: ~4 chars/token is a conservative lower bound on the
# window in characters, so a row above this is at risk of truncation.
_ROW_TRUNCATE_HINT_CHARS = MAX_SEQ_LENGTH * 4


class BuildError(RuntimeError):
    """Build pre-conditions failed: empty corpus, conflicting flags, etc."""


class RefreshError(RuntimeError):
    """Refresh pre-conditions failed: bundled archive, missing source_root,
    wrong store mode, etc."""


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    n_passages: int
    n_files: int
    band_name: str
    encoder_revision: str
    store_mode: str
    elapsed_seconds: float


@dataclass(frozen=True)
class RefreshResult:
    output_path: Path
    n_added: int
    n_changed: int
    n_deleted: int
    n_unchanged: int
    encoder_revision: str
    elapsed_seconds: float

    @property
    def n_passages(self) -> int:
        """New passage count after delta-apply (unchanged + changed + added).
        Deletions reduce the total; old_count - n_deleted + n_added."""
        return self.n_unchanged + self.n_changed + self.n_added

    @property
    def is_noop(self) -> bool:
        return self.n_added == 0 and self.n_changed == 0 and self.n_deleted == 0


def build_rlat(
    walker: SourceWalker,
    output: Path,
    *,
    store_mode: StoreMode = StoreMode.LOCAL,
    kind: Kind = Kind.CORPUS,
    runtime: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    encoder: Encoder | None = None,
    remote_url_base: str | None = None,
    row_mode: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> BuildResult:
    """Build a `.rlat` archive from `walker`.

    `encoder` injection: when None, constructs a fresh `Encoder(runtime=runtime)`
    (CLI default). UDF callers pass their already-warm encoder so build
    inherits the search side's lazy load — no second weight read.

    `row_mode`: the semantic-slicer build. Each walker entry becomes exactly
    ONE passage spanning its whole text (the chunker is bypassed — no split,
    no short-text merge), keyed by the walker's key (its rel-path slot), with
    `passage_id` pinned to that key via `compute_key_id`. Requires
    `store_mode=bundled` — row text comes from a table, not a file tree, so
    it must live inside the archive for retrieval/verify to resolve it. Keys
    must be unique; a duplicate raises `BuildError`. Pair with
    `build.walker.RowSourceWalker`.

    `on_progress`: optional callback invoked with stage strings — "walking",
    "chunked N passages", "encoded N", "writing archive". UDF callers use
    this for diagnostic logging.
    """
    if row_mode and store_mode is not StoreMode.BUNDLED:
        raise BuildError(
            "row_mode requires store_mode=bundled — row text has no source "
            "file tree to re-read, so it must be bundled inside the archive "
            f"(got store_mode={store_mode.value})"
        )
    if store_mode is StoreMode.REMOTE and not remote_url_base:
        raise BuildError(
            "store_mode=remote requires remote_url_base; "
            "e.g. https://example.com/corpus/v1"
        )
    if remote_url_base and store_mode is not StoreMode.REMOTE:
        raise BuildError(
            f"remote_url_base only applies to store_mode=remote "
            f"(got {store_mode.value})"
        )

    started = time.monotonic()

    def _emit(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    _emit("walking sources")
    files: list[tuple[str, str]] = list(walker.iter_files())
    if not files:
        raise BuildError(
            f"no text files found under {walker.source_root_for_metadata}"
        )
    _emit(f"walked {len(files)} files (skipped {len(walker.skipped)})")

    registry: list[PassageCoord] = []
    passage_texts: list[str] = []
    if row_mode:
        seen_keys: set[str] = set()
        n_oversize = 0
        for key, text in files:
            if key in seen_keys:
                raise BuildError(
                    f"row_mode requires unique keys; duplicate key {key!r}"
                )
            seen_keys.add(key)
            if len(text) > _ROW_TRUNCATE_HINT_CHARS:
                n_oversize += 1
            # One passage spanning the whole row — chunker bypassed. id pinned
            # to the key so a text edit doesn't re-key the row.
            registry.append(PassageCoord(
                passage_idx=len(registry),
                source_file=key,
                char_offset=0,
                char_length=len(text),
                content_hash=compute_hash(text),
                passage_id=compute_key_id(key),
                key=key,
            ))
            passage_texts.append(text)
        if n_oversize:
            _emit(
                f"warning: {n_oversize} row(s) exceed ~{_ROW_TRUNCATE_HINT_CHARS} "
                f"chars and may be truncated to the encoder window "
                f"({MAX_SEQ_LENGTH} tokens) — their embeddings represent only "
                f"the head. Consider splitting or pooling these rows."
            )
    else:
        for rel_path, text in files:
            for char_offset, char_length in chunk_text(text, min_chars, max_chars):
                passage_text = text[char_offset:char_offset + char_length]
                registry.append(PassageCoord(
                    passage_idx=len(registry),
                    source_file=rel_path,
                    char_offset=char_offset,
                    char_length=char_length,
                    content_hash=compute_hash(passage_text),
                    passage_id=compute_id(rel_path, char_offset, char_length),
                ))
                passage_texts.append(passage_text)
    if not registry:
        raise BuildError(
            "no passages produced — every row/file may be empty"
        )
    _emit(f"{'rows' if row_mode else 'chunked'} {len(registry)} passages")

    enc = encoder if encoder is not None else Encoder(runtime=runtime)
    base_band = enc.encode_batched(passage_texts, batch_size)
    _emit(f"encoded {len(passage_texts)} passages")

    ann_meta: dict[str, dict[str, int | str]] = {}
    ann_blobs: dict[str, bytes] = {}
    if ann.should_build_ann(len(registry)):
        index = ann.build(base_band)
        ann_blobs["base"] = ann.serialize(index)
        ann_meta["base"] = {
            "type": "hnsw",
            "M": ann.HNSW_M,
            "efConstruction": ann.HNSW_EFCONSTRUCTION,
            "efSearch": ann.HNSW_EFSEARCH,
        }

    build_config: dict[str, Any] = {
        "chunker": "row_v1" if row_mode else "passage_v1",
        "row_mode": row_mode,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "passage_count": len(registry),
        "file_count": len(files),
        "source_root": walker.source_root_for_metadata,
        "batch_size": batch_size,
    }
    build_config.update(walker.build_config_extras)

    metadata = Metadata(
        kind=kind.value,
        backbone=BackboneInfo(
            name=MODEL_ID,
            revision=enc.revision,
            dim=DIM,
            pool=POOLING,
            max_seq_length=MAX_SEQ_LENGTH,
        ),
        bands={
            "base": BandInfo(
                role="retrieval_default",
                dim=DIM,
                l2_norm=True,
                passage_count=len(registry),
            ),
        },
        store_mode=store_mode.value,
        ann=ann_meta,
        build_config=build_config,
        created_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    source_files_payload: dict[str, bytes] | None = None
    if store_mode is StoreMode.BUNDLED:
        source_files_payload = pack_source_files(dict(files))

    remote_manifest_payload: dict[str, dict[str, str]] | None = None
    if store_mode is StoreMode.REMOTE:
        assert remote_url_base is not None  # guarded above; for type narrowing
        remote_manifest_payload = compose_manifest(dict(files), remote_url_base)
        metadata.build_config["upstream_url_base"] = remote_url_base.rstrip("/")

    # Foundational, LLM-free self-audit — computed IN-MEMORY from the band + registry and folded into the SINGLE
    # write below (no second ZIP rewrite). EVERY built .rlat carries its own shape-report. Gaps emerge from use
    # (telemetry), so none at build; drift is empty for a fresh corpus. Best-effort: never breaks a build.
    _emit("self-audit (contradiction candidates)")
    from types import SimpleNamespace

    from ..store.self_audit import compute_self_audit
    try:
        self_audit_report = compute_self_audit(
            contents=SimpleNamespace(bands={"base": base_band}, registry=registry)
        )
    except Exception:
        self_audit_report = None

    _emit(f"writing archive ({store_mode.value})")
    archive.write(
        output,
        metadata=metadata,
        bands={"base": base_band},
        registry=registry,
        ann_blobs=ann_blobs,
        source_files=source_files_payload,
        remote_manifest=remote_manifest_payload,
        self_audit=self_audit_report or None,
    )

    return BuildResult(
        output_path=output,
        n_passages=len(registry),
        n_files=len(files),
        band_name="base",
        encoder_revision=enc.revision,
        store_mode=store_mode.value,
        elapsed_seconds=time.monotonic() - started,
    )


def refresh_rlat(
    walker: SourceWalker,
    rlat_path: Path,
    *,
    runtime: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_chars: int | None = None,
    max_chars: int | None = None,
    encoder: Encoder | None = None,
    dry_run: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> RefreshResult:
    """Incremental delta-apply against an existing local-mode `.rlat`.

    `min_chars` / `max_chars` default to the values recorded in
    `build_config`, keeping the chunker replay-faithful. Pass explicit
    overrides only when migrating chunker bounds.

    `encoder` injection: when None, constructs `Encoder(runtime=runtime)`
    with `runtime="auto"` (matches build behaviour — unlocks ONNX /
    OpenVINO refresh paths). The legacy `cmd_refresh` hardcoded
    `runtime="torch"`; this is a deliberate behaviour upgrade.
    """
    started = time.monotonic()

    if not rlat_path.is_file():
        raise RefreshError(f"{rlat_path} is not a file")
    try:
        contents = archive.read(rlat_path)
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise RefreshError(
            f"{rlat_path} is not a valid v4 knowledge model: {exc}"
        ) from exc

    mode = StoreMode(contents.metadata.store_mode)
    if mode is StoreMode.BUNDLED:
        raise RefreshError(
            "bundled-mode archives are immutable post-build; "
            "run `rlat build` to produce a fresh archive."
        )
    if mode is StoreMode.REMOTE:
        raise RefreshError(
            "refresh_rlat is for local-mode archives; "
            "remote-mode reconciliation uses `rlat sync`."
        )

    bc = contents.metadata.build_config
    eff_min = min_chars if min_chars is not None else int(bc.get("min_chars", DEFAULT_MIN_CHARS))
    eff_max = max_chars if max_chars is not None else int(bc.get("max_chars", DEFAULT_MAX_CHARS))

    def _emit(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    _emit("walking sources")
    files = list(walker.iter_files())
    _emit(f"walked {len(files)} files (skipped {len(walker.skipped)})")
    candidates = incremental.chunk_files(files, eff_min, eff_max)
    delta = incremental.bucketise(contents.registry, candidates)
    _emit(f"delta unchanged={delta.n_unchanged} updated={delta.n_updated} "
          f"added={delta.n_added} removed={delta.n_removed}")

    if dry_run or delta.is_empty:
        # dry_run: report the delta without touching the encoder or archive.
        # delta.is_empty: nothing changed, so no rewrite needed.
        return RefreshResult(
            output_path=rlat_path,
            n_added=delta.n_added,
            n_changed=delta.n_updated,
            n_deleted=delta.n_removed,
            n_unchanged=delta.n_unchanged,
            encoder_revision=contents.metadata.backbone.revision,
            elapsed_seconds=time.monotonic() - started,
        )

    enc = encoder if encoder is not None else Encoder(runtime=runtime)
    _emit(f"re-encoding {delta.n_re_encode} passages")
    result = incremental.apply_delta(
        rlat_path, contents, delta,
        encoder=enc, batch_size=batch_size,
    )
    _emit("archive written")

    return RefreshResult(
        output_path=result.archive_path,
        n_added=delta.n_added,
        n_changed=delta.n_updated,
        n_deleted=delta.n_removed,
        n_unchanged=delta.n_unchanged,
        encoder_revision=enc.revision,
        elapsed_seconds=time.monotonic() - started,
    )
