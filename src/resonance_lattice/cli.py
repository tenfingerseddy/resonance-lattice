# SPDX-License-Identifier: BUSL-1.1
"""CLI interface for the Resonance Lattice.

Usage:
    rlat search corpus.rlat "How does auth work?"
    rlat profile corpus.rlat
    rlat build ./docs ./src -o corpus.rlat
    rlat compare a.rlat b.rlat
"""

from __future__ import annotations

import argparse
import contextlib
import io as _io
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resonance_lattice.composition.composed import ComposedCartridge
    from resonance_lattice.encoder import Encoder
    from resonance_lattice.lattice import Lattice

logger = logging.getLogger(__name__)


# ── Output helpers ─────────────────────────────────────────────────────

def _die(message: str, code: int = 1) -> None:
    """Print error to stderr and exit."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def _warn(message: str) -> None:
    """Print warning to stderr."""
    print(f"Warning: {message}", file=sys.stderr)


def _parse_band_energies(
    coverage: dict,
    fallback_names: list[str] | None = None,
) -> tuple[list[str], list[float]]:
    """Extract (names, values) from a coverage dict's band_energies field."""
    band_energies = coverage.get("band_energies", {})
    band_names = coverage.get("band_names", []) or fallback_names or []
    if isinstance(band_energies, dict):
        names = list(band_energies.keys())
        vals = [float(v) for v in band_energies.values()]
    else:
        names = band_names or [f"b{i}" for i in range(len(band_energies))]
        vals = [float(v) for v in band_energies]
    return names, vals


def _safe_bar(fraction: float, width: int = 20) -> str:
    """Render a progress bar using characters safe for the current terminal encoding."""
    fill_len = int(width * max(0.0, min(1.0, fraction)))
    empty_len = width - fill_len
    try:
        "\u2588".encode(sys.stdout.encoding or "utf-8")
        return "\u2588" * fill_len + "\u2591" * empty_len
    except (UnicodeEncodeError, LookupError):
        return "#" * fill_len + "-" * empty_len


def _display_name(result) -> str:
    """Human-readable name for a result: source_file / heading, falling back to source_id."""
    if result.content and result.content.metadata:
        source_file = result.content.metadata.get("source_file", "")
        heading = result.content.metadata.get("heading", "")
        stem = Path(source_file).stem if source_file else ""
        if stem and heading:
            return f"{stem} / {heading}"
        if stem:
            return stem
        if heading:
            return heading
    return result.source_id


def _truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars, breaking at a word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind(" ")
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut] + "..."


def _clean_passage(text: str) -> str:
    """Strip markdown noise from a passage for clean terminal display."""
    import re
    # Strip heading markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strip bold/italic markers
    text = text.replace('**', '').replace('__', '')
    # Strip markdown links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Strip blockquote markers (both start-of-line and inline after newlines)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s>\s', ' ', text)
    # Strip HTML-like tags (</BR>, :::, etc.)
    text = re.sub(r'</?[A-Za-z]+/?>', '', text)
    text = re.sub(r':::.*?:::', '', text, flags=re.DOTALL)
    # Strip [!NOTE], [!IMPORTANT], etc.
    text = re.sub(r'\[!(?:NOTE|IMPORTANT|WARNING|CAUTION|TIP)\]\s*', '', text)
    # Strip markdown table rows (lines that are mostly pipes and dashes)
    text = re.sub(r'\|[-\s|]+\|', '', text)
    # Clean remaining pipe-delimited table content
    text = re.sub(r'\|', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _quiet_stderr():
    """Context manager that suppresses stderr (encoder loading noise)."""
    if os.environ.get("RLAT_VERBOSE"):
        return contextlib.nullcontext()
    return contextlib.redirect_stderr(_io.StringIO())


def _require_file(path: Path) -> None:
    """Exit with a friendly message if the file does not exist."""
    if not path.exists():
        _die(f"file not found: {path}")


# ── ANSI color helpers ─────────────────────────────────────────────────

def _use_color() -> bool:
    """Check if the terminal supports ANSI color."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


class _C:
    """ANSI color codes. All resolve to empty strings when color is off."""
    _on: bool = False

    @classmethod
    def init(cls):
        cls._on = _use_color()

    @classmethod
    def _code(cls, code: str) -> str:
        return code if cls._on else ""

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls._code(chr(27) + '[1m')}{text}{cls._code(chr(27) + '[0m')}"

    @classmethod
    def dim(cls, text: str) -> str:
        return f"{cls._code(chr(27) + '[2m')}{text}{cls._code(chr(27) + '[0m')}"

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls._code(chr(27) + '[32m')}{text}{cls._code(chr(27) + '[0m')}"

    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls._code(chr(27) + '[33m')}{text}{cls._code(chr(27) + '[0m')}"

    @classmethod
    def cyan(cls, text: str) -> str:
        return f"{cls._code(chr(27) + '[36m')}{text}{cls._code(chr(27) + '[0m')}"

    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls._code(chr(27) + '[31m')}{text}{cls._code(chr(27) + '[0m')}"

    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls._code(chr(27) + '[34m')}{text}{cls._code(chr(27) + '[0m')}"


# ── File manifest for incremental sync ─────────────────────────────────

import hashlib as _hashlib


def _canonical_path(path: Path) -> str:
    """Normalize a file path to a stable canonical form for manifest keys.

    Uses CWD-relative POSIX paths so that the same source tree produces
    identical manifest keys regardless of where the repo is checked out.
    Falls back to the resolved filename if the path is outside CWD.
    Lowercased on Windows for case-insensitive matching.
    """
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        # Path is outside CWD — use the filename only
        rel = Path(path.name)
    posix = rel.as_posix()
    if sys.platform == "win32":
        posix = posix.lower()
    return posix


class FileManifest:
    """Tracks which files are in a knowledge model and which chunks they produced.

    Persisted as a __manifest__ entry in the SourceStore so it survives
    save/load cycles. Keys are canonical POSIX paths (see _canonical_path).
    """

    def __init__(self) -> None:
        # canonical_path -> {hash, chunk_ids, encoder}
        self.entries: dict[str, dict] = {}

    def record(self, file_path: str, content_hash: str, chunk_ids: list[str],
               encoder_fp: str = "") -> None:
        self.entries[file_path] = {
            "hash": content_hash,
            "chunk_ids": chunk_ids,
            "encoder": encoder_fp,
        }

    def remove_file(self, file_path: str) -> list[str]:
        """Remove a file entry and return its chunk IDs for cleanup."""
        entry = self.entries.pop(file_path, None)
        return entry["chunk_ids"] if entry else []

    def needs_update(self, file_path: str, content_hash: str, encoder_fp: str = "") -> bool:
        """Check if a file needs re-ingestion.

        Returns True if the file is new, its content changed, OR the encoder
        fingerprint differs from what was used to encode it.
        """
        entry = self.entries.get(file_path)
        if entry is None:
            return True
        if entry["hash"] != content_hash:
            return True
        # Encoder mismatch = must re-encode
        if encoder_fp and entry.get("encoder") and entry["encoder"] != encoder_fp:
            return True
        return False

    def stored_encoder(self) -> str | None:
        """Return the encoder fingerprint used by existing entries, or None."""
        for entry in self.entries.values():
            fp = entry.get("encoder", "")
            if fp:
                return fp
        return None

    def known_files(self) -> set[str]:
        return set(self.entries.keys())

    def to_json(self) -> str:
        return json.dumps(self.entries, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> FileManifest:
        m = cls()
        m.entries = json.loads(data) if data else {}
        return m

    @staticmethod
    def hash_file(path: Path) -> str:
        return _hashlib.md5(path.read_bytes()).hexdigest()


def _load_manifest(lattice) -> FileManifest:
    """Load the file manifest from a lattice's store.

    If the knowledge model has no manifest (legacy), returns an empty manifest
    and prints a warning. The first add/sync will treat all files as new.
    """
    content = lattice.store.retrieve("__manifest__")
    if content and content.full_text:
        return FileManifest.from_json(content.full_text)
    # Legacy cartridge or first run
    if lattice.source_count > 0:
        print(
            "Warning: this knowledge model has no file manifest (built before manifest support).\n"
            "  The first add/sync will treat all source files as new.\n"
            "  To avoid duplicates, rebuild with: rlat build ... -o knowledge model.rlat",
            file=sys.stderr,
        )
    return FileManifest()


def _check_encoder_consistency(manifest: FileManifest, encoder_fp: str,
                               lattice=None) -> None:
    """Refuse to mix encoders in a single knowledge model.

    Checks the manifest's stored encoder fingerprint first. For legacy
    knowledge models without a manifest, falls back to comparing the stored
    __encoder__ config in the lattice itself.

    Raises SystemExit if the encoders don't match.
    """
    stored = manifest.stored_encoder()

    # Legacy fallback: if no manifest encoder, check the cartridge's stored encoder
    if not stored and lattice is not None and lattice.store.count > 0:
        enc_content = lattice.store.retrieve("__encoder__")
        if enc_content and enc_content.full_text:
            try:
                import json as _json
                enc_config = _json.loads(enc_content.full_text)
                # Build the same fingerprint from stored config
                parts = [
                    enc_config.get("encoder_type", ""),
                    enc_config.get("backbone", ""),
                    str(enc_config.get("bands", "")),
                    str(enc_config.get("dim", "")),
                    str(enc_config.get("sparsities", "")),
                    enc_config.get("query_prefix", ""),
                    enc_config.get("passage_prefix", ""),
                    enc_config.get("pooling", ""),
                ]
                stored = _hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
            except Exception:
                pass

    if stored and encoder_fp and stored != encoder_fp:
        print(
            f"Error: encoder mismatch.\n"
            f"  Cartridge was built with: {stored}\n"
            f"  Current encoder:          {encoder_fp}\n"
            f"  Mixing encoders degrades retrieval quality.\n"
            f"  Use the same --encoder, or rebuild the cartridge.",
            file=sys.stderr,
        )
        sys.exit(1)


def _save_manifest(lattice, manifest: FileManifest) -> None:
    """Persist the file manifest into the lattice's store."""
    from resonance_lattice.store import SourceContent
    lattice.store.remove("__manifest__")
    lattice.store.store(SourceContent(
        source_id="__manifest__",
        summary="file manifest",
        full_text=manifest.to_json(),
    ))


def cmd_init(args: argparse.Namespace) -> None:
    """Create a new empty lattice file."""
    from resonance_lattice.config import Compression, FieldType, LatticeConfig, Precision
    from resonance_lattice.lattice import Lattice

    config = LatticeConfig(
        bands=args.bands,
        dim=args.dim,
        field_type=FieldType(args.field_type),
        pq_subspaces=args.pq_subspaces,
        pq_codebook_size=args.pq_codebook_size,
        svd_rank=args.svd_rank,
        precision=Precision(args.precision),
        compression=Compression(args.compression),
    )

    lattice = Lattice(config=config)

    output = Path(args.output)
    lattice.save(output)
    print(f"Created empty lattice: {output}")
    print(f"  Field type: {config.field_type.value}")
    print(f"  Bands: {config.bands}, Dim: {config.dim}")
    print(f"  Estimated field size: {config.field_size_mb:.1f} MB")


def _describe_encoder(encoder) -> str:
    """Return a short human-readable encoder description."""
    config = encoder.get_config()
    encoder_type = config.get("encoder_type", "random")
    backbone = config.get("backbone", "random")
    return f"{encoder_type} ({backbone})"


def _encoder_fingerprint(encoder) -> str:
    """Return a stable fingerprint that distinguishes different encoder configs.

    Includes encoder_type, backbone, bands, dim, sparsities, and protocol
    fields (query_prefix, passage_prefix, pooling) so that encoders with
    different prefix/pooling contracts produce different fingerprints.
    """
    config = encoder.get_config()
    # Build a canonical key from all identity-relevant fields
    parts = [
        config.get("encoder_type", ""),
        config.get("backbone", ""),
        str(config.get("bands", "")),
        str(config.get("dim", "")),
        str(config.get("sparsities", "")),
        config.get("query_prefix", ""),
        config.get("passage_prefix", ""),
        config.get("pooling", ""),
    ]
    return _hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _has_encoder_override(args: argparse.Namespace) -> bool:
    """True when CLI args explicitly override a knowledge model's stored encoder."""
    return bool(getattr(args, "encoder", None))


def _resolve_preset(name: str) -> dict | None:
    """Look up a named encoder preset. Returns config dict or None."""
    from resonance_lattice.config import ENCODER_PRESETS
    return ENCODER_PRESETS.get(name)


def _pick_device() -> str:
    """Choose an encoder device.

    Override with `RLAT_DEVICE=cpu|cuda|cuda:0|...`. Default: cuda if available,
    else cpu. Before this helper the CLI always ran on CPU even under GPU —
    silently — which wasted Colab/Kaggle hours. Auto by default, explicit wins.
    """
    override = os.environ.get("RLAT_DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


NPU_DEFAULT_SEQ_LEN = 512


def _resolve_openvino_static_seq_len(args, device: str) -> int | None:
    """Return the static sequence length to use, or None for dynamic shapes.

    Priority: explicit `--openvino-static-seq-len N` > env `RLAT_OPENVINO_STATIC_SEQ_LEN` >
    NPU-implied default (512). Dynamic shapes return None — CPU/GPU handle that fine;
    only NPU compile requires static.
    """
    explicit = getattr(args, "openvino_static_seq_len", None)
    if explicit is None:
        env = os.environ.get("RLAT_OPENVINO_STATIC_SEQ_LEN")
        if env:
            try:
                explicit = int(env)
            except ValueError:
                pass
    if explicit is not None:
        return int(explicit)
    if device and device.upper() == "NPU":
        return NPU_DEFAULT_SEQ_LEN
    return None


def _maybe_attach_accelerated_backbone(args, lattice) -> None:
    """Attach OpenVINO or ONNX backbone per flags/env, with auto-detect fallback.

    Decision order:
      1. `--openvino DIR` explicit — wins; warns if --onnx also given.
      2. `--onnx DIR` explicit — use it (no auto-OV attempt).
      3. Auto-detect — if encoder_openvino.auto_enabled() and Arc/NPU present
         and export succeeds, attach OpenVINO silently. "Auto by default,
         explicit override wins": opt out with RLAT_OPENVINO=off.
      4. No attach — stock torch behaviour.

    Static-shape (NPU) handling: NPU compile rejects dynamic sequence dims, so
    when `--openvino-device NPU` is selected (explicit or auto) the exporter
    reshapes to a fixed seq_len (default 512, override via
    --openvino-static-seq-len or RLAT_OPENVINO_STATIC_SEQ_LEN). The static IR
    lives in a separate cache dir so CPU/GPU dynamic IR coexists.

    Any failure logs at WARN and falls through; never blocks a working flow.

    Env:
      RLAT_OPENVINO=off                 disable auto-detect
      RLAT_OPENVINO_DEVICE=<dev>        device choice (CPU | GPU | NPU | AUTO)
      RLAT_OPENVINO_CACHE_DIR=<d>       IR cache location
      RLAT_OPENVINO_STATIC_SEQ_LEN=<N>  force static seq_len for export (default 512 on NPU)
    """
    if lattice.encoder is None:
        return

    ov_dir = getattr(args, "openvino", None)
    onnx_dir = getattr(args, "onnx", None)

    # 1. Explicit --openvino wins.
    if ov_dir:
        if onnx_dir:
            _warn("both --openvino and --onnx given; using OpenVINO (Arc iGPU ~40× faster).")
        try:
            from resonance_lattice.encoder_openvino import attach_openvino_backbone
            device = (
                getattr(args, "openvino_device", None)
                or os.environ.get("RLAT_OPENVINO_DEVICE")
                or "AUTO"
            )
            attach_openvino_backbone(lattice.encoder, ov_dir, device=device)
            print(f"OpenVINO backbone attached (device={device}, dir={ov_dir})", file=sys.stderr)
            return
        except Exception as exc:
            _warn(f"OpenVINO backbone failed to load: {exc}")
            return  # user explicitly asked for OV; don't silently fall through to ONNX

    # 2. Explicit --onnx path (caller opted into ONNX).
    if onnx_dir:
        try:
            from resonance_lattice.encoder_onnx import attach_onnx_backbone
            attach_onnx_backbone(lattice.encoder, onnx_dir)
        except Exception as exc:
            _warn(f"ONNX backbone failed to load: {exc}")
        return

    # 3. Auto-detect — opportunistic Arc/NPU acceleration on this host.
    try:
        from resonance_lattice import encoder_openvino as OV
    except Exception:
        return
    if not OV.auto_enabled():
        return
    device_override = (
        getattr(args, "openvino_device", None)
        or os.environ.get("RLAT_OPENVINO_DEVICE")
    )
    device = device_override or OV.preferred_device() or "AUTO"
    static_seq_len = _resolve_openvino_static_seq_len(args, device)
    auto_dir = OV.auto_get_or_export(lattice.encoder, static_seq_len=static_seq_len)
    if auto_dir is None:
        return
    try:
        OV.attach_openvino_backbone(lattice.encoder, auto_dir, device=device)
        suffix = f", static_seq_len={static_seq_len}" if static_seq_len else ""
        print(f"OpenVINO backbone auto-attached (device={device}, dir={auto_dir}{suffix})", file=sys.stderr)
    except Exception as exc:
        _warn(f"OpenVINO auto-attach failed: {exc}")


def _load_encoder(args: argparse.Namespace, bands: int, dim: int, lattice=None):
    """Helper to load encoder from CLI args, stored config, or default.

    Priority: explicit --encoder > restored lattice encoder > default
    """
    from resonance_lattice.encoder import Encoder

    device = _pick_device()
    if device != "cpu":
        print(f"Encoder device: {device} (override with RLAT_DEVICE=cpu)", file=sys.stderr)

    # Projection head sparsification mode (None → default "threshold")
    sparsify_mode = getattr(args, "sparsify_mode", None)
    soft_topk_tau = getattr(args, "soft_topk_tau", None)
    sparsemax_scale = getattr(args, "sparsemax_scale", None)
    backbone_kwargs: dict = {}
    if sparsify_mode is not None:
        backbone_kwargs["sparsify_mode"] = sparsify_mode
    if soft_topk_tau is not None:
        backbone_kwargs["soft_topk_tau"] = soft_topk_tau
    if sparsemax_scale is not None:
        backbone_kwargs["sparsemax_scale"] = sparsemax_scale

    # Explicit encoder flag (preset name or raw HuggingFace model ID)
    encoder_flag = getattr(args, "encoder", None)
    if encoder_flag is not None:
        if encoder_flag == "random":
            print("Using random encoder", file=sys.stderr)
            return Encoder.random(bands=bands, dim=dim, **backbone_kwargs)
        preset = _resolve_preset(encoder_flag)
        if preset:
            print(f"Loading encoder preset: {encoder_flag} ({preset['backbone']})", file=sys.stderr)
            return Encoder.from_backbone(
                model_name=preset["backbone"],
                bands=bands,
                dim=dim,
                query_prefix=preset["query_prefix"],
                passage_prefix=preset["passage_prefix"],
                pooling=preset["pooling"],
                max_length=preset["max_length"],
                device=device,
                **backbone_kwargs,
            )
        # Not a preset — treat as raw HuggingFace model ID
        print(f"Loading encoder: {encoder_flag}", file=sys.stderr)
        return Encoder.from_backbone(
            model_name=encoder_flag, bands=bands, dim=dim, device=device,
            **backbone_kwargs,
        )

    # 3. Restored encoder on the lattice
    if lattice is not None and lattice.encoder is not None:
        print(f"Using stored encoder: {_describe_encoder(lattice.encoder)}", file=sys.stderr)
        return lattice.encoder

    # 4. Default: BGE-large-en-v1.5 (board item 237, 2026-04-20). BGE
    # measured stronger than E5 on the 5-BEIR launch sweep — see
    # docs/ENCODER_CHOICE.md for the per-corpus numbers. Resolves
    # through the preset so query_prefix / pooling / max_length come
    # from a single source of truth (config.ENCODER_PRESETS). Pass
    # `--encoder e5-large-v2` to keep building with the prior default.
    default_preset_name = "bge-large-en-v1.5"
    default_preset = _resolve_preset(default_preset_name)
    assert default_preset is not None, (
        f"missing preset {default_preset_name!r} in ENCODER_PRESETS"
    )
    print(
        f"Downloading encoder ({default_preset['backbone']}, one-time)...",
        file=sys.stderr,
    )
    return Encoder.from_backbone(
        model_name=default_preset["backbone"],
        bands=bands,
        dim=dim,
        query_prefix=default_preset["query_prefix"],
        passage_prefix=default_preset["passage_prefix"],
        pooling=default_preset["pooling"],
        max_length=default_preset["max_length"],
        device=device,
        **backbone_kwargs,
    )


def _load_lattice_with_encoder(args: argparse.Namespace, lattice_path: Path):
    """Load a lattice while respecting stored encoder ownership.

    Suppresses HuggingFace/BERT loading noise unless RLAT_VERBOSE=1.
    """
    import warnings

    from resonance_lattice.lattice import Lattice
    verbose = os.environ.get("RLAT_VERBOSE")
    if not verbose:
        print(f"Loading {lattice_path.name}...", file=sys.stderr, end="", flush=True)
    source_root = getattr(args, "source_root", None)
    override = _has_encoder_override(args)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with _quiet_stderr():
            lattice = Lattice.load(
                lattice_path,
                restore_encoder=not override,
                source_root=source_root,
            )
            if override or lattice.encoder is None:
                lattice.encoder = _load_encoder(args, lattice.config.bands, lattice.config.dim, lattice=lattice)
    if not verbose:
        print("\r\033[K", file=sys.stderr, end="", flush=True)  # clear loading line
    # Surface critical warnings even when stderr is suppressed
    for w in captured_warnings:
        if issubclass(w.category, RuntimeWarning):
            _warn(str(w.message))
    if verbose and lattice.encoder is not None and not override:
        print(f"Encoder: {_describe_encoder(lattice.encoder)}", file=sys.stderr)
    if source_root:
        print(f"Store: external ({source_root})", file=sys.stderr)

    # Attach accelerated backbone. Priority: explicit --openvino > explicit --onnx >
    # auto-OpenVINO (Arc/NPU host) > auto-ONNX (pre-exported alongside cartridge).
    _maybe_attach_accelerated_backbone(args, lattice)

    # Backstop for deployments that pre-exported an ONNX dir next to the cartridge
    # but have no explicit --onnx flag AND no Arc/NPU (auto-OV didn't attach).
    if (
        lattice.encoder is not None
        and not getattr(args, "openvino", None)
        and not getattr(args, "onnx", None)
        and not _has_accelerated_backbone(lattice.encoder)
    ):
        try:
            from resonance_lattice.worker_main import _find_onnx_dir
            onnx_dir = _find_onnx_dir(lattice_path)
            if onnx_dir:
                from resonance_lattice.encoder_onnx import attach_onnx_backbone
                attach_onnx_backbone(lattice.encoder, onnx_dir)
        except Exception:
            pass

    return lattice


_ACCEL_BACKBONE_NAMES = frozenset({"OnnxBackbone", "OpenVinoBackbone"})
_ACCEL_BACKBONE_MODULE_PREFIXES = (
    "resonance_lattice.encoder_openvino",
    "resonance_lattice.encoder_onnx",
    "optimum.",
    "onnxruntime.",
    "openvino.",
)


def _has_accelerated_backbone(encoder) -> bool:
    """Heuristic: True if the encoder's backbone has already been swapped for
    a non-torch accelerator (ONNX or OpenVINO). Used to avoid double-attach.

    Matches either the concrete wrapper class name or a specific module prefix.
    Substring matches on `__module__` are intentionally NOT used — test module
    names can contain 'openvino' and cause false positives.
    """
    bb = getattr(encoder, "_backbone", None)
    if bb is None:
        return False
    if type(bb).__name__ in _ACCEL_BACKBONE_NAMES:
        return True
    mod = type(bb).__module__ or ""
    return any(mod == p or mod.startswith(p) for p in _ACCEL_BACKBONE_MODULE_PREFIXES)


def _get_system_prompt(mode: str, custom_prompt: str | None = None) -> str:
    """Get the system prompt for the specified injection mode."""
    from resonance_lattice.projector import AugmentProjector
    prompts = {
        "augment": AugmentProjector.SYSTEM_AUGMENT,
        "constrain": AugmentProjector.SYSTEM_CONSTRAIN,
        "knowledge": AugmentProjector.SYSTEM_KNOWLEDGE,
    }
    if mode == "custom":
        if not custom_prompt:
            _die("--custom-prompt required with --mode custom")
        return custom_prompt
    return prompts.get(mode, prompts["augment"])


INGEST_EXTENSIONS = {
    # Text / markup
    ".txt", ".md", ".rst", ".html", ".htm", ".xml",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".hpp", ".go", ".rb", ".rs", ".swift", ".kt", ".lua", ".r", ".php",
    ".sh", ".sql", ".css", ".scss",
    # Data / config
    ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv",
    # Binary (requires optional deps — graceful fallback in LocalStore)
    ".docx", ".pdf", ".xlsx", ".xls",
}

# Directories to skip during recursive file discovery
SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules", "dist", "build",
    ".venv", "venv", ".env", ".tox", ".mypy_cache", ".pytest_cache",
    ".egg-info", ".eggs", "htmlcov", ".ipynb_checkpoints",
}


def _collect_files(paths: list[Path]) -> list[Path]:
    """Collect ingestable files from paths, filtering out noise directories."""
    files = []
    for input_path in paths:
        if input_path.is_file():
            if input_path.suffix.lower() in INGEST_EXTENSIONS:
                files.append(input_path)
            else:
                files.append(input_path)  # Explicit files always included
        elif input_path.is_dir():
            for ext in sorted(INGEST_EXTENSIONS):
                for f in sorted(input_path.rglob(f"*{ext}")):
                    if not any(skip in f.parts for skip in SKIP_DIRS):
                        files.append(f)
        else:
            print(f"Warning: not found, skipping: {input_path}", file=sys.stderr)
    return files


def _auto_detect_inputs() -> list[Path]:
    """Auto-detect common documentation and source directories in the current project."""
    cwd = Path(".")
    candidates = [
        cwd / "docs",
        cwd / "src",
        cwd / "lib",
        cwd / "README.md",
        cwd / "CLAUDE.md",
        cwd / "AGENTS.md",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        # Fallback: scan current directory
        found = [cwd]
    return found


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents into a lattice."""
    lattice_path = Path(args.lattice)
    # Support both positional input and --input flag
    input_str = args.input
    input_path = Path(input_str)

    if not lattice_path.exists():
        _die(f"lattice file not found: {lattice_path}")

    lattice = _load_lattice_with_encoder(args, lattice_path)
    manifest = _load_manifest(lattice)
    encoder_fp = _encoder_fingerprint(lattice.encoder) if lattice.encoder else ""
    _check_encoder_consistency(manifest, encoder_fp, lattice=lattice)

    # Collect input files
    files = _collect_files([input_path])
    if not files:
        _die(f"no ingestable files found in: {input_path}")

    start = time.time()
    count = 0

    from resonance_lattice.chunker import auto_chunk, generate_summary

    # Phase 1: Collect all chunks (fast, no GPU)
    all_texts: list[str] = []
    all_sids: list[str] = []
    all_metas: list[dict] = []
    all_summaries: list[str] = []

    try:
        from tqdm import tqdm
        file_iter = tqdm(files, desc="Chunking", unit="file", file=sys.stderr)
    except ImportError:
        print(f"Chunking {len(files)} files...", file=sys.stderr)
        file_iter = files

    for f in file_iter:
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = auto_chunk(text, source_file=str(f))

        for chunk in chunks:
            slug = chunk.heading[:40].replace(" ", "_").lower() if chunk.heading else ""
            sid = f"{f.stem}_{slug}_{count:06d}" if slug else f"{f.stem}_{count:06d}"
            all_texts.append(chunk.text)
            all_sids.append(sid)
            meta = {
                "source_file": str(f),
                "heading": chunk.heading,
                "chunk_type": chunk.chunk_type,
                # Carry char_offset + content_hash through to SourceContent so
                # Lattice.save() can emit them into __source_manifest__ for
                # external cartridges. A3 foundation for drift detection.
                "char_offset": chunk.char_offset,
                "content_hash": chunk.content_hash,
            }
            if chunk.metadata:
                meta.update(chunk.metadata)
            all_metas.append(meta)
            all_summaries.append(generate_summary(chunk))
            count += 1

    t_chunk = time.time() - start
    print(f"Chunked {count} chunks from {len(files)} files in {t_chunk:.1f}s")

    # Phase 2: Batch encode + superpose (GPU-accelerated)
    print(f"Encoding {count} chunks (batch_size=64, fp16 on GPU)...", file=sys.stderr)
    t_encode = time.time()
    lattice.superpose_text_batch(
        texts=all_texts,
        source_ids=all_sids,
        metadatas=all_metas,
        summaries=all_summaries,
        batch_size=64,
    )

    elapsed = time.time() - start
    encode_elapsed = time.time() - t_encode
    print(f"Ingested {count} chunks from {len(files)} files in {elapsed:.1f}s")
    print(f"  Chunking: {t_chunk:.1f}s, Encoding: {encode_elapsed:.1f}s")
    print(f"  Rate: {count / max(elapsed, 1e-3):.0f} chunks/s")

    lattice.save(lattice_path)
    print(f"Saved: {lattice_path}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query a lattice and display results."""
    lattice_path = Path(args.lattice)
    lattice = _load_lattice_with_encoder(args, lattice_path)

    start = time.time()
    result = lattice.resonate_text(
        query=args.query,
        top_k=args.top_k,
    )
    elapsed = (time.time() - start) * 1000

    mode = getattr(args, "mode", None)
    system_prompt = _get_system_prompt(mode, getattr(args, "custom_prompt", None)) if mode else None

    if args.format == "json":
        output = {
            "query": args.query,
            "latency_ms": round(elapsed, 2),
            "results": [
                {
                    "source_id": r.source_id,
                    "score": round(r.score, 4),
                    "raw_score": round(r.raw_score, 4) if getattr(r, "raw_score", None) is not None else None,
                    "band_scores": r.band_scores.tolist() if r.band_scores is not None else None,
                    "summary": r.content.summary if r.content else None,
                    "full_text": r.content.full_text if r.content else None,
                    "source_file": (r.content.metadata or {}).get("source_file", "") if r.content else "",
                    "heading": (r.content.metadata or {}).get("heading", "") if r.content else "",
                    "provenance": getattr(r, "provenance", "dense"),
                }
                for r in result.results
            ],
            "coverage": None,
            "related": None,
            "contradictions": None,
        }
        if mode:
            output["mode"] = mode
            output["system_prompt"] = system_prompt
        print(json.dumps(output, indent=2))
    elif args.format == "context":
        if system_prompt:
            print(f"[System: {mode}]")
            print(system_prompt)
            print()
        lines = []
        for r in result.results[:args.top_k]:
            if r.content:
                text = r.content.full_text or r.content.summary or ""
                if text:
                    lines.append(f"- [{r.score:.2f}] {text}")
        print("\n".join(lines))
    elif args.format == "prompt":
        payload = {
            "query": args.query,
            "results": [
                {
                    "source_id": r.source_id,
                    "score": round(r.score, 4),
                    "summary": r.content.summary if r.content else None,
                    "full_text": r.content.full_text if r.content else None,
                    "source_file": (r.content.metadata or {}).get("source_file", "") if r.content else "",
                    "heading": (r.content.metadata or {}).get("heading", "") if r.content else "",
                }
                for r in result.results
            ],
            "coverage": {},
            "related_topics": [],
            "contradictions": [],
        }
        print(_dict_to_prompt(payload))
    else:
        if system_prompt:
            print(f"[Mode: {mode}]\n{system_prompt}\n")
        print(f"Query: {args.query}")
        print(f"Latency: {elapsed:.2f} ms")
        print(f"Results ({len(result.results)}):\n")
        for i, r in enumerate(result.results, 1):
            print(f"  {i}. [{r.score:.4f}] {r.source_id}")
            if r.content:
                text = (r.content.full_text or r.content.summary or "")[:500]
                print(f"     {text}")
            print()


def cmd_encoders(_args: argparse.Namespace) -> None:
    """List available encoder presets."""
    from resonance_lattice.config import ENCODER_PRESETS

    # Header
    print(f"{'Preset':<18} {'Backbone':<45} {'Tokens':>8}  {'Pooling':<6}  Query prefix")
    print(f"{'─' * 18} {'─' * 45} {'─' * 8}  {'─' * 6}  {'─' * 30}")

    for name, p in ENCODER_PRESETS.items():
        prefix = p["query_prefix"]
        # Truncate long prefixes for display
        if len(prefix) > 30:
            prefix = prefix[:27] + "..."
        prefix = repr(prefix) if prefix else "(none)"
        print(
            f"{name:<18} {p['backbone']:<45} {p['max_length']:>8}  "
            f"{p['pooling']:<6}  {prefix}"
        )

    print("\nPass any preset name to --encoder, or use a raw HuggingFace model ID.")


def cmd_info(args: argparse.Namespace) -> None:
    """Display lattice metadata."""
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.store import LocalStore

    lattice_path = Path(args.lattice)
    source_root = getattr(args, "source_root", None)
    lattice = Lattice.load(
        lattice_path,
        restore_encoder=False,
        source_root=source_root,
    )
    info = lattice.info()

    print(f"Lattice: {lattice_path}")
    print(f"  Sources:    {info['source_count']}")
    print(f"  Bands:      {info['bands']}")
    print(f"  Dim:        {info['dim']}")
    print(f"  Field type: {info['field_type']}")
    print(f"  Field size: {info['field_size_mb']:.1f} MB")
    print(f"  SNR:        {info['snr']:.1f}")
    print(f"  Energy:     {info['field_energy']}")

    if "svd_rank" in info:
        print(f"  SVD rank:   {info['svd_rank']}")
    if "pq_subspaces" in info:
        print(f"  PQ M:       {info['pq_subspaces']}")
        print(f"  PQ K:       {info['pq_codebook_size']}")

    # A4: drift status for external cartridges. Always printed so users
    # know whether the store is embedded or external; --verify triggers a
    # full per-source hash audit against disk.
    if isinstance(lattice.store, LocalStore):
        print(f"  Store mode: external (source_root={lattice.store.source_root})")
        if getattr(args, "verify", False):
            import warnings
            # Suppress per-file drift warnings — we'll summarise instead.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                counts = lattice.store.verify_all(emit_warnings=False)
            total = sum(counts.values())
            print(f"  Drift check ({total} sources verified):")
            print(f"    ok:       {counts['ok']}")
            print(f"    drifted:  {counts['drifted']}")
            print(f"    missing:  {counts['missing']}")
            print(f"    unknown:  {counts['unknown']}  (pre-A3 build, no content_hash recorded)")
            if counts["drifted"]:
                print(f"  ⚠ {counts['drifted']} chunks have drifted — run `rlat refresh` to re-sync")
        else:
            print("  Drift check: skipped (pass --verify to audit)")
    else:
        print("  Store mode: embedded")


def cmd_primer(args: argparse.Namespace) -> None:
    """Orchestrate primer refresh + status across code and memory tiers.

    Subcommands:
      refresh — regenerate both primers, stamp header with git HEAD + timestamp
      status  — print current stamps and staleness without rebuilding
    """
    sub = getattr(args, "primer_command", None)
    from resonance_lattice.primer.refresh import (
        CODE_PRIMER_DEFAULT,
        MEMORY_PRIMER_DEFAULT,
        is_stale,
        read_stamp,
        refresh_primers,
    )
    repo_root = Path(".").resolve()
    if sub == "refresh":
        status = refresh_primers(
            repo_root=repo_root,
            cartridge=args.cartridge,
            memory_root=args.memory_root,
            source_root=args.source_root,
            wait_for_lock=args.wait_for_lock,
        )
        if status.get("locked"):
            print("primer refresh: skipped — another refresh is in flight",
                  file=sys.stderr)
            sys.exit(0)
        code = status.get("code", {})
        mem = status.get("memory", {})
        print(f"primer refresh: git_head={status.get('git_head', '')[:8]}")
        print(f"  code   — {'ok' if code.get('ok') else 'FAIL'}: "
              f"{code.get('path', '')} ({code.get('bytes', 0)}B) "
              f"{code.get('err', '')}")
        print(f"  memory — {'ok' if mem.get('ok') else 'FAIL'}: "
              f"{mem.get('path', '')} ({mem.get('bytes', 0)}B) "
              f"{mem.get('err', '')}")
        if not (code.get('ok') and mem.get('ok')):
            sys.exit(1)
        return
    if sub == "status":
        for name, path in (("code", CODE_PRIMER_DEFAULT),
                           ("memory", MEMORY_PRIMER_DEFAULT)):
            full = repo_root / path
            stamp = read_stamp(full)
            stale, reason = is_stale(full, repo_root=repo_root)
            print(f"{name:>6}: {path} | stale={stale} ({reason})")
            print(f"         generated-at: {stamp.get('generated_at', '(none)')}")
            print(f"         git-head:     {stamp.get('git_head', '(none)')}")
        return
    print("usage: rlat primer {refresh|status}", file=sys.stderr)
    sys.exit(2)


def cmd_refresh(args: argparse.Namespace) -> None:
    """Re-index drifted / missing / new chunks in an external-mode knowledge model.

    Preserves the field tensor where chunk hashes still match — only
    drifted / new / removed chunks trigger the forget + superpose cycle.
    See A5 / issue #220.
    """
    from resonance_lattice.refresh import refresh_cartridge

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)

    source_root = args.source_root
    if source_root is None:
        sys.exit(
            "rlat refresh requires --source-root so it can locate the "
            "current files. The knowledge model's __source_root_hint__ is "
            "advisory only; pass the directory explicitly."
        )
    output = Path(args.output) if args.output else lattice_path

    # Load the encoder the same way the rest of the CLI does so users can
    # override with --encoder / --openvino / --onnx flags. We hand the
    # encoder to refresh_cartridge rather than letting it re-load to avoid
    # double-loading the lattice.
    from resonance_lattice.lattice import Lattice
    tmp_lattice = Lattice.load(lattice_path, restore_encoder=False)
    encoder = _load_encoder(args, tmp_lattice.config.bands, tmp_lattice.config.dim, lattice=tmp_lattice)
    del tmp_lattice  # free before the real load inside refresh_cartridge

    print(f"Refreshing {lattice_path.name} against {source_root}...", file=sys.stderr)
    report = refresh_cartridge(
        cartridge_path=lattice_path,
        source_root=Path(source_root),
        encoder=encoder,
        output_path=output,
    )
    print(report)
    if output != lattice_path:
        print(f"\nRefreshed cartridge written to: {output}")


def cmd_repoint(args: argparse.Namespace) -> None:
    """Switch a knowledge model's storage mode without re-encoding.

    Works because the field tensor + registry + manifest are shared
    across local / bundled / remote — only the store section changes.
    Currently supports the two high-value transitions:

      - local <-> remote (common case: switch between a local working
        copy and a pinned upstream repo).

    Bundled transitions can be added once there's a clear use case for
    them; for now `rlat build` is the fast path to a fresh bundled cart.
    """
    import json as _json_re

    from resonance_lattice.lattice import Lattice
    from resonance_lattice.serialise import RlatHeader
    from resonance_lattice.store import SourceContent as _SC

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    output = Path(args.output) if args.output else lattice_path
    target = args.to

    with open(lattice_path, "rb") as f:
        header = RlatHeader.from_bytes(f.read(RlatHeader.SIZE))
    current = header.store_mode

    # Normalise: "local" is the canonical spelling, "external" is the
    # historical wire value — treat them as equal for the purposes of
    # "nothing to do" detection.
    def _eq_local(m: str) -> bool:
        return m in ("local", "external")

    if (target == "remote" and current == "remote") or (
        _eq_local(target) and _eq_local(current)
    ) or (target == "bundled" and current == "bundled"):
        print(f"{lattice_path.name} already in {current} mode — nothing to do.")
        return

    lattice = Lattice.load(lattice_path, restore_encoder=False)

    # ── local → remote ────────────────────────────────────────────────
    if target == "remote" and _eq_local(current):
        if not getattr(args, "url", None):
            _die(
                "`rlat repoint --to remote` requires --url <github-url>. "
                "Example: --url https://github.com/MicrosoftDocs/fabric-docs"
            )
        from resonance_lattice.remote.github import (
            GithubFetcher as _GH,
        )
        from resonance_lattice.remote.github import (
            parse_origin as _parse_origin,
        )
        origin = _parse_origin(args.url)
        fetcher = _GH(origin=origin)

        print(f"Resolving {args.url} ...", file=sys.stderr)
        sha, paths_at_sha = fetcher.list_files()

        # Validate coverage: cartridge's manifest paths must overlap
        # with what's in the upstream tree, or the repointed cartridge
        # would serve mostly missing-file placeholders.
        # Resolve the manifest robustly. LocalStore exposes it as
        # `_manifest`; a SourceStore-backed cartridge (auto-detect
        # fallback when source_root wasn't passed at load) carries it
        # as a `__source_manifest__` SourceContent row. Accept both.
        manifest = getattr(lattice.store, "_manifest", {}) or {}
        if not manifest:
            sc_m = lattice.store.retrieve("__source_manifest__")
            if sc_m is not None and sc_m.full_text:
                try:
                    manifest = _json_re.loads(sc_m.full_text)
                except Exception:
                    manifest = {}
        if not manifest:
            _die(
                f"{lattice_path.name} has no source manifest to repoint. "
                f"Rebuild with `rlat build` to produce a modern cartridge."
            )

        cartridge_paths: set[str] = set()
        for sid, entry in manifest.items():
            if sid.startswith("__"):
                continue
            if isinstance(entry, dict):
                sf = entry.get("source_file") or ""
            elif isinstance(entry, str):
                sf = entry
            else:
                sf = ""
            if sf and not Path(sf).is_absolute():
                cartridge_paths.add(sf)

        upstream_set = set(paths_at_sha)
        overlap = cartridge_paths & upstream_set
        coverage = (len(overlap) / len(cartridge_paths) * 100) if cartridge_paths else 0.0
        if not overlap:
            sample_cart = next(iter(cartridge_paths), "(none)")
            sample_up = paths_at_sha[0] if paths_at_sha else "(empty tree)"
            _die(
                f"None of the cartridge's manifest paths match any file "
                f"in {origin.base_url} at {sha[:10]}. Rebuild from the "
                f"URL directly via `rlat build {origin.base_url} ...`.\n"
                f"  Sample cartridge path: {sample_cart}\n"
                f"  Sample upstream path:  {sample_up}"
            )
        if coverage < 80:
            print(
                f"Warning: only {coverage:.0f}% of cartridge paths match "
                f"upstream at {sha[:10]}. Queries touching the missing "
                f"{100 - coverage:.0f}% will get missing-file placeholders.",
                file=sys.stderr,
            )

        origin_meta = {
            "type": "github",
            "org": origin.org,
            "repo": origin.repo,
            "ref": origin.ref,
            "commit_sha": sha,
            "base_url": origin.base_url,
        }

        # Ensure lattice.store is a LocalStore with the manifest attached
        # so save()'s "skip manifest rebuild" guard (isinstance check)
        # fires and preserves the existing 23k-entry manifest. Without
        # this, a SourceStore fallback would cause save() to walk
        # all_ids(), find only metadata rows, and emit an empty manifest.
        from resonance_lattice.store import LocalStore as _LS
        from resonance_lattice.store import SourceStore as _SS
        if not isinstance(lattice.store, _LS):
            source_store = lattice.store if isinstance(lattice.store, _SS) else None
            lattice.store = _LS(
                source_root=lattice_path.parent,  # unused for remote mode
                manifest=manifest,
                meta_store=source_store,
            )

        lattice.store.store(_SC(
            source_id="__remote_origin__",
            summary=f"pinned remote origin ({origin.org}/{origin.repo})",
            full_text=_json_re.dumps(origin_meta),
            metadata=origin_meta,
        ))
        lattice.save(output, store_mode="remote")
        print(
            f"Repointed {lattice_path.name} -> remote "
            f"(pinned at {sha[:10]}, {coverage:.0f}% of cartridge paths matched)."
        )
        if output != lattice_path:
            print(f"Written to: {output}")
        return

    # ── remote → local ────────────────────────────────────────────────
    if _eq_local(target) and current == "remote":
        # Strip __remote_origin__ so future loads don't try to reconstruct
        # a fetcher. The manifest paths stay posix-relative, so queries
        # work with --source-root pointed at a local checkout.
        lattice.store.remove("__remote_origin__")
        lattice.save(output, store_mode="local")
        print(
            f"Repointed {lattice_path.name} -> local. Use `--source-root "
            f"<dir>` at query time to resolve passages from disk."
        )
        if output != lattice_path:
            print(f"Written to: {output}")
        return

    # ── local → bundled ───────────────────────────────────────────────
    if target == "bundled" and _eq_local(current):
        if not getattr(args, "source_root", None):
            _die(
                "`rlat repoint --to bundled` from a local knowledge model requires "
                "--source-root <dir> so the source files can be packed into "
                "the .rlat. Use the same directory the knowledge model was built "
                "against."
            )
        source_root = Path(args.source_root).resolve()
        if not source_root.is_dir():
            _die(f"--source-root {source_root} is not a directory.")

        from resonance_lattice.store import LocalStore as _LS
        from resonance_lattice.store import SourceStore as _SS
        manifest = getattr(lattice.store, "_manifest", {}) or {}
        if not manifest:
            sc_m = lattice.store.retrieve("__source_manifest__")
            if sc_m is not None and sc_m.full_text:
                try:
                    manifest = _json_re.loads(sc_m.full_text)
                except Exception:
                    manifest = {}
        if not manifest:
            _die(
                f"{lattice_path.name} has no source manifest to repack. "
                f"Rebuild with `rlat build` to produce a modern cartridge."
            )

        if not isinstance(lattice.store, _LS):
            source_store = lattice.store if isinstance(lattice.store, _SS) else None
            lattice.store = _LS(
                source_root=source_root, manifest=manifest,
                meta_store=source_store,
            )
        else:
            lattice.store.source_root = source_root

        print(
            f"Packing source files from {source_root} into {output.name} ...",
            file=sys.stderr,
        )
        lattice.save(output, store_mode="bundled")
        print(f"Repointed {lattice_path.name} -> bundled (lossless, self-contained).")
        if output != lattice_path:
            print(f"Written to: {output}")
        return

    # ── remote → bundled ──────────────────────────────────────────────
    if target == "bundled" and current == "remote":
        # Fetch every manifest file from the pinned upstream (cache-first),
        # stage into a temp dir, then swap store to a LocalStore pointing
        # at the temp dir so save() can pack it. The disk cache does most
        # of the work: if the user has queried this cartridge at all, hot
        # blobs are already on disk and don't hit the network.
        import shutil
        import tempfile

        from resonance_lattice.remote import DiskCache as _DC
        from resonance_lattice.remote.github import (
            GithubFetcher as _GHF,
        )
        from resonance_lattice.remote.github import (
            GithubOrigin as _GHO,
        )
        from resonance_lattice.store import LocalStore as _LS
        from resonance_lattice.store import SourceStore as _SS

        origin_sc = lattice.store.retrieve("__remote_origin__")
        if origin_sc is None:
            _die(f"{lattice_path.name} has no __remote_origin__; not a remote cartridge.")
        origin_meta = _json_re.loads(origin_sc.full_text)
        if origin_meta.get("type") != "github":
            _die(
                f"Only github-type remote origins are supported, got "
                f"{origin_meta.get('type')!r}."
            )
        origin = _GHO(
            org=origin_meta["org"], repo=origin_meta["repo"],
            ref=origin_meta.get("ref"),
        )
        fetcher = _GHF(origin=origin)
        cache = _DC()
        sha = origin_meta["commit_sha"]
        origin_key = origin.key
        path_prefix_raw = origin_meta.get("path_prefix")
        path_prefix = (path_prefix_raw.strip("/") + "/") if path_prefix_raw else ""

        manifest = getattr(lattice.store, "_manifest", {}) or {}
        if not manifest:
            sc_m = lattice.store.retrieve("__source_manifest__")
            if sc_m is not None and sc_m.full_text:
                manifest = _json_re.loads(sc_m.full_text)

        rel_paths: set[str] = set()
        for sid, entry in manifest.items():
            if sid.startswith("__"):
                continue
            if isinstance(entry, dict):
                sf = entry.get("source_file", "")
            elif isinstance(entry, str):
                sf = entry
            else:
                continue
            if sf and not Path(sf).is_absolute():
                rel_paths.add(sf)
        if not rel_paths:
            _die("Remote knowledge model has no resolvable manifest paths to pack.")

        tmp_root = Path(tempfile.mkdtemp(prefix=f"rlat-repoint-{origin_key}-"))
        try:
            print(
                f"Staging {len(rel_paths)} files from {origin.base_url}@{sha[:10]} "
                f"(cache-first) ...",
                file=sys.stderr,
            )
            fetched = 0
            skipped = 0
            for i, rel in enumerate(sorted(rel_paths), 1):
                upstream_rel = path_prefix + rel
                cached = cache.get(origin_key, sha, upstream_rel)
                if cached is None:
                    try:
                        cached = fetcher.fetch(sha, upstream_rel)
                    except Exception as e:
                        skipped += 1
                        if skipped <= 5 or i % 500 == 0:
                            print(
                                f"  skip {upstream_rel}: {type(e).__name__}: {e}",
                                file=sys.stderr,
                            )
                        continue
                    try:
                        cache.put(origin_key, sha, upstream_rel, cached)
                    except OSError:
                        pass
                dest = tmp_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(cached)
                fetched += 1
                if i % 500 == 0:
                    print(f"  {i}/{len(rel_paths)} files staged ...", file=sys.stderr)
            print(
                f"  staged {fetched} files ({skipped} skipped). Packing ...",
                file=sys.stderr,
            )

            # Swap store to LocalStore over the temp dir; drop
            # __remote_origin__ since the new cartridge is self-contained.
            source_store = None
            if hasattr(lattice.store, "_meta_store"):
                source_store = lattice.store._meta_store
            if source_store is None and isinstance(lattice.store, _SS):
                source_store = lattice.store
            lattice.store = _LS(
                source_root=tmp_root, manifest=manifest,
                meta_store=source_store,
            )
            lattice.store.remove("__remote_origin__")
            lattice.save(output, store_mode="bundled")
            print(
                f"Repointed {lattice_path.name} -> bundled "
                f"(from remote pin {sha[:10]}, {fetched} files packed, "
                f"{skipped} unreachable)."
            )
            if output != lattice_path:
                print(f"Written to: {output}")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
        return

    _die(
        f"`rlat repoint` does not currently support {current!r} -> {target!r}. "
        f"Supported transitions: local <-> remote, local -> bundled, "
        f"remote -> bundled. Use `rlat build` for a fresh-from-scratch "
        f"mode switch."
    )


def cmd_freshness(args: argparse.Namespace) -> None:
    """Read-only upstream drift check for a remote-mode knowledge model.

    One GitHub API call. Prints a one-line "pinned / upstream / diff"
    summary; never mutates the knowledge model or touches the disk cache.
    Exit code 0 if up-to-date, 1 if drift detected (so CI can gate on it).
    """
    from resonance_lattice.refresh import check_remote_freshness

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    try:
        report = check_remote_freshness(lattice_path)
    except ValueError as e:
        _die(str(e))

    print(report)
    sys.exit(0 if not report.is_stale() else 1)


def cmd_sync_remote(args: argparse.Namespace) -> None:
    """Apply upstream diff to a remote-mode knowledge model — the lockfile upgrade.

    Fetches only changed files via the GitHub compare API, routes them
    through the same chunk-reconciliation helper `rlat refresh` uses,
    and bumps the knowledge model's pinned commit_sha on success. Invoked by
    ``cmd_sync`` when the knowledge model is remote-mode and no source inputs
    are given.
    """
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.refresh import sync_remote_cartridge

    lattice_path = Path(args.lattice)
    output = Path(args.output) if getattr(args, "output", None) else lattice_path

    tmp = Lattice.load(lattice_path, restore_encoder=False)
    encoder = _load_encoder(args, tmp.config.bands, tmp.config.dim, lattice=tmp)
    del tmp

    print(f"Syncing {lattice_path.name} against upstream...", file=sys.stderr)
    report = sync_remote_cartridge(
        cartridge_path=lattice_path,
        encoder=encoder,
        output_path=output,
    )
    print(report)
    if output != lattice_path:
        print(f"\nSynced cartridge written to: {output}")


def cmd_diff(args: argparse.Namespace) -> None:
    """Compute corpus subtraction between two lattices."""
    from resonance_lattice.lattice import Lattice

    path_a, path_b = Path(args.lattice_a), Path(args.lattice_b)
    _require_file(path_a)
    _require_file(path_b)
    a = Lattice.load(path_a, restore_encoder=False)
    b = Lattice.load(path_b, restore_encoder=False)

    delta = a.subtract(b)

    if args.output:
        delta.save(Path(args.output))
        print(f"Delta saved: {args.output}")

    info = delta.info()
    print(f"Delta field energy: {info['field_energy']}")
    print(f"Delta sources: ~{info['source_count']}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the lattice HTTP server."""
    from resonance_lattice.server import serve

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    lattice = _load_lattice_with_encoder(args, lattice_path)

    serve(lattice, host=args.host, port=args.port)


from dataclasses import dataclass as _dc
from datetime import UTC


@_dc
class _RemoteStagingResult:
    """Temp-dir staging for a remote (GitHub) build.

    ``root`` is a posix-style path to a populated temp directory; the
    normal on-disk build pipeline runs against it. ``origin_meta`` is
    the JSON-ready record that gets written into the knowledge model as
    ``__remote_origin__`` so load() can reconstruct the fetcher.
    """

    root: Path
    origin_meta: dict


def _maybe_stage_remote_build(args: argparse.Namespace) -> _RemoteStagingResult | None:
    """Detect GitHub URL inputs and stage a temp directory with the repo.

    Returns None if no input looks like a remote origin. When a URL is
    detected, resolves it to a pinned SHA, fetches every tracked blob
    into a temp tree, and returns the staging metadata. Mixing local
    paths and remote URLs in one build is rejected up front — the
    resulting knowledge model couldn't sanely pin.
    """
    inputs = list(getattr(args, "inputs", []) or [])
    if not inputs:
        return None

    from resonance_lattice.remote.github import (
        GithubFetcher as _GH,
    )
    from resonance_lattice.remote.github import (
        parse_origin as _parse_origin,
    )

    def _is_remote(x: str) -> bool:
        x = x.strip()
        return x.startswith("https://github.com/") or x.startswith("git@github.com:")

    remote_inputs = [i for i in inputs if _is_remote(i)]
    local_inputs = [i for i in inputs if not _is_remote(i)]
    if not remote_inputs:
        return None
    if local_inputs:
        _die(
            f"`rlat build` cannot mix remote URLs with local paths. "
            f"Remote: {remote_inputs}, local: {local_inputs}. "
            f"Build separate cartridges and compose them later."
        )
    if len(remote_inputs) > 1:
        _die(
            "`rlat build` accepts at most one remote URL per knowledge model. "
            f"Got {len(remote_inputs)}: {remote_inputs}."
        )

    url = remote_inputs[0]
    origin = _parse_origin(url)
    fetcher = _GH(origin=origin)
    progress = getattr(args, "progress", False)
    path_prefix_raw = getattr(args, "path", None)
    path_prefix = path_prefix_raw.strip("/") if path_prefix_raw else None

    if progress:
        print(json.dumps({
            "phase": "remote_resolve", "url": url,
            **({"path": path_prefix} if path_prefix else {}),
        }), flush=True)
    else:
        scope_note = f" (scope: {path_prefix}/)" if path_prefix else ""
        print(f"Resolving {url}{scope_note} ...", file=sys.stderr)

    sha, paths = fetcher.list_files(path_prefix=path_prefix)
    if path_prefix and not paths:
        _die(
            f"--path {path_prefix!r} matched zero files in "
            f"{origin.org}/{origin.repo} at {sha[:10]}. Check the "
            f"subdirectory name (case-sensitive; no leading slash)."
        )
    # Filter to ingestible extensions up front so we don't pay network
    # cost for assets that the build pipeline would ignore.
    keep = [p for p in paths if Path(p).suffix.lower() in INGEST_EXTENSIONS]
    skipped = len(paths) - len(keep)
    if progress:
        print(json.dumps({
            "phase": "remote_listed", "sha": sha, "kept": len(keep), "skipped": skipped,
        }), flush=True)
    else:
        print(
            f"Pinned at {sha[:10]}... — {len(keep)} files to fetch "
            f"({skipped} non-ingest skipped)",
            file=sys.stderr,
        )

    # Materialise into a temp dir scoped to this build.
    import tempfile
    root = Path(tempfile.mkdtemp(prefix=f"rlat-remote-{origin.key}-"))
    for i, rel in enumerate(keep, 1):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = fetcher.fetch(sha, rel)
        except Exception as e:
            if progress:
                print(json.dumps({
                    "phase": "remote_fetch_warn", "path": rel, "error": str(e),
                }), flush=True)
            else:
                print(f"  skip {rel}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        dest.write_bytes(data)
        if progress and i % 25 == 0:
            print(json.dumps({
                "phase": "remote_fetch_progress", "fetched": i, "total": len(keep),
            }), flush=True)

    origin_meta = {
        "type": "github",
        "org": origin.org,
        "repo": origin.repo,
        "ref": origin.ref,
        "commit_sha": sha,
        "base_url": origin.base_url,
    }
    if path_prefix:
        origin_meta["path_prefix"] = path_prefix
    return _RemoteStagingResult(root=root, origin_meta=origin_meta)


def cmd_build(args: argparse.Namespace) -> None:
    """Build a knowledge model: create lattice + ingest documents in one step."""
    from resonance_lattice.config import Compression, FieldType, LatticeConfig, Precision
    from resonance_lattice.lattice import Lattice

    # A7: nudge users away from embedded mode. The warning only fires when
    # the caller explicitly chose embedded — the new default (external) is
    # silent. Deprecation points at the v2.0.0 removal and the replacement
    # path (`rlat pack`, landing in v2.0.0) so nobody's stranded.
    if getattr(args, "store_mode", "external") == "embedded":
        import warnings
        warnings.warn(
            "Building with --store-mode embedded is deprecated and will "
            "stop working in v2.0.0. For a self-contained knowledge model, use "
            "--store-mode bundled — same lossless semantics as external "
            "mode but with the source files packed inside the .rlat. See "
            "docs/guides/migration-to-external.md.",
            DeprecationWarning,
            stacklevel=2,
        )

    dim = 512 if getattr(args, "compact", False) else args.dim

    config = LatticeConfig(
        bands=args.bands,
        dim=dim,
        field_type=FieldType(args.field_type),
        dim_key=getattr(args, "dim_key", None),
        dim_value=getattr(args, "dim_value", None),
        precision=Precision(args.precision),
        compression=Compression(args.compression),
    )

    lattice = Lattice(config=config)
    lattice.encoder = _load_encoder(args, config.bands, config.dim)

    # Attach accelerated backbone (OpenVINO preferred, then ONNX) if requested.
    _maybe_attach_accelerated_backbone(args, lattice)

    # Remote-mode build: inputs look like GitHub URLs. Resolve to a
    # pinned commit SHA, materialise the tree into a temp directory,
    # and continue with the normal on-disk build pipeline against that
    # temp root. The __remote_origin__ record + --store-mode=remote
    # override below make load() reconstruct a RemoteStore backed by
    # GithubFetcher + ~/.cache/rlat/remote.
    _remote_staging: _RemoteStagingResult | None = _maybe_stage_remote_build(args)
    if _remote_staging is not None:
        args.inputs = [str(_remote_staging.root)]
        args.store_mode = "remote"
        # Record the pinned origin so save() carries it forward into
        # the cartridge's metadata SQLite. Stored directly on
        # self.store (the in-build SourceStore); the copy-forward loop
        # in Lattice.save picks it up like __encoder__ / __retrieval_config__.
        import json as _json_ro

        from resonance_lattice.store import SourceContent as _SC
        lattice.store.store(_SC(
            source_id="__remote_origin__",
            summary=f"pinned remote origin ({_remote_staging.origin_meta.get('org')}/{_remote_staging.origin_meta.get('repo')})",
            full_text=_json_ro.dumps(_remote_staging.origin_meta),
            metadata=_remote_staging.origin_meta,
        ))

    # Collect input files
    input_paths = [Path(s) for s in args.inputs]
    files = _collect_files(input_paths)

    if not files:
        _die(
            f"no files found to ingest.\n"
            f"  Searched: {', '.join(args.inputs)}\n"
            f"  Extensions: {', '.join(sorted(INGEST_EXTENSIONS))}"
        )

    # Show what we found
    ext_counts: dict[str, int] = {}
    for f in files:
        ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1
    ext_summary = ", ".join(f"{count} {ext}" for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]))
    progress = getattr(args, "progress", False)

    if progress:
        print(json.dumps({"phase": "scanning", "total_files": len(files), "extensions": ext_summary}), flush=True)
    else:
        print(f"Building cartridge from {len(files)} files ({ext_summary})")

    manifest = FileManifest()
    encoder_fp = _encoder_fingerprint(lattice.encoder) if lattice.encoder else ""

    fmt = getattr(args, "input_format", "")
    session = getattr(args, "session", "")
    ts = getattr(args, "timestamp", "")

    ctx_mode = getattr(args, "contextual_chunking", "auto")

    # Chunker overrides (None → auto_chunk defaults apply per format)
    max_chars = getattr(args, "max_chars", None)
    min_chars = getattr(args, "min_chars", None)
    overlap_chars = getattr(args, "overlap_chars", None)

    start = time.time()
    added, updated, skipped = _ingest_files_incremental(
        lattice, files, manifest, encoder_fp, progress=progress,
        format_override=fmt, session=session, timestamp=ts,
        contextual_chunking=ctx_mode,
        max_chars=max_chars, min_chars=min_chars, overlap_chars=overlap_chars,
    )
    elapsed = time.time() - start

    _save_manifest(lattice, manifest)

    output = Path(args.output)
    quant_bits = getattr(args, "quantize_registry", 0)

    store_mode = getattr(args, "store_mode", "embedded")
    if progress:
        print(json.dumps({"phase": "saving"}), flush=True)
    else:
        print("Saving knowledge model...", file=sys.stderr)
    lattice.save(output, registry_quantize=quant_bits, store_mode=store_mode)

    info = lattice.info()
    count = lattice.source_count
    if progress:
        print(json.dumps({"phase": "done", "chunks": count, "files": len(files),
                          "elapsed": round(elapsed, 1),
                          "field_size_mb": round(info["field_size_mb"], 1)}), flush=True)
    else:
        print(f"Built {output.name}: {count} chunks from {len(files)} files in {elapsed:.1f}s")
    print(f"  Field: {info['field_size_mb']:.1f} MB | Bands: {info['bands']} x {info['dim']}d")
    if quant_bits:
        print(f"  Registry: {quant_bits}-bit quantized")
    if store_mode == "external":
        print("  Store: external (use --source-root at query time)")

    # ── 236c: build-time retrieval-mode probe ──
    # When the user supplies a held-out qrels+queries pair, run every
    # candidate retrieval mode and persist the winner into the cartridge
    # as __retrieval_config__. `rlat search --retrieval-mode auto` reads
    # it back so each cartridge ships with its measured best mode.
    if getattr(args, "probe_qrels", None) and getattr(args, "probe_queries", None):
        from resonance_lattice.retrieval import (
            load_qrels_tsv,
            load_queries_jsonl,
            probe_modes,
        )
        from resonance_lattice.store import SourceContent

        qrels = load_qrels_tsv(args.probe_qrels)
        queries = load_queries_jsonl(args.probe_queries, qids=set(qrels.keys()))
        modes_arg = getattr(args, "probe_modes", None)
        modes = [m.strip() for m in modes_arg.split(",")] if modes_arg else None
        bm25_path = getattr(args, "bm25_index", None)
        # 238: optional reranker auto-routing. When set, the probe runs
        # each (CE-mode × reranker) combo and records the winning
        # reranker in __retrieval_config__.reranker_model.
        rerankers_arg = getattr(args, "probe_rerankers", None)
        reranker_candidates = (
            [r.strip() for r in rerankers_arg.split(",") if r.strip()]
            if rerankers_arg else None
        )
        # source_root for plus_full_stack passage expansion: the first
        # input directory is the natural root for BEIR-style builds
        # (one corpus dir → one cartridge).
        probe_source_root = args.inputs[0] if args.inputs else None

        print(f"Probing retrieval modes on {len(queries)} held-out queries...",
              file=sys.stderr)
        if reranker_candidates:
            print(
                f"  Reranker candidates ({len(reranker_candidates)}): "
                f"{', '.join(reranker_candidates)}",
                file=sys.stderr,
            )
        result = probe_modes(
            lattice, queries, qrels,
            modes=modes,
            source_root=probe_source_root,
            bm25_path=bm25_path,
            reranker_candidates=reranker_candidates,
        )
        print(f"  Winner: {result.default_mode}", file=sys.stderr)
        if result.reranker_model:
            print(f"  Reranker: {result.reranker_model}", file=sys.stderr)
        for m, s in sorted(result.scores.items(), key=lambda kv: -kv[1]):
            tag = " *" if m == result.default_mode else ""
            print(f"    {m:<32} nDCG@10 = {s:.5f}{tag}", file=sys.stderr)

        lattice.store.remove("__retrieval_config__")
        lattice.store.store(SourceContent(
            source_id="__retrieval_config__",
            summary=f"build-time probe: default_mode={result.default_mode}",
            full_text=result.to_json(),
        ))
        # Re-save so the new entry is persisted alongside the field.
        lattice.save(output, registry_quantize=quant_bits, store_mode=store_mode)

    # Update manifest if cartridge is in a .rlat/ directory
    if output.parent.name == ".rlat" or output.suffix == ".rlat":
        try:
            from resonance_lattice.discover import update_manifest
            manifest_dir = output.parent if output.parent.name == ".rlat" else output.parent
            manifest_path = manifest_dir / "manifest.json"
            source_strs = [str(f) for f in files]
            update_manifest(
                manifest_path, output, source_files=source_strs,
                source_count=count, bands=info["bands"], dim=info["dim"],
            )
        except Exception:
            pass  # manifest update is best-effort


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge two lattices into one."""
    from resonance_lattice.algebra import FieldAlgebra
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lattice import Lattice

    path_a, path_b = Path(args.lattice_a), Path(args.lattice_b)
    _require_file(path_a)
    _require_file(path_b)
    a = Lattice.load(path_a)
    b = Lattice.load(path_b)

    if not isinstance(a.field, DenseField) or not isinstance(b.field, DenseField):
        _die("merge only supports dense field backends.")

    # Validate encoder compatibility — merging fields built with different
    # heads produces meaningless results.
    fp_a = _encoder_fingerprint(a.encoder) if a.encoder else ""
    fp_b = _encoder_fingerprint(b.encoder) if b.encoder else ""
    if fp_a and fp_b and fp_a != fp_b:
        _warn("knowledge models use different encoders — merged results may be unreliable.")

    result = FieldAlgebra.merge(a.field, b.field)

    merged = Lattice(config=a.config)
    assert isinstance(merged.field, DenseField)
    merged.field.F = result.field.F
    merged.field._source_count = result.source_count
    # Carry the encoder from the first cartridge so the merged .rlat is searchable
    merged.encoder = a.encoder

    output = Path(args.output)
    merged.save(output)

    print(f"Merged: {output}")
    print(f"  Sources:      {result.source_count}")
    print(f"  Total energy: {result.total_energy:.4f}")
    print(f"  Per-band:     {result.per_band_energy.tolist()}")


def cmd_forget(args: argparse.Namespace) -> None:
    """Remove a source from a lattice with a RemovalCertificate."""
    from resonance_lattice.lattice import Lattice

    lattice_path = Path(args.lattice)
    lattice = Lattice.load(lattice_path)

    source_id = args.source

    if source_id not in lattice._phase_cache:
        _die(f"source '{source_id}' not found.")

    success = lattice.remove(source_id)
    if success:
        lattice.save(lattice_path)
        print(f"Removed source '{source_id}' from {lattice_path}")
        print(f"  Remaining sources: {lattice.source_count}")
    else:
        _die(f"failed to remove source '{source_id}'.")


def cmd_summary(args: argparse.Namespace) -> None:
    """Generate a pre-injection context primer from a knowledge model for CLAUDE.md inclusion.

    Runs multiple broad bootstrap queries against the knowledge model and compiles
    the results into a structured markdown document. This is the session-primer
    pattern (TD-10): 3-5 broad queries upfront give the AI instant project
    orientation at 77x less context than file-reading tools.
    """
    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    lattice = _load_lattice_with_encoder(args, lattice_path)
    info = lattice.info()

    fmt = getattr(args, "format", "context")

    if fmt == "stats":
        # Legacy stats-only output
        lines = [
            f"<!-- Resonance Memory: {lattice_path.name} -->",
            f"<!-- Sources: {info['source_count']} | Field: {info['field_size_mb']:.0f} MB | "
            f"Bands: {info['bands']} x {info['dim']}d -->",
        ]
        print("\n".join(lines))
        return

    # Pre-injection context primer: run broad bootstrap queries.
    # Default set is 10 questions spanning the seven lens-router intent
    # classes (factoid / explore / profile / locate / compare / contrast /
    # compose) so the primer surfaces definitions AND adjacencies AND
    # gaps, not just the top-K of a single intent.
    default_queries = [
        # Orientation / profile
        "What is this project about? What problem does it solve and what are the key components?",
        "What is the architecture and how do the main abstractions interact?",
        # Capability / factoid
        "What operations, capabilities, and APIs does this system provide?",
        "What is the primary interface and how do users invoke it?",
        # Decision / constraint
        "What are the key design decisions, tradeoffs, and constraints?",
        "What has been deprecated, removed, or reversed recently and why?",
        # Pattern / convention
        "What are the important patterns, conventions, and workflows?",
        # Locate / coverage
        "Where are the gaps in this codebase and what is not yet covered?",
        # Explore / adjacent
        "What are the related subsystems, experiments, and research threads?",
        # Evidence / measurement
        "What benchmarks, probes, and evidence exist for current quality claims?",
    ]

    queries = default_queries
    if hasattr(args, "queries") and args.queries:
        queries = [q.strip() for q in args.queries.split(";") if q.strip()]

    # Commit-aware bootstrap: derive retrieval anchors from the last N days of
    # git history so work that post-dates the static queries still surfaces.
    # Root priority: explicit --source-root, else the cartridge's parent dir.
    commit_window = getattr(args, "commit_window", 14)
    if commit_window and commit_window > 0:
        from resonance_lattice.primer.commit_topics import (
            derive_commit_topics,
            topics_to_queries,
        )
        repo_root = Path(getattr(args, "source_root", None) or lattice_path.parent).resolve()
        topics = derive_commit_topics(repo_root, since_days=commit_window)
        commit_queries = topics_to_queries(topics, since_days=commit_window)
        if commit_queries:
            print(
                f"  commit-window: seeded {len(commit_queries)} queries from "
                f"{len(topics)} recent topics ({commit_window}d)",
                file=sys.stderr,
            )
            queries = list(queries) + commit_queries

    top_k = getattr(args, "top_k", 20)

    budget = getattr(args, "budget", 2500)

    print(f"Generating context primer from {info['source_count']} sources...", file=sys.stderr)

    # ── Phase 1: Collect results from all bootstrap queries ──
    # Keep the full MaterialisedResult so we have band_scores + metadata
    from resonance_lattice.lens_router import route_query
    from resonance_lattice.materialiser import _estimate_tokens, _truncate_to_tokens

    # Route each query so we can (a) report intent distribution and (b) later
    # use the lens hint to shape ranking. Intent diversity is the point —
    # 10 queries that all land on the same lens are just one query with
    # typos.
    lens_counts: dict[str, int] = {}
    for q in queries:
        choice = route_query(q, num_cartridges=1)
        lens_counts[choice.lens] = lens_counts.get(choice.lens, 0) + 1

    seen: dict[str, tuple[float, object]] = {}  # source_id -> (score, MaterialisedResult)
    for query in queries:
        if lattice.encoder is None:
            print(f"Warning: no encoder available, skipping query: {query[:50]}...", file=sys.stderr)
            break

        result = lattice.resonate_text(query=query, top_k=top_k)
        for r in result.results:
            if not r.content or r.source_id.startswith("__"):
                continue
            text = r.content.full_text or r.content.summary or ""
            if not text:
                continue
            existing = seen.get(r.source_id)
            if existing is None or r.score > existing[0]:
                seen[r.source_id] = (r.score, r)

    # Sort by score descending
    ranked_results = [r for _, r in sorted(seen.values(), key=lambda x: -x[0])]

    if lens_counts:
        lens_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(lens_counts.items(), key=lambda kv: -kv[1])
        )
        print(
            f"  bootstrap: {len(queries)} queries across lenses [{lens_summary}], "
            f"merged to {len(ranked_results)} unique results",
            file=sys.stderr,
        )

    # ── Cross-encoder rerank of the merged bootstrap pool ──
    # Bootstrap merged dense-field wins across 10+ intents into one pool.
    # A cross-encoder pass over a union query pulls the truly primer-worthy
    # passages (high-level orientation, decisions, patterns) to the top.
    # Default ON — escape via --no-rerank or RLAT_PRIMER_RERANK=0.
    rerank_flag = getattr(args, "rerank", True)
    rerank_env = os.environ.get("RLAT_PRIMER_RERANK", "").strip()
    if rerank_env == "0":
        rerank_flag = False
    elif rerank_env == "1":
        rerank_flag = True
    if rerank_flag and len(ranked_results) > 3:
        try:
            from resonance_lattice.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            union_query = (
                "project orientation overview architecture design patterns "
                "decisions capabilities interfaces recent changes"
            )
            pre_len = len(ranked_results)
            reranked, rerank_ms = reranker.rerank(
                query=union_query,
                results=ranked_results,
                top_k=min(60, pre_len),
                skip_margin=0.0,       # always rerank at primer time
                blend_alpha=0.55,      # moderate CE weight — keep bootstrap diversity
                source_root=getattr(args, "source_root", None),
                expand_mode="natural",
            )
            ranked_results = reranked
            print(
                f"  rerank: cross-encoder reordered {pre_len} candidates "
                f"(blend=0.55) in {rerank_ms:.0f}ms",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"  rerank: skipped ({type(exc).__name__}: {exc}); "
                "set RLAT_PRIMER_RERANK=0 to silence",
                file=sys.stderr,
            )

    # ── Memory amplification ──
    # Files and keywords discussed in recent memory sessions amplify the
    # matching retrieval results. This implements the "memory.rlat as
    # primer hint" loop: what we've been working on rises in the code
    # primer. Auto-discovers ./memory next to the cartridge; override
    # via --memory-root.
    mem_root_arg = getattr(args, "memory_root", None)
    if mem_root_arg is None:
        # Auto-discover: ./memory relative to cwd OR cartridge parent
        for candidate in (Path.cwd() / "memory", lattice_path.parent / "memory"):
            if candidate.is_dir() and any(candidate.glob("*.rlat")):
                mem_root_arg = str(candidate)
                break
    if mem_root_arg:
        try:
            from resonance_lattice.layered_memory import LayeredMemory
            from resonance_lattice.primer.memory_amplify import amplify_by_memory
            memory = LayeredMemory.open(mem_root_arg, restore_encoder=False)
            code_fp = (
                _encoder_fingerprint(lattice.encoder) if lattice.encoder else ""
            )
            half_life = float(getattr(args, "recency_weight", 0.0) or 14.0)
            # recency-weight reuses the memory flag's semantics: 0 → default
            # 14-day half-life, 1 → aggressive 3-day, anything else → as-specified.
            rw = getattr(args, "recency_weight", 0.0)
            if rw and 0.0 < rw <= 1.0:
                half_life = max(3.0, 14.0 * (1.0 - rw))
            ranked_results, amp_diag = amplify_by_memory(
                ranked_results,
                memory,
                code_encoder_fp=code_fp,
                half_life_days=half_life,
                max_boost=0.15,
            )
            if amp_diag.get("applied"):
                print(
                    f"  memory-amplify: boosted {amp_diag['boosted']} of "
                    f"{len(ranked_results)} from {amp_diag['entries_seen']} memory entries "
                    f"(half-life {half_life:.1f}d)",
                    file=sys.stderr,
                )
            elif amp_diag.get("reason"):
                print(
                    f"  memory-amplify: skipped ({amp_diag['reason']})",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"  memory-amplify: skipped ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )

    # Detect corpus type early so noise filters can adapt
    _CODE_EXTS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cs", ".rb", ".c", ".cpp", ".h"}
    corpus_has_code = any(
        Path((r.content.metadata or {}).get("source_file", "")).suffix.lower() in _CODE_EXTS
        for r in ranked_results if r.content
    )

    # Code-heavy repos need more token budget for meaningful coverage.
    # When user hasn't overridden the budget (still at default), bump by 25%.
    if corpus_has_code and budget in (2500, 4000):
        budget = int(budget * 1.25)

    # Early noise filter: remove passages that will produce empty summaries
    # so they don't win dedup slots and block useful content from the same file
    def _early_clean(text: str) -> str:
        text = re.sub(r"citeturn\d+view\d+", "", text)
        lines = text.split("\n")
        lines = [l for l in lines if not re.match(r"^\s*#{1,4}\s+\S", l.strip()) and not re.match(r"^[-*_]{3,}\s*$", l.strip())]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def _early_is_noise(text: str) -> bool:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return True
        toc = sum(1 for l in lines if re.match(r"^\d+\.\s*\[", l) or re.match(r"^-\s*\[.*\]\(.*\)", l))
        if len(lines) > 3 and toc / len(lines) > 0.6:
            return True
        # Import blocks — only filter for code corpora
        if corpus_has_code:
            imp = sum(1 for l in lines if re.match(r"^(import |from |#include )", l))
            if len(lines) > 3 and imp / len(lines) > 0.5:
                return True
        _box = set("│┌┐└┘├┤┬┴┼─▼▲")
        bx = sum(1 for l in lines if any(c in _box for c in l))
        if bx > 8 and bx / len(lines) > 0.4:
            return True
        # Changelog / release notes — version headers, date stamps, "Added/Changed/Fixed"
        changelog_pat = re.compile(
            r"^\[?\d+\.\d+(\.\d+)?\]?\s*[-–—]?\s*\d{4}[-/]|"  # [0.6.0] - 2021-03-17
            r"^#{1,3}\s*\[?\d+\.\d+|"                          # ## [1.2.0]
            r"^(Added|Changed|Fixed|Removed|Deprecated|Security)\b"  # keepachangelog headings
        )
        cl = sum(1 for l in lines if changelog_pat.match(l))
        if len(lines) > 2 and cl / len(lines) > 0.3:
            return True
        # License boilerplate
        lower = text.lower()
        if ("provided by the copyright holders" in lower
                or "permission is hereby granted" in lower
                or "as is" in lower and "warranty" in lower):
            return True
        # YAML/TOML config blocks (indented key-value pairs dominating the passage)
        # Softer threshold for code corpora — config files carry architecture info
        yaml_lines = sum(1 for l in lines if re.match(r"^[\w-]+:\s*([\w\"/']|$)", l))
        yaml_threshold = 0.8 if corpus_has_code else 0.5
        if len(lines) > 4 and yaml_lines / len(lines) > yaml_threshold:
            return True
        return False

    # Source files that are almost always low-orientation (changelogs, migration guides)
    _LOW_VALUE_STEMS = {"changelog", "changes", "history", "migration", "migrating",
                        "upgrading", "upgrade", "release", "releases", "news",
                        "license", "licence", "copying", "notice", "authors",
                        "contributing", "contributors", "pull_request_template",
                        "dependabot", "renovate", "code_of_conduct", "codeowners",
                        "issue_template", "bug_report", "feature_request",
                        "funding", "security", "1-issue", "2-bug"}
    _LOW_VALUE_FILENAMES = {"package-lock.json", "cargo.lock", "go.sum",
                            "tsconfig.json", "jest.config.js", "webpack.config.js",
                            ".eslintrc", ".prettierrc", ".editorconfig",
                            "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
                            "pipfile.lock", "FUNDING.yml", "CODEOWNERS",
                            "codecov.yml", ".codecov.yml"}
    # Directories whose content is almost never orientation-relevant
    _LOW_VALUE_DIRS = {".github", ".circleci", ".buildkite", ".husky"}

    def _is_low_value_source(r) -> bool:
        sf = (r.content.metadata or {}).get("source_file", "") if r.content else ""
        if not sf:
            return False
        p = Path(sf)
        if p.stem.lower() in _LOW_VALUE_STEMS:
            return True
        if p.name.lower() in _LOW_VALUE_FILENAMES:
            return True
        # Files inside .github/ and similar dirs
        sf_lower = sf.lower().replace("\\", "/")
        if any(f"/{d}/" in sf_lower or sf_lower.startswith(f"{d}/") for d in _LOW_VALUE_DIRS):
            return True
        # Test files (multiple naming conventions)
        name_lower = p.name.lower()
        stem_lower = p.stem.lower()
        if (stem_lower.startswith("test_") or stem_lower.endswith("_test")
                or name_lower.endswith(".test.ts") or name_lower.endswith(".test.js")
                or name_lower.endswith(".test.tsx") or name_lower.endswith(".test.jsx")
                or name_lower.endswith("_test.go") or name_lower.endswith("_test.rs")
                or "/test/" in sf_lower or "/tests/" in sf_lower
                or "/__tests__/" in sf_lower):
            return True
        # Fixture / mock / snapshot files
        if any(kw in stem_lower for kw in ("fixture", "mock", "snapshot", "testdata")):
            return True
        return False

    ranked_results = [
        r for r in ranked_results
        if not _early_is_noise(_early_clean(r.content.full_text or r.content.summary or ""))
        and not _is_low_value_source(r)
    ]

    # ── Phase 2: Classify each result into a semantic section ──
    # Determine section list: custom, or auto-detect from corpus
    custom_sections = getattr(args, "sections", None)
    if custom_sections:
        section_names = [s.strip() for s in custom_sections.split(";") if s.strip()]
    else:
        if corpus_has_code:
            section_names = ["Overview", "Architecture", "Patterns & Conventions", "Reference"]
        else:
            section_names = ["Overview", "Key Concepts", "Details", "Reference"]

    sections: dict[str, list] = {name: [] for name in section_names}

    # Build keyword sets for each section for text-based classification
    # For well-known section names, use curated keyword expansions
    # For custom names, use the section name words as keywords
    _SECTION_KEYWORDS: dict[str, set[str]] = {
        "Overview": {"readme", "overview", "introduction", "about", "summary", "what is",
                     "getting started", "thesis", "purpose", "mission"},
        "Architecture": {"architecture", "design", "structure", "component", "layer",
                         "module", "abstractions", "system design", "diagram"},
        "Patterns & Conventions": {"pattern", "convention", "workflow", "practice",
                                   "guideline", "rule", "style", "process"},
        "Reference": {"reference", "api", "function", "class", "method", "endpoint",
                      "parameter", "specification"},
        "Key Concepts": {"concept", "definition", "principle", "theory", "foundation",
                         "fundamental", "model", "framework"},
        "Details": {"detail", "implementation", "analysis", "finding", "result",
                    "evidence", "data", "example", "case"},
        "Methods": {"method", "methodology", "approach", "procedure", "protocol",
                    "technique", "algorithm", "process"},
        "Findings": {"finding", "result", "outcome", "conclusion", "evidence",
                     "observation", "discovery", "insight"},
        "Background": {"background", "context", "history", "prior work", "literature",
                       "motivation", "problem statement"},
    }

    def _keywords_for_section(name: str) -> set[str]:
        """Get keyword set for a section — curated if known, derived if custom."""
        if name in _SECTION_KEYWORDS:
            return _SECTION_KEYWORDS[name]
        # For custom section names, split into lowercase words as keywords
        words = set(name.lower().split())
        # Also add the full lowercase name as a phrase
        words.add(name.lower())
        return words

    section_keywords = {name: _keywords_for_section(name) for name in section_names}

    # Global dedup: each source_file goes to one section only (highest score wins)
    global_file_assignment: dict[str, tuple[float, str]] = {}  # source_file -> (score, section)

    # Check if "Reference" is one of the sections (special handling for code files)
    has_reference_section = "Reference" in sections

    for r in ranked_results:
        meta = r.content.metadata if r.content else {}
        source_file = meta.get("source_file", "").lower()
        heading = meta.get("heading", "").lower()
        chunk_type = meta.get("chunk_type", "doc")
        file_ext = Path(source_file).suffix.lower() if source_file else ""
        file_stem = Path(source_file).stem.lower() if source_file else ""

        # Start with default (last non-Reference section, or first section)
        default_section = section_names[-1] if section_names[-1] != "Reference" else (
            section_names[-2] if len(section_names) > 1 else section_names[0]
        )
        section = default_section

        # README and top-level docs always go to Overview (first section)
        _OVERVIEW_STEMS = {"readme", "index", "overview", "introduction", "about",
                           "getting-started", "getting_started", "quickstart"}
        if file_stem in _OVERVIEW_STEMS and chunk_type == "doc":
            section = section_names[0]
        # Source code files: route to Key APIs if heading matches, else Reference
        elif has_reference_section and (chunk_type == "source" or file_ext in _CODE_EXTS):
            section = "Reference"
            if "Key APIs" in sections and heading:
                api_kws = section_keywords.get("Key APIs", set())
                if any(kw in heading for kw in api_kws):
                    section = "Key APIs"
        else:
            # Score each section by keyword matches in heading, file stem, and content
            # Require at least a heading or file_stem match (score >= 2) to override default
            text_snippet = ((r.content.full_text or "")[:300]).lower() if r.content else ""
            best_score = 0
            for sec_name in section_names:
                if sec_name == "Reference" and has_reference_section:
                    continue  # Reference is handled by file type, not keywords
                kws = section_keywords[sec_name]
                score = 0
                for kw in kws:
                    if kw in heading:
                        score += 3  # heading match is strongest signal
                    if kw in file_stem:
                        score += 2  # file stem is strong signal
                    if kw in text_snippet:
                        score += 1  # content match — weak, tiebreaker only
                if score > best_score and score >= 2:
                    best_score = score
                    section = sec_name

        # Band scores bias (only for default codebase sections)
        if not custom_sections and r.band_scores is not None and len(r.band_scores) >= 3:
            nb = len(r.band_scores)
            low = float(sum(r.band_scores[:min(2, nb)]))
            mid = float(r.band_scores[2]) if nb > 2 else 0.0
            high = float(sum(r.band_scores[min(3, nb):])) if nb > 3 else 0.0
            total = low + mid + high
            if total > 1e-8 and section not in ("Reference",):
                if low / total > 0.55:
                    section = section_names[0]  # first section = overview-like
                elif mid / total > 0.45 and len(section_names) > 1:
                    section = section_names[1]  # second section = architecture-like

        sections[section].append(r)

    # ── Phase 3: Deduplicate ──
    # 3a: Global cross-section dedup — each source_file appears in one section only
    for section_name in sections:
        for r in sections[section_name]:
            sf = (r.content.metadata or {}).get("source_file", r.source_id) if r.content else r.source_id
            prev = global_file_assignment.get(sf)
            if prev is None or r.score > prev[0]:
                global_file_assignment[sf] = (r.score, section_name)

    # Rebuild sections with global assignment
    for section_name in sections:
        sections[section_name] = [
            r for r in sections[section_name]
            if global_file_assignment.get(
                (r.content.metadata or {}).get("source_file", r.source_id) if r.content else r.source_id
            ) == (r.score, section_name)
        ]

    # 3b: Within-section dedup by source_file (keep highest-scoring chunk)
    for section_name in sections:
        by_file: dict[str, object] = {}
        for r in sections[section_name]:
            sf = (r.content.metadata or {}).get("source_file", r.source_id) if r.content else r.source_id
            if sf not in by_file or r.score > by_file[sf].score:
                by_file[sf] = r
        sections[section_name] = sorted(by_file.values(), key=lambda r: -r.score)

    # 3c: Content novelty dedup — skip near-duplicate passages across sections.
    # Two-layer gate:
    #   (a) 120-char fingerprint kills exact / near-exact repeats (cheap)
    #   (b) 4-gram shingled Jaccard kills "same content, different markdown"
    #       that slips past the fingerprint (e.g. list reordering, heading
    #       added). Threshold 0.6 is tuned for primer-length passages —
    #       below that, passages usually add genuinely new information.
    def _shingle_set(text: str, n: int = 4, cap_chars: int = 500) -> set[str]:
        norm = re.sub(r"\s+", " ", (text or "")[:cap_chars]).strip().lower()
        norm = re.sub(r"[^a-z0-9 ]", "", norm)
        words = norm.split()
        if len(words) < n:
            return set(words)
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        return inter / len(a | b)

    seen_fingerprints: set[str] = set()
    seen_shingles: list[set[str]] = []
    novelty_skipped = 0
    for section_name in list(sections.keys()):
        filtered = []
        for r in sections[section_name]:
            text = (r.content.full_text or r.content.summary or "") if r.content else ""
            # (a) Fingerprint: first 120 chars, stripped of URLs and links for dedup
            fp_text = re.sub(r"https?://\S+", "", text[:200])
            fp_text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", fp_text)
            fp = re.sub(r"\s+", " ", fp_text[:120]).strip().lower()
            if fp and fp in seen_fingerprints:
                continue
            # (b) Shingled Jaccard novelty gate
            shingles = _shingle_set(text)
            if shingles:
                # Check against only the last ~30 accepted passages to bound O(N^2)
                duplicate = False
                for prior in seen_shingles[-30:]:
                    if _jaccard(shingles, prior) >= 0.6:
                        duplicate = True
                        break
                if duplicate:
                    novelty_skipped += 1
                    continue
                seen_shingles.append(shingles)
            if fp:
                seen_fingerprints.add(fp)
            filtered.append(r)
        sections[section_name] = filtered
    if novelty_skipped:
        print(
            f"  novelty-gate: dropped {novelty_skipped} near-duplicate passages",
            file=sys.stderr,
        )

    # ── Phase 3d: Rebalance starved sections ──
    # Two promotion passes:
    #   (a) If Overview (or any non-Reference section) is under-filled while
    #       Reference is overflowing, lift doc-file passages up into the first
    #       non-Reference section. Previous threshold of <2 almost never
    #       fired; <5 matches the real "sparse" signal.
    #   (b) README-specific force-promote: top-level README / introductory
    #       docs are the project's anchor — make sure at least a few make it
    #       into Overview regardless of how full Reference is.
    _DOC_EXTS = {".md", ".rst", ".txt", ".html", ".htm"}
    _OVERVIEW_HEADING_RE = re.compile(
        r"^#{1,3}\s*(overview|introduction|about|readme|what\s+is|thesis|purpose)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    non_ref = [n for n in section_names if n != "Reference"]
    non_ref_total = sum(len(sections.get(n, [])) for n in non_ref)
    ref_count = len(sections.get("Reference", []))

    # (a) Starved-section promotion — loosened gate
    if non_ref_total < 5 and ref_count > 3 and non_ref:
        ref_results = sections.get("Reference", [])
        promoted = []
        kept = []
        for r in ref_results:
            sf = (r.content.metadata or {}).get("source_file", "") if r.content else ""
            ext = Path(sf).suffix.lower() if sf else ""
            if ext in _DOC_EXTS and len(promoted) < 5:
                promoted.append(r)
            else:
                kept.append(r)
        if promoted:
            sections["Reference"] = kept
            target = non_ref[0]  # Put in Overview
            sections[target] = promoted + sections.get(target, [])

    # (b) README force-promote — at most 3, only those whose first heading
    # looks like an orientation heading. Fires even when Overview already has
    # results, because a well-written README is the project's own primer.
    if non_ref and ref_count > 0:
        overview_sec = non_ref[0]
        ref_results = sections.get("Reference", [])
        readme_picks: list = []
        keep_rest: list = []
        for r in ref_results:
            sf = (r.content.metadata or {}).get("source_file", "") if r.content else ""
            name = Path(sf).name.lower() if sf else ""
            ext = Path(sf).suffix.lower() if sf else ""
            text = (r.content.full_text or r.content.summary or "") if r.content else ""
            is_readme_like = (
                ext in _DOC_EXTS
                and (name.startswith("readme") or _OVERVIEW_HEADING_RE.search(text[:600]) is not None)
            )
            if is_readme_like and len(readme_picks) < 3:
                readme_picks.append(r)
            else:
                keep_rest.append(r)
        if readme_picks:
            sections["Reference"] = keep_rest
            existing = sections.get(overview_sec, [])
            # README first, then whatever was already there
            sections[overview_sec] = readme_picks + existing

    # ── Phase 4: Build output with token budgets ──
    # Distribute budget: first section gets 30% (orientation), Reference gets
    # an adaptive share based on code density, remaining sections split the rest.
    code_result_count = sum(
        1 for sec_results in sections.values()
        for r in sec_results
        if (r.content.metadata or {}).get("chunk_type") == "source" if r.content
    )
    total_result_count = sum(len(v) for v in sections.values())
    code_ratio = code_result_count / max(1, total_result_count)

    def _allocate_budgets(names: list[str], total: int) -> dict[str, int]:
        budgets: dict[str, int] = {}
        remaining = total
        # First section gets orientation bonus
        budgets[names[0]] = int(total * 0.30)
        remaining -= budgets[names[0]]
        # Reference: adaptive — 20% for docs, up to 45% for pure code
        if "Reference" in names:
            ref_pct = 0.20 + 0.25 * code_ratio
            budgets["Reference"] = int(total * ref_pct)
            remaining -= budgets["Reference"]
        # Split the rest evenly among other sections
        others = [n for n in names if n not in budgets]
        if others:
            per_section = remaining // len(others)
            for n in others:
                budgets[n] = per_section
        return budgets

    token_budgets = _allocate_budgets(section_names, budget)

    total_passages = sum(len(v) for v in sections.values())

    lines = [
        f"# Project Memory: {lattice_path.stem}",
        "",
        f"<!-- Auto-generated by `rlat summary` from {lattice_path.name} -->",
        f"<!-- {info['source_count']} sources | {total_passages} passages | "
        f"{info['bands']} bands x {info['dim']}d -->",
        "",
    ]

    def _clean_passage(text: str) -> str:
        """Strip noise from passage text: citation markers, partial tables, embedded headers."""
        # Strip research citation markers (citeturnNviewN patterns)
        text = re.sub(r"citeturn\d+view\d+", "", text)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip orphaned table separator rows
            if re.match(r"^\|[-\s|:]+\|$", stripped):
                if not cleaned_lines or not cleaned_lines[-1].strip().startswith("|"):
                    continue
            # Skip lines that are just a partial table cell
            if stripped.startswith("|") and stripped.count("|") < 2 and len(stripped) < 40:
                continue
            # Skip embedded markdown section headers (## Foo) — these create fake structure
            if re.match(r"^#{1,4}\s+\S", stripped):
                continue
            # Skip horizontal rules (---, ***)
            if re.match(r"^[-*_]{3,}\s*$", stripped):
                continue
            cleaned_lines.append(line)
        # Strip leading partial table rows / sentence fragments
        while cleaned_lines:
            first = cleaned_lines[0].strip()
            # Skip leading blank lines
            if not first:
                cleaned_lines.pop(0)
                continue
            # Orphaned table cell continuation: ends with | but doesn't start with |
            if first.endswith("|") and not first.startswith("|"):
                cleaned_lines.pop(0)
                continue
            # Short fragment before a table row (passage starts mid-table)
            if (len(cleaned_lines) > 1 and cleaned_lines[1].strip().startswith("|")
                    and not first.startswith("|") and len(first) < 50):
                cleaned_lines.pop(0)
                continue
            # Leading sentence fragment: short, starts lowercase or doesn't start a sentence
            if (first and len(first) < 40 and len(cleaned_lines) > 1
                    and not first[0].isupper() and not first.startswith(("-", "*", "|"))):
                cleaned_lines.pop(0)
                continue
            break
        text = "\n".join(cleaned_lines)
        # Strip leading numbered-list prefix noise (e.g. "1. " at start of passage)
        text = re.sub(r"^\d+\.\s+", "", text)
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_low_signal(text: str) -> bool:
        """Check if passage is mostly structural noise (ToC, imports, diagrams, changelogs)."""
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return True
        # Single-char or very short fragments: {, }, ], etc.
        if len(lines) == 1 and len(lines[0]) < 5:
            return True
        # License boilerplate
        lower = text.lower()
        if ("provided by the copyright holders" in lower
                or "permission is hereby granted" in lower
                or "as is" in lower and "warranty" in lower):
            return True
        # YAML config blocks (indented key-value pairs)
        yaml_lines = sum(1 for l in lines if re.match(r"^[\w-]+:\s*([\w\"/']|$)", l) or l.startswith("- "))
        if len(lines) > 3 and yaml_lines / len(lines) > 0.6:
            return True
        # Sphinx/RST directives (:members:, :param:, :type:)
        sphinx_lines = sum(1 for l in lines if re.match(r"^:\w+", l))
        if sphinx_lines > 0 and len(lines) <= 3:
            return True
        # Raw HTTP headers / binary data blobs (b'...' byte string patterns)
        if text.count("b'") > 3:
            return True
        # Table-of-contents: mostly numbered links or bullet links
        toc_lines = sum(1 for l in lines if re.match(r"^\d+\.\s*\[", l) or re.match(r"^-\s*\[.*\]\(.*\)", l))
        if len(lines) > 3 and toc_lines / len(lines) > 0.6:
            return True
        # Changelog / release notes — version entries, date stamps
        _cl_pat = re.compile(
            r"^\[?\d+\.\d+(\.\d+)?\]?\s*[-–—]?\s*\d{4}[-/]|"
            r"^#{1,3}\s*\[?\d+\.\d+|"
            r"^(Added|Changed|Fixed|Removed|Deprecated|Security)\b"
        )
        cl = sum(1 for l in lines if _cl_pat.match(l))
        if len(lines) > 2 and cl / len(lines) > 0.3:
            return True
        # Import blocks: only check for code corpora
        if corpus_has_code:
            import_lines = sum(1 for l in lines if re.match(r"^(import |from |#include )", l))
            if len(lines) > 3 and import_lines / len(lines) > 0.5:
                return True
        # Large ASCII box diagrams: lines containing box-drawing characters
        # Small diagrams (<= 8 lines) can be useful; large ones waste tokens
        _box_chars = set("│┌┐└┘├┤┬┴┼─▼▲")
        box_lines = sum(1 for l in lines if any(c in _box_chars for c in l))
        if box_lines > 8 and box_lines / len(lines) > 0.4:
            return True
        return False

    def _is_config_or_code_fragment(text: str, is_source: bool = False) -> bool:
        """Detect passages that are just config entries or code variable assignments.

        Must distinguish 'Summary: The project does X' (prose — keep) from
        'layout: "diff, flags, files"' (config — filter).  The heuristic:
        if text after the colon contains 5+ space-separated words, it's prose.

        When is_source=True, the content is expected to be code — skip checks
        that penalise brackets and assignments.
        """
        if is_source:
            return False
        stripped = text.strip()
        if not stripped:
            return False
        lines = [l.strip() for l in stripped.split("\n") if l.strip()]
        if not lines:
            return False
        first = lines[0]
        # Short fragments (1-3 lines, <200 chars) — check for config patterns
        if len(lines) <= 3 and len(stripped) < 200:
            m = re.match(r'^([\w.-]+)\s*[:=]\s*(.*)', first)
            if m:
                value_part = m.group(2).strip()
                # Prose with 5+ words is likely documentation, not config
                word_count = len(value_part.split())
                if word_count >= 5:
                    # But pure code lines (type annotations, assignments) look long too.
                    # Heuristic: real prose has articles/prepositions; code has unquoted brackets.
                    # Strip backtick-wrapped code spans before checking for bare code syntax.
                    sans_backticks = re.sub(r'`[^`]+`', '', value_part)
                    has_bare_brackets = bool(re.search(r'[\[\]{}()]', sans_backticks))
                    has_assignment = bool(re.search(r'\b\w+\s*=\s*\w', sans_backticks))
                    if has_bare_brackets or has_assignment:
                        return True
                    return False
                # Short value: likely config (key: "val", key: val)
                return True
        # Mostly code: lines starting with variable assignments, type annotations
        code_pat = re.compile(r'^(\w+\s*[:=]|def |class |if |for |while |return |import |from )')
        code_lines = sum(1 for l in lines if code_pat.match(l))
        if len(lines) > 1 and code_lines / len(lines) > 0.6:
            return True
        return False

    def _trim_to_sentence(text: str) -> str:
        """Trim text to the last complete sentence, avoiding mid-word cuts."""
        if not text:
            return text
        # Already ends with sentence punctuation
        if text.rstrip()[-1:] in ".!?":
            return text.rstrip()
        # Find last sentence boundary (. ! ? followed by space or end)
        for end_char in (".", "!", "?"):
            idx = text.rfind(end_char)
            if idx > len(text) * 0.4:
                return text[:idx + 1]
        # No sentence boundary found — try to end at a newline or list item
        last_nl = text.rfind("\n")
        if last_nl > len(text) * 0.5:
            return text[:last_nl].rstrip()
        return text

    def _effective_summary(r, max_chars: int = 300) -> str:
        """Get a real summary — extract one at read time if summary==full_text."""
        c = r.content
        if not c:
            return ""
        chunk_type = (c.metadata or {}).get("chunk_type", "doc") if c else "doc"
        is_src = chunk_type == "source"
        if c.summary and c.summary != c.full_text:
            cleaned = _clean_passage(c.summary)
            if _is_low_signal(cleaned) or _is_config_or_code_fragment(cleaned, is_source=is_src):
                return ""
            return cleaned
        # Legacy: extract from full_text
        text = c.full_text or c.summary or ""
        if not text:
            return ""
        text = _clean_passage(text)
        if _is_low_signal(text):
            return ""
        # Heading + first sentences
        heading = (c.metadata or {}).get("heading", "")
        # Strip numbered-list prefixes from heading (e.g. "4. Architecture" -> "Architecture")
        if heading:
            heading = re.sub(r"^\d+\.\s+", "", heading)
        body = text
        body_lines = body.split("\n", 1)
        if body_lines and body_lines[0].lstrip().startswith("#"):
            body = body_lines[1].strip() if len(body_lines) > 1 else ""
        if heading and body:
            # Skip headings that are just config keys or variable assignments
            if _is_config_or_code_fragment(heading, is_source=is_src):
                return ""
            prefix = f"{heading}: "
            remaining = max_chars - len(prefix)
            if remaining > 30:
                truncated = _truncate_to_tokens(body, remaining // 4)[:remaining]
                # Ensure we end at a sentence boundary if possible
                truncated = _trim_to_sentence(truncated)
                result = prefix + truncated
                if _is_config_or_code_fragment(result, is_source=is_src):
                    return ""
                return result if len(result) > 30 else ""
        truncated = _truncate_to_tokens(text, max_chars // 4)[:max_chars]
        truncated = _trim_to_sentence(truncated)
        # Skip config-like fragments and bare code assignments
        if _is_config_or_code_fragment(truncated, is_source=is_src):
            return ""
        return truncated if len(truncated) > 30 else ""

    section_order = section_names

    for section_name in section_order:
        results = sections[section_name]
        if not results:
            continue

        token_budget = token_budgets[section_name]
        remaining_tokens = token_budget
        lines.append(f"## {section_name}")
        lines.append("")

        if section_name == "Reference":
            # File list with key headings and summaries
            for r in results:
                if remaining_tokens <= 0:
                    break
                sf = (r.content.metadata or {}).get("source_file", "") if r.content else ""
                heading = (r.content.metadata or {}).get("heading", "") if r.content else ""
                summ = (r.content.summary or "") if r.content else ""
                if sf:
                    # Skip bare entries with no heading and no summary
                    if not heading and (not summ or len(summ) < 20):
                        continue
                    entry = f"- `{Path(sf).name}`"
                    if heading and heading != "module-level":
                        entry += f" — {heading}"
                    # Include summary for richer orientation
                    if summ and len(summ) > 20 and summ != (r.content.full_text or ""):
                        summ_clean = summ.strip().split("\n")[0][:150].strip()
                        # Skip summaries that are just file path comments or imports
                        if (summ_clean and summ_clean not in entry
                                and not summ_clean.startswith("// file:")
                                and not summ_clean.startswith("from ")
                                and not summ_clean.startswith("import ")):
                            entry += f"\n  {summ_clean}"
                    lines.append(entry)
                    remaining_tokens -= _estimate_tokens(entry)
            lines.append("")
        else:
            # Summary bullets for overview/architecture/patterns
            for r in results:
                if remaining_tokens <= 0:
                    break
                summ = _effective_summary(r, max_chars=remaining_tokens * 4)
                if not summ and section_name == section_names[0]:
                    # Overview fallback: strip HTML and use raw text from README/index
                    raw = (r.content.full_text or "") if r.content else ""
                    if raw:
                        # Strip HTML tags, badges, images
                        raw = re.sub(r"<[^>]+>", "", raw)
                        raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)  # markdown images
                        raw = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", raw)  # badge links
                        raw = raw.strip()
                        if len(raw) > 30:
                            summ = _truncate_to_tokens(raw, min(remaining_tokens, 200))
                            summ = _trim_to_sentence(summ)
                if not summ:
                    continue
                summ = summ.strip().replace("\n\n\n", "\n\n")
                # Build provenance tag so readers can grep-verify the claim
                meta = (r.content.metadata or {}) if r.content else {}
                sf = meta.get("source_file", "") or meta.get("path", "")
                heading = meta.get("heading", "") or ""
                prov_parts = []
                if sf:
                    prov_parts.append(Path(sf).name)
                if heading and heading.lower() not in ("", "module-level"):
                    prov_parts.append(heading[:60])
                prov = f" *[src: {':'.join(prov_parts)}]*" if prov_parts else ""
                # Always use bullet format; indent continuation lines
                if "\n" in summ:
                    first, rest = summ.split("\n", 1)
                    indented = "\n".join("  " + l for l in rest.split("\n"))
                    entry = f"- {first}\n{indented}{prov}"
                else:
                    entry = f"- {summ}{prov}"
                tokens_used = _estimate_tokens(entry)
                if tokens_used > remaining_tokens:
                    entry = _truncate_to_tokens(entry, remaining_tokens)
                    tokens_used = _estimate_tokens(entry)
                lines.append(entry)
                lines.append("")
                remaining_tokens -= tokens_used

    source_root = getattr(args, "source_root", None)
    # Use just the filename in the footer command — absolute paths are machine-specific
    resonate_cmd = f'rlat resonate {lattice_path.name} "your question here"'
    if source_root:
        resonate_cmd += f" --source-root {source_root}"
    lines.extend([
        "---",
        "",
        "For deeper context, query the knowledge model directly:",
        "```",
        resonate_cmd,
        "```",
    ])

    summary_text = "\n".join(lines)
    est_tokens = _estimate_tokens(summary_text)

    if args.output:
        Path(args.output).write_text(summary_text, encoding="utf-8")
        print(f"Context primer written to {args.output} "
              f"({len(summary_text)} chars, ~{est_tokens} tokens, "
              f"{total_passages} passages from {len(queries)} queries)",
              file=sys.stderr)
    else:
        sys.stdout.buffer.write(summary_text.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")


def cmd_resonate(args: argparse.Namespace) -> None:
    """Query a lattice and return compressed context (optimised for AI tool injection)."""
    lattice_path = Path(args.lattice)
    lattice = _load_lattice_with_encoder(args, lattice_path)

    start = time.perf_counter()
    result = lattice.resonate_text(query=args.query, top_k=args.top_k)
    elapsed = (time.perf_counter() - start) * 1000

    mode = getattr(args, "mode", None)
    system_prompt = _get_system_prompt(mode, getattr(args, "custom_prompt", None)) if mode else None

    if args.format == "json":
        output = {
            "query": args.query,
            "latency_ms": round(elapsed, 2),
            "results": [
                {
                    "source_id": r.source_id,
                    "score": round(r.score, 4),
                    "raw_score": round(r.raw_score, 4) if getattr(r, "raw_score", None) is not None else None,
                    "band_scores": r.band_scores.tolist() if r.band_scores is not None else None,
                    "summary": r.content.summary if r.content else None,
                    "full_text": r.content.full_text if r.content else None,
                    "source_file": (r.content.metadata or {}).get("source_file", "") if r.content else "",
                    "heading": (r.content.metadata or {}).get("heading", "") if r.content else "",
                    "provenance": getattr(r, "provenance", "dense"),
                }
                for r in result.results
            ],
            "coverage": None,
            "related": None,
            "contradictions": None,
        }
        if system_prompt:
            output["mode"] = mode
            output["system_prompt"] = system_prompt
        print(json.dumps(output, indent=2))
    elif args.format == "context":
        if system_prompt:
            print(f"[System: {mode}]")
            print(system_prompt)
            print()
        lines = []
        for r in result.results[:args.top_k]:
            if r.content:
                text = r.content.full_text or r.content.summary or ""
                if text:
                    lines.append(f"- [{r.score:.2f}] {text}")
        print("\n".join(lines))
    elif args.format == "prompt":
        payload = {
            "query": args.query,
            "results": [
                {
                    "source_id": r.source_id,
                    "score": round(r.score, 4),
                    "summary": r.content.summary if r.content else None,
                    "full_text": r.content.full_text if r.content else None,
                    "source_file": (r.content.metadata or {}).get("source_file", "") if r.content else "",
                    "heading": (r.content.metadata or {}).get("heading", "") if r.content else "",
                }
                for r in result.results
            ],
            "coverage": {},
            "related_topics": [],
            "contradictions": [],
        }
        print(_dict_to_prompt(payload))
    else:
        if system_prompt:
            print(f"[Mode: {mode}]\n")
        for i, r in enumerate(result.results, 1):
            text = (r.content.full_text or r.content.summary or "")[:500] if r.content else ""
            print(f"  {i}. [{r.score:.4f}] {r.source_id}: {text}")


def _read_source_file(path: Path) -> str:
    """Read a source file, dispatching by extension for binary formats."""
    from resonance_lattice.store import LocalStore
    ext = path.suffix.lower()
    if ext == ".csv":
        return LocalStore._read_csv(path)
    elif ext == ".tsv":
        return LocalStore._read_csv(path, delimiter="\t")
    elif ext == ".docx":
        return LocalStore._read_docx(path)
    elif ext == ".pdf":
        return LocalStore._read_pdf(path)
    elif ext in (".xlsx", ".xls"):
        return LocalStore._read_xlsx(path)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def _ingest_files_incremental(lattice, files: list[Path], manifest: FileManifest,
                               encoder_fp: str = "",
                               progress: bool = False,
                               format_override: str = "",
                               session: str = "",
                               timestamp: str = "",
                               contextual_chunking: str = "auto",
                               max_chars: int | None = None,
                               min_chars: int | None = None,
                               overlap_chars: int | None = None) -> tuple[int, int, int]:
    """Ingest files incrementally using the manifest. Returns (added, updated, skipped).

    Args:
        format_override: Force a specific chunker (e.g. "conversation").
        session: Session ID to annotate all chunks with.
        timestamp: ISO timestamp to annotate all chunks with.
        contextual_chunking: "auto" (corpus-aware), "on" (always prefix),
            or "off" (never prefix).
        max_chars, min_chars, overlap_chars: chunker size overrides. When
            None, auto_chunk uses per-format defaults (1200/150/0 for
            markdown). Set via `rlat build --max-chars ... --min-chars ...`.
    """
    ctx_enabled: bool | None
    if contextual_chunking == "on":
        ctx_enabled = True
    elif contextual_chunking == "off":
        ctx_enabled = False
    else:
        ctx_enabled = None
    from resonance_lattice.chunker import (
        auto_chunk,
        chunk_conversation,
        contextualize_chunk,
        generate_summary,
    )

    added = updated = skipped = 0
    total = len(files)

    if progress:
        file_iter = files
    else:
        try:
            from tqdm import tqdm
            file_iter = tqdm(files, desc="Encoding", unit="file", file=sys.stderr)
        except ImportError:
            file_iter = files

    for idx, f in enumerate(file_iter):
        key = _canonical_path(f)
        content_hash = FileManifest.hash_file(f)

        if not manifest.needs_update(key, content_hash, encoder_fp):
            skipped += 1
            if progress:
                print(json.dumps({"phase": "encoding", "current": idx + 1, "total": total,
                                  "file": f.name, "status": "skipped"}), flush=True)
            continue

        # Remove old chunks if updating
        old_ids = manifest.remove_file(key)
        is_update = len(old_ids) > 0
        for sid in old_ids:
            lattice.remove(sid)

        # Chunk and encode — use format-aware reader for binary files
        text = _read_source_file(f)
        if format_override == "conversation":
            conv_kwargs = {}
            if max_chars is not None:
                conv_kwargs["max_chars"] = max_chars
            if min_chars is not None:
                conv_kwargs["min_chars"] = min_chars
            chunks = chunk_conversation(
                text, source_file=str(f),
                session_id=session, timestamp=timestamp,
                **conv_kwargs,
            )
        else:
            ac_kwargs = {}
            if max_chars is not None:
                ac_kwargs["max_chars"] = max_chars
            if min_chars is not None:
                ac_kwargs["min_chars"] = min_chars
            if overlap_chars is not None:
                ac_kwargs["overlap_chars"] = overlap_chars
            chunks = auto_chunk(
                text, source_file=str(f), format_override=format_override,
                **ac_kwargs,
            )

        chunk_ids = []
        batch_texts = []
        batch_encode_texts = []
        batch_sids = []
        batch_metas = []
        batch_summaries = []
        for i, chunk in enumerate(chunks):
            slug = chunk.heading[:40].replace(" ", "_").lower() if chunk.heading else ""
            path_hash = _hashlib.md5(key.encode()).hexdigest()[:4]
            sid = f"{f.stem}_{slug}_{path_hash}_{i:04d}" if slug else f"{f.stem}_{path_hash}_{i:04d}"

            # Build metadata: base fields + chunk-level metadata (conversation fields, etc.)
            meta = {
                "source_file": str(f),
                "heading": chunk.heading,
                "chunk_type": chunk.chunk_type,
                "char_offset": chunk.char_offset,
                "content_hash": chunk.content_hash,
            }
            if chunk.metadata:
                meta.update(chunk.metadata)
            # CLI-level overrides for session/timestamp
            if session and "session_id" not in meta:
                meta["session_id"] = session
            if timestamp and "timestamp" not in meta:
                meta["timestamp"] = timestamp

            # Contextual chunking: prepend file/heading/position context
            # so the encoder captures WHERE the chunk came from.
            # Clean text stored for display; contextual text used for encoding.
            contextual_text = contextualize_chunk(
                chunk, total_chunks=len(chunks), chunk_index=i,
                enabled=ctx_enabled,
            )
            batch_texts.append(chunk.text)
            batch_encode_texts.append(contextual_text)
            batch_sids.append(sid)
            batch_metas.append(meta)
            batch_summaries.append(generate_summary(chunk))
            chunk_ids.append(sid)

        if batch_texts:
            lattice.superpose_text_batch(
                texts=batch_texts,
                source_ids=batch_sids,
                metadatas=batch_metas,
                summaries=batch_summaries,
                encode_texts=batch_encode_texts,
            )

        manifest.record(key, content_hash, chunk_ids, encoder_fp)
        if is_update:
            updated += 1
        else:
            added += 1
        if progress:
            print(json.dumps({"phase": "encoding", "current": idx + 1, "total": total,
                              "file": f.name, "status": "updated" if is_update else "added",
                              "chunks": len(chunk_ids)}), flush=True)

    return added, updated, skipped


def cmd_add(args: argparse.Namespace) -> None:
    """Incrementally add files to an existing knowledge model."""
    lattice_path = Path(args.lattice)
    if not lattice_path.exists():
        _die(f"{lattice_path} not found. Use 'rlat build' to create a new cartridge.")

    lattice = _load_lattice_with_encoder(args, lattice_path)
    manifest = _load_manifest(lattice)
    encoder_fp = _encoder_fingerprint(lattice.encoder) if lattice.encoder else ""
    _check_encoder_consistency(manifest, encoder_fp, lattice=lattice)

    input_paths = [Path(s) for s in args.inputs]
    files = _collect_files(input_paths)
    if not files:
        print("No files found to add.", file=sys.stderr)
        sys.exit(1)

    fmt = getattr(args, "input_format", "")
    session = getattr(args, "session", "")
    ts = getattr(args, "timestamp", "")

    ctx_mode = getattr(args, "contextual_chunking", "auto")

    start = time.time()
    added, updated, skipped = _ingest_files_incremental(
        lattice, files, manifest, encoder_fp,
        format_override=fmt, session=session, timestamp=ts,
        contextual_chunking=ctx_mode,
    )
    elapsed = time.time() - start

    _save_manifest(lattice, manifest)
    lattice.save(lattice_path)

    total = added + updated
    print(f"{_C.green(str(total))} chunks added" if total else "No new files to add.", end="")
    if skipped:
        print(f"  ({_C.dim(f'{skipped} unchanged, skipped')})", end="")
    if updated:
        print(f"  ({_C.yellow(f'{updated} updated')})", end="")
    print(f"  {_C.dim(f'in {elapsed:.1f}s')}")
    print(f"  Sources: {lattice.source_count}")


def cmd_sync(args: argparse.Namespace) -> None:
    """Sync a knowledge model with its source of truth.

    Three-mode dispatch:
      - Remote-mode knowledge model (no local inputs): pull upstream diff from
        GitHub via ``cmd_sync_remote``. Lockfile-style upgrade.
      - External-mode knowledge model (with local inputs): existing behavior —
        add new files, update changed, remove deleted under the supplied
        source directories.
    """
    lattice_path = Path(args.lattice)
    if not lattice_path.exists():
        _die(f"{lattice_path} not found. Use 'rlat build' to create a new cartridge.")

    # Remote-mode dispatch: peek the header before doing any heavy load.
    # Remote cartridges don't take local source inputs — the origin pin
    # is the source of truth.
    inputs = list(getattr(args, "inputs", []) or [])
    try:
        from resonance_lattice.serialise import RlatHeader as _RlatHeaderSync
        with open(lattice_path, "rb") as _f:
            _hdr = _RlatHeaderSync.from_bytes(_f.read(_RlatHeaderSync.SIZE))
        store_mode = _hdr.store_mode
    except Exception:
        store_mode = "external"
    if store_mode == "remote":
        if inputs:
            _die(
                "rlat sync on a remote-mode knowledge model does not take local "
                "inputs — the knowledge model's __remote_origin__ pin is the "
                "source of truth. Re-run without directory arguments to "
                "pull the upstream diff."
            )
        cmd_sync_remote(args)
        return

    lattice = _load_lattice_with_encoder(args, lattice_path)
    manifest = _load_manifest(lattice)
    encoder_fp = _encoder_fingerprint(lattice.encoder) if lattice.encoder else ""
    _check_encoder_consistency(manifest, encoder_fp, lattice=lattice)

    input_paths = [Path(s) for s in args.inputs]
    files = _collect_files(input_paths)
    current_files = {_canonical_path(f) for f in files}

    progress = getattr(args, "progress", False)

    # Phase 1: detect deleted files
    removed = 0
    for known_file in list(manifest.known_files()):
        if known_file not in current_files:
            old_ids = manifest.remove_file(known_file)
            for sid in old_ids:
                lattice.remove(sid)
            removed += 1

    if progress:
        print(json.dumps({"phase": "scanning", "total_files": len(files), "removed": removed}), flush=True)

    # Phase 2: add new + update changed
    fmt = getattr(args, "input_format", "")
    session = getattr(args, "session", "")
    ts = getattr(args, "timestamp", "")
    ctx_mode = getattr(args, "contextual_chunking", "auto")

    start = time.time()
    added, updated, skipped = _ingest_files_incremental(
        lattice, files, manifest, encoder_fp, progress=progress,
        format_override=fmt, session=session, timestamp=ts,
        contextual_chunking=ctx_mode,
    )
    elapsed = time.time() - start

    _save_manifest(lattice, manifest)
    if progress:
        print(json.dumps({"phase": "saving"}), flush=True)
    lattice.save(lattice_path)

    if progress:
        print(json.dumps({"phase": "done", "added": added, "updated": updated,
                          "removed": removed, "skipped": skipped,
                          "sources": lattice.source_count,
                          "elapsed": round(elapsed, 1)}), flush=True)
    else:
        parts = []
        if added:
            parts.append(_C.green(f"+{added} added"))
        if updated:
            parts.append(_C.yellow(f"~{updated} updated"))
        if removed:
            parts.append(_C.red(f"-{removed} removed"))
        if skipped:
            parts.append(_C.dim(f"{skipped} unchanged"))

        print("  ".join(parts) if parts else "Nothing to sync.")
        print(f"  Sources: {lattice.source_count}  {_C.dim(f'in {elapsed:.1f}s')}")

    # Update manifest if cartridge is in a .rlat/ directory
    if lattice_path.parent.name == ".rlat" or lattice_path.suffix == ".rlat":
        try:
            from resonance_lattice.discover import update_manifest
            manifest_dir = lattice_path.parent if lattice_path.parent.name == ".rlat" else lattice_path.parent
            manifest_path = manifest_dir / "manifest.json"
            update_manifest(manifest_path, lattice_path, source_count=lattice.source_count)
        except Exception:
            pass


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two knowledge models: what they share, what's unique to each."""
    from resonance_lattice.lattice import Lattice

    path_a, path_b = Path(args.lattice_a), Path(args.lattice_b)
    _require_file(path_a)
    _require_file(path_b)
    a = Lattice.load(path_a, restore_encoder=False)
    b = Lattice.load(path_b, restore_encoder=False)

    comparison = Lattice.compare(a, b)

    if args.format == "json":
        print(json.dumps(comparison, indent=2))
    else:
        print(f"Comparing: {args.lattice_a} ({comparison['a_sources']} sources) vs {args.lattice_b} ({comparison['b_sources']} sources)")
        print()
        for line in comparison["summary_lines"]:
            print(line)
        print()
        print("Per-band detail:")
        for bp in comparison["per_band"]:
            print(f"  {bp['name']}: overlap={bp['overlap']:.0%}  A={bp['energy_a']:.2f}  B={bp['energy_b']:.2f}  "
                  f"A-B={bp['diff_a_minus_b']:.2f}  B-A={bp['diff_b_minus_a']:.2f}")


def cmd_ls(args: argparse.Namespace) -> None:
    """List sources in a knowledge model."""
    from resonance_lattice.lattice import Lattice

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    lattice = Lattice.load(lattice_path, restore_encoder=False)

    source_ids = lattice.store.all_ids()
    # Filter out internal entries
    source_ids = [sid for sid in source_ids if not sid.startswith("__")]

    # Apply grep filter if provided
    pattern = getattr(args, "grep", None)
    if pattern:
        source_ids = [sid for sid in source_ids if pattern.lower() in sid.lower()]

    total = len(source_ids)

    # Apply head limit
    head = getattr(args, "head", None)
    if head is not None and head > 0:
        source_ids = source_ids[:head]

    if args.format == "json":
        output = {
            "lattice": str(lattice_path),
            "source_count": total,
            "shown": len(source_ids),
            "sources": source_ids,
        }
        print(json.dumps(output, indent=2))
    else:
        label = f"Sources in {lattice_path} ({total})"
        if pattern:
            label += f" matching '{pattern}'"
        if head and head < total:
            label += f" (showing first {head})"
        print(f"{label}:\n")
        for sid in source_ids:
            if args.verbose:
                content = lattice.store.retrieve(sid)
                summary = (content.summary[:80] if content and content.summary else "")
                print(f"  {sid}  {summary}")
            else:
                print(f"  {sid}")


def cmd_profile(args: argparse.Namespace) -> None:
    """Show semantic profile of a knowledge model."""
    from resonance_lattice.calculus import FieldCalculus
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lattice import BAND_NAMES, Lattice

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    lattice = Lattice.load(lattice_path, restore_encoder=False)
    info = lattice.info()

    if not isinstance(lattice.field, DenseField):
        print(f"Profile requires DenseField backend (got {info['field_type']})", file=sys.stderr)
        sys.exit(1)

    B = lattice.config.bands
    band_names = list(BAND_NAMES[:B])
    if B > len(BAND_NAMES):
        band_names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))

    profile = {
        "lattice": str(lattice_path),
        "source_count": info["source_count"],
        "bands": B,
        "dim": lattice.config.dim,
        "field_size_mb": round(info["field_size_mb"], 1),
        "snr": round(info["snr"], 1),
        "band_profiles": [],
    }

    for b in range(B):
        fc = FieldCalculus.field_confidence(lattice.field, band=b, top_k=10)
        topo = lattice.eigendecompose(band=b, top_k=5)
        profile["band_profiles"].append({
            "band": b,
            "name": band_names[b],
            "energy": round(float(info["field_energy"][b]), 4),
            "effective_rank": round(fc.effective_rank, 1),
            "spectral_entropy": round(fc.spectral_entropy, 4),
            "condition_number": round(fc.condition_number, 1),
            "top_eigenvalues": [round(float(v), 4) for v in topo["eigenvalues"][:5]],
        })

    # Spectral community detection
    try:
        communities = lattice.detect_communities(n_communities=8, band=0)
        profile["communities"] = communities
    except Exception:
        logger.debug("Community detection failed", exc_info=True)
        communities = None

    if args.format == "json":
        print(json.dumps(profile, indent=2))
    else:
        print(f"Semantic Profile: {lattice_path}")
        print(f"  Sources: {info['source_count']} | Field: {info['field_size_mb']:.1f} MB | SNR: {info['snr']:.1f}")
        print()
        for bp in profile["band_profiles"]:
            print(f"  Band {bp['band']} ({bp['name']}):")
            print(f"    Energy:           {bp['energy']:.4f}")
            print(f"    Effective rank:   {bp['effective_rank']:.1f}")
            print(f"    Spectral entropy: {bp['spectral_entropy']:.4f}")
            print(f"    Condition number: {bp['condition_number']:.1f}")
            print(f"    Top eigenvalues:  {bp['top_eigenvalues']}")
            print()

        if communities and communities.get("communities"):
            print(f"  Topic Communities ({communities['n_communities']} detected):")
            print()
            for c in communities["communities"]:
                pct = c["fraction"] * 100
                coh = c["coherence"]
                reps = c["representatives"][:3]
                print(f"    Community {c['rank']:2d}: {c['size']:5d} sources ({pct:4.1f}%)  coherence={coh:.2f}")
                print(f"      Representatives: {', '.join(reps[:3])}")
            print()


def _dict_to_prompt(payload: dict) -> str:
    """Render prompt-oriented text from an EnrichedResult dict.

    Mirrors ``EnrichedResult.to_prompt()`` so that warm and cold paths
    produce contract-compatible output for ``--format prompt``.
    """
    lines: list[str] = []
    coverage = payload.get("coverage", {})
    confidence = coverage.get("confidence", 0)
    lines.append(f"## Resonance Results (confidence: {confidence:.0%})")
    lines.append("")

    # Coverage bars
    names, vals = _parse_band_energies(coverage)
    max_e = max(max(vals), 1e-8) if vals else 1e-8
    for name, energy in zip(names, vals):
        frac = energy / max_e
        bar = _safe_bar(frac, width=20)
        lines.append(f"  {name:<12} {bar} {energy:.2f}")
    lines.append("")

    gaps = coverage.get("gaps", [])
    if gaps:
        lines.append(f"**Knowledge gaps**: {', '.join(gaps)}")
        lines.append("")

    # Passages
    lines.append("### Passages")
    for i, r in enumerate(payload.get("results", []), 1):
        text = r.get("full_text", "") or r.get("summary", "") or ""
        band_scores = r.get("band_scores")
        band_str = ""
        if band_scores:
            band_str = f" bands=[{', '.join(f'{s:.2f}' for s in band_scores)}]"
        score = r.get("score", 0)
        raw_score = r.get("raw_score")
        score_str = f"{score:.3f}"
        if raw_score is not None:
            score_str += f", raw={raw_score:.3f}"
        # Display name
        source_file = r.get("source_file", "")
        heading = r.get("heading", "")
        stem = Path(source_file).stem if source_file else ""
        if stem and heading:
            display = f"{stem} / {heading}"
        elif stem:
            display = stem
        elif heading:
            display = heading
        else:
            display = r.get("source_id", "")
        lines.append(f"[{i}] ({score_str}{band_str}) {display}")
        if text:
            lines.append(f"    {_truncate(text, 500)}")
        lines.append("")

    # Related topics
    related = payload.get("related", []) or payload.get("related_topics", [])
    if related:
        lines.append("### Related Topics (via cascade)")
        for rt in related:
            summary = rt.get("summary", "") or ""
            sid = rt.get("source_id", "")
            hop = rt.get("hop", 0)
            score = rt.get("score", 0)
            lines.append(f"  - (hop {hop}, {score:.3f}) {sid}: {summary[:200]}")
        lines.append("")

    # Contradictions
    contradictions = payload.get("contradictions", [])
    if contradictions:
        lines.append("### Contradictions Detected")
        for c in contradictions:
            src_a = c.get("source_a", "")
            src_b = c.get("source_b", "")
            interf = c.get("interference", 0)
            lines.append(f"  \u26a1 {src_a} vs {src_b} (interference: {interf:.3f})")
            if c.get("summary_a"):
                lines.append(f"    A: {c['summary_a'][:150]}")
            if c.get("summary_b"):
                lines.append(f"    B: {c['summary_b'][:150]}")
        lines.append("")

    return "\n".join(lines)


def _output_search(args: argparse.Namespace, payload: dict, load_ms: float, warm: bool) -> None:
    """Format and print search results.

    *payload* is the dict form of an EnrichedResult (from to_dict() or
    from a warm worker JSON response).
    """
    if args.format == "json":
        payload["load_ms"] = round(load_ms, 2)
        payload["warm"] = warm
        print(json.dumps(payload, indent=2))
        return

    if args.format == "prompt":
        mode = getattr(args, "mode", None)
        system_prompt = _get_system_prompt(mode, getattr(args, "custom_prompt", None)) if mode else None
        if system_prompt:
            print(f"[System: {mode}]")
            print(system_prompt)
            print()
        print(_dict_to_prompt(payload))
        return

    if args.format == "context":
        mode = getattr(args, "mode", None)
        system_prompt = _get_system_prompt(mode, getattr(args, "custom_prompt", None)) if mode else None
        if system_prompt:
            print(f"[System: {mode}]")
            print(system_prompt)
            print()
        lines = []
        for r in payload.get("results", []):
            if isinstance(r, dict):
                text = r.get("full_text", "") or r.get("summary", "") or ""
                score = r.get("score", 0)
            else:
                text = (r.content.full_text or r.content.summary or "") if r.content else ""
                score = r.score
            if text:
                lines.append(f"- [{score:.2f}] {text}")
        print("\n".join(lines))
        return

    # ── text format ──
    query = payload.get("query", "")
    results = payload.get("results", [])
    coverage = payload.get("coverage", {})
    latency_ms = payload.get("latency_ms", 0)
    contradictions = payload.get("contradictions", [])
    related = payload.get("related_topics", []) or payload.get("related", [])
    timings_ms = payload.get("timings_ms")

    print()
    print(f"  {_C.bold(query)}")
    print()

    n = len(results)
    conf = coverage.get("confidence", 0)
    warm_label = _C.green("warm") if warm else f"cold, load: {load_ms:.0f}ms"
    print(f"  {_C.dim(f'{latency_ms:.0f}ms')} {_C.dim(f'({warm_label})')} {_C.dim('|')} {n} results {_C.dim('|')} confidence {_C.cyan(f'{conf:.0%}')}")

    # ── Query profile: confidence · source spread · match type ──
    unique_sources = set()
    scores = []
    for r in results:
        if isinstance(r, dict):
            sf = r.get("source_file", "")
            scores.append(r.get("score", 0))
        else:
            sf = (r.content.metadata or {}).get("source_file", "") if r.content else ""
            scores.append(r.score)
        if sf:
            unique_sources.add(Path(sf).stem)
    n_sources = len(unique_sources)
    source_label = f"{n_sources} source{'s' if n_sources != 1 else ''}"

    # Match type from score distribution
    if len(scores) >= 2:
        top5 = scores[:min(5, len(scores))]
        mean_top5 = sum(top5) / len(top5)
        if conf < 0.3:
            match_type = "weak match"
        elif mean_top5 > 0.7:
            match_type = "broad match"
        elif scores[0] - scores[1] > 0.4:
            match_type = "needle"
        else:
            match_type = "focused match"
    elif scores:
        match_type = "single hit"
    else:
        match_type = "no match"

    print(f"  {_C.dim(source_label)} {_C.dim(chr(0xB7))} {_C.dim(match_type)}")

    # ── Topic clustering: extract category from source_file paths ──
    category_counts: dict[str, int] = {}
    all_parts: list[tuple[str, ...]] = []
    for r in results:
        if isinstance(r, dict):
            sf = r.get("source_file", "")
        else:
            sf = (r.content.metadata or {}).get("source_file", "") if r.content else ""
        if sf:
            all_parts.append(Path(sf).parts)
    # Strip the longest common directory prefix so categories reflect
    # the first *differentiating* directory, regardless of corpus depth.
    if all_parts:
        prefix_len = 0
        for level_parts in zip(*all_parts):
            if len(set(level_parts)) == 1:
                prefix_len += 1
            else:
                break
        for parts in all_parts:
            remaining = parts[prefix_len:-1]  # strip prefix and filename
            if remaining:
                cat = remaining[0].replace("_", " ").replace("-", " ").title()
                category_counts[cat] = category_counts.get(cat, 0) + 1
    if category_counts:
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_parts = [f"{cat}({count})" for cat, count in sorted_cats[:5]]
        print(f"  {_C.dim('topics: ' + '  '.join(cluster_parts))}")
    print()

    # ── Results ──
    for i, r in enumerate(results, 1):
        # r may be a dict (from to_dict() or warm worker) or a MaterialisedResult object
        if isinstance(r, dict):
            source_id = r.get("source_id", "")
            score = r.get("score", 0)
            raw_score = r.get("raw_score")
            # to_dict() flattens content fields to top level
            full_text = r.get("full_text", "") or r.get("summary", "") or ""
            source_file = r.get("source_file", "")
            heading = r.get("heading", "")
            provenance = r.get("provenance", "dense")
        else:
            source_id = r.source_id
            score = r.score
            raw_score = getattr(r, "raw_score", None)
            full_text = (r.content.full_text or r.content.summary or "") if r.content else ""
            metadata = (r.content.metadata or {}) if r.content else {}
            source_file = metadata.get("source_file", "")
            heading = metadata.get("heading", "")
            provenance = getattr(r, "provenance", "dense")

        # Human-readable name
        stem = Path(source_file).stem if source_file else ""
        if stem and heading:
            name = f"{stem} / {heading}"
        elif stem:
            name = stem
        elif heading:
            name = heading
        else:
            name = source_id

        text = _truncate(_clean_passage(full_text), 160) if full_text else ""

        score_color = _C.green if score > 0.3 else (_C.yellow if score > 0.1 else _C.dim)
        score_text = f"{score:.3f}"
        if raw_score is not None and getattr(args, "verbose", False):
            score_text += f" (raw {raw_score:.3f})"
        prov_tag = ""
        if provenance in ("keyword", "hybrid"):
            prov_tag = f" {_C.yellow(f'[{provenance}]')}"
        print(f"  {_C.dim(f'{i:>2}.')} {score_color(score_text)}  {_C.bold(name)}{prov_tag}")
        if text:
            print(f"      {_C.dim(text)}")
        if source_file:
            print(f"      {_C.cyan(source_file)}")

    # ── Related ──
    if related:
        print()
        print(f"  {_C.dim('Related:')}")
        for rt in related:
            if isinstance(rt, dict):
                summary = rt.get("summary", "") or ""
                sid = rt.get("source_id", "")
            else:
                summary = ""
                if hasattr(rt, "content") and rt.content:
                    summary = rt.content.summary or ""
                elif hasattr(rt, "summary") and rt.summary:
                    summary = rt.summary
                sid = rt.source_id
            if summary:
                label = _truncate(_clean_passage(summary), 100)
            else:
                sid_parts = sid.split("_")
                label = sid_parts[0] if sid_parts else sid
            print(f"    {_C.dim('-')} {label}")

    # ── Contradictions ──
    if contradictions:
        print()
        print(f"  {_C.red('Contradictions:')}")
        for c in contradictions:
            if isinstance(c, dict):
                sa = _truncate(_clean_passage(c.get("summary_a", "") or ""), 80) or c.get("source_a", "")
                sb = _truncate(_clean_passage(c.get("summary_b", "") or ""), 80) or c.get("source_b", "")
            else:
                sa = _truncate(_clean_passage(c.summary_a), 80) if c.summary_a else c.source_a
                sb = _truncate(_clean_passage(c.summary_b), 80) if c.summary_b else c.source_b
            print(f"    {sa}")
            print(f"    {_C.red('vs')} {sb}")

    print()

    # ── Verbose timing ──
    if os.environ.get("RLAT_VERBOSE") and timings_ms:
        t = timings_ms
        parts = (
            f"load={load_ms:.0f} encode={t['encode']:.0f} resonate={t['resonate']:.0f} "
            f"registry={t['registry']:.0f} store={t['store']:.0f} "
            f"cascade={t['cascade']:.0f} total={t['total']:.0f}ms"
        )
        print(f"  {_C.dim(parts)}")


def _apply_memory_filters(payload: dict, args: argparse.Namespace) -> dict:
    """Apply conversation memory filters (session, time range, speaker, recency) to search results."""
    session_filter = getattr(args, "session", None)
    after_filter = getattr(args, "after", None)
    before_filter = getattr(args, "before", None)
    speaker_filter = getattr(args, "speaker", None)
    recency_weight = getattr(args, "recency_weight", 0.0)

    if not any([session_filter, after_filter, before_filter, speaker_filter, recency_weight]):
        return payload

    results = payload.get("results", [])
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        # If result has no metadata, check source_file and other top-level fields
        if not meta:
            # Try to extract from content metadata if available
            content = r.get("content_metadata", {})
            if content:
                meta = content

        if session_filter and meta.get("session_id") != session_filter:
            continue
        if speaker_filter and meta.get("speaker") != speaker_filter:
            continue
        if after_filter and meta.get("timestamp", "") and meta["timestamp"] < after_filter:
            continue
        if before_filter and meta.get("timestamp", "") and meta["timestamp"] > before_filter:
            continue
        filtered.append(r)

    # Apply recency weighting
    if recency_weight > 0:
        from datetime import datetime
        for r in filtered:
            meta = r.get("metadata", r.get("content_metadata", {}))
            ts = meta.get("timestamp", "")
            if ts:
                try:
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    age_days = (datetime.now(UTC) - t).total_seconds() / 86400
                    # Exponential decay: recency_score = exp(-age_days / 30)
                    import math
                    recency_score = math.exp(-age_days / 30)
                    original_score = r.get("score", 0.0)
                    r["score"] = (1 - recency_weight) * original_score + recency_weight * recency_score
                except (ValueError, TypeError):
                    pass
        # Re-sort by blended score
        filtered.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    payload["results"] = filtered
    return payload


# ─────────────────────────────────────────────────────────────────────
# B4/B5: three-layer retrieval flags (`--expand` + `--hybrid`)
# ─────────────────────────────────────────────────────────────────────
#
# Post-processes a finished search payload so the CLI flags can take
# effect without touching the field / registry / store hot path. Both
# passes are strictly fail-soft: any exception returns the original
# payload — hybrid is opportunistic, expansion is presentational, and
# we never want a broken local rg install or an unreadable source file
# to crash a search that otherwise worked.


def _apply_retrieval_flags(
    args: argparse.Namespace,
    payload: dict,
    lattice_path: Path,
    lattice: Any = None,
) -> dict:
    """Apply --expand and --hybrid post-query passes to `payload`.

    Called after the field + store materialisation has produced results
    but before `_output_search` renders them. Both flags are additive
    — default-off expand and auto-hybrid match the old behaviour when
    the source isn't reachable, so existing callers see no surprise.
    """
    try:
        return _apply_retrieval_flags_inner(args, payload, lattice_path, lattice)
    except Exception:
        return payload


def _apply_retrieval_flags_inner(
    args: argparse.Namespace,
    payload: dict,
    lattice_path: Path,
    lattice: Any,
) -> dict:
    expand_mode = getattr(args, "expand", "off")
    hybrid_arg = getattr(args, "hybrid", "off")

    if expand_mode == "off" and hybrid_arg == "off":
        return payload

    results = payload.get("results") or []
    if not results:
        return payload

    source_root = _resolve_effective_source_root(args, lattice)

    # --hybrid auto: on for external mode when source files are
    # reachable; off for embedded mode (the lexical pass would have
    # nothing to read). An explicit 'on' still requires source files —
    # we can't run rg against in-memory store bytes.
    if hybrid_arg == "auto":
        store_mode = _detect_store_mode(lattice, lattice_path)
        hybrid_on = (store_mode == "external") and (source_root is not None)
    else:
        hybrid_on = (hybrid_arg == "on") and (source_root is not None)

    if expand_mode != "off" and source_root is not None:
        for r in results:
            _try_expand_one_result(r, source_root, expand_mode)

    if hybrid_on:
        query = payload.get("query") or getattr(args, "query", "") or ""
        reordered = _try_hybrid_rerank(results, query, source_root)
        if reordered is not None:
            payload["results"] = reordered

    return payload


def _resolve_effective_source_root(
    args: argparse.Namespace, lattice: Any,
) -> Path | None:
    """Pick the directory that relative `source_file` paths resolve against.

    Priority: explicit --source-root flag > the lattice's inferred
    source_root_hint (A2 manifest metadata) > None. Returns None when
    no usable root is available; callers then skip the passes rather
    than falling back to CWD, which could silently search the wrong
    tree.
    """
    explicit = getattr(args, "source_root", None)
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p.resolve()
    if lattice is not None:
        # A2 records the build-time root in manifest metadata; Lattice
        # rehydrates it as either an attribute or via _infer_source_root.
        for attr in ("source_root_hint", "_source_root", "source_root"):
            val = getattr(lattice, attr, None)
            if val:
                p = Path(val)
                if p.is_dir():
                    return p.resolve()
    return None


def _detect_store_mode(lattice: Any, lattice_path: Path) -> str:
    """Return 'external' or 'embedded'. Cheap — peeks header only."""
    if lattice is not None:
        # Lattice carries it after A1 load.
        mode = getattr(lattice, "store_mode", None)
        if mode:
            return mode
    try:
        from resonance_lattice.serialise import RlatHeader
        with open(lattice_path, "rb") as f:
            header_bytes = f.read(RlatHeader.SIZE)
        header = RlatHeader.from_bytes(header_bytes)
        return header.store_mode
    except Exception:
        return "embedded"


def _try_expand_one_result(
    result: dict, source_root: Path, mode: str,
) -> None:
    """Replace a result's `full_text` with a boundary-expanded view.

    Mutates `result` in place. Adds `expansion_kind` so downstream
    formatters (`_output_search` text / prompt / context) can display
    which rule fired. Silent on any failure — the original full_text
    stays in place so the result still renders.
    """
    from resonance_lattice.retrieval import expand_chunk

    source_file = result.get("source_file") or ""
    full_text = result.get("full_text") or ""
    if not source_file or not full_text:
        return

    src_path = Path(source_file)
    if not src_path.is_absolute():
        src_path = (source_root / source_file).resolve()
    if not src_path.exists():
        # Try platform-separator normalisation — the A2 manifest stores
        # forward slashes regardless of build OS.
        normalised = source_file.replace("\\", "/")
        src_path = (source_root / Path(*normalised.split("/"))).resolve()
        if not src_path.exists():
            return

    try:
        source_text = src_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    # char_offset may not be in the payload on older cartridges. 0 is a
    # degraded but safe fallback — expand still finds *some* boundary,
    # just not necessarily aligned to this specific chunk.
    char_offset = int(result.get("char_offset") or 0)

    expanded = expand_chunk(
        full_text,
        char_offset,
        source_text,
        mode=mode,
        source_file=str(src_path),
    )
    if expanded.text and expanded.expansion_kind != "none":
        result["full_text"] = expanded.text
        result["expansion_kind"] = expanded.expansion_kind
        result["expansion_char_offset"] = expanded.char_offset


def _try_hybrid_rerank(
    results: list[dict], query: str, source_root: Path,
) -> list[dict] | None:
    """Rerank `results` by blending in a ripgrep lexical signal.

    Returns a new list ordered by blended score, or None on any
    failure (caller keeps the input ordering). Updates each result's
    `score` to the blended value so the displayed ranking matches the
    list order.
    """
    from resonance_lattice.retrieval import ScoredHit, lexical_rerank

    if not results or not query.strip():
        return None

    hits = [
        ScoredHit(
            source_file=r.get("source_file") or "",
            char_offset=int(r.get("char_offset") or 0),
            char_length=len(r.get("full_text") or r.get("summary") or ""),
            text=r.get("full_text") or r.get("summary") or "",
            # Prefer raw_score (unblended) when present so we're not
            # double-blending on top of an already-reranked score.
            score=float(r.get("raw_score") or r.get("score") or 0.0),
        )
        for r in results
    ]
    reranked = lexical_rerank(hits, query, source_root=source_root)
    if len(reranked) != len(results):
        return None

    # Position-match each reranked hit back to its result dict via
    # (source_file, char_offset). Two results sharing those keys would
    # be a dedup bug upstream; fall back to first-unused on collision.
    used = [False] * len(results)
    reordered: list[dict] = []
    for h in reranked:
        for i, r in enumerate(results):
            if used[i]:
                continue
            r_key = (r.get("source_file") or "", int(r.get("char_offset") or 0))
            if r_key == (h.source_file, h.char_offset):
                used[i] = True
                r["score"] = round(h.score, 4)
                r["hybrid_reranked"] = True
                reordered.append(r)
                break

    # Any unmatched (shouldn't happen if the list shape is consistent)
    # are appended in original order so no result silently vanishes.
    for i, r in enumerate(results):
        if not used[i]:
            reordered.append(r)

    return reordered


_DENSE_RETRIEVAL_MODES = {
    "auto", "field_only", "plus_cross_encoder",
    "plus_cross_encoder_expanded",
}
_FUSED_RETRIEVAL_MODES = {
    "plus_hybrid", "plus_rrf", "plus_full_stack", "bm25_only",
}


def _retrieval_mode_kwargs(mode: str) -> dict[str, object]:
    """Map --retrieval-mode value to enriched_query kwargs.

    Covers the three dense modes enriched_query handles natively.
    Fused modes (plus_rrf, plus_full_stack, bm25_only, plus_hybrid)
    require BM25 sidecar plumbing landing with 236c — this helper
    raises NotImplementedError for those and points at the follow-up.
    """
    if mode == "auto" or mode == "field_only":
        return {"enable_cross_encoder": False}
    if mode == "plus_cross_encoder":
        return {"enable_cross_encoder": True, "cross_encoder_expand": False}
    if mode == "plus_cross_encoder_expanded":
        return {"enable_cross_encoder": True, "cross_encoder_expand": True}
    if mode in _FUSED_RETRIEVAL_MODES:
        raise NotImplementedError(
            f"--retrieval-mode {mode!r} requires BM25 sidecar + fused-mode "
            f"plumbing in `rlat search` (board item 236c). Today the mode "
            f"is available via the BEIR bench harness "
            f"(`benchmarks/bench_new_arch_beir.py`) and the library "
            f"(`resonance_lattice.retrieval.modes.retrieve`). Use "
            f"--retrieval-mode plus_cross_encoder for the strongest mode "
            f"currently wired to the CLI."
        )
    raise ValueError(f"Unknown --retrieval-mode: {mode!r}")


def cmd_search(args: argparse.Namespace) -> None:
    """Enriched search: passages + coverage + optional cascade and contradictions."""
    if not args.query or not args.query.strip():
        _die("query must not be empty")
    lattice_path = Path(args.lattice)
    _require_file(lattice_path)

    # Resolve --retrieval-mode once; may raise NotImplementedError for
    # fused modes not yet wired to the CLI (see 236c).
    _retrieval_mode = getattr(args, "retrieval_mode", "auto")
    try:
        _retrieval_kwargs = _retrieval_mode_kwargs(_retrieval_mode)
    except NotImplementedError as e:
        _die(str(e))
    worker_ok = not getattr(args, "no_worker", False) and not os.environ.get("RLAT_NO_WORKER")

    # 236c: warm-worker path doesn't yet read __retrieval_config__ (the
    # worker holds the lattice in a separate process). Bypass the warm
    # path on `--retrieval-mode auto` so the cold path's cartridge-
    # config lookup is honored. Explicit modes (field_only, plus_*) keep
    # warm benefits. Worker-side config plumbing is a 236c follow-up.
    if _retrieval_mode == "auto":
        worker_ok = False

    # ── Warm path: try to reuse a running compatible worker ──
    if worker_ok:
        try:
            from resonance_lattice.worker import (
                cleanup_stale,
                probe,
                read_state,
                spawn_worker,
                warm_search,
                worker_key,
            )
            key = worker_key(lattice_path, getattr(args, "encoder", None), source_root=getattr(args, "source_root", None))
            state = read_state(key)
            if state and probe(state["port"]):
                _rerank = getattr(args, "rerank", "auto")
                _rerank_val = "auto" if _rerank == "auto" else _rerank == "true"
                params = {
                    "text": args.query,
                    "top_k": args.top_k,
                    "cascade_depth": args.cascade_depth,
                    "contradiction_threshold": args.contradiction_threshold,
                    "enable_cascade": not args.no_cascade,
                    "enable_contradictions": args.enable_contradictions,
                    "enable_rerank": _rerank_val,
                    "enable_subgraph": getattr(args, "subgraph", False),
                    "subgraph_context_k": getattr(args, "subgraph_k", 3),
                    **_retrieval_kwargs,
                }
                result = warm_search(state["port"], params)
                if result is not None:
                    result = _apply_memory_filters(result, args)
                    # B4/B5: post-query expand + hybrid. No in-process
                    # lattice here; helper peeks the cartridge header
                    # for store_mode detection.
                    result = _apply_retrieval_flags(args, result, lattice_path)
                    _output_search(args, result, load_ms=0, warm=True)
                    return
        except Exception:
            pass  # Any failure in warm path falls through to cold

    # ── Check for composition flags ──
    with_cartridges = getattr(args, "with_cartridges", []) or []
    through_cartridge = getattr(args, "through", None)
    diff_cartridge = getattr(args, "diff_against", None)
    boost_topics = getattr(args, "boost_topics", []) or []
    suppress_topics = getattr(args, "suppress_topics", []) or []
    boost_strength = getattr(args, "boost_strength", 0.5)
    suppress_strength = getattr(args, "suppress_strength", 0.3)
    explain_mode = getattr(args, "explain", False)
    has_composition = bool(with_cartridges or through_cartridge or diff_cartridge)
    lens_arg = getattr(args, "lens", None)
    has_sculpting = bool(boost_topics or suppress_topics)
    has_lens = bool(lens_arg)

    # ── EML corpus transforms ──
    sharpen_strength = getattr(args, "sharpen", None)
    soften_strength = getattr(args, "soften", None)
    contrast_cartridge = getattr(args, "contrast", None)
    tune_preset = getattr(args, "tune", None)
    has_eml = bool(
        sharpen_strength is not None
        or soften_strength is not None
        or contrast_cartridge
        or tune_preset
    )

    if has_composition or has_sculpting or has_lens or has_eml:
        # ── Composition / sculpting path ──
        from resonance_lattice.composition import ComposedCartridge

        t_load_start = time.perf_counter()
        primary = _load_lattice_with_encoder(args, lattice_path)

        if through_cartridge:
            # Semantic projection: primary through lens
            lens_path = Path(through_cartridge)
            _require_file(lens_path)
            lens_lattice = _load_lattice_with_encoder(args, lens_path)
            composed = ComposedCartridge.project(
                source={lattice_path.stem: primary},
                lens={lens_path.stem: lens_lattice},
            )
        elif diff_cartridge:
            # Semantic diff: primary minus baseline
            baseline_path = Path(diff_cartridge)
            _require_file(baseline_path)
            baseline = _load_lattice_with_encoder(args, baseline_path)
            composed = ComposedCartridge.diff(
                newer={lattice_path.stem: primary},
                older={baseline_path.stem: baseline},
            )
        elif with_cartridges:
            # Merge: primary + additional cartridges
            constituents = {lattice_path.stem: primary}
            for extra_path_str in with_cartridges:
                extra_path = Path(extra_path_str)
                _require_file(extra_path)
                extra = _load_lattice_with_encoder(args, extra_path)
                constituents[extra_path.stem] = extra
            composed = ComposedCartridge.merge(constituents)
        else:
            # Sculpting only — wrap single cartridge as composed
            composed = ComposedCartridge.merge({lattice_path.stem: primary})

        # Apply topic sculpting if requested
        if has_sculpting:
            composed = composed.sculpt_topics(
                boost_topics=boost_topics,
                suppress_topics=suppress_topics,
                boost_strength=boost_strength,
                suppress_strength=suppress_strength,
            )

        # Apply knowledge lens if requested
        if has_lens:
            from resonance_lattice.lens import Lens, LensBuilder
            if lens_arg in ("sharpen", "flatten", "denoise"):
                lens = getattr(LensBuilder, lens_arg)()
            elif Path(lens_arg).suffix == ".rlens" and Path(lens_arg).exists():
                lens = Lens.load(lens_arg)
            else:
                _die(f"Unknown lens: {lens_arg}. Use a .rlens file or one of: sharpen, flatten, denoise")
            # Apply lens to the composed field
            from resonance_lattice.composition.composed import ComposedCartridge as _CC
            viewed_field = lens.apply(composed.composed_field)
            composed = _CC(composed._constituents, viewed_field, composed._composition_type)

        # Apply EML corpus transforms if requested
        if has_eml:
            from resonance_lattice.compiler import (
                CompilationContext as _CC_ctx,
            )
            from resonance_lattice.compiler import (
                EmlContrast as _EmlContrast,
            )
            from resonance_lattice.compiler import (
                EmlSharpen as _EmlSharpen,
            )
            from resonance_lattice.compiler import (
                EmlSoften as _EmlSoften,
            )
            from resonance_lattice.compiler import (
                EmlTune as _EmlTune,
            )
            from resonance_lattice.composition.composed import ComposedCartridge as _CC
            eml_ctx = _CC_ctx()
            transformed_field = composed.composed_field

            if contrast_cartridge:
                bg_path = Path(contrast_cartridge)
                _require_file(bg_path)
                bg_lattice = _load_lattice_with_encoder(args, bg_path)
                transformed_field = _EmlContrast(bg_lattice.field).apply(
                    transformed_field, eml_ctx,
                )
            if tune_preset:
                transformed_field = _EmlTune(tune_preset).apply(
                    transformed_field, eml_ctx,
                )
            if sharpen_strength is not None:
                transformed_field = _EmlSharpen(sharpen_strength).apply(
                    transformed_field, eml_ctx,
                )
            if soften_strength is not None:
                transformed_field = _EmlSoften(soften_strength).apply(
                    transformed_field, eml_ctx,
                )

            composed = _CC(composed._constituents, transformed_field, composed._composition_type)

        load_ms = (time.perf_counter() - t_load_start) * 1000

        # Show composition diagnostics if --explain
        if explain_mode:
            from resonance_lattice.composition.diagnostics import (
                diagnose_composition,
                format_diagnostics,
            )
            diag = diagnose_composition(
                composed._constituents,
                composed_field=composed.composed_field,
            )
            sys.stderr.write(format_diagnostics(diag))
            if has_sculpting:
                sculpt_info = []
                if boost_topics:
                    sculpt_info.append(f"Boosted: {', '.join(boost_topics)} (strength={boost_strength})")
                if suppress_topics:
                    sculpt_info.append(f"Suppressed: {', '.join(suppress_topics)} (strength={suppress_strength})")
                sys.stderr.write("Topic sculpting:\n  " + "\n  ".join(sculpt_info) + "\n\n")

        # Search composed cartridge
        results = composed.search(args.query, top_k=args.top_k)

        # Format as enriched-compatible payload for _output_search
        payload = {
            "results": [
                {
                    "source_id": r.source_id,
                    "score": r.score,
                    "raw_score": r.raw_score,
                    "provenance": r.provenance,
                    "knowledge model": r.cartridge,
                    "injection_mode": composed.get_injection_mode(r.cartridge),
                    "band_scores": r.band_scores.tolist() if r.band_scores is not None else None,
                    "content": r.content.full_text[:300] if r.content and r.content.full_text else "",
                    "source_file": (r.content.metadata or {}).get("source_file", "") if r.content else "",
                }
                for r in results
            ],
            "coverage": {"confidence": 0.0, "gaps": []},
            "composition": {
                "type": composed._composition_type,
                "constituents": composed.constituent_names,
                "total_sources": composed.total_sources,
            },
            "timings_ms": {"load": round(load_ms, 2)},
        }
        payload = _apply_memory_filters(payload, args)
        # B4/B5: composition path still goes through expand+hybrid post-
        # processing. The composed lattice may not expose a uniform
        # source_root_hint, so hybrid=auto will usually resolve to off
        # unless --source-root is set — that's fine, composition is a
        # separate dimension from the three-layer retrieval story.
        payload = _apply_retrieval_flags(args, payload, lattice_path, primary)
        _output_search(args, payload, load_ms, warm=False)

    else:
        # ── Standard cold path: load in-process and query directly ──
        t_load_start = time.perf_counter()
        lattice = _load_lattice_with_encoder(args, lattice_path)
        load_ms = (time.perf_counter() - t_load_start) * 1000

        # 236c: when --retrieval-mode auto, read the cartridge's probed
        # default mode (set by `rlat build --probe-qrels`). If absent or
        # malformed, keep the field-only fallback already in
        # _retrieval_kwargs — that's the documented "auto without probe"
        # behavior.
        # 238: also read the probed reranker_model and pre-attach a
        # CrossEncoderReranker(model_name=...) to the lattice so
        # enriched_query uses the per-corpus winning reranker instead
        # of the built-in default.
        if _retrieval_mode == "auto":
            try:
                cfg_content = lattice.store.retrieve("__retrieval_config__")
                if cfg_content and cfg_content.full_text:
                    cfg = json.loads(cfg_content.full_text)
                    resolved_mode = cfg.get("default_mode")
                    if resolved_mode:
                        _retrieval_kwargs = _retrieval_mode_kwargs(resolved_mode)
                    resolved_reranker = cfg.get("reranker_model")
                    if resolved_reranker:
                        from resonance_lattice.reranker import (
                            CrossEncoderReranker,
                        )
                        lattice._cross_encoder = CrossEncoderReranker(
                            model_name=resolved_reranker,
                        )
            except Exception:
                pass  # fall through with the field-only default

        _rerank = getattr(args, "rerank", "auto")
        _rerank_val = "auto" if _rerank == "auto" else _rerank == "true"
        enriched = lattice.enriched_query(
            text=args.query,
            top_k=args.top_k,
            cascade_depth=args.cascade_depth,
            contradiction_threshold=args.contradiction_threshold,
            enable_cascade=not args.no_cascade,
            enable_contradictions=args.enable_contradictions,
            enable_rerank=_rerank_val,
            enable_subgraph=getattr(args, "subgraph", False),
            subgraph_context_k=getattr(args, "subgraph_k", 3),
            **_retrieval_kwargs,
        )

        payload = enriched.to_dict()
        if hasattr(enriched, "timings_ms"):
            payload["timings_ms"] = {k: round(v, 2) for k, v in enriched.timings_ms.items()}

        payload = _apply_memory_filters(payload, args)
        # B4/B5: standard cold path has the loaded lattice in hand so
        # we can detect store mode and source_root_hint without a
        # second file read.
        payload = _apply_retrieval_flags(args, payload, lattice_path, lattice)
        _output_search(args, payload, load_ms, warm=False)

    # ── Spawn worker for next time (fire-and-forget) ──
    if worker_ok:
        try:
            from resonance_lattice.worker import (
                cleanup_stale,
                probe,
                read_state,
                spawn_worker,
                worker_key,
            )
            key = worker_key(lattice_path, getattr(args, "encoder", None), source_root=getattr(args, "source_root", None))
            state = read_state(key)
            if not (state and probe(state["port"])):
                spawn_worker(
                    str(lattice_path),
                    key,
                    encoder=getattr(args, "encoder", None),
                    onnx=getattr(args, "onnx", None),
                )
            # Opportunistic cleanup
            cleanup_stale()
        except Exception:
            pass


def cmd_contradictions(args: argparse.Namespace) -> None:
    """Find contradictions in a lattice for a given query."""
    lattice_path = Path(args.lattice)
    lattice = _load_lattice_with_encoder(args, lattice_path)

    if lattice.encoder is None:
        _die("encoder required for contradiction detection")

    phase = lattice.encoder.encode_query(args.query)
    pairs = lattice.find_contradictions(
        query_phase=phase,
        band=args.band,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    if not pairs:
        print("No contradictions found.")
        return

    print(f"Contradictions (band {args.band}, threshold {args.threshold}):\n")
    for a_id, b_id, score in pairs:
        print(f"  [{score:.4f}] {a_id} <-> {b_id}")

        # Show content if available
        for sid in (a_id, b_id):
            contents = lattice.store.retrieve_batch([sid])
            for c in contents:
                if c.summary:
                    print(f"    {sid}: {c.summary[:100]}")


def cmd_topology(args: argparse.Namespace) -> None:
    """Analyze knowledge structure: identify clusters, robustness, and topological features."""
    from resonance_lattice.lattice import Lattice

    lattice = Lattice.load(Path(args.lattice))

    topo = lattice.eigendecompose(band=args.band, top_k=args.top_k)

    print(f"Knowledge Topology (band {args.band}):")
    print(f"  Total energy:      {topo['total_energy']:.4f}")
    print(f"  Top eigenvalues:   {topo['eigenvalues'][:5].tolist()}")
    print(f"  Explained var (5): {topo['explained_variance'][:5].tolist()}")
    print(f"  Spectral gaps:     {topo['spectral_gaps'][:5].tolist()}")

    if args.output:
        output = {
            "band": args.band,
            "total_energy": topo["total_energy"],
            "eigenvalues": topo["eigenvalues"].tolist(),
            "explained_variance": topo["explained_variance"].tolist(),
            "spectral_gaps": topo["spectral_gaps"].tolist(),
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nSaved: {args.output}")


def cmd_xray(args: argparse.Namespace) -> None:
    """Field X-Ray: corpus-level semantic diagnostics."""
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.xray import FieldXRay

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    lattice = Lattice.load(lattice_path, restore_encoder=False)

    if not isinstance(lattice.field, DenseField):
        _die(f"xray requires DenseField backend (got {lattice.info()['field_type']})")

    info = lattice.info()
    deep = getattr(args, "deep", False)

    if deep:
        result = FieldXRay.deep(lattice.field, info["source_count"], str(lattice_path), lattice=lattice)
    else:
        result = FieldXRay.quick(lattice.field, info["source_count"], str(lattice_path))

    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif fmt == "prompt":
        print(result.to_prompt())
    else:
        print(result.to_text())


def cmd_locate(args: argparse.Namespace) -> None:
    """Query positioning: where does this question sit in the knowledge landscape?"""
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.locate import QueryLocator

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)
    lattice = _load_lattice_with_encoder(args, lattice_path)

    if lattice.encoder is None:
        _die("encoder required for locate")
    if not isinstance(lattice.field, DenseField):
        _die("locate requires DenseField backend")

    phase = lattice.encoder.encode_query(args.query)
    result = QueryLocator.locate(
        lattice.field, phase.vectors, args.query,
        registry=lattice.registry, store=lattice.store,
    )

    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif fmt == "prompt":
        print(result.to_prompt())
    else:
        print(result.to_text())


def cmd_probe(args: argparse.Namespace) -> None:
    """RQL quick insight recipes."""
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.probe import RECIPES, ProbeRecipes

    recipe = args.recipe
    if recipe not in RECIPES:
        _die(f"Unknown recipe '{recipe}'. Available: {', '.join(RECIPES.keys())}")

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)

    needs_query = RECIPES[recipe]["needs_query"]
    query_text = getattr(args, "query", None) or ""
    if needs_query and not query_text.strip():
        _die(f"Recipe '{recipe}' requires a query argument")

    if needs_query:
        lattice = _load_lattice_with_encoder(args, lattice_path)
        if lattice.encoder is None:
            _die("encoder required for this recipe")
    else:
        from resonance_lattice.lattice import Lattice
        lattice = Lattice.load(lattice_path, restore_encoder=False)

    if not isinstance(lattice.field, DenseField):
        _die("probe requires DenseField backend")

    if recipe == "health":
        result = ProbeRecipes.health(lattice.field)
    elif recipe == "novelty":
        phase = lattice.encoder.encode_passage(query_text)
        result = ProbeRecipes.novelty(lattice.field, phase.vectors, query_text)
    elif recipe == "saturation":
        result = ProbeRecipes.saturation(lattice.field)
    elif recipe == "band-flow":
        result = ProbeRecipes.band_flow(lattice.field)
    elif recipe == "anti":
        phase = lattice.encoder.encode_query(query_text)
        result = ProbeRecipes.anti(lattice.field, phase.vectors, query_text)
    elif recipe == "gaps":
        result = ProbeRecipes.gaps(lattice.field)
    else:
        _die(f"Unknown recipe '{recipe}'")

    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif fmt == "prompt":
        print(result.to_prompt())
    else:
        print(result.to_text())


# ═══════════════════════════════════════════════════════════
# Reusable helpers (shared by cmd_init_project and setup wizard)
# ═══════════════════════════════════════════════════════════


def build_project_cartridge(
    files: list[Path],
    cartridge_path: Path,
    *,
    encoder_args: argparse.Namespace | None = None,
    encoder_preset: str | None = None,
    bands: int = 5,
    dim: int = 2048,
    field_type_str: str = "dense",
    precision_str: str = "f32",
    compression_str: str = "none",
    onnx_dir: str | None = None,
    quiet: bool = False,
) -> tuple[Lattice, int]:
    """Build a knowledge model from source files and save it.

    Returns (lattice, chunk_count).
    Can be driven from argparse Namespace (init-project) or from
    explicit kwargs (setup wizard).
    """
    from resonance_lattice.config import Compression, FieldType, LatticeConfig, Precision
    from resonance_lattice.lattice import Lattice

    config = LatticeConfig(
        bands=bands, dim=dim,
        field_type=FieldType(field_type_str),
        precision=Precision(precision_str),
        compression=Compression(compression_str),
    )
    lattice = Lattice(config=config)

    if encoder_args is not None:
        lattice.encoder = _load_encoder(encoder_args, config.bands, config.dim)
    else:
        lattice.encoder = _load_encoder_by_preset(encoder_preset, config.bands, config.dim)

    if onnx_dir and lattice.encoder is not None:
        try:
            from resonance_lattice.encoder_onnx import attach_onnx_backbone
            attach_onnx_backbone(lattice.encoder, onnx_dir)
        except Exception as exc:
            _warn(f"ONNX backbone failed to load: {exc}")

    from resonance_lattice.chunker import auto_chunk, generate_summary

    if not quiet:
        ext_counts: dict[str, int] = {}
        for f in files:
            ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1
        ext_summary = ", ".join(
            f"{v} {k}" for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])
        )
        print(f"Building cartridge from {len(files)} files ({ext_summary})...", file=sys.stderr)

    count = 0
    start = time.time()
    all_texts: list[str] = []
    all_sids: list[str] = []
    all_metas: list[dict] = []
    all_summaries: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = auto_chunk(text, source_file=str(f))
        for chunk in chunks:
            slug = chunk.heading[:40].replace(" ", "_").lower() if chunk.heading else ""
            sid = f"{f.stem}_{slug}_{count:06d}" if slug else f"{f.stem}_{count:06d}"
            all_texts.append(chunk.text)
            all_sids.append(sid)
            all_metas.append({
                "source_file": str(f), "heading": chunk.heading,
                "chunk_type": chunk.chunk_type,
            })
            all_summaries.append(generate_summary(chunk))
            count += 1

    lattice.superpose_text_batch(
        texts=all_texts, source_ids=all_sids, metadatas=all_metas,
        summaries=all_summaries, batch_size=64,
    )

    cartridge_path.parent.mkdir(parents=True, exist_ok=True)
    lattice.save(cartridge_path)
    elapsed = time.time() - start
    if not quiet:
        print(f"Cartridge: {cartridge_path} ({count} chunks in {elapsed:.1f}s)", file=sys.stderr)

    return lattice, count


def generate_project_summary(
    cartridge_path: Path,
    summary_path: Path,
    chunk_count: int,
    *,
    encoder_args: argparse.Namespace | None = None,
    quiet: bool = False,
) -> str:
    """Generate a semantic summary primer from a knowledge model.

    Returns the summary text.
    """
    lattice = _load_lattice_with_encoder(
        encoder_args or argparse.Namespace(encoder=None, onnx=None),
        cartridge_path,
    )

    queries = [
        "What is this project about? What problem does it solve and what are the key components?",
        "What is the architecture and how do the main abstractions interact?",
        "What operations, capabilities, and APIs does this system provide?",
        "What are the key design decisions, tradeoffs, and constraints?",
        "What are the important patterns, conventions, and workflows?",
    ]

    all_passages: dict[str, tuple[float, str, str]] = {}
    for query in queries:
        if lattice.encoder is None:
            break
        result = lattice.resonate_text(query=query, top_k=20)
        for r in result.results:
            if r.content:
                text = r.content.full_text or r.content.summary or ""
                if text and r.source_id != "__encoder__":
                    existing = all_passages.get(r.source_id)
                    if existing is None or r.score > existing[0]:
                        all_passages[r.source_id] = (r.score, text, query)

    ranked = sorted(all_passages.values(), key=lambda x: -x[0])

    section_titles = {
        "What is this project about": "Overview",
        "What is the architecture": "Architecture",
        "What operations": "Capabilities",
        "What are the key design": "Design Decisions",
        "What are the important patterns": "Patterns & Conventions",
    }

    lines = [
        f"# Project Memory: {Path('.').resolve().name}",
        "",
        f"<!-- Auto-generated by rlat from {cartridge_path} -->",
        f"<!-- {chunk_count} chunks | {len(ranked)} passages | "
        f"{len(queries)} bootstrap queries -->",
        "",
    ]

    query_groups: dict[str, list[tuple[float, str]]] = {}
    for score, text, query in ranked:
        query_groups.setdefault(query, []).append((score, text))

    for query in queries:
        passages = query_groups.get(query, [])
        if not passages:
            continue
        title = "Context"
        for prefix, t in section_titles.items():
            if query.startswith(prefix):
                title = t
                break
        lines.append(f"## {title}")
        lines.append("")
        for _score, text in passages[:8]:
            lines.append(text.strip().replace("\n\n\n", "\n\n"))
            lines.append("")

    lines.extend([
        "---",
        "",
        "For deeper context, query the knowledge model directly:",
        "```",
        f'rlat resonate {cartridge_path} "your question here"',
        "```",
    ])

    summary = "\n".join(lines)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    if not quiet:
        print(f"Summary: {summary_path} ({len(summary)} chars, {len(ranked)} passages)", file=sys.stderr)
    return summary


def wire_integration(
    cartridge_path: Path,
    files: list[Path],
    summary_path: Path,
    *,
    update_mcp: bool = True,
    update_claude: bool = True,
    quiet: bool = False,
) -> Path:
    """Generate manifest and optionally update .mcp.json and CLAUDE.md.

    Returns the manifest path.
    """
    from resonance_lattice.discover import update_manifest

    rlat_dir = cartridge_path.parent
    manifest_path = rlat_dir / "manifest.json"
    source_file_strs = [str(f) for f in files]

    manifest = update_manifest(
        manifest_path=manifest_path,
        cartridge_path=cartridge_path,
        source_files=source_file_strs,
        primer_path=str(summary_path),
        project_name=Path(".").resolve().name,
    )
    if not quiet:
        print(f"Manifest: {manifest_path} ({len(manifest.cartridges)} cartridge(s))", file=sys.stderr)

    if update_mcp:
        from resonance_lattice.discover import update_mcp_json
        if update_mcp_json(manifest_path=manifest_path):
            if not quiet:
                print("Updated .mcp.json with rlat MCP server", file=sys.stderr)

    if update_claude:
        from resonance_lattice.discover import inject_claude_md
        if inject_claude_md(manifest_path=manifest_path):
            if not quiet:
                print("Injected knowledge model section into CLAUDE.md", file=sys.stderr)

    return manifest_path


def _load_encoder_by_preset(
    preset: str | None, bands: int, dim: int,
) -> Encoder | None:
    """Load encoder from a preset name without argparse Namespace."""
    ns = argparse.Namespace(encoder=preset, onnx=None)
    return _load_encoder(ns, bands, dim)


# ═══════════════════════════════════════════════════════════
# init-project command
# ═══════════════════════════════════════════════════════════


def cmd_init_project(args: argparse.Namespace) -> None:
    """One-command project setup: build knowledge model + generate summary + suggest integration.

    This is the golden workflow in a single command:
      1. Auto-detect or use provided source directories
      2. Build a .rlat knowledge model
      3. Generate a pre-injection context primer
      4. Write it to .claude/resonance-context.md (or specified path)
      5. Print integration instructions
    """
    # Step 1: Determine input paths
    if args.inputs:
        input_paths = [Path(s) for s in args.inputs]
    else:
        input_paths = _auto_detect_inputs()
        print(f"Auto-detected: {', '.join(str(p) for p in input_paths)}", file=sys.stderr)

    files = _collect_files(input_paths)
    if not files:
        print("Error: no files found. Specify inputs: rlat init-project ./docs ./src", file=sys.stderr)
        sys.exit(1)

    # Step 2: Build cartridge
    rlat_dir = Path(".rlat")
    rlat_dir.mkdir(exist_ok=True)
    cartridge_path = rlat_dir / "project.rlat"

    _lattice, count = build_project_cartridge(
        files, cartridge_path,
        encoder_args=args,
        onnx_dir=getattr(args, "onnx", None),
    )

    # Step 3: Generate summary
    summary_path = Path(args.output) if args.output else Path(".claude") / "resonance-context.md"
    generate_project_summary(cartridge_path, summary_path, count, encoder_args=args)

    # Step 4-5: Manifest + auto-integrate
    auto_integrate = getattr(args, "auto_integrate", False)
    manifest_path = wire_integration(
        cartridge_path, files, summary_path,
        update_mcp=auto_integrate,
        update_claude=auto_integrate,
    )

    # Step 6: Integration instructions
    print(f"\n{'='*60}", file=sys.stderr)
    print("Project memory ready!", file=sys.stderr)
    print("", file=sys.stderr)
    if not auto_integrate:
        print("  Run with --auto-integrate to wire everything in automatically, or:", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Add this line to your CLAUDE.md (or .cursorrules):", file=sys.stderr)
        print(f"    @{summary_path}", file=sys.stderr)
    else:
        print("  Auto-integration complete. Your assistant now has:", file=sys.stderr)
        print(f"    - Cartridge manifest: {manifest_path}", file=sys.stderr)
        print("    - MCP server configured in .mcp.json", file=sys.stderr)
        print("    - Cartridge section in CLAUDE.md", file=sys.stderr)
    print("", file=sys.stderr)
    print("  For deeper queries:", file=sys.stderr)
    print(f'    rlat search {cartridge_path} "your question"', file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


def cmd_setup(args: argparse.Namespace) -> None:
    """Interactive setup wizard for Resonance Lattice projects."""
    from resonance_lattice.setup_wizard import run_wizard

    run_wizard(
        non_interactive=getattr(args, "non_interactive", False),
        config_path=getattr(args, "config", None),
        reconfigure=getattr(args, "reconfigure", False),
        encoder=getattr(args, "encoder", None),
        no_memory=getattr(args, "no_memory", False),
        precision=getattr(args, "precision", None),
        compression=getattr(args, "compression", None),
    )


_HELP_EPILOG = """\
primary commands:
  search        Enriched semantic query (passages + coverage + related)
  profile       Inspect knowledge model semantic shape
  compare       Compare two knowledge models
  ls            List sources in a knowledge model
  info          Show knowledge model metadata

build commands:
  build         Build a knowledge model from source files
  add           Incrementally add files (no full rebuild)
  sync          Sync with source dirs (add new, update changed, remove deleted)
  setup         Guided project setup (interactive wizard)
  init-project  One-command setup (build + summary + integration hints)

serve commands:
  serve         Start HTTP server
  summary       Generate assistant primer

algebra commands:
  merge         Combine two knowledge models
  forget        Remove a source
  diff          Compute corpus delta

skill commands:
  skill build      Build knowledge model from a skill's reference materials
  skill sync       Incrementally sync a skill's knowledge model
  skill search     Search a skill's knowledge models
  skill inject     Four-tier adaptive context injection
  skill route      Rank skills by relevance to a query
  skill profile    Semantic profile of a skill's knowledge model
  skill freshness  Check freshness of skill knowledge models
  skill gaps       Detect knowledge gaps in a skill's knowledge model
  skill compare    Compare two skills' knowledge models for overlap
  skill info       Show skill knowledge model configuration and status

export commands:
  export         Export knowledge model (supports --field-only)

environment:
  RLAT_VERBOSE=1    Show encoder loading details and timing breakdown
  RLAT_NO_WORKER=1  Disable background warm-search worker
"""


class _GroupedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Hide the flat subcommand list — the epilog has the grouped version."""
    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            return ""
        return super()._format_action(action)




def cmd_compose(args: argparse.Namespace) -> None:
    """Search with algebraic composition expressions or .rctx context files."""
    from resonance_lattice.composition import ComposedCartridge
    from resonance_lattice.composition.parser import (
        collect_cartridge_paths,
        parse_expression,
    )

    expr_or_path = args.expression

    # Check if it's a .rctx context file
    if Path(expr_or_path).suffix in (".rctx", ".json", ".yaml", ".yml") and Path(expr_or_path).exists():
        from resonance_lattice.composition.context_file import load_context, validate_context
        config = load_context(expr_or_path)
        warnings = validate_context(config)
        for w in warnings:
            sys.stderr.write(f"Warning: {w}\n")

        # Load cartridges from config
        lattices = {}
        for alias, path in config.cartridges.items():
            lattices[alias] = _load_lattice_with_encoder(args, Path(path))

        # Build composition
        if config.expression:
            # Parse expression using aliases
            ast = parse_expression(config.expression)
            composed = _eval_compose_ast(ast, lattices)
        elif config.weights:
            composed = ComposedCartridge.merge(lattices, weights=config.weights)
        else:
            composed = ComposedCartridge.merge(lattices)

        # Apply config options
        if config.boost or config.suppress:
            composed = composed.sculpt_topics(
                boost_topics=config.boost,
                suppress_topics=config.suppress,
                boost_strength=config.boost_strength,
                suppress_strength=config.suppress_strength,
            )
        if config.injection_modes:
            composed.set_injection_modes(config.injection_modes)
        if config.lens:
            from resonance_lattice.lens import Lens, LensBuilder
            if config.lens in ("sharpen", "flatten", "denoise"):
                lens = getattr(LensBuilder, config.lens)()
            else:
                lens = Lens.load(config.lens)
            viewed = lens.apply(composed.composed_field)
            composed = ComposedCartridge(composed._constituents, viewed, composed._composition_type)

    else:
        # Parse as expression
        ast = parse_expression(expr_or_path)
        paths = collect_cartridge_paths(ast)

        # Load all referenced cartridges
        lattices = {}
        for p in paths:
            path = Path(p)
            if not path.exists():
                _die(f"Cartridge not found: {p}")
            stem = path.stem
            if stem not in lattices:
                lattices[stem] = _load_lattice_with_encoder(args, path)

        composed = _eval_compose_ast(ast, lattices)

    # Show diagnostics if --explain
    if getattr(args, "explain", False):
        from resonance_lattice.composition.diagnostics import (
            diagnose_composition,
            format_diagnostics,
        )
        diag = diagnose_composition(composed._constituents, composed.composed_field)
        sys.stderr.write(format_diagnostics(diag))

    # Search
    results = composed.search(args.query, top_k=args.top_k)

    # Output
    if getattr(args, "format", "text") == "json":
        import json as _json
        payload = [
            {
                "source_id": r.source_id,
                "score": r.score,
                "knowledge model": r.cartridge,
                "content": r.content.full_text[:300] if r.content and r.content.full_text else "",
            }
            for r in results
        ]
        print(_json.dumps(payload, indent=2))
    else:
        for i, r in enumerate(results, 1):
            sf = (r.content.metadata or {}).get("source_file", "") if r.content else ""
            excerpt = r.content.full_text[:200] if r.content and r.content.full_text else ""
            print(f"{i}. [{r.score:.3f}] [{r.cartridge}] {r.source_id}")
            if sf:
                print(f"   {sf}")
            if excerpt:
                print(f"   {excerpt}")
            print()


def _eval_compose_ast(ast, lattices: dict) -> ComposedCartridge:
    """Evaluate a parsed composition AST against loaded lattices."""
    from resonance_lattice.composition import ComposedCartridge
    from resonance_lattice.composition.parser import (
        CartridgeRef,
        ContradictNode,
        DiffNode,
        MergeNode,
        ProjectNode,
        WeightedNode,
    )

    if isinstance(ast, CartridgeRef):
        name = Path(ast.path).stem
        if name not in lattices:
            _die(f"Cartridge '{name}' not loaded")
        return ComposedCartridge.merge({name: lattices[name]})

    elif isinstance(ast, WeightedNode):
        inner = _eval_compose_ast(ast.child, lattices)
        # Scale the composed field
        from resonance_lattice.field.dense import DenseField
        scaled = DenseField(bands=inner.composed_field.bands, dim=inner.composed_field.dim)
        scaled.F = ast.weight * inner.composed_field.F.copy()
        scaled._source_count = inner.composed_field.source_count
        return ComposedCartridge(inner._constituents, scaled, "weighted")

    elif isinstance(ast, MergeNode):
        left = _eval_compose_ast(ast.left, lattices)
        right = _eval_compose_ast(ast.right, lattices)
        from resonance_lattice.algebra import FieldAlgebra
        merged = FieldAlgebra.merge(left.composed_field, right.composed_field)
        all_constituents = {**left._constituents, **right._constituents}
        return ComposedCartridge(all_constituents, merged.field, "merge")

    elif isinstance(ast, DiffNode):
        newer = _eval_compose_ast(ast.newer, lattices)
        older = _eval_compose_ast(ast.older, lattices)
        from resonance_lattice.algebra import FieldAlgebra
        diff = FieldAlgebra.diff(newer.composed_field, older.composed_field)
        all_constituents = {**newer._constituents, **older._constituents}
        return ComposedCartridge(all_constituents, diff.delta_field, "diff")

    elif isinstance(ast, ContradictNode):
        left = _eval_compose_ast(ast.left, lattices)
        right = _eval_compose_ast(ast.right, lattices)
        from resonance_lattice.algebra import FieldAlgebra
        contra = FieldAlgebra.contradict(left.composed_field, right.composed_field)
        all_constituents = {**left._constituents, **right._constituents}
        return ComposedCartridge(all_constituents, contra.contradiction_field, "contradict")

    elif isinstance(ast, ProjectNode):
        source = _eval_compose_ast(ast.source, lattices)
        lens = _eval_compose_ast(ast.lens, lattices)
        from resonance_lattice.algebra import FieldAlgebra
        proj = FieldAlgebra.project(source.composed_field, lens.composed_field)
        all_constituents = {**source._constituents, **lens._constituents}
        return ComposedCartridge(all_constituents, proj.projected_field, "project")

    else:
        _die(f"Unknown AST node type: {type(ast).__name__}")


def cmd_mcp(args) -> None:
    """Start the MCP server for Claude Code integration."""
    import asyncio
    from pathlib import Path

    from resonance_lattice.mcp_server import load_cartridge, run_server
    from resonance_lattice.worker_main import _find_onnx_dir

    # Detect ONNX dir for faster inference (passed to deferred loader)
    onnx_dir = getattr(args, "onnx", None) or _find_onnx_dir(Path(args.cartridge))

    # Configure only — cartridge + encoder load is deferred to first tool call
    # so the MCP handshake completes instantly
    load_cartridge(
        args.cartridge,
        source_root=getattr(args, "source_root", None),
        onnx_dir=str(onnx_dir) if onnx_dir else None,
    )

    asyncio.run(run_server())


def cmd_export(args) -> None:
    """Export a knowledge model, with optional field-only mode."""
    from resonance_lattice.serialise import load_dense_field, save_field_only

    cartridge_path = args.cartridge
    output_path = args.output
    field_only = getattr(args, "field_only", False)

    if field_only:
        print("Exporting field-only knowledge model (no source text)...")
        data = load_dense_field(cartridge_path)
        save_field_only(output_path, data)
        print(f"Field-only cartridge saved to: {output_path}")
    else:
        # Full copy export
        import shutil
        shutil.copy2(cartridge_path, output_path)
        print(f"Cartridge copied to: {output_path}")


# ── skill commands ──────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# C4: `rlat ask` — reader synthesis flow
# ─────────────────────────────────────────────────────────────────────
#
# The launch-gate user-visible face of the three-layer architecture:
#
#   field     -> which region of the corpus is relevant?  (search)
#   retrieval -> what bytes do we return?                 (expand+hybrid)
#   reader    -> what answer does the user see?           (this code)
#
# Lives alongside the existing lens-routing `cmd_ask` behaviour to avoid
# breaking scripts that use `rlat ask` as a dispatcher. New behaviour is
# opt-in via `--reader context|llm`.


def _results_to_evidence(results: list[dict]) -> list[Any]:
    """Adapt search payload result dicts to reader.Evidence instances.

    Pulls the fields the Reader layer needs (source_file, char_offset,
    text, score, heading) and drops the rest. Char offset falls back
    to 0 for older knowledge models without A3 offsets — a degraded anchor,
    but the reader doesn't fail.
    """
    from resonance_lattice.reader import Evidence
    evidence: list[Any] = []
    for r in results:
        evidence.append(Evidence(
            source_file=r.get("source_file") or "",
            char_offset=int(r.get("char_offset") or 0),
            text=r.get("full_text") or r.get("summary") or "",
            score=float(r.get("score") or 0.0),
            heading=r.get("heading") or "",
        ))
    return evidence


def _select_reader_backend(backend: str, reader_model: str | None) -> str:
    """Resolve --reader-backend auto to a concrete backend name.

    Preference order:
      1. local — if OpenVINO/optimum-intel/transformers are installed
         AND the caller supplied --reader-model. Without a model id
         we don't know what to load, so we skip local.
      2. anthropic — if CLAUDE_API or ANTHROPIC_API_KEY is set.
      3. openai — if OPENAI_API_KEY is set.

    Returns the chosen backend or raises ValueError when none of the
    above are available. The error message lists every missing
    precondition so users can pick the easiest remediation.
    """
    if backend != "auto":
        return backend

    try:
        from resonance_lattice.reader.local import is_available as _local_available
        local_ok, _local_diag = _local_available()
    except Exception:
        local_ok = False

    have_anthropic = bool(
        os.environ.get("CLAUDE_API") or os.environ.get("ANTHROPIC_API_KEY")
    )
    have_openai = bool(os.environ.get("OPENAI_API_KEY"))

    # Local only makes sense when the caller specified a model — we
    # don't want to silently download a 3B-param model without explicit
    # opt-in.
    if local_ok and reader_model:
        return "local"
    if have_anthropic:
        return "anthropic"
    if have_openai:
        return "openai"

    raise ValueError(
        "No reader backend available. Options:\n"
        "  - Install openvino + optimum-intel + transformers AND pass "
        "--reader-model <hf-id> for local inference.\n"
        "  - Set CLAUDE_API for Anthropic (recommended).\n"
        "  - Set OPENAI_API_KEY for OpenAI."
    )


def _build_reader(args: argparse.Namespace) -> Any:
    """Construct a Reader from parsed CLI args. Raises with a clear
    diagnostic if the chosen backend isn't available.

    Kept separate from `cmd_ask` so tests can exercise the selection
    logic without building a lattice.
    """
    backend = _select_reader_backend(
        getattr(args, "reader_backend", "auto"),
        getattr(args, "reader_model", None),
    )
    model = getattr(args, "reader_model", None)

    if backend == "local":
        from resonance_lattice.reader.local import LocalReader
        if not model:
            raise ValueError(
                "--reader-backend local requires --reader-model "
                "(e.g. Qwen/Qwen2.5-3B-Instruct)"
            )
        return LocalReader(
            model,
            max_new_tokens=getattr(args, "max_tokens", 1024),
            temperature=getattr(args, "temperature", 0.3),
        )

    # API backends share the same constructor shape.
    from resonance_lattice.reader.api import APIReader
    if not model:
        # Sensible defaults so `--reader llm` works without every user
        # having to look up a model id.
        model = "claude-opus-4-7" if backend == "anthropic" else "gpt-4o-mini"
    return APIReader(
        provider=backend,
        model=model,
        max_tokens=getattr(args, "max_tokens", 1024),
        temperature=getattr(args, "temperature", 0.3),
    )


def _render_answer_text(answer: Any) -> str:
    """Format an Answer for terminal display.

    Layout:
      <answer text>

      Sources:
        [1] file.md (offset 1234)
        [2] other.md (offset 5678)

    Model / latency / evidence count go to stderr (via caller) so
    `rlat ask ... | grep` doesn't pick them up.
    """
    parts = [answer.text.rstrip()]
    if answer.citations:
        parts.append("")
        parts.append("Sources:")
        for i, c in enumerate(answer.citations, start=1):
            parts.append(
                f"  [{i}] {c.source_file or '(unknown)'} "
                f"(offset {c.char_offset})"
            )
    return "\n".join(parts)


def _apply_reader_config_defaults(args: argparse.Namespace) -> None:
    """C6: overlay `.rlat.toml` [reader] values onto `args`.

    Only overrides args that are at their argparse default — an
    explicit CLI flag always wins. This makes the precedence obvious:
    CLI > .rlat.toml > built-in defaults. Invoked exactly once at the
    top of the reader branch so tests that construct `args` manually
    can skip the config lookup by pre-setting values.

    Fail-soft: any exception reading the TOML leaves args untouched
    with a stderr warning. The reader flow should still work if the
    config file is malformed — the user can fix it later.
    """
    from resonance_lattice.config import load_reader_config

    try:
        cfg = load_reader_config()
    except Exception as e:
        sys.stderr.write(f"[rlat ask] ignoring .rlat.toml: {e}\n")
        return

    # The argparse defaults for the reader flags. When `args.<x>` is
    # still the default, and config has a non-default value, adopt
    # config. Keep this dict in sync with the argparse setup for
    # `ask`.
    argparse_defaults = {
        "reader": "off",
        "reader_backend": "auto",
        "reader_model": None,
        "max_tokens": 1024,
        "temperature": 0.3,
        "expand": "natural",
        "hybrid": "auto",
    }
    # Config uses `backend` / `model`; argparse uses `reader_backend` /
    # `reader_model`. The rest are same-named.
    config_map = {
        "reader": cfg.reader,
        "reader_backend": cfg.backend,
        "reader_model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "expand": cfg.expand,
        "hybrid": cfg.hybrid,
    }
    for attr, default in argparse_defaults.items():
        current = getattr(args, attr, default)
        if current == default:
            setattr(args, attr, config_map[attr])


def _run_ask_reader_mode(args: argparse.Namespace) -> None:
    """The reader-mode branch of cmd_ask. Factored out so tests can
    monkeypatch the pieces (lattice load, _build_reader) without
    needing a real knowledge model or API key."""
    import dataclasses

    from resonance_lattice.reader import build_context_pack

    lattice_path = Path(args.lattice)
    _require_file(lattice_path)

    # Load + search (reuses the standard cold path — no warm worker,
    # no composition. Reader flow is deliberately simpler than search's
    # feature matrix).
    lattice = _load_lattice_with_encoder(args, lattice_path)
    _rerank = getattr(args, "rerank", "auto")
    _rerank_val = "auto" if _rerank == "auto" else _rerank == "true"
    enriched = lattice.enriched_query(
        text=args.query,
        top_k=args.top_k,
        enable_cascade=False,
        enable_contradictions=False,
        enable_rerank=_rerank_val,
    )
    payload = enriched.to_dict()

    # Reuse B4/B5 expand + hybrid — same post-processing the search
    # path gets so --reader context/llm sees the same evidence shape
    # as --format context on rlat search.
    payload = _apply_retrieval_flags(args, payload, lattice_path, lattice)

    results = payload.get("results") or []
    evidence = _results_to_evidence(results)

    if args.reader == "context":
        context_pack = build_context_pack(args.query, evidence)
        if args.format == "json":
            out = {
                "query": args.query,
                "evidence": [dataclasses.asdict(e) for e in evidence],
                "context_pack": context_pack,
            }
            print(json.dumps(out, indent=2))
        else:
            print(context_pack)
        return

    # args.reader == "llm": synthesize.
    try:
        reader = _build_reader(args)
    except ValueError as e:
        _die(str(e))

    try:
        answer = reader.answer(args.query, evidence)
    finally:
        # Readers hold GB-scale resources (local) or HTTP sessions
        # (api). Release on the way out.
        try:
            reader.close()
        except Exception:
            pass

    if args.format == "json":
        print(json.dumps(dataclasses.asdict(answer), indent=2))
    else:
        print(_render_answer_text(answer))
        # Status line on stderr — pipable stdout stays clean.
        sys.stderr.write(
            f"\n[model: {answer.model}, "
            f"latency: {answer.latency_ms:.0f}ms, "
            f"evidence: {answer.evidence_used}, "
            f"citations: {len(answer.citations)}]\n"
        )


def cmd_ask(args: argparse.Namespace) -> None:
    """Smart query: auto-selects the best retrieval lens for the question."""
    # C6: apply .rlat.toml [reader] defaults before we decide which
    # branch to take — so a project-level `reader = "llm"` in config
    # can switch the default, not just the flags inside the branch.
    _apply_reader_config_defaults(args)

    # C4: when --reader is explicitly set (or set via config), bypass
    # the lens router and run the reader-synthesis flow. `off` keeps
    # the existing dispatcher behaviour so scripts using `rlat ask`
    # don't break.
    reader_mode = getattr(args, "reader", "off")
    if reader_mode in ("context", "llm"):
        _run_ask_reader_mode(args)
        return

    from resonance_lattice.lens_router import format_explain, route_query

    lattice_path = args.lattice
    query = args.query
    with_carts = getattr(args, "with_cartridges", [])
    background = getattr(args, "background", None)
    encoder = getattr(args, "encoder", None)
    num_cartridges = 1 + len(with_carts)

    choice = route_query(query, num_cartridges=num_cartridges, background_cartridge=background)

    if getattr(args, "explain", False):
        print(format_explain(choice, lattice_path, query))
        return

    print(f"[ask] Lens: {choice.lens}"
          + (f" --tune {choice.args['tune']}" if choice.args.get("tune") else "")
          + (f" --contrast {choice.args['contrast']}" if choice.args.get("contrast") else "")
          + f" | {choice.rationale}", file=sys.stderr)

    # Build CLI args and dispatch via subprocess to avoid fragile
    # argparse namespace construction (main() parser has ~50 flags per command).
    cmd_parts = [sys.executable, "-m", "resonance_lattice.cli"]

    if choice.lens == "search":
        cmd_parts.extend(["search", lattice_path, query,
                          "--top-k", str(args.top_k), "--format", args.format])
        if choice.args.get("tune"):
            cmd_parts.extend(["--tune", choice.args["tune"]])
        if choice.args.get("contrast"):
            cmd_parts.extend(["--contrast", choice.args["contrast"]])
        for wc in with_carts:
            cmd_parts.extend(["--with", wc])
        if encoder:
            cmd_parts.extend(["--encoder", encoder])

    elif choice.lens == "locate":
        cmd_parts.extend(["locate", lattice_path, query, "--format", args.format])
        if encoder:
            cmd_parts.extend(["--encoder", encoder])

    elif choice.lens == "profile":
        cmd_parts.extend(["profile", lattice_path, "--format", args.format])

    elif choice.lens == "negotiate" and num_cartridges >= 2:
        cmd_parts.extend(["negotiate", lattice_path, with_carts[0],
                          "--format", args.format])
        if query:
            cmd_parts.extend(["--query", query])

    elif choice.lens in ("compare", "negotiate"):
        cmd_parts.extend(["search", lattice_path, query,
                          "--top-k", str(args.top_k), "--format", args.format,
                          "--with-contradictions"])
        if encoder:
            cmd_parts.extend(["--encoder", encoder])

    elif choice.lens == "compose_search":
        carts = [lattice_path] + with_carts
        expression = " + ".join(carts)
        cmd_parts.extend(["compose", expression, query,
                          "--top-k", str(args.top_k), "--format", args.format])
        if encoder:
            cmd_parts.extend(["--encoder", encoder])

    else:
        cmd_parts.extend(["search", lattice_path, query,
                          "--top-k", str(args.top_k), "--format", args.format])

    result = subprocess.run(cmd_parts, capture_output=False, text=True)
    sys.exit(result.returncode)


def _find_skills_root() -> Path:
    """Find the .claude/skills/ directory from the current working directory."""
    cwd = Path.cwd()
    candidates = [
        cwd / ".claude" / "skills",
        cwd.parent / ".claude" / "skills",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return cwd / ".claude" / "skills"


def cmd_skill(args: argparse.Namespace) -> None:
    """Dispatch skill subcommands."""
    sub = getattr(args, "skill_command", None)
    if sub is None:
        print("Usage: rlat skill <build|sync|search|inject|route|info> [options]", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "build": cmd_skill_build,
        "sync": cmd_skill_sync,
        "search": cmd_skill_search,
        "info": cmd_skill_info,
        "inject": cmd_skill_inject,
        "route": cmd_skill_route,
        "profile": cmd_skill_profile,
        "freshness": cmd_skill_freshness,
        "gaps": cmd_skill_gaps,
        "compare": cmd_skill_compare,
    }
    if sub not in dispatch:
        print(f"Unknown skill command: {sub}", file=sys.stderr)
        print("Usage: rlat skill <command> [options]", file=sys.stderr)
        sys.exit(1)
    dispatch[sub](args)


def cmd_skill_build(args: argparse.Namespace) -> None:
    """Build a knowledge model from a skill's reference materials."""
    from resonance_lattice.config import LatticeConfig
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.skill import parse_skill_frontmatter

    skill_dir = Path(args.skill_dir)
    if not skill_dir.is_dir():
        _die(f"skill directory not found: {skill_dir}")

    skill = parse_skill_frontmatter(skill_dir)
    if skill is None:
        _die(f"no valid SKILL.md found in {skill_dir}")

    source_dirs = skill.resolve_source_paths()
    if not source_dirs:
        _die(f"no source directories found for skill '{skill.name}' "
             f"(set cartridge-sources in SKILL.md frontmatter, or add a references/ directory)")

    files = _collect_files(source_dirs)
    if not files:
        _die(f"no ingestible files found for skill '{skill.name}' in {[str(d) for d in source_dirs]}")

    ext_counts: dict[str, int] = {}
    for f in files:
        ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1
    ext_summary = ", ".join(f"{count} {ext}" for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]))
    print(f"Building cartridge for skill '{skill.name}' from {len(files)} files ({ext_summary})")

    output_path = skill.local_cartridge_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = LatticeConfig()
    lattice = Lattice(config=config)
    lattice.encoder = _load_encoder(args, config.bands, config.dim)

    onnx_dir = getattr(args, "onnx", None)
    if onnx_dir and lattice.encoder is not None:
        try:
            from resonance_lattice.encoder_onnx import attach_onnx_backbone
            attach_onnx_backbone(lattice.encoder, onnx_dir)
        except Exception as exc:
            _warn(f"ONNX backbone failed to load: {exc}")

    manifest = FileManifest()
    encoder_fp = _encoder_fingerprint(lattice.encoder) if lattice.encoder else ""

    import time as _time
    start = _time.time()
    added, updated, skipped = _ingest_files_incremental(
        lattice, files, manifest, encoder_fp, progress=False,
    )
    elapsed = _time.time() - start

    _save_manifest(lattice, manifest)
    lattice.save(output_path)

    count = lattice.source_count
    info = lattice.info()
    print(f"Built {output_path.name}: {count} chunks from {len(files)} files in {elapsed:.1f}s")
    print(f"  Field: {info['field_size_mb']:.1f} MB | Bands: {info['bands']} x {info['dim']}d")
    print(f"  Path: {output_path}")

    # Generate primer
    _generate_skill_primer(lattice, skill)


def cmd_skill_sync(args: argparse.Namespace) -> None:
    """Incrementally sync a skill's knowledge model with its reference materials."""
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.skill import parse_skill_frontmatter

    skill_dir = Path(args.skill_dir)
    if not skill_dir.is_dir():
        _die(f"skill directory not found: {skill_dir}")

    skill = parse_skill_frontmatter(skill_dir)
    if skill is None:
        _die(f"no valid SKILL.md found in {skill_dir}")

    cartridge_path = skill.local_cartridge_path()
    if not cartridge_path.exists():
        print("No existing cartridge — building from scratch...", file=sys.stderr)
        cmd_skill_build(args)
        return

    source_dirs = skill.resolve_source_paths()
    if not source_dirs:
        _die(f"no source directories for skill '{skill.name}'")

    files = _collect_files(source_dirs)
    if not files:
        _die(f"no ingestible files found for skill '{skill.name}'")

    print(f"Syncing cartridge for skill '{skill.name}'...")

    lattice = Lattice.load(cartridge_path, restore_encoder=True)
    override = _has_encoder_override(args)
    if override or lattice.encoder is None:
        lattice.encoder = _load_encoder(args, lattice.config.bands, lattice.config.dim, lattice=lattice)

    manifest = _load_manifest(lattice)
    encoder_fp = _encoder_fingerprint(lattice.encoder) if lattice.encoder else ""

    import time as _time
    start = _time.time()
    added, updated, skipped = _ingest_files_incremental(
        lattice, files, manifest, encoder_fp, progress=False,
    )
    elapsed = _time.time() - start

    if added + updated == 0:
        print(f"  No changes detected ({skipped} files up-to-date)")
        return

    _save_manifest(lattice, manifest)
    lattice.save(cartridge_path)
    print(f"  Synced: +{added} new, ~{updated} updated, {skipped} unchanged in {elapsed:.1f}s")

    _generate_skill_primer(lattice, skill)


def cmd_skill_search(args: argparse.Namespace) -> None:
    """Search a skill's knowledge models."""
    from resonance_lattice.skill import find_skill, parse_skill_frontmatter

    skill_name = args.skill_name
    skills_root = _find_skills_root()

    # Try as a directory path first
    skill_path = Path(skill_name)
    if skill_path.is_dir():
        skill = parse_skill_frontmatter(skill_path)
    else:
        skill = find_skill(skills_root, skill_name)

    if skill is None:
        _die(f"skill not found: {skill_name}\n"
             f"  Looked in: {skills_root}")

    # Collect all searchable cartridges
    cartridge_paths = skill.resolve_cartridge_paths(project_root=Path.cwd())

    # Add skill-local cartridge if it exists
    local = skill.local_cartridge_path()
    if local.exists() and local not in cartridge_paths:
        cartridge_paths.append(local)

    if not cartridge_paths:
        _die(f"skill '{skill.name}' has no cartridges.\n"
             f"  Run: rlat skill build {skill.skill_dir}")

    # Filter to existing files
    existing = [p for p in cartridge_paths if p.exists()]
    if not existing:
        _die(f"no cartridge files found for skill '{skill.name}'.\n"
             f"  Expected: {[str(p) for p in cartridge_paths]}")

    query = args.query
    top_k = getattr(args, "top_k", 10)

    if len(existing) == 1:
        # Single cartridge: delegate directly to enriched search
        lattice = _load_lattice_with_encoder(args, existing[0])

        import time as _time
        t0 = _time.perf_counter()
        result = lattice.enriched_query(
            text=query,
            top_k=top_k,
            enable_cascade=False,
            enable_contradictions=False,
        )
        load_ms = (_time.perf_counter() - t0) * 1000

        fmt = getattr(args, "format", "text")
        if fmt == "text":
            print(f"Skill: {skill.name}")
        _output_search(args, result.to_dict(), load_ms=load_ms, warm=False)
    else:
        # Multiple cartridges: compose and search
        from resonance_lattice.composition.composed import ComposedCartridge
        from resonance_lattice.lattice import Lattice

        lattices = {}
        for p in existing:
            lat = Lattice.load(p, restore_encoder=True)
            lattices[p.stem] = lat

        composed = ComposedCartridge.merge(lattices)

        # ComposedCartridge.search takes query_text (str), encodes internally
        results = composed.search(query, top_k=top_k)

        fmt = getattr(args, "format", "text")
        if fmt == "json":
            print(json.dumps([
                {
                    "source_id": r.source_id,
                    "score": round(r.score, 4),
                    "knowledge model": getattr(r, "knowledge model", "?"),
                    "summary": (r.content.summary or "")[:200] if r.content else "",
                    "source_file": (r.content.metadata or {}).get("source_file", "") if r.content else "",
                }
                for r in results
            ], indent=2))
        else:
            print(f"Skill: {skill.name} ({len(existing)} cartridges)")
            print(f"Query: {query}")
            print()
            for i, r in enumerate(results, 1):
                score_str = f"{r.score:.3f}"
                cartridge = getattr(r, "knowledge model", "?")
                summary = ""
                if r.content:
                    summary = r.content.summary or (r.content.full_text or "")[:200]
                source_file = ""
                if r.content and r.content.metadata:
                    source_file = r.content.metadata.get("source_file", "")
                print(f"  {i}. [{score_str}] ({cartridge}) {source_file}")
                if summary:
                    print(f"     {summary[:120]}")
                print()


def cmd_skill_info(args: argparse.Namespace) -> None:
    """Show skill knowledge model configuration and status."""
    from resonance_lattice.skill import discover_skills, find_skill, parse_skill_frontmatter

    skill_name = getattr(args, "skill_name", None)
    skills_root = _find_skills_root()

    if skill_name:
        skill_path = Path(skill_name)
        if skill_path.is_dir():
            skill = parse_skill_frontmatter(skill_path)
        else:
            skill = find_skill(skills_root, skill_name)
        if skill is None:
            _die(f"skill not found: {skill_name}")
        _print_skill_info(skill)
    else:
        # List all skills
        skills = discover_skills(skills_root)
        if not skills:
            print(f"No skills found in {skills_root}")
            return
        print(f"Skills in {skills_root}:")
        print()
        for s in skills:
            cart_status = ""
            local = s.local_cartridge_path()
            if s.has_cartridges or local.exists():
                ext_count = len(s.cartridges)
                if local.exists():
                    size_mb = local.stat().st_size / (1024 * 1024)
                    cart_status = f"[local {size_mb:.0f}MB] "
                if ext_count:
                    cart_status += f"[{ext_count} external] "
                if s.cartridge_queries:
                    cart_status += f"[{len(s.cartridge_queries)} queries] "
            else:
                cart_status = "[no knowledge model]"
            print(f"  {s.name:30s} {cart_status}")
        print()
        print(f"  {len(skills)} skills found")


def _print_skill_info(skill) -> None:
    """Print detailed skill knowledge model info."""
    print(f"Skill: {skill.name}")
    print(f"  Directory: {skill.skill_dir}")
    print(f"  Description: {skill.description[:100]}{'...' if len(skill.description) > 100 else ''}")
    print()

    local = skill.local_cartridge_path()
    if not skill.has_cartridges and not local.exists():
        print("  No knowledge model integration configured.")
        print("  To enable: add 'cartridges:' or 'cartridge-sources:' to SKILL.md frontmatter")
        return

    print("  Knowledge Model configuration:")
    if skill.cartridges:
        print(f"    External cartridges: {skill.cartridges}")
    if skill.cartridge_sources:
        print(f"    Sources: {skill.cartridge_sources}")
    print(f"    Mode: {skill.cartridge_mode}")
    print(f"    Budget: {skill.cartridge_budget} lines")
    print(f"    Rebuild: {skill.cartridge_rebuild}")
    print(f"    Derive: {skill.cartridge_derive} (max {skill.cartridge_derive_count} queries)")

    if skill.cartridge_queries:
        print(f"    Foundational queries ({len(skill.cartridge_queries)}):")
        for q in skill.cartridge_queries:
            print(f"      - {q}")

    local = skill.local_cartridge_path()
    if local.exists():
        size_mb = local.stat().st_size / (1024 * 1024)
        print(f"    Local cartridge: {local} ({size_mb:.1f} MB)")
    else:
        sources = skill.resolve_source_paths()
        if sources:
            print(f"    Local cartridge: not built (run: rlat skill build {skill.skill_dir})")

    primer = skill.local_primer_path()
    if primer.exists():
        print(f"    Primer: {primer}")


def cmd_skill_inject(args: argparse.Namespace) -> None:
    """Four-tier adaptive context injection for a skill."""
    from resonance_lattice.skill import find_skill, parse_skill_frontmatter
    from resonance_lattice.skill_projector import SkillProjector
    from resonance_lattice.skill_runtime import SkillRuntime

    skill_name = args.skill_name
    query = args.query
    skills_root = _find_skills_root()

    # Resolve skill
    skill_path = Path(skill_name)
    if skill_path.is_dir():
        skill = parse_skill_frontmatter(skill_path)
    else:
        skill = find_skill(skills_root, skill_name)

    if skill is None:
        _die(
            f"skill not found: {skill_name}\n"
            f"  Looked in: {skills_root}\n"
            f"  Run: rlat skill info  to list available skills"
        )

    # Build runtime and projector. --source-root threads through to
    # Lattice.load so external-mode cartridges can resolve passage content
    # at T2/T3 retrieval time (without it: empty body, 0 content tokens).
    print(f"Injecting context for skill '{skill.name}'...", file=sys.stderr)
    source_root = getattr(args, "source_root", None)
    rt = SkillRuntime(skills_root, Path.cwd(), source_root=source_root)
    projector = SkillProjector(rt)

    # Parse derived queries if provided
    derived = None
    derived_raw = getattr(args, "derived", None)
    if derived_raw:
        derived = [q.strip() for q in derived_raw.split(";") if q.strip()]

    injection = projector.project(skill, query, derived_queries=derived)

    fmt = getattr(args, "format", "text")

    if fmt == "json":
        import json as _json
        print(_json.dumps({
            "mode": injection.mode,
            "gated": injection.gated,
            "total_tokens": injection.total_tokens,
            "tier_tokens": injection.tier_tokens,
            "queries_used": injection.queries_used,
            "cartridge_hits": injection.cartridge_hits,
            "coverage_confidence": injection.coverage_confidence,
            "header_length": len(injection.header),
            "body_length": len(injection.body),
            "freshness": [
                {"name": f.name, "age_hours": round(f.age_hours, 1), "status": f.status}
                for f in injection.freshness
            ],
        }, indent=2))
    elif fmt == "context":
        # Output just the injectable body (for piping into prompts)
        if injection.body:
            print(injection.body)
    else:
        # text format: full diagnostic output
        use_c = _use_color()
        B = "\033[1m" if use_c else ""
        D = "\033[2m" if use_c else ""
        R = "\033[0m" if use_c else ""
        G = "\033[32m" if use_c else ""
        Y = "\033[33m" if use_c else ""

        print(f"{B}Skill Injection: {skill.name}{R}")
        print(f"  Query: {query}")
        print(f"  Mode: {injection.mode}")
        print(f"  Gated: {'yes (dynamic context suppressed)' if injection.gated else 'no'}")
        print(f"  Coverage: {injection.coverage_confidence:.0%}")
        if injection.freshness:
            for f in injection.freshness:
                color = G if f.status == "fresh" else Y
                print(f"  {color}Freshness: {f.label()}{R}")
        print()

        print(f"{B}Token Budget:{R}")
        for tier, tokens in injection.tier_tokens.items():
            label = {"t1": "Tier 1 (static)", "t2": "Tier 2 (foundational)",
                     "t3": "Tier 3 (user query)", "t4": "Tier 4 (derived)"}.get(tier, tier)
            bar = _safe_bar(tokens / max(injection.total_tokens, 1), width=20)
            print(f"  {label:25s} {tokens:5d} tokens  {bar}")
        print(f"  {'Total':25s} {injection.total_tokens:5d} tokens")
        print()

        if injection.queries_used:
            print(f"{B}Queries Used ({len(injection.queries_used)}):{R}")
            for q in injection.queries_used:
                print(f"  - {q}")
            print()

        if injection.cartridge_hits:
            print(f"{B}Cartridge Hits:{R}")
            for cart, count in sorted(injection.cartridge_hits.items(), key=lambda x: -x[1]):
                print(f"  {cart}: {count} passages")
            print()

        if injection.body:
            print(f"{D}--- Dynamic Body ({_estimate_tokens_cli(injection.body)} tokens) ---{R}")
            # Print first 2000 chars to avoid flooding terminal
            preview = injection.body[:2000]
            if len(injection.body) > 2000:
                preview += f"\n... ({len(injection.body) - 2000} chars truncated)"
            print(preview)


def _estimate_tokens_cli(text: str) -> int:
    return max(1, len(text) // 4)


def cmd_skill_route(args: argparse.Namespace) -> None:
    """Rank all knowledge model-backed skills by relevance to a query."""
    from resonance_lattice.skill_runtime import SkillRuntime

    query = args.query
    top_n = getattr(args, "top_n", 5)
    skills_root = _find_skills_root()

    print(f"Routing: \"{query}\"", file=sys.stderr)
    rt = SkillRuntime(skills_root, Path.cwd())

    matches = rt.route(query, top_n=top_n)

    if not matches:
        print("No knowledge model-backed skills found.")
        print(f"  Skills root: {skills_root}")
        print("  Run 'rlat skill build' on skills with references/ directories.")
        return

    fmt = getattr(args, "format", "text")

    if fmt == "json":
        import json as _json
        print(_json.dumps([
            {"name": m.name, "energy": round(m.energy, 1),
             "coverage": m.coverage, "mode": m.mode}
            for m in matches
        ], indent=2))
    else:
        use_c = _use_color()
        B = "\033[1m" if use_c else ""
        R = "\033[0m" if use_c else ""
        G = "\033[32m" if use_c else ""
        Y = "\033[33m" if use_c else ""
        D = "\033[2m" if use_c else ""

        print(f"\n{B}Routing:{R} \"{query}\"\n")
        for i, m in enumerate(matches, 1):
            color = G if m.coverage == "high" else (Y if m.coverage == "medium" else D)
            print(f"  {i}. {color}{m.name:30s}{R}  energy={m.energy:>7.1f}  "
                  f"coverage={m.coverage:6s}  mode={m.mode}")

        print(f"\n  {len(matches)} skill(s) with relevant knowledge. Best: {matches[0].name}")


def cmd_skill_profile(args: argparse.Namespace) -> None:
    """Semantic profile of a skill's knowledge model."""
    from resonance_lattice.skill import find_skill, parse_skill_frontmatter

    skill_name = args.skill_name
    skills_root = _find_skills_root()

    skill_path = Path(skill_name)
    if skill_path.is_dir():
        skill = parse_skill_frontmatter(skill_path)
    else:
        skill = find_skill(skills_root, skill_name)
    if skill is None:
        _die(f"skill not found: {skill_name}")

    cartridge = skill.local_cartridge_path()
    if not cartridge.exists():
        # Check external cartridges
        paths = skill.resolve_cartridge_paths(project_root=Path.cwd())
        existing = [p for p in paths if p.exists()]
        if not existing:
            _die(f"no cartridge found for skill '{skill.name}'.\n"
                 f"  Run: rlat skill build {skill.skill_dir}")
        cartridge = existing[0]

    # Delegate to existing profile logic
    args.lattice = str(cartridge)
    args.format = getattr(args, "format", "text")
    print(f"Skill: {skill.name}")
    cmd_profile(args)


def cmd_skill_freshness(args: argparse.Namespace) -> None:
    """Check freshness of skill knowledge models."""
    from resonance_lattice.skill import discover_skills

    skills_root = _find_skills_root()
    skill_name = getattr(args, "skill_name", None)

    if skill_name:
        from resonance_lattice.skill import find_skill, parse_skill_frontmatter
        skill_path = Path(skill_name)
        if skill_path.is_dir():
            skill = parse_skill_frontmatter(skill_path)
        else:
            skill = find_skill(skills_root, skill_name)
        if skill is None:
            _die(f"skill not found: {skill_name}")
        skills = [skill]
    else:
        skills = discover_skills(skills_root)

    from datetime import datetime

    fmt = getattr(args, "format", "text")
    results = []

    for skill in skills:
        cartridge = skill.local_cartridge_path()
        if not cartridge.exists():
            results.append({
                "name": skill.name, "status": "no knowledge model",
                "age_hours": None, "stale": None,
                "files_changed": None, "recommendation": "build first",
            })
            continue

        # Get cartridge mtime as built_at proxy
        mtime = cartridge.stat().st_mtime
        built = datetime.fromtimestamp(mtime, tz=UTC)
        age_hours = (datetime.now(UTC) - built).total_seconds() / 3600

        # Check source file changes
        source_dirs = skill.resolve_source_paths()
        files_changed = 0
        for src_dir in source_dirs:
            if not src_dir.is_dir():
                continue
            for f in src_dir.rglob("*"):
                if f.is_file() and f.stat().st_mtime > mtime:
                    files_changed += 1

        if age_hours < 24 and files_changed == 0:
            status = "fresh"
            stale = False
        elif age_hours < 72 and files_changed < 10:
            status = "consider rebuilding" if files_changed else "fresh"
            stale = False
        else:
            status = "stale"
            stale = True

        rec = status
        if stale:
            rec = f"stale ({age_hours:.0f}h old, {files_changed} files changed)"

        results.append({
            "name": skill.name, "status": status,
            "age_hours": round(age_hours, 1), "stale": stale,
            "files_changed": files_changed, "recommendation": rec,
        })

    if fmt == "json":
        print(json.dumps(results, indent=2))
    else:
        use_c = _use_color()
        G = "\033[32m" if use_c else ""
        Y = "\033[33m" if use_c else ""
        R_c = "\033[31m" if use_c else ""
        D = "\033[2m" if use_c else ""
        R = "\033[0m" if use_c else ""

        for r in results:
            if r["status"] == "fresh":
                color = G
                icon = "OK"
            elif r["status"] == "no knowledge model":
                color = D
                icon = "--"
            elif r["stale"]:
                color = R_c
                icon = "!!"
            else:
                color = Y
                icon = "~~"

            age_str = f"{r['age_hours']:.0f}h" if r["age_hours"] is not None else "n/a"
            changed_str = str(r["files_changed"]) if r["files_changed"] is not None else "-"
            print(f"  {color}[{icon}]{R} {r['name']:30s}  age={age_str:>5s}  "
                  f"changed={changed_str:>3s}  {r['recommendation']}")


def cmd_skill_gaps(args: argparse.Namespace) -> None:
    """Detect knowledge gaps in a skill's knowledge model."""
    from resonance_lattice.skill import find_skill, parse_skill_frontmatter

    skill_name = args.skill_name
    skills_root = _find_skills_root()

    skill_path = Path(skill_name)
    if skill_path.is_dir():
        skill = parse_skill_frontmatter(skill_path)
    else:
        skill = find_skill(skills_root, skill_name)
    if skill is None:
        _die(f"skill not found: {skill_name}")

    cartridge = skill.local_cartridge_path()
    if not cartridge.exists():
        paths = skill.resolve_cartridge_paths(project_root=Path.cwd())
        existing = [p for p in paths if p.exists()]
        if not existing:
            _die(f"no cartridge for skill '{skill.name}'")
        cartridge = existing[0]

    # Delegate to probe gaps recipe
    args.lattice = str(cartridge)
    args.recipe = "gaps"
    args.query = ""
    args.format = getattr(args, "format", "text")
    args.encoder = getattr(args, "encoder", None)

    print(f"Skill: {skill.name}")
    cmd_probe(args)


def cmd_skill_compare(args: argparse.Namespace) -> None:
    """Compare two skills' knowledge models for overlap."""
    from resonance_lattice.skill import find_skill, parse_skill_frontmatter

    skills_root = _find_skills_root()

    def _resolve_skill_cartridge(name: str) -> Path:
        skill_path = Path(name)
        if skill_path.is_dir():
            skill = parse_skill_frontmatter(skill_path)
        else:
            skill = find_skill(skills_root, name)
        if skill is None:
            _die(f"skill not found: {name}")
        cartridge = skill.local_cartridge_path()
        if not cartridge.exists():
            paths = skill.resolve_cartridge_paths(project_root=Path.cwd())
            existing = [p for p in paths if p.exists()]
            if not existing:
                _die(f"no cartridge for skill '{skill.name}'")
            cartridge = existing[0]
        return cartridge

    cart_a = _resolve_skill_cartridge(args.skill_a)
    cart_b = _resolve_skill_cartridge(args.skill_b)

    # Delegate to existing compare
    args.lattice_a = str(cart_a)
    args.lattice_b = str(cart_b)
    args.format = getattr(args, "format", "text")

    print(f"Comparing skills: {args.skill_a} vs {args.skill_b}")
    cmd_compare(args)


def _generate_skill_primer(lattice, skill) -> None:
    """Generate primer.md alongside the skill's knowledge model."""
    if lattice.encoder is None:
        return

    from resonance_lattice.materialiser import _estimate_tokens, _truncate_to_tokens

    queries = [
        "What are the key concepts and patterns?",
        "What operations and workflows does this cover?",
        "What are the important conventions and best practices?",
    ]

    seen: dict[str, tuple[float, object]] = {}
    for query in queries:
        result = lattice.resonate_text(query=query, top_k=10)
        for r in result.results:
            if not r.content or r.source_id.startswith("__"):
                continue
            text = r.content.full_text or r.content.summary or ""
            if not text:
                continue
            existing = seen.get(r.source_id)
            if existing is None or r.score > existing[0]:
                seen[r.source_id] = (r.score, r)

    ranked = [r for _, r in sorted(seen.values(), key=lambda x: -x[0])]

    lines = [
        f"# Skill Primer: {skill.name}",
        "",
        f"<!-- Auto-generated by rlat skill build | {lattice.source_count} sources -->",
        "",
    ]

    budget_tokens = 1500
    used = 0
    for r in ranked:
        text = r.content.full_text or r.content.summary or ""
        tokens = _estimate_tokens(text)
        if used + tokens > budget_tokens:
            text = _truncate_to_tokens(text, budget_tokens - used)
            if not text.strip():
                break
        source_file = (r.content.metadata or {}).get("source_file", r.source_id)
        lines.append(f"- **{source_file}**: {text.strip()}")
        lines.append("")
        used += _estimate_tokens(text)
        if used >= budget_tokens:
            break

    primer_path = skill.local_primer_path()
    primer_path.parent.mkdir(parents=True, exist_ok=True)
    primer_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Primer: {primer_path.name} ({len(ranked)} passages, ~{used} tokens)")


def cmd_negotiate(args: argparse.Namespace) -> None:
    """Analyze the relationship between two knowledge models."""
    import numpy as np

    from resonance_lattice.algebra import FieldAlgebra
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lattice import Lattice

    path_a, path_b = Path(args.cartridge_a), Path(args.cartridge_b)
    if not path_a.exists():
        _die(f"file not found: {args.cartridge_a}")
    if not path_b.exists():
        _die(f"file not found: {args.cartridge_b}")

    lattice_a = Lattice.load(str(path_a))
    lattice_b = Lattice.load(str(path_b))

    if not isinstance(lattice_a.field, DenseField) or not isinstance(lattice_b.field, DenseField):
        _die("negotiate requires DenseField knowledge models")
    if lattice_a.field.bands != lattice_b.field.bands or lattice_a.field.dim != lattice_b.field.dim:
        _die("knowledge models must have matching dimensions")

    name_a, name_b = path_a.stem, path_b.stem

    # Intersection
    intersection = FieldAlgebra.intersect(lattice_a.field, lattice_b.field)
    print(f"Knowledge Negotiation: {name_a} vs {name_b}")
    print(f"  Shared ground: {intersection.overlap_fraction:.0%} overlap")
    print()

    # Novelty
    novelty_a, novelty_b = [], []
    if lattice_a.registry:
        for sid, entry in list(lattice_a.registry._source_index.items())[:100]:
            n = FieldAlgebra.novelty(lattice_b.field, entry.phase_vectors)
            novelty_a.append(n.score)
    if lattice_b.registry:
        for sid, entry in list(lattice_b.registry._source_index.items())[:100]:
            n = FieldAlgebra.novelty(lattice_a.field, entry.phase_vectors)
            novelty_b.append(n.score)

    avg_a = float(np.mean(novelty_a)) if novelty_a else 0
    avg_b = float(np.mean(novelty_b)) if novelty_b else 0
    print(f"  Unique to {name_a}: {avg_a:.0%} novelty")
    print(f"  Unique to {name_b}: {avg_b:.0%} novelty")
    print()

    # Contradictions
    contradiction = FieldAlgebra.contradict(lattice_a.field, lattice_b.field)
    if contradiction.contradiction_ratio < 0.01:
        print("  Disagreements: none detected")
    else:
        ratio_label = "high" if contradiction.contradiction_ratio > 0.15 else "moderate" if contradiction.contradiction_ratio > 0.05 else "low"
        print(f"  Disagreements: {ratio_label} ({contradiction.contradiction_ratio:.1%} contradiction ratio)")
    print()

    # Recommendation
    print("  Recommendation:")
    if contradiction.contradiction_ratio > 0.05:
        print("    Resolve disagreements before merging.")
    if avg_b > avg_a + 0.1:
        print(f"    {name_b} has significantly more unique knowledge.")
    elif avg_a > avg_b + 0.1:
        print(f"    {name_a} has significantly more unique knowledge.")
    if intersection.overlap_fraction > 0.7:
        print("    High overlap — merging adds limited new knowledge.")
    elif intersection.overlap_fraction < 0.3:
        print("    Low overlap — merge (--with) for broad coverage.")


def cmd_memory(args: argparse.Namespace) -> None:
    """Dispatch memory subcommands."""
    sub = getattr(args, "memory_command", None)
    if sub is None:
        print("Usage: rlat memory {search,add,forget,stats,init,write,recall,consolidate,gc,profile,primer}")
        sys.exit(1)
    dispatch = {
        "search": cmd_memory_search,
        "add": cmd_memory_add,
        "forget": cmd_memory_forget,
        "stats": cmd_memory_stats,
        "init": cmd_memory_init,
        "write": cmd_memory_write,
        "recall": cmd_memory_recall,
        "consolidate": cmd_memory_consolidate,
        "gc": cmd_memory_gc,
        "profile": cmd_memory_profile,
        "primer": cmd_memory_primer,
    }
    handler = dispatch.get(sub)
    if handler is None:
        print(f"Unknown memory subcommand: {sub}")
        sys.exit(1)
    handler(args)


def _find_memory_path() -> Path:
    """Find the standard memory knowledge model path."""
    candidates = [Path(".rlat/memory.rlat"), Path("memory.rlat")]
    for c in candidates:
        if c.exists():
            return c
    return Path(".rlat/memory.rlat")  # default


def cmd_memory_search(args: argparse.Namespace) -> None:
    """Search memory for relevant prior context."""
    from resonance_lattice.lattice import Lattice
    mem_path = _find_memory_path()
    if not mem_path.exists():
        _die(f"No memory cartridge found at {mem_path}. Use 'rlat memory add' to create one.")

    lattice = Lattice.load(str(mem_path))
    result = lattice.enriched_query(args.query, top_k=args.top_k)

    print(f"Memory: {lattice.source_count} memories")
    print()
    for i, r in enumerate(result.results[:args.top_k], 1):
        text = ""
        session = ""
        if r.content:
            text = r.content.summary or (r.content.full_text or "")[:200]
            meta = r.content.metadata or {}
            session = meta.get("session_id", "")
        tag = f" [session: {session}]" if session else ""
        print(f"  {i}. [{r.score:.3f}]{tag} {text}")


def cmd_memory_add(args: argparse.Namespace) -> None:
    """Add content to the memory knowledge model."""
    import tempfile
    from datetime import datetime

    from resonance_lattice.lattice import Lattice

    mem_path = _find_memory_path()
    mem_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).isoformat()
    session_id = getattr(args, "session", "")
    header = f"---\nsession_id: {session_id}\ntimestamp: {timestamp}\n---\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(header + args.content)
        tmp = f.name

    try:
        if mem_path.exists():
            lattice = Lattice.load(str(mem_path))
            lattice.add_files([tmp])
            lattice.save(str(mem_path))
        else:
            from resonance_lattice.encoder import ResonanceEncoder
            encoder = ResonanceEncoder.from_preset("bge-large-en-v1.5")
            lattice = Lattice.build(paths=[tmp], encoder=encoder)
            lattice.save(str(mem_path))
        print(f"Saved to memory ({lattice.source_count} memories).")
    finally:
        Path(tmp).unlink(missing_ok=True)


def cmd_memory_forget(args: argparse.Namespace) -> None:
    """Provably remove a topic or session from memory."""
    from resonance_lattice.algebra import FieldAlgebra
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lattice import Lattice

    mem_path = _find_memory_path()
    if not mem_path.exists():
        _die(f"No memory cartridge found at {mem_path}")

    topic = getattr(args, "topic", None)
    session_id = getattr(args, "session", None)
    if not topic and not session_id:
        _die("Provide --topic or --session to forget")

    lattice = Lattice.load(str(mem_path))
    if not isinstance(lattice.field, DenseField):
        _die("Memory requires DenseField")

    sources_to_forget = []
    if session_id and lattice.store:
        for sid in list(lattice.registry._source_index.keys()):
            content = lattice.store.retrieve(sid)
            if content and content.metadata and content.metadata.get("session_id") == session_id:
                sources_to_forget.append(sid)
    elif topic and lattice.encoder:
        result = lattice.enriched_query(topic, top_k=10)
        for r in result.results:
            if r.score > 0.3:
                sources_to_forget.append(r.source_id)

    if not sources_to_forget:
        print("No matching memories found.")
        return

    total_residual = 0.0
    for sid in sources_to_forget:
        entry = lattice.registry._source_index.get(sid)
        if entry is not None:
            cert = FieldAlgebra.forget(lattice.field, [entry.phase_vectors])
            total_residual = max(total_residual, cert.residual_ratio)
            lattice.registry.unregister(sid)
            if lattice.store:
                lattice.store.remove(sid)

    lattice.save(str(mem_path))

    label = f"topic \"{topic}\"" if topic else f"session \"{session_id}\""
    print(f"Removed {len(sources_to_forget)} memories about {label}.")
    print(f"Removal certificate: residual < {total_residual:.1e} (algebraically exact).")
    print(f"Memory: {lattice.source_count} memories remaining.")


def cmd_memory_stats(args: argparse.Namespace) -> None:
    """Show memory knowledge model statistics."""
    from resonance_lattice.lattice import Lattice
    mem_path = _find_memory_path()
    if not mem_path.exists():
        print(f"No memory cartridge at {mem_path}")
        return

    lattice = Lattice.load(str(mem_path))
    size_mb = mem_path.stat().st_size / (1024 * 1024)
    print(f"Memory: {mem_path}")
    print(f"  Sources: {lattice.source_count}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Bands: {lattice.field.bands}")
    print(f"  Dim: {lattice.field.dim}")


def cmd_memory_init(args: argparse.Namespace) -> None:
    """Initialize a layered memory root directory."""
    from resonance_lattice.layered_memory import LayeredMemory

    root = Path(args.memory_root)
    mem = LayeredMemory.init(root)
    mem.save()
    print(f"Initialized layered memory at {root}/")
    for name in ("working", "episodic", "semantic"):
        print(f"  {name}.rlat")


def cmd_memory_write(args: argparse.Namespace) -> None:
    """Write content to a layered memory tier."""
    import tempfile
    from datetime import datetime

    from resonance_lattice.layered_memory import LayeredMemory

    root = Path(args.memory_root)
    if not root.exists():
        _die(f"Memory root not found: {root}. Run 'rlat memory init {root}' first.")

    mem = LayeredMemory.open(root)

    # Ensure each tier has an encoder attached. Freshly-initialized memory
    # roots load with no encoder, which makes any ``superpose_text`` call
    # fail at runtime — load one default encoder (respects --encoder flag)
    # and share it across tiers.
    if any(t.encoder is None for t in mem.tiers.values()):
        ref = next(iter(mem.tiers.values()))
        shared_encoder = _load_encoder(
            args, bands=ref.config.bands, dim=ref.config.dim, lattice=None,
        )
        for lattice in mem.tiers.values():
            if lattice.encoder is None:
                lattice.encoder = shared_encoder

    # Attach OpenVINO backbone for fast embedding (Intel Arc iGPU = ~40x CPU).
    openvino_dir = getattr(args, "openvino", None)
    openvino_device = getattr(args, "openvino_device", None)
    if openvino_dir:
        try:
            from resonance_lattice.encoder_openvino import attach_openvino_backbone
            for lattice in mem.tiers.values():
                if lattice.encoder is not None:
                    attach_openvino_backbone(
                        lattice.encoder, openvino_dir, device=openvino_device,
                    )
        except ImportError:
            pass  # openvino not installed — silent fallback to PyTorch

    # Attach ONNX backbone if available (optional speedup)
    onnx_dir = getattr(args, "onnx", None)
    if onnx_dir:
        try:
            from resonance_lattice.encoder_onnx import attach_onnx_backbone
            for lattice in mem.tiers.values():
                if lattice.encoder is not None:
                    attach_onnx_backbone(lattice.encoder, onnx_dir)
        except ImportError:
            pass  # onnxruntime not installed — silent fallback to PyTorch

    tier = getattr(args, "tier", "working")
    session_id = getattr(args, "session", "")
    input_file = getattr(args, "input_file", None)
    input_format = getattr(args, "input_format", "auto")

    # Path 1: JSONL transcript file (Claude Code session export)
    if input_file and (input_format == "claude_transcript" or
                       (input_format == "auto" and input_file.endswith(".jsonl"))):
        from resonance_lattice.chunker import chunk_claude_transcript
        chunks = chunk_claude_transcript(
            input_file,
            session_id=session_id,
        )
        if not chunks:
            print("No conversation turns found in transcript.")
            return

        lattice = mem.tiers[tier]
        for chunk in chunks:
            lattice.superpose_text(
                text=chunk.text,
                salience=1.0,
                source_id=f"{session_id or Path(input_file).stem}_{chunk.metadata.get('turn_index', 0):04d}",
                metadata=chunk.metadata,
            )
        mem.save()
        print(f"Ingested {len(chunks)} chunks from transcript → {tier} tier ({lattice.source_count} sources).")
        return

    # Path 2: Plain text content or markdown file
    content_text = getattr(args, "content", None) or ""
    if input_file:
        content_text = Path(input_file).read_text(encoding="utf-8")

    if not content_text:
        _die("Provide content text or --input-file")

    timestamp = datetime.now(UTC).isoformat()
    header = f"---\nsession_id: {session_id}\ntimestamp: {timestamp}\ntier: {tier}\n---\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(header + content_text)
        tmp = f.name

    try:
        lattice = mem.tiers[tier]
        lattice.add_files([tmp])
        mem.save()
        print(f"Written to {tier} tier ({lattice.source_count} sources).")
    finally:
        Path(tmp).unlink(missing_ok=True)


def cmd_memory_recall(args: argparse.Namespace) -> None:
    """Recall across all memory tiers with weighted fusion."""
    from resonance_lattice.layered_memory import LayeredMemory

    root = Path(args.memory_root)
    if not root.exists():
        _die(f"Memory root not found: {root}")

    mem = LayeredMemory.open(root)

    onnx_dir = getattr(args, "onnx", None)
    if onnx_dir:
        try:
            from resonance_lattice.encoder_onnx import attach_onnx_backbone
            for lattice in mem.tiers.values():
                if lattice.encoder is not None:
                    attach_onnx_backbone(lattice.encoder, onnx_dir)
        except ImportError:
            pass

    tier_weights = None
    weights_arg = getattr(args, "tier_weights", None)
    if weights_arg:
        tier_weights = {}
        for pair in weights_arg.split(","):
            name, val = pair.split("=")
            tier_weights[name.strip()] = float(val.strip())

    tiers_arg = getattr(args, "tiers", None)
    tiers_filter = tiers_arg.split(",") if tiers_arg else None

    result = mem.recall_text(
        args.query,
        tier_weights=tier_weights,
        top_k=args.top_k,
        tiers=tiers_filter,
    )

    info = mem.info()
    print(f"Memory: {info['total_sources']} sources across {len(mem.tiers)} tiers")
    print()
    for i, r in enumerate(result.results[:args.top_k], 1):
        text = ""
        tier_label = ""
        if r.content:
            text = r.content.summary or (r.content.full_text or "")[:200]
            meta = r.content.metadata or {}
            tier_label = meta.get("tier", "")
        tag = f" [{tier_label}]" if tier_label else ""
        print(f"  {i}. [{r.score:.3f}]{tag} {text}")


def cmd_memory_consolidate(args: argparse.Namespace) -> None:
    """Consolidate memory tiers (working->episodic, episodic->semantic)."""
    from resonance_lattice.consolidation import (
        consolidate_working_to_episodic,
        promote_to_semantic,
    )
    from resonance_lattice.layered_memory import LayeredMemory

    root = Path(args.memory_root)
    if not root.exists():
        _die(f"Memory root not found: {root}")

    mem = LayeredMemory.open(root)
    session_id = getattr(args, "session", None)

    promoted_we = consolidate_working_to_episodic(
        mem.tiers["working"],
        mem.tiers["episodic"],
        session_id=session_id,
    )
    print(f"working → episodic: {len(promoted_we)} sources promoted")

    if getattr(args, "promote", False):
        threshold = getattr(args, "recurrence_threshold", 3)
        seed = getattr(args, "cold_start_seed", 0)
        semantic_before = mem.tiers["semantic"].source_count
        promoted_es = promote_to_semantic(
            mem.tiers["episodic"],
            mem.tiers["semantic"],
            recurrence_threshold=threshold,
            cold_start_seed=seed,
        )
        suffix = ""
        if seed > 0 and semantic_before == 0 and promoted_es:
            suffix = f" (cold-start seed: top-{seed} by salience)"
        print(f"episodic → semantic: {len(promoted_es)} sources promoted{suffix}")

    mem.save()
    print("Memory saved.")


def cmd_memory_gc(args: argparse.Namespace) -> None:
    """Garbage-collect: enforce TTL and capacity limits on all tiers."""
    from resonance_lattice.layered_memory import LayeredMemory
    from resonance_lattice.retention import enforce

    root = Path(args.memory_root)
    if not root.exists():
        _die(f"Memory root not found: {root}")

    mem = LayeredMemory.open(root)

    for tier_name, policy in mem.config.retention.items():
        if tier_name not in mem.tiers:
            continue
        removed = enforce(mem.tiers[tier_name], policy)
        if removed:
            print(f"  {tier_name}: evicted {len(removed)} sources")
        else:
            print(f"  {tier_name}: clean")

    mem.save()
    print("GC complete.")


def cmd_memory_profile(args: argparse.Namespace) -> None:
    """Show per-tier memory profile."""
    from resonance_lattice.layered_memory import LayeredMemory

    root = Path(args.memory_root)
    if not root.exists():
        _die(f"Memory root not found: {root}")

    mem = LayeredMemory.open(root)
    info = mem.info()

    print(f"Layered Memory: {root}")
    print(f"  Total sources: {info['total_sources']}")
    print(f"  Tier weights: {info['tier_weights']}")
    print()
    for name, tier_info in info["tiers"].items():
        print(f"  [{name}]")
        print(f"    Sources: {tier_info['sources']}")
        print(f"    Field size: {tier_info['field_size_mb']:.1f} MB")
        energy = tier_info["energy"]
        if energy:
            print(f"    Band energies: {[f'{e:.2f}' for e in energy[:5]]}{'...' if len(energy) > 5 else ''}")
        print()


def cmd_memory_primer(args: argparse.Namespace) -> None:
    """Generate a memory primer for CLAUDE.md inclusion."""
    from resonance_lattice.layered_memory import LayeredMemory
    from resonance_lattice.memory_primer import generate_memory_primer

    root = Path(args.memory_root)
    if not root.exists():
        _die(f"Memory root not found: {root}")

    mem = LayeredMemory.open(root)

    # Attach ONNX backbone if specified
    if getattr(args, "onnx", None) and mem.encoder is not None:
        try:
            mem.encoder.attach_onnx(args.onnx)
        except Exception:
            pass  # fall back to PyTorch

    code_cartridge = None
    if getattr(args, "code_cartridge", None):
        from resonance_lattice.lattice import Lattice
        cart_path = Path(args.code_cartridge)
        if not cart_path.exists():
            _die(f"Code cartridge not found: {args.code_cartridge}")
        code_cartridge = Lattice.load(str(cart_path), restore_encoder=False)

    budget = getattr(args, "budget", 2500)
    novelty_threshold = getattr(args, "novelty_threshold", 0.3)

    print(f"Generating memory primer from {mem.total_sources} memories...", file=sys.stderr)

    result = generate_memory_primer(
        memory=mem,
        code_cartridge=code_cartridge,
        budget=budget,
        novelty_threshold=novelty_threshold,
    )

    output_path = Path(getattr(args, "output", ".claude/memory-primer.md"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.markdown, encoding="utf-8")
    print(
        f"Memory primer written to {output_path} "
        f"({result.total_tokens} tokens, {result.passages_used} passages, "
        f"{result.novelty_filtered} novelty-filtered, "
        f"{result.contradictions_found} contradictions)",
        file=sys.stderr,
    )


def cmd_health(args: argparse.Namespace) -> None:
    """Knowledge health check: signal quality, band health, contradictions, drift."""
    from resonance_lattice.health import HealthCheck
    from resonance_lattice.lattice import Lattice

    lattice_path = Path(args.lattice)
    if not lattice_path.exists():
        _die(f"file not found: {args.lattice}")

    lattice = Lattice.load(str(lattice_path))

    baseline = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            _die(f"baseline not found: {args.baseline}")
        baseline = Lattice.load(str(baseline_path))

    report = HealthCheck.run(
        lattice=lattice,
        baseline=baseline,
        lattice_path=str(lattice_path),
    )

    fmt = getattr(args, "format", "text")
    if fmt == "json":
        import json
        print(json.dumps({
            "status": report.status,
            "snr": report.snr,
            "saturation": report.saturation,
            "remaining_capacity": report.remaining_capacity,
            "band_health": report.band_health,
            "contradiction_count": report.contradiction_count,
            "contradiction_ratio": report.contradiction_ratio,
            "coverage_changes": report.coverage_changes,
            "diagnostics": report.diagnostics,
        }, indent=2))
    else:
        print(report.to_text())

    if getattr(args, "ci", False) and report.status == "CRITICAL":
        sys.exit(1)


def cmd_lens(args: argparse.Namespace) -> None:
    """Dispatch lens subcommands."""
    sub = getattr(args, "lens_command", None)
    if sub is None:
        print("Usage: rlat lens {create,list,compose}")
        sys.exit(1)
    {"create": cmd_lens_create, "list": cmd_lens_list, "compose": cmd_lens_compose}[sub](args)


def cmd_lens_create(args: argparse.Namespace) -> None:
    """Create a .rlens file from topics or a knowledge model eigenspace."""
    from resonance_lattice.lens import LensBuilder

    output = Path(args.output).with_suffix(".rlens")

    if args.topics:
        # Build from topic exemplars — requires an encoder
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        if not topics:
            _die("--topics must provide at least one topic")

        from resonance_lattice.encoder import ResonanceEncoder
        encoder = ResonanceEncoder.from_preset(args.encoder or "bge-large-en-v1.5")
        phases = [encoder.encode_passage(t).vectors for t in topics]
        lens = LensBuilder.from_exemplars(
            name=",".join(topics[:3]),
            phase_vectors_list=phases,
            k=args.rank,
        )
        lens.save(output)
        print(f"Lens saved: {output} (from {len(topics)} topics, rank {lens.metadata.get('rank', '?')})")

    elif args.from_cartridge:
        # Build from cartridge eigenspace
        from resonance_lattice.field.dense import DenseField
        from resonance_lattice.lattice import Lattice
        lattice = Lattice.load(args.from_cartridge)
        if not isinstance(lattice.field, DenseField):
            _die("Lens creation requires a DenseField knowledge model")
        lens = LensBuilder.from_field(lattice.field, name=Path(args.from_cartridge).stem, k=args.rank)
        lens.save(output)
        print(f"Lens saved: {output} (from {args.from_cartridge}, rank {lens.metadata.get('rank', '?')})")

    else:
        _die("Provide --topics or --from to create a lens")


def cmd_lens_list(args: argparse.Namespace) -> None:
    """List .rlens files in a directory."""
    search_dir = Path(getattr(args, "dir", "."))
    rlens_files = sorted(search_dir.rglob("*.rlens"))
    if not rlens_files:
        print(f"No .rlens files found in {search_dir}")
        return
    print(f"Found {len(rlens_files)} lens(es):")
    for f in rlens_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f}  ({size_kb:.1f} KB)")


def cmd_lens_compose(args: argparse.Namespace) -> None:
    """Compose two lenses into one (subspace intersection)."""
    from resonance_lattice.lens import Lens as LensBase

    lens_a = LensBase.load(args.lens_a)
    lens_b = LensBase.load(args.lens_b)

    if not hasattr(lens_a, "compose"):
        _die(f"Lens {args.lens_a} does not support composition")

    composed = lens_a.compose(lens_b)
    output = Path(args.output).with_suffix(".rlens")
    composed.save(output)
    print(f"Composed lens saved: {output}")


def main() -> None:
    _C.init()
    # Enable ANSI on Windows 10+ and fix Unicode output
    if sys.platform == "win32":
        try:
            os.system("")  # triggers VT100 mode on Windows
        except Exception:
            pass
        # Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
        # when printing source text containing non-ASCII characters
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="rlat",
        usage="rlat <command> [options]",
        description="Resonance Lattice — portable semantic model for knowledge.",
        epilog=_HELP_EPILOG,
        formatter_class=_GroupedHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── init ──
    p_init = subparsers.add_parser("init", help="Create a new lattice")
    p_init.add_argument("--bands", type=int, default=5)
    p_init.add_argument("--dim", type=int, default=2048)
    p_init.add_argument("--field-type", default="dense", choices=["dense", "factored", "pq"])
    p_init.add_argument("--pq-subspaces", type=int, default=8)
    p_init.add_argument("--pq-codebook-size", type=int, default=1024)
    p_init.add_argument("--svd-rank", type=int, default=512)
    p_init.add_argument("--precision", default="f32", choices=["f16", "bf16", "f32"])
    p_init.add_argument("--compression", default="none", choices=["none", "zstd", "lz4"])
    p_init.add_argument("--output", "-o", required=True, help="Output .rlat file")

    # ── ingest ──
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into a lattice")
    p_ingest.add_argument("lattice", help="Path to .rlat file")
    p_ingest.add_argument("input", help="Input file or directory to ingest")
    p_ingest.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_ingest.add_argument("--onnx", default=None,
                           help="ONNX backbone directory for faster encoding")

    # ── query ──
    p_query = subparsers.add_parser("query", help="Basic retrieval: ranked passages without enrichment (scripts, debugging)")
    p_query.add_argument("lattice", help="Path to .rlat file")
    p_query.add_argument("query", help="Query text")
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_query.add_argument("--format", default="text", choices=["text", "json", "context", "prompt"])
    p_query.add_argument("--mode", default=None, choices=["augment", "constrain", "knowledge", "custom"],
                         help="Injection framing mode (augment=enrich, constrain=zero-hallucination)")
    p_query.add_argument("--custom-prompt", default=None, help="Custom system prompt (requires --mode custom)")

    # ── info ──
    p_info = subparsers.add_parser("info", help="Display lattice metadata")
    p_info.add_argument("lattice", help="Path to .rlat file")
    p_info.add_argument("--verify", action="store_true",
                          help="For external knowledge models: verify every source's content_hash "
                               "against disk. Prints a drift report. Costs one file read "
                               "per indexed source; skip for large knowledge models unless needed.")
    p_info.add_argument("--source-root", default=None,
                          help="Override source_root (overrides __source_root_hint__ embedded "
                               "at build time). Only meaningful with --verify on external knowledge models.")

    # ── refresh ──
    p_refresh = subparsers.add_parser(
        "refresh",
        help="Re-index drifted files in an external-mode knowledge model "
             "(preserves the field tensor where chunk hashes still match).",
    )
    p_refresh.add_argument("lattice", help="Path to the .rlat file to refresh")
    p_refresh.add_argument(
        "--source-root", required=True,
        help="Directory the manifest paths resolve under. Manifest "
             "__source_root_hint__ is advisory; pass the current root "
             "explicitly.",
    )
    p_refresh.add_argument(
        "--output", "-o", default=None,
        help="Where to write the refreshed knowledge model. Defaults to "
             "overwriting the input path in place.",
    )
    # Encoder flags — same set the build/ingest commands accept so the
    # same overrides work when re-embedding drifted chunks.
    p_refresh.add_argument(
        "--encoder", default=None,
        help="Encoder preset or HuggingFace model ID. Defaults to the "
             "one embedded in the knowledge model at build time (guarantees "
             "build/query parity).",
    )
    p_refresh.add_argument("--onnx", default=None, help="ONNX backbone directory.")
    p_refresh.add_argument("--openvino", default=None, help="OpenVINO IR directory.")
    p_refresh.add_argument("--openvino-device", default=None,
                           help="OpenVINO target: CPU | GPU | NPU | AUTO.")
    p_refresh.add_argument("--openvino-static-seq-len", type=int, default=None,
                           help="Fixed sequence length for OpenVINO NPU.")

    # ── freshness (remote) ──
    p_freshness = subparsers.add_parser(
        "freshness",
        help="Check upstream drift for a remote-mode knowledge model (read-only). "
             "One GitHub API call; exit 0 = up-to-date, exit 1 = drift.",
    )
    p_freshness.add_argument("lattice", help="Path to the remote .rlat file")

    # ── repoint ──
    p_repoint = subparsers.add_parser(
        "repoint",
        help=(
            "Switch a knowledge model's storage mode without re-encoding. "
            "Supports local <-> remote and local/remote -> bundled "
            "(field / registry / manifest are shared across modes; "
            "only the store section changes)."
        ),
    )
    p_repoint.add_argument("lattice", help="Path to the .rlat file to repoint")
    p_repoint.add_argument(
        "--to", required=True, choices=["local", "remote", "bundled"],
        help="Target storage mode. `remote` pins at a GitHub repo; "
             "`local` unpins a remote knowledge model to a local working copy; "
             "`bundled` packs raw source files into the knowledge model for a "
             "self-contained artifact.",
    )
    p_repoint.add_argument(
        "--url", default=None,
        help="GitHub URL (required for `--to remote`). Example: "
             "https://github.com/MicrosoftDocs/fabric-docs",
    )
    p_repoint.add_argument(
        "--source-root", default=None,
        help="Directory containing the source files (required for "
             "`--to bundled` from a local knowledge model; not needed when "
             "repointing from remote — files come from the cache + "
             "pinned upstream).",
    )
    p_repoint.add_argument(
        "--output", "-o", default=None,
        help="Where to write the repointed knowledge model. Defaults to "
             "overwriting the input path in place.",
    )

    # ── diff ──
    p_diff = subparsers.add_parser("diff", help="Corpus subtraction between two lattices")
    p_diff.add_argument("lattice_a", help="First .rlat file")
    p_diff.add_argument("lattice_b", help="Second .rlat file")
    p_diff.add_argument("--output", "-o", help="Output delta .rlat file")

    # ── build ──
    p_build = subparsers.add_parser("build", help="Build a knowledge model from source files")
    p_build.add_argument("inputs", nargs="+", help="Input files or directories")
    p_build.add_argument("--output", "-o", required=True, help="Output .rlat file")
    p_build.add_argument("--bands", type=int, default=5)
    p_build.add_argument("--dim", type=int, default=2048)
    p_build.add_argument("--compact", action="store_true",
                          help="Compact mode: 16x smaller field (5 MB vs 80 MB) at equivalent reranked quality. Sets --dim=512. SciFact: -0.6%% dense nDCG@10, +0.4%% reranked.")
    p_build.add_argument("--field-type", default="dense",
                          choices=["dense", "factored", "pq", "asymmetric_dense"],
                          help="Field storage backend. 'asymmetric_dense' splits key and value dimensionalities (see --dim-key / --dim-value).")
    p_build.add_argument("--dim-key", type=int, default=None,
                          help="Asymmetric field: key dimensionality (default: equals --dim). Only used when --field-type=asymmetric_dense.")
    p_build.add_argument("--dim-value", type=int, default=None,
                          help="Asymmetric field: value dimensionality (default: equals --dim). Only used when --field-type=asymmetric_dense.")
    p_build.add_argument("--sparsify-mode", default=None,
                          choices=["threshold", "topk", "sparsemax", "soft_topk"],
                          help="Projection head sparsification mode (default: threshold). Research/advanced.")
    p_build.add_argument("--soft-topk-tau", type=float, default=None,
                          help="Sigmoid sharpness for --sparsify-mode=soft_topk (default: 10.0). Research/advanced.")
    p_build.add_argument("--sparsemax-scale", type=float, default=None,
                          help="Pre-sparsemax magnitude scale for --sparsify-mode=sparsemax (default: 8/sparsity). Research/advanced.")
    p_build.add_argument("--max-chars", type=int, default=None,
                          help="Chunker: max chars per chunk (default: 1200 for markdown). Advanced — tune for specific corpora.")
    p_build.add_argument("--min-chars", type=int, default=None,
                          help="Chunker: min chars per chunk (default: 150 for markdown). Advanced.")
    p_build.add_argument("--overlap-chars", type=int, default=None,
                          help="Chunker: sliding-window overlap chars (default: 0). Advanced.")
    p_build.add_argument("--precision", default="f32", choices=["f16", "bf16", "f32"])
    p_build.add_argument("--compression", default="none", choices=["none", "zstd", "lz4"])
    p_build.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_build.add_argument("--quantize-registry", type=int, default=0, metavar="BITS",
                          help="Quantize registry phases (0=off, 8=recommended 75%% compression, 4=aggressive 87%% compression)")
    p_build.add_argument(
        "--store-mode", default="external",
        choices=["embedded", "external", "local", "bundled", "remote"],
        help=(
            "Where evidence text lives. "
            "external (default): knowledge model carries only the semantic layer — "
            "field + registry + manifest with content_hash + char_offset. "
            "Content is resolved from files on disk at query time via "
            "--source-root, hash-verified for drift. Portable, privacy-"
            "preserving, maps to the dbt / Power BI semantic-layer pattern. "
            "bundled: lossless + self-contained. Raw source files are packed "
            "into the .rlat as zstd frames; re-chunking, window widening, and "
            "drift detection all still work. Pick this for HF Hub demos, "
            "CI, offline use, or when you want a single-file artifact to "
            "hand someone. Not the same as legacy embedded — bundled stores "
            "whole files, not pre-chunked text. "
            "remote: lossless + HTTP-backed. Knowledge Model pins to a commit SHA "
            "on an upstream repo (currently GitHub public repos). Queries "
            "serve from ~/.cache/rlat/remote; `rlat freshness` / `rlat sync` "
            "manage upgrades like a lockfile. Auto-selected when the "
            "build input is a GitHub URL. "
            "embedded: legacy pre-chunked SQLite store. Deprecated and "
            "scheduled for removal in v2.0.0 — prefer bundled for the "
            "self-contained use case."
        ),
    )
    # Convenience alias for users who want to opt back into the old
    # behaviour without typing the full `--store-mode embedded` pair.
    # `dest` is shared with --store-mode so argparse collapses them.
    p_build.add_argument(
        "--embedded", action="store_const", const="embedded", dest="store_mode",
        help="Shortcut for --store-mode embedded (deprecated; see --store-mode).",
    )
    p_build.add_argument(
        "--path", default=None, metavar="SUBDIR",
        help=(
            "Remote builds only: scope the build to a subdirectory of the "
            "repo (e.g., --path Doc/ for just CPython's docs). The scope is "
            "persisted into __remote_origin__ so `rlat freshness` and "
            "`rlat sync` also respect it — upstream changes outside the "
            "scope are silently dropped. Ignored for local builds (pass "
            "specific paths as inputs instead)."
        ),
    )
    p_build.add_argument("--progress", action="store_true",
                          help="Emit JSON progress lines to stdout (for programmatic consumers)")
    p_build.add_argument("--onnx", default=None,
                          help="ONNX backbone directory for faster encoding (2-5x CPU speedup)")
    p_build.add_argument("--openvino", default=None,
                          help="OpenVINO IR directory (~40x faster on Intel Arc iGPU). See encoder_openvino.export_backbone_openvino.")
    p_build.add_argument("--openvino-device", default=None,
                          help="OpenVINO target: CPU | GPU | NPU | AUTO (default AUTO). Overrides RLAT_OPENVINO_DEVICE.")
    p_build.add_argument("--openvino-static-seq-len", type=int, default=None,
                          help="Reshape the OpenVINO IR to a fixed sequence length (required for NPU; default 512 when --openvino-device NPU).")
    p_build.add_argument("--input-format", default="", choices=["", "conversation"],
                          help="Force input format (default: auto-detect). 'conversation' uses turn-aware chunking.")
    p_build.add_argument("--session", default="", help="Session ID to tag all chunks with (conversation memory)")
    p_build.add_argument("--timestamp", default="", help="ISO timestamp to tag all chunks with (conversation memory)")
    p_build.add_argument("--contextual-chunking", default="auto", choices=["auto", "on", "off"],
                         help="Prepend file/heading context before encoding. 'auto' (default): corpus-aware (skipped for conversation chunks).")
    p_build.add_argument("--probe-qrels", default=None,
                         help="TSV path with test qrels for build-time retrieval-mode probe. "
                              "If provided (with --probe-queries), after build runs all modes "
                              "on the held-out set and writes the winning mode to "
                              "__retrieval_config__ in the knowledge model so `rlat search "
                              "--retrieval-mode auto` ships the per-corpus best mode.")
    p_build.add_argument("--probe-queries", default=None,
                         help="JSONL path with probe queries (BEIR queries.jsonl format).")
    p_build.add_argument("--probe-modes", default=None,
                         help="Comma-separated modes to probe (default: auto — dense-only "
                              "if no --bm25-index, all 7 if BM25 available).")
    p_build.add_argument("--bm25-index", default=None,
                         help="Path to BM25 sidecar for probing plus_rrf / plus_full_stack / bm25_only.")
    p_build.add_argument("--probe-rerankers", default=None,
                         help="Comma-separated reranker model IDs to test for "
                              "CE-using modes (plus_cross_encoder, plus_cross_encoder_expanded, "
                              "plus_full_stack). The probe records the best "
                              "(mode, reranker) combo into __retrieval_config__ so "
                              "`rlat search --retrieval-mode auto` ships the per-corpus "
                              "best reranker (board item 238). Default unset = only "
                              "the built-in default reranker is tested.")

    # ── merge ──
    p_merge = subparsers.add_parser("merge", help="Merge two lattices into one")
    p_merge.add_argument("lattice_a", help="First .rlat file")
    p_merge.add_argument("lattice_b", help="Second .rlat file")
    p_merge.add_argument("--output", "-o", required=True, help="Output merged .rlat file")

    # ── forget ──
    p_forget = subparsers.add_parser("forget", help="Remove a source from a lattice")
    p_forget.add_argument("lattice", help="Path to .rlat file")
    p_forget.add_argument("--source", required=True, help="Source ID to remove")

    # ── summary ──
    # ── primer (self-maintaining primer orchestrator) ──
    p_primer = subparsers.add_parser("primer",
        help="Primer orchestration: refresh both code + memory primers with staleness tracking")
    primer_sub = p_primer.add_subparsers(dest="primer_command")
    p_primer_refresh = primer_sub.add_parser("refresh",
        help="Regenerate .claude/resonance-context.md and .claude/memory-primer.md with git-head stamps")
    p_primer_refresh.add_argument("--knowledge model", default="project.rlat",
        help="Code knowledge model to summarize (default: project.rlat)")
    p_primer_refresh.add_argument("--memory-root", default="./memory",
        help="Memory root dir (default: ./memory)")
    p_primer_refresh.add_argument("--source-root", default=".",
        help="Source root for external-mode summaries (default: .)")
    p_primer_refresh.add_argument("--wait-for-lock", action="store_true",
        help="Block until the refresh lockfile is free (default: exit cleanly if locked)")
    primer_sub.add_parser("status",
        help="Show current primer staleness + git-head stamps without rebuilding")

    p_summary = subparsers.add_parser("summary", help="Generate pre-injection context primer for CLAUDE.md")
    p_summary.add_argument("lattice", help="Path to .rlat file")
    p_summary.add_argument("--output", "-o", help="Output file (default: stdout)")
    p_summary.add_argument("--format", default="context", choices=["context", "stats"],
                           help="Output format: context (rich primer, default) or stats (field metadata only)")
    p_summary.add_argument("--queries", help="Custom bootstrap queries separated by semicolons")
    p_summary.add_argument("--top-k", type=int, default=20, help="Results per bootstrap query (default: 20)")
    p_summary.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_summary.add_argument("--source-root", default=None,
                            help="External source root for file-backed store")
    p_summary.add_argument("--budget", type=int, default=4000,
                            help="Target token budget for output (default: 4000)")
    p_summary.add_argument("--sections",
                            help="Custom section names separated by semicolons "
                                 "(e.g. 'Background;Methods;Findings;Reference'). "
                                 "Default: auto-detect from corpus type")
    p_summary.add_argument("--commit-window", type=int, default=14,
                            help="Days of git history to mine for bootstrap query topics "
                                 "(default: 14; set 0 to disable)")
    _rerank_group = p_summary.add_mutually_exclusive_group()
    _rerank_group.add_argument("--rerank", dest="rerank", action="store_true",
                                default=True,
                                help="Cross-encoder rerank bootstrap results (default: on)")
    _rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false",
                                help="Skip cross-encoder rerank (env: RLAT_PRIMER_RERANK=0)")
    p_summary.add_argument("--memory-root", default=None,
                            help="Path to memory/ dir with working/episodic/semantic .rlat tiers. "
                                 "Files and keywords discussed in recent memory amplify code-primer "
                                 "ranking. Auto-discovers ./memory/ when omitted.")
    p_summary.add_argument("--recency-weight", type=float, default=0.0,
                            help="Recency bias for memory amplification in [0,1]. "
                                 "0 = 14-day half-life (default), 1 = aggressive 3-day half-life.")

    # ── resonate ──
    p_resonate = subparsers.add_parser("resonate", help="AI context: compressed output for LLM injection (use --mode to frame)")
    p_resonate.add_argument("lattice", help="Path to .rlat file")
    p_resonate.add_argument("query", help="Query text")
    p_resonate.add_argument("--top-k", type=int, default=10)
    p_resonate.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_resonate.add_argument("--format", default="context", choices=["text", "json", "context", "prompt"])
    p_resonate.add_argument("--mode", default=None, choices=["augment", "constrain", "knowledge", "custom"],
                            help="Injection framing mode (augment=enrich, constrain=zero-hallucination)")
    p_resonate.add_argument("--custom-prompt", default=None, help="Custom system prompt (requires --mode custom)")
    p_resonate.add_argument("--source-root", default=None,
                             help="Resolve source content from local files under this directory instead of the embedded store")
    p_resonate.add_argument("--onnx", default=None,
                             help="ONNX backbone directory for faster encoding")
    p_resonate.add_argument("-v", "--verbose", action="store_true",
                             help="Show raw scores and detailed timings")

    # ── add (incremental) ──
    p_add = subparsers.add_parser("add", help="Incrementally add files to a knowledge model")
    p_add.add_argument("lattice", help="Path to .rlat file")
    p_add.add_argument("inputs", nargs="+", help="Files or directories to add")
    p_add.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_add.add_argument("--onnx", default=None,
                        help="ONNX backbone directory for faster encoding")
    p_add.add_argument("--openvino", default=None,
                        help="OpenVINO IR directory (~40x faster on Intel Arc iGPU).")
    p_add.add_argument("--openvino-device", default=None,
                        help="OpenVINO target: CPU | GPU | NPU | AUTO (default AUTO).")
    p_add.add_argument("--openvino-static-seq-len", type=int, default=None,
                        help="Reshape the OpenVINO IR to a fixed sequence length (NPU requires this; default 512 when --openvino-device NPU).")
    p_add.add_argument("--input-format", default="", choices=["", "conversation"],
                        help="Force input format (default: auto-detect). 'conversation' uses turn-aware chunking.")
    p_add.add_argument("--session", default="", help="Session ID to tag all chunks with (conversation memory)")
    p_add.add_argument("--timestamp", default="", help="ISO timestamp to tag all chunks with (conversation memory)")
    p_add.add_argument("--contextual-chunking", default="auto", choices=["auto", "on", "off"],
                        help="Prepend file/heading context before encoding. 'auto' (default): corpus-aware (skipped for conversation chunks).")

    # ── sync ──
    p_sync = subparsers.add_parser(
        "sync",
        help=(
            "Sync knowledge model with its source of truth. "
            "Local (external) mode: pass source dirs to re-ingest "
            "(add/update/remove). Remote mode: run with no inputs to "
            "pull the upstream GitHub diff (lockfile-style)."
        ),
    )
    p_sync.add_argument("lattice", help="Path to .rlat file")
    p_sync.add_argument(
        "inputs", nargs="*",
        help="Source directories (external mode). Omit for remote-mode "
             "knowledge models — the pinned __remote_origin__ drives the sync.",
    )
    p_sync.add_argument(
        "--output", "-o", default=None,
        help="Remote-mode only: where to write the synced knowledge model. "
             "Defaults to overwriting the input path in place.",
    )
    p_sync.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_sync.add_argument("--progress", action="store_true",
                          help="Emit JSON progress lines to stdout (for programmatic consumers)")
    p_sync.add_argument("--onnx", default=None,
                          help="ONNX backbone directory for faster encoding")
    p_sync.add_argument("--openvino", default=None,
                          help="OpenVINO IR directory (~40x faster on Intel Arc iGPU).")
    p_sync.add_argument("--openvino-device", default=None,
                          help="OpenVINO target: CPU | GPU | NPU | AUTO (default AUTO).")
    p_sync.add_argument("--openvino-static-seq-len", type=int, default=None,
                          help="Reshape the OpenVINO IR to a fixed sequence length (NPU requires this; default 512 when --openvino-device NPU).")
    p_sync.add_argument("--input-format", default="", choices=["", "conversation"],
                          help="Force input format (default: auto-detect). 'conversation' uses turn-aware chunking.")
    p_sync.add_argument("--session", default="", help="Session ID to tag all chunks with (conversation memory)")
    p_sync.add_argument("--timestamp", default="", help="ISO timestamp to tag all chunks with (conversation memory)")
    p_sync.add_argument("--contextual-chunking", default="auto", choices=["auto", "on", "off"],
                          help="Prepend file/heading context before encoding. 'auto' (default): corpus-aware (skipped for conversation chunks).")

    # ── compare ──
    p_compare = subparsers.add_parser("compare", help="Compare two knowledge models: overlap, unique coverage, depth")
    p_compare.add_argument("lattice_a", help="First .rlat file")
    p_compare.add_argument("lattice_b", help="Second .rlat file")
    p_compare.add_argument("--format", default="text", choices=["text", "json"])

    # ── ls ──
    p_ls = subparsers.add_parser("ls", help="List sources in a knowledge model")
    p_ls.add_argument("lattice", help="Path to .rlat file")
    p_ls.add_argument("--format", default="text", choices=["text", "json"])
    p_ls.add_argument("-v", "--verbose", action="store_true", help="Show summaries")
    p_ls.add_argument("--head", type=int, default=None, help="Show only the first N sources")
    p_ls.add_argument("--grep", default=None, help="Filter sources by substring match")

    # ── profile ──
    p_profile = subparsers.add_parser("profile", help="Semantic profile of a knowledge model")
    p_profile.add_argument("lattice", help="Path to .rlat file")
    p_profile.add_argument("--format", default="text", choices=["text", "json"])

    # ── search (enriched query) ──
    p_search = subparsers.add_parser("search", help="Primary search: enriched results with coverage, topics, and source paths")
    p_search.add_argument("lattice", help="Path to .rlat file")
    p_search.add_argument("query", help="Query text")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--format", default="text", choices=["text", "json", "context", "prompt"])
    p_search.add_argument("--cascade-depth", type=int, default=2, help="Cascade hop depth (default: 2)")
    p_search.add_argument("--contradiction-threshold", type=float, default=None,
                           help="Destructive interference threshold (default: -0.3 when enabled)")
    p_search.add_argument("--no-cascade", action="store_true", default=True,
                           help="Disable related topics cascade (default: disabled)")
    p_search.add_argument("--cascade", dest="no_cascade", action="store_false",
                           help="Enable related topics cascade")
    contradictions_group = p_search.add_mutually_exclusive_group()
    contradictions_group.add_argument("--with-contradictions", dest="enable_contradictions",
                                      action="store_true", help="Enable contradiction detection")
    contradictions_group.add_argument("--no-contradictions", dest="enable_contradictions",
                                      action="store_false", help="Disable contradiction detection")
    p_search.set_defaults(enable_contradictions=False)
    p_search.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_search.add_argument("--no-worker", action="store_true",
                           help="Disable background worker (env: RLAT_NO_WORKER=1)")
    p_search.add_argument("--source-root", default=None,
                           help="Resolve source content from local files under this directory instead of the embedded store")
    p_search.add_argument("--subgraph", action="store_true",
                           help="Expand results with spectral neighbours (richer context)")
    p_search.add_argument("--subgraph-k", type=int, default=3,
                           help="Neighbours per result for subgraph expansion (default: 3)")
    p_search.add_argument("--onnx", default=None,
                           help="ONNX backbone directory for faster encoding (auto-detects <stem>_onnx/)")
    p_search.add_argument("--rerank", default="auto", choices=["auto", "true", "false"],
                           help="Reranking mode: auto skips when dense is confident (default: auto)")
    p_search.add_argument(
        "--retrieval-mode", dest="retrieval_mode", default="auto",
        choices=["auto", "field_only", "plus_cross_encoder",
                 "plus_cross_encoder_expanded", "plus_hybrid", "plus_rrf",
                 "plus_full_stack", "bm25_only"],
        help=(
            "Retrieval stack selection (default: auto). "
            "auto: use knowledge model's probed default (set via `rlat build "
            "--probe-modes`) or field_only if none. "
            "field_only: dense retrieval, no post-processing. "
            "plus_cross_encoder: dense + cross-encoder rerank. "
            "plus_cross_encoder_expanded: + B1 passage expansion. "
            "plus_hybrid / plus_rrf / plus_full_stack / bm25_only require "
            "BM25 sidecar plumbing landing with the probe work (board 236c)."
        ),
    )
    p_search.add_argument("-v", "--verbose", action="store_true",
                           help="Show raw scores and detailed timings")
    p_search.add_argument("--mode", default=None, choices=["augment", "constrain", "knowledge", "custom"],
                           help="Injection framing mode for prompt/context formats")
    p_search.add_argument("--custom-prompt", default=None,
                           help="Custom system prompt (requires --mode custom)")
    # ── composition flags ──
    p_search.add_argument("--with", dest="with_cartridges", action="append", default=[],
                           help="Additional .rlat knowledge model(s) to compose with (repeatable). "
                                "Merges fields, dispatches searches to each registry independently.")
    p_search.add_argument("--through", default=None,
                           help="Project primary knowledge model through this .rlat's perspective "
                                "(semantic projection: 'show me A's knowledge through B's lens')")
    p_search.add_argument("--diff", dest="diff_against", default=None,
                           help="Show what's new in the primary knowledge model vs this baseline .rlat "
                                "(queryable semantic diff)")
    # ── topic control flags ──
    p_search.add_argument("--boost", dest="boost_topics", action="append", default=[],
                           help="Boost a semantic topic during search (repeatable). "
                                "Amplifies the topic direction in the field before querying.")
    p_search.add_argument("--suppress", dest="suppress_topics", action="append", default=[],
                           help="Suppress a semantic topic during search (repeatable). "
                                "Attenuates the topic direction — like removing it on the fly.")
    p_search.add_argument("--boost-strength", type=float, default=0.5,
                           help="Strength of topic boosting (default: 0.5)")
    p_search.add_argument("--suppress-strength", type=float, default=0.3,
                           help="Strength of topic suppression (default: 0.3)")
    # ── EML corpus transforms ──
    p_search.add_argument("--sharpen", type=float, default=None, metavar="STRENGTH",
                           help="Sharpen the corpus field for more precise retrieval. "
                                "Nonlinear EML contrast enhancement: amplifies dominant topics, "
                                "compresses noise. Higher = sharper (try 1.0-2.0).")
    p_search.add_argument("--soften", type=float, default=None, metavar="STRENGTH",
                           help="Soften the corpus field for broader exploration. "
                                "Flattens the spectrum so buried topics surface. "
                                "Higher = flatter (try 0.5-1.5).")
    p_search.add_argument("--contrast", default=None, metavar="CARTRIDGE",
                           help="Asymmetric EML contrast against a background knowledge model. "
                                "Shows what THIS knowledge model knows that the background doesn't. "
                                "Uses REML: expm(primary) - logm(background).")
    p_search.add_argument("--tune", default=None, choices=["focus", "explore", "denoise"],
                           help="Tune retrieval mode. focus: precision for specific questions. "
                                "explore: breadth for research queries. "
                                "denoise: clean noisy corpora with boilerplate.")
    p_search.add_argument("--explain", action="store_true",
                           help="Show composition diagnostics before searching "
                                "(overlap, novelty, contradiction ratio between knowledge models)")
    # ── cascade flags ──
    p_search.add_argument("--cascade-through", dest="cascade_route", nargs="+", default=None,
                           help="Cross-knowledge model cascade: follow semantic links through a route of .rlat files. "
                                "E.g. --cascade-through docs.rlat incidents.rlat")
    p_search.add_argument("--access", default=None,
                           help="Apply an access policy (.rlens file or built from field). "
                                "Restricts visible knowledge to the policy's allowed subspace.")
    # ── lens flags ──
    p_search.add_argument("--lens", default=None,
                           help="Apply a named knowledge lens. Either a .rlens file path or a "
                                "built-in name: 'sharpen', 'flatten', 'denoise'")
    # ── conversation memory filters ──
    p_search.add_argument("--session", default=None,
                           help="Filter results by session ID (conversation memory)")
    p_search.add_argument("--after", default=None,
                           help="Filter results after ISO timestamp (conversation memory)")
    p_search.add_argument("--before", default=None,
                           help="Filter results before ISO timestamp (conversation memory)")
    p_search.add_argument("--speaker", default=None, choices=["human", "assistant", "system", "qa_pair"],
                           help="Filter results by speaker role (conversation memory)")
    p_search.add_argument("--recency-weight", type=float, default=0.0,
                           help="Blend recency into ranking (0.0=off, 0.3=moderate, 1.0=recency-only)")
    # ── B4/B5: three-layer retrieval flags (semantic layer v1.0.0) ──
    p_search.add_argument("--expand", default="off",
                           choices=["off", "natural", "max"],
                           help="Grow each result to a natural boundary (section / "
                                "function / paragraph). 'off' returns the chunk as-is; "
                                "'natural' grows to the smallest enclosing unit; "
                                "'max' grows to the top-level unit. Requires external "
                                "store or --source-root. Default: off.")
    p_search.add_argument("--hybrid", default="auto",
                           choices=["off", "on", "auto"],
                           help="Run a lexical (ripgrep) second pass over the retrieved "
                                "neighbourhood and blend the signal into ranking. "
                                "'auto' = on for external-mode knowledge models when source "
                                "files are reachable, off otherwise. Fail-soft when "
                                "ripgrep is unavailable. Default: auto.")

    # ── compose (advanced expression-based composition) ──
    p_compose = subparsers.add_parser(
        "compose",
        help="Search with algebraic composition expressions or .rctx context files",
    )
    p_compose.add_argument("expression", help="Composition expression or path to .rctx file")
    p_compose.add_argument("query", help="Search query")
    p_compose.add_argument("--top-k", type=int, default=10)
    p_compose.add_argument("--format", default="text", choices=["text", "json"])
    p_compose.add_argument("--explain", action="store_true", help="Show composition diagnostics")
    p_compose.add_argument("--encoder", default=None)
    p_compose.add_argument("--onnx", default=None)

    # ── contradictions ──
    p_contra = subparsers.add_parser("contradictions", help="Find contradicting sources")
    p_contra.add_argument("lattice", help="Path to .rlat file")
    p_contra.add_argument("query", help="Query text to check for contradictions")
    p_contra.add_argument("--band", type=int, default=2, help="Band to check (default: relations)")
    p_contra.add_argument("--threshold", type=float, default=0.7, help="Anti-correlation threshold")
    p_contra.add_argument("--top-k", type=int, default=20)
    p_contra.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")

    # ── serve ──
    p_serve = subparsers.add_parser("serve", help="Start lattice HTTP server")
    p_serve.add_argument("lattice", help="Path to .rlat file")
    p_serve.add_argument("--port", type=int, default=8080, help="Port number")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address")
    p_serve.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")

    # ── topology ──
    p_topo = subparsers.add_parser("topology", help="Analyze knowledge structure: clusters, robustness, and topological features")
    p_topo.add_argument("lattice", help="Path to .rlat file")
    p_topo.add_argument("--band", type=int, default=0)
    p_topo.add_argument("--top-k", type=int, default=20)
    p_topo.add_argument("--output", "-o", help="Output JSON file")

    # ── xray ──
    p_xray = subparsers.add_parser("xray", help="Field X-Ray: corpus-level semantic diagnostics")
    p_xray.add_argument("lattice", help="Path to .rlat file")
    p_xray.add_argument("--format", default="text", choices=["text", "json", "prompt"])
    p_xray.add_argument("--deep", action="store_true", help="Add topological analysis and community detection")

    # ── locate ──
    p_locate = subparsers.add_parser("locate", help="Query positioning: where does this question sit?")
    p_locate.add_argument("lattice", help="Path to .rlat file")
    p_locate.add_argument("query", help="Query text")
    p_locate.add_argument("--format", default="text", choices=["text", "json", "prompt"])
    p_locate.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")

    # ── probe ──
    p_probe = subparsers.add_parser("probe", help="RQL quick insight recipes")
    p_probe.add_argument("lattice", help="Path to .rlat file")
    p_probe.add_argument("recipe", help="Recipe name: health, novelty, saturation, band-flow, anti, gaps")
    p_probe.add_argument("query", nargs="?", default="", help="Query text (required for novelty, anti)")
    p_probe.add_argument("--format", default="text", choices=["text", "json", "prompt"])
    p_probe.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")

    # ── init-project ──
    p_initp = subparsers.add_parser("init-project",
        help="One-command setup: build knowledge model + generate summary + integration instructions")
    p_initp.add_argument("inputs", nargs="*", help="Input files or directories (auto-detects if omitted)")
    p_initp.add_argument("--output", "-o", help="Summary output path (default: .claude/resonance-context.md)")
    p_initp.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID (run 'rlat encoders' to list presets)")
    p_initp.add_argument("--onnx", default=None,
                          help="ONNX backbone directory for faster encoding")
    p_initp.add_argument("--auto-integrate", action="store_true",
                          help="Automatically update .mcp.json, inject CLAUDE.md knowledge model section, "
                               "and create .rlat/manifest.json")

    # ── setup (wizard) ──
    p_setup = subparsers.add_parser("setup",
        help="Guided project setup: knowledge models, skills, memory, integration (interactive wizard)")
    p_setup.add_argument("--non-interactive", "--ni", action="store_true",
                         help="Run with defaults (no interactive prompts)")
    p_setup.add_argument("--config", default=None,
                         help="Load setup config from TOML file (e.g. .rlat/setup.toml)")
    p_setup.add_argument("--reconfigure", action="store_true",
                         help="Force re-prompt even with existing setup")
    p_setup.add_argument("--encoder", default=None,
                         help="Override encoder preset")
    p_setup.add_argument("--no-memory", action="store_true",
                         help="Disable layered memory setup")
    p_setup.add_argument("--precision", default=None, choices=["f16", "bf16", "f32"])
    p_setup.add_argument("--compression", default=None, choices=["none", "zstd", "lz4"])

    # ── encoders ──
    subparsers.add_parser("encoders", help="List available encoder presets")

    # ── Export subcommand (field-only cartridges, PR #60) ──────────
    p_export = subparsers.add_parser("export", help="Export a knowledge model (supports field-only mode)")
    p_export.add_argument("knowledge model", help="Source .rlat knowledge model")
    p_export.add_argument("--output", "-o", required=True, help="Output path")
    p_export.add_argument("--field-only", action="store_true",
                          help="Export field+registry without source store (privacy-preserving)")

    # ── MCP server ───────────────────────────────────────────────────
    p_mcp = subparsers.add_parser("mcp", help="Start MCP server (stdio transport) for Claude Code")
    p_mcp.add_argument("knowledge model", help="Path to .rlat knowledge model")
    p_mcp.add_argument("--source-root", default=None,
                        help="External source root for file-backed store")
    p_mcp.add_argument("--onnx", default=None,
                        help="ONNX backbone directory for faster encoding")

    # ── skill ─────────────────────────────────────────────────────────
    p_skill = subparsers.add_parser("skill", help="Skill knowledge model integration (build, sync, search)")
    skill_sub = p_skill.add_subparsers(dest="skill_command")

    p_sk_build = skill_sub.add_parser("build", help="Build knowledge model from skill's reference materials")
    p_sk_build.add_argument("skill_dir", help="Path to skill directory (containing SKILL.md)")
    p_sk_build.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID")
    p_sk_build.add_argument("--onnx", default=None, help="ONNX backbone directory")

    p_sk_sync = skill_sub.add_parser("sync", help="Incrementally sync a skill's knowledge model")
    p_sk_sync.add_argument("skill_dir", help="Path to skill directory (containing SKILL.md)")
    p_sk_sync.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID")
    p_sk_sync.add_argument("--onnx", default=None, help="ONNX backbone directory")

    p_sk_search = skill_sub.add_parser("search", help="Search a skill's knowledge models")
    p_sk_search.add_argument("skill_name", help="Skill name or path to skill directory")
    p_sk_search.add_argument("query", help="Search query")
    p_sk_search.add_argument("--top-k", type=int, default=10)
    p_sk_search.add_argument("--format", default="text", choices=["text", "json", "context", "prompt"])
    p_sk_search.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID")
    p_sk_search.add_argument("--source-root", default=None, help="External source root")
    p_sk_search.add_argument("--onnx", default=None, help="ONNX backbone directory")

    p_sk_info = skill_sub.add_parser("info", help="Show skill knowledge model configuration and status")
    p_sk_info.add_argument("skill_name", nargs="?", default=None,
                           help="Skill name or path (omit to list all)")

    p_sk_inject = skill_sub.add_parser("inject", help="Four-tier adaptive context injection")
    p_sk_inject.add_argument("skill_name", help="Skill name or path to skill directory")
    p_sk_inject.add_argument("query", help="User query to inject context for")
    p_sk_inject.add_argument("--format", default="text", choices=["text", "json", "context"],
                              help="Output format (text=diagnostic, json=structured, context=injectable body)")
    p_sk_inject.add_argument("--derived", default=None,
                              help="Tier 4 derived queries (semicolon-separated). Omit to skip Tier 4.")
    p_sk_inject.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID")
    p_sk_inject.add_argument("--source-root", default=None, help="External source root")
    p_sk_inject.add_argument("--onnx", default=None, help="ONNX backbone directory")

    p_sk_route = skill_sub.add_parser("route", help="Rank skills by relevance to a query")
    p_sk_route.add_argument("query", help="Query to route")
    p_sk_route.add_argument("--top-n", type=int, default=5, help="Max skills to return")
    p_sk_route.add_argument("--format", default="text", choices=["text", "json"])

    p_sk_profile = skill_sub.add_parser("profile", help="Semantic profile of a skill's knowledge model")
    p_sk_profile.add_argument("skill_name", help="Skill name or path to skill directory")
    p_sk_profile.add_argument("--format", default="text", choices=["text", "json"])

    p_sk_fresh = skill_sub.add_parser("freshness", help="Check freshness of skill knowledge models")
    p_sk_fresh.add_argument("skill_name", nargs="?", default=None,
                            help="Skill name or path (omit for all)")
    p_sk_fresh.add_argument("--format", default="text", choices=["text", "json"])

    p_sk_gaps = skill_sub.add_parser("gaps", help="Detect knowledge gaps in a skill's knowledge model")
    p_sk_gaps.add_argument("skill_name", help="Skill name or path to skill directory")
    p_sk_gaps.add_argument("--format", default="text", choices=["text", "json", "prompt"])
    p_sk_gaps.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID")

    p_sk_compare = skill_sub.add_parser("compare", help="Compare two skills' knowledge models")
    p_sk_compare.add_argument("skill_a", help="First skill name or path")
    p_sk_compare.add_argument("skill_b", help="Second skill name or path")
    p_sk_compare.add_argument("--format", default="text", choices=["text", "json"])

    # ── Health ──
    p_health = subparsers.add_parser("health", help="Knowledge health check: signal, bands, contradictions, drift")
    p_health.add_argument("lattice", help="Knowledge Model to check")
    p_health.add_argument("--baseline", default=None, help="Compare against baseline for drift/contradiction detection")
    p_health.add_argument("--ci", action="store_true", help="CI mode: exit 1 on CRITICAL status")
    p_health.add_argument("--format", default="text", choices=["text", "json"])

    # ── Negotiate ──
    p_negotiate = subparsers.add_parser("negotiate",
        help="Analyze relationship between two knowledge models: shared ground, novelty, disagreements")
    p_negotiate.add_argument("cartridge_a", help="First knowledge model")
    p_negotiate.add_argument("cartridge_b", help="Second knowledge model")
    p_negotiate.add_argument("--query", default=None, help="Focus analysis on a specific topic")
    p_negotiate.add_argument("--format", default="text", choices=["text", "json"])

    # ── ask (lens-routed query) ──
    p_ask = subparsers.add_parser(
        "ask",
        help="Smart query: auto-selects the best retrieval lens for your question",
    )
    p_ask.add_argument("lattice", help="Path to .rlat file")
    p_ask.add_argument("query", help="Your question")
    p_ask.add_argument("--explain", action="store_true",
                        help="Show which lens was chosen and why, plus the command to run")
    p_ask.add_argument("--with", dest="with_cartridges", action="append", default=[],
                        help="Additional knowledge model(s) for multi-knowledge model queries (repeatable)")
    p_ask.add_argument("--background", default=None,
                        help="Background knowledge model for contrast queries")
    p_ask.add_argument("--top-k", type=int, default=10)
    p_ask.add_argument("--format", default="text", choices=["text", "json", "context"])
    p_ask.add_argument("--encoder", default=None)
    # ── C4: reader synthesis (three-layer architecture) ──
    p_ask.add_argument("--reader", default="off",
                        choices=["off", "context", "llm"],
                        help="Reader mode. 'off' keeps the existing lens-routing "
                             "behaviour (runs search/locate/profile via subprocess). "
                             "'context' returns the evidence as a labelled prompt "
                             "pack (no LLM). 'llm' synthesizes a grounded answer "
                             "with citations. Default: off.")
    p_ask.add_argument("--reader-backend", default="auto",
                        choices=["auto", "local", "anthropic", "openai"],
                        help="LLM backend for --reader llm. 'auto' picks local "
                             "(if OpenVINO+optimum-intel are installed AND "
                             "--reader-model is set) then anthropic (CLAUDE_API) "
                             "then openai (OPENAI_API_KEY). Default: auto.")
    p_ask.add_argument("--reader-model", default=None,
                        help="Model id for the reader. HF id for local (e.g. "
                             "'Qwen/Qwen2.5-3B-Instruct'); provider model for "
                             "anthropic/openai (default claude-opus-4-7 / gpt-4o-mini).")
    p_ask.add_argument("--source-root", default=None,
                        help="Directory containing source files. Required for "
                             "reader=context/llm when the knowledge model was built in "
                             "external mode with a non-portable hint.")
    p_ask.add_argument("--expand", default="natural",
                        choices=["off", "natural", "max"],
                        help="Context expansion for retrieved evidence "
                             "(reader modes only). Default: natural.")
    p_ask.add_argument("--hybrid", default="auto",
                        choices=["off", "on", "auto"],
                        help="Lexical second-pass reranking "
                             "(reader modes only). Default: auto.")
    p_ask.add_argument("--max-tokens", type=int, default=1024,
                        help="Generation cap for --reader llm.")
    p_ask.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature for --reader llm.")

    # ── Memory subcommands ────────────────────────────────────────────
    p_memory = subparsers.add_parser("memory", help="Agent memory: search, add, forget, stats")
    memory_sub = p_memory.add_subparsers(dest="memory_command")

    p_mem_search = memory_sub.add_parser("search", help="Search memory for relevant prior context")
    p_mem_search.add_argument("query", help="What to recall")
    p_mem_search.add_argument("--top-k", type=int, default=5)
    p_mem_search.add_argument("--format", default="text", choices=["text", "json"])

    p_mem_add = memory_sub.add_parser("add", help="Add content to memory")
    p_mem_add.add_argument("--content", required=True, help="Text to remember")
    p_mem_add.add_argument("--session", default="", help="Session ID tag")

    p_mem_forget = memory_sub.add_parser("forget", help="Provably remove from memory")
    p_mem_forget.add_argument("--topic", default=None, help="Topic to forget")
    p_mem_forget.add_argument("--session", default=None, help="Session ID to forget")

    memory_sub.add_parser("stats", help="Memory knowledge model statistics")

    # Layered memory subcommands
    p_mem_init = memory_sub.add_parser("init", help="Initialize a layered memory root")
    p_mem_init.add_argument("memory_root", help="Directory for the memory root")

    p_mem_write = memory_sub.add_parser("write", help="Write to a layered memory tier")
    p_mem_write.add_argument("memory_root", help="Memory root directory")
    p_mem_write.add_argument("content", nargs="?", default=None, help="Text to remember (optional if --input-file)")
    p_mem_write.add_argument("--tier", default="working", choices=["working", "episodic", "semantic"])
    p_mem_write.add_argument("--session", default="", help="Session ID tag")
    p_mem_write.add_argument("--input-file", default=None, help="Ingest from file (.jsonl for Claude transcript, .md for conversation)")
    p_mem_write.add_argument("--input-format", default="auto", choices=["auto", "conversation", "claude_transcript"],
                             help="Input format (auto-detected from extension by default)")
    p_mem_write.add_argument("--onnx", default=None,
                             help="ONNX backbone directory for faster encoding (2-5x CPU speedup)")
    p_mem_write.add_argument("--encoder", default=None,
                             help="Encoder preset (default: bge-large-en-v1.5)")
    p_mem_write.add_argument("--openvino", default=None,
                             help="OpenVINO IR directory (~40x faster on Intel Arc iGPU)")
    p_mem_write.add_argument("--openvino-device", default=None,
                             help="OpenVINO target: CPU | GPU | NPU | AUTO")

    p_mem_recall = memory_sub.add_parser("recall", help="Recall across all memory tiers")
    p_mem_recall.add_argument("memory_root", help="Memory root directory")
    p_mem_recall.add_argument("query", help="What to recall")
    p_mem_recall.add_argument("--top-k", type=int, default=10)
    p_mem_recall.add_argument("--tier-weights", default=None, help="e.g. working=0.8,episodic=0.15,semantic=0.05")
    p_mem_recall.add_argument("--tiers", default=None, help="Comma-separated tier names to query")
    p_mem_recall.add_argument("--onnx", default=None,
                              help="ONNX backbone directory for faster encoding")

    p_mem_consolidate = memory_sub.add_parser("consolidate", help="Promote sources between tiers")
    p_mem_consolidate.add_argument("memory_root", help="Memory root directory")
    p_mem_consolidate.add_argument("--session", default=None, help="Only consolidate this session")
    p_mem_consolidate.add_argument("--promote", action="store_true", help="Also promote episodic → semantic")
    p_mem_consolidate.add_argument("--recurrence-threshold", type=int, default=3)
    p_mem_consolidate.add_argument(
        "--cold-start-seed", type=int, default=0,
        help="When semantic is empty, seed it with the top-N highest-salience "
             "episodic sources regardless of access_count. Opt-in (default 0); "
             "fires only while semantic.source_count == 0.",
    )

    p_mem_gc = memory_sub.add_parser("gc", help="Enforce TTL and capacity limits")
    p_mem_gc.add_argument("memory_root", help="Memory root directory")

    p_mem_profile = memory_sub.add_parser("profile", help="Per-tier memory profile")
    p_mem_profile.add_argument("memory_root", help="Memory root directory")

    p_mem_primer = memory_sub.add_parser("primer", help="Generate memory primer for CLAUDE.md")
    p_mem_primer.add_argument("memory_root", help="Memory root directory")
    p_mem_primer.add_argument("--output", "-o", default=".claude/memory-primer.md",
                              help="Output file (default: .claude/memory-primer.md)")
    p_mem_primer.add_argument("--code-knowledge model", default=None,
                              help="Code knowledge model for cross-primer novelty filtering")
    p_mem_primer.add_argument("--budget", type=int, default=3500,
                              help="Target token budget (default: 3500)")
    p_mem_primer.add_argument("--novelty-threshold", type=float, default=0.3,
                              help="Novelty threshold for cross-primer filtering (default: 0.3)")
    p_mem_primer.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID")
    p_mem_primer.add_argument("--onnx", default=None, help="ONNX backbone directory")

    # ── Lens subcommands ──────────────────────────────────────────────
    p_lens = subparsers.add_parser("lens", help="Knowledge lenses: named semantic perspectives")
    lens_sub = p_lens.add_subparsers(dest="lens_command")

    p_lens_create = lens_sub.add_parser("create", help="Create a lens from topics or a knowledge model")
    p_lens_create.add_argument("-o", "--output", required=True, help="Output .rlens file path")
    p_lens_create.add_argument("--topics", default=None,
                                help="Comma-separated topic strings (e.g. 'security,auth,encryption')")
    p_lens_create.add_argument("--from", dest="from_cartridge", default=None,
                                help="Build lens from a knowledge model's eigenspace")
    p_lens_create.add_argument("--rank", type=int, default=None,
                                help="Subspace rank (default: auto from effective rank)")
    p_lens_create.add_argument("--encoder", default=None, help="Encoder preset or HuggingFace model ID")
    p_lens_create.add_argument("--onnx", default=None, help="ONNX backbone directory")

    p_lens_list = lens_sub.add_parser("list", help="List available .rlens files")
    p_lens_list.add_argument("--dir", default=".", help="Directory to search (default: current)")

    p_lens_compose = lens_sub.add_parser("compose", help="Compose two lenses (subspace intersection)")
    p_lens_compose.add_argument("lens_a", help="First .rlens file")
    p_lens_compose.add_argument("lens_b", help="Second .rlens file")
    p_lens_compose.add_argument("-o", "--output", required=True, help="Output .rlens file path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "init": cmd_init,
        "build": cmd_build,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "resonate": cmd_resonate,
        "info": cmd_info,
        "summary": cmd_summary,
        "diff": cmd_diff,
        "merge": cmd_merge,
        "forget": cmd_forget,
        "add": cmd_add,
        "sync": cmd_sync,
        "compare": cmd_compare,
        "ls": cmd_ls,
        "profile": cmd_profile,
        "search": cmd_search,
        "compose": cmd_compose,
        "contradictions": cmd_contradictions,
        "serve": cmd_serve,
        "topology": cmd_topology,
        "xray": cmd_xray,
        "locate": cmd_locate,
        "probe": cmd_probe,
        "init-project": cmd_init_project,
        "setup": cmd_setup,

        "encoders": cmd_encoders,
        "export": cmd_export,
        "mcp": cmd_mcp,
        "skill": cmd_skill,
        "lens": cmd_lens,
        "health": cmd_health,
        "negotiate": cmd_negotiate,
        "ask": cmd_ask,
        "memory": cmd_memory,
        "primer": cmd_primer,
        "refresh": cmd_refresh,
        "freshness": cmd_freshness,
        "repoint": cmd_repoint,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
