# SPDX-License-Identifier: BUSL-1.1
"""Lattice — main orchestrator tying field, registry, and store together.

The Lattice class is the primary interface for the Resonance Lattice system.
It coordinates:
  1. Encoding: text -> phase spectrum (via Encoder)
  2. Superposition: phase spectrum -> field update + registry + store
  3. Resonance: query -> field projection -> registry lookup -> materialisation
  4. Removal: algebraically exact rank-1 subtraction
  5. Persistence: save/load via .rlat binary format

Supports all three field backends: Dense, Factored (SVD), and PQ.
"""

from __future__ import annotations

import logging
import os.path
import re
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

from resonance_lattice.config import FieldType, LatticeConfig
from resonance_lattice.encoder import Encoder, PhaseSpectrum
from resonance_lattice.field.asymmetric_dense import AsymmetricDenseField
from resonance_lattice.field.dense import DenseField, ResonanceResult
from resonance_lattice.field.factored import FactoredField
from resonance_lattice.field.pq import PQField
from resonance_lattice.registry import PhaseRegistry, SourcePointer
from resonance_lattice.serialise import (
    RlatHeader,
    load_dense_field,
    load_factored_field,
    load_pq_field,
    save_dense_field,
    save_factored_field,
    save_pq_field,
)
from resonance_lattice.store import LocalStore, SourceContent, SourceStore

# Union type for all field backends
FieldBackend = DenseField | AsymmetricDenseField | FactoredField | PQField


BAND_NAMES = ("domain", "topic", "relations", "entity", "verbatim")


def _load_source_manifest(store_data: bytes) -> dict[str, str | dict]:
    """Extract source manifest from embedded store data (if present).

    Returns a mapping of source_id -> file path (str, legacy) or rich
    entry (dict with source_file, heading, chunk_index).
    """
    if not store_data:
        return {}
    try:
        import json
        tmp_store = SourceStore.from_bytes(store_data)
        entry = tmp_store.retrieve("__source_manifest__")
        if entry and entry.full_text:
            return json.loads(entry.full_text)
    except Exception:
        pass
    return {}


def _is_external_cartridge(store_data: bytes) -> bool:
    """Return True if store_data contains only metadata (external knowledge model)."""
    if not store_data:
        return False
    try:
        tmp_store = SourceStore.from_bytes(store_data)
        ids = tmp_store.all_ids()
        return all(sid.startswith("__") for sid in ids)
    except Exception:
        return False


# Reserved manifest key recording the absolute source_root at build time.
# Starts with "__" so it never collides with a real source_id (all of which
# are derived from file stems) and is inert to LocalStore._resolve_path
# which only looks up specific source_ids.
_MANIFEST_SOURCE_ROOT_KEY = "__source_root_hint__"


def _manifest_entries(manifest: dict) -> list:
    """Return only the real source entries from a manifest, skipping reserved
    metadata keys (any key starting with '__')."""
    return [v for k, v in manifest.items() if not (isinstance(k, str) and k.startswith("__"))]


def _infer_source_root(cartridge_path: Path, manifest: dict[str, str | dict]) -> Path | None:
    """Try to infer source_root from a manifest.

    Preference order:
    1. The explicit `__source_root_hint__` recorded at build time (v2+
       manifests). Works when the corpus lives at the same absolute path
       on the reader's machine (same dev box, mounted share, etc.).
    2. Heuristic: sample manifest entries and see if their paths resolve
       relative to the knowledge model's own directory. Works when the knowledge model
       ships alongside its source tree.

    Returns None if neither yields an existing directory; the caller then
    falls back to requiring `--source-root`.
    """
    if not manifest:
        return None

    # (1) Build-time hint — preferred when it's still valid on this machine.
    hint = manifest.get(_MANIFEST_SOURCE_ROOT_KEY)
    if isinstance(hint, str) and hint:
        hint_path = Path(hint)
        if hint_path.exists() and hint_path.is_dir():
            return hint_path.resolve()

    # (2) Sample real entries (skipping reserved __ keys) to check against
    # the cartridge's own directory.
    for entry in _manifest_entries(manifest)[:5]:
        sf = entry["source_file"] if isinstance(entry, dict) else entry
        candidate = cartridge_path.parent / sf
        if candidate.exists():
            return cartridge_path.parent.resolve()

    return None


def _compute_source_root(paths: list[str]) -> str | None:
    """Compute the common parent directory of a list of source paths.

    All paths are resolved to absolute form before computing the common
    prefix. Returns None if paths cross drives/roots (Windows) or the
    input is empty.
    """
    if not paths:
        return None
    abs_paths: list[str] = []
    for p in paths:
        if not p:
            continue
        try:
            abs_paths.append(str(Path(p).resolve()))
        except (OSError, ValueError):
            continue
    if not abs_paths:
        return None
    try:
        common = os.path.commonpath(abs_paths)
    except ValueError:
        # Paths on different drives (Windows) or no common prefix
        return None
    # If commonpath returned a file (single-path case), use its parent dir.
    if Path(common).is_file():
        common = str(Path(common).parent)
    return common


_CLAIM_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "via", "was", "what", "when", "where", "which", "with",
}


@dataclass
class CoverageProfile:
    """Per-band energy and confidence at the query point."""
    band_energies: NDArray[np.float32]  # (B,) — qᵀ F_b q per band
    band_names: list[str]  # Approximate human labels
    total_energy: float
    confidence: float  # 0-1, based on curvature concentration
    gaps: list[str]  # Bands with low energy (potential knowledge gaps)


@dataclass
class RelatedTopic:
    """A topic discovered via multi-hop cascade."""
    source_id: str
    score: float
    hop: int  # Which hop surfaced this (1=direct, 2=indirect)
    content: SourceContent | None


@dataclass
class ContradictionPair:
    """A pair of sources with destructive interference."""
    source_a: str
    source_b: str
    interference: float  # Negative = stronger contradiction
    summary_a: str
    summary_b: str


@dataclass
class EnrichedResult:
    """Complete enriched query result combining passages + structural metadata."""
    query: str
    results: list[MaterialisedResult]
    coverage: CoverageProfile
    related: list[RelatedTopic]
    contradictions: list[ContradictionPair]
    latency_ms: float
    timings_ms: dict[str, float]
    assessment: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise everything for JSON consumers."""
        d: dict[str, Any] = {
            "query": self.query,
            "latency_ms": round(float(self.latency_ms), 2),
            "results": [
                {
                    "source_id": r.source_id,
                    "score": round(float(r.score), 4),
                    "raw_score": (
                        round(float(r.raw_score), 4)
                        if r.raw_score is not None else None
                    ),
                    "band_scores": r.band_scores.tolist() if r.band_scores is not None else None,
                    "summary": r.content.summary if r.content else None,
                    "full_text": r.content.full_text if r.content else None,
                    "source_file": (
                        r.content.metadata.get("source_file", "")
                        if r.content and r.content.metadata else ""
                    ),
                    "heading": (
                        r.content.metadata.get("heading", "")
                        if r.content and r.content.metadata else ""
                    ),
                    "provenance": getattr(r, "provenance", "dense"),
                }
                for r in self.results
            ],
            "coverage": {
                "band_energies": self.coverage.band_energies.tolist(),
                "band_names": self.coverage.band_names,
                "total_energy": round(float(self.coverage.total_energy), 4),
                "confidence": round(float(self.coverage.confidence), 4),
                "gaps": self.coverage.gaps,
            },
            "related": [
                {
                    "source_id": rt.source_id,
                    "score": round(float(rt.score), 4),
                    "hop": rt.hop,
                    "summary": rt.content.summary if rt.content else None,
                }
                for rt in self.related
            ],
            "contradictions": [
                {
                    "source_a": c.source_a,
                    "source_b": c.source_b,
                    "interference": round(float(c.interference), 4),
                    "summary_a": c.summary_a,
                    "summary_b": c.summary_b,
                }
                for c in self.contradictions
            ],
            "timings_ms": {
                name: round(float(value), 2)
                for name, value in self.timings_ms.items()
            },
        }
        if self.assessment is not None:
            d["assessment"] = self.assessment
        return d

    def to_prompt(self) -> str:
        """Format as LLM-injectable text with confidence, coverage bars, band scores."""
        lines = []

        # Header with confidence
        lines.append(f"## Resonance Results (confidence: {self.coverage.confidence:.0%})")
        lines.append("")

        # Coverage bars
        max_energy = max(self.coverage.band_energies.max(), 1e-8)
        for i, (name, energy) in enumerate(zip(self.coverage.band_names, self.coverage.band_energies)):
            bar_len = int(20 * energy / max_energy)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {name:<12} {bar} {energy:.2f}")
        lines.append("")

        if self.coverage.gaps:
            lines.append(f"**Knowledge gaps**: {', '.join(self.coverage.gaps)}")
            lines.append("")

        # Passages
        lines.append("### Passages")
        for i, r in enumerate(self.results, 1):
            text = ""
            if r.content:
                text = r.content.full_text or r.content.summary or ""
            band_str = ""
            if r.band_scores is not None:
                band_str = f" bands=[{', '.join(f'{s:.2f}' for s in r.band_scores)}]"
            score_str = f"{r.score:.3f}"
            if r.raw_score is not None:
                score_str += f", raw={r.raw_score:.3f}"
            display_name = _result_display_name(r)
            lines.append(f"[{i}] ({score_str}{band_str}) {display_name}")
            if text:
                lines.append(f"    {_truncate_text(text, 500)}")
            lines.append("")

        # Related topics
        if self.related:
            lines.append("### Related Topics (via cascade)")
            for rt in self.related:
                summary = rt.content.summary if rt.content else ""
                lines.append(f"  - (hop {rt.hop}, {rt.score:.3f}) {rt.source_id}: {summary[:200]}")
            lines.append("")

        # Contradictions
        if self.contradictions:
            lines.append("### Contradictions Detected")
            for c in self.contradictions:
                lines.append(f"  ⚡ {c.source_a} vs {c.source_b} (interference: {c.interference:.3f})")
                if c.summary_a:
                    lines.append(f"    A: {c.summary_a[:150]}")
                if c.summary_b:
                    lines.append(f"    B: {c.summary_b[:150]}")
            lines.append("")

        return "\n".join(lines)


class RetrievalResult(NamedTuple):
    """Complete result of a resonance retrieval."""
    results: list[MaterialisedResult]
    resonance: ResonanceResult
    source_pointers: list[SourcePointer]
    timings_ms: dict[str, float] | None = None


def _result_display_name(result) -> str:
    """Human-readable name for a result: file stem / heading, falling back to source_id."""
    if hasattr(result, "content") and result.content and result.content.metadata:
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


def _truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text at a word boundary with ellipsis marker."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind(" ")
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut] + "..."


@dataclass
class MaterialisedResult:
    """A single materialised result with content and scoring.

    `score` is the primary ranking/display score for the current result set.
    `raw_score` preserves the underlying retrieval score when a later rerank
    stage changes the displayed order.
    `provenance` records how the candidate was found: "dense", "lexical",
    or "dense+lexical" (both paths found it).
    `knowledge model` records which .rlat knowledge model this result came from
    (empty string for single-knowledge model queries, populated by ComposedCartridge).
    """
    source_id: str
    score: float
    band_scores: NDArray[np.float32] | None
    content: SourceContent | None
    raw_score: float | None = None
    provenance: str = "dense"
    cartridge: str = ""


def _create_field(config: LatticeConfig) -> FieldBackend:
    """Factory: create the appropriate field backend from config."""
    if config.field_type == FieldType.DENSE:
        return DenseField(bands=config.bands, dim=config.dim)
    elif config.field_type == FieldType.ASYMMETRIC_DENSE:
        dk = config.dim_key if config.dim_key is not None else config.dim
        dv = config.dim_value if config.dim_value is not None else config.dim
        return AsymmetricDenseField(bands=config.bands, dim_key=dk, dim_value=dv)
    elif config.field_type == FieldType.FACTORED:
        return FactoredField(
            bands=config.bands,
            dim=config.dim,
            rank=config.svd_rank,
        )
    elif config.field_type == FieldType.PQ:
        return PQField(
            bands=config.bands,
            dim=config.dim,
            num_subspaces=config.pq_subspaces,
            codebook_size=config.pq_codebook_size,
        )
    else:
        raise ValueError(f"Unknown field type: {config.field_type}")


@dataclass
class Config:
    """Simple configuration for the production SDK.

    Usage:
        lattice = Lattice(Config(bands=5, dim=2048, backbone="e5-large-v2"))

    For edge deployment (CPU-only, minimal memory):
        lattice = Lattice(Config(bands=5, dim=2048))  # random encoder, no backbone download
    """
    bands: int = 5
    dim: int = 2048
    backbone: str = "e5-large-v2"
    field_type: str = "dense"
    device: str = "cpu"

    def to_lattice_config(self) -> LatticeConfig:
        """Convert to the internal LatticeConfig."""
        return LatticeConfig(
            bands=self.bands,
            dim=self.dim,
            field_type=FieldType(self.field_type),
        )


class Lattice:
    """Main Resonance Lattice orchestrator.

    Ties together:
      - Field backend (Dense, Factored, or PQ)
      - PhaseRegistry (LSH-based source lookup)
      - SourceStore (multi-resolution content)
      - Encoder (optional, for text-based encode/query)

    Production SDK usage:
        lattice = Lattice(Config(bands=5, dim=2048, backbone="e5-large-v2"))
        lattice.add("doc_001", "Text content here...")
        results = lattice.query("How does X work?", top_k=10)
        lattice.save("my_kb.rlat")
    """

    def __init__(
        self,
        config: LatticeConfig | Config,
        store_path: str | Path = ":memory:",
        encoder: Encoder | None = None,
    ) -> None:
        # Accept either Config (SDK) or LatticeConfig (internal)
        if isinstance(config, Config):
            self._sdk_config = config
            lattice_config = config.to_lattice_config()
            # Auto-create encoder from Config
            if encoder is None:
                backbone = config.backbone
                if backbone in ("random", "none", ""):
                    encoder = Encoder.random(
                        bands=config.bands,
                        dim=config.dim,
                    )
                else:
                    # Resolve short names to full HuggingFace IDs
                    backbone_map = {
                        "e5-large-v2": "intfloat/e5-large-v2",
                        "e5-base-v2": "intfloat/e5-base-v2",
                        "e5-small-v2": "intfloat/e5-small-v2",
                    }
                    full_name = backbone_map.get(backbone, backbone)
                    encoder = Encoder.from_backbone(
                        model_name=full_name,
                        bands=config.bands,
                        dim=config.dim,
                        device=config.device,
                    )
        else:
            self._sdk_config = None
            lattice_config = config

        self.config = lattice_config
        self.field: FieldBackend = _create_field(lattice_config)
        self.registry = PhaseRegistry(
            dim=lattice_config.dim,
            bands=lattice_config.bands,
        )
        self.store = SourceStore(path=store_path)
        self.encoder = encoder

        # Cache of phase vectors for removal (source_id -> (phase, salience))
        self._phase_cache: dict[str, tuple[NDArray[np.float32], float]] = {}

        # Corpus-level IDF weights for keyword reranking (lazy-loaded from __idf__)
        self._idf: dict[str, float] | None = None

        # Dense backbone index (None until explicitly built; PR #71 rolled back)
        self.dense_index = None

        # Salience reinforcement: touch() hit sources on retrieval.
        # Off by default to preserve query determinism for benchmarks.
        self.reinforce_on_hit: bool = False
        self._hit_boost: float = 1.2

    @property
    def source_count(self) -> int:
        return self.field.source_count

    @property
    def field_type(self) -> FieldType:
        return self.config.field_type

    def superpose(
        self,
        phase_spectrum: PhaseSpectrum | NDArray[np.float32],
        salience: float = 1.0,
        source_id: str = "",
        content: SourceContent | dict[str, Any] | None = None,
    ) -> str:
        """Add a source to the lattice.

        Performs three operations atomically:
          1. Rank-1 outer product update to the field tensor
          2. Register in the phase registry for lookup
          3. Store content for materialisation

        Args:
            phase_spectrum: PhaseSpectrum or raw (B, D) array.
            salience: Importance weight alpha_i.
            source_id: Unique identifier. Auto-generated if empty.
            content: Source content for materialisation.

        Returns:
            The source_id used.
        """
        # Normalise inputs
        if isinstance(phase_spectrum, PhaseSpectrum):
            vectors = phase_spectrum.vectors
        else:
            vectors = phase_spectrum

        if not source_id:
            source_id = f"src_{self.source_count:08d}"

        # Idempotent upsert: if a source with this id already exists, remove
        # the old one before adding the new. Without this, the field tensor's
        # ``_source_count`` is double-incremented (once for the original, once
        # for the duplicate) while the registry and phase cache keep a single
        # entry — silently desyncing ``field.source_count`` from the live set.
        # Symptom in the wild: after bulk ingest with colliding ids,
        # ``rlat memory gc`` reports evictions but can't bring the tier down
        # to policy capacity because the count it reads never matches the
        # entries it can actually remove (DOGFOOD_FINDINGS #316 Bug B).
        if source_id in self._phase_cache:
            self.remove(source_id)

        # 1. Update field
        if isinstance(self.field, AsymmetricDenseField):
            # Asymmetric: pass vectors as both key and value (degenerate mode)
            # When trained key/value heads are available, this path will
            # receive separate key_vectors and value_vectors instead.
            self.field.superpose(vectors, vectors, salience=salience)
        else:
            self.field.superpose(vectors, salience=salience)

        # 2. Register in phase registry
        self.registry.register(source_id, vectors, salience=salience)

        # 3. Store content
        if content is not None:
            if isinstance(content, dict):
                sc = SourceContent(
                    source_id=source_id,
                    summary=content.get("summary", ""),
                    relations=[tuple(r) for r in content.get("relations", [])],
                    full_text=content.get("full_text", ""),
                    metadata=content.get("metadata", {}),
                )
            else:
                sc = content
                sc.source_id = source_id
            self.store.store(sc)

        # Cache for removal
        self._phase_cache[source_id] = (vectors.copy(), salience)

        return source_id

    def superpose_text(
        self,
        text: str,
        salience: float = 1.0,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> str:
        """Encode text and add to the lattice (requires encoder).

        Args:
            text: Input text to encode and store.
            salience: Importance weight.
            source_id: Unique identifier. Auto-generated if empty.
            metadata: Optional metadata dict.
            summary: Short extractive summary. Defaults to full text.

        Returns:
            The source_id used.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder configured. Use superpose() with pre-computed phase vectors.")

        phase = self.encoder.encode_passage(text)

        content = SourceContent(
            source_id=source_id or f"src_{self.source_count:08d}",
            summary=summary or text,
            full_text=text,
            metadata=metadata or {},
        )

        return self.superpose(
            phase_spectrum=phase,
            salience=salience,
            source_id=source_id,
            content=content,
        )

    def superpose_text_batch(
        self,
        texts: list[str],
        source_ids: list[str],
        saliences: list[float] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        summaries: list[str] | None = None,
        batch_size: int = 64,
        encode_texts: list[str] | None = None,
    ) -> list[str]:
        """Encode and superpose a batch of texts in one shot.

        Much faster than calling superpose_text() in a loop because the
        backbone forward pass is batched (better GPU utilisation).

        Args:
            texts: Passage texts to store as full_text (clean, for display).
            source_ids: Unique IDs, one per text.
            saliences: Importance weights (default all 1.0).
            metadatas: Optional metadata dicts, one per text.
            summaries: Short extractive summaries. Defaults to full text.
            batch_size: Backbone batch size (default 64).
            encode_texts: If provided, these are encoded instead of texts.
                Use for contextual chunking: encode_texts has structural
                prefixes (file path, heading, position) while texts has
                the clean original text for display and keyword matching.

        Returns:
            List of source_ids used.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder configured.")
        if len(texts) != len(source_ids):
            raise ValueError("texts and source_ids must have the same length")

        if saliences is None:
            saliences = [1.0] * len(texts)
        if metadatas is None:
            metadatas = [{}] * len(texts)
        if summaries is None:
            summaries = texts

        # Idempotent upsert: remove any source_ids that already exist before
        # we add their replacements. Mirrors the single-call guard in
        # ``superpose``; prevents the field / registry desync on duplicate
        # id batches (see DOGFOOD_FINDINGS #316 Bug B). Duplicates within
        # *this* batch are handled in the register loop below — the latest
        # wins since ``_phase_cache[sid] = (...)`` overwrites.
        for sid in source_ids:
            if sid and sid in self._phase_cache:
                self.remove(sid)

        # Encode contextual texts if provided, otherwise encode raw texts
        to_encode = encode_texts if encode_texts is not None else texts

        # Batch encode through backbone + projection heads
        spectra = self.encoder.encode_passages(to_encode, batch_size=batch_size)

        # Stack phase vectors for batched field update
        phase_arrays = [
            s.vectors if isinstance(s, PhaseSpectrum) else s for s in spectra
        ]
        phase_batch = np.stack(phase_arrays)  # (N, B, D)
        sal_array = np.array(saliences, dtype=np.float32)

        # 1. Batched field update — use BLAS batch when available, else loop
        if isinstance(self.field, AsymmetricDenseField):
            # Asymmetric degenerate: key == value
            if hasattr(self.field, 'superpose_batch'):
                self.field.superpose_batch(phase_batch, phase_batch, sal_array)
            else:
                for vectors, sal in zip(phase_arrays, saliences):
                    self.field.superpose(vectors, vectors, salience=sal)
        elif hasattr(self.field, 'superpose_batch'):
            self.field.superpose_batch(phase_batch, sal_array)
        else:
            for vectors, sal in zip(phase_arrays, saliences):
                self.field.superpose(vectors, salience=sal)

        # 2. Register each in phase registry (in-memory, fast)
        contents = []
        used_ids = []
        for vectors, sid, sal, meta, text, summ in zip(
            phase_arrays, source_ids, saliences, metadatas, texts, summaries
        ):
            self.registry.register(sid, vectors, salience=sal)
            self._phase_cache[sid] = (vectors.copy(), sal)
            contents.append(SourceContent(
                source_id=sid,
                summary=summ,
                full_text=text,
                metadata=meta,
            ))
            used_ids.append(sid)

        # 3. Batched store (single transaction)
        self.store.store_batch(contents)

        return used_ids

    def remove(self, source_id: str) -> bool:
        """Remove a source from the lattice.

        Performs the algebraically exact reverse of superpose:
          1. Rank-1 subtraction from the field tensor
          2. Unregister from the phase registry
          3. Remove from the source store

        Args:
            source_id: The source to remove.

        Returns:
            True if the source existed and was removed.
        """
        if source_id not in self._phase_cache:
            return False

        vectors, salience = self._phase_cache.pop(source_id)

        # 1. Field subtraction
        if isinstance(self.field, AsymmetricDenseField):
            self.field.remove(vectors, vectors, salience=salience)
        else:
            self.field.remove(vectors, salience=salience)

        # 2. Registry removal
        self.registry.unregister(source_id)

        # 3. Store removal
        self.store.remove(source_id)

        return True

    def update(
        self,
        source_id: str,
        new_phase: PhaseSpectrum | NDArray[np.float32],
        new_salience: float = 1.0,
        new_content: SourceContent | dict[str, Any] | None = None,
    ) -> bool:
        """Update a source (remove old + add new).

        Args:
            source_id: The source to update.
            new_phase: New phase spectrum.
            new_salience: New salience weight.
            new_content: New content (optional).

        Returns:
            True if the source existed and was updated.
        """
        if not self.remove(source_id):
            return False
        self.superpose(new_phase, new_salience, source_id, new_content)
        return True

    def reweight(self, source_id: str, new_salience: float) -> bool:
        """Change a source's salience weight without altering its phase vectors.

        Algebraically equivalent to remove(source_id) + superpose(same_phase, new_salience).
        Used by retention enforcement and salience reinforcement on hit.

        Args:
            source_id: The source to reweight.
            new_salience: New salience value (must be > 0).

        Returns:
            True if the source existed and was reweighted.
        """
        if source_id not in self._phase_cache:
            return False

        vectors, old_salience = self._phase_cache[source_id]
        if abs(new_salience - old_salience) < 1e-9:
            return True  # no-op

        content = self.store.retrieve(source_id)
        return self.update(source_id, vectors, new_salience, content)

    # ── Corpus-level IDF for keyword reranking ──────────────────────────────

    def build_idf(self) -> dict[str, float]:
        """Compute corpus-level IDF weights from stored content.

        Scans all source content, tokenizes, and computes
        IDF = log(N / (1 + df)) for each token. Stores the result
        as ``__idf__`` in the knowledge model for persistence.

        Only supported on embedded-store knowledge models — the SQL path below
        needs the in-knowledge model ``sources`` table. External knowledge models
        short-circuit to an empty IDF and let the B3 ripgrep hybrid carry
        the lexical signal (see ``_inject_lexical_matches`` docstring for
        the broader deprecation context).

        Returns:
            Dict mapping token -> IDF weight.
        """
        import math

        doc_freq: dict[str, int] = {}
        n_docs = 0

        # LocalStore has no SQLite connection — text lives on disk, not
        # in the cartridge. IDF computation currently requires an in-cartridge
        # text index; external cartridges get no IDF until a proper external
        # text index lands. Return empty dict so callers degrade gracefully.
        if not hasattr(self.store, "_conn"):
            self._idf = {}
            return self._idf

        cursor = self.store._conn.execute(
            "SELECT source_id, full_text, summary FROM sources "
            "WHERE source_id NOT LIKE '\\_\\_%' ESCAPE '\\'",
        )
        for sid, full_text, summary in cursor:
            text = full_text or summary or ""
            if not text:
                continue
            tokens = self._tokenize_for_match(text)
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1
            n_docs += 1

        if n_docs == 0:
            self._idf = {}
            return self._idf

        idf = {t: math.log(n_docs / (1 + df)) for t, df in doc_freq.items()}
        self._idf = idf

        # Persist in cartridge
        import json
        self.store.remove("__idf__")
        self.store.store(SourceContent(
            source_id="__idf__",
            summary=f"corpus IDF weights ({len(idf)} tokens, {n_docs} docs)",
            full_text=json.dumps(idf),
            metadata={"n_tokens": len(idf), "n_docs": n_docs},
        ))

        return idf

    def _get_idf(self) -> dict[str, float]:
        """Return cached IDF weights, loading from store if needed."""
        if self._idf is not None:
            return self._idf

        # Try loading from cartridge
        content = self.store.retrieve("__idf__")
        if content and content.full_text:
            import json
            try:
                self._idf = json.loads(content.full_text)
                return self._idf
            except (json.JSONDecodeError, TypeError):
                pass

        # No IDF available — return empty (falls back to unweighted)
        return {}

    def _idf_weighted_overlap(
        self,
        query_tokens: set[str],
        doc_tokens: set[str],
    ) -> float:
        """Compute IDF-weighted token overlap between query and document.

        Falls back to unweighted Jaccard-style overlap if no IDF is available.
        """
        if not query_tokens:
            return 0.0

        idf = self._get_idf()
        if not idf:
            # No IDF — unweighted overlap (original behavior)
            return len(query_tokens & doc_tokens) / len(query_tokens)

        # IDF-weighted: rare query tokens contribute more
        matched = query_tokens & doc_tokens
        matched_weight = sum(idf.get(t, 1.0) for t in matched)
        total_weight = sum(idf.get(t, 1.0) for t in query_tokens)
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def resonate(
        self,
        query_phase: PhaseSpectrum | NDArray[np.float32],
        band_weights: NDArray[np.float32] | None = None,
        top_k: int = 20,
        use_lsh: bool = False,
        retrieval_mode: str = "auto",
        resonance_alpha: float = 1.0,
        resonance_mode: str = "normalize",
        scoring_fn: str = "linear",
    ) -> RetrievalResult:
        """Project a query into the field and retrieve matching sources.

        Steps:
          1. Resonance projection: F_b @ q_b per band
          2. Source identification via selected retrieval mode
          3. Content retrieval: materialise from source store

        Retrieval modes:
          - "field": Score via field tensor only (fast, O(D²), good for N < 10K)
          - "registry": Score via exact phase cosine (O(N), best quality at scale)
          - "hybrid": Field generates candidates, registry re-scores (best of both)
          - "auto": Picks field for N < 10K, registry for N >= 10K

        Args:
            query_phase: PhaseSpectrum or raw (B, D) array.
            band_weights: Per-band weights for fusion. Uniform if None.
            top_k: Number of results to return.
            use_lsh: If True, use LSH for approximate lookup (faster but lower recall).
                If False (default), use brute-force exact lookup.
            retrieval_mode: "auto", "field", "registry", or "hybrid".
            resonance_alpha: Blend weight for resonance-guided scoring in [0, 1].
                1.0 (default) = pure sparse dot-product scoring (original behavior).
                0.0 = pure resonance-guided scoring (phases scored against dense
                field resonance vectors instead of sparse query).
                Values between blend both signals.
            scoring_fn: "linear" (default). Separate DEPRECATE verdict applies
                to this parameter per DECISIONS.md v1.4 — kept for now as a
                no-op router signal to registry.py's scoring_fn='eml' branch.

        Returns:
            RetrievalResult with materialised results, raw resonance, and pointers.
        """
        t_total_start = _time.perf_counter()

        if isinstance(query_phase, PhaseSpectrum):
            q_vectors = query_phase.vectors
        else:
            q_vectors = query_phase

        # 1. Resonance projection (always computed — needed for band diagnostics)
        # EML fusion path removed in v1.5 Phase 6 Pass 4b — eml_cluster REMOVE
        # verdict (falsified 3× incl. NaN overflow). field.resonate_eml() kept
        # as library primitive but unreachable from this flow.
        t_resonate_start = _time.perf_counter()
        raw_resonance = self.field.resonate(q_vectors, band_weights=band_weights)
        # Normalize asymmetric result to standard ResonanceResult for downstream compat
        if isinstance(raw_resonance, ResonanceResult):
            resonance = raw_resonance
        else:
            # AsymmetricResonanceResult → ResonanceResult (works in degenerate mode
            # where D_key == D_value; for true asymmetric, value-space resonance
            # vectors serve the same diagnostic role as symmetric ones)
            resonance = ResonanceResult(
                resonance_vectors=raw_resonance.resonance_vectors,
                fused=raw_resonance.fused,
                band_energies=raw_resonance.band_energies,
            )
        resonate_ms = (_time.perf_counter() - t_resonate_start) * 1000

        # 2. Select retrieval mode
        mode = retrieval_mode
        if mode == "auto":
            # Use ANN when available, otherwise field for small or registry for large
            if self.registry.has_ann:
                mode = "ann"
            elif self.source_count < 10_000:
                mode = "field"
            else:
                mode = "registry"

        # Resonance-guided scoring: pass field's dense resonance vectors
        # to the registry so it can blend sparse x sparse with sparse x dense.
        _res_vecs = resonance.resonance_vectors if resonance_alpha < 1.0 else None

        # 3. Source identification
        t_registry_start = _time.perf_counter()
        if mode == "ann":
            pointers = self.registry.lookup_ann(
                query_phase=q_vectors,
                top_k=top_k,
                band_weights=band_weights,
                resonance_vectors=_res_vecs,
                resonance_alpha=resonance_alpha,
                resonance_mode=resonance_mode,
            )

        elif mode == "field":
            if use_lsh:
                pointers = self.registry.lookup(
                    resonance_vector=resonance.fused,
                    top_k=top_k,
                    query_phase=q_vectors,
                )
            else:
                pointers = self.registry.lookup_bruteforce(
                    query_phase=q_vectors,
                    top_k=top_k,
                    band_weights=band_weights,
                    resonance_vectors=_res_vecs,
                    resonance_alpha=resonance_alpha,
                    resonance_mode=resonance_mode,
                    scoring_fn=scoring_fn,
                )

        elif mode == "registry":
            # Direct phase cosine scoring (best quality at scale)
            pointers = self.registry.lookup_bruteforce(
                query_phase=q_vectors,
                top_k=top_k,
                band_weights=band_weights,
                resonance_vectors=_res_vecs,
                resonance_alpha=resonance_alpha,
                resonance_mode=resonance_mode,
                scoring_fn=scoring_fn,
            )

        elif mode == "hybrid":
            # Field generates broad candidate set, registry re-scores
            candidate_k = min(top_k * 10, self.source_count)
            if use_lsh:
                candidates = self.registry.lookup(
                    resonance_vector=resonance.fused,
                    top_k=candidate_k,
                    query_phase=q_vectors,
                )
            else:
                # Use field resonance to score all sources, take top candidate_k
                candidates = self._field_rank_sources(
                    q_vectors, resonance, candidate_k,
                )
            # Re-score candidates with exact phase cosine
            pointers = self._rerank_candidates(
                candidates, q_vectors, top_k,
                band_weights=band_weights,
            )

        else:
            raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

        registry_ms = (_time.perf_counter() - t_registry_start) * 1000

        # 4. Materialise content
        t_store_start = _time.perf_counter()
        materialised = []
        source_ids = [p.source_id for p in pointers]
        contents = {c.source_id: c for c in self.store.retrieve_batch(source_ids)}

        for ptr in pointers:
            materialised.append(MaterialisedResult(
                source_id=ptr.source_id,
                score=ptr.fidelity_score,
                band_scores=ptr.band_signature,
                content=contents.get(ptr.source_id),
            ))
        store_ms = (_time.perf_counter() - t_store_start) * 1000

        # 5. Salience reinforcement (opt-in for memory tiers)
        if self.reinforce_on_hit:
            import time as _tmod
            now = _tmod.time()
            for ptr in pointers:
                self.registry.touch(ptr.source_id, now)
                entry = self.registry._source_index.get(ptr.source_id)
                if entry is not None:
                    boosted = min(1.0, entry.salience * self._hit_boost)
                    if abs(boosted - entry.salience) > 1e-6:
                        self.reweight(ptr.source_id, boosted)

        timings_ms = {
            "resonate": resonate_ms,
            "registry": registry_ms,
            "store": store_ms,
            "total": (_time.perf_counter() - t_total_start) * 1000,
        }

        return RetrievalResult(
            results=materialised,
            resonance=resonance,
            source_pointers=pointers,
            timings_ms=timings_ms,
        )

    def _field_rank_sources(
        self,
        query_phase: NDArray[np.float32],
        resonance: ResonanceResult,
        top_k: int,
    ) -> list[SourcePointer]:
        """Rank sources using field resonance vectors (approximate, O(N*D) per band).

        Scores each registered source by dot product with the field's
        resonance vector (not the raw query). This leverages the field's
        corpus-aware reweighting.
        """
        if not self.registry._source_index:
            return []

        if self.registry._cache_dirty:
            self.registry._rebuild_cache()

        source_ids = self.registry._cached_ids
        n = len(source_ids)
        bands = query_phase.shape[0]

        # Score via resonance vectors (field-weighted)
        phases_3d = self.registry._cached_phases.reshape(n, bands, -1)
        scores = np.zeros(n, dtype=np.float32)
        for b in range(bands):
            r_b = resonance.resonance_vectors[b]
            r_norm = r_b / (np.linalg.norm(r_b) + 1e-8)
            scores += phases_3d[:, b, :] @ r_norm

        scores *= self.registry._cached_saliences

        if n > top_k:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        return [
            SourcePointer(
                source_id=source_ids[i],
                fidelity_score=float(scores[i]),
                band_signature=np.array([
                    float(np.dot(phases_3d[i, b], query_phase[b]))
                    for b in range(bands)
                ], dtype=np.float32),
            )
            for i in top_indices[:top_k]
        ]

    def _rerank_candidates(
        self,
        candidates: list[SourcePointer],
        query_phase: NDArray[np.float32],
        top_k: int,
        band_weights: NDArray[np.float32] | None = None,
    ) -> list[SourcePointer]:
        """Re-score candidates with exact phase cosine similarity."""
        if not candidates:
            return []

        bands = query_phase.shape[0]
        reranked = []

        for ptr in candidates:
            if ptr.source_id not in self.registry._source_index:
                continue
            entry = self.registry._source_index[ptr.source_id]
            phase = entry.phase_vectors  # (B, D)

            band_scores = np.zeros(bands, dtype=np.float32)
            for b in range(bands):
                band_scores[b] = float(np.dot(phase[b], query_phase[b]))

            if band_weights is not None:
                exact_score = float(np.dot(band_scores, band_weights))
            else:
                exact_score = float(band_scores.sum())
            exact_score *= entry.salience

            reranked.append(SourcePointer(
                source_id=ptr.source_id,
                fidelity_score=exact_score,
                band_signature=band_scores,
            ))

        reranked.sort(key=lambda p: p.fidelity_score, reverse=True)
        return reranked[:top_k]

    def resonate_text(
        self,
        query: str,
        band_weights: NDArray[np.float32] | None = None,
        top_k: int = 20,
        use_lsh: bool = False,
        retrieval_mode: str = "auto",
        scoring_fn: str = "linear",
    ) -> RetrievalResult:
        """Encode a text query and resonate (requires encoder).

        Args:
            query: Query text string.
            band_weights: Per-band weights for fusion.
            top_k: Number of results to return.
            use_lsh: Use LSH approximate lookup instead of brute-force.
            retrieval_mode: "auto", "field", "registry", or "hybrid".
            scoring_fn: "linear" (default). See resonate() docstring for scope.

        Returns:
            RetrievalResult.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder configured. Use resonate() with pre-computed phase vectors.")

        phase = self.encoder.encode_query(query)
        return self.resonate(
            phase, band_weights=band_weights, top_k=top_k,
            use_lsh=use_lsh, retrieval_mode=retrieval_mode,
            scoring_fn=scoring_fn,
        )

    def compute_novelty(
        self,
        phase_spectrum: PhaseSpectrum | NDArray[np.float32],
    ) -> float:
        """Check how novel a source is relative to the current field state.

        novelty = 1 - max_b(max(F_b @ phi_b) / norm(phi_b))

        High novelty (~1.0) means the field doesn't "know" this information.
        Low novelty (~0.0) means it's redundant with existing sources.

        Args:
            phase_spectrum: The candidate source's phase spectrum.

        Returns:
            Novelty score in [0, 1].
        """
        if isinstance(phase_spectrum, PhaseSpectrum):
            vectors = phase_spectrum.vectors
        else:
            vectors = phase_spectrum

        resonance = self.field.resonate(vectors)

        max_similarity = 0.0
        for b in range(self.config.bands):
            r_b = resonance.resonance_vectors[b]
            phi_b = vectors[b]
            norm_phi = np.linalg.norm(phi_b)
            if norm_phi > 1e-8:
                similarity = np.max(np.abs(r_b)) / norm_phi
                max_similarity = max(max_similarity, similarity)

        return float(max(0.0, min(1.0, 1.0 - max_similarity)))

    def save(self, path: str | Path, registry_quantize: int = 0,
             store_mode: str = "embedded") -> None:
        """Save the lattice to a .rlat file.

        Args:
            path: Output file path.
            registry_quantize: Bits per value for registry phase quantization.
                0 = full float32 (default, backwards compatible).
                4 = 4-bit quantization (~87% registry compression).
                8 = 8-bit quantization (~50% compression, higher quality).
            store_mode: Serving topology for source content.
                - "embedded" (legacy, deprecated v2.0.0) — pre-chunked text
                  in an in-knowledge model SQLite database.
                - "external" (default in the CLI) — manifest + metadata
                  only; files resolved from disk at query time.
                - "bundled" (v3+) — lossless: raw source files packed into
                  the knowledge model as zstd frames. Same retrieval semantics
                  as external mode but self-contained.
        """
        path = Path(path)

        # Canonical "local" is a synonym for the historical "external"
        # spelling — same wire value, same behaviour. Normalise up front.
        if store_mode == "local":
            store_mode = "external"

        # External, bundled, and remote all build a manifest + minimal
        # metadata store and omit the per-chunk SQLite rows. Bundled
        # additionally packs the raw source files inside the cartridge;
        # remote adds a __remote_origin__ record so load() can
        # reconstruct the fetcher.
        is_external = store_mode == "external"
        is_bundled = store_mode == "bundled"
        is_remote = store_mode == "remote"
        use_manifest = is_external or is_bundled or is_remote

        # Store encoder config so the cartridge is self-describing.
        # External and bundled modes both use a fresh metadata SourceStore;
        # per-chunk text rows never enter the cartridge.
        meta_store = SourceStore() if use_manifest else self.store
        if self.encoder is not None:
            import base64 as _b64
            import json
            encoder_meta = self.encoder.get_config()
            # Embed head weights so the cartridge is fully self-contained
            if hasattr(self.encoder, "heads") and len(self.encoder.heads) > 0:
                weights_blob = self.encoder.get_head_weights_blob(precision="f16")
                encoder_meta["has_embedded_weights"] = True
                meta_store.remove("__encoder_weights__")
                meta_store.store(SourceContent(
                    source_id="__encoder_weights__",
                    summary=f"projection head weights ({len(weights_blob)} bytes, f16+zlib)",
                    full_text=_b64.b64encode(weights_blob).decode("ascii"),
                    metadata={"format": "rlhw_v1", "size_bytes": len(weights_blob)},
                ))
            meta_store.remove("__encoder__")
            meta_store.store(SourceContent(
                source_id="__encoder__",
                summary="encoder configuration",
                full_text=json.dumps(encoder_meta),
                metadata=encoder_meta,
            ))

        # Build and store corpus-level IDF for keyword reranking
        if self.source_count > 0 and not use_manifest:
            try:
                self.build_idf()
            except Exception:
                pass  # IDF is optional — don't fail the save

        # 236c: copy build-time retrieval-mode probe result into the
        # metadata store. In external/bundled/remote mode meta_store is a
        # fresh SourceStore, so __retrieval_config__ written by
        # `rlat build --probe-qrels` to self.store needs to be migrated
        # explicitly — same pattern as __encoder__. Remote mode also
        # carries __remote_origin__; reload-and-resave flows (rlat
        # repoint, rlat sync) rely on __source_manifest__ coming along
        # so the existing manifest survives, since the manifest-rebuild
        # block below skips when self.store is already a LocalStore.
        if use_manifest:
            reserved_ids = [
                "__retrieval_config__",
                "__remote_origin__",
                "__source_manifest__",
                "__profile__",
            ]
            # Preserve __encoder__ / __encoder_weights__ from the source
            # store when no live encoder is attached (e.g. repoint + save
            # cycles that loaded with restore_encoder=False). Guarded so
            # a live encoder's fresh config above isn't overwritten.
            if self.encoder is None:
                reserved_ids.extend(["__encoder__", "__encoder_weights__"])
            for reserved_id in reserved_ids:
                cfg = self.store.retrieve(reserved_id)
                if cfg is not None:
                    meta_store.remove(reserved_id)
                    meta_store.store(cfg)

        # Store semantic profile if DenseField
        if isinstance(self.field, DenseField) and self.source_count > 0:
            import json as _json
            try:
                prof = self.semantic_profile()
                meta_store.remove("__profile__")
                meta_store.store(SourceContent(
                    source_id="__profile__",
                    summary="semantic profile",
                    full_text=_json.dumps(prof),
                    metadata=prof,
                ))
            except Exception:
                pass  # Profile is best-effort, don't block save

        # External and bundled mode both build a source_manifest mapping
        # source_id -> relative path + drift-check metadata. For bundled,
        # the same manifest also drives which files get packed into the
        # blob store. Paths are normalized to be relative-to-source-root
        # wherever possible so the cartridge travels between machines.
        # `__source_root_hint__` records the build-time absolute root for
        # auto-resolution in external mode (bundled doesn't need it).
        manifest: dict[str, str | dict] = {}
        source_root: str | None = None
        if use_manifest and not isinstance(self.store, LocalStore):
            import json as _json2

            # Pass 1: collect raw paths per source_id to compute a common root.
            raw_entries: dict[str, dict] = {}
            for sid in self.store.all_ids():
                if sid.startswith("__"):
                    continue
                sc = self.store.retrieve(sid)
                if sc and sc.metadata.get("source_file"):
                    entry: dict = {
                        "source_file": sc.metadata["source_file"],
                        "heading": sc.metadata.get("heading", ""),
                    }
                    tail = sid.rsplit("_", 1)
                    if len(tail) == 2 and tail[1].isdigit():
                        entry["chunk_index"] = int(tail[1])
                    # A3: carry char_offset + content_hash when the build
                    # pipeline provided them. Older cartridges (built before
                    # A3 landed) have no such metadata — the keys simply
                    # won't appear in the manifest and readers should
                    # tolerate their absence.
                    if "char_offset" in sc.metadata:
                        entry["char_offset"] = sc.metadata["char_offset"]
                    if "content_hash" in sc.metadata:
                        entry["content_hash"] = sc.metadata["content_hash"]
                    raw_entries[sid] = entry

            source_root = _compute_source_root(
                [e["source_file"] for e in raw_entries.values()]
            )

            # Pass 2: emit the manifest. Paths under source_root become
            # relative + posix (so Windows-built cartridges load on Linux
            # and vice versa). Paths outside source_root stay absolute —
            # LocalStore._resolve_path falls back to absolute lookup.
            if source_root:
                manifest[_MANIFEST_SOURCE_ROOT_KEY] = source_root
            for sid, entry in raw_entries.items():
                sf = entry["source_file"]
                if source_root:
                    try:
                        abs_sf = str(Path(sf).resolve())
                        rel = os.path.relpath(abs_sf, source_root)
                        # os.path.relpath returns ".." paths for siblings
                        # of source_root; only normalise when it stays under.
                        if not rel.startswith(".."):
                            entry = {**entry, "source_file": Path(rel).as_posix()}
                    except (OSError, ValueError):
                        pass  # keep path as-recorded
                manifest[sid] = entry

            if manifest:
                meta_store.store(SourceContent(
                    source_id="__source_manifest__",
                    summary="source_id to file path and metadata mapping",
                    full_text=_json2.dumps(manifest),
                ))

        # Serialise registry and store
        registry_data = self.registry.to_bytes(quantize=registry_quantize) if self.registry.source_count > 0 else b""
        store_data = meta_store.to_bytes() if meta_store.count > 0 else b""

        # Reload-and-repoint-to-bundled path: when self.store is
        # already a LocalStore (reloaded cartridge), the manifest-rebuild
        # block above was skipped. Pull manifest + source_root directly
        # off the LocalStore so the bundled-pack block below fires.
        if is_bundled and isinstance(self.store, LocalStore) and not manifest:
            manifest = dict(getattr(self.store, "_manifest", {}) or {})
            source_root = str(self.store.source_root)

        # Bundled mode: wrap the metadata SQLite and the packed source
        # files into a single source_store section. The blob payload
        # sniffs as "RLBD" so the loader can tell bundled from legacy
        # embedded / external SQLite without inspecting header flags.
        if is_bundled and source_root and manifest:
            from resonance_lattice.bundled import pack as _pack_bundled
            files: dict[str, bytes] = {}
            seen_abs: set[str] = set()
            for sid, entry in manifest.items():
                if sid.startswith("__"):
                    continue
                if not isinstance(entry, dict):
                    continue
                rel = entry.get("source_file")
                if not rel:
                    continue
                # Normalise posix-relative -> absolute under source_root.
                # Absolute paths (files outside source_root) land as-is.
                if Path(rel).is_absolute():
                    abs_path = Path(rel)
                    bundle_key = str(abs_path)
                else:
                    abs_path = Path(source_root) / rel
                    bundle_key = rel  # already posix-relative
                if str(abs_path) in seen_abs:
                    continue
                try:
                    files[bundle_key] = abs_path.read_bytes()
                    seen_abs.add(str(abs_path))
                except OSError as e:
                    import warnings
                    warnings.warn(
                        f"Bundled mode: could not read {abs_path} ({e}); "
                        f"this file will be missing from the cartridge.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            store_data = _pack_bundled(files, meta_sqlite=store_data)

        if self.field_type == FieldType.DENSE:
            assert isinstance(self.field, DenseField)
            save_dense_field(
                path=path,
                field_tensor=self.field.F,
                config=self.config,
                source_count=self.source_count,
                registry_data=registry_data,
                store_data=store_data,
                store_mode=store_mode,
            )
        elif self.field_type == FieldType.FACTORED:
            assert isinstance(self.field, FactoredField)
            save_factored_field(
                path=path,
                U_list=self.field._U,
                sigma_list=self.field._sigma,
                V_list=self.field._V,
                config=self.config,
                source_count=self.source_count,
                store_mode=store_mode,
            )
        elif self.field_type == FieldType.PQ:
            assert isinstance(self.field, PQField)
            save_pq_field(
                path=path,
                codebooks=self.field._codebooks,
                qfield=self.field._qfield,
                config=self.config,
                source_count=self.source_count,
                store_mode=store_mode,
            )
        else:
            raise NotImplementedError(f"Save not yet implemented for {self.field_type}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        encoder: Encoder | None = None,
        restore_encoder: bool = True,
        source_root: str | Path | None = None,
    ) -> Lattice:
        """Load a lattice from a .rlat file.

        Args:
            path: Input .rlat file path.
            encoder: Optional encoder to attach.
            restore_encoder: Whether to restore the stored encoder config when
                no explicit encoder is provided.
            source_root: If provided, use a LocalStore that resolves
                source_ids to files under this directory instead of the
                embedded store. Useful for storeless knowledge models or when
                you want to read fresh content from local files.

        Returns:
            Loaded Lattice instance.
        """
        path = Path(path)

        with open(path, "rb") as f:
            header_bytes = f.read(RlatHeader.SIZE)
        header = RlatHeader.from_bytes(header_bytes)

        # A7: flag embedded-mode cartridges so users have time to migrate
        # before v2.0.0 removes the format. Python's default warning filter
        # de-dups by (category, module, lineno) so the message fires once
        # per Python session even if multiple cartridges get loaded — quiet
        # enough not to spam, loud enough to notice.
        if header.store_mode == "embedded":
            import warnings
            warnings.warn(
                f"{path.name} uses embedded store mode, deprecated since "
                f"v1.0.0 and slated for removal in v2.0.0. Rebuild with "
                f"`rlat build ... --store-mode external` (the new default) "
                f"or see docs/guides/migration-to-external.md.",
                DeprecationWarning,
                stacklevel=2,
            )

        config = LatticeConfig(
            bands=header.bands,
            dim=header.dim,
            field_type=header.field_type,
            svd_rank=header.svd_rank,
            pq_subspaces=header.pq_subspaces,
            pq_codebook_size=header.pq_codebook_size,
            precision=header.precision,
            compression=header.compression,
        )

        lattice = cls(config=config, encoder=encoder)

        # Track embedded store data for encoder restore even when using external store
        embedded_store_data = b""

        if header.field_type == FieldType.DENSE:
            _, tensor, registry_data, store_data, ann_index_data = load_dense_field(path)
            assert isinstance(lattice.field, DenseField)
            lattice.field.F = tensor
            lattice.field._source_count = header.source_count
            embedded_store_data = store_data

            # Restore registry
            if registry_data:
                lattice.registry = PhaseRegistry.from_bytes(
                    registry_data, dim=config.dim, bands=config.bands,
                )
                # Rebuild phase_cache from registry entries
                for sid, entry in lattice.registry._source_index.items():
                    lattice._phase_cache[sid] = (entry.phase_vectors, entry.salience)

            # Restore ANN index
            if ann_index_data:
                from resonance_lattice.ann_index import FAISSIndex
                try:
                    ann = FAISSIndex.from_bytes(ann_index_data, dim=config.bands * config.dim)
                    lattice.registry.set_ann_index(ann)
                except Exception:
                    pass  # ANN is optional; fall back to brute-force

            # Restore store — dispatch on store_mode + payload sniff.
            # Bundled payloads start with "RLBD" magic so we can detect
            # them independently of the header flag (format v3 sets both).
            from resonance_lattice.bundled import (
                BlobReader as _BlobReader,
            )
            from resonance_lattice.bundled import (
                is_bundled_payload as _is_bundled_payload,
            )
            from resonance_lattice.bundled import (
                split as _split_bundled,
            )
            from resonance_lattice.store import BundledStore as _BundledStore

            if store_data and _is_bundled_payload(store_data):
                meta_bytes, index, blob_bytes = _split_bundled(store_data)
                meta = SourceStore.from_bytes(meta_bytes) if meta_bytes else SourceStore()
                manifest = _load_source_manifest(meta_bytes) if meta_bytes else {}
                reader = _BlobReader(index=index, blob_bytes=blob_bytes)
                lattice.store = _BundledStore(
                    blob_reader=reader, manifest=manifest, meta_store=meta,
                )
                # embedded_store_data holds the metadata SQLite so the
                # encoder-restore path below can find __encoder__ etc.
                embedded_store_data = meta_bytes
            elif store_data and header.store_mode == "remote":
                # Remote-mode cartridge: metadata SQLite holds
                # __remote_origin__ with {type, org, repo, ref, commit_sha}.
                # Reconstruct the fetcher + cache and instantiate
                # RemoteStore. Never touches the network at load time.
                import json as _json_remote

                from resonance_lattice.remote import DiskCache as _DiskCache
                from resonance_lattice.remote.github import (
                    GithubFetcher as _GithubFetcher,
                )
                from resonance_lattice.remote.github import (
                    GithubOrigin as _GithubOrigin,
                )
                from resonance_lattice.store import RemoteStore as _RemoteStore

                meta = SourceStore.from_bytes(store_data)
                manifest = _load_source_manifest(store_data)
                origin_sc = meta.retrieve("__remote_origin__")
                origin_meta: dict = {}
                if origin_sc is not None:
                    try:
                        origin_meta = _json_remote.loads(origin_sc.full_text or "{}")
                    except Exception:
                        origin_meta = origin_sc.metadata or {}
                if not origin_meta.get("commit_sha") or origin_meta.get("type") != "github":
                    import warnings
                    warnings.warn(
                        f"Remote-mode cartridge {path.name} has no usable "
                        f"__remote_origin__ metadata; falling back to a "
                        f"metadata-only store (queries will miss content).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    lattice.store = meta
                else:
                    origin = _GithubOrigin(
                        org=origin_meta["org"],
                        repo=origin_meta["repo"],
                        ref=origin_meta.get("ref"),
                    )
                    fetcher = _GithubFetcher(origin=origin)
                    cache = _DiskCache()
                    lattice.store = _RemoteStore(
                        fetcher=fetcher,
                        commit_sha=origin_meta["commit_sha"],
                        cache=cache,
                        origin_key=origin.key,
                        manifest=manifest,
                        meta_store=meta,
                        path_prefix=origin_meta.get("path_prefix"),
                    )
                embedded_store_data = store_data
            elif source_root is not None:
                manifest = _load_source_manifest(store_data)
                meta = SourceStore.from_bytes(store_data) if store_data else None
                lattice.store = LocalStore(source_root, manifest=manifest, meta_store=meta)
            elif store_data and _is_external_cartridge(store_data):
                # Auto-detect: cartridge was saved with external store mode
                manifest = _load_source_manifest(store_data)
                inferred_root = _infer_source_root(path, manifest)
                if inferred_root is not None:
                    meta = SourceStore.from_bytes(store_data)
                    lattice.store = LocalStore(inferred_root, manifest=manifest, meta_store=meta)
                else:
                    import warnings
                    warnings.warn(
                        f"Cartridge was built with external store but source files "
                        f"could not be found relative to {path.parent}. "
                        f"Use --source-root to specify the source directory.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    lattice.store = SourceStore.from_bytes(store_data)
            elif store_data:
                lattice.store = SourceStore.from_bytes(store_data)
        elif header.field_type == FieldType.FACTORED:
            _, U_list, sigma_list, V_list = load_factored_field(path)
            assert isinstance(lattice.field, FactoredField)
            lattice.field._U = U_list
            lattice.field._sigma = sigma_list
            lattice.field._V = V_list
            for b in range(config.bands):
                if U_list[b] is not None:
                    lattice.field._current_rank[b] = U_list[b].shape[1]
            lattice.field._source_count = header.source_count
            if source_root is not None:
                # Factored format doesn't embed store data, so no manifest available
                lattice.store = LocalStore(source_root)
        elif header.field_type == FieldType.PQ:
            _, codebooks, qfield = load_pq_field(path)
            assert isinstance(lattice.field, PQField)
            lattice.field._codebooks = codebooks
            lattice.field._qfield = qfield
            lattice.field._codebooks_trained = True
            lattice.field._source_count = header.source_count
            if source_root is not None:
                # PQ format doesn't embed store data, so no manifest available
                lattice.store = LocalStore(source_root)

        # Restore encoder from stored config (if no encoder explicitly provided).
        # Lossless stores (external/local/bundled/remote) keep encoder config in
        # a separate metadata SQLite that lattice.store proxies to via _meta_store.
        # Fall back to rehydrating from the raw metadata bytes captured above.
        if restore_encoder and encoder is None:
            from resonance_lattice.store import LosslessStore as _LosslessStore
            _restore_store = lattice.store
            if isinstance(lattice.store, _LosslessStore) and embedded_store_data:
                _restore_store = SourceStore.from_bytes(embedded_store_data)
            if _restore_store.count > 0:
                encoder_content = _restore_store.retrieve("__encoder__")
                if encoder_content and encoder_content.full_text:
                    import base64 as _b64
                    import json
                    try:
                        from resonance_lattice.encoder import Encoder
                        enc_config = json.loads(encoder_content.full_text)

                        # Prefer embedded head weights (self-contained cartridge)
                        weights_content = _restore_store.retrieve("__encoder_weights__")
                        if weights_content and weights_content.full_text:
                            weights_blob = _b64.b64decode(weights_content.full_text)
                            lattice.encoder = Encoder.from_config_with_weights(
                                enc_config, weights_blob,
                            )
                        else:
                            # Legacy cartridge: no embedded weights — fall back to from_config
                            lattice.encoder = Encoder.from_config(enc_config)
                    except Exception as e:
                        import sys
                        print(f"Warning: could not restore encoder from cartridge: {e}",
                              file=sys.stderr)

        return lattice

    # ═══════════════════════════════════════════════════════
    # Novel primitives (Spec Section 12)
    # ═══════════════════════════════════════════════════════

    def subtract(self, other: Lattice) -> Lattice:
        """Corpus subtraction: compute the delta between two lattices.

        Returns a new Lattice whose field = self.field - other.field.
        Only supported for DenseField backends.

        Use case: "What changed between March and February?"
            delta = lattice_march.subtract(lattice_february)
            changes = delta.resonate(query)

        Args:
            other: The lattice to subtract.

        Returns:
            A new Lattice with the delta field.
        """
        if not isinstance(self.field, DenseField) or not isinstance(other.field, DenseField):
            raise TypeError("Corpus subtraction only supported for DenseField backends")
        if self.config.bands != other.config.bands or self.config.dim != other.config.dim:
            raise ValueError("Lattices must have matching bands and dim for subtraction")

        delta = Lattice(config=self.config)
        assert isinstance(delta.field, DenseField)
        delta.field.F = self.field.F - other.field.F
        delta.field._source_count = abs(self.source_count - other.source_count)
        return delta

    def find_contradictions(
        self,
        query_phase: PhaseSpectrum | NDArray[np.float32],
        band: int = 2,
        threshold: float = 0.7,
        top_k: int = 20,
    ) -> list[tuple[str, str, float]]:
        """Detect contradictions via destructive interference.

        Sources with anti-correlated phase vectors at a given band are
        likely making opposing claims. This finds pairs where both are
        relevant to the query but point in opposite directions.

        Args:
            query_phase: Query phase spectrum.
            band: Which band to check for contradictions (default: Omega_3/relations).
            threshold: Minimum absolute correlation for a contradiction.
            top_k: Number of top sources to check for pairwise contradictions.

        Returns:
            List of (source_id_a, source_id_b, anti_correlation_score) tuples.
        """
        if isinstance(query_phase, PhaseSpectrum):
            q = query_phase.vectors
        else:
            q = query_phase

        # Get top-k sources via brute-force lookup
        pointers = self.registry.lookup_bruteforce(query_phase=q, top_k=top_k)

        if len(pointers) < 2:
            return []

        # Get phase vectors for all top sources
        source_phases = {}
        for ptr in pointers:
            if ptr.source_id in self._phase_cache:
                source_phases[ptr.source_id] = self._phase_cache[ptr.source_id][0]

        # Find anti-correlated pairs at the specified band
        contradictions = []
        source_ids = list(source_phases.keys())
        for i in range(len(source_ids)):
            for j in range(i + 1, len(source_ids)):
                id_a, id_b = source_ids[i], source_ids[j]
                phi_a = source_phases[id_a][band]
                phi_b = source_phases[id_b][band]

                norm_a = np.linalg.norm(phi_a)
                norm_b = np.linalg.norm(phi_b)
                if norm_a < 1e-8 or norm_b < 1e-8:
                    continue

                correlation = np.dot(phi_a, phi_b) / (norm_a * norm_b)
                if correlation < -threshold:
                    contradictions.append((id_a, id_b, float(-correlation)))

        return sorted(contradictions, key=lambda x: x[2], reverse=True)

    def eigendecompose(
        self,
        band: int = 0,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Knowledge topology via eigendecomposition of the field tensor.

        Eigenvalues reveal dominant knowledge clusters. Eigenvectors are
        "concept axes." Eigenvalue gaps indicate natural cluster boundaries.

        Only supported for DenseField.

        Args:
            band: Which band to decompose.
            top_k: Number of top eigenvalues/vectors to return.

        Returns:
            Dict with eigenvalues, eigenvectors, spectral_gap, explained_variance.
        """
        if not isinstance(self.field, DenseField):
            raise TypeError("Eigendecomposition only supported for DenseField")

        from numpy.linalg import eigh

        F_b = self.field.F[band]
        # Symmetrise (should already be symmetric from outer products)
        F_sym = (F_b + F_b.T) / 2.0

        eigenvalues, eigenvectors = eigh(F_sym)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][:top_k]
        eigenvectors = eigenvectors[:, idx][:, :top_k]

        # Spectral gaps (differences between consecutive eigenvalues)
        spectral_gaps = np.diff(eigenvalues)

        # Explained variance
        total_energy = np.sum(np.abs(eigenvalues))
        explained_variance = np.cumsum(np.abs(eigenvalues)) / total_energy if total_energy > 0 else np.zeros_like(eigenvalues)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "spectral_gaps": spectral_gaps,
            "explained_variance": explained_variance,
            "total_energy": float(total_energy),
            "band": band,
        }

    def detect_communities(
        self,
        n_communities: int = 8,
        band: int = 0,
        top_eigvecs: int | None = None,
        max_sources: int = 5000,
    ) -> dict[str, Any]:
        """Detect topic communities via spectral clustering on the field.

        Projects source phase vectors into the field's eigenspace and
        clusters them using k-means on the spectral embedding. Each
        community represents a topic cluster in the corpus.

        Uses only numpy/scipy — no sklearn dependency.

        Args:
            n_communities: Number of topic clusters to detect.
            band: Which band to use for community structure.
            top_eigvecs: Number of eigenvectors for embedding (default: n_communities).
            max_sources: Sample size if registry is very large.

        Returns:
            Dict with community labels, sizes, representative sources, and centroids.
        """
        if not isinstance(self.field, DenseField):
            raise TypeError("Community detection requires DenseField")

        if top_eigvecs is None:
            top_eigvecs = n_communities

        # Get eigendecomposition
        topo = self.eigendecompose(band=band, top_k=top_eigvecs)
        eigvecs = topo["eigenvectors"]  # (D, top_k)

        # Get source phase vectors from registry
        reg = self.registry
        if reg._cache_dirty:
            reg._rebuild_cache()
        if reg._cached_phases is None:
            return {"communities": [], "error": "empty registry"}

        source_ids = reg._cached_ids
        n = len(source_ids)
        dim = reg._cached_phases.shape[1] // reg.bands
        phases_3d = reg._cached_phases.reshape(n, reg.bands, dim)
        band_phases = phases_3d[:, band, :]  # (N, D)

        # Subsample if very large
        if n > max_sources:
            rng = np.random.default_rng(42)
            chosen_idx = rng.choice(n, max_sources, replace=False)
            band_phases_sample = band_phases[chosen_idx]
            source_ids_sample = [source_ids[i] for i in chosen_idx]
        else:
            band_phases_sample = band_phases
            source_ids_sample = list(source_ids)

        n_sample = len(source_ids_sample)

        # Project into spectral space: each source gets a k-dim embedding
        projections = band_phases_sample @ eigvecs  # (N_sample, top_k)

        # Normalize rows for better clustering
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        projections_normed = projections / norms

        # K-means clustering (numpy-only implementation)
        k = min(n_communities, n_sample)
        rng = np.random.default_rng(42)

        # Initialize centroids with k-means++
        centroids = np.zeros((k, top_eigvecs), dtype=np.float32)
        centroids[0] = projections_normed[rng.integers(n_sample)]
        for c in range(1, k):
            dists = np.min([
                np.sum((projections_normed - centroids[j]) ** 2, axis=1)
                for j in range(c)
            ], axis=0)
            probs = dists / (dists.sum() + 1e-12)
            centroids[c] = projections_normed[rng.choice(n_sample, p=probs)]

        # Run k-means iterations
        for _ in range(50):
            # Assign
            dists = np.array([
                np.sum((projections_normed - centroids[j]) ** 2, axis=1)
                for j in range(k)
            ]).T  # (N_sample, k)
            labels = np.argmin(dists, axis=1)

            # Update
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                members = projections_normed[labels == j]
                if len(members) > 0:
                    new_centroids[j] = members.mean(axis=0)
                    norm = np.linalg.norm(new_centroids[j])
                    if norm > 1e-8:
                        new_centroids[j] /= norm
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Build community descriptions
        communities = []
        for j in range(k):
            member_mask = labels == j
            member_ids = [source_ids_sample[i] for i in range(n_sample) if member_mask[i]]
            if not member_ids:
                continue

            # Find representative sources (closest to centroid)
            member_projs = projections_normed[member_mask]
            dists_to_centroid = np.sum((member_projs - centroids[j]) ** 2, axis=1)
            top_rep_idx = np.argsort(dists_to_centroid)[:5]
            representatives = [member_ids[i] for i in top_rep_idx]

            # Community coherence: mean intra-cluster similarity
            if len(member_projs) > 1:
                coherence = float(np.mean(member_projs @ centroids[j]))
            else:
                coherence = 1.0

            communities.append({
                "id": j,
                "size": int(member_mask.sum()),
                "fraction": round(int(member_mask.sum()) / n_sample, 4),
                "coherence": round(coherence, 4),
                "representatives": representatives,
                "centroid_eigenweights": [round(float(v), 4) for v in centroids[j]],
            })

        # Sort by size descending
        communities.sort(key=lambda c: c["size"], reverse=True)

        # Assign final IDs by rank
        for rank, c in enumerate(communities):
            c["rank"] = rank

        return {
            "n_communities": len(communities),
            "n_sources_sampled": n_sample,
            "band": band,
            "top_eigvecs": top_eigvecs,
            "communities": communities,
            "eigenvalues_used": [round(float(v), 6) for v in topo["eigenvalues"][:top_eigvecs]],
        }

    # ═══════════════════════════════════════════════════════
    # Production SDK — clean public API (WS1)
    # ═══════════════════════════════════════════════════════

    def add(
        self,
        source_id: str,
        text: str,
        salience: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a document to the lattice.

        This is the primary ingestion method for the production SDK.
        Requires an encoder to be configured.

        Args:
            source_id: Unique identifier for this document.
            text: Document text to encode and store.
            salience: Importance weight (default 1.0).
            metadata: Optional metadata dict.

        Returns:
            The source_id used.
        """
        return self.superpose_text(
            text=text,
            salience=salience,
            source_id=source_id,
            metadata=metadata,
        )

    def query(
        self,
        text: str,
        top_k: int = 10,
        band_weights: NDArray[np.float32] | None = None,
    ) -> list[MaterialisedResult]:
        """Query the lattice and return ranked results.

        This is the primary retrieval method for the production SDK.
        Requires an encoder to be configured.

        Args:
            text: Query text string.
            top_k: Number of results to return.
            band_weights: Optional per-band weights for fusion.

        Returns:
            List of MaterialisedResult sorted by score descending.
        """
        retrieval = self.resonate_text(
            query=text,
            top_k=top_k,
            band_weights=band_weights,
        )
        return retrieval.results

    def enriched_query(
        self,
        text: str,
        top_k: int = 10,
        cascade_depth: int | None = None,
        contradiction_threshold: float | None = None,
        enable_cascade: bool | None = None,
        enable_contradictions: bool | None = None,
        enable_lexical: bool | None = None,
        enable_rerank: bool | str | None = None,
        enable_dense: bool = True,
        enable_cross_encoder: bool = False,
        cross_encoder_expand: bool = False,
        enable_subgraph: bool = False,
        subgraph_context_k: int = 3,
        lexical_weight: float | None = None,
        phrase_bonus: float = 0.1,
        asymmetric: bool = False,
        scoring_fn: str = "linear",
        enable_assess: bool = False,
        merge_cascade: bool = True,
        mode: str = "auto",
        question_date: str | None = None,
    ) -> EnrichedResult:
        """Query the lattice and return enriched results with coverage, cascade, and contradictions.

        Combines ranked text passages with structural metadata in a single call:
        - Coverage profile (per-band energy, confidence, gaps)
        - Related topics (multi-hop cascade, configurable depth)
        - Contradiction detection (destructive interference between top-k results)

        Requires an encoder and a DenseField backend for full enrichment.

        Args:
            text: Query text string.
            top_k: Number of results to return.
            cascade_depth: Hop depth for related topics (default 2, results degrade beyond 2).
            contradiction_threshold: Threshold for destructive interference.
                None = use the default fixed threshold (-0.3) when enabled.
                More negative = stricter (fewer false positives).
            enable_cascade: Whether to run cascade for related topics.
            enable_contradictions: Whether to run contradiction detection.
            enable_lexical: Whether to inject lexical phrase/keyword matches.
                Set False for dense-only retrieval ablation.
            enable_rerank: Whether to apply lexical reranking to the result pool.
                Set False to skip blended scoring and return field-score order.
                Set "auto" to skip reranking when dense results are already
                well-separated (high discrimination), and apply it otherwise.
            enable_cross_encoder: Whether to apply cross-encoder reranking after
                hybrid candidate generation. Uses a transformer to score each
                query-passage pair for domain-agnostic relevance. Default False.
            enable_subgraph: Whether to expand results with spectral neighbours.
                Improves fact coverage by returning context around each hit.
                Default False.
            subgraph_context_k: Number of neighbours per result (default 3).
            lexical_weight: Weight for lexical score in reranking (0-1).
                Default 0.5. Higher values boost keyword matches.
            phrase_bonus: Bonus added for exact bigram phrase matches.
                Default 0.1. Helps surface results with exact phrase overlap.
            enable_assess: Whether to compute result manifold metrics
                (diversity, concentration, signal strength, coverage).
                Adds an 'assessment' dict to EnrichedResult. Default False.

        Returns:
            EnrichedResult with passages, coverage, related topics, and contradictions.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder configured. enriched_query() requires an encoder.")

        # ── Resolve auto defaults ──────────────────────────────────────
        # mode="auto" (default): unset knobs are filled by adaptive_memory_config
        # based on query surface features. Any explicit non-None value wins.
        # mode="manual": unset knobs fall back to conservative hard defaults.
        if mode == "auto":
            from resonance_lattice.query_router import adaptive_memory_config
            routed = adaptive_memory_config(text, question_date=question_date)
            if enable_cascade is None:
                enable_cascade = bool(routed.get("enable_cascade", False))
            if enable_lexical is None:
                enable_lexical = bool(routed.get("enable_lexical", True))
            if enable_rerank is None:
                enable_rerank = routed.get("enable_rerank", "auto")
            if enable_contradictions is None:
                enable_contradictions = bool(routed.get("enable_contradictions", False))
            if cascade_depth is None:
                cascade_depth = int(routed.get("cascade_depth", 2))
            if lexical_weight is None:
                lexical_weight = float(routed.get("lexical_weight", 0.3))
        else:
            if enable_cascade is None: enable_cascade = False
            if enable_lexical is None: enable_lexical = False
            if enable_rerank is None: enable_rerank = True
            if enable_contradictions is None: enable_contradictions = False
            if cascade_depth is None: cascade_depth = 2
            if lexical_weight is None: lexical_weight = 0.3

        t_start = _time.perf_counter()
        timings_ms = {
            "encode": 0.0,
            "resonate": 0.0,
            "registry": 0.0,
            "store": 0.0,
            "cascade": 0.0,
            "contradictions": 0.0,
            "total": 0.0,
        }

        # Step 1: Encode query
        t_encode_start = _time.perf_counter()
        phase = self.encoder.encode_query(text, asymmetric=asymmetric)
        timings_ms["encode"] = (_time.perf_counter() - t_encode_start) * 1000
        q_vectors = phase.vectors if isinstance(phase, PhaseSpectrum) else phase

        # Step 2: Hybrid retrieval (field candidates + lexical search + rerank)
        candidate_k = min(top_k * 10, self.source_count)
        if enable_dense:
            retrieval = self.resonate(
                q_vectors, top_k=candidate_k,
                scoring_fn=scoring_fn,
            )
            if retrieval.timings_ms:
                for name in ("resonate", "registry", "store"):
                    timings_ms[name] = float(retrieval.timings_ms.get(name, 0.0))
        else:
            # Lexical-only: start with an empty candidate pool
            retrieval = RetrievalResult(
                results=[],
                resonance=ResonanceResult(
                    resonance_vectors=np.zeros_like(q_vectors),
                    fused=np.zeros(q_vectors.shape[1], dtype=np.float32),
                    band_energies=np.zeros(q_vectors.shape[0], dtype=np.float32),
                ),
                source_pointers=[],
                timings_ms=None,
            )

        # Inject lexical matches that the field missed
        if enable_lexical:
            retrieval = self._inject_lexical_matches(text, q_vectors, retrieval, candidate_k)

        # Auto-decide reranking: skip when dense results are well-separated.
        if enable_rerank == "auto":
            enable_rerank = self._should_rerank(retrieval)

        # Rerank the combined pool with keyword boost
        if enable_rerank:
            retrieval = self._lexical_rerank(
                text, retrieval, top_k,
                lexical_weight=lexical_weight,
                phrase_bonus=phrase_bonus,
            )
        elif enable_lexical:
            # Lightweight fusion: blend field score with keyword overlap so
            # lexical injections can compete with dense candidates without
            # the full rerank machinery (phrase bonus, dedup, min_score).
            retrieval = self._keyword_fuse(
                text, retrieval, top_k,
                keyword_weight=lexical_weight,
                phrase_bonus=phrase_bonus,
            )
        else:
            # Pure dense: sort by field score only.
            sorted_results = sorted(
                retrieval.results, key=lambda r: r.score, reverse=True,
            )[:top_k]
            retrieval = RetrievalResult(
                results=sorted_results,
                resonance=retrieval.resonance,
                source_pointers=retrieval.source_pointers[:top_k]
                    if retrieval.source_pointers else retrieval.source_pointers,
                timings_ms=retrieval.timings_ms,
            )

        # Step 2a-ctx: Recover heading context for top passages.
        # Done BEFORE cross-encoder so the transformer scores contextually-
        # rich passages (with heading anchors) rather than raw chunk text.
        retrieval = RetrievalResult(
            results=self._recover_passage_context(retrieval.results),
            resonance=retrieval.resonance,
            source_pointers=retrieval.source_pointers,
            timings_ms=retrieval.timings_ms,
        )

        # Step 2b-ce: Cross-encoder reranking (domain-agnostic)
        if enable_cross_encoder:
            from resonance_lattice.reranker import CrossEncoderReranker
            if not hasattr(self, "_cross_encoder"):
                self._cross_encoder = CrossEncoderReranker()
            # B1 natural-boundary expansion on rerank input is OPT-IN
            # via `cross_encoder_expand`. NFCorpus test (mxbai) showed
            # expansion can hurt on short-prose corpora by adding
            # boilerplate that dilutes the query-relevance signal.
            # Default off preserves the measured CE win.
            _source_root = (
                getattr(getattr(self, "store", None), "source_root", None)
                if cross_encoder_expand else None
            )
            reranked_results, ce_ms = self._cross_encoder.rerank(
                text, retrieval.results, top_k=top_k,
                source_root=_source_root,
            )
            retrieval = RetrievalResult(
                results=reranked_results,
                resonance=retrieval.resonance,
                source_pointers=retrieval.source_pointers,
                timings_ms={**(retrieval.timings_ms or {}), "cross_encoder": ce_ms},
            )

        # Step 2c: Subgraph expansion — add spectral neighbours for context
        if enable_subgraph and enable_dense and self.source_count > 0:
            t_sg_start = _time.perf_counter()
            seen_ids = {r.source_id for r in retrieval.results}
            extra_results: list[MaterialisedResult] = []
            for r in retrieval.results[:min(5, len(retrieval.results))]:
                entry = self.registry._source_index.get(r.source_id)
                if entry is None:
                    continue
                neighbours = self.registry.lookup_bruteforce(
                    query_phase=entry.phase_vectors,
                    top_k=subgraph_context_k + 1,
                )
                for nb in neighbours:
                    if nb.source_id not in seen_ids:
                        seen_ids.add(nb.source_id)
                        content = self.store.retrieve(nb.source_id) if self.store else None
                        extra_results.append(MaterialisedResult(
                            source_id=nb.source_id,
                            score=nb.fidelity_score * 0.5,
                            band_scores=nb.band_signature,
                            content=content,
                            raw_score=nb.fidelity_score,
                            provenance="subgraph",
                        ))
            if extra_results:
                combined = sorted(
                    list(retrieval.results) + extra_results,
                    key=lambda r: r.score,
                    reverse=True,
                )[:top_k]
                retrieval = RetrievalResult(
                    results=combined,
                    resonance=retrieval.resonance,
                    source_pointers=retrieval.source_pointers,
                    timings_ms=retrieval.timings_ms,
                )
            timings_ms["subgraph"] = (_time.perf_counter() - t_sg_start) * 1000

        # Step 2d: Result manifold — assess only (sharpen removed v1.5 Pass 4b)
        assessment_dict: dict[str, Any] | None = None
        if enable_assess and retrieval.results:
            from resonance_lattice.rql.result_manifold import (
                assess_results as _assess_results,
            )
            t_manifold_start = _time.perf_counter()

            # Extract phase vectors from registry for each result
            result_phases = []
            result_scores = []
            for r in retrieval.results:
                entry = self.registry._source_index.get(r.source_id)
                if entry is not None:
                    result_phases.append(entry.phase_vectors)
                    result_scores.append(r.score)

            if result_phases:
                phases_arr = np.stack(result_phases)
                scores_arr = np.array(result_scores, dtype=np.float32)
                corpus_f = (
                    self.field
                    if isinstance(self.field, DenseField)
                    else None
                )
                assessment = _assess_results(
                    phases_arr, scores_arr,
                    corpus_field=corpus_f,
                )
                assessment_dict = assessment.to_dict()

            timings_ms["manifold"] = (
                (_time.perf_counter() - t_manifold_start) * 1000
            )

        # Step 3a: Coverage profile from actual retrieval quality
        coverage = self._compute_coverage(q_vectors, retrieval.results)

        # Step 3b: Cascade for related topics (only on DenseField)
        related: list[RelatedTopic] = []
        if enable_cascade and isinstance(self.field, DenseField) and self.source_count > 0:
            t_cascade_start = _time.perf_counter()
            related = self._compute_related(q_vectors, retrieval, cascade_depth)
            timings_ms["cascade"] = (_time.perf_counter() - t_cascade_start) * 1000

            # Merge cascade hits into the ranked result set so `related` is not
            # just diagnostic — cascade surfaces sibling sessions the direct
            # query missed (core lever for multi-session aggregation). We
            # interleave cascade topics after a protected head of direct
            # results, displacing the weakest tail so total stays <= top_k.
            if merge_cascade and related and retrieval.results:
                seen = {r.source_id for r in retrieval.results}
                direct = list(retrieval.results)
                head_k = min(3, len(direct))  # protect top-3 direct hits
                head = direct[:head_k]
                tail = direct[head_k:]
                direct_min = min((float(r.score) for r in direct), default=0.0)
                cascade_results: list[MaterialisedResult] = []
                for topic in related:
                    if topic.source_id in seen:
                        continue
                    damped = min(float(topic.score) * 0.75, direct_min * 0.99)
                    cascade_results.append(MaterialisedResult(
                        source_id=topic.source_id,
                        score=damped,
                        band_scores=np.zeros(self.config.bands, dtype=np.float32),
                        content=topic.content,
                        raw_score=float(topic.score),
                        provenance="cascade",
                    ))
                    seen.add(topic.source_id)
                # Interleave: one cascade hit after each remaining tail slot,
                # up to top_k total.
                interleaved: list[MaterialisedResult] = list(head)
                ci = ti = 0
                while len(interleaved) < top_k and (ci < len(cascade_results) or ti < len(tail)):
                    if ti < len(tail):
                        interleaved.append(tail[ti])
                        ti += 1
                    if ci < len(cascade_results) and len(interleaved) < top_k:
                        interleaved.append(cascade_results[ci])
                        ci += 1
                retrieval = RetrievalResult(
                    results=interleaved[:top_k],
                    resonance=retrieval.resonance,
                    source_pointers=retrieval.source_pointers,
                    timings_ms=retrieval.timings_ms,
                )

        # Step 3c: Contradiction detection (only on DenseField)
        contradictions: list[ContradictionPair] = []
        if enable_contradictions and isinstance(self.field, DenseField) and len(retrieval.results) >= 2:
            if contradiction_threshold is None:
                contradiction_threshold = -0.3
            t_contradictions_start = _time.perf_counter()
            contradictions = self._compute_contradictions(
                q_vectors, retrieval, contradiction_threshold,
            )
            timings_ms["contradictions"] = (
                (_time.perf_counter() - t_contradictions_start) * 1000
            )

        latency_ms = (_time.perf_counter() - t_start) * 1000
        timings_ms["total"] = latency_ms

        return EnrichedResult(
            query=text,
            results=retrieval.results,
            coverage=coverage,
            related=related,
            contradictions=contradictions,
            latency_ms=latency_ms,
            timings_ms=timings_ms,
            assessment=assessment_dict,
        )

    # Alias for CLI discoverability
    search = enriched_query

    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
        "and", "or", "not", "with", "from", "by", "as", "be", "was", "were",
        "are", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "shall", "may", "might", "can",
        "that", "this", "these", "those", "what", "which", "who", "whom",
        "when", "where", "why", "how", "if", "then", "than", "but", "so",
        "very", "just", "about", "into", "over", "after", "before", "between",
        "through", "during", "each", "some", "any", "all", "both", "more",
        "most", "other", "such", "only", "same", "also", "use", "used",
    })

    @staticmethod
    def _tokenize_for_match(text: str) -> set[str]:
        """Extract lowercase word tokens for keyword matching, excluding stop words."""
        import re
        tokens = set(re.findall(r'[a-z][a-z0-9_-]+', text.lower()))
        return tokens - Lattice._STOP_WORDS

    @staticmethod
    def _extract_phrases(text: str) -> list[str]:
        """Extract likely multi-word phrases from a query (bigrams of content words)."""
        import re
        words = re.findall(r'[a-z][a-z0-9]+', text.lower())
        content = [w for w in words if w not in Lattice._STOP_WORDS and len(w) >= 3]
        phrases = []
        for i in range(len(content) - 1):
            phrases.append(f"{content[i]}%{content[i+1]}")  # SQL LIKE pattern
        return phrases

    def _inject_lexical_matches(
        self,
        query_text: str,
        query_phase: NDArray[np.float32],
        retrieval: RetrievalResult,
        max_inject: int = 50,
    ) -> RetrievalResult:
        """Deprecated: SQL-based phrase/keyword injection for embedded knowledge models.

        **Superseded by B3 ripgrep hybrid** (see
        ``resonance_lattice.retrieval.lexical.lexical_rerank`` and commit
        ``8915b1c``). B3 runs over the retrieved neighbourhood's files via
        ripgrep and blends a saturating lexical score into the dense ranking.
        It works on embedded AND external knowledge models, carries no SQLite
        dependency, and fails soft if ``rg`` is not on PATH.

        This legacy path still runs for embedded knowledge models in
        ``enriched_query`` for back-compat with pre-B3 callers. It is a no-op
        on external knowledge models (no ``_conn`` — see the ``hasattr`` guard
        below). Empirical measurement on 100 queries against dense retrieval
        showed it injects **zero unique top-20 candidates** beyond what dense
        already surfaces, so disabling it on external has no measurable
        cost. A proper inverted-index sidecar for external knowledge models (#313)
        was considered and rejected on those grounds — B3 is the supported
        hybrid path.

        Strategy (legacy):
        1. Extract bigram phrases (e.g. "direct%lake") — strongest signal.
        2. Fall back to individual keywords if no phrase hits.
        3. Search source_id, summary, and full_text (source_id contains
           file stem + heading slug, which is the most reliable signal).
        """
        phrases = self._extract_phrases(query_text)
        keywords = sorted(self._tokenize_for_match(query_text), key=len, reverse=True)
        keywords = [k for k in keywords if len(k) >= 3][:10]

        if not phrases and not keywords:
            return retrieval

        existing_ids = {r.source_id for r in retrieval.results}
        # LocalStore has no SQLite index over source text — lexical
        # phrase/keyword matches would need an external inverted index.
        # Until that lands, skip lexical injection for external cartridges
        # rather than crash.
        if not hasattr(self.store, "_conn"):
            return retrieval
        conn = self.store._conn
        injected: list[MaterialisedResult] = []
        import json as _json

        def _run_search(where: str, params: list, per_limit: int = 0) -> None:
            if len(injected) >= max_inject:
                return
            remaining = max_inject - len(injected)
            # Cap each individual search to avoid one broad query consuming
            # all capacity (default: min(remaining, 15))
            limit = min(remaining, per_limit) if per_limit > 0 else min(remaining, 10)
            sql = (
                f"SELECT source_id, summary, full_text, metadata FROM sources "
                f"WHERE ({where}) AND substr(source_id, 1, 2) != '__' "
                f"LIMIT ?"
            )
            try:
                cursor = conn.execute(sql, params + [limit])
            except Exception:
                return
            for row in cursor:
                if len(injected) >= max_inject:
                    break
                sid = row[0]
                if sid in existing_ids:
                    continue
                existing_ids.add(sid)
                field_score = 0.0
                if sid in self._phase_cache:
                    pvecs, sal = self._phase_cache[sid]
                    for b in range(self.config.bands):
                        field_score += float(np.dot(pvecs[b], query_phase[b]))
                    field_score *= sal
                injected.append(MaterialisedResult(
                    source_id=sid,
                    score=field_score,
                    band_scores=None,
                    content=SourceContent(
                        source_id=sid, summary=row[1], full_text=row[2],
                        metadata=_json.loads(row[3]) if row[3] else {},
                    ),
                    provenance="lexical",
                ))

        # Phase 1: Source-ID phrase search — search ALL phrases, not just first 3.
        # Topic-identifying bigrams often appear later in the query.
        # Use a generous per-search limit here because phrase matches on
        # source_id are the highest-precision signal — don't cap them like
        # the broader keyword searches.
        for phrase in phrases:
            if len(injected) >= max_inject:
                break
            _run_search("source_id LIKE ? COLLATE NOCASE", [f"%{phrase}%"],
                        per_limit=25)

        # Phase 2: Source-ID multi-keyword conjunction search.
        # Search pairs of keywords together (AND) for higher precision than
        # individual keyword matches which swamp the pool with noise.
        if len(injected) < max_inject and len(keywords) >= 2:
            from itertools import combinations
            # Try pairs of content keywords (shorter words first — more specific)
            kw_by_specificity = sorted(keywords, key=len)
            for a, b in combinations(kw_by_specificity[:6], 2):
                if len(injected) >= max_inject:
                    break
                _run_search(
                    "source_id LIKE ? COLLATE NOCASE AND source_id LIKE ? COLLATE NOCASE",
                    [f"%{a}%", f"%{b}%"],
                )

        # Phase 2b: Source-ID individual keyword search (fallback)
        if len(injected) < max_inject and keywords:
            for kw in keywords[:5]:
                if len(injected) >= max_inject:
                    break
                if len(kw) < 3:
                    continue
                _run_search("source_id LIKE ? COLLATE NOCASE", [f"%{kw}%"])

        # Phase 3: Content phrase search — search ALL phrases
        if len(injected) < max_inject and phrases:
            for phrase in phrases:
                if len(injected) >= max_inject:
                    break
                pattern = f"%{phrase}%"
                _run_search(
                    "full_text LIKE ? COLLATE NOCASE OR summary LIKE ? COLLATE NOCASE",
                    [pattern, pattern],
                )

        # Phase 4: Content multi-keyword conjunction search
        if len(injected) < max_inject and len(keywords) >= 2:
            from itertools import combinations
            kw_by_specificity = sorted(keywords, key=len)
            for a, b in combinations(kw_by_specificity[:6], 2):
                if len(injected) >= max_inject:
                    break
                _run_search(
                    "(full_text LIKE ? COLLATE NOCASE AND full_text LIKE ? COLLATE NOCASE)"
                    " OR (summary LIKE ? COLLATE NOCASE AND summary LIKE ? COLLATE NOCASE)",
                    [f"%{a}%", f"%{b}%", f"%{a}%", f"%{b}%"],
                )

        # Phase 4b: Content individual keyword search (fallback)
        if len(injected) < max_inject and keywords:
            for kw in keywords[:3]:
                if len(injected) >= max_inject:
                    break
                if len(kw) < 3:
                    continue
                _run_search(
                    "full_text LIKE ? COLLATE NOCASE OR summary LIKE ? COLLATE NOCASE",
                    [f"%{kw}%", f"%{kw}%"],
                )

        if not injected:
            return retrieval

        return RetrievalResult(
            results=list(retrieval.results) + injected,
            resonance=retrieval.resonance,
            source_pointers=retrieval.source_pointers,
            timings_ms=retrieval.timings_ms,
        )

    def _should_rerank(
        self,
        retrieval: RetrievalResult,
        discrimination_threshold: float = 0.7,
    ) -> bool:
        """Decide whether reranking will help based on dense result separation.

        When the dense field already produces well-separated scores (high
        coefficient of variation), reranking adds latency without improving
        ordering. When scores cluster, reranking helps amplify subtle
        differences via lexical signals.

        Returns True if reranking should be applied.
        """
        results = retrieval.results
        if len(results) < 3:
            return True  # Too few results to judge

        import math
        raw_vals = [
            r.raw_score if r.raw_score is not None else r.score
            for r in results[:10]
        ]
        raw_mean = sum(raw_vals) / len(raw_vals)
        raw_var = sum((v - raw_mean) ** 2 for v in raw_vals) / len(raw_vals)
        raw_cv = math.sqrt(raw_var) / (abs(raw_mean) + 1e-8)
        discrimination = min(raw_cv / 0.002, 1.0)

        return discrimination < discrimination_threshold

    def _keyword_fuse(
        self,
        query_text: str,
        retrieval: RetrievalResult,
        top_k: int,
        keyword_weight: float = 0.5,
        phrase_bonus: float = 0.1,
    ) -> RetrievalResult:
        """Lightweight keyword-overlap fusion for the no-rerank hybrid path.

        Blends normalised field score with keyword overlap so that
        lexical injections (which often have low/zero field scores) can
        surface before the top-k cutoff.  Applies a small phrase bonus
        for exact bigram matches but does not apply per-source dedup or
        min_score gating — those belong to the full rerank path.

        Args:
            query_text: Original query string.
            retrieval: Combined dense + lexical candidate pool.
            top_k: Number of results to return.
            keyword_weight: Weight for keyword overlap (0-1). Default 0.5.
            phrase_bonus: Bonus for exact bigram phrase matches. Default 0.1.
        """
        if not retrieval.results:
            return retrieval

        query_tokens = self._tokenize_for_match(query_text)
        if not query_tokens:
            sorted_results = sorted(
                retrieval.results, key=lambda r: r.score, reverse=True,
            )[:top_k]
            return RetrievalResult(
                results=sorted_results,
                resonance=retrieval.resonance,
                source_pointers=retrieval.source_pointers[:top_k]
                    if retrieval.source_pointers else retrieval.source_pointers,
                timings_ms=retrieval.timings_ms,
            )

        query_phrases = self._extract_phrases(query_text)

        # Normalise field scores to [0, 1]
        scores = [r.score for r in retrieval.results]
        max_field = max(scores) if scores else 1.0
        min_field = min(scores) if scores else 0.0
        score_range = max_field - min_field if max_field > min_field else 1.0

        scored: list[tuple[float, int]] = []
        for i, r in enumerate(retrieval.results):
            field_norm = (r.score - min_field) / score_range

            doc_text = ""
            if r.content:
                doc_text = (r.content.full_text or r.content.summary or "")
            doc_tokens = self._tokenize_for_match(doc_text)
            sid_tokens = self._tokenize_for_match(r.source_id)

            content_overlap = self._idf_weighted_overlap(query_tokens, doc_tokens)
            sid_overlap = self._idf_weighted_overlap(query_tokens, sid_tokens)
            overlap = min(max(content_overlap, sid_overlap * 1.5), 1.0)

            fused = (1 - keyword_weight) * field_norm + keyword_weight * overlap

            # Lightweight phrase bonus: reward exact bigram matches
            if query_phrases and doc_text and phrase_bonus > 0:
                doc_lower = doc_text.lower()
                for phrase in query_phrases:
                    if phrase.replace("%", " ") in doc_lower:
                        fused = min(fused + phrase_bonus, 1.0)
                        break  # One phrase match is enough for the lightweight path

            scored.append((fused, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        fused_results = []
        for fused_score, idx in scored[:top_k]:
            r = retrieval.results[idx]
            fused_results.append(MaterialisedResult(
                source_id=r.source_id,
                score=fused_score,
                band_scores=r.band_scores,
                content=r.content,
                raw_score=r.score,
                provenance=getattr(r, "provenance", "dense"),
            ))

        return RetrievalResult(
            results=fused_results,
            resonance=retrieval.resonance,
            source_pointers=retrieval.source_pointers[:top_k]
                if retrieval.source_pointers else retrieval.source_pointers,
            timings_ms=retrieval.timings_ms,
        )

    def _lexical_rerank(
        self,
        query_text: str,
        retrieval: RetrievalResult,
        top_k: int,
        lexical_weight: float = 0.3,
        phrase_bonus: float = 0.1,
        max_per_source: int = 2,
        min_score: float = 0.0,
        dedup_threshold: float = 0.85,
    ) -> RetrievalResult:
        """Rerank retrieval results with RRF + band-aware fusion.

        Uses Reciprocal Rank Fusion (RRF) to combine field and keyword
        rankings. RRF is rank-based, not score-based, so it is naturally
        calibrated regardless of score distributions — no z-score
        normalization needed. When field and keyword rankings agree, the
        candidate gets boosted; when they disagree, neither dominates.

        Band-aware fusion adjusts how much the keyword signal matters
        based on the query's band energy profile: entity-dominant queries
        weight keywords higher (name matching), topic-dominant queries
        trust the field more (conceptual matching).

        After scoring, applies per-source-file deduplication, near-
        duplicate Jaccard suppression, and exact-phrase bonus.

        Args:
            query_text: Original query string.
            retrieval: Wide candidate retrieval result.
            top_k: Final number of results to return.
            lexical_weight: Baseline weight for lexical signal (0-1).
                Adjusted by band-aware logic. Default 0.5.
            phrase_bonus: Bonus added for exact phrase matches.
                Default 0.1.
            min_score: Minimum blended score to include in results.
                Default 0.0.
            dedup_threshold: Jaccard token similarity threshold for
                near-duplicate suppression (0-1). Default 0.85.

        Returns:
            Reranked RetrievalResult with top_k results.
        """
        if not retrieval.results:
            return retrieval

        query_tokens = self._tokenize_for_match(query_text)
        if not query_tokens:
            results = retrieval.results[:top_k]
            return RetrievalResult(
                results=results,
                resonance=retrieval.resonance,
                source_pointers=retrieval.source_pointers[:top_k],
                timings_ms=retrieval.timings_ms,
            )

        query_phrases = self._extract_phrases(query_text)
        n = len(retrieval.results)

        # --- Band-aware keyword weight (Opt 3) ---
        # Use the resonance's band energies to detect query type and
        # nudge how much the keyword signal matters. Conservative
        # adjustments: ±0.1 from baseline to avoid disrupting ordering.
        # NOTE: Disabled pending calibration — caused 2% R@5 regression
        # on Fabric. Re-enable after per-corpus threshold tuning.
        effective_kw_weight = lexical_weight

        # --- Collect raw signals ---
        field_scores: list[float] = []
        overlaps: list[float] = []
        for r in retrieval.results:
            field_scores.append(r.score)

            doc_text = ""
            if r.content:
                doc_text = (r.content.full_text or r.content.summary or "")
            doc_tokens = self._tokenize_for_match(doc_text)
            sid_tokens = self._tokenize_for_match(r.source_id)

            content_overlap = self._idf_weighted_overlap(query_tokens, doc_tokens)
            sid_overlap = self._idf_weighted_overlap(query_tokens, sid_tokens)
            overlap = min(max(content_overlap, sid_overlap * 1.5), 1.0)
            overlaps.append(overlap)

        # --- Z-score fusion with band-aware weighting ---
        # Z-score normalization converts each signal to zero-mean, unit-variance.
        # When a signal has near-zero variance (uninformative), its z-scores
        # cluster at zero and naturally contribute nothing to the blend.
        # Band-aware keyword weighting (Opt 3) adjusts the lexical weight
        # based on the query's band energy profile.
        import math

        def _z_normalize(vals: list[float]) -> list[float]:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var) if var > 0 else 0.0
            if std < 1e-9:
                return [0.0] * len(vals)
            return [(v - mean) / std for v in vals]

        z_field = _z_normalize(field_scores)
        z_lex = _z_normalize(overlaps)

        field_w = 1.0 - effective_kw_weight
        scored: list[tuple[float, int]] = []
        for i in range(n):
            blended = field_w * z_field[i] + effective_kw_weight * z_lex[i]

            # Exact-phrase bonus
            if query_phrases and phrase_bonus > 0:
                doc_text = ""
                if retrieval.results[i].content:
                    doc_text = (retrieval.results[i].content.full_text or
                                retrieval.results[i].content.summary or "")
                if doc_text:
                    doc_lower = doc_text.lower()
                    n_phrase_hits = 0
                    for phrase in query_phrases:
                        if phrase.replace("%", " ") in doc_lower:
                            n_phrase_hits += 1
                    if n_phrase_hits > 0:
                        bonus = phrase_bonus + phrase_bonus * 0.5 * (n_phrase_hits - 1)
                        blended += bonus

            scored.append((blended, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        # --- Select top_k with dedup and filtering ---
        # Rescale to [0, 1] for downstream consumers that expect normalised scores.
        if scored:
            max_s = scored[0][0]
            min_s = scored[-1][0]
            s_range = max_s - min_s if max_s > min_s else 1.0
        else:
            min_s, s_range = 0.0, 1.0

        reranked_results = []
        reranked_pointers = []
        source_counts: dict[str, int] = {}
        accepted_token_sets: list[set[str]] = []
        for raw_score, idx in scored:
            if len(reranked_results) >= top_k:
                break
            norm_score = (raw_score - min_s) / s_range
            if norm_score < min_score:
                break
            r = retrieval.results[idx]

            # Per-source-file dedup
            if max_per_source > 0:
                source_key = ""
                if r.content and r.content.metadata:
                    source_key = r.content.metadata.get("source_file", "")
                if source_key:
                    count = source_counts.get(source_key, 0)
                    if count >= max_per_source:
                        continue
                    source_counts[source_key] = count + 1

            # Near-duplicate suppression
            if dedup_threshold < 1.0 and r.content:
                candidate_text = r.content.full_text or r.content.summary or ""
                if candidate_text:
                    cand_tokens = self._tokenize_for_match(candidate_text)
                    if cand_tokens:
                        is_near_dup = False
                        for accepted_ts in accepted_token_sets:
                            intersection = len(cand_tokens & accepted_ts)
                            union = len(cand_tokens | accepted_ts)
                            if union > 0 and intersection / union > dedup_threshold:
                                is_near_dup = True
                                break
                        if is_near_dup:
                            continue
                        accepted_token_sets.append(cand_tokens)

            # Classify dominant signal from z-score contributions
            zf = z_field[idx]
            zl = z_lex[idx]
            if abs(zf - zl) < 0.3:
                signal = "hybrid"
            elif zf > zl:
                signal = "semantic"
            else:
                signal = "keyword"

            reranked_results.append(MaterialisedResult(
                source_id=r.source_id,
                score=norm_score,
                band_scores=r.band_scores,
                content=r.content,
                raw_score=r.raw_score if r.raw_score is not None else r.score,
                provenance=signal,
            ))
            if idx < len(retrieval.source_pointers):
                reranked_pointers.append(retrieval.source_pointers[idx])

        return RetrievalResult(
            results=reranked_results,
            resonance=retrieval.resonance,
            source_pointers=reranked_pointers,
            timings_ms=retrieval.timings_ms,
        )

    @staticmethod
    def _recover_passage_context(results: list[MaterialisedResult]) -> list[MaterialisedResult]:
        """Prepend heading context to passages that start mid-paragraph.

        When a chunk was split mid-document, the heading metadata survives
        in the store but the passage text may start abruptly.  Prepending
        ``## heading`` gives the consumer (human or LLM) the structural
        anchor that was lost at chunk boundaries.
        """
        recovered = []
        for r in results:
            if r.content and r.content.full_text and r.content.metadata:
                heading = r.content.metadata.get("heading", "")
                text = r.content.full_text
                # Only prepend if the passage doesn't already start with
                # the heading (or a markdown heading marker)
                if heading and not text.lstrip().startswith(("#", heading)):
                    r = MaterialisedResult(
                        source_id=r.source_id,
                        score=r.score,
                        band_scores=r.band_scores,
                        content=SourceContent(
                            source_id=r.content.source_id,
                            summary=r.content.summary,
                            relations=r.content.relations,
                            full_text=f"## {heading}\n\n{text}",
                            metadata=r.content.metadata,
                        ),
                        raw_score=r.raw_score,
                    )
            recovered.append(r)
        return recovered

    def _compute_coverage(
        self,
        query_phase: NDArray[np.float32],
        results: list[MaterialisedResult] | None = None,
    ) -> CoverageProfile:
        """Compute per-band coverage profile from actual retrieval quality.

        Band energies are derived from the top results' per-band scores,
        showing which semantic dimensions contributed to the actual
        matches. Confidence is the mean retrieval score of the top-5
        results — a direct measure of how well the corpus answered the
        query.
        """
        B = self.config.bands
        band_names = list(BAND_NAMES[:B])
        if B > len(BAND_NAMES):
            band_names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))

        # Use top-k results' band_scores for coverage when available.
        # This shows which semantic bands actually contributed to the
        # matches, not abstract field structure.
        top_k = 5
        top_results = (results or [])[:top_k]

        # Collect band scores and reranked scores from top results
        band_arrays = []
        reranked_scores = []
        for r in top_results:
            if r.band_scores is not None and len(r.band_scores) == B:
                band_arrays.append(np.abs(r.band_scores))
            reranked_scores.append(r.score)

        if band_arrays:
            # Mean per-band contribution across top results,
            # scaled by confidence so nonsense queries show low bars.
            band_energies = np.mean(band_arrays, axis=0).astype(np.float32)
            total_energy = float(np.sum(band_energies))
        else:
            # Fallback: field curvature when no band_scores available
            if isinstance(self.field, DenseField):
                from resonance_lattice.calculus import FieldCalculus
                curvature = FieldCalculus.knowledge_curvature(self.field, query_phase)
                band_energies = np.abs(curvature.per_band_curvature).astype(np.float32)
            else:
                resonance = self.field.resonate(query_phase)
                band_energies = np.abs(resonance.band_energies).astype(np.float32)
            total_energy = float(np.sum(band_energies))

        # Confidence: how well did the corpus actually answer this query?
        # Two signals multiplied:
        #   1. Mean reranked score of top results — consistently high
        #      scores mean multiple strong matches (good query). One high
        #      + many low means weak match (bad query).
        #   2. Raw score discrimination — when the field can't separate
        #      candidates (raw scores within a tiny range), the query
        #      didn't match anything meaningful. z-score normalisation
        #      hides this by amplifying noise, so we check the raw CV.
        if reranked_scores and len(reranked_scores) >= 2:
            import math
            # Signal 1: mean reranked score (0-1, higher = better)
            mean_reranked = sum(reranked_scores) / len(reranked_scores)

            # Signal 2: raw score coefficient of variation
            raw_vals = []
            for r in top_results:
                rs = r.raw_score if r.raw_score is not None else r.score
                raw_vals.append(rs)
            raw_mean = sum(raw_vals) / len(raw_vals)
            raw_var = sum((v - raw_mean) ** 2 for v in raw_vals) / len(raw_vals)
            raw_cv = math.sqrt(raw_var) / (abs(raw_mean) + 1e-8)
            # CV < 0.002 means field can't tell candidates apart.
            # Scale so CV=0 → 0, CV≥0.002 → 1.
            discrimination = min(raw_cv / 0.002, 1.0)

            confidence = float(mean_reranked * discrimination)
            confidence = max(0.0, min(1.0, confidence))
        elif reranked_scores:
            confidence = float(reranked_scores[0])
        else:
            confidence = 0.0

        # Scale band energies by confidence so low-quality queries
        # show proportionally low bars.
        band_energies = band_energies * confidence
        total_energy = float(np.sum(band_energies))

        # Identify gaps: bands with energy below a fraction of max.
        gap_threshold = 0.15 if self.source_count < 500 else 0.10
        max_energy = float(np.max(band_energies)) if len(band_energies) > 0 else 0.0
        gaps = []
        if max_energy > 1e-8:
            for name, energy in zip(band_names, band_energies):
                if energy < gap_threshold * max_energy:
                    gaps.append(name)

        return CoverageProfile(
            band_energies=band_energies,
            band_names=band_names,
            total_energy=total_energy,
            confidence=confidence,
            gaps=gaps,
        )

    def _query_cosine(
        self,
        source_id: str,
        query_phase: NDArray[np.float32],
    ) -> float | None:
        """Compute cosine similarity between a source and the original query.

        Uses the phase cache for O(B*D) dot product — no field traversal.
        Returns None if the source is not in the cache (caller should skip
        the gate rather than penalize the candidate).
        """
        entry = self._phase_cache.get(source_id)
        if entry is None:
            return None  # Not in cache — caller should skip the gate
        pvecs, sal = entry
        q_norm = float(np.linalg.norm(query_phase.ravel()))
        p_norm = float(np.linalg.norm(pvecs.ravel()))
        if q_norm < 1e-8 or p_norm < 1e-8:
            return 0.0
        dot = sum(float(np.dot(pvecs[b], query_phase[b])) for b in range(pvecs.shape[0]))
        return dot / (q_norm * p_norm)

    def _compute_related(
        self,
        query_phase: NDArray[np.float32],
        retrieval: RetrievalResult,
        cascade_depth: int,
        min_query_cosine: float = 0.25,
    ) -> list[RelatedTopic]:
        """Discover related topics via multi-hop cascade.

        Args:
            query_phase: Original query phase vectors.
            retrieval: Direct retrieval results (to exclude from related).
            cascade_depth: Number of cascade hops.
            min_query_cosine: Minimum cosine similarity to the **original
                query** in registry phase space.  Candidates below this
                threshold are dropped, even if the cascade found them.
                This prevents generic high-energy nodes from appearing as
                "related" on large corpora.  Set to 0 to disable.
        """
        from resonance_lattice.cascade import ResonanceCascade

        assert isinstance(self.field, DenseField)

        cascade_result = ResonanceCascade.cascade(
            field=self.field,
            query_phase=query_phase,
            depth=cascade_depth,
            alpha=0.1,
        )

        # Build a phase-like query from the indirect cascade hops. Using the
        # original query here just re-runs direct retrieval and produces a tail
        # of barely-related results once the direct matches are excluded.
        indirect_hops = (
            cascade_result.per_hop[1:]
            if len(cascade_result.per_hop) > 1
            else cascade_result.per_hop
        )
        if not indirect_hops:
            return []

        related_phase = np.zeros_like(query_phase)
        start_hop = 2 if len(cascade_result.per_hop) > 1 else 1
        for hop_idx, hop_phase in enumerate(indirect_hops, start=start_hop):
            related_phase += (cascade_result.alpha ** hop_idx) * hop_phase

        if not np.all(np.isfinite(related_phase)):
            return []
        if float(np.linalg.norm(related_phase.ravel())) < 1e-8:
            return []

        # Use the cascade-expanded phase to find sources not in the direct results.
        direct_ids = {r.source_id for r in retrieval.results}
        # Also track source files from main results to deduplicate at file level
        direct_files: set[str] = set()
        for r in retrieval.results:
            if r.content and r.content.metadata:
                sf = r.content.metadata.get("source_file", "")
                if sf:
                    direct_files.add(sf)
        related: list[RelatedTopic] = []

        def _related_terms(source_id: str, content: SourceContent | None) -> set[str]:
            sid_text = source_id.replace("-", " ").replace("_", " ").replace("/", " ")
            parts = [sid_text]
            if content is not None and content.summary:
                parts.append(content.summary)
            return self._tokenize_for_match(" ".join(parts))

        # Build an anchor vocabulary from the direct result neighborhood. A
        # candidate should overlap with the cluster the user is already in,
        # otherwise it is usually just a generic Fabric neighbor.
        from collections import Counter
        anchor_counts: Counter[str] = Counter()
        for result in retrieval.results[:5]:
            anchor_counts.update(_related_terms(result.source_id, result.content))
        anchor_terms = {
            term for term, count in anchor_counts.items()
            if count >= 2
        }
        if not anchor_terms:
            anchor_terms = {
                term for term, _count in anchor_counts.most_common(6)
            }
        min_anchor_support = 2

        # Pull a slightly wider pool, then trim weak-tail candidates. Returning
        # nothing is better than surfacing unrelated noise as "related".
        cascade_pointers = self.registry.lookup_bruteforce(
            query_phase=related_phase,
            top_k=len(direct_ids) + 20,
        )

        non_direct = [
            ptr for ptr in cascade_pointers
            if ptr.source_id not in direct_ids
        ]
        if not non_direct:
            return []

        max_score = max(float(ptr.fidelity_score) for ptr in non_direct)
        score_floor = max(0.0, max_score * 0.25)
        scored_related: list[tuple[tuple[int, float], RelatedTopic]] = []
        for ptr in non_direct:
            if float(ptr.fidelity_score) < score_floor:
                continue

            # ── Cosine gate (#53): require minimum similarity to the
            # original query, not the cascade-expanded phase.  This
            # filters generic high-energy nodes that the cascade finds
            # via field eigenstructure rather than topical relevance.
            # Sources not in the phase cache (e.g. lexically injected)
            # skip the gate rather than being penalized.
            if min_query_cosine > 0:
                cos_sim = self._query_cosine(ptr.source_id, query_phase)
                if cos_sim is not None and cos_sim < min_query_cosine:
                    continue

            content = self.store.retrieve(ptr.source_id)
            # File-level dedup: skip if same source_file as a main result
            if content and content.metadata:
                cand_file = content.metadata.get("source_file", "")
                if cand_file and cand_file in direct_files:
                    continue
            if anchor_terms:
                support = len(anchor_terms & _related_terms(ptr.source_id, content))
                if support < min_anchor_support:
                    continue
            else:
                support = 0
            if ptr.source_id in direct_ids:
                continue
            scored_related.append(((support, float(ptr.fidelity_score)), RelatedTopic(
                source_id=ptr.source_id,
                score=ptr.fidelity_score,
                hop=2,  # These come from the expanded cascade
                content=content,
            )))

        scored_related.sort(key=lambda item: item[0], reverse=True)
        for _rank_key, topic in scored_related:
            related.append(topic)
            if len(related) >= 5:
                break

        return related

    def _normalise_claim_terms(self, text: str) -> set[str]:
        """Extract coarse claim terms for contradiction gating."""
        tokens = set()
        for token in _CLAIM_TOKEN_RE.findall(text.lower()):
            if len(token) < 3 or token in _STOPWORDS:
                continue
            tokens.add(token)
        return tokens

    def _extract_relation_terms(self, content: SourceContent | None) -> set[str]:
        """Extract normalized terms from relation triples."""
        if content is None:
            return set()

        terms: set[str] = set()
        for relation in content.relations:
            for part in relation[:3]:
                terms.update(self._normalise_claim_terms(str(part)))
        return terms

    def _extract_text_terms(self, content: SourceContent | None) -> set[str]:
        """Extract normalized terms from summaries and evidence text."""
        if content is None:
            return set()
        text = " ".join(part for part in (content.summary, content.full_text) if part)
        return self._normalise_claim_terms(text)

    def _compute_contradictions(
        self,
        query_phase: NDArray[np.float32],
        retrieval: RetrievalResult,
        threshold: float,
    ) -> list[ContradictionPair]:
        """Detect contradictions among top-k results via quantum interference."""
        from resonance_lattice.interference import QuantumScorer

        assert isinstance(self.field, DenseField)

        # Gather phase vectors for top results
        source_phases: dict[str, NDArray[np.float32]] = {}
        for r in retrieval.results:
            if r.source_id in self._phase_cache:
                source_phases[r.source_id] = self._phase_cache[r.source_id][0]

        if len(source_phases) < 2:
            return []

        raw_contradictions = QuantumScorer.detect_contradictions(
            field=self.field,
            source_phases=source_phases,
            threshold=threshold,
        )

        content_cache = {
            r.source_id: self.store.retrieve(r.source_id)
            for r in retrieval.results
        }
        relation_terms = {
            source_id: self._extract_relation_terms(content)
            for source_id, content in content_cache.items()
        }
        text_terms = {
            source_id: self._extract_text_terms(content)
            for source_id, content in content_cache.items()
        }
        relation_band_support = {}
        for r in retrieval.results:
            support = 0.0
            if r.band_scores is not None and len(r.band_scores) > 2:
                support = float(max(r.band_scores[2], 0.0))
            relation_band_support[r.source_id] = support

        ranked_pairs: list[tuple[tuple[int, int, float, float], ContradictionPair]] = []
        for c in raw_contradictions:
            content_a = content_cache.get(c.source_a)
            content_b = content_cache.get(c.source_b)
            shared_relation_terms = relation_terms.get(c.source_a, set()) & relation_terms.get(c.source_b, set())
            shared_text_terms = text_terms.get(c.source_a, set()) & text_terms.get(c.source_b, set())

            if not shared_relation_terms and len(shared_text_terms) < 2:
                continue

            support_key = (
                1 if shared_relation_terms else 0,
                len(shared_text_terms),
                relation_band_support.get(c.source_a, 0.0) + relation_band_support.get(c.source_b, 0.0),
                float(c.strength),
            )
            ranked_pairs.append((support_key, ContradictionPair(
                source_a=c.source_a,
                source_b=c.source_b,
                interference=c.interference,
                summary_a=content_a.summary if content_a else "",
                summary_b=content_b.summary if content_b else "",
            )))

        ranked_pairs.sort(key=lambda item: item[0], reverse=True)
        return [pair for _support, pair in ranked_pairs]

    def semantic_profile(self) -> dict[str, Any]:
        """Compute the semantic profile of this lattice.

        Returns per-band analysis: principal dimensions (top eigenvectors),
        effective rank, spectral entropy, topic density, and band energy.
        Only fully supported for DenseField backends.

        Returns:
            Dict with lattice metadata and per-band profiles.
        """
        B = self.config.bands
        band_names = list(BAND_NAMES[:B])
        if B > len(BAND_NAMES):
            band_names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))

        profile: dict[str, Any] = {
            "source_count": self.source_count,
            "bands": B,
            "dim": self.config.dim,
            "field_type": self.config.field_type.value,
            "field_size_mb": round(float(self.field.size_mb), 1),
            "band_profiles": [],
        }

        if not isinstance(self.field, DenseField):
            return profile

        from resonance_lattice.calculus import FieldCalculus

        total_energy = 0.0
        for b in range(B):
            fc = FieldCalculus.field_confidence(self.field, band=b, top_k=20)
            topo = self.eigendecompose(band=b, top_k=10)
            band_energy = float(topo["total_energy"])
            total_energy += band_energy

            profile["band_profiles"].append({
                "band": b,
                "name": band_names[b],
                "energy": round(band_energy, 4),
                "effective_rank": round(float(fc.effective_rank), 1),
                "spectral_entropy": round(float(fc.spectral_entropy), 4),
                "condition_number": round(float(fc.condition_number), 1),
                "top_eigenvalues": [round(float(v), 6) for v in topo["eigenvalues"][:10]],
                "explained_variance_5": round(float(topo["explained_variance"][4]), 4) if len(topo["explained_variance"]) >= 5 else None,
                "explained_variance_10": round(float(topo["explained_variance"][9]), 4) if len(topo["explained_variance"]) >= 10 else None,
            })

        profile["total_energy"] = round(total_energy, 4)
        profile["snr"] = round(float(self.field.compute_snr()), 1)

        # Spectral community detection (Exp 7)
        try:
            communities = self.detect_communities(n_communities=8, band=0)
            profile["communities"] = communities
        except Exception:
            logger.debug("Community detection failed in semantic_profile", exc_info=True)

        return profile

    @staticmethod
    def compare(a: Lattice, b: Lattice) -> dict[str, Any]:
        """Compare two lattices and produce a human-readable diff/intersect report.

        Analyses what field A covers that B doesn't, what they share,
        and where B has more depth than A.

        Both lattices must use DenseField with matching bands and dim.

        Args:
            a: First lattice.
            b: Second lattice.

        Returns:
            Dict with comparison metrics and a text summary.
        """
        if not isinstance(a.field, DenseField) or not isinstance(b.field, DenseField):
            raise TypeError("compare() requires DenseField backends")
        if a.config.bands != b.config.bands or a.config.dim != b.config.dim:
            raise ValueError("Lattices must have matching bands and dim")

        B = a.config.bands
        band_names = list(BAND_NAMES[:B])
        if B > len(BAND_NAMES):
            band_names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))

        comparison: dict[str, Any] = {
            "a_sources": a.source_count,
            "b_sources": b.source_count,
            "bands": B,
            "dim": a.config.dim,
            "per_band": [],
            "summary_lines": [],
        }

        for band in range(B):
            F_a = a.field.F[band]
            F_b = b.field.F[band]

            energy_a = float(np.linalg.norm(F_a, "fro"))
            energy_b = float(np.linalg.norm(F_b, "fro"))

            # Diff energy: what A has that B doesn't (and vice versa)
            diff_ab = F_a - F_b
            diff_ba = F_b - F_a
            diff_energy_ab = float(np.linalg.norm(diff_ab, "fro"))
            diff_energy_ba = float(np.linalg.norm(diff_ba, "fro"))

            # Overlap via normalised Frobenius inner product
            inner = float(np.sum(F_a * F_b))
            overlap = inner / (energy_a * energy_b + 1e-12)

            comparison["per_band"].append({
                "band": band,
                "name": band_names[band],
                "energy_a": round(energy_a, 4),
                "energy_b": round(energy_b, 4),
                "diff_a_minus_b": round(diff_energy_ab, 4),
                "diff_b_minus_a": round(diff_energy_ba, 4),
                "overlap": round(float(overlap), 4),
            })

            # Generate summary line
            if overlap > 0.9:
                comparison["summary_lines"].append(
                    f"  {band_names[band]}: highly similar (overlap {overlap:.0%})"
                )
            elif energy_a > energy_b * 1.5:
                comparison["summary_lines"].append(
                    f"  {band_names[band]}: A has more depth ({energy_a:.1f} vs {energy_b:.1f})"
                )
            elif energy_b > energy_a * 1.5:
                comparison["summary_lines"].append(
                    f"  {band_names[band]}: B has more depth ({energy_b:.1f} vs {energy_a:.1f})"
                )
            else:
                comparison["summary_lines"].append(
                    f"  {band_names[band]}: comparable (overlap {overlap:.0%}, A={energy_a:.1f}, B={energy_b:.1f})"
                )

        return comparison

    def info(self) -> dict[str, Any]:
        """Return summary information about the lattice."""
        info = {
            "source_count": self.source_count,
            "bands": self.config.bands,
            "dim": self.config.dim,
            "field_type": self.config.field_type.value,
            "field_size_mb": self.field.size_mb,
            "field_energy": self.field.energy().tolist(),
            "snr": self.field.compute_snr(),
            "registry_sources": self.registry.source_count,
            "store_sources": self.store.count,
        }

        # Backend-specific info
        if isinstance(self.field, FactoredField):
            info["svd_rank"] = self.config.svd_rank
            info["current_ranks"] = self.field._current_rank.copy()
        elif isinstance(self.field, PQField):
            info["pq_subspaces"] = self.config.pq_subspaces
            info["pq_codebook_size"] = self.config.pq_codebook_size
            info["codebooks_trained"] = self.field.codebooks_trained

        return info
