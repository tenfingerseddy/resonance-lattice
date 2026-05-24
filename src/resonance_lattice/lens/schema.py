"""Lens schema + portable file I/O.

The lens travels. Its schema is designed-portable from day one — no field
references a specific corpus by identifier. Trust weights are keyed by
source-pattern globs (e.g. `docs/*`), not by passage IDs. Insight
preferences are keyed by `insight_id` content hash, which is stable across
archives. Private insights embed in the same 768d encoder space as any
corpus the lens loads against.

On-disk format: a `.lens` ZIP archive (architecture §5.1 Option B):

  my-engineering-lens.lens (ZIP)
  ├── manifest.json
  ├── stance.md                (optional editorial constitution)
  ├── memory.jsonl             (lens-scoped memory rows)
  ├── intent_history.jsonl
  ├── verdict_log.jsonl
  ├── trust_weights.json
  ├── insight_preferences.json
  ├── private_insights.jsonl
  └── bands/
      └── private_insights.npz

Use `load(path)` / `save(lens, path)` for end-to-end round-trip.
`compose(lenses)` merges multiple lenses into one (team / role
composition).
"""

from __future__ import annotations

import fnmatch
import json
import os
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from ..store import bands as bands_io
from ..store.insight import InsightPassage, VerdictSignal
from ..store.insight import load_jsonl as insight_load_jsonl
from ..store.insight import write_jsonl as insight_write_jsonl

LensScope = Literal["user", "role", "team", "project"]

SCHEMA_VERSION = "1"

# File names inside the .lens ZIP. Single source of truth.
_MANIFEST = "manifest.json"
_STANCE = "stance.md"
_TRUST = "trust_weights.json"
_PREFERENCES = "insight_preferences.json"
_MEMORY = "memory.jsonl"
_INTENT_HISTORY = "intent_history.jsonl"
_VERDICT_LOG = "verdict_log.jsonl"
_PRIVATE_INSIGHTS = "private_insights.jsonl"
_PRIVATE_BAND = "bands/private_insights.npz"


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrustWeight:
    """One trust adjustment for retrieval re-ranking.

    `pattern` is a glob over source-file paths (e.g. `src/*`,
    `docs/external/*`). Patterns are corpus-agnostic — the same pattern
    re-resolves against any corpus the lens loads.

    `weight` multiplies a hit's cosine score; >1 boosts, <1 suppresses,
    0 effectively excludes. Identity is 1.0.
    """
    pattern: str
    weight: float


@dataclass(frozen=True)
class InsightPreference:
    """One per-insight preference. Keyed by insight content hash so the
    preference re-resolves against any corpus that promoted the same
    insight independently."""
    insight_id: str
    weight: float


@dataclass(frozen=True)
class LensManifest:
    """The lens's identifying metadata. Carried in `manifest.json` and
    designed-portable: every field is either a stable id, a corpus-
    agnostic human label, or a schema version."""
    lens_id: str
    scope: LensScope
    name: str
    description: str | None
    created_at: str
    last_active: str
    schema_version: str
    encoder_version: str | None      # populated if private_insights exist
    scope_metadata: dict[str, Any]   # user_id / role_name / team_id / project_id


@dataclass
class Lens:
    """The complete lens object — designed-portable across corpora.

    Frozen-ish at the outer level (replace() rebuilds), but the
    collections inside are mutable lists/tuples to keep the construction
    pipeline ergonomic. Save/load is by value, so any in-flight mutation
    only affects in-memory state.
    """
    manifest: LensManifest
    declared_stance: str | None = None
    trust_weights: list[TrustWeight] = field(default_factory=list)
    insight_preferences: list[InsightPreference] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)
    intent_history: list[dict[str, Any]] = field(default_factory=list)
    verdict_log: list[VerdictSignal] = field(default_factory=list)
    private_insights: list[InsightPassage] = field(default_factory=list)
    private_insights_band: np.ndarray | None = None

    # ----- lookup helpers used at retrieval time -----------------------

    def trust_for_source(self, source_file: str) -> float:
        """Resolve trust weight for a given source path.

        Patterns are matched in declaration order; the first match wins.
        Returns 1.0 (identity) when no pattern matches. Glob semantics
        via `fnmatch`.
        """
        for tw in self.trust_weights:
            if fnmatch.fnmatch(source_file, tw.pattern):
                return tw.weight
        return 1.0

    def preference_for_insight(self, insight_id: str) -> float:
        """Resolve per-insight preference. Returns 1.0 if not in preferences."""
        for pref in self.insight_preferences:
            if pref.insight_id == insight_id:
                return pref.weight
        return 1.0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def new_lens(
    *,
    lens_id: str,
    scope: LensScope,
    name: str,
    description: str | None = None,
    declared_stance: str | None = None,
    scope_metadata: dict[str, Any] | None = None,
) -> Lens:
    """Construct a fresh empty lens with manifest stamped at now()."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Lens(
        manifest=LensManifest(
            lens_id=lens_id,
            scope=scope,
            name=name,
            description=description,
            created_at=now,
            last_active=now,
            schema_version=SCHEMA_VERSION,
            encoder_version=None,
            scope_metadata=scope_metadata or {},
        ),
        declared_stance=declared_stance,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _verdict_log_to_jsonl(signals: Iterable[VerdictSignal]) -> str:
    return "\n".join(json.dumps({
        "source": s.source, "polarity": s.polarity,
        "timestamp": s.timestamp, "lens_id": s.lens_id,
    }, sort_keys=True) for s in signals)


def _verdict_log_from_jsonl(text: str) -> list[VerdictSignal]:
    out: list[VerdictSignal] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(VerdictSignal(
            source=d["source"], polarity=d["polarity"],
            timestamp=d["timestamp"], lens_id=d.get("lens_id"),
        ))
    return out


def _row_jsonl_dump(rows: Iterable[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, sort_keys=True) for r in rows)


def _row_jsonl_load(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def save(lens: Lens, path: str | Path) -> None:
    """Serialise the lens to a `.lens` ZIP at `path`. Atomic write via
    tmp file + `os.replace`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")

    # Stamp last_active on every save so consumers can dedupe stale
    # copies on disk.
    from dataclasses import replace as _replace
    lens = _replace(
        lens,
        manifest=_replace(lens.manifest,
                          last_active=datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )

    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(_MANIFEST, json.dumps(asdict(lens.manifest), indent=2, sort_keys=True))
            if lens.declared_stance is not None:
                zf.writestr(_STANCE, lens.declared_stance)
            if lens.trust_weights:
                zf.writestr(_TRUST, json.dumps(
                    [asdict(tw) for tw in lens.trust_weights],
                    indent=2, sort_keys=True,
                ))
            if lens.insight_preferences:
                zf.writestr(_PREFERENCES, json.dumps(
                    [asdict(p) for p in lens.insight_preferences],
                    indent=2, sort_keys=True,
                ))
            if lens.memory:
                zf.writestr(_MEMORY, _row_jsonl_dump(lens.memory))
            if lens.intent_history:
                zf.writestr(_INTENT_HISTORY, _row_jsonl_dump(lens.intent_history))
            if lens.verdict_log:
                zf.writestr(_VERDICT_LOG, _verdict_log_to_jsonl(lens.verdict_log))
            if lens.private_insights:
                zf.writestr(_PRIVATE_INSIGHTS, insight_write_jsonl(lens.private_insights))
                if lens.private_insights_band is None:
                    raise ValueError(
                        "lens has private_insights but no private_insights_band "
                        f"(need a ({len(lens.private_insights)}, D) array)"
                    )
                if lens.private_insights_band.shape[0] != len(lens.private_insights):
                    raise ValueError(
                        f"private_insights band has {lens.private_insights_band.shape[0]} "
                        f"rows but private_insights list has {len(lens.private_insights)}"
                    )
                bands_io.write_band(zf, _PRIVATE_BAND, lens.private_insights_band)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    os.replace(tmp, p)


def load(path: str | Path) -> Lens:
    """Load a `.lens` ZIP into a Lens object.

    Schema version mismatch raises. Missing optional sections are silently
    treated as empty.
    """
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        manifest_text = zf.read(_MANIFEST).decode("utf-8")
        manifest_raw = json.loads(manifest_text)
        if manifest_raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{p}: lens schema_version is {manifest_raw.get('schema_version')!r}, "
                f"this build only reads {SCHEMA_VERSION!r}"
            )
        manifest = LensManifest(**manifest_raw)

        names = set(zf.namelist())

        stance = (
            zf.read(_STANCE).decode("utf-8") if _STANCE in names else None
        )

        trust_weights: list[TrustWeight] = []
        if _TRUST in names:
            trust_weights = [
                TrustWeight(**d) for d in json.loads(zf.read(_TRUST).decode("utf-8"))
            ]

        prefs: list[InsightPreference] = []
        if _PREFERENCES in names:
            prefs = [
                InsightPreference(**d)
                for d in json.loads(zf.read(_PREFERENCES).decode("utf-8"))
            ]

        memory = _row_jsonl_load(zf.read(_MEMORY).decode("utf-8")) \
                 if _MEMORY in names else []
        intent_history = _row_jsonl_load(zf.read(_INTENT_HISTORY).decode("utf-8")) \
                         if _INTENT_HISTORY in names else []
        verdict_log = _verdict_log_from_jsonl(zf.read(_VERDICT_LOG).decode("utf-8")) \
                      if _VERDICT_LOG in names else []

        private_insights: list[InsightPassage] = []
        private_band: np.ndarray | None = None
        if _PRIVATE_INSIGHTS in names:
            text = zf.read(_PRIVATE_INSIGHTS).decode("utf-8")
            if text.strip():
                private_insights = insight_load_jsonl(text.splitlines())
        if _PRIVATE_BAND in names:
            private_band = bands_io.load_base(zf, _PRIVATE_BAND)
            if private_insights and private_band.shape[0] != len(private_insights):
                raise ValueError(
                    f"{p}: private_insights band has {private_band.shape[0]} rows "
                    f"but private_insights.jsonl has {len(private_insights)}"
                )
        if private_insights and private_band is None:
            raise ValueError(
                f"{p}: private_insights.jsonl present but bands/private_insights.npz "
                f"missing — lens is half-written"
            )

    return Lens(
        manifest=manifest,
        declared_stance=stance,
        trust_weights=trust_weights,
        insight_preferences=prefs,
        memory=memory,
        intent_history=intent_history,
        verdict_log=verdict_log,
        private_insights=private_insights,
        private_insights_band=private_band,
    )


# ---------------------------------------------------------------------------
# Composition (team / role lens overlay)
# ---------------------------------------------------------------------------

def compose(lenses: list[Lens], composed_id: str, name: str,
            *, scope: LensScope = "team") -> Lens:
    """Combine multiple lenses into one team/role/composed lens.

    Composition rules:
    - Manifest: new lens_id + name; created_at = now.
    - Declared stance: concatenated with markdown headings if any present.
    - Trust weights: union (later lenses' patterns shadow earlier ones
      for the same pattern string).
    - Insight preferences: union; on collision, the average of weights
      (a team's collective preference rather than one member's).
    - Memory + intent_history + verdict_log: concatenated in input order.
    - Private insights: union by `insight_id`; on collision, the first
      lens's row wins (consumers can re-promote).
    - Private band: concatenated. Encoder version must match across all
      input lenses with non-empty private insights; mismatch raises.

    Returns the composed Lens. The input lenses are not mutated.
    """
    if not lenses:
        raise ValueError("compose() requires at least one input lens")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Stance: stitch each contributing lens's stance under a heading.
    stance_parts: list[str] = []
    for li in lenses:
        if li.declared_stance:
            stance_parts.append(f"# From {li.manifest.name}\n\n{li.declared_stance}")
    composed_stance = "\n\n---\n\n".join(stance_parts) if stance_parts else None

    # Trust weights: union by pattern; later wins on collision.
    tw_by_pattern: dict[str, TrustWeight] = {}
    for li in lenses:
        for tw in li.trust_weights:
            tw_by_pattern[tw.pattern] = tw
    trust = sorted(tw_by_pattern.values(), key=lambda x: x.pattern)

    # Insight preferences: average on collision.
    pref_acc: dict[str, list[float]] = {}
    for li in lenses:
        for pref in li.insight_preferences:
            pref_acc.setdefault(pref.insight_id, []).append(pref.weight)
    prefs = sorted(
        (InsightPreference(insight_id=k, weight=sum(v) / len(v))
         for k, v in pref_acc.items()),
        key=lambda x: x.insight_id,
    )

    memory = [r for li in lenses for r in li.memory]
    intent_history = [r for li in lenses for r in li.intent_history]
    verdict_log = [s for li in lenses for s in li.verdict_log]

    # Private insights: union by insight_id, first-wins.
    seen_ids: set[str] = set()
    private_insights: list[InsightPassage] = []
    band_rows: list[np.ndarray] = []
    encoder_version: str | None = None
    for li in lenses:
        if li.private_insights_band is None:
            continue
        if encoder_version is None:
            encoder_version = li.manifest.encoder_version
        elif (li.manifest.encoder_version is not None
              and encoder_version != li.manifest.encoder_version):
            raise ValueError(
                f"compose(): encoder version mismatch — "
                f"{encoder_version!r} vs {li.manifest.encoder_version!r}. "
                f"Composed lens requires uniform encoder version across inputs."
            )
        for row_idx, ins in enumerate(li.private_insights):
            if ins.insight_id in seen_ids:
                continue
            seen_ids.add(ins.insight_id)
            # Re-stamp insight_idx to match the new composed order.
            from dataclasses import replace as _replace
            private_insights.append(_replace(ins, insight_idx=len(private_insights)))
            band_rows.append(li.private_insights_band[row_idx])
    private_band = np.stack(band_rows, axis=0) if band_rows else None

    return Lens(
        manifest=LensManifest(
            lens_id=composed_id,
            scope=scope,
            name=name,
            description=f"Composed of {len(lenses)} lens(es): " +
                        ", ".join(li.manifest.name for li in lenses),
            created_at=now,
            last_active=now,
            schema_version=SCHEMA_VERSION,
            encoder_version=encoder_version,
            scope_metadata={"composed_from": [li.manifest.lens_id for li in lenses]},
        ),
        declared_stance=composed_stance,
        trust_weights=trust,
        insight_preferences=prefs,
        memory=memory,
        intent_history=intent_history,
        verdict_log=verdict_log,
        private_insights=private_insights,
        private_insights_band=private_band,
    )
