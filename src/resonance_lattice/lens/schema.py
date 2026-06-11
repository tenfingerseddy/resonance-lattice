"""Lens schema + portable file I/O.

The lens travels. Its schema is designed-portable from day one — no field
references a specific corpus by identifier. Trust weights are keyed by
source-pattern globs (e.g. `docs/*`), not by passage IDs. Insight
preferences are keyed by `insight_id` content hash, which is stable across
archives.

On-disk format: a `.lens` ZIP archive:

  my-engineering-lens.lens (ZIP)
  ├── manifest.json
  ├── stance.md                (optional editorial constitution)
  ├── trust_weights.json
  └── insight_preferences.json

Use `load(path)` / `save(lens, path)` for end-to-end round-trip.
`compose(lenses)` merges multiple lenses into one (team / role
composition).
"""

from __future__ import annotations

import fnmatch
import json
import os
import secrets
import zipfile
from dataclasses import asdict, dataclass, field, fields as _fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


LensScope = Literal["user", "role", "team", "project"]

SCHEMA_VERSION = "1"

# File names inside the .lens ZIP. Single source of truth.
_MANIFEST = "manifest.json"
_STANCE = "stance.md"
_TRUST = "trust_weights.json"
_PREFERENCES = "insight_preferences.json"


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
    """One per-insight preference. Keyed by the corpus claim's
    `content_fingerprint` (a content hash) so the preference re-resolves
    against any corpus that promoted the same insight independently.

    The field is named `insight_id` for lens-file-format stability; the
    value is a corpus claim's `content_fingerprint`."""
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
    scope_metadata: dict[str, Any]   # user_id / role_name / team_id / project_id


_MANIFEST_FIELDS: frozenset[str] = frozenset(f.name for f in _fields(LensManifest))


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

    def preference_for_insight(self, content_fingerprint: str) -> float:
        """Resolve per-insight preference, keyed by a corpus claim's
        `content_fingerprint`. Returns 1.0 if not in preferences."""
        for pref in self.insight_preferences:
            if pref.insight_id == content_fingerprint:
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
            scope_metadata=scope_metadata or {},
        ),
        declared_stance=declared_stance,
    )


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def save(lens: Lens, path: str | Path) -> None:
    """Serialise the lens to a `.lens` ZIP at `path`. Atomic write via a
    per-writer-unique tmp file (pid + random suffix) + `os.replace`, so
    two processes saving the same lens don't collide on `{path}.tmp`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{p}.{os.getpid()}.{secrets.token_hex(4)}.tmp")

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
        os.replace(tmp, p)
    except BaseException:
        # Clean up on any failure (write, close, or `os.replace`) so a
        # Windows concurrent-writer collision doesn't leak an orphan tmp.
        tmp.unlink(missing_ok=True)
        raise


def load(path: str | Path) -> Lens:
    """Load a `.lens` ZIP into a Lens object.

    Schema version mismatch raises. Missing optional sections are silently
    treated as empty. Unknown manifest keys are ignored — accommodates
    .lens files written by earlier development snapshots without
    requiring the loader to track every dropped field.
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
        manifest = LensManifest(**{
            k: v for k, v in manifest_raw.items() if k in _MANIFEST_FIELDS
        })

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

    return Lens(
        manifest=manifest,
        declared_stance=stance,
        trust_weights=trust_weights,
        insight_preferences=prefs,
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
            scope_metadata={"composed_from": [li.manifest.lens_id for li in lenses]},
        ),
        declared_stance=composed_stance,
        trust_weights=trust,
        insight_preferences=prefs,
    )
