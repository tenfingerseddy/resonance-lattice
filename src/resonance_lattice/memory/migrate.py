"""v2.0 LayeredMemory → v2.1 flat-memory one-shot migration.

Per §14.3-14.5 of `.claude/plans/fabric-agent-flat-memory.md`:

- One-way (no two-way bridge — see §14.3 rationale).
- Opt-in (CLI subcommand the user invokes deliberately).
- Lossy by design — the v2.0 → v2.1 axis change is not zero-info-loss
  by construction. The honest list of what's lost is in §14.5.
- Deleted in v2.2 alongside the v2.0 module files (§14.7 timeline).

The polarity heuristic targets ~70% accuracy on natural English; the
balance falls to `--polarity-default` (default `factual`). See §14.5
contract 3 for the calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import Encoder
from ._common import utcnow_iso, workspace_tag_for_cwd
from .store import Memory

# Override via --polarity-default to tilt a preference-heavy v2.0 corpus.
_AVOID_VERBS = re.compile(
    r"\b(avoid|don'?t|never|stop|skip|disable|prohibit|forbid)\b",
    re.IGNORECASE,
)
_PREFER_VERBS = re.compile(
    r"\b(prefer|want|like|use|always|enable|favo[u]?r|recommend)\b",
    re.IGNORECASE,
)

# Order matters for the migrate summary.
_V20_TIERS = ("working", "episodic", "semantic")


@dataclass
class MigrateResult:
    """Outcome of one `migrate(...)` invocation. Returned even on dry-run."""

    rows_migrated: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    by_polarity: dict[str, int] = field(default_factory=dict)
    archived_path: Path | None = None
    dry_run: bool = False

    def summary(self) -> str:
        """Human-readable line for stderr / CLI output."""
        prefix = "would migrate" if self.dry_run else "migrated"
        tier_breakdown = ", ".join(
            f"{tier}: {self.by_tier.get(tier, 0)}" for tier in _V20_TIERS
        )
        pol_breakdown = ", ".join(
            f"{p}: {n}" for p, n in sorted(self.by_polarity.items())
        )
        archived = (
            f"; archived v2.0 root → {self.archived_path}"
            if self.archived_path is not None
            else ""
        )
        return (
            f"[rlat memory migrate] {prefix} {self.rows_migrated} row(s) "
            f"({tier_breakdown}) → polarity ({pol_breakdown}){archived}"
        )


def classify_polarity(text: str, default: str = "factual") -> str:
    """Verb-scan polarity heuristic (§14.4 step 2)."""
    if _AVOID_VERBS.search(text):
        return "avoid"
    if _PREFER_VERBS.search(text):
        return "prefer"
    return default


def migrate(
    v20_root: Path,
    *,
    v21_root: Path,
    user_id: str,
    encoder: Encoder | None = None,
    dry_run: bool = False,
    polarity_default: str = "factual",
    workspace_path: str | None = None,
) -> MigrateResult:
    """Migrate `v20_root` (v2.0 LayeredMemory tree) into the v2.1 flat
    store at `<v21_root>/<user_id>/`.

    - `dry_run=True` prints the summary without writing the v2.1 band
      or moving the v2.0 root. **Recommended first invocation.**
    - On real run, the v2.0 root is moved to `<v20_root>.archived` so
      subsequent v2.0 CLI calls fail loudly.

    The v2.0 → v2.1 encoder is the same `gte-modernbert-base` pinned
    revision; we re-encode rather than copy embeddings because v2.0's
    per-tier `.npy` shape doesn't compose into the v2.1 single-band
    layout — re-encoding is cheap (band is bit-equivalent) and keeps
    the migration explicit.
    """
    from .layered import LayeredMemory

    layered = LayeredMemory(v20_root, encoder=encoder)
    target_dir = v21_root / user_id
    workspace_tag = workspace_tag_for_cwd(workspace_path)

    # Pre-flight idempotency: if the v2.1 target already has rows AND the
    # v2.0 source is still in place (rename hadn't happened), this is a
    # retry of a partial migration — re-running would duplicate every row.
    # Refuse unless dry-run. Recovery: user inspects + deletes the partial
    # v2.1 store, OR finishes the rename manually then re-runs.
    archived_target = v20_root.with_suffix(v20_root.suffix + ".archived")
    if not dry_run:
        sidecar = target_dir / "sidecar.jsonl"
        if sidecar.exists() and sidecar.stat().st_size > 0:
            raise FileExistsError(
                f"v2.1 target {target_dir} already populated; refusing to "
                f"re-run migrate (would duplicate rows). Inspect the "
                f"sidecar and delete it (or pass a different --to) if "
                f"this is a retry after a previous failure."
            )
        # Pre-flight the archive rename target too — if it exists, the
        # final rename would fail AFTER rows are written, leaving a
        # partial v2.1 store + intact v2.0 source that the sidecar guard
        # then refuses to re-run. Cheaper to refuse before any write.
        if archived_target.exists():
            raise FileExistsError(
                f"v2.0 archive target {archived_target} already exists; "
                f"refusing to migrate (final rename would fail after the "
                f"v2.1 rows are written). Move or delete it first."
            )

    result = MigrateResult(dry_run=dry_run)

    # Collect all rows across tiers in memory before writing — single
    # batch into Memory.add_rows_batch avoids the O(N²) per-row sidecar
    # re-read that `add_row` would do for N v2.0 rows.
    pending_texts: list[str] = []
    pending_meta: list[dict] = []

    for tier in _V20_TIERS:
        entries = layered.all_entries(tier)
        result.by_tier[tier] = len(entries)
        for entry in entries:
            primary = classify_polarity(entry.text, default=polarity_default)
            result.by_polarity[primary] = result.by_polarity.get(primary, 0) + 1
            result.rows_migrated += 1
            if dry_run:
                continue
            pending_texts.append(entry.text)
            pending_meta.append({
                "text": entry.text,
                "polarity": [primary, workspace_tag],
                "transcript_hash": f"migrated:v2.0:{tier}",
            })

    if not dry_run and pending_meta:
        live_encoder = layered._ensure_encoder()  # type: ignore[attr-defined]
        # Always re-encode per §14.4 — bit-equivalent under the
        # PINNED_REVISION pin, but the explicit re-encode keeps the
        # migration pipeline honest about its encoder dependency.
        embeddings = live_encoder.encode(pending_texts)
        l2_normalize(embeddings)
        store = Memory(root=target_dir, encoder=live_encoder)
        store.add_rows_batch(pending_meta, embeddings)

    if not dry_run and result.rows_migrated > 0:
        # Atomic rename — v2.0 CLI calls against the original path now
        # FileNotFoundError, surfacing the migration to the user. Target
        # path was pre-flighted above so this rename can't collide with
        # an existing archive.
        v20_root.rename(archived_target)
        result.archived_path = archived_target

    return result
