"""The unified claim record — Phase D of the Memory + Lens convergence.

A `Claim` is the one record type for earned knowledge, whatever its
source: a pattern distilled from the agent's own sessions, or an insight
synthesised from the corpus. It replaces the separate `memory.store.Row`
and `store.insight.InsightPassage`. The full design is
`docs/internal/claim-system-design.md`.

A claim has a **core** — identity, content, Beta trust, lifecycle state,
derivation — that the unified lifecycle, the outcome loop, and recall all
operate on. The genuinely source-specific signals live in a **typed**
`facts` record, discriminated by `source`: `ExperienceFacts` or
`CorpusFacts`. `facts` is two small dataclasses, never an untyped dict.

Experience claims live in `ExperienceClaimStore` — a per-user
filesystem directory. Corpus claims live inside one `.rlat` archive's
insight layer (`ArchiveContents.insights: list[Claim]`), serialised by
`store.corpus_claim_io`. Both share the JSONL row shape and the
core-field serialisers in `state/claim_io.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # annotations only — no runtime import, no import cycle
    from ..memory.store import Criticality, IntentKind, Origin
    from ..store.insight import InsightCitation, VerdictSignal

# Where a claim came from — a provenance tag, not a separate system.
ClaimSource = Literal["experience", "corpus"]

# The writer/owner of a claim or outcome — the H3 (team) identity axis,
# present from v1 so the team horizon is an unlock, not a migration of live
# earned data (architecture §3, invariant 7). v1 is single-writer: every
# claim and outcome defaults to this sentinel until a real identity system
# (H3) sets it.
DEFAULT_WRITER = "local"

# The lifecycle-state axis (distinct from trust, which is the Beta score).
# Experience claims are born `active`; corpus claims born `candidate`.
ClaimState = Literal["candidate", "active", "stale", "retired"]

# State-set partitions over `ClaimState` — the single source of truth for
# the membership checks retrieval, the lifecycle, and verdict routing share.
#   retrievable — surfaces in default retrieval.
#   pending     — on the path to active or retired; surfaces only opt-in
#                 (e.g. `--include-stale`), not yet authoritative.
#   final       — absorbing; never leaves, never surfaces. A corpus reject
#                 and a corpus correction both land here — the distinction
#                 is carried by `parent_ids` + the preserved verdict signals.
RETRIEVABLE_STATES: frozenset[str] = frozenset({"active"})
PENDING_STATES: frozenset[str] = frozenset({"candidate", "stale"})
FINAL_STATES: frozenset[str] = frozenset({"retired"})

# What role the claim plays — the union of the two source vocabularies.
ClaimKind = Literal[
    "event", "attribute", "constraint",                 # experience
    "synthesis", "faq", "mechanism", "boundary",        # corpus
    "negation", "gap",
]

# The `kind` values an experience claim may carry — the three serve-proven
# world content classes plus raw events. `attribute` = a stable fact about
# the world the knowledge model covers (capacity/version/region/policy),
# captured single-shot on first mention — NOT recurrence-gated (recurrence
# is structurally dead for attributes, 0/259; criticality+trust carry it).
# `constraint` = a standing hard rule of that world, served ALL-always (R1:
# blind 62% violation -> served 7%, zero over-blocking). `negation` = a
# tried-and-falsified finding whose VERDICT is the active ingredient (R2:
# 86pp over a topical control); its evidence pointer lives in the content
# text. All three share the attribute capture path: born active, newest-wins
# dedup by `attribute_key`.
EXPERIENCE_KINDS: frozenset[str] = frozenset(
    {"event", "attribute", "constraint", "negation"})

# Polarity is `[primary, *scope_tags]`. The single primary tag is one of
# these; scope tags are `workspace:<hash>` / `cross-workspace`.
PrimaryPolarity = Literal["prefer", "avoid", "factual"]
PRIMARY_POLARITY: frozenset[str] = frozenset({"prefer", "avoid", "factual"})

# `transcript_hash` discriminators. A manual claim carries the literal
# string; a distil-emitted claim carries `<DISTILLED_PREFIX><source_sha>`;
# a v2.0-migrated claim carries `<MIGRATED_PREFIX><tier>`. All three are
# excluded from the distil-input filter so the distiller never reprocesses
# its own output or a migrated claim as raw capture.
MANUAL_TRANSCRIPT_HASH = "manual"
DISTILLED_PREFIX = "distilled:"
MIGRATED_PREFIX = "migrated:"


def derive_origin(transcript_hash: str) -> Origin:
    """The `origin` tag implied by a `transcript_hash` prefix."""
    if transcript_hash.startswith(DISTILLED_PREFIX):
        return "distilled"
    if transcript_hash.startswith(MIGRATED_PREFIX):
        return "migrated"
    return "manual"


@dataclass(frozen=True)
class ExperienceFacts:
    """Source-specific facts for an `experience` claim — the signals the
    experience backend and the experience-side recall gates need
    (workspace scope, recurrence, bad-vote suppression)."""

    polarity: tuple[str, ...]
    recurrence_count: int
    criticality: Criticality
    created_under_intent_kind: IntentKind
    transcript_hash: str
    origin: Origin
    last_corroborated_at: str
    is_bad: bool = False
    # Normalized subject for `attribute` claims (e.g. "powershell version") —
    # the serve-time newest-wins dedup groups by this, never the value. "" = no
    # declared identity → never deduped (a keyless fact can't suppress another).
    # Defaulted + last so older insight.jsonl / experience-store rows load.
    attribute_key: str = ""

    def primary_polarity(self) -> str:
        """The single primary polarity tag — exactly one is present."""
        return next(p for p in self.polarity if p in PRIMARY_POLARITY)

    def is_manual(self) -> bool:
        return self.transcript_hash == MANUAL_TRANSCRIPT_HASH

    def is_distilled(self) -> bool:
        return self.transcript_hash.startswith(DISTILLED_PREFIX)

    def is_migrated(self) -> bool:
        return self.transcript_hash.startswith(MIGRATED_PREFIX)


@dataclass(frozen=True)
class CorpusFacts:
    """Source-specific facts for a `corpus` claim — citations into the
    archive source layer, drift hashes, verdict history, and the
    `content_fingerprint` a portable lens keys preferences on."""

    citations: tuple[InsightCitation, ...]
    content_fingerprint: str
    source_model_hash: str
    source_passage_hashes: tuple[str, ...]
    verdict_signals: tuple[VerdictSignal, ...] = ()
    query: str | None = None
    intent_context: str | None = None
    stale_if_sources_drift: bool = True
    encoder_version: str = ""
    # The born Beta prior (the `seed_confidence(faithfulness)` values set at
    # promotion). IMMUTABLE — the attribution apply mutates `corroboration` /
    # `falsification` but never these, so `consolidate-insights` can re-derive
    # the absolute tally as `seed + full-ledger weight` each run instead of
    # accumulating onto the prior result (idempotency — §B BLOCKER). Defaulted
    # to a negative sentinel meaning "unseeded"; a real seed is always ≥ 1.0
    # (`_PRIOR_BASE`), so the apply falls back to additive only for a claim that
    # never recorded one. Defaulted + last so older `insight.jsonl` rows load.
    seed_corroboration: float = -1.0
    seed_falsification: float = -1.0


@dataclass(frozen=True)
class Claim:
    """One earned-knowledge claim — experience or corpus.

    The core fields are the converged contract. `facts` carries the
    source-specific remainder, typed and discriminated by `source`:
    `source == "experience"` ⇒ `facts` is `ExperienceFacts`,
    `"corpus"` ⇒ `CorpusFacts`. `trust` derives from the Beta tallies —
    the one trust math shared across both sources.
    """

    claim_id: str
    source: ClaimSource
    kind: ClaimKind
    content: str
    created_at: str
    corroboration: float
    falsification: float
    trust_as_of: str
    state: ClaimState
    parent_ids: tuple[str, ...]
    facts: ExperienceFacts | CorpusFacts
    # H3 identity (defaulted single-writer in v1) — who wrote this claim.
    # Defaulted + last so older `insight.jsonl` rows lacking it still load
    # (`core_from_row` falls back to the default).
    writer: str = DEFAULT_WRITER

    def __post_init__(self) -> None:
        """Enforce the discrimination invariant — `source` must be a known
        value, and `facts` must match it. Without this the union is only
        checker-deep; the runtime check makes a mismatched or
        unknown-source `Claim` impossible to construct."""
        if self.source == "experience":
            expected: type = ExperienceFacts
        elif self.source == "corpus":
            expected = CorpusFacts
        else:
            raise ValueError(f"unknown claim source: {self.source!r}")
        if not isinstance(self.facts, expected):
            raise TypeError(
                f"source={self.source!r} requires {expected.__name__}, "
                f"got {type(self.facts).__name__}"
            )

    @property
    def trust(self) -> float:
        """Beta-mean trust over the corroboration / falsification tallies.
        Imported here, not at module scope, to keep `state` free of the
        `store` package's import cost."""
        from ..store.insight import beta_mean
        return beta_mean(self.corroboration, self.falsification)

    @property
    def confidence(self) -> str:
        """The 4-rung confidence label — `low | medium | high | verified`.
        A derived view over `trust`, never a stored field (design §7)."""
        from ..store.insight import confidence_band
        return confidence_band(self.trust)

    @property
    def content_key(self) -> str:
        """Corpus-agnostic identity — a stable 16-hex hash over `content`
        alone, so the same earned text in two corpora shares a key (the H3
        cross-corpus consensus join, invariant 7). Derived, never stored:
        `content` is the source of truth and the key tracks it through
        `evolve`. Distinct from `CorpusFacts.content_fingerprint`, which also
        folds in corpus-specific source hashes and so differs per corpus."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def is_retrievable(self) -> bool:
        """Whether this claim surfaces in default retrieval — `state` is
        `active`. `candidate` / `stale` surface only opt-in; `retired`
        never does."""
        return self.state in RETRIEVABLE_STATES


_EVOLVE_IMMUTABLE: frozenset[str] = frozenset(
    {"claim_id", "created_at", "source", "writer"}
)


def evolve(claim: Claim, **changes: object) -> Claim:
    """Return a copy of `claim` with `changes` applied. Each keyword is
    routed by name to the core record or the typed `facts` sub-record,
    a field merge over a frozen record. Pass `facts` fields directly,
    never `facts` itself.

    Identity and provenance — `claim_id`, `created_at`, `source`, `writer`
    — are immutable; passing one raises (changing `claim_id` would silently
    fork the claim's provenance, re-attributing a claim's `writer` would
    forge ownership)."""
    import dataclasses

    if "facts" in changes:
        raise ValueError("evolve: pass facts fields directly, not `facts`")
    frozen = _EVOLVE_IMMUTABLE & changes.keys()
    if frozen:
        raise ValueError(
            f"evolve: cannot change immutable field(s) {sorted(frozen)}"
        )
    core_names = {f.name for f in dataclasses.fields(Claim)} - {"facts"}
    facts_names = {f.name for f in dataclasses.fields(claim.facts)}
    core: dict = {}
    facts: dict = {}
    for key, value in changes.items():
        if key in core_names:
            core[key] = value
        elif key in facts_names:
            facts[key] = value
        else:
            raise ValueError(f"evolve: unknown claim field {key!r}")
    new_facts = (
        dataclasses.replace(claim.facts, **facts) if facts else claim.facts
    )
    return dataclasses.replace(claim, facts=new_facts, **core)
