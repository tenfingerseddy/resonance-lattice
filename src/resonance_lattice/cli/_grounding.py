"""Grounding modes for LLM-facing markdown emitters.

Three modes for how the consumer LLM should treat retrieved passages:

  augment    — passages are primary context; prefer them over training
               knowledge when the two conflict. **Default mode**.
               Suppresses on genuinely off-corpus retrieval (top1_score
               < 0.30) or heavy drift (drift_fraction > 0.30) so the
               LLM falls back to training instead of grounding on
               noise. Bench 2 v3 (Fabric corpus): 54.9% answerable
               accuracy, 3.9% hallucination, 67% distractor refusal —
               the value default for broad domain corpora the LLM
               partially knows.
  knowledge  — passages supplement training. Lighter gate
               (top1_score < 0.15 only — drift is allowed) because the
               directive already tells the LLM to lean on training.
               Choose this for partial-coverage corpora where the LLM
               already knows the surrounding domain reasonably well.
  constrain  — passages are the ONLY source of truth. No suppression;
               the consumer LLM is instructed to refuse when passages
               are thin. Bench 2 v3 (Fabric): 45.1% accuracy / 2.0%
               hallucination — trades 9.8 pp accuracy for halving the
               hallucination rate vs augment. Recommended for fact-
               extraction / compliance / regulatory / audit work where
               wrong-but-confident is worse than no answer.

Used by `cli.skill_context` and `cli.search --format context`. The
mode header is stamped at the top of the markdown output as a directive
to the consumer LLM. When the gate fires, the dynamic body is replaced
with a marker but the mode header still ships so the directive is
unambiguous.

The modes are CONSUMER-instructions (telling the LLM how to read the
output), not retrieval knobs — they don't change what `rlat` retrieves,
only how it labels the result and whether it gates a low-confidence
body. This keeps the single-recipe retrieval thesis intact.

Bench 2 evidence (2026-04-27): the previous gap-based augment gate
(`top1_top2_gap < 0.05`) over-suppressed on paraphrase-rich corpora
because top-1 ≈ top-2 is the SIGNAL of strong retrieval (same fact
stated multiple ways), not weak retrieval. Switching to absolute-score
floor fixed the inversion. Default is **augment** — bench 2 v3 +
5-lane Fabric bench (relaxed rubric) showed augment 74.5% accuracy /
3.9% hallucination beats constrain on accuracy at single-digit
hallucination, so augment is the right default for broad documentation
corpora; constrain (2.0% hallucination, 9.8 pp lower accuracy) is the
opt-in for compliance / audit / regulatory work where wrong-but-
confident is worse than no answer.

Distinctive-name verification is wired in lockstep at the
emit boundary (`cli/_namecheck.py`) — when a question references a
proper noun / acronym / product ID that doesn't appear in any
retrieved passage, a refusal directive is prepended to the body. This
catches the name-aliasing distractor failure mode (canonical case:
`MVE` question matched to `MLV` passages) that score-based gating
cannot, and is independent of the grounding mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..rql.types import ConfidenceMetrics


class Mode(str, Enum):
    AUGMENT = "augment"
    KNOWLEDGE = "knowledge"
    CONSTRAIN = "constrain"


# Derived from Mode so the argparse `choices=` can never drift from the
# enum. Order is the declaration order of Mode members.
MODE_CHOICES: tuple[str, ...] = tuple(m.value for m in Mode)


# Default mode. Bench 2 v3 evidence (Fabric corpus, partially-known by
# Sonnet 4.6): augment combines best answer accuracy (54.9%) with
# single-digit hallucination (3.9% answerable, 33.3% distractor) — same
# refusal rate as constrain on distractors but +9.8 pp accuracy on
# answerable. The shape that holds for most user corpora — broad
# documentation the LLM partially knows — favours augment as the
# default. Constrain remains the right pick for compliance / audit /
# regulatory work where wrong-but-confident is worse than no answer
# (2.0% answerable hallucination vs augment's 3.9%); pass `--mode
# constrain` to opt in.
DEFAULT_MODE: str = Mode.AUGMENT.value


@dataclass(frozen=True)
class _Thresholds:
    min_top1_score: float
    max_drift_fraction: float


# Augment + knowledge gate on absolute top-1 cosine — the strongest
# "is retrieval working at all" signal. Cosine 0.30 is the empirical
# floor for any topical match on gte-modernbert-base 768d L2-normalized
# embeddings; below it, augment retrieval has genuinely failed and the
# LLM is better off falling back to its training. Knowledge uses a
# lower floor (0.15) because the directive already invites the LLM to
# lean on training, so the gate's job is just to prevent grounding on
# absolute noise. Constrain never gates: refusal is the LLM's job under
# the constrain directive.
_THRESHOLDS: dict[Mode, _Thresholds] = {
    Mode.AUGMENT:   _Thresholds(min_top1_score=0.30, max_drift_fraction=0.30),
    Mode.KNOWLEDGE: _Thresholds(min_top1_score=0.15, max_drift_fraction=1.00),
    Mode.CONSTRAIN: _Thresholds(min_top1_score=0.00, max_drift_fraction=1.00),
}


_HEADERS: dict[Mode, str] = {
    Mode.AUGMENT: (
        "<!-- rlat-mode: augment -->\n"
        "> **Grounding mode: augment.** Use the passages below as primary "
        "context for this corpus's domain. Cite them when answering; prefer "
        "them over your training knowledge when the two conflict."
    ),
    Mode.KNOWLEDGE: (
        "<!-- rlat-mode: knowledge -->\n"
        "> **Grounding mode: knowledge.** The passages below supplement your "
        "existing knowledge. Ground claims about this corpus's domain in "
        "them; you may draw on general knowledge for surrounding context."
    ),
    Mode.CONSTRAIN: (
        "<!-- rlat-mode: constrain -->\n"
        "> **Grounding mode: constrain.** Answer ONLY from the passages "
        "below. If they do not cover the question, refuse explicitly — do "
        "not draw on training knowledge."
    ),
}


def should_suppress(metrics: ConfidenceMetrics, mode: Mode) -> bool:
    """Return True if the dynamic body should be suppressed for this mode.

    Suppression replaces the rendered passages with a 'no confident
    evidence' marker so the consumer LLM sees the directive but isn't
    grounded on noise. `constrain` never suppresses (refusal is the
    LLM's job).
    """
    t = _THRESHOLDS[mode]
    if metrics.top1_score < t.min_top1_score:
        return True
    if metrics.drift_fraction > t.max_drift_fraction:
        return True
    return False


def format_header(mode: Mode) -> str:
    """Return the markdown header instructing the consumer LLM."""
    return _HEADERS[mode]


def suppression_marker(metrics: ConfidenceMetrics, mode: Mode) -> str:
    """Body replacement when the gate fires. Surfaces which gate triggered
    so the corpus author can debug a runaway suppression rate."""
    return (
        f"*(no confident evidence under mode=`{mode.value}`; "
        f"top1_score={metrics.top1_score:.3f}, "
        f"drift_fraction={metrics.drift_fraction:.2f})*"
    )
