"""Distinctive-name verification for grounding-mode output.

The bench-2 v3 distractor analysis (`docs/internal/benchmarks/02_distractor_floor_analysis.md`)
found that retrieval-score-based distractor floors don't work on a
paraphrase-rich documentation corpus — answerable and distractor questions
have overlapping top1_score and top1_top2_gap distributions. The dominant
distractor failure mode is **name aliasing**: the question contains a
distinctive proper noun or acronym that doesn't exist in the corpus, the
encoder semantic-matches it to an adjacent real entity, the LLM reads the
adjacent-entity passages and answers as if the question had asked about
that entity instead.

Canonical case: `Materialized View Express (MVE)` (fake) vs
`Materialized Lake View (MLV)` (real). All three rlat modes hallucinated
the MLV default-action answer because the retrieval surfaced the MLV
passages at high score and the LLM didn't notice the question used MVE.

This module extracts distinctive tokens from a question (acronyms,
capitalised proper nouns, alphanumeric product IDs like `F4096`) and
verifies that each appears verbatim in at least one retrieved passage.
When a distinctive token from the question is missing from all retrieved
passages, the consumer LLM is given a hard refusal directive: the
question may be about a different entity than what the corpus describes.

Cheap (zero LLM calls, ~50 lines), defensible (the heuristic is
conservative — common nouns and stopwords are skipped), and load-bearing
(addresses the only safety-axis failure mode that score-based gating
cannot).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Words that look like proper nouns / brand names but are common enough in
# question phrasing that requiring verbatim presence would over-refuse.
# These are baseline English stopwords plus product-domain words that
# appear in nearly every Fabric / Power BI / Azure documentation passage.
_STOPWORDS: frozenset[str] = frozenset({
    # English function words
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does",
    "for", "from", "had", "has", "have", "how", "if", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you",
    "your", "i", "me", "my", "we", "us", "our", "they", "them", "their",
    "can", "could", "should", "would", "may", "might", "must", "shall",
    "than", "then", "there", "these", "those", "any", "all", "some",
    "no", "not", "yes", "so", "also", "into", "out", "up", "down", "off",
    "about", "after", "before", "during", "while", "until", "among",
    # Common product-domain words that don't need verbatim presence
    "fabric", "microsoft", "azure", "power", "bi", "data", "table",
    "tables", "file", "files", "user", "users", "role", "roles",
    "feature", "features", "service", "services", "item", "items",
    "type", "types", "mode", "modes", "option", "options", "setting",
    "settings", "support", "supports", "use", "uses", "default",
    "configuration", "configure", "configures", "configured",
    "create", "creates", "created", "enable", "enables", "enabled",
    "value", "values", "field", "fields", "section", "sections",
    "document", "documents", "documentation", "guide", "guides",
    "name", "names", "id", "ids",
})


@dataclass(frozen=True)
class NameCheckResult:
    """Outcome of distinctive-name verification.

    `missing_tokens` holds the distinctive question tokens that did NOT
    appear in any retrieved passage. Empty list = check passed; non-empty
    list = the consumer LLM should be told to refuse on potential name
    aliasing.

    `extracted_tokens` is the full set of distinctive tokens the heuristic
    pulled from the question — exposed for debugging / harness assertions.
    """
    missing_tokens: list[str]
    extracted_tokens: list[str]

    @property
    def passed(self) -> bool:
        return not self.missing_tokens


def _is_distinctive_token(tok: str) -> bool:
    """A token is distinctive if it's an ALL-CAPS acronym OR an
    alphanumeric product ID, and it's not a stopword.

    Plain capitalised words (`Compare`, `Pricing`, `Summarize`, even
    `Snowflake`) are NOT treated as distinctive — too prone to over-
    refusal on natural English question phrasing where any word may be
    sentence-initial. Multi-word proper nouns must be quoted by the
    user (`'Quantum Data Lakes'`) to opt into the check; the
    `_extract_distinctive_tokens` quote-handling layer covers that
    path. Codex review (2026-04-27) flagged this — over-refusal on
    `"Summarize login behavior"` style questions is worse than the
    additional Snowflake-style coverage was worth.
    """
    if not tok or len(tok) < 2:
        return False
    lower = tok.lower()
    if lower in _STOPWORDS:
        return False
    # ALL-CAPS acronyms (>= 2 chars): MVE, SKU, CU, MLV, ETL, F4096
    if tok.isupper() and any(c.isalpha() for c in tok):
        return True
    # Alphanumeric product IDs: F32, F4096, P3, gen2 — at least one digit
    # and at least one alpha
    if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        return True
    return False


def _extract_distinctive_tokens(question: str) -> list[str]:
    """Pull distinctive proper-noun-like tokens from a question, preserving
    case so the substring check is exact.

    Splits on whitespace + common punctuation, keeps tokens with internal
    word characters (handles hyphenated names like 'Power-BI' as a single
    token). Quoted strings are kept as-is so multi-word product names like
    `'Quantum Data Lakes'` survive as a single token to match.
    """
    # Pull out single-quoted and double-quoted multi-word names first
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
    quoted_distinctive = [q.strip() for q in quoted if q.strip() and len(q.strip()) >= 3]

    # Strip the quoted portions from the question for word-by-word splitting,
    # so we don't double-count their inner words
    rest = re.sub(r"['\"][^'\"]+['\"]", " ", question)
    # Strip parenthetical glosses like "(MVE)" so MVE shows up as its own token
    rest_with_parens = re.sub(r"[()]", " ", rest)
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", rest_with_parens)

    distinctive_words = [w for w in words if _is_distinctive_token(w)]

    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for tok in quoted_distinctive + distinctive_words:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _passage_contains(passages_text: str, token: str) -> bool:
    """Case-insensitive substring presence check.

    Acronyms are checked case-sensitively (MVE != mve at semantic level —
    a corpus mention of `mve` lowercase is unlikely to be the brand name).
    Multi-word names and capitalised proper nouns use case-insensitive.
    """
    if token.isupper() and len(token) <= 6:
        # Acronym — case-sensitive whole-word match
        return bool(re.search(rf"\b{re.escape(token)}\b", passages_text))
    return token.lower() in passages_text.lower()


def verify_question_in_passages(question: str, passages_text: str) -> NameCheckResult:
    """Check whether each distinctive token from the question appears in the
    concatenated retrieved passage text.

    Args:
        question: The user's question, raw.
        passages_text: The concatenated text of all retrieved passages
            (typically the body of `rlat search --format context`).

    Returns:
        NameCheckResult with `passed=True` when every distinctive token
        appears in the passages, else lists the missing tokens.
    """
    extracted = _extract_distinctive_tokens(question)
    missing = [t for t in extracted if not _passage_contains(passages_text, t)]
    return NameCheckResult(missing_tokens=missing, extracted_tokens=extracted)


# Help text for the `--strict-names` argparse flag. Shared across
# `rlat search`, `rlat skill-context`, and `rlat deep-search` so the
# user-facing description never drifts between subcommands.
STRICT_NAMES_HELP = (
    "Exit non-zero (rc=3) when a distinctive proper noun, acronym, or "
    "alphanumeric ID from the question does not appear verbatim in any "
    "retrieved passage. Default behaviour prepends a refusal directive "
    "to the body and proceeds. Catches the name-aliasing distractor "
    "failure mode that score-based gating cannot."
)


def refusal_directive(missing_tokens: list[str]) -> str:
    """The block prepended to the grounding-mode body when name-check
    fails. Tells the consumer LLM to refuse / explicitly note the
    name mismatch instead of paraphrasing the adjacent-entity content."""
    if not missing_tokens:
        return ""
    quoted = ", ".join(f"`{t}`" for t in missing_tokens)
    return (
        f"<!-- rlat-namecheck: missing {quoted} -->\n"
        f"> ⚠ **Name verification failed.** The question references "
        f"{quoted}, but no retrieved passage contains this exact name. "
        f"The corpus may describe an adjacent or differently-named "
        f"entity. Do NOT answer as if the corpus's content is about "
        f"{quoted}. Either (a) refuse explicitly, noting the name "
        f"mismatch, or (b) ask the user to confirm the name is correct.\n"
    )
