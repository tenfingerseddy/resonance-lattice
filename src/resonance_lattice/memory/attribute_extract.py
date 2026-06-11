"""Extract durable WORLD ATTRIBUTES from a captured session transcript — the PASSIVE source.

The 2nd capture source (capture-frontier charter): standing facts about the world a knowledge
model covers (a tenant's region policy, a garden's water restrictions, a practice's
jurisdiction rule) that the user states IN PASSING get mined from the SessionEnd transcript
the `capture.py` hook already reads — zero added friction. Mirrors `extract.py::extract_events`,
but with a PRECISION-GATED prompt and routed to `attribute_capture.capture_attributes`.

The gate is load-bearing twice over: a wrong/transient fact POLLUTES the band (a polluted band
hurts answers — STEP 13), and a PERSON-fact in the band leaks into a shareable artifact. GATE 4
(the scope gate, Kane's 2026-06-10 direction) therefore drops facts about the individual
speaker entirely — only facts true for anyone using the knowledge model may land. The 3-gate
ancestor of this prompt was A/B-validated on trap-heavy synthetic sessions (E2b, subscription,
zero-API): precision 0.65→0.83 with no recall loss (0.93→0.95). The 4-gate, domain-neutral
variant below was re-validated on the REAL production path (E2c run 1, pre-registered,
benchmarks/attribute_gate_e2c/): precision 0.86, recall 1.00, ZERO person-fact leaks across
7 traps, every domain ≥0.80 — the validation gate on live use is CLEARED.

Integration: in the capture pipeline, run BOTH `extract_events` (event facts) and
`extract_attributes`; route the latter's output to `capture_attributes(km_path, texts,
criticality="high")`. `client is None` is a no-op (None).
"""
from __future__ import annotations

from ._common import parse_llm_json
from ._llm import LLMClient

_PROMPT = """You read a session transcript between a user and an AI assistant. The assistant serves a KNOWLEDGE MODEL — a packaged corpus about some domain (software, a garden, a legal practice, a product line, anything). Extract DURABLE WORLD ATTRIBUTES the user stated — stable facts about the world the knowledge model covers that change the correct answer to future questions (a version, capacity, region, policy, physical condition, or standing CONSTRAINT of that world).

Emit a fact ONLY if it passes ALL FOUR gates. When in doubt, DROP it — a wrong attribute in the band hurts more than a missing one.

  GATE 1 — DURABLE + STANDING. A stable property that was true BEFORE this session and persists after it. NOT transient state, NOT something discovered or changed DURING the session (e.g. a version a command just printed, a temporary workaround, "right now I'm doing X").
  GATE 2 — STATED BY THE USER, first-person / possessive ("our tenant...", "my garden...", "the practice never..."). NOT the assistant's statements, NOT a doc/corpus fact, NOT a hypothetical.
  GATE 3 — PREMISE-BEARING. Knowing it would change the correct answer to a realistic future question. Drop opinions and incidental nouns.
  GATE 4 — ABOUT THE WORLD the knowledge model covers, true for ANYONE who uses it (the tenant, the project, the garden, the practice). NOT about the individual person speaking — drop their role, their personal machine, their habits, and their preferences entirely.

GOOD (emit): "The tenant is EU-only for data residency." / "The garden is on permanent water restrictions." / "The practice takes NSW-law matters only."
BAD (drop): "cargo --version printed 1.87" (discovered this session) / "right now the test is failing" (transient) / "the assistant recommended Postgres 16" (assistant, not user) / "the user's workspace role is Viewer" (person-fact, not world-fact) / "they prefer compact tables" (preference).

OUTPUT — ONLY a JSON object, first char `{`, last char `}`, no prose/markdown/fences:
  {"attributes": ["<attribute sentence>", ...]}
Empty list `[]` if none pass. Each attribute ≤30 words, third-person, self-contained, stated about the WORLD (e.g. "The tenant ...", "The garden ..."), never about the person.
"""

_MAX_TOKENS = 768


def _build_messages(text: str) -> list[dict]:
    return [{
        "role": "user",
        "content": ("USER STATEMENTS from a session transcript (capped at 24K chars):\n\n" + text.strip()
                    + "\n\nExtract the durable world attributes. Empty list if none pass all four gates."),
    }]


def extract_attributes(text: str, *, client: LLMClient | None) -> list[str] | None:
    """One precision-gated LLM pass over `text` → list of durable world-attribute strings, or None.

    `list[str]` on success (possibly empty); `None` on any failure or `client is None` (caller skips attribute
    capture — same optional-LLM convention as `extract_events`).
    """
    if client is None:
        return None
    if not text or not text.strip():
        return []
    try:
        response = client(_PROMPT, _build_messages(text), _MAX_TOKENS)
    except Exception:  # noqa: BLE001 — LLM failure must not raise
        return None
    try:
        payload = parse_llm_json(response.text)
    except Exception:  # noqa: BLE001 — parse failure must not raise
        return None
    if not isinstance(payload, dict):
        return None
    attrs = payload.get("attributes")
    if not isinstance(attrs, list):
        return None
    out: list[str] = []
    for item in attrs:
        if not isinstance(item, str):
            return None
        cleaned = item.strip()
        if cleaned:
            out.append(cleaned)
    return out
