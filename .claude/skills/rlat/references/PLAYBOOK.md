# Context Control Playbook

## Contents

- [Composition decision tree](#composition-decision-tree) — which compose operation for which intent
- [Pre-search assessment](#pre-search-assessment) — reading the Knowledge Assessment block
- [Injection mode selection](#injection-mode-selection) — augment / constrain / knowledge / custom
- [Interpreting diagnostics](#interpreting-diagnostics) — band health, saturation, overlap
- [Proactive composition](#proactive-composition) — when to compose without being asked

Decision rules for *how* to search, not just *what* to search. Read this when the user's question involves multiple knowledge models, confidence / coverage assessment, or LLM grounding framing.

---

## Composition decision tree

Pick composition based on the user's intent:

| User intent | Operation | MCP call |
|-------------|-----------|----------|
| Search across multiple domains | **merge** | `rlat_compose_search(query, with_cartridges=["code.rlat", "docs.rlat"])` |
| Search from a specific perspective | **project** | `rlat_compose_search(query, through="compliance.rlat")` |
| What changed since last version? | **diff** | `rlat_compose_search(query, diff_against="baseline.rlat")` |
| Focus on a specific topic | **boost** | `rlat_compose_search(query, boost=["security", "auth"])` |
| Exclude noise from results | **suppress** | `rlat_compose_search(query, suppress=["deprecated", "legacy"])` |

Decision rules:

- **Merge** when the question spans domains (e.g. "how does the frontend auth flow connect to the backend API?"). Merge is commutative — order doesn't matter.
- **Project** when the question has a lens or perspective (e.g. "search the codebase from a compliance perspective"). The lens knowledge model shapes results but doesn't contribute its own passages.
- **Diff** when the question is about change (e.g. "what's new since the last release?"). Only the newer knowledge model returns results; the baseline is subtracted.
- **Boost** when the user's question could return noisy results and you want to amplify a specific signal (e.g. searching for "configuration" but the user cares about security config specifically).
- **Suppress** when a known noise source dominates results (e.g. deprecated modules appearing in every search).

These compose freely — you can merge + boost + suppress in a single call.

---

## Pre-search assessment

Before answering questions where confidence matters, use the Knowledge Assessment that `rlat_search` returns (the Coverage block at the top of results), or call `rlat_locate` for a standalone assessment:

| Coverage label | What to do |
|---------------|------------|
| **strong** | Answer confidently. Cite sources. |
| **partial** | Answer, but tell the user which aspects are thin. |
| **edge** | Answer with heavy caveats. Use the expansion_hint to suggest where else to look. Consider boost to amplify weak signal, or `--through` another knowledge model. |
| **gap** | Tell the user the knowledge model doesn't cover this. Suggest the nearest covered topic (expansion_hint). Fall back to Grep or your own knowledge. |

Use the **band focus** to understand what kind of question it is:

- Topic band dominant → conceptual question; broad context helps.
- Entity band dominant → specific lookup; exact match matters, consider Grep as supplement.
- Relations band dominant → structural question; how things connect.

When **anti-resonance is high** (>0.3), explicitly mention the gap to the user:

> "The project knowledge model has limited coverage on [topic]. The nearest well-covered area is [expansion_hint]."

---

## Injection mode selection

Pick the injection mode based on the stakes of the question:

| Scenario | Mode | Why |
|----------|------|-----|
| General questions, exploration, brainstorming | `augment` | LLM supplements from own knowledge |
| Compliance, legal, safety-critical, "cite your sources" | `constrain` | Zero hallucination — answer only from evidence |
| Domain-specific or proprietary content | `knowledge` | Trust knowledge model primarily, flag gaps honestly |
| User says "only from docs" or "cite sources" | `constrain` | Explicit user intent |
| User says "what do you think?" or wants reasoning | `augment` | User wants LLM's perspective too |

Default to `knowledge`. Escalate to `constrain` when accuracy is critical. Drop to `augment` when the user wants exploration.

---

## Interpreting diagnostics

When using `rlat_xray`, `rlat_profile`, or reading diagnostic data from search results:

**Band health labels:**

- **rich** — No action. Report as healthy if asked.
- **adequate** — Normal. Mention only if asked.
- **thin** — Flag to user: "The [dimension] is thin — [conceptual/entity/structural] knowledge is sparse. Consider adding more [type] content."
- **noisy** — Flag to user: "The [dimension] signal is noisy. Results in this area may be less reliable."

**Saturation:**

- <50% → "The knowledge model has room for significantly more content."
- 50-80% → "Well-populated."
- \>80% → "Approaching capacity. Consider splitting into domain-specific knowledge models and composing at query time with `--with`."

**When comparing knowledge models (`rlat_compare`):**

- High overlap (>70%) → "These knowledge models cover similar ground — merging adds little."
- Low overlap (<30%) → "Largely different domains. Merge with `--with` for broad coverage."
- Asymmetric energy → "Knowledge Model A has knowledge that B lacks in [band], suggesting [interpretation]."

---

## Proactive composition

After calling `rlat_discover`, think about whether composition would help:

- **Cross-domain question** → Merge relevant knowledge models automatically and explain: "I'm searching across both [domain A] and [domain B] to answer this."
- **Review or audit context** → Suggest `--through` with a relevant knowledge model as lens: "I can search the codebase through a compliance lens — would that help?"
- **Change tracking context** → Use `--diff-against` with baseline if one exists: "I'll compare against the baseline to show what's semantically new."
- **Noisy or unfocused results** → Boost the relevant topic or suppress the noise source. Don't just re-run the same search — sculpt the field.
