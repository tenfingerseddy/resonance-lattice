"""LayeredMemory — three-tier (working / episodic / semantic).

Ported from v0.11 production. Confirmed live (NOT the unwired ResonanceMemory
in v0.11's memory.py — that was deleted).

Tier dynamics:
- Working:  short-lived, current-session context.
- Episodic: per-session records.
- Semantic: consolidated knowledge promoted from episodic.

Phase 5 deliverable. See base plan handoff to Kane's plan; design docs in
docs/internal/MEMORY.md (written this phase).
"""
