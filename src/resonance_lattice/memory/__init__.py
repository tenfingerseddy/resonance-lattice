"""Per-user earned-experience memory.

Backed by a flat `ExperienceClaimStore` (`memory/claim_store.py`) — one
unified `Claim` record per row, polarity-tagged + confidence-graded. The
pipeline:

  capture     extract event(s) from a transcript, redact, dedup, write.
  recall      cosine + workspace gate + confidence gap + recurrence gate
              + manifesto rerank, source-dispatched (experience vs corpus).
  distil      §7 atomic-event extraction; arrows 1/2/3 build pattern /
              learning / principle claims from clusters / attribution /
              cross-domain.
  confidence  Beta tallies → 4-rung label, modulated by user verdicts,
              implicit corroboration, corpus confirm/contradict.
  forget      decay-below-floor + redundant-after-promotion + falsified
              + trivial drop, with provenance / severity / declaration
              protections.
  redactor    Layer-1 PII + denylist scrub at capture time, with an
              append-only audit log.
"""
