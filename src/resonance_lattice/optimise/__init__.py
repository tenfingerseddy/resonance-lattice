"""MRL optimise path — opt-in upgrade.

Trains a (512, 768) MRL projection over a knowledge model's base band.
Adds bands/optimised.npz + bands/optimised_W.npz + ann/optimised.hnsw
in place. Idempotent.

Cost: ~$14-21/knowledge-model in LLM synth queries + 30min GPU (or 4-6hr CPU).
Lift: ~+6.8 pt R@5 on 3-corpus holdout per base plan §C.

Phase 4 deliverable. Base plan §4.
"""
