# Rotor intent operators for rlat retrieval

**Status:** application design with machine-checked mechanism receipts — no product
code changed · **Date:** 2026-08-10 · **Receipts:**
[`demos/demo6_rotor_intent_ops.py`](demos/demo6_rotor_intent_ops.py) (all checks pass) ·
**Parent:** [`ROTOR_DELTA.md`](ROTOR_DELTA.md) (the rotor gate), applied to this
repository's retrieval layer.

Nothing here touches `src/` — the contribution gate (simplify → codex review →
harness → board) belongs to the owner. This document is the design + evidence; a
PR-ready patch is a follow-up on request.

---

## 1. Where it plugs in

All of rlat retrieval flows through one line
([`field/dense.py:44`](../../src/resonance_lattice/field/dense.py)):

```python
scores = band_embeddings @ q        # both sides L2-normalised; cosine == dot
```

A rotor operator transforms **q only**. Because a rotor is exactly orthogonal,
`R q` is exactly unit-norm at every parameter value, so:

- score calibration, dedup thresholds, and `--verified-only` semantics survive
  untouched;
- the corpus side is never modified — **no rebuild, hashes and drift receipts
  intact**;
- the operation is describable in one sentence (plane + angle), printable as a
  receipt under the grounding directive.

## 2. The operators (the Intent Lattice's deferred v2.1 surface)

[`intent/__init__.py`](../../src/resonance_lattice/intent/__init__.py) ships the
Intent Lattice as a kind tag and defers the operators — `anti`, `--toward`,
multi-intent composition — to v2.1+. Those are, mathematically, rotor operations:

| Operator | Definition | Checked property (demo 6) |
|---|---|---|
| `--toward c, strength t` | rotate q by `t·angle(q,c)` in plane span{q,c} | ≡ SLERP exactly; t=1 lands on c; unit-norm at every t |
| `anti` | the same rotation, negative angle | exact inverse to 1e-15 (reversible, auditable) |
| composition | apply rotors in sequence | exact products; order effect shrinks 4× when both angles halve (first-order BCH, measured) — non-abelian in principle, benign at lens strengths |
| **lens rotor** | ONE fixed rotation (plane from two anchors) applied to every query | **exact isometry**: all pairwise query similarities preserved to 1e-15 — conditioning at zero resolution cost |

Guardrail imported from the rotor-gate training experiments: clamp per-query
strengths so the rotation never passes quadrature (θ ≤ π/2) — beyond it the query
rotates *through* the anchor into anti-correlation (the retrieval version of demo 5's
phase-aliasing failure).

## 3. The design finding: target contrasts, not centroids

The first experiment scored `--toward centroid(intent A)` and it **hurt**: group-A
precision@10 fell 53.3 → 46.3 (t=0.3) and the specific target passage's rank
collapsed at higher strength. Diagnosis: a broad intent group's centroid nearly
coincides with the global corpus centroid — aiming there makes a query *generic*,
not intent-flavoured. This is kept in the demo as a documented anti-pattern.

The discriminative object is the **contrast** between intents,
`unit(centroid(A) − centroid(B))` — operationally `--toward A` composed with
`anti B`, which is exactly why the deferred operator triple is *jointly* necessary
rather than three conveniences. Measured on real text (this repository's own docs,
1,011 passages, LSA stand-in embeddings; 250 ambiguous two-topic queries):

| Method | intent precision@10 | median rank of the specific parent |
|---|---:|---:|
| raw query | 53.3 | 1 |
| rotor → centroid t=0.3 *(anti-pattern)* | 46.3 | 1 |
| rotor → centroid t=0.6 *(anti-pattern)* | 43.9 | 29 |
| **rotor → contrast t=0.6** | **83.4** | **1** |
| additive contrast +0.5 | 75.2 | 1 |
| trust-weight ×1.25 *(needs a glob)* | 88.2 | 1 |

Three conclusions, drawn narrowly:

1. **Contrast conditioning works and keeps specificity** — 53.3 → 83.4 with the
   parent passage still at rank 1 (the contrast direction is near-orthogonal to
   within-topic detail).
2. **Rotor beats additive at higher strength here** (83.4 vs 75.2) — but this is
   LSA-scale evidence; the pre-registered bench below decides.
3. **Trust weights and rotors are complementary dials, not competitors.** Trust
   weights re-score *after* aiming (amplitude) and win outright when the intent is
   expressible as a source glob; rotors re-aim *before* scoring (phase) and reach
   semantic regions no glob can name. A lens should carry both.

## 4. Lens serialisation sketch

[`lens/schema.py`](../../src/resonance_lattice/lens/schema.py) already defines a
portable `.lens` ZIP with `trust_weights.json` (amplitude dial). The rotor is the
phase dial, one new member:

```
my-lens.lens
├── manifest.json
├── trust_weights.json          (existing — amplitude)
└── query_rotors.json           (new — phase)
    [{"name": "engineering-not-marketing",
      "anchor_a": {"kind": "centroid", "of": "docs/internal/*", "vector": [...]},
      "anchor_b": {"kind": "centroid", "of": "docs/site/*",     "vector": [...]},
      "angle_max": 0.9,
      "backbone_revision": "<pinned encoder commit>",
      "receipt": "rotates queries toward internal-engineering register"}]
```

- **Portable** by the same argument as trust weights: anchors live in encoder
  space, and every `.rlat` pins `backbone.revision` — a lens declares the revision
  it was fit against and fails loud on mismatch (the existing metadata contract).
- **Composable**: `lens.compose()` extends naturally — rotor lists concatenate;
  application is sequential (order recorded; §2's BCH measurement says small-angle
  order effects are negligible).
- **Removable exactly**: the transpose. No approximate "undo".
- **Auditable**: each application can print `query rotated 14° toward
  ⟨engineering⟩, away from ⟨marketing⟩` under the grounding directive.

Cost: two `d`-vectors per rotor stored; O(d) per query applied; zero rebuild.

## 5. Pre-registered production bench (to run where the real encoder lives)

Design, registered before any production run:

- **Data**: an intent-labelled query slice (the 63-question Fabric set is a
  candidate seed; label each query with a register/intent and a contrast), plus
  BEIR-5 as the no-regression floor (locked at 0.5144 mean nDCG@10 per
  `BENCHMARK_GATE.md`).
- **Arms**: raw · additive-contrast (strength-matched) · trust-weights-only (where
  glob-expressible) · rotor-contrast · rotor+trust.
- **Metrics**: intent precision@10 and nDCG@10 (primary), rank of gold passage
  (specificity), BEIR-5 delta with the lens *off* (must be exactly 0 — the corpus
  side is untouched, so this is a wiring check).
- **Falsification, stated now**: if additive is within 1 point of rotor on the
  primary metric, the rotor claim shrinks to exactness/invertibility/auditability
  (still arguably worth shipping, but claimed as such). If rotor+trust ≤
  trust-only, the phase dial earns no place in the lens schema and this design is
  shelved with its numbers published.

## 6. Honest limits

- All effectiveness numbers above are **LSA-stand-in scale** (see
  [`demos/corpus.py`](demos/corpus.py)); the production encoder may change them in
  either direction — that is what §5 is for.
- Contrast anchors require *negative* exemplars (a `--toward X` UX will quietly
  need `--anti Y` or a sensible default contrast, e.g. the corpus centroid — which
  turns `--toward` into "toward X, away from average", plausibly the right default;
  untested, noted as a design question).
- Per-query `--toward` trades query-space resolution for alignment (measured in
  demo 6b: mean pairwise query cosine rises from ~0.00 to ~0.50 at t=0.5). Only the
  global lens rotor is resolution-free. Both belong in the design; they are
  different tools.
- Training rotors from curation/demand-gap signal (rather than hand-authoring
  anchors) is future work; the gradient of the closed form is already derived and
  finite-difference-verified in demo 5's `rotor_backward`.
