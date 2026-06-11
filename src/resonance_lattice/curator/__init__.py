"""curator — the tiny head + closed-form clauses that read the capture stream.

`the-curator-head.md`: a few-MB head bolted on the frozen gte-modernbert encoder
that *decides* over the telemetry the store already produced (`field.capture`).
This package holds the head's inputs and the head itself, split by the H1 §D
arms (`horizon-1-capture.md`):

- `signals` — the **closed-form clauses** (arm (b)): score-gap, reformulation /
  time-gap, intent clustering. Pure, no model, no learned weights; each emits a
  per-query feature, never a decision.
- `gap` — the **closed-form gap-candidate decision** (arm (b)): the fixed-threshold
  conjunction over the clauses (weak score-gap AND reproducible, plus an optional
  injected lexical veto). The §D baseline the learned combiner must beat. Also the
  **sleep-time recurring-gap queue** (C4): dedups candidates to one entry per
  recurring gap intent — the runaway-compute guard on the only cloud touch.
- (later) the **learnable head** (arm (c)): the cluster→label mapper, the
  calibrated reformulation classifier, the learned gap combiner — the parameters
  arm (b) cannot compute, the delta §D measures.

The §D gate is decisive (c) − (b): the head ships only if its learned parameters
beat the closed-form clauses at finding gaps. Nothing here claims the head helps
until that gate passes.
"""
