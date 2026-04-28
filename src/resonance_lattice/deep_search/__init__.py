"""Multi-hop deep-search: plan → search → refine → synthesize.

The product-headline mode for v2.0 launch. On the Microsoft Fabric
5-lane bench (Sonnet 4.6 inference, relaxed rubric, 63 hand-curated
questions across post-cutoff / fuzzy / stable recency tiers):

    no_retrieval (LLM alone):  54.9% acc / 19.6% halluc / $0.002/q
    augment      (single-shot): 74.5% acc / 3.9%  halluc / $0.004/q
    deep_search  (multi-hop):   92.2% acc / 2.0%  halluc / $0.010/q

Worth 2.5× the cost over augment for ~24 pp accuracy lift and 2× lower
hallucination. Headline number for the launch.

Public surface:

  from resonance_lattice.deep_search import deep_search, DeepSearchResult
  result = deep_search(km_path, question, client=anthropic_client)
  print(result.answer)
  for hop in result.hops:
      print(hop.kind, hop.query)

The loop is purely additive: the same single-recipe retrieval path is
reused per hop (`field.retrieve` against the optimised band when
present, base otherwise). Name-verification is applied across the
union of all retrieved evidence to the original question — when a
distinctive token from the question never appears in any hop's
evidence, the synthesised answer is prefixed with a refusal directive
(or, under `strict_names=True`, the loop returns a refusal answer
without paying for the synth call).

Bench harness: benchmarks/user_bench/hallucination/run.py (5-lane).
"""

from __future__ import annotations

from .loop import deep_search
from .types import DeepSearchHop, DeepSearchResult

__all__ = ["deep_search", "DeepSearchHop", "DeepSearchResult"]
