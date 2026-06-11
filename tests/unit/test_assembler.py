"""Unit test — the relevance-gated assembler's gating behaviour.

Pure + injected retrievers, no I/O. Run directly:
    PYTHONPATH=src python -m tests.unit.test_assembler
"""

from __future__ import annotations

from resonance_lattice.assembler import (
    AssembledContext,
    CorpusHit,
    MemoryHit,
    assemble,
)


def _mem(*relevances):
    return lambda q: [MemoryHit(f"lesson {r}", "prefer", r) for r in relevances]


def _corp(*scores):
    return lambda q: [CorpusHit(f"passage {s}", "doc.md", s) for s in scores]


def test_load_bearing_kept_off_domain_dropped():
    # one relevant lesson (0.78) + two off-domain (0.45, 0.50); floor 0.60
    a = assemble("q", memory_recall=_mem(0.78, 0.50, 0.45),
                 mem_floor=0.60, enable=("memory",))
    assert a.sources_included == ["memory"], a.sources_included
    assert len(a.memory_hits) == 1 and a.memory_hits[0].relevance == 0.78
    assert "lesson 0.78" in a.text and "lesson 0.45" not in a.text


def test_all_below_floor_drops_the_source():
    # every memory hit off-domain → no memory block, no drag (the bug fix)
    a = assemble("q", memory_recall=_mem(0.40, 0.52, 0.55), mem_floor=0.60,
                 enable=("memory",))
    assert a.sources_included == []
    assert a.text == ""
    assert a.memory_hits == []


def test_corpus_gate():
    a = assemble("q", corpus_retrieve=_corp(0.88, 0.40), corpus_floor=0.62,
                 enable=("corpus",))
    assert a.sources_included == ["corpus"]
    assert len(a.corpus_hits) == 1 and a.corpus_hits[0].score == 0.88


def test_assembled_combines_only_relevant_sources():
    # memory relevant, corpus irrelevant → assembled keeps memory only
    a = assemble("q", memory_recall=_mem(0.80), corpus_retrieve=_corp(0.30),
                 mem_floor=0.60, corpus_floor=0.62)
    assert a.sources_included == ["memory"]
    assert [h.relevance for h in a.memory_hits] == [0.80]
    assert a.corpus_hits == [] and "passage" not in a.text


def test_both_relevant():
    a = assemble("q", memory_recall=_mem(0.80), corpus_retrieve=_corp(0.85),
                 mem_floor=0.60, corpus_floor=0.62)
    assert a.sources_included == ["memory", "corpus"]


def test_enable_filters_sources():
    # corpus disabled even though relevant
    a = assemble("q", memory_recall=_mem(0.80), corpus_retrieve=_corp(0.85),
                 enable=("memory",))
    assert a.sources_included == ["memory"]


def test_top_k_caps_hits():
    a = assemble("q", memory_recall=_mem(0.9, 0.85, 0.8, 0.75, 0.7, 0.65),
                 mem_floor=0.60, top_k=3, enable=("memory",))
    assert len(a.memory_hits) == 3


def test_cold_is_empty():
    a = assemble("q", enable=())
    assert isinstance(a, AssembledContext) and a.text == ""


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} assembler tests passed")


if __name__ == "__main__":
    _run()
