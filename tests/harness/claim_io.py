"""claim_io — the shared serialisation + merge plumbing.

Pins the helpers `ExperienceClaimStore` calls through. The transitive
coverage from the experience-store harness exercises the happy path;
this suite pins the edge cases the store never reaches because it
guards upstream:

  (a) empty-batch merge — `(existing=[], new_claims=[])` returns the
      empty `(claims, (0,DIM) band)` shell and **never calls the encoder
      provider** (the lazy-allocate contract the store relies on).
  (b) `embeddings` shape mismatch — `(N, DIM-1)` raises `ValueError`
      before any state mutation.
  (c) duplicate `claim_id` within a single batch raises `ValueError`
      (the encoder provider must not be called).
  (d) empty `claim_ids` short-circuits `delete_claims` with
      `removed == 0` (no scan, no copy).
  (e) `core_to_row` / `core_from_row` round-trip a claim's core fields
      including the tuple-typed `parent_ids` (the shared serialiser
      contract — facts handling is exercised in the
      `experience_claim_store` suite and the `corpus_claim_io` round-trip
      coverage inside `insight_layer`).

Hermetic — pure construction, no I/O.
"""

from __future__ import annotations

import sys

import numpy as np

from ._testutil import check_guarantee

_P = "claim_io"


def _claim(claim_id: str = "01HZCLAIM00000000000000001"):
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    return Claim(
        claim_id=claim_id,
        source="experience",
        kind="event",
        content="prefer the standard library",
        created_at="2026-05-23T00:00:00Z",
        corroboration=2.0,
        falsification=1.0,
        trust_as_of="",
        state="active",
        parent_ids=("01HZEVENT0000000000000000A",),
        facts=ExperienceFacts(
            polarity=("prefer",),
            recurrence_count=1,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash="manual",
            origin="manual",
            last_corroborated_at="2026-05-23T00:00:00Z",
        ),
    )


def _corpus_claim(content: str = "prefer the standard library",
                  fingerprint: str = "fp-corpus-A"):
    from resonance_lattice.state.claim import Claim, CorpusFacts

    return Claim(
        claim_id="01HZCORPUS0000000000000001",
        source="corpus",
        kind="synthesis",
        content=content,
        created_at="2026-05-23T00:00:00Z",
        corroboration=2.0,
        falsification=1.0,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=CorpusFacts(
            citations=(),
            content_fingerprint=fingerprint,
            source_model_hash="m",
            source_passage_hashes=("h1", "h2"),
        ),
    )


def _no_encoder():
    """Encoder provider that fails the test if called — guards the
    lazy-allocate contract."""
    raise AssertionError("encoder_provider must not be called")


def _check_empty_merge() -> int:
    from resonance_lattice.field.encoder import DIM
    from resonance_lattice.state.claim_io import merge_claims

    existing: list = []
    band = np.zeros((0, DIM), dtype=np.float32)
    rows, new_band = merge_claims(
        existing, band, [],
        embeddings=None,
        encoder_provider=_no_encoder,
    )
    ok = (
        rows == []
        and new_band.shape == (0, DIM)
        and new_band.dtype == np.float32
    )
    return 0 if check_guarantee(
        ok, "(a) empty-batch merge + lazy encoder", _P) else 1


def _check_embeddings_shape_reject() -> int:
    from resonance_lattice.field.encoder import DIM
    from resonance_lattice.state.claim_io import merge_claims

    raised = False
    try:
        merge_claims(
            [], np.zeros((0, DIM), dtype=np.float32),
            [_claim()],
            embeddings=np.zeros((1, DIM - 1), dtype=np.float32),
            encoder_provider=_no_encoder,
        )
    except ValueError:
        raised = True
    return 0 if check_guarantee(
        raised, "(b) embeddings shape mismatch rejected", _P) else 1


def _check_duplicate_claim_id_reject() -> int:
    from resonance_lattice.field.encoder import DIM
    from resonance_lattice.state.claim_io import merge_claims

    a = _claim("01HZCLAIM00000000000000DUP")
    b = _claim("01HZCLAIM00000000000000DUP")    # same id
    raised = False
    try:
        merge_claims(
            [], np.zeros((0, DIM), dtype=np.float32),
            [a, b],
            embeddings=None,
            encoder_provider=_no_encoder,    # must not fire — reject is pre-encode
        )
    except ValueError:
        raised = True
    return 0 if check_guarantee(
        raised, "(c) duplicate claim_id in batch rejected", _P) else 1


def _check_empty_delete() -> int:
    from resonance_lattice.field.encoder import DIM
    from resonance_lattice.state.claim_io import delete_claims

    existing = [_claim()]
    band = np.zeros((1, DIM), dtype=np.float32)
    kept, new_band, removed = delete_claims(existing, band, [])
    ok = (
        removed == 0
        and kept is existing          # no copy on empty input
        and new_band is band
    )
    return 0 if check_guarantee(
        ok, "(d) empty claim_ids short-circuits delete", _P) else 1


def _check_core_round_trip() -> int:
    from resonance_lattice.state.claim_io import core_from_row, core_to_row

    c = _claim()
    row = core_to_row(c)
    # `parent_ids` must serialise as a list (JSON has no tuple).
    list_shape = isinstance(row["parent_ids"], list)
    kwargs = core_from_row(row)
    # All core fields recovered, parent_ids restored as tuple.
    recovered = (
        kwargs["claim_id"] == c.claim_id
        and kwargs["source"] == c.source
        and kwargs["kind"] == c.kind
        and kwargs["state"] == c.state
        and kwargs["parent_ids"] == c.parent_ids
        and isinstance(kwargs["parent_ids"], tuple)
        and kwargs["corroboration"] == c.corroboration
        and kwargs["falsification"] == c.falsification
        and "facts" not in kwargs       # facts is the caller's responsibility
    )
    return 0 if check_guarantee(
        list_shape and recovered, "(e) core_to_row / core_from_row round-trip", _P) else 1


def _check_writer_identity() -> int:
    """Invariant 7 (writer half): `writer` defaults single-writer, serialises,
    round-trips, defaults on a pre-S1.5 row, and is immutable through `evolve`;
    a still-required core field absent from a row still raises."""
    import dataclasses

    from resonance_lattice.state.claim import DEFAULT_WRITER, evolve
    from resonance_lattice.state.claim_io import core_from_row, core_to_row

    c = _claim()                                    # constructed without writer=
    row = core_to_row(c)
    default_ok = (
        c.writer == DEFAULT_WRITER
        and row["writer"] == DEFAULT_WRITER
        and core_from_row(row)["writer"] == DEFAULT_WRITER
    )

    explicit = dataclasses.replace(c, writer="alice")
    explicit_ok = core_from_row(core_to_row(explicit))["writer"] == "alice"

    legacy = core_to_row(c)
    del legacy["writer"]                            # simulate a pre-S1.5 row
    legacy_ok = core_from_row(legacy)["writer"] == DEFAULT_WRITER

    required_missing = core_to_row(c)
    del required_missing["claim_id"]                # required core field, no default
    raised = False
    try:
        core_from_row(required_missing)
    except KeyError:
        raised = True

    evolve_raised = False
    try:
        evolve(c, writer="mallory")                 # re-attribution forbidden
    except ValueError:
        evolve_raised = True

    ok = default_ok and explicit_ok and legacy_ok and raised and evolve_raised
    return 0 if check_guarantee(
        ok, "(f) writer: default + serialise + legacy-default + required-KeyError "
            "+ evolve-immutable", _P) else 1


def _check_content_key_identity() -> int:
    """Invariant 7 (content_key half): a stable 16-hex hash over `content`
    alone — corpus- AND source-agnostic (the H3 join), where the corpus
    `content_fingerprint` is NOT — and it tracks `content` through `evolve`."""
    import hashlib

    from resonance_lattice.state.claim import evolve

    text = "prefer the standard library"
    exp = _claim()                                  # experience, same content
    corp_a = _corpus_claim(content=text, fingerprint="fp-A")
    corp_b = _corpus_claim(content=text, fingerprint="fp-B")  # different corpus

    expect = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    shape_ok = exp.content_key == expect and len(exp.content_key) == 16

    join_ok = (
        exp.content_key == corp_a.content_key       # cross-source same content
        and corp_a.content_key == corp_b.content_key  # cross-corpus same content
        and corp_a.facts.content_fingerprint
        != corp_b.facts.content_fingerprint         # fingerprint does NOT join
    )

    evolved = evolve(exp, content="a different earned claim")
    track_ok = (
        evolved.content_key != exp.content_key
        and evolved.content_key
        == hashlib.sha256(b"a different earned claim").hexdigest()[:16]
    )

    ok = shape_ok and join_ok and track_ok
    return 0 if check_guarantee(
        ok, "(g) content_key: stable corpus+source-agnostic join, tracks content", _P) else 1


def _check_experience_claim_round_trip() -> int:
    """(h) The shared experience-facts serialiser the unified band (S3) and
    `ExperienceClaimStore` both call: full claim round-trips, the tuple-typed
    `polarity` serialises as a list and restores to a tuple."""
    from resonance_lattice.state.claim_io import (
        experience_claim_from_row,
        experience_claim_to_row,
    )

    import dataclasses as _dc

    c = _claim()
    row = experience_claim_to_row(c)
    polarity_is_list = isinstance(row["polarity"], list)
    recovered = experience_claim_from_row(row)
    ok = (
        recovered == c
        and isinstance(recovered.facts.polarity, tuple)
        and recovered.facts.is_bad is False      # default preserved
        and recovered.facts.attribute_key == ""  # default preserved
        and row["source"] == "experience"        # the read discriminator
    )

    # attribute_key round-trips when set (the serve-time dedup grouping key)...
    keyed = _dc.replace(c, facts=_dc.replace(c.facts, attribute_key="powershell version"))
    ok_keyed = (experience_claim_from_row(experience_claim_to_row(keyed))
                .facts.attribute_key == "powershell version")

    # ...and a LEGACY row missing attribute_key loads with the "" default — the
    # from_row tolerance (older insight.jsonl / experience-store rows must load).
    legacy_row = experience_claim_to_row(c)
    legacy_row.pop("attribute_key", None)
    ok_legacy = experience_claim_from_row(legacy_row).facts.attribute_key == ""

    return 0 if check_guarantee(
        polarity_is_list and ok and ok_keyed and ok_legacy,
        "(h) experience round-trip + attribute_key round-trip + legacy-row tolerance", _P) else 1


def _check_citation_external_round_trip() -> int:
    """InsightCitation.source_url: a corpus citation OMITS it (legacy byte-identical
    round-trip), an external citation carries the URL and round-trips, and a legacy
    dict without the key deserialises to source_url=None (backward compatible)."""
    from resonance_lattice.store.insight import (
        InsightCitation,
        _citation_from_dict,
        _citation_to_dict,
    )
    corpus = InsightCitation(passage_id="abc123", char_span=(0, 10), confidence=0.9)
    d = _citation_to_dict(corpus)
    ok_corpus = "source_url" not in d and _citation_from_dict(d) == corpus and not corpus.is_external

    ext = InsightCitation(passage_id="external:deadbeef", char_span=None,
                          confidence=1.0, source_url="https://a.example/x")
    de = _citation_to_dict(ext)
    ok_ext = (de.get("source_url") == "https://a.example/x"
              and _citation_from_dict(de) == ext and ext.is_external)

    legacy = _citation_from_dict({"passage_id": "p1", "char_span": None, "confidence": 1.0})
    ok_legacy = legacy.source_url is None and not legacy.is_external

    return 0 if check_guarantee(
        ok_corpus and ok_ext and ok_legacy,
        "(i) InsightCitation.source_url round-trips, omitted-when-None, backward-compatible", _P) else 1


def run() -> int:
    failures = 0
    for check in (
        _check_empty_merge,
        _check_embeddings_shape_reject,
        _check_duplicate_claim_id_reject,
        _check_empty_delete,
        _check_core_round_trip,
        _check_writer_identity,
        _check_content_key_identity,
        _check_experience_claim_round_trip,
        _check_citation_external_round_trip,
    ):
        failures += check()
    if failures:
        print(f"[{_P}] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print(f"[{_P}] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
