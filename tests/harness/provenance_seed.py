"""provenance_seed — the provenance-tier TRUST PRIOR (Claude-in-loop keystone).

A higher-trust SOURCE must seed a higher STARTING confidence: a user-vouched fact > a cross-source-verified
external fact > a single web source ~ a corpus synthesis. Pins:

  (a) `seed_confidence` corroboration is strictly ordered user > verified_external > single_external == corpus,
      and the falsification prior is unchanged across tiers (only corroboration is lifted).
  (b) `provenance_tier` derives the right default from citations (>=2 external -> verified_external; 1 external ->
      single_external; corpus-anchored -> corpus) and NEVER infers "user".
  (c) `new_corpus_claim` lands each tier in the expected confidence band, and an external fill (auto-derived) seeds
      strictly higher trust than a corpus fill at the SAME faithfulness — the trust model, end to end.
  (d) the default ("corpus") is byte-identical to the pre-provenance seed — no existing caller shifts.

Pure (no encoder / no archive) — seeds + claim minting are deterministic maths."""
from __future__ import annotations

import sys

from resonance_lattice.store.insight import (
    InsightCitation,
    beta_mean,
    confidence_band,
    provenance_tier,
    seed_confidence,
)


def _corr(p, f=0.8):
    return seed_confidence(f, provenance=p)[0]


def _trust(p, f=0.8):
    c, fa = seed_confidence(f, provenance=p)
    return beta_mean(c, fa)


def _check_seed_ordering() -> int:
    f = 0.8
    # (a) corroboration strictly ordered user > verified_external > single_external == corpus
    cu, cv, cs, cc = (_corr("user", f), _corr("verified_external", f),
                      _corr("single_external", f), _corr("corpus", f))
    if not (cu > cv > cs == cc):
        print(f"[provenance_seed] (a) corroboration not ordered: user={cu} ver={cv} single={cs} corpus={cc}",
              file=sys.stderr)
        return 1
    # falsification identical across tiers (only corroboration is lifted)
    fals = {p: seed_confidence(f, provenance=p)[1] for p in ("user", "verified_external", "single_external", "corpus")}
    if len(set(round(v, 9) for v in fals.values())) != 1:
        print(f"[provenance_seed] (a) falsification should not vary by tier: {fals}", file=sys.stderr)
        return 1
    # (d) default == corpus == pre-provenance behaviour (no boost)
    if seed_confidence(f) != seed_confidence(f, provenance="corpus"):
        print("[provenance_seed] (d) default must equal corpus tier", file=sys.stderr)
        return 1
    if seed_confidence(f, provenance="corpus") != (1.0 + 2.0 * f, 1.0 + 2.0 * (1.0 - f)):
        print("[provenance_seed] (d) corpus tier must match the unboosted seed", file=sys.stderr)
        return 1
    # an unknown tier degrades to no boost (== corpus)
    if seed_confidence(f, provenance="bogus") != seed_confidence(f, provenance="corpus"):
        print("[provenance_seed] (d) unknown tier must degrade to no boost", file=sys.stderr)
        return 1
    return 0


def _ext(url):
    import hashlib
    return InsightCitation(passage_id="external:" + hashlib.sha256(url.encode()).hexdigest()[:16],
                           char_span=None, confidence=0.9, source_url=url)


def _check_tier_derivation() -> int:
    two_ext = (_ext("https://a.example"), _ext("https://b.example"))
    one_ext = (_ext("https://a.example"),)
    corpus = (InsightCitation(passage_id="corpus-passage-1", char_span=None, confidence=0.9),)
    mixed = (_ext("https://a.example"),
             InsightCitation(passage_id="corpus-passage-1", char_span=None, confidence=0.9))
    cases = {
        "verified_external": provenance_tier(two_ext),
        "single_external": provenance_tier(one_ext),
        "corpus_only": provenance_tier(corpus),
        "empty": provenance_tier(()),
        "mixed": provenance_tier(mixed),  # not all-external -> corpus-anchored
    }
    if cases["verified_external"] != "verified_external":
        print(f"[provenance_seed] (b) two external -> {cases['verified_external']}", file=sys.stderr); return 1
    if cases["single_external"] != "single_external":
        print(f"[provenance_seed] (b) one external -> {cases['single_external']}", file=sys.stderr); return 1
    if cases["corpus_only"] != "corpus" or cases["empty"] != "corpus" or cases["mixed"] != "corpus":
        print(f"[provenance_seed] (b) corpus/empty/mixed misclassified: {cases}", file=sys.stderr); return 1
    # two external citations that are the SAME url are NOT 2 distinct sources -> single_external
    if provenance_tier((_ext("https://a.example"), _ext("https://a.example"))) != "single_external":
        print("[provenance_seed] (b) duplicate-url external should be single_external", file=sys.stderr); return 1
    return 0


def _check_end_to_end_claim() -> int:
    from resonance_lattice.store.corpus_claim_io import new_corpus_claim

    def mint(citations, provenance=None):
        return new_corpus_claim(
            content="A verified fact.", kind="synthesis", citations=tuple(citations),
            source_model_hash="m", source_passage_hashes=("h1",), faithfulness=0.8,
            provenance=provenance)

    two_ext = (_ext("https://a.example"), _ext("https://b.example"))
    corpus = (InsightCitation(passage_id="corpus-passage-1", char_span=None, confidence=0.9),)

    corpus_claim = mint(corpus)                          # auto -> corpus
    ext_claim = mint(two_ext)                             # auto -> verified_external
    user_claim = mint(corpus, provenance="user")         # explicit -> user

    tc = beta_mean(corpus_claim.corroboration, corpus_claim.falsification)
    te = beta_mean(ext_claim.corroboration, ext_claim.falsification)
    tu = beta_mean(user_claim.corroboration, user_claim.falsification)
    # (c) the trust model, end to end: user > verified_external > corpus, at the same faithfulness
    if not (tu > te > tc):
        print(f"[provenance_seed] (c) claim trust not ordered: user={tu:.3f} ext={te:.3f} corpus={tc:.3f}",
              file=sys.stderr)
        return 1
    bands = (confidence_band(tc), confidence_band(te), confidence_band(tu))
    if bands != ("medium", "high", "verified"):
        print(f"[provenance_seed] (c) bands not (medium,high,verified): {bands} "
              f"(trusts {tc:.3f}/{te:.3f}/{tu:.3f})", file=sys.stderr)
        return 1
    # the auto-derived external seed must equal the explicit one (no hidden divergence)
    if mint(two_ext, provenance="verified_external").corroboration != ext_claim.corroboration:
        print("[provenance_seed] (c) auto-derived tier != explicit tier", file=sys.stderr); return 1
    return 0


def run() -> int:
    for check in (_check_seed_ordering, _check_tier_derivation, _check_end_to_end_claim):
        rc = check()
        if rc != 0:
            return rc
    print("[provenance_seed] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
