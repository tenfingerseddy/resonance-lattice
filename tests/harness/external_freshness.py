"""external_freshness — re-fetch an external fill's source and re-judge whether the WORLD still supports it.

The complement of `reverification` (which only re-checks CORPUS drift and SKIPS external claims). Pins:

  (a) `external_claims` enumerates ONLY active external claims + their URLs (a corpus claim and a retired external
      claim are excluded) — the free, no-LLM input the skill drives.
  (b) re-fetch + judge says STILL SUPPORTED -> status "fresh".
  (c) re-fetch + judge says NO LONGER SUPPORTED (the world moved on) -> status "stale" (surfaced, NOT evicted).
  (d) no source re-fetchable -> status "unknown" (never wrongly stale); no client/fetcher -> [].

Real encoder + a real written insight band (the claims must round-trip); the url_fetcher + judge are stubs (no
network, no API)."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

from ._testutil import build_corpus as _build
from ._testutil import make_corpus_claim, unpatch_zero_encoder


class _Block:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class _StubJudge:
    """Returns a canned freshness verdict; `.messages.create` shape only. `calls` counts judge invocations."""

    def __init__(self, supports: bool):
        self._json = json.dumps({"supports": supports, "reason": "stub"})
        self.messages = self
        self.calls = 0

    def create(self, *, model, max_tokens, system, messages, **kwargs):
        self.calls += 1
        return _Resp(self._json)


def _make_external_claim(content, urls, *, state="active"):
    from resonance_lattice.state.claim import Claim, CorpusFacts
    from resonance_lattice.store.insight import (
        InsightCitation,
        compute_insight_id,
        seed_confidence,
    )
    cits = tuple(
        InsightCitation(passage_id="external:" + hashlib.sha256(u.encode()).hexdigest()[:16],
                        char_span=None, confidence=0.9, source_url=u)
        for u in urls
    )
    fp = compute_insight_id(content, tuple(urls), "model-x")
    corr, fals = seed_confidence(0.8)
    return Claim(
        claim_id=fp, source="corpus", kind="synthesis", content=content,
        created_at="2026-06-06T00:00:00Z", corroboration=corr, falsification=fals,
        trust_as_of="", state=state, parent_ids=(),
        facts=CorpusFacts(
            citations=cits, content_fingerprint=fp, source_model_hash="model-x",
            source_passage_hashes=tuple(urls), verdict_signals=(), query="q",
            intent_context=None, stale_if_sources_drift=False,
            encoder_version="gte-mb-768", seed_corroboration=corr, seed_falsification=fals),
    )


def _build_km_with_claims(d):
    """A corpus + an insight band holding: a corpus claim, an ACTIVE external claim (2 urls), a RETIRED external."""
    from resonance_lattice.field.encoder import Encoder
    from resonance_lattice.store import archive

    root = Path(d) / "corpus"
    km = _build(root, {"a.md": "# A\n\nThe sky is documented as blue.", "b.md": "# B\n\nWater is wet."})
    c0 = archive.read(km)
    src_ids = [c.passage_id for c in c0.registry]

    corpus_claim = make_corpus_claim("Synthesised from the corpus.", src_ids[:1], state="active")
    ext_active = _make_external_claim(
        "FooLib's latest stable release is version 3.2.",
        ["https://foolib.example/releases", "https://docs.foolib.example/changelog"], state="active")
    ext_retired = _make_external_claim(
        "An old retired external fact.", ["https://old.example/x", "https://old.example/y"], state="retired")
    insights = [corpus_claim, ext_active, ext_retired]
    band = Encoder().encode([i.content for i in insights]).astype("float32")
    archive.write_insight_layer_in_place(km, insights, band)
    return km, ext_active.claim_id


def _check_enumeration() -> int:
    from resonance_lattice.store import archive
    from resonance_lattice.store.external_freshness import external_claims
    with tempfile.TemporaryDirectory() as d:
        km, ext_id = _build_km_with_claims(d)
        refs = external_claims(archive.read(km))
        if len(refs) != 1:
            print(f"[external_freshness] (a) expected 1 active external claim, got {len(refs)}", file=sys.stderr)
            return 1
        r = refs[0]
        if r.claim_id != ext_id:
            print(f"[external_freshness] (a) wrong claim enumerated: {r.claim_id}", file=sys.stderr); return 1
        if set(r.source_urls) != {"https://foolib.example/releases", "https://docs.foolib.example/changelog"}:
            print(f"[external_freshness] (a) wrong urls: {r.source_urls}", file=sys.stderr); return 1
    return 0


def _check_fresh_and_stale() -> int:
    from resonance_lattice.store.external_freshness import recheck_external_freshness
    fetcher = lambda url: f"current content of {url}"  # noqa: E731 — every url re-fetches fine
    with tempfile.TemporaryDirectory() as d:
        km, ext_id = _build_km_with_claims(d)
        # (b) judge says still supported -> fresh
        fresh = recheck_external_freshness(km, fetcher, _StubJudge(True))
        if len(fresh) != 1 or fresh[0].status != "fresh" or fresh[0].claim_id != ext_id:
            print(f"[external_freshness] (b) expected fresh, got {fresh}", file=sys.stderr); return 1
        if set(fresh[0].refetched_urls) != {"https://foolib.example/releases", "https://docs.foolib.example/changelog"}:
            print(f"[external_freshness] (b) refetched urls wrong: {fresh[0].refetched_urls}", file=sys.stderr)
            return 1
        # (c) judge says no longer supported -> stale (surfaced, not evicted)
        stale = recheck_external_freshness(km, fetcher, _StubJudge(False))
        if len(stale) != 1 or stale[0].status != "stale":
            print(f"[external_freshness] (c) expected stale, got {stale}", file=sys.stderr); return 1
        # v1 surfaces only: the claim is still ACTIVE in the band (not retired/evicted)
        from resonance_lattice.store import archive
        states = {c.claim_id: c.state for c in archive.read(km).insights}
        if states.get(ext_id) != "active":
            print(f"[external_freshness] (c) v1 must NOT mutate the claim; state={states.get(ext_id)}",
                  file=sys.stderr); return 1
        # (b2) PARTIAL fetch — one url returns content, the other fails: still judged, only the fetched url recorded
        partial = lambda url: ("fresh page" if url.endswith("releases") else None)  # noqa: E731
        pj = _StubJudge(True)
        pout = recheck_external_freshness(km, partial, pj)
        if len(pout) != 1 or pout[0].status != "fresh":
            print(f"[external_freshness] (b2) partial fetch should still judge: {pout}", file=sys.stderr); return 1
        if pout[0].refetched_urls != ("https://foolib.example/releases",) or pj.calls != 1:
            print(f"[external_freshness] (b2) only the fetched url should be judged: {pout[0].refetched_urls} "
                  f"calls={pj.calls}", file=sys.stderr); return 1
    return 0


def _check_unknown_and_degrade() -> int:
    from resonance_lattice.store.external_freshness import recheck_external_freshness
    with tempfile.TemporaryDirectory() as d:
        km, _ = _build_km_with_claims(d)
        # (d) no source re-fetchable -> unknown, and the judge is never called
        judge = _StubJudge(False)
        out = recheck_external_freshness(km, lambda url: None, judge)
        if len(out) != 1 or out[0].status != "unknown":
            print(f"[external_freshness] (d) expected unknown, got {out}", file=sys.stderr); return 1
        if judge.calls != 0:
            print(f"[external_freshness] (d) judge must not be called when nothing re-fetched ({judge.calls})",
                  file=sys.stderr); return 1
        # no client / no fetcher -> []
        if recheck_external_freshness(km, lambda u: "x", None) != []:
            print("[external_freshness] (d) no client must yield []", file=sys.stderr); return 1
        if recheck_external_freshness(km, None, _StubJudge(True)) != []:
            print("[external_freshness] (d) no fetcher must yield []", file=sys.stderr); return 1
    return 0


def run() -> int:
    unpatch_zero_encoder()
    for check in (_check_enumeration, _check_fresh_and_stale, _check_unknown_and_degrade):
        rc = check()
        if rc != 0:
            return rc
    print("[external_freshness] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
