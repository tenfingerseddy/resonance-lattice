"""agent_fill — LAND an agent-fetched, agent-verified external fact with NO metered LLM gate (the free loop).

`curator.agent_fill.land_external_fact(km, question, claim, sources)` builds external evidence from the agent's
fetched sources and lands it via the caller-verified promotion path. Pins:

  (a) two distinct agreeing sources -> the claim LANDS, carrying external provenance (is_external citations), at
      the verified_external trust tier — no API client used.
  (b) fewer than 2 DISTINCT sources (one source, or two with the same url) -> does NOT land (cross-source guard).
  (c) an empty claim -> does NOT land.

Real encoder + real corpus (the landing must round-trip through promotion); no API, no network."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ._testutil import build_corpus as _build
from ._testutil import unpatch_zero_encoder


_SRC_A = {"url": "https://a.example/spec", "text": "The FooWidget API rate limit is 5000 requests per hour."}
_SRC_B = {"url": "https://b.example/docs", "text": "FooWidget enforces a cap of 5000 requests per hour per key."}
_CLAIM = "The FooWidget API rate limit is 5000 requests per hour."
_Q = "what is the FooWidget API rate limit"


def _km(d):
    return _build(Path(d) / "corpus", {
        "a.md": "# FooWidget\n\nFooWidget is a service for managing widgets.",
        "b.md": "# Usage\n\nCall the FooWidget API with an authenticated key.",
    })


def _check_lands() -> int:
    from resonance_lattice.curator.agent_fill import land_external_fact
    from resonance_lattice.store import archive
    from resonance_lattice.store.insight import beta_mean, confidence_band
    with tempfile.TemporaryDirectory() as d:
        km = _km(d)
        landed, outcomes = land_external_fact(km, _Q, _CLAIM, [_SRC_A, _SRC_B], faithfulness=0.9)
        if not landed or not any(o.promoted for o in outcomes):
            print(f"[agent_fill] (a) verified external fact did not land: {outcomes}", file=sys.stderr); return 1
        ins = archive.read(km).insights
        ext = [c for c in ins if c.facts.citations and all(cit.is_external for cit in c.facts.citations)]
        if len(ext) != 1:
            print(f"[agent_fill] (a) expected 1 external claim in band, got {len(ext)}", file=sys.stderr); return 1
        claim = ext[0]
        if claim.content != _CLAIM:
            print(f"[agent_fill] (a) wrong content landed: {claim.content!r}", file=sys.stderr); return 1
        band = confidence_band(beta_mean(claim.corroboration, claim.falsification))
        if band not in ("high", "verified"):
            print(f"[agent_fill] (a) verified_external should seed >= high, got {band}", file=sys.stderr); return 1
        # provenance is visible: every citation carries a source_url
        if not all(cit.source_url for cit in claim.facts.citations):
            print("[agent_fill] (a) landed external claim missing source_url provenance", file=sys.stderr); return 1
    return 0


def _check_cross_source_guard() -> int:
    from resonance_lattice.curator.agent_fill import land_external_fact
    from resonance_lattice.store import archive
    with tempfile.TemporaryDirectory() as d:
        km = _km(d)
        # one source -> cannot cross-verify
        if land_external_fact(km, _Q, _CLAIM, [_SRC_A], faithfulness=0.9)[0]:
            print("[agent_fill] (b) single source must NOT land", file=sys.stderr); return 1
        # two sources but SAME url -> not 2 distinct sources
        dup = {"url": _SRC_A["url"], "text": "A second blurb, same source url."}
        if land_external_fact(km, _Q, _CLAIM, [_SRC_A, dup], faithfulness=0.9)[0]:
            print("[agent_fill] (b) duplicate-url sources must NOT land", file=sys.stderr); return 1
        if archive.read(km).insights:
            print("[agent_fill] (b) nothing should have landed", file=sys.stderr); return 1
    return 0


def _check_empty_claim() -> int:
    from resonance_lattice.curator.agent_fill import land_external_fact
    with tempfile.TemporaryDirectory() as d:
        km = _km(d)
        if land_external_fact(km, _Q, "  ", [_SRC_A, _SRC_B], faithfulness=0.9)[0]:
            print("[agent_fill] (c) empty claim must NOT land", file=sys.stderr); return 1
    return 0


def run() -> int:
    unpatch_zero_encoder()
    for check in (_check_lands, _check_cross_source_guard, _check_empty_claim):
        rc = check()
        if rc != 0:
            return rc
    print("[agent_fill] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
