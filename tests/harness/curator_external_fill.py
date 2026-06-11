"""curator_external_fill — author a VERIFIED claim from EXTERNAL sources for a TRUE gap.

`curator.external_fill.author_external_fill(question, client, fetcher)` is the injected-capability path that fills a
gap the corpus cannot ground: fetch candidate sources, require >= min_sources AGREEING ones, distil one claim, and
return a `PendingFill` in the same shape the corpus author returns. Pins (the never-serve-an-unverified-fact guard
is the load-bearing one):

  (a) two agreeing sources + an author that confirms agreement -> a PendingFill grounded in BOTH, tagged external.
  (b) a single fetched source -> None (cannot cross-verify).
  (c) author says agree=false -> None.
  (d) author returns an empty claim -> None.
  (e) author claims a fact but only ONE source actually supports it -> None (the structural cross-source guard
      overrides the model's own agreement flag).
  (f) no client / no fetcher / empty question -> None.
  (g) a fetcher that raises -> None (never breaks the growth loop).

Hermetic: the fetcher and client are deterministic stubs; no network, no encoder, no API.
"""
from __future__ import annotations

import json
import sys

from resonance_lattice.curator.external_fill import author_external_fill


class _Block:
    def __init__(self, text): self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _StubClient:
    """Returns a canned JSON for the external-author system prompt; `.messages.create` shape only."""

    def __init__(self, author_json: str):
        self._json = author_json
        self.messages = self

    def create(self, *, model, max_tokens, system, messages):
        return _Resp(self._json)


def _fetch_two(_q):
    return [{"url": "https://a.example/x", "text": "The widget ships in March 2027 per the vendor announcement."},
            {"url": "https://b.example/y", "text": "Vendor confirms a March 2027 ship date for the widget."}]


def _fetch_one(_q):
    return [{"url": "https://a.example/x", "text": "The widget ships in March 2027."}]


def _agree(claim="The widget ships in March 2027.", support=(1, 2), agree=True):
    return json.dumps({"claim": claim, "supporting_sources": list(support), "agree": agree})


def _check_two_agreeing() -> int:
    pf = author_external_fill("When does the widget ship?", _StubClient(_agree()), _fetch_two)
    if pf is None:
        print("[curator_external_fill] (a) two agreeing sources should author a fill", file=sys.stderr); return 1
    if len(pf.evidence_passages) != 2:
        print(f"[curator_external_fill] (a) expected 2 cited sources, got {len(pf.evidence_passages)}",
              file=sys.stderr); return 1
    if any(e.get("drift_status") != "external" for e in pf.evidence_passages):
        print("[curator_external_fill] (a) evidence must be tagged external", file=sys.stderr); return 1
    if not all(str(e.get("content_hash", "")).startswith("sha256:") for e in pf.evidence_passages):
        print("[curator_external_fill] (a) evidence must carry a content hash", file=sys.stderr); return 1
    if pf.intent != "When does the widget ship?" or not pf.claim:
        print("[curator_external_fill] (a) intent/claim not set", file=sys.stderr); return 1
    return 0


def _check_single_source() -> int:
    if author_external_fill("q?", _StubClient(_agree(support=(1,))), _fetch_one) is not None:
        print("[curator_external_fill] (b) a single fetched source cannot cross-verify -> None", file=sys.stderr)
        return 1
    return 0


def _check_disagree() -> int:
    if author_external_fill("q?", _StubClient(_agree(agree=False)), _fetch_two) is not None:
        print("[curator_external_fill] (c) agree=false must yield None", file=sys.stderr); return 1
    return 0


def _check_empty_claim() -> int:
    if author_external_fill("q?", _StubClient(_agree(claim="")), _fetch_two) is not None:
        print("[curator_external_fill] (d) empty claim must yield None", file=sys.stderr); return 1
    return 0


def _check_single_support_overrides() -> int:
    # Two sources fetched, but the author flags only ONE as supporting -> the structural guard rejects.
    if author_external_fill("q?", _StubClient(_agree(support=(1,))), _fetch_two) is not None:
        print("[curator_external_fill] (e) a single supporting source must yield None even with agree=true",
              file=sys.stderr); return 1
    return 0


def _check_missing_capabilities() -> int:
    if author_external_fill("q?", None, _fetch_two) is not None:
        print("[curator_external_fill] (f) no client -> None", file=sys.stderr); return 1
    if author_external_fill("q?", _StubClient(_agree()), None) is not None:
        print("[curator_external_fill] (f) no fetcher -> None", file=sys.stderr); return 1
    if author_external_fill("   ", _StubClient(_agree()), _fetch_two) is not None:
        print("[curator_external_fill] (f) empty question -> None", file=sys.stderr); return 1
    return 0


def _check_fetcher_raises() -> int:
    def _boom(_q):
        raise RuntimeError("network down")
    if author_external_fill("q?", _StubClient(_agree()), _boom) is not None:
        print("[curator_external_fill] (g) a raising fetcher must degrade to None", file=sys.stderr); return 1
    return 0


def run() -> int:
    for check in [
        _check_two_agreeing,
        _check_single_source,
        _check_disagree,
        _check_empty_claim,
        _check_single_support_overrides,
        _check_missing_capabilities,
        _check_fetcher_raises,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[curator_external_fill] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
