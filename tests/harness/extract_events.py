"""extract_events — atomic event extraction at capture, contract pins.

Pins Phase E/3a's fals.1: the extractor must either return
`list[(fact, polarity)]` (possibly empty) or `None` (failure path the
caller falls back on). It must never raise, never return a malformed
shape. Polarity ("factual"/"prefer"/"avoid") landed 2026-06 — the
valence the recall rerank weighs.

Thirteen contracts:

  (a) Successful extraction: well-formed LLM response → list of
      `(fact, polarity)` pairs; bare-string entries (the pre-polarity
      shape) are tolerated as "factual".
  (b) Empty extraction: LLM returns `{"facts": []}` → `[]`. A no-op
      session is distinct from a failed extraction.
  (c) `client is None` → `None` (caller falls back). The optional-LLM
      seam is preserved.
  (d) Empty/whitespace input → `[]` without an LLM call.
  (e) LLM raises → `None`. Failure must not propagate.
  (f) Non-JSON LLM response → `None`.
  (g) Non-dict JSON payload (e.g., a bare list) → `None`.
  (h) Missing `facts` key → `None`.
  (i) `facts` value not a list → `None`.
  (j) `facts` entry that is neither a string nor a fact dict → `None`
      (strict — partial extraction would silently lose data).
  (k) Strings are stripped; empty strings after stripping are dropped.
  (l) Empty fenced JSON block (e.g. ```json\\n```) → `None`. Pins the
      "never raise" contract against `parse_llm_json` edge cases.
  (m) Polarity classification: dict entries carry their polarity through;
      an unknown polarity value coerces to "factual"; a dict without
      "text" → `None`.

Hermetic — fake LLM client; no encoder, no network.
"""

from __future__ import annotations

import json
import sys

from resonance_lattice.memory.extract import extract_events

from ._testutil import (
    make_stub_llm_client, make_stub_llm_facts, make_stub_llm_raising,
)


def _check_successful_extraction() -> int:
    out = extract_events(
        "Session text about choosing the medallion architecture.",
        client=make_stub_llm_facts([
            "The project's data lake uses the medallion architecture.",
            "Bronze layer ingests Event Hubs JSON via Spark Structured Streaming.",
        ]),
    )
    if (out is None or len(out) != 2 or "medallion" not in out[0][0]
            or {pol for _, pol in out} != {"factual"}):
        print(f"[extract_events] FAIL (a): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (a) successful extraction (strings -> factual) OK",
          file=sys.stderr)
    return 0


def _check_empty_extraction() -> int:
    out = extract_events(
        "We just chatted about how the weather is.",
        client=make_stub_llm_facts([]),
    )
    if out != []:
        print(f"[extract_events] FAIL (b): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (b) empty extraction OK", file=sys.stderr)
    return 0


def _check_no_client() -> int:
    out = extract_events("anything", client=None)
    if out is not None:
        print(f"[extract_events] FAIL (c): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (c) no-client returns None OK", file=sys.stderr)
    return 0


def _check_empty_input() -> int:
    out_blank = extract_events("", client=make_stub_llm_facts(["should not see this"]))
    out_ws = extract_events("   \n\t  ", client=make_stub_llm_facts(["nope"]))
    if out_blank != [] or out_ws != []:
        print(f"[extract_events] FAIL (d): blank={out_blank!r} ws={out_ws!r}",
              file=sys.stderr)
        return 1
    print("[extract_events] (d) empty input short-circuits OK",
          file=sys.stderr)
    return 0


def _check_llm_raises() -> int:
    out = extract_events("text", client=make_stub_llm_raising(RuntimeError("net down")))
    if out is not None:
        print(f"[extract_events] FAIL (e): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (e) LLM exception returns None OK",
          file=sys.stderr)
    return 0


def _check_non_json() -> int:
    out = extract_events("text", client=make_stub_llm_client("not json at all"))
    if out is not None:
        print(f"[extract_events] FAIL (f): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (f) non-JSON returns None OK", file=sys.stderr)
    return 0


def _check_non_dict_payload() -> int:
    out = extract_events("text", client=make_stub_llm_client(
        json.dumps(["a", "b", "c"])
    ))
    if out is not None:
        print(f"[extract_events] FAIL (g): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (g) non-dict payload returns None OK",
          file=sys.stderr)
    return 0


def _check_missing_facts_key() -> int:
    out = extract_events("text", client=make_stub_llm_client(
        json.dumps({"steps": ["a", "b"]})
    ))
    if out is not None:
        print(f"[extract_events] FAIL (h): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (h) missing facts key returns None OK",
          file=sys.stderr)
    return 0


def _check_facts_not_a_list() -> int:
    out = extract_events("text", client=make_stub_llm_client(
        json.dumps({"facts": "a single string, not a list"})
    ))
    if out is not None:
        print(f"[extract_events] FAIL (i): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (i) facts not a list returns None OK",
          file=sys.stderr)
    return 0


def _check_non_string_entry() -> int:
    out = extract_events("text", client=make_stub_llm_client(
        json.dumps({"facts": ["valid string", 42, "another"]})
    ))
    if out is not None:
        print(f"[extract_events] FAIL (j): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (j) non-string entry returns None OK",
          file=sys.stderr)
    return 0


def _check_whitespace_handling() -> int:
    out = extract_events("text", client=make_stub_llm_facts([
        "  fact one with leading/trailing space  ",
        "",
        "   ",
        "fact two",
    ]))
    if out != [("fact one with leading/trailing space", "factual"),
               ("fact two", "factual")]:
        print(f"[extract_events] FAIL (k): {out!r}", file=sys.stderr)
        return 1
    print("[extract_events] (k) whitespace stripped, empties dropped OK",
          file=sys.stderr)
    return 0


def _check_empty_fenced_block() -> int:
    # Pins "never raise" against `parse_llm_json` edge cases where the
    # fence-stripping path can end with a non-JSON-shaped empty string.
    # Any exception from the parser must degrade to None, not propagate.
    edge_cases = [
        "```json\n```",       # empty fenced block with language tag
        "```\n```",            # empty fenced block, no language tag
        "```",                 # opening fence only
        "```json\n  \n```",   # whitespace-only fenced block
    ]
    for case in edge_cases:
        out = extract_events("text", client=make_stub_llm_client(case))
        if out is not None:
            print(f"[extract_events] FAIL (l): {case!r} returned {out!r}",
                  file=sys.stderr)
            return 1
    print("[extract_events] (l) parse_llm_json edge cases return None OK",
          file=sys.stderr)
    return 0


def _check_polarity_classification() -> int:
    out = extract_events("text", client=make_stub_llm_client(json.dumps({
        "facts": [
            {"text": "Tokens rotate weekly.", "polarity": "factual"},
            {"text": "User prefers compact tables.", "polarity": "prefer"},
            {"text": "Do not use wildcard imports.", "polarity": "avoid"},
            {"text": "Mystery valence.", "polarity": "sideways"},
            "bare string from an older-shaped reply",
        ],
    })))
    want = [
        ("Tokens rotate weekly.", "factual"),
        ("User prefers compact tables.", "prefer"),
        ("Do not use wildcard imports.", "avoid"),
        ("Mystery valence.", "factual"),
        ("bare string from an older-shaped reply", "factual"),
    ]
    if out != want:
        print(f"[extract_events] FAIL (m): {out!r}", file=sys.stderr)
        return 1
    missing_text = extract_events("text", client=make_stub_llm_client(
        json.dumps({"facts": [{"polarity": "avoid"}]})
    ))
    if missing_text is not None:
        print(f"[extract_events] FAIL (m): dict without text should be None, "
              f"got {missing_text!r}", file=sys.stderr)
        return 1
    print("[extract_events] (m) polarity classification + coercion OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_successful_extraction,
        _check_empty_extraction,
        _check_no_client,
        _check_empty_input,
        _check_llm_raises,
        _check_non_json,
        _check_non_dict_payload,
        _check_missing_facts_key,
        _check_facts_not_a_list,
        _check_non_string_entry,
        _check_whitespace_handling,
        _check_empty_fenced_block,
        _check_polarity_classification,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[extract_events] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
