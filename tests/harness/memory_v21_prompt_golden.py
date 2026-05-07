"""memory_v21_prompt_golden — distil prompt golden (Appendix D D.7).

Pins four guarantees on `src/resonance_lattice/memory/prompts/distil.md`:

  (a) Prompt file loads + parses; required `{transcript}`,
      `{user_id}`, `{workspace_path}`, `{workspace_hash}` placeholders
      are present.

  (b) Version header `# version: N` + `# last-modified: YYYY-MM-DD` is
      present in the locked shape from §22 closing block.

  (c) For each fixture transcript, the parsed JSON output from the
      canned LLM response, after the orchestrator's validation +
      workspace-hash interpolation, satisfies the structural
      expectations in `_fixtures/distil_golden.json` (primary
      polarity match, required scope tags, required text keywords).

  (d) Bumping the prompt version without re-judging the golden flips
      the suite to a deterministic FAIL with a clear "golden requires
      re-judge" message. Prevents §18.7 distil drift from sneaking in.

Hermetic — no live LLM, mock LLMClient returns canned responses.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

from ._testutil import make_stub_llm_client, patch_zero_encoder


_GOLDEN_PATH = Path(__file__).resolve().parent / "_fixtures" / "distil_golden.json"
_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "resonance_lattice" / "memory" / "prompts" / "distil.md"
)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _load_golden() -> dict:
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (a) Prompt loads + required placeholders present
# ---------------------------------------------------------------------------


def _check_prompt_loads() -> int:
    if not _PROMPT_PATH.exists():
        print(f"[memory_v21_prompt_golden] FAIL (a): prompt missing at "
              f"{_PROMPT_PATH}", file=sys.stderr)
        return 1
    text = _load_prompt()
    required = ["{transcript}", "{user_id}", "{workspace_path}", "{workspace_hash}"]
    missing = [tok for tok in required if tok not in text]
    if missing:
        print(f"[memory_v21_prompt_golden] FAIL (a): missing placeholders "
              f"{missing} in {_PROMPT_PATH}", file=sys.stderr)
        return 1
    print("[memory_v21_prompt_golden] (a) prompt loads + placeholders "
          "present OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) Version header in the locked shape
# ---------------------------------------------------------------------------


_VERSION_RE = re.compile(r"^#\s*version:\s*(\d+)\s*$", re.MULTILINE)
_LAST_MODIFIED_RE = re.compile(
    r"^#\s*last-modified:\s*\d{4}-\d{2}-\d{2}\s*$", re.MULTILINE
)


def _check_version_header() -> int:
    text = _load_prompt()
    version_match = _VERSION_RE.search(text)
    if not version_match:
        print(f"[memory_v21_prompt_golden] FAIL (b): no `# version: N` "
              f"line found in {_PROMPT_PATH}", file=sys.stderr)
        return 1
    if not _LAST_MODIFIED_RE.search(text):
        print(f"[memory_v21_prompt_golden] FAIL (b): no `# last-modified: "
              f"YYYY-MM-DD` line found", file=sys.stderr)
        return 1
    print(f"[memory_v21_prompt_golden] (b) version header present "
          f"(version={version_match.group(1)}) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) Fixture transcripts satisfy golden expectations
# ---------------------------------------------------------------------------


def _check_golden_match() -> int:
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.distil import distil
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    golden = _load_golden()
    cwd_tag = workspace_tag_for_cwd("/proj")

    for tx_id, fixture in golden["transcripts"].items():
        with tempfile.TemporaryDirectory() as td:
            memory = Memory(root=Path(td) / "u")
            for cap in fixture["captures"]:
                memory.add_row(
                    text=cap["text"],
                    polarity=["factual", cwd_tag],
                    transcript_hash=cap["transcript_hash"],
                )

            result = distil(
                store=memory,
                redactor=Redactor(),
                client=make_stub_llm_client(fixture["canned_llm_response"]),
                workspace_path="/proj",
                all_rows=True,
            )

            expected = fixture["expected"]
            # Distilled rows = those returned in `result.written_row_ids`.
            # Walking the sidecar would require subtracting captures;
            # the result already names the new ones we care about.
            rows, _ = memory.read_all()
            distilled = [r for r in rows if r.row_id in result.written_row_ids]

            if len(distilled) != len(expected):
                print(f"[memory_v21_prompt_golden] FAIL (c) [{tx_id}]: "
                      f"expected {len(expected)} distilled rows; got "
                      f"{len(distilled)}: "
                      f"{[(r.text[:40], r.polarity) for r in distilled]}",
                      file=sys.stderr)
                return 1

            for got_row, exp in zip(distilled, expected):
                primary = next(p for p in got_row.polarity
                               if p in {"prefer", "avoid", "factual"})
                if primary != exp["primary_polarity"]:
                    print(f"[memory_v21_prompt_golden] FAIL (c) [{tx_id}]: "
                          f"primary polarity {primary!r} != "
                          f"{exp['primary_polarity']!r}", file=sys.stderr)
                    return 1
                # Required scope-tag literal:
                for required in exp.get("scope_tags_must_include", []):
                    if required not in got_row.polarity:
                        print(f"[memory_v21_prompt_golden] FAIL (c) "
                              f"[{tx_id}]: scope tag {required!r} missing "
                              f"from {got_row.polarity}", file=sys.stderr)
                        return 1
                # Required scope-tag pattern (e.g., "workspace:"):
                pat = exp.get("scope_tags_must_include_pattern")
                if pat is not None:
                    if not any(t.startswith(pat) for t in got_row.polarity):
                        print(f"[memory_v21_prompt_golden] FAIL (c) "
                              f"[{tx_id}]: no tag matching pattern "
                              f"{pat!r} in {got_row.polarity}",
                              file=sys.stderr)
                        return 1
                # Required text keywords:
                lowered = got_row.text.lower()
                for kw in exp.get("text_keywords_required", []):
                    if kw.lower() not in lowered:
                        print(f"[memory_v21_prompt_golden] FAIL (c) "
                              f"[{tx_id}]: keyword {kw!r} missing from "
                              f"text {got_row.text!r}", file=sys.stderr)
                        return 1
    print(f"[memory_v21_prompt_golden] (c) all "
          f"{len(golden['transcripts'])} fixtures satisfy golden OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) Version bump without golden re-judge → suite fails
# ---------------------------------------------------------------------------


def _check_version_bump_invalidates() -> int:
    text = _load_prompt()
    version_match = _VERSION_RE.search(text)
    if not version_match:
        print(f"[memory_v21_prompt_golden] FAIL (d): version header missing",
              file=sys.stderr)
        return 1
    prompt_version = int(version_match.group(1))
    golden_version = _load_golden().get("prompt_version")
    if golden_version != prompt_version:
        print(f"[memory_v21_prompt_golden] FAIL (d): prompt version "
              f"{prompt_version} != golden version {golden_version}. "
              f"Re-judge the golden in tests/harness/_fixtures/"
              f"distil_golden.json against prompt v{prompt_version} and "
              f"bump `prompt_version` to match.", file=sys.stderr)
        return 1
    print(f"[memory_v21_prompt_golden] (d) golden tracks prompt version "
          f"v{prompt_version} (no drift) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_prompt_loads,
        _check_version_header,
        _check_golden_match,
        _check_version_bump_invalidates,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_prompt_golden] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
