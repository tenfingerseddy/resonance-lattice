"""Docs-truth harness — docs/site behavioural claims validated against code.

Three static checks, all fast, no network and no encoder:

1. Command existence — every ``rlat <command> [<subcommand>]`` mention
   inside a ``<code>``/``<pre>`` block in ``docs/site/*.html`` resolves
   against the live argparse surface, so a renamed or retired verb cannot
   survive in the docs.
2. LLM-usage truth — the set of CLI modules that reach an LLM client is
   derived from source (the lazy ``import anthropic`` sites and the
   ``_maybe_llm_client`` resolver) and asserted against the declared
   table below. A new LLM call site fails this suite until the table —
   and therefore the docs — are updated.
3. Keyed-verb presence — every keyed verb in the declared table is named
   on ``faq.html`` and ``api-keys.html``, the two pages users read to
   answer "do I need a key?".

This suite exists because behavioural claims (key requirements, "runs
automatically") drifted from code on three separate audit cycles that
each read pages in isolation.
"""

from __future__ import annotations

import argparse
import html as html_mod
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SITE = _REPO_ROOT / "docs" / "site"
_CLI = _REPO_ROOT / "src" / "resonance_lattice" / "cli"

# The declared truth table. Check 2 keeps this honest against the code;
# check 3 keeps the docs honest against this.
LLM_TOUCHING_CLI_MODULES = {
    "deep_search.py",   # the multi-hop research loop
    "grow.py",          # gap-fill authoring (--dry-run is LLM-free)
    "reverify.py",      # stale-insight re-judge
    "probe.py",         # idle-cycle self-probe (runs deep-search)
    "intent.py",        # decompose requires a key; what-next degrades
    "memory.py",        # verify requires a key (via intent's resolver)
}
KEYED_VERB_DOC_STRINGS = (
    "rlat deep-search",
    "rlat grow",
    "rlat reverify",
    "rlat probe",
    "rlat memory verify",
    "rlat intent decompose",
)
_KEY_ANSWER_PAGES = ("faq.html", "api-keys.html")

_LLM_SITE_RE = re.compile(r"^\s*(?:import anthropic\b|from .intent import _maybe_llm_client|def _maybe_llm_client)", re.M)
_CODE_BLOCK_RE = re.compile(r"<(code|pre)[^>]*>(.*?)</\1>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
# Only command position counts: the snippet (or shell line) must START
# with `rlat ` — so `.rlat` filenames, `pip install rlat`, and HF repo
# names never parse as commands. The second token must end at
# whitespace/EOL, so `km.rlat`, `fabric://…`, `--flags`, and
# placeholders like &lt;subcmd&gt; never validate as subcommands.
_MENTION_RE = re.compile(r"^rlat\s+([a-z][a-z-]*)(?:\s+([a-z][a-z-]*)(?![\w:./@-]))?")


def _cli_surface() -> dict[str, set[str]]:
    from resonance_lattice.cli.app import build_parser

    parser = build_parser()
    subs = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    surface: dict[str, set[str]] = {}
    for name, sub in subs.choices.items():
        inner = [
            a for a in sub._actions if isinstance(a, argparse._SubParsersAction)
        ]
        surface[name] = set(inner[0].choices) if inner else set()
    return surface


def _command_lines(page: Path) -> list[str]:
    raw = page.read_text(encoding="utf-8")
    lines: list[str] = []
    for m in _CODE_BLOCK_RE.finditer(raw):
        text = html_mod.unescape(_TAG_RE.sub(" ", m.group(2)))
        for line in text.splitlines():
            line = line.strip().removeprefix("$ ").lstrip()
            if line.startswith("rlat "):
                lines.append(line)
    return lines


def _check_command_mentions(surface: dict[str, set[str]]) -> list[str]:
    errors: list[str] = []
    for page in sorted(_SITE.glob("*.html")):
        for line in _command_lines(page):
            m = _MENTION_RE.match(line)
            if not m:
                continue
            cmd, sub = m.group(1), m.group(2)
            if cmd not in surface:
                errors.append(f"{page.name}: unknown command `rlat {cmd}`")
                continue
            if sub and surface[cmd] and sub not in surface[cmd]:
                errors.append(
                    f"{page.name}: `rlat {cmd} {sub}` — no such subcommand"
                )
    return errors


def _check_llm_truth() -> list[str]:
    derived = {
        p.name
        for p in _CLI.glob("*.py")
        if _LLM_SITE_RE.search(p.read_text(encoding="utf-8"))
    }
    if derived == LLM_TOUCHING_CLI_MODULES:
        return []
    return [
        "LLM-touching CLI modules drifted from the declared table — "
        f"update LLM_TOUCHING_CLI_MODULES and the docs. derived={sorted(derived)} "
        f"declared={sorted(LLM_TOUCHING_CLI_MODULES)}"
    ]


def _check_keyed_verbs_documented() -> list[str]:
    errors: list[str] = []
    for page_name in _KEY_ANSWER_PAGES:
        text = html_mod.unescape(
            _TAG_RE.sub(" ", (_SITE / page_name).read_text(encoding="utf-8"))
        )
        text = re.sub(r"\s+", " ", text)
        for verb in KEYED_VERB_DOC_STRINGS:
            if verb not in text:
                errors.append(f"{page_name}: keyed verb `{verb}` not named")
    return errors


def run() -> int:
    if not _SITE.is_dir():
        print("[docs_truth] SKIP — docs/site not present", file=sys.stderr)
        return 2
    errors = (
        _check_command_mentions(_cli_surface())
        + _check_llm_truth()
        + _check_keyed_verbs_documented()
    )
    for e in errors:
        print(f"[docs_truth] FAIL {e}", file=sys.stderr)
    if errors:
        return 1
    print("[docs_truth] OK — command mentions, LLM table, keyed-verb docs")
    return 0


if __name__ == "__main__":
    sys.exit(run())
