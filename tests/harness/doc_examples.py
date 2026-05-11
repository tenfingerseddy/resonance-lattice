"""Doc-example harness — placeholder.

The eventual contract: parse ```bash and ```python fenced blocks under
`docs/user/*.md`; run each in a sandboxed tempdir; compare stdout
against embedded `# expect:` comments. A static linter (parse fenced
shell blocks, validate referenced subcommands + flags against the
argparse surface) is the cheaper interim step; full execution is the
right-shape end state.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[doc_examples] SKIP — execution harness not built", file=sys.stderr)
    return 0
