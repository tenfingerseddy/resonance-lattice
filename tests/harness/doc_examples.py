"""Doc-example harness — placeholder, deferred to v2.0.1.

The eventual contract: parse ```bash and ```python fenced blocks under
`docs/user/*.md`; run each in a sandboxed tempdir; compare stdout against
embedded `# expect:` comments. Until that lands, this suite is a no-op
that always passes — it is intentionally NOT counted toward the harness
"green on every commit" tally in REBUILD_PLAN.md. A static linter
(parse fenced shell blocks, validate referenced subcommands + flags
against the argparse surface) is the cheaper interim step on the v2.0.1
list; full execution is the right-shape end state.
"""


def run() -> int:
    return 0
