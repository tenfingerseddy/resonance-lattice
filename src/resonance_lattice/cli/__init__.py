"""CLI surface — `rlat <command>`.

Each command lives in its own module under `cli/` and registers its parser
via `add_subparser(sub)`; the `add_subparser(sub)` call sequence in
`cli/app.build_parser()` is the authoritative registration list.

`serve` was dropped (no current consumer — MCP is the planned bridge
for a future version).
"""
