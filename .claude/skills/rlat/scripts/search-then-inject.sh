#!/usr/bin/env bash
# Search a knowledge model and output context-format results for LLM injection.
# Usage: search-then-inject.sh <knowledge-model> <query> [mode]
#
# Defaults:
#   mode = augment
#
# Uses --no-worker since this is typically called from scripts or subagents.
# Output is context format, ready for piping into an LLM prompt.

set -euo pipefail

CARTRIDGE="${1:?Usage: search-then-inject.sh <knowledge-model> <query> [mode]}"  # variable name retained for back-compat
QUERY="${2:?Usage: search-then-inject.sh <knowledge-model> <query> [mode]}"
MODE="${3:-augment}"

rlat search "$CARTRIDGE" "$QUERY" \
    --format context \
    --mode "$MODE" \
    --no-worker
