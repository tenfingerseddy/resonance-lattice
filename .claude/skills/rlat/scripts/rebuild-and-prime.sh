#!/usr/bin/env bash
# Rebuild (or sync) a knowledge model and regenerate the assistant primer.
# Usage: rebuild-and-prime.sh [knowledge-model] [sources...]
#
# Defaults:
#   knowledge-model = .rlat/project.rlat
#   sources         = ./docs ./src
#
# If the knowledge model does not exist, runs `rlat build`.
# If it exists, runs `rlat sync` for incremental update.
# Then regenerates the summary primer.

set -euo pipefail

CARTRIDGE="${1:-.rlat/project.rlat}"  # variable name retained for back-compat
shift 2>/dev/null || true
SOURCES="${@:-./docs ./src}"

if [ ! -f "$CARTRIDGE" ]; then
    echo "Building new knowledge model: $CARTRIDGE"
    mkdir -p "$(dirname "$CARTRIDGE")"
    rlat build $SOURCES -o "$CARTRIDGE"
else
    echo "Syncing knowledge model: $CARTRIDGE"
    rlat sync "$CARTRIDGE" $SOURCES
fi

PRIMER=".claude/resonance-context.md"
mkdir -p "$(dirname "$PRIMER")"
echo "Generating primer: $PRIMER"
rlat summary "$CARTRIDGE" -o "$PRIMER"

echo "Done. Knowledge model: $CARTRIDGE | Primer: $PRIMER"
