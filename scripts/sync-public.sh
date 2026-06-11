#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# sync-public.sh — Export public-facing subset to resonance-lattice-cli
#
# Usage:
#   ./scripts/sync-public.sh              # dry-run (shows diff)
#   ./scripts/sync-public.sh --push       # commit and push to public remote
#
# Uses git worktree — never touches your working directory.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PUBLIC_REMOTE="public"
PUBLIC_BRANCH="main"
PUSH="${1:-}"
STAGING="__sync_staging"
WORK="$REPO_ROOT/.sync-worktree"

cleanup() {
    cd "$REPO_ROOT"
    git worktree remove --force "$WORK" 2>/dev/null || true
    git branch -D "$STAGING" 2>/dev/null || true
    rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

# ── Verify remote ────────────────────────────────────────────────────

if ! git remote get-url "$PUBLIC_REMOTE" &>/dev/null; then
    echo "ERROR: Remote '$PUBLIC_REMOTE' not found."
    echo "Add it: git remote add public https://github.com/tenfingerseddy/resonance-lattice.git"
    exit 1
fi

echo "Fetching $PUBLIC_REMOTE/$PUBLIC_BRANCH..."
git fetch "$PUBLIC_REMOTE" "$PUBLIC_BRANCH" 2>/dev/null || true

# ── Paths to include ─────────────────────────────────────────────────

INCLUDE=(
    # ── Top-level metadata ─────────────────────────────────────────
    ".gitignore"
    # Pages deploy workflow — the public repo serves the docs site
    # (v3: the public home; the private deployment is retired after
    # cutover). Pushing workflow files needs a token with `workflow`
    # scope (gh auth default has it).
    ".github/workflows/docs-pages.yml"
    # This script itself — tracked since v3 so the sync is reproducible.
    "scripts/sync-public.sh"
    "docs/assets/"
    # HTML documentation site (rendered on GitHub Pages). Includes the
    # landing page, shared stylesheet, and per-topic pages including
    # the Fabric subfolder (Resonance Lattice on Microsoft Fabric).
    "docs/site/"
    "CHANGELOG.md"
    "CODE_OF_CONDUCT.md"
    "CONTRIBUTING.md"
    "LICENSE.md"
    "NOTICE"
    "README.md"
    "SECURITY.md"
    "pyproject.toml"

    # ── Python source (BSL is source-available; Kane decision 2026-04-22) ──
    "src/"

    # ── Test harness (so users can run the contract suite) ─────────
    "tests/"

    # ── Internal docs that are public-facing reference material.
    #    (docs/user/ + OPTIMISE.md retired; the HTML site at docs/site/
    #    is the user-facing home.)
    "docs/internal/ARCHITECTURE.md"
    "docs/internal/BENCHMARK_GATE.md"
    "docs/internal/FIELD.md"
    "docs/internal/GROUNDING_MODEL.md"
    "docs/internal/HONEST_CLAIMS.md"
    "docs/internal/KNOWLEDGE_MODEL_FORMAT.md"
    "docs/internal/MEMORY.md"
    "docs/internal/RQL.md"
    "docs/internal/SKILL_INTEGRATION.md"
    "docs/internal/STORE.md"
    "docs/VISION.md"

    # ── User-facing benchmark methodology + reproduction harnesses.
    #    Ships the harnesses + tasks + fixtures + result JSONs that
    #    BENCHMARKS.md cites as reproducible. Only the ship numbers
    #    ship publicly — historical run variants (v2, v3, _postfix,
    #    5-lane intermediates) stay in the private repo as audit trail.
    "docs/internal/benchmarks/"
    "benchmarks/user_bench/"
    "benchmarks/results/user_bench/build_query_speed.json"
    "benchmarks/results/user_bench/hallucination_fabric_11lane.json"
    "benchmarks/results/user_bench/hallucination_fabric_11lane_relaxed.json"
    "benchmarks/results/user_bench/primer_effectiveness.json"
    "benchmarks/results/user_bench/token_usage_v2.json"
    "benchmarks/results/optimised/beir_fiqa_probe_v1.json"

    # ── v3 world-knowledge evidence ("no claim without a public
    #    receipt") — pre-registered designs, verdicts, items, and raw
    #    run artifacts, including the honest R4 failure record. The
    #    docs-site benchmarks page links into these paths.
    "benchmarks/constraint_band/"
    "benchmarks/constraint_band_xdomain/"
    "benchmarks/falsification_ledger/"
    "benchmarks/attribute_gate_e2c/"
    "benchmarks/r4_continuous_credit/"

    # ── Ecosystem: Claude skills ───────────────────────────────────
    ".claude/skills/rlat/"
    ".claude/skills/deep-research/"
    ".claude/skills/rlat-build-on-kaggle/"

    # ── Fabric integration (UDF only) ──────────────────────────────
    # UDF code users publish to their workspace + the user-facing
    # example notebooks (build, demo, analytics) FABRIC.md walks
    # through. The Fabric data agent (Eventhouse/KQL substrate) is
    # in development and is excluded from sync — its planning docs
    # (`.claude/plans/fabric-data-agent-*.md`), the deploy notebook
    # (`notebooks/examples/fabric_deploy.ipynb`), the data-agent UDF
    # variant (`fabric/data-agent/`), and the data-agent docs site
    # (`docs/site/fabric/`) are private until the design ships.
    "fabric/udf/"
    "notebooks/examples/"
)

# ── Create worktree from public branch ───────────────────────────────

cleanup 2>/dev/null || true

if git rev-parse "$PUBLIC_REMOTE/$PUBLIC_BRANCH" &>/dev/null; then
    git branch -f "$STAGING" "$PUBLIC_REMOTE/$PUBLIC_BRANCH"
else
    # First-ever sync: create an empty root commit
    local EMPTY_TREE
    EMPTY_TREE="$(git hash-object -t tree /dev/null)"
    local ROOT
    ROOT="$(git commit-tree "$EMPTY_TREE" -m 'Initial empty commit')"
    git branch -f "$STAGING" "$ROOT"
fi

git worktree add "$WORK" "$STAGING"

# ── Sync files into worktree ─────────────────────────────────────────

echo "Syncing files..."

# Clear everything in worktree except .git
find "$WORK" -mindepth 1 -maxdepth 1 -not -name ".git" -exec rm -rf {} + 2>/dev/null || true

COUNT=0
for path in "${INCLUDE[@]}"; do
    # Use git ls-tree to find committed files (works for dirs and files)
    while IFS= read -r f; do
        case "$f" in
            */node_modules/*|*/dist/*|*/.cache/*|*/main.js) continue ;;
            # Fabric data agent — in development, excluded from public sync.
            notebooks/examples/fabric_deploy.ipynb) continue ;;
            docs/site/fabric/*) continue ;;
        esac
        mkdir -p "$WORK/$(dirname "$f")"
        git show "HEAD:$f" > "$WORK/$f" 2>/dev/null && COUNT=$((COUNT + 1)) || true
    done < <(git ls-tree -r --name-only HEAD "$path" 2>/dev/null)
done

echo "Synced $COUNT files."

# ── Check for changes ─────────────────────────────────────────────────

cd "$WORK"
git add -A

DIFF="$(git diff --cached --stat)"
if [ -z "$DIFF" ]; then
    echo "Public repo is already up to date."
    exit 0
fi

echo ""
echo "Changes:"
echo "$DIFF"
echo ""

# ── Commit and push ──────────────────────────────────────────────────

PRIVATE_SHORT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
PRIVATE_MSG="$(git -C "$REPO_ROOT" log -1 --format='%s')"

git commit -q -m "Sync from private repo ($PRIVATE_SHORT)

Source: $PRIVATE_MSG
Synced by scripts/sync-public.sh"

if [ "$PUSH" = "--push" ]; then
    git push "$PUBLIC_REMOTE" "$STAGING:$PUBLIC_BRANCH"
    echo "Pushed to $PUBLIC_REMOTE/$PUBLIC_BRANCH"
else
    echo "Dry run. Push with: ./scripts/sync-public.sh --push"
fi
