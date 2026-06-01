#!/bin/bash
# Install git hooks to prevent accidental trace/DB commits

set -e

HOOKS_DIR=".git/hooks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing git hooks..."

# Pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'HOOK'
#!/bin/bash
# Pre-commit hook: prevent accidental trace file commits

set -e

STAGED=$(git diff --cached --name-only)

TRACE_PATTERNS=(
    "var/inbox/traces/*.json"
    "var/inbox/traces/processed/*.json"
    "var/inbox/traces/failed/*.json"
    "var/omega_traces.db*"
    "omega_traces.db*"
)

BLOCKED=0
for pattern in "${TRACE_PATTERNS[@]}"; do
    for file in $STAGED; do
        if [[ "$file" == $pattern ]]; then
            echo "[ERROR] Cannot commit trace/database files: $file"
            echo "        Traces are runtime artifacts, not source code."
            BLOCKED=1
        fi
    done
done

if [ $BLOCKED -eq 1 ]; then
    echo ""
    echo "Unstage with: git reset HEAD var/inbox/traces/ var/omega_traces.db*"
    exit 1
fi

exit 0
HOOK

chmod +x "$HOOKS_DIR/pre-commit"
echo "✓ Pre-commit hook installed"

# Post-merge hook (optional: auto-audit)
cat > "$HOOKS_DIR/post-merge" << 'HOOK'
#!/bin/bash
# Post-merge hook: warn if traces were merged (they shouldn't be)

if git diff --cached --name-only | grep -E "var/inbox/traces|omega_traces.db"; then
    echo "[WARNING] Merge included trace files (runtime artifacts)"
    echo "          These should not be in version control"
fi

exit 0
HOOK

chmod +x "$HOOKS_DIR/post-merge"
echo "✓ Post-merge hook installed (optional)"

echo ""
echo "Git hooks installed successfully."
echo "Run this script again if hooks are ever reset."
