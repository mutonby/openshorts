#!/usr/bin/env bash
# Installs the pre-commit hook that keeps CLAUDE.md in sync with the codebase.
#
# Run once after cloning:    bash scripts/install_hooks.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v pre-commit > /dev/null 2>&1; then
    echo "⚠️  pre-commit not installed."
    echo "    Install it first:  pip install pre-commit"
    echo "    Then re-run:       bash scripts/install_hooks.sh"
    exit 1
fi

pre-commit install
echo "✅ pre-commit hooks installed."
echo "   CLAUDE.md will be regenerated on every commit."
echo "   To run manually:  python scripts/update_claude_md.py"
