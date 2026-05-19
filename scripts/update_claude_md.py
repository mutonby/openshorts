#!/usr/bin/env python3
"""
Regenerate the auto-managed sections of CLAUDE.md.

What this does
==============

CLAUDE.md is split into hand-written prose and three auto-managed tables. The
auto-managed sections live between marker comments and get rewritten by this
script on every commit (via the pre-commit hook in .pre-commit-config.yaml):

    <!-- AUTO:REPO-MAP:START --> ... <!-- AUTO:REPO-MAP:END -->
    <!-- AUTO:MODULE-MAP:START --> ... <!-- AUTO:MODULE-MAP:END -->
    <!-- AUTO:ENV:START --> ... <!-- AUTO:ENV:END -->

The script:
1. Walks the repo and lists top-level folders (REPO-MAP).
2. Parses every backend/app/*.py module via ast, extracting the one-line
   docstring + the names of public functions/classes (MODULE-MAP).
3. Reads .env.example and renders the env-vars table (ENV).
4. Locates the markers in CLAUDE.md and rewrites only the content between them.

It is idempotent: running it twice with no source changes is a no-op.

Convention enforcement
======================

This script exits non-zero (with a list of offenders) if any module under
backend/app/ is missing a module docstring. The pre-commit hook will fail the
commit until the developer adds one. This is how the "every module has a
one-liner" convention from CLAUDE.md becomes mechanically enforced — without
this, CLAUDE.md is just an advisory and drifts.

Usage
=====

    python scripts/update_claude_md.py        # rewrite CLAUDE.md in place
    python scripts/update_claude_md.py --check # exit non-zero if a rewrite is needed (CI mode)
"""

from __future__ import annotations

import ast
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "backend" / "app"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


# ---------------------------------------------------------------------------
# Marker handling
# ---------------------------------------------------------------------------

MARKERS = {
    "REPO-MAP": ("<!-- AUTO:REPO-MAP:START -->", "<!-- AUTO:REPO-MAP:END -->"),
    "MODULE-MAP": ("<!-- AUTO:MODULE-MAP:START -->", "<!-- AUTO:MODULE-MAP:END -->"),
    "ENV": ("<!-- AUTO:ENV:START -->", "<!-- AUTO:ENV:END -->"),
}


def replace_between(text: str, start: str, end: str, body: str) -> str:
    """Replace whatever is between ``start`` and ``end`` markers with ``body``."""
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end),
        re.DOTALL,
    )
    replacement = f"{start}\n{body}\n{end}"
    if not pattern.search(text):
        # Marker block doesn't exist yet — append it at the end.
        return text.rstrip() + "\n\n" + replacement + "\n"
    return pattern.sub(replacement, text)


# ---------------------------------------------------------------------------
# REPO-MAP
# ---------------------------------------------------------------------------

TOP_LEVEL_DESCRIPTIONS = {
    "backend": "Python FastAPI service — the API, video pipeline, and tests.",
    "frontend": "React + Vite dashboard — the UI users interact with.",
    "renderer": "Remotion render microservice (TypeScript) + compositions.",
    "assets": "Committed static assets (fonts, screenshots).",
    "scripts": "Developer tooling (update_claude_md.py, install_hooks.sh).",
    "uploads": "Runtime: incoming video uploads (gitignored).",
    "output": "Runtime: generated clips and thumbnails (gitignored).",
}


def build_repo_map() -> str:
    lines = ["| Folder | What it is |", "| --- | --- |"]
    for entry in sorted(os.listdir(REPO_ROOT)):
        full = REPO_ROOT / entry
        if not full.is_dir():
            continue
        if entry.startswith("."):
            continue
        if entry in {"__pycache__", ".venv", ".git", "node_modules"}:
            continue
        if entry.endswith(".egg-info") or entry.endswith(".dist-info"):
            continue
        desc = TOP_LEVEL_DESCRIPTIONS.get(entry, "_(undocumented — add to TOP_LEVEL_DESCRIPTIONS in scripts/update_claude_md.py)_")
        lines.append(f"| `{entry}/` | {desc} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MODULE-MAP
# ---------------------------------------------------------------------------


def parse_module(path: Path) -> Tuple[str, List[str]]:
    """Return ``(one_line_docstring, public_symbols)`` for a Python module.

    Raises ``ValueError`` if the module has no docstring (so the pre-commit
    hook fails the commit and the developer is forced to add one).
    """
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"{path}: syntax error — {e}")

    raw_doc = ast.get_docstring(tree)
    if not raw_doc:
        raise ValueError(f"{path}: missing module docstring")

    one_line = raw_doc.strip().splitlines()[0].strip()

    public: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public.append(node.name)

    return one_line, public


def build_module_map() -> Tuple[str, List[str]]:
    """Walk the package and return ``(markdown_table, [errors])``."""
    rows: List[str] = ["| Module | Purpose | Public surface |", "| --- | --- | --- |"]
    errors: List[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        try:
            doc, symbols = parse_module(path)
        except ValueError as e:
            errors.append(str(e))
            continue

        if path.name == "__init__.py" and not symbols:
            # Skip empty __init__.py rows to keep the table scannable; the doc
            # is captured implicitly via the folder description in REPO-MAP.
            continue

        symbol_text = ", ".join(f"`{s}`" for s in symbols) if symbols else "_(none)_"
        rows.append(f"| `{rel.as_posix()}` | {doc} | {symbol_text} |")

    return "\n".join(rows), errors


# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------

ENV_LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=([^\n]*)$")


def build_env_table() -> str:
    if not ENV_EXAMPLE.exists():
        return "_(.env.example not found)_"

    rows: List[str] = ["| Variable | Default | Notes |", "| --- | --- | --- |"]
    current_section = ""

    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        # Section headers are written as `# --- Name -------------`
        m = re.match(r"^#\s*-+\s*(.+?)\s*-+\s*$", line)
        if m:
            current_section = m.group(1)
            continue
        # Comment lines starting with `# VAR=` are documented optional vars.
        m = re.match(r"^#\s*([A-Z][A-Z0-9_]*)=(.*)$", line)
        if m:
            name, default = m.group(1), m.group(2)
            rows.append(f"| `{name}` | _(unset)_ | {current_section} (commented — optional) |")
            continue
        m = ENV_LINE_RE.match(line)
        if m:
            name, default = m.group(1), m.group(2)
            shown_default = default if default else "_(empty — must set)_"
            rows.append(f"| `{name}` | `{shown_default}` | {current_section} |")

    return "\n".join(rows) if len(rows) > 2 else "_(no env vars in .env.example)_"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate auto-managed CLAUDE.md sections.")
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if CLAUDE.md would change (CI mode).")
    args = parser.parse_args()

    if not CLAUDE_MD.exists():
        print(f"❌ {CLAUDE_MD} not found.", file=sys.stderr)
        return 1

    repo_map = build_repo_map()
    module_map, errors = build_module_map()
    env_table = build_env_table()

    if errors:
        print("❌ Module docstrings missing — every .py file under backend/app/ "
              "must start with a one-line module docstring:", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        return 2

    current = CLAUDE_MD.read_text(encoding="utf-8")
    updated = current
    updated = replace_between(updated, *MARKERS["REPO-MAP"], repo_map)
    updated = replace_between(updated, *MARKERS["MODULE-MAP"], module_map)
    updated = replace_between(updated, *MARKERS["ENV"], env_table)

    if updated == current:
        print("✓ CLAUDE.md already up to date.")
        return 0

    if args.check:
        print("❌ CLAUDE.md is out of date — run `python scripts/update_claude_md.py` "
              "to regenerate the auto-managed sections.", file=sys.stderr)
        return 1

    CLAUDE_MD.write_text(updated, encoding="utf-8")
    print(f"✓ Rewrote {CLAUDE_MD.relative_to(REPO_ROOT)} "
          f"(REPO-MAP, MODULE-MAP, ENV sections).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
