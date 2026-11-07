#!/usr/bin/env bash
# LongLoom release CLI — one command from source to PyPI.
#
#   scripts/release.sh 1.2.0            # test, bump, changelog, commit, tag v1.2.0
#   scripts/release.sh 1.2.0 --push     # ...and push the tag (triggers the publish workflow)
#   scripts/release.sh 1.2.0 --testpypi # ...and upload locally to test.pypi.org (needs a token)
#   scripts/release.sh 1.2.0 --publish  # ...and upload locally to pypi.org (needs a token)
#   scripts/release.sh 1.2.0 --dry-run  # show every step without changing anything
#
# Publish credentials: set PYPI_TOKEN (or TESTPYPI_TOKEN for --testpypi),
# or configure them in ~/.pypirc. Never commit tokens.
set -euo pipefail

VERSION="${1:-}"
MODE="${2:-none}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
say() { printf "%b\n" "$2${1}${NC}"; }

[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$ ]] || {
  say "usage: scripts/release.sh <version> [--push|--publish|--testpypi|--dry-run]  (e.g. 1.2.0)" "$RED"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="v$VERSION"
DRY=""; [[ "$MODE" == "--dry-run" ]] && DRY="1"
run() { if [[ -n "$DRY" ]]; then say "  [dry-run] $*" "$YELLOW"; else "$@"; fi }

# ── pre-flight ────────────────────────────────────────────────────────
[[ -z "$DRY" ]] && { [[ -n "$(git status --porcelain)" ]] && { say "✗ working tree not clean — commit or stash first" "$RED"; exit 1; }; }
git rev-parse --verify "refs/tags/$TAG" >/dev/null 2>&1 && { say "✗ tag $TAG already exists" "$RED"; exit 1; }

say "▸ running offline test suite" "$GREEN"
run python -m pytest tests/ -q

say "▸ bumping longloom/__init__.py → $VERSION" "$GREEN"
[[ -n "$DRY" ]] || sed -i "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" longloom/__init__.py
grep -q "^__version__ = \"$VERSION\"" longloom/__init__.py || { say "✗ version bump failed" "$RED"; exit 1; }

say "▸ moving CHANGELOG 'Unreleased' → [$VERSION]" "$GREEN"
if [[ -z "$DRY" ]]; then
  TODAY=$(date +%Y-%m-%d)
  python - "$VERSION" "$TODAY" <<'PY'
import re, sys
version, today = sys.argv[1], sys.argv[2]
s = open("CHANGELOG.md").read()
s = s.replace("## [Unreleased]", f"## [{version}] — {today}", 1)
open("CHANGELOG.md", "w").write(s)
PY
fi

say "▸ committing + tagging $TAG" "$GREEN"
run git add longloom/__init__.py CHANGELOG.md
run git commit -m "Release $VERSION"
run git tag "$TAG"

# ── publish paths ─────────────────────────────────────────────────────
if [[ "$MODE" == "--testpypi" || "$MODE" == "--publish" ]]; then
  REPO="pypi"; [[ "$MODE" == "--testpypi" ]] && REPO="testpypi"
  TOKEN_VAR="PYPI_TOKEN"; [[ "$MODE" == "--testpypi" ]] && TOKEN_VAR="TESTPYPI_TOKEN"
  [[ -n "${!TOKEN_VAR:-}" ]] || { say "✗ set $TOKEN_VAR to upload locally" "$RED"; exit 1; }
  say "▸ building + uploading to $REPO" "$GREEN"
  run python -m build
  run python -m twine check dist/*
  run python -m twine upload --repository "$REPO" --non-interactive \
        -u __token__ -p "${!TOKEN_VAR}" dist/*
elif [[ "$MODE" == "--push" ]]; then
  say "▸ pushing main + $TAG (triggers .github/workflows/release.yml → PyPI)" "$GREEN"
  run git push origin main "$TAG"
else
  say "▸ next: git push origin main $TAG   (or re-run with --push / --publish)" "$YELLOW"
fi

say "✔ release $VERSION complete" "$GREEN"
