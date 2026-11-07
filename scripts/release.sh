#!/usr/bin/env bash
# LongLoom release CLI — one command from source to PyPI. Idempotent:
# re-running an already-published version is a safe no-op.
#
#   scripts/release.sh 1.2.0            # test, bump, changelog, commit, tag v1.2.0
#   scripts/release.sh 1.2.0 --push     # ...and push the tag (triggers the publish workflow)
#   scripts/release.sh 1.2.0 --publish  # ...and upload locally to pypi.org (needs PYPI_TOKEN)
#   scripts/release.sh 1.2.0 --testpypi # ...and upload locally to test.pypi.org (TESTPYPI_TOKEN)
#   scripts/release.sh 1.2.0 --dry-run  # show every step without changing anything
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
TARGET="pypi"; [[ "$MODE" == "--testpypi" ]] && TARGET="testpypi"
DRY=""; [[ "$MODE" == "--dry-run" ]] && DRY="1"
run() { if [[ -n "$DRY" ]]; then say "  [dry-run] $*" "$YELLOW"; else "$@"; fi }

# ── idempotency checks ────────────────────────────────────────────────
on_pypi() {  # $1=version, $2=target registry
  local host="pypi.org"; [[ "$2" == "testpypi" ]] && host="test.pypi.org"
  curl -fsSL --max-time 15 "https://$host/pypi/longloom/$1/json" >/dev/null 2>&1
}

if on_pypi "$VERSION" "$TARGET"; then
  say "✔ $VERSION is already published on $TARGET — nothing to do" "$GREEN"
  exit 0
fi

TAG_LOCAL=0;    git rev-parse --verify "refs/tags/$TAG" >/dev/null 2>&1 && TAG_LOCAL=1
TAG_REMOTE=$(git ls-remote --tags origin "refs/tags/$TAG" 2>/dev/null | wc -l)

# ── pre-flight (only when there is something to release) ─────────────
[[ -n "$DRY" ]] || { [[ -n "$(git status --porcelain)" ]] && { say "✗ working tree not clean — commit or stash first" "$RED"; exit 1; }; }

say "▸ running offline test suite" "$GREEN"
run python -m pytest tests/ -q

say "▸ bumping longloom/__init__.py → $VERSION" "$GREEN"
if [[ -n "$DRY" ]]; then
  say "  [dry-run] sed longloom/__init__.py → __version__ = \"$VERSION\"" "$YELLOW"
else
  sed -i "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" longloom/__init__.py
  grep -q "^__version__ = \"$VERSION\"" longloom/__init__.py || { say "✗ version bump failed" "$RED"; exit 1; }
fi

say "▸ moving CHANGELOG 'Unreleased' → [$VERSION]" "$GREEN"
if [[ -z "$DRY" ]]; then
  TODAY=$(date +%Y-%m-%d)
  python - "$VERSION" "$TODAY" <<'PY'
import sys
version, today = sys.argv[1], sys.argv[2]
s = open("CHANGELOG.md").read()
s = s.replace("## [Unreleased]", f"## [{version}] — {today}", 1)
open("CHANGELOG.md", "w").write(s)
PY
fi

# ── tag: create only when missing, never fail on re-run ──────────────
if [[ $TAG_LOCAL == 1 ]]; then
  say "▸ tag $TAG already exists locally — skipping creation" "$YELLOW"
else
  say "▸ committing + tagging $TAG" "$GREEN"
  run git add longloom/__init__.py CHANGELOG.md
  run git commit -m "Release $VERSION"
  run git tag "$TAG"
fi

# ── publish paths ─────────────────────────────────────────────────────
if [[ "$MODE" == "--testpypi" || "$MODE" == "--publish" ]]; then
  TOKEN_VAR="PYPI_TOKEN"; [[ "$MODE" == "--testpypi" ]] && TOKEN_VAR="TESTPYPI_TOKEN"
  [[ -n "${!TOKEN_VAR:-}" ]] || { say "✗ set $TOKEN_VAR to upload locally (or use --push for CI trusted publishing)" "$RED"; exit 1; }
  say "▸ building + uploading to $TARGET" "$GREEN"
  run python -m build
  run python -m twine check dist/*
  # Set TWINE_PASSWORD instead of passing -p on the command line:
  # the token must never appear in `ps` output or shell history.
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="${!TOKEN_VAR}"
  run python -m twine upload --repository "$TARGET" --non-interactive dist/*
elif [[ "$MODE" == "--push" ]]; then
  say "▸ pushing main + $TAG (triggers .github/workflows/release.yml → PyPI)" "$GREEN"
  run git push origin main
  if [[ $TAG_REMOTE -gt 0 ]]; then
    say "▸ tag $TAG already on origin — skipping tag push" "$YELLOW"
  else
    run git push origin "$TAG"
  fi
else
  PUSH_HINT="git push origin main"
  [[ $TAG_LOCAL == 1 && $TAG_REMOTE == 0 ]] && PUSH_HINT="$PUSH_HINT $TAG"
  say "▸ next: $PUSH_HINT   (or re-run with --push / --publish)" "$YELLOW"
fi

say "✔ release $VERSION complete" "$GREEN"
