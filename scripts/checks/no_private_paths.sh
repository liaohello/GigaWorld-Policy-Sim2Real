#!/usr/bin/env bash
# Fail if tracked content contains private paths, host addresses, or SSH details.
#
# This repository is public. AGENTS.md forbids committing personal filesystem
# paths, public IPs, SSH commands, credentials, and machine identifiers. Run
# this before every commit; it is also cheap enough to run in CI.
#
# Usage: bash scripts/checks/no_private_paths.sh [rev]
#   rev defaults to the index (staged) view: `git ls-files`
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Files that legitimately contain a person's name (license attribution).
ALLOWLIST='^(LICENSE|\.gitignore|scripts/checks/no_private_paths\.sh)$'

# Loopback and wildcard addresses are not private; a public dotted quad is.
ALLOW_LINE='127\.0\.0\.1|0\.0\.0\.0|255\.255\.|^[0-9]+:[A-Za-z0-9_.+-]+=='

# Each pattern is an ERE. Keep them narrow so ordinary prose does not trip them.
PATTERNS=(
  '/home/[a-z]'                       # absolute personal home paths
  '/Users/[A-Za-z]'                   # macOS home paths
  'dataAssets'                        # machine-specific asset root used before sanitizing
  '[0-9]{1,3}(\.[0-9]{1,3}){3}'       # any dotted-quad address, incl. 127.0.0.1
  'ssh[[:space:]]+-i'                 # ssh invocations with a key
  '\.ssh/'                            # key paths
  'ssh[[:space:]].*-p[[:space:]]*[0-9]{4,5}'  # ssh with a non-default port
  'BEGIN [A-Z ]*PRIVATE KEY'
)

fail=0
while IFS= read -r f; do
  [[ "$f" =~ $ALLOWLIST ]] && continue
  [ -f "$f" ] || continue
  # Skip binaries; the tracked tree should be text only.
  if ! grep -Iq . "$f" 2>/dev/null; then
    echo "skip (binary): $f" >&2
    continue
  fi
  for p in "${PATTERNS[@]}"; do
    if hits=$(grep -nEI -- "$p" "$f" 2>/dev/null); then
      while IFS= read -r line; do
        # A dotted-quad hit on loopback/wildcard is fine; anything else is not.
        printf '%s' "$line" | grep -qE "$ALLOW_LINE" && continue
        printf 'VIOLATION %s:%s  [%s]\n' "$f" "${line%%:*}" "$p" >&2
        fail=1
      done <<<"$hits"
    fi
  done
done < <(git ls-files)

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "Refusing to ship private data. Replace the value with an environment" >&2
  echo "variable documented in .env.example, or drop the line." >&2
  exit 1
fi

echo "ok: no private paths, addresses, or SSH material in tracked files"
