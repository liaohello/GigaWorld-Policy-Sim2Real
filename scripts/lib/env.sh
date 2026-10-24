#!/usr/bin/env bash
# Load $REPO_ROOT/.env into the shell, then export it.
#
# Sourced by every shell entrypoint:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"
#
# Why not just rely on the caller: the scripts are run one at a time, often from
# a different directory, and a missing variable must fail at the first line that
# needs it rather than halfway through a conversion. `set -a` marks everything
# defined by the file for export, so child processes (python, wam-predict) see
# the same values without an explicit `export` per variable.
#
# Values already present in the environment win: `.env` is a default, not an
# override, so `RUNS_DIR=/tmp/x bash scripts/...` keeps working.

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export REPO_ROOT

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi
