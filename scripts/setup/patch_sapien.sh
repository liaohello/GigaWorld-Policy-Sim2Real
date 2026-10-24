#!/usr/bin/env bash
# Stage A: apply the one site-packages patch RoboTwin's installer makes that this
# project actually needs, and record the before/after.
#
# Applied:  sapien/wrapper/urdf_loader.py -- open the URDF/SRDF files as UTF-8.
#           Upstream scripts/_install.sh does the same sed.
#
# NOT applied: mplib/planner.py "drop the collide term from the screw-plan abort
#           condition". That patch weakens a collision check, and the code path is
#           unreachable here: plan_screw only runs when an embodiment config sets
#           `planner: mplib_screw`, and assets/embodiments/aloha-agilex/config.yml
#           sets `planner: curobo`. Recorded in patch/NOTES.md.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENVS=${ENVS_DIR:?set ENVS_DIR}
PY=$ENVS/robotwin_gwp_sim/bin/python
PATCH_DIR=$REPO_ROOT/patch
# The pristine copy of an upstream file is not redistributed here; keep it with
# the other locks. patch/upstream-manifest.json records its SHA-256.
PRISTINE_DIR=${LOCKS_DIR:-$REPO_ROOT/locks}/upstream
mkdir -p "$PRISTINE_DIR"
PRISTINE=$PRISTINE_DIR/sapien_urdf_loader.py

SAPIEN_LOCATION=$("$PY" -m pip show sapien | awk '/^Location:/{print $2}')/sapien
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
echo "sapien urdf_loader: $URDF_LOADER"
[ -f "$URDF_LOADER" ] || { echo "missing $URDF_LOADER" >&2; exit 1; }

if [ ! -f "$PRISTINE" ]; then
  cp "$URDF_LOADER" "$PRISTINE"
  echo "kept pristine copy outside Git: $PRISTINE"
fi

echo "--- occurrences before ---"
grep -n 'open(urdf_file\|open(srdf_file' "$URDF_LOADER" || true

sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"

echo "--- occurrences after ---"
grep -n 'open(urdf_file\|open(srdf_file' "$URDF_LOADER" || true

# Keep the recorded diff path-free: `patch -p1` from the site-packages root
# applies it, and no absolute path ends up in the repository.
diff -u "$PRISTINE" "$URDF_LOADER" \
  | sed -E 's#^--- .*#--- a/sapien/wrapper/urdf_loader.py#; s#^\+\+\+ .*#+++ b/sapien/wrapper/urdf_loader.py#' \
  > "$PATCH_DIR/sapien_urdf_loader.diff" || true
echo "--- diff ---"
cat "$PATCH_DIR/sapien_urdf_loader.diff"

echo "--- sanity import ---"
"$PY" -c "import sapien, sapien.core, inspect; print('sapien', sapien.__version__ if hasattr(sapien,'__version__') else 'ok')"
