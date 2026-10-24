#!/usr/bin/env bash
# Create the shared binary-tools env (ffmpeg with libx264/libsvtav1) used by
# RoboTwin video writing, eval video capture and LeRobot encoding.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

CONDA=${CONDA_ROOT:?set CONDA_ROOT}/bin/conda
ENVS=${ENVS_DIR:?set ENVS_DIR}
TOOLS_ENV=$ENVS/robotwin_gwp_tools

if [ ! -x "$TOOLS_ENV/bin/ffmpeg" ]; then
  "$CONDA" create -p "$TOOLS_ENV" -y -c conda-forge ffmpeg >/dev/null
fi

"$TOOLS_ENV/bin/ffmpeg" -hide_banner -version | head -2
echo "--- encoders ---"
"$TOOLS_ENV/bin/ffmpeg" -hide_banner -encoders 2>/dev/null | grep -E "libx264|libsvtav1|h264_nvenc" || true
