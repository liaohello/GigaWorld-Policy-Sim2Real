#!/usr/bin/env bash
# Stage B2/B3: convert native RoboTwin HDF5 -> LeRobot v3.0 packed dataset.
#
# IMPORTANT: create_empty_dataset() calls shutil.rmtree(dataset_root) when the
# target exists, so this wrapper refuses to run onto an existing repo_id unless
# --force is passed explicitly. Output always lives under this project's
# datasets/ tree (HF_LEROBOT_HOME), never under the shared RoboTwin data dir.
#
# Usage:
#   bash scripts/data/convert_lerobot.sh smoke            # 1 episode
#   bash scripts/data/convert_lerobot.sh full             # all 50 episodes
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

MODE=${1:-smoke}
FORCE=${2:-}

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RT=${ROBOTWIN_SOURCE_DIR:?set ROBOTWIN_SOURCE_DIR}
ENVS=${ENVS_DIR:?set ENVS_DIR}
# The converter needs lerobot 0.4.4, which pins diffusers<0.36 and therefore
# cannot live in the policy env (GWP pins diffusers==0.36.0). Dedicated data env.
PY=$ENVS/robotwin_gwp_data/bin/python
export PATH=$ENVS/robotwin_gwp_tools/bin:$PATH
export HF_LEROBOT_HOME=${DATASETS_DIR:?set DATASETS_DIR}

case "$MODE" in
  smoke) REPO_ID=robotwin_gwp/beat_block_hammer_smoke_v3; MAX_EP=1 ;;
  full)  REPO_ID=robotwin_gwp/beat_block_hammer_50_v3;    MAX_EP=50 ;;
  *) echo "usage: $0 {smoke|full} [--force]" >&2; exit 2 ;;
esac

TARGET=$HF_LEROBOT_HOME/$REPO_ID
if [ -d "$TARGET" ]; then
  if [ "$FORCE" != "--force" ]; then
    echo "REFUSING: $TARGET already exists (the converter would delete it). Pass --force to recreate." >&2
    exit 3
  fi
  echo "WARNING: removing existing $TARGET because --force was passed"
  rm -rf "$TARGET"
fi

mkdir -p "$HF_LEROBOT_HOME"
cd "$RT"
"$PY" XPolicyLab/scripts/transform_lerobot_v30_format.py \
  'demo_clean.beat_block_hammer.aloha_agilex' \
  --repo_id "$REPO_ID" \
  --max_episode "$MAX_EP" \
  2>&1 | tee "${LOGS_DIR:?set LOGS_DIR}/B_convert_${MODE}.log"

echo "--- result tree ---"
find "$TARGET" -maxdepth 3 -type d | sort | head -40
du -sh "$TARGET"
