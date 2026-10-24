#!/usr/bin/env bash
# Stage A4: one expert episode through the native collector.
# Output goes to <workspace>/generated/demo_clean_smoke/beat_block_hammer/aloha_agilex
# (never into the shipped data/ directory, which stays read-only for this stage).
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RT=${ROBOTWIN_SOURCE_DIR:?set ROBOTWIN_SOURCE_DIR}
ENVS=${ENVS_DIR:?set ENVS_DIR}
PY=$ENVS/robotwin_gwp_sim/bin/python
export PATH=$ENVS/robotwin_gwp_tools/bin:$PATH
export PYTHONWARNINGS=ignore::UserWarning

GENERATED_DIR=${GENERATED_DIR:?set GENERATED_DIR}
LOGS_DIR=${LOGS_DIR:?set LOGS_DIR}
mkdir -p "$GENERATED_DIR" "$LOGS_DIR"

# The shipped task config carries __GENERATED_DIR__ instead of a path so that a
# checkout cannot write into someone else's dataset directory by accident.
TASK_CFG=$RT/env_cfg/task_config/demo_clean_smoke.yml
sed "s#__GENERATED_DIR__#${GENERATED_DIR}#" \
  "$REPO_ROOT/configs/environments/robotwin_task_demo_clean_smoke.yml" > "$TASK_CFG"
echo "[collect] installed $TASK_CFG (save_path=$GENERATED_DIR)"

# RoboTwin resolves ./assets/... relative to the CWD at import time.
cd "$RT"
"$PY" scripts/collect_data.py beat_block_hammer demo_clean_smoke 2>&1 | tee "$LOGS_DIR/A_collect_smoke.log"
