#!/usr/bin/env bash
# Stage E: RoboTwin -> GWP05Native adapter -> GWP PyTorch server -> RoboTwin.
#
# Two processes are involved:
#   1. the GWP 0.5 PyTorch server (policy env, ZMQ port 11444) - started by
#      scripts/infer/gwp_server.sh and expected to be up already;
#   2. the XPolicyLab ws policy server hosting the GWP05Native adapter
#      (simulation env, ws port 19100) - started here.
#
# Usage:
#   bash scripts/evaluate/closed_loop.sh 1 0        # 1 episode, seed 0
#   bash scripts/evaluate/closed_loop.sh 3 0        # 3 episodes starting at seed 0
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

COUNT=${1:-1}
SEED=${2:-0}

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RT=${ROBOTWIN_SOURCE_DIR:?set ROBOTWIN_SOURCE_DIR}
XPL=$RT/XPolicyLab
ENVS=${ENVS_DIR:?set ENVS_DIR}
SIMPY=$ENVS/robotwin_gwp_sim/bin/python
POLICYPY=$ENVS/robotwin_gwp_policy/bin/python
export PATH=$ENVS/robotwin_gwp_tools/bin:$PATH
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONUNBUFFERED=1

ADAPTER_PORT=19100
GWP_PORT=11444
EVAL_GPU=${EVAL_GPU:-0}
export CUDA_VISIBLE_DEVICES=$EVAL_GPU
# TAG_SUFFIX keeps two runs of the same (count, seed) from overwriting
# each other, e.g. the published min-max convention vs the z-score one.
TAG="n${COUNT}_seed${SEED}${TAG_SUFFIX:-}"
LOG=${LOGS_DIR:?set LOGS_DIR}/E_closed_loop_${TAG}.log
ADAPTER_LOG=${LOGS_DIR:?set LOGS_DIR}/E_adapter_${TAG}.log
OUT_DIR=${RUNS_DIR:?set RUNS_DIR}/E_closed_loop/${TAG}
mkdir -p "$OUT_DIR"
# full commanded-action trace (48x14 per call) for the trajectory deliverable
export GWP05_TRACE=${GWP05_TRACE:-$OUT_DIR/action_trace.jsonl}

echo "=== check GWP server on ${GWP_PORT} ==="
GIGA_MODELS_DIR=${GIGA_MODELS_DIR:-$GWP_SOURCE_DIR/third_party/giga-models}
export GIGA_MODELS_DIR
"$POLICYPY" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["GIGA_MODELS_DIR"])
from giga_models.sockets import RobotInferenceClient
c = RobotInferenceClient(host="127.0.0.1", port=11444, timeout_ms=5000)
alive = c.ping()
print("gwp_server_alive =", alive)
raise SystemExit(0 if alive else 1)
PYEOF

cleanup() {
  if [ -n "${ADAPTER_PID:-}" ]; then
    echo "[cleanup] stopping adapter server pid=$ADAPTER_PID"
    kill "$ADAPTER_PID" 2>/dev/null || true
    wait "$ADAPTER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== start XPolicyLab adapter server on ${ADAPTER_PORT} ==="
cd "$XPL"
setsid nohup "$SIMPY" setup_policy_server.py \
  --config_path policy/GWP05Native/deploy.yml \
  --protocol ws --host 127.0.0.1 --port "$ADAPTER_PORT" \
  > "$ADAPTER_LOG" 2>&1 &
ADAPTER_PID=$!
echo "adapter pid=$ADAPTER_PID"

for i in $(seq 1 120); do
  if (echo > /dev/tcp/127.0.0.1/$ADAPTER_PORT) >/dev/null 2>&1; then
    echo "adapter port open after ${i}s"; break
  fi
  if ! kill -0 "$ADAPTER_PID" 2>/dev/null; then
    echo "adapter server died; log:"; tail -50 "$ADAPTER_LOG"; exit 1
  fi
  sleep 1
done

echo "=== run eval: ${COUNT} episode(s) starting at seed ${SEED} ==="
# RoboTwin resolves ./assets/... relative to the CWD at import time, so the eval
# must run from the repo root (same as scripts/eval_policy.sh does).
cd "$RT"
"$SIMPY" "$RT/scripts/eval_policy_xpolicylab.py" \
  --bench_name RoboTwin \
  --task_name beat_block_hammer \
  --env_cfg_type aloha_agilex \
  --task_config demo_clean \
  --policy_name GWP05Native \
  --host 127.0.0.1 --port "$ADAPTER_PORT" --protocol ws \
  --test_num "$COUNT" --seed "$SEED" \
  2>&1 | tee "$LOG" || true

echo "=== collect artefacts ==="
LATEST=$(ls -dt "$RT"/eval_result/beat_block_hammer/GWP05Native/demo_clean/*/*/ 2>/dev/null | head -1)
echo "eval_result dir: $LATEST"
if [ -n "$LATEST" ]; then
  cp -r "$LATEST"/. "$OUT_DIR"/ 2>/dev/null || true
fi
cp "$ADAPTER_LOG" "$OUT_DIR/adapter.log" 2>/dev/null || true
find "$OUT_DIR" -maxdepth 2 -type f | sort
