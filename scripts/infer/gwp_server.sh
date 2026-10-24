#!/usr/bin/env bash
# Stage D: start the GWP 0.5 PyTorch inference server (ZMQ) in the policy env.
#
# Explicit component placement: UMT5 stays on CPU (11.36 GB BF16 that a
# fixed-embedding run never executes), only the MoT transformer + VAE go to the
# GPU. torch.compile is off for the first functional pass.
#
# Usage:
#   GPU=2 bash scripts/infer/gwp_server.sh [extra args ...]
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GWP=${GWP_SOURCE_DIR:?set GWP_SOURCE_DIR}
ENVS=${ENVS_DIR:?set ENVS_DIR}
PY=$ENVS/robotwin_gwp_policy/bin/python
DATASET=${DATASETS_DIR:?set DATASETS_DIR}/robotwin_gwp/beat_block_hammer_50_v3
GPU=${GPU:-2}

export CHECKPOINT=${GWP_CHECKPOINT_DIR:?set GWP_CHECKPOINT_DIR}
export BASE_MODEL=${WAN_BASE_MODEL_DIR:?set WAN_BASE_MODEL_DIR}
# Which statistics file is loaded decides the normalization convention:
#   norm_stats_delta.json               q01/q99 min-max (script default, matches
#                                       the official open-loop client)
#   norm_stats_delta_zscore_equiv.json  z-score-equivalent bounds (matches wam.cpp)
# Both were run; the closed-loop report says which produced which numbers.
export NORM_STATS=${NORM_STATS:-${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/norm_stats_delta.json}
export DATA_PATH=$DATASET
export DATA_IDX=${DATA_IDX:-0}
export FIXED_T5_PATH=$DATASET/t5_embedding/episode_000000.pt
export HOST=127.0.0.1
export PORT=${PORT:-11444}
export DEVICE=cuda:0

mkdir -p "${RUNS_DIR:?set RUNS_DIR}/D_openloop" "${LOGS_DIR:?set LOGS_DIR}"
cd "$GWP"

export CUDA_VISIBLE_DEVICES=$GPU
# The official wrapper calls bare `python`, so the policy env must be first on
# PATH (the ffmpeg tools env also ships a python -- it must not win).
export PATH=$ENVS/robotwin_gwp_policy/bin:$ENVS/robotwin_gwp_tools/bin:$PATH

echo "[gwp-server] CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT checkpoint=$CHECKPOINT"
# Official entrypoint (scripts/run_inference_openloop.sh), plus this project's
# placement/artefact flags, which the wrapper forwards to the python module.
exec bash scripts/run_inference_openloop.sh server \
  --model-dtype bf16 \
  --no-compile-transformer \
  --text-encoder-device cpu \
  --output-dir "${RUNS_DIR:?set RUNS_DIR}/D_openloop" \
  --timing-jsonl "${RUNS_DIR:?set RUNS_DIR}/D_openloop/server_timing.jsonl" \
  "$@"
