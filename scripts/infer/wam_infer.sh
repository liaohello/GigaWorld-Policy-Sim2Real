#!/usr/bin/env bash
# Stage F3/F4: inspect, validate and actually run the converted GGUF.
#
# Order matters and is recorded honestly:
#   1. wam-inspect / wam-validate   (CPU metadata + contract)
#   2. wam-predict --backend cuda   (expected to run out of memory on a 24 GiB
#      card: the BF16 artifact is ~23.7 GB and ~18 GiB of each card is already
#      held by other tenants; the failure is captured, not hidden)
#   3. wam-predict --backend cpu    (real inference on the GGML CPU backend --
#      slower, but the same artifact, same inputs, same seed). CPU cannot use
#      the BF16 latency profile ("the BF16 latency profile requires CUDA"), so it
#      runs as f32 math over the BF16-stored weights.
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
WAM=${WAM_CPP_SOURCE_DIR:?set WAM_CPP_SOURCE_DIR}
ENVS=${ENVS_DIR:?set ENVS_DIR}
PY=$ENVS/robotwin_gwp_policy/bin/python
GGUF=${GGUF:-${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/gwp05_robotwin_14d_mot_vae_bf16.gguf}
LIB=$WAM/build-cuda/libwam_c_api.so
# wam-predict is the Python console script (pyproject [project.scripts]); the
# C++ build only produces wam-inspect / wam-validate and the C ABI library.
PREDICT=$ENVS/robotwin_gwp_policy/bin/wam-predict
CUDNN_LIB=${CUDNN_ROOT:?set CUDNN_ROOT}/lib
INPUT=${INPUT:-${RUNS_DIR:?set RUNS_DIR}/F_compare/cpp_input.npz}
OUTDIR=${OUTDIR:-${RUNS_DIR:?set RUNS_DIR}/F_compare}
GPU=${GPU:-0}

export LD_LIBRARY_PATH=$CUDNN_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
mkdir -p "$OUTDIR" "${LOGS_DIR:?set LOGS_DIR}"

echo "=== artifact ==="
ls -la "$GGUF" "$GGUF.manifest.json" 2>/dev/null

echo "=== inspect ==="
"$WAM/build-cuda/wam-inspect" "$GGUF" 2>&1 | tee "$OUTDIR/inspect.txt" | head -60
echo "inspect_rc=${PIPESTATUS[0]}"

echo "=== validate ==="
"$WAM/build-cuda/wam-validate" "$GGUF" 2>&1 | tee "$OUTDIR/validate.txt" | head -40
echo "validate_rc=${PIPESTATUS[0]}"

echo "=== wam-predict (cuda) ==="
set +e
CUDA_VISIBLE_DEVICES=$GPU timeout 900 "$PREDICT" \
  "$GGUF" "$INPUT" \
  --library "$LIB" \
  --backend cuda --precision bf16 --device 0 --seed 0 \
  --output "$OUTDIR/cpp_action_cuda.npy" > "$OUTDIR/predict_cuda.log" 2>&1
echo "cuda_rc=$?"
tail -15 "$OUTDIR/predict_cuda.log"
set -e

echo "=== wam-predict (cpu) ==="
set +e
timeout 5400 "$PREDICT" \
  "$GGUF" "$INPUT" \
  --library "$LIB" \
  --backend cpu --precision f32 --seed 0 \
  --output "$OUTDIR/cpp_action.npy" > "$OUTDIR/predict_cpu.log" 2>&1
echo "cpu_rc=$?"
tail -20 "$OUTDIR/predict_cpu.log"
ls -la "$OUTDIR"/cpp_action*.npy 2>/dev/null
