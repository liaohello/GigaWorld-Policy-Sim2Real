#!/usr/bin/env bash
# Stage F3/F4: inspect, validate, run the converted GGUF, then compare.
#
# The three prediction attempts are separate on purpose, and their exit codes are
# not swallowed: on a shared 24 GiB card the expected result is "no output
# produced", and this script must say so with a non-zero status instead of
# finishing quietly.
#
#   1. wam-inspect / wam-validate   metadata and contract, CPU only
#   2. wam-predict --backend cuda   bf16 artifact; expected to exhaust the card
#   3. wam-predict --backend cpu    f32 artifact on the GGML CPU backend
#
# Exit codes:
#   0  at least one C++ action file was produced (comparison runs if a reference
#      exists; its own verdict is printed)
#   2  inspect/validate rejected the artifact
#   3  no C++ action file (BLOCKED - see docs/specifications/v0.3-wam-cpp-parity.md)
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

WAM=${WAM_CPP_SOURCE_DIR:?set WAM_CPP_SOURCE_DIR}
ENVS=${ENVS_DIR:?set ENVS_DIR}
PY=$ENVS/robotwin_gwp_policy/bin/python
GGUF=${GGUF:-${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/gwp05_robotwin_14d_mot_vae_bf16.gguf}
F32_GGUF=${F32_GGUF:-${ARTIFACTS_DIR:?}/gwp05_robotwin_14d_source_f32.gguf}
LIB=$WAM/build-cuda/libwam_c_api.so
# wam-predict is a Python console script; the C++ build only produces
# wam-inspect / wam-validate and the C ABI library.
PREDICT=$ENVS/robotwin_gwp_policy/bin/wam-predict
CUDNN_LIB=${CUDNN_ROOT:?set CUDNN_ROOT}/lib
INPUT=${INPUT:-${RUNS_DIR:?set RUNS_DIR}/F_compare/cpp_input.npz}
OUTDIR=${OUTDIR:-${RUNS_DIR:?set RUNS_DIR}/F_compare}
GPU=${GPU:-0}
MAX_ABS_TOL=${MAX_ABS_TOL:-}

export LD_LIBRARY_PATH=$CUDNN_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
mkdir -p "$OUTDIR" "${LOGS_DIR:?set LOGS_DIR}"

# Only count outputs from this invocation. A stale action file left by an
# earlier run must not turn a failed prediction into a false success.
rm -f "$OUTDIR/cpp_action_cuda.npy" "$OUTDIR/cpp_action.npy"

echo "=== artifact ==="
ls -la "$GGUF" "$GGUF.manifest.json" 2>/dev/null

echo "=== inspect ==="
"$WAM/build-cuda/wam-inspect" "$GGUF" > "$OUTDIR/inspect.txt" 2>&1
inspect_rc=$?
echo "inspect_rc=$inspect_rc"
head -20 "$OUTDIR/inspect.txt"

echo "=== validate ==="
"$WAM/build-cuda/wam-validate" "$GGUF" > "$OUTDIR/validate.txt" 2>&1
validate_rc=$?
echo "validate_rc=$validate_rc"
head -10 "$OUTDIR/validate.txt"

if [ "$inspect_rc" -ne 0 ] || [ "$validate_rc" -ne 0 ]; then
  echo "artifact rejected by the upstream self-check; stopping before prediction"
  exit 2
fi

echo "=== wam-predict (cuda, bf16 artifact) ==="
CUDA_VISIBLE_DEVICES=$GPU timeout 900 "$PREDICT" \
  "$GGUF" "$INPUT" \
  --library "$LIB" \
  --backend cuda --precision bf16 --device 0 --seed 0 \
  --output "$OUTDIR/cpp_action_cuda.npy" > "$OUTDIR/predict_cuda.log" 2>&1
cuda_rc=$?
echo "cuda_rc=$cuda_rc"
tail -5 "$OUTDIR/predict_cuda.log"

echo "=== wam-predict (cpu, f32 artifact) ==="
if [ -f "$F32_GGUF" ]; then
  timeout 5400 "$PREDICT" \
    "$F32_GGUF" "$INPUT" \
    --library "$LIB" \
    --backend cpu --precision f32 --seed 0 \
    --output "$OUTDIR/cpp_action.npy" > "$OUTDIR/predict_cpu.log" 2>&1
  cpu_rc=$?
  echo "cpu_rc=$cpu_rc"
  tail -5 "$OUTDIR/predict_cpu.log"
else
  cpu_rc=127
  echo "skip: no f32 artifact at $F32_GGUF (needed because bf16 cannot run off CUDA)"
fi

shopt -s nullglob
produced=("$OUTDIR"/cpp_action*.npy)
shopt -u nullglob

echo "=== summary ==="
echo "cuda_rc=$cuda_rc cpu_rc=$cpu_rc produced=${#produced[@]}"

if [ "${#produced[@]}" -eq 0 ]; then
  echo "no C++ action output was produced."
  echo "on a 24 GiB card shared with other tenants this is the expected outcome;"
  echo "see docs/specifications/v0.3-wam-cpp-parity.md for the precondition."
  exit 3
fi

if [ -f "$OUTDIR/torch_action.npy" ]; then
  compare_args=(--cpp "${produced[0]}" --out "$OUTDIR/compare.json")
  if [ -n "$MAX_ABS_TOL" ]; then
    compare_args+=(--max-abs-tol "$MAX_ABS_TOL")
  fi
  "$PY" "$REPO_ROOT/scripts/infer/wam_compare.py" "${compare_args[@]}"
  exit $?
fi

echo "no torch_action.npy in $OUTDIR; run scripts/infer/gwp_torch_reference.py first"
exit 0
