#!/usr/bin/env bash
# Stage F1/F2: merge the diffusers checkpoint into the layout wam.cpp's converter
# expects, derive the PolicySpec, then convert to GGUF.
#
# Why the merge: convert_gwp05._load_transformer() reads
#   <checkpoint>/config.json + <checkpoint>/diffusion_pytorch_model.bin
# with torch.load(mmap=True). The public release is sharded safetensors
# (diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors + index), so the
# converter stops with "missing GWP weights: .../diffusion_pytorch_model.bin".
# The merge is a pure container change (shards -> one F32 pickle); no arithmetic.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
WAM=${WAM_CPP_SOURCE_DIR:?set WAM_CPP_SOURCE_DIR}
ENVS=${ENVS_DIR:?set ENVS_DIR}
PY=$ENVS/robotwin_gwp_policy/bin/python
CKPT=${GWP_CHECKPOINT_DIR:?set GWP_CHECKPOINT_DIR}
BASE=${WAN_BASE_MODEL_DIR:?set WAN_BASE_MODEL_DIR}
MERGED=${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/gwp05_ckpt_merged
PROFILE=${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/wam_profiles/gwp05_robotwin_14d_zscore_public05.json
OUT=${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/gwp05_robotwin_14d_mot_vae_bf16.gguf
GGUF_PY=$WAM/build-cuda/_deps/llama-src/gguf-py

mkdir -p "${ARTIFACTS_DIR:?set ARTIFACTS_DIR}" "${LOGS_DIR:?set LOGS_DIR}"

echo "=== [1/4] merge sharded safetensors -> diffusion_pytorch_model.bin ==="
if [ ! -f "$MERGED/diffusion_pytorch_model.bin" ]; then
  mkdir -p "$MERGED"
  cp -n "$CKPT/config.json" "$MERGED/"
  "$PY" - "$CKPT" "$MERGED" <<'PYEOF'
import json, os, sys, torch
from safetensors.torch import load_file
src, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json")))
weight_map = idx["weight_map"]
files = sorted(set(weight_map.values()))
print("shards:", files)
state = {}
for f in files:
    part = load_file(os.path.join(src, f), device="cpu")
    print(f"  {f}: {len(part)} tensors")
    state.update(part)
missing = [k for k in weight_map if k not in state]
print("tensors:", len(state), "missing:", len(missing))
assert not missing, missing[:5]
total = sum(t.numel() * t.element_size() for t in state.values())
print(f"total bytes: {total/1e9:.2f} GB")
torch.save(state, os.path.join(dst, "diffusion_pytorch_model.bin"))
print("wrote", os.path.join(dst, "diffusion_pytorch_model.bin"))
PYEOF
fi
ls -la "$MERGED"

echo "=== [2/4] derive PolicySpec profile ==="
"$PY" "$REPO_ROOT/scripts/infer/make_wam_profile.py" \
  --src "$WAM/profiles/gwp05_robotwin_14d_zscore.json" \
  --out "$PROFILE"

echo "=== [2b/4] runtime-acceptable norm stats ==="
# wam-inspect rejects zero-std dims ("normalization std is too small", index 14);
# the padded dims 14/15 are zero-std by construction. See scripts/86_*.py.
"$PY" "$REPO_ROOT/scripts/infer/norm_for_gguf.py" \
  --src "${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/norm_stats_delta.json" \
  --out "${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/norm_stats_delta_for_gguf.json" --model-dim 16

echo "=== [3/4] dry run ==="
cd "$WAM"
export PYTHONPATH="$GGUF_PY${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/convert/convert_gwp05.py \
  --checkpoint "$MERGED" \
  --base-model "$BASE" \
  --norm-stats "${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/norm_stats_delta_for_gguf.json" \
  --policy-spec "$PROFILE" \
  --weight-policy mot-vae-bf16 \
  --out "$OUT" \
  --dry-run 2>&1 | tee "${LOGS_DIR:?set LOGS_DIR}/F_convert_dryrun.log"

echo "=== [4/4] convert ==="
if [ -f "$OUT" ] && [ "${RECONVERT:-0}" = "1" ]; then
  echo "RECONVERT=1: removing our own derived artifact $OUT"
  rm -f "$OUT" "$OUT.manifest.json"
fi
if [ -f "$OUT" ]; then
  echo "existing $OUT present; converter refuses to overwrite -- leaving it in place"
else
  "$PY" scripts/convert/convert_gwp05.py \
    --checkpoint "$MERGED" \
    --base-model "$BASE" \
    --norm-stats "${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/norm_stats_delta_for_gguf.json" \
    --policy-spec "$PROFILE" \
    --weight-policy mot-vae-bf16 \
    --out "$OUT" 2>&1 | tee "${LOGS_DIR:?set LOGS_DIR}/F_convert.log"
fi
ls -la "$OUT" "$OUT.manifest.json" 2>/dev/null || true
