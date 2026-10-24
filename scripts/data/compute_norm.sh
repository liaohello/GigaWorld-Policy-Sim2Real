#!/usr/bin/env bash
# Compute action/state normalization statistics from the converted dataset.
#
# Delegates to the upstream script (it knows the delta mask), but pins the two
# arguments that matter:
#   --model-dim 32      what the official script expects for a 32-channel layout;
#                       dims 14/15 come out with zero spread and are later
#                       rewritten for the GGUF path (see infer/norm_for_gguf.py).
#   --action-horizon 48 the policy's horizon.
#
# Output: $ARTIFACTS_DIR/norm_stats_delta.json
set -euo pipefail

ENVS=${ENVS_DIR:?set ENVS_DIR}
PY=$ENVS/robotwin_gwp_policy/bin/python
GWP=${GWP_SOURCE_DIR:?set GWP_SOURCE_DIR}
DATA=${DATASETS_DIR:?set DATASETS_DIR}/robotwin_gwp/beat_block_hammer_50_v3
OUT=${ARTIFACTS_DIR:?set ARTIFACTS_DIR}/norm_stats_delta.json

mkdir -p "$(dirname "$OUT")"
cd "$GWP"
"$PY" scripts/compute_wam_task_norm.py \
  --data-root "$DATA" \
  --output "$OUT" \
  --model-dim "${MODEL_DIM:-32}" \
  --action-horizon 48 2>&1 | tail -20

"$PY" - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["norm_stats"]
for key in ("observation.state", "action"):
    q01, q99 = d[key]["q01"], d[key]["q99"]
    live = [i for i, (a, b) in enumerate(zip(q01, q99)) if abs(b - a) > 1e-6]
    print(f"{key:20s} dims={len(q01):3d} non-degenerate={len(live)}")
PY
echo "[compute_norm] $OUT"
