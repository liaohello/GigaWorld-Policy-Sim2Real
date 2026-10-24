#!/usr/bin/env bash
# Build the isolated GigaWorld-Policy 0.5 environment.
#
#   python 3.11 / torch 2.7.1+cu126 / diffusers 0.36.0 / transformers 4.54.1
#   lerobot 0.4.4 (also drives the HDF5 -> LeRobot v3 conversion)
#
# No NumPy 1.x pin here: GWP pins numpy 2.4.6, and the sim env keeps its own
# numpy 1.26.4. Mixing the two would break the RoboTwin native extensions.
set -euo pipefail

CONDA=${CONDA_ROOT:?set CONDA_ROOT}/bin/conda
ENVS=${ENVS_DIR:?set ENVS_DIR}
POLICY_ENV=$ENVS/robotwin_gwp_policy
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GWP=${GWP_SOURCE_DIR:?set GWP_SOURCE_DIR}
TOOLS_ENV=$ENVS/robotwin_gwp_tools
export PATH=$TOOLS_ENV/bin:$PATH

echo "=== [1/6] conda env ($POLICY_ENV) ==="
if [ ! -x "$POLICY_ENV/bin/python" ]; then
  "$CONDA" create -p "$POLICY_ENV" -y python=3.11
fi
PY=$POLICY_ENV/bin/python
"$PY" -V

echo "=== [2/6] pip bootstrap ==="
"$PY" -m pip install -U pip setuptools wheel 2>&1 | tail -2

echo "=== [3/6] torch 2.7.1 (cu126) ==="
# download.pytorch.org is not usable from this host (403 / stalls), so the
# wheels come from the Tsinghua PyPI mirror. torch 2.7.1 on PyPI IS the cu126
# build for linux x86_64 -- asserted below instead of assumed.
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 2>&1 | tail -8
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available());
assert torch.version.cuda=='12.6', torch.version.cuda"

echo "=== [4/6] GWP requirements (lerobot excluded -- see configs/gwp_requirements_no_lerobot.txt) ==="
cd "$GWP"
"$PY" -m pip install -r "$REPO_ROOT/configs/gwp_requirements_no_lerobot.txt" 2>&1 | tail -20

echo "=== [5/6] third_party editable packages ==="
for pkg in giga-train giga-models giga-datasets; do
  echo "--- $pkg ---"
  if [ -f "$GWP/third_party/$pkg/setup.py" ]; then
    "$PY" -m pip install --no-deps -e "$GWP/third_party/$pkg" 2>&1 | tail -3 || \
      "$PY" -m pip install -e "$GWP/third_party/$pkg" 2>&1 | tail -3
  fi
done

echo "=== [6/6] verification ==="
"$PY" -m pip check || true
"$PY" - <<'PYEOF'
import importlib
mods = ["torch","diffusers","transformers","lerobot","zmq","av","pandas","pyarrow","h5py","torchvision","einops","safetensors"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:12s} FAIL {type(e).__name__}: {e}")
PYEOF

echo "=== freeze ==="
"$PY" -m pip freeze > "${LOCKS_DIR:?set LOCKS_DIR}/policy_env_pip_freeze.txt"
"$CONDA" list -p "$POLICY_ENV" --export > "${LOCKS_DIR:?set LOCKS_DIR}/policy_env_conda_explicit.txt" || true
echo "DONE policy env"
