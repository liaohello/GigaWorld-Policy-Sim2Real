#!/usr/bin/env bash
# Dedicated HDF5 -> LeRobot v3 data-processing environment.
#
# Exists because lerobot 0.4.4 pins diffusers<0.36.0 while GigaWorld-Policy
# pins diffusers==0.36.0 (pip: ResolutionImpossible). Bending either side would
# silently change the policy runtime or the dataset writer, so the converter
# gets its own env. torch is the CPU build -- the converter only needs it as a
# lerobot import dependency, and no GPU work happens here.
set -euo pipefail

CONDA=${CONDA_ROOT:?set CONDA_ROOT}/bin/conda
ENVS=${ENVS_DIR:?set ENVS_DIR}
DATA_ENV=$ENVS/robotwin_gwp_data
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
TOOLS_ENV=$ENVS/robotwin_gwp_tools
export PATH=$TOOLS_ENV/bin:$PATH

echo "=== [1/4] conda env ($DATA_ENV) ==="
if [ ! -x "$DATA_ENV/bin/python" ]; then
  CONDA_PKGS_DIRS=${CONDA_ROOT:?set CONDA_ROOT}/pkgs_robotwin_gwp "$CONDA" create -p "$DATA_ENV" -y python=3.11
fi
PY=$DATA_ENV/bin/python
"$PY" -V

echo "=== [2/4] torch (CPU) for lerobot import ==="
"$PY" -m pip install -U pip setuptools wheel 2>&1 | tail -1
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3 || \
  "$PY" -m pip install torch torchvision 2>&1 | tail -3

echo "=== [3/4] lerobot 0.4.4 + conversion deps ==="
"$PY" -m pip install "lerobot==0.4.4" h5py opencv-python-headless pyyaml tqdm 2>&1 | tail -8

echo "=== [4/4] verify ==="
"$PY" - <<'PYEOF'
import importlib
for m in ("torch", "lerobot", "diffusers", "h5py", "cv2", "pandas", "pyarrow", "av"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:10s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:10s} FAIL {type(e).__name__}: {e}")
PYEOF
"$PY" -c "import lerobot, importlib.metadata as md; print('lerobot', md.version('lerobot')); print('diffusers', md.version('diffusers'))"
"$PY" -m pip freeze > "${LOCKS_DIR:?set LOCKS_DIR}/data_env_pip_freeze.txt"
echo "DONE data env"
