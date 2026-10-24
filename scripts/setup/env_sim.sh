#!/usr/bin/env bash
# Build the isolated RoboTwin simulation environment.
#
#   python 3.10 / torch 2.4.1+cu121 / numpy 1.26.4 / sapien 3.0.0b1 / mplib 0.2.1
#   + curobo v0.7.8 (planner aloha-agilex/config.yml selects) + warp-lang 1.12.0
#
# Deliberate deviations from scripts/_install.sh, recorded in patch/:
#   * pytorch3d is NOT installed: nothing on the executed code path (collect_data,
#     eval_policy_xpolicylab, envs/*) imports it -- verified with grep; only
#     scripts/_install.sh lists it. Revisit if an import error says otherwise.
#   * The upstream submodule refresh (scripts/update_xpolicylab.sh) is skipped: the
#     archived tree already ships a complete XPolicyLab and has no .git metadata.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

CONDA=${CONDA_ROOT:?set CONDA_ROOT}/bin/conda
ENVS=${ENVS_DIR:?set ENVS_DIR}
SIM_ENV=$ENVS/robotwin_gwp_sim
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RT=${ROBOTWIN_SOURCE_DIR:?set ROBOTWIN_SOURCE_DIR}
TOOLS_ENV=$ENVS/robotwin_gwp_tools
export PATH=$TOOLS_ENV/bin:$PATH

echo "=== [1/7] conda env ($SIM_ENV) ==="
if [ ! -x "$SIM_ENV/bin/python" ]; then
  "$CONDA" create -p "$SIM_ENV" -y python=3.10
fi
PY=$SIM_ENV/bin/python
"$PY" -V

echo "=== [2/7] pip bootstrap ==="
"$PY" -m pip install -U pip setuptools wheel 2>&1 | tail -2

echo "=== [3/7] torch 2.4.1 (PyPI linux wheel is cu121) ==="
"$PY" -m pip install torch==2.4.1 torchvision==0.19.1 2>&1 | tail -5
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

echo "=== [4/7] RoboTwin requirements ==="
cd "$RT"
"$PY" -m pip install -r scripts/requirements.txt 2>&1 | tail -15

echo "=== [5/7] toppra (imported by envs/robot/{robot,planner}.py, absent from requirements) ==="
"$PY" -m pip install toppra 2>&1 | tail -3
"$PY" -c "import toppra, importlib.metadata as m; print('toppra', m.version('toppra'))"

echo "=== [6/7] XPolicyLab editable install ==="
"$PY" -m pip install -e "$RT/XPolicyLab" 2>&1 | tail -5

echo "=== [7/7] curobo v0.7.8 + warp ==="
mkdir -p "$RT/envs"
if [ ! -d "$RT/envs/curobo" ]; then
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$RT/envs/curobo"
fi
"$PY" -m pip install warp-lang==1.12.0 2>&1 | tail -2
"$PY" -m pip install setuptools==69.5.1 2>&1 | tail -2
cd "$RT/envs/curobo"
"$PY" -m pip install -e . --no-build-isolation 2>&1 | tail -20
"$PY" -c "import warp;print('warp',warp.__version__)"

echo "=== freeze ==="
"$PY" -m pip freeze > "${LOCKS_DIR:?set LOCKS_DIR}/sim_env_pip_freeze.txt"
"$PY" -c "import torch,sys;print(sys.version);print('torch.version.cuda',torch.version.cuda)" >> "${LOCKS_DIR:?set LOCKS_DIR}/sim_env_pip_freeze.txt"
"$CONDA" list -p "$SIM_ENV" --export > "${LOCKS_DIR:?set LOCKS_DIR}/sim_env_conda_explicit.txt" || true
echo "DONE sim env"
