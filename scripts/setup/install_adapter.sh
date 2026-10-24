#!/usr/bin/env bash
# Install the GWP05Native adapter into the RoboTwin checkout.
#
# XPolicyLab discovers policies as local packages under
# `XPolicyLab/policy/<name>/` with a `deploy.yml` beside `model.py`, so the
# adapter cannot simply be imported from this repository. This script copies it
# in and renders the three environment-dependent values.
#
# Idempotent: re-running overwrites only these three files.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/env.sh"

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RT=${ROBOTWIN_SOURCE_DIR:?set ROBOTWIN_SOURCE_DIR}
GWP=${GWP_SOURCE_DIR:?set GWP_SOURCE_DIR}

# Where the fixed text embeddings live (produced by scripts/data/compute_t5.py).
T5_DIR=${T5_DIR:-${DATASETS_DIR:?set DATASETS_DIR}/robotwin_gwp/beat_block_hammer_50_v3/t5_embedding}
GIGA_MODELS_DIR=${GIGA_MODELS_DIR:-$GWP/third_party/giga-models}

DEST=$RT/XPolicyLab/policy/GWP05Native
mkdir -p "$DEST"

cp "$REPO_ROOT/src/gwp_sim2real/adapters/robotwin_gwp05.py" "$DEST/model.py"
sed -e "s#__T5_DIR__#${T5_DIR}#g" \
    -e "s#__GIGA_MODELS_DIR__#${GIGA_MODELS_DIR}#g" \
    "$REPO_ROOT/src/gwp_sim2real/adapters/deploy.example.yml" > "$DEST/deploy.yml"
cat > "$DEST/__init__.py" <<'PY'
try:
    from .deploy import *
except ImportError:
    pass
try:
    from .model import *
except ImportError:
    pass
PY

echo "[install_adapter] $DEST"
echo "  model.py     <- src/gwp_sim2real/adapters/robotwin_gwp05.py"
echo "  deploy.yml   <- rendered (t5_dir=$T5_DIR)"
echo "  giga-models  <- $GIGA_MODELS_DIR"
