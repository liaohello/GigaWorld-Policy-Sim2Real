#!/usr/bin/env python
"""Make the norm stats acceptable to the wam.cpp GWP05 runtime.

`wam-inspect` / `wam-validate` reject the artifact with
    {"code":"incompatible_artifact","message":"normalization std is too small",
     "details":[{"field":"wam.norm.state.std","reason":"index 14"}]}

because ``compute_wam_task_norm.py`` pads the statistics to ``--model-dim 32``
and dims 14..31 are all zeros (mean=std=q01=q99=0), i.e. std < epsilon.

The padded dims carry no information: the PyTorch pipeline normalises only the
14 real dims and then pads the *normalised* vector with literal zeros. Setting
``mean=0, std=1`` (and bounds +-1) for every zero-std dim makes the C++ z-score
of that dim exactly ``(0-0)/1 = 0`` -- the same value PyTorch's padding
produces. Real dims are untouched.

Usage:
  python scripts/infer/norm_for_gguf.py --src artifacts/norm_stats_delta.json \
      --out artifacts/norm_stats_delta_for_gguf.json --model-dim 16
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gwp_sim2real.normalization import floor_zero_std  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dim", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    with open(args.src) as f:
        data = json.load(f)
    ns = data["norm_stats"]

    changed = {}
    for domain in ("observation.state", "action"):
        ns[domain], changed[domain] = floor_zero_std(
            ns[domain], model_dim=args.model_dim, eps=args.eps
        )

    data["derivation"] = (
        "zero-std / padded dims set to mean=0,std=1,bounds=+-1 so the C++ z-score "
        "of a padded dim is 0, matching PyTorch's zero padding; see "
        "scripts/infer/norm_for_gguf.py"
    )
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps({"out": args.out, "model_dim": args.model_dim,
                      "zero_std_dims_within_model_dim": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
