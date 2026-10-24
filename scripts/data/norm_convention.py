#!/usr/bin/env python
"""Rewrite a norm-stats file so min-max becomes z-score, without touching weights.

Why: ``scripts/inference_openloop.py`` normalises with ``q01``/``q99`` min-max
while wam.cpp's GWP05 converter requires *unclipped z_score* for RoboTwin
(``scripts/common/gwp05_gguf.py``). Those are different affine maps, so the two
runtimes cannot be compared on the same observation.

Setting ``q01' = mean - std`` and ``q99' = mean + std`` makes them coincide::

    2*(x - (mean-std)) / (2*std) - 1  ==  (x - mean)/std
    ((a+1)/2)*(2*std) + (mean-std)    ==  a*std + mean

The transformation lives in :mod:`gwp_sim2real.normalization` so the tests
exercise the same code this script runs. ``mean`` and ``std`` are the *measured*
statistics read from the file; deriving them from q01/q99 would make the whole
thing an identity, which is a mistake this project already made once.

Usage:
  python3 scripts/data/norm_convention.py \
      --src "$ARTIFACTS_DIR/norm_stats_delta.json" \
      --out "$ARTIFACTS_DIR/norm_stats_delta_zscore_equiv.json"
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

import env as project_env  # noqa: E402
from gwp_sim2real.normalization import convert_block  # noqa: E402

project_env.load_env()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="norm stats produced by compute_norm.sh")
    ap.add_argument("--out", required=True, help="destination stats file")
    args = ap.parse_args()

    with open(args.src) as f:
        data = json.load(f)

    report = {"source": args.src, "domains": {}}
    for domain in ("observation.state", "action"):
        block = data["norm_stats"][domain]
        mean, std = block["mean"], block["std"]
        data["norm_stats"][domain] = convert_block(block)
        zero = [i for i, s in enumerate(std) if s == 0.0]
        report["domains"][domain] = {
            "dim": len(mean),
            "std_min": min(std),
            "std_zero_dims": zero,
        }
        if zero:
            print(
                f"WARNING: {domain} has zero-std dims {zero}; the equivalence is "
                "degenerate there until they are floored (see infer/norm_for_gguf.py)"
            )

    data["derivation"] = (
        "q01/q99 replaced by mean-std / mean+std so the published min-max pipeline "
        "computes exactly z-score; see scripts/data/norm_convention.py"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
