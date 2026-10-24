#!/usr/bin/env python
"""Build a z-score-equivalent norm file for the published PyTorch pipeline.

Why: ``scripts/inference_openloop.py`` normalises with ``q01``/``q99`` min-max
(``2*(x-q01)/(q99-q01)-1``) while wam.cpp's GWP05 converter/PolicySpec requires
*unclipped z_score* for RoboTwin ("RoboTwin PolicySpec {domain} normalization
must be unclipped z_score", scripts/common/gwp05_gguf.py). Those are different
affine maps, so the two runtimes cannot be compared on the same observation.

Setting ``q01' = mean - std`` and ``q99' = mean + std`` makes the min-max map
algebraically identical to z-score:

    2*(x - (mean-std)) / (2*std) - 1  ==  (x - mean)/std
    ((a+1)/2)*(2*std) + (mean-std)    ==  a*std + mean

The model weights are untouched; only the affine normalization the *code*
applies changes, and it now matches what the C++ runtime does.

Usage:
  python scripts/data/norm_convention.py \
      --src artifacts/norm_stats_delta.json \
      --out artifacts/norm_stats_delta_zscore_equiv.json
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.src) as f:
        data = json.load(f)
    ns = data["norm_stats"]

    report = {"source": args.src, "domains": {}}
    for domain in ("observation.state", "action"):
        block = ns[domain]
        mean = block["mean"]
        std = block["std"]
        block["q01"] = [m - s for m, s in zip(mean, std)]
        block["q99"] = [m + s for m, s in zip(mean, std)]
        report["domains"][domain] = {
            "dim": len(mean),
            "std_min": min(std),
            "std_zero_dims": [i for i, s in enumerate(std) if s == 0.0],
        }
        if report["domains"][domain]["std_zero_dims"]:
            # a zero std makes the equivalence degenerate; report instead of hiding
            print(f"WARNING: {domain} has zero-std dims: {report['domains'][domain]['std_zero_dims']}")

    data["derivation"] = (
        "q01/q99 replaced by mean-std / mean+std so that the published min-max "
        "pipeline computes exactly z-score; see scripts/data/norm_convention.py"
    )
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
