#!/usr/bin/env python
"""Numeric comparison between the PyTorch reference and the C++ run.

Both sides are driven with the identical NPZ (state, three RGB frames, the bf16
language embedding, and the exact initial action noise captured from the PyTorch
pipeline), so any difference is implementation, not input.

Honesty note that belongs in the numbers: the PyTorch reference runs bf16 on
CUDA while the C++ run can only execute the f32 artifact on CPU on 24 GiB cards
(the bf16 artifact does not fit, and the runtime refuses bf16 off CUDA). Part of
the delta is therefore bf16-vs-f32, not implementation error. That is why a
tolerance has to be chosen by the caller (`--max-abs-tol`) instead of being
baked in here: without C++ output, no one has earned the right to pick one.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

import env as project_env  # noqa: E402

project_env.load_env()

RUNS_DIR = pathlib.Path(os.environ.get("RUNS_DIR") or REPO_ROOT / "runs")
DEFAULT_DIR = RUNS_DIR / "F_compare"


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(np.float64) - b.astype(np.float64)
    absd = np.abs(d)
    return {
        "shape": list(a.shape),
        "max_abs": float(absd.max()),
        "mean_abs": float(absd.mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "p50_abs": float(np.percentile(absd, 50)),
        "p95_abs": float(np.percentile(absd, 95)),
        "per_dim_max_abs": absd.max(axis=0).round(6).tolist(),
        "per_dim_mean_abs": absd.mean(axis=0).round(6).tolist(),
        "corr_per_dim": [
            float(np.corrcoef(a[:, i], b[:, i])[0, 1])
            if np.std(a[:, i]) > 0 and np.std(b[:, i]) > 0
            else None
            for i in range(a.shape[1])
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--torch", default=str(DEFAULT_DIR / "torch_action.npy"))
    ap.add_argument("--cpp", default=str(DEFAULT_DIR / "cpp_action.npy"))
    ap.add_argument("--out", default=str(DEFAULT_DIR / "compare.json"))
    ap.add_argument(
        "--max-abs-tol",
        type=float,
        default=None,
        help="if given, exit 1 when max |torch - cpp| exceeds this value in rad",
    )
    args = ap.parse_args()

    for path in (args.torch, args.cpp):
        if not os.path.isfile(path):
            print(f"missing input: {path}")
            return 2

    torch_action = np.load(args.torch)
    cpp_action = np.load(args.cpp)
    if torch_action.shape != cpp_action.shape:
        print(f"shape mismatch: {torch_action.shape} vs {cpp_action.shape}")
        return 2

    report = {
        "torch": {
            "path": args.torch,
            "shape": list(torch_action.shape),
            "min": float(torch_action.min()),
            "max": float(torch_action.max()),
        },
        "cpp": {
            "path": args.cpp,
            "shape": list(cpp_action.shape),
            "min": float(cpp_action.min()),
            "max": float(cpp_action.max()),
        },
        "delta": stats(torch_action, cpp_action),
        "max_abs_tolerance": args.max_abs_tol,
        "verdict": "reported",
    }
    if args.max_abs_tol is not None:
        report["verdict"] = "PASS" if report["delta"]["max_abs"] <= args.max_abs_tol else "FAIL"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    d = report["delta"]
    print(f"shape      {d['shape']}")
    print(f"max_abs    {d['max_abs']:.6f} rad")
    print(f"mean_abs   {d['mean_abs']:.6f}")
    print(f"p95_abs    {d['p95_abs']:.6f}")
    print(f"verdict    {report['verdict']}")
    print(f"wrote      {args.out}")
    return 0 if report["verdict"] in ("PASS", "reported") else 1


if __name__ == "__main__":
    raise SystemExit(main())
