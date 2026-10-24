#!/usr/bin/env python
"""Stage D: fixed-sample open-loop benchmark of the GWP 0.5 server.

Sends the *same* observation (episode 0, frame 0 of the derived LeRobot v3
dataset) 10 times as warm-up and then 30 times as measured calls, through the
official ``RobotInferenceClient``. Records, per call, the client-side
round-trip and the server-side breakdown written by the patched
``inference()`` into ``--timing-jsonl``.

Reports p50/p95 for:
  * model_s        - the transformer/VAE forward only (server-side)
  * preprocess_s   - image/state prep before the forward (server-side)
  * comm_s         - client round-trip minus server total (serialisation + ZMQ)
  * e2e_s          - client-observed round-trip
plus GPU allocated/reserved at the end and the server's reported peak.

Usage (server must already be running):
  python scripts/infer/openloop_bench.py --data-root datasets/robotwin_gwp/... \
      --t5 datasets/.../t5_embedding/episode_000000.pt \
      --host 127.0.0.1 --port 11444 --warmup 10 --iters 30 \
      --out runs/D_openloop
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "lib"))

import env as project_env  # noqa: E402

project_env.load_env()
# The GWP checkout supplies both the wire format and the dataset loader; it is
# an input to this script, not part of the repository.
GWP = project_env.require("GWP_SOURCE_DIR")["GWP_SOURCE_DIR"]


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return values[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--t5", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11444)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="runs/D_openloop")
    ap.add_argument("--server-timing", default=None, help="path of the server --timing-jsonl")
    args = ap.parse_args()

    sys.path.insert(0, GWP)
    sys.path.insert(0, os.path.join(GWP, "scripts"))
    os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")
    os.environ["WAM_GIGA_ROOT"] = os.path.join(GWP, "third_party")

    import torch
    from einops import rearrange

    from giga_models.sockets import RobotInferenceClient
    from inference_openloop import load_lerobot_v3_episode

    out_dir = os.path.join(WS, args.out)
    os.makedirs(out_dir, exist_ok=True)

    root = args.data_root if os.path.isabs(args.data_root) else os.path.join(WS, args.data_root)
    front, left, right, state, action, t5_ep, task = load_lerobot_v3_episode(
        root, args.episode
    )
    f = args.frame
    print(f"episode={args.episode} frame={f} task={task!r} state={state.shape} action={action.shape}")

    img_front = np.array(front[f]) / 255.0
    img_left = np.array(left[f]) / 255.0
    img_right = np.array(right[f]) / 255.0
    obs = {
        "observation.state": torch.from_numpy(state[f][:14]).float().unsqueeze(0),
        "observation.images.cam_high": torch.from_numpy(rearrange(img_front, "h w c -> c h w")),
        "observation.images.cam_left_wrist": torch.from_numpy(rearrange(img_left, "h w c -> c h w")),
        "observation.images.cam_right_wrist": torch.from_numpy(rearrange(img_right, "h w c -> c h w")),
        "prompt_embedding": torch.load(args.t5, map_location="cpu").float()[:64].unsqueeze(0),
        "chunk_idx": 0,
        "episode_idx": args.episode,
        "_quiet": True,
        "_skip_ref_image_save": True,
    }

    client = RobotInferenceClient(host=args.host, port=args.port, timeout_ms=180000)
    if not client.ping():
        raise SystemExit(f"server not reachable at {args.host}:{args.port}")

    timings = []
    shapes = set()
    ranges = []
    for i in range(args.warmup + args.iters):
        t0 = time.perf_counter()
        out = client.inference(obs)
        dt = time.perf_counter() - t0
        arr = np.asarray(out)
        shapes.add(tuple(arr.shape))
        ranges.append((float(arr.min()), float(arr.max())))
        assert np.isfinite(arr).all(), "non-finite action"
        if i >= args.warmup:
            timings.append({"iter": i - args.warmup, "e2e_s": dt})

    # server-side records
    server_recs = []
    if args.server_timing and os.path.isfile(args.server_timing):
        with open(args.server_timing, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    server_recs.append(json.loads(line))
    server_recs = server_recs[-args.iters:]

    for i, rec in enumerate(server_recs):
        if i < len(timings):
            timings[i]["server_total_s"] = rec.get("total_s")
            timings[i]["model_s"] = rec.get("model_s")
            timings[i]["preprocess_s"] = rec.get("preprocess_s")
            timings[i]["post_s"] = rec.get("post_s")
            timings[i]["comm_s"] = (
                timings[i]["e2e_s"] - rec.get("total_s") if rec.get("total_s") else None
            )
            timings[i]["gpu_allocated_gb"] = rec.get("gpu_allocated_gb")
            timings[i]["gpu_reserved_gb"] = rec.get("gpu_reserved_gb")
            timings[i]["gpu_peak_reserved_gb"] = rec.get("gpu_peak_reserved_gb")

    def summary(key):
        vals = [t[key] for t in timings if t.get(key) is not None]
        if not vals:
            return None
        return {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "p50": pct(vals, 50),
            "p95": pct(vals, 95),
            "min": min(vals),
            "max": max(vals),
        }

    report = {
        "episode": args.episode,
        "frame": args.frame,
        "task": task,
        "warmup": args.warmup,
        "iters": args.iters,
        "output_shapes": sorted(list(shapes)),
        "output_finite": True,
        "output_range_min": min(r[0] for r in ranges),
        "output_range_max": max(r[1] for r in ranges),
        "summary": {
            k: summary(k)
            for k in ("e2e_s", "server_total_s", "model_s", "preprocess_s", "post_s", "comm_s")
        },
        "gpu_last": {
            "allocated_gb": timings[-1].get("gpu_allocated_gb"),
            "reserved_gb": timings[-1].get("gpu_reserved_gb"),
        },
        "gpu_peak_reserved_gb": max(
            [t.get("gpu_peak_reserved_gb") or 0 for t in timings] or [0]
        ),
        "raw": timings,
    }

    with open(os.path.join(out_dir, "openloop_bench.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "openloop_bench.csv"), "w", encoding="utf-8") as fh:
        cols = ["iter", "e2e_s", "server_total_s", "model_s", "preprocess_s", "post_s", "comm_s",
                "gpu_allocated_gb", "gpu_reserved_gb", "gpu_peak_reserved_gb"]
        fh.write(",".join(cols) + "\n")
        for t in timings:
            fh.write(",".join("" if t.get(c) is None else f"{t[c]:.6f}" for c in cols) + "\n")

    np.save(os.path.join(out_dir, "openloop_last_action.npy"), np.asarray(out))
    print(json.dumps({k: v for k, v in report.items() if k != "raw"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
