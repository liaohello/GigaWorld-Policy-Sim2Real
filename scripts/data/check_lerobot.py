#!/usr/bin/env python
"""Stage B: verify the produced LeRobot v3.0 dataset against the source HDF5.

Checks, per episode:
  * declared features (observation.state / action dim, three cameras, fps)
  * state/action shape and finiteness
  * the exact task string LeRobot stored (the converter picks one instruction at
    random per episode -- this is the episode -> text mapping the T5 cache needs)
  * action[t] == state[t+1] inside the converted parquet (time alignment)
  * a decoded video frame per camera compared against the HDF5 stored frame at
    the same index, to prove camera order and RGB order survived the conversion

Usage:
  python scripts/data/check_lerobot.py --repo datasets/robotwin_gwp/beat_block_hammer_smoke_v3 --episodes 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(WS, "src", "RoboTwin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path relative to workspace or absolute")
    ap.add_argument("--episodes", default="0", help="comma separated episode indices")
    ap.add_argument("--out", default="runs/B_contract")
    ap.add_argument("--compare-hdf5", action="store_true", default=True)
    args_cli = ap.parse_args()

    repo = args_cli.repo
    if not os.path.isabs(repo):
        repo = os.path.join(WS, repo)

    import pandas as pd
    import av

    out_dir = os.path.join(WS, args_cli.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(repo, "meta", "info.json")) as f:
        info = json.load(f)

    report: dict = {"repo": repo, "info": info, "episodes": {}}

    ep_meta = []
    for dirpath, _, files in os.walk(os.path.join(repo, "meta", "episodes")):
        for fn in files:
            if fn.endswith(".parquet"):
                ep_meta.append(os.path.join(dirpath, fn))
    ep_df = pd.concat([pd.read_parquet(p) for p in sorted(ep_meta)], ignore_index=True)
    report["num_episodes"] = int(len(ep_df))

    sys.path.insert(0, RT)
    from XPolicyLab.utils.process_data import decode_image_bit
    import h5py

    for ep in [int(x) for x in args_cli.episodes.split(",")]:
        row = ep_df[ep_df["episode_index"] == ep].iloc[0]
        length = int(row["length"])
        data_path = os.path.join(
            repo, "data", f"chunk-{int(row['data/chunk_index']):03d}",
            f"file-{int(row['data/file_index']):03d}.parquet",
        )
        df = pd.read_parquet(data_path)
        df = df[df["episode_index"] == ep].sort_values("frame_index")
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        tasks = row["tasks"]
        if isinstance(tasks, (list, np.ndarray)):
            tasks = list(tasks)
        ep_rep = {
            "length": length,
            "state_shape": list(state.shape),
            "action_shape": list(action.shape),
            "state_finite": bool(np.isfinite(state).all()),
            "action_finite": bool(np.isfinite(action).all()),
            "state_min": state.min(0).round(4).tolist(),
            "state_max": state.max(0).round(4).tolist(),
            "action_min": action.min(0).round(4).tolist(),
            "action_max": action.max(0).round(4).tolist(),
            "task": tasks[0] if tasks else None,
            "tasks_all": tasks,
            "action_t_eq_state_t1_max_abs": float(np.abs(action[:-1] - state[1:]).max()),
        }

        # video frame 0 per camera vs HDF5 stored frame 0
        if args_cli.compare_hdf5:
            # feature key -> source HDF5 camera group
            cam_pairs = {
                "cam_high": "cam_head",
                "cam_left_wrist": "cam_left_wrist",
                "cam_right_wrist": "cam_right_wrist",
            }
            h5_path = os.path.join(
                RT, "data", "demo_clean", "beat_block_hammer", "aloha_agilex",
                "data", f"episode_{ep:07d}.hdf5",
            )
            with h5py.File(h5_path, "r") as f:
                stored = {
                    src: decode_image_bit(np.array(f["vision"][src]["colors"])[:1])[0]
                    for src in cam_pairs.values()
                }
            cmp = {}
            for feat_cam, src_cam in cam_pairs.items():
                vpath = os.path.join(
                    repo, "videos", f"observation.images.{feat_cam}",
                    f"chunk-{int(row[f'videos/observation.images.{feat_cam}/chunk_index']):03d}",
                    f"file-{int(row[f'videos/observation.images.{feat_cam}/file_index']):03d}.mp4",
                )
                container = av.open(vpath)
                try:
                    frame = next(iter(container.decode(video=0))).to_ndarray(format="rgb24")
                finally:
                    container.close()
                ref = stored[src_cam]
                if ref.shape == frame.shape:
                    d = np.abs(ref.astype(np.int16) - frame.astype(np.int16))
                    cmp[feat_cam] = {
                        "source_hdf5_camera": src_cam,
                        "mean_abs_diff": float(d.mean()),
                        "max_abs_diff": int(d.max()),
                        "shape": list(frame.shape),
                    }
                else:
                    cmp[feat_cam] = {"source_hdf5_camera": src_cam,
                                     "error": f"shape {ref.shape} vs {frame.shape}"}
            ep_rep["frame0_vs_hdf5"] = cmp

        report["episodes"][str(ep)] = ep_rep
        print(json.dumps(ep_rep, indent=2)[:2000])

    with open(os.path.join(out_dir, "lerobot_contract.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {os.path.join(out_dir, 'lerobot_contract.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
