#!/usr/bin/env python
"""Stage B1: data-contract report for the 50 native RoboTwin demonstrations.

Establishes, from the files themselves (nothing assumed):
  * the full HDF5 schema (dataset paths, shapes, dtypes)
  * that the effective action dimension is 14 and what each column means
  * joint/state physical ranges, gripper range and semantics
  * the state/action time offset
  * the camera set, stored encoding, resolution and byte format
  * the declared sampling frequency
  * cross-episode consistency over all 50 files, plus the seed mapping

Usage:
  python scripts/data/probe_hdf5.py --out runs/B_contract
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("RUNS_DIR", os.path.join(REPO_ROOT, "runs"))
# The demonstrations themselves live outside the repo; point at them explicitly.
RT = os.environ.get("ROBOTWIN_SOURCE_DIR", "")
DATA_ROOT = os.environ.get("ROBOTWIN_DATA_DIR") or os.path.join(
    RT, "data", "demo_clean", "beat_block_hammer", "aloha_agilex"
)
EP_DIR = DATA_ROOT


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS_DIR, "B_contract"))
    ap.add_argument("--ep-dir", default=EP_DIR)
    args = ap.parse_args()

    import h5py

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(RUNS_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, RT)
    from XPolicyLab.utils.process_data import decode_image_bit

    files = sorted(glob.glob(os.path.join(args.ep_dir, "data", "*.hdf5")))
    seeds = [int(x) for x in open(os.path.join(args.ep_dir, "seed.txt")).read().split()]
    report: dict = {
        "episode_dir": args.ep_dir,
        "num_hdf5": len(files),
        "num_seeds": len(seeds),
        "videos": len(glob.glob(os.path.join(args.ep_dir, "video", "*.mp4"))),
        "instructions": len(glob.glob(os.path.join(args.ep_dir, "instruction", "*.json"))),
        "seed_list_first10": seeds[:10],
    }

    # ---- full schema of the first file
    with h5py.File(files[0], "r") as f:
        schema = {}
        f.visititems(
            lambda name, obj: schema.__setitem__(
                name,
                {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                if isinstance(obj, h5py.Dataset)
                else None,
            )
            if isinstance(obj, h5py.Dataset)
            else None
        )
        report["schema"] = {k: v for k, v in schema.items() if v is not None}
        report["attrs"] = {k: str(v) for k, v in f.attrs.items()}
        report["root_groups"] = sorted(f.keys())
        report["cameras"] = sorted(f["vision"].keys())
        report["frequency"] = int(np.asarray(f["additional_info"]["frequency"]))
        report["camera_shape_declared"] = {
            cam: [int(v) for v in np.asarray(f["vision"][cam]["shape"])]
            for cam in f["vision"].keys()
        }
        colors_ds = f["vision"]["cam_head"]["colors"]
        raw = np.asarray(colors_ds[:1])
        decoded = decode_image_bit(raw)[0]
        report["camera_encoding"] = {
            "stored_dtype": str(colors_ds.dtype),
            "stored_shape": list(colors_ds.shape),
            "stored_bytes_per_frame": int(colors_ds.dtype.itemsize),
            "first_bytes": bytes(raw[0][:8]).hex(),
            "decoded_shape": list(decoded.shape),
            "decoded_dtype": str(decoded.dtype),
            "is_jpeg_soi": bool(bytes(raw[0][:2]) == b"\xff\xd8"),
            "has_xpl_rgb_marker": bool(b"XPL-RGB1" in bytes(raw[0][:128])),
        }
        instr = json.loads(f["instructions"][()])
        report["instructions_sample"] = instr[:2]

    # ---- per-episode scan over all 50 files
    rows = []
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            st = np.concatenate([
                np.asarray(f["state"]["left_arm_joint_states"], dtype=np.float32),
                np.asarray(f["state"]["left_ee_joint_states"], dtype=np.float32),
                np.asarray(f["state"]["right_arm_joint_states"], dtype=np.float32),
                np.asarray(f["state"]["right_ee_joint_states"], dtype=np.float32),
            ], axis=1)
            ac = np.concatenate([
                np.asarray(f["action"]["left_arm_joint_states"], dtype=np.float32),
                np.asarray(f["action"]["left_ee_joint_states"], dtype=np.float32),
                np.asarray(f["action"]["right_arm_joint_states"], dtype=np.float32),
                np.asarray(f["action"]["right_ee_joint_states"], dtype=np.float32),
            ], axis=1)
            rows.append({
                "episode": int(os.path.basename(path)[len("episode_"):-len(".hdf5")]),
                "frames": int(st.shape[0]),
                "dim": int(st.shape[1]),
                "finite": bool(np.isfinite(st).all() and np.isfinite(ac).all()),
                "action_shift_max_abs": float(np.abs(ac[:-1] - st[1:]).max()) if len(st) > 1 else None,
                "state_min": st.min(0).round(4).tolist(),
                "state_max": st.max(0).round(4).tolist(),
            })

    dims = sorted({r["dim"] for r in rows})
    frames = [r["frames"] for r in rows]
    report["episodes_summary"] = {
        "dims": dims,
        "all_finite": all(r["finite"] for r in rows),
        "frames_min": min(frames), "frames_max": max(frames),
        "frames_mean": float(np.mean(frames)),
        "action_equals_next_state_max_abs": max(
            r["action_shift_max_abs"] for r in rows if r["action_shift_max_abs"] is not None
        ),
        "seed_episode_pairs": list(zip([r["episode"] for r in rows], seeds))[:10],
    }

    # global physical ranges (per column) across all episodes
    mins = np.min([r["state_min"] for r in rows], axis=0)
    maxs = np.max([r["state_max"] for r in rows], axis=0)
    report["columns"] = {
        "order": ["left_joint_0..5", "left_gripper", "right_joint_0..5", "right_gripper"],
        "state_min": [round(float(v), 4) for v in mins],
        "state_max": [round(float(v), 4) for v in maxs],
    }
    report["per_episode"] = rows

    with open(os.path.join(out_dir, "hdf5_contract.json"), "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    slim = {k: v for k, v in report.items() if k != "per_episode"}
    print(json.dumps(slim, indent=2, default=_json_default)[:4000])
    print(f"\nreport -> {os.path.join(out_dir, 'hdf5_contract.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
