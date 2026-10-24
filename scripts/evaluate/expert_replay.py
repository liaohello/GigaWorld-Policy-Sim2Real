#!/usr/bin/env python
"""Stage B4: expert action replay through the native RoboTwin control interface.

Loads one recorded HDF5 episode, rebuilds the *same* scene with the seed that
produced it, and replays the stored 14-D absolute joint/gripper actions through
``task.take_action(action, action_type="qpos")`` -- the exact interface the
policy adapter will use later. Success is judged by the environment's own
``check_success()``.

It also re-renders frame 0 at that seed and compares it against the frame stored
in the HDF5, which is the camera-order / RGB-order cross-check for the data
contract.

Usage:
  python scripts/evaluate/expert_replay.py --episode 0 --out runs/B_replay
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "lib"))

import env as project_env  # noqa: E402

project_env.load_env()
# RoboTwin is an external checkout; never a subdirectory of this repository.
RT = project_env.require("ROBOTWIN_SOURCE_DIR")["ROBOTWIN_SOURCE_DIR"]
sys.path.insert(0, RT)

import rt_args  # noqa: E402


def read_seed_list(ep_dir: str) -> list[int]:
    with open(os.path.join(ep_dir, "seed.txt")) as f:
        return [int(x) for x in f.read().split()]


def write_video(path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError("no frames captured")
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg") or "${ENVS_DIR:?set ENVS_DIR}/robotwin_gwp_tools/bin/ffmpeg"
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "-",
         "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", path],
        stdin=subprocess.PIPE,
    )
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="beat_block_hammer")
    ap.add_argument("--task-config", default="demo_clean_smoke")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="runs/B_replay")
    ap.add_argument("--step-stride", type=int, default=1,
                    help="replay every Nth recorded step (1 = full fidelity)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = all recorded steps")
    ap.add_argument("--video-fps", type=int, default=10)
    args_cli = ap.parse_args()

    import h5py

    out_dir = os.path.join(WS, args_cli.out)
    os.makedirs(out_dir, exist_ok=True)

    ep_dir = os.path.join(RT, "data", "demo_clean", args_cli.task, "aloha_agilex")
    h5_path = os.path.join(ep_dir, "data", f"episode_{args_cli.episode:07d}.hdf5")
    seeds = read_seed_list(ep_dir)
    seed = seeds[args_cli.episode]

    with h5py.File(h5_path, "r") as f:
        state = np.concatenate([
            np.array(f["state"]["left_arm_joint_states"]),
            np.array(f["state"]["left_ee_joint_states"]),
            np.array(f["state"]["right_arm_joint_states"]),
            np.array(f["state"]["right_ee_joint_states"]),
        ], axis=1).astype(np.float32)
        action = np.concatenate([
            np.array(f["action"]["left_arm_joint_states"]),
            np.array(f["action"]["left_ee_joint_states"]),
            np.array(f["action"]["right_arm_joint_states"]),
            np.array(f["action"]["right_ee_joint_states"]),
        ], axis=1).astype(np.float32)
        stored_bits = np.array(f["vision"]["cam_head"]["colors"])
        stored_shape = np.array(f["vision"]["cam_head"]["shape"])
        instructions = json.loads(f["instructions"][()])

    sys.path.insert(0, RT)
    from XPolicyLab.utils.process_data import decode_image_bit

    stored_head = decode_image_bit(stored_bits[:1])[0]

    report = {
        "episode": args_cli.episode,
        "hdf5": h5_path,
        "seed": int(seed),
        "frames": int(state.shape[0]),
        "state_dim": int(state.shape[1]),
        "action_t_equals_state_t+1_max_abs": float(np.abs(action[:-1] - state[1:]).max()),
        "stored_cam_head_shape": stored_shape.tolist(),
        "instruction_sample": instructions[0] if instructions else None,
    }

    task = rt_args.build_task(args_cli.task, RT)
    args, _ = rt_args.load_args(
        args_cli.task, args_cli.task_config, RT,
        save_path=os.path.join(WS, "generated", "B_replay"),
        episode_num=1, use_seed=True, render_freq=0,
    )
    args["need_plan"] = False
    args["save_data"] = False
    args["eval_mode"] = False
    args["left_joint_path"] = []
    args["right_joint_path"] = []

    try:
        task.setup_demo(now_ep_num=0, seed=seed, **args)
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        with open(os.path.join(out_dir, f"replay_ep{args_cli.episode}.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(report["traceback"])
        return 1

    obs0 = task.get_obs()
    live_head = np.asarray(obs0["observation"]["head_camera"]["rgb"])
    live_left = np.asarray(obs0["observation"]["left_camera"]["rgb"])
    live_right = np.asarray(obs0["observation"]["right_camera"]["rgb"])
    report["live_shapes"] = {
        "head": list(live_head.shape), "left": list(live_left.shape), "right": list(live_right.shape),
    }
    if stored_head.shape == live_head.shape:
        diff = np.abs(stored_head.astype(np.int16) - live_head.astype(np.int16))
        report["frame0_head_vs_stored"] = {
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": int(diff.max()),
            "p99_abs_diff": float(np.percentile(diff, 99)),
        }
    else:
        report["frame0_head_vs_stored"] = {"error": f"shape mismatch {stored_head.shape} vs {live_head.shape}"}

    # state cross-check: live joint vector vs stored state[0]
    live_state = np.asarray(obs0["joint_action"]["vector"], dtype=np.float32)
    report["live_state_dim"] = int(live_state.size)
    if live_state.size == state.shape[1]:
        report["live_state_vs_stored_t0_max_abs"] = float(np.abs(live_state - state[0]).max())

    n = state.shape[0]
    steps = list(range(0, n, args_cli.step_stride))
    if args_cli.max_steps:
        steps = steps[: args_cli.max_steps]

    frames = []
    applied = 0
    try:
        for idx in steps:
            task.take_action(action[idx], action_type="qpos")
            applied += 1
            cur = task.get_obs()
            frames.append(np.asarray(cur["observation"]["head_camera"]["rgb"]).copy())
            if task.eval_success:
                break
    except Exception:
        report["replay_traceback"] = traceback.format_exc()

    report["steps_applied"] = applied
    report["success"] = bool(task.check_success())
    report["eval_success_flag"] = bool(task.eval_success)
    report["final_state_max_abs_vs_stored_last"] = float(
        np.abs(np.asarray(task.get_obs()["joint_action"]["vector"], dtype=np.float32) - state[-1]).max()
    )
    report["status"] = "PASS" if report["success"] else "FAIL"

    video_path = os.path.join(out_dir, f"replay_ep{args_cli.episode}_seed{seed}.mp4")
    write_video(video_path, frames, args_cli.video_fps)
    report["video"] = video_path
    np.save(os.path.join(out_dir, f"replay_ep{args_cli.episode}_applied_actions.npy"), action[: applied])

    with open(os.path.join(out_dir, f"replay_ep{args_cli.episode}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    task.close_env()
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
