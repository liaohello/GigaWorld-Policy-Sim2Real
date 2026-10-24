#!/usr/bin/env python
"""Stage A3: environment / render smoke for beat_block_hammer + Aloha-AgileX.

Builds the real task env through the same code path as scripts/collect_data.py
(config resolution, embodiment, SAPIEN scene, curobo planner, three cameras),
then captures real RGB frames from head/left-wrist/right-wrist.

scripts/test_render.py only checks that the ray-tracing renderer initialises and
exits 0 even on failure, so it cannot be used as an acceptance gate. This script
fails loudly and writes the frames it actually captured.

Usage:
  python scripts/evaluate/smoke_env.py --task beat_block_hammer \
      --task-config demo_clean_smoke --seed 0 --out runs/A_smoke
"""
from __future__ import annotations

import argparse
import json
import os
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


def save_png(path: str, rgb: np.ndarray) -> None:
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="beat_block_hammer")
    ap.add_argument("--task-config", default="demo_clean_smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/A_smoke")
    ap.add_argument("--frames", type=int, default=3)
    args_cli = ap.parse_args()

    out_dir = os.path.join(WS, args_cli.out)
    os.makedirs(out_dir, exist_ok=True)

    task = rt_args.build_task(args_cli.task, RT)
    args, embodiment_dir = rt_args.load_args(
        args_cli.task, args_cli.task_config, RT,
        save_path=os.path.join(WS, "generated", "A_smoke"),
        episode_num=1, use_seed=True, render_freq=0,
    )
    args["need_plan"] = True
    args["save_data"] = False
    args["eval_mode"] = False

    report: dict = {
        "task": args_cli.task,
        "task_config": args_cli.task_config,
        "seed": args_cli.seed,
        "embodiment": args["embodiment_name"],
        "embodiment_dir": embodiment_dir,
        "save_path": args["save_path"],
        "camera_config": {
            "head_camera_type": args["camera"]["head_camera_type"],
            "wrist_camera_type": args["camera"]["wrist_camera_type"],
        },
    }

    try:
        task.setup_demo(now_ep_num=0, seed=args_cli.seed, **args)
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        with open(os.path.join(out_dir, "smoke_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(report["traceback"], flush=True)
        return 1

    # Robot / camera introspection straight from the live sim objects.
    report["left_arm_joints"] = [j.get_name() for j in task.robot.left_arm_joints]
    report["right_arm_joints"] = [j.get_name() for j in task.robot.right_arm_joints]
    report["left_gripper_joint"] = task.robot.left_gripper_name if hasattr(task.robot, "left_gripper_name") else None
    cam_names = sorted(task.cameras.camera_list.keys()) if hasattr(task.cameras, "camera_list") else []
    report["camera_names"] = cam_names

    frames = []
    head_pixels = []
    for i in range(args_cli.frames):
        obs = task.get_obs()
        cameras = obs["observation"]
        shot = {}
        for name in ("head_camera", "left_camera", "right_camera"):
            if name in cameras and "rgb" in cameras[name]:
                rgb = np.asarray(cameras[name]["rgb"])
                shot[name] = {"shape": list(rgb.shape), "dtype": str(rgb.dtype),
                              "min": int(rgb.min()), "max": int(rgb.max()),
                              "mean": float(rgb.mean())}
                save_png(os.path.join(out_dir, f"frame{i}_{name}.png"), rgb)
                if name == "head_camera":
                    head_pixels.append(rgb.copy())
        frames.append(shot)
        # advance the sim so successive frames are genuinely re-rendered
        for _ in range(5):
            task.scene.step()

    report["frames"] = frames
    report["joint_action_vector_dim"] = int(len(obs["joint_action"]["vector"])) if "joint_action" in obs else None
    report["joint_action_keys"] = sorted(obs["joint_action"].keys()) if "joint_action" in obs else None
    report["cameras_have_rgb"] = all(
        name in frames[0] for name in ("head_camera", "left_camera", "right_camera")
    )
    nonconstant = all(shot["head_camera"]["max"] > shot["head_camera"]["min"] for shot in frames)
    report["frames_non_constant"] = bool(nonconstant)
    report["head_frames_distinct"] = bool(
        len(head_pixels) > 1 and any(
            not np.array_equal(head_pixels[0], px) for px in head_pixels[1:]
        )
    )
    report["status"] = "PASS" if (report["cameras_have_rgb"] and nonconstant) else "FAIL"

    with open(os.path.join(out_dir, "smoke_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)

    task.close_env()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
