"""Shared helpers to build RoboTwin task args outside scripts/collect_data.py.

Mirrors ``collect_data.main()`` argument assembly so that ad-hoc smoke/replay
scripts use exactly the same config resolution as the native collector
(embodiment lookup, camera type, domain randomisation flags, save_path layout).
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from typing import Any

import yaml


def _import_envs(robotwin_root: str):
    # RoboTwin resolves some asset paths relative to the *current directory*
    # (envs/utils/rand_create_cluttered_actor.py opens
    # "./assets/objects/objaverse/list.json" at import time), so the native
    # entrypoints must run from the repo root. Mirror that here instead of
    # duplicating the repo.
    os.chdir(robotwin_root)
    if robotwin_root not in sys.path:
        sys.path.insert(0, robotwin_root)
    envs = importlib.import_module("envs")
    return envs


def build_task(task_name: str, robotwin_root: str):
    _import_envs(robotwin_root)
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def load_args(
    task_name: str,
    task_config: str,
    robotwin_root: str,
    *,
    save_path: str | None = None,
    episode_num: int | None = None,
    use_seed: bool | None = None,
    render_freq: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Return (args, embodiment_dir_name) exactly as collect_data.main builds them."""
    envs = _import_envs(robotwin_root)
    configs_path = envs.CONFIGS_PATH

    with open(os.path.join(configs_path, f"{task_config}.yml"), "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    if save_path is not None:
        args["save_path"] = save_path
    if episode_num is not None:
        args["episode_num"] = int(episode_num)
    if use_seed is not None:
        args["use_seed"] = bool(use_seed)
    if render_freq is not None:
        args["render_freq"] = int(render_freq)

    embodiment_type = args.get("embodiment")
    with open(os.path.join(configs_path, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(kind: str) -> str:
        robot_file = _embodiment_types[kind]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    def get_embodiment_config(robot_file: str) -> dict:
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name

    embodiment_dir = re.sub(r"[^A-Za-z0-9_]+", "_", embodiment_name).lower()
    args["save_path"] = os.path.join(
        args["save_path"], args["task_config"], str(args["task_name"]), embodiment_dir
    )
    return args, embodiment_dir
