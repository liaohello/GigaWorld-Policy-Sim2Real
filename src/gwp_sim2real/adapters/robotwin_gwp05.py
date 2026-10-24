"""GWP 0.5 adapter for RoboTwin's XPolicyLab policy interface.

This file is *installed into* the RoboTwin tree by
``scripts/setup/install_adapter.sh`` (as ``XPolicyLab/policy/GWP05Native/model.py``)
because XPolicyLab imports the policy as a local package with a ``deploy.yml``
beside it. Keep it free of local paths: everything machine-specific comes from
``deploy.yml`` or environment variables.

Interface contract, in one paragraph: the simulator hands over three RGB
cameras, a 14-D packed joint state and an instruction; this adapter forwards
them to the GWP PyTorch server over ``giga_models.sockets`` (ZMQ, torch.save
payload) and returns *physical* 14-D joint targets, already denormalized and
already absolute. Nothing downstream may denormalize or add state again.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
# Importable root is the parent of the XPolicyLab checkout (AGENTS.md rule).
_XPL_ROOT = _CUR_DIR.parents[2]

if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    unpack_robot_state,
)

# Camera names as delivered by the RoboTwin eval bridge
# (scripts/eval_policy_xpolicylab.py: CAMERA_NAME_MAP).
CAM_HIGH = "cam_head"
CAM_LEFT = "cam_left_wrist"
CAM_RIGHT = "cam_right_wrist"

# The GWP 0.5 pipeline already denormalises its action output and adds the
# current state back on the joint dims (inference_openloop.get_policy ->
# inference(): `((pred+1)/2)*delta_range+delta_min` then `+ state * mask` with
# mask = [T]*6+[F]+[T]*6+[F]). Anything this adapter did on top of that would be
# a double conversion, so it is deliberately absent -- see GWP05Native/README.md.


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class Model(ModelTemplate):
    """GWP 0.5 adapter: forward RoboTwin observations to the validated PyTorch
    GigaWorld-Policy server over the official ``RobotInferenceClient`` (ZMQ +
    ``torch.save``), and hand back physical 14-D joint targets.

    This adapter does not load the model itself. The model process
    (``src/giga-world-policy`` + ``--fixed-t5-path``) owns the GPU; this process
    only speaks the wire protocol, so it can run in the simulation environment.
    """

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name") or "beat_block_hammer"
        self.env_cfg_type = self.model_cfg.get("env_cfg_type") or "aloha_agilex"
        self.action_type = self.model_cfg.get("action_type") or "joint"

        self.gwp_host = str(self.model_cfg.get("gwp_host") or "127.0.0.1")
        self.gwp_port = int(self.model_cfg.get("gwp_port") or 11444)
        self.timeout_ms = int(self.model_cfg.get("gwp_timeout_ms") or 60000)

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.packed_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )

        self.action_chunk = int(self.model_cfg.get("action_chunk") or 48)
        exec_chunk = self.model_cfg.get("execute_action_chunk")
        self.execute_action_chunk = (
            self.action_chunk if exec_chunk in (None, "") else max(1, min(int(exec_chunk), self.action_chunk))
        )

        self.prompt_mode = str(self.model_cfg.get("prompt_mode") or "embedding").lower()
        self.t5_embedding_path = self.model_cfg.get("t5_embedding_path")
        self.t5_index_path = self.model_cfg.get("t5_index_path")
        self._t5_cache: dict[str, np.ndarray] = {}
        self._t5_index: dict[str, str] = {}
        if self.t5_index_path and os.path.isfile(self.t5_index_path):
            with open(self.t5_index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for rec in payload.get("episodes", []):
                for key in ("instruction_raw", "instruction_clean"):
                    text = rec.get(key)
                    if text:
                        self._t5_index.setdefault(str(text), rec["path"])
        if not _as_bool(self.model_cfg.get("load_model"), True):
            raise ValueError("GWP05Native requires load_model=true (it is the remote client).")

        # Optional per-call trace of the commanded chunk, for the trajectory
        # deliverable. Off unless GWP05_TRACE is set, so normal runs stay quiet.
        self.trace_path = os.environ.get("GWP05_TRACE") or None
        self._client = None
        self._latest_payload: dict[str, Any] | None = None
        self._instruction: str | None = None
        self.request_count = 0
        print(
            "[GWP05Native] initialized",
            f"server={self.gwp_host}:{self.gwp_port}",
            f"packed_dim={self.packed_dim}",
            f"action_chunk={self.action_chunk} exec={self.execute_action_chunk}",
            f"prompt_mode={self.prompt_mode}",
            f"t5_index_entries={len(self._t5_index)}",
            flush=True,
        )

    # ---------------------------------------------------------------- client
    def _get_client(self):
        if self._client is None:
            try:
                from giga_models.sockets import RobotInferenceClient
            except ModuleNotFoundError:
                # giga-models is only pip-installed in the policy env; make the
                # workspace checkout importable here instead of duplicating it.
                gm = self.model_cfg.get("giga_models_path") or os.environ.get(
                    "GIGA_MODELS_DIR"
                )
                if not gm:
                    raise
                gm = str(gm)
                if not os.path.isabs(gm):
                    gm = os.path.join(_XPL_ROOT, "..", gm)
                gm = os.path.abspath(gm)
                if gm not in sys.path:
                    sys.path.insert(0, gm)
                from giga_models.sockets import RobotInferenceClient

            self._client = RobotInferenceClient(
                host=self.gwp_host, port=self.gwp_port, timeout_ms=self.timeout_ms
            )
            if not self._client.ping():
                raise RuntimeError(f"GWP policy server not reachable at {self.gwp_host}:{self.gwp_port}")
            print(f"[GWP05Native] connected to {self.gwp_host}:{self.gwp_port}", flush=True)
        return self._client

    # ------------------------------------------------------------- obs encode
    @staticmethod
    def _image_chw_float01(item: Any) -> np.ndarray:
        """Server expects the exact array the official open-loop client sends:
        float64 CHW in [0, 1] (it multiplies by 255 and casts to uint8)."""
        arr = np.asarray(item)
        if arr.ndim != 3:
            raise ValueError(f"expected 3-D camera array, got {arr.shape}")
        if arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(arr)) <= 1.5 else 1.0
            arr = np.clip(arr * scale, 0, 255)
        return (arr.astype(np.float64) / 255.0)

    def _pack_state(self, obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("state", {})
        parts = [
            np.asarray(state["left_arm_joint_state"], dtype=np.float32).reshape(-1),
            np.asarray(state["left_ee_joint_state"], dtype=np.float32).reshape(-1),
            np.asarray(state["right_arm_joint_state"], dtype=np.float32).reshape(-1),
            np.asarray(state["right_ee_joint_state"], dtype=np.float32).reshape(-1),
        ]
        packed = np.concatenate(parts)
        if packed.size != self.packed_dim:
            raise ValueError(f"state dim {packed.size} != expected {self.packed_dim}")
        return packed

    def _embedding_for(self, instruction: str) -> np.ndarray:
        import torch

        text = instruction or self.task_name
        cached = self._t5_cache.get(text)
        if cached is None:
            path = self._t5_index.get(text)
            if path is None:
                # fall back to the configured default embedding
                path = self.t5_embedding_path
            if path is None or not os.path.isfile(path):
                raise FileNotFoundError(
                    f"no T5 embedding for instruction {text!r}; index={self.t5_index_path}"
                )
            tensor = torch.load(path, map_location="cpu")
            if isinstance(tensor, dict):
                tensor = tensor.get("t5_embedding", next(iter(tensor.values())))
            tensor = tensor[:64].float()
            if tensor.shape[0] < 64:
                tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 64 - tensor.shape[0]))
            cached = tensor.numpy()
            self._t5_cache[text] = cached
        return cached

    def update_obs(self, obs):
        import torch

        vision = obs.get("vision", {})
        for key in (CAM_HIGH, CAM_LEFT, CAM_RIGHT):
            if key not in vision:
                raise KeyError(f"observation is missing camera {key!r}; got {sorted(vision)}")

        instruction = obs.get("instruction") or obs.get("instructions") or self.task_name
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else self.task_name
        self._instruction = str(instruction)

        payload: dict[str, Any] = {
            "observation.images.cam_high": torch.from_numpy(self._image_chw_float01(vision[CAM_HIGH]["color"])),
            "observation.images.cam_left_wrist": torch.from_numpy(self._image_chw_float01(vision[CAM_LEFT]["color"])),
            "observation.images.cam_right_wrist": torch.from_numpy(self._image_chw_float01(vision[CAM_RIGHT]["color"])),
            "observation.state": torch.from_numpy(self._pack_state(obs)),
            "chunk_idx": int(self.request_count),
            "_quiet": True,
            "_skip_ref_image_save": True,
        }
        if self.prompt_mode == "text":
            payload["prompt"] = self._instruction
        else:
            payload["prompt_embedding"] = torch.from_numpy(
                self._embedding_for(self._instruction)
            ).unsqueeze(0)
        self._latest_payload = payload

    # ------------------------------------------------------------- inference
    def get_action(self):
        if self._latest_payload is None:
            raise AssertionError("update_obs must be called before get_action")
        client = self._get_client()
        t0 = time.perf_counter()
        raw = client.inference(self._latest_payload)
        latency = time.perf_counter() - t0
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[0] < 1 or actions.shape[1] != self.packed_dim:
            raise ValueError(f"unexpected action shape {actions.shape}, expected [*,{self.packed_dim}]")
        if not np.isfinite(actions).all():
            raise ValueError("policy returned non-finite actions")

        self.request_count += 1
        print(
            "[GWP05Native] action chunk",
            f"#{self.request_count}",
            f"shape={tuple(actions.shape)}",
            f"latency_ms={latency * 1000:.1f}",
            f"first6={np.array2string(actions[0, :6], precision=4, separator=',')}",
            flush=True,
        )

        if self.trace_path:
            rec = {
                "call": self.request_count,
                "instruction": self._instruction,
                "state": np.asarray(self._latest_payload["observation.state"]).reshape(-1).tolist(),
                "chunk": actions.tolist(),
                "executed": int(self.execute_action_chunk),
                "latency_ms": latency * 1000.0,
            }
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        chunk = actions[: self.execute_action_chunk]
        out = []
        for row in chunk:
            out.append(
                unpack_robot_state(
                    row, self.action_type, self.robot_action_dim_info, source_type="obs"
                )
            )
        return out

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        self._latest_payload = None
        self._instruction = None
        self.request_count = 0
        client = self._client
        if client is not None:
            # A stale action-only prefix cache on the server would condition the
            # next episode on the previous one. Reconnect-per-episode is the
            # cheapest reliable reset: the ZMQ REP socket is stateless.
            try:
                del client
            except Exception:
                pass
            self._client = None

    def prepare_case(self, case_meta=None):
        if not case_meta:
            return None
        instruction = case_meta.get("instruction")
        if instruction:
            self._instruction = str(instruction)
        return {"instruction": self._instruction}

    def on_trial_end(self, result=None):
        return None
