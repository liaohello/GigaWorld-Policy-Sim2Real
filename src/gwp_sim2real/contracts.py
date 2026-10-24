"""Observation and action contracts for the native 14-D Aloha-AgileX interface.

Normalization is not here: see `gwp_sim2real.normalization`, which operates on
the measured mean/std and is exercised by tests against the real transforms.

None of these numbers are guesses. They were measured from the stored
demonstrations and the checkpoint:

- 14 action dims = left arm 6 + left gripper 1 + right arm 6 + right gripper 1
  (`action/left_arm_joint_states` etc. in the HDF5 episodes);
- horizon 48 = the checkpoint's action horizon, and what the policy actually
  returns per call;
- camera names are what the simulator emits (`cam_head`, `cam_left_wrist`,
  `cam_right_wrist`), which is *not* what the LeRobot writer calls them
  (`cam_high`, ...) — see `LEROBOT_CAMERA_KEYS`.

Keeping these in one importable module means the adapter, the evaluation
scripts and the tests cannot drift apart silently.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

ACTION_DIM = 14
ACTION_HORIZON = 48

CAMERA_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

# The simulator and the dataset writer disagree on one name. Mapping in one
# place beats a scattered rename.
LEROBOT_CAMERA_KEYS = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

# Order matters: it is the order the policy was trained on and the order the
# stored demonstrations use.
STATE_SEGMENTS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)

# The checkpoint's action *channel* width is 16: 14 real dims plus 2 zero-padded
# dims. Do not confuse the two numbers; the padding exists only to match the
# pretrained projection layer.
MODEL_ACTION_CHANNELS = 16


class ContractError(ValueError):
    """Raised when an observation or an action chunk violates the native contract."""


def assert_state(state: object) -> np.ndarray:
    """Return `state` as a float32 vector of length 14, or raise."""
    vec = np.asarray(state, dtype=np.float32).reshape(-1)
    if vec.size != ACTION_DIM:
        raise ContractError(f"state has {vec.size} dims, expected {ACTION_DIM}")
    if not np.isfinite(vec).all():
        raise ContractError("state contains non-finite values")
    return vec


def assert_action_chunk(
    actions: object, *, horizon: int | None = None, dim: int = ACTION_DIM
) -> np.ndarray:
    """Return `actions` as a `[T, 14]` float32 array, or raise.

    `T` is checked against `horizon` only when one is given: the policy returns
    48 rows, but the simulator executes a prefix.
    """
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ContractError(f"action chunk must be 2-D, got shape {arr.shape}")
    if arr.shape[1] != dim:
        raise ContractError(f"action chunk has {arr.shape[1]} dims, expected {dim}")
    if horizon is not None and arr.shape[0] != horizon:
        raise ContractError(f"action chunk has {arr.shape[0]} steps, expected {horizon}")
    if not np.isfinite(arr).all():
        raise ContractError("action chunk contains non-finite values")
    return arr


def split_state(packed: object) -> dict[str, np.ndarray]:
    """Split a packed 14-D state into its four named segments."""
    vec = assert_state(packed)
    out: dict[str, np.ndarray] = {}
    i = 0
    for name, width in STATE_SEGMENTS:
        out[name] = vec[i : i + width]
        i += width
    return out


def split_action(row: object) -> dict[str, np.ndarray]:
    """Split one 14-D action row into arm and gripper segments."""
    vec = assert_action_chunk(row)[0]
    return {
        "left_arm": vec[0:6],
        "left_gripper": vec[6:7],
        "right_arm": vec[7:13],
        "right_gripper": vec[13:14],
    }


def assert_images(vision: Mapping[str, object]) -> None:
    """Raise if any of the three required cameras is missing."""
    missing = [k for k in CAMERA_KEYS if k not in vision]
    if missing:
        raise ContractError(f"observation is missing cameras {missing}; got {sorted(vision)}")
