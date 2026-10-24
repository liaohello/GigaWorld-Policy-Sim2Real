"""The 14-D contract must fail loudly, not silently reshape."""

import numpy as np
import pytest

from gwp_sim2real import contracts


def test_action_dim_is_the_sum_of_the_segments():
    assert sum(w for _, w in contracts.STATE_SEGMENTS) == contracts.ACTION_DIM == 14


def test_accepts_a_well_formed_chunk():
    chunk = np.zeros((48, 14), dtype=np.float32)
    out = contracts.assert_action_chunk(chunk, horizon=contracts.ACTION_HORIZON)
    assert out.shape == (48, 14)
    assert out.dtype == np.float32


def test_rejects_wrong_width():
    with pytest.raises(contracts.ContractError, match="expected 14"):
        contracts.assert_action_chunk(np.zeros((48, 32)))


def test_rejects_wrong_horizon_only_when_asked():
    chunk = np.zeros((8, 14))
    # A prefix is legal for execution...
    assert contracts.assert_action_chunk(chunk).shape == (8, 14)
    # ...but not when the caller wants a full policy chunk.
    with pytest.raises(contracts.ContractError, match="expected 48"):
        contracts.assert_action_chunk(chunk, horizon=48)


def test_rejects_non_finite():
    bad = np.zeros((48, 14))
    bad[3, 2] = np.nan
    with pytest.raises(contracts.ContractError, match="non-finite"):
        contracts.assert_action_chunk(bad)


def test_state_splits_in_the_recorded_order():
    packed = np.arange(14, dtype=np.float32)
    parts = contracts.split_state(packed)
    assert list(parts) == [name for name, _ in contracts.STATE_SEGMENTS]
    assert parts["left_arm_joint_state"].tolist() == [0, 1, 2, 3, 4, 5]
    assert parts["left_ee_joint_state"].tolist() == [6]
    assert parts["right_arm_joint_state"].tolist() == [7, 8, 9, 10, 11, 12]
    assert parts["right_ee_joint_state"].tolist() == [13]


def test_camera_names_differ_only_where_documented():
    assert contracts.LEROBOT_CAMERA_KEYS["cam_head"] == "cam_high"
    assert contracts.LEROBOT_CAMERA_KEYS["cam_left_wrist"] == "cam_left_wrist"
    assert set(contracts.LEROBOT_CAMERA_KEYS) == set(contracts.CAMERA_KEYS)


def test_missing_camera_is_an_error():
    with pytest.raises(contracts.ContractError, match="missing cameras"):
        contracts.assert_images({"cam_head": None})
