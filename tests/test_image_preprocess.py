"""Image handling: colour order, channel order, and scale detection.

The simulator hands over HxWx3 uint8 in RGB. The server expects CHW float64 in
[0, 1] because it multiplies by 255 and casts back to uint8. An extra BGR swap
or a double 255 multiply would still "work" and quietly degrade every policy
input, which is exactly the kind of bug a test should catch.
"""

import numpy as np

from gwp_sim2real.inference.client import _to_chw_float01


def test_hwc_uint8_becomes_chw_unit_float():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[..., 0] = 255  # pure red in RGB

    out = _to_chw_float01(img)

    assert out.shape == (3, 240, 320)
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isclose(out[0].mean(), 1.0)   # red stays in channel 0
    assert np.isclose(out[1].mean(), 0.0)
    assert np.isclose(out[2].mean(), 0.0)


def test_chw_uint8_passes_through_untouched():
    img = np.full((3, 8, 8), 128, dtype=np.uint8)
    out = _to_chw_float01(img)
    assert out.shape == (3, 8, 8)
    assert np.isclose(out.mean(), 128 / 255.0)


def test_unit_range_float_is_not_scaled_twice():
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out = _to_chw_float01(img)
    assert np.isclose(out.mean(), 0.5)


def test_grayscale_is_replicated_to_three_channels():
    img = np.full((8, 8, 1), 200, dtype=np.uint8)
    out = _to_chw_float01(img)
    assert out.shape[0] == 3
    assert np.allclose(out[0], out[1]) and np.allclose(out[1], out[2])


def test_rejects_non_image():
    import pytest

    with pytest.raises(Exception):
        _to_chw_float01(np.zeros((8, 8)))
