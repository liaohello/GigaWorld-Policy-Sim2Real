"""The min-max / z-score reconciliation must be exact, not approximate.

wam.cpp's converter forces unclipped z-score normalization for RoboTwin while
the official PyTorch script uses q01/q99 min-max. Rather than changing either
side's weights, the statistics file handed to the converter is rewritten so the
two transforms coincide. If this test ever fails, the C++ and PyTorch paths
would silently disagree on the same observation.
"""

import numpy as np

from gwp_sim2real.contracts import minmax_to_zscore_equiv


def _minmax(x, q01, q99):
    return 2.0 * (x - q01) / (q99 - q01) - 1.0


def _zscore(x, mean, std):
    return (x - mean) / std


def test_rewritten_bounds_reproduce_zscore_exactly():
    rng = np.random.default_rng(0)
    q01 = rng.normal(-1.0, 0.3, size=14)
    q99 = q01 + rng.uniform(1.0, 4.0, size=14)
    mean = (q01 + q99) / 2.0
    std = (q99 - q01) / 2.0

    q01_new, q99_new = minmax_to_zscore_equiv(q01, q99)
    x = rng.normal(0.0, 1.5, size=14)

    assert np.allclose(_minmax(x, q01_new, q99_new), _zscore(x, mean, std), atol=1e-12)


def test_degenerate_dims_stay_degenerate_but_finite():
    # Padding dims 14/15 have std 0 in the real statistics file. The rewrite must
    # not turn them into NaN; callers substitute std = 1 for the converter.
    q01 = np.array([0.0, 0.0])
    q99 = np.array([0.0, 0.0])
    q01_new, q99_new = minmax_to_zscore_equiv(q01, q99)
    assert np.allclose(q01_new, 0.0) and np.allclose(q99_new, 0.0)


def test_mapping_is_involutive_on_symmetric_bounds():
    q01 = np.array([-2.0, 0.5])
    q99 = np.array([4.0, 1.5])
    a, b = minmax_to_zscore_equiv(q01, q99)
    c, d = minmax_to_zscore_equiv(a, b)
    assert np.allclose([c, d], [q01, q99])
