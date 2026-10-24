"""Normalization equivalence, tested against the real transform.

The bug this file exists to prevent: deriving `mean`/`std` from `q01`/`q99` and
then "converting" the bounds back — which is an identity, passes every test that
reuses the same derivation, and leaves the two runtimes disagreeing.

So the fixtures below carry mean/std that are *independent* of q01/q99, exactly
like a real statistics file, and the assertions go through
`gwp_sim2real.normalization` — the same module `scripts/data/norm_convention.py`
runs in production.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from gwp_sim2real.normalization import (
    convert_block,
    denormalize_minmax,
    denormalize_zscore,
    floor_zero_std,
    normalize_minmax,
    normalize_zscore,
    zscore_equivalent_stats,
)

# Measured-looking statistics: mean/std are not the midpoint/half-width of
# q01/q99, because real distributions are skewed. Keeping them independent is
# the whole point of the fixture.
MEAN = np.array(
    [0.12, -0.40, 0.03, 0.51, -0.08, 0.27, 0.00, 0.19, -0.31, 0.44, 0.02, -0.15, 0.36, -0.02]
)
STD = np.array(
    [0.71, 0.33, 0.95, 1.20, 0.44, 0.61, 0.05, 0.88, 0.52, 1.05, 0.29, 0.73, 0.41, 0.66]
)
Q01 = np.array(
    [-1.90, -1.05, -1.42, -2.60, -0.98, -1.30, -0.10, -1.55, -1.70, -2.10, -0.60, -1.45, -0.42, -1.30]
)
Q99 = np.array(
    [1.75, 0.62, 1.88, 3.20, 0.71, 2.15, 0.11, 2.10, 0.85, 2.60, 0.80, 1.60, 1.35, 1.05]
)


def _block() -> dict:
    return {
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "min": (MEAN - 4 * STD).tolist(),
        "max": (MEAN + 4 * STD).tolist(),
        "q01": Q01.tolist(),
        "q99": Q99.tolist(),
    }


def test_conversion_is_not_an_identity():
    """If this fails, the 'conversion' does nothing and the tests below are moot."""
    out = convert_block(_block())
    new_q01 = np.array(out["q01"])
    new_q99 = np.array(out["q99"])

    assert np.allclose(new_q01, MEAN - STD)
    assert np.allclose(new_q99, MEAN + STD)
    # The rewritten bounds must differ from the measured quantiles.
    assert not np.allclose(new_q01, Q01)
    assert not np.allclose(new_q99, Q99)


def test_input_normalization_matches_zscore():
    out = convert_block(_block())
    x = np.linspace(-3.0, 3.0, 97)[:, None] * STD[None, :] + MEAN[None, :]

    from_minmax = normalize_minmax(x, out["q01"], out["q99"])
    from_zscore = normalize_zscore(x, MEAN, STD)

    assert np.allclose(from_minmax, from_zscore, atol=1e-12)


def test_output_denormalization_matches_zscore():
    out = convert_block(_block())
    a = np.linspace(-1.0, 1.0, 41)[:, None] * np.ones((1, MEAN.size))

    from_minmax = denormalize_minmax(a, out["q01"], out["q99"])
    from_zscore = denormalize_zscore(a, MEAN, STD)

    assert np.allclose(from_minmax, from_zscore, atol=1e-12)


def test_round_trip_within_the_converted_frame():
    out = convert_block(_block())
    x = MEAN + 0.7 * STD
    a = normalize_minmax(x, out["q01"], out["q99"])
    assert np.allclose(denormalize_minmax(a, out["q01"], out["q99"]), x, atol=1e-12)


def test_zero_bounds_come_from_measured_statistics_only():
    mean = np.array([2.0, -1.0])
    std = np.array([0.5, 0.25])
    q01, q99 = zscore_equivalent_stats(mean, std)
    assert np.allclose(q01, [1.5, -1.25])
    assert np.allclose(q99, [2.5, -0.75])


def test_padding_dims_are_floored_and_stay_zero():
    block = _block()
    pad = 12
    block["std"][pad] = 0.0
    block["mean"][pad] = 0.0
    block["q01"][pad] = 0.0
    block["q99"][pad] = 0.0

    out, changed = floor_zero_std(block, model_dim=14)

    assert changed == [pad]
    assert out["std"][pad] == 1.0 and out["mean"][pad] == 0.0
    assert out["q01"][pad] == -1.0 and out["q99"][pad] == 1.0
    # A padded value is exactly 0 in the PyTorch pipeline; both runtimes must
    # still produce 0 after flooring.
    assert normalize_zscore(np.zeros(14), out["mean"], out["std"])[pad] == 0.0
    assert normalize_minmax(np.zeros(14), out["q01"], out["q99"])[pad] == 0.0


def test_padding_beyond_model_dim_is_not_reported():
    block = _block()
    for key in ("std", "mean", "q01", "q99", "min", "max"):
        block[key] = block[key] + [0.0, 0.0]
    _, changed = floor_zero_std(block, model_dim=14)
    assert changed == []


def test_the_shipped_script_reproduces_the_same_bounds(tmp_path: Path):
    """Integration: run scripts/data/norm_convention.py as the pipeline does."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "data" / "norm_convention.py"
    src = tmp_path / "stats.json"
    dst = tmp_path / "stats_z.json"
    payload = {"norm_stats": {"observation.state": _block(), "action": _block()}}
    src.write_text(json.dumps(payload))

    subprocess.run(
        [sys.executable, str(script), "--src", str(src), "--out", str(dst)],
        check=True,
        capture_output=True,
    )

    out = json.loads(dst.read_text())["norm_stats"]["action"]
    assert np.allclose(out["q01"], MEAN - STD)
    assert np.allclose(out["q99"], MEAN + STD)
    # Untouched fields must stay untouched.
    assert np.allclose(out["mean"], MEAN)
    assert np.allclose(out["std"], STD)
