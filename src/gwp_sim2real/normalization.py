"""Normalization used by the two GWP runtimes, and how they are reconciled.

The official PyTorch path normalizes actions with q01/q99 min-max::

    2 * (x - q01) / (q99 - q01) - 1

wam.cpp's GWP05 converter requires unclipped z-score for RoboTwin::

    (x - mean) / std

These are different affine maps, so the same observation produces different
model inputs. Rather than changing either implementation, the *statistics file*
handed to the converter is rewritten: with ``q01' = mean - std`` and
``q99' = mean + std`` the min-max map becomes::

    2 * (x - (mean - std)) / (2 * std) - 1  ==  (x - mean) / std

and the inverse (denormalization) coincides as well. Weights never change.

The important subtlety, and the reason this module exists instead of a one-line
helper: ``mean`` and ``std`` must be the *measured* statistics. Deriving them
from q01/q99 makes the transformation an identity and the "reconciliation"
vacuous — which is exactly what an earlier version of this code did.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_minmax",
    "denormalize_minmax",
    "normalize_zscore",
    "denormalize_zscore",
    "zscore_equivalent_stats",
    "convert_block",
    "floor_zero_std",
]


def normalize_minmax(x, q01, q99) -> np.ndarray:
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    return 2.0 * (np.asarray(x, dtype=np.float64) - q01) / (q99 - q01) - 1.0


def denormalize_minmax(a, q01, q99) -> np.ndarray:
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    return (np.asarray(a, dtype=np.float64) + 1.0) / 2.0 * (q99 - q01) + q01


def normalize_zscore(x, mean, std) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / np.asarray(
        std, dtype=np.float64
    )


def denormalize_zscore(a, mean, std) -> np.ndarray:
    return (
        np.asarray(a, dtype=np.float64) * np.asarray(std, dtype=np.float64)
        + np.asarray(mean, dtype=np.float64)
    )


def zscore_equivalent_stats(mean, std):
    """Bounds whose min-max normalization equals z-score with `mean`/`std`."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return mean - std, mean + std


def convert_block(block: dict) -> dict:
    """Return a copy of one `norm_stats` domain block with z-score-equal bounds.

    `mean` and `std` are read, never derived: they are the measured statistics.
    `min`/`max` are left untouched, since neither runtime uses them.
    """
    mean = np.asarray(block["mean"], dtype=np.float64)
    std = np.asarray(block["std"], dtype=np.float64)
    q01, q99 = zscore_equivalent_stats(mean, std)
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in block.items()}
    out["q01"] = q01.tolist()
    out["q99"] = q99.tolist()
    return out


def floor_zero_std(block: dict, model_dim: int | None = None, eps: float = 1e-6):
    """Replace degenerate dims with mean=0, std=1, bounds ±1.

    Needed because `wam-inspect` rejects ``std < eps``. A padded dim normalized
    by PyTorch is literally 0; with mean=0/std=1 the C++ z-score of that dim is
    also 0, so the padding is preserved exactly rather than approximately.

    Returns `(block, changed_dims)`, where `changed_dims` lists indices inside
    `model_dim` that were degenerate (dims beyond `model_dim` are padding by
    construction and are not reported).
    """
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in block.items()}
    std = list(out["std"])
    changed: list[int] = []
    for i, s in enumerate(std):
        if abs(s) > eps:
            continue
        if model_dim is not None and i >= model_dim:
            pass  # padding beyond the model width; floored but not reported
        else:
            changed.append(i)
        for key, value in (
            ("mean", 0.0),
            ("std", 1.0),
            ("min", -1.0),
            ("max", 1.0),
            ("q01", -1.0),
            ("q99", 1.0),
        ):
            if key in out and i < len(out[key]):
                out[key][i] = value
    return out, changed
