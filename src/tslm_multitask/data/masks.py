"""Mask utilities shared by training, inference, and the desktop bridge."""

from __future__ import annotations

import numpy as np


def observed_mask_from_values(values: np.ndarray) -> np.ndarray:
    """Return True for real observations; numeric zero remains a valid value."""
    return ~np.isnan(np.asarray(values, dtype=np.float32))


def missing_indices_from_mask(observed_mask: np.ndarray) -> np.ndarray:
    """Return integer indices where observations are missing."""
    return np.where(~np.asarray(observed_mask, dtype=bool))[0]
