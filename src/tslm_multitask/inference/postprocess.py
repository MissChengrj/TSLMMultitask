"""Post-processing helpers for reconstruction, imputation, and anomaly scores."""

from __future__ import annotations

import numpy as np


def prediction_to_numpy(predictions) -> np.ndarray:
    """Convert Chronos/PyTorch/list predictions to a NumPy array."""
    if isinstance(predictions, list):
        predictions = predictions[0]
    if hasattr(predictions, "detach"):
        predictions = predictions.detach()
    if hasattr(predictions, "cpu"):
        predictions = predictions.cpu()
    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()
    return np.asarray(predictions)


def median_reconstruction(predictions: np.ndarray) -> np.ndarray:
    """Return the middle quantile reconstruction vector from q x time predictions."""
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = np.squeeze(predictions, axis=0)
    if predictions.ndim != 2:
        raise ValueError(f"期望二维分位数预测，实际形状: {predictions.shape}")
    median = predictions[len(predictions) // 2]
    if np.isfinite(median).all():
        return median

    fallback = np.ma.median(np.ma.masked_invalid(predictions), axis=0).filled(0.0)
    fallback = np.nan_to_num(fallback, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.where(np.isfinite(median), median, fallback).astype(np.float32)


def interpolate_missing(
    masked_series: np.ndarray,
    predictions: np.ndarray,
    missing_indices: np.ndarray,
    use_median: bool = True,
) -> np.ndarray:
    """Fill only the provided missing indices from reconstruction predictions."""
    interpolated_series = np.asarray(masked_series, dtype=np.float32).copy()
    predictions = np.asarray(predictions)
    fill_values = median_reconstruction(predictions) if use_median else predictions.mean(axis=0)
    fill_values = np.nan_to_num(fill_values, nan=0.0, posinf=1e6, neginf=-1e6)

    for idx in missing_indices:
        if idx < len(fill_values):
            interpolated_series[idx] = fill_values[idx]

    return interpolated_series


def detect_anomalies(series: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute non-negative reconstruction residual scores and envelope breaches."""
    series = np.asarray(series, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = np.squeeze(predictions, axis=0)
    if predictions.ndim != 2:
        raise ValueError(f"期望二维分位数预测，实际形状: {predictions.shape}")

    median = median_reconstruction(predictions)
    lower_bound = np.nan_to_num(predictions[0], nan=np.nan, posinf=np.nan, neginf=np.nan)
    upper_bound = np.nan_to_num(predictions[-1], nan=np.nan, posinf=np.nan, neginf=np.nan)
    length = min(len(series), len(median), len(lower_bound), len(upper_bound))

    scores = np.full(len(series), np.nan, dtype=np.float32)
    finite_bounds = np.isfinite(lower_bound[:length]) & np.isfinite(upper_bound[:length])
    width = np.where(finite_bounds, upper_bound[:length] - lower_bound[:length], np.nan)
    width = np.where(np.isfinite(width) & (np.abs(width) > 1e-8), np.abs(width), np.nan)
    raw_scores = np.abs(series[:length] - median[:length]) / width
    scores[:length] = raw_scores

    valid = np.isfinite(series[:length]) & np.isfinite(scores[:length]) & finite_bounds
    anomaly_indices = np.where(valid & ((series[:length] < lower_bound[:length]) | (series[:length] > upper_bound[:length])))[0]
    return anomaly_indices, np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
