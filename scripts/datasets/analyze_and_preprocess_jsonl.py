# -*- coding: utf-8 -*-
"""Analyze signal characteristics and build a leakage-safe de-spiked JSONL split."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("forecast", "interpolation", "anomaly_detection")
SPLITS = ("train", "val", "test")
QAR_VALID_RANGES = {
    "ALT": (-2000.0, 60000.0),
    "CAS": (0.0, 500.0),
    "GS": (0.0, 700.0),
    "FF1": (0.0, 6000.0),
    "FF2": (0.0, 6000.0),
    "EGT1": (-100.0, 1200.0),
    "EGT2": (-100.0, 1200.0),
    "N21": (0.0, 120.0),
    "N22": (0.0, 120.0),
    "MACH": (0.0, 1.5),
}


def finite_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def as_array(values) -> np.ndarray:
    return np.asarray([finite_float(value) for value in values], dtype=np.float64)


def json_values(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def fill_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    good = np.isfinite(values)
    if not good.any():
        return np.zeros_like(values)
    indices = np.arange(len(values))
    values[~good] = np.interp(indices[~good], indices[good], values[good])
    return values


def median3_despike(values: np.ndarray, threshold: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated impulses while retaining steps that persist for at least two samples."""
    original = np.asarray(values, dtype=np.float64)
    filled = fill_linear(original)
    if len(filled) < 3:
        return original.copy(), np.zeros(len(original), dtype=bool)
    local_median = np.median(np.stack([np.r_[filled[0], filled[:-1]], filled, np.r_[filled[1:], filled[-1]]]), axis=0)
    differences = np.diff(filled)
    center = float(np.median(differences)) if len(differences) else 0.0
    diff_mad = 1.4826 * float(np.median(np.abs(differences - center))) if len(differences) else 0.0
    nonzero = np.abs(differences[np.abs(differences) > 1e-12])
    quantization = float(np.median(nonzero)) if len(nonzero) else 0.0
    scale = max(diff_mad, quantization, 1e-6)
    isolated = np.abs(filled - local_median) > threshold * scale
    isolated[[0, -1]] = False
    result = filled.copy()
    result[isolated] = local_median[isolated]
    result[~np.isfinite(original)] = np.nan
    return result, isolated


def unwrap_period(values: np.ndarray, period: float) -> np.ndarray:
    original = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(original)
    if not finite.any():
        return original.copy()
    filled = fill_linear(original)
    unwrapped = np.unwrap(filled * (2.0 * np.pi / period), discont=np.pi) * (period / (2.0 * np.pi))
    unwrapped[~finite] = np.nan
    return unwrapped


def preprocess_channel(values: np.ndarray, source: str, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct known QAR encoding artifacts, then remove only isolated impulses."""
    values = np.asarray(values, dtype=np.float64).copy()
    invalid = np.zeros(len(values), dtype=bool)
    if source == "qar":
        if column == "ALT":
            values = unwrap_period(values, 65536.0)
            finite = values[np.isfinite(values)]
            if len(finite):
                center = float(np.median(finite))
                values += 65536.0 * math.ceil((-2000.0 - center) / 65536.0) if center < -2000.0 else 0.0
                finite = values[np.isfinite(values)]
                center = float(np.median(finite))
                values -= 65536.0 * math.ceil((center - 60000.0) / 65536.0) if center > 60000.0 else 0.0
        elif column.endswith("IMB_ANG_L") or column.endswith("IMB_ANG_R"):
            values = unwrap_period(values, 360.0)
        bounds = QAR_VALID_RANGES.get(column)
        if bounds is not None:
            invalid = np.isfinite(values) & ((values < bounds[0]) | (values > bounds[1]))
            values[invalid] = np.nan
    processed, spikes = median3_despike(values)
    return processed, invalid, spikes


def robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 1.0
    q10, q90 = np.quantile(finite, [0.1, 0.9])
    return max(float(q90 - q10), float(np.std(finite)) * 0.1, 1e-6)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def channel_stats(history: np.ndarray, future: np.ndarray | None = None) -> dict[str, float]:
    history_fill = fill_linear(history)
    smoothed, spikes = median3_despike(history)
    smoothed_fill = fill_linear(smoothed)
    differences = np.diff(history_fill)
    nonzero = np.abs(differences[np.abs(differences) > 1e-12])
    diff_center = float(np.median(differences)) if len(differences) else 0.0
    diff_mad = 1.4826 * float(np.median(np.abs(differences - diff_center))) if len(differences) else 0.0
    result = {
        "points": float(np.isfinite(history).sum()),
        "missing_rate": float(1.0 - np.isfinite(history).mean()),
        "abs_level_median": float(np.median(np.abs(history_fill))),
        "value_iqr": float(np.subtract(*np.quantile(history_fill, [0.75, 0.25]))),
        "robust_range_p90_p10": robust_scale(history_fill),
        "std": float(np.std(history_fill)),
        "quantization_step": float(np.median(nonzero)) if len(nonzero) else 0.0,
        "median_abs_diff": float(np.median(np.abs(differences))) if len(differences) else 0.0,
        "diff_mad": diff_mad,
        "lag1_corr": safe_corr(history_fill[:-1], history_fill[1:]),
        "isolated_spike_rate": float(spikes.mean()),
        "despike_delta_mae": float(np.mean(np.abs(history_fill - smoothed_fill))),
    }
    if future is not None:
        future_fill = fill_linear(future)
        threshold = max(6.0 * diff_mad, 3.0 * result["quantization_step"], 1e-6)
        future_diffs = np.diff(np.r_[history_fill[-1], future_fill])
        persistence_mae = float(np.mean(np.abs(future_fill - history_fill[-1])))
        future_smooth = fill_linear(median3_despike(future)[0])
        result.update(
            {
                "persistence_mae": persistence_mae,
                "persistence_nmae": persistence_mae / robust_scale(history_fill),
                "future_jump_rate": float(np.mean(np.abs(future_diffs) > threshold)),
                "boundary_jump_abs": float(abs(future_fill[0] - history_fill[-1])),
                "future_despike_delta_mae": float(np.mean(np.abs(future_fill - future_smooth))),
            }
        )
    return result


def iter_jsonl(path: Path, limit: int | None = None):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def complete_interpolation(row: dict, column: str) -> np.ndarray:
    values = as_array(row["observed_context"][column])
    for index, target in zip(row["missing_indices"].get(column, []), row["target_values"].get(column, [])):
        values[int(index)] = finite_float(target)
    return values


def analyze(input_dir: Path, output_dir: Path, max_records: int | None) -> pd.DataFrame:
    rows = []
    baseline_errors: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"absolute": [], "normalized": []}
    )
    for source in ("acars", "qar"):
        for split_name in SPLITS:
            for task in TASKS:
                path = input_dir / source / split_name / f"{task}.jsonl"
                for record in iter_jsonl(path, max_records):
                    for column in record["source_columns"]:
                        if task == "forecast":
                            history = as_array(record["history"][column])
                            future = as_array(record["target_future"][column])
                            stats = channel_stats(history, future)
                            valid_targets = np.isfinite(future)
                            errors = np.abs(future[valid_targets] - fill_linear(history)[-1])
                            scale = robust_scale(fill_linear(history))
                            accumulator = baseline_errors[(source, split_name, task, column)]
                            accumulator["absolute"].extend(errors.tolist())
                            accumulator["normalized"].extend((errors / scale).tolist())
                        elif task == "interpolation":
                            complete = complete_interpolation(record, column)
                            stats = channel_stats(complete)
                            observed = as_array(record["observed_context"][column])
                            prediction = fill_linear(observed)
                            indices = np.asarray(record["missing_indices"].get(column, []), dtype=int)
                            valid_targets = np.isfinite(complete[indices])
                            errors = np.abs(prediction[indices][valid_targets] - complete[indices][valid_targets])
                            scale = robust_scale(complete)
                            accumulator = baseline_errors[(source, split_name, task, column)]
                            accumulator["absolute"].extend(errors.tolist())
                            accumulator["normalized"].extend((errors / scale).tolist())
                        else:
                            stats = channel_stats(as_array(record["clean_context"][column]))
                        rows.append({"source": source, "split": split_name, "task": task, "channel": column, **stats})
    frame = pd.DataFrame(rows)
    metric_columns = [column for column in frame.columns if column not in {"source", "split", "task", "channel"}]
    summary = frame.groupby(["source", "split", "task", "channel"], as_index=False)[metric_columns].median()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "channel_characteristics.csv", index=False)

    error_rows = []
    for (source, split_name, task, channel), values in baseline_errors.items():
        absolute = np.asarray(values["absolute"], dtype=np.float64)
        normalized = np.asarray(values["normalized"], dtype=np.float64)
        if len(absolute) == 0:
            continue
        error_rows.append(
            {
                "source": source,
                "split": split_name,
                "task": task,
                "channel": channel,
                "points": len(absolute),
                "mae": float(np.mean(absolute)),
                "p90_abs_error": float(np.quantile(absolute, 0.90)),
                "p99_abs_error": float(np.quantile(absolute, 0.99)),
                "max_abs_error": float(np.max(absolute)),
                "mean_normalized_abs_error": float(np.mean(normalized)),
                "p90_normalized_abs_error": float(np.quantile(normalized, 0.90)),
                "absolute_error_sum": float(np.sum(absolute)),
            }
        )
    error_frame = pd.DataFrame(error_rows)
    error_frame["source_error_share"] = error_frame.groupby(
        ["source", "split", "task"]
    )["absolute_error_sum"].transform(lambda values: values / max(float(values.sum()), 1e-12))
    error_frame.to_csv(output_dir / "baseline_error_by_channel.csv", index=False)

    forecast = summary[(summary["split"] == "test") & (summary["task"] == "forecast")]
    source_summary = forecast.groupby("source").agg(
        channels=("channel", "nunique"),
        median_level=("abs_level_median", "median"),
        median_robust_range=("robust_range_p90_p10", "median"),
        median_lag1_corr=("lag1_corr", "median"),
        median_spike_rate=("isolated_spike_rate", "median"),
        median_future_jump_rate=("future_jump_rate", "median"),
        median_persistence_mae=("persistence_mae", "median"),
        median_persistence_nmae=("persistence_nmae", "median"),
        median_noise_floor=("future_despike_delta_mae", "median"),
    ).reset_index()
    source_summary.to_csv(output_dir / "source_characteristics.csv", index=False)
    print(source_summary.to_string(index=False))
    return summary


def preprocess_record(row: dict) -> tuple[dict, int, int, int]:
    changed, invalid_points, points = 0, 0, 0
    task = row["task_type"]
    source = row["source_domain"]
    columns = row["source_columns"]
    if task == "forecast":
        for column in columns:
            history = as_array(row["history"][column])
            future = as_array(row["target_future"][column])
            combined, invalid, spikes = preprocess_channel(np.r_[history, future], source, column)
            row["history"][column] = json_values(combined[: len(history)])
            row["target_future"][column] = json_values(combined[len(history) :])
            changed += int(spikes.sum())
            invalid_points += int(invalid.sum())
            points += len(combined)
    elif task == "interpolation":
        for column in columns:
            complete = complete_interpolation(row, column)
            processed, invalid, spikes = preprocess_channel(complete, source, column)
            missing = {int(index) for index in row["missing_indices"].get(column, [])}
            row["observed_context"][column] = [
                None if index in missing or not np.isfinite(processed[index]) else float(processed[index])
                for index in range(len(processed))
            ]
            row["target_values"][column] = [
                float(processed[int(index)]) if np.isfinite(processed[int(index)]) else None
                for index in row["missing_indices"].get(column, [])
            ]
            changed += int(spikes.sum())
            invalid_points += int(invalid.sum())
            points += len(processed)
    else:
        for column in columns:
            clean = as_array(row["clean_context"][column])
            original_observed = as_array(row["observed_context"][column])
            processed, invalid, spikes = preprocess_channel(clean, source, column)
            observed = processed.copy()
            for index in row["anomaly_indices"].get(column, []):
                index = int(index)
                delta = original_observed[index] - clean[index]
                if np.isfinite(processed[index]) and np.isfinite(delta):
                    observed[index] = processed[index] + delta
            row["clean_context"][column] = json_values(processed)
            row["observed_context"][column] = json_values(observed)
            row["anomaly_values"][column] = [
                float(observed[int(index)]) if np.isfinite(observed[int(index)]) else None
                for index in row["anomaly_indices"].get(column, [])
            ]
            changed += int(spikes.sum())
            invalid_points += int(invalid.sum())
            points += len(processed)
    row["preprocessing"] = {
        "method": "qar_encoding_correction_and_median3_isolated_despike",
        "despike_threshold": 4.0,
        "altitude_period": 65536.0,
        "angle_period": 360.0,
    }
    return row, changed, invalid_points, points


def build_preprocessed(input_dir: Path, output_dir: Path) -> None:
    summary = defaultdict(lambda: {"records": 0, "changed_points": 0, "points": 0})
    for source in ("acars", "qar"):
        for split_name in SPLITS:
            for task in TASKS:
                input_path = input_dir / source / split_name / f"{task}.jsonl"
                output_path = output_dir / source / split_name / f"{task}.jsonl"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                    for row in iter_jsonl(input_path):
                        row, changed, invalid, points = preprocess_record(row)
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        key = f"{source}/{split_name}/{task}"
                        summary[key]["records"] += 1
                        summary[key]["changed_points"] += changed
                        summary[key].setdefault("invalid_points", 0)
                        summary[key]["invalid_points"] += invalid
                        summary[key]["points"] += points
    serializable = {}
    for key, values in summary.items():
        serializable[key] = {
            **values,
            "changed_rate": values["changed_points"] / max(1, values["points"]),
            "invalid_rate": values.get("invalid_points", 0) / max(1, values["points"]),
        }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/multitask_eval_splits")
    parser.add_argument("--analysis-dir", default="results/data_characteristics")
    parser.add_argument("--output-split-dir")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not args.skip_analysis:
        analyze(input_dir, Path(args.analysis_dir), args.max_records)
    if args.output_split_dir:
        build_preprocessed(input_dir, Path(args.output_split_dir))


if __name__ == "__main__":
    main()
