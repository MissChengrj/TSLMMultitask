# -*- coding: utf-8 -*-
"""Build fixed multitask evaluation datasets from local project CSV files.

The generated JSONL files are model-agnostic fixtures for evaluating:
forecasting, interpolation/imputation, and anomaly detection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TASKS = ("forecast", "interpolation", "anomaly_detection")
SOURCE_DOMAINS = ("acars", "qar")


@dataclass(frozen=True)
class BuildConfig:
    input_dir: str
    output_dir: str
    context_length: int = 128
    prediction_length: int = 16
    max_samples_per_task: int = 120
    min_valid_ratio: float = 0.9
    max_columns_per_sample: int = 4
    interpolation_mask_ratio: float = 0.2
    anomaly_ratio: float = 0.05
    anomaly_sigma: float = 4.0
    seed: int = 42
    source_domain: str = "all"


@dataclass
class SourceSeries:
    file_path: Path
    source_domain: str
    columns: list[str]
    values: np.ndarray
    original_valid_ratio: float


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_float(value: float) -> float | None:
    return None if not math.isfinite(float(value)) else float(value)


def _as_json_list(values: Iterable[float]) -> list[float | None]:
    return [_safe_float(v) for v in values]


def _fill_missing(values: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs so generated labels are complete and reproducible."""
    values = np.asarray(values, dtype=np.float32).copy()
    for row_idx in range(values.shape[0]):
        row = values[row_idx]
        valid = np.isfinite(row)
        if valid.all():
            continue
        if not valid.any():
            row[:] = 0.0
            continue
        x = np.arange(len(row))
        row[~valid] = np.interp(x[~valid], x[valid], row[valid]).astype(np.float32)
    return values


def _read_numeric_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df.apply(pd.to_numeric, errors="coerce")


def infer_source_domain(path: Path) -> str:
    """Infer source domain from project naming conventions."""
    name = path.name.lower()
    if "features" in name:
        return "acars"
    if ".qar" in name:
        return "qar"
    return "unknown"


def _select_columns(df: pd.DataFrame, min_length: int, min_valid_ratio: float) -> list[str]:
    selected: list[str] = []
    for col in df.columns:
        valid_ratio = float(df[col].notna().mean()) if len(df) else 0.0
        if len(df) >= min_length and int(df[col].notna().sum()) >= min_length and valid_ratio >= min_valid_ratio:
            selected.append(str(col))
    return selected


def load_source_series(input_dir: Path, config: BuildConfig) -> list[SourceSeries]:
    min_length = config.context_length + config.prediction_length
    sources: list[SourceSeries] = []

    for path in sorted(input_dir.glob("*.csv")):
        source_domain = infer_source_domain(path)
        if source_domain == "unknown":
            continue
        if config.source_domain != "all" and source_domain != config.source_domain:
            continue

        df = _read_numeric_csv(path)
        columns = _select_columns(df, min_length=min_length, min_valid_ratio=config.min_valid_ratio)
        if not columns:
            continue

        for start in range(0, len(columns), config.max_columns_per_sample):
            group_columns = columns[start : start + config.max_columns_per_sample]
            values = df[group_columns].to_numpy(dtype=np.float32).T
            valid_ratio = float(np.isfinite(values).mean())
            sources.append(
                SourceSeries(
                    file_path=path,
                    source_domain=source_domain,
                    columns=group_columns,
                    values=_fill_missing(values),
                    original_valid_ratio=valid_ratio,
                )
            )

    return sources


def _window_starts(length: int, total_length: int, rng: np.random.Generator, max_windows: int) -> list[int]:
    if length < total_length:
        return []
    possible = np.arange(0, length - total_length + 1)
    if len(possible) <= max_windows:
        return possible.astype(int).tolist()
    return sorted(rng.choice(possible, size=max_windows, replace=False).astype(int).tolist())


def _sample_sources(
    sources: list[SourceSeries],
    config: BuildConfig,
    task: str,
    rng: np.random.Generator,
) -> list[tuple[SourceSeries, int]]:
    total_length = config.context_length + (config.prediction_length if task == "forecast" else 0)
    per_source = max(1, math.ceil(config.max_samples_per_task / max(1, len(sources))))
    candidates: list[tuple[SourceSeries, int]] = []

    for source in sources:
        starts = _window_starts(source.values.shape[1], total_length, rng, per_source)
        candidates.extend((source, start) for start in starts)

    rng.shuffle(candidates)
    return candidates[: config.max_samples_per_task]


def _base_record(task: str, idx: int, source: SourceSeries, start: int, config: BuildConfig) -> dict:
    return {
        "sample_id": f"{source.source_domain}_{task}_{idx:05d}",
        "task_type": task,
        "source_domain": source.source_domain,
        "source_file": source.file_path.name,
        "source_columns": source.columns,
        "start_index": int(start),
        "context_length": config.context_length,
        "original_valid_ratio": round(source.original_valid_ratio, 6),
    }


def build_forecast_records(sources: list[SourceSeries], config: BuildConfig, rng: np.random.Generator) -> list[dict]:
    records: list[dict] = []
    for idx, (source, start) in enumerate(_sample_sources(sources, config, "forecast", rng)):
        context_end = start + config.context_length
        future_end = context_end + config.prediction_length
        history = source.values[:, start:context_end]
        future = source.values[:, context_end:future_end]
        record = _base_record("forecast", idx, source, start, config)
        record.update(
            {
                "prediction_length": config.prediction_length,
                "history": {col: _as_json_list(history[col_idx]) for col_idx, col in enumerate(source.columns)},
                "target_future": {col: _as_json_list(future[col_idx]) for col_idx, col in enumerate(source.columns)},
                "metric_hints": ["mae", "rmse", "smape"],
            }
        )
        records.append(record)
    return records


def build_interpolation_records(sources: list[SourceSeries], config: BuildConfig, rng: np.random.Generator) -> list[dict]:
    records: list[dict] = []
    for idx, (source, start) in enumerate(_sample_sources(sources, config, "interpolation", rng)):
        clean = source.values[:, start : start + config.context_length].copy()
        observed = clean.copy()
        missing_count = max(1, int(round(config.context_length * config.interpolation_mask_ratio)))
        edge_guard = min(8, max(0, config.context_length // 8))
        candidates = np.arange(edge_guard, config.context_length - edge_guard)
        if len(candidates) < missing_count:
            candidates = np.arange(config.context_length)

        missing_indices: dict[str, list[int]] = {}
        targets: dict[str, list[float | None]] = {}
        observed_context: dict[str, list[float | None]] = {}

        for col_idx, col in enumerate(source.columns):
            chosen = sorted(rng.choice(candidates, size=min(missing_count, len(candidates)), replace=False).astype(int).tolist())
            missing_indices[col] = chosen
            targets[col] = _as_json_list(clean[col_idx, chosen])
            observed[col_idx, chosen] = np.nan
            observed_context[col] = _as_json_list(observed[col_idx])

        record = _base_record("interpolation", idx, source, start, config)
        record.update(
            {
                "observed_context": observed_context,
                "missing_indices": missing_indices,
                "target_values": targets,
                "mask_ratio": config.interpolation_mask_ratio,
                "metric_hints": ["mae_on_missing", "rmse_on_missing"],
            }
        )
        records.append(record)
    return records


def build_anomaly_records(sources: list[SourceSeries], config: BuildConfig, rng: np.random.Generator) -> list[dict]:
    records: list[dict] = []
    for idx, (source, start) in enumerate(_sample_sources(sources, config, "anomaly_detection", rng)):
        clean = source.values[:, start : start + config.context_length].copy()
        corrupted = clean.copy()
        anomaly_count = max(1, int(round(config.context_length * config.anomaly_ratio)))
        edge_guard = min(8, max(0, config.context_length // 8))
        candidates = np.arange(edge_guard, config.context_length - edge_guard)
        if len(candidates) < anomaly_count:
            candidates = np.arange(config.context_length)

        labels: dict[str, list[int]] = {}
        anomaly_indices: dict[str, list[int]] = {}
        anomaly_values: dict[str, list[float | None]] = {}

        for col_idx, col in enumerate(source.columns):
            chosen = sorted(rng.choice(candidates, size=min(anomaly_count, len(candidates)), replace=False).astype(int).tolist())
            scale = float(np.nanstd(clean[col_idx]))
            if not math.isfinite(scale) or scale <= 1e-6:
                scale = max(float(np.nanmean(np.abs(clean[col_idx]))), 1.0) * 0.1
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(chosen))
            offsets = signs * config.anomaly_sigma * scale
            corrupted[col_idx, chosen] = corrupted[col_idx, chosen] + offsets.astype(np.float32)
            col_labels = np.zeros(config.context_length, dtype=np.int8)
            col_labels[chosen] = 1
            labels[col] = col_labels.astype(int).tolist()
            anomaly_indices[col] = chosen
            anomaly_values[col] = _as_json_list(corrupted[col_idx, chosen])

        record = _base_record("anomaly_detection", idx, source, start, config)
        record.update(
            {
                "clean_context": {col: _as_json_list(clean[col_idx]) for col_idx, col in enumerate(source.columns)},
                "observed_context": {col: _as_json_list(corrupted[col_idx]) for col_idx, col in enumerate(source.columns)},
                "anomaly_indices": anomaly_indices,
                "anomaly_values": anomaly_values,
                "labels": labels,
                "anomaly_ratio": config.anomaly_ratio,
                "anomaly_sigma": config.anomaly_sigma,
                "metric_hints": ["precision", "recall", "f1", "auroc"],
            }
        )
        records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _write_readme(path: Path, config: BuildConfig, counts: dict[str, int], source_domain: str) -> None:
    domain_label = {
        "acars": "ACARS `*features*.csv` files",
        "qar": "QAR `*.qar.csv` files",
        "all": "all supported source domains",
    }.get(source_domain, source_domain)
    text = f"""# Multitask Evaluation Dataset

Generated from {domain_label} in `{config.input_dir}`.

Files:
- `forecast.jsonl`: history windows and future targets for forecasting.
- `interpolation.jsonl`: windows with synthetic missing values plus held-out labels.
- `anomaly_detection.jsonl`: windows with injected point anomalies plus binary labels.
- `manifest.json`: build configuration, task counts, and source coverage.

Each JSONL row is a self-contained sample with `sample_id`, `task_type`,
`source_file`, `source_columns`, and task-specific inputs/labels.

Task counts:
- forecast: {counts.get("forecast", 0)}
- interpolation: {counts.get("interpolation", 0)}
- anomaly_detection: {counts.get("anomaly_detection", 0)}
"""
    path.write_text(text, encoding="utf-8")


def _build_domain_dataset(
    sources: list[SourceSeries],
    config: BuildConfig,
    output_dir: Path,
    source_domain: str,
    seed_offset: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sources:
        raise ValueError(f"No usable {source_domain} numeric series found in {config.input_dir}")

    rng = np.random.default_rng(config.seed + seed_offset)
    records_by_task = {
        "forecast": build_forecast_records(sources, config, rng),
        "interpolation": build_interpolation_records(sources, config, rng),
        "anomaly_detection": build_anomaly_records(sources, config, rng),
    }

    for task, records in records_by_task.items():
        _write_jsonl(output_dir / f"{task}.jsonl", records)

    counts = {task: len(records) for task, records in records_by_task.items()}
    manifest = {
        "config": asdict(config),
        "source_domain": source_domain,
        "task_counts": counts,
        "source_count": len(sources),
        "source_files": sorted({source.file_path.name for source in sources}),
        "schema_version": 1,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(output_dir / "README.md", config, counts, source_domain)
    return manifest


def build_datasets(config: BuildConfig) -> dict:
    if config.source_domain not in {"all", *SOURCE_DOMAINS}:
        raise ValueError(f"source_domain must be one of all, acars, qar; got {config.source_domain}")

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    sources = load_source_series(input_dir, config)
    if not sources:
        raise ValueError(f"No usable numeric series found in {input_dir} for source_domain={config.source_domain}")

    if config.source_domain != "all":
        return _build_domain_dataset(sources, config, output_dir, config.source_domain)

    manifests: dict[str, dict] = {}
    for idx, domain in enumerate(SOURCE_DOMAINS):
        domain_sources = [source for source in sources if source.source_domain == domain]
        if not domain_sources:
            continue
        manifests[domain] = _build_domain_dataset(
            sources=domain_sources,
            config=config,
            output_dir=output_dir / domain,
            source_domain=domain,
            seed_offset=idx * 100_000,
        )

    if not manifests:
        raise ValueError(f"No usable ACARS or QAR numeric series found in {input_dir}")

    root_manifest = {
        "config": asdict(config),
        "source_domains": list(manifests.keys()),
        "domain_manifests": {domain: str(Path(domain) / "manifest.json") for domain in manifests},
        "task_counts_by_domain": {domain: manifest["task_counts"] for domain, manifest in manifests.items()},
        "source_count_by_domain": {domain: manifest["source_count"] for domain, manifest in manifests.items()},
        "schema_version": 2,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(root_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Multitask Evaluation Datasets\n\n"
        "ACARS and QAR sources are built separately.\n\n"
        "- `acars/`: samples from files whose names contain `features`.\n"
        "- `qar/`: samples from files whose names contain `.qar`.\n"
        "- `manifest.json`: top-level summary pointing to each domain manifest.\n",
        encoding="utf-8",
    )
    return root_manifest


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build multitask evaluation JSONL datasets from project CSV data")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default=str(Path("data") / "multitask_eval"))
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=16)
    parser.add_argument("--max-samples-per-task", type=int, default=120)
    parser.add_argument("--min-valid-ratio", type=float, default=0.9)
    parser.add_argument("--max-columns-per-sample", type=int, default=4)
    parser.add_argument("--interpolation-mask-ratio", type=float, default=0.2)
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--anomaly-sigma", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-domain", choices=["all", "acars", "qar"], default="all")
    args = parser.parse_args()
    return BuildConfig(**vars(args))


def main() -> None:
    manifest = build_datasets(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
