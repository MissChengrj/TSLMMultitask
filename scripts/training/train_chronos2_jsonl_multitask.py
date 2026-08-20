# -*- coding: utf-8 -*-
"""Fine-tune Chronos2 on multitask JSONL splits for one source domain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(os.environ.get("TSLM_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scripts.training.train_multitask_chronos2 import TrainConfig, main as csv_train_main
from scripts.training import train_multitask_chronos2 as base_train


TASKS = ("forecast", "interpolation", "anomaly_detection")


def _finite_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _matrix_from_mapping(mapping: dict, columns: list[str]) -> np.ndarray:
    rows = []
    for col in columns:
        rows.append([_finite_float(x) for x in mapping[col]])
    return np.asarray(rows, dtype=np.float32)


def _series_from_forecast(row: dict) -> torch.Tensor:
    columns = list(row["source_columns"])
    history = _matrix_from_mapping(row["history"], columns)
    future = _matrix_from_mapping(row["target_future"], columns)
    return torch.from_numpy(np.concatenate([history, future], axis=1))


def _series_from_interpolation(row: dict) -> torch.Tensor:
    columns = list(row["source_columns"])
    values = _matrix_from_mapping(row["observed_context"], columns)
    for row_idx, col in enumerate(columns):
        for idx, value in zip(row["missing_indices"].get(col, []), row["target_values"].get(col, [])):
            if 0 <= int(idx) < values.shape[1]:
                values[row_idx, int(idx)] = _finite_float(value)
    return torch.from_numpy(values)


def _series_from_anomaly(row: dict) -> torch.Tensor:
    columns = list(row["source_columns"])
    return torch.from_numpy(_matrix_from_mapping(row["clean_context"], columns))


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_split(config: TrainConfig):
    split_dir = Path(config.data_dir)
    source = getattr(config, "source_domain")
    max_targets = max(1, int(config.max_targets_per_item))

    train_series: list[torch.Tensor] = []
    val_series: list[torch.Tensor] = []
    selected_task = getattr(config, "selected_task", "all")
    selected_tasks = TASKS if selected_task == "all" else (selected_task,)
    stats = {
        "source_domain": source,
        "data_mode": "jsonl_multitask",
        "selected_task": selected_task,
        "max_targets_per_item": max_targets,
        "tasks": {},
        "train_series_count": 0,
        "val_series_count": 0,
        "total_nan_count_preserved": 0,
    }

    converters = {
        "forecast": _series_from_forecast,
        "interpolation": _series_from_interpolation,
        "anomaly_detection": _series_from_anomaly,
    }
    for split_name, bucket in (("train", train_series), ("val", val_series)):
        for task in selected_tasks:
            records = 0
            for row in _read_jsonl(split_dir / source / split_name / f"{task}.jsonl"):
                tensor = converters[task](row)
                if tensor.shape[-1] < config.min_series_length:
                    continue
                stats["total_nan_count_preserved"] += int(torch.isnan(tensor).sum().item())
                for start in range(0, tensor.shape[0], max_targets):
                    chunk = tensor[start : start + max_targets]
                    if chunk.numel() > 0:
                        bucket.append(chunk)
                records += 1
            stats["tasks"].setdefault(task, {})[split_name] = records

    if not train_series:
        raise ValueError(f"Training split is empty for source={source}: {split_dir}")
    if not val_series:
        raise ValueError(f"Validation split is empty for source={source}: {split_dir}")

    stats["train_series_count"] = len(train_series)
    stats["val_series_count"] = len(val_series)
    stats["total_feature_count"] = int(sum(x.shape[0] for x in train_series + val_series))
    return train_series, val_series, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronos2 multitask JSONL fine-tuning")
    parser.add_argument("--source", choices=["acars", "qar"], required=True)
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--split-dir", default="data/multitask_eval_splits")
    parser.add_argument("--model-path", default="weights/chronos-2")
    parser.add_argument("--output-dir")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--max-targets-per-item", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        choices=["output_head", "feed_forward_and_output", "attention_and_output"],
        default="output_head",
    )
    parser.add_argument("--normalized-clip-value", type=float, default=100.0)
    parser.add_argument("--no-sanitize-nonfinite-grads", action="store_true")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--forecast-loss-weight", type=float, default=1.0)
    parser.add_argument("--recon-loss-weight", type=float, default=0.3)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    config.source_domain = args.source
    config.selected_task = args.task
    config.data_dir = str((PROJECT_ROOT / args.split_dir).resolve())
    config.model_path = str((PROJECT_ROOT / args.model_path).resolve())
    task_suffix = "multitask" if args.task == "all" else args.task
    default_out = PROJECT_ROOT / "weights" / f"chronos2_jsonl_{args.source}_{task_suffix}"
    config.output_dir = str((PROJECT_ROOT / args.output_dir).resolve()) if args.output_dir else str(default_out)
    config.context_length = args.context_length
    config.prediction_length = args.prediction_length
    config.max_steps = args.max_steps
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_targets_per_item = args.max_targets_per_item
    config.finetune_mode = "lora"
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.lora_target_modules = args.lora_target_modules
    config.normalized_clip_value = args.normalized_clip_value
    config.sanitize_nonfinite_grads = not args.no_sanitize_nonfinite_grads
    config.mask_ratio = args.mask_ratio
    config.forecast_loss_weight = args.forecast_loss_weight
    config.recon_loss_weight = args.recon_loss_weight
    if args.task == "forecast":
        config.mask_ratio = 0.0
        config.recon_loss_weight = 0.0
    elif args.task in ("interpolation", "anomaly_detection"):
        config.forecast_loss_weight = 0.0
    config.use_cpu = args.use_cpu
    config.skip_final_eval = args.skip_final_eval
    config.seed = args.seed
    config.min_series_length = args.prediction_length + 16

    original_loader = base_train.load_and_split_data
    original_argv = sys.argv
    try:
        base_train.load_and_split_data = load_jsonl_split
        sys.argv = [sys.argv[0]]
        base_train.load_config_from_args = lambda: config
        csv_train_main()
    finally:
        base_train.load_and_split_data = original_loader
        sys.argv = original_argv


if __name__ == "__main__":
    main()
