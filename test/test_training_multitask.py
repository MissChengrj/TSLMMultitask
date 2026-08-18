from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "train_multitask_chronos2.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_multitask_chronos2", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_multivariate_loader_keeps_file_as_target_group(tmp_path):
    module = _load_training_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "machine_a_features.csv").write_text(
        "s1,s2,short\n"
        "1.0,10.0,\n"
        "2.0,,\n"
        "3.0,30.0,3.0\n"
        "4.0,40.0,\n",
        encoding="utf-8",
    )
    (data_dir / "machine_b_features.csv").write_text(
        "s1,s2\n"
        "5.0,50.0\n"
        "6.0,60.0\n"
        "7.0,\n"
        "8.0,80.0\n",
        encoding="utf-8",
    )

    config = module.TrainConfig()
    config.data_dir = str(data_dir)
    config.data_mode = "multivariate"
    config.min_series_length = 3
    config.val_ratio = 0.5
    config.seed = 7

    train_series, val_series, stats = module.load_and_split_data(config)
    all_series = train_series + val_series

    assert stats["data_mode"] == "multivariate"
    assert stats["train_series_count"] == 1
    assert stats["val_series_count"] == 1
    assert all(tensor.ndim == 2 for tensor in all_series)
    assert {tuple(tensor.shape) for tensor in all_series} == {(2, 4)}
    assert any(torch.isnan(tensor).any() for tensor in all_series)


def test_single_column_loader_keeps_legacy_series_shape(tmp_path):
    module = _load_training_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "machine_a_features.csv").write_text(
        "s1,s2\n"
        "1.0,10.0\n"
        "2.0,20.0\n"
        "3.0,30.0\n",
        encoding="utf-8",
    )
    (data_dir / "machine_b_features.csv").write_text(
        "s1,s2\n"
        "5.0,50.0\n"
        "6.0,60.0\n"
        "7.0,70.0\n",
        encoding="utf-8",
    )

    config = module.TrainConfig()
    config.data_dir = str(data_dir)
    config.data_mode = "single_column"
    config.min_series_length = 3
    config.val_ratio = 0.5
    config.seed = 7

    train_series, val_series, stats = module.load_and_split_data(config)

    assert stats["data_mode"] == "single_column"
    assert stats["train_series_count"] == 2
    assert stats["val_series_count"] == 2
    assert all(tensor.ndim == 1 for tensor in train_series + val_series)


def test_multivariate_loader_splits_large_target_groups(tmp_path):
    module = _load_training_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    header = ",".join(f"s{i}" for i in range(5))
    rows = ["1,2,3,4,5", "2,3,4,5,6", "3,4,5,6,7"]
    (data_dir / "machine_a_features.csv").write_text(header + "\n" + "\n".join(rows), encoding="utf-8")
    (data_dir / "machine_b_features.csv").write_text(header + "\n" + "\n".join(rows), encoding="utf-8")

    config = module.TrainConfig()
    config.data_dir = str(data_dir)
    config.data_mode = "multivariate"
    config.max_targets_per_item = 2
    config.min_series_length = 3
    config.val_ratio = 0.5
    config.seed = 7

    train_series, val_series, stats = module.load_and_split_data(config)

    assert stats["total_feature_count"] == 10
    assert stats["train_series_count"] == 3
    assert stats["val_series_count"] == 3
    assert all(tensor.ndim == 2 for tensor in train_series + val_series)
    assert max(tensor.shape[0] for tensor in train_series + val_series) == 2
