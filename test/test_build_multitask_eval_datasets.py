from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_SCRIPT = PROJECT_ROOT / "scripts" / "datasets" / "build_multitask_eval_datasets.py"


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("build_multitask_eval_datasets", BUILDER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_multitask_eval_datasets_outputs_three_task_files(tmp_path):
    module = _load_builder_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "eval"
    data_dir.mkdir()
    rows = ["sensor_a,sensor_b,sparse"]
    for idx in range(80):
        sparse = "" if idx % 10 == 0 else f"{idx * 0.5:.1f}"
        rows.append(f"{idx:.1f},{idx * 2.0:.1f},{sparse}")
    (data_dir / "flight_001_features.csv").write_text("\n".join(rows), encoding="utf-8")
    (data_dir / "flight_001.qar.csv").write_text("\n".join(rows), encoding="utf-8")

    config = module.BuildConfig(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        context_length=16,
        prediction_length=4,
        max_samples_per_task=3,
        min_valid_ratio=0.8,
        max_columns_per_sample=2,
        interpolation_mask_ratio=0.25,
        anomaly_ratio=0.125,
        seed=7,
    )

    manifest = module.build_datasets(config)

    assert manifest["source_domains"] == ["acars", "qar"]
    assert manifest["task_counts_by_domain"]["acars"] == {
        "forecast": 3,
        "interpolation": 3,
        "anomaly_detection": 3,
    }
    assert manifest["task_counts_by_domain"]["qar"] == {
        "forecast": 3,
        "interpolation": 3,
        "anomaly_detection": 3,
    }
    assert (output_dir / "README.md").exists()

    forecast = _read_jsonl(output_dir / "acars" / "forecast.jsonl")[0]
    forecast_col = forecast["source_columns"][0]
    assert forecast["task_type"] == "forecast"
    assert forecast["source_domain"] == "acars"
    assert forecast["prediction_length"] == 4
    assert len(forecast["history"][forecast_col]) == 16
    assert len(forecast["target_future"][forecast_col]) == 4

    interpolation = _read_jsonl(output_dir / "acars" / "interpolation.jsonl")[0]
    interpolation_col = interpolation["source_columns"][0]
    assert interpolation["task_type"] == "interpolation"
    assert len(interpolation["observed_context"][interpolation_col]) == 16
    assert len(interpolation["missing_indices"][interpolation_col]) == 4
    assert len(interpolation["target_values"][interpolation_col]) == 4
    assert any(value is None for value in interpolation["observed_context"][interpolation_col])

    anomaly = _read_jsonl(output_dir / "qar" / "anomaly_detection.jsonl")[0]
    anomaly_col = anomaly["source_columns"][0]
    assert anomaly["task_type"] == "anomaly_detection"
    assert anomaly["source_domain"] == "qar"
    assert len(anomaly["observed_context"][anomaly_col]) == 16
    assert len(anomaly["clean_context"][anomaly_col]) == 16
    assert len(anomaly["anomaly_indices"][anomaly_col]) == 2
    assert sum(anomaly["labels"][anomaly_col]) == 2
