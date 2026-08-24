from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_SCRIPT = PROJECT_ROOT / "scripts" / "datasets" / "build_multitask_eval_datasets.py"
MERGE_SCRIPT = PROJECT_ROOT / "scripts" / "datasets" / "merge_acars_sources.py"


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("build_multitask_eval_datasets", BUILDER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_acars_sources", MERGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_multitask_eval_datasets_outputs_three_task_files(tmp_path):
    module = _load_builder_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "eval"
    data_dir.mkdir()
    rows = ["DEGT-CRUISE,DEGT_SMOOTHED-CRUISE"]
    for idx in range(80):
        rows.append(f"{idx:.1f},{idx * 2.0:.1f}")
    (data_dir / "flight_001_features.csv").write_text("\n".join(rows), encoding="utf-8")
    qar_rows = ["FF1,FF2"] + [f"{idx:.1f},{idx * 2.0:.1f}" for idx in range(80)]
    (data_dir / "flight_001.qar.csv").write_text("\n".join(qar_rows), encoding="utf-8")

    config = module.BuildConfig(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        context_length=16,
        prediction_length=4,
        windows_per_source=1,
        min_valid_points=8,
        interpolation_mask_ratio=0.25,
        anomaly_ratio=0.125,
        seed=7,
    )

    manifest = module.build_datasets(config)

    assert manifest["source_domains"] == ["acars", "qar"]
    assert manifest["task_counts_by_domain"]["acars"] == {"forecast": 1, "interpolation": 1, "anomaly_detection": 1}
    assert manifest["task_counts_by_domain"]["qar"] == {"forecast": 1, "interpolation": 1, "anomaly_detection": 1}
    assert (output_dir / "README.md").exists()
    assert (output_dir / "channel_schema.yaml").exists()
    assert (output_dir / "source_manifest.jsonl").exists()
    assert (output_dir / "canonical" / "source_manifest.jsonl").exists()
    assert (output_dir / "train_scalers.json").exists()
    assert (data_dir / "source_catalog.yaml").exists()

    forecast = _read_jsonl(output_dir / "tasks" / "acars" / "forecast.jsonl")[0]
    forecast_col = forecast["source_columns"][0]
    assert forecast["task_type"] == "forecast"
    assert forecast["source_domain"] == "acars"
    assert forecast["split"] in {"train", "val", "test"}
    assert forecast["prediction_length"] == 4
    assert len(forecast["history"][forecast_col]) == 16
    assert len(forecast["target_future"][forecast_col]) == 4
    assert len(forecast["target_observation_mask"][forecast_col]) == 4
    assert len(forecast["history_quality_mask"][forecast_col]) == 16
    assert forecast["sampling_interval_unit"] == "flight_cycle"
    assert forecast["sampling_interval"][forecast_col] == 1.0
    assert len(forecast["history_native_sampling_mask"][forecast_col]) == 16
    assert len(forecast["history_time_delta"][forecast_col]) == 16

    interpolation = _read_jsonl(output_dir / "tasks" / "acars" / "interpolation.jsonl")[0]
    interpolation_col = interpolation["source_columns"][0]
    assert interpolation["task_type"] == "interpolation"
    assert len(interpolation["observed_context"][interpolation_col]) == 16
    assert len(interpolation["missing_indices"][interpolation_col]) == 4
    assert len(interpolation["target_values"][interpolation_col]) == 4
    assert any(value is None for value in interpolation["observed_context"][interpolation_col])
    assert "task_mask" in interpolation

    anomaly = _read_jsonl(output_dir / "tasks" / "qar" / "anomaly_detection.jsonl")[0]
    anomaly_col = anomaly["source_columns"][0]
    assert anomaly["task_type"] == "anomaly_detection"
    assert anomaly["source_domain"] == "qar"
    assert len(anomaly["observed_context"][anomaly_col]) == 16
    assert len(anomaly["clean_context"][anomaly_col]) == 16
    assert len(anomaly["anomaly_indices"][anomaly_col]) == 2
    assert sum(anomaly["labels"][anomaly_col]) == 2
    assert "quality_mask" in anomaly
    assert anomaly["sampling_interval_unit"] == "second"
    assert len(anomaly["native_sampling_mask"][anomaly_col]) == 16
    assert len(anomaly["time_delta"][anomaly_col]) == 16


def test_alternative_interpolation_and_anomaly_modes(tmp_path):
    module = _load_builder_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "eval"
    data_dir.mkdir()
    rows = ["FF1,FF2"] + [f"{idx:.1f},{idx * 2.0:.1f}" for idx in range(80)]
    (data_dir / "B-1400_20260101000000.qar.csv").write_text("\n".join(rows), encoding="utf-8")

    config = module.BuildConfig(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        context_length=16,
        prediction_length=4,
        windows_per_source=1,
        interpolation_mode="block",
        anomaly_mode="cross_channel",
        seed=9,
    )
    module.build_datasets(config)

    interpolation = _read_jsonl(output_dir / "tasks" / "qar" / "interpolation.jsonl")[0]
    anomaly = _read_jsonl(output_dir / "tasks" / "qar" / "anomaly_detection.jsonl")[0]
    assert interpolation["interpolation_mode"] == "block"
    assert anomaly["anomaly_mode"] == "cross_channel"
    assert sum(anomaly["labels"]["FF1"]) > 0
    assert sum(anomaly["labels"]["FF2"]) == 0


def test_latest_advice_channel_schema_mappings():
    module = _load_builder_module()

    oil_pressure = module.channel_metadata("qar", "OIL_PRS1_R")
    assert oil_pressure["group_id"] == "G4"
    assert oil_pressure["physical_variable"] == "OIL_PRESSURE"
    assert oil_pressure["engine_id"] == 1

    vibration = module.channel_metadata("qar", "CN2_(HPC)_VIB_L")
    assert vibration["group_id"] == "G5"
    assert vibration["component"] == "HPC"
    assert vibration["engine_id"] == "global"
    assert vibration["relation_group_id"] == "HPC_VIBRATION_PAIR"

    duct_pressure = module.channel_metadata("qar", "DUCTPRS1C")
    assert duct_pressure["group_id"] == "G6"
    assert duct_pressure["role"] == "auxiliary_target"
    assert duct_pressure["physical_variable"] == "DUCT_PRESSURE"
    assert duct_pressure["engine_id"] == 1
    assert duct_pressure["relation_group_ids"] == ["DUCT_PRESSURE_ENGINE_1"]

    n1_command = module.channel_metadata("qar", "N1_#1_CMD_INDICATED")
    n1_actual = module.channel_metadata("qar", "SELECTED_N1_INDICATED_#1")
    assert n1_command["physical_variable"] == "N1_COMMAND"
    assert n1_actual["physical_variable"] == "N1"
    assert "N1_COMMAND_RESPONSE_ENGINE_1" in n1_command["relation_group_ids"]
    assert "N1_COMMAND_RESPONSE_ENGINE_1" in n1_actual["relation_group_ids"]

    avm = module.channel_metadata("qar", "AVM_SYSTEM_FAULT")
    assert avm["role"] == "label_or_metadata"
    assert avm["anomaly_input"] is False

    acars_divergence = module.channel_metadata("acars", "DEGT_D_SMOOTHED-CRUISE")
    assert acars_divergence["source_variable"] == "DEGT"
    assert acars_divergence["physical_variable"] == "EGT"
    assert acars_divergence["feature_type"] == "smoothed"
    assert acars_divergence["delta_semantics"] == "oem_deviation"
    assert acars_divergence["transformation"] == "divergence"
    assert acars_divergence["is_delta"] is True
    assert acars_divergence["is_divergence"] is True
    assert acars_divergence["divergence_group_id"] == "EGT_DIVERGENCE"
    assert "EGT_DIVERGENCE" in acars_divergence["relation_group_ids"]

    acars_raw = module.channel_metadata("acars", "DEGT-CRUISE")
    assert acars_raw["physical_variable"] == "EGT"
    assert acars_raw["delta_semantics"] == "oem_deviation"
    assert acars_raw["transformation"] == "none"

    acars_oil_delta = module.channel_metadata("acars", "DPOIL-CRUISE")
    assert acars_oil_delta["physical_variable"] == "OIL_PRESSURE"
    assert acars_oil_delta["delta_semantics"] == "oem_deviation"


def test_sampling_interval_and_native_mask_are_inferred(tmp_path):
    module = _load_builder_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "eval"
    data_dir.mkdir()
    rows = ["FF1,PS3_SEL_1"]
    for idx in range(80):
        rows.append(f"{idx},{idx * 2 if idx % 2 == 0 else ''}")
    (data_dir / "B-1400_20260101000000.qar.csv").write_text("\n".join(rows), encoding="utf-8")

    config = module.BuildConfig(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        context_length=16,
        prediction_length=4,
        windows_per_source=1,
        seed=11,
    )
    module.build_datasets(config)

    manifest = _read_jsonl(output_dir / "source_manifest.jsonl")
    qar_group = next(item for item in manifest if item["source_domain"] == "qar")
    assert qar_group["time_unit"] == "second"
    assert qar_group["base_time_interval"] == 1.0
    assert qar_group["sampling_interval"]["FF1"] == 1.0
    assert qar_group["sampling_interval"]["PS3_SEL_1"] == 2.0
    assert qar_group["native_sampling_ratio"] == 0.75

    forecast = _read_jsonl(output_dir / "tasks" / "qar" / "forecast.jsonl")[0]
    assert forecast["sampling_interval"]["PS3_SEL_1"] == 2.0
    assert forecast["history_native_sampling_mask"]["PS3_SEL_1"][:4] == [1, 0, 1, 0]
    assert forecast["history_time_delta"]["PS3_SEL_1"][:4] == [0.0, 1.0, 0.0, 1.0]


def test_canonical_task_generator_balances_without_jsonl_materialization(tmp_path):
    generator_path = PROJECT_ROOT / "scripts" / "datasets" / "canonical_task_generator.py"
    spec = importlib.util.spec_from_file_location("canonical_task_generator", generator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    acars_rows = ["DEGT-CRUISE"] + [str(idx) for idx in range(80)]
    qar_rows = ["FF1"] + [str(idx) for idx in range(80)]
    (data_dir / "flight_001_features.csv").write_text("\n".join(acars_rows), encoding="utf-8")
    (data_dir / "B-1400_20260101000000.qar.csv").write_text("\n".join(qar_rows), encoding="utf-8")

    builder = _load_builder_module()
    config = builder.BuildConfig(input_dir=str(data_dir), output_dir=str(tmp_path / "eval"), context_length=16, prediction_length=4)
    generator = module.CanonicalTaskGenerator(data_dir, config=config)
    records = list(generator.iter_balanced(4, tasks=("forecast",), split=None, seed=3))
    assert len(records) == 4
    assert {record["sampling_policy"] for record in records} == {"balanced_domain_task"}
    assert {record["source_domain"] for record in records} <= {"acars", "qar"}


def test_merge_acars_triplet_preserves_row_union_and_source_identity(tmp_path):
    module = _load_merge_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "038382_1_features.csv").write_text("DEGT-CRUISE\n1\n2\n", encoding="utf-8")
    (data_dir / "038382_2_features.csv").write_text("ZVB1F-CRUISE\n3\n", encoding="utf-8")
    (data_dir / "038382_3_features.csv").write_text("DEGT-TAKEOFF\n4\n5\n6\n", encoding="utf-8")
    (data_dir / "B-1400_20260130044340.qar.csv").write_text("FF1\n1\n", encoding="utf-8")

    result = module.merge_acars_sources(data_dir)

    assert result["engine_count"] == 1
    merged = pd.read_csv(data_dir / "038382_1_acars.csv")
    assert len(merged) == 3
    assert not list(data_dir.glob("*_features.csv"))
    manifest = json.loads((data_dir / "acars_merge_manifest.json").read_text(encoding="utf-8"))
    item = manifest["sources"][0]
    assert item["merge_alignment"] == "row_index_outer"
    assert item["column_availability_lengths"]["ZVB1F-CRUISE"] == 1

    builder = _load_builder_module()
    identity = builder._source_identity(Path("B-1400_20260130044340.qar.csv"), "qar")
    assert identity["aircraft_id"] == "B-1400"
    assert identity["flight_id"] == "B-1400_20260130044340"
    assert identity["source_timestamp"] == "2026-01-30T04:43:40"
    acars_identity = builder._source_identity(Path("038382_1_acars.csv"), "acars")
    assert acars_identity["aircraft_id"] is None
    assert acars_identity["source_engine_id"] == "038382"
