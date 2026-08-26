# -*- coding: utf-8 -*-
"""Build a semantic, quality-aware aero-engine multitask dataset.

The builder keeps ACARS and QAR separate while exposing a shared G0-G7
channel-group schema. Native missing values are preserved as nulls and are
represented by observation masks; only task-specific masks are synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TASKS = ("forecast", "interpolation", "anomaly_detection")
DOMAINS = ("acars", "qar")
INTERPOLATION_MODES = ("random_point", "block", "periodic", "channel_dropout")
ANOMALY_MODES = ("spike", "bias", "drift", "frozen", "variance", "cross_channel")
QUALITY_SENTINELS = (768.0, 1536.0, 12288.0)
GROUPS = {
    "G0": {"subsystem": "FLIGHT_CONDITION", "role": "covariate"},
    "G1": {"subsystem": "ENGINE_COMMAND", "role": "covariate"},
    "G2": {"subsystem": "GAS_PATH", "role": "target_candidate"},
    "G3": {"subsystem": "FUEL_ACTUATION", "role": "intermediate"},
    "G4": {"subsystem": "LUBRICATION", "role": "target_candidate"},
    "G5": {"subsystem": "VIBRATION", "role": "target_candidate"},
    "G6": {"subsystem": "PNEUMATIC_CONFIG", "role": "covariate"},
    "G7": {"subsystem": "DIAGNOSTIC_AUX", "role": "diagnostic_aux"},
}

# Task boundaries are deliberately narrower than the complete sensor list. A
# covariate can condition a response target without becoming a health target.
FORECAST_CORE_VARIABLES = {
    "EGT", "N1", "N2", "FUEL_FLOW", "PS3", "T25",
}
FORECAST_SECONDARY_VARIABLES = {
    "OIL_PRESSURE", "OIL_TEMPERATURE", "OIL_QUANTITY",
}
ANOMALY_CORE_VARIABLES = FORECAST_CORE_VARIABLES | {
    "FAN_VIBRATION", "HPC_VIBRATION", "HPT_VIBRATION", "LPT_VIBRATION", "N1_VIBRATION", "N2_VIBRATION",
}
ANOMALY_SECONDARY_VARIABLES = {
    "OIL_TEMPERATURE", "OIL_QUANTITY", "FAN_IMBALANCE_ANGLE", "LPT_IMBALANCE_ANGLE", "FMV_POSITION", "VSV_POSITION", "VBV_POSITION", "DUCT_PRESSURE",
}


@dataclass(frozen=True)
class BuildConfig:
    input_dir: str
    output_dir: str
    context_length: int = 128
    acars_context_length: int | None = None
    qar_context_length: int | None = None
    max_group_channels: int = 16
    max_condition_channels: int = 4
    prediction_length: int = 16
    windows_per_source: int = 1
    interpolation_mask_ratio: float = 0.2
    interpolation_mode: str = "random_point"
    anomaly_ratio: float = 0.05
    anomaly_sigma: float = 4.0
    anomaly_mode: str = "spike"
    min_valid_points: int = 8
    seed: int = 42
    source_domain: str = "all"
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class SourceSeries:
    file_path: Path
    source_domain: str
    source_group_id: str
    split: str
    group_id: str
    phase: str | None
    phases: list[str]
    aircraft_id: str | None
    source_engine_id: str | None
    flight_id: str | None
    source_timestamp: str | None
    source_type: str
    time_index_type: str
    time_column: str | None
    time_unit: str
    base_time_interval: float
    columns: list[str]
    channel_metadata: list[dict]
    values: np.ndarray
    observation_mask: np.ndarray
    native_sampling_mask: np.ndarray
    quality_mask: np.ndarray
    time_delta: np.ndarray
    sampling_intervals: dict[str, float | None]
    native_missing_count: int
    quality_invalid_count: int


def _safe_float(value: float) -> float | None:
    return None if not math.isfinite(float(value)) else float(value)


def _json_values(values: Iterable[float]) -> list[float | None]:
    return [_safe_float(value) for value in values]


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _context_length_for_domain(config: BuildConfig, domain: str) -> int:
    configured = config.acars_context_length if domain == "acars" else config.qar_context_length
    return int(configured or config.context_length)


def _time_scale_token(source: SourceSeries, column: str) -> str:
    interval = source.sampling_intervals.get(column)
    if interval is None:
        interval_token = "UNKNOWN"
    elif float(interval).is_integer():
        interval_token = str(int(interval))
    else:
        interval_token = str(interval).replace(".", "p")
    unit_token = {"second": "S", "flight_cycle": "FLIGHT_CYCLE"}.get(source.time_unit, source.time_unit.upper())
    return f"{source.source_domain.upper()}_{interval_token}{unit_token}"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def infer_source_domain(path: Path) -> str:
    """Classify merged ACARS and explicitly named QAR files."""
    name = path.name.lower()
    if re.match(r"^\d+_1_acars\.csv$", name):
        return "acars"
    if re.match(r"^b-2694_.*_features\.csv$", name):
        return "qar"
    if name.endswith(".qar.csv"):
        return "qar"
    if "features" in name:
        return "acars"
    return "unknown"


def _source_identity(path: Path, domain: str) -> dict[str, str | None]:
    """Decode identity-bearing filename fields without conflating aircraft and engine ids."""
    name = path.name
    if domain == "qar":
        match = re.match(r"^(?P<aircraft>B-[^_]+)_(?P<timestamp>\d{14})\.qar\.csv$", name, flags=re.IGNORECASE)
        if match:
            raw_timestamp = match.group("timestamp")
            timestamp = pd.to_datetime(raw_timestamp, format="%Y%m%d%H%M%S", errors="coerce")
            return {
                "aircraft_id": match.group("aircraft"),
                "source_engine_id": None,
                "flight_id": f"{match.group('aircraft')}_{raw_timestamp}",
                "source_timestamp": timestamp.isoformat() if not pd.isna(timestamp) else None,
                "source_type": "qar_flight",
            }
    if domain == "acars":
        match = re.match(r"^(?P<engine>\d+)_1_acars\.csv$", name, flags=re.IGNORECASE)
        if match:
            return {
                "aircraft_id": None,
                "source_engine_id": match.group("engine"),
                "flight_id": None,
                "source_timestamp": None,
                "source_type": "acars_engine_series",
            }
    return {
        "aircraft_id": path.name.split("_")[0] if domain == "qar" else None,
        "source_engine_id": None,
        "flight_id": path.stem if domain == "qar" else None,
        "source_timestamp": None,
        "source_type": f"{domain}_source",
    }


def _load_source_catalog(input_dir: Path) -> dict[str, dict]:
    catalog_path = input_dir / "source_catalog.yaml"
    if catalog_path.exists():
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            normalized = {}
            for item in catalog.get("sources", []):
                source_file = str(item["source_file"])
                path = input_dir / source_file
                domain = str(item.get("source_domain") or infer_source_domain(path))
                identity = _source_identity(path, domain)
                item = {**item, **identity}
                normalized[source_file] = item
            return normalized
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid source catalog: {catalog_path}") from exc

    entries = []
    for path in sorted(input_dir.glob("*.csv")):
        domain = infer_source_domain(path)
        if domain == "unknown":
            continue
        entries.append(
            {
                "source_file": path.name,
                "source_domain": domain,
                **_source_identity(path, domain),
                "classification": "explicit_b2694_qar" if path.name.lower().startswith("b-2694_") else "project_naming_rule",
            }
        )
    catalog_path.write_text(
        json.dumps({"schema_version": 1, "sources": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {item["source_file"]: item for item in entries}


def _phase_from_column(column: str) -> str | None:
    match = re.search(r"-(TAKEOFF|CRUISE)$", column.upper())
    return match.group(1) if match else None


def _acars_metadata(column: str) -> dict:
    phase = _phase_from_column(column)
    base = re.sub(r"-(TAKEOFF|CRUISE)$", "", column, flags=re.IGNORECASE)
    feature_type = "raw"
    transformation = "none"
    for suffix, kind in (
        ("_D_SMOOTHED", "smoothed"),
        ("_SMOOTHED", "smoothed"),
        ("_D", "raw"),
    ):
        if base.upper().endswith(suffix):
            base = base[: -len(suffix)]
            feature_type = kind
            if suffix.startswith("_D"):
                transformation = "divergence"
            break

    delta_semantics = "none"
    source_variable = base
    canonical_overrides = {
        "DEGT": "EGT",
        "DPOIL": "OIL_PRESSURE",
        "ZPOIL": "OIL_PRESSURE",
        "ZTOIL": "OIL_TEMPERATURE",
    }
    if base.startswith("D") and len(base) > 1:
        delta_semantics = "oem_deviation"
        physical_variable = canonical_overrides.get(base, base[1:])
    else:
        physical_variable = canonical_overrides.get(base, base)

    if source_variable in {"DEGT", "GPCN25", "GWFM", "EGTHDM"}:
        group_id = "G2"
        pair_id = f"{physical_variable}_FAMILY"
    elif source_variable in {"ZPOIL", "DPOIL", "ZTOIL"}:
        group_id = "G4"
        pair_id = f"{physical_variable}_FAMILY"
    elif source_variable in {"ZVB1F", "ZVB1R"}:
        group_id = "G5"
        pair_id = "FAN_VIBRATION_PAIR"
    else:
        group_id = "G7"
        pair_id = None
    if source_variable in {"ZVB1F", "ZVB1R"}:
        physical_variable = "FAN_VIBRATION"

    relation_group_ids = [pair_id] if pair_id else []
    divergence_group_id = None
    if transformation == "divergence":
        divergence_group_id = f"{physical_variable}_DIVERGENCE"
        relation_group_ids.append(divergence_group_id)

    return {
        "channel": column,
        "domain": "ACARS",
        "group_id": group_id,
        "subsystem": GROUPS[group_id]["subsystem"],
        "role": GROUPS[group_id]["role"],
        "physical_variable": physical_variable,
        "source_variable": source_variable,
        "feature_type": feature_type,
        "delta_semantics": delta_semantics,
        "transformation": transformation,
        "divergence_group_id": divergence_group_id,
        "is_delta": delta_semantics != "none",
        "is_divergence": transformation == "divergence",
        "phase": phase,
        "engine_id": "global",
        "pair_id": pair_id,
        "relation_group_ids": relation_group_ids,
        "component": "FAN" if source_variable in {"ZVB1F", "ZVB1R"} else None,
        "target_candidate": group_id in {"G2", "G4", "G5"},
        "anomaly_input": group_id != "G7",
        "semantic_confidence": "high" if group_id != "G7" else "low",
    }


def _engine_id(column: str) -> str | int:
    match = re.search(r"(?:#|_)([12])(?:$|[^0-9])", column)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:OIL_TMP|OIL_QTY|OIL_PRS)([12])", column.upper())
    if match:
        return int(match.group(1))
    if re.search(r"(?:FF|EGT|T25_SEL|PS3_SEL)[12]$", column):
        return int(column[-1])
    if re.search(r"N1[12]$|N2[12]$", column):
        return int(column[-1])
    match = re.search(r"DUCTPRS([12])C?$", column.upper())
    if match:
        return int(match.group(1))
    return "global"


def _qar_metadata(column: str) -> dict:
    upper = column.upper()
    group_id = "G7"
    physical_variable = column
    pair_id = None
    component = None

    if upper in {"ALT", "FMC_SAT", "TAT", "CAS", "GS", "MACH", "ALTITUDE_STD", "AIR/GROUND"}:
        group_id = "G0"
        physical_variable = upper
    elif (
        "CUTOFF" in upper
        or "CMD_INDICATED" in upper
        or upper.startswith("TARGET_N1_")
        or upper.startswith("SELECTED_TRA_FILTERED_")
    ):
        group_id = "G1"
        physical_variable = re.sub(r"_[#_]?[12]$", "", column)
    elif (
        upper.startswith(("FF", "EGT", "N1", "N2", "T25_SEL", "PS3_SEL"))
        or upper.startswith(("SELECTED_FUEL_FLOW", "SELECTED_EGT", "SELECTED_N1_INDICATED", "SELECTED_N2_ACTUAL"))
        or upper.startswith(("SELECTED_PS3", "SELECTED_T25"))
    ):
        group_id = "G2"
        if "FUEL_FLOW" in upper or upper.startswith("FF"):
            physical_variable = "FUEL_FLOW"
        elif "EGT" in upper:
            physical_variable = "EGT"
        elif "N1" in upper:
            physical_variable = "N1"
        elif "N2" in upper:
            physical_variable = "N2"
        elif "PS3" in upper:
            physical_variable = "PS3"
        else:
            physical_variable = "T25"
    elif upper.startswith(("SELECTED_FMV", "SELECTED_VSV", "SELECTED_VBV")):
        group_id = "G3"
        physical_variable = upper.split("_")[1] + "_POSITION"
    elif upper.startswith(("OIL_", "ENG_OIL")) or upper.startswith("SELECTED_OIL"):
        group_id = "G4"
        physical_variable = "OIL_" + (
            "PRESSURE" if "PRESS" in upper or "PRS" in upper else
            "QUANTITY" if "QTY" in upper or "QUANTITY" in upper else
            "TEMPERATURE"
        )
    elif upper.startswith(("VIB_", "FAN_IMB", "LPT_IMB", "CN1_", "CN2_", "TN1_", "TN2_")):
        group_id = "G5"
        if upper.startswith("FAN_IMB"):
            component = "FAN"
            physical_variable = "FAN_IMBALANCE_ANGLE"
        elif upper.startswith("LPT_IMB"):
            component = "LPT"
            physical_variable = "LPT_IMBALANCE_ANGLE"
        elif upper.startswith(("VIB_N11", "VIB_N12")):
            component = "N1"
        elif upper.startswith(("VIB_N21", "VIB_N22")):
            component = "N2"
        elif upper.startswith("VIB_N1FNT"):
            component = "FAN"
        elif upper.startswith("TN1_"):
            component = "LPT"
        elif upper.startswith("CN1_"):
            component = "FAN"
        elif upper.startswith("CN2_"):
            component = "HPC"
        elif upper.startswith("TN2_"):
            component = "HPT"
        physical_variable = f"{component}_VIBRATION" if component else "VIBRATION"
    elif (
        upper.startswith(("DUCTPRS", "ENGINE_1_BLEED", "ENGINE_2_BLEED"))
        or "ANTI-ICE" in upper
        or "STARTER" in upper
        or upper == "ON_DC"
        or upper.startswith("T/R_POSN")
    ):
        group_id = "G6"
        if upper.startswith("DUCTPRS"):
            physical_variable = "DUCT_PRESSURE"
        elif "BLEED" in upper:
            physical_variable = "BLEED_SWITCH"
        else:
            physical_variable = upper

    if group_id == "G2":
        if "FUEL_FLOW" in upper or upper.startswith("FF"):
            pair_id = "FF_PAIR"
        elif "EGT" in upper:
            pair_id = "EGT_PAIR"
        elif "N1" in upper:
            pair_id = "N1_PAIR"
        elif "N2" in upper:
            pair_id = "N2_PAIR"
        elif "PS3" in upper:
            pair_id = "PS3_PAIR"
        elif "T25" in upper:
            pair_id = "T25_PAIR"
        if "SELECTED_N1_INDICATED" in upper or upper.startswith("N11") or upper.startswith("N12"):
            component = "N1"
        elif "SELECTED_N2_ACTUAL" in upper or upper.startswith("N21") or upper.startswith("N22"):
            component = "N2"
    elif group_id == "G4":
        pair_id = f"{physical_variable}_PAIR"
    elif group_id == "G5":
        pair_id = f"{component}_VIBRATION_PAIR" if component else "VIBRATION_PAIR"

    role = GROUPS[group_id]["role"]
    semantic_confidence = "high"
    if group_id == "G6" and upper.startswith("DUCTPRS"):
        role = "auxiliary_target"
        semantic_confidence = "medium"
    if group_id == "G7" and upper.startswith("CONT_REG"):
        role = "auxiliary"
        semantic_confidence = "low"
    if group_id == "G7" and upper.startswith("AVM_"):
        role = "label_or_metadata"

    if group_id == "G1" and "CMD_INDICATED" in upper:
        physical_variable = "N1_COMMAND"
    elif group_id == "G1" and upper.startswith("TARGET_N1"):
        physical_variable = "N1_TARGET"
    elif group_id == "G1" and upper.startswith("SELECTED_TRA"):
        physical_variable = "TRA"

    relation_group_ids = [pair_id] if pair_id else []
    engine_id = _engine_id(column)
    if group_id == "G1" and physical_variable in {"N1_COMMAND", "N1_TARGET"} and engine_id in {1, 2}:
        relation_group_ids.append(f"N1_COMMAND_RESPONSE_ENGINE_{engine_id}")
    if group_id == "G2" and physical_variable == "N1" and engine_id in {1, 2}:
        relation_group_ids.append(f"N1_COMMAND_RESPONSE_ENGINE_{engine_id}")
    if group_id == "G6" and physical_variable == "DUCT_PRESSURE" and engine_id in {1, 2}:
        relation_group_ids.append(f"DUCT_PRESSURE_ENGINE_{engine_id}")
    if group_id == "G1" and physical_variable == "TRA" and engine_id in {1, 2}:
        relation_group_ids.append(f"TRA_RESPONSE_ENGINE_{engine_id}")
    if group_id == "G2" and physical_variable in {"EGT", "N1", "N2", "FUEL_FLOW", "PS3", "T25"} and engine_id in {1, 2}:
        relation_group_ids.append(f"TRA_RESPONSE_ENGINE_{engine_id}")
    if group_id == "G3" and physical_variable in {"FMV_POSITION", "VSV_POSITION", "VBV_POSITION"} and engine_id in {1, 2}:
        relation_group_ids.append(
            f"{'FMV' if physical_variable == 'FMV_POSITION' else 'N2_CONTROL'}_RESPONSE_ENGINE_{engine_id}"
        )
    if group_id == "G2" and physical_variable == "FUEL_FLOW" and engine_id in {1, 2}:
        relation_group_ids.append(f"FMV_RESPONSE_ENGINE_{engine_id}")
    if group_id == "G2" and physical_variable == "N2" and engine_id in {1, 2}:
        relation_group_ids.append(f"N2_CONTROL_RESPONSE_ENGINE_{engine_id}")
    if group_id == "G2" and physical_variable in {"N2", "PS3"} and engine_id in {1, 2}:
        relation_group_ids.append(f"N2_RESPONSE_ENGINE_{engine_id}")
    if group_id == "G6" and physical_variable == "BLEED_SWITCH" and engine_id in {1, 2}:
        relation_group_ids.append(f"DUCT_PRESSURE_ENGINE_{engine_id}")


    return {
        "channel": column,
        "domain": "QAR",
        "group_id": group_id,
        "subsystem": GROUPS[group_id]["subsystem"],
        "role": role,
        "physical_variable": physical_variable,
        "feature_type": "raw",
        "phase": None,
        "engine_id": engine_id,
        "pair_id": pair_id,
        "relation_group_ids": relation_group_ids,
        "component": component,
        "target_candidate": group_id in {"G2", "G4", "G5"} or role == "auxiliary_target",
        "anomaly_input": group_id != "G7",
        "semantic_confidence": semantic_confidence,
    }


def channel_metadata(domain: str, column: str) -> dict:
    item = _acars_metadata(column) if domain == "acars" else _qar_metadata(column)
    item.setdefault("source_variable", item["physical_variable"])
    item.setdefault("delta_semantics", "none")
    item.setdefault("transformation", "none")
    item.setdefault("divergence_group_id", None)
    item.setdefault("is_delta", False)
    item.setdefault("is_divergence", False)
    variable = str(item["physical_variable"]).upper()
    if any(token in variable for token in ("TEMP", "TEMPERATURE", "EGT", "DEGT", "T25", "TAT")):
        physical_type = "temperature"
    elif any(token in variable for token in ("PRESS", "PRESSURE", "PS3", "DUCT")):
        physical_type = "pressure"
    elif any(token in variable for token in ("VIB", "ZVB")):
        physical_type = "vibration"
    elif any(token in variable for token in ("FLOW", "FF", "GWFM")):
        physical_type = "flow"
    elif any(token in variable for token in ("QTY", "QUANTITY")):
        physical_type = "quantity"
    elif any(token in variable for token in ("POSITION", "TRA", "ANGLE")):
        physical_type = "position_or_angle"
    elif any(token in variable for token in ("N1", "N2", "MACH", "CAS", "GS", "ALT")):
        physical_type = "speed_or_navigation"
    elif any(token in variable for token in ("SW", "SWITCH", "CUTOFF", "OPEN", "ON_DC", "GROUND", "ALERT", "FAULT")):
        physical_type = "status"
    else:
        physical_type = "unknown"
    item.update(
        {
            "canonical_variable": item["physical_variable"],
            "channel_family": item["physical_variable"],
            "physical_type": physical_type,
            "relation_group_id": item["pair_id"],
            "relation_group_ids": item.get("relation_group_ids", [item["pair_id"]] if item.get("pair_id") else []),
        }
    )
    return _assign_task_roles(item)


def _assign_task_roles(item: dict) -> dict:
    variable = str(item["physical_variable"]).upper()
    forecast_tier = (
        "core" if variable in FORECAST_CORE_VARIABLES
        else "secondary" if variable in FORECAST_SECONDARY_VARIABLES
        else "none"
    )
    anomaly_tier = (
        "core" if variable in ANOMALY_CORE_VARIABLES
        else "secondary" if variable in ANOMALY_SECONDARY_VARIABLES
        else "none"
    )
    label_only = item.get("role") == "label_or_metadata" or item["group_id"] == "G7"
    item.update(
        {
            "condition_variable": bool(item.get("role") in {"covariate", "intermediate"}) and not label_only,
            "forecast_tier": forecast_tier,
            "forecast_target": bool(forecast_tier != "none" and item["group_id"] in {"G2", "G4"}),
            "interpolation_target": bool(anomaly_tier != "none" or item["group_id"] in {"G2", "G4"}),
            "anomaly_tier": anomaly_tier,
            "anomaly_target": bool(anomaly_tier != "none" or item.get("role") == "auxiliary_target") and not label_only,
            "label_only": label_only,
        }
    )
    return item


def _clean_values(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    raw = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).T
    finite = np.isfinite(raw)
    sentinel = np.zeros_like(raw, dtype=bool)
    for value in QUALITY_SENTINELS:
        sentinel |= np.isclose(raw, value, rtol=0.0, atol=1e-5)
    invalid = ~finite | sentinel
    values = raw.copy()
    values[invalid] = np.nan
    return values, finite, finite & ~sentinel, int((finite & sentinel).sum()), int((~finite).sum())


def _infer_sampling_profile(
    domain: str,
    columns: list[str],
    observation_mask: np.ndarray,
    availability_lengths: dict[str, int] | None = None,
) -> tuple[str, float, dict[str, float | None], np.ndarray, np.ndarray]:
    """Infer native multi-rate availability without filling native gaps.

    QAR rows are one second apart. The modal distance between finite samples
    estimates each channel's native period; larger occasional gaps are kept as
    missing scheduled samples. ACARS rows represent flight cycles instead of
    seconds, so each declared channel has one observation opportunity per row.
    """
    channel_count, length = observation_mask.shape
    if domain == "acars":
        time_unit = "flight_cycle"
        base_interval = 1.0
        intervals = {column: 1.0 for column in columns}
        native_mask = np.ones_like(observation_mask, dtype=bool)
        if availability_lengths:
            row_indices = np.arange(length, dtype=np.int64)
            for index, column in enumerate(columns):
                declared_length = availability_lengths.get(column)
                if declared_length is not None:
                    native_mask[index] = row_indices < int(declared_length)
    else:
        time_unit = "second"
        base_interval = 1.0
        intervals: dict[str, float | None] = {}
        native_mask = np.zeros_like(observation_mask, dtype=bool)
        row_indices = np.arange(length, dtype=np.int64)
        for index, column in enumerate(columns):
            observed_positions = np.flatnonzero(observation_mask[index])
            if len(observed_positions) == 0:
                intervals[column] = None
                continue
            if len(observed_positions) < 2:
                interval = 1
            else:
                differences = np.diff(observed_positions)
                differences = differences[differences > 0]
                counts = Counter(int(value) for value in differences)
                highest_count = max(counts.values(), default=1)
                interval = min(value for value, count in counts.items() if count == highest_count)
                interval = max(1, int(interval))
            intervals[column] = float(interval * base_interval)
            first_observation = int(observed_positions[0])
            native_mask[index] = (row_indices >= first_observation) & ((row_indices - first_observation) % interval == 0)

    time_delta = np.full(observation_mask.shape, np.nan, dtype=np.float32)
    for index in range(channel_count):
        last_observation = None
        for position in range(length):
            if observation_mask[index, position]:
                last_observation = position
                time_delta[index, position] = 0.0
            elif last_observation is not None:
                time_delta[index, position] = float((position - last_observation) * base_interval)
    return time_unit, base_interval, intervals, native_mask, time_delta


def load_source_series(input_dir: Path, config: BuildConfig, catalog: dict[str, dict] | None = None) -> list[SourceSeries]:
    sources: list[SourceSeries] = []
    catalog = catalog or _load_source_catalog(input_dir)
    paths = sorted(input_dir.glob("*.csv"))
    split_map = _source_split_map(input_dir, paths, catalog, config)
    for path in paths:
        domain = catalog.get(path.name, {}).get("source_domain", infer_source_domain(path))
        if domain == "unknown" or (config.source_domain != "all" and domain != config.source_domain):
            continue

        df = pd.read_csv(path, low_memory=False)
        frame_columns = [str(column) for column in df.columns if str(column).upper() in {"FRAME_COUNTER", "TIMESTAMP", "TIME"}]
        time_column = frame_columns[0] if frame_columns else None
        time_index_type = "frame_counter" if time_column == "FRAME_COUNTER" else "timestamp" if time_column else "row_index"
        if frame_columns:
            df = df.drop(columns=frame_columns)
        values, observation_mask, quality_mask, quality_invalid_count, native_missing_count = _clean_values(df)
        columns = [str(column) for column in df.columns]
        metadata = [channel_metadata(domain, column) for column in columns]
        catalog_item = catalog.get(path.name, {})
        availability_lengths = {
            str(column): int(length)
            for column, length in catalog_item.get("column_availability_lengths", {}).items()
            if length is not None
        }
        time_unit, base_time_interval, sampling_intervals, native_sampling_mask, time_delta = _infer_sampling_profile(
            domain, columns, observation_mask, availability_lengths=availability_lengths
        )

        grouped: dict[str, list[int]] = {}
        for index, item in enumerate(metadata):
            grouped.setdefault(item["group_id"], []).append(index)
        phase_values = {item["phase"] for item in metadata if item["phase"]}
        phase = next(iter(phase_values)) if len(phase_values) == 1 else None
        phases = sorted(phase_values)
        identity = _source_identity(path, domain)
        condition_priority = {
            name: rank
            for rank, name in enumerate(
                ("ALT", "ALTITUDE_STD", "MACH", "TAT", "FMC_SAT", "CAS", "GS", "AIR/GROUND")
            )
        }
        condition_indices = sorted(
            grouped.get("G0", []),
            key=lambda index: (condition_priority.get(columns[index].upper(), 999), index),
        )

        for group_id, indices in sorted(grouped.items()):
            primary_indices = list(indices)
            primary_relation_ids = {
                relation
                for index in primary_indices
                for relation in (metadata[index].get("relation_group_ids") or [])
            }
            related_indices = [
                index
                for index, item in enumerate(metadata)
                if index not in primary_indices
                and primary_relation_ids.intersection(item.get("relation_group_ids") or [])
            ]
            semantic_indices = primary_indices + related_indices
            semantic_indices = semantic_indices[: config.max_group_channels]
            if not any(
                metadata[index].get("forecast_target")
                or metadata[index].get("interpolation_target")
                or metadata[index].get("anomaly_target")
                for index in semantic_indices
            ):
                continue
            if domain == "qar" and group_id != "G0" and condition_indices:
                available_slots = max(0, config.max_group_channels - len(semantic_indices))
                condition_candidates = [
                    index for index in condition_indices
                    if index not in semantic_indices
                ]
                condition_count = min(config.max_condition_channels, available_slots)
                indices = condition_candidates[:condition_count] + semantic_indices
            else:
                indices = semantic_indices
            group_values = values[indices]
            group_observation_mask = observation_mask[indices]
            group_native_sampling_mask = native_sampling_mask[indices]
            group_quality_mask = quality_mask[indices]
            group_time_delta = time_delta[indices]
            if group_values.shape[1] < config.context_length:
                continue
            if int(group_quality_mask.sum()) < config.min_valid_points:
                continue
            group_columns = [columns[index] for index in indices]
            group_sampling_intervals = {column: sampling_intervals[column] for column in group_columns}
            source_group_id = f"{path.stem}::{group_id}"
            sources.append(
                SourceSeries(
                    file_path=path,
                    source_domain=domain,
                    source_group_id=source_group_id,
                    split=split_map.get(path.name, _stable_split(path.name, config)),
                    group_id=group_id,
                    phase=phase,
                    phases=phases,
                    aircraft_id=identity["aircraft_id"],
                    source_engine_id=identity["source_engine_id"],
                    flight_id=identity["flight_id"],
                    source_timestamp=identity["source_timestamp"],
                    source_type=str(identity["source_type"]),
                    time_index_type=time_index_type,
                    time_column=time_column,
                    time_unit=time_unit,
                    base_time_interval=base_time_interval,
                    columns=group_columns,
                    channel_metadata=[metadata[index] for index in indices],
                    values=group_values,
                    observation_mask=group_observation_mask,
                    native_sampling_mask=group_native_sampling_mask,
                    quality_mask=group_quality_mask,
                    time_delta=group_time_delta,
                    sampling_intervals=group_sampling_intervals,
                    native_missing_count=int((~group_observation_mask).sum()),
                    quality_invalid_count=int((group_observation_mask & ~group_quality_mask).sum()),
                )
            )
    return sources


def _candidate_starts(source: SourceSeries, task: str, config: BuildConfig) -> np.ndarray:
    total_length = config.context_length + (config.prediction_length if task == "forecast" else 0)
    if source.values.shape[1] < total_length:
        return np.array([], dtype=int)
    starts = np.arange(0, source.values.shape[1] - total_length + 1)
    valid_counts = np.array(
        [source.quality_mask[:, start : start + total_length].sum() for start in starts], dtype=np.int64
    )
    minimum = max(config.min_valid_points, int(0.05 * total_length * len(source.columns)))
    return starts[valid_counts >= minimum]


def _sample_windows(sources: list[SourceSeries], task: str, config: BuildConfig, rng: np.random.Generator):
    candidates: list[tuple[SourceSeries, int]] = []
    for source in sources:
        if task == "anomaly_detection" and source.group_id == "G7":
            continue
        starts = _candidate_starts(source, task, config)
        if len(starts) == 0:
            continue
        count = min(config.windows_per_source, len(starts))
        selected = rng.choice(starts, size=count, replace=False)
        candidates.extend((source, int(start)) for start in sorted(selected.tolist()))
    return candidates


def _base_record(task: str, index: int, source: SourceSeries, start: int, config: BuildConfig) -> dict:
    condition_channels = [
        item["channel"] for item in source.channel_metadata
        if item.get("condition_variable")
    ]
    forecast_target_channels = [
        item["channel"] for item in source.channel_metadata
        if item.get("forecast_target")
    ]
    interpolation_target_channels = [
        item["channel"] for item in source.channel_metadata
        if item.get("interpolation_target")
    ]
    anomaly_target_channels = [
        item["channel"] for item in source.channel_metadata
        if item.get("anomaly_target")
    ]
    state_channels = interpolation_target_channels
    relation_groups: dict[str, list[str]] = {}
    for item in source.channel_metadata:
        relation_ids = item.get("relation_group_ids") or ([item["relation_group_id"]] if item.get("relation_group_id") else [])
        for relation_id in relation_ids:
            relation_groups.setdefault(relation_id, []).append(item["channel"])
    return {
        "sample_id": f"{source.source_domain}_{task}_{index:06d}",
        "task_type": task,
        "source_domain": source.source_domain,
        "schema_id": (
            "B-2694" if source.source_domain == "qar" and source.file_path.name.upper().startswith("B-2694_")
            else "B-1400" if source.source_domain == "qar" and source.file_path.name.upper().startswith("B-1400_")
            else "ACARS" if source.source_domain == "acars" else "UNKNOWN"
        ),
        "source_file": source.file_path.name,
        "aircraft_id": source.aircraft_id,
        "source_engine_id": source.source_engine_id,
        "source_type": source.source_type,
        "source_timestamp": source.source_timestamp,
        "source_group_id": source.source_group_id,
        "split": source.split,
        "group_id": source.group_id,
        "subsystem": GROUPS[source.group_id]["subsystem"],
        "phase": source.phase,
        "phases": source.phases,
        "time_index_type": source.time_index_type,
        "time_column": source.time_column,
        "time_unit": source.time_unit,
        "base_time_interval": source.base_time_interval,
        "source_columns": source.columns,
        "time_scale_tokens": {column: _time_scale_token(source, column) for column in source.columns},
        "channel_metadata": source.channel_metadata,
        "channel_ids": source.columns,
        "subsystem_ids": [item["subsystem"] for item in source.channel_metadata],
        "channel_group_ids": [item["group_id"] for item in source.channel_metadata],
        "engine_ids": [item["engine_id"] for item in source.channel_metadata],
        "relation_groups": relation_groups,
        "condition_channels": condition_channels,
        "state_channels": state_channels,
        "forecast_target_channels": forecast_target_channels,
        "interpolation_target_channels": interpolation_target_channels,
        "anomaly_target_channels": anomaly_target_channels,
        "condition_state_relation": {
            "condition_channels": condition_channels,
            "state_channels": state_channels,
            "forecast_target_channels": forecast_target_channels,
            "interpolation_target_channels": interpolation_target_channels,
            "anomaly_target_channels": anomaly_target_channels,
        },
        "target_policy": {
            "forecast": "forecast_target",
            "interpolation": "interpolation_target",
            "anomaly_detection": "anomaly_target",
            "forecast_core": [item["channel"] for item in source.channel_metadata if item.get("forecast_tier") == "core"],
            "forecast_secondary": [item["channel"] for item in source.channel_metadata if item.get("forecast_tier") == "secondary"],
            "anomaly_core": [item["channel"] for item in source.channel_metadata if item.get("anomaly_tier") == "core"],
            "anomaly_secondary": [item["channel"] for item in source.channel_metadata if item.get("anomaly_tier") == "secondary"],
        },
        "flight_id": source.flight_id,
        "sampling_interval": source.sampling_intervals,
        "sampling_interval_unit": source.time_unit,
        "start_index": int(start),
        "context_length": config.context_length,
    }


def _temporal_window_metadata(source: SourceSeries, start: int, end: int) -> dict:
    native_mask = source.native_sampling_mask[:, start:end]
    deltas = source.time_delta[:, start:end]
    availability = {column: native_mask[index].astype(int).tolist() for index, column in enumerate(source.columns)}
    time_delta = {column: _json_values(deltas[index]) for index, column in enumerate(source.columns)}
    return {
        "native_sampling_mask": availability,
        "availability_mask": availability,
        "time_delta": time_delta,
    }


def build_forecast_records(sources: list[SourceSeries], config: BuildConfig, rng: np.random.Generator) -> list[dict]:
    records = []
    for index, (source, start) in enumerate(_sample_windows(sources, "forecast", config, rng)):
        context_end = start + config.context_length
        future_end = context_end + config.prediction_length
        context = source.values[:, start:context_end]
        future = source.values[:, context_end:future_end]
        context_observation_mask = source.observation_mask[:, start:context_end]
        future_observation_mask = source.observation_mask[:, context_end:future_end]
        context_quality_mask = source.quality_mask[:, start:context_end]
        future_quality_mask = source.quality_mask[:, context_end:future_end]
        history_temporal = _temporal_window_metadata(source, start, context_end)
        target_temporal = _temporal_window_metadata(source, context_end, future_end)
        record = _base_record("forecast", index, source, start, config)
        has_state_targets = bool(record["forecast_target_channels"])
        forecast_task_mask = {}
        for idx, col in enumerate(source.columns):
            is_target = bool(source.channel_metadata[idx].get("forecast_target"))
            forecast_task_mask[col] = (
                future_quality_mask[idx].astype(int).tolist()
                if is_target or not has_state_targets
                else np.zeros(config.prediction_length, dtype=np.int8).tolist()
            )
        record.update(
            {
                "history": {col: _json_values(context[idx]) for idx, col in enumerate(source.columns)},
                "history_observation_mask": {col: context_observation_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                "history_quality_mask": {col: context_quality_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                "history_native_sampling_mask": history_temporal["native_sampling_mask"],
                "history_availability_mask": history_temporal["availability_mask"],
                "history_time_delta": history_temporal["time_delta"],
                "native_sampling_mask": history_temporal["native_sampling_mask"],
                "availability_mask": history_temporal["availability_mask"],
                "time_delta": history_temporal["time_delta"],
                "prediction_length": config.prediction_length,
                "target_future": {col: _json_values(future[idx]) for idx, col in enumerate(source.columns)},
                "target_observation_mask": {col: future_observation_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                "target_quality_mask": {col: future_quality_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                "target_native_sampling_mask": target_temporal["native_sampling_mask"],
                "target_availability_mask": target_temporal["availability_mask"],
                "target_time_delta": target_temporal["time_delta"],
                "task_mask": forecast_task_mask,
                "metric_hints": ["smae_on_observed_future", "smape_on_observed_future"],
            }
        )
        records.append(record)
    return records


def _choose_interpolation_indices(valid_candidates: np.ndarray, count: int, context_length: int, mode: str, rng: np.random.Generator) -> list[int]:
    if len(valid_candidates) == 0:
        return []
    if mode == "channel_dropout":
        return valid_candidates.astype(int).tolist()
    if mode == "block":
        block_length = min(count, len(valid_candidates))
        possible_starts = [start for start in valid_candidates if start + block_length - 1 in valid_candidates]
        if possible_starts:
            start = int(rng.choice(np.asarray(possible_starts)))
            return list(range(start, start + block_length))
    if mode == "periodic":
        stride = max(2, context_length // max(count, 1))
        offset = int(rng.choice(np.arange(min(stride, max(context_length, 1)))))
        selected = [int(index) for index in valid_candidates if (int(index) - offset) % stride == 0]
        if len(selected) >= count:
            return selected[:count]
    return sorted(rng.choice(valid_candidates, size=min(count, len(valid_candidates)), replace=False).astype(int).tolist())


def build_interpolation_records(sources: list[SourceSeries], config: BuildConfig, rng: np.random.Generator) -> list[dict]:
    records = []
    for index, (source, start) in enumerate(_sample_windows(sources, "interpolation", config, rng)):
        interpolation_mode = (
            str(rng.choice(INTERPOLATION_MODES))
            if config.interpolation_mode == "mixed"
            else config.interpolation_mode
        )
        dropout_channel = (
            int(rng.integers(0, len(source.columns)))
            if interpolation_mode == "channel_dropout"
            else None
        )
        clean = source.values[:, start : start + config.context_length].copy()
        original_observation_mask = source.observation_mask[:, start : start + config.context_length].copy()
        original_quality_mask = source.quality_mask[:, start : start + config.context_length].copy()
        observed = clean.copy()
        observed_observation_mask = original_observation_mask.copy()
        observed_quality_mask = original_quality_mask.copy()
        missing_indices: dict[str, list[int]] = {}
        targets: dict[str, list[float | None]] = {}
        evaluation_masks: dict[str, list[int]] = {}
        edge_guard = min(8, max(0, config.context_length // 8))
        candidates = np.arange(edge_guard, config.context_length - edge_guard)

        for idx, col in enumerate(source.columns):
            valid_candidates = candidates[original_quality_mask[idx, candidates]]
            count = min(max(1, int(round(config.context_length * config.interpolation_mask_ratio))), len(valid_candidates))
            chosen = (
                _choose_interpolation_indices(
                    valid_candidates,
                    count,
                    config.context_length,
                    interpolation_mode,
                    rng,
                )
                if count
                and source.channel_metadata[idx].get("interpolation_target")
                and (dropout_channel is None or idx == dropout_channel)
                else []
            )
            missing_indices[col] = chosen
            targets[col] = _json_values(clean[idx, chosen])
            evaluation_mask = np.zeros(config.context_length, dtype=np.int8)
            evaluation_mask[chosen] = 1
            evaluation_masks[col] = evaluation_mask.tolist()
            observed[idx, chosen] = np.nan
            observed_observation_mask[idx, chosen] = False
            observed_quality_mask[idx, chosen] = False

        record = _base_record("interpolation", index, source, start, config)
        record.update(
            {
                "observed_context": {col: _json_values(observed[idx]) for idx, col in enumerate(source.columns)},
                "observed_observation_mask": {col: observed_observation_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                "observed_quality_mask": {col: observed_quality_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                **_temporal_window_metadata(source, start, start + config.context_length),
                "missing_indices": missing_indices,
                "target_values": targets,
                "evaluation_mask": evaluation_masks,
                "task_mask": evaluation_masks,
                "mask_ratio": config.interpolation_mask_ratio,
                "interpolation_mode": interpolation_mode,
                "metric_hints": ["smae_on_masked_observed", "smape_on_masked_observed"],
            }
        )
        records.append(record)
    return records


def build_anomaly_records(sources: list[SourceSeries], config: BuildConfig, rng: np.random.Generator) -> list[dict]:
    records = []
    for index, (source, start) in enumerate(_sample_windows(sources, "anomaly_detection", config, rng)):
        anomaly_mode = (
            str(rng.choice(ANOMALY_MODES))
            if config.anomaly_mode == "mixed"
            else config.anomaly_mode
        )
        cross_channel_index = None
        if anomaly_mode == "cross_channel":
            relation_candidates = [
                index
                for index, item in enumerate(source.channel_metadata)
                if item.get("anomaly_target")
                and any(
                    str(relation).startswith("N1_COMMAND_RESPONSE_ENGINE_")
                    for relation in (item.get("relation_group_ids") or [])
                )
            ]
            if relation_candidates:
                cross_channel_index = int(rng.choice(np.asarray(relation_candidates)))
            else:
                anomaly_candidates = [
                    index for index, item in enumerate(source.channel_metadata)
                    if item.get("anomaly_target")
                ]
                cross_channel_index = (
                    anomaly_candidates[0]
                    if config.anomaly_mode != "mixed" and anomaly_candidates
                    else int(rng.choice(np.asarray(anomaly_candidates)))
                    if anomaly_candidates
                    else None
                )
        clean = source.values[:, start : start + config.context_length].copy()
        original_observation_mask = source.observation_mask[:, start : start + config.context_length].copy()
        original_quality_mask = source.quality_mask[:, start : start + config.context_length].copy()
        corrupted = clean.copy()
        labels: dict[str, list[int]] = {}
        anomaly_indices: dict[str, list[int]] = {}
        anomaly_values: dict[str, list[float | None]] = {}
        evaluation_masks: dict[str, list[int]] = {}
        edge_guard = min(8, max(0, config.context_length // 8))
        candidates = np.arange(edge_guard, config.context_length - edge_guard)

        for idx, col in enumerate(source.columns):
            valid_candidates = candidates[original_quality_mask[idx, candidates]]
            count = min(max(1, int(round(config.context_length * config.anomaly_ratio))), len(valid_candidates))
            if not source.channel_metadata[idx].get("anomaly_target"):
                chosen = []
            elif cross_channel_index is not None and idx != cross_channel_index:
                chosen = []
            else:
                selection_mode = "block" if anomaly_mode in {"bias", "drift", "frozen"} else "random_point"
                chosen = _choose_interpolation_indices(valid_candidates, count, config.context_length, selection_mode, rng) if count else []
            scale = float(np.nanstd(clean[idx]))
            if not math.isfinite(scale) or scale <= 1e-6:
                valid_values = clean[idx][original_quality_mask[idx]]
                scale = max(float(np.nanmean(np.abs(valid_values))) if len(valid_values) else 1.0, 1.0) * 0.1
            if chosen:
                sign = float(rng.choice(np.array([-1.0, 1.0], dtype=np.float32)))
                if anomaly_mode == "frozen":
                    reference_index = max(chosen[0] - 1, 0)
                    corrupted[idx, chosen] = clean[idx, reference_index]
                elif anomaly_mode == "drift":
                    offsets = sign * config.anomaly_sigma * scale * np.linspace(0.0, 1.0, len(chosen), dtype=np.float32)
                    corrupted[idx, chosen] = corrupted[idx, chosen] + offsets
                elif anomaly_mode == "variance":
                    noise = rng.normal(0.0, config.anomaly_sigma * scale, size=len(chosen)).astype(np.float32)
                    corrupted[idx, chosen] = corrupted[idx, chosen] + noise
                else:
                    offset = sign * config.anomaly_sigma * scale
                    corrupted[idx, chosen] = corrupted[idx, chosen] + offset
            col_labels = np.zeros(config.context_length, dtype=np.int8)
            col_labels[chosen] = 1
            eval_mask = (
                original_quality_mask[idx].astype(np.int8)
                if source.channel_metadata[idx].get("anomaly_target")
                else np.zeros(config.context_length, dtype=np.int8)
            )
            labels[col] = col_labels.tolist()
            anomaly_indices[col] = chosen
            anomaly_values[col] = _json_values(corrupted[idx, chosen])
            evaluation_masks[col] = eval_mask.tolist()

        record = _base_record("anomaly_detection", index, source, start, config)
        record.update(
            {
                "clean_context": {col: _json_values(clean[idx]) for idx, col in enumerate(source.columns)},
                "observed_context": {col: _json_values(corrupted[idx]) for idx, col in enumerate(source.columns)},
                "observation_mask": {col: original_observation_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                "quality_mask": {col: original_quality_mask[idx].astype(int).tolist() for idx, col in enumerate(source.columns)},
                **_temporal_window_metadata(source, start, start + config.context_length),
                "evaluation_mask": evaluation_masks,
                "task_mask": labels,
                "anomaly_indices": anomaly_indices,
                "anomaly_values": anomaly_values,
                "labels": labels,
                "anomaly_ratio": config.anomaly_ratio,
                "anomaly_sigma": config.anomaly_sigma,
                "anomaly_mode": anomaly_mode,
                "anomaly_mechanism": "physical_inconsistency" if anomaly_mode == "cross_channel" else anomaly_mode,
                "metric_hints": ["precision", "recall", "f1", "auroc_on_observed_points"],
            }
        )
        records.append(record)
    return records


def _stable_split(split_key: str, config: BuildConfig) -> str:
    value = int(hashlib.sha256(f"{config.seed}:{split_key}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < config.train_ratio:
        return "train"
    if value < config.train_ratio + config.val_ratio:
        return "val"
    return "test"


def _source_split_map(input_dir: Path, paths: list[Path], catalog: dict[str, dict], config: BuildConfig) -> dict[str, str]:
    """Assign source files without splitting one aircraft/engine across sets.

    QAR has a filename timestamp, so each aircraft is split chronologically.
    Merged ACARS has an engine identifier but no absolute timestamp, so all
    groups from one engine use one deterministic entity split.
    """
    identity_rows: list[tuple[Path, str, dict[str, str | None]]] = []
    for path in paths:
        domain = catalog.get(path.name, {}).get("source_domain", infer_source_domain(path))
        identity_rows.append((path, domain, _source_identity(path, domain)))

    assignments: dict[str, str] = {}
    qar_by_aircraft: dict[str, list[tuple[Path, str | None]]] = {}
    for path, domain, identity in identity_rows:
        if domain == "qar" and identity.get("aircraft_id"):
            qar_by_aircraft.setdefault(str(identity["aircraft_id"]), []).append((path, identity.get("source_timestamp")))
        elif domain == "acars" and identity.get("source_engine_id"):
            assignments[path.name] = _stable_split(f"acars_engine::{identity['source_engine_id']}", config)
        else:
            assignments[path.name] = _stable_split(path.name, config)

    for aircraft, rows in qar_by_aircraft.items():
        rows.sort(key=lambda item: (item[1] or "", item[0].name))
        count = len(rows)
        for index, (path, _) in enumerate(rows):
            fraction = index / max(count, 1)
            if fraction < config.train_ratio:
                split = "train"
            elif fraction < config.train_ratio + config.val_ratio:
                split = "val"
            else:
                split = "test"
            assignments[path.name] = split
    return assignments


def _write_splits(output_dir: Path, records_by_domain_task: dict[tuple[str, str], list[dict]], config: BuildConfig) -> dict:
    summary: dict[str, dict] = {}
    for (domain, task), records in records_by_domain_task.items():
        summary.setdefault(domain, {})[task] = {}
        for split in ("train", "val", "test"):
            split_records = [record for record in records if record["split"] == split]
            split_records.sort(key=lambda record: record["sample_id"])
            _write_jsonl(output_dir / "splits" / split / domain / f"{task}.jsonl", split_records)
            summary[domain][task][split] = len(split_records)
    return summary


def _source_manifest(sources: list[SourceSeries]) -> list[dict]:
    manifest = []
    for source in sources:
        manifest.append(
            {
                "source_group_id": source.source_group_id,
                "source_file": source.file_path.name,
                "source_domain": source.source_domain,
                "split": source.split,
                "aircraft_id": source.aircraft_id,
                "source_engine_id": source.source_engine_id,
                "source_type": source.source_type,
                "source_timestamp": source.source_timestamp,
                "flight_id": source.flight_id,
                "group_id": source.group_id,
                "subsystem": GROUPS[source.group_id]["subsystem"],
                "phase": source.phase,
                "phases": source.phases,
                "time_index_type": source.time_index_type,
                "time_column": source.time_column,
                "time_unit": source.time_unit,
                "base_time_interval": source.base_time_interval,
                "sampling_interval": source.sampling_intervals,
                "sampling_interval_unit": source.time_unit,
                "time_scale_tokens": {column: _time_scale_token(source, column) for column in source.columns},
                "time_delta_definition": "elapsed time since the most recent finite observation",
                "columns": source.columns,
                "channel_metadata": source.channel_metadata,
                "length": int(source.values.shape[1]),
                "channel_count": len(source.columns),
                "observed_ratio": float(source.observation_mask.mean()),
                "native_sampling_ratio": float(source.native_sampling_mask.mean()),
                "quality_ratio": float(source.quality_mask.mean()),
                "native_missing_count": source.native_missing_count,
                "quality_invalid_count": source.quality_invalid_count,
            }
        )
    return manifest


def _channel_schema(sources: list[SourceSeries]) -> dict:
    channels: dict[str, dict] = {}
    sampling_profiles: dict[str, dict] = {}
    for source in sources:
        for index, item in enumerate(source.channel_metadata):
            key = f"{item['domain']}::{item['channel']}"
            channels[key] = item
            profile = sampling_profiles.setdefault(
                key,
                {
                    "units": set(),
                    "intervals": set(),
                    "time_scale_tokens": set(),
                    "source_count": 0,
                    "native_sampling_ratio_sum": 0.0,
                },
            )
            profile["units"].add(source.time_unit)
            interval = source.sampling_intervals.get(item["channel"])
            if interval is not None:
                profile["intervals"].add(float(interval))
            profile["time_scale_tokens"].add(_time_scale_token(source, item["channel"]))
            profile["source_count"] += 1
            profile["native_sampling_ratio_sum"] += float(source.native_sampling_mask[index].mean())
    for profile in sampling_profiles.values():
        profile["units"] = sorted(profile["units"])
        profile["intervals"] = sorted(profile["intervals"])
        profile["time_scale_tokens"] = sorted(profile["time_scale_tokens"])
        profile["mean_native_sampling_ratio"] = profile.pop("native_sampling_ratio_sum") / profile["source_count"]
    return {
        "schema_version": 8,
        "groups": GROUPS,
        "channels": channels,
        "sampling_profiles": sampling_profiles,
        "quality_sentinels": QUALITY_SENTINELS,
        "task_policy": {
            "forecast": "forecast_target: core/secondary engine response only",
            "interpolation": "interpolation_target: health and response channels",
            "anomaly_detection": "anomaly_target: engine health plus actuator/pneumatic auxiliary channels",
            "covariates": "condition_variable channels are context-only",
            "labels": "label_only channels are excluded from model targets and anomaly scores",
        },
    }


def _fit_train_scalers(sources: list[SourceSeries], config: BuildConfig) -> dict:
    chunks: dict[str, list[np.ndarray]] = {}
    metadata: dict[str, dict[str, str | int]] = {}
    train_files = {source.file_path.name for source in sources if source.split == "train"}
    for source in sources:
        if source.file_path.name not in train_files:
            continue
        for index, item in enumerate(source.channel_metadata):
            valid = source.quality_mask[index]
            values = source.values[index, valid].astype(np.float64)
            if len(values) == 0:
                continue
            phase = item.get("phase") or source.phase or "UNSPECIFIED"
            key = (
                f"{item['domain']}::{item['canonical_variable']}::{item['engine_id']}::"
                f"{item['feature_type']}::{item.get('delta_semantics', 'none')}::"
                f"{item.get('transformation', 'none')}::{phase}"
            )
            chunks.setdefault(key, []).append(values)
            metadata[key] = {
                "domain": item["domain"],
                "canonical_variable": item["canonical_variable"],
                "engine_id": item["engine_id"],
                "feature_type": item["feature_type"],
                "delta_semantics": item.get("delta_semantics", "none"),
                "transformation": item.get("transformation", "none"),
                "phase": phase,
            }

    finalized = {}
    for key, value_chunks in chunks.items():
        values = np.concatenate(value_chunks)
        domain = str(metadata[key]["domain"])
        mean = float(values.mean())
        std = float(values.std())
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75]).tolist()
        finalized[key] = {
            **metadata[key],
            "count": int(len(values)),
            "mean": mean,
            "std": std,
            "mean_abs": float(np.abs(values).mean()),
            "median": median,
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
            "min": float(values.min()),
            "max": float(values.max()),
            "recommended_method": "robust_median_iqr" if domain == "ACARS" else "zscore_or_mean_abs",
        }
    return {
        "fit_scope": "train_source_files_only",
        "split_unit": "aircraft_or_engine_entity",
        "grouping": "domain_x_canonical_variable_x_engine_x_feature_type_x_delta_semantics_x_transformation_x_phase",
        "methods": {
            "ACARS": "robust_median_iqr",
            "QAR": "zscore_or_mean_abs",
        },
        "scalers": finalized,
    }


def build_datasets(config: BuildConfig) -> dict:
    if config.source_domain not in {"all", *DOMAINS}:
        raise ValueError(f"source_domain must be one of all, acars, qar; got {config.source_domain}")
    if config.windows_per_source < 1:
        raise ValueError("windows_per_source must be >= 1")
    if config.train_ratio <= 0 or config.val_ratio < 0 or config.train_ratio + config.val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1")

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    source_catalog = _load_source_catalog(input_dir)
    sources = load_source_series(input_dir, config, source_catalog)
    if not sources:
        raise ValueError(f"No usable source groups found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_schema = _channel_schema(sources)
    _write_json(output_dir / "channel_schema.json", channel_schema)
    _write_json(output_dir / "channel_schema.yaml", channel_schema)
    source_manifest = _source_manifest(sources)
    _write_jsonl(output_dir / "source_manifest.jsonl", source_manifest)
    _write_jsonl(output_dir / "canonical" / "source_manifest.jsonl", source_manifest)
    (output_dir / "canonical" / "README.md").write_text(
        "# Canonical source index\n\n"
        "This index preserves source-file, phase, channel, group, quality, and time metadata.\n"
        "For ACARS, a leading D is recorded as OEM delta/deviation semantics, while a trailing _D is recorded as divergence.\n"
        "The raw CSV values remain in the project `data` directory; task JSONL files are generated from this index.\n",
        encoding="utf-8",
    )

    counts: dict[str, dict[str, int]] = {}
    split_summary: dict[str, dict[str, dict[str, int]]] = {}
    for domain_index, domain in enumerate(DOMAINS):
        domain_sources = [source for source in sources if source.source_domain == domain]
        if not domain_sources:
            continue
        counts[domain] = {}
        split_summary[domain] = {}
        domain_config = replace(config, context_length=_context_length_for_domain(config, domain))
        rng = np.random.default_rng(config.seed + domain_index * 100_003)
        builders = {
            "forecast": build_forecast_records,
            "interpolation": build_interpolation_records,
            "anomaly_detection": build_anomaly_records,
        }
        for task, builder in builders.items():
            records = builder(domain_sources, domain_config, rng)
            _write_jsonl(output_dir / "tasks" / domain / f"{task}.jsonl", records)
            counts[domain][task] = len(records)
            split_summary[domain][task] = {}
            for split in ("train", "val", "test"):
                split_records = [record for record in records if record["split"] == split]
                split_records.sort(key=lambda record: record["sample_id"])
                _write_jsonl(
                    output_dir / "splits" / split / domain / f"{task}.jsonl",
                    split_records,
                )
                split_summary[domain][task][split] = len(split_records)
            del records
            del split_records

    scalers = _fit_train_scalers(sources, config)
    _write_json(output_dir / "train_scalers.json", scalers)
    manifest = {
        "schema_version": 8,
        "config": asdict(config),
        "source_domains": sorted({source.source_domain for source in sources}),
        "source_catalog": "data/source_catalog.yaml",
        "source_group_count": len(sources),
        "source_file_count": len({source.file_path.name for source in sources}),
        "source_file_count_by_domain": {
            domain: len({source.file_path.name for source in sources if source.source_domain == domain}) for domain in DOMAINS
        },
        "source_group_count_by_domain": {
            domain: sum(source.source_domain == domain for source in sources) for domain in DOMAINS
        },
        "task_counts_by_domain": counts,
        "split_counts": split_summary,
        "split_unit": "aircraft_or_engine_entity",
        "split_isolation": {
            "implemented": "QAR aircraft chronological split; ACARS engine entity split",
            "aircraft_level": "implemented_from_QAR_filename",
            "engine_level": "implemented_for_merged_ACARS_engine_id; QAR_engine_serial_not_available",
            "reason": "QAR aircraft id and timestamp are encoded in the filename; ACARS engine id is encoded in the merged source filename.",
        },
        "flight_segmentation": {
            "implemented": "one source CSV is one flight/source segment",
            "intra_file_phase_segmentation": "not_inferred",
            "reason": "The raw files do not provide a validated phase-boundary dictionary; ACARS phase comes from column suffixes.",
        },
        "quality_sentinels_removed_from_observation": QUALITY_SENTINELS,
        "sampling_schema": {
            "qar": "one row equals one second; per-channel native interval is inferred from the modal finite-observation gap",
            "acars": "one row equals one flight cycle; phase is metadata and is never cross-filled",
            "acars_delta": "a leading D denotes OEM delta/deviation relative to the canonical physical variable",
            "acars_divergence": "a trailing _D denotes divergence and is represented by a separate dual-engine relation group",
            "time_scale_token": "domain plus channel sampling interval, for example QAR_1S, QAR_4S, or ACARS_1FLIGHT_CYCLE",
            "native_sampling_mask": "scheduled sampling opportunity before task masking",
            "time_delta": "elapsed time since the most recent finite observation, in the source time unit",
        },
        "sampling_policy": {
            "recommended": "balanced_domain_task",
            "domain_weights": {"acars": 0.5, "qar": 0.5},
            "generator": "scripts/datasets/canonical_task_generator.py",
        },
        "evaluation_policy": {
            "forecast_primary": sorted(FORECAST_CORE_VARIABLES),
            "forecast_secondary": sorted(FORECAST_SECONDARY_VARIABLES),
            "anomaly_primary": sorted(ANOMALY_CORE_VARIABLES),
            "anomaly_secondary": sorted(ANOMALY_SECONDARY_VARIABLES),
            "anomaly_score": "temporal residual + cross-engine residual + physical-response residual",
            "unavailable": [
                "QAR engine serial isolation requires source_file-to-engine_serial_id mapping",
                "fault-type metrics require validated physical fault labels",
            ],
        },
        "paths": {
            "channel_schema": "channel_schema.json",
            "source_manifest": "source_manifest.jsonl",
            "canonical_source_manifest": "canonical/source_manifest.jsonl",
            "train_scalers": "train_scalers.json",
            "tasks": "tasks/{domain}/{task}.jsonl",
            "splits": "splits/{split}/{domain}/{task}.jsonl",
            "online_task_generator": "scripts/datasets/canonical_task_generator.py",
            "manifest_dir": "manifest/",
            "acars_merge_manifest": "data/acars_merge_manifest.json",
        },
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    _write_json(output_dir / "dataset_manifest.yaml", manifest)
    _write_json(output_dir / "manifest" / "dataset_manifest.json", manifest)
    _write_json(output_dir / "manifest" / "channel_schema.json", channel_schema)
    _write_json(output_dir / "manifest" / "source_catalog.json", {"schema_version": 2, "sources": list(source_catalog.values())})
    (output_dir / "manifest" / "README.md").write_text(
        "# Dataset manifests\n\n"
        "This directory contains the machine-readable dataset, channel, and source manifests.\n"
        "The canonical source index remains under `../canonical/`.\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Aero-engine monitoring multitask dataset\n\n"
        "ACARS and QAR are stored separately under `tasks/` and `splits/`.\n"
        "Each source file is divided by semantic group G0-G7, not by arbitrary column chunks.\n"
        "Native missing values remain null; observation and quality masks are separate.\n"
        "QAR sampling metadata includes per-channel native intervals and time deltas; ACARS uses flight-cycle units.\n"
        "ACARS triplets are merged by native row order into one engine source; the merge manifest records column availability lengths.\n"
        "QAR source filenames provide aircraft id, normalized flight id, and source timestamp; splits are aircraft-chronological for QAR and engine-isolated for ACARS.\n"
        "Each channel also has a time-scale token such as `QAR_1S` or `QAR_4S`; domain-specific context lengths can be configured.\n"
        "For ACARS, a leading D is an OEM deviation marker and a trailing _D is a divergence transformation.\n"
        "The former `B-2694_*_features.csv` files are QAR sources and are named `*.qar.csv`.\n"
        "G7 diagnostic and unknown channels are excluded from anomaly-detection inputs.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build semantic aero-engine multitask JSONL datasets")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default=str(Path("data") / "aero_engine_dataset"))
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--acars-context-length", type=int, default=None)
    parser.add_argument("--qar-context-length", type=int, default=None)
    parser.add_argument("--max-group-channels", type=int, default=16)
    parser.add_argument("--max-condition-channels", type=int, default=4)
    parser.add_argument("--prediction-length", type=int, default=16)
    parser.add_argument("--windows-per-source", type=int, default=1)
    parser.add_argument("--interpolation-mask-ratio", type=float, default=0.2)
    parser.add_argument(
        "--interpolation-mode",
        choices=[*INTERPOLATION_MODES, "mixed"],
        default="random_point",
    )
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--anomaly-sigma", type=float, default=4.0)
    parser.add_argument(
        "--anomaly-mode",
        choices=[*ANOMALY_MODES, "mixed"],
        default="spike",
    )
    parser.add_argument("--min-valid-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-domain", choices=["all", "acars", "qar"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return BuildConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    print(json.dumps(build_datasets(parse_args()), indent=2, ensure_ascii=False))
