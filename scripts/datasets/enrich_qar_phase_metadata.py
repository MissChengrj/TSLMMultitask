# -*- coding: utf-8 -*-
"""Attach row-level QAR flight-phase sidecars to canonical/task JSONL data."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd


PHASE_COLUMNS = (
    "flight_phase", "vertical_state", "power_state", "segment_id",
    "phase_confidence", "height_ft", "height_agl_departure_ft",
    "height_agl_arrival_ft", "vertical_speed_fpm",
)


def _load_sidecars(segments_dir: Path) -> dict[str, dict[str, list]]:
    sidecars = {}
    for path in sorted(segments_dir.glob("*.segments.csv")):
        frame = pd.read_csv(path, low_memory=False)
        sidecars[path.name.replace(".segments.csv", ".qar.csv")] = {
            column: frame[column].tolist() for column in PHASE_COLUMNS if column in frame.columns
        }
    return sidecars


def _window(values: list, start: int, length: int) -> list:
    return values[start:start + length]


def _phase_summary(values: list) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(value for value in values if value == value).items()}


def _enrich_record(record: dict, sidecar: dict[str, list]) -> dict:
    start = int(record.get("start_index", 0))
    context_length = int(record.get("context_length", 0))
    target_length = int(record.get("prediction_length", 0)) if record.get("task_type") == "forecast" else 0
    context = {key: _window(values, start, context_length) for key, values in sidecar.items()}
    record["engine_serial_id"] = record.get("source_engine_id")
    record["entity_id"] = (f"engine:{record['source_engine_id']}" if record.get("source_engine_id") else f"aircraft:{record.get('aircraft_id')}")
    record["entity_level"] = "engine" if record.get("source_engine_id") else "aircraft"
    record["phase_source"] = "qar_segmenter_v1"
    record["phase_labels"] = sorted(set(value for value in context.get("flight_phase", []) if value == value))
    record["phase_distribution"] = _phase_summary(context.get("flight_phase", []))
    phase_values = context.get("flight_phase", [])
    record["phase_boundary_mask"] = [0] + [int(phase_values[index] != phase_values[index - 1]) for index in range(1, len(phase_values))] if phase_values else []
    if record.get("task_type") == "forecast":
        target = {key: _window(values, start + context_length, target_length) for key, values in sidecar.items()}
        for key, values in context.items():
            record[f"history_{key}"] = values
        for key, values in target.items():
            record[f"target_{key}"] = values
    else:
        for key, values in context.items():
            record[key] = values
    return record


def _rewrite_jsonl(path: Path, sidecars: dict[str, dict[str, list]]) -> int:
    count = 0
    fd, temporary_name = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".jsonl", dir=str(path.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for line in source:
                record = json.loads(line)
                record["engine_serial_id"] = record.get("source_engine_id")
                record["entity_id"] = (f"engine:{record['source_engine_id']}" if record.get("source_engine_id") else f"aircraft:{record.get('aircraft_id')}")
                record["entity_level"] = "engine" if record.get("source_engine_id") else "aircraft"
                if record.get("source_domain") == "qar" and record.get("source_file") in sidecars:
                    record = _enrich_record(record, sidecars[record["source_file"]])
                    count += 1
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _update_source_manifest(path: Path, metrics: dict[str, dict]) -> int:
    records = []
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        item = metrics.get(record.get("source_file"))
        if item and record.get("source_domain") == "qar":
            record.update(
                {
                    "engine_serial_id": record.get("source_engine_id"),
                    "entity_id": (f"engine:{record['source_engine_id']}" if record.get("source_engine_id") else f"aircraft:{record.get('aircraft_id')}"),
                    "entity_level": "engine" if record.get("source_engine_id") else "aircraft",
                    "phase_source": "qar_segmenter_v1",
                    "flight_phase_labels": sorted(item.get("phase_duration_seconds", {}).keys()),
                    "phase_duration_seconds": item.get("phase_duration_seconds", {}),
                    "flight_phase_segment_count": int(item.get("phase_transition_count", 0)) + 1,
                    "cruise_height_reference_ft": item.get("cruise_height_reference_ft"),
                    "phase_boundary_indices": {
                        "liftoff": item.get("liftoff_index"),
                        "initial_climb_end": item.get("initial_climb_end_index"),
                        "cruise_start": item.get("cruise_start_index"),
                        "top_of_descent": item.get("top_of_descent_index"),
                        "approach_start": item.get("approach_start_index"),
                        "touchdown": item.get("touchdown_index"),
                    },
                }
            )
            count += 1
        records.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return count


def run(output_dir: Path, segments_dir: Path) -> dict:
    sidecars = _load_sidecars(segments_dir)
    task_count = 0
    for path in sorted((output_dir / "tasks").glob("*/*.jsonl")):
        task_count += _rewrite_jsonl(path, sidecars)
    for path in sorted((output_dir / "splits").glob("*/*/*.jsonl")):
        task_count += _rewrite_jsonl(path, sidecars)

    summary_path = segments_dir.parent / "segmentation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = {item["source_file"]: item for item in summary.get("metrics", [])}
    manifest_count = 0
    for path in (output_dir / "source_manifest.jsonl", output_dir / "canonical" / "source_manifest.jsonl"):
        if path.exists():
            manifest_count += _update_source_manifest(path, metrics)

    for path in (output_dir / "dataset_manifest.json", output_dir / "dataset_manifest.yaml", output_dir / "manifest" / "dataset_manifest.json"):
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = max(int(manifest.get("schema_version", 0)), 8)
        manifest["flight_segmentation"] = {
            "implemented": "QAR row-level phase sidecars generated by segment_qar_flight_phases.py and attached to canonical/task JSONL",
            "phase_source": "qar_segmenter_v1",
            "phase_order": ["Taxi-out", "Takeoff", "Initial Climb", "Climb", "Cruise", "Descent", "Approach", "Landing", "Taxi-in"],
            "boundary_fields": ["liftoff", "initial_climb_end", "cruise_start", "top_of_descent", "approach_start", "touchdown"],
            "limitations": ["B-1400 uses low-height/low-speed touchdown proxies because AIR/GROUND is absent", "FLAP/GEAR labels are unavailable in the supplied QAR files"],
        }
        manifest.setdefault("paths", {})["phase_segments"] = "flight_phase_segments/segments/{source}.segments.csv"
        manifest["paths"]["phase_summary"] = "flight_phase_segments/segmentation_summary.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    enrichment = {
        "schema_version": 1,
        "phase_source": "qar_segmenter_v1",
        "sidecar_count": len(sidecars),
        "task_records_enriched": task_count,
        "source_manifest_records_enriched": manifest_count,
        "fields": list(PHASE_COLUMNS),
    }
    (output_dir / "phase_enrichment_manifest.json").write_text(json.dumps(enrichment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return enrichment


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach QAR phase sidecars to task and canonical JSONL files")
    parser.add_argument("--output-dir", default="data/aero_engine_dataset")
    parser.add_argument("--segments-dir", default="data/aero_engine_dataset/flight_phase_segments/segments")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.output_dir), Path(args.segments_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
