# -*- coding: utf-8 -*-
"""Run reproducible JOINT-context ablations on the aero-engine corpus.

The script creates lightweight, channel-filtered dataset views and invokes the
existing Chronos-2 trainer for each view. Raw CSV files and model weights are
never copied. Profiles are defined by semantic group ids from channel_metadata:
G0 condition, G1 command, G2 gas path, G3 actuation, G4 lubrication,
G5 vibration, G6 pneumatic/configuration, and G7 diagnostic auxiliary.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


GROUP_NAMES = {
    "G0": "condition",
    "G1": "command",
    "G2": "gas",
    "G3": "actuation",
    "G4": "oil",
    "G5": "vibration",
    "G6": "pneumatic",
    "G7": "auxiliary",
}

PROFILE_GROUPS = {
    "gas_only": {"G2"},
    "conditions_gas": {"G0", "G2"},
    "conditions_gas_control": {"G0", "G1", "G2"},
    "gas_oil_vibration": {"G2", "G4", "G5"},
    "conditions_gas_oil": {"G0", "G2", "G4"},
    "conditions_gas_vibration": {"G0", "G2", "G5"},
    "conditions_gas_oil_vibration": {"G0", "G2", "G4", "G5"},
    "core_joint": {"G0", "G1", "G2", "G4", "G5"},
    "operational_joint": {"G0", "G1", "G2", "G3", "G4", "G5", "G6"},
    "full_joint": set(GROUP_NAMES),
    "no_condition": set(GROUP_NAMES) - {"G0"},
    "no_oil": set(GROUP_NAMES) - {"G4"},
    "no_vibration": set(GROUP_NAMES) - {"G5"},
}

DEFAULT_PROFILES = (
    "gas_only",
    "conditions_gas",
    "conditions_gas_oil_vibration",
    "core_joint",
    "full_joint",
    "no_condition",
    "no_oil",
    "no_vibration",
)

CHANNEL_DICT_FIELDS = {
    "history",
    "history_observation_mask",
    "history_quality_mask",
    "history_native_sampling_mask",
    "history_availability_mask",
    "history_time_delta",
    "target_future",
    "target_observation_mask",
    "target_quality_mask",
    "target_native_sampling_mask",
    "target_availability_mask",
    "target_time_delta",
    "observed_context",
    "observed_observation_mask",
    "observed_quality_mask",
    "clean_context",
    "observation_mask",
    "quality_mask",
    "task_mask",
    "evaluation_mask",
    "missing_indices",
    "target_values",
    "sampling_interval",
    "time_scale_tokens",
}


def _record_has_target(record: dict, columns: list[str]) -> bool:
    task_mask = record.get("task_mask") or {}
    for column in columns:
        values = task_mask.get(column, [])
        if any(bool(value) for value in values):
            return True
    return False


def _schema_id(record: dict) -> str:
    explicit = record.get("schema_id")
    if explicit:
        return str(explicit)
    if str(record.get("source_domain", "")).lower() == "acars":
        return "ACARS"
    source_file = str(record.get("source_file", "")).upper()
    if source_file.startswith("B-2694_"):
        return "B-2694"
    if source_file.startswith("B-1400_"):
        return "B-1400"
    return "QAR-OTHER"


def _stratified_sample_ids(records_by_schema: dict[str, list[dict]], limit: int, rng: random.Random) -> list[str]:
    """Select a deterministic, approximately proportional sample per schema."""
    total = sum(len(records) for records in records_by_schema.values())
    if total <= limit:
        selected = [record["sample_id"] for records in records_by_schema.values() for record in records]
        rng.shuffle(selected)
        return selected

    schemas = sorted(records_by_schema)
    allocations = {
        schema: min(len(records_by_schema[schema]), int(limit * len(records_by_schema[schema]) / total))
        for schema in schemas
    }
    if limit >= len(schemas):
        for schema in schemas:
            if records_by_schema[schema] and allocations[schema] == 0:
                allocations[schema] = 1

    while sum(allocations.values()) < limit:
        candidates = [schema for schema in schemas if allocations[schema] < len(records_by_schema[schema])]
        if not candidates:
            break
        schema = max(
            candidates,
            key=lambda item: (
                len(records_by_schema[item]) / total - allocations[item] / limit,
                -schemas.index(item),
            ),
        )
        allocations[schema] += 1

    selected: list[str] = []
    for schema in schemas:
        candidates = list(records_by_schema[schema])
        rng.shuffle(candidates)
        selected.extend(record["sample_id"] for record in candidates[:allocations[schema]])
    rng.shuffle(selected)
    return selected


def _build_eval_manifest(dataset_root: Path, output_path: Path, limit: int, seed: int) -> dict:
    """Build one frozen validation manifest shared by every profile and seed."""
    records_by_pool: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    split_root = dataset_root / "splits" / "val"
    for source in sorted(split_root.glob("*/*.jsonl")):
        pool_name = f"{source.parent.name}/{source.stem}"
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records_by_pool[pool_name][_schema_id(record)].append(record)

    manifest = {
        "version": 1,
        "split": "val",
        "selection": "deterministic proportional stratification by schema",
        "selection_seed": seed,
        "max_records_per_domain_task": limit,
        "records": {},
        "strata_counts": {},
    }
    for pool_name in sorted(records_by_pool):
        groups = records_by_pool[pool_name]
        pool_seed = seed + sum(ord(char) for char in pool_name) * 1009
        ids = _stratified_sample_ids(groups, limit, random.Random(pool_seed))
        manifest["records"][pool_name] = ids
        selected = set(ids)
        manifest["strata_counts"][pool_name] = {
            schema: sum(record["sample_id"] in selected for record in records)
            for schema, records in sorted(groups.items())
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _apply_target_groups(
    record: dict,
    target_groups: set[str] | None,
    target_domain: str,
) -> None:
    """Keep context channels intact while restricting supervised QAR targets."""
    domain = str(record.get("source_domain", "")).lower()
    if not target_groups or (target_domain != "all" and domain != target_domain):
        return

    columns = list(record.get("source_columns") or [])
    metadata = record.get("channel_metadata") or [{} for _ in columns]
    target_columns = {
        column
        for column, item in zip(columns, metadata)
        if str(item.get("group_id", "G7")) in target_groups
    }
    for field in ("task_mask", "evaluation_mask"):
        values_by_column = record.get(field)
        if not isinstance(values_by_column, dict):
            continue
        record[field] = {
            column: [bool(value) and column in target_columns for value in values]
            for column, values in values_by_column.items()
        }
    for field in (
        "forecast_target_channels",
        "interpolation_target_channels",
        "anomaly_target_channels",
    ):
        values = record.get(field)
        if isinstance(values, list):
            record[field] = [column for column in values if column in target_columns]


def _filter_record(
    record: dict,
    allowed_groups: set[str],
    target_groups: set[str] | None,
    target_domain: str,
) -> dict | None:
    metadata = record.get("channel_metadata") or []
    source_columns = record.get("source_columns") or []
    keep_indices = [
        index
        for index, item in enumerate(metadata)
        if str(item.get("group_id", "G7")) in allowed_groups
    ]
    if not keep_indices:
        return None

    columns = [source_columns[index] for index in keep_indices]
    filtered = dict(record)
    filtered["source_columns"] = columns
    filtered["channel_ids"] = [record.get("channel_ids", source_columns)[index] for index in keep_indices]
    filtered["channel_metadata"] = [metadata[index] for index in keep_indices]
    for field in ("subsystem_ids", "channel_group_ids", "engine_ids"):
        values = record.get(field)
        if values is not None:
            filtered[field] = [values[index] for index in keep_indices]

    for field in CHANNEL_DICT_FIELDS:
        value = record.get(field)
        if not isinstance(value, dict):
            continue
        filtered[field] = {column: value[column] for column in columns if column in value}

    relation_groups = record.get("relation_groups") or {}
    filtered["relation_groups"] = {
        relation: [column for column in values if column in columns]
        for relation, values in relation_groups.items()
        if any(column in columns for column in values)
    }
    filtered["condition_channels"] = [column for column in record.get("condition_channels", []) if column in columns]
    filtered["state_channels"] = [column for column in record.get("state_channels", []) if column in columns]
    filtered["condition_state_relation"] = {
        "condition_channels": filtered["condition_channels"],
        "state_channels": filtered["state_channels"],
    }

    _apply_target_groups(filtered, target_groups, target_domain)

    # A profile that removes every task target cannot produce a valid loss.
    if record.get("task_type") in {"forecast", "interpolation", "anomaly_detection"}:
        if not _record_has_target(filtered, columns):
            return None
    return filtered


def _write_view(
    dataset_root: Path,
    view_root: Path,
    allowed_groups: set[str],
    profile: str,
    target_groups: set[str] | None,
    target_domain: str,
) -> dict:
    if view_root.exists():
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True)
    for filename in (
        "channel_schema.json",
        "channel_schema.yaml",
        "dataset_manifest.json",
        "dataset_manifest.yaml",
        "train_scalers.json",
    ):
        source = dataset_root / filename
        if source.exists():
            shutil.copy2(source, view_root / filename)

    counts: dict[str, int] = {}
    for source in sorted((dataset_root / "splits").glob("*/*/*.jsonl")):
        relative = source.relative_to(dataset_root)
        target = view_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with source.open("r", encoding="utf-8") as input_stream, target.open("w", encoding="utf-8") as output_stream:
            for line in input_stream:
                if not line.strip():
                    continue
                record = _filter_record(
                    json.loads(line), allowed_groups, target_groups, target_domain
                )
                if record is None:
                    continue
                record["context_profile"] = profile
                record["context_profile_groups"] = sorted(allowed_groups)
                output_stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
        counts[str(relative)] = kept

    manifest = {
        "profile": profile,
        "groups": sorted(allowed_groups),
        "target_groups": sorted(target_groups) if target_groups else None,
        "target_domain": target_domain if target_groups else None,
        "group_names": {group: GROUP_NAMES[group] for group in sorted(allowed_groups)},
        "source_dataset": str(dataset_root),
        "split_record_counts": counts,
    }
    (view_root / "context_profile.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _trainer_command(args, view_root: Path, output_root: Path) -> list[str]:
    trainer = Path(args.trainer).resolve()
    return [
        sys.executable,
        str(trainer),
        "--dataset-dir", str(view_root),
        "--model-path", args.model_path,
        "--output-dir", str(output_root),
        "--stage1-steps", str(args.stage1_steps),
        "--stage2-steps", str(args.stage2_steps),
        "--eval-records", str(args.eval_records),
        "--eval-manifest", str(args.eval_manifest),
        "--trainable-mode", args.trainable_mode,
        "--stage2-trainable-mode", args.stage2_trainable_mode,
        "--anomaly-inference", args.anomaly_inference,
        "--device", args.device,
        "--seed", str(args.seed),
        "--log-steps", str(args.log_steps),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Core-JOINT and subsystem-removal experiments")
    parser.add_argument("--dataset-dir", default="data/aero_engine_dataset_joint_familywise_v2")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="results/joint_context_experiments")
    parser.add_argument("--trainer", default="scripts/training/train_aero_multitask_chronos2.py")
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILE_GROUPS), default=list(DEFAULT_PROFILES))
    parser.add_argument(
        "--target-groups",
        nargs="+",
        choices=sorted(GROUP_NAMES),
        help="Restrict supervised targets while retaining every selected input channel.",
    )
    parser.add_argument(
        "--target-domain",
        choices=["qar", "acars", "all"],
        default="qar",
        help="Apply --target-groups only to this domain (QAR by default).",
    )
    parser.add_argument("--stage1-steps", type=int, default=0)
    parser.add_argument("--stage2-steps", type=int, default=1000)
    parser.add_argument("--eval-records", type=int, default=64)
    parser.add_argument("--eval-manifest", default=None, help="Frozen validation sample manifest shared across profiles and seeds.")
    parser.add_argument("--eval-manifest-seed", type=int, default=20260828)
    parser.add_argument("--trainable-mode", default="heads", choices=["reconstruction_head", "heads", "adapter_blocks", "full"])
    parser.add_argument("--stage2-trainable-mode", default="heads", choices=["same", "reconstruction_head", "heads", "adapter_blocks", "full"])
    parser.add_argument("--anomaly-inference", default="direct", choices=["direct", "lopo"])
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--keep-views", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    view_root = output_root / "dataset_views"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.eval_manifest:
        eval_manifest_path = Path(args.eval_manifest).resolve()
        if not eval_manifest_path.exists():
            _build_eval_manifest(dataset_root, eval_manifest_path, args.eval_records, args.eval_manifest_seed)
    else:
        eval_manifest_path = output_root / "eval_manifest.json"
        _build_eval_manifest(dataset_root, eval_manifest_path, args.eval_records, args.eval_manifest_seed)
    args.eval_manifest = str(eval_manifest_path)
    summaries = []

    for profile in args.profiles:
        profile_root = output_root / profile
        profile_root.mkdir(parents=True, exist_ok=True)
        manifest = _write_view(
            dataset_root,
            view_root / profile,
            PROFILE_GROUPS[profile],
            profile,
            set(args.target_groups) if args.target_groups else None,
            args.target_domain,
        )
        (profile_root / "context_profile.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        command = _trainer_command(args, view_root / profile, profile_root)
        print(json.dumps({"profile": profile, "command": command}, ensure_ascii=False), flush=True)
        completed = subprocess.run(command, check=False)
        summary = {
            "profile": profile,
            "return_code": completed.returncode,
            "output_dir": str(profile_root),
            "groups": sorted(PROFILE_GROUPS[profile]),
        }
        summaries.append(summary)
        (profile_root / "experiment_status.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0:
            raise SystemExit(f"Experiment {profile} failed with return code {completed.returncode}")

    (output_root / "experiment_summary.json").write_text(
        json.dumps({"args": vars(args), "experiments": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.keep_views:
        shutil.rmtree(view_root, ignore_errors=True)


if __name__ == "__main__":
    main()
