# -*- coding: utf-8 -*-
"""Merge the three raw ACARS feature files belonging to each engine.

ACARS has no row-level timestamp in these files, so the only defensible
alignment available here is the native row order.  An outer join preserves
the longer phase/type series and the manifest records each column's original
row count so downstream availability masks do not turn padding into data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ACARS_NAME = re.compile(r"^(?P<engine>\d+)_(?P<kind>[123])_features\.csv$", re.IGNORECASE)
QAR_NAME = re.compile(r"^(?P<aircraft>B-[^_]+)_(?P<timestamp>\d{14})\.qar\.csv$", re.IGNORECASE)


def _source_catalog(data_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for path in sorted(data_dir.glob("*.csv")):
        name = path.name
        if re.match(r"^\d+_1_acars\.csv$", name, re.IGNORECASE):
            engine = name.split("_", 1)[0]
            entries.append(
                {
                    "source_file": name,
                    "source_domain": "acars",
                    "aircraft_id": None,
                    "source_engine_id": engine,
                    "flight_id": None,
                    "source_timestamp": None,
                    "source_type": "acars_engine_series",
                    "classification": "merged_acars_1_2_3",
                }
            )
            continue
        match = QAR_NAME.match(name)
        if match:
            raw_timestamp = match.group("timestamp")
            timestamp = pd.to_datetime(raw_timestamp, format="%Y%m%d%H%M%S", errors="coerce")
            entries.append(
                {
                    "source_file": name,
                    "source_domain": "qar",
                    "aircraft_id": match.group("aircraft"),
                    "source_engine_id": None,
                    "flight_id": f"{match.group('aircraft')}_{raw_timestamp}",
                    "source_timestamp": timestamp.isoformat() if not pd.isna(timestamp) else None,
                    "source_type": "qar_flight",
                    "classification": "filename_aircraft_timestamp",
                }
            )
    return entries


def merge_acars_sources(data_dir: Path, delete_originals: bool = True) -> dict:
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*.csv")):
        match = ACARS_NAME.match(path.name)
        if match:
            groups.setdefault(match.group("engine"), {})[match.group("kind")] = path

    if not groups:
        raise ValueError(f"No raw ACARS triplets found in {data_dir}")
    incomplete = {engine: sorted(set("123") - set(files)) for engine, files in groups.items() if set(files) != set("123")}
    if incomplete:
        details = ", ".join(f"{engine}: missing {','.join(missing)}" for engine, missing in sorted(incomplete.items()))
        raise ValueError(f"Incomplete ACARS triplets: {details}")

    merged_records: list[dict] = []
    output_paths: list[Path] = []
    for engine, files in sorted(groups.items()):
        frames: list[pd.DataFrame] = []
        source_row_counts: dict[str, int] = {}
        column_row_counts: dict[str, int] = {}
        source_files: list[str] = []
        for kind in ("1", "2", "3"):
            path = files[kind]
            frame = pd.read_csv(path, low_memory=False)
            frame.columns = [str(column) for column in frame.columns]
            if frame.columns.duplicated().any():
                duplicates = sorted(set(frame.columns[frame.columns.duplicated()].tolist()))
                raise ValueError(f"Duplicate columns in {path.name}: {duplicates}")
            source_files.append(path.name)
            source_row_counts[path.name] = int(len(frame))
            for column in frame.columns:
                column_row_counts[column] = int(len(frame))
            frames.append(frame.reset_index(drop=True))

        merged = pd.concat(frames, axis=1, join="outer", sort=False)
        if merged.columns.duplicated().any():
            duplicates = sorted(set(merged.columns[merged.columns.duplicated()].tolist()))
            raise ValueError(f"Duplicate columns after merging engine {engine}: {duplicates}")
        output = data_dir / f"{engine}_1_acars.csv"
        merged.to_csv(output, index=False)
        output_paths.append(output)
        merged_records.append(
            {
                "source_file": output.name,
                "source_domain": "acars",
                "source_engine_id": engine,
                "source_files": source_files,
                "source_row_counts": source_row_counts,
                "column_availability_lengths": column_row_counts,
                "row_count": int(len(merged)),
                "column_count": int(len(merged.columns)),
                "merge_alignment": "row_index_outer",
            }
        )

    manifest = {
        "schema_version": 1,
        "merge_alignment": "row_index_outer",
        "reason": "ACARS files provide no validated row-level timestamp; rows are aligned by native sequence order.",
        "sources": merged_records,
    }
    (data_dir / "acars_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    catalog_entries = _source_catalog(data_dir)
    by_name = {item["source_file"]: item for item in catalog_entries}
    for item in merged_records:
        by_name[item["source_file"]].update(
            {
                "column_availability_lengths": item["column_availability_lengths"],
                "acars_merge_alignment": item["merge_alignment"],
                "acars_source_files": item["source_files"],
            }
        )
    (data_dir / "source_catalog.yaml").write_text(
        json.dumps({"schema_version": 2, "sources": list(by_name.values())}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if delete_originals:
        for files in groups.values():
            for path in files.values():
                path.unlink()

    return {
        "engine_count": len(merged_records),
        "merged_files": [item["source_file"] for item in merged_records],
        "deleted_original_count": sum(len(files) for files in groups.values()) if delete_originals else 0,
        "qar_source_count": sum(item["source_domain"] == "qar" for item in catalog_entries),
    }


def refresh_source_catalog(data_dir: Path) -> dict:
    """Refresh identity fields after a merge has already removed raw triplets."""
    existing_path = data_dir / "source_catalog.yaml"
    existing = {}
    if existing_path.exists():
        payload = json.loads(existing_path.read_text(encoding="utf-8"))
        existing = {item["source_file"]: item for item in payload.get("sources", [])}
    entries = _source_catalog(data_dir)
    merge_manifest_path = data_dir / "acars_merge_manifest.json"
    merge_manifest = {}
    if merge_manifest_path.exists():
        merge_manifest = {
            item["source_file"]: item
            for item in json.loads(merge_manifest_path.read_text(encoding="utf-8")).get("sources", [])
        }
    for item in entries:
        old = existing.get(item["source_file"], {})
        item.update({key: value for key, value in old.items() if key.startswith("column_") or key.startswith("acars_")})
        if item["source_file"] in merge_manifest:
            merged = merge_manifest[item["source_file"]]
            item.update(
                {
                    "column_availability_lengths": merged["column_availability_lengths"],
                    "acars_merge_alignment": merged["merge_alignment"],
                    "acars_source_files": merged["source_files"],
                }
            )
    existing_path.write_text(
        json.dumps({"schema_version": 2, "sources": entries}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"source_count": len(entries), "catalog": str(existing_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ACARS *_features.csv triplets into *_acars.csv files")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--keep-originals", action="store_true")
    parser.add_argument("--refresh-only", action="store_true")
    args = parser.parse_args()
    result = refresh_source_catalog(Path(args.data_dir)) if args.refresh_only else merge_acars_sources(
        Path(args.data_dir), delete_originals=not args.keep_originals
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
