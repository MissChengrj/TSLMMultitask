# -*- coding: utf-8 -*-
"""Split multitask JSONL evaluation data by source file.

The source data under data/multitask_eval_all_sources is already organized as:

    <source>/<task>.jsonl

This script creates train/val/test files for each source and task.  Splitting is
done by source_file, not by individual row, to reduce flight/file leakage across
splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


TASKS = ("forecast", "interpolation", "anomaly_detection")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_group_keys(keys: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 2) if n >= 3 else max(1, n - 1)
        n_val = 1 if n >= 3 else 0

    train = set(keys[:n_train])
    val = set(keys[n_train : n_train + n_val])
    test = set(keys[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description="Split multitask JSONL data by source_file")
    parser.add_argument("--input-dir", default="data/multitask_eval_all_sources")
    parser.add_argument("--output-dir", default="data/multitask_eval_splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1")

    summary: dict[str, dict] = {}
    for source_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        source = source_dir.name
        summary[source] = {}
        for task in TASKS:
            path = source_dir / f"{task}.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                groups[str(row.get("source_file", row.get("sample_id", "unknown")))].append(row)

            split_keys = split_group_keys(
                sorted(groups),
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed + abs(hash((source, task))) % 100_000,
            )

            task_summary = {
                "total_records": len(rows),
                "total_source_files": len(groups),
                "splits": {},
            }
            for split_name, keys in split_keys.items():
                split_rows = [row for key in sorted(keys) for row in groups[key]]
                split_rows.sort(key=lambda x: str(x.get("sample_id", "")))
                write_jsonl(output_dir / source / split_name / f"{task}.jsonl", split_rows)
                task_summary["splits"][split_name] = {
                    "records": len(split_rows),
                    "source_files": len(keys),
                }
            summary[source][task] = task_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
