# -*- coding: utf-8 -*-
"""Run QAR phase segmentation and multitask dataset construction together.

The builder writes the canonical/task JSONL snapshots, so phase metadata is
attached in a final enrichment pass after every rebuild.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    segmenter = _load("qar_phase_segmenter", ROOT / "scripts" / "datasets" / "segment_qar_flight_phases.py")
    builder = _load("aero_engine_dataset_builder", ROOT / "scripts" / "datasets" / "build_multitask_eval_datasets.py")
    enricher = _load("aero_engine_phase_enricher", ROOT / "scripts" / "datasets" / "enrich_qar_phase_metadata.py")

    phase_dir = output_dir / "flight_phase_segments"
    phase_summary = segmenter.run(input_dir, phase_dir, args.plot_file)
    config = builder.BuildConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        context_length=args.context_length,
        acars_context_length=args.acars_context_length,
        qar_context_length=args.qar_context_length,
        max_group_channels=args.max_group_channels,
        max_condition_channels=args.max_condition_channels,
        prediction_length=args.prediction_length,
        windows_per_source=args.windows_per_source,
        interpolation_mask_ratio=args.interpolation_mask_ratio,
        interpolation_mode=args.interpolation_mode,
        anomaly_ratio=args.anomaly_ratio,
        anomaly_sigma=args.anomaly_sigma,
        anomaly_mode=args.anomaly_mode,
        min_valid_points=args.min_valid_points,
        seed=args.seed,
        source_domain=args.source_domain,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    dataset_summary = builder.build_datasets(config)
    enrichment_summary = enricher.run(output_dir, phase_dir / "segments")
    return {
        "phase_segmentation": {
            "source_count": phase_summary["source_count"],
            "output_dir": str(phase_dir),
        },
        "dataset": {
            "source_file_count": dataset_summary["source_file_count"],
            "source_group_count": dataset_summary["source_group_count"],
            "output_dir": str(output_dir),
        },
        "phase_enrichment": enrichment_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the aero-engine dataset with QAR phase metadata")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default="data/aero_engine_dataset")
    parser.add_argument("--plot-file", action="append", default=[])
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--acars-context-length", type=int, default=None)
    parser.add_argument("--qar-context-length", type=int, default=None)
    parser.add_argument("--max-group-channels", type=int, default=16)
    parser.add_argument("--max-condition-channels", type=int, default=4)
    parser.add_argument("--prediction-length", type=int, default=16)
    parser.add_argument("--windows-per-source", type=int, default=1)
    parser.add_argument("--interpolation-mask-ratio", type=float, default=0.2)
    parser.add_argument("--interpolation-mode", choices=["random_point", "block", "periodic", "channel_dropout", "mixed"], default="random_point")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--anomaly-sigma", type=float, default=4.0)
    parser.add_argument("--anomaly-mode", choices=["spike", "bias", "drift", "frozen", "variance", "cross_channel", "mixed"], default="spike")
    parser.add_argument("--min-valid-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-domain", choices=["all", "acars", "qar"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))



