# -*- coding: utf-8 -*-
"""Generate multitask samples from canonical source files on demand.

The JSONL files are reproducible snapshots for evaluation. This generator is
the training-oriented path: it keeps task generation separate from the raw
source index and can balance domains and tasks without concatenating datasets.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = PROJECT_ROOT / "scripts" / "datasets" / "build_multitask_eval_datasets.py"
ENRICH_PATH = PROJECT_ROOT / "scripts" / "datasets" / "enrich_qar_phase_metadata.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("aero_engine_dataset_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module




def _load_phase_enricher():
    spec = importlib.util.spec_from_file_location("aero_engine_phase_enricher", ENRICH_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load phase enricher: {ENRICH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CanonicalTaskGenerator:
    """On-demand task generator over source-file-split canonical data."""

    def __init__(self, input_dir: str | Path = PROJECT_ROOT / "data", config=None, source_domain: str = "all"):
        self.builder = _load_builder()
        input_path = Path(input_dir)
        self.config = config or self.builder.BuildConfig(
            input_dir=str(input_path),
            output_dir=str(PROJECT_ROOT / "data" / "aero_engine_dataset"),
        )
        catalog = self.builder._load_source_catalog(input_path)
        self.sources = self.builder.load_source_series(input_path, self.config, catalog)
        if source_domain != "all":
            self.sources = [source for source in self.sources if source.source_domain == source_domain]
        phase_dir = Path(self.config.phase_segments_dir) if getattr(self.config, "phase_segments_dir", None) else input_path / "aero_engine_dataset" / "flight_phase_segments" / "segments"
        self.phase_enricher = _load_phase_enricher()
        self.phase_sidecars = self.phase_enricher._load_sidecars(phase_dir) if phase_dir.exists() else {}
        if not self.sources:
            raise ValueError("No usable canonical source groups found")

    def _domain_config(self, domain: str):
        context_length = self.builder._context_length_for_domain(self.config, domain)
        return replace(self.config, context_length=context_length)

    def _records_for_source(self, source, task: str, rng: np.random.Generator):
        builders = {
            "forecast": self.builder.build_forecast_records,
            "interpolation": self.builder.build_interpolation_records,
            "anomaly_detection": self.builder.build_anomaly_records,
        }
        if task not in builders:
            raise ValueError(f"Unsupported task: {task}")
        records = builders[task]([source], self._domain_config(source.source_domain), rng)
        enriched_records = []
        for record in records:
            record["engine_serial_id"] = record.get("source_engine_id")
            record["entity_id"] = (f"engine:{record['source_engine_id']}" if record.get("source_engine_id") else f"aircraft:{record.get('aircraft_id')}")
            record["entity_level"] = "engine" if record.get("source_engine_id") else "aircraft"
            if source.source_domain == "qar" and source.file_path.name in self.phase_sidecars:
                record = self.phase_enricher._enrich_record(record, self.phase_sidecars[source.file_path.name])
            enriched_records.append(record)
        return enriched_records

    def iter_records(self, task: str, split: str | None = "train", domain: str | None = None):
        """Yield records source-by-source without materializing a task JSONL."""
        selected = [source for source in self.sources if domain is None or source.source_domain == domain]
        counter = 0
        for source in selected:
            if split is not None and source.split != split:
                continue
            rng = np.random.default_rng(self.config.seed + counter + 17)
            for record in self._records_for_source(source, task, rng):
                record["sample_id"] = f"{source.source_domain}_{task}_online_{counter:08d}"
                counter += 1
                yield record

    def iter_balanced(
        self,
        steps: int,
        tasks: tuple[str, ...] = ("forecast", "interpolation", "anomaly_detection"),
        split: str | None = "train",
        domain_weights: dict[str, float] | None = None,
        task_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        """Yield balanced domain-task samples with replacement at sampler level."""
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        domains = sorted({source.source_domain for source in self.sources if split is None or source.split == split})
        domains = [domain for domain in domains if domain_weights is None or domain_weights.get(domain, 0.0) > 0]
        if not domains:
            raise ValueError("No domains available for balanced sampling")
        domain_prob = np.asarray(
            [domain_weights.get(domain, 1.0) if domain_weights else 1.0 for domain in domains], dtype=np.float64
        )
        domain_prob /= domain_prob.sum()
        task_prob = np.asarray([task_weights.get(task, 1.0) if task_weights else 1.0 for task in tasks], dtype=np.float64)
        task_prob /= task_prob.sum()
        source_by_domain = {
            domain: [source for source in self.sources if source.source_domain == domain and (split is None or source.split == split)]
            for domain in domains
        }
        for step in range(steps):
            domain = str(rng.choice(domains, p=domain_prob))
            task = str(rng.choice(tasks, p=task_prob))
            source = source_by_domain[domain][int(rng.integers(0, len(source_by_domain[domain])))]
            records = self._records_for_source(source, task, rng)
            if not records:
                continue
            record = records[int(rng.integers(0, len(records)))]
            record["sample_id"] = f"{domain}_{task}_balanced_{step:08d}"
            record["sampling_policy"] = "balanced_domain_task"
            yield record


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical aero-engine task samples on demand")
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--task", choices=["forecast", "interpolation", "anomaly_detection"], default="forecast")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--domain", choices=["all", "acars", "qar"], default="all")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    generator = CanonicalTaskGenerator(args.input_dir, source_domain=args.domain)
    for record in generator.iter_balanced(args.steps, tasks=(args.task,), split=args.split):
        print(record["sample_id"])


if __name__ == "__main__":
    main()
