# -*- coding: utf-8 -*-
"""Two-stage Chronos-2 adaptation on the canonical aero-engine corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("TSLM_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from chronos.chronos2.config import Chronos2CoreConfig
from tslm_multitask.models import Chronos2MultiTaskModel


TASKS = ("forecast", "interpolation", "anomaly_detection")
DOMAINS = ("acars", "qar")
METADATA_FIELDS = (
    "domain",
    "task",
    "phase",
    "channel",
    "subsystem",
    "engine",
    "time_scale",
    "feature_type",
    "target_role",
    "relation",
    "time_gap",
)


class JsonlIndex:
    """Byte-offset index for random access without materializing large JSONL files."""

    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = []
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def read(self, index: int) -> dict:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


class FilteredJsonlIndex:
    def __init__(self, base: JsonlIndex, indices: list[int]):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def read(self, index: int) -> dict:
        return self.base.read(self.indices[index])


def _row_schema_id(row: dict) -> str:
    explicit = row.get("schema_id")
    if explicit:
        return str(explicit)
    if str(row.get("source_domain", "")).lower() == "acars":
        return "ACARS"
    source_file = str(row.get("source_file", "")).upper()
    if source_file.startswith("B-2694_"):
        return "B-2694"
    if source_file.startswith("B-1400_"):
        return "B-1400"
    return "QAR-OTHER"


class CanonicalPools:
    def __init__(self, root: Path, split: str):
        self.pools: dict[tuple[str, str], JsonlIndex] = {}
        self.schema_pools: dict[tuple[str, str, str], FilteredJsonlIndex] = {}
        for domain in DOMAINS:
            for task in TASKS:
                path = root / "splits" / split / domain / f"{task}.jsonl"
                if path.exists() and path.stat().st_size:
                    pool = JsonlIndex(path)
                    if len(pool):
                        self.pools[(domain, task)] = pool
                        grouped = defaultdict(list)
                        for index in range(len(pool)):
                            grouped[_row_schema_id(pool.read(index))].append(index)
                        for schema_id, indices in grouped.items():
                            self.schema_pools[(domain, schema_id, task)] = FilteredJsonlIndex(pool, indices)

    def sample(
        self,
        rng: np.random.Generator,
        tasks: tuple[str, ...],
        domain_weights: dict[str, float],
        task_weights: dict[str, float],
        schema_weights: dict[str, float] | None = None,
    ) -> dict:
        domains = [
            domain for domain in DOMAINS
            if any((domain, schema, task) in self.schema_pools
                   for schema in self.schemas(domain) for task in tasks)
        ]
        domain_probs = np.asarray([domain_weights.get(domain, 0.0) for domain in domains], dtype=np.float64)
        domain_probs /= domain_probs.sum()
        domain = str(rng.choice(domains, p=domain_probs))
        available_schemas = self.schemas(domain, tasks)
        schema_probs = np.asarray(
            [(schema_weights or {}).get(schema, 1.0) for schema in available_schemas], dtype=np.float64
        )
        schema_probs /= schema_probs.sum()
        schema = str(rng.choice(available_schemas, p=schema_probs))
        available_tasks = [task for task in tasks if (domain, schema, task) in self.schema_pools]
        task_probs = np.asarray([task_weights.get(task, 0.0) for task in available_tasks], dtype=np.float64)
        task_probs /= task_probs.sum()
        task = str(rng.choice(available_tasks, p=task_probs))
        pool = self.schema_pools[(domain, schema, task)]
        return pool.read(int(rng.integers(0, len(pool))))

    def schemas(self, domain: str, tasks: tuple[str, ...] = TASKS) -> list[str]:
        return sorted({
            schema for candidate_domain, schema, task in self.schema_pools
            if candidate_domain == domain and task in tasks
        })


class MetadataVocabulary:
    def __init__(self, dataset_root: Path):
        self.values = {field: ["<UNK>"] for field in METADATA_FIELDS}
        self.index = {field: {"<UNK>": 0} for field in METADATA_FIELDS}
        schema = json.loads((dataset_root / "channel_schema.json").read_text(encoding="utf-8"))
        manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
        scaler_path = dataset_root / "train_scalers.json"
        scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8")) if scaler_path.exists() else {}
        self.train_scalers = scaler_payload.get("scalers", {})
        for value in ("ACARS", "QAR"):
            self.add("domain", value)
        for value in TASKS:
            self.add("task", value)
        for bucket in range(16):
            self.add("time_gap", f"gap_{bucket}")
        self.add("time_gap", "gap_unknown")
        for value in manifest.get("flight_segmentation", {}).get("phase_order", []):
            self.add("phase", value)
        for key, item in schema.get("channels", {}).items():
            self.add("channel", key)
            self.add("subsystem", item.get("subsystem"))
            self.add("engine", item.get("engine_id"))
            self.add("feature_type", item.get("feature_type"))
            self.add(
                "target_role",
                f"forecast_{item.get('forecast_tier', 'none')}_anomaly_{item.get('anomaly_tier', 'none')}",
            )
            for relation in item.get("relation_group_ids") or []:
                self.add("relation", relation)
        for profile in schema.get("sampling_profiles", {}).values():
            for token in profile.get("time_scale_tokens", []):
                self.add("time_scale", token)

    def smae_scale(self, row: dict, item: dict, values: torch.Tensor | None = None) -> torch.Tensor:
        domain = str(item.get("domain") or row.get("source_domain") or "").upper()
        phase = item.get("phase") or row.get("phase") or "UNSPECIFIED"
        key = (
            f"{domain}::{item.get('canonical_variable', item.get('physical_variable', 'UNKNOWN'))}::"
            f"{item.get('engine_id', 'global')}::{item.get('feature_type', 'raw')}::"
            f"{item.get('delta_semantics', 'none')}::{item.get('transformation', 'none')}::{phase}"
        )
        stats = self.train_scalers.get(key, {})
        preferred = (
            (stats.get("iqr"), stats.get("std"), stats.get("mean_abs"))
            if domain == "ACARS"
            else (stats.get("std"), stats.get("mean_abs"), stats.get("iqr"))
        )
        device = values.device if values is not None else torch.device("cpu")
        for candidate in preferred:
            try:
                candidate = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate > 1e-8:
                return torch.as_tensor(candidate, dtype=torch.float32, device=device)
        if values is not None:
            finite = values[torch.isfinite(values)]
            if finite.numel():
                scale = (finite - finite.median()).abs().median() * 1.4826
                if not torch.isfinite(scale) or scale <= 1e-8:
                    scale = finite.std(unbiased=False)
                if not torch.isfinite(scale) or scale <= 1e-8:
                    scale = finite.abs().mean()
                if torch.isfinite(scale) and scale > 1e-8:
                    return scale
        return torch.as_tensor(1.0, dtype=torch.float32, device=device)

    def add(self, field: str, value) -> int:
        value = "<UNK>" if value is None or value == "" else str(value)
        if value not in self.index[field]:
            self.index[field][value] = len(self.values[field])
            self.values[field].append(value)
        return self.index[field][value]

    def get(self, field: str, value) -> int:
        value = "<UNK>" if value is None or value == "" else str(value)
        return self.index[field].get(value, 0)

    @property
    def sizes(self) -> dict[str, int]:
        return {field: len(values) for field, values in self.values.items()}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.values, ensure_ascii=False, indent=2), encoding="utf-8")


def _float(value) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _matrix(mapping: dict, columns: list[str], dtype=np.float32) -> np.ndarray:
    return np.asarray([[_float(value) for value in mapping[column]] for column in columns], dtype=dtype)


def _mask(mapping: dict, columns: list[str]) -> np.ndarray:
    return np.asarray([mapping[column] for column in columns], dtype=bool)


def _time_gap_token(row: dict, column: str) -> str:
    interval = (row.get("sampling_interval") or {}).get(column)
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        interval = 0.0
    if not math.isfinite(interval) or interval <= 0:
        return "gap_unknown"
    bucket = min(15, max(0, int(math.ceil(math.log2(max(1.0, interval))))))
    return f"gap_{bucket}"


def _dominant_phase(row: dict) -> str:
    if row.get("phase"):
        return str(row["phase"])
    distribution = row.get("phase_distribution") or {}
    return str(max(distribution, key=distribution.get)) if distribution else "<UNK>"


def _metadata_ids(row: dict, vocab: MetadataVocabulary, device: torch.device) -> dict[str, torch.Tensor]:
    columns = list(row["source_columns"])
    domain = str(row["source_domain"]).upper()
    metadata = row.get("channel_metadata") or [{} for _ in columns]
    time_scales = row.get("time_scale_tokens") or {}
    phase = _dominant_phase(row)
    ids = defaultdict(list)
    for column, item in zip(columns, metadata):
        relations = item.get("relation_group_ids") or []
        ids["domain"].append(vocab.get("domain", domain))
        ids["task"].append(vocab.get("task", row["task_type"]))
        ids["phase"].append(vocab.get("phase", item.get("phase") or phase))
        ids["channel"].append(vocab.get("channel", f"{domain}::{column}"))
        ids["subsystem"].append(vocab.get("subsystem", item.get("subsystem") or row.get("subsystem")))
        ids["engine"].append(vocab.get("engine", item.get("engine_id")))
        ids["time_scale"].append(vocab.get("time_scale", time_scales.get(column)))
        ids["feature_type"].append(vocab.get("feature_type", item.get("feature_type")))
        ids["target_role"].append(
            vocab.get(
                "target_role",
                f"forecast_{item.get('forecast_tier', 'none')}_anomaly_{item.get('anomaly_tier', 'none')}",
            )
        )
        ids["relation"].append(vocab.get("relation", relations[0] if relations else None))
        ids["time_gap"].append(vocab.get("time_gap", _time_gap_token(row, column)))
    return {
        field: torch.as_tensor(ids[field], dtype=torch.long, device=device)
        for field in METADATA_FIELDS
    }


def record_to_batch(
    row: dict,
    vocab: MetadataVocabulary,
    device: torch.device,
    rng: np.random.Generator,
    masked_anomaly_probability: float,
) -> dict:
    columns = list(row["source_columns"])
    task = row["task_type"]
    result: dict[str, object] = {
        "task": task,
        "domain": row["source_domain"],
        "metadata_ids": _metadata_ids(row, vocab, device),
    }

    if task == "forecast":
        context = _matrix(row["history"], columns)
        context_mask = _mask(row["history_observation_mask"], columns) & _mask(
            row["history_quality_mask"], columns
        )
        future = _matrix(row["target_future"], columns)
        target_mask = (
            _mask(row["target_observation_mask"], columns)
            & _mask(row["target_quality_mask"], columns)
            & _mask(row["task_mask"], columns)
            & np.isfinite(future)
        )
        result.update(
            context=torch.as_tensor(context, device=device),
            context_mask=torch.as_tensor(context_mask, device=device),
            future_target=torch.as_tensor(future, device=device),
            future_target_mask=torch.as_tensor(target_mask, device=device),
            reconstruction_target=None,
            reconstruction_target_mask=None,
            reconstruction_mask=None,
        )
    elif task == "interpolation":
        context = _matrix(row["observed_context"], columns)
        context_mask = _mask(row["observed_observation_mask"], columns) & _mask(
            row["observed_quality_mask"], columns
        )
        target = context.copy()
        for channel_index, column in enumerate(columns):
            for index, value in zip(row["missing_indices"][column], row["target_values"][column]):
                target[channel_index, int(index)] = _float(value)
        target_mask = _mask(row["task_mask"], columns) & np.isfinite(target)
        result.update(
            context=torch.as_tensor(context, device=device),
            context_mask=torch.as_tensor(context_mask, device=device),
            future_target=None,
            future_target_mask=None,
            reconstruction_target=torch.as_tensor(target, device=device),
            reconstruction_target_mask=torch.as_tensor(target_mask, device=device),
            reconstruction_mask=torch.as_tensor(target_mask, device=device),
        )
    else:
        context = _matrix(row["observed_context"], columns)
        clean = _matrix(row["clean_context"], columns)
        context_mask = _mask(row["observation_mask"], columns) & _mask(row["quality_mask"], columns)
        target_mask = _mask(row["task_mask"], columns) & np.isfinite(clean)
        hide_anomaly = bool(rng.random() < masked_anomaly_probability)
        input_mask = target_mask if hide_anomaly else np.zeros_like(target_mask)
        if hide_anomaly:
            context[input_mask] = np.nan
            context_mask[input_mask] = False
        result.update(
            context=torch.as_tensor(context, device=device),
            context_mask=torch.as_tensor(context_mask, device=device),
            future_target=None,
            future_target_mask=None,
            reconstruction_target=torch.as_tensor(clean, device=device),
            reconstruction_target_mask=torch.as_tensor(target_mask, device=device),
            reconstruction_mask=torch.as_tensor(input_mask, device=device),
            anomaly_labels=torch.as_tensor(target_mask, device=device),
            evaluation_mask=torch.as_tensor(
                _mask(row["evaluation_mask"], columns) & np.isfinite(clean), device=device
            ),
        )

    series_count = len(columns)
    result["group_ids"] = torch.zeros(series_count, dtype=torch.long, device=device)
    result["num_output_patches"] = max(
        1,
        math.ceil(
            (result["future_target"].shape[-1] if result["future_target"] is not None else 1) / 16
        ),
    )
    return result


def _model_kwargs(batch: dict) -> dict:
    return {
        "context": batch["context"],
        "context_mask": batch["context_mask"],
        "group_ids": batch["group_ids"],
        "num_output_patches": batch["num_output_patches"],
        "future_target": batch["future_target"],
        "future_target_mask": batch["future_target_mask"],
        "metadata_ids": batch["metadata_ids"],
        "reconstruction_target": batch["reconstruction_target"],
        "reconstruction_target_mask": batch["reconstruction_target_mask"],
        "reconstruction_mask": batch["reconstruction_mask"],
    }


def configure_trainable(model: Chronos2MultiTaskModel, mode: str) -> dict[str, int]:
    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if mode == "reconstruction_head":
            prefixes = ["reconstruction_head"]
        else:
            prefixes = [
                "output_patch_embedding",
                "reconstruction_head",
                "metadata_embeddings",
                "metadata_gates",
            ]
            if mode == "adapter_blocks":
                prefixes.append("input_patch_embedding")
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                parameter.requires_grad = True
    return {
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def make_optimizer(model: Chronos2MultiTaskModel, backbone_lr: float, new_lr: float):
    new_parameters = []
    adapted_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("metadata_", "reconstruction_head")):
            new_parameters.append(parameter)
        else:
            adapted_parameters.append(parameter)
    groups = []
    if adapted_parameters:
        groups.append({"params": adapted_parameters, "lr": backbone_lr})
    if new_parameters:
        groups.append({"params": new_parameters, "lr": new_lr})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_stage(
    model: Chronos2MultiTaskModel,
    pools: CanonicalPools,
    vocab: MetadataVocabulary,
    optimizer,
    steps: int,
    stage: str,
    tasks: tuple[str, ...],
    task_weights: dict[str, float],
    args,
    global_step: int,
    history: list[dict],
) -> int:
    if steps <= 0:
        return global_step
    model.train()
    rng = np.random.default_rng(args.seed + global_step)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    running = defaultdict(float)
    counts = defaultdict(int)
    for local_step in range(1, steps + 1):
        row = pools.sample(
            rng,
            tasks,
            {"acars": args.acars_weight, "qar": args.qar_weight},
            task_weights,
            {"B-1400": args.qar_b1400_weight, "B-2694": args.qar_b2694_weight, "ACARS": 1.0},
        )
        batch = record_to_batch(row, vocab, model.device, rng, args.masked_anomaly_probability)
        task = str(batch["task"])
        model.forecast_loss_weight = 1.0 if task == "forecast" else 0.0
        model.recon_loss_weight = 0.0 if task == "forecast" else 1.0
        model.mask_ratio = 0.0
        with autocast_context(model.device, args.amp):
            output = model(**_model_kwargs(batch))
            loss = output.loss / args.gradient_accumulation
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {stage} step {local_step}: {loss.item()}")
        loss.backward()
        running[task] += float(loss.detach()) * args.gradient_accumulation
        counts[task] += 1

        if local_step % args.gradient_accumulation == 0 or local_step == steps:
            nonfinite = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            if nonfinite:
                raise FloatingPointError(
                    f"Non-finite gradients at {stage} step {local_step}: {nonfinite} elements"
                )
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if local_step % args.log_steps == 0 or local_step == steps:
            entry = {
                "stage": stage,
                "local_step": local_step,
                "global_optimizer_step": global_step,
                "elapsed_seconds": time.time() - started,
                "mean_loss_by_task": {
                    task_name: running[task_name] / max(counts[task_name], 1)
                    for task_name in sorted(counts)
                },
                "samples_by_task": dict(counts),
            }
            history.append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            running.clear()
            counts.clear()
    return global_step


def _median_index(model: Chronos2MultiTaskModel) -> int:
    return model.chronos_config.quantiles.index(0.5)


def _binary_auc(labels: list[int], scores: list[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positive = labels_array == 1
    negative = labels_array == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores_array, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(scores_array, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def _top_k_f1(labels: list[int], scores: list[float]) -> float:
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels_array.sum())
    if positive_count == 0 or positive_count == len(labels_array):
        return float("nan")
    predicted = np.zeros_like(labels_array)
    predicted[np.argpartition(scores_array, -positive_count)[-positive_count:]] = True
    true_positive = int((predicted & labels_array).sum())
    precision = true_positive / max(int(predicted.sum()), 1)
    recall = true_positive / positive_count
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def _channel_normalized_residual(
    prediction: torch.Tensor,
    target: torch.Tensor,
    evaluation_mask: torch.Tensor,
    quantile_preds: torch.Tensor | None = None,
    lower_index: int | None = None,
    upper_index: int | None = None,
) -> torch.Tensor:
    """Use predictive interval width first, with robust channel fallback."""
    residual = (prediction - target).abs()
    normalized = torch.full_like(residual, float("nan"))
    interval = None
    if quantile_preds is not None and lower_index is not None and upper_index is not None:
        interval = (
            quantile_preds[:, upper_index, :] - quantile_preds[:, lower_index, :]
        ).abs().clamp_min(1e-6)
    for channel_index in range(target.shape[0]):
        valid = evaluation_mask[channel_index] & torch.isfinite(target[channel_index])
        values = target[channel_index][valid]
        if not valid.any():
            continue
        fallback = (values - values.median()).abs().median() * 1.4826
        if not torch.isfinite(fallback) or fallback <= 1e-6:
            fallback = values.std(unbiased=False)
        if not torch.isfinite(fallback) or fallback <= 1e-6:
            fallback = values.abs().mean().clamp_min(1.0)
        if interval is None:
            scale = torch.full_like(residual[channel_index], fallback)
        else:
            scale = torch.where(
                torch.isfinite(interval[channel_index]) & (interval[channel_index] > 1e-6),
                interval[channel_index],
                torch.full_like(interval[channel_index], fallback),
            )
        normalized[channel_index] = residual[channel_index] / scale
    return normalized


def _robust_scale(values: torch.Tensor) -> torch.Tensor:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return torch.as_tensor(1.0, device=values.device if values.numel() else "cpu")
    scale = (values - values.median()).abs().median() * 1.4826
    if not torch.isfinite(scale) or scale <= 1e-6:
        scale = values.std(unbiased=False)
    if not torch.isfinite(scale) or scale <= 1e-6:
        scale = values.abs().mean().clamp_min(1.0)
    return scale


def _relation_residuals(
    row: dict,
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    evaluation_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cross_engine = torch.zeros_like(prediction)
    physical_response = torch.zeros_like(prediction)
    metadata = row.get("channel_metadata") or []
    index_by_name = {name: index for index, name in enumerate(row["source_columns"])}
    for relation, names in (row.get("relation_groups") or {}).items():
        indices = [index_by_name[name] for name in names if name in index_by_name]
        if len(indices) < 2:
            continue
        if relation.endswith("_PAIR"):
            engine_indices = [
                index for index in indices
                if metadata[index].get("engine_id") in {1, 2}
            ]
            if len(engine_indices) >= 2:
                left = next((index for index in engine_indices if metadata[index].get("engine_id") == 1), None)
                right = next((index for index in engine_indices if metadata[index].get("engine_id") == 2), None)
                if left is not None and right is not None:
                    valid = (
                        evaluation_mask[left] & evaluation_mask[right]
                        & torch.isfinite(target[left]) & torch.isfinite(target[right])
                    )
                    if valid.any():
                        target_delta = target[left] - target[right]
                        prediction_delta = prediction[left] - prediction[right]
                        scale = _robust_scale(target_delta[valid])
                        relation_score = (target_delta - prediction_delta).abs() / scale
                        cross_engine[left] = torch.where(valid, relation_score, cross_engine[left])
                        cross_engine[right] = torch.where(valid, relation_score, cross_engine[right])
        elif "RESPONSE_ENGINE_" in relation:
            if "N1_COMMAND_RESPONSE_ENGINE_" in relation:
                condition_variables = {"N1_COMMAND", "N1_TARGET"}
            elif "TRA_RESPONSE_ENGINE_" in relation:
                condition_variables = {"TRA"}
            elif "FMV_RESPONSE_ENGINE_" in relation:
                condition_variables = {"FMV_POSITION"}
            elif "N2_CONTROL_RESPONSE_ENGINE_" in relation:
                condition_variables = {"VSV_POSITION", "VBV_POSITION"}
            elif "N2_RESPONSE_ENGINE_" in relation:
                condition_variables = {"N2"}
            elif "DUCT_PRESSURE_ENGINE_" in relation:
                condition_variables = {"BLEED_SWITCH"}
            else:
                condition_variables = set()
            response_indices = [
                index for index in indices
                if metadata[index].get("anomaly_target")
                and metadata[index].get("physical_variable") not in condition_variables
            ]
            condition_indices = [
                index for index in indices
                if metadata[index].get("physical_variable") in condition_variables
            ]
            if response_indices and condition_indices:
                response = response_indices[0]
                condition = condition_indices[0]
                valid = (
                    evaluation_mask[response]
                    & torch.isfinite(target[response])
                    & torch.isfinite(target[condition])
                )
                if valid.any():
                    target_response = target[response] - target[condition]
                    prediction_response = prediction[response] - target[condition]
                    scale = _robust_scale(target_response[valid])
                    physical_response[response] = torch.where(
                        valid,
                        (target_response - prediction_response).abs() / scale,
                        physical_response[response],
                    )
    return cross_engine, physical_response


def evaluate(model, pools: CanonicalPools, vocab: MetadataVocabulary, args) -> dict:
    model.eval()
    rng = np.random.default_rng(args.seed + 900_001)
    aggregates = defaultdict(list)
    feature_aggregates = defaultdict(lambda: defaultdict(list))
    feature_metadata = {}
    threshold_groups: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def feature_key(domain: str, row: dict, channel_index: int) -> str:
        column = row["source_columns"][channel_index]
        item = (row.get("channel_metadata") or [])[channel_index]
        key = f"{domain}|{item.get('physical_variable', column)}|{column}"
        feature_metadata[key] = {
            "domain": domain,
            "channel": column,
            "physical_variable": item.get("physical_variable", column),
            "engine_id": item.get("engine_id"),
            "forecast_tier": item.get("forecast_tier", "none"),
            "anomaly_tier": item.get("anomaly_tier", "none"),
        }
        return key
    with torch.no_grad():
        for (domain, task), pool in pools.pools.items():
            for index in range(min(args.eval_records, len(pool))):
                row = pool.read(index)
                batch = record_to_batch(row, vocab, model.device, rng, 1.0)
                if task == "forecast":
                    model.forecast_loss_weight = 1.0
                    model.recon_loss_weight = 0.0
                    output = model(**_model_kwargs(batch))
                    prediction = output.quantile_preds[:, _median_index(model), :]
                    target = batch["future_target"]
                    mask = batch["future_target_mask"] & torch.isfinite(prediction)
                    if mask.any():
                        for channel_index, item in enumerate(row.get("channel_metadata") or []):
                            channel_mask = mask[channel_index]
                            if not channel_mask.any():
                                continue
                            channel_error = (prediction[channel_index][channel_mask] - target[channel_index][channel_mask]).abs()
                            scale = vocab.smae_scale(row, item, target[channel_index][channel_mask])
                            channel_smae = channel_error / scale
                            channel_smape = 2 * channel_error / (
                                prediction[channel_index][channel_mask].abs() + target[channel_index][channel_mask].abs()
                            ).clamp_min(1e-6)
                            aggregates[(domain, task, "smae")].extend(channel_smae.cpu().tolist())
                            aggregates[(domain, task, "smape")].extend(channel_smape.cpu().tolist())
                            key = feature_key(domain, row, channel_index)
                            feature_aggregates[key]["forecast_smae"].extend(channel_smae.cpu().tolist())
                            feature_aggregates[key]["forecast_smape"].extend(channel_smape.cpu().tolist())
                            tier = item.get("forecast_tier", "none")
                            if tier != "none":
                                aggregates[(domain, task, f"smae_{tier}")].extend(channel_smae.cpu().tolist())
                else:
                    if task == "anomaly_detection" and args.anomaly_inference == "lopo":
                        lopo = model.reconstruct_context_lopo(
                            context=batch["context"],
                            context_mask=batch["context_mask"],
                            group_ids=batch["group_ids"],
                            metadata_ids=batch["metadata_ids"],
                        )
                        length = lopo.shape[-1]
                        patch_index = torch.arange(length, device=lopo.device) // int(model.chronos_config.input_patch_size)
                        patch_index = patch_index.clamp_max(lopo.shape[1] - 1)
                        batch_index = torch.arange(lopo.shape[0], device=lopo.device)[:, None]
                        time_index = torch.arange(length, device=lopo.device)[None, :]
                        quantile_preds = lopo[batch_index, patch_index[None, :], :, time_index]
                        quantile_preds = quantile_preds.permute(0, 2, 1)
                    else:
                        quantile_preds = model.reconstruct_context(
                            context=batch["context"],
                            context_mask=batch["context_mask"],
                            reconstruction_mask=batch["reconstruction_mask"],
                            group_ids=batch["group_ids"],
                            metadata_ids=batch["metadata_ids"],
                        )
                    prediction = quantile_preds[:, _median_index(model), :]
                    target = batch["reconstruction_target"]
                    mask = batch["reconstruction_target_mask"] & torch.isfinite(prediction)
                    if mask.any():
                        for channel_index, item in enumerate(row.get("channel_metadata") or []):
                            channel_mask = mask[channel_index]
                            if not channel_mask.any():
                                continue
                            channel_error = (prediction[channel_index][channel_mask] - target[channel_index][channel_mask]).abs()
                            scale = vocab.smae_scale(row, item, target[channel_index][channel_mask])
                            channel_smae = channel_error / scale
                            channel_smape = 2 * channel_error / (
                                prediction[channel_index][channel_mask].abs() + target[channel_index][channel_mask].abs()
                            ).clamp_min(1e-6)
                            aggregates[(domain, task, "smae")].extend(channel_smae.cpu().tolist())
                            aggregates[(domain, task, "smape")].extend(channel_smape.cpu().tolist())
                            key = feature_key(domain, row, channel_index)
                            feature_aggregates[key]["interpolation_smae"].extend(channel_smae.cpu().tolist())
                            feature_aggregates[key]["interpolation_smape"].extend(channel_smape.cpu().tolist())
                            tier = item.get("anomaly_tier", "none")
                            if tier != "none":
                                aggregates[(domain, task, f"smae_{tier}")].extend(channel_smae.cpu().tolist())
                    if task == "anomaly_detection":
                        evaluation_mask = batch["evaluation_mask"] & torch.isfinite(prediction)
                        temporal_residual = _channel_normalized_residual(
                            prediction,
                            target,
                            evaluation_mask,
                            quantile_preds=quantile_preds,
                            lower_index=min(range(model.num_quantiles), key=lambda i: abs(model.chronos_config.quantiles[i] - 0.1)),
                            upper_index=min(range(model.num_quantiles), key=lambda i: abs(model.chronos_config.quantiles[i] - 0.9)),
                        )
                        cross_engine_residual, physical_response_residual = _relation_residuals(
                            row,
                            prediction,
                            target,
                            batch["context"],
                            evaluation_mask,
                        )
                        combined_residual = (
                            temporal_residual
                            + float(getattr(args, "cross_engine_weight", 0.35)) * cross_engine_residual
                            + float(getattr(args, "cross_variable_weight", 0.50)) * physical_response_residual
                        )
                        scores = combined_residual[evaluation_mask]
                        labels = batch["anomaly_labels"][evaluation_mask]
                        aggregates[(domain, task, "scores")].extend(scores.cpu().tolist())
                        aggregates[(domain, task, "labels")].extend(labels.long().cpu().tolist())
                        aggregates[(domain, task, "temporal_scores")].extend(
                            temporal_residual[evaluation_mask].cpu().tolist()
                        )
                        aggregates[(domain, task, "cross_engine_scores")].extend(
                            cross_engine_residual[evaluation_mask].cpu().tolist()
                        )
                        aggregates[(domain, task, "physical_response_scores")].extend(
                            physical_response_residual[evaluation_mask].cpu().tolist()
                        )
                        for channel_index, item in enumerate(row.get("channel_metadata") or []):
                            channel_mask = evaluation_mask[channel_index]
                            tier = item.get("anomaly_tier", "none")
                            if channel_mask.any() and tier != "none":
                                key = feature_key(domain, row, channel_index)
                                feature_aggregates[key]["anomaly_scores"].extend(
                                    combined_residual[channel_index][channel_mask].cpu().tolist()
                                )
                                feature_aggregates[key]["anomaly_labels"].extend(
                                    batch["anomaly_labels"][channel_index][channel_mask].long().cpu().tolist()
                                )
                                feature_aggregates[key]["temporal_scores"].extend(
                                    temporal_residual[channel_index][channel_mask].cpu().tolist()
                                )
                                feature_aggregates[key]["cross_engine_scores"].extend(
                                    cross_engine_residual[channel_index][channel_mask].cpu().tolist()
                                )
                                feature_aggregates[key]["physical_response_scores"].extend(
                                    physical_response_residual[channel_index][channel_mask].cpu().tolist()
                                )
                                aggregates[(domain, task, f"scores_{tier}")].extend(
                                    combined_residual[channel_index][channel_mask].cpu().tolist()
                                )
                                aggregates[(domain, task, f"labels_{tier}")].extend(
                                    batch["anomaly_labels"][channel_index][channel_mask].long().cpu().tolist()
                                )
                        phase = _dominant_phase(row)
                        for channel_index, column in enumerate(row["source_columns"]):
                            channel_mask = evaluation_mask[channel_index]
                            channel_scores = combined_residual[channel_index][channel_mask]
                            channel_labels = batch["anomaly_labels"][channel_index][channel_mask]
                            key = f"{domain}|{column}|{phase}"
                            threshold_groups[key].extend(
                                zip(channel_scores.cpu().tolist(), channel_labels.long().cpu().tolist())
                            )
    metrics = {}
    for (domain, task, metric), values in aggregates.items():
        if "scores" in metric or "labels" in metric:
            continue
        metrics[f"{domain}/{task}/{metric}"] = float(np.mean(values)) if values else float("nan")
    thresholds: dict[str, float] = {}
    threshold_scores: list[float] = []
    threshold_labels: list[int] = []
    for key, pairs in threshold_groups.items():
        normal_scores = [score for score, label in pairs if label == 0 and math.isfinite(score)]
        if not normal_scores:
            continue
        threshold = float(np.percentile(normal_scores, 99.0))
        thresholds[key] = threshold
        for score, label in pairs:
            if math.isfinite(score):
                threshold_scores.append(score)
                threshold_labels.append(int(label))
    if thresholds and threshold_scores:
        predicted = [
            score > thresholds.get(key, float("inf"))
            for key, pairs in threshold_groups.items()
            for score, _ in pairs
            if math.isfinite(score)
        ]
        labels_flat = [
            int(label)
            for key, pairs in threshold_groups.items()
            for score, label in pairs
            if math.isfinite(score)
        ]
        tp = sum(int(pred and label) for pred, label in zip(predicted, labels_flat))
        fp = sum(int(pred and not label) for pred, label in zip(predicted, labels_flat))
        fn = sum(int((not pred) and label) for pred, label in zip(predicted, labels_flat))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics["anomaly_detection/threshold_f1"] = float(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
        metrics["anomaly_thresholds"] = thresholds
    for domain in DOMAINS:
        scores = aggregates.get((domain, "anomaly_detection", "scores"), [])
        labels = aggregates.get((domain, "anomaly_detection", "labels"), [])
        if scores and labels:
            metrics[f"{domain}/anomaly_detection/auroc"] = _binary_auc(labels, scores)
            metrics[f"{domain}/anomaly_detection/top_k_f1"] = _top_k_f1(labels, scores)
            for component, metric_name in (
                ("temporal_scores", "temporal_auroc"),
                ("cross_engine_scores", "cross_engine_auroc"),
                ("physical_response_scores", "physical_response_auroc"),
            ):
                component_scores = aggregates.get((domain, "anomaly_detection", component), [])
                if component_scores and len(component_scores) == len(labels):
                    metrics[f"{domain}/anomaly_detection/{metric_name}"] = _binary_auc(labels, component_scores)
            for tier in ("core", "secondary"):
                tier_scores = aggregates.get((domain, "anomaly_detection", f"scores_{tier}"), [])
                tier_labels = aggregates.get((domain, "anomaly_detection", f"labels_{tier}"), [])
                if tier_scores and tier_labels:
                    metrics[f"{domain}/anomaly_detection/{tier}_auroc"] = _binary_auc(tier_labels, tier_scores)
                    metrics[f"{domain}/anomaly_detection/{tier}_top_k_f1"] = _top_k_f1(tier_labels, tier_scores)
    per_feature = {}
    for key, values in feature_aggregates.items():
        entry = dict(feature_metadata[key])
        if values.get("forecast_smae"):
            entry["forecast"] = {
                "count": len(values["forecast_smae"]),
                "smae": float(np.mean(values["forecast_smae"])),
                "smape": float(np.mean(values["forecast_smape"])),
            }
        if values.get("interpolation_smae"):
            entry["interpolation"] = {
                "count": len(values["interpolation_smae"]),
                "smae": float(np.mean(values["interpolation_smae"])),
                "smape": float(np.mean(values["interpolation_smape"])),
            }
        if values.get("anomaly_scores") and values.get("anomaly_labels"):
            entry["anomaly_detection"] = {
                "count": len(values["anomaly_scores"]),
                "positive_count": int(sum(values["anomaly_labels"])),
                "auroc": _binary_auc(values["anomaly_labels"], values["anomaly_scores"]),
                "top_k_f1": _top_k_f1(values["anomaly_labels"], values["anomaly_scores"]),
                "temporal_auroc": _binary_auc(values["anomaly_labels"], values["temporal_scores"]),
                "cross_engine_auroc": _binary_auc(values["anomaly_labels"], values["cross_engine_scores"]),
                "physical_response_auroc": _binary_auc(values["anomaly_labels"], values["physical_response_scores"]),
            }
        per_feature[key] = entry
    metrics["metric_definition"] = {
        "smae": "mean(abs(prediction - target) / train_only_channel_scale)",
        "scale_selection": "ACARS IQR then std/mean_abs; QAR std then mean_abs/IQR",
        "smape": "2 * abs(prediction - target) / max(abs(prediction) + abs(target), 1e-6)",
    }
    metrics["per_feature"] = per_feature
    return metrics


def save_checkpoint(model, output_dir: Path, vocab: MetadataVocabulary, args, history, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.architectures = ["Chronos2MultiTaskModel"]
    model.config.aero_metadata_vocab_sizes = vocab.sizes
    model.save_pretrained(output_dir)
    vocab.save(output_dir / "metadata_vocab.json")
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage canonical aero-engine Chronos-2 training")
    parser.add_argument("--dataset-dir", default="data/aero_engine_dataset")
    parser.add_argument("--model-path", default="weight/chronos-2")
    parser.add_argument("--output-dir", default="weights/chronos2_aero_canonical_multitask")
    parser.add_argument("--stage1-steps", type=int, default=1000)
    parser.add_argument("--stage2-steps", type=int, default=2000)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--new-module-learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--trainable-mode",
        choices=["reconstruction_head", "heads", "adapter_blocks", "full"],
        default="adapter_blocks",
    )
    parser.add_argument("--stage2-trainable-mode", choices=["same", "reconstruction_head", "heads", "adapter_blocks", "full"], default="reconstruction_head")
    parser.add_argument("--acars-weight", type=float, default=0.5)
    parser.add_argument("--qar-weight", type=float, default=0.5)
    parser.add_argument("--qar-b1400-weight", type=float, default=0.5)
    parser.add_argument("--qar-b2694-weight", type=float, default=0.5)
    parser.add_argument("--forecast-weight", type=float, default=0.40)
    parser.add_argument("--interpolation-weight", type=float, default=0.35)
    parser.add_argument("--anomaly-weight", type=float, default=0.25)
    parser.add_argument("--masked-anomaly-probability", type=float, default=0.70)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--eval-records", type=int, default=24)
    parser.add_argument("--anomaly-inference", choices=["direct", "lopo"], default="lopo")
    parser.add_argument("--cross-engine-weight", type=float, default=0.35)
    parser.add_argument("--cross-variable-weight", type=float, default=0.50)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_root = (PROJECT_ROOT / args.dataset_dir).resolve()
    model_path = (PROJECT_ROOT / args.model_path).resolve()
    if not model_path.exists() and args.model_path == "weight/chronos-2":
        fallback = PROJECT_ROOT / "weights" / "chronos-2"
        if fallback.exists():
            model_path = fallback.resolve()
    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    vocab = MetadataVocabulary(dataset_root)
    source_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    source_architectures = source_config.get("architectures") or []
    loading_base_chronos = "Chronos2MultiTaskModel" not in source_architectures
    config = Chronos2CoreConfig.from_pretrained(model_path)
    config.aero_metadata_vocab_sizes = vocab.sizes
    config.architectures = ["Chronos2MultiTaskModel"]
    model = Chronos2MultiTaskModel.from_pretrained(model_path, config=config).to(device)
    if loading_base_chronos:
        model.initialize_aero_modules_from_base()
    parameter_stats = configure_trainable(model, args.trainable_mode)
    print(
        json.dumps(
            {
                "device": str(device),
                "parameter_stats": parameter_stats,
                "reconstruction_head_initialized_from_forecast_head": loading_base_chronos,
            },
            indent=2,
        )
    )

    train_pools = CanonicalPools(dataset_root, "train")
    val_pools = CanonicalPools(dataset_root, "val")
    optimizer = make_optimizer(model, args.backbone_learning_rate, args.new_module_learning_rate)
    history: list[dict] = []
    baseline_metrics = evaluate(model, val_pools, vocab, args)
    (output_root / "baseline_metrics.json").write_text(
        json.dumps(baseline_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"baseline_metrics": baseline_metrics}, ensure_ascii=False, indent=2))
    global_step = train_stage(
        model,
        train_pools,
        vocab,
        optimizer,
        args.stage1_steps,
        "domain_forecast",
        ("forecast",),
        {"forecast": 1.0},
        args,
        0,
        history,
    )
    stage1_metrics = evaluate(model, val_pools, vocab, args)
    save_checkpoint(model, output_root / "checkpoint-stage1", vocab, args, history, stage1_metrics)

    if args.stage2_trainable_mode != "same":
        configure_trainable(model, args.stage2_trainable_mode)
        optimizer = make_optimizer(model, args.backbone_learning_rate, args.new_module_learning_rate)

    global_step = train_stage(
        model,
        train_pools,
        vocab,
        optimizer,
        args.stage2_steps,
        "multitask",
        TASKS,
        {
            "forecast": args.forecast_weight,
            "interpolation": args.interpolation_weight,
            "anomaly_detection": args.anomaly_weight,
        },
        args,
        global_step,
        history,
    )
    final_metrics = evaluate(model, val_pools, vocab, args)
    final_metrics["global_optimizer_steps"] = global_step
    save_checkpoint(model, output_root / "checkpoint-final", vocab, args, history, final_metrics)
    print(json.dumps({"saved": str(output_root / "checkpoint-final"), "metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
