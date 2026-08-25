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
    "relation",
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


class CanonicalPools:
    def __init__(self, root: Path, split: str):
        self.pools: dict[tuple[str, str], JsonlIndex] = {}
        for domain in DOMAINS:
            for task in TASKS:
                path = root / "splits" / split / domain / f"{task}.jsonl"
                if path.exists() and path.stat().st_size:
                    pool = JsonlIndex(path)
                    if len(pool):
                        self.pools[(domain, task)] = pool

    def sample(
        self,
        rng: np.random.Generator,
        tasks: tuple[str, ...],
        domain_weights: dict[str, float],
        task_weights: dict[str, float],
    ) -> dict:
        domains = [domain for domain in DOMAINS if any((domain, task) in self.pools for task in tasks)]
        domain_probs = np.asarray([domain_weights.get(domain, 0.0) for domain in domains], dtype=np.float64)
        domain_probs /= domain_probs.sum()
        domain = str(rng.choice(domains, p=domain_probs))
        available_tasks = [task for task in tasks if (domain, task) in self.pools]
        task_probs = np.asarray([task_weights.get(task, 0.0) for task in available_tasks], dtype=np.float64)
        task_probs /= task_probs.sum()
        task = str(rng.choice(available_tasks, p=task_probs))
        pool = self.pools[(domain, task)]
        return pool.read(int(rng.integers(0, len(pool))))


class MetadataVocabulary:
    def __init__(self, dataset_root: Path):
        self.values = {field: ["<UNK>"] for field in METADATA_FIELDS}
        self.index = {field: {"<UNK>": 0} for field in METADATA_FIELDS}
        schema = json.loads((dataset_root / "channel_schema.json").read_text(encoding="utf-8"))
        manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
        for value in ("ACARS", "QAR"):
            self.add("domain", value)
        for value in TASKS:
            self.add("task", value)
        for value in manifest.get("flight_segmentation", {}).get("phase_order", []):
            self.add("phase", value)
        for key, item in schema.get("channels", {}).items():
            self.add("channel", key)
            self.add("subsystem", item.get("subsystem"))
            self.add("engine", item.get("engine_id"))
            self.add("feature_type", item.get("feature_type"))
            for relation in item.get("relation_group_ids") or []:
                self.add("relation", relation)
        for profile in schema.get("sampling_profiles", {}).values():
            for token in profile.get("time_scale_tokens", []):
                self.add("time_scale", token)

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
        ids["relation"].append(vocab.get("relation", relations[0] if relations else None))
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
) -> torch.Tensor:
    """Scale residuals per channel so heterogeneous units do not dominate ranking."""
    residual = (prediction - target).abs()
    normalized = torch.full_like(residual, float("nan"))
    for channel_index in range(target.shape[0]):
        valid = evaluation_mask[channel_index] & torch.isfinite(target[channel_index])
        values = target[channel_index][valid]
        if not valid.any():
            continue
        center = values.median()
        scale = (values - center).abs().median() * 1.4826
        if not torch.isfinite(scale) or scale <= 1e-6:
            scale = values.std(unbiased=False)
        if not torch.isfinite(scale) or scale <= 1e-6:
            scale = values.abs().mean().clamp_min(1.0)
        normalized[channel_index] = residual[channel_index] / scale
    return normalized


def evaluate(model, pools: CanonicalPools, vocab: MetadataVocabulary, args) -> dict:
    model.eval()
    rng = np.random.default_rng(args.seed + 900_001)
    aggregates = defaultdict(list)
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
                        error = (prediction[mask] - target[mask]).abs()
                        smape = 2 * error / (prediction[mask].abs() + target[mask].abs()).clamp_min(1e-6)
                        aggregates[(domain, task, "mae")].extend(error.cpu().tolist())
                        aggregates[(domain, task, "smape")].extend(smape.cpu().tolist())
                else:
                    prediction = model.reconstruct_context(
                        context=batch["context"],
                        context_mask=batch["context_mask"],
                        reconstruction_mask=batch["reconstruction_mask"],
                        group_ids=batch["group_ids"],
                        metadata_ids=batch["metadata_ids"],
                    )[:, _median_index(model), :]
                    target = batch["reconstruction_target"]
                    mask = batch["reconstruction_target_mask"] & torch.isfinite(prediction)
                    if mask.any():
                        error = (prediction[mask] - target[mask]).abs()
                        smape = 2 * error / (prediction[mask].abs() + target[mask].abs()).clamp_min(1e-6)
                        aggregates[(domain, task, "mae")].extend(error.cpu().tolist())
                        aggregates[(domain, task, "smape")].extend(smape.cpu().tolist())
                    if task == "anomaly_detection":
                        evaluation_mask = batch["evaluation_mask"] & torch.isfinite(prediction)
                        normalized_residual = _channel_normalized_residual(
                            prediction, target, evaluation_mask
                        )
                        scores = normalized_residual[evaluation_mask]
                        labels = batch["anomaly_labels"][evaluation_mask]
                        aggregates[(domain, task, "scores")].extend(scores.cpu().tolist())
                        aggregates[(domain, task, "labels")].extend(labels.long().cpu().tolist())
    metrics = {}
    for (domain, task, metric), values in aggregates.items():
        if metric in {"scores", "labels"}:
            continue
        metrics[f"{domain}/{task}/{metric}"] = float(np.mean(values)) if values else float("nan")
    for domain in DOMAINS:
        scores = aggregates.get((domain, "anomaly_detection", "scores"), [])
        labels = aggregates.get((domain, "anomaly_detection", "labels"), [])
        if scores and labels:
            metrics[f"{domain}/anomaly_detection/auroc"] = _binary_auc(labels, scores)
            metrics[f"{domain}/anomaly_detection/top_k_f1"] = _top_k_f1(labels, scores)
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
    parser.add_argument("--acars-weight", type=float, default=0.5)
    parser.add_argument("--qar-weight", type=float, default=0.5)
    parser.add_argument("--forecast-weight", type=float, default=0.40)
    parser.add_argument("--interpolation-weight", type=float, default=0.35)
    parser.add_argument("--anomaly-weight", type=float, default=0.25)
    parser.add_argument("--masked-anomaly-probability", type=float, default=0.70)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--eval-records", type=int, default=24)
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
