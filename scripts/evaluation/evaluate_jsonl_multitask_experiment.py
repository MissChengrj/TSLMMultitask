# -*- coding: utf-8 -*-
"""Evaluate Chronos2 and classic baselines on JSONL multitask splits."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("TSLM_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.evaluation.evaluate_multitask_checkpoint import load_checkpoint
from tslm_multitask.models import Chronos2MultiTaskModel


TASKS = ("forecast", "interpolation", "anomaly_detection")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def finite_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def matrix_from_mapping(mapping: dict, columns: list[str]) -> np.ndarray:
    return np.asarray([[finite_float(x) for x in mapping[col]] for col in columns], dtype=np.float32)


def fill_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    if values.ndim == 1:
        x = np.arange(len(values))
        good = np.isfinite(values)
        if good.any():
            values[~good] = np.interp(x[~good], x[good], values[good]).astype(np.float32)
            values[: int(x[good][0])] = values[int(x[good][0])]
            values[int(x[good][-1]) + 1 :] = values[int(x[good][-1])]
        else:
            values[:] = 0.0
        return values
    return np.stack([fill_linear(row) for row in values]).astype(np.float32)


def norm1d(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(x)
    if finite.any():
        mu = float(np.nanmean(x))
        sigma = float(np.nanstd(x))
    else:
        mu, sigma = 0.0, 1.0
    sigma = sigma if sigma > 1e-6 else 1.0
    return ((fill_linear(x) - mu) / sigma).astype(np.float32), mu, sigma


def robust_score_scale(scores: np.ndarray) -> np.ndarray:
    """Put heterogeneous anomaly scores on a comparable robust scale."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    scaled = np.zeros_like(scores, dtype=np.float64)
    if not finite.any():
        return scaled
    values = scores[finite]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.1, 1e-6)
    scaled[finite] = np.clip((values - median) / scale, 0.0, 20.0)
    return scaled.astype(np.float32)


def rolling_zscore(values: np.ndarray) -> np.ndarray:
    filled = fill_linear(values)
    baseline = pd.Series(filled).rolling(16, min_periods=4, center=True).median().bfill().ffill().to_numpy()
    resid = np.abs(filled - baseline)
    scale = pd.Series(resid).rolling(32, min_periods=8, center=True).std().bfill().ffill().to_numpy()
    return np.nan_to_num(resid / (scale + 1e-6), nan=0.0, posinf=20.0, neginf=0.0)


def mse_mae(preds: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
    if not preds:
        return {}
    p = np.concatenate([np.ravel(x) for x in preds]).astype(np.float64)
    y = np.concatenate([np.ravel(x) for x in targets]).astype(np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return {}
    err = p[mask] - y[mask]
    smape = 2.0 * np.abs(err) / np.maximum(np.abs(p[mask]) + np.abs(y[mask]), 1e-6)
    return {
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "smape": float(np.mean(smape)),
        "points": int(mask.sum()),
    }


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def anomaly_metrics(labels_list: list[np.ndarray], scores_list: list[np.ndarray]) -> dict[str, float]:
    if not labels_list:
        return {}
    labels = np.concatenate([np.ravel(x).astype(bool) for x in labels_list])
    scores = np.concatenate([np.ravel(x).astype(np.float64) for x in scores_list])
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    positives = int(labels.sum())
    if len(labels) == 0 or positives == 0:
        return {}
    k = min(positives, len(scores))
    pred = np.zeros(len(scores), dtype=bool)
    pred[np.argsort(scores)[-k:]] = True
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision_topk": float(precision),
        "recall_topk": float(recall),
        "f1_topk": float(f1),
        "auroc": rank_auc(labels, scores),
        "points": int(len(labels)),
        "positives": positives,
    }


class LSTMForecast(nn.Module):
    def __init__(self, pred_len: int, hidden: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x):
        _, (h, _) = self.rnn(x.unsqueeze(-1))
        return self.head(h[-1])


class TransformerForecast(nn.Module):
    def __init__(self, pred_len: int, hidden: int = 32):
        super().__init__()
        self.inp = nn.Linear(1, hidden)
        enc_layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=64, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x):
        z = self.enc(self.inp(x.unsqueeze(-1)))
        return self.head(z[:, -1])


class LSTMAutoEncoder(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        z, _ = self.rnn(x.unsqueeze(-1))
        return self.head(z).squeeze(-1)


class TransformerAutoEncoder(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.inp = nn.Linear(1, hidden)
        enc_layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=64, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.enc(self.inp(x.unsqueeze(-1)))).squeeze(-1)


@dataclass
class NeuralBaselines:
    lstm_forecast: nn.Module
    transformer_forecast: nn.Module
    lstm_ae: nn.Module
    transformer_ae: nn.Module


def collect_training_arrays(split_dir: Path, source: str, max_records: int, pred_len: int):
    forecast_x, forecast_y, clean_contexts = [], [], []
    for row in read_jsonl(split_dir / source / "train" / "forecast.jsonl", max_records):
        cols = list(row["source_columns"])
        hist = matrix_from_mapping(row["history"], cols)
        fut = matrix_from_mapping(row["target_future"], cols)
        for x, y in zip(hist, fut):
            xn, mu, sigma = norm1d(x)
            forecast_x.append(xn)
            forecast_y.append(((fill_linear(y) - mu) / sigma).astype(np.float32)[:pred_len])
            clean_contexts.append(xn)
    for task, field in (("interpolation", "observed_context"), ("anomaly_detection", "clean_context")):
        for row in read_jsonl(split_dir / source / "train" / f"{task}.jsonl", max_records):
            cols = list(row["source_columns"])
            mat = matrix_from_mapping(row[field], cols)
            if task == "interpolation":
                for row_idx, col in enumerate(cols):
                    for idx, value in zip(row["missing_indices"].get(col, []), row["target_values"].get(col, [])):
                        if 0 <= int(idx) < mat.shape[1]:
                            mat[row_idx, int(idx)] = finite_float(value)
            for series in mat:
                clean_contexts.append(norm1d(series)[0])
    return (
        np.asarray(forecast_x, dtype=np.float32),
        np.asarray(forecast_y, dtype=np.float32),
        np.asarray(clean_contexts, dtype=np.float32),
    )


def train_model(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, lr: float = 1e-3) -> nn.Module:
    if len(x) == 0:
        return model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    batch_size = min(64, len(x))
    for _ in range(epochs):
        perm = torch.randperm(len(x_t), device=device)
        for start in range(0, len(x_t), batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_t[idx])
            loss = torch.mean((pred - y_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def train_neural_baselines(split_dir: Path, source: str, max_records: int, pred_len: int, epochs: int) -> NeuralBaselines:
    fx, fy, contexts = collect_training_arrays(split_dir, source, max_records, pred_len)
    rng = np.random.default_rng(42)
    masked = contexts.copy()
    if len(masked):
        mask = rng.random(masked.shape) < 0.2
        masked[mask] = 0.0
    return NeuralBaselines(
        lstm_forecast=train_model(LSTMForecast(pred_len), fx, fy, epochs),
        transformer_forecast=train_model(TransformerForecast(pred_len), fx, fy, epochs),
        lstm_ae=train_model(LSTMAutoEncoder(), masked, contexts, epochs),
        transformer_ae=train_model(TransformerAutoEncoder(), masked, contexts, epochs),
    )


def predict_nn(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x[None].astype(np.float32))).squeeze(0).cpu().numpy()


def load_chronos_model(path: str, base_model: str, device: str):
    if path == base_model:
        model = Chronos2MultiTaskModel.from_pretrained(base_model)
        model.to(torch.device(device))
        model.eval()
        return model
    return load_checkpoint(base_model, path, device)


def chronos_forecast(model, history: np.ndarray, pred_len: int) -> np.ndarray:
    device = next(model.parameters()).device
    core = getattr(getattr(model, "base_model", None), "model", model)
    median_idx = core.chronos_config.quantiles.index(0.5)
    output_patch = core.chronos_config.output_patch_size
    num_patches = math.ceil(pred_len / output_patch)
    context = torch.from_numpy(history).to(device)
    with torch.no_grad():
        out = model(
            context=context,
            context_mask=torch.isfinite(context),
            group_ids=torch.zeros(context.shape[0], dtype=torch.long, device=device),
            future_covariates=torch.full((context.shape[0], pred_len), float("nan"), device=device),
            num_output_patches=num_patches,
        )
    return out.quantile_preds[:, median_idx, :pred_len].detach().cpu().numpy()


def chronos_reconstruct(model, observed: np.ndarray, reconstruction_mask: np.ndarray | None = None) -> np.ndarray:
    device = next(model.parameters()).device
    core = getattr(getattr(model, "base_model", None), "model", model)
    median_idx = core.chronos_config.quantiles.index(0.5)
    context = torch.from_numpy(observed).to(device)
    rec_mask = None if reconstruction_mask is None else torch.from_numpy(reconstruction_mask).to(device)
    with torch.no_grad():
        preds = core.reconstruct_context(
            context=context,
            context_mask=torch.isfinite(context),
            reconstruction_mask=rec_mask,
            group_ids=torch.zeros(context.shape[0], dtype=torch.long, device=device),
        )
    return preds[:, median_idx, : observed.shape[1]].detach().cpu().numpy()


def evaluate_classic(split_dir: Path, source: str, max_records: int | None, neural: NeuralBaselines | None) -> dict:
    result = {}
    pred_acc: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "last_value": ([], []),
        "linear_trend": ([], []),
    }
    if neural:
        pred_acc["lstm"] = ([], [])
        pred_acc["transformer"] = ([], [])
    for row in read_jsonl(split_dir / source / "test" / "forecast.jsonl", max_records):
        cols = list(row["source_columns"])
        hist = matrix_from_mapping(row["history"], cols)
        fut = matrix_from_mapping(row["target_future"], cols)
        pred_len = fut.shape[1]
        for x, y in zip(hist, fut):
            x_fill = fill_linear(x)
            last = np.full(pred_len, x_fill[-1], dtype=np.float32)
            slope = (x_fill[-1] - x_fill[max(0, len(x_fill) - min(16, len(x_fill)))]) / max(1, min(16, len(x_fill)) - 1)
            trend = x_fill[-1] + slope * np.arange(1, pred_len + 1, dtype=np.float32)
            pred_acc["last_value"][0].append(last); pred_acc["last_value"][1].append(y)
            pred_acc["linear_trend"][0].append(trend); pred_acc["linear_trend"][1].append(y)
            if neural:
                xn, mu, sigma = norm1d(x)
                pred_acc["lstm"][0].append(predict_nn(neural.lstm_forecast, xn) * sigma + mu); pred_acc["lstm"][1].append(y)
                pred_acc["transformer"][0].append(predict_nn(neural.transformer_forecast, xn) * sigma + mu); pred_acc["transformer"][1].append(y)
    for name, (preds, targets) in pred_acc.items():
        result[f"{name}/forecast"] = mse_mae(preds, targets)

    interp_acc: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "linear_interpolation": ([], []),
        "mean_fill": ([], []),
    }
    anomaly_acc: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "rolling_zscore": ([], []),
    }
    if neural:
        interp_acc["lstm_ae"] = ([], [])
        interp_acc["transformer_ae"] = ([], [])
        anomaly_acc["lstm_ae"] = ([], [])
        anomaly_acc["transformer_ae"] = ([], [])

    for row in read_jsonl(split_dir / source / "test" / "interpolation.jsonl", max_records):
        cols = list(row["source_columns"])
        obs = matrix_from_mapping(row["observed_context"], cols)
        lin = fill_linear(obs)
        for i, col in enumerate(cols):
            idx = np.asarray(row["missing_indices"].get(col, []), dtype=int)
            tgt = np.asarray([finite_float(x) for x in row["target_values"].get(col, [])], dtype=np.float32)
            if len(idx) == 0:
                continue
            interp_acc["linear_interpolation"][0].append(lin[i, idx]); interp_acc["linear_interpolation"][1].append(tgt)
            mean_value = float(np.nanmean(obs[i])) if np.isfinite(obs[i]).any() else 0.0
            interp_acc["mean_fill"][0].append(np.full_like(tgt, mean_value)); interp_acc["mean_fill"][1].append(tgt)
            if neural:
                xn, mu, sigma = norm1d(obs[i])
                interp_acc["lstm_ae"][0].append(predict_nn(neural.lstm_ae, xn)[idx] * sigma + mu); interp_acc["lstm_ae"][1].append(tgt)
                interp_acc["transformer_ae"][0].append(predict_nn(neural.transformer_ae, xn)[idx] * sigma + mu); interp_acc["transformer_ae"][1].append(tgt)
    for name, (preds, targets) in interp_acc.items():
        result[f"{name}/interpolation"] = mse_mae(preds, targets)

    for row in read_jsonl(split_dir / source / "test" / "anomaly_detection.jsonl", max_records):
        cols = list(row["source_columns"])
        obs = matrix_from_mapping(row["observed_context"], cols)
        filled = fill_linear(obs)
        for i, col in enumerate(cols):
            labels = np.asarray(row["labels"].get(col, []), dtype=bool)
            if labels.size == 0:
                continue
            anomaly_acc["rolling_zscore"][0].append(labels)
            anomaly_acc["rolling_zscore"][1].append(rolling_zscore(filled[i]))
            if neural:
                xn, mu, sigma = norm1d(obs[i])
                rec = predict_nn(neural.lstm_ae, xn) * sigma + mu
                anomaly_acc["lstm_ae"][0].append(labels); anomaly_acc["lstm_ae"][1].append(np.abs(filled[i] - rec))
                rec = predict_nn(neural.transformer_ae, xn) * sigma + mu
                anomaly_acc["transformer_ae"][0].append(labels); anomaly_acc["transformer_ae"][1].append(np.abs(filled[i] - rec))
    for name, (labels, scores) in anomaly_acc.items():
        result[f"{name}/anomaly_detection"] = anomaly_metrics(labels, scores)

    return result


def collect_chronos_anomaly_scores(
    model,
    split_dir: Path,
    source: str,
    split_name: str,
    max_records: int | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    labels, chronos_scores, rolling_scores = [], [], []
    for row in read_jsonl(split_dir / source / split_name / "anomaly_detection.jsonl", max_records):
        cols = list(row["source_columns"])
        obs = matrix_from_mapping(row["observed_context"], cols)
        rec = chronos_reconstruct(model, obs, None)
        filled = fill_linear(obs)
        for i, col in enumerate(cols):
            lab = np.asarray(row["labels"].get(col, []), dtype=bool)
            if len(lab):
                labels.append(lab)
                chronos_scores.append(robust_score_scale(np.abs(filled[i] - rec[i])))
                rolling_scores.append(robust_score_scale(rolling_zscore(filled[i])))
    return labels, chronos_scores, rolling_scores


def select_fusion_alpha(
    labels: list[np.ndarray],
    chronos_scores: list[np.ndarray],
    rolling_scores: list[np.ndarray],
) -> float:
    best_alpha, best_auc = 0.0, -float("inf")
    for alpha in np.linspace(0.0, 1.0, 11):
        fused = [alpha * c + (1.0 - alpha) * r for c, r in zip(chronos_scores, rolling_scores)]
        auc = anomaly_metrics(labels, fused).get("auroc", float("nan"))
        if np.isfinite(auc) and auc > best_auc:
            best_alpha, best_auc = float(alpha), float(auc)
    return best_alpha


def evaluate_chronos(
    split_dir: Path,
    source: str,
    model_name: str,
    model_path: str,
    base_model: str,
    max_records: int | None,
    device: str,
    tasks: tuple[str, ...] = TASKS,
) -> dict:
    model = load_chronos_model(model_path, base_model, device)
    result = {}

    if "forecast" in tasks:
        preds, targets = [], []
        for row in read_jsonl(split_dir / source / "test" / "forecast.jsonl", max_records):
            cols = list(row["source_columns"])
            hist = matrix_from_mapping(row["history"], cols)
            fut = matrix_from_mapping(row["target_future"], cols)
            pred = chronos_forecast(model, hist, fut.shape[1])
            preds.append(pred); targets.append(fut)
        result[f"{model_name}/forecast"] = mse_mae(preds, targets)

    if "interpolation" in tasks:
        preds, targets = [], []
        for row in read_jsonl(split_dir / source / "test" / "interpolation.jsonl", max_records):
            cols = list(row["source_columns"])
            obs = matrix_from_mapping(row["observed_context"], cols)
            mask = ~np.isfinite(obs)
            rec = chronos_reconstruct(model, obs, mask)
            for i, col in enumerate(cols):
                idx = np.asarray(row["missing_indices"].get(col, []), dtype=int)
                tgt = np.asarray([finite_float(x) for x in row["target_values"].get(col, [])], dtype=np.float32)
                if len(idx):
                    preds.append(rec[i, idx]); targets.append(tgt)
        result[f"{model_name}/interpolation"] = mse_mae(preds, targets)

    if "anomaly_detection" in tasks:
        val = collect_chronos_anomaly_scores(model, split_dir, source, "val", max_records)
        alpha = select_fusion_alpha(*val)
        labels, chronos_scores, rolling_scores = collect_chronos_anomaly_scores(
            model, split_dir, source, "test", max_records
        )
        result[f"{model_name}/anomaly_detection"] = anomaly_metrics(labels, chronos_scores)
        fused = [alpha * c + (1.0 - alpha) * r for c, r in zip(chronos_scores, rolling_scores)]
        hybrid_metrics = anomaly_metrics(labels, fused)
        hybrid_metrics["chronos_weight"] = alpha
        result[f"{model_name}_hybrid/anomaly_detection"] = hybrid_metrics
    model.to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def flatten_results(source: str, results: dict) -> pd.DataFrame:
    rows = []
    for key, metrics in results.items():
        model, task = key.split("/", 1)
        rows.append({"source": source, "model": model, "task": task, **metrics})
    return pd.DataFrame(rows).sort_values(["source", "task", "model"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multitask JSONL experiments")
    parser.add_argument("--split-dir", default="data/multitask_eval_splits")
    parser.add_argument("--source", choices=["acars", "qar"], required=True)
    parser.add_argument("--base-model", default="weights/chronos-2")
    parser.add_argument("--skip-base-model", action="store_true")
    parser.add_argument("--finetuned-model")
    parser.add_argument("--forecast-model")
    parser.add_argument("--interpolation-model")
    parser.add_argument("--anomaly-model")
    parser.add_argument("--output-dir", default="results/multitask_jsonl_experiment")
    parser.add_argument("--max-records-per-task", type=int, default=80)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--include-neural-baselines", action="store_true")
    parser.add_argument("--neural-train-records", type=int, default=128)
    parser.add_argument("--neural-epochs", type=int, default=1)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = str((PROJECT_ROOT / args.base_model).resolve())

    neural = None
    if args.include_neural_baselines:
        neural = train_neural_baselines(split_dir, args.source, args.neural_train_records, pred_len=16, epochs=args.neural_epochs)

    results = evaluate_classic(split_dir, args.source, args.max_records_per_task, neural)
    if not args.skip_base_model:
        results.update(
            evaluate_chronos(
                split_dir,
                args.source,
                "chronos2_base",
                base_model,
                base_model,
                args.max_records_per_task,
                args.device,
            )
        )
    if args.finetuned_model:
        finetuned = str((PROJECT_ROOT / args.finetuned_model).resolve())
        results.update(
            evaluate_chronos(split_dir, args.source, "chronos2_finetuned", finetuned, base_model, args.max_records_per_task, args.device)
        )
    task_models = {
        "forecast": args.forecast_model,
        "interpolation": args.interpolation_model,
        "anomaly_detection": args.anomaly_model,
    }
    for task, model_path in task_models.items():
        if model_path:
            resolved = str((PROJECT_ROOT / model_path).resolve())
            results.update(
                evaluate_chronos(
                    split_dir,
                    args.source,
                    f"chronos2_{task}_adapter",
                    resolved,
                    base_model,
                    args.max_records_per_task,
                    args.device,
                    tasks=(task,),
                )
            )

    df = flatten_results(args.source, results)
    json_path = output_dir / f"{args.source}_metrics.json"
    csv_path = output_dir / f"{args.source}_metrics.csv"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
