"""Unified inference engine used by scripts and the desktop bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from tslm_multitask.data.loaders import load_multivariate_values
from tslm_multitask.data.masks import missing_indices_from_mask
from tslm_multitask.inference.postprocess import detect_anomalies, interpolate_missing, median_reconstruction, prediction_to_numpy


FORECAST_TASKS = {"multivariate_forecast", "univariate_forecast", "covariate_forecast"}


@dataclass(frozen=True)
class InferenceRequest:
    task_type: str
    file_path: str
    columns: str
    covariates: str
    model_path: str
    prediction_length: int
    confidence_interval: float
    anomaly_threshold: float
    context_length: int
    data_start: int
    data_end: int
    data_type: str


def json_safe_list(values) -> list[float | None]:
    return [None if pd.isna(value) or not np.isfinite(value) else float(value) for value in values]


class MultitaskInferenceEngine:
    """Task-router for forecast, imputation, and anomaly reconstruction."""

    def __init__(
        self,
        project_root: str | Path,
        pipeline_factory: Callable[[str, str], object] | None = None,
        safe_max_context: int = 2048,
        chunk_size: int = 512,
        enable_cpu_quantization: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.pipeline_factory = pipeline_factory or self._default_pipeline_factory
        self.safe_max_context = safe_max_context
        self.chunk_size = chunk_size
        self.enable_cpu_quantization = enable_cpu_quantization

    def run(self, request: InferenceRequest) -> dict:
        model_path = self._resolve_path(request.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"找不到本地模型权重: {model_path}")

        device = self._select_device()
        pipeline = self.pipeline_factory(str(model_path), device)

        target_columns = [c.strip() for c in request.columns.split(",") if c.strip()]
        covariate_columns = [c.strip() for c in request.covariates.split(",") if c.strip() and c.strip() != "-"]
        all_columns = target_columns + covariate_columns

        data_path = self._resolve_path(request.file_path)
        values, observed_mask = load_multivariate_values(data_path, all_columns, request.data_type)
        values = values[:, request.data_start : request.data_end]
        observed_mask = observed_mask[:, request.data_start : request.data_end]

        result = {
            "status": "success",
            "model_used": model_path.name,
            "device_used": device,
            "multi_results": [],
        }

        if request.task_type == "data_repair":
            result["multi_results"] = self._run_data_repair(pipeline, values, observed_mask, target_columns, request)
        elif request.task_type == "anomaly_detection":
            result["multi_results"] = self._run_anomaly_detection(pipeline, values, target_columns, request)
        elif request.task_type in FORECAST_TASKS:
            result["multi_results"] = self._run_forecast(pipeline, values, target_columns, request)
        else:
            raise ValueError(f"Unsupported task_type: {request.task_type}")

        return result

    def _run_data_repair(self, pipeline, values, observed_mask, target_columns, request: InferenceRequest) -> list[dict]:
        timestamps = self._row_timestamps(request.data_start, values.shape[1])
        results = []

        for idx, target_name in enumerate(target_columns):
            series = values[idx]
            missing_indices = missing_indices_from_mask(observed_mask[idx])
            repaired = series.copy()

            for start in range(0, len(series), self.chunk_size):
                end = min(start + self.chunk_size, len(series))
                chunk_missing = missing_indices_from_mask(observed_mask[idx, start:end])
                if len(chunk_missing) == 0:
                    continue
                chunk = series[start:end]
                padded_chunk, actual_len = self._pad_chunk(chunk)
                padded_observed_mask, _ = self._pad_chunk(observed_mask[idx, start:end].astype(np.float32))
                reconstruction_mask = np.zeros_like(padded_chunk, dtype=bool)
                reconstruction_mask[chunk_missing] = True
                preds = self._reconstruct_context(
                    pipeline=pipeline,
                    context=padded_chunk,
                    observed_mask=padded_observed_mask.astype(bool),
                    reconstruction_mask=reconstruction_mask,
                )
                preds = self._squeeze_prediction(preds)[..., :actual_len]
                repaired[start:end] = interpolate_missing(chunk, preds, chunk_missing, use_median=True)

            results.append(
                {
                    "target_name": target_name,
                    "original_data": json_safe_list(series),
                    "repaired_data": json_safe_list(repaired),
                    "missing_indices": missing_indices.astype(int).tolist(),
                    "repaired_count": int(len(missing_indices)),
                    "timestamps": timestamps,
                }
            )

        return results

    def _run_anomaly_detection(self, pipeline, values, target_columns, request: InferenceRequest) -> list[dict]:
        timestamps = self._row_timestamps(request.data_start, values.shape[1])
        results = []

        for idx, target_name in enumerate(target_columns):
            series = values[idx]
            scores = np.zeros(len(series), dtype=np.float32)
            reconstructed = np.full(len(series), np.nan, dtype=np.float32)
            reconstruction_gap_count = 0

            for start in range(0, len(series), self.chunk_size):
                end = min(start + self.chunk_size, len(series))
                chunk = series[start:end]
                padded_chunk, actual_len = self._pad_chunk(chunk)
                preds = self._reconstruct_context(
                    pipeline=pipeline,
                    context=padded_chunk,
                    observed_mask=np.isfinite(padded_chunk),
                    reconstruction_mask=None,
                )
                preds = self._squeeze_prediction(preds)[..., :actual_len]
                reconstruction_gap_count += int((~np.isfinite(preds)).all(axis=0).sum())
                _, chunk_scores = detect_anomalies(chunk, preds)
                chunk_reconstruction = median_reconstruction(preds)[:actual_len]
                scores[start:end] = chunk_scores[:actual_len]
                reconstructed[start:end] = chunk_reconstruction

            reconstructed = self._fill_reconstruction_gaps(reconstructed, series)

            mean_score = float(np.nanmean(scores))
            std_score = float(np.nanstd(scores))
            threshold = mean_score + request.anomaly_threshold * std_score
            anomaly_indices = np.where(scores > threshold)[0]

            results.append(
                {
                    "target_name": target_name,
                    "original_data": json_safe_list(series),
                    "reconstructed_data": json_safe_list(reconstructed),
                    "anomaly_scores": json_safe_list(scores),
                    "threshold": float(threshold),
                    "reconstruction_gap_count": int(reconstruction_gap_count),
                    "anomalies": [
                        {"index": int(anomaly_idx), "value": float(series[anomaly_idx]), "score": float(scores[anomaly_idx])}
                        for anomaly_idx in anomaly_indices
                    ],
                    "total_anomalies": int(len(anomaly_indices)),
                    "timestamps": timestamps,
                }
            )

        return results

    def _run_forecast(self, pipeline, values, target_columns, request: InferenceRequest) -> list[dict]:
        ctx_len = min(request.context_length, values.shape[1], self.safe_max_context)
        history_data = values[:, -ctx_len:]

        if request.task_type == "covariate_forecast":
            inputs = [history_data]
        elif request.task_type == "multivariate_forecast":
            inputs = [history_data[: len(target_columns)]]
        else:
            inputs = [history_data[0]]

        preds = prediction_to_numpy(pipeline.predict(inputs=inputs, prediction_length=request.prediction_length))
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=0)

        lower_q = (1.0 - request.confidence_interval) / 2.0 * 100
        upper_q = (1.0 + request.confidence_interval) / 2.0 * 100
        history_timestamps = [f"ROW_{request.data_end - ctx_len + idx}" for idx in range(ctx_len)]
        future_timestamps = [f"PRED_{request.data_end + idx}" for idx in range(request.prediction_length)]

        results = []
        for idx in range(min(len(target_columns), preds.shape[0])):
            target_pred = preds[idx]
            results.append(
                {
                    "target_name": target_columns[idx],
                    "predictions": json_safe_list(np.percentile(target_pred, 50, axis=0)),
                    "confidence_lower": json_safe_list(np.percentile(target_pred, lower_q, axis=0)),
                    "confidence_upper": json_safe_list(np.percentile(target_pred, upper_q, axis=0)),
                    "timestamps": future_timestamps,
                    "history_data": json_safe_list(history_data[idx]),
                    "history_timestamps": history_timestamps,
                }
            )
        return results

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    def _row_timestamps(self, data_start: int, length: int) -> list[str]:
        return [f"ROW_{data_start + idx}" for idx in range(length)]

    def _pad_chunk(self, chunk: np.ndarray) -> tuple[np.ndarray, int]:
        actual_len = len(chunk)
        if actual_len >= self.chunk_size:
            return chunk, actual_len
        if actual_len == 0:
            return chunk, actual_len
        fill_value = float(chunk[-1]) if np.isfinite(chunk[-1]) else 0.0
        return np.pad(chunk, (0, self.chunk_size - actual_len), mode="constant", constant_values=fill_value), actual_len

    def _fill_reconstruction_gaps(self, reconstructed: np.ndarray, series: np.ndarray) -> np.ndarray:
        """Keep the baseline drawable even when a model chunk returns NaN values."""
        reconstructed = np.asarray(reconstructed, dtype=np.float32).copy()
        series = np.asarray(series, dtype=np.float32)
        missing = ~np.isfinite(reconstructed)
        if not missing.any():
            return reconstructed

        x = np.arange(len(reconstructed))
        valid = ~missing
        if valid.any():
            reconstructed[missing] = np.interp(x[missing], x[valid], reconstructed[valid]).astype(np.float32)
            first_valid = int(x[valid][0])
            last_valid = int(x[valid][-1])
            series_baseline = self._series_baseline(series)
            if first_valid > 0:
                reconstructed[:first_valid] = series_baseline[:first_valid]
            if last_valid + 1 < len(reconstructed):
                reconstructed[last_valid + 1 :] = series_baseline[last_valid + 1 :]
            return reconstructed

        reconstructed[:] = self._series_baseline(series)
        return reconstructed

    def _series_baseline(self, series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=np.float32)
        x = np.arange(len(series))
        valid = np.isfinite(series)
        if valid.any():
            return np.interp(x, x[valid], series[valid]).astype(np.float32)
        return np.zeros(len(series), dtype=np.float32)

    def _squeeze_prediction(self, preds: np.ndarray) -> np.ndarray:
        if preds.ndim == 3 and preds.shape[0] == 1:
            return np.squeeze(preds, axis=0)
        return preds

    def _reconstruct_context(
        self,
        pipeline,
        context: np.ndarray,
        observed_mask: np.ndarray,
        reconstruction_mask: np.ndarray | None,
    ) -> np.ndarray:
        reconstruct_fn = getattr(pipeline, "reconstruct_context", None)
        if reconstruct_fn is None and hasattr(pipeline, "model"):
            reconstruct_fn = getattr(pipeline.model, "reconstruct_context", None)
        if reconstruct_fn is None:
            raise AttributeError(
                "数据修复和异常检测需要带 reconstruct_context() 的 Chronos2 多任务模型；"
                "普通 forecasting pipeline.predict() 不再用于历史重构。"
            )

        import torch

        model = getattr(pipeline, "model", None)
        try:
            device = next(model.parameters()).device if model is not None else torch.device(self._select_device())
        except StopIteration:
            device = torch.device(self._select_device())

        context_tensor = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
        observed_mask_tensor = torch.as_tensor(observed_mask, dtype=torch.float32, device=device).unsqueeze(0)
        reconstruction_mask_tensor = (
            torch.as_tensor(reconstruction_mask, dtype=torch.bool, device=device).unsqueeze(0)
            if reconstruction_mask is not None
            else None
        )

        was_training = bool(model.training) if model is not None else False
        if model is not None:
            model.eval()
        with torch.no_grad():
            preds = reconstruct_fn(
                context=context_tensor,
                context_mask=observed_mask_tensor,
                reconstruction_mask=reconstruction_mask_tensor,
            )
        if model is not None and was_training:
            model.train()
        return prediction_to_numpy(preds)

    def _select_device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _default_pipeline_factory(self, model_path: str, device: str):
        import torch
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=device, dtype=torch.float32)
        if device == "cpu" and self.enable_cpu_quantization:
            pipeline.model = torch.quantization.quantize_dynamic(pipeline.model, {torch.nn.Linear}, dtype=torch.qint8)
        return pipeline
