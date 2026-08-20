# -*- coding: utf-8 -*-
"""
Chronos-2 多任务微调训练脚本

使用随机掩码技术对 Chronos-2 模型进行微调，使其同时支持：
1. 时间序列预测（标准分位数损失）
2. 数据插值（掩码位置重建）
3. 异常检测（基于 P10-P90 动态置信包络线）

训练流程：
- 加载预训练权重
- 保留原始数据中的 NaN，由模型原生 context_mask 处理
- 对上下文应用随机掩码
- 同时优化预测损失和重建损失
- 监控验证集指标并保存模型
"""

import os
import json
import random
import logging
import glob
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

PROJECT_ROOT = Path(os.environ.get("TSLM_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments, set_seed
from transformers.trainer_callback import TrainerCallback
from einops import rearrange

from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback
from tslm_multitask.models import Chronos2MultiTaskModel

logger = logging.getLogger(__name__)

# ============================================================
# 超参数配置
# ============================================================
class TrainConfig:
    # 路径配置
    model_path: str = str(PROJECT_ROOT / "weights" / "chronos-2")
    data_dir: str = str(PROJECT_ROOT / "data")
    output_dir: str = str(PROJECT_ROOT / "weights" / "multitask_chronos2")

    # 数据集参数
    context_length: int = 128        # 上下文长度（8个patch，每个16步）
    prediction_length: int = 16      # 预测长度（1个patch）
    val_ratio: float = 0.2           # 验证集比例（8:2划分）
    min_series_length: int = 50      # 最短序列长度
    data_mode: str = "multivariate"  # multivariate: 每个文件作为多传感器序列; single_column: 逐列单变量
    selected_task: str = "all"       # JSONL workflow: all/forecast/interpolation/anomaly_detection
    max_targets_per_item: int = 16   # 单个 group 的最大变量数，避免低显存 GPU 上 group attention 数值不稳

    # 训练超参数
    batch_size: int = 16             # 批处理大小
    max_steps: int = 500             # 最大训练步数
    learning_rate: float = 1e-4      # 学习率
    warmup_ratio: float = 0.1        # 预热比例
    log_steps: int = 20              # 日志记录间隔
    save_steps: int = 100            # 模型保存间隔
    eval_steps: int = 50             # 评估间隔
    finetune_mode: str = "lora"      # lora 或 full
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: str = "output_head"
    normalized_clip_value: float = 100.0
    sanitize_nonfinite_grads: bool = True
    use_cpu: bool = False
    skip_final_eval: bool = False

    # 掩码参数
    forecast_loss_weight: float = 1.0
    mask_ratio: float = 0.2          # 随机掩码比例（20%）
    recon_loss_weight: float = 0.5   # 重建损失权重
    anomaly_loss_weight: float = 0.0 # 预留：需标注异常或校准集后启用显式异常损失

    # 随机种子
    seed: int = 42


def _parse_scalar(value: str):
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_flat_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    config: Dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            config[key] = _parse_scalar(value)
    return config


def _resolve_project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def load_config_from_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Chronos-2 multitask fine-tuning")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_chronos2_multitask.yaml"))
    parser.add_argument("--model-path")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--prediction-length", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--data-mode", choices=["multivariate", "single_column"])
    parser.add_argument("--max-targets-per-item", type=int)
    parser.add_argument("--finetune-mode", choices=["lora", "full"])
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument(
        "--lora-target-modules",
        choices=["output_head", "feed_forward_and_output", "attention_and_output"],
    )
    parser.add_argument("--normalized-clip-value", type=float)
    parser.add_argument("--no-sanitize-nonfinite-grads", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--mask-ratio", type=float)
    parser.add_argument("--forecast-loss-weight", type=float)
    parser.add_argument("--recon-loss-weight", type=float)
    args = parser.parse_args()

    config = TrainConfig()
    config_path = Path(args.config)
    if config_path.exists():
        for key, value in _load_flat_config(config_path).items():
            if hasattr(config, key):
                setattr(config, key, value)

    for path_key in ("model_path", "data_dir", "output_dir"):
        setattr(config, path_key, _resolve_project_path(getattr(config, path_key)))

    for arg_name, config_name in {
        "model_path": "model_path",
        "data_dir": "data_dir",
        "output_dir": "output_dir",
        "context_length": "context_length",
        "prediction_length": "prediction_length",
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "data_mode": "data_mode",
        "max_targets_per_item": "max_targets_per_item",
        "finetune_mode": "finetune_mode",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_target_modules": "lora_target_modules",
        "normalized_clip_value": "normalized_clip_value",
        "mask_ratio": "mask_ratio",
        "forecast_loss_weight": "forecast_loss_weight",
        "recon_loss_weight": "recon_loss_weight",
    }.items():
        value = getattr(args, arg_name)
        if value is not None:
            if config_name in {"model_path", "data_dir", "output_dir"}:
                value = _resolve_project_path(value)
            setattr(config, config_name, value)

    if args.use_cpu:
        config.use_cpu = True
    if args.skip_final_eval:
        config.skip_final_eval = True
    if args.no_sanitize_nonfinite_grads:
        config.sanitize_nonfinite_grads = False

    return config


# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_split_data(config: TrainConfig):
    """
    加载 CSV 数据文件，直接保留原始数据中的 NaN 缺失值。
    利用 Chronos-2 原生的 context_mask 机制处理缺失数据，避免人为填充破坏真实分布。
    """
    if config.data_mode not in {"multivariate", "single_column"}:
        raise ValueError(f"data_mode 必须是 multivariate 或 single_column，当前为: {config.data_mode}")

    data_dir = config.data_dir
    feature_files = sorted(glob.glob(os.path.join(data_dir, "*_features.csv")))
    if len(feature_files) == 0:
        raise ValueError(f"未在 {data_dir} 中找到 *_features.csv 文件")

    logger.info(f"找到 {len(feature_files)} 个特征文件")

    rng = random.Random(config.seed)
    file_indices = list(range(len(feature_files)))
    rng.shuffle(file_indices)
    n_val = max(1, int(len(feature_files) * config.val_ratio))
    val_file_indices = set(file_indices[:n_val])

    train_series: List[torch.Tensor] = []
    val_series: List[torch.Tensor] = []
    nan_count = 0
    total_feature_count = 0

    for idx, filepath in enumerate(feature_files):
        df = pd.read_csv(filepath)
        is_val = idx in val_file_indices
        numeric_df = df.apply(pd.to_numeric, errors="coerce")

        if config.data_mode == "multivariate":
            valid_columns = [
                col for col in numeric_df.columns
                if int(numeric_df[col].notna().sum()) >= config.min_series_length
            ]
            if len(numeric_df) < config.min_series_length or not valid_columns:
                continue

            total_feature_count += len(valid_columns)
            max_targets = max(1, int(config.max_targets_per_item))

            for col_start in range(0, len(valid_columns), max_targets):
                column_group = valid_columns[col_start : col_start + max_targets]
                values = numeric_df[column_group].to_numpy(dtype=np.float32).T
                nan_count += int(np.isnan(values).sum())

                # Chronos2Dataset 支持 2-D tensor: (n_targets, time)，group_ids 会按目标维度自动构造。
                tensor = torch.from_numpy(values)
                if is_val:
                    val_series.append(tensor)
                else:
                    train_series.append(tensor)
        else:
            for col in numeric_df.columns:
                values = numeric_df[col].values.astype(np.float32)
                valid_count = int(np.sum(~np.isnan(values)))

                if len(values) >= config.min_series_length and valid_count >= config.min_series_length:
                    nan_count += int(np.sum(np.isnan(values)))
                    total_feature_count += 1

                    # 直接转换为 Tensor，保留 NaN 供原生 context_mask 处理
                    tensor = torch.from_numpy(values)
                    if is_val:
                        val_series.append(tensor)
                    else:
                        train_series.append(tensor)

    if not train_series:
        raise ValueError("训练集为空，请检查 data_dir、min_series_length 或 val_ratio 配置")
    if not val_series:
        raise ValueError("验证集为空，请降低 val_ratio/min_series_length 或增加数据文件")

    file_stats = {
        "data_mode": config.data_mode,
        "max_targets_per_item": config.max_targets_per_item if config.data_mode == "multivariate" else None,
        "total_files": len(feature_files),
        "train_series_count": len(train_series),
        "val_series_count": len(val_series),
        "total_feature_count": total_feature_count,
        "val_ratio": config.val_ratio,
        "total_nan_count_preserved": nan_count,
    }

    logger.info(
        f"数据加载完成: 训练集 {len(train_series)} 条序列, "
        f"验证集 {len(val_series)} 条序列, "
        f"保留原生 NaN 数量: {nan_count}"
    )

    return train_series, val_series, file_stats


# ============================================================
# 训练日志回调
# ============================================================
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.history: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
                **{k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in logs.items()},
            }
            self.history.append(entry)
            self._save()

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
                "event": "evaluation",
                **{k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in logs.items()},
            }
            self.history.append(entry)
            self._save()

    def _save(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)


class FiniteGradientCallback(TrainerCallback):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.total_cleaned = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if not self.enabled or model is None:
            return

        cleaned = 0
        for param in model.parameters():
            grad = param.grad
            if grad is None:
                continue
            finite_mask = torch.isfinite(grad)
            if finite_mask.all():
                continue
            cleaned += int((~finite_mask).sum().item())
            grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        if cleaned:
            self.total_cleaned += cleaned
            logger.warning(
                "清理非有限梯度: step=%s, cleaned_elements=%s, total_cleaned=%s",
                state.global_step,
                cleaned,
                self.total_cleaned,
            )


def unwrap_multitask_model(model):
    """Return the injected Chronos2MultiTaskModel when PEFT wraps it."""
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def apply_lora_if_requested(model: Chronos2MultiTaskModel, config: TrainConfig):
    if config.finetune_mode == "full":
        logger.info("使用全量微调模式")
        return model
    if config.finetune_mode != "lora":
        raise ValueError(f"finetune_mode 必须是 lora 或 full，当前为: {config.finetune_mode}")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "当前环境未安装 peft，无法启用 LoRA。请安装 peft，或将 finetune_mode 设置为 full。"
        ) from exc

    if config.lora_target_modules == "attention_and_output":
        target_modules = [
            "self_attention.q",
            "self_attention.k",
            "self_attention.v",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ]
    elif config.lora_target_modules == "output_head":
        target_modules = ["output_patch_embedding.output_layer"]
    elif config.lora_target_modules == "feed_forward_and_output":
        target_modules = ["wi", "wo", "output_patch_embedding.output_layer"]
    else:
        raise ValueError(
            "lora_target_modules 必须是 output_head、feed_forward_and_output 或 attention_and_output，"
            f"当前为: {config.lora_target_modules}"
        )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    logger.info(
        "启用 LoRA: r=%s, alpha=%s, dropout=%s, target_modules=%s",
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
        target_modules,
    )
    if hasattr(lora_model, "print_trainable_parameters"):
        lora_model.print_trainable_parameters()
    return lora_model


# ============================================================
# 训练后评估
# ============================================================
def evaluate_model(
    model,
    val_series: List[torch.Tensor],
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    core_model = unwrap_multitask_model(model)
    device = next(model.parameters()).device

    output_patch_size = core_model.chronos_config.output_patch_size
    num_output_patches = (config.prediction_length + output_patch_size - 1) // output_patch_size

    val_dataset = Chronos2Dataset(
        inputs=val_series,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        batch_size=config.batch_size,
        output_patch_size=output_patch_size,
        min_past=config.prediction_length,
        mode=DatasetMode.VALIDATION,
    )

    forecast_preds_all = []
    forecast_targets_all = []
    
    # 动态阈值评估集合
    anomaly_flags_all = []

    median_idx = core_model.chronos_config.quantiles.index(0.5)
    
    # 获取 P10 和 P90 分位数的索引，用于动态置信包络线
    p10_idx = core_model.chronos_config.quantiles.index(0.1)
    p90_idx = core_model.chronos_config.quantiles.index(0.9)

    with torch.no_grad():
        for batch in val_dataset:
            context = batch["context"].to(device)
            future_target = batch["future_target"].to(device)
            future_covariates = batch["future_covariates"].to(device)
            group_ids = batch["group_ids"].to(device)

            original_context = context.clone()

            # 模拟数据丢失
            valid_mask = torch.isfinite(context)
            rand = torch.rand_like(context)
            masked_positions = valid_mask & (rand < config.mask_ratio)

            outputs = model(
                context=context,
                context_mask=valid_mask,
                future_target=future_target,
                future_covariates=future_covariates,
                group_ids=group_ids,
                num_output_patches=num_output_patches,
            )

            preds = outputs.quantile_preds[:, median_idx, :]
            target_mask = torch.isfinite(future_target) & torch.isfinite(preds)
            if target_mask.any():
                forecast_preds_all.append(preds[target_mask].cpu())
                forecast_targets_all.append(future_target[target_mask].cpu())

            # 评估动态置信包络线 (P10 - P90)
            recon_preds = core_model.reconstruct_context(
                context=context,
                context_mask=valid_mask,
                reconstruction_mask=masked_positions,
                group_ids=group_ids,
            )
            
            # 提取边界并进行逆归一化
            recon_p10 = recon_preds[:, p10_idx, :]
            recon_p90 = recon_preds[:, p90_idx, :]

            recon_length = recon_preds.shape[-1]
            ctx_len = context.shape[-1]
            
            if recon_length > ctx_len:
                offset = recon_length - ctx_len
                recon_p10 = recon_p10[:, offset:]
                recon_p90 = recon_p90[:, offset:]
            elif recon_length < ctx_len:
                min_len = min(recon_p10.shape[-1], ctx_len)
                recon_p10 = recon_p10[:, :min_len]
                recon_p90 = recon_p90[:, :min_len]
                original_context = original_context[:, :min_len]
                masked_positions = masked_positions[:, :min_len]

            if recon_p10.shape[-1] == original_context.shape[-1] and masked_positions.any():
                finite_positions = (
                    masked_positions
                    & torch.isfinite(original_context)
                    & torch.isfinite(recon_p10)
                    & torch.isfinite(recon_p90)
                )
                if not finite_positions.any():
                    continue

                true_vals = original_context[finite_positions].cpu()
                lower_bound = recon_p10[finite_positions].cpu()
                upper_bound = recon_p90[finite_positions].cpu()

                # 若真实值游离于 P10-P90 包络线之外，则判定为异常
                is_anomaly = (true_vals < lower_bound) | (true_vals > upper_bound)
                anomaly_flags_all.append(is_anomaly)

    metrics = {}

    if forecast_preds_all:
        preds = torch.cat(forecast_preds_all)
        targets = torch.cat(forecast_targets_all)
        metrics["forecast_valid_points"] = int(targets.numel())
        metrics["forecast_mse"] = ((preds - targets) ** 2).mean().item()
        metrics["forecast_mae"] = (preds - targets).abs().mean().item()

    if anomaly_flags_all:
        all_flags = torch.cat(anomaly_flags_all)
        # 基于分位数动态包络的异常检测率
        metrics["anomaly_detection_ratio_p10_p90"] = all_flags.float().mean().item()

    return metrics


# ============================================================
# 主训练函数
# ============================================================
def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    config = load_config_from_args()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Chronos-2 多任务微调训练")
    logger.info("=" * 60)

    logger.info("[1/6] 加载和划分数据...")
    train_series, val_series, file_stats = load_and_split_data(config)
    
    logger.info(f"[2/6] 加载预训练权重: {config.model_path}")
    model = Chronos2MultiTaskModel.from_pretrained(config.model_path)
    model.forecast_loss_weight = config.forecast_loss_weight
    model.recon_loss_weight = config.recon_loss_weight
    model.mask_ratio = config.mask_ratio
    model.normalized_clip_value = config.normalized_clip_value
    model = apply_lora_if_requested(model, config)

    logger.info("[3/6] 创建训练和验证数据集...")
    core_model = unwrap_multitask_model(model)
    output_patch_size = core_model.chronos_config.output_patch_size

    train_dataset = Chronos2Dataset(
        inputs=train_series,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        batch_size=config.batch_size,
        output_patch_size=output_patch_size,
        min_past=config.prediction_length,
        mode=DatasetMode.TRAIN,
    )

    val_dataset = Chronos2Dataset(
        inputs=val_series,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        batch_size=config.batch_size,
        output_patch_size=output_patch_size,
        min_past=config.prediction_length,
        mode=DatasetMode.VALIDATION,
    )

    logger.info("[4/6] 配置训练参数...")
    warmup_steps = max(0, int(config.max_steps * config.warmup_ratio))
    use_tf32 = (not config.use_cpu) and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_steps=warmup_steps,
        logging_steps=config.log_steps,
        save_steps=config.save_steps,
        eval_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        prediction_loss_only=True,
        max_grad_norm=1.0,
        report_to=["none"],
        remove_unused_columns=False,
        seed=config.seed,
        use_cpu=config.use_cpu,
        tf32=use_tf32,
        torch_compile=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
    )

    log_file = str(output_dir / "training_log.json")
    metrics_logger = MetricsLoggerCallback(log_file)
    finite_grad_callback = FiniteGradientCallback(config.sanitize_nonfinite_grads)

    logger.info("[5/6] 开始训练...")
    trainer = Chronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[metrics_logger, finite_grad_callback, EvaluateAndSaveFinalStepCallback()],
    )

    train_result = trainer.train()

    logger.info("[6/6] 保存模型和训练日志...")
    final_dir = output_dir / "checkpoint-final"
    if not config.use_cpu and torch.cuda.is_available():
        logger.info("训练完成，迁移模型到 CPU 后再合并/保存/评估，降低低显存 GPU 的后处理风险...")
        model.to(torch.device("cpu"))
        torch.cuda.empty_cache()

    if config.finetune_mode == "lora" and hasattr(model, "merge_and_unload"):
        logger.info("合并 LoRA 权重并保存完整 checkpoint: %s", final_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(final_dir))
        model = merged_model
    else:
        trainer.save_model(str(final_dir))

    train_info = {
        "train_result": {k: float(v) if isinstance(v, (int, float)) else str(v)
                         for k, v in train_result.metrics.items()},
        "config": {
            "data_mode": config.data_mode,
            "selected_task": getattr(config, "selected_task", "all"),
            "max_targets_per_item": config.max_targets_per_item,
            "context_length": config.context_length,
            "prediction_length": config.prediction_length,
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "finetune_mode": config.finetune_mode,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": config.lora_target_modules,
            "normalized_clip_value": config.normalized_clip_value,
            "sanitize_nonfinite_grads": config.sanitize_nonfinite_grads,
            "use_cpu": config.use_cpu,
            "skip_final_eval": config.skip_final_eval,
            "forecast_loss_weight": config.forecast_loss_weight,
            "mask_ratio": config.mask_ratio,
            "recon_loss_weight": config.recon_loss_weight,
            "anomaly_loss_weight": config.anomaly_loss_weight,
            "seed": config.seed,
            "model_path": config.model_path,
        },
        "data_stats": file_stats,
        "training_time": datetime.now().isoformat(),
    }

    with open(output_dir / "training_info.json", 'w', encoding='utf-8') as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)

    if config.skip_final_eval:
        logger.info("跳过训练脚本内置详细评估；请使用 JSONL 评估脚本在测试集上统一评估。")
    else:
        logger.info("进行训练后详细评估...")
        detailed_metrics = evaluate_model(model, val_series, config)
        logger.info(f"详细指标: {json.dumps(detailed_metrics, indent=2)}")

        with open(output_dir / "evaluation_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
