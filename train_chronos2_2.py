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
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments, set_seed
from transformers.trainer_callback import TrainerCallback
from einops import rearrange

# 修正导入路径：直接从 model 模块导入，避免触发外层的循环依赖
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback
from chronos.chronos2.model import Chronos2Model, Chronos2Output

logger = logging.getLogger(__name__)

# ============================================================
# 超参数配置
# ============================================================
class TrainConfig:
    # 路径配置
    model_path: str = r"e:\Cursor Code\Chronos\weights\chronos-2"
    data_dir: str = r"e:\Cursor Code\Chronos\data"
    output_dir: str = r"e:\Cursor Code\Chronos\finetuned-chronos-2_v2" # 更改输出目录以区分

    # 数据集参数
    context_length: int = 128        # 上下文长度（8个patch，每个16步）
    prediction_length: int = 16      # 预测长度（1个patch）
    val_ratio: float = 0.2           # 验证集比例（8:2划分）
    min_series_length: int = 50      # 最短序列长度

    # 训练超参数
    batch_size: int = 16             # 批处理大小
    max_steps: int = 500             # 最大训练步数
    learning_rate: float = 1e-4      # 学习率
    warmup_ratio: float = 0.1        # 预热比例
    log_steps: int = 20              # 日志记录间隔
    save_steps: int = 100            # 模型保存间隔
    eval_steps: int = 50             # 评估间隔

    # 掩码参数
    mask_ratio: float = 0.2          # 随机掩码比例（20%）
    recon_loss_weight: float = 0.5   # 重建损失权重

    # 随机种子
    seed: int = 42


# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_split_data(config: TrainConfig):
    """
    加载 CSV 数据文件，直接保留原始数据中的 NaN 缺失值。
    利用 Chronos-2 原生的 context_mask 机制处理缺失数据，避免人为填充破坏真实分布。
    """
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

    for idx, filepath in enumerate(feature_files):
        df = pd.read_csv(filepath)
        is_val = idx in val_file_indices
        for col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            valid_count = int(np.sum(~np.isnan(values)))
            
            if len(values) >= config.min_series_length and valid_count >= config.min_series_length:
                nan_count += int(np.sum(np.isnan(values)))
                
                # 直接转换为 Tensor，保留 NaN 供原生 context_mask 处理
                tensor = torch.from_numpy(values)
                if is_val:
                    val_series.append(tensor)
                else:
                    train_series.append(tensor)

    file_stats = {
        "total_files": len(feature_files),
        "train_series_count": len(train_series),
        "val_series_count": len(val_series),
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
# 多任务 Chronos-2 模型
# ============================================================
class Chronos2MultiTaskModel(Chronos2Model):
    def __init__(self, config):
        super().__init__(config)
        self.recon_loss_weight = 0.5

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        future_covariates_mask: Optional[torch.Tensor] = None,
        num_output_patches: int = 1,
        future_target: Optional[torch.Tensor] = None,
        future_target_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        is_anomaly_detection: bool = False,
    ) -> Chronos2Output:
        
        batch_size = context.shape[0]

        # 贯穿控制标志，底层 _prepare_patched_context 将自动接收并处理 is_anomaly_detection
        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches, mlm_mask = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
            is_anomaly_detection=is_anomaly_detection,
        )
        hidden_states: torch.Tensor = encoder_outputs[0]

        # ---- 重建任务（context patches）----
        recon_embeds = hidden_states[:, :num_context_patches]
        recon_preds = self.output_patch_embedding(recon_embeds)
        recon_preds = rearrange(
            recon_preds,
            "b n (q p) -> b q (n p)",
            n=num_context_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )

        mlm_loss = self._compute_mlm_loss(
            quantile_preds=recon_preds,
            original_context=context,
            context_mask=context_mask,
            mlm_mask=mlm_mask,
            loc_scale=loc_scale,
        )

        # ---- 预测任务（future patches）----
        forecast_embeds = hidden_states[:, -num_output_patches:]
        forecast_preds = self.output_patch_embedding(forecast_embeds)
        forecast_preds = rearrange(
            forecast_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )

        forecast_loss = None
        if future_target is not None:
            forecast_loss = self._compute_loss(
                quantile_preds=forecast_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )

        # ---- 总损失 ----
        if forecast_loss is not None:
            total_loss = forecast_loss + self.recon_loss_weight * mlm_loss
        else:
            total_loss = mlm_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e4, neginf=-1e4)

        # 逆归一化预测结果
        forecast_preds_unscaled = rearrange(
            forecast_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )
        forecast_preds_unscaled = self.instance_norm.inverse(forecast_preds_unscaled, loc_scale)
        forecast_preds_unscaled = rearrange(
            forecast_preds_unscaled,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )

        return Chronos2Output(
            loss=total_loss,
            quantile_preds=forecast_preds_unscaled,
            enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        )


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


# ============================================================
# 训练后评估
# ============================================================
def evaluate_model(
    model: Chronos2MultiTaskModel,
    val_series: List[torch.Tensor],
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    output_patch_size = model.chronos_config.output_patch_size
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

    median_idx = model.chronos_config.quantiles.index(0.5)
    
    # 获取 P10 和 P90 分位数的索引，用于动态置信包络线
    p10_idx = model.chronos_config.quantiles.index(0.1)
    p90_idx = model.chronos_config.quantiles.index(0.9)

    with torch.no_grad():
        for batch in val_dataset:
            context = batch["context"].to(device)
            future_target = batch["future_target"].to(device)
            future_covariates = batch["future_covariates"].to(device)
            group_ids = batch["group_ids"].to(device)

            original_context = context.clone()

            # 模拟数据丢失
            valid_mask = ~torch.isnan(context)
            rand = torch.rand_like(context)
            masked_positions = valid_mask & (rand < config.mask_ratio)
            masked_context = context.clone()
            masked_context[masked_positions] = float('nan')

            outputs = model(
                context=masked_context,
                future_target=future_target,
                future_covariates=future_covariates,
                group_ids=group_ids,
                num_output_patches=num_output_patches,
            )

            preds = outputs.quantile_preds[:, median_idx, :]
            target_mask = ~torch.isnan(future_target)
            forecast_preds_all.append(preds[target_mask].cpu())
            forecast_targets_all.append(future_target[target_mask].cpu())

            # 评估动态置信包络线 (P10 - P90)
            encoder_outputs, loc_scale, _, num_context_patches, mlm_mask = model.encode(
                context=masked_context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                num_output_patches=num_output_patches,
            )
            hidden_states = encoder_outputs[0]
            recon_embeds = hidden_states[:, :num_context_patches]
            recon_preds = model.output_patch_embedding(recon_embeds)
            recon_preds = rearrange(
                recon_preds,
                "b n (q p) -> b q (n p)",
                n=num_context_patches,
                q=model.num_quantiles,
                p=output_patch_size,
            )
            
            # 提取边界并进行逆归一化
            recon_p10 = model.instance_norm.inverse(recon_preds[:, p10_idx, :], loc_scale)
            recon_p90 = model.instance_norm.inverse(recon_preds[:, p90_idx, :], loc_scale)

            recon_length = num_context_patches * model.chronos_config.input_patch_size
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
                true_vals = original_context[masked_positions].cpu()
                lower_bound = recon_p10[masked_positions].cpu()
                upper_bound = recon_p90[masked_positions].cpu()
                
                # 若真实值游离于 P10-P90 包络线之外，则判定为异常
                is_anomaly = (true_vals < lower_bound) | (true_vals > upper_bound)
                anomaly_flags_all.append(is_anomaly)

    metrics = {}

    if forecast_preds_all:
        preds = torch.cat(forecast_preds_all)
        targets = torch.cat(forecast_targets_all)
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

    config = TrainConfig()
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
    model.recon_loss_weight = config.recon_loss_weight

    logger.info("[3/6] 创建训练和验证数据集...")
    output_patch_size = model.chronos_config.output_patch_size

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
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
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
        use_cpu=False,
        tf32=True,
        torch_compile=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
    )

    log_file = str(output_dir / "training_log.json")
    metrics_logger = MetricsLoggerCallback(log_file)

    logger.info("[5/6] 开始训练...")
    trainer = Chronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[metrics_logger, EvaluateAndSaveFinalStepCallback()],
    )

    train_result = trainer.train()

    logger.info("[6/6] 保存模型和训练日志...")
    final_dir = output_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))

    train_info = {
        "train_result": {k: float(v) if isinstance(v, (int, float)) else str(v)
                         for k, v in train_result.metrics.items()},
        "config": {
            "context_length": config.context_length,
            "prediction_length": config.prediction_length,
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "mask_ratio": config.mask_ratio,
            "recon_loss_weight": config.recon_loss_weight,
            "seed": config.seed,
            "model_path": config.model_path,
        },
        "data_stats": file_stats,
        "training_time": datetime.now().isoformat(),
    }

    with open(output_dir / "training_info.json", 'w', encoding='utf-8') as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)

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