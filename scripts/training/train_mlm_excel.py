"""
使用 GluonTS 加载 Excel 数据进行 Chronos-2 MLM LoRA 微调

数据格式: 航空发动机传感器时序数据
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

logger = logging.getLogger(__name__)

# 本地权重目录
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"


def backup_weights(source_path: Path, backup_dir: Path, tag: str = "") -> Path:
    """备份权重文件到指定目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source_path.name}_{tag}_{timestamp}" if tag else f"{source_path.name}_{timestamp}"
    backup_path = backup_dir / backup_name
    backup_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        shutil.copytree(source_path, backup_path)
    else:
        shutil.copy2(source_path, backup_path)

    logger.info(f"权重已备份到: {backup_path}")
    return backup_path


def load_excel_data(
    file_path: str,
    target_columns: list[str] | None = None,
    freq: str = "H",
) -> list[dict]:
    """
    从 Excel 文件加载时序数据并转换为 Chronos 格式

    Parameters
    ----------
    file_path : str
        Excel 文件路径
    target_columns : list[str] | None
        要作为目标的列名列表。如果为 None，自动选择所有数值列
    freq : str
        时间频率，默认 "H" (小时)

    Returns
    -------
    list[dict]
        Chronos 格式的数据列表，每个元素包含 "target" 和可选的 "start"
    """
    # 读取 Excel，跳过头部信息行
    df = pd.read_excel(file_path, header=6, skiprows=1)

    logger.info(f"原始数据形状: {df.shape}")
    logger.info(f"列名: {df.columns.tolist()}")

    # 解析时间列
    if "Flight DateTime" in df.columns:
        df["Flight DateTime"] = pd.to_datetime(df["Flight DateTime"])
        df = df.sort_values("Flight DateTime")
        start_time = df["Flight DateTime"].iloc[0]
    else:
        start_time = pd.Timestamp("2024-03-12", freq=freq)

    # 选择数值列作为目标
    if target_columns is None:
        # 自动选择数值类型的列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除一些非传感器列
        exclude_cols = ["ENG POS"]
        target_columns = [c for c in numeric_cols if c not in exclude_cols]

    logger.info(f"目标列: {target_columns}")

    # 提取目标数据
    target_data = df[target_columns].values

    # 处理缺失值
    target_data = np.where(np.isnan(target_data), 0.0, target_data)

    # 转换为 Chronos 格式
    # 对于多变量时序，我们将其拆分为多个单变量序列
    inputs = []

    # 方式1: 每个传感器作为一个独立的单变量时序
    for i, col in enumerate(target_columns):
        series = target_data[:, i]
        inputs.append({
            "target": series.astype(np.float32),
        })

    logger.info(f"创建了 {len(inputs)} 个单变量时序序列")
    logger.info(f"每个序列长度: {len(series)}")

    return inputs


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 MLM LoRA 微调 - Excel 数据")

    # 数据参数
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/038227_1.xlsx",
        help="Excel 数据文件路径",
    )
    parser.add_argument(
        "--target_columns",
        type=str,
        nargs="+",
        default=None,
        help="目标列名（如不指定则自动选择所有数值列）",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="H",
        help="时间频率 (H=小时, D=天, etc.)",
    )

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(WEIGHTS_DIR / "chronos-2"),
        help="本地模型权重路径",
    )

    # 训练参数
    parser.add_argument("--finetune_mode", type=str, default="lora", choices=["full", "lora"])
    parser.add_argument("--num_steps", type=int, default=2000, help="训练步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--context_length", type=int, default=512, help="上下文长度")
    parser.add_argument("--prediction_length", type=int, default=64, help="预测长度")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--backup_original", action="store_true", default=True)
    parser.add_argument("--no_backup_original", action="store_true", default=False)

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # 1. 备份原始权重
    model_path = Path(args.model_path)
    if args.backup_original and not args.no_backup_original and model_path.exists():
        backup_dir = WEIGHTS_DIR / "backups"
        backup_weights(model_path, backup_dir, tag="original")

    # 2. 加载模型到 GPU
    logger.info(f"从本地加载模型: {model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(str(model_path), device_map="cuda")
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")

    # 3. 加载 Excel 数据
    logger.info(f"加载 Excel 数据: {args.data_path}")
    train_inputs = load_excel_data(
        file_path=args.data_path,
        target_columns=args.target_columns,
        freq=args.freq,
    )

    # 4. 配置 LoRA
    lora_config = None
    if args.finetune_mode == "lora":
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "self_attention.q",
                "self_attention.v",
                "self_attention.k",
                "self_attention.o",
                "output_patch_embedding.output_layer",
            ],
        )
        logger.info(f"使用 LoRA 微调: r={args.lora_r}, alpha={args.lora_alpha}")

    # 5. 设置输出目录
    if args.output_dir is None:
        output_dir = WEIGHTS_DIR / "mlm_finetuned" / time.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. 开始 MLM 微调
    logger.info("=" * 60)
    logger.info("开始 MLM (Masked Language Modeling) LoRA 微调")
    logger.info(f"  数据文件: {args.data_path}")
    logger.info(f"  序列数量: {len(train_inputs)}")
    logger.info(f"  微调模式: {args.finetune_mode}")
    logger.info(f"  训练步数: {args.num_steps}")
    logger.info(f"  学习率: {args.learning_rate}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  上下文长度: {args.context_length}")
    logger.info("=" * 60)

    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        finetune_mode=args.finetune_mode,
        lora_config=lora_config,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=output_dir,
        mlm_mode=True,  # 启用 MLM 模式
    )

    # 7. 保存微调后的模型
    final_save_path = output_dir / "mlm-lora-finetuned"
    finetuned_pipeline.save_pretrained(final_save_path)
    logger.info(f"微调后的模型已保存到: {final_save_path}")

    # 8. 备份微调后的权重
    backup_dir = WEIGHTS_DIR / "backups"
    backup_weights(final_save_path, backup_dir, tag="mlm_lora_finetuned")

    logger.info("MLM LoRA 微调训练完成!")

    # 9. 测试推理
    logger.info("测试推理...")
    test_series = train_inputs[0]["target"][:args.context_length]
    test_input = [{"target": test_series}]

    # 使用微调后的模型进行推理（插值/重构）
    prediction = finetuned_pipeline.predict(
        inputs=test_input,
        prediction_length=args.prediction_length,
    )

    # prediction 是一个 list，每个元素是一个 tensor (batch, quantiles, horizon)
    if isinstance(prediction, list):
        logger.info(f"推理结果数量: {len(prediction)}")
        logger.info(f"第一个预测形状: {prediction[0].shape}")
    else:
        logger.info(f"推理结果形状: {prediction.shape}")
    logger.info("推理测试完成!")


if __name__ == "__main__":
    main()