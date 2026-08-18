"""
Chronos-2 MLM (Masked Language Modeling) 微调训练脚本

将 Chronos-2 从自回归预测模型改造为类似 BERT 的随机掩码模型，
用于时序插值和异常检测。

使用方法:
    python scripts/training/train_mlm.py --data_path <数据路径> [选项]

示例:
    # 使用本地权重进行 LoRA 微调
    python scripts/training/train_mlm.py \
        --model_path ./weights/chronos-2 \
        --data_path ./data/train.json \
        --finetune_mode lora \
        --num_steps 2000 \
        --learning_rate 1e-5

    # 全量微调
    python scripts/training/train_mlm.py \
        --model_path ./weights/chronos-2 \
        --data_path ./data/train.json \
        --finetune_mode full \
        --num_steps 1000 \
        --learning_rate 1e-6
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np
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


def load_pipeline_from_local(model_path: str) -> Chronos2Pipeline:
    """从本地路径加载 Chronos2Pipeline"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"本地权重路径不存在: {model_path}")

    logger.info(f"从本地加载模型: {model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(str(model_path))
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")
    return pipeline


def generate_sample_data(context_length: int = 512, num_series: int = 100):
    """生成示例训练数据（正弦波 + 噪声），用于测试"""
    data = []
    for i in range(num_series):
        freq = np.random.uniform(0.01, 0.1)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 2.0)
        noise_level = np.random.uniform(0.01, 0.1)
        t = np.arange(context_length + 64)
        series = amplitude * np.sin(2 * np.pi * freq * t + phase) + noise_level * np.random.randn(len(t))
        data.append(series)
    return data


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 MLM 微调训练")

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(WEIGHTS_DIR / "chronos-2"),
        help="本地模型权重路径 (默认: ./weights/chronos-2)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="训练数据路径 (JSON/CSV 格式)。如不提供则使用示例数据",
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
        default=None,
        help="验证数据路径",
    )

    # 训练参数
    parser.add_argument("--finetune_mode", type=str, default="lora", choices=["full", "lora"], help="微调模式")
    parser.add_argument("--num_steps", type=int, default=2000, help="训练步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率 (LoRA 推荐 1e-5, 全量推荐 1e-6)")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--context_length", type=int, default=None, help="上下文长度 (默认使用模型默认值)")
    parser.add_argument("--prediction_length", type=int, default=64, help="预测长度 (MLM 模式下仅影响数据切片)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录 (默认: ./weights/mlm_finetuned/<时间戳>)",
    )
    parser.add_argument(
        "--backup_original",
        action="store_true",
        default=True,
        help="训练前备份原始权重 (默认开启)",
    )
    parser.add_argument(
        "--no_backup_original",
        action="store_true",
        default=False,
        help="不备份原始权重",
    )

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

    # 2. 加载模型
    pipeline = load_pipeline_from_local(args.model_path)

    # 3. 准备训练数据
    if args.data_path is not None:
        # 从文件加载数据
        import json

        data_path = Path(args.data_path)
        if data_path.suffix == ".json":
            with open(data_path, "r") as f:
                raw_data = json.load(f)
            # 假设数据格式为 list of dict, 每个包含 "target" 字段
            if isinstance(raw_data, list) and len(raw_data) > 0:
                if isinstance(raw_data[0], dict) and "target" in raw_data[0]:
                    train_inputs = raw_data
                else:
                    # 假设是简单的数值列表
                    train_inputs = [{"target": np.array(s)} for s in raw_data]
            else:
                raise ValueError(f"无法解析数据格式: {data_path}")
        else:
            raise ValueError(f"目前仅支持 JSON 格式的数据文件, 收到: {data_path.suffix}")
    else:
        logger.info("未提供训练数据路径, 使用示例正弦波数据进行测试")
        sample_series = generate_sample_data()
        train_inputs = [{"target": s} for s in sample_series]

    # 准备验证数据
    validation_inputs = None
    if args.validation_data_path is not None:
        import json

        with open(args.validation_data_path, "r") as f:
            val_data = json.load(f)
        if isinstance(val_data, list) and len(val_data) > 0:
            if isinstance(val_data[0], dict) and "target" in val_data[0]:
                validation_inputs = val_data
            else:
                validation_inputs = [{"target": np.array(s)} for s in val_data]

    # 4. 配置 LoRA (如果使用)
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
    logger.info("开始 MLM (Masked Language Modeling) 微调训练")
    logger.info(f"  模型路径: {args.model_path}")
    logger.info(f"  微调模式: {args.finetune_mode}")
    logger.info(f"  训练步数: {args.num_steps}")
    logger.info(f"  学习率: {args.learning_rate}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)

    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        validation_inputs=validation_inputs,
        finetune_mode=args.finetune_mode,
        lora_config=lora_config,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=output_dir,
        mlm_mode=True,
    )

    # 7. 保存微调后的模型到本地权重目录
    final_save_path = output_dir / "mlm-finetuned-final"
    finetuned_pipeline.save_pretrained(final_save_path)
    logger.info(f"微调后的模型已保存到: {final_save_path}")

    # 8. 备份微调后的权重
    backup_dir = WEIGHTS_DIR / "backups"
    backup_weights(final_save_path, backup_dir, tag="mlm_finetuned")

    logger.info("MLM 微调训练完成!")


if __name__ == "__main__":
    main()
