"""
批量 MLM 微调脚本

使用所有数据文件对 Chronos-2 模型进行 MLM 微调。

使用方法:
    python scripts/training/batch_train_mlm.py --data_dir data --num_steps 2000 --finetune_mode lora
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 导入批量数据加载器
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from batch_data_loader import create_training_dataset, get_dataset_statistics

from chronos import Chronos2Pipeline
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# 本地权重目录
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
BACKUP_DIR = WEIGHTS_DIR / "backups"


class MLMTrainingDataset(Dataset):
    """MLM 训练数据集"""
    
    def __init__(self, data_generator, context_length: int = 256):
        """
        Parameters
        ----------
        data_generator : Iterator
            数据生成器
        context_length : int
            上下文长度
        """
        self.samples = list(data_generator)
        self.context_length = context_length
        logger.info(f"数据集大小: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = sample["target"]
        
        # 确保长度正确
        if len(target) < self.context_length:
            # 填充
            target = np.pad(target, (0, self.context_length - len(target)), mode="edge")
        elif len(target) > self.context_length:
            # 截断
            target = target[:self.context_length]
        
        return {
            "target": torch.tensor(target, dtype=torch.float32),
            "file": sample["file"],
            "column": sample["column"],
        }


def collate_fn(batch):
    """批次整理函数"""
    targets = torch.stack([item["target"] for item in batch])
    return {"target": targets}


def setup_lora(model, r: int = 8, alpha: int = 16):
    """配置 LoRA"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
        lora_dropout=0.1,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def backup_weights(model_path: Path, backup_name: str):
    """备份权重"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / backup_name
    if model_path.is_dir():
        shutil.copytree(model_path, backup_path, dirs_exist_ok=True)
        logger.info(f"权重已备份到: {backup_path}")
    return backup_path


def train_mlm(
    model_path: str,
    data_dir: str,
    output_dir: str,
    context_length: int = 256,
    stride: int = 64,
    num_steps: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    finetune_mode: str = "lora",
    lora_r: int = 8,
    lora_alpha: int = 16,
    max_files: int = None,
    mlm_prob: float = 0.15,
):
    """
    执行 MLM 微调训练
    
    Parameters
    ----------
    model_path : str
        模型路径
    data_dir : str
        数据目录
    output_dir : str
        输出目录
    context_length : int
        上下文长度
    stride : int
        滑动窗口步长
    num_steps : int
        训练步数
    batch_size : int
        批次大小
    learning_rate : float
        学习率
    finetune_mode : str
        微调模式 (lora/full)
    lora_r : int
        LoRA rank
    lora_alpha : int
        LoRA alpha
    max_files : int
        最大文件数量
    mlm_prob : float
        MLM 掩码概率
    """
    # 1. 获取数据集统计信息
    stats = get_dataset_statistics(data_dir)
    logger.info("=" * 60)
    logger.info("数据集统计信息:")
    logger.info(f"  文件数: {stats['num_files']}")
    logger.info(f"  时序数: {stats['total_series']}")
    logger.info(f"  最小长度: {stats['min_length']}")
    logger.info(f"  最大长度: {stats['max_length']}")
    logger.info(f"  平均长度: {stats['avg_length']:.1f}")
    logger.info("=" * 60)
    
    # 2. 备份原始权重
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"chronos-2_original_batch_{timestamp}"
    backup_weights(Path(model_path), backup_name)
    
    # 3. 加载模型
    logger.info(f"加载模型: {model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map="cuda")
    model = pipeline.model
    
    # 设置 MLM 模式和掩码概率
    if hasattr(model, 'mlm_prob'):
        model.mlm_prob = mlm_prob
    logger.info(f"MLM 掩码概率: {mlm_prob}")
    
    # 4. 配置微调
    if finetune_mode == "lora":
        logger.info("配置 LoRA 微调...")
        model = setup_lora(model, r=lora_r, alpha=lora_alpha)
    else:
        logger.info("全量微调模式...")
        for param in model.parameters():
            param.requires_grad = True
    
    # 5. 创建数据集
    logger.info("创建训练数据集...")
    data_generator = create_training_dataset(
        data_dir=data_dir,
        context_length=context_length,
        stride=stride,
        max_files=max_files,
    )
    dataset = MLMTrainingDataset(data_generator, context_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    # 6. 配置优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # 7. 训练循环
    logger.info("=" * 60)
    logger.info(f"开始训练: {num_steps} 步, 批次大小: {batch_size}")
    logger.info("=" * 60)
    
    model.train()
    global_step = 0
    total_loss = 0.0
    
    progress_bar = tqdm(total=num_steps, desc="训练进度")
    
    while global_step < num_steps:
        for batch in dataloader:
            if global_step >= num_steps:
                break
            
            # 准备输入
            context = batch["target"].to(model.device)
            
            # 前向传播（MLM 模式）
            outputs = model(
                context=context,
                context_mask=None,
            )
            
            loss = outputs.loss
            if loss is None:
                logger.warning("损失为 None，跳过此批次")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 每 100 步打印日志
            if global_step % 100 == 0:
                avg_loss = total_loss / global_step
                logger.info(f"Step {global_step}: 平均损失 = {avg_loss:.4f}")
    
    progress_bar.close()
    
    # 8. 保存微调后的模型
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if finetune_mode == "lora":
        # 保存 LoRA 权重
        lora_path = output_path / "mlm-lora-batch-finetuned"
        model.save_pretrained(str(lora_path))
        logger.info(f"LoRA 权重已保存到: {lora_path}")
        
        # 同时保存完整模型配置
        pipeline.save_pretrained(str(output_path / "mlm-batch-finetuned"))
    else:
        # 保存完整模型
        pipeline.save_pretrained(str(output_path / "mlm-batch-finetuned"))
        logger.info(f"完整模型已保存到: {output_path / 'mlm-batch-finetuned'}")
    
    # 9. 备份微调权重
    backup_name = f"mlm-batch-finetuned_{timestamp}"
    backup_weights(output_path, backup_name)
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"总步数: {global_step}")
    logger.info(f"平均损失: {total_loss / global_step:.4f}")
    logger.info(f"模型保存位置: {output_path}")
    logger.info("=" * 60)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="批量 MLM 微调训练")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(WEIGHTS_DIR / "chronos-2"),
        help="模型路径",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="上下文长度",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="滑动窗口步长",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=2000,
        help="训练步数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批次大小",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率",
    )
    parser.add_argument(
        "--finetune_mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="微调模式",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="最大文件数量（用于调试）",
    )
    parser.add_argument(
        "--mlm_prob",
        type=float,
        default=0.15,
        help="MLM 掩码概率",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = str(WEIGHTS_DIR / "mlm_batch_finetuned" / timestamp)
    else:
        output_dir = args.output_dir
    
    # 执行训练
    train_mlm(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=output_dir,
        context_length=args.context_length,
        stride=args.stride,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        finetune_mode=args.finetune_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_files=args.max_files,
        mlm_prob=args.mlm_prob,
    )


if __name__ == "__main__":
    main()