"""
B-1400 数据 MLM 微调脚本

使用 B-1400 数据对 Chronos-2 模型进行 MLM 微调。

使用方法:
    python scripts/training/train_b1400_mlm.py --data_dir data/B-1400 --num_steps 1000 --batch_size 64
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 设置项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from chronos import Chronos2Pipeline
from peft import LoraConfig, get_peft_model
from scripts.data.load_b1400_data import load_all_data, DEFAULT_VARIABLES

logger = logging.getLogger(__name__)


class MLMDataset(Dataset):
    """MLM 数据集"""
    
    def __init__(
        self,
        data_list: list,
        context_length: int = 256,
    ):
        self.data_list = data_list
        self.context_length = context_length
        
        # 过滤掉长度不足的数据
        self.valid_data = []
        for item in data_list:
            if len(item["target"]) >= context_length:
                self.valid_data.append(item)
        
        logger.info(f"有效数据: {len(self.valid_data)} / {len(data_list)}")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        data = item["target"]
        
        # 随机选择一个片段
        max_start = len(data) - self.context_length
        start_idx = np.random.randint(0, max_start + 1)
        segment = data[start_idx:start_idx + self.context_length].copy()
        
        return {
            "target": torch.tensor(segment, dtype=torch.float32),
            "file": item["file"],
            "column": item["column"],
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


def train_mlm(
    model,
    train_dataset,
    num_steps: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    device: str = "cuda",
    output_dir: str = None,
):
    """MLM 训练"""
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
    )
    
    # 训练
    model.train()
    global_step = 0
    total_loss = 0
    losses = []
    
    logger.info(f"开始训练, 总步数: {num_steps}")
    logger.info(f"设备: {device}")
    
    progress_bar = tqdm(total=num_steps, desc="训练进度")
    
    while global_step < num_steps:
        for batch in train_loader:
            if global_step >= num_steps:
                break
            
            # 准备输入
            context = batch["target"].to(device)
            
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
            scheduler.step()
            
            global_step += 1
            total_loss += loss.item()
            losses.append(loss.item())
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % 100 == 0:
                avg_loss = total_loss / global_step
                logger.info(f"Step {global_step}: 平均损失 = {avg_loss:.4f}")
    
    progress_bar.close()
    
    avg_loss = total_loss / num_steps
    logger.info(f"训练完成: 平均损失 = {avg_loss:.4f}")
    
    # 保存模型
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 LoRA 权重
        model.save_pretrained(output_path / "mlm-lora-b1400-finetuned")
        logger.info(f"模型已保存到: {output_path / 'mlm-lora-b1400-finetuned'}")
        
        # 保存训练日志
        log_path = output_path / "training_log.txt"
        with open(log_path, "w") as f:
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"训练步数: {num_steps}\n")
            f.write(f"批次大小: {batch_size}\n")
            f.write(f"学习率: {learning_rate}\n")
            f.write(f"平均损失: {avg_loss:.4f}\n")
            f.write(f"数据集大小: {len(train_dataset)}\n")
            f.write(f"\n损失历史:\n")
            for i, loss in enumerate(losses):
                if i % 10 == 0:
                    f.write(f"Step {i}: {loss:.4f}\n")
        
        logger.info(f"训练日志已保存到: {log_path}")
    
    return losses


def main():
    parser = argparse.ArgumentParser(description="B-1400 数据 MLM 微调")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/chronos-2",
        help="模型路径",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/B-1400",
        help="数据目录",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="要提取的变量列表",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="最大文件数",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="上下文长度",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.15,
        help="mask 比例",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="训练步数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="warmup 步数",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA r 参数",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha 参数",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weights/mlm_b1400_finetuned",
        help="输出目录",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda")
    model = pipeline.model
    logger.info(f"模型加载成功, 设备: {model.device}")
    
    # 配置 LoRA
    logger.info("配置 LoRA...")
    model = setup_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
    )
    
    # 加载数据
    logger.info(f"加载数据: {args.data_dir}")
    all_data, stats = load_all_data(
        args.data_dir,
        args.variables,
        args.max_files,
        min_length=args.context_length,
    )
    
    if not all_data:
        logger.error("没有找到有效数据!")
        return
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = MLMDataset(
        all_data,
        context_length=args.context_length,
    )
    
    logger.info(f"数据集大小: {len(train_dataset)}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / timestamp
    
    # 训练
    logger.info("开始训练...")
    losses = train_mlm(
        model,
        train_dataset,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        device=str(model.device),
        output_dir=str(output_dir),
    )
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"模型保存路径: {output_dir / 'mlm-lora-b1400-finetuned'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()