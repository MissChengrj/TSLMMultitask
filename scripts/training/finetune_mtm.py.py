import os
import sys
from pathlib import Path

# ==========================================
# 【路径配置】：将 src 目录加入环境变量最高优先级
# 确保能够识别 chronos 包及其子模块
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import glob
import logging
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model

# 导入采用 interrelated encoding and independent decoding 架构的多任务外壳
from chronos.chronos2.ieid_chronos import IEIDMultitaskModel 

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 动态数据加载器 (处理多文件夹、变长、变特征)
# ==========================================
class EngineLocalDataset(Dataset):
    def __init__(self, data_dir, seq_len=256, patch_size=16):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.tasks = ['forecast', 'impute', 'anomaly']
        
        # 递归扫描所有子文件夹中的 CSV 文件
        self.file_paths = list(self.data_dir.rglob('*.csv'))
        logger.info(f"初始化数据集：在 {data_dir} 中找到 {len(self.file_paths)} 个 CSV 文件")

    def __len__(self):
        return len(self.file_paths)

    def generate_patch_mask(self, task_type, num_patches):
        """生成基于 Patch 层级的掩码"""
        mask = torch.zeros(num_patches, dtype=torch.bool)
        if task_type == 'forecast':
            mask[-int(num_patches * 0.2):] = True
        elif task_type == 'impute':
            start_idx = random.randint(0, int(num_patches * 0.6))
            mask[start_idx : start_idx + int(num_patches * 0.15)] = True
        elif task_type == 'anomaly':
            if random.random() > 0.5:
                mask_indices = torch.randperm(num_patches)[:int(num_patches * 0.05)]
                mask[mask_indices] = True
        return mask

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        try:
            # 读取CSV，pandas默认将第一行作为列名
            df = pd.read_csv(file_path)
            # 处理异常值 (前向填充 -> 后向填充 -> 补零)
            df = df.ffill().bfill().fillna(0)
            data = torch.tensor(df.values, dtype=torch.float32).T 
        except Exception as e:
            logger.warning(f"读取文件失败跳过 {file_path}: {e}")
            data = torch.zeros((1, self.seq_len))
            
        if data.numel() == 0:
            data = torch.zeros((1, self.seq_len))
            
        num_features, current_len = data.shape
        
        # 解决长度不一致：随机裁剪或补零
        if current_len > self.seq_len:
            start = random.randint(0, current_len - self.seq_len)
            data = data[:, start : start + self.seq_len]
        elif current_len < self.seq_len:
            pad_len = self.seq_len - current_len
            padding = torch.zeros(num_features, pad_len)
            data = torch.cat([data, padding], dim=1)
            
        task_type = random.choice(self.tasks)
        num_patches = self.seq_len // self.patch_size
        mask = self.generate_patch_mask(task_type, num_patches)
        
        return data, mask, task_type

def collate_multivariate_fn(batch):
    """
    将不同特征数量的样本展平，并分配 Group ID，以适应 GroupSelfAttention。
    """
    all_series, all_masks, all_group_ids, task_types = [], [], [], []
    
    group_id_counter = 0
    for data, mask, task_type in batch:
        num_features = data.shape[0]
        for i in range(num_features):
            all_series.append(data[i])
            all_masks.append(mask)
            all_group_ids.append(group_id_counter)
        
        group_id_counter += 1
        task_types.append(task_type)
        
    return {
        "context_series": torch.stack(all_series),
        "mask_indices": torch.stack(all_masks),
        "group_ids": torch.tensor(all_group_ids, dtype=torch.long),
        "task_type": task_types[0] # 取 Batch 第一个任务作为当前批次任务
    }

# ==========================================
# 2. LoRA 模型配置
# ==========================================
def setup_lora_model(model):
    """配置低秩微调 (PEFT)，锁定原版权重。"""
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q", "v", "o", "wi", "wo"], # 匹配 Chronos-2 的注意力与MLP层
        lora_dropout=0.1, bias="none"
    )
    model.backbone = get_peft_model(model.backbone, lora_config)
    
    # 解冻自定义的 Token 和独立解码头
    model.mask_token.requires_grad = True
    for head in [model.impute_head, model.anomaly_head]:
        for param in head.parameters():
            param.requires_grad = True
    return model

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    # 定义目录路径 
    model_path = project_root / "weights" / "chronos-2"
    data_path = project_root / "data"
    backup_path = project_root / "weights" / "backups"
    backup_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. 加载本地模型并应用 LoRA
    logger.info(f"Loading local weights from: {model_path}")
    model = IEIDMultitaskModel(pretrained_model_path=str(model_path))
    model = setup_lora_model(model)
    model.to(device)
    
    # 2. 准备数据集与 DataLoader
    dataset = EngineLocalDataset(data_dir=data_path, seq_len=256, patch_size=model.patch_size)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_multivariate_fn)
    
    # 3. 优化器与损失函数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    mse_loss_fn = nn.MSELoss(reduction='none')
    
    # 【修复警告】：使用 PyTorch 2.x 推荐的设备无关缩放器 API
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    epochs = 10
    model.train()
    
    logger.info("Starting Multi-task Fine-tuning...")
    for epoch in range(epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            # 清理缓存碎片
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            context = batch_data["context_series"].to(device)
            mask = batch_data["mask_indices"].to(device)
            group_ids = batch_data["group_ids"].to(device)
            task_type = batch_data["task_type"]
            
            optimizer.zero_grad()
            
            # 【修复警告】：使用新版 torch.amp.autocast API
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                preds, loc_scale = model(
                    context=context, 
                    mask_indices=mask, 
                    group_ids=group_ids, 
                    task_type=task_type
                )
                
                # 真实训练中，此处应将原始 context 切分为对应的 patch_size 块作为 target
                target_patches = torch.randn_like(preds) 
                
                if task_type in ["impute", "forecast"]:
                    loss = mse_loss_fn(preds[mask], target_patches[mask]).mean()
                else:
                    loss = mse_loss_fn(preds, target_patches).mean()
                    
            # 混合精度反向传播与优化
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx} | Task: {task_type:8s} | Loss: {loss.item():.4f}")
                
    # 4. 保存微调权重
    logger.info(f"Saving fine-tuned weights to: {backup_path}")
    model.backbone.save_pretrained(backup_path / "lora_adapter")
    
    custom_state = {
        'mask_token': model.mask_token,
        'impute_head': model.impute_head.state_dict(),
        'anomaly_head': model.anomaly_head.state_dict(),
    }
    torch.save(custom_state, backup_path / "ieid_custom_heads.pt")
    logger.info("Backup complete!")

if __name__ == "__main__":
    main()