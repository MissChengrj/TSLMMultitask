import torch
import torch.nn as nn
from einops import rearrange
from chronos.chronos2.model import Chronos2Model

class IEIDMultitaskModel(nn.Module):
    """
    基于真实 Chronos-2 预训练权重的多任务时序模型
    采用 interrelated encoding and independent decoding 范式
    """
    def __init__(self, pretrained_model_path: str):
        super().__init__()
        
        # 1. 正确加载官方预训练模型作为 Backbone
        self.backbone = Chronos2Model.from_pretrained(pretrained_model_path)
        self.config = self.backbone.config
        self.patch_size = self.backbone.chronos_config.input_patch_size
        self.num_quantiles = self.backbone.num_quantiles
        
        # 2. 引入 MASK Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.config.d_model))
        
        # 3. 冻结基础预处理层 (InstanceNorm 与 Patch Embedding)
        for param in self.backbone.instance_norm.parameters():
            param.requires_grad = False
        for param in self.backbone.input_patch_embedding.parameters():
            param.requires_grad = False
            
        # 4. 独立解码头 (Independent Decoding)
        self.forecast_head = self.backbone.output_patch_embedding
        
        self.impute_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.patch_size)
        )
        
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.patch_size)
        )

    def forward(self, context, mask_indices, group_ids=None, task_type="forecast"):
        """
        context: (batch_size, context_length) 原始时序数据
        mask_indices: (batch_size, num_patches) 针对 Patch 的掩码布尔矩阵
        """
        # ========================================================
        # 【核心修复 1】：获取真实的底层模型，绕过 PEFT 包装器的代理干扰
        # ========================================================
        base_model = self.backbone
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model.model
            
        # ========================================================
        # 【核心修复 2】：采用安全索引解包，防止代理导致返回值长度变化
        # ========================================================
        preprocess_out = base_model._prepare_patched_context(context)
        patched_context = preprocess_out[0]
        attention_mask = preprocess_out[1]
        loc_scale = preprocess_out[2]
        
        # 获取底层 Embedding
        input_embeds = base_model.input_patch_embedding(patched_context)
        
        # 将被 MASK 的 Patch 替换为 mask_token
        batch_size, num_patches, _ = input_embeds.shape
        mask_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        mask_tokens = self.mask_token.expand(batch_size, num_patches, -1)
        input_embeds = torch.where(mask_expanded, mask_tokens, input_embeds)
        
        # ========================================================
        # 【核心修复 3】：兼容 Chronos-2 原生的 [REG] 特殊标记，对齐预训练维度
        # ========================================================
        if base_model.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), base_model.config.reg_token_id, device=input_embeds.device)
            reg_embeds = base_model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(base_model.dtype), torch.ones_like(reg_input_ids).to(base_model.dtype)], dim=-1
            )
        
        # 若没有 group_ids，默认独立处理
        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=context.device)
            
        # 经过强大的 Chronos2Encoder 提取时序特征
        encoder_outputs = base_model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # 解码前剥离 [REG] token (因为它不属于需要预测的时序 Patch)
        if base_model.chronos_config.use_reg_token:
            hidden_states = hidden_states[:, :-1, :]
        
        # 任务路由解码
        if task_type == "forecast":
            out = self.forecast_head(hidden_states)
            out = rearrange(out, 'b n (q p) -> b n q p', q=self.num_quantiles, p=self.patch_size)
        elif task_type == "impute":
            out = self.impute_head(hidden_states)
        elif task_type == "anomaly":
            out = self.anomaly_head(hidden_states)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
            
        return out, loc_scale