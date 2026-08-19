"""Chronos-2 wrapper with an auxiliary masked-reconstruction objective."""

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange, repeat

from chronos.chronos2.model import Chronos2Model, Chronos2Output


class Chronos2MultiTaskModel(Chronos2Model):
    """Forecast with the base future head while also optimizing context reconstruction."""

    def __init__(self, config):
        super().__init__(config)
        self.forecast_loss_weight = 1.0
        self.recon_loss_weight = 0.5
        self.mask_ratio = 0.15
        self.normalized_clip_value = 100.0
        self._forced_reconstruction_mask: torch.Tensor | None = None

    def _clip_normalized(self, tensor: torch.Tensor) -> torch.Tensor:
        clip_value = float(getattr(self, "normalized_clip_value", 0.0) or 0.0)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        if clip_value > 0:
            tensor = tensor.clamp(-clip_value, clip_value)
        return tensor

    def _prepare_patched_context(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        finite_context_mask = torch.isfinite(context).to(context.dtype)
        context_mask = (
            context_mask.to(context.dtype) * finite_context_mask
            if context_mask is not None
            else finite_context_mask
        )
        context = torch.where(finite_context_mask > 0, context, torch.full_like(context, torch.nan))

        batch_size, context_length = context.shape
        if context_length > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            context_mask = context_mask[..., -self.chronos_config.context_length :]

        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)
        context = torch.where(context_mask > 0, context, torch.zeros_like(context))
        context = self._clip_normalized(context)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        attention_mask = patched_mask.sum(dim=-1) > 0
        num_context_patches = attention_mask.shape[-1]

        if self._forced_reconstruction_mask is not None:
            mlm_mask = self._coerce_patch_mask(self._forced_reconstruction_mask, num_context_patches, context.device)
        elif self.training and self.mask_ratio > 0:
            mlm_mask = (torch.rand(patched_context.shape[:-1], device=self.device) < self.mask_ratio) & attention_mask
        else:
            mlm_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

        attention_mask = attention_mask | mlm_mask
        patched_context = torch.where(mlm_mask.unsqueeze(-1), torch.zeros_like(patched_context), patched_context)

        final_context_length = num_context_patches * self.chronos_config.input_patch_size
        context_time_enc = torch.arange(start=-final_context_length, end=0, device=self.device, dtype=torch.float32)
        context_time_enc = (
            repeat(
                context_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_context_patches,
                p=self.chronos_config.input_patch_size,
            )
            .div(int(self.chronos_config.time_encoding_scale))
            .to(self.dtype)
        )

        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)
        return patched_context, attention_mask, loc_scale, mlm_mask

    def _prepare_patched_future(
        self,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_patch_size = self.chronos_config.output_patch_size
        if future_covariates is not None:
            finite_future_mask = torch.isfinite(future_covariates).to(future_covariates.dtype)
            future_covariates_mask = (
                future_covariates_mask.to(future_covariates.dtype) * finite_future_mask
                if future_covariates_mask is not None
                else finite_future_mask
            )
            future_covariates = torch.where(
                finite_future_mask > 0,
                future_covariates,
                torch.full_like(future_covariates, torch.nan),
            )
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = future_covariates.to(self.dtype)
            future_covariates_mask = future_covariates_mask.to(self.dtype)

            # Chronos-2 tokenization: after the mask channel is constructed, missing values are zero-filled.
            future_covariates = torch.where(
                future_covariates_mask > 0.0,
                future_covariates,
                torch.zeros_like(future_covariates),
            )
            future_covariates = self._clip_normalized(future_covariates)

            if num_output_patches * output_patch_size > future_covariates.shape[-1]:
                padding_shape = (
                    *future_covariates.shape[:-1],
                    num_output_patches * output_patch_size - future_covariates.shape[-1],
                )
                future_covariates = torch.cat(
                    [future_covariates, torch.zeros(padding_shape).to(future_covariates)], dim=-1
                )
                future_covariates_mask = torch.cat(
                    [future_covariates_mask, torch.zeros(padding_shape).to(future_covariates_mask)], dim=-1
                )

            patched_future_covariates = rearrange(
                future_covariates, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
            )
            patched_future_covariates_mask = rearrange(
                future_covariates_mask, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
            )
        else:
            patched_future_covariates = torch.zeros(
                batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype
            )
            patched_future_covariates_mask = torch.zeros(
                batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype
            )

        final_future_length = num_output_patches * output_patch_size
        future_time_enc = torch.arange(start=0, end=final_future_length, device=self.device, dtype=torch.float32)
        future_time_enc = (
            repeat(
                future_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_output_patches,
                p=output_patch_size,
            )
            .div(int(self.chronos_config.time_encoding_scale))
            .to(self.dtype)
        )

        patched_future = torch.cat(
            [future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1
        )
        return patched_future, patched_future_covariates_mask

    def _coerce_patch_mask(self, mask: torch.Tensor, num_context_patches: int, device: torch.device) -> torch.Tensor:
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 2:
            raise ValueError(f"reconstruction mask must be 2-D, found: {tuple(mask.shape)}")
        if mask.shape[-1] == num_context_patches:
            return mask
        if mask.shape[-1] == num_context_patches * self.chronos_config.input_patch_size:
            return rearrange(mask, "b (n p) -> b n p", n=num_context_patches, p=self.chronos_config.input_patch_size).any(dim=-1)
        raise ValueError(
            "reconstruction mask length must equal number of context patches or context timesteps, "
            f"found {mask.shape[-1]} for {num_context_patches} patches"
        )

    def reconstruct_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        reconstruction_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Reconstruct context patches without using the future forecasting head."""
        previous_mask = self._forced_reconstruction_mask
        self._forced_reconstruction_mask = reconstruction_mask
        try:
            encoder_outputs, loc_scale, _, num_context_patches, _ = self.encode(
                context=context,
                context_mask=context_mask,
                group_ids=group_ids,
                num_output_patches=1,
                output_attentions=output_attentions,
            )
        finally:
            self._forced_reconstruction_mask = previous_mask

        hidden_states: torch.Tensor = encoder_outputs[0]
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        recon_embeds = hidden_states[:, :num_context_patches]
        recon_preds = self.output_patch_embedding(recon_embeds)
        recon_preds = rearrange(
            recon_preds,
            "b n (q p) -> b q (n p)",
            n=num_context_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )
        recon_preds = torch.nan_to_num(recon_preds, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        recon_preds = rearrange(
            recon_preds,
            "b q h -> b (q h)",
            q=self.num_quantiles,
            h=num_context_patches * self.chronos_config.output_patch_size,
        )
        recon_preds = self.instance_norm.inverse(recon_preds, loc_scale)
        recon_preds = rearrange(
            recon_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_context_patches * self.chronos_config.output_patch_size,
        )
        return recon_preds[..., -context.shape[-1] :]

    def _compute_mlm_loss(
        self,
        quantile_preds: torch.Tensor,
        original_context: torch.Tensor,
        context_mask: torch.Tensor | None,
        mlm_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        target, _ = self.instance_norm(original_context, loc_scale)
        target = target.unsqueeze(1).to(self.device)

        mlm_mask_expanded = repeat(
            mlm_mask,
            "b n -> b 1 (n p)",
            p=self.chronos_config.input_patch_size,
        ).float()

        target_length = target.shape[-1]
        if quantile_preds.shape[-1] > target_length:
            quantile_preds = quantile_preds[..., -target_length:]
            mlm_mask_expanded = mlm_mask_expanded[..., -target_length:]
        elif quantile_preds.shape[-1] < target_length:
            target = target[..., -quantile_preds.shape[-1]:]
            mlm_mask_expanded = mlm_mask_expanded[..., -quantile_preds.shape[-1]:]
            if context_mask is not None:
                context_mask = context_mask[..., -quantile_preds.shape[-1]:]

        valid_mask = (
            context_mask.unsqueeze(1).to(self.device)
            if context_mask is not None
            else ~torch.isnan(target)
        ).float()
        loss_mask = valid_mask * mlm_mask_expanded

        target = self._clip_normalized(target)
        target = torch.where(loss_mask > 0, target, torch.zeros_like(target))
        quantile_preds = self._clip_normalized(quantile_preds)

        quantiles = rearrange(self.quantiles, "num_quantiles -> 1 num_quantiles 1")
        quantile_loss = 2 * torch.abs(
            (target - quantile_preds) * ((target <= quantile_preds).float() - quantiles)
        )
        loss = quantile_loss * loss_mask

        sum_loss = loss.sum(dim=-1).sum(dim=-1)
        num_masked = loss_mask.sum(dim=-1).squeeze(1) + 1e-8
        return (sum_loss / num_masked).mean()

    def _compute_loss(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        future_target_mask: torch.Tensor | None,
        patched_future_covariates_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
    ) -> torch.Tensor:
        batch_size = future_target.shape[0]
        output_patch_size = self.chronos_config.output_patch_size
        assert quantile_preds.shape[0] == batch_size

        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1).to(self.device)
        finite_target_mask = torch.isfinite(future_target)
        if future_target_mask is not None:
            future_target_mask = future_target_mask.unsqueeze(1).to(self.device).bool() & finite_target_mask
        else:
            future_target_mask = finite_target_mask

        target_length = min(future_target.shape[-1], quantile_preds.shape[-1])
        future_target = future_target[..., :target_length]
        future_target_mask = future_target_mask[..., :target_length]
        quantile_preds = quantile_preds[..., :target_length]

        inv_future_covariate_mask = ~rearrange(
            patched_future_covariates_mask,
            "b n p -> b 1 (n p)",
            b=batch_size,
            n=num_output_patches,
            p=output_patch_size,
        ).bool()
        inv_future_covariate_mask = inv_future_covariate_mask[..., :target_length]

        loss_mask = future_target_mask & inv_future_covariate_mask & torch.isfinite(quantile_preds).all(dim=1, keepdim=True)
        future_target = self._clip_normalized(future_target)
        future_target = torch.where(loss_mask, future_target, torch.zeros_like(future_target))
        quantile_preds = self._clip_normalized(quantile_preds)

        quantiles = rearrange(self.quantiles, "num_quantiles -> 1 num_quantiles 1")
        quantile_loss = 2 * torch.abs(
            (future_target - quantile_preds) * ((future_target <= quantile_preds).float() - quantiles)
        )
        loss = quantile_loss * loss_mask.float()
        denom = loss_mask.float().sum(dim=-1).clamp_min(1.0)
        return (loss.sum(dim=-1).sum(dim=-1) / denom.squeeze(1)).mean()

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
    ) -> Chronos2Output:
        batch_size = context.shape[0]

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
        )
        hidden_states: torch.Tensor = encoder_outputs[0]
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

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

        forecast_embeds = hidden_states[:, -num_output_patches:]
        forecast_preds = self.output_patch_embedding(forecast_embeds)
        forecast_preds = rearrange(
            forecast_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )
        forecast_preds = torch.nan_to_num(forecast_preds, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

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

        if forecast_loss is not None:
            total_loss = self.forecast_loss_weight * forecast_loss + self.recon_loss_weight * mlm_loss
        else:
            total_loss = self.recon_loss_weight * mlm_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e4, neginf=-1e4)

        forecast_preds_unscaled = rearrange(
            forecast_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )
        forecast_preds_unscaled = self.instance_norm.inverse(forecast_preds_unscaled, loc_scale)
        forecast_preds_unscaled = torch.nan_to_num(
            forecast_preds_unscaled,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
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
