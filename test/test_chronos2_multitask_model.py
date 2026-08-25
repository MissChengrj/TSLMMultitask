import json
from pathlib import Path

import torch

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.pipeline import Chronos2Pipeline
from tslm_multitask.models import Chronos2MultiTaskModel
from scripts.training.train_aero_multitask_chronos2 import (
    _channel_normalized_residual,
    configure_trainable,
)


def _dummy_config() -> Chronos2CoreConfig:
    config_path = Path(__file__).parent / "dummy-chronos2-model" / "config.json"
    return Chronos2CoreConfig(**json.loads(config_path.read_text(encoding="utf-8")))


def test_base_chronos2_prepare_context_has_no_mlm_mask():
    model = Chronos2Model(_dummy_config())
    model.train()

    _, _, _, mlm_mask = model._prepare_patched_context(torch.randn(2, 32))

    assert not mlm_mask.any()


def test_multitask_mask_ratio_controls_training_mask():
    model = Chronos2MultiTaskModel(_dummy_config())
    model.train()
    model.mask_ratio = 1.0

    _, _, _, mlm_mask = model._prepare_patched_context(torch.randn(2, 32))

    assert mlm_mask.all()


def test_reconstruct_context_returns_context_length():
    model = Chronos2MultiTaskModel(_dummy_config())
    model.eval()
    context = torch.randn(2, 32)
    reconstruction_mask = torch.zeros_like(context, dtype=torch.bool)
    reconstruction_mask[:, 8:16] = True

    with torch.no_grad():
        preds = model.reconstruct_context(context=context, reconstruction_mask=reconstruction_mask)

    assert preds.shape == (2, len(model.chronos_config.quantiles), 32)


def test_reconstruct_context_with_missing_values_is_finite():
    model = Chronos2MultiTaskModel(_dummy_config())
    model.eval()
    context = torch.randn(2, 32)
    context[:, :8] = float("nan")
    context_mask = torch.isfinite(context)

    with torch.no_grad():
        preds = model.reconstruct_context(context=context, context_mask=context_mask)

    assert preds.shape == (2, len(model.chronos_config.quantiles), 32)
    assert torch.isfinite(preds).all()


def test_forward_with_nan_future_covariates_is_finite():
    model = Chronos2MultiTaskModel(_dummy_config())
    model.eval()
    context = torch.randn(2, 32)
    future_target = torch.randn(2, 16)
    future_covariates = torch.full((2, 16), float("nan"))

    with torch.no_grad():
        outputs = model(
            context=context,
            future_target=future_target,
            future_covariates=future_covariates,
            num_output_patches=1,
        )

    assert torch.isfinite(outputs.loss)
    assert torch.isfinite(outputs.quantile_preds).all()


def test_training_backward_with_missing_values_has_finite_gradients():
    model = Chronos2MultiTaskModel(_dummy_config())
    model.train()
    model.mask_ratio = 0.5
    context = torch.randn(2, 32)
    context[:, 4:8] = float("nan")
    future_target = torch.randn(2, 16)
    future_target[:, 2:5] = float("nan")
    future_covariates = torch.full((2, 16), float("nan"))

    outputs = model(
        context=context,
        future_target=future_target,
        future_covariates=future_covariates,
        num_output_patches=1,
    )
    outputs.loss.backward()

    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)


def test_clean_reconstruction_target_keeps_masked_nan_loss_finite():
    model = Chronos2MultiTaskModel(_dummy_config())
    model.train()
    model.forecast_loss_weight = 0.0
    model.recon_loss_weight = 1.0
    clean_target = torch.randn(2, 32)
    reconstruction_mask = torch.zeros_like(clean_target, dtype=torch.bool)
    reconstruction_mask[:, 8:12] = True
    context = clean_target.clone()
    context[reconstruction_mask] = float("nan")
    context_mask = torch.isfinite(context)

    output = model(
        context=context,
        context_mask=context_mask,
        reconstruction_target=clean_target,
        reconstruction_target_mask=reconstruction_mask,
        reconstruction_mask=reconstruction_mask,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)


def test_metadata_gates_preserve_base_behavior_until_enabled():
    model = Chronos2MultiTaskModel(_dummy_config())
    model.eval()
    context = torch.randn(2, 32)
    first_ids = {"domain": torch.tensor([1, 1])}
    second_ids = {"domain": torch.tensor([2, 2])}

    with torch.no_grad():
        first = model(context=context, metadata_ids=first_ids).quantile_preds
        second = model(context=context, metadata_ids=second_ids).quantile_preds
        model.metadata_gates["domain"].fill_(1.0)
        enabled = model(context=context, metadata_ids=second_ids).quantile_preds

    assert torch.equal(first, second)
    assert not torch.equal(first, enabled)


def test_reconstruction_head_is_independent_from_forecast_head():
    model = Chronos2MultiTaskModel(_dummy_config())

    assert model.reconstruction_head is not model.output_patch_embedding
    assert all(
        reconstruction is not forecast
        for reconstruction, forecast in zip(
            model.reconstruction_head.parameters(), model.output_patch_embedding.parameters()
        )
    )


def test_base_initialization_copies_head_and_zeros_metadata_gates():
    model = Chronos2MultiTaskModel(_dummy_config())
    with torch.no_grad():
        for gate in model.metadata_gates.values():
            gate.fill_(123.0)
        for parameter in model.reconstruction_head.parameters():
            parameter.zero_()

    model.initialize_aero_modules_from_base()

    assert all(gate.item() == 0.0 for gate in model.metadata_gates.values())
    assert all(
        torch.equal(reconstruction, forecast)
        for reconstruction, forecast in zip(
            model.reconstruction_head.parameters(), model.output_patch_embedding.parameters()
        )
    )


def test_reconstruction_only_mode_preserves_forecast_parameters():
    model = Chronos2MultiTaskModel(_dummy_config())

    configure_trainable(model, "reconstruction_head")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("reconstruction_head") for name in trainable)
    assert not any(name.startswith("output_patch_embedding") for name in trainable)
    assert not any(name.startswith("input_patch_embedding") for name in trainable)


def test_anomaly_residual_is_normalized_per_channel():
    target = torch.tensor(
        [[0.0, 100.0, 200.0, 300.0], [0.0, 1.0, 2.0, 3.0]]
    )
    prediction = target + torch.tensor(
        [[10.0, 10.0, 10.0, 20.0], [0.1, 0.1, 0.1, 0.2]]
    )
    mask = torch.ones_like(target, dtype=torch.bool)

    scores = _channel_normalized_residual(prediction, target, mask)

    assert torch.allclose(scores[0], scores[1], atol=1e-6)
    assert scores[0, -1] > scores[0, 0]


def test_multitask_checkpoint_loads_through_default_pipeline(tmp_path):
    model = Chronos2MultiTaskModel(_dummy_config())
    model.save_pretrained(tmp_path)

    pipeline = Chronos2Pipeline.from_pretrained(tmp_path, device_map="cpu")

    assert isinstance(pipeline.model, Chronos2MultiTaskModel)
    assert hasattr(pipeline.model, "reconstruct_context")
