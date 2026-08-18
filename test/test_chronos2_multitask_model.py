import json
from pathlib import Path

import torch

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.pipeline import Chronos2Pipeline
from tslm_multitask.models import Chronos2MultiTaskModel


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


def test_multitask_checkpoint_loads_through_default_pipeline(tmp_path):
    model = Chronos2MultiTaskModel(_dummy_config())
    model.save_pretrained(tmp_path)

    pipeline = Chronos2Pipeline.from_pretrained(tmp_path, device_map="cpu")

    assert isinstance(pipeline.model, Chronos2MultiTaskModel)
    assert hasattr(pipeline.model, "reconstruct_context")
