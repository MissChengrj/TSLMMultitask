from pathlib import Path

import numpy as np

from tslm_multitask.data.loaders import load_multivariate_values
from tslm_multitask.inference.engine import InferenceRequest, MultitaskInferenceEngine


class DummyPipeline:
    def predict(self, inputs, prediction_length):
        preds = np.vstack(
            [
                np.full(prediction_length, 1.0, dtype=np.float32),
                np.full(prediction_length, 9.0, dtype=np.float32),
                np.full(prediction_length, 17.0, dtype=np.float32),
            ]
        )
        return [preds]

    def reconstruct_context(self, context, context_mask=None, reconstruction_mask=None):
        length = context.shape[-1]
        preds = np.vstack(
            [
                np.full(length, 1.0, dtype=np.float32),
                np.full(length, 9.0, dtype=np.float32),
                np.full(length, 17.0, dtype=np.float32),
            ]
        )
        return preds[None, ...]


class ForecastOnlyPipeline:
    def predict(self, inputs, prediction_length):
        return [np.zeros((3, prediction_length), dtype=np.float32)]


class NaNFirstChunkPipeline:
    def __init__(self):
        self.calls = 0

    def reconstruct_context(self, context, context_mask=None, reconstruction_mask=None):
        self.calls += 1
        length = context.shape[-1]
        preds = np.vstack(
            [
                np.full(length, 1.0, dtype=np.float32),
                np.full(length, 9.0, dtype=np.float32),
                np.full(length, 17.0, dtype=np.float32),
            ]
        )
        if self.calls == 1:
            preds[:, :] = np.nan
        return preds[None, ...]


def _write_baseline_csv(path: Path):
    path.write_text(
        "\n".join(
            [
                "meta,,",
                "FLIGHT,A,B",
                "1,0,2",
                "2,,3",
                "3,4,5",
            ]
        ),
        encoding="utf-8",
    )


def test_loader_preserves_nan_and_zero(tmp_path):
    data_path = tmp_path / "sample.csv"
    _write_baseline_csv(data_path)

    values, observed_mask = load_multivariate_values(data_path, ["A"], "baseline")

    assert values[0].tolist()[0] == 0.0
    assert np.isnan(values[0, 1])
    assert observed_mask[0].tolist() == [True, False, True]


def test_data_repair_only_fills_missing_positions(tmp_path):
    data_path = tmp_path / "sample.csv"
    model_dir = tmp_path / "model"
    _write_baseline_csv(data_path)
    model_dir.mkdir()

    engine = MultitaskInferenceEngine(
        project_root=tmp_path,
        pipeline_factory=lambda model_path, device: DummyPipeline(),
        chunk_size=8,
    )
    result = engine.run(
        InferenceRequest(
            task_type="data_repair",
            file_path=str(data_path),
            columns="A",
            covariates="-",
            model_path=str(model_dir),
            prediction_length=2,
            confidence_interval=0.9,
            anomaly_threshold=3.0,
            context_length=2,
            data_start=0,
            data_end=3,
            data_type="baseline",
        )
    )

    repaired = result["multi_results"][0]["repaired_data"]
    assert result["status"] == "success"
    assert result["multi_results"][0]["missing_indices"] == [1]
    assert repaired == [0.0, 9.0, 4.0]


def test_data_repair_requires_reconstruction_interface(tmp_path):
    data_path = tmp_path / "sample.csv"
    model_dir = tmp_path / "model"
    _write_baseline_csv(data_path)
    model_dir.mkdir()

    engine = MultitaskInferenceEngine(
        project_root=tmp_path,
        pipeline_factory=lambda model_path, device: ForecastOnlyPipeline(),
        chunk_size=8,
    )

    try:
        engine.run(
            InferenceRequest(
                task_type="data_repair",
                file_path=str(data_path),
                columns="A",
                covariates="-",
                model_path=str(model_dir),
                prediction_length=2,
                confidence_interval=0.9,
                anomaly_threshold=3.0,
                context_length=2,
                data_start=0,
                data_end=3,
                data_type="baseline",
            )
        )
    except AttributeError as exc:
        assert "reconstruct_context" in str(exc)
    else:
        raise AssertionError("data_repair should require reconstruct_context()")


def test_anomaly_detection_fills_nan_reconstruction_in_first_chunk(tmp_path):
    data_path = tmp_path / "sample.csv"
    model_dir = tmp_path / "model"
    data_path.write_text(
        "\n".join(
            [
                "FLIGHT,A",
                "1,1",
                "2,2",
                "3,3",
                "4,4",
                "5,5",
            ]
        ),
        encoding="utf-8",
    )
    model_dir.mkdir()

    engine = MultitaskInferenceEngine(
        project_root=tmp_path,
        pipeline_factory=lambda model_path, device: NaNFirstChunkPipeline(),
        chunk_size=3,
    )
    result = engine.run(
        InferenceRequest(
            task_type="anomaly_detection",
            file_path=str(data_path),
            columns="A",
            covariates="-",
            model_path=str(model_dir),
            prediction_length=2,
            confidence_interval=0.9,
            anomaly_threshold=3.0,
            context_length=2,
            data_start=0,
            data_end=5,
            data_type="baseline",
        )
    )

    item = result["multi_results"][0]
    assert item["reconstruction_gap_count"] == 3
    assert all(value is not None for value in item["reconstructed_data"])
    assert all(value is not None for value in item["anomaly_scores"])
