# -*- coding: utf-8 -*-
"""Evaluate a Chronos2 multitask full checkpoint or LoRA adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(os.environ.get("TSLM_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_multitask_chronos2 import TrainConfig, evaluate_model, load_and_split_data
from tslm_multitask.models import Chronos2MultiTaskModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Chronos2 multitask checkpoint")
    parser.add_argument("--base-model", default=str(PROJECT_ROOT / "weights" / "chronos-2"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-file")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--prediction-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def load_checkpoint(base_model: str, checkpoint: str, device: str):
    adapter_config = Path(checkpoint) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        model = Chronos2MultiTaskModel.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        model = Chronos2MultiTaskModel.from_pretrained(checkpoint)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    model.to(torch.device(device))
    model.eval()
    return model


def main():
    args = parse_args()

    config = TrainConfig()
    config.model_path = str(Path(args.base_model).resolve())
    config.data_dir = str(Path(args.data_dir).resolve())
    config.context_length = args.context_length
    config.prediction_length = args.prediction_length
    config.batch_size = args.batch_size
    config.mask_ratio = args.mask_ratio

    _, val_series, data_stats = load_and_split_data(config)
    model = load_checkpoint(config.model_path, args.checkpoint, args.device)
    metrics = evaluate_model(model, val_series, config)

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_model": config.model_path,
        "device": args.device,
        "data_stats": data_stats,
        "metrics": metrics,
    }

    if args.output_file:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
