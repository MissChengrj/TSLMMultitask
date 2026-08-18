# Remote 4060 Training Workflow

This project keeps code in GitHub and keeps large local artifacts out of Git:

- `data/`
- `weights/`
- `results/`
- `app/node_modules/`
- `app/src-tauri/target/`

## Local Machine

Commit and push only source code, scripts, configs, tests, and documentation.

```powershell
git status --short
git add .gitignore docs scripts src configs test README.md pyproject.toml
git commit -m "add jsonl multitask training and evaluation workflow"
git push
```

## 4060 Desktop

Pull the latest code:

```powershell
git pull
```

Prepare the environment:

```powershell
conda create -n tslm-multitask python=3.10 -y
conda activate tslm-multitask
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install transformers peft accelerate safetensors pandas numpy einops tqdm
```

Keep the pretrained Chronos2 weights and datasets on the desktop outside Git:

```text
weights/chronos-2/
data/multitask_eval_all_sources/
```

Split the datasets:

```powershell
python scripts/datasets/split_multitask_eval_all_sources.py --input-dir data/multitask_eval_all_sources --output-dir data/multitask_eval_splits
```

Run longer fine-tuning on the RTX 4060:

```powershell
python scripts/training/train_chronos2_jsonl_multitask.py --source acars --split-dir data/multitask_eval_splits --model-path weights/chronos-2 --output-dir weights/chronos2_jsonl_acars_multitask_1000 --max-steps 1000 --batch-size 2 --learning-rate 0.0000005 --lora-r 4 --lora-alpha 8 --mask-ratio 0.15 --forecast-loss-weight 1.0 --recon-loss-weight 0.3
```

```powershell
python scripts/training/train_chronos2_jsonl_multitask.py --source qar --split-dir data/multitask_eval_splits --model-path weights/chronos-2 --output-dir weights/chronos2_jsonl_qar_multitask_1000 --max-steps 1000 --batch-size 2 --learning-rate 0.0000005 --lora-r 4 --lora-alpha 8 --mask-ratio 0.15 --forecast-loss-weight 1.0 --recon-loss-weight 0.3
```

Evaluate:

```powershell
python scripts/evaluation/evaluate_jsonl_multitask_experiment.py --source acars --split-dir data/multitask_eval_splits --base-model weights/chronos-2 --finetuned-model weights/chronos2_jsonl_acars_multitask_1000/checkpoint-final --output-dir results/multitask_jsonl_experiment_1000 --max-records-per-task 300 --device cuda --include-neural-baselines
```

```powershell
python scripts/evaluation/evaluate_jsonl_multitask_experiment.py --source qar --split-dir data/multitask_eval_splits --base-model weights/chronos-2 --finetuned-model weights/chronos2_jsonl_qar_multitask_1000/checkpoint-final --output-dir results/multitask_jsonl_experiment_1000 --max-records-per-task 300 --device cuda --include-neural-baselines
```
