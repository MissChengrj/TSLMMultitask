"""
批量测试脚本

对微调后的模型进行插值、异常检测、预测等任务的批量测试。

使用方法:
    python scripts/testing/batch_test.py --model_path weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned --data_dir data
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入批量数据加载器
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from batch_data_loader import load_all_data, find_all_data_files

from chronos import Chronos2Pipeline

logger = logging.getLogger(__name__)


def test_interpolation(
    pipeline: Chronos2Pipeline,
    target: np.ndarray,
    missing_ratio: float = 0.1,
    context_length: int = 256,
) -> dict:
    """
    测试插值任务
    
    Parameters
    ----------
    pipeline : Chronos2Pipeline
        模型管道
    target : np.ndarray
        目标数据
    missing_ratio : float
        缺失比例
    context_length : int
        上下文长度
        
    Returns
    -------
    dict
        测试结果
    """
    # 截取数据
    if len(target) < context_length:
        return {"error": "数据长度不足"}
    
    data = target[:context_length].copy()
    
    # 随机生成缺失位置
    num_missing = int(context_length * missing_ratio)
    missing_indices = np.random.choice(context_length, num_missing, replace=False)
    
    # 创建缺失数据
    data_with_missing = data.copy()
    data_with_missing[missing_indices] = np.nan
    
    # 推理 - MLM 模式输出是对 Context 的重构
    predictions = pipeline.predict(
        [{"target": data_with_missing}],
        prediction_length=context_length,  # MLM 模式下输出长度等于 context_length
    )
    
    # 处理预测结果
    # MLM 模式输出形状: (batch, n_quantiles, context_length) = (1, 21, 256)
    if hasattr(predictions[0], 'cpu'):
        pred = predictions[0].cpu().numpy()
    else:
        pred = predictions[0]
    
    # 去掉批次维度
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)  # (n_quantiles, context_length)
    
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    
    # MLM 模式输出: (n_quantiles, context_length)
    # pred.shape[0] = n_quantiles (21), pred.shape[1] = context_length (256)
    median_idx = pred.shape[0] // 2
    median_pred = pred[median_idx]  # (context_length,)
    
    # 计算插值误差
    actual_values = data[missing_indices]
    predicted_values = median_pred[missing_indices]
    mae = np.mean(np.abs(actual_values - predicted_values))
    
    return {
        "mae": float(mae),
        "num_missing": num_missing,
        "missing_ratio": missing_ratio,
    }


def test_anomaly_detection(
    pipeline: Chronos2Pipeline,
    target: np.ndarray,
    anomaly_ratio: float = 0.05,
    context_length: int = 256,
    confidence_level: float = 0.90,
) -> dict:
    """
    测试异常检测任务
    
    Parameters
    ----------
    pipeline : Chronos2Pipeline
        模型管道
    target : np.ndarray
        目标数据
    anomaly_ratio : float
        异常比例
    context_length : int
        上下文长度
    confidence_level : float
        置信区间水平
        
    Returns
    -------
    dict
        测试结果
    """
    # 截取数据
    if len(target) < context_length:
        return {"error": "数据长度不足"}
    
    data = target[:context_length].copy()
    
    # 随机生成异常位置
    num_anomalies = int(context_length * anomaly_ratio)
    anomaly_indices = np.random.choice(context_length, num_anomalies, replace=False)
    
    # 创建异常数据（添加大幅偏移）
    data_with_anomalies = data.copy()
    anomaly_offsets = np.random.choice([-1, 1], num_anomalies) * np.random.uniform(5, 10, num_anomalies) * np.std(data)
    data_with_anomalies[anomaly_indices] += anomaly_offsets
    
    # 推理 - MLM 模式输出是对 Context 的重构
    predictions = pipeline.predict(
        [{"target": data_with_anomalies}],
        prediction_length=context_length,
    )
    
    # 处理预测结果
    if hasattr(predictions[0], 'cpu'):
        pred = predictions[0].cpu().numpy()
    else:
        pred = predictions[0]
    
    # 去掉批次维度
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)  # (n_quantiles, context_length)
    
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    
    # 计算置信区间索引
    n_quantiles = pred.shape[0]
    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = (1 + confidence_level) / 2
    lower_idx = int((lower_quantile - 0.05) / 0.05)
    upper_idx = int((upper_quantile - 0.05) / 0.05)
    
    # 确保索引在有效范围内
    lower_idx = max(0, min(lower_idx, n_quantiles - 1))
    upper_idx = max(0, min(upper_idx, n_quantiles - 1))
    
    lower_bound = pred[lower_idx]
    upper_bound = pred[upper_idx]
    
    # 检测异常
    detected_anomalies = (data_with_anomalies < lower_bound) | (data_with_anomalies > upper_bound)
    
    # 计算检测性能
    true_anomaly_mask = np.zeros(context_length, dtype=bool)
    true_anomaly_mask[anomaly_indices] = True
    
    true_positives = np.sum(detected_anomalies & true_anomaly_mask)
    false_positives = np.sum(detected_anomalies & ~true_anomaly_mask)
    false_negatives = np.sum(~detected_anomalies & true_anomaly_mask)
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "num_anomalies": num_anomalies,
    }


def test_forecast(
    pipeline: Chronos2Pipeline,
    target: np.ndarray,
    context_length: int = 256,
    forecast_length: int = 64,
    use_original_model: bool = False,
) -> dict:
    """
    测试预测任务
    
    注意: MLM 微调后的模型主要用于插值和异常检测，预测任务建议使用原始模型
    
    Parameters
    ----------
    pipeline : Chronos2Pipeline
        模型管道
    target : np.ndarray
        目标数据
    context_length : int
        上下文长度
    forecast_length : int
        预测长度
    use_original_model : bool
        是否使用原始模型（而非 MLM 微调模型）
        
    Returns
    -------
    dict
        测试结果
    """
    total_length = context_length + forecast_length
    
    if len(target) < total_length:
        return {"error": "数据长度不足"}
    
    # 分割数据
    history = target[:context_length]
    future = target[context_length:context_length + forecast_length]
    
    # 推理
    predictions = pipeline.predict(
        [{"target": history}],
        prediction_length=forecast_length,
    )
    
    # 处理预测结果
    if hasattr(predictions[0], 'cpu'):
        pred = predictions[0].cpu().numpy()
    else:
        pred = predictions[0]
    
    # 去掉批次维度
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)  # (n_quantiles, forecast_length)
    
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    
    # 检查预测长度
    if pred.shape[1] < forecast_length:
        logger.warning(f"预测长度 {pred.shape[1]} < 期望长度 {forecast_length}")
        return {"error": "预测长度不足"}
    
    median_idx = pred.shape[0] // 2
    median_pred = pred[median_idx][:forecast_length]
    
    # 计算预测误差
    mae = np.mean(np.abs(median_pred - future))
    mse = np.mean((median_pred - future) ** 2)
    rmse = np.sqrt(mse)
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "forecast_length": forecast_length,
    }


def run_batch_tests(
    model_path: str,
    data_dir: str,
    output_dir: str,
    context_length: int = 256,
    forecast_length: int = 64,
    missing_ratio: float = 0.1,
    anomaly_ratio: float = 0.05,
    confidence_level: float = 0.90,
    max_files: int = None,
    max_series: int = None,
):
    """
    执行批量测试
    
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
    forecast_length : int
        预测长度
    missing_ratio : float
        缺失比例
    anomaly_ratio : float
        异常比例
    confidence_level : float
        置信区间水平
    max_files : int
        最大文件数量
    max_series : int
        最大时序数量
    """
    # 1. 加载模型
    logger.info(f"加载模型: {model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map="cuda")
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")
    
    # 2. 加载所有数据
    logger.info(f"加载测试数据: {data_dir}")
    all_data = load_all_data(data_dir, max_files)
    
    if max_series:
        all_data = all_data[:max_series]
    
    logger.info(f"测试数据量: {len(all_data)} 个时序")
    
    # 3. 执行测试
    results = {
        "interpolation": [],
        "anomaly_detection": [],
        "forecast": [],
        "metadata": {
            "model_path": model_path,
            "data_dir": data_dir,
            "context_length": context_length,
            "forecast_length": forecast_length,
            "missing_ratio": missing_ratio,
            "anomaly_ratio": anomaly_ratio,
            "confidence_level": confidence_level,
            "num_series": len(all_data),
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    logger.info("=" * 60)
    logger.info("开始批量测试...")
    logger.info("=" * 60)
    
    for item in tqdm(all_data, desc="测试进度"):
        target = item["target"]
        file_name = item["file"]
        column_name = item["column"]
        
        # 插值测试
        interp_result = test_interpolation(
            pipeline, target, missing_ratio, context_length
        )
        interp_result["file"] = file_name
        interp_result["column"] = column_name
        results["interpolation"].append(interp_result)
        
        # 异常检测测试
        anomaly_result = test_anomaly_detection(
            pipeline, target, anomaly_ratio, context_length, confidence_level
        )
        anomaly_result["file"] = file_name
        anomaly_result["column"] = column_name
        results["anomaly_detection"].append(anomaly_result)
        
        # 预测测试
        forecast_result = test_forecast(
            pipeline, target, context_length, forecast_length
        )
        forecast_result["file"] = file_name
        forecast_result["column"] = column_name
        results["forecast"].append(forecast_result)
    
    # 4. 计算汇总统计
    logger.info("=" * 60)
    logger.info("计算汇总统计...")
    logger.info("=" * 60)
    
    # 插值统计
    interp_maes = [r["mae"] for r in results["interpolation"] if "mae" in r]
    avg_interp_mae = np.mean(interp_maes) if interp_maes else 0
    
    # 异常检测统计
    anomaly_f1s = [r["f1"] for r in results["anomaly_detection"] if "f1" in r]
    avg_anomaly_f1 = np.mean(anomaly_f1s) if anomaly_f1s else 0
    avg_precision = np.mean([r["precision"] for r in results["anomaly_detection"] if "precision" in r])
    avg_recall = np.mean([r["recall"] for r in results["anomaly_detection"] if "recall" in r])
    
    # 预测统计
    forecast_maes = [r["mae"] for r in results["forecast"] if "mae" in r]
    forecast_rmses = [r["rmse"] for r in results["forecast"] if "rmse" in r]
    avg_forecast_mae = np.mean(forecast_maes) if forecast_maes else 0
    avg_forecast_rmse = np.mean(forecast_rmses) if forecast_rmses else 0
    
    summary = {
        "interpolation": {
            "avg_mae": avg_interp_mae,
            "std_mae": np.std(interp_maes) if interp_maes else 0,
            "num_tests": len(interp_maes),
        },
        "anomaly_detection": {
            "avg_f1": avg_anomaly_f1,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "num_tests": len(anomaly_f1s),
        },
        "forecast": {
            "avg_mae": avg_forecast_mae,
            "avg_rmse": avg_forecast_rmse,
            "std_mae": np.std(forecast_maes) if forecast_maes else 0,
            "num_tests": len(forecast_maes),
        },
    }
    
    results["summary"] = summary
    
    # 5. 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存 JSON 结果
    with open(output_path / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"测试结果已保存到: {output_path / 'test_results.json'}")
    
    # 6. 生成可视化报告
    generate_report(results, output_path)
    
    # 7. 打印汇总
    logger.info("=" * 60)
    logger.info("测试汇总:")
    logger.info(f"  插值任务: 平均 MAE = {avg_interp_mae:.4f}")
    logger.info(f"  异常检测: 平均 F1 = {avg_anomaly_f1:.4f}, Precision = {avg_precision:.4f}, Recall = {avg_recall:.4f}")
    logger.info(f"  预测任务: 平均 MAE = {avg_forecast_mae:.4f}, RMSE = {avg_forecast_rmse:.4f}")
    logger.info("=" * 60)
    
    return results


def generate_report(results: dict, output_path: Path):
    """生成可视化报告"""
    summary = results["summary"]
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 插值误差分布
    interp_maes = [r["mae"] for r in results["interpolation"] if "mae" in r]
    axes[0].hist(interp_maes, bins=30, color="blue", alpha=0.7, edgecolor="black")
    axes[0].axvline(summary["interpolation"]["avg_mae"], color="red", linestyle="--", linewidth=2, label=f"平均: {summary['interpolation']['avg_mae']:.2f}")
    axes[0].set_title("插值误差分布 (MAE)", fontsize=14)
    axes[0].set_xlabel("MAE", fontsize=12)
    axes[0].set_ylabel("频数", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. 异常检测性能分布
    anomaly_f1s = [r["f1"] for r in results["anomaly_detection"] if "f1" in r]
    axes[1].hist(anomaly_f1s, bins=30, color="green", alpha=0.7, edgecolor="black")
    axes[1].axvline(summary["anomaly_detection"]["avg_f1"], color="red", linestyle="--", linewidth=2, label=f"平均: {summary['anomaly_detection']['avg_f1']:.2f}")
    axes[1].set_title("异常检测性能分布 (F1)", fontsize=14)
    axes[1].set_xlabel("F1 Score", fontsize=12)
    axes[1].set_ylabel("频数", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 预测误差分布
    forecast_maes = [r["mae"] for r in results["forecast"] if "mae" in r]
    axes[2].hist(forecast_maes, bins=30, color="orange", alpha=0.7, edgecolor="black")
    axes[2].axvline(summary["forecast"]["avg_mae"], color="red", linestyle="--", linewidth=2, label=f"平均: {summary['forecast']['avg_mae']:.2f}")
    axes[2].set_title("预测误差分布 (MAE)", fontsize=14)
    axes[2].set_xlabel("MAE", fontsize=12)
    axes[2].set_ylabel("频数", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "test_report.png", dpi=150, bbox_inches="tight")
    logger.info(f"测试报告已保存到: {output_path / 'test_report.png'}")
    
    # 创建汇总表格图表
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    
    table_data = [
        ["插值任务", f"{summary['interpolation']['avg_mae']:.4f}", f"{summary['interpolation']['std_mae']:.4f}", f"{summary['interpolation']['num_tests']}"],
        ["异常检测", f"{summary['anomaly_detection']['avg_f1']:.4f}", f"{summary['anomaly_detection']['avg_precision']:.4f}", f"{summary['anomaly_detection']['num_tests']}"],
        ["预测任务", f"{summary['forecast']['avg_mae']:.4f}", f"{summary['forecast']['avg_rmse']:.4f}", f"{summary['forecast']['num_tests']}"],
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=["任务", "主要指标", "次要指标", "测试数量"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    
    plt.title("批量测试汇总报告", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(output_path / "summary_table.png", dpi=150, bbox_inches="tight")
    logger.info(f"汇总表格已保存到: {output_path / 'summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="批量测试脚本")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned",
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
        default="results/batch_test",
        help="输出目录",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="上下文长度",
    )
    parser.add_argument(
        "--forecast_length",
        type=int,
        default=64,
        help="预测长度",
    )
    parser.add_argument(
        "--missing_ratio",
        type=float,
        default=0.1,
        help="插值缺失比例",
    )
    parser.add_argument(
        "--anomaly_ratio",
        type=float,
        default=0.05,
        help="异常比例",
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.90,
        help="异常检测置信区间水平",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="最大文件数量",
    )
    parser.add_argument(
        "--max_series",
        type=int,
        default=None,
        help="最大时序数量",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    # 执行测试
    run_batch_tests(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        missing_ratio=args.missing_ratio,
        anomaly_ratio=args.anomaly_ratio,
        confidence_level=args.confidence_level,
        max_files=args.max_files,
        max_series=args.max_series,
    )


if __name__ == "__main__":
    main()