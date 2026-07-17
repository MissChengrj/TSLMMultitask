"""
综合测试脚本：展示微调后模型的完整功能

包括：
1. 异常检测
2. 插值
3. 协变量预测
4. 多变量预测
5. 单变量预测

使用方法:
    python scripts/testing/comprehensive_test.py --model_path weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned --data_path data/038227/038227_1.xlsx
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from chronos import Chronos2Pipeline

logger = logging.getLogger(__name__)


def load_data(file_path: str, target_column: str = None, covariate_columns: list = None):
    """加载测试数据"""
    df = pd.read_excel(file_path, header=6, skiprows=1)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["ENG POS"]
    
    if target_column is None:
        available_cols = [c for c in numeric_cols if c not in exclude_cols]
        target_column = available_cols[0]
    
    if covariate_columns is None:
        covariate_columns = [c for c in numeric_cols if c not in exclude_cols and c != target_column][:3]
    
    target = df[target_column].values.astype(np.float32)
    target = np.where(np.isnan(target), 0.0, target)
    
    covariates = {}
    for col in covariate_columns:
        cov_data = df[col].values.astype(np.float32)
        cov_data = np.where(np.isnan(cov_data), 0.0, cov_data)
        covariates[col] = cov_data
    
    return target, covariates, target_column, covariate_columns


def test_anomaly_detection(pipeline, data, context_length=256, confidence_level=0.90):
    """异常检测测试"""
    logger.info("=" * 60)
    logger.info("任务 1: 异常检测")
    logger.info("=" * 60)
    
    # 截取数据
    test_data = data[:context_length].copy()
    
    # 添加人工异常
    anomaly_indices = [50, 100, 150, 200]
    anomaly_offsets = [10, -15, 20, -12]
    test_data_with_anomalies = test_data.copy()
    for idx, offset in zip(anomaly_indices, anomaly_offsets):
        test_data_with_anomalies[idx] += offset * np.std(test_data)
    
    # 推理
    predictions = pipeline.predict(
        [{"target": test_data_with_anomalies}],
        prediction_length=context_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    
    # 计算置信区间
    n_quantiles = pred.shape[0]
    lower_idx = int(((1 - confidence_level) / 2 - 0.05) / 0.05)
    upper_idx = int(((1 + confidence_level) / 2 - 0.05) / 0.05)
    lower_idx = max(0, min(lower_idx, n_quantiles - 1))
    upper_idx = max(0, min(upper_idx, n_quantiles - 1))
    
    lower_bound = pred[lower_idx]
    upper_bound = pred[upper_idx]
    median_pred = pred[n_quantiles // 2]
    
    # 检测异常
    detected = (test_data_with_anomalies < lower_bound) | (test_data_with_anomalies > upper_bound)
    
    # 计算性能
    true_anomaly_mask = np.zeros(context_length, dtype=bool)
    true_anomaly_mask[anomaly_indices] = True
    
    tp = np.sum(detected & true_anomaly_mask)
    fp = np.sum(detected & ~true_anomaly_mask)
    fn = np.sum(~detected & true_anomaly_mask)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    logger.info(f"检测到的异常点: {np.sum(detected)} 个")
    logger.info(f"真实异常点: {len(anomaly_indices)} 个")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {
        "test_data": test_data_with_anomalies,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "median_pred": median_pred,
        "detected": detected,
        "anomaly_indices": anomaly_indices,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def test_interpolation(pipeline, data, context_length=256, missing_ratio=0.1):
    """插值测试"""
    logger.info("=" * 60)
    logger.info("任务 2: 插值")
    logger.info("=" * 60)
    
    test_data = data[:context_length].copy()
    
    # 创建缺失数据
    num_missing = int(context_length * missing_ratio)
    missing_indices = np.random.choice(context_length, num_missing, replace=False)
    
    test_data_with_missing = test_data.copy()
    test_data_with_missing[missing_indices] = np.nan
    
    # 推理
    predictions = pipeline.predict(
        [{"target": test_data_with_missing}],
        prediction_length=context_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    
    median_pred = pred[pred.shape[0] // 2]
    
    # 计算误差
    actual_values = test_data[missing_indices]
    predicted_values = median_pred[missing_indices]
    mae = np.mean(np.abs(actual_values - predicted_values))
    
    logger.info(f"缺失点数: {num_missing}")
    logger.info(f"插值 MAE: {mae:.4f}")
    
    return {
        "original_data": test_data,
        "data_with_missing": test_data_with_missing,
        "predicted": median_pred,
        "missing_indices": missing_indices,
        "mae": mae,
    }


def test_forecast(pipeline, data, context_length=256, forecast_length=64):
    """单变量预测测试"""
    logger.info("=" * 60)
    logger.info("任务 3: 单变量预测")
    logger.info("=" * 60)
    
    total_length = context_length + forecast_length
    if len(data) < total_length:
        logger.warning("数据长度不足")
        return None
    
    history = data[:context_length]
    future = data[context_length:context_length + forecast_length]
    
    # 推理
    predictions = pipeline.predict(
        [{"target": history}],
        prediction_length=forecast_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    
    median_pred = pred[pred.shape[0] // 2]
    
    # 计算误差
    mae = np.mean(np.abs(median_pred - future))
    rmse = np.sqrt(np.mean((median_pred - future) ** 2))
    
    logger.info(f"预测长度: {forecast_length}")
    logger.info(f"预测 MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    return {
        "history": history,
        "future": future,
        "predicted": pred,
        "median_pred": median_pred,
        "mae": mae,
        "rmse": rmse,
    }


def test_covariate_forecast(pipeline, target, covariates, context_length=256, forecast_length=64):
    """协变量预测测试"""
    logger.info("=" * 60)
    logger.info("任务 4: 协变量预测")
    logger.info("=" * 60)
    
    total_length = context_length + forecast_length
    if len(target) < total_length:
        logger.warning("数据长度不足")
        return None
    
    history = target[:context_length]
    future = target[context_length:context_length + forecast_length]
    
    # 构建协变量输入
    past_cov = {k: v[:context_length] for k, v in covariates.items()}
    future_cov = {k: v[context_length:context_length + forecast_length] for k, v in covariates.items()}
    
    # 有协变量预测
    inputs_with_cov = [{
        "target": history,
        "past_covariates": past_cov,
        "future_covariates": future_cov,
    }]
    
    predictions_with_cov = pipeline.predict(inputs_with_cov, prediction_length=forecast_length)
    pred_with_cov = predictions_with_cov[0].cpu().numpy()
    if pred_with_cov.ndim == 3:
        pred_with_cov = pred_with_cov.squeeze(0)
    median_with_cov = pred_with_cov[pred_with_cov.shape[0] // 2]
    
    # 无协变量预测（对比）
    predictions_no_cov = pipeline.predict([{"target": history}], prediction_length=forecast_length)
    pred_no_cov = predictions_no_cov[0].cpu().numpy()
    if pred_no_cov.ndim == 3:
        pred_no_cov = pred_no_cov.squeeze(0)
    median_no_cov = pred_no_cov[pred_no_cov.shape[0] // 2]
    
    # 计算误差
    mae_with_cov = np.mean(np.abs(median_with_cov - future))
    mae_no_cov = np.mean(np.abs(median_no_cov - future))
    
    logger.info(f"有协变量 MAE: {mae_with_cov:.4f}")
    logger.info(f"无协变量 MAE: {mae_no_cov:.4f}")
    logger.info(f"协变量带来的改进: {(mae_no_cov - mae_with_cov) / mae_no_cov * 100:.2f}%")
    
    return {
        "history": history,
        "future": future,
        "pred_with_cov": pred_with_cov,
        "pred_no_cov": pred_no_cov,
        "mae_with_cov": mae_with_cov,
        "mae_no_cov": mae_no_cov,
    }


def test_multivariate_forecast(pipeline, multivariate_data, context_length=256, forecast_length=64):
    """多变量预测测试"""
    logger.info("=" * 60)
    logger.info("任务 5: 多变量预测")
    logger.info("=" * 60)
    
    n_variates = multivariate_data.shape[0]
    total_length = context_length + forecast_length
    
    if multivariate_data.shape[1] < total_length:
        logger.warning("数据长度不足")
        return None
    
    history = multivariate_data[:, :context_length]
    future = multivariate_data[:, context_length:context_length + forecast_length]
    
    # 多变量预测
    predictions_multi = pipeline.predict([history], prediction_length=forecast_length)
    pred_multi = predictions_multi[0].cpu().numpy()
    
    # 处理形状: 可能是 (n_quantiles, forecast_length) 或 (n_variates, n_quantiles, forecast_length)
    if pred_multi.ndim == 3 and pred_multi.shape[0] == 1:
        pred_multi = pred_multi.squeeze(0)  # (n_quantiles, forecast_length)
    
    # 单变量预测（对比）
    univariate_preds = []
    for i in range(n_variates):
        pred_uni = pipeline.predict([history[i]], prediction_length=forecast_length)[0].cpu().numpy()
        if pred_uni.ndim == 3 and pred_uni.shape[0] == 1:
            pred_uni = pred_uni.squeeze(0)
        univariate_preds.append(pred_uni)
    
    univariate_preds = np.stack(univariate_preds, axis=0)  # (n_variates, n_quantiles, forecast_length)
    
    # 计算误差 - 注意MLM模式下预测的是context的重构，不是未来预测
    # 对于预测任务，需要使用不同的方法
    median_idx = pred_multi.shape[0] // 2 if pred_multi.ndim == 2 else pred_multi.shape[1] // 2
    
    if pred_multi.ndim == 2:
        # 单变量输出: (n_quantiles, forecast_length)
        median_multi = pred_multi[median_idx]  # (forecast_length,)
        multi_errors = [np.mean(np.abs(median_multi - future[0]))]
        avg_multi_error = multi_errors[0]
    else:
        # 多变量输出
        median_multi = pred_multi[:, median_idx, :]  # (n_variates, forecast_length)
        multi_errors = [np.mean(np.abs(median_multi[i] - future[i])) for i in range(n_variates)]
        avg_multi_error = np.mean(multi_errors)
    
    if univariate_preds.ndim == 3:
        median_uni = univariate_preds[:, median_idx, :]  # (n_variates, forecast_length)
        uni_errors = [np.mean(np.abs(median_uni[i] - future[i])) for i in range(n_variates)]
    else:
        median_uni = univariate_preds[median_idx]
        uni_errors = [np.mean(np.abs(median_uni - future[0]))]
    
    avg_uni_error = np.mean(uni_errors)
    
    logger.info(f"多变量预测平均误差: {avg_multi_error:.4f}")
    logger.info(f"单变量预测平均误差: {avg_uni_error:.4f}")
    logger.info(f"多变量带来的改进: {(avg_uni_error - avg_multi_error) / avg_uni_error * 100:.2f}%")
    
    return {
        "history": history,
        "future": future,
        "pred_multi": pred_multi,
        "pred_uni": univariate_preds,
        "multi_errors": multi_errors,
        "uni_errors": uni_errors,
    }


def visualize_all_results(results, output_dir, target_column, covariate_columns):
    """可视化所有结果"""
    
    fig = plt.figure(figsize=(20, 24))
    
    # 1. 异常检测
    ax1 = fig.add_subplot(5, 1, 1)
    anomaly = results["anomaly_detection"]
    ax1.plot(anomaly["test_data"], 'k-', linewidth=1.5, label='测试数据（含异常）')
    ax1.fill_between(range(len(anomaly["lower_bound"])), 
                     anomaly["lower_bound"], anomaly["upper_bound"],
                     alpha=0.3, color='blue', label='置信区间 (P5-P95)')
    ax1.plot(anomaly["median_pred"], 'b-', linewidth=2, alpha=0.7, label='预测中位数')
    
    # 标记真实异常点
    for idx in anomaly["anomaly_indices"]:
        ax1.scatter(idx, anomaly["test_data"][idx], c='red', s=100, marker='x', zorder=5)
    
    # 标记检测到的异常点
    detected_indices = np.where(anomaly["detected"])[0]
    ax1.scatter(detected_indices, anomaly["test_data"][detected_indices], 
                c='orange', s=50, marker='o', alpha=0.7, label='检测到的异常')
    
    ax1.set_title(f'异常检测 - Precision: {anomaly["precision"]:.2f}, Recall: {anomaly["recall"]:.2f}, F1: {anomaly["f1"]:.2f}', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 插值
    ax2 = fig.add_subplot(5, 1, 2)
    interp = results["interpolation"]
    ax2.plot(interp["original_data"], 'k-', linewidth=1.5, label='原始数据')
    ax2.plot(interp["predicted"], 'b-', linewidth=2, alpha=0.7, label='插值结果')
    
    # 标记缺失点
    for idx in interp["missing_indices"]:
        ax2.scatter(idx, interp["predicted"][idx], c='green', s=50, marker='o', zorder=5)
    
    ax2.set_title(f'插值 - MAE: {interp["mae"]:.2f} (缺失点: {len(interp["missing_indices"])})', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 单变量预测
    ax3 = fig.add_subplot(5, 1, 3)
    forecast = results["forecast"]
    if forecast:
        context_length = len(forecast["history"])
        ax3.plot(range(context_length), forecast["history"], 'k-', linewidth=1.5, label='历史数据')
        ax3.plot(range(context_length, context_length + len(forecast["future"])), 
                 forecast["future"], 'g--', linewidth=1.5, label='真实未来')
        
        n_quantiles = forecast["predicted"].shape[0]
        ax3.fill_between(range(context_length, context_length + len(forecast["future"])),
                         forecast["predicted"][0], forecast["predicted"][-1],
                         alpha=0.3, color='blue', label='置信区间')
        ax3.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["median_pred"], 'b-', linewidth=2, label='预测中位数')
        
        ax3.axvline(x=context_length - 0.5, color='red', linestyle=':', linewidth=2)
        ax3.set_title(f'单变量预测 - MAE: {forecast["mae"]:.2f}, RMSE: {forecast["rmse"]:.2f}', fontsize=14)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    
    # 4. 协变量预测
    ax4 = fig.add_subplot(5, 1, 4)
    cov_forecast = results["covariate_forecast"]
    if cov_forecast:
        context_length = len(cov_forecast["history"])
        ax4.plot(range(context_length), cov_forecast["history"], 'k-', linewidth=1.5, label='历史数据')
        ax4.plot(range(context_length, context_length + len(cov_forecast["future"])),
                 cov_forecast["future"], 'g--', linewidth=1.5, label='真实未来')
        
        n_quantiles = cov_forecast["pred_with_cov"].shape[0]
        median_idx = n_quantiles // 2
        
        ax4.plot(range(context_length, context_length + len(cov_forecast["future"])),
                 cov_forecast["pred_with_cov"][median_idx], 'b-', linewidth=2, label='有协变量预测')
        ax4.plot(range(context_length, context_length + len(cov_forecast["future"])),
                 cov_forecast["pred_no_cov"][median_idx], 'orange', linewidth=2, linestyle='--', label='无协变量预测')
        
        ax4.axvline(x=context_length - 0.5, color='red', linestyle=':', linewidth=2)
        improvement = (cov_forecast["mae_no_cov"] - cov_forecast["mae_with_cov"]) / cov_forecast["mae_no_cov"] * 100
        ax4.set_title(f'协变量预测 - 有协变量MAE: {cov_forecast["mae_with_cov"]:.2f}, 无协变量MAE: {cov_forecast["mae_no_cov"]:.2f}, 改进: {improvement:.1f}%', fontsize=14)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
    
    # 5. 多变量预测
    ax5 = fig.add_subplot(5, 1, 5)
    multi_forecast = results["multivariate_forecast"]
    if multi_forecast:
        n_variates = len(multi_forecast["multi_errors"])
        x = np.arange(n_variates)
        width = 0.35
        
        ax5.bar(x - width/2, multi_forecast["multi_errors"], width, label='多变量预测', color='steelblue', alpha=0.7)
        ax5.bar(x + width/2, multi_forecast["uni_errors"], width, label='单变量预测', color='coral', alpha=0.7)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'变量{i+1}' for i in range(n_variates)], fontsize=10)
        avg_multi = np.mean(multi_forecast["multi_errors"])
        avg_uni = np.mean(multi_forecast["uni_errors"])
        improvement = (avg_uni - avg_multi) / avg_uni * 100
        ax5.set_title(f'多变量预测误差对比 - 平均多变量误差: {avg_multi:.2f}, 平均单变量误差: {avg_uni:.2f}, 改进: {improvement:.1f}%', fontsize=14)
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_test_results.png', dpi=150, bbox_inches='tight')
    logger.info(f"综合测试结果图表已保存: {output_dir / 'comprehensive_test_results.png'}")
    
    # 创建汇总表格
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    
    table_data = [
        ['异常检测', f'F1: {results["anomaly_detection"]["f1"]:.4f}', 
         f'Precision: {results["anomaly_detection"]["precision"]:.4f}', 
         f'Recall: {results["anomaly_detection"]["recall"]:.4f}'],
        ['插值', f'MAE: {results["interpolation"]["mae"]:.4f}', 
         f'缺失点: {len(results["interpolation"]["missing_indices"])}', '-'],
        ['单变量预测', f'MAE: {results["forecast"]["mae"]:.4f}', 
         f'RMSE: {results["forecast"]["rmse"]:.4f}', '-'],
        ['协变量预测', f'有协变量MAE: {results["covariate_forecast"]["mae_with_cov"]:.4f}', 
         f'无协变量MAE: {results["covariate_forecast"]["mae_no_cov"]:.4f}',
         f'改进: {(results["covariate_forecast"]["mae_no_cov"] - results["covariate_forecast"]["mae_with_cov"]) / results["covariate_forecast"]["mae_no_cov"] * 100:.1f}%'],
        ['多变量预测', f'平均误差: {np.mean(results["multivariate_forecast"]["multi_errors"]):.4f}', 
         f'单变量误差: {np.mean(results["multivariate_forecast"]["uni_errors"]):.4f}',
         f'改进: {(np.mean(results["multivariate_forecast"]["uni_errors"]) - np.mean(results["multivariate_forecast"]["multi_errors"])) / np.mean(results["multivariate_forecast"]["uni_errors"]) * 100:.1f}%'],
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['任务', '主要指标', '次要指标', '改进'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title(f'综合测试汇总报告 - 目标变量: {target_column}', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'comprehensive_summary_table.png', dpi=150, bbox_inches='tight')
    logger.info(f"汇总表格已保存: {output_dir / 'comprehensive_summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="综合测试脚本")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned",
        help="模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/038227/038227_1.xlsx",
        help="数据文件路径",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="DEGT-CRUISE",
        help="目标列名",
    )
    parser.add_argument(
        "--covariate_columns",
        type=str,
        nargs="+",
        default=["DEGT_SMOOTHED-CRUISE", "DEGT_D-CRUISE", "DEGT_D_SMOOTHED-CRUISE"],
        help="协变量列名",
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
        "--output_dir",
        type=str,
        default="results/comprehensive_test",
        help="输出目录",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda")
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")
    
    # 加载数据
    logger.info(f"加载测试数据: {args.data_path}")
    target, covariates, target_column, covariate_columns = load_data(
        args.data_path, args.target_column, args.covariate_columns
    )
    logger.info(f"目标列: {target_column}, 协变量列: {covariate_columns}")
    
    # 准备多变量数据
    multivariate_data = np.stack([target] + [covariates[c] for c in covariate_columns], axis=0)
    
    results = {}
    
    # 1. 异常检测
    results["anomaly_detection"] = test_anomaly_detection(
        pipeline, target, args.context_length
    )
    
    # 2. 插值
    results["interpolation"] = test_interpolation(
        pipeline, target, args.context_length, missing_ratio=0.1
    )
    
    # 3. 单变量预测
    results["forecast"] = test_forecast(
        pipeline, target, args.context_length, args.forecast_length
    )
    
    # 4. 协变量预测
    results["covariate_forecast"] = test_covariate_forecast(
        pipeline, target, covariates, args.context_length, args.forecast_length
    )
    
    # 5. 多变量预测
    results["multivariate_forecast"] = test_multivariate_forecast(
        pipeline, multivariate_data, args.context_length, args.forecast_length
    )
    
    # 可视化
    visualize_all_results(results, output_dir, target_column, covariate_columns)
    
    logger.info("=" * 60)
    logger.info("综合测试完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()