"""
B-1400 微调模型测试脚本

使用微调后的模型对 B-1400 数据进行插值、异常检测和预测测试。

使用方法:
    python scripts/testing/test_b1400_model.py --model_path weights/mlm_b1400_finetuned/2026-06-23_10-45-34/mlm-lora-b1400-finetuned --data_path data/B-1400/B-1400_20260101005255.qar.csv --variable EGT1
"""

import argparse
import logging
import sys
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from chronos import Chronos2Pipeline
from scripts.data.load_b1400_data import parse_value

logger = logging.getLogger(__name__)


def load_single_variable(file_path: str, variable: str):
    """加载单个变量的数据"""
    import pandas as pd
    
    df = pd.read_csv(file_path)
    
    if variable not in df.columns:
        logger.error(f"变量 {variable} 不存在于文件中")
        return None
    
    values = df[variable].apply(parse_value).values.astype(np.float32)
    
    # 处理 NaN 值
    nan_mask = np.isnan(values)
    if np.all(nan_mask):
        logger.error(f"变量 {variable} 全部为 NaN")
        return None
    
    # 使用线性插值填充 NaN
    if np.any(nan_mask):
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 1:
            nan_indices = np.where(nan_mask)[0]
            for idx in nan_indices:
                left_idx = valid_indices[valid_indices < idx]
                right_idx = valid_indices[valid_indices > idx]
                
                if len(left_idx) > 0 and len(right_idx) > 0:
                    left_val = values[left_idx[-1]]
                    right_val = values[right_idx[0]]
                    ratio = (idx - left_idx[-1]) / (right_idx[0] - left_idx[-1])
                    values[idx] = left_val + ratio * (right_val - left_val)
                elif len(left_idx) > 0:
                    values[idx] = values[left_idx[-1]]
                elif len(right_idx) > 0:
                    values[idx] = values[right_idx[0]]
    
    return values


def run_interpolation(pipeline, data, context_length: int, num_missing: int = 25):
    """运行插值任务
    
    根据代码审查建议：
    1. 推理时使用 batch_size=1，避免左侧填充误判
    2. 直接传入包含 NaN 的数据，模型会自动识别需要插值的位置
    3. 模型在 eval 模式下会强制唤醒包含 NaN 的 Patch
    """
    total_length = context_length
    if len(data) < total_length:
        logger.warning(f"数据长度不足: {len(data)} < {total_length}")
        return None
    
    # 随机选择一段数据
    start_idx = np.random.randint(0, len(data) - total_length)
    segment = data[start_idx:start_idx + total_length].copy()
    
    # 创建缺失值
    missing_indices = np.random.choice(
        context_length - 20,  # 避开开头和结尾
        size=num_missing,
        replace=False,
    ) + 10
    
    original_segment = segment.copy()
    segment[missing_indices] = np.nan
    
    # 使用模型进行插值
    # 【关键修改】：直接传入包含 NaN 的数据，模型会自动识别需要插值的位置
    # 不再手动替换 NaN 为 0，让模型的推理模式自动处理
    pipeline.model.eval()  # 确保模型在推理模式
    
    # MLM 模式: 使用 prediction_length=context_length 来获取完整的重构
    # batch_size=1 避免左侧填充导致的误判
    predictions = pipeline.predict(
        [{"target": segment}],  # 直接传入包含 NaN 的数据
        prediction_length=context_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    
    # MLM 输出: (n_quantiles, context_length + prediction_length)
    # 取最后 context_length 个点作为重构结果
    median_pred = pred[pred.shape[0] // 2][-context_length:]
    
    # 计算插值误差
    interpolated_values = median_pred[missing_indices]
    original_values = original_segment[missing_indices]
    
    mae = np.mean(np.abs(interpolated_values - original_values))
    
    return {
        "original": original_segment,
        "missing_indices": missing_indices,
        "interpolated": median_pred,
        "mae": mae,
        "num_missing": num_missing,
    }


def run_anomaly_detection(pipeline, data, context_length: int, anomaly_ratio: float = 0.05):
    """运行异常检测任务"""
    total_length = context_length
    if len(data) < total_length:
        logger.warning(f"数据长度不足: {len(data)} < {total_length}")
        return None
    
    # 随机选择一段数据
    start_idx = np.random.randint(0, len(data) - total_length)
    segment = data[start_idx:start_idx + total_length].copy()
    
    # 创建人工异常
    num_anomalies = int(context_length * anomaly_ratio)
    anomaly_indices = np.random.choice(
        context_length - 20,
        size=num_anomalies,
        replace=False,
    ) + 10
    
    original_segment = segment.copy()
    # 添加异常（大幅偏离）
    for idx in anomaly_indices:
        segment[idx] += np.random.choice([-1, 1]) * np.std(original_segment) * 3
    
    # 使用模型进行预测
    # MLM 模式: 使用 prediction_length=context_length 来获取完整的重构
    predictions = pipeline.predict(
        [{"target": segment}],
        prediction_length=context_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    
    # 获取置信区间 (取最后 context_length 个点)
    lower_bound = pred[0][-context_length:]  # P10
    upper_bound = pred[-1][-context_length:]  # P90
    median_pred = pred[pred.shape[0] // 2][-context_length:]
    
    # 检测异常
    detected_anomalies = (segment < lower_bound) | (segment > upper_bound)
    
    # 计算指标
    true_positives = np.sum(detected_anomalies[anomaly_indices])
    false_positives = np.sum(detected_anomalies) - true_positives
    false_negatives = num_anomalies - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / num_anomalies if num_anomalies > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "original": original_segment,
        "anomaly_indices": anomaly_indices,
        "segment_with_anomalies": segment,
        "predicted": median_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "detected_anomalies": detected_anomalies,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_anomalies": num_anomalies,
    }


def run_forecast(pipeline, data, context_length: int, forecast_length: int = 16):
    """运行预测任务"""
    total_length = context_length + forecast_length
    if len(data) < total_length:
        logger.warning(f"数据长度不足: {len(data)} < {total_length}")
        return None
    
    history = data[:context_length]
    future = data[context_length:context_length + forecast_length]
    
    # 推理
    predictions = pipeline.predict(
        [{"target": history}],
        prediction_length=forecast_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    
    # MLM 模式输出: (n_quantiles, context_length + prediction_length)
    # 需要取最后 forecast_length 个点
    median_pred = pred[pred.shape[0] // 2][-forecast_length:]
    
    # 计算误差
    mae = np.mean(np.abs(median_pred - future))
    rmse = np.sqrt(np.mean((median_pred - future) ** 2))
    
    return {
        "history": history,
        "future": future,
        "predicted": pred,
        "median_pred": median_pred,
        "mae": mae,
        "rmse": rmse,
    }


def run_covariate_forecast(pipeline, target_data, covariate_data_list, context_length: int, forecast_length: int = 16):
    """运行协变量预测任务"""
    total_length = context_length + forecast_length
    if len(target_data) < total_length:
        logger.warning(f"目标数据长度不足: {len(target_data)} < {total_length}")
        return None
    
    for cov_data in covariate_data_list:
        if len(cov_data) < total_length:
            logger.warning(f"协变量数据长度不足: {len(cov_data)} < {total_length}")
            return None
    
    # 准备数据
    target_history = target_data[:context_length]
    target_future = target_data[context_length:context_length + forecast_length]
    
    # 协变量数据
    covariate_histories = [cov_data[:context_length] for cov_data in covariate_data_list]
    covariate_futures = [cov_data[context_length:context_length + forecast_length] for cov_data in covariate_data_list]
    
    # 有协变量预测
    # Chronos-2 支持协变量，需要构造正确的输入格式
    # past_covariates 应该是字典格式 {"feat_1": tensor_1, ...}
    covariates_dict = {}
    for i, cov_history in enumerate(covariate_histories):
        covariates_dict[f"feat_{i+1}"] = torch.tensor(cov_history, dtype=torch.float32)
    
    context_with_cov = {
        "target": target_history,
        "past_covariates": covariates_dict,
    }
    
    predictions_with_cov = pipeline.predict(
        [context_with_cov],
        prediction_length=forecast_length,
    )
    
    pred_with_cov = predictions_with_cov[0].cpu().numpy()
    if pred_with_cov.ndim == 3 and pred_with_cov.shape[0] == 1:
        pred_with_cov = pred_with_cov.squeeze(0)
    
    median_pred_with_cov = pred_with_cov[pred_with_cov.shape[0] // 2][-forecast_length:]
    
    # 无协变量预测（对比）
    predictions_no_cov = pipeline.predict(
        [{"target": target_history}],
        prediction_length=forecast_length,
    )
    
    pred_no_cov = predictions_no_cov[0].cpu().numpy()
    if pred_no_cov.ndim == 3 and pred_no_cov.shape[0] == 1:
        pred_no_cov = pred_no_cov.squeeze(0)
    
    median_pred_no_cov = pred_no_cov[pred_no_cov.shape[0] // 2][-forecast_length:]
    
    # 计算误差
    mae_with_cov = np.mean(np.abs(median_pred_with_cov - target_future))
    mae_no_cov = np.mean(np.abs(median_pred_no_cov - target_future))
    
    improvement = (mae_no_cov - mae_with_cov) / mae_no_cov * 100 if mae_no_cov > 0 else 0
    
    return {
        "history": target_history,
        "future": target_future,
        "predicted_with_cov": pred_with_cov,
        "median_pred_with_cov": median_pred_with_cov,
        "predicted_no_cov": pred_no_cov,
        "median_pred_no_cov": median_pred_no_cov,
        "mae_with_cov": mae_with_cov,
        "mae_no_cov": mae_no_cov,
        "improvement": improvement,
        "covariate_histories": covariate_histories,
        "covariate_futures": covariate_futures,
    }


def run_multivariate_forecast(pipeline, variables_data_list, context_length: int, forecast_length: int = 16):
    """运行多变量预测任务"""
    total_length = context_length + forecast_length
    
    for var_data in variables_data_list:
        if len(var_data) < total_length:
            logger.warning(f"变量数据长度不足: {len(var_data)} < {total_length}")
            return None
    
    # 准备数据
    histories = [var_data[:context_length] for var_data in variables_data_list]
    futures = [var_data[context_length:context_length + forecast_length] for var_data in variables_data_list]
    
    # 多变量预测
    # 构造多变量输入
    multivariate_context = {
        "target": np.stack(histories, axis=0),  # (n_variables, context_length)
    }
    
    predictions_multi = pipeline.predict(
        [multivariate_context],
        prediction_length=forecast_length,
    )
    
    pred_multi = predictions_multi[0].cpu().numpy()
    if pred_multi.ndim == 3 and pred_multi.shape[0] == 1:
        pred_multi = pred_multi.squeeze(0)
    
    # 多变量输出: (n_variables, n_quantiles, context_length + prediction_length)
    # 或者 (n_quantiles, context_length + prediction_length)
    
    # 单变量预测（对比）
    single_var_results = []
    for i, history in enumerate(histories):
        predictions_single = pipeline.predict(
            [{"target": history}],
            prediction_length=forecast_length,
        )
        
        pred_single = predictions_single[0].cpu().numpy()
        if pred_single.ndim == 3 and pred_single.shape[0] == 1:
            pred_single = pred_single.squeeze(0)
        
        median_pred_single = pred_single[pred_single.shape[0] // 2][-forecast_length:]
        single_var_results.append({
            "median_pred": median_pred_single,
            "mae": np.mean(np.abs(median_pred_single - futures[i])),
        })
    
    # 计算多变量预测误差
    multi_var_results = []
    n_variables = len(histories)
    
    # 检查输出形状
    if pred_multi.ndim == 3:
        # (n_variables, n_quantiles, seq_len)
        for i in range(n_variables):
            median_pred_multi = pred_multi[i, pred_multi.shape[1] // 2, -forecast_length:]
            multi_var_results.append({
                "median_pred": median_pred_multi,
                "mae": np.mean(np.abs(median_pred_multi - futures[i])),
            })
    else:
        # (n_quantiles, seq_len) - 所有变量共享同一个预测
        median_pred_multi = pred_multi[pred_multi.shape[0] // 2][-forecast_length:]
        for i in range(n_variables):
            multi_var_results.append({
                "median_pred": median_pred_multi,
                "mae": np.mean(np.abs(median_pred_multi - futures[i])),
            })
    
    # 计算改进
    total_mae_multi = sum([r["mae"] for r in multi_var_results]) / n_variables
    total_mae_single = sum([r["mae"] for r in single_var_results]) / n_variables
    improvement = (total_mae_single - total_mae_multi) / total_mae_single * 100 if total_mae_single > 0 else 0
    
    return {
        "histories": histories,
        "futures": futures,
        "pred_multi": pred_multi,
        "multi_var_results": multi_var_results,
        "single_var_results": single_var_results,
        "total_mae_multi": total_mae_multi,
        "total_mae_single": total_mae_single,
        "improvement": improvement,
        "n_variables": n_variables,
    }


def visualize_results(interp_result, anomaly_result, forecast_result, covariate_result, multivariate_result, output_dir: Path, variable: str):
    """可视化测试结果"""
    
    # 第一张图：插值、异常检测、单变量预测
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 插值结果
    ax1 = fig.add_subplot(3, 1, 1)
    if interp_result:
        context_length = len(interp_result["original"])
        ax1.plot(range(context_length), interp_result["original"], 'k-', linewidth=1.5, label='原始数据', alpha=0.7)
        
        # 标记缺失位置
        missing_indices = interp_result["missing_indices"]
        ax1.scatter(missing_indices, interp_result["original"][missing_indices], 
                   c='red', s=50, marker='x', label=f'缺失点 ({interp_result["num_missing"]}个)', zorder=5)
        
        # 插值结果
        ax1.plot(range(context_length), interp_result["interpolated"], 'b-', linewidth=2, label='插值结果', alpha=0.8)
        
        ax1.set_title(f'{variable} - 插值任务\nMAE: {interp_result["mae"]:.4f}', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('值')
    
    # 2. 异常检测结果
    ax2 = fig.add_subplot(3, 1, 2)
    if anomaly_result:
        context_length = len(anomaly_result["original"])
        
        # 绘制置信区间
        ax2.fill_between(range(context_length), 
                        anomaly_result["lower_bound"], anomaly_result["upper_bound"],
                        alpha=0.3, color='blue', label='置信区间 (P10-P90)')
        
        # 原始数据
        ax2.plot(range(context_length), anomaly_result["original"], 'g-', linewidth=1.5, label='原始数据', alpha=0.7)
        
        # 异常数据
        ax2.plot(range(context_length), anomaly_result["segment_with_anomalies"], 'k-', linewidth=1, label='含异常数据', alpha=0.5)
        
        # 真实异常点
        anomaly_indices = anomaly_result["anomaly_indices"]
        ax2.scatter(anomaly_indices, anomaly_result["segment_with_anomalies"][anomaly_indices],
                   c='red', s=100, marker='o', label=f'真实异常 ({anomaly_result["num_anomalies"]}个)', zorder=5)
        
        # 检测到的异常点
        detected_indices = np.where(anomaly_result["detected_anomalies"])[0]
        ax2.scatter(detected_indices, anomaly_result["segment_with_anomalies"][detected_indices],
                   c='orange', s=50, marker='*', label=f'检测异常 ({len(detected_indices)}个)', zorder=4)
        
        ax2.set_title(f'{variable} - 异常检测任务\nPrecision: {anomaly_result["precision"]:.2f}, Recall: {anomaly_result["recall"]:.2f}, F1: {anomaly_result["f1"]:.2f}',
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('值')
    
    # 3. 单变量预测结果
    ax3 = fig.add_subplot(3, 1, 3)
    if forecast_result:
        context_length = len(forecast_result["history"])
        forecast_length = len(forecast_result["future"])
        
        # 历史数据
        ax3.plot(range(context_length), forecast_result["history"], 'k-', linewidth=1.5, label='历史数据')
        
        # 真实未来
        ax3.plot(range(context_length, context_length + forecast_length), 
                forecast_result["future"], 'g-', linewidth=2, label='真实未来')
        
        # 预测结果
        ax3.plot(range(context_length, context_length + forecast_length),
                forecast_result["median_pred"], 'b-', linewidth=2, label='预测中位数')
        
        # 置信区间
        if forecast_result["predicted"].shape[0] > 2:
            lower = forecast_result["predicted"][0][-forecast_length:]
            upper = forecast_result["predicted"][-1][-forecast_length:]
            ax3.fill_between(range(context_length, context_length + forecast_length),
                            lower, upper, alpha=0.3, color='blue', label='置信区间')
        
        ax3.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax3.set_title(f'{variable} - 单变量预测任务 (Patch Size={forecast_length})\nMAE: {forecast_result["mae"]:.4f}, RMSE: {forecast_result["rmse"]:.4f}',
                     fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('值')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'b1400_basic_test_results.png', dpi=150, bbox_inches='tight')
    logger.info(f"基础测试结果图表已保存: {output_dir / 'b1400_basic_test_results.png'}")
    
    # 第二张图：协变量和多变量预测
    fig2 = plt.figure(figsize=(20, 12))
    
    # 4. 协变量预测结果
    ax4 = fig2.add_subplot(2, 1, 1)
    if covariate_result:
        context_length = len(covariate_result["history"])
        forecast_length = len(covariate_result["future"])
        
        # 历史数据
        ax4.plot(range(context_length), covariate_result["history"], 'k-', linewidth=1.5, label='历史数据')
        
        # 真实未来
        ax4.plot(range(context_length, context_length + forecast_length), 
                covariate_result["future"], 'g-', linewidth=2, label='真实未来')
        
        # 有协变量预测
        ax4.plot(range(context_length, context_length + forecast_length),
                covariate_result["median_pred_with_cov"], 'b-', linewidth=2, label='有协变量预测')
        
        # 无协变量预测
        ax4.plot(range(context_length, context_length + forecast_length),
                covariate_result["median_pred_no_cov"], 'r--', linewidth=2, label='无协变量预测')
        
        ax4.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax4.set_title(f'{variable} - 协变量预测任务 (Patch Size={forecast_length})\n有协变量 MAE: {covariate_result["mae_with_cov"]:.4f}, 无协变量 MAE: {covariate_result["mae_no_cov"]:.4f}, 改进: {covariate_result["improvement"]:.2f}%',
                     fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('值')
    
    # 5. 多变量预测结果
    ax5 = fig2.add_subplot(2, 1, 2)
    if multivariate_result:
        n_variables = multivariate_result["n_variables"]
        forecast_length = len(multivariate_result["futures"][0])
        
        # 绘制每个变量的预测对比
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i in range(n_variables):
            history = multivariate_result["histories"][i]
            future = multivariate_result["futures"][i]
            context_length = len(history)
            
            # 只绘制预测部分
            offset = i * (forecast_length + 5)
            
            # 真实未来
            ax5.plot(range(offset, offset + forecast_length), future, 
                    color=colors[i % len(colors)], linewidth=2, label=f'变量{i+1}真实')
            
            # 多变量预测
            median_multi = multivariate_result["multi_var_results"][i]["median_pred"]
            ax5.plot(range(offset, offset + forecast_length), median_multi,
                    color=colors[i % len(colors)], linestyle='--', linewidth=2, label=f'变量{i+1}多变量')
            
            # 单变量预测
            median_single = multivariate_result["single_var_results"][i]["median_pred"]
            ax5.plot(range(offset, offset + forecast_length), median_single,
                    color=colors[i % len(colors)], linestyle=':', linewidth=1.5, alpha=0.7, label=f'变量{i+1}单变量')
        
        ax5.set_title(f'多变量预测任务 (Patch Size={forecast_length})\n多变量平均 MAE: {multivariate_result["total_mae_multi"]:.4f}, 单变量平均 MAE: {multivariate_result["total_mae_single"]:.4f}, 改进: {multivariate_result["improvement"]:.2f}%',
                     fontsize=14, fontweight='bold')
        ax5.legend(loc='upper right', ncol=3)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlabel('时间步 (按变量分组)')
        ax5.set_ylabel('值')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'b1400_advanced_test_results.png', dpi=150, bbox_inches='tight')
    logger.info(f"高级测试结果图表已保存: {output_dir / 'b1400_advanced_test_results.png'}")
    
    # 创建汇总表格
    fig_table, ax_table = plt.subplots(figsize=(14, 8))
    ax_table.axis('off')
    
    table_data = []
    if interp_result:
        table_data.append(['插值任务', f'{interp_result["num_missing"]}个缺失点', f'MAE: {interp_result["mae"]:.4f}', '-'])
    if anomaly_result:
        table_data.append(['异常检测', f'{anomaly_result["num_anomalies"]}个异常', 
                          f'Precision: {anomaly_result["precision"]:.2f}', f'Recall: {anomaly_result["recall"]:.2f}'])
    if forecast_result:
        table_data.append(['单变量预测', f'{len(forecast_result["future"])}步 (Patch)', 
                          f'MAE: {forecast_result["mae"]:.4f}', f'RMSE: {forecast_result["rmse"]:.4f}'])
    if covariate_result:
        table_data.append(['协变量预测', f'{len(covariate_result["future"])}步 (Patch)', 
                          f'有协变量 MAE: {covariate_result["mae_with_cov"]:.4f}', f'改进: {covariate_result["improvement"]:.2f}%'])
    if multivariate_result:
        table_data.append(['多变量预测', f'{multivariate_result["n_variables"]}个变量', 
                          f'多变量 MAE: {multivariate_result["total_mae_multi"]:.4f}', f'改进: {multivariate_result["improvement"]:.2f}%'])
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=['任务', '设置', '指标1', '指标2'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title(f'{variable} - B-1400 微调模型完整测试结果汇总', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'b1400_complete_summary_table.png', dpi=150, bbox_inches='tight')
    logger.info(f"完整汇总表格已保存: {output_dir / 'b1400_complete_summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="B-1400 微调模型完整测试")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/mlm_b1400_finetuned/2026-06-23_10-45-34/mlm-lora-b1400-finetuned",
        help="微调模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/B-1400/B-1400_20260101005255.qar.csv",
        help="数据文件路径",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="EGT1",
        help="要测试的目标变量",
    )
    parser.add_argument(
        "--covariate_variables",
        type=str,
        nargs="+",
        default=["EGT2", "N11", "N12"],
        help="协变量列表",
    )
    parser.add_argument(
        "--multivariate_variables",
        type=str,
        nargs="+",
        default=["EGT1", "EGT2", "N11", "N12"],
        help="多变量预测变量列表",
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
        default=16,
        help="预测长度 (默认为1个patch大小)",
    )
    parser.add_argument(
        "--num_missing",
        type=int,
        default=25,
        help="插值缺失点数",
    )
    parser.add_argument(
        "--anomaly_ratio",
        type=float,
        default=0.05,
        help="异常比例",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/b1400_test",
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
    logger.info(f"加载微调模型: {args.model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda")
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")
    
    # 加载目标变量数据
    logger.info(f"加载数据: {args.data_path}, 变量: {args.variable}")
    target_data = load_single_variable(args.data_path, args.variable)
    
    if target_data is None:
        logger.error("目标数据加载失败!")
        return
    
    logger.info(f"目标数据长度: {len(target_data)}, 均值: {np.mean(target_data):.4f}, 标准差: {np.std(target_data):.4f}")
    
    # 加载协变量数据
    covariate_data_list = []
    for cov_var in args.covariate_variables:
        cov_data = load_single_variable(args.data_path, cov_var)
        if cov_data is not None:
            covariate_data_list.append(cov_data)
            logger.info(f"协变量 {cov_var}: 长度={len(cov_data)}, 均值={np.mean(cov_data):.4f}")
    
    # 加载多变量数据
    multivariate_data_list = []
    for multi_var in args.multivariate_variables:
        multi_data = load_single_variable(args.data_path, multi_var)
        if multi_data is not None:
            multivariate_data_list.append(multi_data)
            logger.info(f"多变量 {multi_var}: 长度={len(multi_data)}, 均值={np.mean(multi_data):.4f}")
    
    # 运行测试
    logger.info("=" * 60)
    logger.info("开始完整测试")
    logger.info("=" * 60)
    
    # 1. 插值任务
    logger.info("运行插值任务...")
    interp_result = run_interpolation(
        pipeline, target_data, args.context_length, args.num_missing
    )
    if interp_result:
        logger.info(f"插值 MAE: {interp_result['mae']:.4f}")
    
    # 2. 异常检测任务
    logger.info("运行异常检测任务...")
    anomaly_result = run_anomaly_detection(
        pipeline, target_data, args.context_length, args.anomaly_ratio
    )
    if anomaly_result:
        logger.info(f"异常检测 Precision: {anomaly_result['precision']:.2f}")
        logger.info(f"异常检测 Recall: {anomaly_result['recall']:.2f}")
        logger.info(f"异常检测 F1: {anomaly_result['f1']:.2f}")
    
    # 3. 单变量预测任务
    logger.info("运行单变量预测任务...")
    forecast_result = run_forecast(
        pipeline, target_data, args.context_length, args.forecast_length
    )
    if forecast_result:
        logger.info(f"单变量预测 MAE: {forecast_result['mae']:.4f}")
        logger.info(f"单变量预测 RMSE: {forecast_result['rmse']:.4f}")
    
    # 4. 协变量预测任务
    logger.info("运行协变量预测任务...")
    covariate_result = None
    if covariate_data_list:
        covariate_result = run_covariate_forecast(
            pipeline, target_data, covariate_data_list, args.context_length, args.forecast_length
        )
        if covariate_result:
            logger.info(f"有协变量 MAE: {covariate_result['mae_with_cov']:.4f}")
            logger.info(f"无协变量 MAE: {covariate_result['mae_no_cov']:.4f}")
            logger.info(f"改进: {covariate_result['improvement']:.2f}%")
    else:
        logger.warning("没有可用的协变量数据")
    
    # 5. 多变量预测任务
    logger.info("运行多变量预测任务...")
    multivariate_result = None
    if multivariate_data_list:
        multivariate_result = run_multivariate_forecast(
            pipeline, multivariate_data_list, args.context_length, args.forecast_length
        )
        if multivariate_result:
            logger.info(f"多变量平均 MAE: {multivariate_result['total_mae_multi']:.4f}")
            logger.info(f"单变量平均 MAE: {multivariate_result['total_mae_single']:.4f}")
            logger.info(f"改进: {multivariate_result['improvement']:.2f}%")
    else:
        logger.warning("没有可用的多变量数据")
    
    # 可视化
    visualize_results(interp_result, anomaly_result, forecast_result, covariate_result, multivariate_result, output_dir, args.variable)
    
    # 打印汇总
    logger.info("=" * 60)
    logger.info("完整测试汇总")
    logger.info("=" * 60)
    logger.info(f"目标变量: {args.variable}")
    logger.info(f"数据长度: {len(target_data)}")
    logger.info(f"预测长度: {args.forecast_length} (Patch Size)")
    if interp_result:
        logger.info(f"插值: {interp_result['num_missing']}个缺失点, MAE={interp_result['mae']:.4f}")
    if anomaly_result:
        logger.info(f"异常检测: {anomaly_result['num_anomalies']}个异常, F1={anomaly_result['f1']:.2f}")
    if forecast_result:
        logger.info(f"单变量预测: {args.forecast_length}步, MAE={forecast_result['mae']:.4f}")
    if covariate_result:
        logger.info(f"协变量预测: 有协变量MAE={covariate_result['mae_with_cov']:.4f}, 改进={covariate_result['improvement']:.2f}%")
    if multivariate_result:
        logger.info(f"多变量预测: {multivariate_result['n_variables']}个变量, 改进={multivariate_result['improvement']:.2f}%")
    logger.info("=" * 60)
    logger.info("测试完成!")
    logger.info(f"结果保存: {output_dir}")


if __name__ == "__main__":
    main()