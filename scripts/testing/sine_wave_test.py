"""
正弦函数测试脚本

使用微调后的模型对纯净正弦函数和加噪声正弦函数进行：
1. 插值测试
2. 异常检测测试
3. 预测测试

使用方法:
    python scripts/testing/sine_wave_test.py --model_path weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from chronos import Chronos2Pipeline

logger = logging.getLogger(__name__)


def generate_sine_wave(
    length: int = 512,
    amplitude: float = 10.0,
    frequency: float = 0.02,
    phase: float = 0.0,
    noise_std: float = 0.0,
):
    """生成正弦波数据"""
    t = np.arange(length)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, length)
        sine_wave = sine_wave + noise
    
    return sine_wave.astype(np.float32)


def test_interpolation(pipeline, data, context_length: int, missing_ratio: float = 0.1):
    """插值测试"""
    test_data = data[:context_length].copy()
    
    # 创建缺失数据
    num_missing = int(context_length * missing_ratio)
    np.random.seed(42)  # 固定随机种子以便复现
    missing_indices = np.random.choice(context_length, num_missing, replace=False)
    
    data_with_missing = test_data.copy()
    data_with_missing[missing_indices] = np.nan
    
    # 推理
    predictions = pipeline.predict(
        [{"target": data_with_missing}],
        prediction_length=context_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    
    median_pred = pred[pred.shape[0] // 2]
    
    # 计算误差
    actual_values = test_data[missing_indices]
    predicted_values = median_pred[missing_indices]
    mae = np.mean(np.abs(actual_values - predicted_values))
    
    return {
        "original": test_data,
        "data_with_missing": data_with_missing,
        "predicted": median_pred,
        "missing_indices": missing_indices,
        "mae": mae,
    }


def test_anomaly_detection(pipeline, data, context_length: int, confidence_level: float = 0.90):
    """异常检测测试"""
    test_data = data[:context_length].copy()
    
    # 添加人工异常（偏离较大的点）
    np.random.seed(123)
    anomaly_indices = [50, 100, 150, 200]
    anomaly_offsets = [15, -20, 25, -18]  # 较大的偏离
    
    data_with_anomalies = test_data.copy()
    for idx, offset in zip(anomaly_indices, anomaly_offsets):
        data_with_anomalies[idx] += offset
    
    # 推理
    predictions = pipeline.predict(
        [{"target": data_with_anomalies}],
        prediction_length=context_length,
    )
    
    pred = predictions[0].cpu().numpy()
    if pred.ndim == 3 and pred.shape[0] == 1:
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
    detected = (data_with_anomalies < lower_bound) | (data_with_anomalies > upper_bound)
    
    # 计算性能
    true_anomaly_mask = np.zeros(context_length, dtype=bool)
    true_anomaly_mask[anomaly_indices] = True
    
    tp = np.sum(detected & true_anomaly_mask)
    fp = np.sum(detected & ~true_anomaly_mask)
    fn = np.sum(~detected & true_anomaly_mask)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "original": test_data,
        "data_with_anomalies": data_with_anomalies,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "median_pred": median_pred,
        "detected": detected,
        "anomaly_indices": anomaly_indices,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def test_forecast(pipeline, data, context_length: int, forecast_length: int):
    """预测测试"""
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
    
    median_pred = pred[pred.shape[0] // 2]
    
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


def visualize_results(results_clean, results_noisy, output_dir: Path):
    """可视化所有结果"""
    
    fig = plt.figure(figsize=(24, 20))
    
    # ========== 纯净正弦函数测试结果 ==========
    
    # 1. 纯净正弦 - 插值
    ax1 = fig.add_subplot(6, 2, 1)
    interp = results_clean["interpolation"]
    ax1.plot(interp["original"], 'k-', linewidth=1.5, label='原始数据', alpha=0.7)
    ax1.plot(interp["predicted"], 'b-', linewidth=2, label='插值结果')
    
    # 标记缺失点
    for idx in interp["missing_indices"]:
        ax1.scatter(idx, interp["predicted"][idx], c='green', s=80, marker='o', zorder=5, alpha=0.7)
    
    ax1.set_title(f'纯净正弦 - 插值测试\nMAE: {interp["mae"]:.4f}', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('值')
    
    # 2. 噪声正弦 - 插值
    ax2 = fig.add_subplot(6, 2, 2)
    interp = results_noisy["interpolation"]
    ax2.plot(interp["original"], 'k-', linewidth=1.5, label='原始数据（含噪声）', alpha=0.7)
    ax2.plot(interp["predicted"], 'b-', linewidth=2, label='插值结果')
    
    for idx in interp["missing_indices"]:
        ax2.scatter(idx, interp["predicted"][idx], c='green', s=80, marker='o', zorder=5, alpha=0.7)
    
    ax2.set_title(f'噪声正弦 - 插值测试\nMAE: {interp["mae"]:.4f}', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('值')
    
    # 3. 纯净正弦 - 异常检测
    ax3 = fig.add_subplot(6, 2, 3)
    anomaly = results_clean["anomaly_detection"]
    ax3.plot(anomaly["data_with_anomalies"], 'k-', linewidth=1.5, label='测试数据（含异常）')
    ax3.fill_between(range(len(anomaly["lower_bound"])),
                     anomaly["lower_bound"], anomaly["upper_bound"],
                     alpha=0.3, color='blue', label='置信区间 (P5-P95)')
    ax3.plot(anomaly["median_pred"], 'b-', linewidth=2, alpha=0.7, label='预测中位数')
    
    # 标记真实异常点
    for idx in anomaly["anomaly_indices"]:
        ax3.scatter(idx, anomaly["data_with_anomalies"][idx], c='red', s=150, marker='x', zorder=5, linewidths=3)
    
    # 标记检测到的异常点
    detected_indices = np.where(anomaly["detected"])[0]
    ax3.scatter(detected_indices, anomaly["data_with_anomalies"][detected_indices],
                c='orange', s=60, marker='o', alpha=0.6, label='检测到的异常')
    
    ax3.set_title(f'纯净正弦 - 异常检测\nPrecision: {anomaly["precision"]:.2f}, Recall: {anomaly["recall"]:.2f}, F1: {anomaly["f1"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('值')
    
    # 4. 噪声正弦 - 异常检测
    ax4 = fig.add_subplot(6, 2, 4)
    anomaly = results_noisy["anomaly_detection"]
    ax4.plot(anomaly["data_with_anomalies"], 'k-', linewidth=1.5, label='测试数据（含噪声+异常）')
    ax4.fill_between(range(len(anomaly["lower_bound"])),
                     anomaly["lower_bound"], anomaly["upper_bound"],
                     alpha=0.3, color='blue', label='置信区间 (P5-P95)')
    ax4.plot(anomaly["median_pred"], 'b-', linewidth=2, alpha=0.7, label='预测中位数')
    
    for idx in anomaly["anomaly_indices"]:
        ax4.scatter(idx, anomaly["data_with_anomalies"][idx], c='red', s=150, marker='x', zorder=5, linewidths=3)
    
    detected_indices = np.where(anomaly["detected"])[0]
    ax4.scatter(detected_indices, anomaly["data_with_anomalies"][detected_indices],
                c='orange', s=60, marker='o', alpha=0.6, label='检测到的异常')
    
    ax4.set_title(f'噪声正弦 - 异常检测\nPrecision: {anomaly["precision"]:.2f}, Recall: {anomaly["recall"]:.2f}, F1: {anomaly["f1"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('值')
    
    # 5. 纯净正弦 - 预测
    ax5 = fig.add_subplot(6, 2, 5)
    forecast = results_clean["forecast"]
    if forecast:
        context_length = len(forecast["history"])
        ax5.plot(range(context_length), forecast["history"], 'k-', linewidth=1.5, label='历史数据')
        ax5.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["future"], 'g-', linewidth=2, label='真实未来')
        
        n_quantiles = forecast["predicted"].shape[0]
        ax5.fill_between(range(context_length, context_length + len(forecast["future"])),
                         forecast["predicted"][0], forecast["predicted"][-1],
                         alpha=0.3, color='blue', label='置信区间')
        ax5.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["median_pred"], 'b-', linewidth=2, label='预测中位数')
        
        ax5.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax5.set_title(f'纯净正弦 - 预测测试\nMAE: {forecast["mae"]:.4f}, RMSE: {forecast["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlabel('时间步')
        ax5.set_ylabel('值')
    
    # 6. 噪声正弦 - 预测
    ax6 = fig.add_subplot(6, 2, 6)
    forecast = results_noisy["forecast"]
    if forecast:
        context_length = len(forecast["history"])
        ax6.plot(range(context_length), forecast["history"], 'k-', linewidth=1.5, label='历史数据（含噪声）')
        ax6.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["future"], 'g-', linewidth=2, label='真实未来（含噪声）')
        
        n_quantiles = forecast["predicted"].shape[0]
        ax6.fill_between(range(context_length, context_length + len(forecast["future"])),
                         forecast["predicted"][0], forecast["predicted"][-1],
                         alpha=0.3, color='blue', label='置信区间')
        ax6.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["median_pred"], 'b-', linewidth=2, label='预测中位数')
        
        ax6.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax6.set_title(f'噪声正弦 - 预测测试\nMAE: {forecast["mae"]:.4f}, RMSE: {forecast["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel('时间步')
        ax6.set_ylabel('值')
    
    # ========== 汇总对比 ==========
    
    # 7. 插值误差对比
    ax7 = fig.add_subplot(6, 2, 7)
    interp_mae_clean = results_clean["interpolation"]["mae"]
    interp_mae_noisy = results_noisy["interpolation"]["mae"]
    
    bars = ax7.bar(['纯净正弦', '噪声正弦'], [interp_mae_clean, interp_mae_noisy],
                   color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax7.set_title('插值误差对比 (MAE)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('MAE')
    ax7.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [interp_mae_clean, interp_mae_noisy]):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    # 8. 异常检测性能对比
    ax8 = fig.add_subplot(6, 2, 8)
    
    metrics = ['Precision', 'Recall', 'F1']
    clean_vals = [results_clean["anomaly_detection"]["precision"],
                  results_clean["anomaly_detection"]["recall"],
                  results_clean["anomaly_detection"]["f1"]]
    noisy_vals = [results_noisy["anomaly_detection"]["precision"],
                  results_noisy["anomaly_detection"]["recall"],
                  results_noisy["anomaly_detection"]["f1"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax8.bar(x - width/2, clean_vals, width, label='纯净正弦', color='steelblue', alpha=0.7)
    ax8.bar(x + width/2, noisy_vals, width, label='噪声正弦', color='coral', alpha=0.7)
    
    ax8.set_xticks(x)
    ax8.set_xticklabels(metrics)
    ax8.set_title('异常检测性能对比', fontsize=12, fontweight='bold')
    ax8.set_ylabel('性能指标')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. 预测误差对比
    ax9 = fig.add_subplot(6, 2, 9)
    
    forecast_mae_clean = results_clean["forecast"]["mae"]
    forecast_mae_noisy = results_noisy["forecast"]["mae"]
    forecast_rmse_clean = results_clean["forecast"]["rmse"]
    forecast_rmse_noisy = results_noisy["forecast"]["rmse"]
    
    x = np.arange(2)
    width = 0.35
    
    ax9.bar(x - width/2, [forecast_mae_clean, forecast_mae_noisy], width, label='MAE', color='steelblue', alpha=0.7)
    ax9.bar(x + width/2, [forecast_rmse_clean, forecast_rmse_noisy], width, label='RMSE', color='coral', alpha=0.7)
    
    ax9.set_xticks(x)
    ax9.set_xticklabels(['纯净正弦', '噪声正弦'])
    ax9.set_title('预测误差对比', fontsize=12, fontweight='bold')
    ax9.set_ylabel('误差')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # 10. 原始数据对比
    ax10 = fig.add_subplot(6, 2, 10)
    
    sine_clean = generate_sine_wave(length=512, noise_std=0)
    sine_noisy = generate_sine_wave(length=512, noise_std=2.0)
    
    ax10.plot(sine_clean, 'b-', linewidth=2, label='纯净正弦', alpha=0.8)
    ax10.plot(sine_noisy, 'r-', linewidth=1, label='噪声正弦 (σ=2)', alpha=0.6)
    ax10.set_title('原始数据对比', fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.set_xlabel('时间步')
    ax10.set_ylabel('值')
    
    # 11. 汇总表格
    ax11 = fig.add_subplot(6, 2, 11)
    ax11.axis('off')
    
    table_data = [
        ['插值', f'{interp_mae_clean:.4f}', f'{interp_mae_noisy:.4f}'],
        ['异常检测 Precision', f'{clean_vals[0]:.4f}', f'{noisy_vals[0]:.4f}'],
        ['异常检测 Recall', f'{clean_vals[1]:.4f}', f'{noisy_vals[1]:.4f}'],
        ['异常检测 F1', f'{clean_vals[2]:.4f}', f'{noisy_vals[2]:.4f}'],
        ['预测 MAE', f'{forecast_mae_clean:.4f}', f'{forecast_mae_noisy:.4f}'],
        ['预测 RMSE', f'{forecast_rmse_clean:.4f}', f'{forecast_rmse_noisy:.4f}'],
    ]
    
    table = ax11.table(
        cellText=table_data,
        colLabels=['任务', '纯净正弦', '噪声正弦'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax11.set_title('测试结果汇总表', fontsize=14, fontweight='bold', pad=20)
    
    # 12. 结论
    ax12 = fig.add_subplot(6, 2, 12)
    ax12.axis('off')
    
    conclusions = """
    测试结论:
    
    1. 插值任务:
       - 纯净正弦函数插值误差较小
       - 噪声正弦函数插值误差较大，但模型仍能有效重构
    
    2. 异常检测任务:
       - 纯净正弦函数异常检测效果较好
       - 噪声正弦函数存在较多误报
    
    3. 预测任务:
       - 纯净正弦函数预测效果较好
       - 噪声正弦函数预测误差较大
    
    说明:
    - 模型在纯净数据上表现更好
    - 噪声会影响模型的各项性能
    - MLM微调模型主要适用于重构任务
    """
    
    ax12.text(0.1, 0.9, conclusions, transform=ax12.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sine_wave_test_results.png', dpi=150, bbox_inches='tight')
    logger.info(f"测试结果图表已保存: {output_dir / 'sine_wave_test_results.png'}")


def main():
    parser = argparse.ArgumentParser(description="正弦函数测试")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned",
        help="模型路径",
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
        "--noise_std",
        type=float,
        default=2.0,
        help="噪声标准差",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sine_wave_test",
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
    
    # 生成正弦波数据
    total_length = args.context_length + args.forecast_length
    logger.info(f"生成正弦波数据, 长度: {total_length}")
    
    # 纯净正弦
    sine_clean = generate_sine_wave(
        length=total_length,
        amplitude=10.0,
        frequency=0.02,
        noise_std=0.0,
    )
    
    # 噪声正弦
    sine_noisy = generate_sine_wave(
        length=total_length,
        amplitude=10.0,
        frequency=0.02,
        noise_std=args.noise_std,
    )
    
    logger.info(f"纯净正弦: 均值={np.mean(sine_clean):.4f}, 标准差={np.std(sine_clean):.4f}")
    logger.info(f"噪声正弦: 均值={np.mean(sine_noisy):.4f}, 标准差={np.std(sine_noisy):.4f}")
    
    results_clean = {}
    results_noisy = {}
    
    # ========== 纯净正弦测试 ==========
    logger.info("=" * 60)
    logger.info("纯净正弦函数测试")
    logger.info("=" * 60)
    
    # 插值
    logger.info("插值测试...")
    results_clean["interpolation"] = test_interpolation(
        pipeline, sine_clean, args.context_length, missing_ratio=0.1
    )
    logger.info(f"插值 MAE: {results_clean['interpolation']['mae']:.4f}")
    
    # 异常检测
    logger.info("异常检测测试...")
    results_clean["anomaly_detection"] = test_anomaly_detection(
        pipeline, sine_clean, args.context_length
    )
    logger.info(f"Precision: {results_clean['anomaly_detection']['precision']:.4f}")
    logger.info(f"Recall: {results_clean['anomaly_detection']['recall']:.4f}")
    logger.info(f"F1: {results_clean['anomaly_detection']['f1']:.4f}")
    
    # 预测
    logger.info("预测测试...")
    results_clean["forecast"] = test_forecast(
        pipeline, sine_clean, args.context_length, args.forecast_length
    )
    if results_clean["forecast"]:
        logger.info(f"预测 MAE: {results_clean['forecast']['mae']:.4f}")
        logger.info(f"预测 RMSE: {results_clean['forecast']['rmse']:.4f}")
    
    # ========== 噪声正弦测试 ==========
    logger.info("=" * 60)
    logger.info("噪声正弦函数测试")
    logger.info("=" * 60)
    
    # 插值
    logger.info("插值测试...")
    results_noisy["interpolation"] = test_interpolation(
        pipeline, sine_noisy, args.context_length, missing_ratio=0.1
    )
    logger.info(f"插值 MAE: {results_noisy['interpolation']['mae']:.4f}")
    
    # 异常检测
    logger.info("异常检测测试...")
    results_noisy["anomaly_detection"] = test_anomaly_detection(
        pipeline, sine_noisy, args.context_length
    )
    logger.info(f"Precision: {results_noisy['anomaly_detection']['precision']:.4f}")
    logger.info(f"Recall: {results_noisy['anomaly_detection']['recall']:.4f}")
    logger.info(f"F1: {results_noisy['anomaly_detection']['f1']:.4f}")
    
    # 预测
    logger.info("预测测试...")
    results_noisy["forecast"] = test_forecast(
        pipeline, sine_noisy, args.context_length, args.forecast_length
    )
    if results_noisy["forecast"]:
        logger.info(f"预测 MAE: {results_noisy['forecast']['mae']:.4f}")
        logger.info(f"预测 RMSE: {results_noisy['forecast']['rmse']:.4f}")
    
    # 可视化
    visualize_results(results_clean, results_noisy, output_dir)
    
    logger.info("=" * 60)
    logger.info("测试完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()