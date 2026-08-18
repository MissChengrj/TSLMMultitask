"""
使用微调后的 Chronos-2 MLM 模型进行异常检测、插值和预测任务

功能:
1. 插值任务: 填补时序数据中的缺失值
2. 异常检测: 检测偏离预测分布的异常点
3. 预测任务: 预测未来数据

使用方法:
    python scripts/inference/mlm_inference.py --model_path <微调模型路径> --data_path <数据路径>
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from chronos import Chronos2Pipeline

logger = logging.getLogger(__name__)

# 本地权重目录
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"


def load_excel_data(file_path: str, target_column: str = None) -> np.ndarray:
    """从 Excel 文件加载时序数据"""
    df = pd.read_excel(file_path, header=6, skiprows=1)

    if target_column is None:
        # 默认选择第一个数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["ENG POS"]
        target_column = [c for c in numeric_cols if c not in exclude_cols][0]

    logger.info(f"目标列: {target_column}")
    series = df[target_column].values.astype(np.float32)

    # 处理缺失值
    series = np.where(np.isnan(series), 0.0, series)

    return series, target_column


def create_missing_data(series: np.ndarray, missing_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    创建模拟缺失数据用于插值测试
    
    Parameters
    ----------
    series : np.ndarray
        原始完整数据
    missing_ratio : float
        缺失比例
        
    Returns
    -------
    masked_series : np.ndarray
        带缺失值的数据（NaN 表示缺失）
    missing_indices : np.ndarray
        缺失位置的索引
    """
    length = len(series)
    num_missing = int(length * missing_ratio)

    # 随机选择缺失位置
    np.random.seed(42)
    missing_indices = np.random.choice(length, num_missing, replace=False)

    masked_series = series.copy()
    masked_series[missing_indices] = np.nan

    logger.info(f"创建了 {num_missing} 个缺失点 ({missing_ratio*100:.1f}%)")
    return masked_series, missing_indices


def detect_anomalies(
    series: np.ndarray,
    predictions: np.ndarray,
    # ... 保持入参不变 ...
) -> tuple[np.ndarray, np.ndarray]:
    
    # 提取预测中位数作为“最理想的重构基准”
    median_idx = len(predictions) // 2
    reconstructed_baseline = predictions[median_idx]
    
    # 取上下界仅用于归一化宽度（可选）
    lower_bound = predictions[0]  
    upper_bound = predictions[-1]  
    width = upper_bound - lower_bound + 1e-8

    # 计算与中位数的绝对残差作为异常得分的分子（连续得分，不再全是0）
    absolute_error = np.abs(series - reconstructed_baseline)
    
    # 标准化异常分数
    anomaly_scores = absolute_error / width
    
    # 依然可以利用上下界判断是否“真正”越界报警
    anomaly_indices = np.where((series < lower_bound) | (series > upper_bound))[0]

    logger.info(f"检测到 {len(anomaly_indices)} 个异常点")
    return anomaly_indices, anomaly_scores


def interpolate_missing(
    masked_series: np.ndarray,
    predictions: np.ndarray,
    missing_indices: np.ndarray,
    use_median: bool = True,
) -> np.ndarray:
    """
    使用模型预测填补缺失值
    
    Parameters
    ----------
    masked_series : np.ndarray
        带缺失值的数据
    predictions : np.ndarray
        模型预测的分位数
    missing_indices : np.ndarray
        缺失位置索引
    use_median : bool
        是否使用中位数 (P50) 填补，否则使用均值
        
    Returns
    -------
    interpolated_series : np.ndarray
        填补后的数据
    """
    interpolated_series = masked_series.copy()

    # 使用中位数预测填补
    if use_median:
        # 找到中位数索引 (通常是中间的分位数)
        median_idx = len(predictions) // 2
        fill_values = predictions[median_idx]
    else:
        # 使用所有分位数的均值
        fill_values = predictions.mean(axis=0)

    # 填补缺失位置
    for idx in missing_indices:
        if idx < len(fill_values):
            interpolated_series[idx] = fill_values[idx]

    return interpolated_series


def visualize_all_tasks(
    original_series: np.ndarray,
    masked_series: np.ndarray,
    interpolated_series: np.ndarray,
    reconstruction_preds: np.ndarray,
    forecast_preds: np.ndarray,
    anomaly_indices: np.ndarray,
    missing_indices: np.ndarray,
    context_length: int,
    forecast_length: int,
    title: str = "MLM Model Results",
    save_path: str = None,
):
    """可视化所有任务结果"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # ========== 上图：插值任务 ==========
    ax1 = axes[0]
    length = min(len(original_series), reconstruction_preds.shape[1])

    # 绘制置信区间
    ax1.fill_between(
        range(length),
        reconstruction_preds[0][:length],
        reconstruction_preds[-1][:length],
        alpha=0.3,
        color="blue",
        label="90% 置信区间",
    )

    # 绘制中位数预测
    median_idx = len(reconstruction_preds) // 2
    ax1.plot(reconstruction_preds[median_idx][:length], "b-", linewidth=1, label="预测中位数 (P50)")

    # 绘制原始数据
    ax1.plot(original_series[:length], "g-", linewidth=1.5, alpha=0.7, label="原始数据")

    # 标记缺失位置
    valid_missing = missing_indices[missing_indices < length]
    if len(valid_missing) > 0:
        ax1.scatter(valid_missing, original_series[valid_missing], c="red", s=50, marker="x", label="缺失点 (原始值)")

    # 绘制填补后的数据
    ax1.plot(interpolated_series[:length], "orange", linewidth=1.5, linestyle="--", alpha=0.8, label="填补后数据")

    ax1.set_title(f"{title} - 插值任务 (重构历史数据)")
    ax1.set_xlabel("时间步")
    ax1.set_ylabel("数值")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ========== 中图：异常检测 ==========
    ax2 = axes[1]

    # 绘制置信区间
    ax2.fill_between(
        range(length),
        reconstruction_preds[0][:length],
        reconstruction_preds[-1][:length],
        alpha=0.3,
        color="green",
        label="正常范围 (P10-P90)",
    )

    # 绘制数据
    ax2.plot(original_series[:length], "k-", linewidth=1, label="数据")

    # 标记异常点
    valid_anomalies = anomaly_indices[anomaly_indices < length]
    if len(valid_anomalies) > 0:
        ax2.scatter(
            valid_anomalies,
            original_series[valid_anomalies],
            c="red",
            s=100,
            marker="o",
            label=f"异常点 ({len(valid_anomalies)}个)",
            zorder=5,
        )

    ax2.set_title(f"{title} - 异常检测 (超出置信区间的点)")
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("数值")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ========== 下图：预测任务 ==========
    ax3 = axes[2]

    # 历史数据范围
    hist_length = min(context_length, len(original_series))

    # 绘制历史数据
    ax3.plot(range(hist_length), original_series[:hist_length], "k-", linewidth=1.5, label="历史数据")

    # 绘制预测置信区间
    forecast_start = hist_length
    forecast_end = forecast_start + forecast_length

    ax3.fill_between(
        range(forecast_start, forecast_end),
        forecast_preds[0][:forecast_length],
        forecast_preds[-1][:forecast_length],
        alpha=0.3,
        color="purple",
        label="预测置信区间 (P10-P90)",
    )

    # 绘制预测中位数
    median_idx = len(forecast_preds) // 2
    ax3.plot(
        range(forecast_start, forecast_end),
        forecast_preds[median_idx][:forecast_length],
        "purple",
        linewidth=2,
        label="预测中位数 (P50)",
    )

    # 如果有真实未来数据，绘制对比
    if len(original_series) > hist_length:
        actual_future_length = min(len(original_series) - hist_length, forecast_length)
        ax3.plot(
            range(forecast_start, forecast_start + actual_future_length),
            original_series[hist_length:hist_length + actual_future_length],
            "g--",
            linewidth=1.5,
            alpha=0.7,
            label="真实未来数据",
        )

    # 添加分隔线
    ax3.axvline(x=hist_length - 0.5, color="red", linestyle=":", linewidth=2, label="预测起点")

    ax3.set_title(f"{title} - 预测任务 (预测未来 {forecast_length} 步)")
    ax3.set_xlabel("时间步")
    ax3.set_ylabel("数值")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图表已保存到: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="MLM 模型推理 - 异常检测、插值与预测")

    parser.add_argument(
        "--model_path",
        type=str,
        default=str(WEIGHTS_DIR / "mlm_finetuned" / "2026-06-22_16-51-10" / "mlm-lora-finetuned"),
        help="微调后的模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/038227_1.xlsx",
        help="数据文件路径",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default=None,
        help="目标列名（如不指定则自动选择）",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="上下文长度（历史数据长度）",
    )
    parser.add_argument(
        "--forecast_length",
        type=int,
        default=64,
        help="预测长度（未来数据长度）",
    )
    parser.add_argument(
        "--missing_ratio",
        type=float,
        default=0.1,
        help="模拟缺失比例（用于插值测试）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        default=False,
        help="不显示可视化图表",
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # 设置输出目录
    if args.output_dir is None:
        output_dir = Path("results/mlm_inference")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载微调后的模型
    logger.info(f"加载微调后的模型: {args.model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda")
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")

    # 2. 加载测试数据
    logger.info(f"加载测试数据: {args.data_path}")
    series, target_column = load_excel_data(args.data_path, args.target_column)
    logger.info(f"数据长度: {len(series)}, 目标列: {target_column}")

    # 取一段数据进行测试
    test_series = series[:args.context_length + args.forecast_length]
    logger.info(f"测试数据长度: {len(test_series)} (历史: {args.context_length}, 预测: {args.forecast_length})")

    # 3. 创建模拟缺失数据（用于插值测试）
    context_series = test_series[:args.context_length]
    masked_series, missing_indices = create_missing_data(context_series, args.missing_ratio)

    # ========== 任务1 & 2: 插值和异常检测（重构历史数据）==========
    logger.info("=" * 60)
    logger.info("任务 1 & 2: 插值和异常检测（重构历史数据）")
    logger.info("=" * 60)

    # 将缺失值设为 NaN，模型会尝试重构
    input_with_missing = [{"target": np.where(np.isnan(masked_series), np.nan, masked_series)}]

    reconstruction_preds = pipeline.predict(
        inputs=input_with_missing,
        prediction_length=len(context_series),
    )

    # 处理预测结果
    if isinstance(reconstruction_preds, list):
        recon_pred_tensor = reconstruction_preds[0]
    else:
        recon_pred_tensor = reconstruction_preds

    if recon_pred_tensor.dim() == 3:
        recon_pred_tensor = recon_pred_tensor.squeeze(0)

    recon_pred_tensor = recon_pred_tensor.cpu().numpy()
    logger.info(f"重构预测形状: {recon_pred_tensor.shape}")

    # 插值任务：填补缺失值
    logger.info("执行插值任务...")
    interpolated_series = interpolate_missing(masked_series, recon_pred_tensor, missing_indices)

    # 计算插值误差
    interpolation_errors = np.abs(interpolated_series[missing_indices] - context_series[missing_indices])
    mean_error = np.mean(interpolation_errors)
    logger.info(f"插值平均绝对误差: {mean_error:.4f}")

    # 异常检测任务
    logger.info("执行异常检测任务...")
    anomaly_indices, anomaly_scores = detect_anomalies(context_series, recon_pred_tensor)

    # ========== 任务3: 预测未来数据 ==========
    logger.info("=" * 60)
    logger.info("任务 3: 预测未来数据")
    logger.info("=" * 60)

    # 使用完整的历史数据进行预测
    forecast_input = [{"target": context_series}]
    forecast_preds = pipeline.predict(
        inputs=forecast_input,
        prediction_length=args.forecast_length,
    )

    # 处理预测结果
    if isinstance(forecast_preds, list):
        forecast_pred_tensor = forecast_preds[0]
    else:
        forecast_pred_tensor = forecast_preds

    if forecast_pred_tensor.dim() == 3:
        forecast_pred_tensor = forecast_pred_tensor.squeeze(0)

    forecast_pred_tensor = forecast_pred_tensor.cpu().numpy()
    logger.info(f"预测形状: {forecast_pred_tensor.shape}")

    # 如果有真实未来数据，计算预测误差
    if len(test_series) > args.context_length:
        actual_future = test_series[args.context_length:args.context_length + args.forecast_length]
        median_idx = len(forecast_pred_tensor) // 2
        forecast_median = forecast_pred_tensor[median_idx][:len(actual_future)]
        forecast_error = np.mean(np.abs(forecast_median - actual_future))
        logger.info(f"预测平均绝对误差: {forecast_error:.4f}")

    # 保存结果
    results = {
        "original_series": test_series,
        "context_series": context_series,
        "masked_series": masked_series,
        "interpolated_series": interpolated_series,
        "missing_indices": missing_indices,
        "anomaly_indices": anomaly_indices,
        "anomaly_scores": anomaly_scores,
        "reconstruction_predictions": recon_pred_tensor,
        "forecast_predictions": forecast_pred_tensor,
        "interpolation_error": mean_error,
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
    }

    # 保存为 numpy 文件
    np.savez(output_dir / "results.npz", **results)
    logger.info(f"结果已保存到: {output_dir / 'results.npz'}")

    # 可视化结果
    if not args.no_visualize:
        visualize_all_tasks(
            original_series=test_series,
            masked_series=masked_series,
            interpolated_series=interpolated_series,
            reconstruction_preds=recon_pred_tensor,
            forecast_preds=forecast_pred_tensor,
            anomaly_indices=anomaly_indices,
            missing_indices=missing_indices,
            context_length=args.context_length,
            forecast_length=args.forecast_length,
            title=f"MLM Model - {target_column}",
            save_path=str(output_dir / "visualization.png"),
        )

    # 打印统计信息
    logger.info("=" * 60)
    logger.info("结果统计:")
    logger.info(f"  数据长度: {len(test_series)} (历史: {args.context_length}, 未来: {args.forecast_length})")
    logger.info(f"  缺失点数: {len(missing_indices)}")
    logger.info(f"  异常点数: {len(anomaly_indices)}")
    logger.info(f"  插值误差: {mean_error:.4f}")
    if len(test_series) > args.context_length:
        logger.info(f"  预测误差: {forecast_error:.4f}")
    logger.info("=" * 60)

    logger.info("推理完成!")


if __name__ == "__main__":
    main()