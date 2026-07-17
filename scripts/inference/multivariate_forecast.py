"""
展示 Chronos-2 多变量预测功能

Chronos-2 支持多变量预测，可以同时对多个相关变量进行联合预测，
模型会利用变量之间的相关性来提高预测精度。

使用方法:
    python scripts/inference/multivariate_forecast.py --model_path <模型路径> --data_path <数据路径>
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
matplotlib.rcParams['axes.unicode_minus'] = False

from chronos import Chronos2Pipeline

logger = logging.getLogger(__name__)

# 本地权重目录
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"


def load_multivariate_data(
    file_path: str,
    target_columns: list = None,
) -> tuple[np.ndarray, list]:
    """
    从 Excel 文件加载多变量数据
    
    Parameters
    ----------
    file_path : str
        Excel 文件路径
    target_columns : list
        目标列名列表
        
    Returns
    -------
    multivariate_data : np.ndarray
        多变量数据 (n_variates, history_length)
    target_columns : list
        目标列名列表
    """
    df = pd.read_excel(file_path, header=6, skiprows=1)
    
    # 获取所有数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["ENG POS"]  # 排除的列
    
    # 选择目标列
    if target_columns is None:
        available_cols = [c for c in numeric_cols if c not in exclude_cols]
        target_columns = available_cols[:4]  # 默认选择前4列
    
    logger.info(f"目标列: {target_columns}")
    
    # 提取多变量数据
    multivariate_data = []
    for col in target_columns:
        data = df[col].values.astype(np.float32)
        data = np.where(np.isnan(data), 0.0, data)
        multivariate_data.append(data)
    
    # 转换为 (n_variates, history_length) 形状
    multivariate_data = np.stack(multivariate_data, axis=0)
    
    return multivariate_data, target_columns


def visualize_multivariate_forecast(
    history_data: np.ndarray,
    future_data: np.ndarray,
    forecast_preds: np.ndarray,
    univariate_forecast_preds: np.ndarray,
    target_columns: list,
    context_length: int,
    forecast_length: int,
    confidence_level: float = 0.50,
    title: str = "Multivariate Forecast",
    save_path: str = None,
):
    """可视化多变量预测结果
    
    Parameters
    ----------
    confidence_level : float
        置信区间水平，如 0.50 表示 P25-P75（50%置信区间，即 P50-P90 的一半）
    """
    n_variates = len(target_columns)
    fig, axes = plt.subplots(n_variates, 1, figsize=(16, 4 * n_variates))
    
    if n_variates == 1:
        axes = [axes]
    
    n_quantiles = forecast_preds.shape[1]
    median_idx = n_quantiles // 2
    
    # 计算置信区间对应的分位数索引
    # P30-P70 置信区间：从 P30 到 P70（即 40% 的区间）
    # 用户要求 P29-P70，但 Chronos 分位数步长为 5%，最接近的是 P30-P70
    lower_quantile = 0.30
    upper_quantile = 0.70
    
    # 计算索引（假设分位数从 0.05 到 0.95，共 21 个）
    # 索引 = (quantile - 0.05) / 0.05
    lower_idx = int((lower_quantile - 0.05) / 0.05)
    upper_idx = int((upper_quantile - 0.05) / 0.05)
    
    # 确保索引在有效范围内
    lower_idx = max(0, min(lower_idx, n_quantiles - 1))
    upper_idx = max(0, min(upper_idx, n_quantiles - 1))
    
    logger.info(f"置信区间 P{int(lower_quantile*100)}-P{int(upper_quantile*100)}，索引: {lower_idx}-{upper_idx}")
    
    for i, (ax, col_name) in enumerate(zip(axes, target_columns)):
        # 绘制历史数据
        ax.plot(range(context_length), history_data[i], "k-", linewidth=1.5)
        
        # 绘制真实未来数据
        forecast_start = context_length
        actual_length = min(len(future_data[i]), forecast_length)
        ax.plot(
            range(forecast_start, forecast_start + actual_length),
            future_data[i][:actual_length],
            "g--",
            linewidth=1.5,
            alpha=0.7,
        )
        
        # 绘制多变量预测置信区间
        ax.fill_between(
            range(forecast_start, forecast_start + forecast_length),
            forecast_preds[i, lower_idx][:forecast_length],
            forecast_preds[i, upper_idx][:forecast_length],
            alpha=0.3,
            color="blue",
        )
        
        # 绘制多变量预测中位数
        ax.plot(
            range(forecast_start, forecast_start + forecast_length),
            forecast_preds[i, median_idx][:forecast_length],
            "b-",
            linewidth=2,
        )
        
        # 绘制单变量预测置信区间（对比）
        ax.fill_between(
            range(forecast_start, forecast_start + forecast_length),
            univariate_forecast_preds[i, lower_idx][:forecast_length],
            univariate_forecast_preds[i, upper_idx][:forecast_length],
            alpha=0.2,
            color="orange",
        )
        
        # 绘制单变量预测中位数
        ax.plot(
            range(forecast_start, forecast_start + forecast_length),
            univariate_forecast_preds[i, median_idx][:forecast_length],
            "orange",
            linewidth=2,
            linestyle="--",
        )
        
        # 添加分隔线
        ax.axvline(x=context_length - 0.5, color="red", linestyle=":", linewidth=2)
        
        # 计算误差
        multi_error = np.mean(np.abs(forecast_preds[i, median_idx][:actual_length] - future_data[i][:actual_length]))
        uni_error = np.mean(np.abs(univariate_forecast_preds[i, median_idx][:actual_length] - future_data[i][:actual_length]))
        improvement = (uni_error - multi_error) / uni_error * 100
        
        ax.set_title(f"{col_name} - 多变量误差: {multi_error:.2f}, 单变量误差: {uni_error:.2f}, 改进: {improvement:.1f}%")
        ax.set_xlabel("时间步")
        ax.set_ylabel("数值")
        ax.grid(True, alpha=0.3)
    
    # 在图片下方添加统一的图例
    fig.legend(
        labels=[
            "历史数据",
            "真实未来数据",
            f"多变量预测置信区间 (P{int(lower_quantile*100)}-P{int(upper_quantile*100)})",
            "多变量预测 (P50)",
            f"单变量预测置信区间 (P{int(lower_quantile*100)}-P{int(upper_quantile*100)})",
            "单变量预测 (P50)",
            "预测起点",
        ],
        loc="lower center",
        ncol=7,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # 为底部图例留出空间
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图表已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 多变量预测演示")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(WEIGHTS_DIR / "chronos-2"),
        help="模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/038227_1.xlsx",
        help="数据文件路径",
    )
    parser.add_argument(
        "--target_columns",
        type=str,
        nargs="+",
        default=["DEGT-CRUISE", "DEGT_SMOOTHED-CRUISE", "DEGT_D-CRUISE", "DEGT_D_SMOOTHED-CRUISE"],
        help="目标列名列表（多个列名用空格分隔）",
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
        default=32,
        help="预测长度（未来数据长度）",
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.80,
        help="置信区间水平（如 0.80 表示 P10-P90）",
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
        output_dir = Path("results/multivariate_forecast")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载模型
    logger.info(f"加载模型: {args.model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda")
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")
    
    # 2. 加载多变量数据
    logger.info(f"加载测试数据: {args.data_path}")
    multivariate_data, target_columns = load_multivariate_data(
        args.data_path, args.target_columns
    )
    logger.info(f"数据形状: {multivariate_data.shape} (n_variates={len(target_columns)}, length={multivariate_data.shape[1]})")
    
    # 截取数据
    total_length = args.context_length + args.forecast_length
    if multivariate_data.shape[1] < total_length:
        logger.warning(f"数据长度不足，调整预测长度")
        args.forecast_length = multivariate_data.shape[1] - args.context_length
        total_length = multivariate_data.shape[1]
    
    # 分割历史和未来数据
    history_data = multivariate_data[:, :args.context_length]
    future_data = multivariate_data[:, args.context_length:args.context_length + args.forecast_length]
    
    logger.info(f"历史数据长度: {args.context_length}, 预测长度: {args.forecast_length}")
    
    # ========== 实验1: 多变量联合预测 ==========
    logger.info("=" * 60)
    logger.info("实验 1: 多变量联合预测")
    logger.info("=" * 60)
    
    # 构建多变量输入 (n_variates, history_length)
    inputs_multivariate = [history_data]  # 2D array: (n_variates, history_length)
    
    # 进行预测
    predictions_multivariate = pipeline.predict(
        inputs=inputs_multivariate,
        prediction_length=args.forecast_length,
    )
    
    # 处理预测结果
    # 多变量预测返回: list of tensors, 每个 tensor 形状为 (n_variates, n_quantiles, prediction_length)
    if isinstance(predictions_multivariate, list):
        pred_multi = predictions_multivariate[0]
    else:
        pred_multi = predictions_multivariate
    
    pred_multi = pred_multi.cpu().numpy()
    logger.info(f"多变量预测形状: {pred_multi.shape} (n_variates={pred_multi.shape[0]}, n_quantiles={pred_multi.shape[1]}, horizon={pred_multi.shape[2]})")
    
    # 计算每个变量的误差
    median_idx = pred_multi.shape[1] // 2
    multi_errors = []
    for i in range(len(target_columns)):
        actual_length = min(future_data.shape[1], args.forecast_length)
        error = np.mean(np.abs(pred_multi[i, median_idx][:actual_length] - future_data[i][:actual_length]))
        multi_errors.append(error)
        logger.info(f"  {target_columns[i]}: 误差 = {error:.4f}")
    
    avg_multi_error = np.mean(multi_errors)
    logger.info(f"多变量预测平均误差: {avg_multi_error:.4f}")
    
    # ========== 实验2: 单变量独立预测（对比）==========
    logger.info("=" * 60)
    logger.info("实验 2: 单变量独立预测（对比）")
    logger.info("=" * 60)
    
    univariate_preds = []
    uni_errors = []
    
    for i, col_name in enumerate(target_columns):
        # 单变量输入
        inputs_univariate = [history_data[i]]  # 1D array
        
        # 进行预测
        predictions_univariate = pipeline.predict(
            inputs=inputs_univariate,
            prediction_length=args.forecast_length,
        )
        
        # 处理预测结果
        if isinstance(predictions_univariate, list):
            pred_uni = predictions_univariate[0]
        else:
            pred_uni = predictions_univariate
        
        if pred_uni.dim() == 3:
            pred_uni = pred_uni.squeeze(0)
        
        pred_uni = pred_uni.cpu().numpy()
        univariate_preds.append(pred_uni)
        
        # 计算误差
        actual_length = min(future_data.shape[1], args.forecast_length)
        error = np.mean(np.abs(pred_uni[median_idx][:actual_length] - future_data[i][:actual_length]))
        uni_errors.append(error)
        logger.info(f"  {col_name}: 误差 = {error:.4f}")
    
    avg_uni_error = np.mean(uni_errors)
    logger.info(f"单变量预测平均误差: {avg_uni_error:.4f}")
    
    # 转换为 numpy 数组
    univariate_preds = np.stack(univariate_preds, axis=0)  # (n_variates, n_quantiles, horizon)
    
    # ========== 结果对比 ==========
    logger.info("=" * 60)
    logger.info("结果对比:")
    logger.info(f"  多变量预测平均误差: {avg_multi_error:.4f}")
    logger.info(f"  单变量预测平均误差: {avg_uni_error:.4f}")
    
    improvement = (avg_uni_error - avg_multi_error) / avg_uni_error * 100
    logger.info(f"  多变量带来的改进: {improvement:.2f}%")
    
    for i, col_name in enumerate(target_columns):
        var_improvement = (uni_errors[i] - multi_errors[i]) / uni_errors[i] * 100
        logger.info(f"  {col_name}: 多变量误差 {multi_errors[i]:.4f}, 单变量误差 {uni_errors[i]:.4f}, 改进 {var_improvement:.2f}%")
    
    logger.info("=" * 60)
    
    # 保存结果
    results = {
        "history_data": history_data,
        "future_data": future_data,
        "multivariate_predictions": pred_multi,
        "univariate_predictions": univariate_preds,
        "multivariate_errors": np.array(multi_errors),
        "univariate_errors": np.array(uni_errors),
        "avg_multivariate_error": avg_multi_error,
        "avg_univariate_error": avg_uni_error,
        "improvement_percent": improvement,
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
        "target_columns": target_columns,
    }
    
    np.savez(output_dir / "results.npz", **results)
    logger.info(f"结果已保存到: {output_dir / 'results.npz'}")
    
    # 可视化结果
    if not args.no_visualize:
        visualize_multivariate_forecast(
            history_data=history_data,
            future_data=future_data,
            forecast_preds=pred_multi,
            univariate_forecast_preds=univariate_preds,
            target_columns=target_columns,
            context_length=args.context_length,
            forecast_length=args.forecast_length,
            confidence_level=args.confidence_level,
            title=f"Chronos-2 Multivariate Forecast (P{int((1-args.confidence_level)/2*100)}-P{int((1+args.confidence_level)/2*100)})",
            save_path=str(output_dir / "visualization.png"),
        )
    
    logger.info("多变量预测演示完成!")


if __name__ == "__main__":
    main()