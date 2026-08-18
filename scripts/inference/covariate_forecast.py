"""
展示 Chronos-2 协变量预测功能

Chronos-2 支持两种协变量:
1. past_covariates: 过去协变量（历史数据）
2. future_covariates: 未来协变量（已知未来的数据，如天气预报、节假日等）

使用方法:
    python scripts/inference/covariate_forecast.py --model_path <模型路径> --data_path <数据路径>
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


def load_excel_data_with_covariates(
    file_path: str, 
    target_column: str = None,
    covariate_columns: list = None
) -> tuple[np.ndarray, dict, dict, str]:
    """
    从 Excel 文件加载时序数据和协变量
    
    Parameters
    ----------
    file_path : str
        Excel 文件路径
    target_column : str
        目标列名
    covariate_columns : list
        协变量列名列表
        
    Returns
    -------
    target : np.ndarray
        目标时序数据
    past_covariates : dict
        过去协变量（历史数据）
    future_covariates : dict
        未来协变量（预测期间的已知数据）
    target_column : str
        目标列名
    """
    df = pd.read_excel(file_path, header=6, skiprows=1)
    
    # 获取所有数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["ENG POS"]  # 排除的列
    
    # 选择目标列
    if target_column is None:
        available_cols = [c for c in numeric_cols if c not in exclude_cols]
        target_column = available_cols[0]
    
    # 选择协变量列
    if covariate_columns is None:
        # 默认选择除目标列外的其他列（最多3个）
        covariate_cols = [c for c in numeric_cols if c not in exclude_cols and c != target_column][:3]
    else:
        # 使用用户指定的协变量列
        covariate_cols = covariate_columns
    
    logger.info(f"目标列: {target_column}")
    logger.info(f"协变量列: {covariate_cols}")
    
    # 提取目标数据
    target = df[target_column].values.astype(np.float32)
    target = np.where(np.isnan(target), 0.0, target)
    
    # 提取协变量数据
    past_covariates = {}
    future_covariates = {}
    
    for col in covariate_cols:
        cov_data = df[col].values.astype(np.float32)
        cov_data = np.where(np.isnan(cov_data), 0.0, cov_data)
        
        # 过去协变量：历史数据
        past_covariates[col] = cov_data
        
        # 未来协变量：假设协变量在未来也是已知的（例如周期性数据）
        # 这里我们简单地复制最后一段数据作为未来协变量的示例
        # 实际应用中，未来协变量应该是真正已知的未来数据（如天气预报）
        future_covariates[col] = cov_data  # 完整数据，后续会截取预测长度
    
    return target, past_covariates, future_covariates, target_column, covariate_cols


def visualize_covariate_forecast(
    target_history: np.ndarray,
    target_future: np.ndarray,
    forecast_preds: np.ndarray,
    forecast_preds_no_cov: np.ndarray,
    past_covariates: dict,
    future_covariates: dict,
    context_length: int,
    forecast_length: int,
    title: str = "Covariate Forecast",
    save_path: str = None,
):
    """可视化协变量预测结果"""
    fig, axes = plt.subplots(2 + len(past_covariates), 1, figsize=(16, 4 * (2 + len(past_covariates))))
    
    # ========== 上图：预测对比 ==========
    ax1 = axes[0]
    
    # 绘制历史数据
    ax1.plot(range(context_length), target_history, "k-", linewidth=1.5, label="历史数据")
    
    # 绘制真实未来数据
    forecast_start = context_length
    ax1.plot(
        range(forecast_start, forecast_start + len(target_future)),
        target_future,
        "g--",
        linewidth=1.5,
        alpha=0.7,
        label="真实未来数据",
    )
    
    # 绘制有协变量的预测置信区间
    ax1.fill_between(
        range(forecast_start, forecast_start + forecast_length),
        forecast_preds[0][:forecast_length],
        forecast_preds[-1][:forecast_length],
        alpha=0.3,
        color="blue",
        label="有协变量预测置信区间",
    )
    
    # 绘制有协变量的预测中位数
    median_idx = len(forecast_preds) // 2
    ax1.plot(
        range(forecast_start, forecast_start + forecast_length),
        forecast_preds[median_idx][:forecast_length],
        "b-",
        linewidth=2,
        label="有协变量预测 (P50)",
    )
    
    # 绘制无协变量的预测置信区间
    ax1.fill_between(
        range(forecast_start, forecast_start + forecast_length),
        forecast_preds_no_cov[0][:forecast_length],
        forecast_preds_no_cov[-1][:forecast_length],
        alpha=0.2,
        color="orange",
        label="无协变量预测置信区间",
    )
    
    # 绘制无协变量的预测中位数
    ax1.plot(
        range(forecast_start, forecast_start + forecast_length),
        forecast_preds_no_cov[median_idx][:forecast_length],
        "orange",
        linewidth=2,
        linestyle="--",
        label="无协变量预测 (P50)",
    )
    
    # 添加分隔线
    ax1.axvline(x=context_length - 0.5, color="red", linestyle=":", linewidth=2, label="预测起点")
    
    ax1.set_title(f"{title} - 协变量预测对比")
    ax1.set_xlabel("时间步")
    ax1.set_ylabel("数值")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ========== 中图：预测误差对比 ==========
    ax2 = axes[1]
    
    # 计算预测误差
    actual_length = min(len(target_future), forecast_length)
    
    # 有协变量的误差
    with_cov_error = np.abs(forecast_preds[median_idx][:actual_length] - target_future[:actual_length])
    
    # 无协变量的误差
    no_cov_error = np.abs(forecast_preds_no_cov[median_idx][:actual_length] - target_future[:actual_length])
    
    ax2.plot(range(actual_length), with_cov_error, "b-", linewidth=1.5, label=f"有协变量误差 (平均: {np.mean(with_cov_error):.2f})")
    ax2.plot(range(actual_length), no_cov_error, "orange", linewidth=1.5, linestyle="--", label=f"无协变量误差 (平均: {np.mean(no_cov_error):.2f})")
    
    ax2.set_title("预测误差对比")
    ax2.set_xlabel("预测步数")
    ax2.set_ylabel("绝对误差")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # ========== 下图：协变量数据 ==========
    for i, (cov_name, cov_data) in enumerate(past_covariates.items()):
        ax = axes[2 + i]
        
        # 绘制协变量历史数据
        ax.plot(range(context_length), cov_data[:context_length], "k-", linewidth=1, label="历史协变量")
        
        # 绘制协变量未来数据（已知）
        if len(cov_data) > context_length:
            future_cov_length = min(len(cov_data) - context_length, forecast_length)
            ax.plot(
                range(context_length, context_length + future_cov_length),
                cov_data[context_length:context_length + future_cov_length],
                "g--",
                linewidth=1,
                alpha=0.7,
                label="已知未来协变量",
            )
        
        ax.axvline(x=context_length - 0.5, color="red", linestyle=":", linewidth=2)
        ax.set_title(f"协变量: {cov_name}")
        ax.set_xlabel("时间步")
        ax.set_ylabel("数值")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图表已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 协变量预测演示")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(WEIGHTS_DIR / "chronos-2"),
        help="模型路径（使用原始 Chronos-2 模型）",
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
        "--covariate_columns",
        type=str,
        nargs="+",
        default=None,
        help="协变量列名列表（多个列名用空格分隔）",
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
        output_dir = Path("results/covariate_forecast")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载模型
    logger.info(f"加载模型: {args.model_path}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda")
    logger.info(f"模型加载成功, 设备: {pipeline.model.device}")
    
    # 2. 加载数据和协变量
    logger.info(f"加载测试数据: {args.data_path}")
    target, past_covariates, future_covariates, target_column, covariate_cols = load_excel_data_with_covariates(
        args.data_path, args.target_column, args.covariate_columns
    )
    logger.info(f"数据长度: {len(target)}")
    
    # 截取数据
    total_length = args.context_length + args.forecast_length
    if len(target) < total_length:
        logger.warning(f"数据长度不足，调整预测长度")
        args.forecast_length = len(target) - args.context_length
        total_length = len(target)
    
    # 分割历史和未来数据
    target_history = target[:args.context_length]
    target_future = target[args.context_length:args.context_length + args.forecast_length]
    
    # 准备协变量数据
    past_cov_subset = {}
    future_cov_subset = {}
    
    for cov_name, cov_data in past_covariates.items():
        past_cov_subset[cov_name] = cov_data[:args.context_length]
        # 未来协变量：假设已知未来的协变量值
        future_cov_subset[cov_name] = cov_data[args.context_length:args.context_length + args.forecast_length]
    
    logger.info(f"历史数据长度: {args.context_length}, 预测长度: {args.forecast_length}")
    logger.info(f"协变量数量: {len(past_cov_subset)}")
    
    # ========== 实验1: 有协变量的预测 ==========
    logger.info("=" * 60)
    logger.info("实验 1: 使用协变量进行预测")
    logger.info("=" * 60)
    
    # 构建带协变量的输入
    inputs_with_cov = [
        {
            "target": target_history,
            "past_covariates": past_cov_subset,
            "future_covariates": future_cov_subset,
        }
    ]
    
    # 进行预测
    predictions_with_cov = pipeline.predict(
        inputs=inputs_with_cov,
        prediction_length=args.forecast_length,
    )
    
    # 处理预测结果
    if isinstance(predictions_with_cov, list):
        pred_with_cov = predictions_with_cov[0]
    else:
        pred_with_cov = predictions_with_cov
    
    if pred_with_cov.dim() == 3:
        pred_with_cov = pred_with_cov.squeeze(0)
    
    pred_with_cov = pred_with_cov.cpu().numpy()
    logger.info(f"有协变量预测形状: {pred_with_cov.shape}")
    
    # 计算误差
    median_idx = len(pred_with_cov) // 2
    with_cov_error = np.mean(np.abs(pred_with_cov[median_idx][:len(target_future)] - target_future))
    logger.info(f"有协变量预测误差: {with_cov_error:.4f}")
    
    # ========== 实验2: 无协变量的预测 ==========
    logger.info("=" * 60)
    logger.info("实验 2: 不使用协变量进行预测（对比）")
    logger.info("=" * 60)
    
    # 构建不带协变量的输入
    inputs_no_cov = [{"target": target_history}]
    
    # 进行预测
    predictions_no_cov = pipeline.predict(
        inputs=inputs_no_cov,
        prediction_length=args.forecast_length,
    )
    
    # 处理预测结果
    if isinstance(predictions_no_cov, list):
        pred_no_cov = predictions_no_cov[0]
    else:
        pred_no_cov = predictions_no_cov
    
    if pred_no_cov.dim() == 3:
        pred_no_cov = pred_no_cov.squeeze(0)
    
    pred_no_cov = pred_no_cov.cpu().numpy()
    logger.info(f"无协变量预测形状: {pred_no_cov.shape}")
    
    # 计算误差
    no_cov_error = np.mean(np.abs(pred_no_cov[median_idx][:len(target_future)] - target_future))
    logger.info(f"无协变量预测误差: {no_cov_error:.4f}")
    
    # ========== 结果对比 ==========
    logger.info("=" * 60)
    logger.info("结果对比:")
    logger.info(f"  有协变量预测误差: {with_cov_error:.4f}")
    logger.info(f"  无协变量预测误差: {no_cov_error:.4f}")
    improvement = (no_cov_error - with_cov_error) / no_cov_error * 100
    logger.info(f"  协变量带来的改进: {improvement:.2f}%")
    logger.info("=" * 60)
    
    # 保存结果
    results = {
        "target_history": target_history,
        "target_future": target_future,
        "predictions_with_covariates": pred_with_cov,
        "predictions_no_covariates": pred_no_cov,
        "past_covariates": past_cov_subset,
        "future_covariates": future_cov_subset,
        "with_cov_error": with_cov_error,
        "no_cov_error": no_cov_error,
        "improvement_percent": improvement,
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
        "covariate_names": list(past_cov_subset.keys()),
    }
    
    np.savez(output_dir / "results.npz", **results)
    logger.info(f"结果已保存到: {output_dir / 'results.npz'}")
    
    # 可视化结果
    if not args.no_visualize:
        visualize_covariate_forecast(
            target_history=target_history,
            target_future=target_future,
            forecast_preds=pred_with_cov,
            forecast_preds_no_cov=pred_no_cov,
            past_covariates=past_cov_subset,
            future_covariates=future_cov_subset,
            context_length=args.context_length,
            forecast_length=args.forecast_length,
            title=f"Chronos-2 Covariate Forecast - {target_column}",
            save_path=str(output_dir / "visualization.png"),
        )
    
    logger.info("协变量预测演示完成!")


if __name__ == "__main__":
    main()