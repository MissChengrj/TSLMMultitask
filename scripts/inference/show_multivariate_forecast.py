"""
多变量预测展示脚本

展示微调后模型对多个变量的预测结果，并与单变量预测进行对比。

使用方法:
    python scripts/inference/show_multivariate_forecast.py --model_path weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned --data_path data/038227/038227_1.xlsx
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


def load_multivariate_data(file_path: str, variable_columns: list):
    """加载多变量数据"""
    df = pd.read_excel(file_path, header=6, skiprows=1)
    
    data = {}
    for col in variable_columns:
        if col in df.columns:
            values = df[col].values.astype(np.float32)
            values = np.where(np.isnan(values), 0.0, values)
            data[col] = values
    
    return data


def run_multivariate_forecast(
    pipeline: Chronos2Pipeline,
    multivariate_data: np.ndarray,
    variable_names: list,
    context_length: int = 256,
    forecast_length: int = 64,
):
    """运行多变量预测"""
    
    n_variates = multivariate_data.shape[0]
    total_length = context_length + forecast_length
    
    if multivariate_data.shape[1] < total_length:
        logger.warning(f"数据长度 {multivariate_data.shape[1]} < 需要长度 {total_length}")
        forecast_length = multivariate_data.shape[1] - context_length
        if forecast_length <= 0:
            return None
    
    # 分割数据
    history = multivariate_data[:, :context_length]
    future = multivariate_data[:, context_length:context_length + forecast_length]
    
    logger.info(f"历史数据形状: {history.shape}")
    logger.info(f"未来数据形状: {future.shape}")
    
    # 多变量预测
    logger.info("执行多变量预测...")
    predictions_multi = pipeline.predict([history], prediction_length=forecast_length)
    pred_multi = predictions_multi[0].cpu().numpy()
    
    # 处理形状
    if pred_multi.ndim == 3 and pred_multi.shape[0] == 1:
        pred_multi = pred_multi.squeeze(0)
    
    logger.info(f"多变量预测输出形状: {pred_multi.shape}")
    
    # 单变量预测（对比）
    logger.info("执行单变量预测（对比）...")
    univariate_preds = []
    for i in range(n_variates):
        pred_uni = pipeline.predict([history[i]], prediction_length=forecast_length)[0].cpu().numpy()
        if pred_uni.ndim == 3 and pred_uni.shape[0] == 1:
            pred_uni = pred_uni.squeeze(0)
        univariate_preds.append(pred_uni)
        logger.info(f"变量 {i} 单变量预测形状: {pred_uni.shape}")
    
    univariate_preds = np.stack(univariate_preds, axis=0)
    logger.info(f"单变量预测堆叠形状: {univariate_preds.shape}")
    
    # 提取中位数和置信区间
    results = {
        "history": history,
        "future": future,
        "variable_names": variable_names,
        "context_length": context_length,
        "forecast_length": forecast_length,
        "multivariate_pred": pred_multi,
        "univariate_pred": univariate_preds,
    }
    
    # 计算每个变量的误差
    errors = []
    for i in range(n_variates):
        # 多变量预测误差
        if pred_multi.ndim == 2:
            # 如果输出是单变量形式，只计算第一个变量
            median_multi = pred_multi[pred_multi.shape[0] // 2]
            mae_multi = np.mean(np.abs(median_multi - future[0])) if i == 0 else 0
        else:
            median_multi = pred_multi[i, pred_multi.shape[1] // 2, :]
            mae_multi = np.mean(np.abs(median_multi - future[i]))
        
        # 单变量预测误差
        if univariate_preds.ndim == 3:
            median_uni = univariate_preds[i, univariate_preds.shape[1] // 2, :]
        else:
            median_uni = univariate_preds[univariate_preds.shape[0] // 2]
        mae_uni = np.mean(np.abs(median_uni - future[i]))
        
        errors.append({
            "variable": variable_names[i],
            "mae_multi": mae_multi,
            "mae_uni": mae_uni,
            "improvement": (mae_uni - mae_multi) / mae_uni * 100 if mae_uni > 0 else 0,
        })
        
        logger.info(f"变量 {variable_names[i]}: 多变量MAE={mae_multi:.4f}, 单变量MAE={mae_uni:.4f}, 改进={errors[-1]['improvement']:.2f}%")
    
    results["errors"] = errors
    
    return results


def visualize_multivariate_forecast(results: dict, output_dir: Path):
    """可视化多变量预测结果"""
    
    n_variates = len(results["variable_names"])
    context_length = results["context_length"]
    forecast_length = results["forecast_length"]
    
    # 创建大图表
    fig = plt.figure(figsize=(20, 5 * n_variates + 4))
    
    # 为每个变量创建子图
    for i, var_name in enumerate(results["variable_names"]):
        ax = fig.add_subplot(n_variates + 2, 1, i + 1)
        
        history = results["history"][i]
        future = results["future"][i]
        
        # 绘制历史数据
        ax.plot(range(context_length), history, 'k-', linewidth=1.5, label='历史数据')
        
        # 绘制真实未来数据
        ax.plot(range(context_length, context_length + forecast_length),
                future, 'g-', linewidth=2, label='真实未来', alpha=0.8)
        
        # 绘制多变量预测
        pred_multi = results["multivariate_pred"]
        if pred_multi.ndim == 2:
            # 单变量输出形式
            if i == 0:
                n_quantiles = pred_multi.shape[0]
                median_idx = n_quantiles // 2
                lower_idx = max(0, int((0.1 - 0.05) / 0.05))
                upper_idx = min(n_quantiles - 1, int((0.9 - 0.05) / 0.05))
                
                # 置信区间
                ax.fill_between(range(context_length, context_length + forecast_length),
                                pred_multi[lower_idx], pred_multi[upper_idx],
                                alpha=0.2, color='blue', label='置信区间 (P10-P90)')
                
                # 中位数
                ax.plot(range(context_length, context_length + forecast_length),
                        pred_multi[median_idx], 'b-', linewidth=2, label='多变量预测')
        else:
            # 多变量输出形式
            n_quantiles = pred_multi.shape[1]
            median_idx = n_quantiles // 2
            lower_idx = max(0, int((0.1 - 0.05) / 0.05))
            upper_idx = min(n_quantiles - 1, int((0.9 - 0.05) / 0.05))
            
            # 置信区间
            ax.fill_between(range(context_length, context_length + forecast_length),
                            pred_multi[i, lower_idx], pred_multi[i, upper_idx],
                            alpha=0.2, color='blue', label='置信区间 (P10-P90)')
            
            # 中位数
            ax.plot(range(context_length, context_length + forecast_length),
                    pred_multi[i, median_idx], 'b-', linewidth=2, label='多变量预测')
        
        # 绘制单变量预测（对比）
        pred_uni = results["univariate_pred"]
        if pred_uni.ndim == 3:
            n_quantiles_uni = pred_uni.shape[1]
            median_idx_uni = n_quantiles_uni // 2
            ax.plot(range(context_length, context_length + forecast_length),
                    pred_uni[i, median_idx_uni], 'r--', linewidth=2, 
                    label='单变量预测', alpha=0.7)
        else:
            median_idx_uni = pred_uni.shape[0] // 2
            ax.plot(range(context_length, context_length + forecast_length),
                    pred_uni[median_idx_uni], 'r--', linewidth=2, 
                    label='单变量预测', alpha=0.7)
        
        # 分割线
        ax.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        
        # 标题和标签
        error = results["errors"][i]
        title = f'{var_name} - 多变量MAE: {error["mae_multi"]:.2f}, 单变量MAE: {error["mae_uni"]:.2f}'
        if error["improvement"] > 0:
            title += f', 改进: {error["improvement"]:.1f}%'
        else:
            title += f', 改进: {error["improvement"]:.1f}%'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('值', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 误差对比柱状图
    ax_bar = fig.add_subplot(n_variates + 2, 1, n_variates + 1)
    
    var_names = [e["variable"] for e in results["errors"]]
    mae_multi = [e["mae_multi"] for e in results["errors"]]
    mae_uni = [e["mae_uni"] for e in results["errors"]]
    
    x = np.arange(len(var_names))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, mae_multi, width, label='多变量预测', color='steelblue', alpha=0.7)
    bars2 = ax_bar.bar(x + width/2, mae_uni, width, label='单变量预测', color='coral', alpha=0.7)
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(var_names, fontsize=10, rotation=15, ha='right')
    ax_bar.set_ylabel('MAE', fontsize=10)
    ax_bar.set_title('各变量预测误差对比', fontsize=12, fontweight='bold')
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 改进百分比柱状图
    ax_improve = fig.add_subplot(n_variates + 2, 1, n_variates + 2)
    
    improvements = [e["improvement"] for e in results["errors"]]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax_improve.bar(var_names, improvements, color=colors, alpha=0.7)
    ax_improve.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax_improve.set_xticklabels(var_names, fontsize=10, rotation=15, ha='right')
    ax_improve.set_ylabel('改进百分比 (%)', fontsize=10)
    ax_improve.set_title('多变量预测相对于单变量预测的改进', fontsize=12, fontweight='bold')
    ax_improve.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax_improve.text(bar.get_x() + bar.get_width()/2, height + 0.5 if height >= 0 else height - 1.5,
                        f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multivariate_forecast_results.png', dpi=150, bbox_inches='tight')
    logger.info(f"多变量预测结果图表已保存: {output_dir / 'multivariate_forecast_results.png'}")
    
    # 创建汇总表格
    fig_table, ax_table = plt.subplots(figsize=(14, 6))
    ax_table.axis('off')
    
    table_data = []
    for e in results["errors"]:
        improvement_str = f'{e["improvement"]:.1f}%'
        if e["improvement"] > 0:
            improvement_str = f'+{improvement_str}'
        table_data.append([
            e["variable"],
            f'{e["mae_multi"]:.4f}',
            f'{e["mae_uni"]:.4f}',
            improvement_str,
        ])
    
    # 添加汇总行
    avg_multi = np.mean([e["mae_multi"] for e in results["errors"]])
    avg_uni = np.mean([e["mae_uni"] for e in results["errors"]])
    avg_improve = (avg_uni - avg_multi) / avg_uni * 100 if avg_uni > 0 else 0
    
    table_data.append([
        '平均',
        f'{avg_multi:.4f}',
        f'{avg_uni:.4f}',
        f'{avg_improve:.1f}%',
    ])
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=['变量', '多变量MAE', '单变量MAE', '改进'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 设置最后一行（汇总）样式
    for i in range(4):
        table[(len(table_data), i)].set_facecolor('#E8F5E9')
        table[(len(table_data), i)].set_text_props(fontweight='bold')
    
    plt.title('多变量预测结果汇总', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'multivariate_summary_table.png', dpi=150, bbox_inches='tight')
    logger.info(f"汇总表格已保存: {output_dir / 'multivariate_summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="多变量预测展示")
    
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
        "--variables",
        type=str,
        nargs="+",
        default=["DEGT-CRUISE", "DEGT_SMOOTHED-CRUISE", "DEGT_D-CRUISE", "DEGT_D_SMOOTHED-CRUISE"],
        help="变量列表",
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
        default="results/multivariate_forecast",
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
    
    # 加载多变量数据
    logger.info(f"加载测试数据: {args.data_path}")
    data_dict = load_multivariate_data(args.data_path, args.variables)
    
    # 检查变量是否存在
    available_vars = [v for v in args.variables if v in data_dict]
    if len(available_vars) < len(args.variables):
        missing_vars = [v for v in args.variables if v not in data_dict]
        logger.warning(f"以下变量不存在: {missing_vars}")
    
    logger.info(f"可用变量: {available_vars}")
    
    # 构建多变量数组
    multivariate_data = np.stack([data_dict[v] for v in available_vars], axis=0)
    logger.info(f"多变量数据形状: {multivariate_data.shape}")
    
    # 运行多变量预测
    results = run_multivariate_forecast(
        pipeline,
        multivariate_data,
        available_vars,
        args.context_length,
        args.forecast_length,
    )
    
    if results is None:
        logger.error("预测失败")
        return
    
    # 可视化
    visualize_multivariate_forecast(results, output_dir)
    
    # 打印汇总
    logger.info("=" * 60)
    logger.info("多变量预测汇总")
    logger.info("=" * 60)
    for e in results["errors"]:
        logger.info(f"  {e['variable']}: 多变量MAE={e['mae_multi']:.4f}, 单变量MAE={e['mae_uni']:.4f}, 改进={e['improvement']:.2f}%")
    
    avg_multi = np.mean([e["mae_multi"] for e in results["errors"]])
    avg_uni = np.mean([e["mae_uni"] for e in results["errors"]])
    avg_improve = (avg_uni - avg_multi) / avg_uni * 100 if avg_uni > 0 else 0
    logger.info(f"  平均: 多变量MAE={avg_multi:.4f}, 单变量MAE={avg_uni:.4f}, 改进={avg_improve:.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()