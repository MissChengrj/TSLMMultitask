"""
原始模型与微调模型对比测试

使用原始 Chronos-2 模型和微调后的模型对正弦函数进行预测对比。

使用方法:
    python scripts/testing/original_vs_finetuned_test.py --original_model weights/chronos2-base --finetuned_model weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned
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


def run_forecast(pipeline, data, context_length: int, forecast_length: int):
    """运行预测"""
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


def visualize_comparison(results_original, results_finetuned, output_dir: Path, noise_std: float):
    """可视化对比结果"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 纯净正弦 - 原始模型预测
    ax1 = fig.add_subplot(4, 2, 1)
    forecast = results_original["clean"]
    if forecast:
        context_length = len(forecast["history"])
        ax1.plot(range(context_length), forecast["history"], 'k-', linewidth=1.5, label='历史数据')
        ax1.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["future"], 'g-', linewidth=2, label='真实未来')
        
        n_quantiles = forecast["predicted"].shape[0]
        ax1.fill_between(range(context_length, context_length + len(forecast["future"])),
                         forecast["predicted"][0], forecast["predicted"][-1],
                         alpha=0.3, color='blue', label='置信区间')
        ax1.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["median_pred"], 'b-', linewidth=2, label='预测中位数')
        
        ax1.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax1.set_title(f'纯净正弦 - 原始模型预测\nMAE: {forecast["mae"]:.4f}, RMSE: {forecast["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('值')
    
    # 2. 纯净正弦 - 微调模型预测
    ax2 = fig.add_subplot(4, 2, 2)
    forecast = results_finetuned["clean"]
    if forecast:
        context_length = len(forecast["history"])
        ax2.plot(range(context_length), forecast["history"], 'k-', linewidth=1.5, label='历史数据')
        ax2.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["future"], 'g-', linewidth=2, label='真实未来')
        
        n_quantiles = forecast["predicted"].shape[0]
        ax2.fill_between(range(context_length, context_length + len(forecast["future"])),
                         forecast["predicted"][0], forecast["predicted"][-1],
                         alpha=0.3, color='orange', label='置信区间')
        ax2.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["median_pred"], 'orange', linewidth=2, label='预测中位数')
        
        ax2.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax2.set_title(f'纯净正弦 - 微调模型预测\nMAE: {forecast["mae"]:.4f}, RMSE: {forecast["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('值')
    
    # 3. 噪声正弦 - 原始模型预测
    ax3 = fig.add_subplot(4, 2, 3)
    forecast = results_original["noisy"]
    if forecast:
        context_length = len(forecast["history"])
        ax3.plot(range(context_length), forecast["history"], 'k-', linewidth=1.5, label='历史数据（含噪声）')
        ax3.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["future"], 'g-', linewidth=2, label='真实未来（含噪声）')
        
        n_quantiles = forecast["predicted"].shape[0]
        ax3.fill_between(range(context_length, context_length + len(forecast["future"])),
                         forecast["predicted"][0], forecast["predicted"][-1],
                         alpha=0.3, color='blue', label='置信区间')
        ax3.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["median_pred"], 'b-', linewidth=2, label='预测中位数')
        
        ax3.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax3.set_title(f'噪声正弦 (σ={noise_std}) - 原始模型预测\nMAE: {forecast["mae"]:.4f}, RMSE: {forecast["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('值')
    
    # 4. 噪声正弦 - 微调模型预测
    ax4 = fig.add_subplot(4, 2, 4)
    forecast = results_finetuned["noisy"]
    if forecast:
        context_length = len(forecast["history"])
        ax4.plot(range(context_length), forecast["history"], 'k-', linewidth=1.5, label='历史数据（含噪声）')
        ax4.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["future"], 'g-', linewidth=2, label='真实未来（含噪声）')
        
        n_quantiles = forecast["predicted"].shape[0]
        ax4.fill_between(range(context_length, context_length + len(forecast["future"])),
                         forecast["predicted"][0], forecast["predicted"][-1],
                         alpha=0.3, color='orange', label='置信区间')
        ax4.plot(range(context_length, context_length + len(forecast["future"])),
                 forecast["median_pred"], 'orange', linewidth=2, label='预测中位数')
        
        ax4.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax4.set_title(f'噪声正弦 (σ={noise_std}) - 微调模型预测\nMAE: {forecast["mae"]:.4f}, RMSE: {forecast["rmse"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('值')
    
    # 5. 纯净正弦 - 预测对比
    ax5 = fig.add_subplot(4, 2, 5)
    forecast_orig = results_original["clean"]
    forecast_finetuned = results_finetuned["clean"]
    
    if forecast_orig and forecast_finetuned:
        context_length = len(forecast_orig["history"])
        ax5.plot(range(context_length, context_length + len(forecast_orig["future"])),
                 forecast_orig["future"], 'g-', linewidth=2, label='真实未来')
        ax5.plot(range(context_length, context_length + len(forecast_orig["future"])),
                 forecast_orig["median_pred"], 'b-', linewidth=2, label='原始模型')
        ax5.plot(range(context_length, context_length + len(forecast_finetuned["future"])),
                 forecast_finetuned["median_pred"], 'orange', linewidth=2, label='微调模型')
        
        ax5.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax5.set_title(f'纯净正弦 - 预测对比\n原始MAE: {forecast_orig["mae"]:.4f}, 微调MAE: {forecast_finetuned["mae"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlabel('时间步')
        ax5.set_ylabel('值')
    
    # 6. 噪声正弦 - 预测对比
    ax6 = fig.add_subplot(4, 2, 6)
    forecast_orig = results_original["noisy"]
    forecast_finetuned = results_finetuned["noisy"]
    
    if forecast_orig and forecast_finetuned:
        context_length = len(forecast_orig["history"])
        ax6.plot(range(context_length, context_length + len(forecast_orig["future"])),
                 forecast_orig["future"], 'g-', linewidth=2, label='真实未来（含噪声）')
        ax6.plot(range(context_length, context_length + len(forecast_orig["future"])),
                 forecast_orig["median_pred"], 'b-', linewidth=2, label='原始模型')
        ax6.plot(range(context_length, context_length + len(forecast_finetuned["future"])),
                 forecast_finetuned["median_pred"], 'orange', linewidth=2, label='微调模型')
        
        ax6.axvline(x=context_length - 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax6.set_title(f'噪声正弦 - 预测对比\n原始MAE: {forecast_orig["mae"]:.4f}, 微调MAE: {forecast_finetuned["mae"]:.4f}',
                      fontsize=12, fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel('时间步')
        ax6.set_ylabel('值')
    
    # 7. MAE 对比柱状图
    ax7 = fig.add_subplot(4, 2, 7)
    
    mae_clean_orig = results_original["clean"]["mae"] if results_original["clean"] else 0
    mae_clean_finetuned = results_finetuned["clean"]["mae"] if results_finetuned["clean"] else 0
    mae_noisy_orig = results_original["noisy"]["mae"] if results_original["noisy"] else 0
    mae_noisy_finetuned = results_finetuned["noisy"]["mae"] if results_finetuned["noisy"] else 0
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, [mae_clean_orig, mae_noisy_orig], width, 
                    label='原始模型', color='steelblue', alpha=0.7)
    bars2 = ax7.bar(x + width/2, [mae_clean_finetuned, mae_noisy_finetuned], width,
                    label='微调模型', color='coral', alpha=0.7)
    
    ax7.set_xticks(x)
    ax7.set_xticklabels(['纯净正弦', '噪声正弦'])
    ax7.set_title('MAE 对比', fontsize=12, fontweight='bold')
    ax7.set_ylabel('MAE')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, [mae_clean_orig, mae_noisy_orig]):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, [mae_clean_finetuned, mae_noisy_finetuned]):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 8. RMSE 对比柱状图
    ax8 = fig.add_subplot(4, 2, 8)
    
    rmse_clean_orig = results_original["clean"]["rmse"] if results_original["clean"] else 0
    rmse_clean_finetuned = results_finetuned["clean"]["rmse"] if results_finetuned["clean"] else 0
    rmse_noisy_orig = results_original["noisy"]["rmse"] if results_original["noisy"] else 0
    rmse_noisy_finetuned = results_finetuned["noisy"]["rmse"] if results_finetuned["noisy"] else 0
    
    bars1 = ax8.bar(x - width/2, [rmse_clean_orig, rmse_noisy_orig], width,
                    label='原始模型', color='steelblue', alpha=0.7)
    bars2 = ax8.bar(x + width/2, [rmse_clean_finetuned, rmse_noisy_finetuned], width,
                    label='微调模型', color='coral', alpha=0.7)
    
    ax8.set_xticks(x)
    ax8.set_xticklabels(['纯净正弦', '噪声正弦'])
    ax8.set_title('RMSE 对比', fontsize=12, fontweight='bold')
    ax8.set_ylabel('RMSE')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, [rmse_clean_orig, rmse_noisy_orig]):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, [rmse_clean_finetuned, rmse_noisy_finetuned]):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'original_vs_finetuned_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"对比结果图表已保存: {output_dir / 'original_vs_finetuned_comparison.png'}")
    
    # 创建汇总表格
    fig_table, ax_table = plt.subplots(figsize=(12, 6))
    ax_table.axis('off')
    
    # 计算改进百分比
    improvement_clean_mae = (mae_clean_orig - mae_clean_finetuned) / mae_clean_orig * 100 if mae_clean_orig > 0 else 0
    improvement_noisy_mae = (mae_noisy_orig - mae_noisy_finetuned) / mae_noisy_orig * 100 if mae_noisy_orig > 0 else 0
    improvement_clean_rmse = (rmse_clean_orig - rmse_clean_finetuned) / rmse_clean_orig * 100 if rmse_clean_orig > 0 else 0
    improvement_noisy_rmse = (rmse_noisy_orig - rmse_noisy_finetuned) / rmse_noisy_orig * 100 if rmse_noisy_orig > 0 else 0
    
    table_data = [
        ['纯净正弦 MAE', f'{mae_clean_orig:.4f}', f'{mae_clean_finetuned:.4f}', f'{improvement_clean_mae:.1f}%'],
        ['纯净正弦 RMSE', f'{rmse_clean_orig:.4f}', f'{rmse_clean_finetuned:.4f}', f'{improvement_clean_rmse:.1f}%'],
        ['噪声正弦 MAE', f'{mae_noisy_orig:.4f}', f'{mae_noisy_finetuned:.4f}', f'{improvement_noisy_mae:.1f}%'],
        ['噪声正弦 RMSE', f'{rmse_noisy_orig:.4f}', f'{rmse_noisy_finetuned:.4f}', f'{improvement_noisy_rmse:.1f}%'],
    ]
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=['指标', '原始模型', '微调模型', '改进'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title('原始模型 vs 微调模型预测对比', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'comparison_summary_table.png', dpi=150, bbox_inches='tight')
    logger.info(f"汇总表格已保存: {output_dir / 'comparison_summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="原始模型与微调模型对比测试")
    
    parser.add_argument(
        "--original_model",
        type=str,
        default="weights/chronos-2",
        help="原始模型路径",
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="weights/mlm_batch_finetuned/2026-06-22_21-29-01/mlm-lora-batch-finetuned",
        help="微调模型路径",
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
        "--noise_std",
        type=float,
        default=2.0,
        help="噪声标准差",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/original_vs_finetuned",
        help="输出目录",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载原始模型
    logger.info(f"加载原始模型: {args.original_model}")
    pipeline_original = Chronos2Pipeline.from_pretrained(args.original_model, device_map="cuda")
    logger.info(f"原始模型加载成功, 设备: {pipeline_original.model.device}")
    
    # 加载微调模型
    logger.info(f"加载微调模型: {args.finetuned_model}")
    pipeline_finetuned = Chronos2Pipeline.from_pretrained(args.finetuned_model, device_map="cuda")
    logger.info(f"微调模型加载成功, 设备: {pipeline_finetuned.model.device}")
    
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
    
    results_original = {}
    results_finetuned = {}
    
    # ========== 原始模型测试 ==========
    logger.info("=" * 60)
    logger.info("原始模型预测测试")
    logger.info("=" * 60)
    
    # 纯净正弦预测
    logger.info("纯净正弦预测...")
    results_original["clean"] = run_forecast(
        pipeline_original, sine_clean, args.context_length, args.forecast_length
    )
    if results_original["clean"]:
        logger.info(f"MAE: {results_original['clean']['mae']:.4f}")
        logger.info(f"RMSE: {results_original['clean']['rmse']:.4f}")
    
    # 噪声正弦预测
    logger.info("噪声正弦预测...")
    results_original["noisy"] = run_forecast(
        pipeline_original, sine_noisy, args.context_length, args.forecast_length
    )
    if results_original["noisy"]:
        logger.info(f"MAE: {results_original['noisy']['mae']:.4f}")
        logger.info(f"RMSE: {results_original['noisy']['rmse']:.4f}")
    
    # ========== 微调模型测试 ==========
    logger.info("=" * 60)
    logger.info("微调模型预测测试")
    logger.info("=" * 60)
    
    # 纯净正弦预测
    logger.info("纯净正弦预测...")
    results_finetuned["clean"] = run_forecast(
        pipeline_finetuned, sine_clean, args.context_length, args.forecast_length
    )
    if results_finetuned["clean"]:
        logger.info(f"MAE: {results_finetuned['clean']['mae']:.4f}")
        logger.info(f"RMSE: {results_finetuned['clean']['rmse']:.4f}")
    
    # 噪声正弦预测
    logger.info("噪声正弦预测...")
    results_finetuned["noisy"] = run_forecast(
        pipeline_finetuned, sine_noisy, args.context_length, args.forecast_length
    )
    if results_finetuned["noisy"]:
        logger.info(f"MAE: {results_finetuned['noisy']['mae']:.4f}")
        logger.info(f"RMSE: {results_finetuned['noisy']['rmse']:.4f}")
    
    # 可视化对比
    visualize_comparison(results_original, results_finetuned, output_dir, args.noise_std)
    
    # 打印汇总
    logger.info("=" * 60)
    logger.info("对比汇总")
    logger.info("=" * 60)
    
    if results_original["clean"] and results_finetuned["clean"]:
        improvement = (results_original["clean"]["mae"] - results_finetuned["clean"]["mae"]) / results_original["clean"]["mae"] * 100
        logger.info(f"纯净正弦: 原始MAE={results_original['clean']['mae']:.4f}, 微调MAE={results_finetuned['clean']['mae']:.4f}, 改进={improvement:.2f}%")
    
    if results_original["noisy"] and results_finetuned["noisy"]:
        improvement = (results_original["noisy"]["mae"] - results_finetuned["noisy"]["mae"]) / results_original["noisy"]["mae"] * 100
        logger.info(f"噪声正弦: 原始MAE={results_original['noisy']['mae']:.4f}, 微调MAE={results_finetuned['noisy']['mae']:.4f}, 改进={improvement:.2f}%")
    
    logger.info("=" * 60)
    logger.info("测试完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()