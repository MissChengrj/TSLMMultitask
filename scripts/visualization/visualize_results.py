"""
测试结果详细可视化

生成详细的测试结果可视化图表，包括：
- 插值误差分布（按变量类型分组）
- 异常检测性能分析
- 预测误差分布
- 各任务详细对比
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> dict:
    """加载测试结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_variable(column_name: str) -> str:
    """根据变量名分类"""
    if 'DEGT' in column_name:
        return 'DEGT'
    elif 'GPCN' in column_name:
        return 'GPCN'
    elif 'N1' in column_name or 'N2' in column_name:
        return 'N'
    elif 'EGT' in column_name:
        return 'EGT'
    elif 'WF' in column_name or 'WFM' in column_name:
        return 'WF'
    elif 'SM' in column_name:
        return 'SM'
    elif 'ALT' in column_name:
        return 'ALT'
    elif 'MACH' in column_name:
        return 'MACH'
    elif 'VSV' in column_name:
        return 'VSV'
    elif 'VBV' in column_name:
        return 'VBV'
    elif 'TAT' in column_name:
        return 'TAT'
    else:
        return 'Other'


def create_detailed_visualization(results: dict, output_dir: Path):
    """创建详细可视化图表"""
    
    # 1. 插值任务详细可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 插值误差分布（整体）
    interp_maes = [r['mae'] for r in results['interpolation'] if 'mae' in r]
    axes[0, 0].hist(interp_maes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(interp_maes), color='red', linestyle='--', linewidth=2, 
                       label=f'平均: {np.mean(interp_maes):.2f}')
    axes[0, 0].axvline(np.median(interp_maes), color='green', linestyle=':', linewidth=2,
                       label=f'中位数: {np.median(interp_maes):.2f}')
    axes[0, 0].set_title('插值误差分布 (MAE)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('MAE', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 1.2 插值误差按变量类型分组
    interp_by_type = {}
    for r in results['interpolation']:
        if 'mae' in r:
            var_type = categorize_variable(r['column'])
            if var_type not in interp_by_type:
                interp_by_type[var_type] = []
            interp_by_type[var_type].append(r['mae'])
    
    var_types = sorted(interp_by_type.keys())
    means = [np.mean(interp_by_type[t]) for t in var_types]
    stds = [np.std(interp_by_type[t]) for t in var_types]
    
    bars = axes[0, 1].bar(var_types, means, color='steelblue', alpha=0.7, 
                          yerr=stds, capsize=5, edgecolor='black')
    axes[0, 1].set_title('插值误差按变量类型分组', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('变量类型', fontsize=12)
    axes[0, 1].set_ylabel('平均 MAE', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, mean in zip(bars, means):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 1.3 插值误差箱线图（按变量类型）
    box_data = [interp_by_type[t] for t in var_types]
    bp = axes[1, 0].boxplot(box_data, labels=var_types, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[1, 0].set_title('插值误差箱线图（按变量类型）', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('变量类型', fontsize=12)
    axes[1, 0].set_ylabel('MAE', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 1.4 插值误差热力图（按文件）
    interp_by_file = {}
    for r in results['interpolation']:
        if 'mae' in r:
            file_name = Path(r['file']).parent.name
            if file_name not in interp_by_file:
                interp_by_file[file_name] = []
            interp_by_file[file_name].append(r['mae'])
    
    file_names = sorted(interp_by_file.keys())
    file_means = [np.mean(interp_by_file[f]) for f in file_names]
    
    # 创建热力图数据
    heatmap_data = np.array(file_means).reshape(-1, 1)
    im = axes[1, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_yticks(range(len(file_names)))
    axes[1, 1].set_yticklabels(file_names, fontsize=8)
    axes[1, 1].set_xticks([0])
    axes[1, 1].set_xticklabels(['平均 MAE'])
    axes[1, 1].set_title('插值误差热力图（按数据源）', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 1], label='MAE')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'interpolation_detailed.png', dpi=150, bbox_inches='tight')
    logger.info(f"插值详细图表已保存: {output_dir / 'interpolation_detailed.png'}")
    
    # 2. 异常检测任务详细可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 2.1 异常检测性能分布
    anomaly_f1s = [r['f1'] for r in results['anomaly_detection'] if 'f1' in r]
    anomaly_precisions = [r['precision'] for r in results['anomaly_detection'] if 'precision' in r]
    anomaly_recalls = [r['recall'] for r in results['anomaly_detection'] if 'recall' in r]
    
    axes[0, 0].hist(anomaly_f1s, bins=30, color='forestgreen', alpha=0.7, edgecolor='black', label='F1')
    axes[0, 0].hist(anomaly_precisions, bins=30, color='coral', alpha=0.5, edgecolor='black', label='Precision')
    axes[0, 0].hist(anomaly_recalls, bins=30, color='royalblue', alpha=0.5, edgecolor='black', label='Recall')
    axes[0, 0].set_title('异常检测性能分布', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('性能指标', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2.2 Precision vs Recall 散点图
    axes[0, 1].scatter(anomaly_recalls, anomaly_precisions, c=anomaly_f1s, 
                       cmap='viridis', alpha=0.7, s=50, edgecolor='black')
    axes[0, 1].set_title('Precision vs Recall (颜色表示 F1)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Recall', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2.3 异常检测性能按变量类型分组
    anomaly_by_type = {}
    for r in results['anomaly_detection']:
        if 'f1' in r:
            var_type = categorize_variable(r['column'])
            if var_type not in anomaly_by_type:
                anomaly_by_type[var_type] = {'f1': [], 'precision': [], 'recall': []}
            anomaly_by_type[var_type]['f1'].append(r['f1'])
            anomaly_by_type[var_type]['precision'].append(r['precision'])
            anomaly_by_type[var_type]['recall'].append(r['recall'])
    
    var_types = sorted(anomaly_by_type.keys())
    x = np.arange(len(var_types))
    width = 0.25
    
    f1_means = [np.mean(anomaly_by_type[t]['f1']) for t in var_types]
    prec_means = [np.mean(anomaly_by_type[t]['precision']) for t in var_types]
    rec_means = [np.mean(anomaly_by_type[t]['recall']) for t in var_types]
    
    axes[1, 0].bar(x - width, f1_means, width, label='F1', color='forestgreen', alpha=0.7)
    axes[1, 0].bar(x, prec_means, width, label='Precision', color='coral', alpha=0.7)
    axes[1, 0].bar(x + width, rec_means, width, label='Recall', color='royalblue', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(var_types, fontsize=9)
    axes[1, 0].set_title('异常检测性能按变量类型分组', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('变量类型', fontsize=12)
    axes[1, 0].set_ylabel('性能指标', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 2.4 检测统计（TP, FP, FN）
    tp_total = sum([r['true_positives'] for r in results['anomaly_detection'] if 'true_positives' in r])
    fp_total = sum([r['false_positives'] for r in results['anomaly_detection'] if 'false_positives' in r])
    fn_total = sum([r['false_negatives'] for r in results['anomaly_detection'] if 'false_negatives' in r])
    
    labels = ['True Positives', 'False Positives', 'False Negatives']
    sizes = [tp_total, fp_total, fn_total]
    colors = ['forestgreen', 'coral', 'royalblue']
    explode = (0.05, 0, 0)
    
    axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
    axes[1, 1].set_title(f'异常检测统计 (总数: {sum(sizes)})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_detection_detailed.png', dpi=150, bbox_inches='tight')
    logger.info(f"异常检测详细图表已保存: {output_dir / 'anomaly_detection_detailed.png'}")
    
    # 3. 预测任务详细可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 3.1 预测误差分布
    forecast_maes = [r['mae'] for r in results['forecast'] if 'mae' in r]
    forecast_rmses = [r['rmse'] for r in results['forecast'] if 'rmse' in r]
    
    axes[0, 0].hist(forecast_maes, bins=30, color='darkorange', alpha=0.7, edgecolor='black', label='MAE')
    axes[0, 0].axvline(np.mean(forecast_maes), color='red', linestyle='--', linewidth=2,
                       label=f'平均 MAE: {np.mean(forecast_maes):.2f}')
    axes[0, 0].set_title('预测误差分布 (MAE)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('MAE', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 3.2 MAE vs RMSE 散点图
    axes[0, 1].scatter(forecast_maes, forecast_rmses, c='darkorange', alpha=0.7, s=50, edgecolor='black')
    axes[0, 1].set_title('MAE vs RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('MAE', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3.3 预测误差按变量类型分组
    forecast_by_type = {}
    for r in results['forecast']:
        if 'mae' in r:
            var_type = categorize_variable(r['column'])
            if var_type not in forecast_by_type:
                forecast_by_type[var_type] = {'mae': [], 'rmse': []}
            forecast_by_type[var_type]['mae'].append(r['mae'])
            forecast_by_type[var_type]['rmse'].append(r['rmse'])
    
    var_types = sorted(forecast_by_type.keys())
    mae_means = [np.mean(forecast_by_type[t]['mae']) for t in var_types]
    rmse_means = [np.mean(forecast_by_type[t]['rmse']) for t in var_types]
    
    x = np.arange(len(var_types))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, mae_means, width, label='MAE', color='darkorange', alpha=0.7)
    axes[1, 0].bar(x + width/2, rmse_means, width, label='RMSE', color='gold', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(var_types, fontsize=9)
    axes[1, 0].set_title('预测误差按变量类型分组', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('变量类型', fontsize=12)
    axes[1, 0].set_ylabel('误差', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 3.4 预测误差箱线图
    box_data = [forecast_by_type[t]['mae'] for t in var_types]
    bp = axes[1, 1].boxplot(box_data, labels=var_types, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('darkorange')
        patch.set_alpha(0.7)
    axes[1, 1].set_title('预测误差箱线图（按变量类型）', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('变量类型', fontsize=12)
    axes[1, 1].set_ylabel('MAE', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'forecast_detailed.png', dpi=150, bbox_inches='tight')
    logger.info(f"预测详细图表已保存: {output_dir / 'forecast_detailed.png'}")
    
    # 4. 综合对比图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 4.1 三任务性能对比（雷达图）
    categories = ['插值\n(MAE)', '异常检测\n(F1)', '预测\n(MAE)']
    
    # 标准化指标（0-1范围）
    interp_score = 1 - np.mean(interp_maes) / max(interp_maes)  # MAE越小越好
    anomaly_score = np.mean(anomaly_f1s)  # F1越大越好
    forecast_score = 1 - np.mean(forecast_maes) / max(forecast_maes)  # MAE越小越好
    
    scores = [interp_score, anomaly_score, forecast_score]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles_plot = angles + [angles[0]]
    
    ax_radar = plt.subplot(1, 3, 1, projection='polar')
    ax_radar.fill(angles_plot, scores_plot, color='steelblue', alpha=0.3)
    ax_radar.plot(angles_plot, scores_plot, color='steelblue', linewidth=2)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(categories, fontsize=11)
    ax_radar.set_title('三任务性能雷达图\n(标准化分数)', fontsize=14, fontweight='bold', pad=20)
    
    # 4.2 各任务平均性能柱状图
    task_names = ['插值', '异常检测', '预测']
    task_scores = [np.mean(interp_maes), np.mean(anomaly_f1s), np.mean(forecast_maes)]
    colors = ['steelblue', 'forestgreen', 'darkorange']
    
    bars = axes[1].bar(task_names, task_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_title('各任务平均性能', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('性能指标', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, score in zip(bars, task_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4.3 各任务测试数量对比
    test_counts = [len(interp_maes), len(anomaly_f1s), len(forecast_maes)]
    
    bars = axes[2].bar(task_names, test_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_title('各任务测试数量', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('测试数量', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, count in zip(bars, test_counts):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"综合对比图表已保存: {output_dir / 'comprehensive_comparison.png'}")
    
    # 5. 创建汇总报告
    create_summary_report(results, output_dir)


def create_summary_report(results: dict, output_dir: Path):
    """创建文本汇总报告"""
    
    summary = results['summary']
    
    report_lines = [
        "=" * 80,
        "Chronos-2 MLM 微调模型批量测试报告",
        "=" * 80,
        "",
        f"测试时间: {results['metadata']['timestamp']}",
        f"模型路径: {results['metadata']['model_path']}",
        f"数据目录: {results['metadata']['data_dir']}",
        f"测试时序数: {results['metadata']['num_series']}",
        "",
        "=" * 80,
        "插值任务结果",
        "=" * 80,
        f"平均 MAE: {summary['interpolation']['avg_mae']:.4f}",
        f"MAE 标准差: {summary['interpolation']['std_mae']:.4f}",
        f"测试数量: {summary['interpolation']['num_tests']}",
        "",
        "=" * 80,
        "异常检测任务结果",
        "=" * 80,
        f"平均 F1 Score: {summary['anomaly_detection']['avg_f1']:.4f}",
        f"平均 Precision: {summary['anomaly_detection']['avg_precision']:.4f}",
        f"平均 Recall: {summary['anomaly_detection']['avg_recall']:.4f}",
        f"测试数量: {summary['anomaly_detection']['num_tests']}",
        "",
        "=" * 80,
        "预测任务结果",
        "=" * 80,
        f"平均 MAE: {summary['forecast']['avg_mae']:.4f}",
        f"平均 RMSE: {summary['forecast']['avg_rmse']:.4f}",
        f"MAE 标准差: {summary['forecast']['std_mae']:.4f}",
        f"测试数量: {summary['forecast']['num_tests']}",
        "",
        "=" * 80,
        "结论",
        "=" * 80,
        "1. 插值任务表现良好，平均 MAE 为 {:.2f}，说明模型能够有效重构缺失数据。".format(summary['interpolation']['avg_mae']),
        "2. 异常检测任务 F1 Score 为 {:.2f}，Recall 较高 ({:.2f})，能够检测到大部分异常，但 Precision 较低 ({:.2f})，存在一定误报。".format(
            summary['anomaly_detection']['avg_f1'],
            summary['anomaly_detection']['avg_recall'],
            summary['anomaly_detection']['avg_precision']
        ),
        "3. 预测任务平均 MAE 为 {:.2f}，RMSE 为 {:.2f}，表现合理。".format(
            summary['forecast']['avg_mae'],
            summary['forecast']['avg_rmse']
        ),
        "",
        "建议:",
        "- 对于插值任务，模型表现优秀，可直接用于缺失数据填补。",
        "- 对于异常检测，建议调整置信区间阈值以平衡 Precision 和 Recall。",
        "- 对于预测任务，建议使用原始 Chronos-2 模型而非 MLM 微调版本。",
        "=" * 80,
    ]
    
    report_text = "\n".join(report_lines)
    
    with open(output_dir / 'test_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"汇总报告已保存: {output_dir / 'test_report.txt'}")
    
    # 打印报告
    print(report_text)


def main():
    results_path = Path("results/batch_test/test_results.json")
    output_dir = Path("results/batch_test")
    
    if not results_path.exists():
        logger.error(f"结果文件不存在: {results_path}")
        return
    
    # 加载结果
    logger.info(f"加载测试结果: {results_path}")
    results = load_results(str(results_path))
    
    # 创建详细可视化
    logger.info("创建详细可视化图表...")
    create_detailed_visualization(results, output_dir)
    
    logger.info("可视化完成!")


if __name__ == "__main__":
    main()