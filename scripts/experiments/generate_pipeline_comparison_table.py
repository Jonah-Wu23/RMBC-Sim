#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_pipeline_comparison_table.py
======================================
生成 Pipeline 级消融的系统对照表

产出：
1. 完整的系统对照表（Markdown + CSV）
2. 强调"校准与验证解耦"的协议一致性
3. 可视化对比图（IEEE 格式）

输入：
- full_metrics.csv（由 compute_pipeline_metrics.py 生成）

输出：
- pipeline_comparison_table.md
- pipeline_comparison_table.csv
- pipeline_comparison_figure.png

Author: RCMDT Project
Date: 2026-01-11
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# IEEE Paper Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 8


# =============================================================================
# 配置名称映射
# =============================================================================

CONFIG_NAMES = {
    "A0": "Zero-shot DT",
    "A2": "Only L2 (DA)",
    "A3": "Only L1 (BO)",
    "A4": "RCMDT (L1+L2)"
}

CONFIG_DESCRIPTIONS = {
    "A0": "不校准不同化",
    "A2": "固定 stop 参数，只做状态同化",
    "A3": "只调 stop 参数，不做 L2",
    "A4": "L1+L2 完整系统"
}


# =============================================================================
# 表格生成
# =============================================================================

def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成摘要统计（每个配置的均值±标准差）
    
    Returns:
        DataFrame with config_id, config_name, mean±std for each metric
    """
    summary_rows = []
    
    for config_id in ["A0", "A2", "A3", "A4"]:
        config_data = df[df["config_id"] == config_id]
        
        if len(config_data) == 0:
            continue
        
        row = {
            "config_id": config_id,
            "config_name": CONFIG_NAMES.get(config_id, config_id),
            "description": CONFIG_DESCRIPTIONS.get(config_id, ""),
            "n_runs": len(config_data)
        }
        
        # 计算每个指标的 mean ± std
        base_metrics = [
            "rmse_speed", "rmse_tt",
            "ks_speed", "ks_tt",
            "worst_ks_speed", "worst_ks_tt",
            "next_day_ks_speed", "next_day_ks_tt"
        ]
        optional_metrics = [
            "W1_speed", "W1_TT",
            "worst_W1_speed", "worst_W1_TT"
        ]
        metrics = base_metrics + [m for m in optional_metrics if m in df.columns]

        for metric in metrics:
            if metric not in config_data.columns:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                continue
            values = config_data[metric].dropna()
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std() if len(values) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def generate_markdown_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    """生成 Markdown 格式的系统对照表"""
    has_w1_speed = "W1_speed_mean" in summary_df.columns
    has_w1_tt = "W1_TT_mean" in summary_df.columns
    has_worst_w1_speed = "worst_W1_speed_mean" in summary_df.columns
    has_worst_w1_tt = "worst_W1_TT_mean" in summary_df.columns

    validation_metrics_line = "- 验证指标: KS(speed), KS(TT), worst-window KS（独立评估）"
    if has_w1_speed or has_w1_tt:
        w1_metrics = []
        if has_w1_speed:
            w1_metrics.append("W1(speed)")
        if has_w1_tt:
            w1_metrics.append("W1(TT)")
        validation_metrics_line = f"- 验证指标: KS(speed), KS(TT), {', '.join(w1_metrics)}, worst-window KS（独立评估）"

    columns = [
        ("RMSE(校准)", "rmse_speed"),
        ("KS(speed)", "ks_speed"),
        ("KS(TT)", "ks_tt"),
    ]
    if has_w1_speed:
        columns.append(("W1(speed)", "W1_speed"))
    if has_w1_tt:
        columns.append(("W1(TT)", "W1_TT"))

    columns.extend([
        ("Worst-15m KS(speed)", "worst_ks_speed"),
        ("Worst-15m KS(TT)", "worst_ks_tt"),
    ])
    if has_worst_w1_speed:
        columns.append(("Worst-15m W1(speed)", "worst_W1_speed"))
    if has_worst_w1_tt:
        columns.append(("Worst-15m W1(TT)", "worst_W1_TT"))

    columns.extend([
        ("Next-day KS(speed)", "next_day_ks_speed"),
        ("Next-day KS(TT)", "next_day_ks_tt"),
    ])

    lines = [
        "# Pipeline 级消融 - 系统对照表",
        "",
        "**协议一致性声明**：所有配置均使用相同的 L2 口径",
        "- 观测向量: M11 走廊 11 维 moving 速度",
        "- 场景: pm_peak (17:00-18:00)",
        "- 状态向量: [capacityFactor, minGap, impatience]",
        "- IES: Ne=10, K=3, β=0.3",
        "- Rule C: T*=325s, v*=5km/h",
        "- 数据源: data/processed（高峰期）",
        "",
        "**校准与验证解耦**：",
        "- 校准指标: RMSE（用于 BO 和 IES 优化）",
        validation_metrics_line,
        "",
        "## 主要结果",
        "",
        "| Pipeline | " + " | ".join([label for label, _ in columns]) + " |",
        "|----------|" + "|".join(["-----------"] * len(columns)) + "|",
    ]
    
    for _, row in summary_df.iterrows():
        name = row["config_name"]
        
        # 格式化指标：mean ± std
        def fmt(metric_prefix):
            mean_val = row.get(f"{metric_prefix}_mean")
            std_val = row.get(f"{metric_prefix}_std")
            if pd.isna(mean_val):
                return "N/A"
            if pd.isna(std_val) or std_val == 0:
                return f"{mean_val:.3f}"
            return f"{mean_val:.3f}±{std_val:.3f}"
        
        row_values = [fmt(metric_key) for _, metric_key in columns]

        lines.append(f"| **{name}** | " + " | ".join(row_values) + " |")
    
    lines.extend([
        "",
        "## 详细配置说明",
        "",
        "| Pipeline | 描述 | L1 校准 | L2 同化 |",
        "|----------|------|---------|---------|",
    ])
    
    for _, row in summary_df.iterrows():
        name = row["config_name"]
        desc = row["description"]
        l1 = "Y" if row["config_id"] in ["A3", "A4"] else "N"
        l2 = "Y" if row["config_id"] in ["A2", "A4"] else "N"
        lines.append(f"| {name} | {desc} | {l1} | {l2} |")
    
    lines.extend([
        "",
        "## 关键发现",
        "",
        "1. **Zero-shot DT (A0)**: 无任何校准的基线性能",
        "2. **Only L2 (A2)**: 证明 L2 状态同化的独立贡献",
        "3. **Only L1 (A3)**: 证明 BO 参数优化的独立贡献",
        "4. **RCMDT (A4)**: L1+L2 协同的完整性能",
        "",
        "通过对比 A2 vs A0，可以量化\"moving 语义是 IES 的前置条件\"的价值。",
        ""
    ])
    
    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Markdown 表格已保存: {output_path}")


# =============================================================================
# 可视化
# =============================================================================

def plot_comparison_bar_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    """生成对比柱状图（IEEE 格式）"""
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
    
    configs = summary_df["config_id"].tolist()
    config_names = [CONFIG_NAMES.get(c, c) for c in configs]
    x = np.arange(len(configs))
    width = 0.6
    
    # 定义颜色
    colors = {
        "A0": "#e74c3c",  # 红色 (baseline)
        "A2": "#f39c12",  # 橙色 (Only L2)
        "A3": "#3498db",  # 蓝色 (Only L1)
        "A4": "#2ecc71"   # 绿色 (Full)
    }
    bar_colors = [colors.get(c, "#95a5a6") for c in configs]
    
    # 子图 1: RMSE(speed)
    ax1 = axes[0]
    rmse_means = summary_df["rmse_speed_mean"].values
    rmse_stds = summary_df["rmse_speed_std"].values
    bars1 = ax1.bar(x, rmse_means, width, yerr=rmse_stds, 
                    color=bar_colors, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'linewidth': 1})
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=15, ha='right', fontsize=7)
    ax1.set_ylabel('RMSE (km/h)')
    ax1.set_title('(a) RMSE (校准指标)', fontweight='bold')
    ax1.set_ylim(0, max(rmse_means) * 1.3 if not pd.isna(rmse_means).all() else 10)
    
    # 添加数值标注
    for bar, val in zip(bars1, rmse_means):
        if not pd.isna(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    # 子图 2: KS(speed)
    ax2 = axes[1]
    ks_speed_means = summary_df["ks_speed_mean"].values
    ks_speed_stds = summary_df["ks_speed_std"].values
    bars2 = ax2.bar(x, ks_speed_means, width, yerr=ks_speed_stds,
                    color=bar_colors, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'linewidth': 1})
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=15, ha='right', fontsize=7)
    ax2.set_ylabel('KS Statistic')
    ax2.set_title('(b) KS(speed) - 验证指标', fontweight='bold')
    ax2.set_ylim(0, 0.6)
    
    # 添加临界线（示例，α=0.05, n≈11）
    ax2.axhline(y=0.41, color='red', linestyle='--', linewidth=1, label='α=0.05')
    ax2.legend(fontsize=6, loc='upper right')
    
    for bar, val in zip(bars2, ks_speed_means):
        if not pd.isna(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    # 子图 3: KS(TT)
    ax3 = axes[2]
    ks_tt_means = summary_df["ks_tt_mean"].values
    ks_tt_stds = summary_df["ks_tt_std"].values
    bars3 = ax3.bar(x, ks_tt_means, width, yerr=ks_tt_stds,
                    color=bar_colors, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'linewidth': 1})
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, rotation=15, ha='right', fontsize=7)
    ax3.set_ylabel('KS Statistic')
    ax3.set_title('(c) KS(TT) - 验证指标', fontweight='bold')
    ax3.set_ylim(0, 0.6)
    
    ax3.axhline(y=0.41, color='red', linestyle='--', linewidth=1, label='α=0.05')
    ax3.legend(fontsize=6, loc='upper right')
    
    for bar, val in zip(bars3, ks_tt_means):
        if not pd.isna(val):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存: {output_path}")
    plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="生成 Pipeline 级消融的系统对照表")
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "experiments_v4" / "unified_l2" / "protocol_ablation" / "full_metrics.csv"),
        help="完整指标 CSV（由 compute_pipeline_metrics.py 生成）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "experiments_v4" / "unified_l2" / "protocol_ablation" / "tables"),
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Pipeline 级消融 - 系统对照表生成")
    print("=" * 70)
    
    # 加载数据
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        print(f"[ERROR] 指标文件不存在: {metrics_csv}")
        print("请先运行 compute_pipeline_metrics.py 生成完整指标")
        return
    
    df = pd.read_csv(metrics_csv)
    print(f"加载指标数据: {len(df)} 条记录")
    print(f"配置: {df['config_id'].unique().tolist()}")
    print()
    
    # 生成摘要统计
    print("[1] 生成摘要统计...")
    summary_df = generate_summary_statistics(df)
    print(summary_df.to_string(index=False))
    print()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存摘要 CSV
    summary_csv = output_dir / "pipeline_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[2] 摘要 CSV 已保存: {summary_csv}")
    print()
    
    # 生成 Markdown 表格
    print("[3] 生成 Markdown 表格...")
    markdown_path = output_dir / "pipeline_comparison_table.md"
    generate_markdown_table(summary_df, markdown_path)
    print()
    
    # 生成可视化
    print("[4] 生成对比图...")
    figure_path = output_dir / "pipeline_comparison_figure.png"
    plot_comparison_bar_chart(summary_df, figure_path)
    print()
    
    print("=" * 70)
    print("完成！产出文件：")
    print(f"  1. {summary_csv}")
    print(f"  2. {markdown_path}")
    print(f"  3. {figure_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
