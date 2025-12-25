#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_distribution.py
======================
统计分布检验: 对比仿真与真实路网速度分布

功能:
    1. 绘制全路网速度的 CDF (累积分布函数) 对比图
    2. 计算 K-S 统计量 (Kolmogorov-Smirnov Test)
    3. 目标: D_KS < 0.15 或 p-value > 0.05

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-23
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# IEEE Paper Style: Times New Roman font (8pt with small figure size)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7


def load_observation_vector(path: str) -> np.ndarray:
    """加载真实观测速度向量"""
    df = pd.read_csv(path)
    return df['mean_speed_kmh'].values


def load_simulation_speeds_from_edgedata(edgedata_path: str) -> np.ndarray:
    """从 edgedata.out.xml 解析仿真速度"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(edgedata_path)
    root = tree.getroot()
    
    speeds = []
    for interval in root.findall('interval'):
        for edge in interval.findall('edge'):
            speed_str = edge.get('speed')
            if speed_str:
                try:
                    speed_mps = float(speed_str)
                    speed_kmh = speed_mps * 3.6
                    if speed_kmh > 0:  # 过滤零速度边
                        speeds.append(speed_kmh)
                except ValueError:
                    continue
    
    return np.array(speeds)


def load_ies_best_iteration_speeds(ies_output_dir: str, best_iter: int) -> np.ndarray:
    """加载 IES 最优迭代的所有仿真速度"""
    all_speeds = []
    
    iter_pattern = f"iter{best_iter:02d}_run*"
    output_path = Path(ies_output_dir)
    
    for run_dir in sorted(output_path.glob(iter_pattern)):
        edgedata_path = run_dir / "edgedata.out.xml"
        if edgedata_path.exists():
            try:
                speeds = load_simulation_speeds_from_edgedata(str(edgedata_path))
                all_speeds.extend(speeds)
            except Exception as e:
                print(f"[WARN] 解析失败 {edgedata_path}: {e}")
    
    return np.array(all_speeds)


def compute_ks_test(real_speeds: np.ndarray, sim_speeds: np.ndarray) -> dict:
    """
    计算 K-S 检验统计量
    
    Returns:
        dict: 包含 ks_statistic, p_value, 和判定结果
    """
    ks_stat, p_value = stats.ks_2samp(real_speeds, sim_speeds)
    
    # 判定标准
    passed_ks = ks_stat < 0.15
    passed_pvalue = p_value > 0.05
    
    return {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'passed_ks_threshold': passed_ks,
        'passed_pvalue_threshold': passed_pvalue,
        'overall_passed': passed_ks or passed_pvalue
    }


def plot_cdf_comparison(
    real_speeds: np.ndarray,
    sim_speeds: np.ndarray,
    ks_result: dict,
    output_path: str
):
    """Plot CDF comparison figure (IEEE paper style)"""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))  # IEEE double-column width
    
    # --- Left: CDF Comparison ---
    ax1 = axes[0]
    
    # Compute CDF
    real_sorted = np.sort(real_speeds)
    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    
    sim_sorted = np.sort(sim_speeds)
    sim_cdf = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted)
    
    ax1.plot(real_sorted, real_cdf, color='#1f77b4', linestyle='-', linewidth=1.5, label='Real-world Observation')
    ax1.plot(sim_sorted, sim_cdf, color='#ff7f0e', linestyle='--', linewidth=1.5, label='Simulation')
    
    ax1.set_xlabel('Speed (km/h)')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title('(a) Speed Distribution CDF Comparison', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(real_sorted.max(), sim_sorted.max()) * 1.05)
    
    # K-S statistics - NOT displayed on figure per user request
    
    # --- Right: Histogram Comparison ---
    ax2 = axes[1]
    
    # Common bins
    all_speeds = np.concatenate([real_speeds, sim_speeds])
    bins = np.linspace(0, np.percentile(all_speeds, 99), 30)
    
    ax2.hist(real_speeds, bins=bins, alpha=0.6, label='Real-world', color='#1f77b4', density=True)
    ax2.hist(sim_speeds, bins=bins, alpha=0.6, label='Simulation', color='#ff7f0e', density=True)
    
    ax2.set_xlabel('Speed (km/h)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('(b) Speed Distribution Histogram', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Sample size annotation (moved to upper-left to avoid legend overlap)
    ax2.text(0.05, 0.95, f"$N_{{real}}$={len(real_speeds)}\n$N_{{sim}}$={len(sim_speeds)}",
             transform=ax2.transAxes, verticalalignment='top',
             horizontalalignment='left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] CDF comparison figure saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='统计分布检验: K-S 检验与 CDF 对比'
    )
    parser.add_argument(
        '--obs', '-o',
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector.csv"),
        help='真实观测向量文件路径'
    )
    parser.add_argument(
        '--ies-output', '-i',
        type=str,
        default=str(PROJECT_ROOT / "sumo" / "output" / "ies_runs"),
        help='IES 仿真输出目录'
    )
    parser.add_argument(
        '--best-iter', '-b',
        type=int,
        default=5,
        help='最优迭代轮次 (默认: 5)'
    )
    parser.add_argument(
        '--out', '-O',
        type=str,
        default=str(PROJECT_ROOT / "plots" / "distribution_verification.png"),
        help='输出图像路径'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("统计分布检验 (K-S Test)")
    print("=" * 60)
    
    # 1. 加载真实观测
    print(f"\n[1] 加载真实观测: {args.obs}")
    real_speeds = load_observation_vector(args.obs)
    print(f"    真实观测样本数: {len(real_speeds)}")
    print(f"    速度范围: [{real_speeds.min():.2f}, {real_speeds.max():.2f}] km/h")
    
    # 2. 加载仿真结果
    print(f"\n[2] 加载 IES 迭代 {args.best_iter} 仿真结果: {args.ies_output}")
    sim_speeds = load_ies_best_iteration_speeds(args.ies_output, args.best_iter)
    
    if len(sim_speeds) == 0:
        print("[ERROR] 未找到仿真速度数据!")
        # 尝试加载单个 edgedata 文件作为后备
        fallback_path = PROJECT_ROOT / "sumo" / "output" / "edgedata.out.xml"
        if fallback_path.exists():
            print(f"    尝试后备路径: {fallback_path}")
            sim_speeds = load_simulation_speeds_from_edgedata(str(fallback_path))
    
    if len(sim_speeds) == 0:
        print("[ERROR] 无法加载仿真数据，退出。")
        return
    
    print(f"    仿真样本数: {len(sim_speeds)}")
    print(f"    速度范围: [{sim_speeds.min():.2f}, {sim_speeds.max():.2f}] km/h")
    
    # 3. K-S 检验
    print("\n[3] 执行 K-S 检验...")
    ks_result = compute_ks_test(real_speeds, sim_speeds)
    
    print(f"\n{'='*40}")
    print("K-S 检验结果")
    print(f"{'='*40}")
    print(f"  K-S 统计量 D_KS = {ks_result['ks_statistic']:.4f}")
    print(f"  p-value         = {ks_result['p_value']:.4f}")
    print(f"  D_KS < 0.15?    : {'通过 ✓' if ks_result['passed_ks_threshold'] else '未通过 ✗'}")
    print(f"  p-value > 0.05? : {'通过 ✓' if ks_result['passed_pvalue_threshold'] else '未通过 ✗'}")
    print(f"  总体判定        : {'通过 ✓' if ks_result['overall_passed'] else '未通过 ✗'}")
    print(f"{'='*40}")
    
    # 4. 绘制 CDF 对比图
    print(f"\n[4] 绘制 CDF 对比图: {args.out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_cdf_comparison(real_speeds, sim_speeds, ks_result, args.out)
    
    print("\n[完成] 统计分布检验完成!")


if __name__ == "__main__":
    main()
