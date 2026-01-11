#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_threshold_sensitivity_pmpeak.py
====================================
为 pm_peak 数据生成阈值敏感性分析（3×3 网格）

网格配置:
    v* ∈ {4, 5, 6} km/h
    T* ∈ {275, 325, 375} s

使用 A2 (Only-L2) 的 pm_peak 数据

Author: RCMDT Project
Date: 2026-01-11
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "calibration"))

from metrics_v3 import (
    load_real_link_stats, compute_sim_link_data, apply_rule_c_audit,
    compute_ks_with_stats, compute_worst_window_exhaustive,
    RULE_C_MAX_DIST_M
)

# PM Peak 时间窗口配置 (17:00-18:00)
PMPEAK_DURATION_SEC = 3600
SUBWINDOW_DURATION_SEC = 900  # 15 min
SUBWINDOW_STEP_SEC = 60  # 1 min
PMPEAK_START_SEC = 17 * 3600  # 17:00

# 3×3 网格
V_STAR_GRID = [4, 5, 6]  # km/h
T_STAR_GRID = [275, 325, 375]  # seconds

MIN_CLEAN_SAMPLES = 10

# 数据路径 (data/processed/link_stats.csv 已经是 PM Peak 17:00-18:00 数据)
REAL_STATS = PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
SIM_BASE_DIR = PROJECT_ROOT / "data" / "experiments_v4" / "protocol_ablation" / "pm_peak" / "A2"
DIST_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration_v3" / "sensitivity"
NUM_SEEDS = 5  # seed0-4


def run_sensitivity_analysis(
    df_real: pd.DataFrame,
    sim_speeds_list: List[np.ndarray],
    v_star_grid: List[float] = V_STAR_GRID,
    t_star_grid: List[float] = T_STAR_GRID,
    max_dist_m: float = RULE_C_MAX_DIST_M
) -> pd.DataFrame:
    """
    运行二维敏感性分析（多seed平均）
    
    Args:
        df_real: 真实数据
        sim_speeds_list: 多个seed的仿真速度数组列表
    """
    results = []
    
    for t_star in t_star_grid:
        for v_star in v_star_grid:
            print(f"  处理 T*={t_star}s, v*={v_star}km/h...")
            
            # 应用 Rule C
            raw_speeds, clean_speeds, raw_tt, clean_tt, flagged_frac, n_clean = apply_rule_c_audit(
                df_real, t_star, v_star, max_dist_m
            )
            
            # Full-hour KS(speed) - 对每个seed计算，然后取mean/std
            if n_clean >= MIN_CLEAN_SAMPLES:
                ks_values = []
                worst_ks_values = []
                
                for sim_speeds in sim_speeds_list:
                    # KS(speed)
                    ks_result = compute_ks_with_stats(clean_speeds, sim_speeds)
                    ks_values.append(ks_result.ks_stat)
                    
                    # Worst-window (exhaustive)
                    worst_result = compute_worst_window_exhaustive(
                        clean_speeds, sim_speeds,
                        total_duration_sec=PMPEAK_DURATION_SEC,
                        window_duration_sec=SUBWINDOW_DURATION_SEC,
                        step_sec=SUBWINDOW_STEP_SEC,
                        base_time_sec=PMPEAK_START_SEC
                    )
                    worst_ks_values.append(worst_result.worst_ks)
                
                ks_clean_mean = np.mean(ks_values)
                ks_clean_std = np.std(ks_values, ddof=1) if len(ks_values) > 1 else 0.0
                worst_ks_mean = np.mean(worst_ks_values)
                worst_ks_std = np.std(worst_ks_values, ddof=1) if len(worst_ks_values) > 1 else 0.0
                
                # 使用第一个seed的worst_window_time作为代表
                worst_result_ref = compute_worst_window_exhaustive(
                    clean_speeds, sim_speeds_list[0],
                    total_duration_sec=PMPEAK_DURATION_SEC,
                    window_duration_sec=SUBWINDOW_DURATION_SEC,
                    step_sec=SUBWINDOW_STEP_SEC,
                    base_time_sec=PMPEAK_START_SEC
                )
                worst_window_time = f"{worst_result_ref.window_start_time}-{worst_result_ref.window_end_time}" if worst_result_ref.window_start_time else None
            else:
                ks_clean_mean = None
                ks_clean_std = None
                worst_ks_mean = None
                worst_ks_std = None
                worst_window_time = None
            
            results.append({
                "T_star": t_star,
                "v_star": v_star,
                "flagged_pct": flagged_frac * 100,
                "n_clean": n_clean,
                "ks_clean": ks_clean_mean,
                "ks_clean_std": ks_clean_std,
                "worst_window_ks": worst_ks_mean,
                "worst_window_ks_std": worst_ks_std,
                "worst_window_time": worst_window_time
            })
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("PM Peak 阈值敏感性分析 (3×3 网格)")
    print("=" * 70)
    print(f"场景: PM Peak (17:00-18:00)")
    print(f"数据源: A2 (Only-L2) protocol_ablation/pm_peak")
    print(f"Seeds: 0-{NUM_SEEDS-1}")
    print(f"v* 网格: {V_STAR_GRID} km/h")
    print(f"T* 网格: {T_STAR_GRID} s")
    print()
    
    # 加载数据
    print("[1] 加载真实数据...")
    df_real = load_real_link_stats(str(REAL_STATS))
    print(f"    样本数: {len(df_real)}")
    
    print(f"\n[2] 加载仿真数据 (seed0-{NUM_SEEDS-1})...")
    sim_speeds_list = []
    for seed_id in range(NUM_SEEDS):
        stopinfo_path = SIM_BASE_DIR / f"seed{seed_id}" / "stopinfo.xml"
        print(f"    加载 seed{seed_id}...")
        sim_speeds, sim_tt, _ = compute_sim_link_data(str(stopinfo_path), str(DIST_FILE))
        print(f"      样本数: {len(sim_speeds)}")
        sim_speeds_list.append(sim_speeds)
    
    print(f"\n    总计 {len(sim_speeds_list)} 个seeds")
    
    # 运行敏感性分析
    print("\n[3] 运行敏感性分析（计算跨seed的KS mean/std）...")
    df_results = run_sensitivity_analysis(df_real, sim_speeds_list)
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_path = OUTPUT_DIR / "threshold_sensitivity_results_pmpeak.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[4] 结果已保存: {csv_path}")
    
    # 显示结果
    print("\n" + "=" * 70)
    print("敏感性分析结果")
    print("=" * 70)
    print(df_results.to_string(index=False))
    
    # 论文选择的配置
    paper_config = df_results[
        (df_results["T_star"] == 325) & (df_results["v_star"] == 5)
    ]
    
    if not paper_config.empty:
        row = paper_config.iloc[0]
        print("\n" + "=" * 70)
        print("论文选择 (T*=325s, v*=5km/h)")
        print("=" * 70)
        print(f"  - Flagged: {row['flagged_pct']:.1f}%")
        print(f"  - n_clean: {int(row['n_clean'])}\n")
        if pd.notna(row['ks_clean']):
            print(f"  - KS(speed) mean: {row['ks_clean']:.4f}")
            if pd.notna(row['ks_clean_std']):
                print(f"  - KS(speed) std: {row['ks_clean_std']:.4f}")
        if pd.notna(row['worst_window_ks']):
            print(f"  - Worst-15min mean: {row['worst_window_ks']:.4f}")
            if pd.notna(row['worst_window_ks_std']):
                print(f"  - Worst-15min std: {row['worst_window_ks_std']:.4f}")
            print(f"  - Worst window (seed0): {row['worst_window_time']}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
