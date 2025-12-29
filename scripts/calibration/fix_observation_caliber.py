#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_observation_caliber.py
==========================
修复观测向量口径：用 L1 校准的 dwell 参数估计并剔除停站时间

核心公式:
  t_dwell = t_fixed + t_board * avg_boarding_passengers
  t_drive = travel_time - t_dwell
  v_drive = dist / t_drive * 3.6 (km/h)

这使得 Y_obs 定义与 SUMO edgedata (行驶速度) 同口径。
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_l1_params():
    """加载 L1 校准参数"""
    l1_path = PROJECT_ROOT / "config" / "calibration" / "l1_optimized.json"
    if l1_path.exists():
        with open(l1_path) as f:
            params = json.load(f)
        print(f"[INFO] 加载 L1 校准参数: {l1_path}")
        return params
    
    # Fallback to default
    print("[WARN] L1 优化参数不存在，使用默认值")
    return {
        "t_fixed": 12.0,
        "t_board": 1.3
    }


def estimate_dwell_time(t_fixed: float, t_board: float, avg_passengers: float = 2.5) -> float:
    """
    估计单站停靠时间
    
    Args:
        t_fixed: 固定停站时间 (s)
        t_board: 每乘客上车时间 (s)
        avg_passengers: 平均每站上车人数 (default 2.5)
    
    Returns:
        估计的停站时间 (s)
    """
    return t_fixed + t_board * avg_passengers


def fix_observation_caliber(
    input_path: str,
    output_path: str,
    l1_params: dict,
    avg_passengers_per_stop: float = 2.5,
    min_sample_count: int = 2
) -> pd.DataFrame:
    """
    修复观测向量口径：剔除估计的停站时间
    
    Args:
        input_path: l2_observation_vector_corridor_M11.csv 路径
        output_path: 输出路径
        l1_params: L1 校准参数
        avg_passengers_per_stop: 每站平均上车人数
        min_sample_count: 最小样本量
    
    Returns:
        修复后的观测向量
    """
    print(f"[INFO] 读取观测向量: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] 原始观测点: {len(df)}")
    
    # L1 参数
    t_fixed = l1_params.get("t_fixed", 12.0)
    t_board = l1_params.get("t_board", 1.3)
    t_dwell_per_stop = estimate_dwell_time(t_fixed, t_board, avg_passengers_per_stop)
    print(f"[INFO] L1 参数: t_fixed={t_fixed:.2f}s, t_board={t_board:.2f}s")
    print(f"[INFO] 估计每站停靠时间: {t_dwell_per_stop:.1f}s")
    
    # 计算原始旅行时间（从速度反推）
    # speed_kmh = dist_m / travel_time_s * 3.6
    # travel_time_s = dist_m / (speed_kmh / 3.6)
    df['travel_time_s'] = df['dist_m'] / (df['mean_speed_kmh'] / 3.6)
    
    # 每个观测段包含一个站点（to_seq 站），估计该站的停靠时间
    # 简化假设：每段剔除 1 个站的停靠时间
    n_stops_per_segment = 1
    df['t_dwell_estimated'] = n_stops_per_segment * t_dwell_per_stop
    
    # 计算行驶时间
    df['t_drive_s'] = df['travel_time_s'] - df['t_dwell_estimated']
    # 防止负值
    df['t_drive_s'] = df['t_drive_s'].clip(lower=5.0)  # 最小 5 秒
    
    # 计算行驶速度
    df['speed_drive_kmh'] = df['dist_m'] / df['t_drive_s'] * 3.6
    
    # 打印诊断
    print("\n[DIAG] 速度修正诊断:")
    print(f"  原始速度范围: {df['mean_speed_kmh'].min():.2f} - {df['mean_speed_kmh'].max():.2f} km/h")
    print(f"  原始速度中位数: {df['mean_speed_kmh'].median():.2f} km/h")
    print(f"  修正速度范围: {df['speed_drive_kmh'].min():.2f} - {df['speed_drive_kmh'].max():.2f} km/h")
    print(f"  修正速度中位数: {df['speed_drive_kmh'].median():.2f} km/h")
    
    # 速度提升倍数
    ratio = df['speed_drive_kmh'].median() / df['mean_speed_kmh'].median()
    print(f"  速度提升倍数: {ratio:.2f}x")
    
    # 更新观测向量：用修正后的速度替换原始速度
    df_out = df[['observation_id', 'route', 'bound', 'from_seq', 'to_seq', 
                  'speed_drive_kmh', 'std_speed_kmh', 'sample_count', 'dist_m']].copy()
    df_out.rename(columns={'speed_drive_kmh': 'mean_speed_kmh'}, inplace=True)
    
    # 同时调整 std（保持 CV 不变）
    # std_new = std_old * (speed_new / speed_old)
    cv = df['std_speed_kmh'] / df['mean_speed_kmh']
    df_out['std_speed_kmh'] = cv * df['speed_drive_kmh']
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\n[INFO] 修正后观测向量已保存: {output_path}")
    
    return df_out


def main():
    # 输入/输出路径
    input_path = PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector_corridor_M11.csv"
    output_path = PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector_corridor_M11_drive.csv"
    
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        return
    
    # 加载 L1 参数
    l1_params = load_l1_params()
    
    # 修复口径
    fix_observation_caliber(
        input_path=str(input_path),
        output_path=str(output_path),
        l1_params=l1_params,
        avg_passengers_per_stop=2.5  # 可调参数
    )
    
    print("\n[INFO] 完成！下一步:")
    print("  1. 更新 IES 使用 _drive 版本的观测向量")
    print("  2. 重跑 B4 验证 K-S 下降")


if __name__ == "__main__":
    main()
