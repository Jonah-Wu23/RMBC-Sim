#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_l2_observation_vector.py
==============================
从 enriched_link_stats.csv 构建 L2 校准的高置信度观测向量。

筛选逻辑:
1. 仅保留 Main Corridor 路段 (route = 68X 或 960)
2. 剔除样本量 N < 10 的噪声路段
3. 剔除长度 < 50m 的短路段

输出: data/calibration/l2_observation_vector.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_observation_vector(
    input_path: str,
    output_path: str,
    min_sample_count: int = 10,
    min_distance_m: float = 50.0,
    routes: list = None
) -> pd.DataFrame:
    """
    构建高置信度观测向量。
    
    Args:
        input_path: enriched_link_stats.csv 路径
        output_path: 输出的观测向量 CSV 路径
        min_sample_count: 最小样本量阈值
        min_distance_m: 最小路段长度阈值 (米)
        routes: 目标路线列表，默认 ['68X', '960']
    
    Returns:
        过滤后的观测向量 DataFrame
    """
    if routes is None:
        routes = ['68X', '960']
    
    print(f"[INFO] 读取输入文件: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] 原始记录数: {len(df)}")
    
    # Step 1: 筛选 Main Corridor 路线
    df_filtered = df[df['route'].isin(routes)].copy()
    print(f"[INFO] 筛选路线 {routes} 后: {len(df_filtered)} 条记录")
    
    # Step 2: 按路段聚合 (route, bound, from_seq, to_seq)
    # 计算每个路段的平均速度、标准差、样本量
    agg_df = df_filtered.groupby(['route', 'bound', 'from_seq', 'to_seq']).agg(
        mean_speed_kmh=('speed_kmh', 'mean'),
        std_speed_kmh=('speed_kmh', 'std'),
        sample_count=('speed_kmh', 'count'),
        dist_m=('dist_m', 'first')  # 距离应该是固定的
    ).reset_index()
    
    print(f"[INFO] 聚合后路段数: {len(agg_df)}")
    
    # 填充 NaN 标准差 (单样本情况)
    agg_df['std_speed_kmh'] = agg_df['std_speed_kmh'].fillna(0.0)
    
    # Step 3: 筛选样本量 >= min_sample_count
    before_count = len(agg_df)
    agg_df = agg_df[agg_df['sample_count'] >= min_sample_count]
    print(f"[INFO] 剔除样本量 < {min_sample_count} 后: {len(agg_df)} 条 (剔除 {before_count - len(agg_df)} 条)")
    
    # Step 4: 筛选路段长度 >= min_distance_m
    before_count = len(agg_df)
    agg_df = agg_df[agg_df['dist_m'] >= min_distance_m]
    print(f"[INFO] 剔除长度 < {min_distance_m}m 后: {len(agg_df)} 条 (剔除 {before_count - len(agg_df)} 条)")
    
    # 添加观测 ID 并排序
    agg_df = agg_df.sort_values(['route', 'bound', 'from_seq']).reset_index(drop=True)
    agg_df.insert(0, 'observation_id', range(1, len(agg_df) + 1))
    
    # 保存输出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    print(f"[INFO] 观测向量已保存: {output_path}")
    print(f"[INFO] 最终观测向量维度: M = {len(agg_df)}")
    
    # 统计信息
    print("\n[SUMMARY] 观测向量统计:")
    print(f"  - 68X 路段数: {len(agg_df[agg_df['route'] == '68X'])}")
    print(f"  - 960 路段数: {len(agg_df[agg_df['route'] == '960'])}")
    print(f"  - 平均速度范围: {agg_df['mean_speed_kmh'].min():.2f} - {agg_df['mean_speed_kmh'].max():.2f} km/h")
    print(f"  - 样本量范围: {agg_df['sample_count'].min()} - {agg_df['sample_count'].max()}")
    
    return agg_df


def main():
    input_path = PROJECT_ROOT / "data" / "processed" / "enriched_link_stats.csv"
    output_path = PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector.csv"
    
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)
    
    build_observation_vector(
        input_path=str(input_path),
        output_path=str(output_path),
        min_sample_count=2,  # 降低阈值以适应当前数据集（最大样本量仅 7）
        min_distance_m=50.0,
        routes=['68X', '960']
    )


if __name__ == "__main__":
    main()
