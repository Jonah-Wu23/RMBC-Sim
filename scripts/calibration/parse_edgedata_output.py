#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_edgedata_output.py
========================
解析 SUMO 的 edgedata.out.xml 输出，提取与观测向量对应的仿真速度。

功能:
1. 读取 l2_observation_vector.csv 获取目标路段列表
2. 解析 edgedata.out.xml 中的 edge 速度数据
3. 按观测向量顺序组装同维度的仿真向量 Y_sim

输出: 仿真速度向量 (与观测向量同维度)
"""

import os
import sys
import argparse
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_edgedata_xml(edgedata_path: str) -> Dict[str, float]:
    """
    解析 SUMO edgedata.out.xml 文件，提取每条边的平均速度。
    
    Args:
        edgedata_path: edgedata.out.xml 文件路径
    
    Returns:
        Dict[edge_id, speed_m_s]: 边 ID 到速度的映射
    
    XML 格式示例:
    <meandata>
        <interval begin="0.00" end="3600.00">
            <edge id="edge_123" speed="12.5" .../>
        </interval>
    </meandata>
    """
    edge_speeds = {}
    
    print(f"[INFO] 解析 edgedata 文件: {edgedata_path}")
    tree = ET.parse(edgedata_path)
    root = tree.getroot()
    
    # 遍历所有 interval (通常只有一个，覆盖整个仿真时段)
    for interval in root.findall('.//interval'):
        begin = interval.get('begin', '0')
        end = interval.get('end', 'unknown')
        print(f"[INFO] 处理时间段: {begin} - {end}")
        
        for edge in interval.findall('edge'):
            edge_id = edge.get('id')
            speed_str = edge.get('speed')
            
            if edge_id and speed_str:
                try:
                    speed_m_s = float(speed_str)
                    edge_speeds[edge_id] = speed_m_s
                except ValueError:
                    pass
    
    print(f"[INFO] 解析到 {len(edge_speeds)} 条边的速度数据")
    return edge_speeds


def load_link_edge_mapping(mapping_path: Optional[str] = None) -> Dict[tuple, str]:
    """
    加载路段 (route, bound, from_seq, to_seq) 到 SUMO edge_id 的映射。
    
    如果没有提供映射文件，将使用基于站点距离文件的推断逻辑。
    
    Returns:
        Dict[(route, bound, from_seq, to_seq), edge_id]
    """
    # TODO: 实现完整的映射逻辑
    # 当前返回空字典，需要根据实际路网结构完善
    
    # 可以从 kmb_route_stop_dist.csv 推断，或者手动建立映射表
    print("[WARN] 路段-边映射表尚未实现，返回空映射")
    return {}


def build_simulation_vector(
    edgedata_path: str,
    observation_csv: str,
    mapping_path: Optional[str] = None
) -> pd.DataFrame:
    """
    根据观测向量和 edgedata 输出，构建仿真速度向量。
    
    Args:
        edgedata_path: SUMO edgedata.out.xml 路径
        observation_csv: l2_observation_vector.csv 路径
        mapping_path: 可选的路段-边映射文件路径
    
    Returns:
        DataFrame with columns: observation_id, sim_speed_kmh, matched
    """
    # 1. 加载观测向量
    obs_df = pd.read_csv(observation_csv)
    print(f"[INFO] 加载观测向量: {len(obs_df)} 个观测点")
    
    # 2. 解析 edgedata
    edge_speeds = parse_edgedata_xml(edgedata_path)
    
    # 3. 加载映射表
    link_edge_map = load_link_edge_mapping(mapping_path)
    
    # 4. 构建仿真向量
    results = []
    for _, row in obs_df.iterrows():
        obs_id = row['observation_id']
        key = (row['route'], row['bound'], row['from_seq'], row['to_seq'])
        
        # 查找对应的 edge_id
        edge_id = link_edge_map.get(key)
        
        if edge_id and edge_id in edge_speeds:
            # 转换 m/s -> km/h
            sim_speed_kmh = edge_speeds[edge_id] * 3.6
            matched = True
        else:
            # 未匹配到，使用 NaN
            sim_speed_kmh = float('nan')
            matched = False
        
        results.append({
            'observation_id': obs_id,
            'route': row['route'],
            'bound': row['bound'],
            'from_seq': row['from_seq'],
            'to_seq': row['to_seq'],
            'obs_speed_kmh': row['mean_speed_kmh'],
            'sim_speed_kmh': sim_speed_kmh,
            'matched': matched
        })
    
    result_df = pd.DataFrame(results)
    
    # 统计匹配情况
    matched_count = result_df['matched'].sum()
    print(f"[INFO] 匹配成功: {matched_count}/{len(result_df)} ({100*matched_count/len(result_df):.1f}%)")
    
    return result_df


def calculate_residuals(sim_vector_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算观测与仿真之间的残差统计量。
    
    Returns:
        Dict with 'rmse', 'mae', 'bias' etc.
    """
    df = sim_vector_df.dropna(subset=['sim_speed_kmh'])
    
    if len(df) == 0:
        return {'rmse': float('nan'), 'mae': float('nan'), 'bias': float('nan')}
    
    residuals = df['obs_speed_kmh'] - df['sim_speed_kmh']
    
    rmse = (residuals ** 2).mean() ** 0.5
    mae = residuals.abs().mean()
    bias = residuals.mean()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'n_matched': len(df)
    }


def main():
    parser = argparse.ArgumentParser(
        description='解析 SUMO edgedata 输出，构建仿真速度向量'
    )
    parser.add_argument(
        '--edgedata', '-e',
        type=str,
        required=True,
        help='SUMO edgedata.out.xml 文件路径'
    )
    parser.add_argument(
        '--observation', '-o',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'calibration' / 'l2_observation_vector.csv'),
        help='观测向量 CSV 路径 (默认: data/calibration/l2_observation_vector.csv)'
    )
    parser.add_argument(
        '--output', '-O',
        type=str,
        default=None,
        help='输出仿真向量 CSV 路径 (默认: 不保存)'
    )
    parser.add_argument(
        '--mapping', '-m',
        type=str,
        default=None,
        help='路段-边映射文件路径 (可选)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.edgedata):
        print(f"[ERROR] edgedata 文件不存在: {args.edgedata}")
        sys.exit(1)
    
    if not os.path.exists(args.observation):
        print(f"[ERROR] 观测向量文件不存在: {args.observation}")
        sys.exit(1)
    
    # 构建仿真向量
    sim_df = build_simulation_vector(
        edgedata_path=args.edgedata,
        observation_csv=args.observation,
        mapping_path=args.mapping
    )
    
    # 计算残差
    stats = calculate_residuals(sim_df)
    print(f"\n[RESULT] 残差统计:")
    print(f"  - RMSE: {stats['rmse']:.2f} km/h")
    print(f"  - MAE:  {stats['mae']:.2f} km/h")
    print(f"  - Bias: {stats['bias']:.2f} km/h")
    print(f"  - 匹配数: {stats['n_matched']}")
    
    # 可选保存
    if args.output:
        sim_df.to_csv(args.output, index=False)
        print(f"\n[INFO] 仿真向量已保存: {args.output}")


if __name__ == "__main__":
    main()
