#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_obs_edges.py
==================
诊断 observation 的边速度（v2：从 mapping CSV 自动读取 edge_ids）

改进:
- 不再硬编码 edge list，直接从 link_edge_mapping_corridor.csv 读取
- 支持诊断所有 observation（默认）或指定某几个
- 自动聚合跨 interval 的速度数据
"""

import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_mapping(mapping_path: str) -> dict:
    """
    从 mapping CSV 加载 observation_id -> edge_ids 的映射
    
    Returns:
        Dict[observation_id, List[edge_id]]
    """
    df = pd.read_csv(mapping_path)
    mapping = {}
    for _, row in df.iterrows():
        obs_id = row['observation_id']
        edges_str = row['edge_ids']
        # 解析 JSON 格式的 edge list
        try:
            edges = json.loads(edges_str)
        except (json.JSONDecodeError, TypeError):
            edges = []
        mapping[obs_id] = edges
    return mapping


def load_observation_info(obs_path: str) -> dict:
    """
    加载观测向量信息（用于显示描述性名称）
    
    Returns:
        Dict[observation_id, {route, bound, from_seq, to_seq, obs_speed}]
    """
    df = pd.read_csv(obs_path)
    info = {}
    for _, row in df.iterrows():
        # 兼容多种列名
        obs_speed = 0
        for col in ['mean_speed_kmh', 'speed_km_h', 'observed_value']:
            if col in row and pd.notna(row[col]):
                obs_speed = row[col]
                break
        
        info[row['observation_id']] = {
            'route': row['route'],
            'bound': row['bound'],
            'from_seq': int(row['from_seq']),
            'to_seq': int(row['to_seq']),
            'obs_speed': obs_speed
        }
    return info


def parse_edgedata(edgedata_path: str) -> dict:
    """
    解析 edgedata.out.xml，聚合跨 interval 的速度数据
    
    Returns:
        Dict[edge_id, {sampled: total_seconds, speed_weighted: sum(speed * seconds)}]
    """
    tree = ET.parse(edgedata_path)
    root = tree.getroot()
    
    edge_data = {}
    for interval in root.findall('.//interval'):
        for edge in interval.findall('edge'):
            eid = edge.get('id')
            sampled = float(edge.get('sampledSeconds', 0))
            speed = edge.get('speed')
            if speed and sampled > 0:
                if eid not in edge_data:
                    edge_data[eid] = {'sampled': 0, 'speed_weighted': 0}
                edge_data[eid]['sampled'] += sampled
                edge_data[eid]['speed_weighted'] += float(speed) * sampled
    
    return edge_data


def check_obs(obs_id: int, obs_edges: list, obs_speed: float, 
              obs_info: dict, edge_data: dict, verbose: bool = True) -> dict:
    """
    检查单个 observation 的仿真边速度
    
    Returns:
        Dict with aggregated statistics
    """
    info = obs_info.get(obs_id, {})
    route = info.get('route', '?')
    bound = info.get('bound', '?')
    from_seq = info.get('from_seq', '?')
    to_seq = info.get('to_seq', '?')
    
    if verbose:
        print(f'=== Obs {obs_id}: {route} {bound} (站点 {from_seq}→{to_seq}), 观测速度={obs_speed:.2f} km/h ===')
    
    found = 0
    total_sampled = 0
    total_speed_weighted = 0
    edge_speeds = []
    
    for eid in obs_edges:
        # 检查原始和 _rev 变体
        candidates = [eid]
        if eid.endswith('_rev'):
            candidates.append(eid[:-4])
        else:
            candidates.append(eid + '_rev')
        
        for cand in candidates:
            if cand in edge_data:
                d = edge_data[cand]
                avg_speed = d['speed_weighted'] / d['sampled'] * 3.6
                if verbose:
                    print(f'  {cand}: sampled={d["sampled"]:.0f}s, speed={avg_speed:.1f} km/h')
                found += 1
                total_sampled += d['sampled']
                total_speed_weighted += d['speed_weighted']
                edge_speeds.append(avg_speed)
                break
    
    # 计算加权平均速度
    if total_sampled > 0:
        weighted_avg_speed = total_speed_weighted / total_sampled * 3.6
    else:
        weighted_avg_speed = 0
    
    if verbose:
        print(f'  >> 找到 {found}/{len(obs_edges)} 边, 总采样 {total_sampled:.0f}s')
        if found > 0:
            print(f'  >> 加权平均速度: {weighted_avg_speed:.1f} km/h (观测: {obs_speed:.1f} km/h)')
        print()
    
    return {
        'obs_id': obs_id,
        'route': route,
        'bound': bound,
        'from_seq': from_seq,
        'to_seq': to_seq,
        'obs_speed': obs_speed,
        'sim_speed': weighted_avg_speed,
        'found_edges': found,
        'total_edges': len(obs_edges),
        'total_sampled': total_sampled,
        'edge_speeds': edge_speeds
    }


def main():
    parser = argparse.ArgumentParser(description='诊断 observation 的边速度')
    parser.add_argument('--edgedata', '-e', 
                        default='sumo/output/ies_runs/iter05_run00/edgedata.out.xml',
                        help='edgedata.out.xml 路径')
    parser.add_argument('--mapping', '-m',
                        default='config/calibration/link_edge_mapping_corridor.csv',
                        help='mapping CSV 路径')
    parser.add_argument('--obs', '-o',
                        default='data/calibration/l2_observation_vector_corridor.csv',
                        help='观测向量 CSV 路径')
    parser.add_argument('--obs-ids', '-i', type=int, nargs='*',
                        help='指定要诊断的 observation IDs（默认全部）')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='只显示摘要表格，不显示边详情')
    args = parser.parse_args()
    
    # 解析路径
    edgedata_path = PROJECT_ROOT / args.edgedata
    mapping_path = PROJECT_ROOT / args.mapping
    obs_path = PROJECT_ROOT / args.obs
    
    # 加载数据
    print('[INFO] 加载数据...')
    print(f'  - Mapping: {mapping_path}')
    print(f'  - Observation: {obs_path}')
    print(f'  - Edgedata: {edgedata_path}')
    print()
    
    mapping = load_mapping(str(mapping_path))
    obs_info = load_observation_info(str(obs_path))
    edge_data = parse_edgedata(str(edgedata_path))
    
    # 确定要诊断的 observation IDs
    if args.obs_ids:
        obs_ids = args.obs_ids
    else:
        obs_ids = sorted(mapping.keys())
    
    # 执行诊断
    results = []
    for obs_id in obs_ids:
        edges = mapping.get(obs_id, [])
        obs_speed = obs_info.get(obs_id, {}).get('obs_speed', 0)
        
        if not edges:
            print(f'[WARN] Obs {obs_id} 没有映射边，跳过\n')
            continue
        
        result = check_obs(
            obs_id, edges, obs_speed, obs_info, edge_data,
            verbose=not args.summary
        )
        results.append(result)
    
    # 打印摘要
    if results:
        print('=' * 80)
        print('[摘要] 所有 Observation 诊断结果')
        print('-' * 80)
        print(f'{"Obs":>4} | {"Route":>5} | {"Bound":>8} | {"Seq":>6} | {"Obs_spd":>8} | {"Sim_spd":>8} | {"Diff":>8} | {"Edges":>10} | {"Sampled":>10}')
        print('-' * 80)
        
        for r in results:
            seq_str = f"{r['from_seq']}→{r['to_seq']}"
            diff = r['sim_speed'] - r['obs_speed']
            diff_str = f"{diff:+.1f}"
            edges_str = f"{r['found_edges']}/{r['total_edges']}"
            
            # 标记异常值
            flag = ""
            if abs(diff) > 10:
                flag = " ⚠️"
            if r['sim_speed'] > 50:
                flag += " 🚗"  # 可能有高速边
            
            print(f"{r['obs_id']:>4} | {r['route']:>5} | {r['bound']:>8} | {seq_str:>6} | {r['obs_speed']:>7.1f} | {r['sim_speed']:>7.1f} | {diff_str:>7} | {edges_str:>10} | {r['total_sampled']:>9.0f}s{flag}")
        
        print('-' * 80)
        
        # 总体统计
        import statistics
        obs_speeds = [r['obs_speed'] for r in results if r['obs_speed'] > 0]
        sim_speeds = [r['sim_speed'] for r in results if r['sim_speed'] > 0]
        diffs = [r['sim_speed'] - r['obs_speed'] for r in results if r['sim_speed'] > 0 and r['obs_speed'] > 0]
        
        if diffs:
            print(f'[统计] 差异 (sim - obs): mean={statistics.mean(diffs):+.1f}, std={statistics.stdev(diffs):.1f}, range=[{min(diffs):+.1f}, {max(diffs):+.1f}]')
        if obs_speeds and sim_speeds:
            print(f'[统计] 观测速度: mean={statistics.mean(obs_speeds):.1f} km/h')
            print(f'[统计] 仿真速度: mean={statistics.mean(sim_speeds):.1f} km/h')


if __name__ == "__main__":
    main()
