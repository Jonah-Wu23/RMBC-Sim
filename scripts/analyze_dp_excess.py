#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_dp_excess.py
====================
分析 DP 结果下的剩余超长段贡献

目标：对 68X inbound (scale=1.72) 和 960 inbound (scale=2.02) 做 Pareto 分析
输出：Top-5 超长段列表，用于验证剩余 excess 是否来自真实单向约束

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-29
"""

import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import sumolib
    HAS_SUMOLIB = True
except ImportError:
    HAS_SUMOLIB = False
    print("⚠️ sumolib 未安装")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_net_topology(net_path):
    """加载路网拓扑"""
    tree = ET.parse(str(net_path))
    root = tree.getroot()
    
    edge_lengths = {}
    edge_from_to = {}
    
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):
            from_node = edge.get('from')
            to_node = edge.get('to')
            lane = edge.find('lane')
            if lane is not None:
                edge_lengths[eid] = float(lane.get('length', 0))
            edge_from_to[eid] = (from_node, to_node)
    
    # 构建 reverse 映射
    ft_to_edge = {v: k for k, v in edge_from_to.items()}
    reverse_map = {}
    for eid, (f, t) in edge_from_to.items():
        rev_eid = ft_to_edge.get((t, f))
        if rev_eid:
            reverse_map[eid] = rev_eid
    
    # junction
    junction_edges = {}
    for junction in root.findall('.//junction'):
        jid = junction.get('id')
        inc_lanes = junction.get('incLanes', '').split()
        inc_edges = list(set(l.rsplit('_', 1)[0] for l in inc_lanes if l and not l.startswith(':')))
        out_edges = [eid for eid, (f, t) in edge_from_to.items() if f == jid]
        junction_edges[jid] = {'incoming': inc_edges, 'outgoing': out_edges}
    
    return edge_lengths, edge_from_to, reverse_map, junction_edges


def load_stop_edges(bus_stops_path):
    """加载站点边映射"""
    tree = ET.parse(str(bus_stops_path))
    root = tree.getroot()
    
    stop_to_edge = {}
    for stop in root.findall('.//busStop'):
        stop_id = stop.get('id')
        lane = stop.get('lane', '')
        if lane.startswith(':'):
            edge = lane.rsplit('_', 1)[0]
        else:
            edge = lane.rsplit('_', 1)[0]
        stop_to_edge[stop_id] = edge
        
    return stop_to_edge


def load_dp_results(csv_path):
    """加载 DP 结果"""
    df = pd.read_csv(csv_path)
    return df


def get_shortest_path_sumolib(net, from_edge_id: str, to_edge_id: str, reverse_map: dict):
    """用 sumolib 计算最短路径"""
    if from_edge_id == to_edge_id:
        return 0, [], False
    
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except Exception:
        return float('inf'), [], False
    
    try:
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            total_len = sum(e.getLength() for e in route)
            edge_ids = [e.getID() for e in route]
            
            # 检测 U-turn
            has_uturn = False
            for i in range(len(edge_ids) - 1):
                e1, e2 = edge_ids[i], edge_ids[i + 1]
                if reverse_map.get(e1) == e2 or reverse_map.get(e2) == e1:
                    has_uturn = True
                    break
            
            return total_len, edge_ids, has_uturn
    except Exception:
        pass
    
    return float('inf'), [], False


def analyze_route_segments(net, stop_sequence, dp_edges, edge_lengths, reverse_map):
    """分析每段的超长贡献"""
    segments = []
    
    for i in range(len(stop_sequence) - 1):
        seq_i, stop_i, name_i, cum_dist_i = stop_sequence[i]
        seq_j, stop_j, name_j, cum_dist_j = stop_sequence[i + 1]
        
        edge_i = dp_edges.get(stop_i, '')
        edge_j = dp_edges.get(stop_j, '')
        
        kmb_len = cum_dist_j - cum_dist_i
        if kmb_len <= 0:
            kmb_len = 100
        
        # 计算路径
        if edge_i == edge_j:
            path_len = 0
            path_edges = [edge_i]
            has_uturn = False
        elif net and HAS_SUMOLIB:
            path_len, path_edges, has_uturn = get_shortest_path_sumolib(
                net, edge_i, edge_j, reverse_map
            )
        else:
            path_len = edge_lengths.get(edge_i, 500) + edge_lengths.get(edge_j, 500)
            path_edges = [edge_i, edge_j]
            has_uturn = False
        
        if path_len == float('inf'):
            excess = float('inf')
            ratio = float('inf')
        else:
            excess = path_len - kmb_len
            ratio = path_len / kmb_len if kmb_len > 0 else 0
        
        segments.append({
            'seg_idx': i + 1,
            'from_seq': seq_i,
            'to_seq': seq_j,
            'from_name': name_i[:25],
            'to_name': name_j[:25],
            'from_edge': edge_i,
            'to_edge': edge_j,
            'kmb_len': kmb_len,
            'path_len': path_len,
            'excess': excess,
            'ratio': ratio,
            'has_uturn': has_uturn,
            'path_edges': path_edges,
        })
    
    return segments


def main():
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml"
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    dp_results_path = PROJECT_ROOT / "config" / "calibration" / "stop_edge_comparison.csv"
    
    print("=" * 80)
    print("[DP 结果超长段 Pareto 分析]")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    edge_lengths, edge_from_to, reverse_map, junction_edges = load_net_topology(net_path)
    print(f"  - 边数: {len(edge_lengths)}")
    
    net = None
    if HAS_SUMOLIB:
        print("  - 加载 sumolib.net...")
        net = sumolib.net.readNet(str(net_path), withInternal=False)
    
    stop_to_edge = load_stop_edges(bus_stops_path)
    print(f"  - 站点: {len(stop_to_edge)}")
    
    df_kmb = pd.read_csv(kmb_csv_path)
    df_dp = load_dp_results(dp_results_path)
    print(f"  - DP 结果: {len(df_dp)} 条")
    
    # 分析目标线路
    targets = [
        ('68X', 'inbound', 1.72),
        ('960', 'inbound', 2.02),
    ]
    
    for route, bound, current_scale in targets:
        print(f"\n{'='*80}")
        print(f"[{route} {bound}] 当前 scale = {current_scale}")
        print("-" * 80)
        
        # 获取 KMB 站点序列
        subset_kmb = df_kmb[(df_kmb['route'] == route) & (df_kmb['bound'] == bound)].sort_values('seq')
        
        # 获取 DP 落边结果
        subset_dp = df_dp[(df_dp['route'] == route) & (df_dp['bound'] == bound)]
        dp_edges = dict(zip(subset_dp['stop_id'], subset_dp['dp_edge']))
        
        # 过滤 core 段（边在 cropped 网络中的站点）
        core_sequence = []
        for _, row in subset_kmb.iterrows():
            stop_id = row['stop_id']
            orig_edge = stop_to_edge.get(stop_id, '')
            
            edge_valid = False
            if orig_edge.startswith(':'):
                parts = orig_edge[1:].rsplit('_', 1)
                junction_id = parts[0] if len(parts) >= 1 else orig_edge[1:]
                jinfo = junction_edges.get(junction_id, {})
                candidates = jinfo.get('incoming', []) + jinfo.get('outgoing', [])
                edge_valid = any(e in edge_lengths for e in candidates if not e.startswith(':'))
            else:
                edge_valid = orig_edge in edge_lengths
            
            if edge_valid:
                core_sequence.append((
                    row['seq'], row['stop_id'], 
                    row['stop_name_en'], row['cum_dist_m']
                ))
        
        if len(core_sequence) < 2:
            print(f"  ⚠️ Core 段站点不足")
            continue
        
        print(f"  Core 段: seq {core_sequence[0][0]} - {core_sequence[-1][0]} ({len(core_sequence)} 站)")
        
        # 分析每段
        segments = analyze_route_segments(net, core_sequence, dp_edges, edge_lengths, reverse_map)
        
        # 按 excess 排序（Pareto 分析）
        segments_sorted = sorted(segments, key=lambda x: x['excess'] if x['excess'] != float('inf') else 1e10, reverse=True)
        
        # 统计
        total_excess = sum(s['excess'] for s in segments if s['excess'] != float('inf'))
        total_kmb = sum(s['kmb_len'] for s in segments)
        total_path = sum(s['path_len'] for s in segments if s['path_len'] != float('inf'))
        
        print(f"\n  总 KMB 长度: {total_kmb:.0f}m")
        print(f"  总 DP 路径长度: {total_path:.0f}m")
        print(f"  总 Excess: {total_excess:.0f}m")
        print(f"  Scale: {total_path/total_kmb:.2f}")
        
        # Top-5 超长段
        print(f"\n  [Top-5 超长段 Pareto 贡献]")
        print(f"  {'#':<3} {'seq':<6} {'from→to':<30} {'KMB(m)':<8} {'Path(m)':<10} {'Excess(m)':<10} {'Ratio':<6} {'贡献%':<8}")
        print("  " + "-" * 100)
        
        cumulative_pct = 0
        for i, seg in enumerate(segments_sorted[:5]):
            if seg['excess'] == float('inf'):
                continue
            pct = seg['excess'] / total_excess * 100 if total_excess > 0 else 0
            cumulative_pct += pct
            
            from_to = f"{seg['from_name'][:12]}→{seg['to_name'][:12]}"
            print(f"  {i+1:<3} {seg['from_seq']}-{seg['to_seq']:<4} {from_to:<30} {seg['kmb_len']:<8.0f} {seg['path_len']:<10.0f} {seg['excess']:<10.0f} {seg['ratio']:<6.2f} {pct:<8.1f}")
        
        print(f"\n  前 5 段累计贡献: {cumulative_pct:.1f}% of total excess")
        
        # 详细路径信息（最大超长段）
        if segments_sorted and segments_sorted[0]['excess'] != float('inf'):
            top_seg = segments_sorted[0]
            print(f"\n  [最大超长段详情]")
            print(f"  段: {top_seg['from_name']} → {top_seg['to_name']}")
            print(f"  DP 落边: {top_seg['from_edge']} → {top_seg['to_edge']}")
            print(f"  KMB 长度: {top_seg['kmb_len']:.0f}m")
            print(f"  DP 路径长度: {top_seg['path_len']:.0f}m")
            print(f"  Excess: {top_seg['excess']:.0f}m (贡献 {top_seg['excess']/total_excess*100:.1f}%)")
            if top_seg['path_edges']:
                print(f"  路径边数: {len(top_seg['path_edges'])}")
                if len(top_seg['path_edges']) <= 10:
                    print(f"  路径: {' → '.join(top_seg['path_edges'])}")
                else:
                    print(f"  路径 (前5+后5): {' → '.join(top_seg['path_edges'][:5])} ... {' → '.join(top_seg['path_edges'][-5:])}")
    
    print("\n" + "=" * 80)
    print("[验证建议]")
    print("-" * 80)
    print("对最大超长段，建议验证:")
    print("  1. GDB 查询路径边的 DIR 字段（单向性）")
    print("  2. 检查是否存在平行替代路径")
    print("  3. 确认是否为真实交通管制约束")
    print("=" * 80)


if __name__ == '__main__':
    main()
