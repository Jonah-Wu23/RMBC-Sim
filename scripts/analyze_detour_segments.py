#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_detour_segments.py
==========================
分段 Detour 贡献分析脚本

对每条线路计算分段 detour 贡献：
- 每段计算: bus_path_len / kmb_len, bus_path_len - kmb_len
- 输出 top 5 贡献段
- Pareto 覆盖率: top1/top3 贡献占总超长里程的比例

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-27
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v1.net.xml"
ROUTE_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_via.rou.xml"
KMB_CSV_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
CORRECTIONS_FILE = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"
OUTPUT_FILE = PROJECT_ROOT / "logs" / "detour_contribution.csv"


def load_edge_lengths():
    """加载边长度"""
    tree = ET.parse(str(NET_FILE))
    root = tree.getroot()
    lengths = {}
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid:
            lane = edge.find('lane')
            if lane is not None:
                lengths[eid] = float(lane.get('length', 0))
    return lengths


def load_corrections():
    """加载站点边纠偏"""
    df = pd.read_csv(CORRECTIONS_FILE)
    corrections = {}
    for _, row in df.iterrows():
        key = (row['route'], row['bound'], row['stop_id'])
        edge = row['fixed_edge'] if pd.notna(row['fixed_edge']) else row['orig_edge']
        if pd.notna(edge) and edge:
            corrections[key] = str(edge)
    return corrections


def parse_routes():
    """解析路由文件"""
    tree = ET.parse(str(ROUTE_FILE))
    root = tree.getroot()
    
    routes = {}
    for vehicle in root.findall('.//vehicle'):
        vid = vehicle.get('id')
        parts = vid.split('_')
        if len(parts) >= 3:
            route_name = parts[1]
            bound = parts[2].split('.')[0]
            
            route_elem = vehicle.find('route')
            if route_elem is not None:
                edges = route_elem.get('edges', '').split()
                if (route_name, bound) not in routes:
                    routes[(route_name, bound)] = edges
    
    return routes


def get_edge_in_route(route_edges, target_edge):
    """找到目标边在路由中的位置"""
    base_target = target_edge[:-4] if target_edge.endswith('_rev') else target_edge
    
    for i, edge in enumerate(route_edges):
        base_edge = edge[:-4] if edge.endswith('_rev') else edge
        if base_edge == base_target or edge == target_edge:
            return i
    return -1


def compute_segment_length(edges, start_idx, end_idx, edge_lengths):
    """计算路由段长度"""
    total = 0
    for i in range(start_idx, min(end_idx + 1, len(edges))):
        e = edges[i]
        if e in edge_lengths:
            total += edge_lengths[e]
        elif e.endswith('_rev') and e[:-4] in edge_lengths:
            total += edge_lengths[e[:-4]]
        elif e + '_rev' in edge_lengths:
            total += edge_lengths[e + '_rev']
    return total


def analyze_route(route_name, bound, route_edges, kmb_stops, edge_lengths, corrections):
    """分析单条线路的分段 detour"""
    segments = []
    
    stops = kmb_stops[(kmb_stops['route'] == route_name) & 
                      (kmb_stops['bound'] == bound)].sort_values('seq')
    
    if len(stops) < 2:
        return []
    
    stops_list = list(stops.itertuples())
    
    for i in range(len(stops_list) - 1):
        stop_from = stops_list[i]
        stop_to = stops_list[i + 1]
        
        # KMB 距离
        kmb_len = stop_to.cum_dist_m - stop_from.cum_dist_m
        if kmb_len <= 0:
            continue
        
        # 获取站点对应的边
        edge_from = corrections.get((route_name, bound, stop_from.stop_id), '')
        edge_to = corrections.get((route_name, bound, stop_to.stop_id), '')
        
        if not edge_from or not edge_to:
            continue
        
        # 在路由中找到位置
        idx_from = get_edge_in_route(route_edges, edge_from)
        idx_to = get_edge_in_route(route_edges, edge_to)
        
        if idx_from < 0 or idx_to < 0 or idx_from >= idx_to:
            continue
        
        # 计算路由段长度
        bus_len = compute_segment_length(route_edges, idx_from, idx_to, edge_lengths)
        
        # 计算 detour
        ratio = bus_len / kmb_len if kmb_len > 0 else 1.0
        excess = bus_len - kmb_len
        
        segments.append({
            'route': route_name,
            'bound': bound,
            'seg_idx': i,
            'stop_from_seq': stop_from.seq,
            'stop_to_seq': stop_to.seq,
            'stop_from_id': stop_from.stop_id,
            'stop_to_id': stop_to.stop_id,
            'edge_from': edge_from,
            'edge_to': edge_to,
            'kmb_len_m': round(kmb_len, 1),
            'bus_len_m': round(bus_len, 1),
            'excess_m': round(excess, 1),
            'ratio': round(ratio, 3)
        })
    
    return segments


def main():
    print("=" * 80)
    print("[Detour 贡献分析] analyze_detour_segments.py")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    edge_lengths = load_edge_lengths()
    print(f"  - 边长度: {len(edge_lengths)} 条")
    
    corrections = load_corrections()
    print(f"  - 站点纠偏: {len(corrections)} 条")
    
    routes = parse_routes()
    print(f"  - 路由: {len(routes)} 条")
    
    kmb_stops = pd.read_csv(KMB_CSV_FILE)
    print(f"  - KMB 站点: {len(kmb_stops)} 条")
    
    # 分析每条线路
    all_segments = []
    
    for (route_name, bound), route_edges in sorted(routes.items()):
        segments = analyze_route(route_name, bound, route_edges, 
                                  kmb_stops, edge_lengths, corrections)
        all_segments.extend(segments)
    
    df = pd.DataFrame(all_segments)
    
    # 保存完整报表
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[输出] {OUTPUT_FILE}")
    
    # 分析每条线路
    print("\n" + "=" * 80)
    print("[分段 Detour 贡献分析]")
    print("=" * 80)
    
    for (route_name, bound), group in df.groupby(['route', 'bound']):
        print(f"\n{'='*60}")
        print(f"  {route_name} {bound}")
        print(f"{'='*60}")
        
        # 总计
        total_kmb = group['kmb_len_m'].sum()
        total_bus = group['bus_len_m'].sum()
        total_excess = group['excess_m'].sum()
        overall_ratio = total_bus / total_kmb if total_kmb > 0 else 1.0
        
        print(f"\n  总览:")
        print(f"    KMB 总长: {total_kmb/1000:.1f} km")
        print(f"    BUS 路由总长: {total_bus/1000:.1f} km")
        print(f"    总超长: {total_excess/1000:.1f} km")
        print(f"    Overall Scale: {overall_ratio:.3f}")
        
        # Top 5 detour 段
        top5 = group.nlargest(5, 'excess_m')
        
        print(f"\n  Top 5 Detour 贡献段:")
        print(f"  {'Rank':<4} {'Seg':>4} {'KMB(m)':>8} {'BUS(m)':>8} {'Excess(m)':>10} {'Ratio':>6}")
        print(f"  {'-'*44}")
        
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"  {rank:<4} {row['seg_idx']:>4} {row['kmb_len_m']:>8.0f} "
                  f"{row['bus_len_m']:>8.0f} {row['excess_m']:>10.0f} {row['ratio']:>6.2f}")
        
        # Pareto 覆盖率
        if total_excess > 0:
            sorted_excess = group['excess_m'].sort_values(ascending=False)
            top1_cover = sorted_excess.iloc[0] / total_excess * 100 if len(sorted_excess) >= 1 else 0
            top3_cover = sorted_excess.iloc[:3].sum() / total_excess * 100 if len(sorted_excess) >= 3 else 0
            top5_cover = sorted_excess.iloc[:5].sum() / total_excess * 100 if len(sorted_excess) >= 5 else 0
            
            print(f"\n  Pareto 覆盖率:")
            print(f"    Top 1 覆盖: {top1_cover:.1f}%")
            print(f"    Top 3 覆盖: {top3_cover:.1f}%")
            print(f"    Top 5 覆盖: {top5_cover:.1f}%")
            
            if top3_cover >= 70:
                print(f"    ✅ 建议: Top 3 覆盖率 ≥70%，只需针对 top 3 段加 via")
            else:
                print(f"    ⚠️ 建议: Top 3 覆盖率 <70%，可能是系统性偏差")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
