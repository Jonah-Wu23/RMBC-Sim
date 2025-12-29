#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_core_detour.py
======================
分析 68X inbound 核心段的偏离贡献，输出 Top5 超长段
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml"
ROUTE_FILE = PROJECT_ROOT / "build" / "68X_inbound_core.rou.xml"
KMB_CSV = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
CORRECTIONS_FILE = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"


def load_edge_lengths():
    """加载边长度"""
    tree = ET.parse(str(NET_FILE))
    lengths = {}
    for edge in tree.getroot().findall('.//edge'):
        eid = edge.get('id')
        if eid:
            lane = edge.find('lane')
            if lane is not None:
                lengths[eid] = float(lane.get('length', 0))
    return lengths


def get_core_stops():
    """获取核心段 stops"""
    df = pd.read_csv(KMB_CSV)
    corrections = pd.read_csv(CORRECTIONS_FILE)
    
    # 加载 cropped 网络边集合
    tree = ET.parse(str(NET_FILE))
    cropped_edges = {e.get('id') for e in tree.getroot().findall('.//edge')}
    
    # 构建 corrections 字典
    corr_dict = {}
    for _, row in corrections.iterrows():
        key = (row['route'], row['bound'], row['stop_id'])
        edge = row['fixed_edge'] if pd.notna(row['fixed_edge']) else row['orig_edge']
        if pd.notna(edge):
            corr_dict[key] = str(edge)
    
    # 获取 68X inbound 核心段 stops
    subset = df[(df['route'] == '68X') & (df['bound'] == 'inbound')].sort_values('seq')
    
    core_stops = []
    for _, row in subset.iterrows():
        edge = corr_dict.get(('68X', 'inbound', row['stop_id']), '')
        if edge and edge in cropped_edges:
            core_stops.append({
                'seq': row['seq'],
                'stop_id': row['stop_id'],
                'edge': edge,
                'cum_dist_m': row['cum_dist_m'],
                'stop_name': row.get('name_en', '') if 'name_en' in row else ''
            })
    
    return core_stops


def analyze_segments(route_edges, core_stops, edge_lengths):
    """分析每个站点间隔的偏离"""
    segments = []
    
    for i in range(len(core_stops) - 1):
        from_stop = core_stops[i]
        to_stop = core_stops[i + 1]
        
        # KMB 长度
        kmb_len = to_stop['cum_dist_m'] - from_stop['cum_dist_m']
        
        # 找到路由中对应的边
        try:
            from_idx = route_edges.index(from_stop['edge'])
            to_idx = route_edges.index(to_stop['edge'])
        except ValueError:
            # 尝试 _rev 版本
            from_edge = from_stop['edge']
            to_edge = to_stop['edge']
            
            from_idx = -1
            to_idx = -1
            for idx, e in enumerate(route_edges):
                if e == from_edge or e == from_edge + '_rev' or e == from_edge.replace('_rev', ''):
                    from_idx = idx
                if e == to_edge or e == to_edge + '_rev' or e == to_edge.replace('_rev', ''):
                    to_idx = idx
            
            if from_idx == -1 or to_idx == -1:
                continue
        
        # 确保 to_idx > from_idx
        if to_idx <= from_idx:
            # 可能是反向，跳过
            continue
        
        # 计算 SUMO 长度
        bus_len = 0
        segment_edges = route_edges[from_idx:to_idx + 1]
        for e in segment_edges:
            if e in edge_lengths:
                bus_len += edge_lengths[e]
            elif e.endswith('_rev') and e[:-4] in edge_lengths:
                bus_len += edge_lengths[e[:-4]]
            elif e + '_rev' in edge_lengths:
                bus_len += edge_lengths[e + '_rev']
        
        ratio = bus_len / kmb_len if kmb_len > 0 else 0
        excess = bus_len - kmb_len
        
        segments.append({
            'segment': f"seg{i+1}",
            'from_stop': from_stop['stop_id'][:8],
            'to_stop': to_stop['stop_id'][:8],
            'from_edge': from_stop['edge'],
            'to_edge': to_stop['edge'],
            'kmb_len': kmb_len,
            'bus_len': bus_len,
            'ratio': ratio,
            'excess': excess,
            'edge_count': len(segment_edges)
        })
    
    return segments


def main():
    print("="*70)
    print("68X Inbound Core 偏离贡献分析")
    print("="*70)
    
    # 加载数据
    edge_lengths = load_edge_lengths()
    core_stops = get_core_stops()
    
    print(f"\n核心段站点数: {len(core_stops)}")
    
    # 加载路由
    tree = ET.parse(str(ROUTE_FILE))
    route_elem = tree.getroot().find('.//vehicle/route')
    if route_elem is None:
        print("未找到路由")
        return
    
    route_edges = route_elem.get('edges', '').split()
    print(f"路由边数: {len(route_edges)}")
    
    # 计算总长度
    total_bus_len = sum(edge_lengths.get(e, edge_lengths.get(e[:-4] if e.endswith('_rev') else e+'_rev', 0)) for e in route_edges)
    total_kmb_len = core_stops[-1]['cum_dist_m'] - core_stops[0]['cum_dist_m']
    total_scale = total_bus_len / total_kmb_len
    
    print(f"\n总 SUMO 长度: {total_bus_len:.0f}m")
    print(f"总 KMB 长度: {total_kmb_len:.0f}m")
    print(f"Scale: {total_scale:.3f}")
    print(f"目标: <1.700")
    print(f"需降低: {total_bus_len - total_kmb_len * 1.7:.0f}m")
    
    # 分析各段
    segments = analyze_segments(route_edges, core_stops, edge_lengths)
    
    # 排序：按 excess 降序
    segments_sorted = sorted(segments, key=lambda x: x['excess'], reverse=True)
    
    # 计算 Top5 覆盖率
    total_excess = sum(s['excess'] for s in segments if s['excess'] > 0)
    
    print("\n" + "="*70)
    print("Top 5 超长段 (按 excess 排序)")
    print("="*70)
    
    cumulative_excess = 0
    for i, seg in enumerate(segments_sorted[:5]):
        cumulative_excess += max(0, seg['excess'])
        coverage = cumulative_excess / total_excess * 100 if total_excess > 0 else 0
        
        print(f"\n#{i+1} {seg['segment']}: {seg['from_edge']} → {seg['to_edge']}")
        print(f"   bus_len={seg['bus_len']:.0f}m, kmb_len={seg['kmb_len']:.0f}m")
        print(f"   ratio={seg['ratio']:.2f}, excess={seg['excess']:.0f}m")
        print(f"   edges={seg['edge_count']}")
        print(f"   累计覆盖: {coverage:.1f}%")
    
    # Pareto 分析
    print("\n" + "="*70)
    print("Pareto 分析")
    print("="*70)
    
    top3_excess = sum(max(0, s['excess']) for s in segments_sorted[:3])
    top5_excess = sum(max(0, s['excess']) for s in segments_sorted[:5])
    
    print(f"总超长: {total_excess:.0f}m")
    print(f"Top 3 覆盖: {top3_excess/total_excess*100:.1f}% ({top3_excess:.0f}m)")
    print(f"Top 5 覆盖: {top5_excess/total_excess*100:.1f}% ({top5_excess:.0f}m)")
    
    if top3_excess/total_excess >= 0.7:
        print("\n➡️  Top3 覆盖 ≥70%: 只需动 1-2 个 via 点即可达标")
    else:
        print("\n➡️  需检查 stop 映射/方向一致性或边界问题")


if __name__ == '__main__':
    main()
