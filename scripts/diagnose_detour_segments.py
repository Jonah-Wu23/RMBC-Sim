#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_detour_segments.py
===========================
定位哪些相邻站点段导致 scale 爆炸

输出：
1. 68X inbound 的前 8 个 stop 的 id + edge
2. ratio 最大的 Top 10 段（包含是否出现折返）
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 加载边长度
def load_edge_lengths(net_path):
    tree = ET.parse(str(net_path))
    root = tree.getroot()
    lengths = {}
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):
            lane = edge.find('lane')
            if lane is not None:
                lengths[eid] = float(lane.get('length', 0))
    return lengths

# 加载站点边映射
def load_stop_edges(bus_stops_path):
    tree = ET.parse(str(bus_stops_path))
    root = tree.getroot()
    stop_to_edge = {}
    stop_to_lane = {}
    for stop in root.findall('.//busStop'):
        stop_id = stop.get('id')
        lane = stop.get('lane', '')
        stop_to_lane[stop_id] = lane
        if lane.startswith(':'):
            edge = lane.rsplit('_', 1)[0]
        else:
            edge = lane.rsplit('_', 1)[0]
        stop_to_edge[stop_id] = edge
    return stop_to_edge, stop_to_lane

# 加载分段路由
def load_segment_routes(seg_route_file):
    tree = ET.parse(str(seg_route_file))
    root = tree.getroot()
    segments = {}
    for vehicle in root.findall('.//vehicle'):
        vid = vehicle.get('id')
        route_elem = vehicle.find('route')
        if route_elem is not None:
            edges = route_elem.get('edges', '').split()
            segments[vid] = edges
    return segments

# 检查折返模式
def check_reversal(edges):
    reversals = []
    for i in range(len(edges) - 1):
        e, n = edges[i], edges[i+1]
        if e.endswith('_rev') and e[:-4] == n:
            reversals.append(f"{e}→{n}")
        elif n.endswith('_rev') and n[:-4] == e:
            reversals.append(f"{e}→{n}")
        elif e == n:
            reversals.append(f"{e}(dup)")
    return reversals

def main():
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    seg_route_file = PROJECT_ROOT / "build" / "68X_inbound.seg_routes.xml"
    
    print("=" * 80)
    print("Top-K Detour 段定位器")
    print("=" * 80)
    
    # 加载数据
    edge_lengths = load_edge_lengths(net_path)
    stop_to_edge, stop_to_lane = load_stop_edges(bus_stops_path)
    
    df = pd.read_csv(kmb_csv_path)
    df_68x_in = df[(df['route'] == '68X') & (df['bound'] == 'inbound')].sort_values('seq')
    
    # 1. 输出前 8 个 stop
    print("\n## 68X inbound 前 8 个 stop (id + lane)")
    print("-" * 60)
    for i, (_, row) in enumerate(df_68x_in.iterrows()):
        if i >= 8:
            break
        stop_id = row['stop_id']
        lane = stop_to_lane.get(stop_id, 'N/A')
        edge = stop_to_edge.get(stop_id, 'N/A')
        name = str(row['stop_name_en'])[:30]
        print(f"  seq={int(row['seq']):2d}: {stop_id} | lane={lane:20s} | edge={edge:15s} | {name}")
    
    # 2. 加载分段路由并计算每段 ratio
    print("\n## 分段 ratio 分析")
    print("-" * 60)
    
    if not seg_route_file.exists():
        print(f"  ⚠️  分段路由文件不存在: {seg_route_file}")
        return
    
    segments = load_segment_routes(seg_route_file)
    
    # 准备 KMB 距离
    stops_list = list(df_68x_in.itertuples())
    
    results = []
    for i in range(len(stops_list) - 1):
        row1, row2 = stops_list[i], stops_list[i+1]
        seq1, seq2 = int(row1.seq), int(row2.seq)
        stop1_id, stop2_id = row1.stop_id, row2.stop_id
        edge1 = stop_to_edge.get(stop1_id, 'N/A')
        edge2 = stop_to_edge.get(stop2_id, 'N/A')
        
        # KMB 距离
        kmb_len = row2.cum_dist_m - row1.cum_dist_m
        if kmb_len <= 0:
            kmb_len = 100  # 避免除零
        
        # 找对应的分段路由
        seg_id = None
        for vid in segments:
            if f"_{seq1:02d}_{seq2:02d}" in vid:
                seg_id = vid
                break
        
        if seg_id is None:
            # 同边或失败
            results.append({
                'seq': f"{seq1}->{seq2}",
                'edge1': edge1,
                'edge2': edge2,
                'kmb_len': kmb_len,
                'sumo_len': 0,
                'ratio': 0,
                'reversals': 'NO_ROUTE',
                'stop1_name': str(row1.stop_name_en)[:20],
                'stop2_name': str(row2.stop_name_en)[:20],
            })
            continue
        
        seg_edges = segments[seg_id]
        
        # 计算 SUMO 路径长度
        sumo_len = 0
        for e in seg_edges:
            if e in edge_lengths:
                sumo_len += edge_lengths[e]
            elif e.endswith('_rev') and e[:-4] in edge_lengths:
                sumo_len += edge_lengths[e[:-4]]
            elif e + '_rev' in edge_lengths:
                sumo_len += edge_lengths[e + '_rev']
        
        ratio = sumo_len / kmb_len if kmb_len > 0 else 0
        reversals = check_reversal(seg_edges)
        
        results.append({
            'seq': f"{seq1}->{seq2}",
            'edge1': edge1,
            'edge2': edge2,
            'kmb_len': kmb_len,
            'sumo_len': sumo_len,
            'ratio': ratio,
            'reversals': ', '.join(reversals) if reversals else 'none',
            'stop1_name': str(row1.stop_name_en)[:20],
            'stop2_name': str(row2.stop_name_en)[:20],
        })
    
    # 排序并输出 Top 10
    results_sorted = sorted(results, key=lambda x: -x['ratio'])
    
    print("\n## Top 10 Detour 段 (ratio 最大)")
    print("-" * 100)
    print(f"{'SEQ':<8} {'EDGE1':<18} {'EDGE2':<18} {'KMB':>8} {'SUMO':>10} {'RATIO':>7} REVERSALS")
    print("-" * 100)
    
    for r in results_sorted[:10]:
        print(f"{r['seq']:<8} {r['edge1']:<18} {r['edge2']:<18} {r['kmb_len']:>8.0f} {r['sumo_len']:>10.0f} {r['ratio']:>7.2f} {r['reversals'][:40]}")
    
    print("\n## Top 5 详细信息")
    print("-" * 80)
    for r in results_sorted[:5]:
        print(f"\n段 {r['seq']}: {r['stop1_name']} -> {r['stop2_name']}")
        print(f"  edge1={r['edge1']}, edge2={r['edge2']}")
        print(f"  KMB={r['kmb_len']:.0f}m, SUMO={r['sumo_len']:.0f}m, ratio={r['ratio']:.2f}")
        print(f"  折返: {r['reversals']}")


if __name__ == '__main__':
    main()
