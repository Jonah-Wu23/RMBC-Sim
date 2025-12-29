#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
locate_reversals.py
===================
折返定位脚本 - 识别路线中的折返点并输出报表

折返定义：
- 同一 road 的相反方向边 (e.g., `95649` → `95649_rev`)
- 4 edge 内重复经过同一 edge (短环回)

输出：
- 每个折返的 edge 对
- 最近 stop ID 及距离
- 折返分类 (STOP_DRIVEN / PATH_DRIVEN)

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-27
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v1.net.xml"
ROUTE_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_via.rou.xml"
CORRECTIONS_FILE = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"
KMB_CSV_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
OUTPUT_FILE = PROJECT_ROOT / "logs" / "reversal_report.csv"

STOP_DRIVEN_THRESHOLD = 150  # meters


def load_edge_positions():
    """加载边的中点坐标（用于距离计算）"""
    tree = ET.parse(str(NET_FILE))
    root = tree.getroot()
    
    edge_pos = {}
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):
            lane = edge.find('lane')
            if lane is not None:
                shape = lane.get('shape', '')
                if shape:
                    points = shape.split()
                    if len(points) >= 2:
                        # 取中点
                        mid_idx = len(points) // 2
                        x, y = map(float, points[mid_idx].split(','))
                        edge_pos[eid] = (x, y)
    return edge_pos


def load_stop_positions(corrections):
    """加载站点位置（基于其所在 edge）"""
    edge_pos = load_edge_positions()
    
    stop_pos = {}
    for (route, bound, stop_id), edge in corrections.items():
        if edge in edge_pos:
            stop_pos[(route, bound, stop_id)] = {
                'edge': edge,
                'pos': edge_pos[edge],
                'is_rev': edge.endswith('_rev')
            }
    return stop_pos, edge_pos


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


def get_base_edge(edge_id):
    """获取边的基础 ID（去掉 _rev 后缀）"""
    if edge_id.endswith('_rev'):
        return edge_id[:-4]
    return edge_id


def find_reversals(edges):
    """
    找出路径中的折返点
    
    折返定义：
    1. 同一 road 的相反方向边
    2. 4 edge 内重复经过同一 edge
    """
    reversals = []
    
    # 检查同 road 反向边
    for i in range(len(edges) - 1):
        e1, e2 = edges[i], edges[i + 1]
        base1, base2 = get_base_edge(e1), get_base_edge(e2)
        
        if base1 == base2 and e1 != e2:
            reversals.append({
                'index': i,
                'edge_from': e1,
                'edge_to': e2,
                'type': 'DIRECTION_SWITCH'
            })
    
    # 检查短环回（4 edge 内重复）
    for i in range(len(edges)):
        base_i = get_base_edge(edges[i])
        for j in range(i + 2, min(i + 5, len(edges))):
            if get_base_edge(edges[j]) == base_i:
                # 检查是否已记录为反向折返
                already_recorded = any(
                    r['index'] >= i and r['index'] < j 
                    for r in reversals
                )
                if not already_recorded:
                    reversals.append({
                        'index': i,
                        'edge_from': edges[i],
                        'edge_to': edges[j],
                        'type': 'SHORT_LOOP',
                        'loop_length': j - i
                    })
    
    return reversals


def find_nearest_stop(reversal_edge, stop_pos, edge_pos, route, bound):
    """找到距离折返点最近的站点"""
    if reversal_edge not in edge_pos:
        return None, None, None
    
    rev_pos = edge_pos[reversal_edge]
    
    nearest_stop = None
    nearest_dist = float('inf')
    nearest_is_rev = None
    
    for (r, b, stop_id), info in stop_pos.items():
        if r == route and b == bound:
            dx = rev_pos[0] - info['pos'][0]
            dy = rev_pos[1] - info['pos'][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_stop = stop_id
                nearest_is_rev = info['is_rev']
    
    return nearest_stop, nearest_dist, nearest_is_rev


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


def main():
    print("=" * 80)
    print("[折返定位分析] locate_reversals.py")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    corrections = load_corrections()
    print(f"  - 站点纠偏: {len(corrections)} 条")
    
    stop_pos, edge_pos = load_stop_positions(corrections)
    print(f"  - 站点位置: {len(stop_pos)} 个")
    print(f"  - 边位置: {len(edge_pos)} 条")
    
    routes = parse_routes()
    print(f"  - 路由: {len(routes)} 条")
    
    # 分析折返
    print("\n[折返分析]")
    all_reversals = []
    
    for (route_name, bound), edges in sorted(routes.items()):
        reversals = find_reversals(edges)
        
        print(f"\n  {route_name} {bound}: {len(reversals)} 个折返")
        
        for rev in reversals:
            # 找最近站点
            nearest_stop, dist, is_rev = find_nearest_stop(
                rev['edge_from'], stop_pos, edge_pos, route_name, bound
            )
            
            # 分类
            if dist is not None and dist <= STOP_DRIVEN_THRESHOLD:
                reversal_type = 'STOP_DRIVEN'
            else:
                reversal_type = 'PATH_DRIVEN'
            
            rev_record = {
                'route': route_name,
                'bound': bound,
                'index': rev['index'],
                'edge_from': rev['edge_from'],
                'edge_to': rev['edge_to'],
                'reversal_pattern': rev['type'],
                'nearest_stop': nearest_stop,
                'stop_distance_m': round(dist, 1) if dist else None,
                'stop_edge_is_rev': is_rev,
                'reversal_type': reversal_type,
            }
            all_reversals.append(rev_record)
            
            # 打印详情
            stop_info = f"stop={nearest_stop}, dist={dist:.0f}m" if nearest_stop else "no stop"
            is_rev_str = "⚠️_rev" if is_rev else "✅正向"
            print(f"    [{rev['index']:3d}] {rev['edge_from']} → {rev['edge_to']}")
            print(f"         {reversal_type} | {stop_info} | {is_rev_str}")
    
    # 保存报表
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_reversals)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[输出] {OUTPUT_FILE}")
    
    # 汇总
    print("\n" + "=" * 80)
    print("[汇总]")
    print("-" * 80)
    
    for (route_name, bound), group in df.groupby(['route', 'bound']):
        total = len(group)
        stop_driven = len(group[group['reversal_type'] == 'STOP_DRIVEN'])
        path_driven = len(group[group['reversal_type'] == 'PATH_DRIVEN'])
        rev_stops = len(group[group['stop_edge_is_rev'] == True])
        
        print(f"  {route_name} {bound}:")
        print(f"    折返总数: {total}")
        print(f"    STOP_DRIVEN: {stop_driven} | PATH_DRIVEN: {path_driven}")
        print(f"    站点在 _rev 边: {rev_stops}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
