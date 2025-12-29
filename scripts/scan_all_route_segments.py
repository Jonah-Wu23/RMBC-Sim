#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scan_all_route_segments.py
==========================
全量扫描 68X/960 的 inbound/outbound 所有相邻站点段
诊断问题类型：NoPath / RuleBlocked / Detour / OK

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-27
"""

import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import math
import argparse

try:
    import sumolib
except ImportError:
    print("⚠️ sumolib 未安装")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def get_reverse_edge_id(edge_id):
    """获取反向边 ID"""
    if edge_id.endswith('_rev'):
        return edge_id[:-4]
    else:
        return edge_id + '_rev'


def has_immediate_reversal(edge_ids):
    """检查路径是否包含立即折返"""
    reversals = []
    for i in range(len(edge_ids) - 1):
        e1, e2 = edge_ids[i], edge_ids[i + 1]
        if (e1.endswith('_rev') and e1[:-4] == e2) or \
           (e2.endswith('_rev') and e2[:-4] == e1):
            reversals.append(f"{e1}→{e2}")
    return reversals


def check_alternative_path(net, from_edge, to_edge, forbidden_edge):
    """尝试绕过某条边"""
    try:
        outgoing = list(from_edge.getOutgoing())
        best_len = float('inf')
        best_route = []
        
        for next_edge in outgoing:
            if next_edge.getID() == forbidden_edge:
                continue
            route, cost = net.getShortestPath(next_edge, to_edge)
            if route:
                total_len = from_edge.getLength() + sum(e.getLength() for e in route)
                if total_len < best_len:
                    best_len = total_len
                    best_route = [from_edge.getID()] + [e.getID() for e in route]
        
        return best_len, best_route
    except Exception:
        return float('inf'), []


def diagnose_segment(net, from_edge_id, to_edge_id, kmb_len):
    """
    诊断一个段的问题类型
    
    Returns:
        dict with diagnosis results
    """
    result = {
        'from_edge': from_edge_id,
        'to_edge': to_edge_id,
        'kmb_len': kmb_len,
        'test1_len': None,
        'test1_ratio': None,
        'test1_reversals': [],
        'test2_len': None,
        'test2_ratio': None,
        'node_dist': None,
        'type': 'UNKNOWN',
        'recommendation': ''
    }
    
    if from_edge_id == to_edge_id:
        result['type'] = 'SAME_EDGE'
        result['test1_len'] = 0
        result['test1_ratio'] = 0
        return result
    
    # 处理 internal edge
    if from_edge_id.startswith(':'):
        parts = from_edge_id[1:].rsplit('_', 1)
        junction_id = parts[0] if len(parts) >= 1 else from_edge_id[1:]
        try:
            junction = net.getNode(junction_id)
            inc_edges = [e.getID() for e in junction.getIncoming() if not e.getID().startswith(':')]
            if inc_edges:
                from_edge_id = inc_edges[0]
        except Exception:
            result['type'] = 'INTERNAL_EDGE_ERROR'
            return result
    
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except Exception as e:
        result['type'] = 'EDGE_NOT_FOUND'
        return result
    
    # 计算节点距离
    try:
        from_to_node = from_edge.getToNode()
        to_from_node = to_edge.getFromNode()
        from_coord = from_to_node.getCoord()
        to_coord = to_from_node.getCoord()
        result['node_dist'] = math.sqrt((from_coord[0] - to_coord[0])**2 + (from_coord[1] - to_coord[1])**2)
    except Exception:
        pass
    
    # Test-1: 正常最短路
    try:
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            edge_ids = [e.getID() for e in route]
            total_len = sum(e.getLength() for e in route)
            result['test1_len'] = total_len
            result['test1_ratio'] = total_len / kmb_len if kmb_len > 0 else 0
            result['test1_reversals'] = has_immediate_reversal(edge_ids)
        else:
            result['type'] = 'NO_PATH'
            result['recommendation'] = 'BRIDGE_EDGE'
            return result
    except Exception as e:
        result['type'] = 'ROUTING_ERROR'
        return result
    
    # 如果没有折返，直接返回
    if not result['test1_reversals']:
        if result['test1_ratio'] <= 1.5:
            result['type'] = 'OK'
        elif result['test1_ratio'] <= 3.0:
            result['type'] = 'MINOR_DETOUR'
            result['recommendation'] = 'ACCEPTABLE'
        else:
            result['type'] = 'MAJOR_DETOUR'
            result['recommendation'] = 'CHECK_MAPPING'
        return result
    
    # Test-2: 禁止立即折返
    rev_edge = result['test1_reversals'][0].split('→')[1] if '→' in result['test1_reversals'][0] else None
    if rev_edge:
        test2_len, test2_route = check_alternative_path(net, from_edge, to_edge, rev_edge)
        if test2_len < float('inf'):
            result['test2_len'] = test2_len
            result['test2_ratio'] = test2_len / kmb_len if kmb_len > 0 else 0
            
            if result['test2_ratio'] <= 2.0:
                result['type'] = 'ROUTING_STRATEGY'
                result['recommendation'] = 'VIA_ROUTING'
            elif result['test2_ratio'] <= 5.0:
                # Changed: This is reachable but long detour, not RULE_BLOCKED
                result['type'] = 'LONG_DETOUR'
                result['recommendation'] = 'VIA_ROUTING'
            else:
                # Changed: This is major detour, not TRAFFIC_RULE
                result['type'] = 'MAJOR_DETOUR'
                result['recommendation'] = 'CHECK_MAPPING'
        else:
            result['type'] = 'NO_ALT_PATH'
            result['recommendation'] = 'BRIDGE_EDGE'
    else:
        result['type'] = 'REVERSAL_DETECTED'
        result['recommendation'] = 'VIA_ROUTING'
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default=str(PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched.net.xml"),
                       help='Path to net.xml file')
    args = parser.parse_args()
    
    net_path = Path(args.net)
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops_irn.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    
    print("=" * 100)
    print("[全量路由问题扫描] 68X + 960 (in/out)")
    print(f"  网络: {net_path.name}")
    print("=" * 100)
    
    # 加载数据
    print("\n[加载数据]")
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    print(f"  路网边数: {len(list(net.getEdges()))}")
    
    stop_to_edge = load_stop_edges(bus_stops_path)
    print(f"  站点数: {len(stop_to_edge)}")
    
    df = pd.read_csv(kmb_csv_path)
    
    # 扫描所有路线的所有段
    results = []
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            route_df = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            
            if len(route_df) == 0:
                continue
            
            print(f"\n[{route} {bound}] {len(route_df)} 站点")
            
            for i in range(len(route_df) - 1):
                row1 = route_df.iloc[i]
                row2 = route_df.iloc[i + 1]
                
                stop1 = row1['stop_id']
                stop2 = row2['stop_id']
                seq_from = int(row1['seq'])
                seq_to = int(row2['seq'])
                kmb_len = row2['link_dist_m']
                
                edge1 = stop_to_edge.get(stop1, 'UNKNOWN')
                edge2 = stop_to_edge.get(stop2, 'UNKNOWN')
                
                if edge1 == 'UNKNOWN' or edge2 == 'UNKNOWN':
                    results.append({
                        'route': route,
                        'bound': bound,
                        'seq': f"{seq_from}→{seq_to}",
                        'from_edge': edge1,
                        'to_edge': edge2,
                        'kmb_len': kmb_len,
                        'type': 'STOP_NOT_MAPPED',
                        'ratio': None,
                        'node_dist': None,
                        'recommendation': 'FIX_STOP_MAPPING'
                    })
                    continue
                
                diag = diagnose_segment(net, edge1, edge2, kmb_len)
                
                results.append({
                    'route': route,
                    'bound': bound,
                    'seq': f"{seq_from}→{seq_to}",
                    'from_edge': diag['from_edge'],
                    'to_edge': diag['to_edge'],
                    'kmb_len': kmb_len,
                    'type': diag['type'],
                    'ratio': diag.get('test2_ratio') or diag.get('test1_ratio'),
                    'node_dist': diag.get('node_dist'),
                    'recommendation': diag['recommendation']
                })
    
    # 输出结果
    results_df = pd.DataFrame(results)
    
    # 按类型统计
    print("\n" + "=" * 100)
    print("[统计汇总]")
    print("=" * 100)
    
    type_counts = results_df['type'].value_counts()
    print(f"\n类型分布:")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")
    
    # 输出需要修复的段
    print("\n" + "=" * 100)
    print("[需要修复的段 (NoPath / NO_ALT_PATH)]")
    print("=" * 100)
    
    nopath_df = results_df[results_df['type'].isin(['NO_PATH', 'NO_ALT_PATH'])]
    if len(nopath_df) > 0:
        for _, row in nopath_df.iterrows():
            print(f"  {row['route']} {row['bound']} 段{row['seq']}: {row['from_edge']} → {row['to_edge']}")
            print(f"    KMB={row['kmb_len']:.0f}m, node_dist={row['node_dist']:.1f}m" if row['node_dist'] else f"    KMB={row['kmb_len']:.0f}m")
            print(f"    推荐: {row['recommendation']}")
    else:
        print("  无 (所有硬缺陷已修复!)")
    
    # 输出规则限制/大绕行段
    print("\n" + "=" * 100)
    print("[规则限制/大绕行段 (ratio > 2)]")
    print("=" * 100)
    
    detour_df = results_df[(results_df['ratio'].notna()) & (results_df['ratio'] > 2.0)]
    detour_df = detour_df.sort_values('ratio', ascending=False)
    
    if len(detour_df) > 0:
        for _, row in detour_df.iterrows():
            print(f"  {row['route']} {row['bound']} 段{row['seq']}: ratio={row['ratio']:.2f}")
            print(f"    {row['from_edge']} → {row['to_edge']}")
            print(f"    type={row['type']}, 推荐: {row['recommendation']}")
    else:
        print("  无")
    
    # 保存完整结果
    output_path = PROJECT_ROOT / "logs" / "route_segment_diagnosis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[完整诊断结果已保存] {output_path}")
    
    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
