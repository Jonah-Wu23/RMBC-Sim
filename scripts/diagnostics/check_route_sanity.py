#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_route_sanity.py
=====================
检查 SUMO 路由与 KMB 累计里程的一致性

诊断内容：
1. 各 (route, bound) 的 SUMO 总长度 vs KMB 总长度
2. scale factor 是否异常（>1.3 表示可能有 loop/回头段）
3. 检测边序列是否有回头段（同一边多次出现、正反向混杂）

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-26
"""

import json
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_edge_lengths(net_path: str) -> dict:
    """从路网加载边长度"""
    tree = ET.parse(net_path)
    root = tree.getroot()
    
    lengths = {}
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):
            lane = edge.find('lane')
            if lane is not None:
                lengths[eid] = float(lane.get('length', 0))
    return lengths


def parse_route_file(route_path: str) -> dict:
    """解析路由文件，提取边序列"""
    tree = ET.parse(route_path)
    root = tree.getroot()
    
    routes = {}  # (route, bound) -> {edges, stops, vehicle_ids}
    
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id')
        if not vid:
            continue
        
        # 解析线路名和方向
        parts = vid.split('_')
        if len(parts) >= 3:
            route_name = parts[1]
            bound = parts[2].split('.')[0]
        else:
            continue
        
        key = (route_name, bound)
        
        # 获取边序列
        route_elem = vehicle.find('route')
        if route_elem is not None:
            edges = route_elem.get('edges', '').split()
        else:
            edges = []
        
        # 获取站点序列
        stops = [stop.get('busStop') for stop in vehicle.findall('stop') if stop.get('busStop')]
        
        if key not in routes:
            routes[key] = {'edges': edges, 'stops': stops, 'vehicle_ids': []}
        routes[key]['vehicle_ids'].append(vid)
    
    return routes


def load_kmb_distances(kmb_csv_path: str) -> dict:
    """加载 KMB 累计里程数据"""
    df = pd.read_csv(kmb_csv_path)
    
    # 每个 (route, bound) 的最大累计里程
    distances = {}
    for (route, bound), g in df.groupby(['route', 'bound']):
        distances[(route, bound)] = {
            'max_cum_dist': g['cum_dist_m'].max(),
            'n_stops': len(g),
            'stops': g['stop_id'].tolist()
        }
    return distances


def check_edge_anomalies(edges: list) -> dict:
    """检查边序列异常"""
    # 统计边出现次数
    edge_counts = Counter(edges)
    duplicates = {e: c for e, c in edge_counts.items() if c > 1}
    
    # 检查正反向混杂
    base_edges = set()
    rev_conflicts = []
    for e in edges:
        if e.endswith('_rev'):
            base = e[:-4]
            if base in base_edges:
                rev_conflicts.append((base, e))
        else:
            if e + '_rev' in base_edges:
                rev_conflicts.append((e, e + '_rev'))
        base_edges.add(e.replace('_rev', ''))
    
    return {
        'duplicates': duplicates,
        'reverse_conflicts': rev_conflicts,
        'total_edges': len(edges),
        'unique_edges': len(set(edges))
    }


def main():
    route_path = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes.rou.xml"
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    
    print("=" * 80)
    print("[Sanity Check] SUMO 路由与 KMB 累计里程一致性检查")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    edge_lengths = load_edge_lengths(str(net_path))
    print(f"  - 边长度: {len(edge_lengths)} 条")
    
    routes = parse_route_file(str(route_path))
    print(f"  - 路由: {len(routes)} 个 (route, bound)")
    
    kmb_data = load_kmb_distances(str(kmb_csv_path))
    print(f"  - KMB: {len(kmb_data)} 个 (route, bound)")
    
    # 检查每个路由
    print("\n" + "=" * 80)
    print("[详细诊断]")
    print("-" * 80)
    
    warnings = []
    
    for key, rdata in sorted(routes.items()):
        route, bound = key
        edges = rdata['edges']
        
        # 计算 SUMO 总长度
        sumo_length = 0
        missing_cnt = 0
        for e in edges:
            if e in edge_lengths:
                sumo_length += edge_lengths[e]
            elif e + '_rev' in edge_lengths:
                sumo_length += edge_lengths[e + '_rev']
            elif e[:-4] in edge_lengths and e.endswith('_rev'):
                sumo_length += edge_lengths[e[:-4]]
            else:
                missing_cnt += 1
        
        # 获取 KMB 数据
        kmb_info = kmb_data.get(key, {})
        kmb_length = kmb_info.get('max_cum_dist', 0)
        n_stops_kmb = kmb_info.get('n_stops', 0)
        
        # 计算 scale
        scale = sumo_length / kmb_length if kmb_length > 0 else float('inf')
        
        # 检查边异常
        anomalies = check_edge_anomalies(edges)
        
        # 输出
        print(f"\n[{route} {bound}]")
        print(f"  - SUMO: {sumo_length/1000:.1f} km ({len(edges)} edges, {missing_cnt} missing)")
        print(f"  - KMB:  {kmb_length/1000:.1f} km ({n_stops_kmb} stops)")
        print(f"  - Scale: {scale:.3f}")
        
        # 边异常
        if anomalies['duplicates']:
            top_dups = sorted(anomalies['duplicates'].items(), key=lambda x: -x[1])[:5]
            print(f"  - ⚠️ 重复边: {len(anomalies['duplicates'])} 条")
            for e, c in top_dups:
                print(f"       {e}: {c}x")
        
        if anomalies['reverse_conflicts']:
            print(f"  - ⚠️ 正反向混杂: {len(anomalies['reverse_conflicts'])} 对")
            for base, rev in anomalies['reverse_conflicts'][:3]:
                print(f"       {base} ↔ {rev}")
        
        # 警告
        if scale > 1.5:
            warn_msg = f"[{route} {bound}] scale={scale:.2f} >> 1.5，疑似包含往返/环线"
            warnings.append(warn_msg)
            print(f"  - 🚨 {warn_msg}")
        elif scale < 0.7:
            warn_msg = f"[{route} {bound}] scale={scale:.2f} << 0.7，疑似路由不完整"
            warnings.append(warn_msg)
            print(f"  - 🚨 {warn_msg}")
        
        # 输出前10条边和后10条边
        if len(edges) > 20:
            print(f"  - 边序列: {edges[:5]}...{edges[-5:]}")
        else:
            print(f"  - 边序列: {edges[:10]}{'...' if len(edges) > 10 else ''}")
    
    # 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("-" * 80)
    
    if warnings:
        print(f"⚠️ 发现 {len(warnings)} 个潜在问题：")
        for w in warnings:
            print(f"  - {w}")
        print("\n建议检查：")
        print("  1. rou.xml 中的 vehicle/route 定义是否包含往返")
        print("  2. bound 字段是否正确（inbound vs outbound）")
        print("  3. route edges 是否来自 routeDistribution（可能包含多个备选路线）")
    else:
        print("✅ 所有路由 scale factor 在合理范围内 (0.7-1.5)")


if __name__ == "__main__":
    main()
