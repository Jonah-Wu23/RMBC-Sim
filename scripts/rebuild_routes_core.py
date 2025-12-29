#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_routes_core.py
======================
核心段路由生成：只使用 bbox 内的 stops，基于 hk_cropped.net.xml
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 使用 CROPPED 网络！
NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml"
KMB_CSV_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
CORRECTIONS_FILE = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"
BUILD_DIR = PROJECT_ROOT / "build"
OUTPUT_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_core.rou.xml"


def load_cropped_edges():
    """加载 cropped 网络的边集合"""
    tree = ET.parse(str(NET_FILE))
    edges = set()
    for edge in tree.getroot().findall('.//edge'):
        eid = edge.get('id')
        if eid:
            edges.add(eid)
    return edges


def load_corrections():
    """加载纠偏后的端点"""
    df = pd.read_csv(CORRECTIONS_FILE)
    corrections = {}
    for _, row in df.iterrows():
        key = (row['route'], row['bound'], row['stop_id'])
        edge = row['fixed_edge'] if pd.notna(row['fixed_edge']) else row['orig_edge']
        if pd.notna(edge):
            corrections[key] = str(edge)
    return corrections


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


def get_core_stops(route, bound, corrections, cropped_edges):
    """获取在核心区内的 stops"""
    df = pd.read_csv(KMB_CSV_FILE)
    subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
    
    core_stops = []
    for _, row in subset.iterrows():
        edge = corrections.get((route, bound, row['stop_id']), '')
        if not edge or edge.startswith(':'):
            continue
        
        # 检查边是否在 cropped 网络中
        if edge in cropped_edges or edge.replace('_rev', '') in cropped_edges:
            core_stops.append({
                'seq': row['seq'],
                'stop_id': row['stop_id'],
                'edge': edge,
                'cum_dist_m': row['cum_dist_m']
            })
    
    return core_stops


def generate_core_trips(corrections, cropped_edges, build_dir):
    """生成核心段 trips"""
    build_dir.mkdir(parents=True, exist_ok=True)
    trip_files = {}
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            core_stops = get_core_stops(route, bound, corrections, cropped_edges)
            
            if len(core_stops) < 2:
                print(f"  ⚠️ {route} {bound}: 核心区内站点不足")
                continue
            
            # 去重连续相同边
            edges = [core_stops[0]['edge']]
            for s in core_stops[1:]:
                if s['edge'] != edges[-1] and not s['edge'].startswith('bridge_'):
                    edges.append(s['edge'])
            
            if len(edges) < 2:
                continue
            
            # 生成 trips 文件
            trips_path = build_dir / f"{route}_{bound}_core.trips.xml"
            root = ET.Element('trips')
            
            # 添加 vType（带 vClass=bus）
            vtype = ET.SubElement(root, 'vType')
            vtype.set('id', 'kmb_double_decker')
            vtype.set('vClass', 'bus')
            vtype.set('length', '12.00')
            vtype.set('width', '2.55')
            vtype.set('maxSpeed', '20.00')
            vtype.set('guiShape', 'bus')
            
            trip = ET.SubElement(root, 'trip')
            trip.set('id', f"CORE_{route}_{bound}")
            trip.set('type', 'kmb_double_decker')
            trip.set('depart', '0')
            trip.set('from', edges[0])
            trip.set('to', edges[-1])
            
            if len(edges) > 2:
                trip.set('via', ' '.join(edges[1:-1]))
            
            tree = ET.ElementTree(root)
            ET.indent(tree, space='    ')
            tree.write(str(trips_path), encoding='utf-8', xml_declaration=True)
            
            trip_files[(route, bound)] = trips_path
            
            # 计算核心段 KMB 长度
            core_len = core_stops[-1]['cum_dist_m'] - core_stops[0]['cum_dist_m']
            print(f"  {route} {bound}: {len(core_stops)} stops, {len(edges)} via点, KMB_len={core_len:.0f}m")
    
    return trip_files


def run_duarouter(trip_files, build_dir):
    """运行 duarouter"""
    route_files = {}
    
    for (route, bound), trips_path in trip_files.items():
        output_path = build_dir / f"{route}_{bound}_core.rou.xml"
        
        cmd = [
            'duarouter',
            '-n', str(NET_FILE),
            '--trip-files', str(trips_path),
            '-o', str(output_path),
            '--ignore-errors',
            '--no-warnings',
            '--routing-algorithm', 'dijkstra',
        ]
        
        print(f"  duarouter: {route} {bound}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if output_path.exists() and output_path.stat().st_size > 100:
            route_files[(route, bound)] = output_path
            print(f"    ✅ 输出: {output_path.name}")
        else:
            print(f"    ❌ 失败")
            if result.stderr:
                print(f"    stderr: {result.stderr[:300]}")
    
    return route_files


def calculate_core_scale(route_files, edge_lengths, corrections, cropped_edges):
    """计算核心段 scale"""
    print("\n[核心段 Scale 计算]")
    print("="*60)
    
    for (route, bound), route_file in route_files.items():
        tree = ET.parse(str(route_file))
        root = tree.getroot()
        
        for vehicle in root.findall('.//vehicle'):
            route_elem = vehicle.find('route')
            if route_elem is not None:
                edges = route_elem.get('edges', '').split()
                
                # 计算 SUMO 路径长度
                sumo_len = 0
                for e in edges:
                    if e in edge_lengths:
                        sumo_len += edge_lengths[e]
                    elif e.endswith('_rev') and e[:-4] in edge_lengths:
                        sumo_len += edge_lengths[e[:-4]]
                
                # 计算核心段 KMB 长度
                core_stops = get_core_stops(route, bound, corrections, cropped_edges)
                if len(core_stops) >= 2:
                    kmb_len = core_stops[-1]['cum_dist_m'] - core_stops[0]['cum_dist_m']
                else:
                    kmb_len = 1
                
                scale = sumo_len / kmb_len if kmb_len > 0 else 0
                bridges = [e for e in edges if 'bridge' in e.lower()]
                
                status = "✅" if scale < 1.7 else "⚠️"
                print(f"{status} {route} {bound}: scale={scale:.3f}, edges={len(edges)}, bridges={len(bridges)}")


def main():
    print("="*60)
    print("核心段路由生成 (使用 hk_cropped.net.xml)")
    print("="*60)
    
    print(f"\n使用网络: {NET_FILE.name}")
    
    print("\n[加载数据]")
    cropped_edges = load_cropped_edges()
    corrections = load_corrections()
    edge_lengths = load_edge_lengths()
    print(f"  Cropped 网络边数: {len(cropped_edges)}")
    print(f"  纠偏记录: {len(corrections)}")
    
    print("\n[Step 1] 生成核心段 trips")
    trip_files = generate_core_trips(corrections, cropped_edges, BUILD_DIR)
    
    print("\n[Step 2] 运行 duarouter")
    route_files = run_duarouter(trip_files, BUILD_DIR)
    
    if route_files:
        calculate_core_scale(route_files, edge_lengths, corrections, cropped_edges)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
