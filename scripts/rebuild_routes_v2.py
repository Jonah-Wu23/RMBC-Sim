#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_routes_v2.py
====================
采用纠偏后的端点重建 fixed_routes.rou.xml (P0.2)

基于 fix_stop_edge_binding.py 生成的 stop_edge_corrections.csv
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
BUS_STOPS_FILE = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
KMB_CSV_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
CORRECTIONS_FILE = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"
OLD_ROUTE_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes.rou.xml"
BUILD_DIR = PROJECT_ROOT / "build"
OUTPUT_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_v2.rou.xml"


def load_corrections():
    """加载纠偏后的端点"""
    df = pd.read_csv(CORRECTIONS_FILE)
    # 创建 (route, bound, stop_id) -> fixed_edge 的映射
    corrections = {}
    for _, row in df.iterrows():
        key = (row['route'], row['bound'], row['stop_id'])
        corrections[key] = row['fixed_edge'] if pd.notna(row['fixed_edge']) else row['orig_edge']
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


def generate_segment_trips_with_corrections(corrections, build_dir):
    """使用纠偏后的端点生成 trips 文件"""
    build_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(KMB_CSV_FILE)
    trip_files = {}
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            
            if len(subset) < 2:
                continue
            
            trips_path = build_dir / f"{route}_{bound}_fixed.trips.xml"
            root = ET.Element('trips')
            
            stops = list(subset.itertuples())
            valid_trips = 0
            
            for i in range(len(stops) - 1):
                stop1, stop2 = stops[i], stops[i+1]
                
                edge1 = corrections.get((route, bound, stop1.stop_id), '')
                edge2 = corrections.get((route, bound, stop2.stop_id), '')
                
                # 处理 NaN 值
                if pd.isna(edge1) or edge1 == '':
                    edge1 = ''
                if pd.isna(edge2) or edge2 == '':
                    edge2 = ''
                
                # 跳过空边或同边
                if not edge1 or not edge2 or edge1 == edge2:
                    continue
                
                # 跳过 bridge gap 边作为起点（这些是虚拟桥接边）
                if str(edge1).startswith('bridge_') or str(edge2).startswith('bridge_'):
                    # 但如果是 bridge gap，仍然生成（让 duarouter 尝试）
                    pass
                
                trip = ET.SubElement(root, 'trip')
                trip.set('id', f"SEG_{route}_{bound}_{int(stop1.seq):02d}_{int(stop2.seq):02d}")
                trip.set('type', 'kmb_double_decker')
                trip.set('depart', '0')
                trip.set('from', edge1)
                trip.set('to', edge2)
                valid_trips += 1
            
            tree = ET.ElementTree(root)
            ET.indent(tree, space='    ')
            tree.write(str(trips_path), encoding='utf-8', xml_declaration=True)
            
            trip_files[(route, bound)] = trips_path
            print(f"  {route} {bound}: {valid_trips} trips")
    
    return trip_files


def run_duarouter(trip_files, build_dir):
    """运行 duarouter"""
    seg_route_files = {}
    
    for (route, bound), trips_path in trip_files.items():
        output_path = build_dir / f"{route}_{bound}_fixed.seg_routes.xml"
        
        cmd = [
            'duarouter',
            '-n', str(NET_FILE),
            '--trip-files', str(trips_path),
            '-o', str(output_path),
            '--ignore-errors',
            '--no-warnings',
        ]
        
        print(f"  duarouter: {route} {bound}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if output_path.exists() and output_path.stat().st_size > 100:
            seg_route_files[(route, bound)] = output_path
    
    return seg_route_files


def merge_segments(seg_route_file, corrections, route, bound, edge_lengths):
    """拼接分段路由"""
    tree = ET.parse(str(seg_route_file))
    root = tree.getroot()
    
    # 解析所有分段
    segments = {}
    for vehicle in root.findall('.//vehicle'):
        vid = vehicle.get('id')
        route_elem = vehicle.find('route')
        if route_elem is not None:
            edges = route_elem.get('edges', '').split()
            segments[vid] = edges
    
    # 按序号排序拼接
    df = pd.read_csv(KMB_CSV_FILE)
    subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
    stops = list(subset.itertuples())
    
    merged = []
    for i in range(len(stops) - 1):
        seq1, seq2 = int(stops[i].seq), int(stops[i+1].seq)
        seg_id = f"SEG_{route}_{bound}_{seq1:02d}_{seq2:02d}"
        
        if seg_id not in segments:
            continue
        
        seg_edges = segments[seg_id]
        
        if not merged:
            merged.extend(seg_edges)
        else:
            start_idx = 0
            if seg_edges and seg_edges[0] == merged[-1]:
                start_idx = 1
            merged.extend(seg_edges[start_idx:])
    
    # 去重复边
    cleaned = []
    for e in merged:
        if cleaned and cleaned[-1] == e:
            continue
        cleaned.append(e)
    
    return cleaned


def calculate_scale(edges, kmb_len, edge_lengths):
    """计算 scale"""
    sumo_len = 0
    for e in edges:
        if e in edge_lengths:
            sumo_len += edge_lengths[e]
        elif e.endswith('_rev') and e[:-4] in edge_lengths:
            sumo_len += edge_lengths[e[:-4]]
        elif e + '_rev' in edge_lengths:
            sumo_len += edge_lengths[e + '_rev']
    
    scale = sumo_len / kmb_len if kmb_len > 0 else 0
    return sumo_len, scale


def load_old_routes():
    """加载旧路由的 stop 定义"""
    tree = ET.parse(str(OLD_ROUTE_FILE))
    root = tree.getroot()
    
    route_stops = {}
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id')
        parts = vid.split('_')
        if len(parts) >= 3:
            route_name = parts[1]
            bound = parts[2].split('.')[0]
            key = (route_name, bound)
            
            if key not in route_stops:
                stops = []
                for stop in vehicle.findall('stop'):
                    stops.append({
                        'busStop': stop.get('busStop'),
                        'duration': stop.get('duration', '30')
                    })
                route_stops[key] = stops
    
    return route_stops


def write_routes(merged_routes, old_route_stops, output_path, depart_interval=600, n_vehicles=6):
    """生成最终路由文件"""
    root = ET.Element('routes')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
    
    # vType
    vtype = ET.SubElement(root, 'vType')
    vtype.set('id', 'kmb_double_decker')
    vtype.set('length', '12.00')
    vtype.set('maxSpeed', '20.00')
    vtype.set('guiShape', 'bus')
    vtype.set('width', '2.55')
    vtype.set('color', 'red')
    vtype.set('personCapacity', '137')
    
    for (route_name, bound), edges in sorted(merged_routes.items()):
        if not edges:
            continue
        
        stops = old_route_stops.get((route_name, bound), [])
        
        for i in range(n_vehicles):
            vehicle = ET.SubElement(root, 'vehicle')
            vehicle.set('id', f'flow_{route_name}_{bound}.{i}')
            vehicle.set('type', 'kmb_double_decker')
            vehicle.set('depart', f'{i * depart_interval:.2f}')
            
            route_elem = ET.SubElement(vehicle, 'route')
            route_elem.set('edges', ' '.join(edges))
            
            for stop_info in stops:
                stop = ET.SubElement(vehicle, 'stop')
                stop.set('busStop', stop_info['busStop'])
                stop.set('duration', stop_info['duration'])
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space='    ')
    tree.write(str(output_path), encoding='utf-8', xml_declaration=True)


def main():
    print("=" * 80)
    print("[P0.2] 使用纠偏后端点重建路由")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    corrections = load_corrections()
    print(f"  - 纠偏记录: {len(corrections)} 条")
    
    edge_lengths = load_edge_lengths()
    print(f"  - 边长度: {len(edge_lengths)} 条")
    
    # KMB 总距离
    df = pd.read_csv(KMB_CSV_FILE)
    kmb_distances = {}
    for (route, bound), g in df.groupby(['route', 'bound']):
        kmb_distances[(route, bound)] = g['cum_dist_m'].max()
    
    # 生成 trips
    print("\n[Step 1] 生成 trips 文件（使用纠偏端点）")
    trip_files = generate_segment_trips_with_corrections(corrections, BUILD_DIR)
    
    # 运行 duarouter
    print("\n[Step 2] 运行 duarouter")
    seg_route_files = run_duarouter(trip_files, BUILD_DIR)
    
    # 拼接
    print("\n[Step 3] 拼接路由")
    merged_routes = {}
    
    for (route, bound), seg_file in seg_route_files.items():
        edges = merge_segments(seg_file, corrections, route, bound, edge_lengths)
        merged_routes[(route, bound)] = edges
        
        kmb_len = kmb_distances.get((route, bound), 1)
        sumo_len, scale = calculate_scale(edges, kmb_len, edge_lengths)
        
        print(f"  {route} {bound}: {len(edges)} edges, SUMO={sumo_len/1000:.1f}km, KMB={kmb_len/1000:.1f}km, scale={scale:.3f}")
    
    # 输出
    print("\n[Step 4] 输出路由文件")
    old_route_stops = load_old_routes()
    write_routes(merged_routes, old_route_stops, OUTPUT_FILE)
    print(f"  -> {OUTPUT_FILE}")
    
    print("\n" + "=" * 80)
    print("[完成] 请运行 check_route_sanity.py 验证")
    print("=" * 80)


if __name__ == '__main__':
    main()
