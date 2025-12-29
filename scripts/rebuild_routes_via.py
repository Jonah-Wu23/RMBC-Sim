#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_routes_via.py
=====================
P0.6 via routing：一次路由穿过所有 stops

替代分段拼接方法，使用 duarouter 的 via 参数一次性求解整条路

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-27
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v1.net.xml"
BUS_STOPS_FILE = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
KMB_CSV_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
CORRECTIONS_FILE = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"
OLD_ROUTE_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes.rou.xml"
BUILD_DIR = PROJECT_ROOT / "build"
OUTPUT_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_via.rou.xml"

# 额外 via 点配置：用于解决回环折返/走廊偏离
# 格式：(route, bound): [(after_edge, insert_via_edge), ...]
EXTRA_VIA = {
    # 68X inbound 的 scale 2.094 是系统性偏差，via 效果有限
}


def load_corrections():
    """加载纠偏后的端点"""
    df = pd.read_csv(CORRECTIONS_FILE)
    corrections = {}
    for _, row in df.iterrows():
        key = (row['route'], row['bound'], row['stop_id'])
        edge = row['fixed_edge'] if pd.notna(row['fixed_edge']) else row['orig_edge']
        if pd.notna(edge) and edge and not str(edge).startswith(':'):
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


def generate_via_trips(corrections, build_dir):
    """
    生成使用 via 参数的 trips 文件
    
    每条线路只生成一个 trip：
    - from = 首站 edge
    - to = 末站 edge
    - via = 中间站点 edges（用空格分隔）
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(KMB_CSV_FILE)
    trip_files = {}
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            
            if len(subset) < 2:
                continue
            
            stops = list(subset.itertuples())
            
            # 获取所有站点的 edge
            edges = []
            for stop in stops:
                edge = corrections.get((route, bound, stop.stop_id), '')
                if edge and not edge.startswith('bridge_'):
                    edges.append(edge)
            
            if len(edges) < 2:
                print(f"  ⚠️ {route} {bound}: 有效边不足")
                continue
            
            # 去重连续相同边
            deduped_edges = [edges[0]]
            for e in edges[1:]:
                if e != deduped_edges[-1]:
                    deduped_edges.append(e)
            
            # 应用额外 via 点
            extra_via_list = EXTRA_VIA.get((route, bound), [])
            if extra_via_list:
                new_edges = []
                for e in deduped_edges:
                    new_edges.append(e)
                    for after_edge, insert_via in extra_via_list:
                        if e == after_edge:
                            new_edges.append(insert_via)
                deduped_edges = new_edges
            
            # 生成 trips 文件
            trips_path = build_dir / f"{route}_{bound}_via.trips.xml"
            root = ET.Element('trips')
            
            # Inject vType to enforce vClass="bus" permissions
            vtype = ET.SubElement(root, 'vType')
            vtype.set('id', 'kmb_double_decker')
            vtype.set('vClass', 'bus')
            vtype.set('length', '12.00')
            vtype.set('width', '2.55')
            vtype.set('maxSpeed', '20.00')
            vtype.set('guiShape', 'bus')
            vtype.set('color', 'red')
            vtype.set('personCapacity', '137')
            
            trip = ET.SubElement(root, 'trip')
            trip.set('id', f"VIA_{route}_{bound}")
            trip.set('type', 'kmb_double_decker')
            trip.set('depart', '0')
            trip.set('from', deduped_edges[0])
            trip.set('to', deduped_edges[-1])
            
            # via 参数：中间站点（不包括首尾）
            if len(deduped_edges) > 2:
                via_edges = deduped_edges[1:-1]
                trip.set('via', ' '.join(via_edges))
            
            tree = ET.ElementTree(root)
            ET.indent(tree, space='    ')
            tree.write(str(trips_path), encoding='utf-8', xml_declaration=True)
            
            trip_files[(route, bound)] = trips_path
            print(f"  {route} {bound}: from={deduped_edges[0]}, to={deduped_edges[-1]}, via count={len(deduped_edges)-2}")
    
    return trip_files


def run_duarouter(trip_files, build_dir):
    """运行 duarouter"""
    route_files = {}
    
    for (route, bound), trips_path in trip_files.items():
        output_path = build_dir / f"{route}_{bound}_via.rou.xml"
        
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
                print(f"    stderr: {result.stderr[:500]}")
    
    return route_files


def parse_routes(route_files, edge_lengths):
    """解析路由并计算 scale"""
    df = pd.read_csv(KMB_CSV_FILE)
    kmb_data = {}
    for (route, bound), g in df.groupby(['route', 'bound']):
        kmb_data[(route, bound)] = g['cum_dist_m'].max()
    
    results = {}
    
    for (route, bound), route_file in route_files.items():
        tree = ET.parse(str(route_file))
        root = tree.getroot()
        
        for vehicle in root.findall('.//vehicle'):
            route_elem = vehicle.find('route')
            if route_elem is not None:
                edges = route_elem.get('edges', '').split()
                
                # 计算长度
                sumo_len = 0
                for e in edges:
                    if e in edge_lengths:
                        sumo_len += edge_lengths[e]
                    elif e.endswith('_rev') and e[:-4] in edge_lengths:
                        sumo_len += edge_lengths[e[:-4]]
                    elif e + '_rev' in edge_lengths:
                        sumo_len += edge_lengths[e + '_rev']
                
                kmb_len = kmb_data.get((route, bound), 1)
                scale = sumo_len / kmb_len
                
                # 检查折返
                reversals = []
                for i in range(len(edges) - 1):
                    e1, e2 = edges[i], edges[i + 1]
                    if (e1.endswith('_rev') and e1[:-4] == e2) or \
                       (e2.endswith('_rev') and e2[:-4] == e1):
                        reversals.append(f"{e1}→{e2}")
                
                results[(route, bound)] = {
                    'edges': edges,
                    'sumo_len': sumo_len,
                    'kmb_len': kmb_len,
                    'scale': scale,
                    'reversals': reversals,
                }
                
                print(f"\n  {route} {bound}:")
                print(f"    - 边数: {len(edges)}")
                print(f"    - SUMO长度: {sumo_len/1000:.1f} km")
                print(f"    - KMB长度: {kmb_len/1000:.1f} km")
                print(f"    - Scale: {scale:.3f}")
                if reversals:
                    print(f"    - ⚠️ 折返: {len(reversals)} 个")
                    for r in reversals[:5]:
                        print(f"         {r}")
                else:
                    print(f"    - ✅ 无折返")
    
    return results


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


def write_routes(results, old_route_stops, output_path, n_vehicles=6, depart_interval=600):
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
    
    for (route_name, bound), data in sorted(results.items()):
        edges = data['edges']
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
    print("[P0.6] Via Routing：一次路由穿过所有 stops")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    corrections = load_corrections()
    print(f"  - 纠偏记录: {len(corrections)} 条")
    
    edge_lengths = load_edge_lengths()
    print(f"  - 边长度: {len(edge_lengths)} 条")
    
    # 生成 via trips
    print("\n[Step 1] 生成 via trips 文件")
    trip_files = generate_via_trips(corrections, BUILD_DIR)
    
    # 运行 duarouter
    print("\n[Step 2] 运行 duarouter")
    route_files = run_duarouter(trip_files, BUILD_DIR)
    
    # 解析结果
    print("\n[Step 3] 解析路由结果")
    results = parse_routes(route_files, edge_lengths)
    
    # 输出
    print("\n[Step 4] 输出路由文件")
    old_route_stops = load_old_routes()
    write_routes(results, old_route_stops, OUTPUT_FILE)
    print(f"  -> {OUTPUT_FILE}")
    
    # 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("-" * 80)
    
    for (route, bound), data in sorted(results.items()):
        status = "✅" if data['scale'] < 1.3 and not data['reversals'] else "⚠️"
        print(f"  {status} {route} {bound}: scale={data['scale']:.3f}, 折返={len(data['reversals'])}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
