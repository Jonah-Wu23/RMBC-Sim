#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_routes.py
=================
采用"站点到站点分段路由 + 拼接"策略重建 fixed_routes.rou.xml

策略：
1. 从 bus_stops.add.xml 读取站点→边映射
2. 从 kmb_route_stop_dist.csv 获取站点序列
3. 为每对相邻站点生成 trips 文件
4. 调用 duarouter 生成分段路由
5. 拼接边序列并去毛刺
6. 输出新的 fixed_routes.rou.xml

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-26
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 文件路径
NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
BUS_STOPS_FILE = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
KMB_CSV_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
OLD_ROUTE_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes.rou.xml"
BUILD_DIR = PROJECT_ROOT / "build"
OUTPUT_FILE = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_v2.rou.xml"


def parse_bus_stops(bus_stops_path: str) -> dict:
    """
    从 bus_stops.add.xml 读取 busStop lane -> edge 映射
    
    Returns:
        {stop_id: edge_id}
    """
    tree = ET.parse(bus_stops_path)
    root = tree.getroot()
    
    stop_to_edge = {}
    for stop in root.findall('.//busStop'):
        stop_id = stop.get('id')
        lane = stop.get('lane', '')
        if lane:
            # lane 格式: "EDGE_0" 或 "EDGE_rev_0" 或 ":junction_0"
            if lane.startswith(':'):
                # junction 内部边，提取 junction 名
                edge = lane.rsplit('_', 1)[0]
            else:
                # 普通边，去掉最后的 _N (lane index)
                edge = lane.rsplit('_', 1)[0]
            stop_to_edge[stop_id] = edge
    
    print(f"[Step A] 解析 bus_stops.add.xml: {len(stop_to_edge)} 个站点")
    return stop_to_edge


def get_stop_sequences(kmb_csv_path: str, stop_to_edge: dict) -> dict:
    """
    从 kmb_route_stop_dist.csv 获取排序后的站点序列
    
    Returns:
        {(route, bound): [(seq, stop_id, edge_id), ...]}
    """
    df = pd.read_csv(kmb_csv_path)
    
    sequences = {}
    for (route, bound), g in df.groupby(['route', 'bound']):
        g = g.sort_values('seq')
        seq_list = []
        for _, row in g.iterrows():
            stop_id = row['stop_id']
            edge = stop_to_edge.get(stop_id, None)
            if edge is None:
                print(f"  ⚠️  站点 {stop_id} 在 bus_stops.add.xml 中未找到")
                continue
            seq_list.append((row['seq'], stop_id, edge))
        sequences[(route, bound)] = seq_list
    
    print(f"[Step B] 加载站点序列: {len(sequences)} 个 (route, bound)")
    return sequences


def generate_segment_trips(sequences: dict, build_dir: Path) -> dict:
    """
    为每对相邻站点生成 SUMO trips 文件
    
    Returns:
        {(route, bound): trips_file_path}
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    trip_files = {}
    
    for (route, bound), seq_list in sequences.items():
        if len(seq_list) < 2:
            print(f"  ⚠️  {route} {bound}: 站点少于2个，跳过")
            continue
        
        trips_path = build_dir / f"{route}_{bound}.trips.xml"
        
        root = ET.Element('trips')
        
        for i in range(len(seq_list) - 1):
            seq1, stop1, edge1 = seq_list[i]
            seq2, stop2, edge2 = seq_list[i + 1]
            
            # 跳过同边的相邻站点
            if edge1 == edge2:
                continue
            
            trip = ET.SubElement(root, 'trip')
            trip.set('id', f"SEG_{route}_{bound}_{int(seq1):02d}_{int(seq2):02d}")
            trip.set('type', 'kmb_double_decker')
            trip.set('depart', '0')
            trip.set('from', edge1)
            trip.set('to', edge2)
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space='    ')
        tree.write(str(trips_path), encoding='utf-8', xml_declaration=True)
        
        trip_files[(route, bound)] = trips_path
    
    print(f"[Step C] 生成 trips 文件: {len(trip_files)} 个")
    return trip_files


def run_duarouter(trip_files: dict, net_file: Path, build_dir: Path) -> dict:
    """
    调用 duarouter 生成分段路由
    
    Returns:
        {(route, bound): seg_routes_file_path}
    """
    seg_route_files = {}
    
    for (route, bound), trips_path in trip_files.items():
        output_path = build_dir / f"{route}_{bound}.seg_routes.xml"
        
        cmd = [
            'duarouter',
            '-n', str(net_file),
            '--trip-files', str(trips_path),
            '-o', str(output_path),
            '--ignore-errors',
            '--no-warnings',
        ]
        
        print(f"  运行 duarouter: {route} {bound}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ⚠️  duarouter 失败: {result.stderr[:200]}")
            continue
        
        if output_path.exists():
            seg_route_files[(route, bound)] = output_path
    
    print(f"[Step D] duarouter 完成: {len(seg_route_files)} 个成功")
    return seg_route_files


def parse_segment_routes(seg_route_file: Path) -> dict:
    """
    解析分段路由文件，提取每段的边序列
    
    Returns:
        {trip_id: [edge1, edge2, ...]}
    """
    if not seg_route_file.exists():
        return {}
    
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


def merge_segments(segments: dict, seq_list: list) -> tuple:
    """
    拼接分段路由并去毛刺
    
    Returns:
        (merged_edges, anomalies_log)
    """
    merged = []
    anomalies = []
    
    # 按序列顺序拼接
    for i in range(len(seq_list) - 1):
        seq1, stop1, edge1 = seq_list[i]
        seq2, stop2, edge2 = seq_list[i + 1]
        
        # 查找对应的段
        seg_id = None
        for vid in segments:
            if f"_{int(seq1):02d}_{int(seq2):02d}" in vid:
                seg_id = vid
                break
        
        if seg_id is None:
            # 同边段或 duarouter 失败
            continue
        
        seg_edges = segments[seg_id]
        
        if not merged:
            # 第一段全收
            merged.extend(seg_edges)
        else:
            # 后续段：跳过与前一段最后边相同的开头边
            start_idx = 0
            if seg_edges and seg_edges[0] == merged[-1]:
                start_idx = 1
            merged.extend(seg_edges[start_idx:])
    
    # 去毛刺：连续重复边压缩
    cleaned = []
    for e in merged:
        if cleaned and cleaned[-1] == e:
            anomalies.append(f"连续重复边: {e}")
            continue
        cleaned.append(e)
    
    # 注意：不再删除"立即折返"(e -> e_rev)
    # 因为公交路线可能需要折返才能到达某些站点（如站点在单向道的另一侧）
    # duarouter 生成的路径是最短路径，折返通常是必要的
    
    # 只记录折返情况供诊断
    for i in range(len(cleaned) - 1):
        e, next_e = cleaned[i], cleaned[i + 1]
        if e.endswith('_rev') and e[:-4] == next_e:
            anomalies.append(f"折返(保留): {e} -> {next_e}")
        elif next_e.endswith('_rev') and next_e[:-4] == e:
            anomalies.append(f"折返(保留): {e} -> {next_e}")
    
    return cleaned, anomalies


def load_old_routes(old_route_file: Path) -> dict:
    """
    从旧路由文件加载 stop 定义
    
    Returns:
        {(route, bound): [{busStop, duration}, ...]}
    """
    tree = ET.parse(str(old_route_file))
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


def write_routes(
    merged_routes: dict,
    old_route_stops: dict,
    output_path: Path,
    depart_interval: int = 600,
    n_vehicles: int = 6
):
    """
    生成最终路由文件
    """
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
    
    # 生成 vehicles
    for (route_name, bound), edges in sorted(merged_routes.items()):
        if not edges:
            print(f"  ⚠️  {route_name} {bound}: 边序列为空，跳过")
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
    
    print(f"[Step F] 输出路由文件: {output_path}")


def verify_stop_coverage(merged_routes: dict, old_route_stops: dict, stop_to_edge: dict):
    """
    验证 busStop edge 是否在 route edges 中
    """
    print("\n[验证] busStop 一致性检查:")
    all_ok = True
    
    for (route_name, bound), edges in merged_routes.items():
        edge_set = set(edges)
        stops = old_route_stops.get((route_name, bound), [])
        
        missing = []
        for stop_info in stops:
            stop_id = stop_info['busStop']
            edge = stop_to_edge.get(stop_id, None)
            if edge and edge not in edge_set:
                # 也检查 _rev 版本
                if edge.endswith('_rev'):
                    base = edge[:-4]
                else:
                    base = edge + '_rev'
                if base not in edge_set:
                    missing.append((stop_id, edge))
        
        if missing:
            all_ok = False
            print(f"  ⚠️  {route_name} {bound}: {len(missing)} 个站点边不在路由中")
            for stop_id, edge in missing[:3]:
                print(f"       - {stop_id} ({edge})")
        else:
            print(f"  ✅ {route_name} {bound}: 所有站点边都在路由中")
    
    return all_ok


def main():
    print("=" * 80)
    print("[rebuild_routes.py] 站点到站点分段路由重建")
    print("=" * 80)
    
    # Step A: 解析站点→边映射
    stop_to_edge = parse_bus_stops(str(BUS_STOPS_FILE))
    
    # Step B: 获取站点序列
    sequences = get_stop_sequences(str(KMB_CSV_FILE), stop_to_edge)
    
    # Step C: 生成分段 trips 文件
    trip_files = generate_segment_trips(sequences, BUILD_DIR)
    
    # Step D: 调用 duarouter
    seg_route_files = run_duarouter(trip_files, NET_FILE, BUILD_DIR)
    
    # Step E: 拼接边序列
    print("\n[Step E] 拼接边序列并去毛刺:")
    merged_routes = {}
    all_anomalies = []
    
    for (route, bound), seg_file in seg_route_files.items():
        segments = parse_segment_routes(seg_file)
        seq_list = sequences.get((route, bound), [])
        
        merged, anomalies = merge_segments(segments, seq_list)
        merged_routes[(route, bound)] = merged
        
        if anomalies:
            all_anomalies.extend([(route, bound, a) for a in anomalies])
            print(f"  {route} {bound}: {len(merged)} 边, {len(anomalies)} 个异常")
        else:
            print(f"  {route} {bound}: {len(merged)} 边, 无异常")
    
    if all_anomalies:
        print(f"\n  ⚠️  共发现 {len(all_anomalies)} 个异常:")
        for route, bound, a in all_anomalies[:10]:
            print(f"       [{route} {bound}] {a}")
    
    # 加载旧路由的 stop 定义
    old_route_stops = load_old_routes(OLD_ROUTE_FILE)
    
    # Step F: 输出新路由文件
    write_routes(merged_routes, old_route_stops, OUTPUT_FILE)
    
    # 验证
    verify_stop_coverage(merged_routes, old_route_stops, stop_to_edge)
    
    print("\n" + "=" * 80)
    print("[完成] 请运行 check_route_sanity.py 验证修复效果")
    print("=" * 80)


if __name__ == '__main__':
    main()
