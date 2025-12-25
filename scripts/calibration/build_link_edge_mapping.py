#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_link_edge_mapping.py
==========================
构建 L2 观测路段到 SUMO edge_id 的映射关系。

核心思路:
1. 解析公交路由文件，提取每条线路的 edge 序列和 busStop 序列
2. 根据 busStop 在 edge 序列中的位置，确定相邻站点之间经过的边
3. 输出映射表: (route, bound, from_seq, to_seq) -> [edge_ids]

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-22
"""

import os
import sys
import json
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_bus_stops_additional(add_path: str) -> Dict[str, str]:
    """
    解析 bus_stops.add.xml，获取 busStop ID 到 lane 的映射
    
    Returns:
        Dict[busStop_id, lane_id]
    """
    tree = ET.parse(add_path)
    root = tree.getroot()
    
    stop_to_lane = {}
    for bus_stop in root.findall('.//busStop'):
        stop_id = bus_stop.get('id')
        lane = bus_stop.get('lane')
        if stop_id and lane:
            stop_to_lane[stop_id] = lane
    
    return stop_to_lane


def lane_to_edge(lane_id: str) -> str:
    """从 lane ID 提取 edge ID (去除 _0, _1 等后缀)"""
    if '_' in lane_id:
        parts = lane_id.rsplit('_', 1)
        if parts[-1].isdigit():
            return parts[0]
    return lane_id


def parse_route_file(route_path: str) -> Dict[str, Dict]:
    """
    解析公交路由文件，提取每条线路的信息
    
    Returns:
        Dict[vehicle_id, {
            'route': str,
            'bound': str,
            'edges': List[str],
            'stops': List[str]  # busStop IDs in order
        }]
    """
    tree = ET.parse(route_path)
    root = tree.getroot()
    
    vehicles = {}
    
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id')
        if not vid:
            continue
        
        # 解析线路名和方向 (如 flow_68X_inbound.0)
        parts = vid.split('_')
        if len(parts) >= 3:
            route_name = parts[1]  # 68X or 960
            bound = parts[2].split('.')[0]  # inbound or outbound
        else:
            continue
        
        # 获取 edge 序列
        route_elem = vehicle.find('route')
        if route_elem is not None:
            edges_str = route_elem.get('edges', '')
            edges = edges_str.split()
        else:
            edges = []
        
        # 获取 stop 序列
        stops = []
        for stop in vehicle.findall('stop'):
            bus_stop = stop.get('busStop')
            if bus_stop:
                stops.append(bus_stop)
        
        vehicles[vid] = {
            'route': route_name,
            'bound': bound,
            'edges': edges,
            'stops': stops
        }
    
    return vehicles


def load_stop_sequence(kmb_csv_path: str) -> Dict[Tuple[str, str, int], str]:
    """
    加载站点序列表
    
    Returns:
        Dict[(route, bound, seq), stop_id]
    """
    df = pd.read_csv(kmb_csv_path)
    
    stop_map = {}
    for _, row in df.iterrows():
        key = (row['route'], row['bound'], int(row['seq']))
        stop_map[key] = row['stop_id']
    
    return stop_map


def build_link_edge_mapping(
    route_path: str,
    bus_stops_path: str,
    kmb_csv_path: str,
    obs_csv_path: str
) -> pd.DataFrame:
    """
    构建完整的路段-边映射表
    
    Args:
        route_path: 公交路由 XML 路径
        bus_stops_path: bus_stops.add.xml 路径
        kmb_csv_path: kmb_route_stop_dist.csv 路径
        obs_csv_path: l2_observation_vector.csv 路径
    
    Returns:
        DataFrame with columns: observation_id, edge_ids (JSON string)
    """
    # 1. 加载站点信息
    stop_to_lane = parse_bus_stops_additional(bus_stops_path)
    stop_seq = load_stop_sequence(kmb_csv_path)
    
    # 2. 解析路由文件
    vehicles = parse_route_file(route_path)
    
    # 3. 为每条线路/方向建立站点->边索引映射
    # 使用第一班车的数据 (假设所有班次路线相同)
    route_edge_maps = {}  # (route, bound) -> Dict[stop_id, edge_index_range]
    
    for vid, vinfo in vehicles.items():
        key = (vinfo['route'], vinfo['bound'])
        if key in route_edge_maps:
            continue  # 只处理第一班车
        
        edges = vinfo['edges']
        stops = vinfo['stops']
        
        # 为每个站点找到其在 edge 序列中的大致位置
        # 方法: 根据 busStop 绑定的 lane -> edge，在边序列中查找
        stop_edge_indices = {}
        
        for i, stop_id in enumerate(stops):
            lane = stop_to_lane.get(stop_id)
            if lane:
                edge = lane_to_edge(lane)
                # 在边序列中查找这个 edge
                try:
                    # 找第一个匹配 (或最接近的)
                    indices = [j for j, e in enumerate(edges) if e == edge or e == f"{edge}_rev"]
                    if indices:
                        # 取中位数位置
                        stop_edge_indices[stop_id] = indices[len(indices) // 2]
                except Exception:
                    pass
        
        route_edge_maps[key] = {
            'edges': edges,
            'stops': stops,
            'stop_indices': stop_edge_indices
        }
    
    # 4. 加载观测向量
    obs_df = pd.read_csv(obs_csv_path)
    
    # 5. 为每个观测点匹配边
    results = []
    
    for _, row in obs_df.iterrows():
        obs_id = row['observation_id']
        route = row['route']
        bound = row['bound']
        from_seq = int(row['from_seq'])
        to_seq = int(row['to_seq'])
        
        key = (route, bound)
        if key not in route_edge_maps:
            results.append({
                'observation_id': obs_id,
                'edge_ids': '[]',
                'matched': False
            })
            continue
        
        rmap = route_edge_maps[key]
        edges = rmap['edges']
        stops = rmap['stops']
        stop_indices = rmap['stop_indices']
        
        # 获取 from_seq 和 to_seq 对应的站点 ID
        from_stop = stop_seq.get((route, bound, from_seq))
        to_stop = stop_seq.get((route, bound, to_seq))
        
        if not from_stop or not to_stop:
            results.append({
                'observation_id': obs_id,
                'edge_ids': '[]',
                'matched': False
            })
            continue
        
        # 获取站点在 edge 序列中的位置
        from_idx = stop_indices.get(from_stop, -1)
        to_idx = stop_indices.get(to_stop, -1)
        
        if from_idx < 0 or to_idx < 0:
            # 尝试在 stops 列表中找
            try:
                from_stop_order = stops.index(from_stop)
                to_stop_order = stops.index(to_stop)
                
                # 用站点顺序估算边的位置
                total_edges = len(edges)
                total_stops = len(stops)
                
                if total_stops > 1:
                    from_idx = int(from_stop_order / total_stops * total_edges)
                    to_idx = int(to_stop_order / total_stops * total_edges)
                else:
                    from_idx = 0
                    to_idx = total_edges - 1
                    
            except ValueError:
                results.append({
                    'observation_id': obs_id,
                    'edge_ids': '[]',
                    'matched': False
                })
                continue
        
        # 提取这段路程的边
        if from_idx <= to_idx:
            link_edges = edges[from_idx:to_idx + 1]
        else:
            # 逆序情况 (不太可能发生)
            link_edges = edges[to_idx:from_idx + 1]
        
        # 去重并过滤掉 bridge_ 等虚拟边
        unique_edges = []
        for e in link_edges:
            if not e.startswith('bridge_') and e not in unique_edges:
                unique_edges.append(e)
        
        results.append({
            'observation_id': obs_id,
            'edge_ids': json.dumps(unique_edges),
            'matched': len(unique_edges) > 0
        })
    
    return pd.DataFrame(results)


def main():
    """生成并保存映射表"""
    route_path = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes.rou.xml"
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    obs_csv_path = PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector.csv"
    output_path = PROJECT_ROOT / "config" / "calibration" / "link_edge_mapping.csv"
    
    print("[INFO] 构建路段-边映射表...")
    print(f"  - 路由文件: {route_path}")
    print(f"  - 站点文件: {bus_stops_path}")
    print(f"  - 站点距离表: {kmb_csv_path}")
    print(f"  - 观测向量: {obs_csv_path}")
    
    mapping_df = build_link_edge_mapping(
        str(route_path),
        str(bus_stops_path),
        str(kmb_csv_path),
        str(obs_csv_path)
    )
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_path, index=False)
    
    matched = mapping_df['matched'].sum()
    total = len(mapping_df)
    print(f"\n[INFO] 映射完成: {matched}/{total} ({100*matched/total:.1f}%)")
    print(f"[INFO] 输出文件: {output_path}")


if __name__ == "__main__":
    main()
