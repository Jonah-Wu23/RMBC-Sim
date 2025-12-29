#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clean_kmb_shapes_directed.py
============================
生成有向 KMB 距离（遵守单向约束）。

与 clean_kmb_shapes.py 的区别：
1. 使用 nx.DiGraph() 而非 nx.Graph()
2. 按 TRAVEL_DIRECTION 建边：
   - DIR=1: 只加正向边 (u→v)
   - DIR=2: 只加反向边 (v→u)
   - DIR=3: 双向
   - DIR=4: 禁止通行

输出：更新 kmb_route_stop_dist.csv，新增 _dir 和 _undir 双列

兜底策略：
A. 投影节点兜底：如果 directed 不可达，尝试最近 5 个候选投影节点
B. 标记而不是硬算：仍不可达则 link_dist_m_dir = NaN, dir_status = DIR_BLOCKED

Author: Auto-generated for RMBC-Sim project  
Date: 2025-12-29
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import pandas as pd
import pyproj

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRN_GDB = PROJECT_ROOT / "data" / "RdNet_IRNP.gdb"
KMB_CSV_IN = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
KMB_CSV_OUT = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"

# 研究区域边界 (HK1980 坐标)
# 从 hk_cropped.net.xml 的 origBoundary
STUDY_AREA_BOUNDS = {
    'min_x': 832316.13,
    'min_y': 815196.72,
    'max_x': 836765.25,
    'max_y': 823216.13,
}


def normalize_xy(x: float, y: float, ndigits: int = 3) -> Tuple[float, float]:
    return (round(x, ndigits), round(y, ndigits))


def is_in_study_area(x: float, y: float) -> bool:
    """检查坐标是否在研究区域内"""
    return (STUDY_AREA_BOUNDS['min_x'] <= x <= STUDY_AREA_BOUNDS['max_x'] and
            STUDY_AREA_BOUNDS['min_y'] <= y <= STUDY_AREA_BOUNDS['max_y'])


def build_graphs() -> Tuple[nx.Graph, nx.DiGraph, Dict]:
    """
    构建无向图和有向图。
    
    Returns:
        G_undir: 无向图（legacy）
        G_dir: 有向图（主口径）
        node_coords: 节点坐标映射
    """
    print("[info] 加载 CENTERLINE 层...")
    gdf = gpd.read_file(str(IRN_GDB), layer="CENTERLINE")
    gdf = gdf.explode(index_parts=False)
    print(f"[info] 共 {len(gdf)} 条记录")
    
    G_undir = nx.Graph()
    G_dir = nx.DiGraph()
    node_coords = {}
    
    dir_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    study_area_edges = 0
    
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type not in ("LineString", "MultiLineString"):
            continue
            
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        travel_dir = int(row.get('TRAVEL_DIRECTION', 3))
        dir_counts[travel_dir] = dir_counts.get(travel_dir, 0) + 1
        
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u = normalize_xy(coords[i][0], coords[i][1])
                v = normalize_xy(coords[i + 1][0], coords[i + 1][1])
                
                # 检查是否在研究区域内（至少一端在区域内）
                u_in = is_in_study_area(u[0], u[1])
                v_in = is_in_study_area(v[0], v[1])
                
                seg_length = math.sqrt((v[0] - u[0])**2 + (v[1] - u[1])**2)
                
                node_coords[u] = u
                node_coords[v] = v
                
                # 无向图：始终双向
                G_undir.add_edge(u, v, length=seg_length)
                
                # 有向图：按 TRAVEL_DIRECTION
                if travel_dir == 1:
                    # 只允许正向 (from→to)
                    G_dir.add_edge(u, v, length=seg_length)
                elif travel_dir == 2:
                    # 只允许反向 (to→from)
                    G_dir.add_edge(v, u, length=seg_length)
                elif travel_dir == 3:
                    # 双向
                    G_dir.add_edge(u, v, length=seg_length)
                    G_dir.add_edge(v, u, length=seg_length)
                # DIR=4 禁止通行，不加边
                
                if u_in or v_in:
                    study_area_edges += 1
    
    print(f"[info] TRAVEL_DIRECTION 分布: {dir_counts}")
    print(f"[info] 无向图: {G_undir.number_of_nodes()} 节点, {G_undir.number_of_edges()} 边")
    print(f"[info] 有向图: {G_dir.number_of_nodes()} 节点, {G_dir.number_of_edges()} 边")
    print(f"[info] 研究区域内边数: {study_area_edges}")
    
    return G_undir, G_dir, node_coords


def find_nearest_nodes(pt: Tuple[float, float], node_coords: Dict, k: int = 5) -> List[Tuple[float, float]]:
    """找到最近的 k 个节点"""
    distances = []
    for node in node_coords:
        d = math.sqrt((node[0] - pt[0])**2 + (node[1] - pt[1])**2)
        distances.append((d, node))
    distances.sort(key=lambda x: x[0])
    return [n for _, n in distances[:k]]


def project_stop(lon: float, lat: float, transformer: pyproj.Transformer,
                 node_coords: Dict) -> Tuple[float, float]:
    """将站点投影到最近的图节点"""
    x, y = transformer.transform(lon, lat)
    pt = normalize_xy(x, y)
    
    # 找最近节点
    min_dist = float('inf')
    nearest = None
    for node in node_coords:
        d = math.sqrt((node[0] - pt[0])**2 + (node[1] - pt[1])**2)
        if d < min_dist:
            min_dist = d
            nearest = node
    
    return nearest


def compute_path_length(G: nx.Graph, from_node, to_node) -> float:
    """计算最短路径长度"""
    if from_node == to_node:
        return 0.0
    try:
        return nx.shortest_path_length(G, from_node, to_node, weight='length')
    except nx.NetworkXNoPath:
        return float('nan')


def compute_directed_path_with_fallback(G_dir: nx.DiGraph, from_node, to_node,
                                         node_coords: Dict, k: int = 5) -> Tuple[float, str]:
    """
    计算有向最短路径，带兜底策略。
    
    Returns:
        (path_length, status)
        status: 'OK', 'FALLBACK', 'DIR_BLOCKED'
    """
    if from_node == to_node:
        return 0.0, 'OK'
    
    # 尝试直接计算
    try:
        length = nx.shortest_path_length(G_dir, from_node, to_node, weight='length')
        return length, 'OK'
    except nx.NetworkXNoPath:
        pass
    
    # 兜底 A：尝试最近 k 个候选投影节点
    from_candidates = find_nearest_nodes(from_node, node_coords, k)
    to_candidates = find_nearest_nodes(to_node, node_coords, k)
    
    best_length = float('inf')
    
    for fc in from_candidates:
        for tc in to_candidates:
            try:
                length = nx.shortest_path_length(G_dir, fc, tc, weight='length')
                # 加上投影距离
                from_offset = math.sqrt((fc[0] - from_node[0])**2 + (fc[1] - from_node[1])**2)
                to_offset = math.sqrt((tc[0] - to_node[0])**2 + (tc[1] - to_node[1])**2)
                total = length + from_offset + to_offset
                if total < best_length:
                    best_length = total
            except nx.NetworkXNoPath:
                continue
    
    if best_length < float('inf'):
        return best_length, 'FALLBACK'
    
    # 兜底 B：标记为不可达
    return float('nan'), 'DIR_BLOCKED'


def main():
    print("=" * 80)
    print("生成有向 KMB 距离（遵守单向约束）")
    print("=" * 80)
    
    # 构建图
    G_undir, G_dir, node_coords = build_graphs()
    
    # 加载现有 KMB 数据
    df = pd.read_csv(KMB_CSV_IN)
    print(f"\n[info] 加载 {len(df)} 行 KMB 数据")
    
    # 保留原始列作为 legacy
    df['link_dist_m_undir'] = df['link_dist_m']
    df['cum_dist_m_undir'] = df['cum_dist_m']
    
    # 坐标转换器
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)
    
    # 按 route + bound 分组处理
    new_link_dist_dir = []
    new_cum_dist_dir = []
    dir_status_list = []
    
    for (route, bound), group in df.groupby(['route', 'bound']):
        print(f"\n[处理] {route} {bound}")
        group = group.sort_values('seq').reset_index(drop=True)
        
        prev_node = None
        cum_dist = 0.0
        
        for idx, row in group.iterrows():
            lon, lat = row['long'], row['lat']
            node = project_stop(lon, lat, transformer, node_coords)
            
            if prev_node is None:
                # 第一站
                new_link_dist_dir.append(0.0)
                new_cum_dist_dir.append(0.0)
                dir_status_list.append('OK')
            else:
                # 计算有向距离
                link_dist, status = compute_directed_path_with_fallback(
                    G_dir, prev_node, node, node_coords
                )
                
                if math.isnan(link_dist):
                    new_link_dist_dir.append(float('nan'))
                    cum_dist = float('nan')
                else:
                    new_link_dist_dir.append(link_dist)
                    if not math.isnan(cum_dist):
                        cum_dist += link_dist
                
                new_cum_dist_dir.append(cum_dist)
                dir_status_list.append(status)
                
                if status != 'OK':
                    print(f"  [!] seq {row['seq']} ({row['stop_name_tc']}): {status}")
            
            prev_node = node
    
    df['link_dist_m_dir'] = new_link_dist_dir
    df['cum_dist_m_dir'] = new_cum_dist_dir
    df['dir_status'] = dir_status_list
    
    # 统计
    print("\n" + "=" * 80)
    print("统计")
    print("=" * 80)
    
    total = len(df)
    ok_count = (df['dir_status'] == 'OK').sum()
    fallback_count = (df['dir_status'] == 'FALLBACK').sum()
    blocked_count = (df['dir_status'] == 'DIR_BLOCKED').sum()
    
    print(f"总行数: {total}")
    print(f"OK: {ok_count} ({ok_count/total*100:.1f}%)")
    print(f"FALLBACK: {fallback_count} ({fallback_count/total*100:.1f}%)")
    print(f"DIR_BLOCKED: {blocked_count} ({blocked_count/total*100:.1f}%)")
    
    # 对比 undir 和 dir 的差异
    df['link_diff'] = df['link_dist_m_dir'] - df['link_dist_m_undir']
    diff_over_100 = df[df['link_diff'].abs() > 100]
    print(f"\n差异 > 100m 的段数: {len(diff_over_100)}")
    
    # 保存
    df.to_csv(KMB_CSV_OUT, index=False)
    print(f"\n[ok] 已保存到: {KMB_CSV_OUT}")
    
    # 输出列说明
    print("""
列说明:
- link_dist_m_undir, cum_dist_m_undir: 无向距离（legacy）
- link_dist_m_dir, cum_dist_m_dir: 有向距离（主口径）
- dir_status: OK / FALLBACK / DIR_BLOCKED
- link_diff: dir - undir 差异
""")


if __name__ == "__main__":
    main()
