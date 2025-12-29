#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_irn_direction.py
=======================
验证 IRN CENTERLINE 无向最短路是否违反单向约束。

对 Pareto Top 段进行验证：
- 68X inbound: Top 5 excess 段
- 960 inbound: Top 2 excess 段

输出：
1. 违规边清单（ROUTE_ID, road name, DIR, from/to, used direction）
2. 违规边长度与累计占比
3. 论文级结论

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-29
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import pandas as pd
import pyproj

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRN_GDB = PROJECT_ROOT / "data" / "RdNet_IRNP.gdb"
KMB_CSV = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"


@dataclass
class EdgeInfo:
    """CENTERLINE 边信息"""
    route_id: int
    street_name: str
    travel_dir: int  # 1=正向only, 2=反向only, 3=双向, 4=禁止
    from_node: Tuple[float, float]
    to_node: Tuple[float, float]
    length: float


@dataclass
class ViolationReport:
    """违规报告"""
    route_id: int
    street_name: str
    travel_dir: int
    dir_str: str
    from_node: str
    to_node: str
    used_direction: str
    length: float


def normalize_xy(x: float, y: float, ndigits: int = 3) -> Tuple[float, float]:
    """Round coordinates to merge floating duplicates."""
    return (round(x, ndigits), round(y, ndigits))


def load_centerline_with_direction() -> Tuple[nx.Graph, Dict[Tuple, EdgeInfo]]:
    """
    加载 CENTERLINE 并构建无向图，同时记录每条边的方向信息。
    
    Returns:
        G: 无向图
        edge_info: (from_node, to_node) -> EdgeInfo
    """
    print("[info] 加载 CENTERLINE 层...")
    gdf = gpd.read_file(str(IRN_GDB), layer="CENTERLINE")
    gdf = gdf.explode(index_parts=False)
    print(f"[info] 共 {len(gdf)} 条记录")
    
    G = nx.Graph()
    edge_info: Dict[Tuple, EdgeInfo] = {}
    
    dir_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type not in ("LineString", "MultiLineString"):
            continue
            
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        
        route_id = row.get('ROUTE_ID', 0)
        street_name = row.get('STREET_ENAME', 'Unknown')
        travel_dir = int(row.get('TRAVEL_DIRECTION', 3))  # 默认双向
        
        dir_counts[travel_dir] = dir_counts.get(travel_dir, 0) + 1
        
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u = normalize_xy(coords[i][0], coords[i][1])
                v = normalize_xy(coords[i + 1][0], coords[i + 1][1])
                
                seg_length = math.sqrt((v[0] - u[0])**2 + (v[1] - u[1])**2)
                
                # 添加无向边
                G.add_edge(u, v, length=seg_length)
                
                # 记录边的方向信息（以规范化的 key 存储）
                key = (u, v) if u < v else (v, u)
                edge_info[key] = EdgeInfo(
                    route_id=route_id,
                    street_name=street_name,
                    travel_dir=travel_dir,
                    from_node=u,  # 几何方向的起点
                    to_node=v,    # 几何方向的终点
                    length=seg_length
                )
    
    print(f"[info] TRAVEL_DIRECTION 分布: {dir_counts}")
    print(f"[info] 图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    return G, edge_info


def project_point_to_graph(G: nx.Graph, lon: float, lat: float, 
                           transformer: pyproj.Transformer) -> Tuple[float, float]:
    """将 WGS84 坐标投影到 HK1980 并找到最近的图节点"""
    x, y = transformer.transform(lon, lat)
    pt = normalize_xy(x, y)
    
    # 找最近节点
    min_dist = float('inf')
    nearest = None
    for node in G.nodes():
        d = math.sqrt((node[0] - pt[0])**2 + (node[1] - pt[1])**2)
        if d < min_dist:
            min_dist = d
            nearest = node
    
    return nearest


def check_path_violations(path: List[Tuple], edge_info: Dict[Tuple, EdgeInfo]) -> List[ViolationReport]:
    """
    检查路径中的违规边。
    
    违规定义：
    - DIR=1（只允许正向）：若实际走了反向 (to→from) 则违规
    - DIR=2（只允许反向）：若实际走了正向 (from→to) 则违规
    - DIR=3（双向）：不违规
    - DIR=4（禁止通行）：任何方向都违规
    """
    violations = []
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        
        # 查找边信息
        key = (u, v) if u < v else (v, u)
        info = edge_info.get(key)
        
        if info is None:
            continue
        
        # 判断实际使用的方向
        # info.from_node 和 info.to_node 是几何方向（digitize 方向）
        # 我们需要判断 u→v 是否与几何方向一致
        
        walking_forward = (u == info.from_node and v == info.to_node)
        walking_backward = (u == info.to_node and v == info.from_node)
        
        is_violation = False
        dir_str = ""
        used_dir = ""
        
        if info.travel_dir == 1:
            # 只允许正向（from→to）
            dir_str = "正向only"
            if walking_backward:
                is_violation = True
                used_dir = "反向使用"
            else:
                used_dir = "正向使用"
                
        elif info.travel_dir == 2:
            # 只允许反向（to→from）
            dir_str = "反向only"
            if walking_forward:
                is_violation = True
                used_dir = "正向使用"
            else:
                used_dir = "反向使用"
                
        elif info.travel_dir == 3:
            # 双向
            dir_str = "双向"
            used_dir = "正向使用" if walking_forward else "反向使用"
            
        elif info.travel_dir == 4:
            # 禁止通行
            dir_str = "禁止通行"
            is_violation = True
            used_dir = "违规通行"
        
        if is_violation:
            violations.append(ViolationReport(
                route_id=info.route_id,
                street_name=info.street_name,
                travel_dir=info.travel_dir,
                dir_str=dir_str,
                from_node=f"({info.from_node[0]:.1f}, {info.from_node[1]:.1f})",
                to_node=f"({info.to_node[0]:.1f}, {info.to_node[1]:.1f})",
                used_direction=used_dir,
                length=info.length
            ))
    
    return violations


def verify_segment(G: nx.Graph, edge_info: Dict[Tuple, EdgeInfo],
                   from_stop: dict, to_stop: dict, 
                   transformer: pyproj.Transformer) -> dict:
    """
    验证一个站点段的无向最短路是否违反单向约束。
    
    Returns:
        {
            'from_stop': str,
            'to_stop': str,
            'kmb_dist': float,
            'path_length': float,
            'path_node_count': int,
            'violations': List[ViolationReport],
            'violation_length': float,
            'violation_ratio': float
        }
    """
    # 投影站点到图
    from_node = project_point_to_graph(G, from_stop['long'], from_stop['lat'], transformer)
    to_node = project_point_to_graph(G, to_stop['long'], to_stop['lat'], transformer)
    
    if from_node is None or to_node is None:
        return {
            'from_stop': from_stop['stop_name_tc'],
            'to_stop': to_stop['stop_name_tc'],
            'kmb_dist': to_stop['cum_dist_m'] - from_stop['cum_dist_m'],
            'path_length': float('nan'),
            'path_node_count': 0,
            'violations': [],
            'violation_length': 0,
            'violation_ratio': 0,
            'error': '无法投影站点到图'
        }
    
    try:
        path = nx.shortest_path(G, from_node, to_node, weight='length')
        path_length = nx.shortest_path_length(G, from_node, to_node, weight='length')
    except nx.NetworkXNoPath:
        return {
            'from_stop': from_stop['stop_name_tc'],
            'to_stop': to_stop['stop_name_tc'],
            'kmb_dist': to_stop['cum_dist_m'] - from_stop['cum_dist_m'],
            'path_length': float('nan'),
            'path_node_count': 0,
            'violations': [],
            'violation_length': 0,
            'violation_ratio': 0,
            'error': '无向图中无路径'
        }
    
    # 检查违规
    violations = check_path_violations(path, edge_info)
    violation_length = sum(v.length for v in violations)
    
    return {
        'from_stop': from_stop['stop_name_tc'],
        'to_stop': to_stop['stop_name_tc'],
        'from_seq': from_stop['seq'],
        'to_seq': to_stop['seq'],
        'kmb_dist': to_stop['cum_dist_m'] - from_stop['cum_dist_m'],
        'path_length': path_length,
        'path_node_count': len(path),
        'violations': violations,
        'violation_count': len(violations),
        'violation_length': violation_length,
        'violation_ratio': violation_length / path_length if path_length > 0 else 0
    }


def main():
    print("=" * 80)
    print("IRN CENTERLINE 无向最短路 - 单向约束违规验证")
    print("=" * 80)
    
    # 加载数据
    G, edge_info = load_centerline_with_direction()
    
    # 加载 KMB 站点数据
    df = pd.read_csv(KMB_CSV)
    
    # 坐标转换器
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)
    
    # Core 段定义（基于 hk_cropped.net.xml 覆盖范围）
    # 68X inbound: seq 1-14 (14站)
    # 68X outbound: seq 19-31 (13站)
    # 960 inbound: seq 1-10 (9站)
    # 960 outbound: seq 15-22 (8站)
    CORE_RANGES = {
        ('68X', 'inbound'): (1, 14),
        ('68X', 'outbound'): (19, 31),
        ('960', 'inbound'): (1, 10),
        ('960', 'outbound'): (15, 22),
    }
    
    # Pareto Top 段定义（基于用户提供的 excess 贡献%）
    # 注意：只验证 Core 段范围内的 Pareto 段
    pareto_segments = [
        # 68X inbound Top 5 (Core: seq 1-14)
        {'route': '68X', 'bound': 'inbound', 'from_seq': 1, 'to_seq': 2, 'excess_pct': 42.1},
        {'route': '68X', 'bound': 'inbound', 'from_seq': 11, 'to_seq': 12, 'excess_pct': 18.2},
        {'route': '68X', 'bound': 'inbound', 'from_seq': 12, 'to_seq': 13, 'excess_pct': 12.2},
        {'route': '68X', 'bound': 'inbound', 'from_seq': 9, 'to_seq': 10, 'excess_pct': 7.4},
        {'route': '68X', 'bound': 'inbound', 'from_seq': 10, 'to_seq': 11, 'excess_pct': 6.6},
        # 960 inbound Top 2 (Core: seq 1-10)
        {'route': '960', 'bound': 'inbound', 'from_seq': 9, 'to_seq': 10, 'excess_pct': 80.3},
        {'route': '960', 'bound': 'inbound', 'from_seq': 1, 'to_seq': 2, 'excess_pct': 14.6},
    ]
    
    # 验证所有 Pareto 段都在 Core 范围内
    for seg in pareto_segments:
        core_range = CORE_RANGES.get((seg['route'], seg['bound']))
        if core_range:
            assert core_range[0] <= seg['from_seq'] <= core_range[1], \
                f"{seg['route']} {seg['bound']} seq {seg['from_seq']} 不在 Core 范围 {core_range}"
            assert core_range[0] <= seg['to_seq'] <= core_range[1], \
                f"{seg['route']} {seg['bound']} seq {seg['to_seq']} 不在 Core 范围 {core_range}"
    
    print(f"\n[info] Core 段口径:")
    for key, val in CORE_RANGES.items():
        print(f"  {key[0]} {key[1]}: seq {val[0]}-{val[1]}")
    
    print("\n" + "=" * 80)
    print("Pareto Top 段验证")
    print("=" * 80)
    
    all_results = []
    total_violations = 0
    total_violation_length = 0.0
    
    for seg in pareto_segments:
        subset = df[(df['route'] == seg['route']) & (df['bound'] == seg['bound'])].sort_values('seq')
        
        from_row = subset[subset['seq'] == seg['from_seq']].iloc[0].to_dict()
        to_row = subset[subset['seq'] == seg['to_seq']].iloc[0].to_dict()
        
        result = verify_segment(G, edge_info, from_row, to_row, transformer)
        result['route'] = seg['route']
        result['bound'] = seg['bound']
        result['excess_pct'] = seg['excess_pct']
        
        all_results.append(result)
        
        print(f"\n[{seg['route']} {seg['bound']} seq {seg['from_seq']}→{seg['to_seq']}] (excess 贡献 {seg['excess_pct']:.1f}%)")
        print(f"  站点: {result['from_stop']} → {result['to_stop']}")
        print(f"  KMB 距离: {result['kmb_dist']:.1f}m")
        print(f"  无向最短路: {result['path_length']:.1f}m ({result['path_node_count']} 节点)")
        print(f"  违规边数: {result['violation_count']}")
        print(f"  违规长度: {result['violation_length']:.1f}m ({result['violation_ratio']*100:.1f}%)")
        
        if result['violations']:
            print(f"  违规边清单:")
            for v in result['violations']:
                print(f"    - ROUTE_ID={v.route_id}, {v.street_name}, DIR={v.travel_dir}({v.dir_str})")
                print(f"      几何方向: {v.from_node} → {v.to_node}")
                print(f"      实际使用: {v.used_direction}, 长度={v.length:.1f}m")
        
        total_violations += result['violation_count']
        total_violation_length += result['violation_length']
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)
    
    print(f"\n验证段数: {len(pareto_segments)}")
    print(f"总违规边数: {total_violations}")
    print(f"总违规长度: {total_violation_length:.1f}m")
    
    segments_with_violations = [r for r in all_results if r['violation_count'] > 0]
    print(f"包含违规的段数: {len(segments_with_violations)}/{len(all_results)}")
    
    # 论文级结论
    print("\n" + "=" * 80)
    print("论文级结论")
    print("=" * 80)
    
    if total_violations > 0:
        print("""
✅ 验证确认：无向 KMB 距离包含"不可行驶"路径

现有 kmb_route_stop_dist.csv 使用无向图计算距离，允许逆向穿越单向道路。
这导致 KMB "真值"距离被低估，而 SUMO（有向图）路径相应被高估。

建议：
1. 将 KMB 距离真值升级为"有向最短路"
2. 保留无向距离作为 legacy 以供历史对比
3. 重新计算 Scale 因子
""")
    else:
        print("""
⚠️ 无向最短路未检测到单向约束违规

可能原因：
1. 路径确实不经过单向道路
2. 违规发生在边界区域（未包含在 CENTERLINE 中）
3. 需要进一步检查 bus-only 通道或其他特殊路权

建议：继续调查其他可能的差距来源。
""")
    
    # 保存详细结果
    output_path = PROJECT_ROOT / "data" / "processed" / "irn_direction_verification.csv"
    rows = []
    for r in all_results:
        rows.append({
            'route': r['route'],
            'bound': r['bound'],
            'from_seq': r.get('from_seq', ''),
            'to_seq': r.get('to_seq', ''),
            'from_stop': r['from_stop'],
            'to_stop': r['to_stop'],
            'kmb_dist_m': r['kmb_dist'],
            'undir_path_m': r['path_length'],
            'violation_count': r['violation_count'],
            'violation_length_m': r['violation_length'],
            'violation_ratio': r['violation_ratio'],
            'excess_pct': r['excess_pct']
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n[ok] 详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
