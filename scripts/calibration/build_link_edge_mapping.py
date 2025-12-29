#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_link_edge_mapping.py
==========================
构建 L2 观测路段到 SUMO edge_id 的映射关系。

核心改进 (v2):
1. 使用累计里程匹配替代线性站点数插值
2. 全局尺度对齐（scale = sumo_total / kmb_total）
3. 方向/异常区间保护
4. 稳健的边 ID 查找（不依赖 _rev 猜测）
5. 增强诊断输出（包含 max_speed 检测高速路）

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-26
"""

import os
import sys
import json
import bisect
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_edge_info(net_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    从路网 XML 加载每条边的长度和限速
    
    Returns:
        Tuple[Dict[edge_id, length_m], Dict[edge_id, speed_limit_ms]]
    """
    tree = ET.parse(net_path)
    root = tree.getroot()
    
    edge_lengths = {}
    edge_speeds = {}
    
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):  # 排除内部边
            lane = edge.find('lane')
            if lane is not None:
                length = float(lane.get('length', 0))
                speed = float(lane.get('speed', 13.89))  # 默认 50 km/h
                edge_lengths[eid] = length
                edge_speeds[eid] = speed
    
    return edge_lengths, edge_speeds


def get_edge_length(edge_id: str, edge_lengths: Dict[str, float]) -> Tuple[float, bool]:
    """
    稳健地获取边长度（不依赖 _rev 猜测）
    
    Returns:
        Tuple[length, found]
    """
    # 1. 直接查找
    if edge_id in edge_lengths:
        return edge_lengths[edge_id], True
    
    # 2. 尝试 _rev 变体（仅作为 fallback）
    if edge_id.endswith('_rev'):
        base_id = edge_id[:-4]
        if base_id in edge_lengths:
            return edge_lengths[base_id], True
    else:
        rev_id = f"{edge_id}_rev"
        if rev_id in edge_lengths:
            return edge_lengths[rev_id], True
    
    # 3. 未找到
    return 0.0, False


def get_edge_speed(edge_id: str, edge_speeds: Dict[str, float]) -> float:
    """稳健地获取边限速"""
    if edge_id in edge_speeds:
        return edge_speeds[edge_id]
    
    if edge_id.endswith('_rev'):
        base_id = edge_id[:-4]
        if base_id in edge_speeds:
            return edge_speeds[base_id]
    else:
        rev_id = f"{edge_id}_rev"
        if rev_id in edge_speeds:
            return edge_speeds[rev_id]
    
    return 13.89  # 默认 50 km/h


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


def load_stop_sequence_with_dist(kmb_csv_path: str) -> Tuple[Dict, Dict]:
    """
    加载站点序列表和累计里程
    
    Returns:
        Tuple[
            Dict[(route, bound, seq), stop_id],
            Dict[(route, bound, seq), cum_dist_m]
        ]
    """
    df = pd.read_csv(kmb_csv_path)
    
    stop_map = {}
    dist_map = {}
    for _, row in df.iterrows():
        key = (row['route'], row['bound'], int(row['seq']))
        stop_map[key] = row['stop_id']
        dist_map[key] = row['cum_dist_m']
    
    return stop_map, dist_map


def compute_route_cum_distances(
    edges: List[str],
    edge_lengths: Dict[str, float]
) -> Tuple[List[float], int]:
    """
    计算边序列的累计里程
    
    Returns:
        Tuple[List[cum_dist], missing_count]
    """
    cum_dist = 0.0
    cum_dists = []
    missing_count = 0
    
    for e in edges:
        length, found = get_edge_length(e, edge_lengths)
        if not found:
            missing_count += 1
        cum_dist += length
        cum_dists.append(cum_dist)
    
    return cum_dists, missing_count


def find_edge_range_by_distance(
    edge_cum_dists: List[float],
    from_dist: float,
    to_dist: float,
    tolerance: float = 100.0
) -> Tuple[int, int]:
    """
    根据累计里程找到边序列中的对应范围
    
    Args:
        edge_cum_dists: 边序列的累计里程（预计算）
        from_dist: 起始站点累计里程（已缩放）
        to_dist: 终止站点累计里程（已缩放）
        tolerance: 匹配容差（米）
    
    Returns:
        (from_idx, to_idx) 边索引范围
    """
    n_edges = len(edge_cum_dists)
    if n_edges == 0:
        return 0, 0
    
    # 二分查找起始位置：第一条累计里程 >= from_dist - tolerance 的边
    target_from = max(0, from_dist - tolerance)
    from_idx = bisect.bisect_left(edge_cum_dists, target_from)
    
    # 二分查找终止位置：最后一条累计里程 <= to_dist + tolerance 的边
    target_to = to_dist + tolerance
    to_idx = bisect.bisect_right(edge_cum_dists, target_to) - 1
    
    # 边界保护
    from_idx = max(0, min(from_idx, n_edges - 1))
    to_idx = max(0, min(to_idx, n_edges - 1))
    
    # 确保 from_idx <= to_idx
    if from_idx > to_idx:
        from_idx, to_idx = to_idx, from_idx
    
    return from_idx, to_idx


def adjust_edge_range_for_length(
    edges: List[str],
    from_idx: int,
    to_idx: int,
    expected_length: float,
    edge_lengths: Dict[str, float],
    edge_speeds: Dict[str, float],
    len_tol: float = 0.25,
    max_steps: int = 5,
    speed_limit_kmh: float = 60.0
) -> Tuple[int, int, str]:
    """
    长度一致性自修正：调整边窗口使 map_len 接近 exp_len
    
    策略：
    - 若 map_len < exp_len*(1-len_tol): 扩展窗口 (to_idx+=1 或 from_idx-=1)
    - 若 map_len > exp_len*(1+len_tol): 收缩窗口 (优先收缩高速边)
    
    Args:
        edges: 完整边序列
        from_idx, to_idx: 初始窗口索引
        expected_length: 期望长度（米）
        edge_lengths: 边长度字典
        edge_speeds: 边限速字典
        len_tol: 长度容差比例 (0.25 = ±25%)
        max_steps: 最大调整步数
        speed_limit_kmh: 判断高速边的限速阈值
    
    Returns:
        (adjusted_from_idx, adjusted_to_idx, adjustment_note)
    """
    if expected_length <= 0:
        return from_idx, to_idx, "NO_EXPECTED_LEN"
    
    n_edges = len(edges)
    adjustments = []
    
    for step in range(max_steps):
        # 计算当前窗口长度
        current_edges = edges[from_idx:to_idx + 1]
        current_length = sum(get_edge_length(e, edge_lengths)[0] for e in current_edges)
        ratio = current_length / expected_length
        
        # 检查是否在容差范围内
        if (1 - len_tol) <= ratio <= (1 + len_tol):
            break
        
        if ratio < (1 - len_tol):
            # 太短，需要扩展
            expanded = False
            # 优先向后扩展
            if to_idx < n_edges - 1:
                next_edge = edges[to_idx + 1]
                next_speed = get_edge_speed(next_edge, edge_speeds) * 3.6
                # 只扩展非高速边
                if next_speed <= speed_limit_kmh:
                    to_idx += 1
                    adjustments.append(f"+to({step})")
                    expanded = True
            # 如果后面不行，尝试向前扩展
            if not expanded and from_idx > 0:
                prev_edge = edges[from_idx - 1]
                prev_speed = get_edge_speed(prev_edge, edge_speeds) * 3.6
                if prev_speed <= speed_limit_kmh:
                    from_idx -= 1
                    adjustments.append(f"-from({step})")
                    expanded = True
            if not expanded:
                # 无法继续扩展（两端都是高速边或边界）
                break
                
        elif ratio > (1 + len_tol):
            # 太长，需要收缩
            # 策略：只收缩真正的高速边（>55 km/h），不做均匀收缩
            shrunk = False
            highspeed_threshold = 55.0  # 只收缩真正的快速路/高速边
            
            if to_idx > from_idx:
                tail_edge = edges[to_idx]
                head_edge = edges[from_idx]
                tail_speed = get_edge_speed(tail_edge, edge_speeds) * 3.6
                head_speed = get_edge_speed(head_edge, edge_speeds) * 3.6
                
                # 只收缩超过阈值的高速边
                if tail_speed > highspeed_threshold:
                    to_idx -= 1
                    adjustments.append(f"-to({step})")
                    shrunk = True
                elif head_speed > highspeed_threshold:
                    from_idx += 1
                    adjustments.append(f"+from({step})")
                    shrunk = True
                # 不再做均匀收缩，避免过度收缩正常市区边
            if not shrunk:
                break
    
    note = ";".join(adjustments) if adjustments else "NO_ADJ"
    return from_idx, to_idx, note


def build_link_edge_mapping(
    route_path: str,
    bus_stops_path: str,
    kmb_csv_path: str,
    obs_csv_path: str,
    net_path: str
) -> pd.DataFrame:
    """
    构建完整的路段-边映射表（v2：累计里程匹配版）
    
    Args:
        route_path: 公交路由 XML 路径
        bus_stops_path: bus_stops.add.xml 路径
        kmb_csv_path: kmb_route_stop_dist.csv 路径
        obs_csv_path: l2_observation_vector.csv 路径
        net_path: 路网 XML 路径
    
    Returns:
        DataFrame with columns: observation_id, edge_ids, matched, ...（诊断字段）
    """
    # 1. 加载边信息（长度 + 限速）
    print("[INFO] 加载路网边信息...")
    edge_lengths, edge_speeds = load_edge_info(net_path)
    print(f"  - 加载了 {len(edge_lengths)} 条边的信息")
    
    # 2. 加载站点信息
    stop_to_lane = parse_bus_stops_additional(bus_stops_path)
    stop_seq, stop_dist = load_stop_sequence_with_dist(kmb_csv_path)
    
    # 3. 解析路由文件
    vehicles = parse_route_file(route_path)
    
    # 4. 为每条线路/方向预计算边序列累计里程和全局尺度
    route_data = {}  # (route, bound) -> {edges, cum_dists, sumo_total, kmb_total, scale}
    
    for vid, vinfo in vehicles.items():
        key = (vinfo['route'], vinfo['bound'])
        if key in route_data:
            continue  # 只处理第一班车
        
        edges = vinfo['edges']
        stops = vinfo['stops']
        
        # 计算边序列累计里程
        edge_cum_dists, missing_len_cnt = compute_route_cum_distances(edges, edge_lengths)
        sumo_total = edge_cum_dists[-1] if edge_cum_dists else 0
        
        # 获取该 route/bound 的 KMB 累计里程总长
        # 找到该组合的最大 seq
        max_seq = max((k[2] for k in stop_dist.keys() 
                       if k[0] == vinfo['route'] and k[1] == vinfo['bound']), default=0)
        kmb_total = stop_dist.get((vinfo['route'], vinfo['bound'], max_seq), 0)
        
        # 计算缩放因子（关键点 A：全局尺度对齐）
        if kmb_total > 0:
            scale = sumo_total / kmb_total
        else:
            scale = 1.0
        
        # 为每个站点找到其在 edge 序列中的大致位置
        stop_edge_indices = {}
        for i, stop_id in enumerate(stops):
            lane = stop_to_lane.get(stop_id)
            if lane:
                edge = lane_to_edge(lane)
                indices = [j for j, e in enumerate(edges) if e == edge or e == f"{edge}_rev"]
                if indices:
                    stop_edge_indices[stop_id] = indices[len(indices) // 2]
        
        route_data[key] = {
            'edges': edges,
            'stops': stops,
            'stop_indices': stop_edge_indices,
            'edge_cum_dists': edge_cum_dists,
            'sumo_total': sumo_total,
            'kmb_total': kmb_total,
            'scale': scale,
            'missing_len_cnt': missing_len_cnt
        }
        
        print(f"  - {key}: {len(edges)} 边, SUMO={sumo_total/1000:.1f}km, KMB={kmb_total/1000:.1f}km, scale={scale:.3f}")
    
    # 5. 加载观测向量
    obs_df = pd.read_csv(obs_csv_path)
    
    # 6. 为每个观测点匹配边
    results = []
    
    for _, row in obs_df.iterrows():
        obs_id = row['observation_id']
        route = row['route']
        bound = row['bound']
        from_seq = int(row['from_seq'])
        to_seq = int(row['to_seq'])
        
        key = (route, bound)
        if key not in route_data:
            results.append({
                'observation_id': obs_id,
                'edge_ids': '[]',
                'matched': False,
                'reason': 'NO_ROUTE_DATA',
                'edge_count': 0,
                'total_length': 0,
                'expected_length': 0,
                'max_speed_kmh': 0,
                'missing_len_cnt': 0,
                'match_method': 'FAILED'
            })
            continue
        
        rdata = route_data[key]
        edges = rdata['edges']
        stops = rdata['stops']
        stop_indices = rdata['stop_indices']
        edge_cum_dists = rdata['edge_cum_dists']
        scale = rdata['scale']
        
        # 获取 from_seq 和 to_seq 对应的站点 ID
        from_stop = stop_seq.get((route, bound, from_seq))
        to_stop = stop_seq.get((route, bound, to_seq))
        
        if not from_stop or not to_stop:
            results.append({
                'observation_id': obs_id,
                'edge_ids': '[]',
                'matched': False,
                'reason': 'NO_STOP_DATA',
                'edge_count': 0,
                'total_length': 0,
                'expected_length': 0,
                'max_speed_kmh': 0,
                'missing_len_cnt': 0,
                'match_method': 'FAILED'
            })
            continue
        
        # 获取 KMB 累计里程
        from_dist_kmb = stop_dist.get((route, bound, from_seq), 0)
        to_dist_kmb = stop_dist.get((route, bound, to_seq), 0)
        expected_length = abs(to_dist_kmb - from_dist_kmb)
        
        # 关键点 B：方向保护
        if to_dist_kmb < from_dist_kmb:
            # 距离数据异常，尝试交换
            from_dist_kmb, to_dist_kmb = to_dist_kmb, from_dist_kmb
        
        # ============================================================
        # 改动 A：只要有有效的距离差值就走 distance_bisect
        # 修复：from_dist=0 (第一站) 也应该走 distance_bisect
        # ============================================================
        adjustment_note = "NO_ADJ"
        
        if to_dist_kmb > from_dist_kmb:  # 关键修改：不再要求 from_dist > 0
            # 关键点 A：全局尺度对齐
            from_dist_scaled = from_dist_kmb * scale
            to_dist_scaled = to_dist_kmb * scale
            
            from_idx, to_idx = find_edge_range_by_distance(
                edge_cum_dists, from_dist_scaled, to_dist_scaled
            )
            match_method = 'distance_bisect'
            
            # ============================================================
            # 改动 B：长度一致性自修正
            # ============================================================
            from_idx, to_idx, adjustment_note = adjust_edge_range_for_length(
                edges=edges,
                from_idx=from_idx,
                to_idx=to_idx,
                expected_length=expected_length,
                edge_lengths=edge_lengths,
                edge_speeds=edge_speeds,
                len_tol=0.25,
                max_steps=5,
                speed_limit_kmh=60.0
            )
            if adjustment_note != "NO_ADJ":
                match_method = 'distance_bisect+adj'
        else:
            # 回退到 stop_index（仅当里程数据完全缺失时）
            from_idx = stop_indices.get(from_stop, -1)
            to_idx = stop_indices.get(to_stop, -1)
            match_method = 'stop_index'
            
            if from_idx < 0 or to_idx < 0:
                results.append({
                    'observation_id': obs_id,
                    'edge_ids': '[]',
                    'matched': False,
                    'reason': 'NO_DISTANCE_DATA',
                    'edge_count': 0,
                    'total_length': 0,
                    'expected_length': expected_length,
                    'length_ratio': 0,
                    'max_speed_kmh': 0,
                    'missing_len_cnt': 0,
                    'removed_highspeed_cnt': 0,
                    'adjustment_note': adjustment_note,
                    'match_method': 'FAILED'
                })
                continue
            
            # stop_index 也做长度自修正
            from_idx, to_idx, adjustment_note = adjust_edge_range_for_length(
                edges=edges,
                from_idx=from_idx,
                to_idx=to_idx,
                expected_length=expected_length,
                edge_lengths=edge_lengths,
                edge_speeds=edge_speeds,
                len_tol=0.25,
                max_steps=5,
                speed_limit_kmh=60.0
            )
            if adjustment_note != "NO_ADJ":
                match_method = 'stop_index+adj'

        
        # 提取这段路程的边
        if from_idx <= to_idx:
            link_edges = edges[from_idx:to_idx + 1]
        else:
            link_edges = edges[to_idx:from_idx + 1]
        
        # 去重并过滤掉 bridge_ 等虚拟边
        unique_edges = []
        for e in link_edges:
            if not e.startswith('bridge_') and e not in unique_edges:
                unique_edges.append(e)
        
        # 高速边过滤器：对于短距离市区路段（<1km），移除限速>60 km/h 的边
        # 原因：SUMO 路由可能包含不合理的高速绕行
        filtered_edges = unique_edges
        removed_highspeed_cnt = 0
        if expected_length > 0 and expected_length < 1000:
            filtered_edges = [e for e in unique_edges 
                              if get_edge_speed(e, edge_speeds) * 3.6 <= 60]
            removed_highspeed_cnt = len(unique_edges) - len(filtered_edges)
            # 如果过滤后没有边了，保留原始边
            if not filtered_edges:
                filtered_edges = unique_edges
                removed_highspeed_cnt = 0  # 回退，不算过滤
        
        # 计算映射结果的统计信息
        total_length = sum(get_edge_length(e, edge_lengths)[0] for e in filtered_edges)
        max_speed = max((get_edge_speed(e, edge_speeds) for e in filtered_edges), default=13.89)
        max_speed_kmh = max_speed * 3.6
        
        # 统计长度缺失的边数
        missing_cnt = sum(1 for e in filtered_edges if not get_edge_length(e, edge_lengths)[1])
        
        # ============================================================
        # 改动 D：增强 reason 诊断
        # ============================================================
        reason = 'OK'
        length_ratio = total_length / expected_length if expected_length > 0 else 0
        
        # 质量门槛（使用已经过自修正后的比例）
        if expected_length > 0 and expected_length < 1000:
            if length_ratio < 0.8:
                reason = 'LENGTH_SHORT'
            elif length_ratio > 1.3:
                reason = 'LENGTH_LONG'
        elif expected_length > 0 and total_length > 3 * expected_length + 200:
            reason = 'DIST_MISMATCH_LARGE'
        
        # 如果有高速边被过滤，附加标记
        if removed_highspeed_cnt > 0 and reason == 'OK':
            reason = 'HIGHSPEED_FILTERED'
        
        results.append({
            'observation_id': obs_id,
            'edge_ids': json.dumps(filtered_edges),
            'matched': len(filtered_edges) > 0,
            'reason': reason,
            'edge_count': len(filtered_edges),
            'total_length': round(total_length, 1),
            'expected_length': round(expected_length, 1),
            'length_ratio': round(length_ratio, 2),
            'max_speed_kmh': round(max_speed_kmh, 1),
            'missing_len_cnt': missing_cnt,
            'removed_highspeed_cnt': removed_highspeed_cnt,
            'adjustment_note': adjustment_note,
            'match_method': match_method
        })
    
    return pd.DataFrame(results)


def main():
    """生成并保存映射表"""
    route_path = PROJECT_ROOT / "sumo" / "routes" / "fixed_routes.rou.xml"
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
    
    # 默认使用 corridor 观测向量
    obs_csv_path = PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector_corridor.csv"
    output_path = PROJECT_ROOT / "config" / "calibration" / "link_edge_mapping_corridor.csv"
    
    # 检查是否有命令行参数指定全量观测向量
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        obs_csv_path = PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector.csv"
        output_path = PROJECT_ROOT / "config" / "calibration" / "link_edge_mapping.csv"
    
    print("=" * 60)
    print("[INFO] 构建路段-边映射表 (v2: 累计里程匹配)")
    print("=" * 60)
    print(f"  - 路由文件: {route_path}")
    print(f"  - 站点文件: {bus_stops_path}")
    print(f"  - 站点距离表: {kmb_csv_path}")
    print(f"  - 路网文件: {net_path}")
    print(f"  - 观测向量: {obs_csv_path}")
    print()
    
    mapping_df = build_link_edge_mapping(
        str(route_path),
        str(bus_stops_path),
        str(kmb_csv_path),
        str(obs_csv_path),
        str(net_path)
    )
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_path, index=False)
    
    # 打印统计信息
    matched = mapping_df['matched'].sum()
    total = len(mapping_df)
    print()
    print("=" * 60)
    print(f"[INFO] 映射完成: {matched}/{total} ({100*matched/total:.1f}%)")
    print(f"[INFO] 输出文件: {output_path}")
    print()
    
    # 打印详细诊断
    print("[诊断] 各观测点映射详情:")
    print("-" * 125)
    print(f"{'obs_id':>6} | {'method':<18} | {'edges':>5} | {'map_len':>8} | {'exp_len':>8} | {'len_rat':>7} | {'rm_hs':>5} | {'reason':<18} | {'adjustment':<15}")
    print("-" * 125)
    for _, r in mapping_df.iterrows():
        flag = ""
        if r['max_speed_kmh'] > 60:
            flag = " ⚠️高速"
        if r['reason'] not in ['OK', 'FAILED', 'HIGHSPEED_FILTERED']:
            flag += " ⚠️"
        ratio_str = f"{r['length_ratio']:.2f}" if r['length_ratio'] > 0 else "N/A"
        rm_hs = r.get('removed_highspeed_cnt', 0)
        adj = r.get('adjustment_note', 'N/A')[:15]
        print(f"{r['observation_id']:>6} | {r['match_method']:<18} | {r['edge_count']:>5} | {r['total_length']:>7.0f}m | {r['expected_length']:>7.0f}m | {ratio_str:>7} | {rm_hs:>5} | {r['reason']:<18} | {adj:<15}{flag}")
    print("-" * 125)
    
    # 质量统计
    ok_count = len(mapping_df[mapping_df['reason'] == 'OK'])
    short_count = len(mapping_df[mapping_df['reason'] == 'LENGTH_SHORT'])
    long_count = len(mapping_df[mapping_df['reason'] == 'LENGTH_LONG'])
    filtered_count = len(mapping_df[mapping_df['reason'] == 'HIGHSPEED_FILTERED'])
    print(f"\n[质量] OK={ok_count}, LENGTH_SHORT={short_count}, LENGTH_LONG={long_count}, HIGHSPEED_FILTERED={filtered_count}")
    
    # 特别检查 Observation 1 和 2
    for obs_id in [1, 2]:
        obs_row = mapping_df[mapping_df['observation_id'] == obs_id]
        if not obs_row.empty:
            print()
            print(f"[关键检查] Observation {obs_id}:")
            edges = json.loads(obs_row['edge_ids'].iloc[0])
            print(f"  - 映射方法: {obs_row['match_method'].iloc[0]}")
            print(f"  - 映射边数: {len(edges)}")
            print(f"  - 映射总长: {obs_row['total_length'].iloc[0]:.0f}m (预期 {obs_row['expected_length'].iloc[0]:.0f}m)")
            print(f"  - 长度比例: {obs_row['length_ratio'].iloc[0]:.2f}")
            print(f"  - 自修正: {obs_row.get('adjustment_note', pd.Series(['N/A'])).iloc[0]}")
            print(f"  - 边列表: {edges[:8]}{'...' if len(edges) > 8 else ''}")


if __name__ == "__main__":
    main()
