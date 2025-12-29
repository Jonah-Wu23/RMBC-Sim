#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_stop_edge_binding.py
========================
P0.1 Stop 端点纠偏

规则：
1. internal edge（以 ':' 开头）直接禁止，换成邻接外部边
2. edge / reverse(edge) 二选一，选使得相邻段最短的组合
3. 输出修正后的端点列表，用于 rebuild_routes.py

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-26
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from collections import defaultdict
import subprocess
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_net_topology(net_path):
    """
    加载路网拓扑：边长度、from/to 节点、reverse 映射
    """
    tree = ET.parse(str(net_path))
    root = tree.getroot()
    
    edge_lengths = {}
    edge_from_to = {}  # edge_id -> (from_node, to_node)
    
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):
            from_node = edge.get('from')
            to_node = edge.get('to')
            lane = edge.find('lane')
            if lane is not None:
                edge_lengths[eid] = float(lane.get('length', 0))
            edge_from_to[eid] = (from_node, to_node)
    
    # 构建 reverse 映射：(from, to) -> edge_id
    ft_to_edge = {v: k for k, v in edge_from_to.items()}
    reverse_map = {}
    for eid, (f, t) in edge_from_to.items():
        rev_eid = ft_to_edge.get((t, f))
        if rev_eid:
            reverse_map[eid] = rev_eid
    
    # 加载 junction 的 incoming/outgoing edges
    junction_edges = {}  # junction_id -> {'incoming': [...], 'outgoing': [...]}
    for junction in root.findall('.//junction'):
        jid = junction.get('id')
        inc_lanes = junction.get('incLanes', '').split()
        # 提取 edge id（去掉 lane 后缀）
        inc_edges = list(set(l.rsplit('_', 1)[0] for l in inc_lanes if l and not l.startswith(':')))
        
        # outgoing edges: 从该 junction 出发的边
        out_edges = [eid for eid, (f, t) in edge_from_to.items() if f == jid]
        
        junction_edges[jid] = {'incoming': inc_edges, 'outgoing': out_edges}
    
    return edge_lengths, edge_from_to, reverse_map, junction_edges


def load_stop_edges(bus_stops_path):
    """加载站点边映射"""
    tree = ET.parse(str(bus_stops_path))
    root = tree.getroot()
    stop_to_edge = {}
    stop_to_lane = {}
    for stop in root.findall('.//busStop'):
        stop_id = stop.get('id')
        lane = stop.get('lane', '')
        stop_to_lane[stop_id] = lane
        if lane.startswith(':'):
            # internal edge
            edge = lane.rsplit('_', 1)[0]
        else:
            edge = lane.rsplit('_', 1)[0]
        stop_to_edge[stop_id] = edge
    return stop_to_edge, stop_to_lane


def get_shortest_path_length(net_path, from_edge, to_edge, edge_lengths):
    """
    用 duarouter 计算最短路径长度
    返回 (长度, 是否有立即折返)
    """
    if from_edge == to_edge:
        return 0, False
    
    # 简单方法：用现有分段路由结果，或者调用 duarouter
    # 这里用简化方法：直接返回边长度（后续可以改进）
    # 实际生产中应该用 duarouter 或 sumolib.net.getShortestPath
    
    # 暂时返回一个简单估计
    return edge_lengths.get(from_edge, 500) + edge_lengths.get(to_edge, 500), False


def fix_internal_edge(edge_id, junction_edges, edge_from_to):
    """
    规则 1：如果是 internal edge，换成邻接外部边
    """
    if not edge_id.startswith(':'):
        return [edge_id]
    
    # 提取 junction id
    # internal edge 格式: :junction_id_N
    parts = edge_id[1:].rsplit('_', 1)
    if len(parts) >= 1:
        junction_id = parts[0]
    else:
        junction_id = edge_id[1:]
    
    # 获取该 junction 的邻接边
    jinfo = junction_edges.get(junction_id, {})
    candidates = jinfo.get('incoming', []) + jinfo.get('outgoing', [])
    
    # 过滤掉 internal edge
    candidates = [e for e in candidates if not e.startswith(':')]
    
    if candidates:
        return candidates
    else:
        # 回退：尝试从 junction_id 名字猜测
        return [edge_id]  # 无法修复


def fix_stop_edges_for_route(
    route: str,
    bound: str,
    stop_sequence: list,
    stop_to_edge: dict,
    edge_lengths: dict,
    reverse_map: dict,
    junction_edges: dict,
    edge_from_to: dict,
):
    """
    对一条线路的所有站点做端点纠偏
    
    stop_sequence: [(seq, stop_id, stop_name), ...]
    
    返回: {stop_id: fixed_edge}
    """
    fixed_edges = {}
    
    # 第一步：处理 internal edge
    for seq, stop_id, _ in stop_sequence:
        orig_edge = stop_to_edge.get(stop_id, '')
        
        if orig_edge.startswith(':'):
            # 规则 1：internal edge
            candidates = fix_internal_edge(orig_edge, junction_edges, edge_from_to)
            if len(candidates) == 1:
                fixed_edges[stop_id] = candidates[0]
            else:
                # 多个候选，暂时选第一个（后续会被规则 2 优化）
                fixed_edges[stop_id] = candidates[0] if candidates else orig_edge
        else:
            fixed_edges[stop_id] = orig_edge
    
    # 第二步：edge/rev(edge) 二选一（贪心优化）
    # 从前向后，对每个 stop 选择使得该段最短的 edge
    
    for i in range(len(stop_sequence) - 1):
        seq_i, stop_i, _ = stop_sequence[i]
        seq_j, stop_j, _ = stop_sequence[i + 1]
        
        edge_i = fixed_edges[stop_i]
        edge_j = fixed_edges[stop_j]
        
        # 获取候选
        rev_i = reverse_map.get(edge_i)
        rev_j = reverse_map.get(edge_j)
        
        candidates_i = [edge_i]
        if rev_i and rev_i != edge_i:
            candidates_i.append(rev_i)
        
        candidates_j = [edge_j]
        if rev_j and rev_j != edge_j:
            candidates_j.append(rev_j)
        
        # 选最短组合（简化版：用边长度之和估计）
        best_pair = (edge_i, edge_j)
        best_score = float('inf')
        
        for ci in candidates_i:
            for cj in candidates_j:
                # 简单评估：如果 ci 和 cj 方向一致（都是 _rev 或都不是），得分更高
                ci_is_rev = ci.endswith('_rev')
                cj_is_rev = cj.endswith('_rev')
                
                # 方向一致性奖励
                direction_bonus = 0 if ci_is_rev == cj_is_rev else 1000
                
                # 边长度估计
                len_ci = edge_lengths.get(ci, edge_lengths.get(ci.replace('_rev', ''), 500))
                len_cj = edge_lengths.get(cj, edge_lengths.get(cj.replace('_rev', ''), 500))
                
                score = len_ci + len_cj + direction_bonus
                
                if score < best_score:
                    best_score = score
                    best_pair = (ci, cj)
        
        # 更新 fixed_edges（注意：只更新 stop_j，因为 stop_i 在前一轮已固定）
        if i == 0:
            fixed_edges[stop_i] = best_pair[0]
        fixed_edges[stop_j] = best_pair[1]
    
    return fixed_edges


def main():
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    output_path = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"
    
    print("=" * 80)
    print("[P0.1] Stop 端点纠偏")
    print("=" * 80)
    
    # 加载路网拓扑
    print("\n[加载数据]")
    edge_lengths, edge_from_to, reverse_map, junction_edges = load_net_topology(net_path)
    print(f"  - 边长度: {len(edge_lengths)} 条")
    print(f"  - reverse 映射: {len(reverse_map)} 对")
    print(f"  - junction: {len(junction_edges)} 个")
    
    # 加载站点边映射
    stop_to_edge, stop_to_lane = load_stop_edges(bus_stops_path)
    print(f"  - 站点: {len(stop_to_edge)} 个")
    
    # 加载 KMB 站点序列
    df = pd.read_csv(kmb_csv_path)
    
    all_corrections = []
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            
            stop_sequence = [
                (row['seq'], row['stop_id'], row['stop_name_en'])
                for _, row in subset.iterrows()
            ]
            
            # 纠偏
            fixed_edges = fix_stop_edges_for_route(
                route, bound, stop_sequence,
                stop_to_edge, edge_lengths, reverse_map, junction_edges, edge_from_to
            )
            
            # 记录变化
            changes = 0
            for seq, stop_id, name in stop_sequence:
                orig = stop_to_edge.get(stop_id, '')
                fixed = fixed_edges.get(stop_id, orig)
                changed = orig != fixed
                if changed:
                    changes += 1
                
                all_corrections.append({
                    'route': route,
                    'bound': bound,
                    'seq': seq,
                    'stop_id': stop_id,
                    'stop_name': name[:30],
                    'orig_edge': orig,
                    'fixed_edge': fixed,
                    'changed': changed,
                })
            
            print(f"\n  {route} {bound}: {changes} 个站点被纠偏")
    
    # 输出修正表
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrections_df = pd.DataFrame(all_corrections)
    corrections_df.to_csv(output_path, index=False)
    print(f"\n[输出] {output_path}")
    
    # 显示变化详情
    print("\n" + "=" * 80)
    print("[纠偏详情]")
    print("-" * 80)
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            changed = corrections_df[
                (corrections_df['route'] == route) & 
                (corrections_df['bound'] == bound) & 
                (corrections_df['changed'] == True)
            ]
            
            if len(changed) > 0:
                print(f"\n{route} {bound} ({len(changed)} 个变化):")
                for _, row in changed.iterrows():
                    print(f"  seq={int(row['seq']):2d}: {row['orig_edge']:20s} -> {row['fixed_edge']:20s} | {row['stop_name']}")


if __name__ == '__main__':
    main()
