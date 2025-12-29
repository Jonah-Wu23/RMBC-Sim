#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_stop_edge_binding_v2.py
===========================
P0.1 Stop 端点纠偏 - 增强版

规则：
1. internal edge（以 ':' 开头）直接禁止，用距离准则选邻接外部边
2. edge / reverse(edge) 二选一，用 sumolib 计算真实最短路
3. KMB 段长约束：>3x 判非法
4. DP 全局优化（替代贪心）

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-26
"""

import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 尝试导入 sumolib
try:
    import sumolib
    HAS_SUMOLIB = True
except ImportError:
    HAS_SUMOLIB = False
    print("⚠️ sumolib 未安装，将使用简化最短路估计")

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
            # internal edge: 格式 :junction_idx_lane
            edge = lane.rsplit('_', 1)[0]
        else:
            edge = lane.rsplit('_', 1)[0]
        stop_to_edge[stop_id] = edge
    return stop_to_edge, stop_to_lane


def get_shortest_path_sumolib(net, from_edge_id, to_edge_id):
    """
    用 sumolib 计算真实最短路径长度
    返回: (长度, 边列表, 是否有折返)
    """
    if from_edge_id == to_edge_id:
        return 0, [from_edge_id], False
    
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except Exception:
        return float('inf'), [], False
    
    try:
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            total_len = sum(e.getLength() for e in route)
            edge_ids = [e.getID() for e in route]
            
            # 检查是否有折返（相邻边是反向）
            has_reversal = False
            for i in range(len(edge_ids) - 1):
                e1, e2 = edge_ids[i], edge_ids[i + 1]
                if e1.endswith('_rev') and e1[:-4] == e2:
                    has_reversal = True
                    break
                if e2.endswith('_rev') and e2[:-4] == e1:
                    has_reversal = True
                    break
            
            return total_len, edge_ids, has_reversal
    except Exception:
        pass
    
    return float('inf'), [], False


def get_shortest_path_simple(edge_lengths, from_edge, to_edge):
    """
    简化最短路估计（没有 sumolib 时使用）
    """
    if from_edge == to_edge:
        return 0, [from_edge], False
    
    len_from = edge_lengths.get(from_edge, 500)
    len_to = edge_lengths.get(to_edge, 500)
    return len_from + len_to, [from_edge, to_edge], False


def fix_internal_edge_with_distance(edge_id, junction_edges, edge_from_to, edge_lengths, 
                                     prev_edge=None, next_edge=None, net=None):
    """
    规则 1 增强：如果是 internal edge，用距离准则选邻接外部边
    
    优先选择：能形成最短路（从 prev_edge 到 candidate 再到 next_edge）的候选边
    """
    if not edge_id.startswith(':'):
        return edge_id
    
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
    candidates = [e for e in candidates if not e.startswith(':') and e in edge_lengths]
    
    if not candidates:
        print(f"  ⚠️ internal edge {edge_id} 无法找到邻接外部边")
        return edge_id
    
    if len(candidates) == 1:
        return candidates[0]
    
    # 用距离准则选最佳候选
    best_candidate = candidates[0]
    best_score = float('inf')
    
    for c in candidates:
        score = 0
        
        if net and HAS_SUMOLIB:
            # 用 sumolib 计算真实距离
            if prev_edge and prev_edge in edge_lengths:
                len1, _, _ = get_shortest_path_sumolib(net, prev_edge, c)
                score += len1 if len1 < float('inf') else 10000
            
            if next_edge and next_edge in edge_lengths:
                len2, _, _ = get_shortest_path_sumolib(net, c, next_edge)
                score += len2 if len2 < float('inf') else 10000
        else:
            # 简化估计
            if prev_edge:
                score += edge_lengths.get(prev_edge, 500) + edge_lengths.get(c, 500)
            if next_edge:
                score += edge_lengths.get(c, 500) + edge_lengths.get(next_edge, 500)
        
        if score < best_score:
            best_score = score
            best_candidate = c
    
    return best_candidate


def dp_optimize_edges(
    stop_sequence,  # [(seq, stop_id, name, kmb_cum_dist), ...]
    stop_to_edge,
    edge_lengths,
    reverse_map,
    junction_edges,
    edge_from_to,
    net=None,
    max_ratio=3.0,
):
    """
    DP 全局优化：选择每个站点的 edge 方向，最小化总路径长度
    
    DP[i][dir_i] = 到达第 i 站使用方向 dir_i 的最小总 cost
    dir_i ∈ {0: original, 1: reversed}
    """
    n = len(stop_sequence)
    if n == 0:
        return {}
    
    INF = float('inf')
    
    # 预处理每个站点的候选边
    def get_candidates(stop_id):
        orig_edge = stop_to_edge.get(stop_id, '')
        
        # 处理 internal edge
        if orig_edge.startswith(':'):
            fixed = fix_internal_edge_with_distance(
                orig_edge, junction_edges, edge_from_to, edge_lengths, net=net
            )
            orig_edge = fixed
        
        candidates = [orig_edge]
        rev_edge = reverse_map.get(orig_edge)
        if rev_edge and rev_edge != orig_edge:
            candidates.append(rev_edge)
        
        return candidates
    
    # 获取所有站点的候选边
    all_candidates = []
    for seq, stop_id, name, cum_dist in stop_sequence:
        cands = get_candidates(stop_id)
        all_candidates.append(cands)
    
    # DP
    # dp[i][j] = 到达第 i 站，选择第 j 个候选边的最小总 cost
    dp = [[INF] * 2 for _ in range(n)]
    parent = [[(-1, -1)] * 2 for _ in range(n)]
    
    # 初始化
    for j in range(len(all_candidates[0])):
        dp[0][j] = 0
    
    # 转移
    for i in range(1, n):
        kmb_len = stop_sequence[i][3] - stop_sequence[i - 1][3]
        if kmb_len <= 0:
            kmb_len = 100  # 避免除零
        
        for j_curr, e_curr in enumerate(all_candidates[i]):
            if j_curr >= 2:
                break
            
            for j_prev, e_prev in enumerate(all_candidates[i - 1]):
                if j_prev >= 2:
                    break
                
                if e_prev == e_curr:
                    # 同边
                    path_len = 0
                    has_reversal = False
                elif net and HAS_SUMOLIB:
                    path_len, _, has_reversal = get_shortest_path_sumolib(net, e_prev, e_curr)
                else:
                    path_len, _, has_reversal = get_shortest_path_simple(edge_lengths, e_prev, e_curr)
                
                # 惩罚
                cost = path_len
                
                # 折返惩罚
                if has_reversal:
                    cost += 5000
                
                # 超长段惩罚
                if path_len > max_ratio * kmb_len and kmb_len > 50:
                    cost += 10000
                
                # 不可达惩罚
                if path_len == INF:
                    cost = INF
                
                total = dp[i - 1][j_prev] + cost
                if total < dp[i][j_curr]:
                    dp[i][j_curr] = total
                    parent[i][j_curr] = (i - 1, j_prev)
    
    # 回溯
    best_end = 0
    if len(all_candidates[-1]) > 1 and dp[n - 1][1] < dp[n - 1][0]:
        best_end = 1
    
    # 回溯路径
    path = []
    i, j = n - 1, best_end
    while i >= 0:
        path.append((i, j))
        if i == 0:
            break
        i, j = parent[i][j]
    path.reverse()
    
    # 构建结果
    fixed_edges = {}
    for idx, j in path:
        seq, stop_id, name, cum_dist = stop_sequence[idx]
        if j < len(all_candidates[idx]):
            fixed_edges[stop_id] = all_candidates[idx][j]
        else:
            fixed_edges[stop_id] = all_candidates[idx][0]
    
    return fixed_edges


def main():
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    output_path = PROJECT_ROOT / "config" / "calibration" / "stop_edge_corrections.csv"
    
    print("=" * 80)
    print("[P0.1] Stop 端点纠偏 - 增强版")
    print("=" * 80)
    
    # 加载路网拓扑
    print("\n[加载数据]")
    edge_lengths, edge_from_to, reverse_map, junction_edges = load_net_topology(net_path)
    print(f"  - 边长度: {len(edge_lengths)} 条")
    print(f"  - reverse 映射: {len(reverse_map)} 对")
    print(f"  - junction: {len(junction_edges)} 个")
    
    # 加载 sumolib net
    net = None
    if HAS_SUMOLIB:
        print("  - 加载 sumolib.net...")
        net = sumolib.net.readNet(str(net_path), withInternal=False)
        print(f"  - sumolib 边数: {len(list(net.getEdges()))}")
    
    # 加载站点边映射
    stop_to_edge, stop_to_lane = load_stop_edges(bus_stops_path)
    print(f"  - 站点: {len(stop_to_edge)} 个")
    
    # 统计 internal edge
    internal_stops = [(sid, stop_to_edge[sid]) for sid in stop_to_edge if stop_to_edge[sid].startswith(':')]
    print(f"  - internal edge 站点: {len(internal_stops)} 个")
    for sid, edge in internal_stops:
        print(f"       {sid}: {edge}")
    
    # 加载 KMB 站点序列
    df = pd.read_csv(kmb_csv_path)
    
    all_corrections = []
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            
            stop_sequence = [
                (row['seq'], row['stop_id'], row['stop_name_en'], row['cum_dist_m'])
                for _, row in subset.iterrows()
            ]
            
            print(f"\n[{route} {bound}] {len(stop_sequence)} 站")
            
            # DP 优化
            fixed_edges = dp_optimize_edges(
                stop_sequence,
                stop_to_edge,
                edge_lengths,
                reverse_map,
                junction_edges,
                edge_from_to,
                net=net,
                max_ratio=3.0,
            )
            
            # 记录变化
            changes = 0
            internal_fixes = 0
            rev_fixes = 0
            
            for seq, stop_id, name, cum_dist in stop_sequence:
                orig = stop_to_edge.get(stop_id, '')
                fixed = fixed_edges.get(stop_id, orig)
                changed = orig != fixed
                
                if changed:
                    changes += 1
                    if orig.startswith(':'):
                        internal_fixes += 1
                    else:
                        rev_fixes += 1
                
                all_corrections.append({
                    'route': route,
                    'bound': bound,
                    'seq': seq,
                    'stop_id': stop_id,
                    'stop_name': name[:30] if name else '',
                    'orig_edge': orig,
                    'fixed_edge': fixed,
                    'changed': changed,
                    'is_internal_fix': orig.startswith(':'),
                })
            
            print(f"  📊 纠偏: {changes} 个 (internal: {internal_fixes}, rev: {rev_fixes})")
    
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
                    marker = "🔧" if row['is_internal_fix'] else "↔️"
                    print(f"  {marker} seq={int(row['seq']):2d}: {row['orig_edge']:20s} -> {row['fixed_edge']:20s} | {row['stop_name']}")
    
    # 验证：计算纠偏后预估的 scale 改善
    print("\n" + "=" * 80)
    print("[预估改善]")
    print("-" * 80)
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            subset = corrections_df[
                (corrections_df['route'] == route) & 
                (corrections_df['bound'] == bound)
            ].sort_values('seq')
            
            fixed_edges_list = subset['fixed_edge'].tolist()
            
            # 简单估计总长度
            total_len = 0
            for i in range(len(fixed_edges_list) - 1):
                e1, e2 = fixed_edges_list[i], fixed_edges_list[i + 1]
                if e1 == e2:
                    continue
                if net and HAS_SUMOLIB:
                    seg_len, _, _ = get_shortest_path_sumolib(net, e1, e2)
                else:
                    seg_len = edge_lengths.get(e1, 500) + edge_lengths.get(e2, 500)
                if seg_len < float('inf'):
                    total_len += seg_len
            
            # KMB 总长度
            kmb_subset = df[(df['route'] == route) & (df['bound'] == bound)]
            kmb_total = kmb_subset['cum_dist_m'].max() if len(kmb_subset) > 0 else 1
            
            scale = total_len / kmb_total if kmb_total > 0 else 0
            print(f"  {route} {bound}: 预估 scale ≈ {scale:.2f}")


if __name__ == '__main__':
    main()
