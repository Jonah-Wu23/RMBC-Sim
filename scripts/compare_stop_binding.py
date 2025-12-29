#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_stop_binding.py
=======================
对比实验：贪心 vs DP 全局站点落边优化

按用户审批意见设计：
1. 候选集扩展：orig + rev + nearby parallel edges（每站 ≤5 候选）
2. 惩罚项：不可达(∞) + 立即回穿(重罚) + rev_switch(轻罚)
3. KMB ratio：只做诊断报告，不进代价函数

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-29
"""

import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

# 尝试导入 sumolib
try:
    import sumolib
    HAS_SUMOLIB = True
except ImportError:
    HAS_SUMOLIB = False
    print("⚠️ sumolib 未安装，将使用简化最短路估计")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# 配置参数
# =============================================================================

@dataclass
class OptimizationConfig:
    """优化配置参数"""
    # 候选集配置
    max_candidates_per_stop: int = 5  # 每站最大候选数
    max_parallel_edges: int = 3       # 最多添加的平行边数
    parallel_edge_max_dist: float = 100.0  # 平行边最大距离（米）
    
    # 惩罚项配置（按用户意见调整）
    penalty_unreachable: float = float('inf')  # 不可达：无穷大
    penalty_immediate_uturn: float = 5000.0    # 立即回穿（非结构性必要）
    penalty_rev_switch: float = 300.0          # rev 切换（稳定用，不压死可达性）
    
    # 诊断阈值（不进代价）
    kmb_ratio_warning: float = 3.0    # KMB 比率预警阈值
    
    # 缓存
    use_path_cache: bool = True


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class PathResult:
    """最短路径结果"""
    length: float
    edges: List[str] = field(default_factory=list)
    has_immediate_uturn: bool = False
    is_structural_reversal: bool = False  # 结构性折返标记


@dataclass
class BindingResult:
    """站点落边结果"""
    stop_id: str
    stop_name: str
    seq: int
    orig_edge: str
    fixed_edge: str
    changed: bool
    candidates_count: int
    

@dataclass
class RouteResult:
    """线路结果汇总"""
    route: str
    bound: str
    method: str  # 'greedy' or 'dp'
    bindings: List[BindingResult]
    total_path_length: float
    kmb_total_length: float
    scale_factor: float
    uturn_count: int  # 立即回穿次数
    structural_reversal_count: int  # 结构性折返次数
    rev_switch_count: int
    unreachable_count: int
    pocket_conflict_count: int = 0  # 口袋对消次数


# =============================================================================
# 路网加载
# =============================================================================

def load_net_topology(net_path):
    """加载路网拓扑"""
    tree = ET.parse(str(net_path))
    root = tree.getroot()
    
    edge_lengths = {}
    edge_from_to = {}  # edge_id -> (from_node, to_node)
    edge_coords = {}   # edge_id -> (x, y) 中心点坐标
    
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        if eid and not eid.startswith(':'):
            from_node = edge.get('from')
            to_node = edge.get('to')
            lane = edge.find('lane')
            if lane is not None:
                edge_lengths[eid] = float(lane.get('length', 0))
                # 解析 shape 获取中心点
                shape = lane.get('shape', '')
                if shape:
                    points = [tuple(map(float, p.split(','))) for p in shape.split()]
                    if points:
                        mid_idx = len(points) // 2
                        edge_coords[eid] = points[mid_idx]
            edge_from_to[eid] = (from_node, to_node)
    
    # 构建 reverse 映射
    ft_to_edge = {v: k for k, v in edge_from_to.items()}
    reverse_map = {}
    for eid, (f, t) in edge_from_to.items():
        rev_eid = ft_to_edge.get((t, f))
        if rev_eid:
            reverse_map[eid] = rev_eid
    
    # 加载 junction
    junction_edges = {}
    for junction in root.findall('.//junction'):
        jid = junction.get('id')
        inc_lanes = junction.get('incLanes', '').split()
        inc_edges = list(set(l.rsplit('_', 1)[0] for l in inc_lanes if l and not l.startswith(':')))
        out_edges = [eid for eid, (f, t) in edge_from_to.items() if f == jid]
        junction_edges[jid] = {'incoming': inc_edges, 'outgoing': out_edges}
    
    return edge_lengths, edge_from_to, reverse_map, junction_edges, edge_coords


def load_stop_edges(bus_stops_path):
    """加载站点边映射和位置"""
    tree = ET.parse(str(bus_stops_path))
    root = tree.getroot()
    
    stop_to_edge = {}
    stop_to_lane = {}
    stop_pos = {}  # stop_id -> lane 上的位置
    
    for stop in root.findall('.//busStop'):
        stop_id = stop.get('id')
        lane = stop.get('lane', '')
        stop_to_lane[stop_id] = lane
        
        start_pos = float(stop.get('startPos', 0))
        end_pos = float(stop.get('endPos', 20))
        stop_pos[stop_id] = (start_pos + end_pos) / 2
        
        if lane.startswith(':'):
            edge = lane.rsplit('_', 1)[0]
        else:
            edge = lane.rsplit('_', 1)[0]
        stop_to_edge[stop_id] = edge
        
    return stop_to_edge, stop_to_lane, stop_pos


# =============================================================================
# 最短路径计算
# =============================================================================

class PathCache:
    """路径代价缓存"""
    def __init__(self):
        self.cache: Dict[Tuple[str, str], PathResult] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, from_edge: str, to_edge: str) -> Optional[PathResult]:
        key = (from_edge, to_edge)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, from_edge: str, to_edge: str, result: PathResult):
        self.cache[(from_edge, to_edge)] = result
    
    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {self.hits}/{total} hits ({hit_rate:.1f}%)"


def is_immediate_uturn(edge1: str, edge2: str, reverse_map: dict) -> bool:
    """检测是否为立即 U-turn（A -> A_rev 或 A_rev -> A）"""
    if edge1 == edge2:
        return False
    
    rev1 = reverse_map.get(edge1)
    if rev1 == edge2:
        return True
    
    # 检查名字模式
    if edge1.endswith('_rev'):
        base1 = edge1[:-4]
        if base1 == edge2:
            return True
    if edge2.endswith('_rev'):
        base2 = edge2[:-4]
        if base2 == edge1:
            return True
    
    return False


def detect_uturn_in_path(path_edges: List[str], reverse_map: dict) -> Tuple[int, int]:
    """
    检测路径中的 U-turn
    返回: (立即回穿次数, 结构性折返次数)
    """
    immediate_count = 0
    structural_count = 0
    
    for i in range(len(path_edges) - 1):
        e1, e2 = path_edges[i], path_edges[i + 1]
        if is_immediate_uturn(e1, e2, reverse_map):
            # 判断是否为结构性折返（简化判断：如果发生在路径中段则可能是结构性的）
            # 更精确的判断需要检查是否有其他可选路径
            if i > 0 and i < len(path_edges) - 2:
                structural_count += 1
            else:
                immediate_count += 1
    
    return immediate_count, structural_count


def get_shortest_path_sumolib(net, from_edge_id: str, to_edge_id: str, 
                               reverse_map: dict, cache: Optional[PathCache] = None) -> PathResult:
    """用 sumolib 计算最短路径"""
    if from_edge_id == to_edge_id:
        return PathResult(length=0, edges=[from_edge_id])
    
    # 检查缓存
    if cache:
        cached = cache.get(from_edge_id, to_edge_id)
        if cached:
            return cached
    
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except Exception:
        result = PathResult(length=float('inf'))
        if cache:
            cache.put(from_edge_id, to_edge_id, result)
        return result
    
    try:
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            total_len = sum(e.getLength() for e in route)
            edge_ids = [e.getID() for e in route]
            
            immediate, structural = detect_uturn_in_path(edge_ids, reverse_map)
            
            result = PathResult(
                length=total_len,
                edges=edge_ids,
                has_immediate_uturn=(immediate > 0),
                is_structural_reversal=(structural > 0)
            )
            if cache:
                cache.put(from_edge_id, to_edge_id, result)
            return result
    except Exception:
        pass
    
    result = PathResult(length=float('inf'))
    if cache:
        cache.put(from_edge_id, to_edge_id, result)
    return result


def get_shortest_path_simple(edge_lengths: dict, from_edge: str, to_edge: str) -> PathResult:
    """简化最短路估计"""
    if from_edge == to_edge:
        return PathResult(length=0, edges=[from_edge])
    
    len_from = edge_lengths.get(from_edge, 500)
    len_to = edge_lengths.get(to_edge, 500)
    return PathResult(length=len_from + len_to, edges=[from_edge, to_edge])


# =============================================================================
# 候选集生成
# =============================================================================

def get_nearby_edges(stop_coord: Tuple[float, float], edge_coords: dict, 
                     edge_lengths: dict, exclude_edges: Set[str],
                     max_dist: float, max_count: int) -> List[str]:
    """获取距离站点最近的边（平行边候选）"""
    if not stop_coord:
        return []
    
    distances = []
    for eid, coord in edge_coords.items():
        if eid in exclude_edges or eid.startswith(':'):
            continue
        if eid not in edge_lengths:
            continue
        
        dist = np.sqrt((coord[0] - stop_coord[0])**2 + (coord[1] - stop_coord[1])**2)
        if dist <= max_dist:
            distances.append((dist, eid))
    
    distances.sort(key=lambda x: x[0])
    return [eid for _, eid in distances[:max_count]]


def fix_internal_edge(edge_id: str, junction_edges: dict, edge_from_to: dict, 
                      edge_lengths: dict) -> str:
    """修复 internal edge"""
    if not edge_id.startswith(':'):
        return edge_id
    
    # 提取 junction id
    parts = edge_id[1:].rsplit('_', 1)
    junction_id = parts[0] if len(parts) >= 1 else edge_id[1:]
    
    # 获取邻接边
    jinfo = junction_edges.get(junction_id, {})
    candidates = jinfo.get('incoming', []) + jinfo.get('outgoing', [])
    candidates = [e for e in candidates if not e.startswith(':') and e in edge_lengths]
    
    if candidates:
        # 选最短的
        return min(candidates, key=lambda e: edge_lengths.get(e, float('inf')))
    
    return edge_id


def generate_candidates(stop_id: str, stop_to_edge: dict, reverse_map: dict,
                        junction_edges: dict, edge_from_to: dict, edge_lengths: dict,
                        edge_coords: dict, kmb_lat_long: Tuple[float, float],
                        config: OptimizationConfig) -> List[str]:
    """
    生成站点的候选边列表
    
    按用户意见：orig + rev + nearby parallel edges（每站 ≤5 候选）
    """
    orig_edge = stop_to_edge.get(stop_id, '')
    
    # 修复 internal edge
    if orig_edge.startswith(':'):
        orig_edge = fix_internal_edge(orig_edge, junction_edges, edge_from_to, edge_lengths)
    
    candidates = set()
    
    # 只添加有效的边（非空且在 edge_lengths 中）
    if orig_edge and orig_edge in edge_lengths:
        candidates.add(orig_edge)
    
    # 添加反向边
    rev_edge = reverse_map.get(orig_edge)
    if rev_edge and rev_edge != orig_edge and rev_edge in edge_lengths:
        candidates.add(rev_edge)
    
    # 添加平行边（nearby edges）
    if orig_edge and orig_edge in edge_coords:
        stop_coord = edge_coords[orig_edge]
        nearby = get_nearby_edges(
            stop_coord, edge_coords, edge_lengths,
            exclude_edges=candidates,
            max_dist=config.parallel_edge_max_dist,
            max_count=config.max_parallel_edges
        )
        candidates.update(nearby)
    
    # 过滤掉空字符串和无效边
    candidates = {c for c in candidates if c and c in edge_lengths}
    
    # 限制候选数
    candidates_list = list(candidates)
    if len(candidates_list) > config.max_candidates_per_stop:
        # 优先保留 orig 和 rev
        priority = []
        if orig_edge and orig_edge in candidates:
            priority.append(orig_edge)
        if rev_edge and rev_edge in candidates and rev_edge not in priority:
            priority.append(rev_edge)
        remaining = [e for e in candidates_list if e not in priority]
        candidates_list = priority + remaining[:config.max_candidates_per_stop - len(priority)]
    
    # 如果候选为空，尝试返回原始边（即使无效）
    if not candidates_list:
        candidates_list = [orig_edge] if orig_edge else []
    
    return candidates_list


# =============================================================================
# 贪心算法（基线）
# =============================================================================

def run_greedy_binding(stop_sequence: List[Tuple], stop_to_edge: dict,
                       edge_lengths: dict, reverse_map: dict, junction_edges: dict,
                       edge_from_to: dict, edge_coords: dict, net,
                       config: OptimizationConfig) -> Dict[str, str]:
    """
    贪心算法（复现原始 fix_stop_edge_binding.py 的逻辑）
    """
    fixed_edges = {}
    
    # 第一步：处理 internal edge
    for seq, stop_id, name, cum_dist, lat, long in stop_sequence:
        orig_edge = stop_to_edge.get(stop_id, '')
        
        if orig_edge.startswith(':'):
            orig_edge = fix_internal_edge(orig_edge, junction_edges, edge_from_to, edge_lengths)
        
        fixed_edges[stop_id] = orig_edge
    
    # 第二步：贪心选择 edge/rev（原始逻辑）
    for i in range(len(stop_sequence) - 1):
        seq_i, stop_i, _, _, _, _ = stop_sequence[i]
        seq_j, stop_j, _, _, _, _ = stop_sequence[i + 1]
        
        edge_i = fixed_edges[stop_i]
        edge_j = fixed_edges[stop_j]
        
        rev_i = reverse_map.get(edge_i)
        rev_j = reverse_map.get(edge_j)
        
        candidates_i = [edge_i]
        if rev_i and rev_i != edge_i:
            candidates_i.append(rev_i)
        
        candidates_j = [edge_j]
        if rev_j and rev_j != edge_j:
            candidates_j.append(rev_j)
        
        # 贪心：边长度 + 方向一致性惩罚
        best_pair = (edge_i, edge_j)
        best_score = float('inf')
        
        for ci in candidates_i:
            for cj in candidates_j:
                ci_is_rev = ci.endswith('_rev')
                cj_is_rev = cj.endswith('_rev')
                
                direction_bonus = 0 if ci_is_rev == cj_is_rev else 1000
                
                len_ci = edge_lengths.get(ci, edge_lengths.get(ci.replace('_rev', ''), 500))
                len_cj = edge_lengths.get(cj, edge_lengths.get(cj.replace('_rev', ''), 500))
                
                score = len_ci + len_cj + direction_bonus
                
                if score < best_score:
                    best_score = score
                    best_pair = (ci, cj)
        
        if i == 0:
            fixed_edges[stop_i] = best_pair[0]
        fixed_edges[stop_j] = best_pair[1]
    
    return fixed_edges


# =============================================================================
# DP 全局优化算法
# =============================================================================

def run_dp_binding(stop_sequence: List[Tuple], stop_to_edge: dict,
                   edge_lengths: dict, reverse_map: dict, junction_edges: dict,
                   edge_from_to: dict, edge_coords: dict, net,
                   config: OptimizationConfig, path_cache: PathCache) -> Dict[str, str]:
    """
    DP 全局优化算法
    
    状态: dp[i][j] = 到达第 i 站选择第 j 个候选边的最小总代价
    代价: 路径长度 + 惩罚项（不包含 KMB ratio）
    """
    n = len(stop_sequence)
    if n == 0:
        return {}
    
    INF = float('inf')
    
    # 生成所有站点的候选边
    all_candidates = []
    for seq, stop_id, name, cum_dist, lat, long in stop_sequence:
        candidates = generate_candidates(
            stop_id, stop_to_edge, reverse_map,
            junction_edges, edge_from_to, edge_lengths,
            edge_coords, (lat, long), config
        )
        all_candidates.append(candidates)
    
    # DP 表大小
    max_cands = max(len(c) for c in all_candidates)
    dp = [[INF] * max_cands for _ in range(n)]
    parent = [[(-1, -1)] * max_cands for _ in range(n)]
    
    # 初始化
    for j in range(len(all_candidates[0])):
        dp[0][j] = 0
    
    # 转移
    for i in range(1, n):
        for j_curr, e_curr in enumerate(all_candidates[i]):
            for j_prev, e_prev in enumerate(all_candidates[i - 1]):
                # 计算路径代价
                if e_prev == e_curr:
                    path_result = PathResult(length=0, edges=[e_prev])
                elif net and HAS_SUMOLIB:
                    path_result = get_shortest_path_sumolib(
                        net, e_prev, e_curr, reverse_map, path_cache
                    )
                else:
                    path_result = get_shortest_path_simple(edge_lengths, e_prev, e_curr)
                
                cost = path_result.length
                
                # 不可达惩罚
                if cost == INF:
                    continue  # 跳过不可达
                
                # 立即回穿惩罚（非结构性必要）
                if path_result.has_immediate_uturn and not path_result.is_structural_reversal:
                    cost += config.penalty_immediate_uturn
                
                # Rev 切换惩罚
                prev_is_rev = e_prev.endswith('_rev')
                curr_is_rev = e_curr.endswith('_rev')
                if prev_is_rev != curr_is_rev:
                    cost += config.penalty_rev_switch
                
                # 更新 DP
                total = dp[i - 1][j_prev] + cost
                if total < dp[i][j_curr]:
                    dp[i][j_curr] = total
                    parent[i][j_curr] = (i - 1, j_prev)
    
    # 找最优终点
    best_end = 0
    best_cost = INF
    for j in range(len(all_candidates[-1])):
        if dp[n - 1][j] < best_cost:
            best_cost = dp[n - 1][j]
            best_end = j
    
    # 如果没有可达路径，回退到使用第一个候选
    if best_cost == INF:
        # 无有效路径，使用每站的第一个候选
        fixed_edges = {}
        for idx, (seq, stop_id, name, cum_dist, lat, long) in enumerate(stop_sequence):
            if all_candidates[idx]:
                fixed_edges[stop_id] = all_candidates[idx][0]
            else:
                fixed_edges[stop_id] = stop_to_edge.get(stop_id, '')
        return fixed_edges
    
    # 回溯
    path = []
    i, j = n - 1, best_end
    visited = set()  # 防止无限循环
    while i >= 0:
        if (i, j) in visited:
            break  # 检测到循环，退出
        visited.add((i, j))
        path.append((i, j))
        if i == 0:
            break
        prev_i, prev_j = parent[i][j]
        if prev_i < 0 or prev_j < 0:
            break  # 无效 parent，退出
        i, j = prev_i, prev_j
    path.reverse()
    
    # 构建结果
    fixed_edges = {}
    for idx, j in path:
        seq, stop_id, name, cum_dist, lat, long = stop_sequence[idx]
        if j < len(all_candidates[idx]) and all_candidates[idx][j]:
            fixed_edges[stop_id] = all_candidates[idx][j]
        elif all_candidates[idx]:
            fixed_edges[stop_id] = all_candidates[idx][0]
        else:
            fixed_edges[stop_id] = stop_to_edge.get(stop_id, '')
    
    # 确保所有站点都有落边
    for seq, stop_id, name, cum_dist, lat, long in stop_sequence:
        if stop_id not in fixed_edges:
            if all_candidates and len(all_candidates) > 0:
                idx = next((i for i, (s, sid, *_) in enumerate(stop_sequence) if sid == stop_id), 0)
                if idx < len(all_candidates) and all_candidates[idx]:
                    fixed_edges[stop_id] = all_candidates[idx][0]
                else:
                    fixed_edges[stop_id] = stop_to_edge.get(stop_id, '')
            else:
                fixed_edges[stop_id] = stop_to_edge.get(stop_id, '')
    
    return fixed_edges


# =============================================================================
# 结果评估
# =============================================================================

def evaluate_result(stop_sequence: List[Tuple], fixed_edges: Dict[str, str],
                    stop_to_edge: dict, edge_lengths: dict, reverse_map: dict,
                    net, config: OptimizationConfig, 
                    path_cache: PathCache) -> Tuple[float, int, int, int, int, List[Tuple]]:
    """
    评估落边结果
    
    返回: (总路径长度, 立即回穿次数, 结构性折返次数, rev切换次数, 不可达次数, 段详情列表)
    """
    total_length = 0.0
    uturn_count = 0
    structural_count = 0
    rev_switch_count = 0
    unreachable_count = 0
    segment_details = []  # [(from_stop, to_stop, path_len, kmb_len, ratio, flags)]
    
    for i in range(len(stop_sequence) - 1):
        seq_i, stop_i, name_i, cum_dist_i, _, _ = stop_sequence[i]
        seq_j, stop_j, name_j, cum_dist_j, _, _ = stop_sequence[i + 1]
        
        edge_i = fixed_edges.get(stop_i, '')
        edge_j = fixed_edges.get(stop_j, '')
        
        kmb_len = cum_dist_j - cum_dist_i
        if kmb_len <= 0:
            kmb_len = 100
        
        # 计算路径
        if edge_i == edge_j:
            path_result = PathResult(length=0, edges=[edge_i])
        elif net and HAS_SUMOLIB:
            path_result = get_shortest_path_sumolib(
                net, edge_i, edge_j, reverse_map, path_cache
            )
        else:
            path_result = get_shortest_path_simple(edge_lengths, edge_i, edge_j)
        
        path_len = path_result.length
        
        if path_len == float('inf'):
            unreachable_count += 1
            ratio = float('inf')
        else:
            total_length += path_len
            ratio = path_len / kmb_len if kmb_len > 0 else 0
        
        if path_result.has_immediate_uturn:
            uturn_count += 1
        if path_result.is_structural_reversal:
            structural_count += 1
        
        # Rev 切换
        prev_is_rev = edge_i.endswith('_rev')
        curr_is_rev = edge_j.endswith('_rev')
        if prev_is_rev != curr_is_rev:
            rev_switch_count += 1
        
        # 标记
        flags = []
        if path_result.has_immediate_uturn:
            flags.append('U-TURN')
        if path_result.is_structural_reversal:
            flags.append('STRUCT')
        if ratio > config.kmb_ratio_warning:
            flags.append(f'RATIO>{config.kmb_ratio_warning:.0f}x')
        if path_len == float('inf'):
            flags.append('UNREACHABLE')
        
        segment_details.append((
            name_i[:20], name_j[:20], 
            path_len, kmb_len, ratio, 
            ','.join(flags) if flags else '-'
        ))
    
    return (total_length, uturn_count, structural_count, 
            rev_switch_count, unreachable_count, segment_details)


def detect_pocket_conflicts(segment_details_greedy: List, 
                            segment_details_dp: List) -> int:
    """
    检测口袋对消现象
    
    定义：如果 greedy 某段"省"但下一段"多"，且差异接近对消
    """
    conflicts = 0
    
    for i in range(len(segment_details_greedy) - 1):
        _, _, len_g1, kmb1, _, _ = segment_details_greedy[i]
        _, _, len_g2, kmb2, _, _ = segment_details_greedy[i + 1]
        _, _, len_d1, _, _, _ = segment_details_dp[i]
        _, _, len_d2, _, _, _ = segment_details_dp[i + 1]
        
        diff1 = len_g1 - len_d1  # 正=greedy更长
        diff2 = len_g2 - len_d2
        
        # 检测对消模式：一个省、一个多，且绝对值相近
        if diff1 * diff2 < 0:  # 符号相反
            if abs(abs(diff1) - abs(diff2)) < min(abs(diff1), abs(diff2)) * 0.5:
                conflicts += 1
    
    return conflicts


# =============================================================================
# 主函数
# =============================================================================

def main():
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml"
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    output_path = PROJECT_ROOT / "config" / "calibration" / "stop_edge_comparison.csv"
    
    config = OptimizationConfig()
    
    print("=" * 80)
    print("[对比实验] 贪心 vs DP 全局站点落边优化")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    edge_lengths, edge_from_to, reverse_map, junction_edges, edge_coords = load_net_topology(net_path)
    print(f"  - 边数: {len(edge_lengths)}")
    print(f"  - reverse 映射: {len(reverse_map)}")
    print(f"  - 边坐标: {len(edge_coords)}")
    
    # 加载 sumolib net
    net = None
    if HAS_SUMOLIB:
        print("  - 加载 sumolib.net...")
        net = sumolib.net.readNet(str(net_path), withInternal=False)
        print(f"  - sumolib 边数: {len(list(net.getEdges()))}")
    
    stop_to_edge, stop_to_lane, stop_pos = load_stop_edges(bus_stops_path)
    print(f"  - 站点: {len(stop_to_edge)}")
    
    df = pd.read_csv(kmb_csv_path)
    
    # 路径缓存
    path_cache = PathCache()
    
    all_results = []
    
    for route in ['68X', '960']:
        for bound in ['inbound', 'outbound']:
            print(f"\n{'='*80}")
            print(f"[{route} {bound}]")
            print("-" * 80)
            
            subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            
            stop_sequence = [
                (row['seq'], row['stop_id'], row['stop_name_en'], 
                 row['cum_dist_m'], row['lat'], row['long'])
                for _, row in subset.iterrows()
            ]
            
            # =====================
            # 过滤只保留 core 段（边在 cropped 网络中的站点）
            # =====================
            core_stop_sequence = []
            for seq, stop_id, name, cum_dist, lat, long in stop_sequence:
                orig_edge = stop_to_edge.get(stop_id, '')
                # 检查边是否在 cropped 网络中
                edge_valid = False
                if orig_edge.startswith(':'):
                    # internal edge - 检查其 junction 的邻接边是否存在
                    parts = orig_edge[1:].rsplit('_', 1)
                    junction_id = parts[0] if len(parts) >= 1 else orig_edge[1:]
                    jinfo = junction_edges.get(junction_id, {})
                    candidates = jinfo.get('incoming', []) + jinfo.get('outgoing', [])
                    edge_valid = any(e in edge_lengths for e in candidates if not e.startswith(':'))
                else:
                    edge_valid = orig_edge in edge_lengths
                
                if edge_valid:
                    core_stop_sequence.append((seq, stop_id, name, cum_dist, lat, long))
            
            # 如果 core 段太短，跳过
            if len(core_stop_sequence) < 3:
                print(f"  ⚠️ Core 段只有 {len(core_stop_sequence)} 站，跳过")
                continue
            
            # 重新计算 KMB 总长度（core 段）
            if core_stop_sequence:
                core_cum_start = core_stop_sequence[0][3]
                core_cum_end = core_stop_sequence[-1][3]
                kmb_total = core_cum_end - core_cum_start
            else:
                kmb_total = 1
            
            print(f"  全线 {len(stop_sequence)} 站 → Core 段 {len(core_stop_sequence)} 站")
            print(f"  Core 范围: seq {core_stop_sequence[0][0]} - {core_stop_sequence[-1][0]}")
            print(f"  KMB Core 长度: {kmb_total:.0f}m")
            
            # 使用 core_stop_sequence 替代 stop_sequence
            stop_sequence = core_stop_sequence
            
            # =====================
            # 贪心算法
            # =====================
            greedy_edges = run_greedy_binding(
                stop_sequence, stop_to_edge, edge_lengths, reverse_map,
                junction_edges, edge_from_to, edge_coords, net, config
            )
            
            (greedy_len, greedy_uturn, greedy_struct, greedy_rev, 
             greedy_unreach, greedy_details) = evaluate_result(
                stop_sequence, greedy_edges, stop_to_edge, edge_lengths,
                reverse_map, net, config, path_cache
            )
            
            # =====================
            # DP 优化
            # =====================
            dp_edges = run_dp_binding(
                stop_sequence, stop_to_edge, edge_lengths, reverse_map,
                junction_edges, edge_from_to, edge_coords, net, config, path_cache
            )
            
            (dp_len, dp_uturn, dp_struct, dp_rev,
             dp_unreach, dp_details) = evaluate_result(
                stop_sequence, dp_edges, stop_to_edge, edge_lengths,
                reverse_map, net, config, path_cache
            )
            
            # 口袋对消检测
            pocket_conflicts = detect_pocket_conflicts(greedy_details, dp_details)
            
            # =====================
            # 输出比较
            # =====================
            greedy_scale = greedy_len / kmb_total if kmb_total > 0 else 0
            dp_scale = dp_len / kmb_total if kmb_total > 0 else 0
            
            print(f"\n{'组':<4} {'方法':<8} {'U-turn':<8} {'结构折返':<10} {'Rev切换':<8} {'不可达':<8} {'总长度(m)':<12} {'Scale':<8}")
            print("-" * 80)
            print(f"{'A':<4} {'贪心':<8} {greedy_uturn:<8} {greedy_struct:<10} {greedy_rev:<8} {greedy_unreach:<8} {greedy_len:<12.0f} {greedy_scale:<8.2f}")
            print(f"{'B':<4} {'DP':<8} {dp_uturn:<8} {dp_struct:<10} {dp_rev:<8} {dp_unreach:<8} {dp_len:<12.0f} {dp_scale:<8.2f}")
            
            # 改善
            len_diff = greedy_len - dp_len
            uturn_diff = greedy_uturn - dp_uturn
            
            print(f"\n[改善] ")
            print(f"  - 路径长度: {len_diff:+.0f}m ({len_diff/greedy_len*100:+.1f}%)" if greedy_len > 0 else "")
            print(f"  - U-turn: {greedy_uturn} → {dp_uturn} ({uturn_diff:+d})")
            print(f"  - 口袋对消检测: {pocket_conflicts} 处")
            
            # 落边变化详情
            changes = []
            for seq, stop_id, name, cum_dist, lat, long in stop_sequence:
                g_edge = greedy_edges.get(stop_id, '')
                d_edge = dp_edges.get(stop_id, '')
                if g_edge != d_edge:
                    changes.append((seq, name[:25], g_edge, d_edge))
            
            if changes:
                print(f"\n[落边变化] {len(changes)} 处:")
                for seq, name, g, d in changes[:10]:  # 只显示前10个
                    print(f"  seq={seq:2d}: {g:20s} → {d:20s} | {name}")
                if len(changes) > 10:
                    print(f"  ... 还有 {len(changes) - 10} 处")
            
            # 保存结果
            for seq, stop_id, name, cum_dist, lat, long in stop_sequence:
                orig = stop_to_edge.get(stop_id, '')
                g_edge = greedy_edges.get(stop_id, orig)
                d_edge = dp_edges.get(stop_id, orig)
                
                all_results.append({
                    'route': route,
                    'bound': bound,
                    'seq': seq,
                    'stop_id': stop_id,
                    'stop_name': name[:30],
                    'orig_edge': orig,
                    'greedy_edge': g_edge,
                    'dp_edge': d_edge,
                    'greedy_changed': orig != g_edge,
                    'dp_changed': orig != d_edge,
                    'methods_differ': g_edge != d_edge,
                })
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\n[输出] {output_path}")
    
    # 缓存统计
    print(f"\n[缓存] {path_cache.stats()}")
    
    print("\n" + "=" * 80)
    print("[验证成功标准]")
    print("-" * 80)
    print("✅ 硬成功 (必须):")
    print("   - U-turn (立即回穿) 次数显著下降")
    print("   - 口袋对消现象消失")
    print("   - 不可达 = 0")
    print("\n⭐ 软成功 (尽力):")
    print("   - 总长度下降 / scale 下降")
    print("=" * 80)


if __name__ == '__main__':
    main()
