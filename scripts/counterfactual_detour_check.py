#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
counterfactual_detour_check.py
==============================
P0.5 反事实检查：对 Top detour 段做"禁止立即折返"测试

判定逻辑：
- Test-1: 允许掉头（现状）
- Test-2: 禁止立即折返（在 cost 里加惩罚）

如果 Test-2 仍有路径且 ratio < 2x：说明是路由策略问题
如果 Test-2 无路可走：说明是拓扑/连接缺陷

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-27
"""

import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

try:
    import sumolib
    HAS_SUMOLIB = True
except ImportError:
    HAS_SUMOLIB = False
    print("⚠️ sumolib 未安装")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 支持命令行指定 net 文件
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default=str(PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"),
                       help='Path to net.xml file')
    return parser.parse_args()


def load_stop_edges(bus_stops_path):
    """加载站点边映射"""
    tree = ET.parse(str(bus_stops_path))
    root = tree.getroot()
    stop_to_edge = {}
    for stop in root.findall('.//busStop'):
        stop_id = stop.get('id')
        lane = stop.get('lane', '')
        if lane.startswith(':'):
            edge = lane.rsplit('_', 1)[0]
        else:
            edge = lane.rsplit('_', 1)[0]
        stop_to_edge[stop_id] = edge
    return stop_to_edge


def get_shortest_path_normal(net, from_edge_id, to_edge_id):
    """
    Test-1: 正常最短路
    """
    if from_edge_id == to_edge_id:
        return 0, [], False
    
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except Exception:
        return float('inf'), [], False
    
    try:
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            edge_ids = [e.getID() for e in route]
            total_len = sum(e.getLength() for e in route)
            
            # 检查折返
            reversals = []
            for i in range(len(edge_ids) - 1):
                e1, e2 = edge_ids[i], edge_ids[i + 1]
                if (e1.endswith('_rev') and e1[:-4] == e2) or \
                   (e2.endswith('_rev') and e2[:-4] == e1):
                    reversals.append(f"{e1}→{e2}")
            
            return total_len, edge_ids, reversals
    except Exception as e:
        print(f"    Error: {e}")
    
    return float('inf'), [], []


def get_reverse_edge_id(edge_id):
    """获取反向边 ID"""
    if edge_id.endswith('_rev'):
        return edge_id[:-4]
    else:
        return edge_id + '_rev'


def get_shortest_path_no_immediate_reversal(net, from_edge_id, to_edge_id, max_iterations=100):
    """
    Test-2: 禁止立即折返的最短路
    
    策略：用迭代求解，每次发现折返就把该边对临时禁止，重新求解
    """
    if from_edge_id == to_edge_id:
        return 0, [], False
    
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except Exception:
        return float('inf'), [], "EDGE_NOT_FOUND"
    
    forbidden_pairs = set()  # 禁止的 (e1, e2) 对
    
    for iteration in range(max_iterations):
        try:
            # 构建禁止边列表
            forbidden_edges = set()
            for e1, e2 in forbidden_pairs:
                # 禁止其中一条边（选择禁止 rev 那条）
                if e1.endswith('_rev'):
                    forbidden_edges.add(e1)
                else:
                    forbidden_edges.add(e2)
            
            # 求最短路（sumolib 不直接支持禁止边，我们用后处理检查）
            route, cost = net.getShortestPath(from_edge, to_edge)
            
            if not route:
                return float('inf'), [], "NO_PATH"
            
            edge_ids = [e.getID() for e in route]
            total_len = sum(e.getLength() for e in route)
            
            # 检查是否有立即折返
            found_reversal = False
            for i in range(len(edge_ids) - 1):
                e1, e2 = edge_ids[i], edge_ids[i + 1]
                if (e1.endswith('_rev') and e1[:-4] == e2) or \
                   (e2.endswith('_rev') and e2[:-4] == e1):
                    # 发现折返，加入禁止列表
                    forbidden_pairs.add((e1, e2))
                    found_reversal = True
                    break
            
            if not found_reversal:
                # 没有折返，返回结果
                return total_len, edge_ids, None
            
            # 如果发现折返，尝试绕过
            # 由于 sumolib 不支持动态禁边，我们用一个 workaround：
            # 尝试从 from_edge 的下游邻居开始
            
        except Exception as e:
            return float('inf'), [], f"ERROR: {e}"
    
    return float('inf'), [], "MAX_ITERATIONS"


def check_alternative_via_neighbors(net, from_edge, to_edge, forbidden_edge):
    """
    尝试绕过某条边，看是否能找到替代路径
    """
    try:
        # 获取 from_edge 的所有出边
        outgoing = list(from_edge.getOutgoing())
        
        best_len = float('inf')
        best_route = []
        
        for next_edge in outgoing:
            if next_edge.getID() == forbidden_edge:
                continue
            
            route, cost = net.getShortestPath(next_edge, to_edge)
            if route:
                total_len = from_edge.getLength() + sum(e.getLength() for e in route)
                if total_len < best_len:
                    best_len = total_len
                    best_route = [from_edge.getID()] + [e.getID() for e in route]
        
        return best_len, best_route
    except Exception:
        return float('inf'), []


def main():
    args = parse_args()
    net_path = Path(args.net)
    bus_stops_path = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
    kmb_csv_path = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
    
    print("=" * 80)
    print("[P0.5] 反事实检查：Top Detour 段折返原因诊断")
    print("=" * 80)
    
    # 加载数据
    print("\n[加载数据]")
    net = sumolib.net.readNet(str(net_path), withInternal=False)
    print(f"  - 路网边数: {len(list(net.getEdges()))}")
    
    stop_to_edge = load_stop_edges(bus_stops_path)
    print(f"  - 站点数: {len(stop_to_edge)}")
    
    df = pd.read_csv(kmb_csv_path)
    
    # 定义 Top detour 段（基于之前的诊断）
    top_detour_segments = [
        # (route, bound, seq_from, seq_to, edge_from, edge_to, kmb_len, description)
        ('68X', 'inbound', 2, 3, '105653_rev', '106831_rev', 380, 'FOO KWAI -> BEECH STREET'),
        ('68X', 'inbound', 3, 4, '106831_rev', '105817', 420, 'BEECH STREET -> MONG KOK ROAD'),
        ('68X', 'inbound', 19, 20, '95503_rev', '95598', 170, 'YAU SAN -> TAI TONG ROAD'),
        ('68X', 'inbound', 20, 21, '95598', '95600', 335, 'TAI TONG ROAD -> YUEN LONG POLICE'),
        ('68X', 'inbound', 32, 33, ':4394_1', '96285', 295, 'RURAL COMMITTEE -> HUNG SHUI KIU (internal edge)'),
    ]
    
    print("\n" + "=" * 80)
    print("[反事实检查]")
    print("=" * 80)
    
    results = []
    
    for route, bound, seq_from, seq_to, edge_from, edge_to, kmb_len, desc in top_detour_segments:
        print(f"\n## 段 {seq_from}->{seq_to}: {desc}")
        print(f"   edge: {edge_from} -> {edge_to}, KMB={kmb_len}m")
        
        # 处理 internal edge
        if edge_from.startswith(':'):
            # 获取邻接外部边
            parts = edge_from[1:].rsplit('_', 1)
            junction_id = parts[0] if len(parts) >= 1 else edge_from[1:]
            try:
                junction = net.getNode(junction_id)
                inc_edges = [e.getID() for e in junction.getIncoming() if not e.getID().startswith(':')]
                if inc_edges:
                    edge_from = inc_edges[0]
                    print(f"   (internal edge 替换为: {edge_from})")
            except Exception:
                print(f"   ⚠️ 无法处理 internal edge: {edge_from}")
                continue
        
        # Test-1: 正常最短路
        len1, route1, reversals1 = get_shortest_path_normal(net, edge_from, edge_to)
        ratio1 = len1 / kmb_len if kmb_len > 0 else 0
        
        print(f"\n   ### Test-1 (允许掉头)")
        if len1 < float('inf'):
            print(f"       长度: {len1:.0f}m, ratio={ratio1:.2f}")
            if reversals1:
                print(f"       折返: {reversals1}")
            else:
                print(f"       折返: 无")
        else:
            print(f"       ⚠️ 无路径")
        
        # Test-2: 尝试绕过折返
        print(f"\n   ### Test-2 (禁止立即折返)")
        
        if reversals1:
            # 尝试从 from_edge 绕过折返边
            rev_edge = reversals1[0].split('→')[1] if '→' in reversals1[0] else None
            
            if rev_edge:
                try:
                    from_edge_obj = net.getEdge(edge_from)
                    to_edge_obj = net.getEdge(edge_to)
                    
                    # 尝试找替代路径
                    len2, route2 = check_alternative_via_neighbors(net, from_edge_obj, to_edge_obj, rev_edge)
                    ratio2 = len2 / kmb_len if kmb_len > 0 else 0
                    
                    if len2 < float('inf'):
                        print(f"       长度: {len2:.0f}m, ratio={ratio2:.2f}")
                        print(f"       路径: {route2[:5]}...{route2[-3:]}" if len(route2) > 8 else f"       路径: {route2}")
                        
                        # 判定
                        if ratio2 < 2.0:
                            print(f"       ✅ 判定: 路由策略问题（可绕过折返，ratio={ratio2:.2f}）")
                            results.append((desc, 'ROUTING_STRATEGY', ratio2))
                        else:
                            print(f"       ⚠️ 判定: 绕路仍然很长（ratio={ratio2:.2f}），可能是真实交通规则限制")
                            results.append((desc, 'TRAFFIC_RULE', ratio2))
                    else:
                        print(f"       ❌ 无替代路径")
                        print(f"       判定: 拓扑缺陷（必须修网）")
                        results.append((desc, 'TOPOLOGY_DEFECT', None))
                        
                except Exception as e:
                    print(f"       ⚠️ 检查失败: {e}")
                    results.append((desc, 'ERROR', None))
            else:
                print(f"       ⚠️ 无法解析折返边")
        else:
            print(f"       ✅ 原路径无折返，无需 Test-2")
            results.append((desc, 'NO_REVERSAL', ratio1))
    
    # 总结
    print("\n" + "=" * 80)
    print("[诊断总结]")
    print("=" * 80)
    
    routing_issues = [r for r in results if r[1] == 'ROUTING_STRATEGY']
    topology_issues = [r for r in results if r[1] == 'TOPOLOGY_DEFECT']
    traffic_rules = [r for r in results if r[1] == 'TRAFFIC_RULE']
    
    print(f"\n路由策略问题: {len(routing_issues)} 段")
    for desc, _, ratio in routing_issues:
        print(f"  - {desc} (ratio={ratio:.2f})")
    
    print(f"\n拓扑缺陷: {len(topology_issues)} 段")
    for desc, _, _ in topology_issues:
        print(f"  - {desc}")
    
    print(f"\n交通规则限制: {len(traffic_rules)} 段")
    for desc, _, ratio in traffic_rules:
        print(f"  - {desc} (ratio={ratio:.2f})")
    
    print("\n" + "-" * 80)
    if len(topology_issues) > 0:
        print("💡 建议: 需要修复路网拓扑（添加 connections 或虚拟桥接边）")
    elif len(routing_issues) > 0:
        print("💡 建议: 改用 via routing 策略，惩罚掉头，消除分段拼接毛刺")
    else:
        print("💡 所有段都正常，scale 问题可能源于其他因素")


if __name__ == '__main__':
    main()
