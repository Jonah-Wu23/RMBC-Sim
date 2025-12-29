#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_topology_defect.py
===========================
诊断网络拓扑缺陷：
1. 检查两条边是否共享节点
2. 检查节点上是否存在 connection
3. 检查 allow/disallow 约束

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-27
"""

import sys
from pathlib import Path

try:
    import sumolib
    HAS_SUMOLIB = True
except ImportError:
    HAS_SUMOLIB = False
    print("⚠️ sumolib 未安装")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def diagnose_edge_pair(net, from_edge_id, to_edge_id):
    """
    诊断一对边的连接问题
    
    Returns:
        dict with diagnosis results
    """
    result = {
        'from_edge': from_edge_id,
        'to_edge': to_edge_id,
        'from_edge_exists': False,
        'to_edge_exists': False,
        'shares_node': False,
        'shared_node_id': None,
        'has_connection': False,
        'connection_details': [],
        'from_outgoing': [],
        'alternative_next_edges': [],
        'from_to_node': None,
        'to_from_node': None,
        'node_distance': None,
        'diagnosis': 'UNKNOWN'
    }
    
    # 检查边是否存在
    try:
        from_edge = net.getEdge(from_edge_id)
        result['from_edge_exists'] = True
        result['from_to_node'] = from_edge.getToNode().getID()
    except Exception as e:
        result['diagnosis'] = f'FROM_EDGE_NOT_FOUND: {e}'
        return result
    
    try:
        to_edge = net.getEdge(to_edge_id)
        result['to_edge_exists'] = True
        result['to_from_node'] = to_edge.getFromNode().getID()
    except Exception as e:
        result['diagnosis'] = f'TO_EDGE_NOT_FOUND: {e}'
        return result
    
    # 检查是否共享节点
    from_to_node = from_edge.getToNode()
    to_from_node = to_edge.getFromNode()
    
    if from_to_node.getID() == to_from_node.getID():
        result['shares_node'] = True
        result['shared_node_id'] = from_to_node.getID()
        
        # 检查共享节点上的 connection
        junction = from_to_node
        connections = junction.getConnections()
        
        for conn in connections:
            conn_from = conn.getFrom().getID() if conn.getFrom() else None
            conn_to = conn.getTo().getID() if conn.getTo() else None
            
            if conn_from == from_edge_id and conn_to == to_edge_id:
                result['has_connection'] = True
                result['connection_details'].append({
                    'fromLane': conn.getFromLane().getIndex(),
                    'toLane': conn.getToLane().getIndex(),
                    'direction': conn.getDirection(),
                })
        
        if result['has_connection']:
            result['diagnosis'] = 'CONNECTION_EXISTS_BUT_MAYBE_DISALLOWED'
        else:
            result['diagnosis'] = 'SHARES_NODE_BUT_NO_CONNECTION'
    else:
        # 不共享节点 - 计算距离
        from_coords = from_to_node.getCoord()
        to_coords = to_from_node.getCoord()
        import math
        dist = math.sqrt((from_coords[0] - to_coords[0])**2 + (from_coords[1] - to_coords[1])**2)
        result['node_distance'] = dist
        
        if dist < 20:  # 20米以内可能是拼接误差
            result['diagnosis'] = 'NODES_CLOSE_BUT_NOT_SHARED'
        else:
            result['diagnosis'] = 'NODES_FAR_APART'
    
    # 获取 from_edge 的所有出边
    outgoing = list(from_edge.getOutgoing())
    result['from_outgoing'] = [e.getID() for e in outgoing]
    
    # 找可能的替代出边（不是立即折返的）
    for next_edge in outgoing:
        next_id = next_edge.getID()
        # 跳过立即折返
        if (from_edge_id.endswith('_rev') and from_edge_id[:-4] == next_id) or \
           (next_id.endswith('_rev') and next_id[:-4] == from_edge_id):
            continue
        result['alternative_next_edges'].append(next_id)
    
    return result


def main():
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
    
    print("=" * 80)
    print("[拓扑缺陷深度诊断]")
    print("=" * 80)
    
    print("\n[加载路网]")
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    print(f"  - 路网边数: {len(list(net.getEdges()))}")
    
    # 问题段列表
    problem_pairs = [
        ('106831_rev', '105817', 'BEECH STREET -> MONG KOK ROAD (Test-2 无替代路径)'),
        ('105653_rev', '106831_rev', 'FOO KWAI -> BEECH STREET'),
        ('95503_rev', '95598', 'YAU SAN -> TAI TONG ROAD'),
        ('95598', '95600', 'TAI TONG ROAD -> YUEN LONG POLICE'),
    ]
    
    for from_edge, to_edge, desc in problem_pairs:
        print(f"\n{'='*80}")
        print(f"## {desc}")
        print(f"   from: {from_edge} -> to: {to_edge}")
        print("-" * 80)
        
        result = diagnose_edge_pair(net, from_edge, to_edge)
        
        print(f"\n[边存在性]")
        print(f"  from_edge exists: {result['from_edge_exists']}")
        print(f"  to_edge exists: {result['to_edge_exists']}")
        
        print(f"\n[节点信息]")
        print(f"  from_edge.toNode: {result['from_to_node']}")
        print(f"  to_edge.fromNode: {result['to_from_node']}")
        print(f"  shares_node: {result['shares_node']}")
        if result['shared_node_id']:
            print(f"  shared_node_id: {result['shared_node_id']}")
        if result['node_distance'] is not None:
            print(f"  node_distance: {result['node_distance']:.2f}m")
        
        print(f"\n[Connection 信息]")
        print(f"  has_connection: {result['has_connection']}")
        if result['connection_details']:
            for conn in result['connection_details']:
                print(f"    - Lane {conn['fromLane']} -> Lane {conn['toLane']}, dir={conn['direction']}")
        
        print(f"\n[出边分析]")
        print(f"  from_edge 的所有出边 ({len(result['from_outgoing'])}): {result['from_outgoing'][:10]}...")
        print(f"  非折返出边 ({len(result['alternative_next_edges'])}): {result['alternative_next_edges'][:5]}...")
        
        print(f"\n[诊断结论]")
        print(f"  >>> {result['diagnosis']} <<<")
        
        # 给出修复建议
        if result['diagnosis'] == 'SHARES_NODE_BUT_NO_CONNECTION':
            print(f"\n[修复建议]")
            print(f"  在 con.xml 中添加:")
            print(f'  <connection from="{from_edge}" to="{to_edge}" fromLane="0" toLane="0" />')
        elif result['diagnosis'] == 'NODES_CLOSE_BUT_NOT_SHARED':
            print(f"\n[修复建议]")
            print(f"  尝试 netconvert --junctions.join --junctions.join-dist {max(10, int(result['node_distance'])+5)}")
        elif result['diagnosis'] == 'NODES_FAR_APART':
            print(f"\n[修复建议]")
            print(f"  需要添加桥接边 (bridge edge) 连接两个节点")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
