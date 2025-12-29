#!/usr/bin/env python
"""seg12 深度诊断: 272309 → 9023_rev"""
import sumolib

net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')

from_e = '272309'
to_e = '9023_rev'
to_e_alt = '9023'

print('='*70)
print('seg12 深度诊断: 272309 → 9023_rev')
print('='*70)

# 1) bus vs passenger 最短路
print('\n1) Bus & Passenger 最短路对比:')
print('-'*50)

path_bus, cost_bus = net.getOptimalPath(net.getEdge(from_e), net.getEdge(to_e), vClass='bus')
path_pass, cost_pass = net.getOptimalPath(net.getEdge(from_e), net.getEdge(to_e), vClass='passenger')

print(f'To 9023_rev:')
print(f'  Bus:       {cost_bus:.0f}m ({len(path_bus) if path_bus else 0} edges)')
print(f'  Passenger: {cost_pass:.0f}m ({len(path_pass) if path_pass else 0} edges)')

# 测试 to=9023 对比
path_bus_alt, cost_bus_alt = net.getOptimalPath(net.getEdge(from_e), net.getEdge(to_e_alt), vClass='bus')
path_pass_alt, cost_pass_alt = net.getOptimalPath(net.getEdge(from_e), net.getEdge(to_e_alt), vClass='passenger')

print(f'To 9023 (对比):')
print(f'  Bus:       {cost_bus_alt:.0f}m ({len(path_bus_alt) if path_bus_alt else 0} edges)')
print(f'  Passenger: {cost_pass_alt:.0f}m ({len(path_pass_alt) if path_pass_alt else 0} edges)')

print(f'\nKMB 长度:  506m')

# 端点距离
e1 = net.getEdge(from_e)
e2 = net.getEdge(to_e)
c1 = e1.getToNode().getCoord()
c2 = e2.getFromNode().getCoord()
dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
print(f'端点直线距离: {dist:.0f}m')

# 2) 最短路径 edge 序列
print('\n2) 最短路径 edge 序列 (to 9023_rev):')
print('-'*50)
if path_pass:
    edge_ids = [e.getID() for e in path_pass]
    print(f'前12: {edge_ids[:12]}')
    if len(edge_ids) > 12:
        print(f'后12: {edge_ids[-12:]}')

# 3) 端点节点信息
print('\n3) 端点节点信息:')
print('-'*50)

e_272309 = net.getEdge('272309')
tn_272309 = e_272309.getToNode()
print(f'272309.toNode: {tn_272309.getID()} at {tn_272309.getCoord()}')
print(f'  出边: {[e.getID() for e in tn_272309.getOutgoing()]}')

e_9023_rev = net.getEdge('9023_rev')
fn_9023_rev = e_9023_rev.getFromNode()
print(f'9023_rev.fromNode: {fn_9023_rev.getID()} at {fn_9023_rev.getCoord()}')
print(f'  入边: {[e.getID() for e in fn_9023_rev.getIncoming()]}')

e_9023 = net.getEdge('9023')
fn_9023 = e_9023.getFromNode()
print(f'9023.fromNode: {fn_9023.getID()} at {fn_9023.getCoord()}')
print(f'  入边: {[e.getID() for e in fn_9023.getIncoming()]}')

# 检查是否有反向边缺失
print('\n4) 路径中单向边检查:')
print('-'*50)
missing_rev = []
if path_pass:
    for e in path_pass:
        eid = e.getID()
        if eid.endswith('_rev'):
            rev_eid = eid[:-4]
        else:
            rev_eid = eid + '_rev'
        exists = net.hasEdge(rev_eid)
        if not exists:
            missing_rev.append(eid)
            print(f'  {eid}: 无反向边 (单向)')

if not missing_rev:
    print('  所有边都有反向边（双向道路）')
else:
    print(f'\n共 {len(missing_rev)} 条单向边')
