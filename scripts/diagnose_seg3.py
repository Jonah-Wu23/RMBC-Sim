#!/usr/bin/env python
"""seg3 深度诊断: 106831 → 105817"""
import sumolib

net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')

from_e = '106831'
to_e = '105817'

print('='*70)
print('seg3 深度诊断: 106831 → 105817')
print('='*70)

# A) bus & passenger 最短路对比
print('\nA) Bus & Passenger 最短路对比:')
print('-'*50)

path_bus, cost_bus = net.getOptimalPath(net.getEdge(from_e), net.getEdge(to_e), vClass='bus')
path_pass, cost_pass = net.getOptimalPath(net.getEdge(from_e), net.getEdge(to_e), vClass='passenger')

print(f'Bus:       {cost_bus:.0f}m ({len(path_bus) if path_bus else 0} edges)')
print(f'Passenger: {cost_pass:.0f}m ({len(path_pass) if path_pass else 0} edges)')
print(f'KMB 长度:  483m')

# 端点距离
e1 = net.getEdge(from_e)
e2 = net.getEdge(to_e)
c1 = e1.getToNode().getCoord()
c2 = e2.getFromNode().getCoord()
dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
print(f'端点直线距离: {dist:.0f}m')

# B) 最短路径 edge 序列
print('\nB) 最短路径 edge 序列:')
print('-'*50)
if path_pass:
    edge_ids = [e.getID() for e in path_pass]
    print(f'前10: {edge_ids[:10]}')
    if len(edge_ids) > 10:
        print(f'后10: {edge_ids[-10:]}')

# C) 检查关键单向边
print('\nC) 路径中边的反向边检查:')
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
            print(f'  {eid} -> {rev_eid}: 不存在 ← 单向!')

if not missing_rev:
    print('  所有边都有反向边（双向道路）')
else:
    print(f'\n共 {len(missing_rev)} 条单向边')

# D) 检查是否存在"口袋回头"结构
print('\nD) 口袋回头结构检查:')
print('-'*50)
if path_pass:
    edge_ids = [e.getID() for e in path_pass]
    # 检查是否出现 A_rev -> A 这种模式
    for i in range(len(edge_ids)-1):
        e1 = edge_ids[i]
        e2 = edge_ids[i+1]
        if e1.endswith('_rev'):
            base = e1[:-4]
            if e2 == base:
                print(f'  口袋回头: {e1} -> {e2}')
        else:
            if e2 == e1 + '_rev':
                print(f'  口袋回头: {e1} -> {e2}')

# 特别检查 106831 附近
print('\n106831 口袋结构详情:')
e = net.getEdge('106831')
print(f'  106831 出边: {[o.getID() for o in e.getOutgoing().keys()]}')
e_rev = net.getEdge('106831_rev')
print(f'  106831_rev 出边: {[o.getID() for o in e_rev.getOutgoing().keys()]}')
