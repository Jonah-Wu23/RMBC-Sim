#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""seg1 Stop 落边分析"""

import pandas as pd
import sumolib

net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')
corr = pd.read_csv('config/calibration/stop_edge_corrections.csv')

print('=== seg1 Stop 落边分析 ===')
print('seg1: 105735_rev -> 105653')
print()

for seq in [1, 2]:
    row = corr[(corr['route']=='68X') & (corr['bound']=='inbound') & (corr['seq']==seq)].iloc[0]
    fixed = row['fixed_edge']
    orig = row['orig_edge']
    stop_name = row['stop_name']
    stop_id = row['stop_id'][:10]
    
    print(f'=== seq {seq}: {stop_name} ===')
    print(f'stop_id: {stop_id}')
    print(f'orig_edge: {orig}')
    print(f'fixed_edge: {fixed}')
    
    # 检查 _rev 版本
    base = fixed.replace('_rev', '') if '_rev' in fixed else fixed
    rev_edge = base + '_rev' if '_rev' not in fixed else base
    
    has_rev = net.hasEdge(rev_edge)
    print(f'反向边 {rev_edge}: {"存在" if has_rev else "不存在"}')
    
    # 边详情
    if net.hasEdge(fixed):
        e = net.getEdge(fixed)
        print(f'{fixed}: node {e.getFromNode().getID()} -> {e.getToNode().getID()}, len={e.getLength():.0f}m')
    if has_rev:
        e_rev = net.getEdge(rev_edge)
        print(f'{rev_edge}: node {e_rev.getFromNode().getID()} -> {e_rev.getToNode().getID()}, len={e_rev.getLength():.0f}m')
    
    # 查询 GDB 方向
    print()

# 检查 seg1 路径
print('=== seg1 当前路径 ===')
from_e = net.getEdge('105735_rev')
to_e = net.getEdge('105653')
route, cost = net.getShortestPath(from_e, to_e, vClass='bus')
if route:
    total = sum(e.getLength() for e in route)
    print(f'105735_rev -> 105653: {total:.0f}m ({len(route)} edges)')
    print('前 8 条边:')
    for i, e in enumerate(route[:8]):
        print(f'  {i}: {e.getID()} ({e.getLength():.0f}m)')

# 测试备选落边
print()
print('=== 备选落边测试 ===')

# 如果起点改为 105735 (不是 _rev)
if net.hasEdge('105735'):
    from_alt = net.getEdge('105735')
    route_alt, _ = net.getShortestPath(from_alt, to_e, vClass='bus')
    if route_alt:
        total_alt = sum(e.getLength() for e in route_alt)
        print(f'105735 -> 105653: {total_alt:.0f}m ({len(route_alt)} edges)')

# 如果终点改为 105653_rev
if net.hasEdge('105653_rev'):
    to_alt = net.getEdge('105653_rev')
    route_alt2, _ = net.getShortestPath(from_e, to_alt, vClass='bus')
    if route_alt2:
        total_alt2 = sum(e.getLength() for e in route_alt2)
        print(f'105735_rev -> 105653_rev: {total_alt2:.0f}m ({len(route_alt2)} edges)')
