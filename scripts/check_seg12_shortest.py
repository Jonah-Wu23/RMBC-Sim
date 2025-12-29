#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 seg12 最短路径"""

import sumolib

net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')

# 使用 sumolib 的最短路径计算从 272309 到 9023_rev
print('=== Calculating shortest paths ===')

from_edge = net.getEdge('272309')
to_edge = net.getEdge('9023_rev')

# passenger 类型（无 vType 限制）
route_p, cost_p = net.getShortestPath(from_edge, to_edge, vClass='passenger')
if route_p:
    path_len = sum(e.getLength() for e in route_p)
    print(f'Passenger: {len(route_p)} edges, {path_len:.0f}m')
else:
    print('Passenger: No path!')

# bus 类型
route_b, cost_b = net.getShortestPath(from_edge, to_edge, vClass='bus')
if route_b:
    path_len = sum(e.getLength() for e in route_b)
    print(f'Bus: {len(route_b)} edges, {path_len:.0f}m')
    print('  Full path:')
    for i, e in enumerate(route_b):
        print(f'    {i}: {e.getID()} ({e.getLength():.0f}m)')
else:
    print('Bus: No path!')
