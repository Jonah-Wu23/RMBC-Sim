#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_seg2_gdb.py
====================
seg2 GDB 闸门验证：查询 GDB 中从 105653 到 106831 的可行车行路径
"""

import sys
sys.path.insert(0, '.')

import geopandas as gpd
import sumolib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml"
GDB_PATH = PROJECT_ROOT / "data" / "external" / "iRN" / "IRN_GDB.gdb"

print("=" * 70)
print("seg2 GDB 闸门验证")
print("=" * 70)

# 1. 获取 seg2 的 SUMO 边信息
print("\n[1] SUMO seg2 边信息")
net = sumolib.net.readNet(str(NET_FILE))

edges_info = {}
for eid in ['105653', '106831']:
    e = net.getEdge(eid)
    if e:
        from_node = e.getFromNode()
        to_node = e.getToNode()
        shape = e.getShape()
        edges_info[eid] = {
            'from_node': from_node.getID(),
            'to_node': to_node.getID(),
            'length': e.getLength(),
            'from_coord': from_node.getCoord(),
            'to_coord': to_node.getCoord(),
            'start_point': shape[0],
            'end_point': shape[-1]
        }
        print(f"  {eid}: {from_node.getID()} → {to_node.getID()}, len={e.getLength():.0f}m")
        print(f"    coords: ({shape[0][0]:.0f}, {shape[0][1]:.0f}) → ({shape[-1][0]:.0f}, {shape[-1][1]:.0f})")

# 2. 计算 SUMO 最短路径
print("\n[2] SUMO 最短路径 (105653 → 106831)")
from_edge = net.getEdge('105653')
to_edge = net.getEdge('106831')

route, cost = net.getShortestPath(from_edge, to_edge, vClass='bus')
if route:
    total_len = sum(e.getLength() for e in route)
    print(f"  路径长度: {total_len:.0f}m ({len(route)} edges)")
    print("  前 10 条边:")
    for i, e in enumerate(route[:10]):
        print(f"    {i}: {e.getID()} ({e.getLength():.0f}m)")
    if len(route) > 10:
        print(f"    ... ({len(route) - 10} more edges)")
        print(f"    {len(route)-1}: {route[-1].getID()} ({route[-1].getLength():.0f}m)")
else:
    print("  无路径!")

# 3. 读取 GDB 数据
print("\n[3] GDB 路网查询")
try:
    irn = gpd.read_file(str(GDB_PATH), layer="IRN")
    print(f"  GDB 记录数: {len(irn)}")
    
    # 查找 105653 和 106831 对应的 GDB 记录
    for eid in ['105653', '106831']:
        base_id = int(eid.replace('_rev', ''))
        matches = irn[irn['OBJECTID'] == base_id]
        if len(matches) > 0:
            row = matches.iloc[0]
            print(f"\n  GDB {eid}:")
            print(f"    ROAD_NAME: {row.get('ROAD_NAME', 'N/A')}")
            print(f"    TRAVEL_DIRECTION: {row.get('TRAVEL_DIRECTION', 'N/A')}")
            geom = row.geometry
            if geom:
                coords = list(geom.coords)
                print(f"    起点: ({coords[0][0]:.0f}, {coords[0][1]:.0f})")
                print(f"    终点: ({coords[-1][0]:.0f}, {coords[-1][1]:.0f})")
        else:
            print(f"\n  GDB {eid}: 未找到")

    # 4. 查找 seg2 区域内的道路 (基于起终点坐标)
    print("\n[4] seg2 区域 GDB 道路搜索")
    # seg2: 105653 终点附近 到 106831 起点附近
    if route:
        # 使用第一条非起始边的起点作为搜索区域中心
        start_edge = route[0]
        end_edge = route[-1]
        
        # 搜索半径 (米)
        search_radius = 300
        
        # 从起终点坐标搜索
        start_coord = edges_info['105653']['end_point']  # 105653 终点
        end_coord = edges_info['106831']['start_point']  # 106831 起点
        
        print(f"  起点区域: ({start_coord[0]:.0f}, {start_coord[1]:.0f})")
        print(f"  终点区域: ({end_coord[0]:.0f}, {end_coord[1]:.0f})")
        
        # 计算直线距离
        import math
        direct_dist = math.sqrt((end_coord[0] - start_coord[0])**2 + (end_coord[1] - start_coord[1])**2)
        print(f"  直线距离: {direct_dist:.0f}m")
        print(f"  KMB 长度: 206m")
        print(f"  SUMO 长度: {total_len:.0f}m")
        print(f"  Ratio: {total_len/206:.2f}")

except Exception as e:
    print(f"  GDB 读取错误: {e}")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
