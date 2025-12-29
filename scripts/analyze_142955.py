#!/usr/bin/env python
"""分析 GDB 中 142955 道路的拓扑连接情况"""
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
import numpy as np

gdb_path = 'data/RdNet_IRNP.gdb'
print('读取 CENTERLINE 图层...')
gdf = gpd.read_file(gdb_path, layer='CENTERLINE')

# 找到 ROUTE_ID = 142955 的道路
road_142955 = gdf[gdf['ROUTE_ID'] == 142955]
if len(road_142955) == 0:
    print('未找到 ROUTE_ID=142955')
    exit()

print(f'\n道路 142955 信息:')
row = road_142955.iloc[0]
print(f'  街道名: {row["STREET_ENAME"]} / {row["STREET_CNAME"]}')
print(f'  长度: {row["SHAPE_Length"]:.1f}m')
geom = row.geometry

# 获取道路的端点
if geom.geom_type == 'MultiLineString':
    # 获取所有线段的起点和终点
    coords = []
    for line in geom.geoms:
        coords.extend(list(line.coords))
    start_pt = Point(coords[0])
    end_pt = Point(coords[-1])
else:
    coords = list(geom.coords)
    start_pt = Point(coords[0])
    end_pt = Point(coords[-1])

print(f'  起点: ({start_pt.x:.0f}, {start_pt.y:.0f})')
print(f'  终点: ({end_pt.x:.0f}, {end_pt.y:.0f})')

# 查找与 142955 端点相连的其他道路
print('\n查找与 142955 相连的道路:')

def find_connected_roads(point, gdf, exclude_route_id, threshold=5):
    """找出在 point 附近（threshold 米内）且端点相连的道路"""
    connected = []
    for idx, row in gdf.iterrows():
        if row['ROUTE_ID'] == exclude_route_id:
            continue
        geom = row.geometry
        if geom is None:
            continue
        # 获取道路端点
        if geom.geom_type == 'MultiLineString':
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            coords = list(geom.coords)
        
        road_start = Point(coords[0])
        road_end = Point(coords[-1])
        
        # 检查是否与 point 相连
        dist_start = point.distance(road_start)
        dist_end = point.distance(road_end)
        
        if dist_start < threshold or dist_end < threshold:
            connected.append({
                'route_id': row['ROUTE_ID'],
                'name': row['STREET_ENAME'] or 'N/A',
                'length': row['SHAPE_Length'],
                'connection': 'start' if dist_start < threshold else 'end',
                'distance': min(dist_start, dist_end)
            })
    return connected

# 检查起点连接
print('\n142955 起点相连的道路:')
connected_start = find_connected_roads(start_pt, gdf, 142955)
for r in connected_start:
    print(f'  {r["route_id"]}: {r["name"]} ({r["length"]:.1f}m)')

# 检查终点连接
print('\n142955 终点相连的道路:')
connected_end = find_connected_roads(end_pt, gdf, 142955)
for r in connected_end:
    print(f'  {r["route_id"]}: {r["name"]} ({r["length"]:.1f}m)')

if len(connected_end) == 0:
    print('  ❌ 终点无连接！这是原始数据的死胡同！')
