#!/usr/bin/env python
"""
检查 GDB 中 105735_rev 到 105653 之间的连通走廊
目标：判断是"节点未融合"、"连接缺失"还是"道路缺失"
"""
import geopandas as gpd
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union
import numpy as np

# SUMO 偏移量
OFFSET_X, OFFSET_Y = -812340.81, -812121.37

# 两个端点（SUMO 坐标）
P1_SUMO = (22601, 7466)  # 105735_rev.toNode
P2_SUMO = (22285, 7752)  # 105653.fromNode

# 转换到 GDB 坐标
P1_GDB = (P1_SUMO[0] - OFFSET_X, P1_SUMO[1] - OFFSET_Y)
P2_GDB = (P2_SUMO[0] - OFFSET_X, P2_SUMO[1] - OFFSET_Y)

print("="*70)
print("GDB 连通走廊分析: 105735_rev → 105653")
print("="*70)
print(f"\nP1 (105735_rev.toNode): SUMO{P1_SUMO} → GDB({P1_GDB[0]:.0f}, {P1_GDB[1]:.0f})")
print(f"P2 (105653.fromNode):   SUMO{P2_SUMO} → GDB({P2_GDB[0]:.0f}, {P2_GDB[1]:.0f})")
print(f"直线距离: {((P1_GDB[0]-P2_GDB[0])**2 + (P1_GDB[1]-P2_GDB[1])**2)**0.5:.0f}m")

# 加载 GDB
gdb_path = 'data/RdNet_IRNP.gdb'
print(f"\n加载 GDB: {gdb_path}")
gdf = gpd.read_file(gdb_path, layer='CENTERLINE')

# 创建搜索区域（以两点为中心，半径 700m）
center = ((P1_GDB[0]+P2_GDB[0])/2, (P1_GDB[1]+P2_GDB[1])/2)
radius = 700
search_box = box(center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius)

# 过滤该区域的道路
roads_in_area = gdf[gdf.geometry.intersects(search_box)]
print(f"\n搜索区域 (半径{radius}m) 内道路数: {len(roads_in_area)}")

# 检查 P1 和 P2 附近的道路端点
def find_nearby_road_endpoints(point, roads, threshold=50):
    """找出距离 point 在 threshold 米内的道路端点"""
    nearby = []
    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == 'MultiLineString':
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            coords = list(geom.coords)
        
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])
        
        dist_start = point.distance(start_pt)
        dist_end = point.distance(end_pt)
        
        if dist_start < threshold:
            nearby.append({
                'route_id': row['ROUTE_ID'],
                'name': row['STREET_ENAME'] or 'N/A',
                'endpoint': 'start',
                'distance': dist_start
            })
        if dist_end < threshold:
            nearby.append({
                'route_id': row['ROUTE_ID'],
                'name': row['STREET_ENAME'] or 'N/A',
                'endpoint': 'end',
                'distance': dist_end
            })
    return sorted(nearby, key=lambda x: x['distance'])

print("\n" + "="*70)
print("Step 1: P1 附近 (50m) 的道路端点")
print("="*70)
p1_nearby = find_nearby_road_endpoints(Point(P1_GDB), roads_in_area)
for r in p1_nearby[:8]:
    print(f"  {r['route_id']}: {r['name']} ({r['endpoint']}, dist={r['distance']:.1f}m)")

print("\n" + "="*70)
print("Step 2: P2 附近 (50m) 的道路端点")
print("="*70)
p2_nearby = find_nearby_road_endpoints(Point(P2_GDB), roads_in_area)
for r in p2_nearby[:8]:
    print(f"  {r['route_id']}: {r['name']} ({r['endpoint']}, dist={r['distance']:.1f}m)")

# 检查是否有道路连接 P1 和 P2 附近的道路
print("\n" + "="*70)
print("Step 3: 检查 P1-P2 之间的主干道走廊")
print("="*70)

# 创建连接线
connection_line = LineString([P1_GDB, P2_GDB])
buffer = connection_line.buffer(100)  # 100m 缓冲

# 找出与连接线缓冲区相交的道路
roads_on_corridor = roads_in_area[roads_in_area.geometry.intersects(buffer)]
print(f"连接走廊 (100m 缓冲) 内的道路: {len(roads_on_corridor)}")

# 按道路名分组
road_names = roads_on_corridor.groupby('STREET_ENAME').size().sort_values(ascending=False)
print("\n走廊内主要道路:")
for name, count in road_names.head(10).items():
    print(f"  {name}: {count} 段")
