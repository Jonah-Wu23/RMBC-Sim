#!/usr/bin/env python
"""在 GDB 中查找应连接节点 8936 和 1970 的道路"""
import geopandas as gpd
import sumolib
from shapely.geometry import Point, LineString

# 加载 SUMO 网络
net = sumolib.net.readNet('sumo/net/hk_irn_v3_patched_v1.net.xml')

# SUMO 偏移量
offset_x, offset_y = -812340.81, -812121.37

# 获取两个节点的坐标并转换
n8936 = net.getNode('8936').getCoord()
n1970 = net.getNode('1970').getCoord()

# 转换到 GDB 坐标
gdb_8936 = (n8936[0] - offset_x, n8936[1] - offset_y)
gdb_1970 = (n1970[0] - offset_x, n1970[1] - offset_y)

print(f'节点 8936 GDB坐标: ({gdb_8936[0]:.0f}, {gdb_8936[1]:.0f})')
print(f'节点 1970 GDB坐标: ({gdb_1970[0]:.0f}, {gdb_1970[1]:.0f})')

# 创建连接线
connection_line = LineString([gdb_8936, gdb_1970])
print(f'两点距离: {connection_line.length:.0f}m')

# 加载 GDB
gdb_path = 'data/RdNet_IRNP.gdb'
print('\n读取 CENTERLINE 图层...')
gdf = gpd.read_file(gdb_path, layer='CENTERLINE')

# 创建缓冲区搜索（在连接线周围 100m 范围内）
buffer = connection_line.buffer(100)

# 查找在缓冲区内的道路
roads_near = gdf[gdf.geometry.intersects(buffer)]
print(f'\n在 8936→1970 连接线 100m 范围内的道路: {len(roads_near)} 条')

if len(roads_near) > 0:
    print('\n道路列表 (按 ROUTE_ID 排序):')
    for idx, row in roads_near.sort_values('ROUTE_ID').iterrows():
        ename = row['STREET_ENAME'] or 'N/A'
        cname = row['STREET_CNAME'] or 'N/A'
        length = row['SHAPE_Length']
        route_id = row['ROUTE_ID']
        # 检查该 ROUTE_ID 是否在 SUMO 网络中存在
        exists = net.hasEdge(str(int(route_id))) if route_id else False
        status = '✅ SUMO存在' if exists else '❌ SUMO缺失'
        print(f'  {route_id}: {ename} ({length:.1f}m) - {status}')
