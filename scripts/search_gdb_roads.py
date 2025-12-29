#!/usr/bin/env python
"""在 GDB 中搜索洪水桥区域的道路数据"""
import geopandas as gpd
from shapely.geometry import box

gdb_path = 'data/RdNet_IRNP.gdb'
print('读取 CENTERLINE 图层...')
gdf = gpd.read_file(gdb_path, layer='CENTERLINE')

# SUMO 偏移量
offset_x, offset_y = -812340.81, -812121.37

# SUMO 中的区域坐标转换到原始坐标
x_min = 4900 - offset_x
x_max = 5900 - offset_x
y_min = 20200 - offset_y
y_max = 20900 - offset_y

print(f'搜索区域 (原始坐标):')
print(f'  X: {x_min:.0f} - {x_max:.0f}')
print(f'  Y: {y_min:.0f} - {y_max:.0f}')

# 创建搜索区域
search_box = box(x_min, y_min, x_max, y_max)

# 过滤该区域的道路
roads_in_area = gdf[gdf.geometry.intersects(search_box)]
print(f'\n该区域道路数: {len(roads_in_area)}')

if len(roads_in_area) > 0:
    print('\n区域内所有道路:')
    for idx, row in roads_in_area.iterrows():
        ename = row['STREET_ENAME'] or 'N/A'
        cname = row['STREET_CNAME'] or 'N/A'
        length = row['SHAPE_Length']
        route_id = row['ROUTE_ID'] if 'ROUTE_ID' in row else 'N/A'
        print(f'  - {ename} / {cname} (len={length:.1f}m, route={route_id})')
else:
    print('\n该区域无道路数据！这确认了原始数据缺口。')
