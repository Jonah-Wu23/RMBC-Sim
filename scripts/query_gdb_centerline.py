#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查询 GDB CENTERLINE 验证 seg2 边的方向"""

import geopandas as gpd

gdf = gpd.read_file('data/RdNet_IRNP.gdb', layer='CENTERLINE')
print(f'CENTERLINE: {len(gdf)} records')
print(f'Columns: {list(gdf.columns)[:10]}')
print()

# 查找 ROUTE_ID 匹配
targets = [106838, 106831, 107154, 105653]
for oid in targets:
    matches = gdf[gdf['ROUTE_ID'] == oid]
    if len(matches) > 0:
        r = matches.iloc[0]
        dir_val = r['TRAVEL_DIRECTION']
        name = r['STREET_ENAME']
        print(f'ROUTE_ID {oid}: DIR={dir_val}, NAME={name}')
    else:
        print(f'ROUTE_ID {oid}: Not found')

print()
print('=== 分析 ===')
print('TRAVEL_DIRECTION 值含义:')
print('  1 = 只能沿几何方向行驶')
print('  2 = 只能逆几何方向行驶')
print('  3 = 双向')
print('  4 = 禁止通行')
