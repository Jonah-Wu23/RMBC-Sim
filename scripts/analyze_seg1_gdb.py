#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""seg1 GDB 走廊分析 - 检查是否有平行道路"""

import geopandas as gpd
import sumolib

net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')
gdf = gpd.read_file('data/RdNet_IRNP.gdb', layer='CENTERLINE')

print('=== seg1 路径边的 GDB 信息 ===')
print()

# seg1 前 10 条边
seg1_edges = ['105735_rev', '273264_rev', '105528', '105501', '105502', 
              '106883', '106884_rev', '106894_rev', '105656', '105732_rev']

for eid in seg1_edges:
    base_id = int(eid.replace('_rev', ''))
    matches = gdf[gdf['ROUTE_ID'] == base_id]
    if len(matches) > 0:
        r = matches.iloc[0]
        name = str(r['STREET_ENAME'])[:25]
        dir_val = r['TRAVEL_DIRECTION']
        print(f'{eid}: DIR={dir_val}, {name}')
    else:
        print(f'{eid}: Not in GDB')

print()
print('=== 关键问题：seg1 是否经过单向道路？ ===')
print('TRAVEL_DIRECTION: 1=正向, 2=反向, 3=双向, 4=禁行')
print()

# 检查 105735/105653 所在道路
for rid in [105735, 105653, 273264]:
    matches = gdf[gdf['ROUTE_ID'] == rid]
    if len(matches) > 0:
        r = matches.iloc[0]
        print(f'{rid}: DIR={r["TRAVEL_DIRECTION"]}, {r["STREET_ENAME"]}')
