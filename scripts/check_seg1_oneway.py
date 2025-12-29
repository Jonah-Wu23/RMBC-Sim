#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 seg1 单向边一致性"""

import geopandas as gpd
import sumolib

net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')
gdf = gpd.read_file('data/RdNet_IRNP.gdb', layer='CENTERLINE')

print('=== 单向边一致性检查 ===')
print('GDB DIR=1 表示只能沿几何方向行驶')
print()

for rid in [105656, 105732, 105528]:
    matches = gdf[gdf['ROUTE_ID'] == rid]
    if len(matches) > 0:
        r = matches.iloc[0]
        dir_val = r['TRAVEL_DIRECTION']
        name = r['STREET_ENAME']
        print(f'GDB {rid}: DIR={dir_val}, {name}')
    
    # SUMO 边检查
    has_normal = net.hasEdge(str(rid))
    has_rev = net.hasEdge(str(rid) + '_rev')
    print(f'  SUMO: {rid}存在={has_normal}, {rid}_rev存在={has_rev}')
    
    if dir_val == 1 and has_rev:
        print(f'  ⚠️ 问题: GDB单向但SUMO有反向边!')
    print()

# 完整 seg1 路径检查
print('=== seg1 完整路径单向性 ===')
from_e = net.getEdge('105735_rev')
to_e = net.getEdge('105653')
route, _ = net.getShortestPath(from_e, to_e, vClass='bus')

if route:
    oneway_violations = []
    for e in route:
        eid = e.getID()
        base_id = int(eid.replace('_rev', ''))
        is_rev = '_rev' in eid
        
        matches = gdf[gdf['ROUTE_ID'] == base_id]
        if len(matches) > 0:
            dir_val = matches.iloc[0]['TRAVEL_DIRECTION']
            # DIR=1 只能正向, DIR=2 只能反向
            if dir_val == 1 and is_rev:
                oneway_violations.append(f'{eid} (用_rev但GDB是DIR=1正向单行)')
            elif dir_val == 2 and not is_rev:
                oneway_violations.append(f'{eid} (用正向但GDB是DIR=2反向单行)')
    
    if oneway_violations:
        print(f'发现 {len(oneway_violations)} 个潜在逆行:')
        for v in oneway_violations[:5]:
            print(f'  {v}')
    else:
        print('无逆行问题')
