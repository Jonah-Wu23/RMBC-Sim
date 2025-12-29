#!/usr/bin/env python
"""检查 seg3 关键边在 GDB 中的双向性"""
import geopandas as gpd

gdf = gpd.read_file('data/RdNet_IRNP.gdb', layer='CENTERLINE')

# 查找关键边的双向性
route_ids = [285166, 105817, 106838, 107154, 106831]

print('GDB 双向性检查: seg3 关键边')
print('='*70)

for route_id in route_ids:
    road = gdf[gdf['ROUTE_ID'] == route_id]
    if len(road) > 0:
        row = road.iloc[0]
        name = row['STREET_ENAME'] or 'N/A'
        length = row['SHAPE_Length']
        direction = row['TRAVEL_DIRECTION']
        dir_str = '单向' if direction == 1 else '双向' if direction == 3 else f'未知({direction})'
        print(f'{route_id}: {name}, len={length:.1f}m, TRAVEL_DIRECTION={direction} ({dir_str})')
    else:
        print(f'{route_id}: 未找到')
