#!/usr/bin/env python
"""检查缺失边 106787/106788 的详情"""
import geopandas as gpd
from shapely.geometry import Point

gdf = gpd.read_file('data/RdNet_IRNP.gdb', layer='CENTERLINE')
missing = ['106787', '106788']

print('缺失道路详情:')
print('='*70)

for route_id in missing:
    road = gdf[gdf['ROUTE_ID'] == int(route_id)]
    if len(road) > 0:
        row = road.iloc[0]
        geom = row.geometry
        if geom.geom_type == 'MultiLineString':
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            coords = list(geom.coords)
        
        print(f'{route_id}: {row["STREET_ENAME"]}')
        print(f'  长度: {row["SHAPE_Length"]:.1f}m')
        print(f'  起点: ({coords[0][0]:.0f}, {coords[0][1]:.0f})')
        print(f'  终点: ({coords[-1][0]:.0f}, {coords[-1][1]:.0f})')
        print()

# P1 和 P2 的 GDB 坐标
P1_GDB = (834942, 819587)
P2_GDB = (834626, 819873)

print('到 P1/P2 的距离:')
print('='*70)

for route_id in missing:
    road = gdf[gdf['ROUTE_ID'] == int(route_id)]
    if len(road) > 0:
        geom = road.iloc[0].geometry
        dist_p1 = Point(P1_GDB).distance(geom)
        dist_p2 = Point(P2_GDB).distance(geom)
        print(f'{route_id}: 到P1={dist_p1:.0f}m, 到P2={dist_p2:.0f}m')
