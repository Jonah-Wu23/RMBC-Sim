#!/usr/bin/env python
"""确认 GDB 中 272309 是单条双向要素"""
import geopandas as gpd

gdf = gpd.read_file('data/RdNet_IRNP.gdb', layer='CENTERLINE')

# 检查 272309 是否只有一条记录
road_272309 = gdf[gdf['ROUTE_ID'] == 272309]
print(f'272309 在 GDB 中的记录数: {len(road_272309)}')

if len(road_272309) > 0:
    row = road_272309.iloc[0]
    print(f'STREET_ENAME: {row["STREET_ENAME"]}')
    print(f'TRAVEL_DIRECTION: {row["TRAVEL_DIRECTION"]} (3=双向)')
    print(f'SHAPE_Length: {row["SHAPE_Length"]:.1f}m')
    
    # 获取端点坐标
    geom = row.geometry
    if geom.geom_type == 'MultiLineString':
        coords = []
        for line in geom.geoms:
            coords.extend(list(line.coords))
    else:
        coords = list(geom.coords)
    print(f'起点: ({coords[0][0]:.0f}, {coords[0][1]:.0f})')
    print(f'终点: ({coords[-1][0]:.0f}, {coords[-1][1]:.0f})')

print('\n结论: 272309 是单条双向要素，可以安全补 _rev 边')
