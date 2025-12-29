#!/usr/bin/env python
"""检查 GDB 中 105656/105732 的双向性"""
import geopandas as gpd

gdf = gpd.read_file('data/RdNet_IRNP.gdb', layer='CENTERLINE')

# 查找这两条边
route_ids = [105656, 105732]

print('GDB 双向性检查: 105656, 105732')
print('='*70)

# 先看一下所有字段名
print('CENTERLINE 图层列名:')
print(gdf.columns.tolist())
print()

for route_id in route_ids:
    road = gdf[gdf['ROUTE_ID'] == route_id]
    if len(road) > 0:
        row = road.iloc[0]
        print(f'ROUTE_ID: {route_id}')
        print(f'  STREET_ENAME: {row["STREET_ENAME"]}')
        print(f'  STREET_CNAME: {row["STREET_CNAME"]}')
        print(f'  SHAPE_Length: {row["SHAPE_Length"]:.1f}m')
        
        # 检查方向相关字段
        direction_fields = ['TRAVEL_DIRECTION', 'ONEWAY', 'DIR', 'DIRECTION', 'ONE_WAY']
        for field in direction_fields:
            if field in gdf.columns:
                print(f'  {field}: {row[field]}')
        
        # 其他可能有用的字段
        other_fields = ['ELEVATION', 'ST_CODE', 'ROUTE_NUM', 'REMARKS']
        for field in other_fields:
            if field in gdf.columns and row[field] is not None:
                val = row[field]
                if val != '' and not (isinstance(val, float) and val != val):  # 排除 NaN
                    print(f'  {field}: {val}')
        print()
    else:
        print(f'ROUTE_ID {route_id}: 未找到')
        print()

# 检查 TRAVEL_DIRECTION 字段的可能值
print('='*70)
print('TRAVEL_DIRECTION 字段值分布:')
if 'TRAVEL_DIRECTION' in gdf.columns:
    print(gdf['TRAVEL_DIRECTION'].value_counts())
