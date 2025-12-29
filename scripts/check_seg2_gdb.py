#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查询 GDB 验证 106838 和 106831 的方向信息"""

import geopandas as gpd
from pathlib import Path

gdb_path = Path('data/RdNet_IRNP.gdb')
print(f'GDB path: {gdb_path}')
print(f'Exists: {gdb_path.exists()}')

if gdb_path.exists():
    import fiona
    layers = fiona.listlayers(str(gdb_path))
    print(f'Layers: {layers}')
    
    # 读取主要图层
    for layer in layers[:3]:
        try:
            gdf = gpd.read_file(str(gdb_path), layer=layer)
            print(f'\nLayer {layer}: {len(gdf)} records')
            print(f'Columns: {list(gdf.columns)[:10]}')
            
            # 查找关键边
            if 'OBJECTID' in gdf.columns:
                for oid in [106838, 106831, 107154, 105653]:
                    matches = gdf[gdf['OBJECTID'] == oid]
                    if len(matches) > 0:
                        r = matches.iloc[0]
                        dir_val = r.get('TRAVEL_DIRECTION', '?')
                        name = str(r.get('ROAD_NAME_EN', '?'))[:40]
                        print(f'  {oid}: DIR={dir_val}, NAME={name}')
        except Exception as e:
            print(f'Layer {layer}: Error - {e}')
