#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断站点边映射问题"""

import pandas as pd
import xml.etree.ElementTree as ET

# 加载站点边映射
tree = ET.parse('sumo/additional/bus_stops.add.xml')
root = tree.getroot()
stop_to_edge = {}
for stop in root.findall('.//busStop'):
    stop_id = stop.get('id')
    lane = stop.get('lane', '')
    if lane.startswith(':'):
        edge = lane.rsplit('_', 1)[0]
    else:
        edge = lane.rsplit('_', 1)[0]
    stop_to_edge[stop_id] = edge

# 加载站点序列
df = pd.read_csv('data/processed/kmb_route_stop_dist.csv')

for route in ['68X', '960']:
    for bound in ['inbound', 'outbound']:
        subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
        
        print(f"\n{route} {bound} 站点序列:")
        prev_edge = None
        for _, row in subset.iterrows():
            stop_id = row['stop_id']
            edge = stop_to_edge.get(stop_id, 'N/A')
            name = str(row['stop_name_en'])[:35]
            
            # 检查是否是折返
            flag = ""
            if prev_edge:
                if prev_edge.endswith('_rev') and prev_edge[:-4] == edge:
                    flag = " <== REVERSAL"
                elif edge.endswith('_rev') and edge[:-4] == prev_edge:
                    flag = " <== REVERSAL"
            
            print(f"  seq={int(row['seq']):2d}: {edge:20s} | {name}{flag}")
            prev_edge = edge
