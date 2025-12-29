#!/usr/bin/env python
"""验证 68X inbound scale"""
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# 加载边长度
net_file = Path('sumo/net/hk_irn_v3_patched_v1.net.xml')
tree = ET.parse(str(net_file))
lengths = {}
for edge in tree.getroot().findall('.//edge'):
    eid = edge.get('id')
    if eid:
        lane = edge.find('lane')
        if lane is not None:
            lengths[eid] = float(lane.get('length', 0))

# 加载 KMB 数据
df = pd.read_csv('data/processed/kmb_route_stop_dist.csv')

for route in ['68X', '960']:
    for bound in ['inbound', 'outbound']:
        route_file = Path(f'build/{route}_{bound}_via.rou.xml')
        if not route_file.exists():
            continue
            
        kmb_len = df[(df['route']==route) & (df['bound']==bound)]['cum_dist_m'].max()
        
        tree = ET.parse(str(route_file))
        for vehicle in tree.getroot().findall('.//vehicle'):
            route_elem = vehicle.find('route')
            if route_elem is not None:
                edges = route_elem.get('edges', '').split()
                
                sumo_len = 0
                for e in edges:
                    if e in lengths:
                        sumo_len += lengths[e]
                    elif e.endswith('_rev') and e[:-4] in lengths:
                        sumo_len += lengths[e[:-4]]
                
                scale = sumo_len / kmb_len if kmb_len > 0 else 0
                bridges = [e for e in edges if 'bridge' in e.lower()]
                
                status = "✅" if scale < 1.7 else "⚠️"
                print(f'{status} {route} {bound}: scale={scale:.3f}, edges={len(edges)}, bridges={len(bridges)}')
                if bridges:
                    print(f'   Bridges: {bridges}')
