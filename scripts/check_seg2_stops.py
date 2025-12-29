#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 seg2 stop 映射"""

import pandas as pd

corr = pd.read_csv('config/calibration/stop_edge_corrections.csv')

# 查找 68X inbound 的 stop 对应的边
route_stops = corr[(corr['route']=='68X') & (corr['bound']=='inbound')]
print('68X inbound stop corrections:')
for _, r in route_stops.iterrows():
    sid = str(r['stop_id'])[:10]
    orig = r['orig_edge']
    fixed = r['fixed_edge']
    print(f'  {sid}: orig={orig}, fixed={fixed}')

# 特别关注 106831 相关
print()
print('涉及 106831 的映射:')
matches = route_stops[route_stops['fixed_edge'].astype(str).str.contains('106831')]
for _, r in matches.iterrows():
    print(f'  stop_id={r["stop_id"]}')
    print(f'  orig_edge={r["orig_edge"]}')
    print(f'  fixed_edge={r["fixed_edge"]}')
    print()

# 也检查 105653 (seg2 起点)
print('涉及 105653 的映射:')
matches = route_stops[route_stops['fixed_edge'].astype(str).str.contains('105653')]
for _, r in matches.iterrows():
    print(f'  stop_id={r["stop_id"]}')
    print(f'  orig_edge={r["orig_edge"]}')
    print(f'  fixed_edge={r["fixed_edge"]}')
