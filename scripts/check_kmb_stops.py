#!/usr/bin/env python
"""检查 68X inbound 前几站的 KMB 数据和 edge 映射"""
import pandas as pd

# 加载 KMB 数据
df = pd.read_csv('data/processed/kmb_route_stop_dist.csv')
subset = df[(df['route']=='68X') & (df['bound']=='inbound')].sort_values('seq')

# 加载 corrections
corr = pd.read_csv('config/calibration/stop_edge_corrections.csv')
corr_68x = corr[(corr['route']=='68X') & (corr['bound']=='inbound')]

print('68X inbound 前 5 站 KMB 数据:')
print('='*70)

for _, row in subset.head(5).iterrows():
    # 获取映射的 edge
    edge_row = corr_68x[corr_68x['stop_id'] == row['stop_id']]
    if len(edge_row) > 0:
        edge = edge_row.iloc[0]['fixed_edge'] if pd.notna(edge_row.iloc[0]['fixed_edge']) else edge_row.iloc[0]['orig_edge']
    else:
        edge = 'N/A'
    
    name = row.get('name_en', 'N/A') if 'name_en' in row else 'N/A'
    print(f"seq={row['seq']:2d}: {row['stop_id'][:16]}... cum_dist={row['cum_dist_m']:7.0f}m -> {edge}")

print()
print('段长度计算 (KMB cum_dist 差值):')
stops = list(subset.head(5).itertuples())
for i in range(len(stops)-1):
    seg_len = stops[i+1].cum_dist_m - stops[i].cum_dist_m
    print(f'  seg{i+1}: {seg_len:.0f}m (stop{i+1} → stop{i+2})')
