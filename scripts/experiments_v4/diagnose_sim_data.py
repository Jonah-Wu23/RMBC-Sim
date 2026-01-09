#!/usr/bin/env python3
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.metrics_v4 import load_sim_stopinfo, load_dist_mapping, compute_sim_link_data

# 测试文件
stopinfo_pm = 'data/experiments_v4/scale_sweep/pm_peak/1h/scale0.00/A3/seed0/stopinfo.xml'
stopinfo_off = 'data/experiments_v4/scale_sweep/off_peak/1h/scale0.00/A3/seed0/stopinfo.xml'
dist_csv = 'data2/processed/kmb_route_stop_dist.csv'

print("="*70)
print("诊断 PM Peak")
print("="*70)

# 1. 检查 stopinfo
df_stops = load_sim_stopinfo(stopinfo_pm)
print(f"\n1. Stopinfo 记录数: {len(df_stops)}")
if len(df_stops) > 0:
    print(f"   Vehicle IDs sample: {df_stops['vehicle_id'].unique()[:5].tolist()}")
    print(f"   Stop IDs sample: {df_stops['stop_id'].unique()[:5].tolist()}")

# 2. 检查 dist_map
dist_map = load_dist_mapping(dist_csv)
print(f"\n2. Dist mapping 键数: {len(dist_map)}")
if len(dist_map) > 0:
    sample_keys = list(dist_map.keys())[:5]
    print(f"   Sample keys: {sample_keys}")

# 3. 检查匹配
if len(df_stops) > 0 and len(dist_map) > 0:
    stop_pairs = []
    for _, veh_data in df_stops.groupby("vehicle_id"):
        veh_data = veh_data.reset_index(drop=True)
        for i in range(len(veh_data) - 1):
            from_stop = str(veh_data.loc[i, "stop_id"])
            to_stop = str(veh_data.loc[i + 1, "stop_id"])
            stop_pairs.append((from_stop, to_stop))
    
    print(f"\n3. Stop pairs 数量: {len(stop_pairs)}")
    if stop_pairs:
        print(f"   Sample pairs: {stop_pairs[:5]}")
        matched = sum(1 for pair in stop_pairs if pair in dist_map)
        print(f"   匹配到 dist_map: {matched}/{len(stop_pairs)}")

# 4. 完整计算
speeds, tt, timestamps = compute_sim_link_data(stopinfo_pm, dist_csv)
print(f"\n4. 最终结果:")
print(f"   Speeds: {len(speeds)}")
print(f"   TT: {len(tt)}")
print(f"   Timestamps: {len(timestamps)}")

print("\n" + "="*70)
print("诊断 Off Peak")
print("="*70)

df_stops_off = load_sim_stopinfo(stopinfo_off)
print(f"\n1. Stopinfo 记录数: {len(df_stops_off)}")
if len(df_stops_off) > 0:
    print(f"   Vehicle IDs sample: {df_stops_off['vehicle_id'].unique()[:5].tolist()}")
    print(f"   Stop IDs sample: {df_stops_off['stop_id'].unique()[:5].tolist()}")

speeds_off, tt_off, ts_off = compute_sim_link_data(stopinfo_off, dist_csv)
print(f"\n2. 最终结果:")
print(f"   Speeds: {len(speeds_off)}")
print(f"   TT: {len(tt_off)}")
