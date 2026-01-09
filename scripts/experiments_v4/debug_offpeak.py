#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.metrics_v4 import compute_metrics_v4, load_real_link_stats, AuditConfig

# 测试 off_peak
real_df = load_real_link_stats('data2/processed/link_stats_offpeak.csv')
print(f'Real data loaded: {len(real_df)} rows')
print(f'Columns: {real_df.columns.tolist()}')

stopinfo = 'data/experiments_v4/scale_sweep/off_peak/1h/scale0.00/A3/seed0/stopinfo.xml'
print(f'\nTesting: {stopinfo}')
print(f'File exists: {Path(stopinfo).exists()}')
print(f'File size: {Path(stopinfo).stat().st_size} bytes')

result = compute_metrics_v4(
    real_data=real_df,
    sim_data=stopinfo,
    dist_file='data2/processed/kmb_route_stop_dist.csv',
    audit_config=AuditConfig.from_protocol(),
    scenario='off_peak',
    route='68X'
)

if result:
    print(f'\n✓ Result obtained')
    print(f'n_sim: {result.n_sim}')
    print(f'n_clean: {result.audit_stats.n_clean}')
    print(f'n_events: {result.audit_stats.n_raw}')
    print(f'ks_speed: {result.ks_speed_clean.ks_stat}')
    print(f'ks_tt: {result.ks_tt_clean.ks_stat}')
else:
    print('\n✗ Result is None')
