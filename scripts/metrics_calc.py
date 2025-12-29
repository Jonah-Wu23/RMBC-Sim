
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import sys
import os
import argparse

# Ensure we can import common_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds, get_dist_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_links', required=True, help="Path to link_speeds.csv")
    parser.add_argument('--real_dist', required=True, help="Path to kmb_route_stop_dist.csv")
    parser.add_argument('--sim', required=True, help="Path to stopinfo.xml")
    parser.add_argument('--out', required=True, help="Output CSV path")
    parser.add_argument('--route', default='68X', help="Route to evaluate")
    parser.add_argument('--bound', default='I', help="Bound (I or O)")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading Sim: {args.sim}")
    sim_raw = load_sim_data(args.sim)
    
    print(f"Loading Real Dist: {args.real_dist}")
    dist_df = load_route_stop_dist(args.real_dist)
    dist_df = dist_df[(dist_df['route'] == args.route) & (dist_df['bound'] == 'inbound')] # CSV uses 'inbound'/'outbound' usually?
    if dist_df.empty:
         # Try 'I'/'O' if inbound failed
         dist_df = load_route_stop_dist(args.real_dist)
         dist_df = dist_df[(dist_df['route'] == args.route) & (dist_df['bound'] == args.bound)]

    print(f"Loading Real Links: {args.real_links}")
    real_links = load_real_link_speeds(args.real_links)
    # Filter Real Links
    # Note: link_speeds.csv bound might be 'inbound' or 'I'. Check previously cat output: 'inbound'.
    bound_map = {'I': 'inbound', 'O': 'outbound'}
    target_bound = bound_map.get(args.bound, args.bound)
    real_links = real_links[(real_links['route'] == args.route) & (real_links['bound'] == target_bound)]

    # 2. Process Sim Data
    # For baseline comparison, we compare Travel Time from Stop 1 to Stop X
    sim_traj = build_sim_trajectory(sim_raw, dist_df)
    if sim_traj.empty:
        print("ERROR: No matching stops found in Sim data!")
        return

    # 3. Calculate Metrics
    
    # --- Metric 1: Travel Time RMSE & MAPE ---
    # Real: Average travel time from Seq 1 to Seq X
    # Real Links contains travel time for each segment (from_seq -> to_seq).
    # We construct "Cumulative Travel Time" from Seq 1.
    
    # Filter real links within 17:00-18:00 (approximate) - Optional, user requires strict window but CSV might cover more.
    # Assuming CSV is already relevant or we filter by hour.
    # real_links['hour'] = real_links['departure_ts'].dt.hour
    # real_links = real_links[real_links['hour'] == 17]
    
    # Group by link (from -> to) to get avg travel time
    real_link_stats = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().reset_index()
    
    # Reconstruct Cumulative Time for Real
    # Sort by sequence
    real_link_stats = real_link_stats.sort_values('from_seq')
    real_cum_time = {1: 0}
    current_time = 0
    # Assuming continuous chain 1->2, 2->3...
    for _, row in real_link_stats.iterrows():
        f = row['from_seq']
        t = row['to_seq']
        if f in real_cum_time:
            real_cum_time[t] = real_cum_time[f] + row['travel_time_s']
            
    real_time_df = pd.DataFrame(list(real_cum_time.items()), columns=['seq', 'real_time_s'])

    # Sim: Cumulative Time
    # Group by seq, mean arrival time relative to Seq 1 arrival time for each TRIP
    sim_trips = []
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        if group.empty: continue
        
        # Check if trip starts at Seq 1 (or close)
        # We find the min seq in this trip
        min_seq = group['seq'].min()
        start_time = group.loc[group['seq'] == min_seq, 'arrival_time'].values[0]
        
        group['rel_time_s'] = group['arrival_time'] - start_time
        # Re-index rel_time to represent travel time FROM min_seq
        # ideally min_seq should be 1. If sim starts at 5, we can only compare from 5.
        
        sim_trips.append(group[['seq', 'rel_time_s']])
    
    if not sim_trips:
        print("No valid sim trips found.")
        return

    sim_all = pd.concat(sim_trips)
    sim_stats = sim_all.groupby('seq')['rel_time_s'].mean().reset_index().rename(columns={'rel_time_s': 'sim_time_s'})
    
    # Merge
    comparison = pd.merge(real_time_df, sim_stats, on='seq', how='inner')
    
    # RMSE & MAPE
    comparison['error'] = comparison['sim_time_s'] - comparison['real_time_s']
    comparison['pct_error'] = (comparison['sim_time_s'] - comparison['real_time_s']) / comparison['real_time_s']
    
    # Filter out seq 1 (error is 0 by definition)
    valid_comp = comparison[comparison['seq'] > 1]
    
    rmse = np.sqrt((valid_comp['error'] ** 2).mean())
    mape = np.abs(valid_comp['pct_error']).mean() * 100
    
    # --- Metric 2: Wasserstein on Link Speeds ---
    # Real speeds
    real_speeds = real_links['speed_kmh'].dropna()
    
    # Sim speeds
    # Recompute sim link speeds: Dist / (Arr_next - Dep_curr)
    sim_speed_list = []
    # 使用 get_dist_map 处理 NaN
    dist_map = get_dist_map(dist_df, 'seq')
    
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        # Calculate link speed between stops
        # Shift to get next
        group['next_seq'] = group['seq'].shift(-1)
        group['next_arr'] = group['arrival_time'].shift(-1)
        group['curr_dep'] = group['departure_time']
        
        valid = group.dropna(subset=['next_seq'])
        for _, row in valid.iterrows():
            s1 = row['seq']
            s2 = int(row['next_seq'])
            if s1 in dist_map and s2 in dist_map:
                dist = dist_map[s2] - dist_map[s1]
                t = row['next_arr'] - row['curr_dep']
                if t > 0 and dist > 0:
                    speed = (dist / t) * 3.6 # m/s to km/h
                    sim_speed_list.append(speed)
                    
    sim_speeds = pd.Series(sim_speed_list)
    
    w_dist = wasserstein_distance(real_speeds, sim_speeds) if not sim_speeds.empty else 0
    
    # Save
    metrics = pd.DataFrame({
        'Metric': ['RMSE (Travel Time)', 'MAPE (Travel Time)', 'Wasserstein (Speed)'],
        'Value': [rmse, mape, w_dist],
        'Unit': ['s', '%', 'Distance']
    })
    
    metrics.to_csv(args.out, index=False)
    print("Metrics Calculated:")
    print(metrics)

if __name__ == "__main__":
    main()
