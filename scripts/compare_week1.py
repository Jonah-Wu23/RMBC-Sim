import pandas as pd
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime
import argparse

def compare_week1():
    parser = argparse.ArgumentParser(description="Compare real vs sim bus data.")
    parser.add_argument("--real", default="data/processed/link_times.csv", help="Real link times CSV")
    parser.add_argument("--sim", default="sumo/output/stopinfo.xml", help="Simulation stopinfo XML")
    parser.add_argument("--dist", default="data/processed/kmb_route_stop_dist.csv", help="Route distance CSV")
    parser.add_argument("--route", default="68X", help="Route ID to analyze (e.g., 68X, 960)")
    parser.add_argument("--plot", default=None, help="Output plot path (optional)")
    
    args = parser.parse_args()

    # File Paths
    real_link_file = args.real
    sim_stopinfo = args.sim
    route_dist_file = args.dist
    target_route = args.route
    
    if args.plot:
        plot_path = args.plot
    else:
        plot_path = f"sumo/output/l1_trajectory_comparison_{target_route}.png"
    
    if not os.path.exists(sim_stopinfo):
        print(f"Error: {sim_stopinfo} not found. Run simulation first.")
        return

    # 1. Load Real Data
    print(f"Loading real link data from {real_link_file}...")
    try:
        df_real = pd.read_csv(real_link_file)
    except FileNotFoundError:
        print(f"File not found: {real_link_file}")
        return

    # Filter for Target Route Inbound
    print(f"Analyzing Route: {target_route} Inbound")
    # Note: 'bound' is 'inbound'/'outbound'.
    df_real_target = df_real[(df_real['route'] == target_route) & (df_real['bound'] == 'inbound')]
    
    if df_real_target.empty:
        print(f"Warning: No real data for {target_route} inbound.")
    
    # 2. Load Sim Data
    print("Loading simulation data...")
    try:
        tree = ET.parse(sim_stopinfo)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return
    
    sim_records = []
    # group by vehicle to get link times
    veh_stops = {}
    
    count_veh = 0
    for stop in root.findall('stopinfo'):
        veh = stop.get('id')
        # Basic filtering: Check if route ID is part of vehicle ID 
        # (e.g. 'flow_68X_inbound' contains '68X')
        if target_route not in veh:
            continue
            
        if veh not in veh_stops: 
            veh_stops[veh] = []
            count_veh += 1
            
        veh_stops[veh].append({
            'stop_id': stop.get('busStop'),
            'arrival': float(stop.get('started')),
            'departure': float(stop.get('ended'))
        })
        
    print(f"Found {count_veh} vehicles matching '{target_route}' in simulation.")
    
    # Calculate link travel times in sim
    sim_links = []
    for veh, stops in veh_stops.items():
        # Sort by arrival
        stops.sort(key=lambda x: x['arrival'])
        for i in range(len(stops) - 1):
            s1 = stops[i]
            s2 = stops[i+1]
            travel_time = s2['arrival'] - s1['departure']
            sim_links.append({
                'from_stop': s1['stop_id'],
                'to_stop': s2['stop_id'],
                'travel_time_s': travel_time,
                'type': 'sim'
            })
    df_sim = pd.DataFrame(sim_links)
    
    # 3. Join with Distance to get Speed
    print("Loading distances...")
    try:
        df_dist = pd.read_csv(route_dist_file)
    except FileNotFoundError:
        print(f"File not found: {route_dist_file}")
        return

    # Create mapping from (from_stop_id, to_stop_id) to link_dist_m
    dist_map = {}
    last_stop = None
    # Filter distance file for target route
    df_target_dist = df_dist[(df_dist['route'] == target_route) & (df_dist['bound'] == 'inbound')].sort_values('seq')
    
    if df_target_dist.empty:
        print(f"Error: No distance data found for {target_route} inbound.")
        return

    for _, row in df_target_dist.iterrows():
        this_stop = row['stop_id']
        if last_stop:
            dist_map[(last_stop, this_stop)] = row['link_dist_m']
        last_stop = this_stop
        
    if df_sim.empty:
        print("No simulation link data found (parsed 0 links).")
        return

    df_sim['speed_kmh'] = df_sim.apply(lambda r: (dist_map.get((r['from_stop'], r['to_stop']), 0) / 1000.0) / (r['travel_time_s'] / 3600.0) if dist_map.get((r['from_stop'], r['to_stop']), 0) > 0 else None, axis=1)
    df_sim = df_sim.dropna(subset=['speed_kmh'])

    # 4. Compare Distributions (L2)
    print(f"\n--- L2 Link Speed Comparison ({target_route} Inbound) ---")
    
    # Overall distribution
    real_speeds = df_real_target['speed_kmh'].values
    sim_speeds = df_sim['speed_kmh'].values
    
    if len(real_speeds) > 0 and len(sim_speeds) > 0:
        ks_stat, p_val = ks_2samp(real_speeds, sim_speeds)
        
        print(f"Real Mean Speed: {np.mean(real_speeds):.2f} km/h (N={len(real_speeds)})")
        print(f"Sim Mean Speed:  {np.mean(sim_speeds):.2f} km/h (N={len(sim_speeds)})")
        print(f"KS Statistic:    {ks_stat:.4f}")
        print(f"P-Value:        {p_val:.4e}")
        
        # Percentiles
        for p in [10, 50, 90]:
            r_p = np.percentile(real_speeds, p)
            s_p = np.percentile(sim_speeds, p)
            print(f"P{p}: Real={r_p:.1f}, Sim={s_p:.1f} (Diff={s_p-r_p:.1f})")
    else:
        print("Insufficient data for KS test.")

    # 5. L1 Comparison (Station-to-Station & Cumulative)
    print("\n--- L1 Comparison: Link Travel Time & Cumulative Arrival ---")
    
    # 5.1 Real Link Stats Aggregation
    real_link_agg = df_real_target.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().reset_index()
    real_link_agg.rename(columns={'travel_time_s': 'real_avg_time_s'}, inplace=True)
    
    # 5.2 Sim Link Stats Aggregation
    sim_link_agg = df_sim.groupby(['from_stop', 'to_stop'])['travel_time_s'].mean().reset_index()
    sim_link_agg.rename(columns={'travel_time_s': 'sim_avg_time_s'}, inplace=True)
    
    # Merge for Comparison Table
    seq_map = df_target_dist.set_index('stop_id')['seq'].to_dict()
    
    # Enrich sim_link_agg with sequences
    sim_link_agg['from_seq'] = sim_link_agg['from_stop'].map(seq_map)
    sim_link_agg['to_seq'] = sim_link_agg['to_stop'].map(seq_map)
    sim_link_agg = sim_link_agg.dropna(subset=['from_seq', 'to_seq'])
    
    # Final L1 Table
    l1_table = pd.merge(real_link_agg, sim_link_agg, on=['from_seq', 'to_seq'], suffixes=('', '_sim'))
    l1_table['diff_s'] = l1_table['sim_avg_time_s'] - l1_table['real_avg_time_s']
    l1_table['diff_percent'] = (l1_table['diff_s'] / l1_table['real_avg_time_s']) * 100
    
    print("Top 10 Links with largest time difference:")
    print(l1_table[['from_seq', 'to_seq', 'real_avg_time_s', 'sim_avg_time_s', 'diff_percent']].sort_values('diff_percent', ascending=False).head(10))

    # 5.3 Cumulative Trajectory Calculation
    
    all_seqs = sorted(df_target_dist['seq'].tolist())
    
    # Re-calculate to ensure alignment
    real_times_map = {row['to_seq']: row['real_avg_time_s'] for _, row in real_link_agg.iterrows()}
    sim_times_map = {row['to_seq']: row['sim_avg_time_s'] for _, row in sim_link_agg.iterrows()}
    
    real_cum = [0.0]
    real_dists = [0.0]
    
    sim_cum = [0.0]
    sim_dists = [0.0]
    
    current_dist = 0
    
    # Truncate Real curve to where Sim curve ends logic
    max_sim_seq = max(sim_times_map.keys()) if sim_times_map else 0
    
    for i in range(1, len(all_seqs)):
        seq = all_seqs[i]
        d = df_target_dist[df_target_dist['seq'] == seq]['link_dist_m'].values[0]
        current_dist += d
        
        # Only append Real if within Sim range
        if seq <= max_sim_seq:
            if seq in real_times_map: 
                 real_dists.append(current_dist)
                 real_cum.append(real_cum[-1] + real_times_map[seq])
            else:
                 real_dists.append(current_dist)
                 # Interpolate or fill missing real data 
                 last_t = real_cum[-1] if real_cum else 0
                 real_cum.append(last_t + real_times_map.get(seq, 0))

        # SIM: Only append if we actually have data
        if seq in sim_times_map:
            sim_dists.append(current_dist)
            sim_cum.append(sim_cum[-1] + sim_times_map[seq])

    # 6. Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(real_dists, real_cum, 'b-o', label='Real (Average)')
    plt.plot(sim_dists, sim_cum, 'r-s', label='Sim (Baseline)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Cumulative Time (s)')
    plt.title(f'Bus Route Trajectory (Distance-Time) - {target_route} Inbound')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(plot_path)
    print(f"\nTrajectory plot saved to: {plot_path}")
    
    # 7. Summary Report
    if real_cum and sim_cum:
        total_real = real_cum[-1]
        total_sim = sim_cum[-1]
        print(f"\nFinal Arrival Window (End-to-End, Truncated to Sim):")
        print(f"Real: {total_real:.1f}s")
        print(f"Sim:  {total_sim:.1f}s")
        difference = total_sim - total_real
        error_pct = (difference / total_real * 100) if total_real > 0 else 0
        print(f"Error: {difference:.1f}s ({error_pct:.1f}%)")

if __name__ == "__main__":
    compare_week1()
