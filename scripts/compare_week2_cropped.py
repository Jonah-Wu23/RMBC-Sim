import pandas as pd
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def compare_week2_cropped():
    # File Paths
    real_link_file = r"data/processed/enriched_link_stats.csv"
    sim_stopinfo = r"sumo/output/stopinfo_exp2_cropped.xml"
    route_dist_file = r"data/processed/kmb_route_stop_dist.csv"
    output_plot = r"sumo/output/week2_cropped_trajectory.png"
    
    if not os.path.exists(sim_stopinfo):
        print(f"Error: {sim_stopinfo} not found. Run simulation first.")
        return

    # 1. Load Real Data
    print("Loading real link data...")
    df_real = pd.read_csv(real_link_file)
    df_real_68x = df_real[(df_real['route'] == '68X') & (df_real['bound'] == 'inbound')]
    
    if df_real_68x.empty:
        print("Warning: No real data for 68X inbound.")
        return

    # 2. Load Sim Data
    print(f"Loading simulation data from {sim_stopinfo}...")
    try:
        tree = ET.parse(sim_stopinfo)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing xml: {e}")
        return
    
    veh_stops = {}
    for stop in root.findall('stopinfo'):
        # Filter for 68X vehicles if necessary, or assume fixed_routes contain 68X
        # Our fixed_routes.rou.xml uses vehicle IDs generally starting with relevant prefixes or we handle all
        veh = stop.get('id')
        if veh not in veh_stops: veh_stops[veh] = []
        veh_stops[veh].append({
            'stop_id': stop.get('busStop'),
            'arrival': float(stop.get('started')), # SUMO stopinfo: started=arrival at stop
            'departure': float(stop.get('ended'))  # ended=departure from stop
        })
    
    # Calculate link travel times in sim
    sim_links = []
    for veh, stops in veh_stops.items():
        stops.sort(key=lambda x: x['arrival'])
        for i in range(len(stops) - 1):
            s1 = stops[i]
            s2 = stops[i+1]
            travel_time = s2['arrival'] - s1['departure']
            sim_links.append({
                'from_stop': s1['stop_id'],
                'to_stop': s2['stop_id'],
                'travel_time_s': travel_time,
                'veh_id': veh
            })
    
    df_sim = pd.DataFrame(sim_links)
    if df_sim.empty:
        print("Error: No link data extracted from simulation (Empty stopinfo?).")
        return

    # 3. Preparation for Comparison
    # Map stop_id to seq using distance file
    df_dist = pd.read_csv(route_dist_file)
    df_dist_68x = df_dist[(df_dist['route'] == '68X') & (df_dist['bound'] == 'inbound')].sort_values('seq')
    
    stop_to_seq = df_dist_68x.set_index('stop_id')['seq'].to_dict()
    
    # Calculate Link Pairs (from_seq, to_seq) -> dist
    link_dists = {}
    stop_ids = df_dist_68x['stop_id'].tolist()
    seqs = df_dist_68x['seq'].tolist()
    dists = df_dist_68x['link_dist_m'].tolist() # Distance from previous stop
    
    # Calculate cumulative distance map
    curr_dist = 0.0
    seq_to_dist = {}
    for i, seq in enumerate(seqs):
        curr_dist += dists[i] # simplistic cumulative
        seq_to_dist[seq] = curr_dist
        
        if i < len(seqs) - 1:
            # Distance between seq[i] and seq[i+1] is dists[i+1]
            # Wait, link_dist_m is usually distance of the stop from start? OR segment length?
            # doc says "distance FROM previous stop to this stop".
            # So dists[i+1] is the distance from i to i+1.
            link_dists[(seqs[i], seqs[i+1])] = dists[i+1]

    # Aggregating Real Data
    real_agg = df_real_68x.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().reset_index()
    real_agg.columns = ['from_seq', 'to_seq', 'real_time_s']
    
    # Aggregating Sim Data
    # First map stops to seq
    df_sim['from_seq'] = df_sim['from_stop'].map(stop_to_seq)
    df_sim['to_seq'] = df_sim['to_stop'].map(stop_to_seq)
    df_sim = df_sim.dropna(subset=['from_seq', 'to_seq']) # Drop links involving unknown stops
    
    sim_agg = df_sim.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().reset_index()
    sim_agg.columns = ['from_seq', 'to_seq', 'sim_time_s']
    
    # Merge on Link (from_seq, to_seq)
    merged = pd.merge(real_agg, sim_agg, on=['from_seq', 'to_seq'], how='inner')
    
    if merged.empty:
        print("Error: No overlapping links found between Real and Sim data.")
        return
        
    # Add distances and speeds
    merged['dist_m'] = merged.apply(lambda x: link_dists.get((x['from_seq'], x['to_seq']), 0), axis=1)
    merged = merged[merged['dist_m'] > 0]
    
    merged['real_speed_kmh'] = (merged['dist_m']/1000) / (merged['real_time_s']/3600)
    merged['sim_speed_kmh'] = (merged['dist_m']/1000) / (merged['sim_time_s']/3600)
    
    # 4. Trajectory Plotting (Aligned)
    # Find the sequence of connected links that we can plot
    # Since network is cropped, we might have a contiguous chain in the middle
    sorted_links = merged.sort_values('from_seq')
    
    # Reconstruct trajectory for the overlapping part
    # We start from the first 'from_seq' in the merged data
    start_seq = sorted_links.iloc[0]['from_seq']
    
    # Initial offset based on Real world distance/time
    start_dist = seq_to_dist.get(start_seq, 0)
    # We align both to time=0 at this start point for clear comparison of relative pace
    # OR we align Sim to Real's cumulative time at this point. 
    # Let's align both to 0 time at the entry of the cropped network.
    
    traj_dist = [start_dist]
    traj_real = [0.0]
    traj_sim = [0.0]
    
    curr_seq = start_seq
    
    # Chain traversal
    while True:
        # Find link starting at curr_seq
        row = sorted_links[sorted_links['from_seq'] == curr_seq]
        if row.empty:
            break
        
        row = row.iloc[0] # Assume one next hop (linear route)
        next_seq = row['to_seq']
        
        d = row['dist_m']
        t_r = row['real_time_s']
        t_s = row['sim_time_s']
        
        traj_dist.append(traj_dist[-1] + d)
        traj_real.append(traj_real[-1] + t_r)
        traj_sim.append(traj_sim[-1] + t_s)
        
        curr_seq = next_seq
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(traj_dist, traj_real, 'b-o', label='Real Data (Aligned)')
    plt.plot(traj_dist, traj_sim, 'r-x', label='Simulation (Cropped)')
    plt.xlabel('Cumulative Distance (m)')
    plt.ylabel('Cumulative Travel Time (s) (from crop entry)')
    plt.title(f'Trajectory Comparison - 68X Inbound (Cropped Area)\nStart Seq: {start_seq}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # 5. Stats Output
    print("\n--- Statistics (Overlapping Segments) ---")
    total_real = traj_real[-1]
    total_sim = traj_sim[-1]
    diff = total_sim - total_real
    print(f"Total Time in Cropped Area: Real={total_real:.1f}s, Sim={total_sim:.1f}s")
    print(f"Difference: {diff:.1f}s ({diff/total_real*100:.1f}%)")
    
    # L2 Speed Distribution
    ks_stat, p_val = ks_2samp(merged['real_speed_kmh'], merged['sim_speed_kmh'])
    print(f"\nSpeed Distribution (Link-based averages):")
    print(f"Real Mean: {merged['real_speed_kmh'].mean():.1f} km/h")
    print(f"Sim Mean:  {merged['sim_speed_kmh'].mean():.1f} km/h")
    print(f"KS Stat: {ks_stat:.4f}, P-Value: {p_val:.4f}")

if __name__ == "__main__":
    compare_week2_cropped()
