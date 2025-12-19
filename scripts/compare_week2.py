import pandas as pd
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime
import matplotlib.pyplot as plt

def compare_week2():
    # File Paths
    real_link_file = r"data/processed/enriched_link_stats.csv"
    sim_stopinfo = r"sumo/output/stopinfo_irn.xml"  # UPDATED for Week 2
    route_dist_file = r"data/processed/kmb_route_stop_dist.csv"
    
    if not os.path.exists(sim_stopinfo):
        print(f"Error: {sim_stopinfo} not found. Run simulation first.")
        return

    # 1. Load Real Data
    print("Loading real link data...")
    df_real = pd.read_csv(real_link_file)
    # Filter for Route 68X Inbound (our primary test case for now)
    df_real_68x = df_real[(df_real['route'] == '68X') & (df_real['bound'] == 'inbound')]
    
    if df_real_68x.empty:
        print("Warning: No real data for 68X inbound.")
    
    # 2. Load Sim Data
    print(f"Loading simulation data from {sim_stopinfo}...")
    tree = ET.parse(sim_stopinfo)
    root = tree.getroot()
    
    sim_records = []
    # group by vehicle to get link times
    veh_stops = {}
    for stop in root.findall('stopinfo'):
        veh = stop.get('id')
        if veh not in veh_stops: veh_stops[veh] = []
        veh_stops[veh].append({
            'stop_id': stop.get('busStop'),
            'arrival': float(stop.get('started')),
            'departure': float(stop.get('ended'))
        })
    
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
    
    if df_sim.empty:
        print("Error: No link data extracted from simulation.")
        return

    # 3. Join with Distance to get Speed
    print("Loading distances...")
    df_dist = pd.read_csv(route_dist_file)
    # Create mapping from (from_stop_id, to_stop_id) to link_dist_m
    # In df_dist, each row has stop_id and link_dist_m (distance FROM previous stop to this stop)
    dist_map = {}
    last_stop = None
    for _, row in df_dist[(df_dist['route'] == '68X') & (df_dist['bound'] == 'inbound')].iterrows():
        this_stop = row['stop_id']
        if last_stop:
            dist_map[(last_stop, this_stop)] = row['link_dist_m']
        last_stop = this_stop
        
    df_sim['speed_kmh'] = df_sim.apply(lambda r: (dist_map.get((r['from_stop'], r['to_stop']), 0) / 1000.0) / (r['travel_time_s'] / 3600.0) if dist_map.get((r['from_stop'], r['to_stop']), 0) > 0 and r['travel_time_s'] > 0 else None, axis=1)
    df_sim = df_sim.dropna(subset=['speed_kmh'])

    # 4. Compare Distributions (L2)
    print("\n--- L2 Link Speed Comparison (68X Inbound) - Week 2 (IRN) ---")
    
    # Overall distribution
    real_speeds = df_real_68x['speed_kmh'].values
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
    real_link_agg = df_real_68x.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().reset_index()
    real_link_agg.rename(columns={'travel_time_s': 'real_avg_time_s'}, inplace=True)
    
    # 5.2 Sim Link Stats Aggregation
    sim_link_agg = df_sim.groupby(['from_stop', 'to_stop'])['travel_time_s'].mean().reset_index()
    sim_link_agg.rename(columns={'travel_time_s': 'sim_avg_time_s'}, inplace=True)
    
    # Merge for Comparison Table
    df_68x_dist = df_dist[(df_dist['route'] == '68X') & (df_dist['bound'] == 'inbound')].sort_values('seq')
    seq_map = df_68x_dist.set_index('stop_id')['seq'].to_dict()
    
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
    real_link_avg = real_link_agg.sort_values('from_seq')
    real_cum_times = [0.0]
    total_t = 0
    for _, row in real_link_avg.iterrows():
        total_t += row['real_avg_time_s']
        real_cum_times.append(total_t)
    
    # Re-calculate to ensure alignment
    real_times_map = {row['to_seq']: row['real_avg_time_s'] for _, row in real_link_agg.iterrows()}
    sim_times_map = {row['to_seq']: row['sim_avg_time_s'] for _, row in sim_link_agg.iterrows()}
    
    all_seqs = sorted(df_68x_dist['seq'].tolist())
    real_cum = [0.0]
    sim_cum = [0.0]
    dists = [0.0]
    current_dist = 0
    
    for i in range(1, len(all_seqs)):
        seq = all_seqs[i]
        d = df_68x_dist[df_68x_dist['seq'] == seq]['link_dist_m'].values[0]
        current_dist += d
        dists.append(current_dist)
        
        real_cum.append(real_cum[-1] + real_times_map.get(seq, 0))
        sim_cum.append(sim_cum[-1] + sim_times_map.get(seq, 0))

    # 6. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dists, real_cum, 'b-o', label='Real (Average)')
    plt.plot(dists, sim_cum, 'g-s', label='Sim (Week 2 IRN)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Cumulative Time (s)')
    plt.title('Bus Route Trajectory (Distance-Time) - 68X Inbound (Week 2)')
    plt.legend()
    plt.grid(True)
    
    plot_path = "sumo/output/week2_trajectory_comparison.png"
    plt.savefig(plot_path)
    print(f"\nTrajectory plot saved to: {plot_path}")
    
    # 7. Summary Report
    total_real = real_cum[-1]
    total_sim = sim_cum[-1]
    print(f"\nFinal Arrival Window (End-to-End):")
    print(f"Real: {total_real:.1f}s")
    print(f"Sim:  {total_sim:.1f}s")
    print(f"Error: {total_sim - total_real:.1f}s ({(total_sim - total_real)/total_real*100:.1f}%)")

if __name__ == "__main__":
    compare_week2()
