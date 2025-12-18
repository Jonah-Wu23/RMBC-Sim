import pandas as pd
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime

def compare_week1():
    # File Paths
    real_link_file = r"data/processed/enriched_link_stats.csv"
    sim_stopinfo = r"sumo/output/stopinfo.xml"
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
    print("Loading simulation data...")
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
        
    def get_speed(row):
        dist = dist_map.get((row['from_stop'], row['to_stop'])) # Note: real data uses stop_id, sim uses busStop ID
        if dist and row['travel_time_s'] > 0:
            return (dist / 1000.0) / (row['travel_time_s'] / 3600.0)
        return None

    # We need to map real stop_id to sim busStop ID if they differ, but here they should be identical.
    # Let's check a few real ones
    
    # Enrich sim with speed
    # In sim scripts, we might HAVE normalized stop IDs.
    # Let's assume they match for now.
    
    df_sim['speed_kmh'] = df_sim.apply(lambda r: (dist_map.get((r['from_stop'], r['to_stop']), 0) / 1000.0) / (r['travel_time_s'] / 3600.0) if dist_map.get((r['from_stop'], r['to_stop']), 0) > 0 else None, axis=1)
    df_sim = df_sim.dropna(subset=['speed_kmh'])

    # 4. Compare Distributions (L2)
    print("\n--- L2 Link Speed Comparison (68X Inbound) ---")
    
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

    # 5. L1 Comparison (Average Arrival Error)
    print("\n--- L1 Arrival Time Comparison ---")
    # This is trickier because of absolute vs relative time.
    # Let's compare "Cumulative Travel Time from Seq 1"
    
    # Real avg cumulative time
    # We need seq for this.
    pass

if __name__ == "__main__":
    compare_week1()
