import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from scipy.stats import ks_2samp, wasserstein_distance
import os
import argparse

def parse_stopinfo(xml_file):
    print(f"Parsing {xml_file}...")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return pd.DataFrame()
    
    trips = {}
    
    # Iterate over all stopinfo elements
    for stop in root.findall('stopinfo'):
        veh_id = stop.get('id')
        # stop_id = stop.get('busStop') # Not strictly needed if we rely on sequence order
        started = float(stop.get('started')) # Arrival time at stop
        ended = float(stop.get('ended'))     # Departure time from stop
        
        if veh_id not in trips:
            trips[veh_id] = []
        
        trips[veh_id].append({
            'arrival': started,
            'departure': ended
        })
        
    # Process trips to get links
    links = []
    for veh_id, stops in trips.items():
        # Sort by arrival time to ensure correct sequence
        stops.sort(key=lambda x: x['arrival'])
        
        for i in range(len(stops) - 1):
            s1 = stops[i]
            s2 = stops[i+1]
            
            # Link Travel Time = Arrival at S2 - Departure from S1
            # Note: This ignores dwell time at S1, which is correct for "Link Travel Time"
            travel_time = s2['arrival'] - s1['departure']
            
            # Simulation time starts at 0. We can map this to an "Hour of Simulation".
            # Assuming sim starts at a specific hour (e.g. 7 AM) or just relative.
            # For now, we take int(depart_time / 3600) as the hour index relative to sim start.
            sim_hour = int(s1['departure'] / 3600)
            
            if travel_time > 0:
                links.append({
                    'veh_id': veh_id,
                    'from_seq': i + 1, # assuming sequence starts at 1
                    'to_seq': i + 2,
                    'travel_time_s': travel_time,
                    'sim_hour': sim_hour
                })
                
    return pd.DataFrame(links)

def calculate_overall_metrics(real_df, sim_df, dist_map):
    results = []
    
    unique_links = pd.concat([
        real_df[['from_seq', 'to_seq']], 
        sim_df[['from_seq', 'to_seq']]
    ]).drop_duplicates().sort_values(['from_seq', 'to_seq'])
    
    print(f"Calculating overall metrics for {len(unique_links)} links...")
    
    for _, row in unique_links.iterrows():
        f = row['from_seq']
        t = row['to_seq']
        
        r_data = real_df[(real_df['from_seq'] == f) & (real_df['to_seq'] == t)]
        s_data = sim_df[(sim_df['from_seq'] == f) & (sim_df['to_seq'] == t)]
        
        link_dist = dist_map.get((f, t), None)
        
        # Add speeds to Sim Data
        if not s_data.empty and link_dist:
             s_data = s_data.copy()
             s_data['speed_kmh'] = (link_dist / 1000.0) / (s_data['travel_time_s'] / 3600.0)
        
        metric_row = {
            'from_seq': int(f),
            'to_seq': int(t),
            'dist_m': round(link_dist, 2) if link_dist else None,
            'real_count': len(r_data),
            'sim_count': len(s_data)
        }
        
        # Real Stats
        if not r_data.empty:
            metric_row['real_p10_spd'] = round(r_data['speed_kmh'].quantile(0.1), 2)
            metric_row['real_p50_spd'] = round(r_data['speed_kmh'].quantile(0.5), 2)
            metric_row['real_p90_spd'] = round(r_data['speed_kmh'].quantile(0.9), 2)
            metric_row['real_mean_spd'] = round(r_data['speed_kmh'].mean(), 2)
            metric_row['real_std_spd'] = round(r_data['speed_kmh'].std(), 2)

        # Sim Stats
        if not s_data.empty and 'speed_kmh' in s_data.columns:
            metric_row['sim_p10_spd'] = round(s_data['speed_kmh'].quantile(0.1), 2)
            metric_row['sim_p50_spd'] = round(s_data['speed_kmh'].quantile(0.5), 2)
            metric_row['sim_p90_spd'] = round(s_data['speed_kmh'].quantile(0.9), 2)
            metric_row['sim_mean_spd'] = round(s_data['speed_kmh'].mean(), 2)
            metric_row['sim_std_spd'] = round(s_data['speed_kmh'].std(), 2)
            
        # Distance Metrics (KS & EMD)
        if not r_data.empty and not s_data.empty and 'speed_kmh' in s_data.columns:
            r_vals = r_data['speed_kmh'].dropna().values
            s_vals = s_data['speed_kmh'].dropna().values
            
            if len(r_vals) > 0 and len(s_vals) > 0:
                metric_row['emd'] = round(wasserstein_distance(r_vals, s_vals), 4)
                ks_stat, p_val = ks_2samp(r_vals, s_vals)
                metric_row['ks_stat'] = round(ks_stat, 4)
                metric_row['ks_p'] = round(p_val, 4)
                
        results.append(metric_row)
        
    return pd.DataFrame(results)

def calculate_hourly_metrics(real_df, sim_df, dist_map):
    # Prepare Hourly Data
    # Real Data already has timestamps, extract hour
    real_df['hour'] = pd.to_datetime(real_df['departure_ts']).dt.hour
    
    # Sim Data has 'sim_hour' (0, 1, 2...)
    # We need to map sim_hour to real hour if the simulation represents a specific time.
    # For now, let's assume sim_hour 0 = Real Hour X (e.g. 7AM or 17PM).
    # Since we might not know, we might just compare "Trends" or output raw index.
    # Let's just output raw Sim Hour data alongside Real Hour data (grouped).
    
    # We will output a row per (Link, Hour) -> Real Mean, Sim Mean
    # But matching the hours requires alignment.
    # Let's simple output: Link, Hour, Real_Mean, Real_Count
    # And separately: Link, Sim_Hour, Sim_Mean, Sim_Count
    # Merging is hard without knowing the start time.
    
    real_hourly = real_df.groupby(['from_seq', 'to_seq', 'hour'])['speed_kmh'].agg(['mean', 'count']).reset_index()
    real_hourly.columns = ['from_seq', 'to_seq', 'hour', 'real_mean_spd', 'real_count']
    
    # Calculate Sim Means (needs distance)
    # We join dist_map manually or calculate speed per record first
    sim_w_speed = []
    for _, row in sim_df.iterrows():
        f, t = row['from_seq'], row['to_seq']
        dist = dist_map.get((f, t), None)
        if dist:
            speed = (dist/1000) / (row['travel_time_s']/3600)
            sim_w_speed.append({
                'from_seq': f, 'to_seq': t, 'hour': row['sim_hour'], 'speed': speed
            })
    
    if sim_w_speed:
        sim_df_spd = pd.DataFrame(sim_w_speed)
        sim_hourly = sim_df_spd.groupby(['from_seq', 'to_seq', 'hour'])['speed'].agg(['mean', 'count']).reset_index()
        sim_hourly.columns = ['from_seq', 'to_seq', 'sim_hour', 'sim_mean_spd', 'sim_count']
    else:
        sim_hourly = pd.DataFrame(columns=['from_seq', 'to_seq', 'sim_hour', 'sim_mean_spd', 'sim_count'])
        
    return real_hourly, sim_hourly

def main():
    real_file = "data/processed/link_times.csv"
    sim_file = "sumo/output/stopinfo.xml"
    
    out_overall = "data/processed/link_metrics_overall.csv"
    out_hourly_real = "data/processed/link_metrics_hourly_real.csv"
    out_hourly_sim = "data/processed/link_metrics_hourly_sim.csv"
    
    print("Loading Real Data...")
    if not os.path.exists(real_file):
        print(f"Error: {real_file} not found.")
        return
        
    df_real = pd.read_csv(real_file)
    # Filter for 68X Inbound
    df_real = df_real[(df_real['route'] == '68X') & (df_real['bound'] == 'inbound')].copy()
    print(f"Loaded {len(df_real)} real records.")

    print("Loading Sim Data...")
    if not os.path.exists(sim_file):
        print(f"Error: {sim_file} not found.")
        return
    df_sim = parse_stopinfo(sim_file)
    print(f"Loaded {len(df_sim)} sim link records.")
    
    print("Loading Distance Data...")
    dist_file = "data/processed/kmb_route_stop_dist.csv"
    if not os.path.exists(dist_file):
        print(f"Error: {dist_file} not found. Using inferred distances from real data.")
        # Fallback
        dist_map = df_real.groupby(['from_seq', 'to_seq'])['dist_m'].mean().to_dict()
    else:
        df_dist = pd.read_csv(dist_file)
        # Filter 68X Inbound Service 1
        df_dist = df_dist[(df_dist['route'] == '68X') & (df_dist['bound'] == 'inbound') & (df_dist['service_type'] == 1)]
        # Map: from_seq (seq-1) -> to_seq (seq) : dist (link_dist_m)
        dist_map = {}
        for _, row in df_dist.iterrows():
            seq = row['seq']
            dist = row['link_dist_m']
            if seq > 1 and pd.notna(dist) and dist > 0:
                dist_map[(seq-1, seq)] = dist
        print(f"Loaded distance map for {len(dist_map)} links.")

    # 1. Overall Metrics (Distribution & EMD/KS)
    df_overall = calculate_overall_metrics(df_real, df_sim, dist_map)
    df_overall.to_csv(out_overall, index=False)
    print(f"Overall metrics saved to {out_overall}")
    
    # 2. Hourly Means (Time-varying)
    df_h_real, df_h_sim = calculate_hourly_metrics(df_real, df_sim, dist_map)
    df_h_real.to_csv(out_hourly_real, index=False)
    df_h_sim.to_csv(out_hourly_sim, index=False)
    print(f"Hourly metrics saved to {out_hourly_real} and {out_hourly_sim}")

    if not df_overall.empty and 'emd' in df_overall.columns:
        print("\nTop 5 Links with Largest Simulation-Reality Gap (EMD):")
        print(df_overall[['from_seq', 'to_seq', 'real_mean_spd', 'sim_mean_spd', 'emd']].sort_values('emd', ascending=False).head(5))

if __name__ == "__main__":
    main()
