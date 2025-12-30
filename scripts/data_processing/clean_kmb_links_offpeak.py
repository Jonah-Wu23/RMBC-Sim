"""
Generate link-level D2D travel time and speed metrics for off-peak period.
Based on clean_kmb_links.py but reads from data2/ directory.

Output: data2/processed/link_stats_offpeak.csv
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import timedelta

# Default paths
ETA_FILE = "data2/processed/station_eta_offpeak.csv"
DIST_FILE = "data/processed/kmb_route_stop_dist.csv"  # Reuse peak distance data
OUTPUT_TIMES = "data2/processed/link_times_offpeak.csv"
OUTPUT_SPEEDS = "data2/processed/link_speeds_offpeak.csv"
OUTPUT_STATS = "data2/processed/link_stats_offpeak.csv"


def clean_kmb_links_offpeak(eta_file, dist_file, output_times, output_speeds, output_stats):
    print(f"Loading {eta_file}...")
    df_eta = pd.read_csv(eta_file)
    print(f"Loading {dist_file}...")
    df_dist = pd.read_csv(dist_file)

    # Ensure timestamps are datetime
    df_eta['capture_ts'] = pd.to_datetime(df_eta['capture_ts'])
    
    # We only care about eta_seq=1 for tracking the 'next' bus
    df_seq1 = df_eta[df_eta['eta_seq'] == 1].copy()
    
    # Map bounds
    bound_map = {'I': 'inbound', 'O': 'outbound'}
    df_seq1['bound'] = df_seq1['bound'].map(bound_map)
    
    # Sort by route, bound, service_type, seq, and capture_ts
    df_seq1 = df_seq1.sort_values(['route', 'bound', 'service_type', 'stop_seq', 'capture_ts'])

    results = []

    # Group by route, bound, service_type
    groups = df_seq1.groupby(['route', 'bound', 'service_type'])
    
    for (route, bound, st), group in groups:
        if bound is None: 
            continue
        print(f"Processing {route} {bound} (Service {st})...")
        
        # Merge distances
        route_dist = df_dist[(df_dist['route'] == route) & 
                             (df_dist['bound'] == bound) & 
                             (df_dist['service_type'] == int(st))]
        
        if route_dist.empty:
            print(f"  Warning: No distance data for {route} {bound} type {st}")
            continue
            
        # For each stop_seq, identify arrival/departure times
        stop_events = {}
        for seq in sorted(group['stop_seq'].unique()):
            stop_data = group[group['stop_seq'] == seq].copy()
            
            if 'has_departed' not in stop_data.columns:
                print(f"  Warning: No 'has_departed' in data for seq {seq}")
                continue
                
            stop_data['bus_index'] = stop_data['has_departed'].shift(1).fillna(False).cumsum()
            
            passes = []
            for b_idx, bus_pass in stop_data.groupby('bus_index'):
                arrival = bus_pass[bus_pass['is_arrived'] == True]['capture_ts'].min()
                last_record = bus_pass['capture_ts'].max()
                departure = bus_pass[bus_pass['has_departed'] == True]['capture_ts'].min()
                
                if pd.notna(arrival):
                    passes.append({
                        'arrival': arrival,
                        'departure': departure if pd.notna(departure) else last_record,
                        'stop_seq': seq
                    })
            stop_events[seq] = passes

        # Match passes across consecutive stops
        seqs = sorted(stop_events.keys())
        for i in range(len(seqs) - 1):
            s1 = seqs[i]
            s2 = seqs[i+1]
            
            dist_info = route_dist[route_dist['seq'] == s2]
            if dist_info.empty: 
                continue
            link_dist = dist_info['link_dist_m'].values[0]
            if pd.isna(link_dist) or link_dist == 0: 
                continue
            
            events1 = stop_events.get(s1, [])
            events2 = stop_events.get(s2, [])
            
            for e1 in events1:
                start_time = e1['departure']
                
                # Look for first arrival at N+1 after start_time within 40 min
                matches = [e2 for e2 in events2 
                          if e2['arrival'] > start_time 
                          and (e2['arrival'] - start_time).total_seconds() < 2400]
                
                if matches:
                    e2 = matches[0]
                    travel_time = (e2['arrival'] - start_time).total_seconds()
                    
                    if travel_time > 10:  # Minimum 10 seconds for a link
                        speed_kms = (link_dist / 1000.0) / (travel_time / 3600.0)
                        
                        if 2 <= speed_kms <= 100:  # 2km/h to 100km/h
                            results.append({
                                'route': route,
                                'bound': bound,
                                'service_type': st,
                                'from_seq': s1,
                                'to_seq': s2,
                                'departure_ts': start_time,
                                'arrival_ts': e2['arrival'],
                                'travel_time_s': round(travel_time, 2),
                                'dist_m': round(link_dist, 2),
                                'speed_kmh': round(speed_kms, 2)
                            })

    if not results:
        print("No link data generated.")
        return

    df_results = pd.DataFrame(results)
    
    # Save raw link times
    df_results.to_csv(output_times, index=False)
    df_results.to_csv(output_speeds, index=False)
    
    # Generate aggregated link stats (for comparison with simulation)
    stats = df_results.groupby(['route', 'bound', 'from_seq', 'to_seq']).agg({
        'travel_time_s': ['mean', 'median', 'std', 'count', lambda x: x.quantile(0.9)],
        'speed_kmh': ['mean', 'median', 'std', lambda x: x.quantile(0.9)],
        'dist_m': 'first'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['route', 'bound', 'from_seq', 'to_seq',
                     'tt_mean', 'tt_median', 'tt_std', 'sample_count', 'tt_p90',
                     'speed_mean', 'speed_median', 'speed_std', 'speed_p90',
                     'dist_m']
    
    stats.to_csv(output_stats, index=False)
    
    print(f"\n--- Link Statistics Summary (Off-Peak) ---")
    print(f"Total link records: {len(df_results)}")
    print(f"Unique links: {len(stats)}")
    print(f"Routes: {df_results['route'].unique().tolist()}")
    print(f"\nOverall Speed Stats:")
    print(f"  Mean: {df_results['speed_kmh'].mean():.2f} km/h")
    print(f"  Median: {df_results['speed_kmh'].median():.2f} km/h")
    print(f"  P90: {df_results['speed_kmh'].quantile(0.9):.2f} km/h")
    print(f"\nOverall Travel Time Stats:")
    print(f"  Mean: {df_results['travel_time_s'].mean():.2f} s")
    print(f"  Median: {df_results['travel_time_s'].median():.2f} s")
    print(f"  P90: {df_results['travel_time_s'].quantile(0.9):.2f} s")
    
    print(f"\nOutput files:")
    print(f"  Times: {output_times}")
    print(f"  Speeds: {output_speeds}")  
    print(f"  Stats: {output_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate link travel times and speeds for off-peak.")
    parser.add_argument("--eta", default=ETA_FILE)
    parser.add_argument("--dist", default=DIST_FILE)
    parser.add_argument("--out-times", default=OUTPUT_TIMES)
    parser.add_argument("--out-speeds", default=OUTPUT_SPEEDS)
    parser.add_argument("--out-stats", default=OUTPUT_STATS)
    
    args = parser.parse_args()
    
    clean_kmb_links_offpeak(args.eta, args.dist, args.out_times, args.out_speeds, args.out_stats)
