import pandas as pd
import numpy as np
import os
import argparse
from datetime import timedelta

def clean_kmb_links(eta_file, dist_file, output_times, output_speeds):
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
        if bound is None: continue
        print(f"Processing {route} {bound} (Service {st})...")
        
        # Merge distances
        route_dist = df_dist[(df_dist['route'] == route) & 
                             (df_dist['bound'] == bound) & 
                             (df_dist['service_type'] == int(st))]
        
        if route_dist.empty:
            print(f"  Warning: No distance data for {route} {bound} type {st}")
            continue
            
        # For each stop_seq, identify arrival/departure times of discrete bus passes
        stop_events = {}
        for seq in sorted(group['stop_seq'].unique()):
            stop_data = group[group['stop_seq'] == seq].copy()
            
            # Identify discrete bus passes using has_departed as a delimiter
            # Shift has_departed to mark the start of the *next* bus
            # If has_departed is missing, we use ETA jump as backup
            if 'has_departed' not in stop_data.columns:
                print(f"  Warning: No 'has_departed' in data for seq {seq}")
                continue
                
            stop_data['bus_index'] = stop_data['has_departed'].shift(1).fillna(False).cumsum()
            
            passes = []
            for b_idx, bus_pass in stop_data.groupby('bus_index'):
                arrival = bus_pass[bus_pass['is_arrived'] == True]['capture_ts'].min()
                # Departure is marked by has_departed == True in the current bus_pass
                # Or the last record of this bus_pass
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
            if dist_info.empty: continue
            link_dist = dist_info['link_dist_m'].values[0]
            if pd.isna(link_dist) or link_dist == 0: continue
            
            events1 = stop_events.get(s1, [])
            events2 = stop_events.get(s2, [])
            
            for e1 in events1:
                # Start time is when it left s1
                start_time = e1['departure']
                
                # Look for the first arrival at N+1 that is after start_time 
                # and within a reasonable window (e.g. 5 to 60 mins depending on distance)
                # For very short segments, 5 mins is enough. For long ones, more.
                # Let's use 40 minutes as a broad upper bound for a single link.
                matches = [e2 for e2 in events2 if e2['arrival'] > start_time and (e2['arrival'] - start_time).total_seconds() < 2400]
                
                if matches:
                    e2 = matches[0]
                    travel_time = (e2['arrival'] - start_time).total_seconds()
                    
                    if travel_time > 10: # Minimum 10 seconds for a link
                        speed_kms = (link_dist / 1000.0) / (travel_time / 3600.0)
                        
                        if 2 <= speed_kms <= 100: # 2km/h to 100km/h
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
    
    # Save link times
    df_results.to_csv(output_times, index=False)
    
    # Generate link speeds (could be the same or aggregated)
    # For now, output the same enriched table as speeds
    df_results.to_csv(output_speeds, index=False)
    
    print(f"Generated {len(df_results)} link records.")
    print(f"Times saved to {output_times}")
    print(f"Speeds saved to {output_speeds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate link travel times and speeds.")
    parser.add_argument("--eta", default="data/processed/station_eta.csv")
    parser.add_argument("--dist", default="data/processed/kmb_route_stop_dist.csv")
    parser.add_argument("--out-times", default="data/processed/link_times.csv")
    parser.add_argument("--out-speeds", default="data/processed/link_speeds.csv")
    
    args = parser.parse_args()
    
    clean_kmb_links(args.eta, args.dist, args.out_times, args.out_speeds)
