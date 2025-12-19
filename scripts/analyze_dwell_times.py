
import pandas as pd
import numpy as np
import os
import sys

def analyze_dwell_times(eta_file):
    print(f"Loading {eta_file}...")
    try:
        df = pd.read_csv(eta_file)
    except FileNotFoundError:
        print(f"Error: File {eta_file} not found.")
        return

    # Check columns
    required_cols = ['stop_id', 'is_arrived', 'has_departed', 'capture_ts', 'route', 'bound']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column {col} missing in {eta_file}")
            return

    df['capture_ts'] = pd.to_datetime(df['capture_ts'])

    # Filter for events
    # We need to pair arrival and departure for the same vehicle visit.
    # Since we don't have vehicle_id, we infer visits by grouping consecutive events.
    # A simplified approach: 
    # For each (route, bound, stop_id), sort by time.
    # Identify blocks of "Arrived" status vs "Departed" status.
    # Dwell Time = Departure Time - Arrival Time.
    
    # However, 'station_eta.csv' as processed by `clean_kmb_eta.py` (from my knowledge) 
    # has booleans 'is_arrived' and 'has_departed' on the time series.
    # We extracted 'arrival_ts' and 'departure_ts' in the cleaning logic, but did we save it?
    # Let's check the schema of station_eta.csv. 
    # If it's the raw time series with flags, we need to aggregate.
    
    print("Analyzing Dwell Times...")
    dwell_times = []

    # Group by unique trip identifier proxies
    # (route, bound, service_type, stop_seq) - this defines a specific stop on a route.
    # But for TIME dimension, we need to separate different bus runs.
    # We can just look at the 'is_arrived' and 'has_departed' flags.
    
    # Iterate groups
    groups = df.groupby(['route', 'bound', 'service_type', 'stop_seq', 'stop_id'])
    
    for name, group in groups:
        group = group.sort_values('capture_ts')
        
        # Find arrival times (first timestamp where is_arrived=True in a block)
        # We need to distinguish separate bus arrivals.
        # Assuming buses are separated by some time gap (e.g. > 2 mins).
        
        # Get indices of arrivals and departures
        arrivals = group[group['is_arrived'] == True]
        departures = group[group['has_departed'] == True]
        
        if arrivals.empty or departures.empty:
            continue
            
        # This is tricky without vehicle IDs. 
        # But `clean_kmb_eta.py` logic was:
        # Arrival = First time eta_seq=1 and (eta <= now or rmk='Arrived')
        # Departure = First time eta_seq=1 jumps forward.
        
        # Let's try to match pairs greedily timeframe.
        # For each departure, find the latest arrival that happened BEFORE it.
        
        # Convert to list
        arr_times = arrivals['capture_ts'].tolist()
        dep_times = departures['capture_ts'].tolist()
        
        # Simple matching:
        # Iterate departures, find closest preceding arrival within reasonable dwell limit (e.g. 10 mins)
        for dep in dep_times:
            # Filter arrs before dep
            possible_arrs = [t for t in arr_times if t < dep and (dep - t).total_seconds() < 600]
            if possible_arrs:
                arr = possible_arrs[-1] # closest one
                dwell = (dep - arr).total_seconds()
                if dwell > 0:
                    dwell_times.append({
                        'route': name[0],
                        'bound': name[1],
                        'stop_id': name[4],
                        'dwell_time': dwell
                    })
                    # Remove used arrival to avoid double counting? 
                    # Actually valid: multiple departures could reuse same arrival if data is noisy, 
                    # but theoretically 1 arr -> 1 dep. 
                    arr_times.remove(arr) 

    if not dwell_times:
        print("No valid dwell times extracted.")
        return

    res_df = pd.DataFrame(dwell_times)
    
    # Global Stats
    mean_dwell = res_df['dwell_time'].mean()
    std_dwell = res_df['dwell_time'].std()
    
    print("\n[Global Results]")
    print(f"Total Samples: {len(res_df)}")
    print(f"Mean Dwell Time: {mean_dwell:.2f} s")
    print(f"Std Dev: {std_dwell:.2f} s")
    
    # Per Stop Stats (Top 10)
    print("\n[Top 10 Stops by Sample Count]")
    stop_stats = res_df.groupby('stop_id')['dwell_time'].agg(['count', 'mean', 'std']).sort_values('count', ascending=False)
    print(stop_stats.head(10))
    
    # Distribution for code generation
    # We want to recommend parameters for SUMO
    # Usually LogNormal is good for dwell time.
    # Calculate LogMean and LogStd from data
    # log_data = np.log(res_df['dwell_time'] + 1e-9) # avoid zero
    # mu = log_data.mean()
    # sigma = log_data.std()
    
    print("\n[Recommended SUMO Parameters]")
    print(f"Default duration (mean): {mean_dwell:.1f}")
    # print(f"LogNormal mu: {mu:.2f}, sigma: {sigma:.2f}")

if __name__ == "__main__":
    ETA_FILE = "data/processed/station_eta.csv"
    analyze_dwell_times(ETA_FILE)
