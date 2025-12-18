import pandas as pd
import numpy as np
import os

def calculate_link_stats(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Ensure departure_ts is datetime
    df['departure_ts'] = pd.to_datetime(df['departure_ts'])
    
    # Extract hour or time slot (e.g., 15-min intervals)
    # For Week 1, let's aggregate by Route, Bound, From_Seq, and Hour
    df['hour'] = df['departure_ts'].dt.hour
    
    # Calculate stats per link
    stats = df.groupby(['route', 'bound', 'service_type', 'from_seq', 'to_seq', 'hour']).agg(
        count=('speed_kmh', 'count'),
        avg_speed=('speed_kmh', 'mean'),
        std_speed=('speed_kmh', 'std'),
        p10_speed=('speed_kmh', lambda x: x.quantile(0.1)),
        p50_speed=('speed_kmh', lambda x: x.quantile(0.5)),
        p90_speed=('speed_kmh', lambda x: x.quantile(1.0 if len(x) == 0 else 0.9))
    ).reset_index()
    
    # Round values
    for col in ['avg_speed', 'std_speed', 'p10_speed', 'p50_speed', 'p90_speed']:
        stats[col] = stats[col].round(2)
        
    stats.to_csv(output_file, index=False)
    print(f"Stats saved to {output_file}")
    print(f"Total links processed: {len(stats)}")

if __name__ == "__main__":
    INPUT_FILE = "data/processed/link_speeds.csv"
    OUTPUT_FILE = "data/processed/link_speed_stats.csv"
    
    if os.path.exists(INPUT_FILE):
        calculate_link_stats(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"Error: {INPUT_FILE} not found.")
