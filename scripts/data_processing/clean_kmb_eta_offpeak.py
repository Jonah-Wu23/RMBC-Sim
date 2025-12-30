"""
Clean KMB ETA data for off-peak period (2025-12-30 15:00-16:00).
Based on clean_kmb_eta.py but reads from data2/ directory.

Output: data2/processed/station_eta_offpeak.csv
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone

# Off-peak specific directories
ROUTE_STOP_DIR = "data/raw/kmb/route-stop"  # Reuse peak data for stop mapping
ETA_DIR = "data2/raw/kmb/route-eta"
OUTPUT_FILE = "data2/processed/station_eta_offpeak.csv"


def load_stop_mapping(route_stop_dir):
    """Builds a mapping from (route, bound, service_type, seq) to stop_id."""
    mapping = {}
    files = glob.glob(os.path.join(route_stop_dir, "*.json"))
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                for item in content.get('data', []):
                    key = (
                        item['route'],
                        item['bound'],
                        str(item['service_type']),
                        str(item['seq'])
                    )
                    mapping[key] = item['stop']
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return mapping


def parse_capture_ts(filename):
    """Extracts capture_ts from filename like 'kmb-route-eta-68X-20251230-150006.json'."""
    basename = os.path.basename(filename)
    parts = basename.replace('.json', '').split('-')
    if len(parts) >= 2:
        ts_str = f"{parts[-2]}-{parts[-1]}"
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
            # Return timezone-aware datetime (HKT = UTC+8)
            return dt.replace(tzinfo=timezone(timedelta(hours=8)))
        except ValueError:
            return None
    return None


def clean_kmb_eta_offpeak(eta_dir, stop_mapping, output_file):
    all_rows = []
    eta_files = sorted(glob.glob(os.path.join(eta_dir, "*.json")))
    
    if not eta_files:
        print("No ETA files found.")
        return

    print(f"Found {len(eta_files)} ETA files in {eta_dir}")
    
    # Track data gaps
    prev_ts = None
    gaps = []

    for file in eta_files:
        capture_ts = parse_capture_ts(file)
        if not capture_ts:
            continue
            
        if prev_ts and (capture_ts - prev_ts) > timedelta(minutes=5):
            gaps.append((prev_ts, capture_ts))
        prev_ts = capture_ts

        with open(file, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                for item in content.get('data', []):
                    key = (
                        item['route'],
                        item['dir'],
                        str(item['service_type']),
                        str(item['seq'])
                    )
                    stop_id = stop_mapping.get(key, None)
                    
                    row = {
                        'capture_ts': capture_ts,
                        'route': item.get('route'),
                        'bound': item.get('dir'),
                        'service_type': item.get('service_type'),
                        'stop_seq': item.get('seq'),
                        'stop_id': stop_id,
                        'eta': item.get('eta'),
                        'eta_seq': item.get('eta_seq'),
                        'data_timestamp': item.get('data_timestamp'),
                        'gps': item.get('gps'),
                        'rmk_en': item.get('rmk_en')
                    }
                    all_rows.append(row)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if not all_rows:
        print("No ETA data extracted.")
        return

    df = pd.DataFrame(all_rows)
    
    # Pre-process timestamps
    df['capture_ts'] = pd.to_datetime(df['capture_ts'], utc=True)
    df['eta'] = pd.to_datetime(df['eta'], utc=True, errors='coerce')
    df['data_timestamp'] = pd.to_datetime(df['data_timestamp'], utc=True, errors='coerce')

    # Sort for time-series analysis
    df = df.sort_values(['route', 'bound', 'service_type', 'stop_seq', 'capture_ts', 'eta_seq'])

    # Arrival/Departure Detection Logic (same as peak version)
    seq1 = df[df['eta_seq'] == 1].copy()
    
    seq1['is_arrived'] = (seq1['eta'] <= seq1['capture_ts']) | (seq1['rmk_en'].str.contains('Arrived', case=False, na=False))
    seq1['eta_jump'] = seq1.groupby(['route', 'bound', 'service_type', 'stop_seq'])['eta'].diff().dt.total_seconds()
    seq1['has_departed'] = (seq1['eta_jump'] > 120) & (seq1.groupby(['route', 'bound', 'service_type', 'stop_seq'])['is_arrived'].shift(1) == True)

    # Merge detection back to main dataframe
    df = df.merge(
        seq1[['route', 'bound', 'service_type', 'stop_seq', 'capture_ts', 'is_arrived', 'has_departed']],
        on=['route', 'bound', 'service_type', 'stop_seq', 'capture_ts'],
        how='left'
    )

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\nProcessed {len(eta_files)} files. Output saved to {output_file}")
    
    # Generate Statistics
    num_arrivals = seq1['is_arrived'].sum()
    num_departures = seq1['has_departed'].sum()
    
    print("\n--- Data Quality Report (Off-Peak) ---")
    print(f"Total ETA records: {len(df)}")
    print(f"Total files processed: {len(eta_files)}")
    print(f"Arrival events detected (eta_seq=1): {num_arrivals}")
    print(f"Departure events detected (eta_seq=1): {num_departures}")
    print(f"Unique routes: {df['route'].unique().tolist()}")
    print(f"Time range: {df['capture_ts'].min()} to {df['capture_ts'].max()}")
    
    if gaps:
        print(f"Data gaps (> 5 mins) detected: {len(gaps)}")
        for start, end in gaps:
            print(f"  {start} to {end} (Duration: {end - start})")
    else:
        print("No significant data gaps detected.")
    
    # Missing timestamps check
    missing_eta = df['eta'].isna().sum()
    if missing_eta > 0:
        print(f"Records with missing ETA: {missing_eta}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("Loading stop mapping from peak data...")
    stop_map = load_stop_mapping(ROUTE_STOP_DIR)
    print(f"Loaded {len(stop_map)} stop mappings")
    
    print("\nCleaning ETA data for off-peak period...")
    clean_kmb_eta_offpeak(ETA_DIR, stop_map, OUTPUT_FILE)
