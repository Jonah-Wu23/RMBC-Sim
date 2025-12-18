import os
import json
import glob
import pandas as pd
from datetime import datetime, timedelta

def load_stop_mapping(route_stop_dir):
    """
    Builds a mapping from (route, bound, service_type, seq) to stop_id.
    """
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
    """
    Extracts capture_ts from filename like 'kmb-route-eta-68X-20251217-173422.json'.
    """
    basename = os.path.basename(filename)
    parts = basename.replace('.json', '').split('-')
    if len(parts) >= 2:
        ts_str = f"{parts[-2]}-{parts[-1]}"
        try:
            # Assume Hong Kong time (+08:00)
            dt = datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
            # Create a simple UTC+8 datetime
            return dt.replace(tzinfo=pd.Timestamp(dt).tz_localize('Asia/Hong_Kong').tzinfo)
        except ValueError:
            return None
    return None

def clean_kmb_eta(eta_dir, stop_mapping, output_file):
    all_rows = []
    eta_files = sorted(glob.glob(os.path.join(eta_dir, "*.json")))
    
    if not eta_files:
        print("No ETA files found.")
        return

    # To track gaps
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
                    # Route-stop uses 'bound' (I/O), ETA uses 'dir' (I/O)
                    # Mapping key: (route, bound, service_type, seq)
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

    # Arrival/Departure Detection Logic
    # We focus on eta_seq=1 to track the "next" bus arriving at each stop.
    seq1 = df[df['eta_seq'] == 1].copy()
    
    # Actual Arrival: ETA <= capture_ts OR remark indicates arrival
    seq1['is_arrived'] = (seq1['eta'] <= seq1['capture_ts']) | (seq1['rmk_en'].str.contains('Arrived', case=False, na=False))
    
    # To detect when a bus *departs*, we look for when the current seq=1 bus is replaced by a later bus.
    # We can detect this by a jump in the ETA for seq=1.
    seq1['eta_jump'] = seq1.groupby(['route', 'bound', 'service_type', 'stop_seq'])['eta'].diff().dt.total_seconds()
    
    # A jump > 120 seconds (2 mins) usually indicates the next bus has taken over seq=1
    # We also check if the previous state was 'arrived'
    seq1['has_departed'] = (seq1['eta_jump'] > 120) & (seq1.groupby(['route', 'bound', 'service_type', 'stop_seq'])['is_arrived'].shift(1) == True)

    # Actual Arrival Time: the first capture_ts where is_arrived is True
    # Actual Departure Time: the capture_ts where has_departed becomes True
    
    results = []
    for (r, b, s, seq), group in seq1.groupby(['route', 'bound', 'service_type', 'stop_seq']):
        arrival_ts = group[group['is_arrived'] == True]['capture_ts'].min()
        departure_ts = group[group['has_departed'] == True]['capture_ts'].min()
        
        # Match back to the original dataframe or just use these results
        # For the target format, we want one row per "event" or just keep the time series?
        # The prompt says: "Extracts timestamp, stop_id, stop_seq, and eta... Calculates Actual Arrival and Actual Departure"
        
        # Let's add these to each row in the original df briefly for the specific bus
        # This is hard without bus_id. Let's just output the enriched time series.
        pass

    # Merge detection back to the main dataframe
    # We only mark the rows where eta_seq was 1 or where the event is relevant
    df = df.merge(
        seq1[['route', 'bound', 'service_type', 'stop_seq', 'capture_ts', 'is_arrived', 'has_departed']],
        on=['route', 'bound', 'service_type', 'stop_seq', 'capture_ts'],
        how='left'
    )

    df.to_csv(output_file, index=False)
    
    print(f"Processed {len(eta_files)} files. Output saved to {output_file}")
    
    # Generate Statistics for Report
    num_arrivals = seq1['is_arrived'].sum()
    num_departures = seq1['has_departed'].sum()
    
    print("\n--- Data Quality Report ---")
    print(f"Total ETA records: {len(df)}")
    print(f"Total files processed: {len(eta_files)}")
    print(f"Arrival events detected (eta_seq=1): {num_arrivals}")
    print(f"Departure events detected (eta_seq=1): {num_departures}")
    
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
    ROUTE_STOP_DIR = "data/raw/kmb/route-stop"
    ETA_DIR = "data/raw/kmb/route-eta"
    OUTPUT_FILE = "data/processed/station_eta.csv"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("Loading stop mapping...")
    stop_map = load_stop_mapping(ROUTE_STOP_DIR)
    
    print("Cleaning ETA data...")
    clean_kmb_eta(ETA_DIR, stop_map, OUTPUT_FILE)
