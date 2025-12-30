"""
Find Critical Threshold T* and Hardest 15-min Window for P14 Robustness Test.
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

# Import evaluation logic if possible, or reimplement KS calc
# Reimplementing KS local logic for simplicity
def calculate_ks_distance(real_values, sim_values):
    if len(real_values) < 5 or len(sim_values) < 5:
        return 999.0, 0.0 # Invalid
    ks_stat, p_value = stats.ks_2samp(real_values, sim_values)
    return ks_stat, p_value

SIM_STOPINFO = "sumo/output/offpeak_v2_offpeak_stopinfo.xml"
REAL_LINKS_AGG = "data2/processed/link_stats_offpeak.csv"
REAL_LINKS_RAW = "data2/processed/link_times_offpeak.csv"

def load_sim_speeds():
    # Load Sim speeds once (stationary assumption)
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(SIM_STOPINFO)
        df_stops = []
        for stop in tree.getroot().findall('.//stopinfo'):
             df_stops.append({
                'vehicle_id': stop.get('id'),
                'stop_id': stop.get('busStop'),
                'arrival': float(stop.get('started', 0)),
                'departure': float(stop.get('ended', 0))
            })
        df = pd.DataFrame(df_stops)
        
        # Calc link times same as evaluate_robustness.py
        results = []
        for veh_id, veh_data in df.groupby('vehicle_id'):
            veh_data = veh_data.sort_values('arrival').reset_index(drop=True)
            for i in range(len(veh_data) - 1):
                departure = veh_data.loc[i, 'departure']
                arrival = veh_data.loc[i+1, 'arrival']
                tt = arrival - departure
                if tt > 0:
                    # Speed approx (500m / tt)
                    speed_kmh = (500 / 1000) / (tt / 3600)
                    if 2 < speed_kmh < 100:
                        results.append(speed_kmh)
        return np.array(results)
    except Exception as e:
        print(f"Error loading sim: {e}")
        return np.array([])

def apply_rule_c(df_stats, threshold_s):
    # Rule C: Time > T & Speed < 5 & Dist < 1500
    cond_ghost = (df_stats['tt_median'] > threshold_s) & \
                 (df_stats['speed_median'] < 5.0) & \
                 (df_stats['dist_m'] < 1500)
    return df_stats[~cond_ghost].copy()

def search_critical_threshold(sim_speeds):
    print("\n=== 1. Searching for Critical Threshold T* ===")
    df_raw_agg = pd.read_csv(REAL_LINKS_AGG)
    
    thresholds = list(range(300, 601, 25))
    results = []
    
    for t in thresholds:
        clean_df = apply_rule_c(df_raw_agg, t)
        real_speeds = clean_df['speed_median'].dropna().values
        
        ks, p = calculate_ks_distance(real_speeds, sim_speeds)
        
        # Check condition
        status = "FAIL" if ks > 0.35 else "PASS"
        results.append({'T': t, 'KS': ks, 'N': len(real_speeds), 'Status': status})
        print(f"T={t:3d}s | KS={ks:.4f} | N={len(real_speeds)} | {status}")
    
    # Find Best T (Max KS <= 0.35)
    valid_results = [r for r in results if r['KS'] <= 0.35]
    if valid_results:
        best = max(valid_results, key=lambda x: x['KS'])
        print(f"\n>>> FOUND T* = {best['T']}s (KS={best['KS']:.4f})")
        return best['T']
    else:
        print("\n>>> NO Valid T Found in range!")
        return 300 # Fallback

def analyze_subwindows(t_critical, sim_speeds):
    print(f"\n=== 2. Sub-Window Analysis (Rule C: T > {t_critical}s) ===")
    
    if not os.path.exists(REAL_LINKS_RAW):
        print("Raw link times file not found.")
        return

    df_raw = pd.read_csv(REAL_LINKS_RAW)
    # Ensure datetime
    if 'departure_ts' in df_raw.columns:
        df_raw['ts'] = pd.to_datetime(df_raw['departure_ts'])
    else:
        # Fallback if capture_ts or similar
        print("Error: No timestamp column found")
        return

    # Define windows (15:00 is start. Data might have different date but time is key)
    # Assuming data is from single hour.
    min_ts = df_raw['ts'].min()
    windows = []
    for i in range(4):
        start = min_ts + pd.Timedelta(minutes=i*15)
        end = start + pd.Timedelta(minutes=15)
        windows.append((i+1, start, end))
    
    best_window = None
    max_ks = -1.0
    
    for idx, start, end in windows:
        # Filter raw data by time
        df_win = df_raw[(df_raw['ts'] >= start) & (df_raw['ts'] < end)]
        
        if df_win.empty:
            print(f"Window {idx}: No data")
            continue
            
        # Aggregate stats for this window
        stats_win = df_win.groupby(['route', 'bound', 'from_seq', 'to_seq']).agg({
            'travel_time_s': 'median',
            'speed_kmh': 'median',
            'dist_m': 'first'
        }).reset_index()
        stats_win.rename(columns={'travel_time_s': 'tt_median', 'speed_kmh': 'speed_median'}, inplace=True)
        
        # Apply Rule C
        clean_win = apply_rule_c(stats_win, t_critical)
        real_speeds = clean_win['speed_median'].dropna().values
        
        # Calc KS
        ks, _ = calculate_ks_distance(real_speeds, sim_speeds)
        
        status = "FAIL" if ks > 0.35 else "PASS"
        print(f"Window {idx} ({start.time()}-{end.time()}): N_raw={len(stats_win)}, N_clean={len(clean_win)} | KS={ks:.4f} | {status}")
        
        if ks <= 0.35 and ks > max_ks:
            max_ks = ks
            best_window = {'Window': idx, 'KS': ks, 'N': len(clean_win)}

    if best_window:
        print(f"\n>>> HARDEST WINDOW: Window {best_window['Window']} (KS={best_window['KS']:.4f})")
    else:
        print("\n>>> No passing window found.")

def main():
    sim_speeds = load_sim_speeds()
    if len(sim_speeds) == 0:
        print("Sim data empty")
        return
        
    t_star = search_critical_threshold(sim_speeds)
    analyze_subwindows(t_star, sim_speeds)

if __name__ == "__main__":
    main()
