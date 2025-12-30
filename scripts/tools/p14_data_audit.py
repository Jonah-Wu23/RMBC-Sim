"""
P14 Data Audit Script
Analyze the source of low speed records in off-peak data and compare dwell times.
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os

REAL_LINKS = "data2/processed/link_stats_offpeak.csv"
REAL_ETA = "data2/processed/station_eta_offpeak.csv"
SIM_STOPINFO = "sumo/output/offpeak_v2_offpeak_stopinfo.xml" # v2 output

def analyze_slow_links():
    print("\n=== 1. Slow Link Analysis (Real Data) ===")
    if not os.path.exists(REAL_LINKS):
        print("Real link stats file not found.")
        return

    df = pd.read_csv(REAL_LINKS)
    print(f"Loaded {len(df)} records. Columns: {df.columns.tolist()}")
    total = len(df)
    
    # Speed thresholds
    slow_5 = df[df['speed_median'] < 5].copy()
    slow_3 = df[df['speed_median'] < 3].copy()
    
    print(f"Total links: {total}")
    print(f"Speed < 5 km/h: {len(slow_5)} ({len(slow_5)/total:.1%})")
    print(f"Speed < 3 km/h: {len(slow_3)} ({len(slow_3)/total:.1%})")
    
    if not slow_5.empty:
        # Check distribution
        print("\n--- Slow Link Characteristics (< 5km/h) ---")
        short_links = slow_5[slow_5['dist_m'] < 200]
        long_links = slow_5[slow_5['dist_m'] > 1000]
        print(f"Short links (<200m): {len(short_links)} ({len(short_links)/len(slow_5):.1%})")
        print(f"Long links (>1000m): {len(long_links)} ({len(long_links)/len(slow_5):.1%})")
        
        print("\n--- Top 10 Slowest Links ---")
        top_slow = slow_5.sort_values('speed_median').head(10)
        for _, row in top_slow.iterrows():
            print(f"Route {row.get('route', '?')} ({row.get('from_seq', '?')}->{row.get('to_seq', '?')}): "
                  f"Dist={row.get('dist_m', 0):.1f}m, "
                  f"Time={row.get('tt_median', 0):.1f}s, "
                  f"Speed={row.get('speed_median', 0):.2f}km/h")

def analyze_dwell_times():
    print("\n=== 2. Dwell Time Analysis (Real vs Sim) ===")
    
    # --- Real Dwell ---
    # Since we cannot easily compute real dwell from current clean files, we skip real dwell precise calc for now.
    # We focus on Sim dwell to see if it is reasonable (e.g. 10-20s).
    
    # --- Sim Dwell ---
    if os.path.exists(SIM_STOPINFO):
        try:
            tree = ET.parse(SIM_STOPINFO)
            root = tree.getroot()
            sim_dwells = []
            for stop in root.findall(".//stopinfo"):
                val = stop.get('duration')
                if val is not None:
                    sim_dwells.append(float(val))
            
            if sim_dwells:
                print(f"\nSimulation Dwell Stats (v2):")
                print(f"  Count: {len(sim_dwells)}")
                print(f"  Mean: {np.mean(sim_dwells):.2f}s")
                print(f"  Median: {np.median(sim_dwells):.2f}s")
                print(f"  P90: {np.percentile(sim_dwells, 90):.2f}s")
                print(f"  Max: {np.max(sim_dwells):.2f}s")
                
                long_dwells = [d for d in sim_dwells if d > 60]
                print(f"  Dwells > 60s: {len(long_dwells)} ({len(long_dwells)/len(sim_dwells):.1%})")
            else:
                print("No stopinfo records found in XML.")
        except Exception as e:
            print(f"Error parse sim xml: {e}")
    else:
        print("Sim stopinfo file not found.")

def main():
    analyze_slow_links()
    analyze_dwell_times()

if __name__ == "__main__":
    main()
