#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
long_detour_compressibility.py
==============================
Check if LONG_DETOUR segments can be compressed by fixing connections/allows

Outputs:
- passenger_path_len vs bus_path_len
- gap = bus_path_len - passenger_path_len
- If gap > 0: bus has additional restrictions vs passenger
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sumolib
except ImportError:
    print("[ERROR] sumolib not installed")
    sys.exit(1)

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v1.net.xml"
RECLASSIFY_CSV = PROJECT_ROOT / "logs" / "v31_rule_blocked_reclassify.csv"
OUTPUT_CSV = PROJECT_ROOT / "logs" / "long_detour_compressibility.csv"

# 4 LONG_DETOUR segments to analyze
LONG_DETOURS = [
    ("68X", "inbound", "1→2", "105735_rev", "105653_rev"),
    ("68X", "inbound", "2→3", "105653_rev", "106831_rev"),
    ("68X", "outbound", "6→7", "97407", "96273_rev"),
    ("960", "inbound", "9→10", "4833", "261296"),
]


def get_path_info(net, from_edge_id, to_edge_id, vclass=None):
    """Get path length for given vClass"""
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            return sum(e.getLength() for e in route), [e.getID() for e in route]
        return None, []
    except:
        return None, []


def main():
    print("="*80)
    print("[LONG_DETOUR] Compressibility Analysis")
    print("="*80)
    
    print(f"\n[LOAD] Network: {NET_FILE.name}")
    net = sumolib.net.readNet(str(NET_FILE))
    
    print(f"[LOAD] Reclassify data: {RECLASSIFY_CSV.name}")
    df = pd.read_csv(RECLASSIFY_CSV)
    
    long_detours = df[df['true_category'] == 'LONG_DETOUR']
    print(f"\n[INFO] Found {len(long_detours)} LONG_DETOUR segments")
    
    results = []
    for _, row in long_detours.iterrows():
        fe, te = row['from_edge'], row['to_edge']
        kmb_len = row['kmb_len']
        bus_path_len = row['bus_path_len']
        
        print(f"\n[ANALYZE] {row['route']} {row['bound']} {row['seq']}")
        print(f"  {fe} -> {te}")
        print(f"  KMB len: {kmb_len:.0f}m")
        print(f"  Bus path: {bus_path_len:.0f}m (ratio={bus_path_len/kmb_len:.2f})")
        
        # Get passenger path (should be same since we're using default shortest path)
        passenger_len, passenger_path = get_path_info(net, fe, te)
        
        if passenger_len:
            gap = bus_path_len - passenger_len
            print(f"  Passenger path: {passenger_len:.0f}m")
            print(f"  Gap (bus - passenger): {gap:.0f}m")
            
            if abs(gap) < 1:
                verdict = "NO_BUS_PENALTY"
                fix = "Issue is network structure, not bus restrictions"
            elif gap > 100:
                verdict = "BUS_RESTRICTED"
                fix = "Check bus allow/disallow on path edges"
            else:
                verdict = "MINIMAL_DIFFERENCE"
                fix = "VIA_ROUTING acceptable"
            
            results.append({
                'route': row['route'],
                'bound': row['bound'],
                'seq': row['seq'],
                'from_edge': fe,
                'to_edge': te,
                'kmb_len': kmb_len,
                'bus_path_len': bus_path_len,
                'passenger_path_len': passenger_len,
                'gap': gap,
                'bus_ratio': bus_path_len / kmb_len,
                'passenger_ratio': passenger_len / kmb_len,
                'verdict': verdict,
                'fix': fix
            })
            
            print(f"  Verdict: {verdict}")
        else:
            print(f"  [WARN] Could not get passenger path")
    
    # Save
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[SAVED] {OUTPUT_CSV}")
        
        # Summary
        print("\n" + "="*80)
        print("[SUMMARY]")
        print("="*80)
        
        for verdict in results_df['verdict'].unique():
            count = len(results_df[results_df['verdict'] == verdict])
            print(f"  {verdict}: {count}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
