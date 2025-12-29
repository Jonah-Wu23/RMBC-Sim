#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_stop_mapping.py
===================
V2: Fix STOP_NOT_MAPPED issues

For stops not in bus_stops.add.xml, try to find nearest edge with larger radius
"""

import sys
import math
import xml.etree.ElementTree as ET
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
BUS_STOPS_FILE = PROJECT_ROOT / "sumo" / "additional" / "bus_stops.add.xml"
BUS_STOPS_IRN_FILE = PROJECT_ROOT / "sumo" / "additional" / "bus_stops_irn.add.xml"
KMB_CSV = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
OUTPUT_FILE = PROJECT_ROOT / "sumo" / "additional" / "bus_stops_irn_fixed.add.xml"
OVERRIDE_CSV = PROJECT_ROOT / "logs" / "stop_mapping_override.csv"

# STOP_NOT_MAPPED segments (route, bound, seq)
UNMAPPED_SEGMENTS = [
    ("68X", "outbound", 17),  # 16->17 to_edge is UNKNOWN
    ("960", "inbound", 3),    # 2->3 to_edge is UNKNOWN
    ("960", "inbound", 12),   # 11->12 to_edge is UNKNOWN
]

def main():
    print("="*80)
    print("[V2] Stop Mapping Fix")
    print("="*80)
    
    # Load data
    print(f"\n[LOAD] Network: {NET_FILE.name}")
    net = sumolib.net.readNet(str(NET_FILE))
    
    print(f"[LOAD] KMB stop data: {KMB_CSV.name}")
    kmb = pd.read_csv(KMB_CSV)
    
    print(f"[LOAD] Existing stops: {BUS_STOPS_FILE.name}")
    tree = ET.parse(str(BUS_STOPS_FILE))
    existing_stops = {s.get('id'): s for s in tree.getroot().findall('.//busStop')}
    
    # Also load IRN version
    tree_irn = ET.parse(str(BUS_STOPS_IRN_FILE))
    irn_stops = {s.get('id'): s for s in tree_irn.getroot().findall('.//busStop')}
    
    print(f"   Existing stops: {len(existing_stops)} (add.xml), {len(irn_stops)} (irn.add.xml)")
    
    # Analyze STOP_NOT_MAPPED stops
    print("\n" + "="*80)
    print("Analyzing STOP_NOT_MAPPED stops")
    print("="*80)
    
    overrides = []
    new_stops = []
    
    for route, bound, seq in UNMAPPED_SEGMENTS:
        stop_row = kmb[(kmb['route'] == route) & (kmb['bound'] == bound) & (kmb['seq'] == seq)]
        if len(stop_row) == 0:
            print(f"\n[ERROR] Cannot find {route} {bound} seq={seq}")
            continue
        
        stop = stop_row.iloc[0]
        stop_id = stop['stop_id']
        lat, lon = stop['lat'], stop['long']
        name = stop['stop_name_en']
        
        print(f"\n[SEARCH] {route} {bound} seq={seq}: {stop_id}")
        print(f"   Name: {name}")
        print(f"   Coords: ({lat:.6f}, {lon:.6f})")
        
        # Check if in bus_stops.add.xml
        if stop_id in existing_stops:
            elem = existing_stops[stop_id]
            lane = elem.get('lane')
            print(f"   [OK] Exists in bus_stops.add.xml: lane={lane}")
            continue
        
        # Check if in bus_stops_irn.add.xml
        if stop_id in irn_stops:
            elem = irn_stops[stop_id]
            lane = elem.get('lane')
            print(f"   [OK] Exists in bus_stops_irn.add.xml: lane={lane}")
            continue
        
        print(f"   [MISS] Stop not in any add.xml")
        
        # Try to find nearest edge
        x, y = net.convertLonLat2XY(lon, lat)
        print(f"   Converted coords: ({x:.2f}, {y:.2f})")
        
        for radius in [50, 100, 200, 500]:
            edges = net.getNeighboringEdges(x, y, radius)
            if edges:
                # Filter out internal edges
                edges = [(e, d) for e, d in edges if not e.getID().startswith(':')]
                if edges:
                    edge, dist = min(edges, key=lambda x: x[1])
                    lane = f"{edge.getID()}_0"
                    print(f"   [FOUND] Nearest edge (r={radius}m): {edge.getID()} dist={dist:.1f}m")
                    
                    # Record override
                    overrides.append({
                        'route': route,
                        'bound': bound,
                        'seq': seq,
                        'stop_id': stop_id,
                        'stop_name': name,
                        'lane': lane,
                        'startPos': 0,
                        'endPos': min(20, edge.getLength()),
                        'distance': dist
                    })
                    
                    # Generate new busStop element
                    new_stops.append({
                        'id': stop_id,
                        'name': name,
                        'lane': lane,
                        'startPos': "0.00",
                        'endPos': f"{min(20, edge.getLength()):.2f}"
                    })
                    break
        else:
            print(f"   [FAIL] No edge found within 500m")
    
    # Save override CSV
    if overrides:
        override_df = pd.DataFrame(overrides)
        override_df.to_csv(OVERRIDE_CSV, index=False)
        print(f"\n[SAVED] Override CSV: {OVERRIDE_CSV}")
        print(override_df.to_string())
    
    # Generate fixed add.xml
    if new_stops:
        print(f"\n[GEN] Creating new stop definitions...")
        
        # Copy IRN version and add new stops
        root = tree_irn.getroot()
        for stop in new_stops:
            elem = ET.SubElement(root, 'busStop')
            elem.set('id', stop['id'])
            elem.set('name', stop['name'])
            elem.set('lane', stop['lane'])
            elem.set('startPos', stop['startPos'])
            elem.set('endPos', stop['endPos'])
            elem.set('friendlyPos', 'true')
        
        # Save
        tree_irn.write(str(OUTPUT_FILE), encoding='unicode', xml_declaration=True)
        print(f"[SAVED] Fixed stops file: {OUTPUT_FILE}")
    else:
        print("\n[INFO] No new stops to add")
    
    print("\n" + "="*80)
    print("[DONE]")
    print("="*80)

if __name__ == "__main__":
    main()
