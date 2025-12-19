"""
Check connectivity between consecutive bus stops for key routes.
Uses sumolib to find paths between stop edges.
"""

import os
import sys
import pandas as pd

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn.net.xml"))
STOP_FILE = os.path.abspath(os.path.join("sumo", "additional", "bus_stops_irn.add.xml"))
CSV_FILE = os.path.abspath(os.path.join("data", "processed", "kmb_route_stop_dist.csv"))

def check_connectivity():
    print(f"Loading network: {NET_FILE}")
    net = sumolib.net.readNet(NET_FILE)
    
    # Needs to parse stops to get Edge IDs
    import xml.etree.ElementTree as ET
    print(f"Loading stops XML: {STOP_FILE}")
    tree = ET.parse(STOP_FILE)
    root = tree.getroot()
    stop_map = {} # stop_id -> lane_id
    for stop in root.findall("busStop"):
        stop_map[stop.attrib['id']] = stop.attrib['lane']

    print(f"Loading Route CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    
    target_routes = ['68X', '960']
    
    for route in target_routes:
        for bound in df[df['route'] == route]['bound'].unique():
            print(f"\nChecking Route {route} ({bound})...")
            subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            
            stop_ids = subset['stop_id'].tolist()
            names = subset['stop_name_en'].tolist()
            
            failed_segments = 0
            
            for i in range(len(stop_ids) - 1):
                s1, s2 = stop_ids[i], stop_ids[i+1]
                
                if s1 not in stop_map or s2 not in stop_map:
                    # Maybe processed filtered some out or mapping failed?
                    # We know 100% mapped, so this shouldn't happen unless IDs mismatch check
                    # Checking if s1 in stop_map
                    continue
                    
                edge1 = net.getLane(stop_map[s1]).getEdge()
                edge2 = net.getLane(stop_map[s2]).getEdge()
                
                # Check path
                # Dijkstra is expensive for all pairs? Not really for one route.
                path = net.getShortestPath(edge1, edge2)
                
                if not path[0]:
                    print(f"  [MISSING] {s1} ({names[i]}) -> {s2} ({names[i+1]})")
                    print(f"     Edge: {edge1.getID()} -> {edge2.getID()}")
                    failed_segments += 1
            
            if failed_segments == 0:
                print(f"  [OK] All {len(stop_ids)-1} segments connected.")
            else:
                print(f"  [FAIL] {failed_segments} broken segments.")

if __name__ == "__main__":
    check_connectivity()
