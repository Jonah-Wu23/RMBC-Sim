"""
Generate SUMO bus stops on the new IRN network.
Matches stops from `data/processed/kmb_route_stop_dist.csv` to `sumo/net/hk_irn.net.xml`.
"""

import os
import sys
import pandas as pd
# Add SUMO_HOME/tools to path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

import pyproj

NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn.net.xml"))
STOP_FILE = os.path.abspath(os.path.join("data", "processed", "kmb_route_stop_dist.csv"))
OUTPUT_FILE = os.path.abspath(os.path.join("sumo", "additional", "bus_stops.add.xml"))

def generate_stops():
    print(f"Loading network: {NET_FILE}")
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"Failed to load network: {e}")
        return

    # Transformer: WGS84 (4326) -> HK1980 Grid (2326)
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:2326", always_xy=True)

    print(f"Loading stops: {STOP_FILE}")
    df_stops = pd.read_csv(STOP_FILE)
    
    # Aggregate routes per stop
    # df_stops has 'route' and 'stop_id'
    stop_routes = df_stops.groupby('stop_id')['route'].apply(lambda x: " ".join(sorted(set(x.astype(str))))).to_dict()
    
    unique_stops = df_stops.groupby(['stop_id', 'stop_name_en']).first().reset_index()
    print(f"Processing {len(unique_stops)} unique stops...")
    
    stops_xml = []
    mapped_count = 0
    
    for _, row in unique_stops.iterrows():
        stop_id = str(row['stop_id'])
        name = row['stop_name_en']
        
        # CSV is WGS84 (lon, lat)
        lon, lat = row['long'], row['lat']
        
        # Transform to EPSG:2326 (HK Grid)
        x, y = transformer.transform(lon, lat)
        
        # Find nearest edge
        # radius=50m to start
        lanes = net.getNeighboringLanes(x, y, r=50)
        
        if not lanes:
            print(f"Warning: No lane found for stop {stop_id} ({name}) within 50m.")
            continue
            
        # Select best lane
        # Heuristic: Closest distance
        best_lane, dist = lanes[0] 
        # dist is distance to the lane shape
        
        # Create busStop element
        # startPos: projected position on lane
        # We need the offset along the lane.
        # getNeighboringLanes returns (lane, dist). 
        # We need to calculate pos on lane.
        
        # Finding the position on the lane
        # project(x,y) returns offset from start
        lane_shape = best_lane.getShape()
        pos_on_lane = sumolib.geomhelper.polygonOffsetWithMinimumDistanceToPoint((x, y), lane_shape)
        
        # Ensure legal positions
        lane_len = best_lane.getLength()
        start_pos = max(0, pos_on_lane - 10) # 10m length stop
        end_pos = min(lane_len, pos_on_lane + 10)
        
        if end_pos - start_pos < 10:
             # Adjust to fit
             if start_pos > 10: start_pos = end_pos - 20
             else: end_pos = start_pos + 20
             
        lines_str = stop_routes.get(stop_id, "")
             
        # Format XML
        stop_elem = (f'    <busStop id="{stop_id}" name="{name}" lane="{best_lane.getID()}" '
                     f'startPos="{start_pos:.2f}" endPos="{end_pos:.2f}" lines="{lines_str}" friendlyPos="true"/>')
        stops_xml.append(stop_elem)
        mapped_count += 1
        
    print(f"Mapped {mapped_count}/{len(unique_stops)} stops.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('<additional>\n')
        f.write('\n'.join(stops_xml))
        f.write('\n</additional>')
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_stops()
