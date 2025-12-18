import os
import pandas as pd
import xml.etree.ElementTree as ET

def generate_sumo_routes():
    # Paths
    stops_csv = r"d:\Documents\Bus Project\Sorce code\data\processed\kmb_route_stop_dist.csv"
    stops_xml = r"d:\Documents\Bus Project\Sorce code\sumo\additional\bus_stops.add.xml"
    output_rou = r"d:\Documents\Bus Project\Sorce code\sumo\routes\baseline.rou.xml"
    
    os.makedirs(os.path.dirname(output_rou), exist_ok=True)
    
    # 1. Parse bus_stops.add.xml to get stop_id -> lane mapping
    stop_lane_map = {}
    try:
        tree = ET.parse(stops_xml)
        root = tree.getroot()
        for stop in root.findall('busStop'):
            s_id = stop.get('id')
            lane = stop.get('lane')
            if lane and lane != "UNKNOWN_LANE":
                stop_lane_map[s_id] = lane
    except FileNotFoundError:
        print(f"Error: {stops_xml} not found.")
        return

    # 2. Read Stop Sequence from CSV
    try:
        df = pd.read_csv(stops_csv)
    except FileNotFoundError:
        print(f"Error: {stops_csv} not found.")
        return
        
    # Filter for 68X (inbound) as primary case
    # Assuming 'inbound' corresponds to 'I' or 'O' in KM data, usually 'I' is Inbound to Kowloon.
    # Check CSV content structure (from previous view): cols include route,bound(inbound/outbound),seq,stop_id...
    route_data = df[(df['route'] == '68X') & (df['bound'] == 'inbound')].sort_values('seq')
    
    if route_data.empty:
        print("No route data found for 68X inbound.")
        return

    # 3. Build Route Edges
    edges = []
    valid_stops = []
    
    for _, row in route_data.iterrows():
        stop_id = row['stop_id']
        if stop_id in stop_lane_map:
            lane = stop_lane_map[stop_id]
            edge = lane.rpartition('_')[0] # remove lane index e.g. edge_0 -> edge
            
            # Check for gap filling
            if edges:
                prev_edge = edges[-1]
                # Assuming naming edge_inbound_X or edge_outbound_X
                # Let's try to extract number
                try:
                    prev_parts = prev_edge.rsplit('_', 1)
                    curr_parts = edge.rsplit('_', 1)
                    if prev_parts[0] == curr_parts[0]: # Same prefix (e.g. edge_inbound)
                        prev_idx = int(prev_parts[1])
                        curr_idx = int(curr_parts[1])
                        if curr_idx > prev_idx + 1:
                            # Fill gaps
                            for i in range(prev_idx + 1, curr_idx):
                                edges.append(f"{prev_parts[0]}_{i}")
                except ValueError:
                    pass # Not numeric or standard pattern, ignore
            
            if not edges or edges[-1] != edge:
                edges.append(edge)
            valid_stops.append(stop_id)
        else:
            print(f"Skipping stop {stop_id} (not mapped to network).")

    if not edges:
        print("No valid edges found for route.")
        return

    route_edges_str = " ".join(edges)
    
    # 4. Generate Routes XML
    with open(output_rou, 'w', encoding='utf-8') as f:
        f.write('<routes>\n')
        f.write('    <vType id="bus_kmb" vClass="bus" accel="2.5" decel="4.5" sigma="0.5" length="12" minGap="2.5" maxSpeed="20.0" color="1,0,0"/>\n')
        f.write(f'    <route id="route_68X_inbound" edges="{route_edges_str}"/>\n')
        
        # Create a single flow for testing
        # 5 buses/hour roughly for testing
        f.write('    <flow id="flow_68X" type="bus_kmb" route="route_68X_inbound" begin="0" end="3600" period="600" line="68X">\n')
        
        # Add stops
        for s_id in valid_stops:
            f.write(f'        <stop busStop="{s_id}" duration="20"/>\n')
            
        f.write('    </flow>\n')
        f.write('</routes>\n')

    print(f"Generated {output_rou} for 68X Inbound with {len(edges)} edges and {len(valid_stops)} stops.")

if __name__ == "__main__":
    generate_sumo_routes()
