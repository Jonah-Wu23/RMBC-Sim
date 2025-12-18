import os
import pandas as pd

def generate_sumo_stops():
    # Input/Output paths
    input_csv = r"d:\Documents\Bus Project\Sorce code\data\processed\kmb_route_stop_dist.csv"
    output_xml = r"d:\Documents\Bus Project\Sorce code\sumo\additional\bus_stops.add.xml"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_xml), exist_ok=True)
    
    # Read Stop Data
    # Expected cols: route,bound,service_type,seq,stop_id,name_en,name_tc,lat,long,cum_dist_m,link_dist_m
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Please ensure Data Agent has run 'clean_kmb_shapes.py'.")
        return

    print(f"Loaded {len(df)} stops.")
    
    # Generate XML Content
    xml_content = ['<additional>']
    
    # We need unique stops. A stop ID might appear multiple times for different routes.
    # We should create one <busStop> per unique stop_id (or stop_id + lane combination).
    # For Week 1, we will assume a simple mapping or just dump all points.
    # Since we don't have the .net.xml loaded to find lanes, we will use a PLACEHOLDER lane 
    # or rely on a subsequent step to map them. 
    # HOWEVER, the prompt asked to "Map each stop to the nearest edge". 
    # Without traci or sumolib AND the net file, we can't do this accurate geometry check here easily 
    # unless we use `sumolib` if available.
    
    # Let's try to import sumolib to see if we can do it properly, otherwise use dummy lanes.
    try:
        import sumolib
        net_file = r"d:\Documents\Bus Project\Sorce code\sumo\net\hk_baseline.net.xml"
        if os.path.exists(net_file):
            net = sumolib.net.readNet(net_file)
            print("Loaded SUMO network for mapping.")
            has_net = True
        else:
            print("SUMO network not found. Using placeholder lanes.")
            has_net = False
    except ImportError:
        print("sumolib not found. Using placeholder lanes.")
        has_net = False

    unique_stops = df.drop_duplicates(subset=['stop_id'])
    
    for _, row in unique_stops.iterrows():
        stop_id = row['stop_id']
        name = row.get('name_en', 'Unknown')
        lat = row['lat']
        lon = row['long']
        
        lane_id = "UNKNOWN_LANE"
        pos = 0
        
        if has_net:
            # Map (lon, lat) to nearest lane
            # Note: net.convertLonLat2XY returns (x, y)
            x, y = net.convertLonLat2XY(lon, lat)
            edges = net.getNeighboringEdges(x, y, 100) # Search within 100m
            if edges:
                # edges is list of (edge, dist)
                # pick closest
                closest_edge, dist = edges[0]
                # get a lane (usually index 0, bus stops often on outer lane)
                lane_id = closest_edge.getLanes()[0].getID()
                # project point to lane for pos
                pos = sumolib.geomhelper.polygonOffsetWithMinimumDistanceToPoint((x,y), closest_edge.getShape())
            else:
                print(f"Warning: Could not map stop {stop_id} ({name}) to network.")
        
        # Write busStop entry ONLY if mapped
        if lane_id != "UNKNOWN_LANE":
            xml_content.append(f'    <busStop id="{stop_id}" name="{name}" lane="{lane_id}" startPos="{pos}" endPos="{pos+15}" friendlyPos="true" lines="{row["route"]}"/>')
        else:
            print(f"Skipping unmapped stop {stop_id}")

    xml_content.append('</additional>')
    
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_content))
    
    print(f"Generated {output_xml} with {len(unique_stops)} stops.")

if __name__ == "__main__":
    generate_sumo_stops()
