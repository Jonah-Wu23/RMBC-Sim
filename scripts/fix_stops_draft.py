import xml.etree.ElementTree as ET
import os
import math

def fix_invalid_stops():
    NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn_v2.net.xml"))
    STOPS_FILE = os.path.abspath(os.path.join("sumo", "additional", "bus_stops.add.xml"))
    OUTPUT_STOPS = os.path.abspath(os.path.join("sumo", "additional", "bus_stops_fixed.add.xml"))
    
    if not os.path.exists(NET_FILE) or not os.path.exists(STOPS_FILE):
        print("Files not found.")
        return

    print("Loading edges and shapes from net file...")
    edges = {}
    context = ET.iterparse(NET_FILE, events=('end',))
    for event, elem in context:
        if elem.tag == 'edge':
            # Simplified: just take the first lane's shape
            lane = elem.find('lane')
            if lane is not None:
                shape_str = lane.get('shape')
                # Parse start coord just for simple distance check
                # Shape format: "x1,y1 x2,y2 ..."
                start_coord = shape_str.split()[0].split(',')
                edges[elem.get('id')] = (float(start_coord[0]), float(start_coord[1]))
            elem.clear()
            
    print(f"Loaded {len(edges)} valid edges.")
    
    tree = ET.parse(STOPS_FILE)
    root = tree.getroot()
    
    fixed_count = 0
    
    for stop in root.findall('busStop'):
        original_lane = stop.get('lane')
        edge_id = original_lane.split('_')[0]
        
        if edge_id not in edges:
            print(f"Fixing Stop {stop.get('id')} on missing edge {edge_id}...")
            # Ideally we use the stop's independent coordinate, but busStop doesn't always have x/y if it relies on lane.
            # Assuming busStop usually has NO x/y in this file, we can't easily find "nearest".
            # BUT, we might know the coordinate from the original INVALID edge if we had it? No.
            # Workaround: Parsing explicit x/y from CSV if available? 
            # Or just assuming we can drop it? No, must fix.
            # Let's check if the busStop element HAS x/y/lon/lat attributes? The user's stops usually map to lane.
            
            # Since I don't have the coordinates easily loaded here, I will try a simple heuristic:
            # If edge '96273' is missing, maybe '96273_rev' exists? Or edges nearby?
            # Actually, the user's data `kmb_route_stop_dist.csv` has coordinates.
            pass
            
    # Re-approach: Read CSV to get coords of all stops, then find nearest valid edge in NET.
    
if __name__ == "__main__":
    pass
