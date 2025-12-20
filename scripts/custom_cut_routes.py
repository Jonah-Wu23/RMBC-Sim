import xml.etree.ElementTree as ET
import os

# Files
NET_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\net\hk_cropped.net.xml"
INPUT_ROUTES = r"d:\Documents\Bus Project\Sorce code\sumo\routes\fixed_routes.rou.xml"
OUTPUT_ROUTES = r"d:\Documents\Bus Project\Sorce code\sumo\routes\fixed_routes_cropped.rou.xml"
STOPS_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\additional\bus_stops.add.xml"

def get_valid_edges(net_file):
    print(f"Reading network: {net_file}")
    valid_edges = set()
    valid_lanes = set()
    context = ET.iterparse(net_file, events=("start", "end"))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in context:
        if event == "end" and elem.tag == "edge":
            eid = elem.get("id")
            # Skip internal edges if needed, but keeping them is safer
            valid_edges.add(eid)
            
            # Also track lanes for stop validation
            for lane in elem.findall("lane"):
                valid_lanes.add(lane.get("id"))
                
            root.clear()
            
    print(f"Loaded {len(valid_edges)} valid edges and {len(valid_lanes)} valid lanes.")
    return valid_edges, valid_lanes

def get_valid_stops(stops_file, valid_lanes):
    print(f"Reading stops: {stops_file}")
    valid_stop_ids = set()
    tree = ET.parse(stops_file)
    root = tree.getroot()
    
    for stop in root.findall("busStop"):
        lane = stop.get("lane")
        if lane in valid_lanes:
            valid_stop_ids.add(stop.get("id"))
            
    print(f"Identified {len(valid_stop_ids)} valid bus stops within cropped area.")
    return valid_stop_ids

def clean_routes():
    valid_edges, valid_lanes = get_valid_edges(NET_FILE)
    valid_stops = get_valid_stops(STOPS_FILE, valid_lanes)
    
    print(f"Processing routes from: {INPUT_ROUTES}")
    tree = ET.parse(INPUT_ROUTES)
    root = tree.getroot()
    
    items_kept = 0
    items_removed = 0
    
    # Iterate over vType, vehicle, flow, route
    # We want to keep vTypes always? Yes.
    
    # We need to filter <route> elements inside <routes> or encoded in <vehicle>/<flow>
    
    # Using list() to allow modification during iteration
    for elem in list(root):
        if elem.tag == "vType":
            continue # Keep vTypes
            
        if elem.tag in ["vehicle", "flow", "route"]:
            # Check edges
            # Edges can be a child element <route edges="..."> or attribute 'edges' if it's a route tag
            
            route_node = None
            if elem.tag == "route":
                route_node = elem
            else:
                route_node = elem.find("route")
                
            if route_node is not None:
                edges_str = route_node.get("edges", "")
                edges_list = edges_str.split()
                if not edges_list:
                    # Maybe stored as text content? Not standard SUMO but possible
                    pass
                
                # Filter edges
                new_edges = [e for e in edges_list if e in valid_edges]
                
                # If filtered route is too short or empty/disconnected, handle it.
                # For fixed routes (bus lines), if we cut the middle, we might just want the longest segment?
                # Or simply keep the sequence. SUMO might complain about gaps (teleport).
                # But at least files will be valid.
                
                if len(new_edges) < 2:
                    # Remove this vehicle/flow entirely if it has no path
                    root.remove(elem)
                    items_removed += 1
                    continue
                    
                # Update edges
                route_node.set("edges", " ".join(new_edges))
                
                # Check Stops
                # Stops are children of vehicle/flow usually, or route? 
                # "stops can be child of vehicle, flow, route or ply"
                # Let's check both elem and route_node
                
                for parent in [elem, route_node]:
                    for stop in list(parent.findall("stop")):
                        stop_id = stop.get("busStop")
                        lane_id = stop.get("lane")
                        
                        is_valid = False
                        if stop_id and stop_id in valid_stops:
                            is_valid = True
                        elif lane_id and lane_id in valid_lanes:
                            is_valid = True
                        elif stop.get("edge") and stop.get("edge") in valid_edges:
                             is_valid = True
                             
                        if not is_valid:
                            parent.remove(stop)
                            
                items_kept += 1
            else:
                # Flow without route? Maybe detached route?
                # If it references a route ID, we assume that route is checked separately.
                pass

    print(f"Processed. Kept {items_kept} items, Removed {items_removed} items.")
    
    print(f"Writing to: {OUTPUT_ROUTES}")
    tree.write(OUTPUT_ROUTES, encoding="UTF-8", xml_declaration=True)
    print("Done.")

if __name__ == "__main__":
    clean_routes()
