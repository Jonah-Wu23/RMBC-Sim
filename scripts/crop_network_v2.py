import os
import subprocess
import xml.etree.ElementTree as ET
import sys

# --- Configuration ---
SUMO_HOME = os.environ.get('SUMO_HOME', r"C:\Program Files (x86)\Eclipse\Sumo")
NETCONVERT_BIN = os.path.join(SUMO_HOME, 'bin', 'netconvert')

BASE_DIR = r"d:\Documents\Bus Project\Sorce code"
NET_FILE_IN = os.path.join(BASE_DIR, "sumo", "net", "hk_irn_v3.net.xml")
NET_FILE_OUT = os.path.join(BASE_DIR, "sumo", "net", "hk_cropped.net.xml")

ROUTES_IN = os.path.join(BASE_DIR, "sumo", "routes", "fixed_routes.rou.xml")
ROUTES_OUT = os.path.join(BASE_DIR, "sumo", "routes", "fixed_routes_cropped.rou.xml")

BACKGROUND_IN = os.path.join(BASE_DIR, "sumo", "routes", "background_clipped.rou.xml")
BACKGROUND_OUT = os.path.join(BASE_DIR, "sumo", "routes", "background_cropped.rou.xml")

STOPS_FILE_IN = os.path.join(BASE_DIR, "sumo", "additional", "bus_stops.add.xml")
STOPS_FILE_OUT = os.path.join(BASE_DIR, "sumo", "additional", "bus_stops_cropped.add.xml")

# Bounding Box: [MinX, MinY, MaxX, MaxY]
# City core: From Mei Foo/Sai Ying Pun to Wan Chai/Mong Kok
BBOX = "20000,3000,26000,11000"

def run_netconvert():
    print(f"--- 1. Cropping Network (netconvert) ---")
    cmd = [
        NETCONVERT_BIN,
        "-s", NET_FILE_IN,
        "--keep-edges.in-boundary", BBOX,
        "--keep-edges.components", "1",
        "-o", NET_FILE_OUT,
        "--output.street-names", "true",
        "--offset.disable-normalization", "true"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing netconvert:\n{result.stderr}")
        return False
    print("Network cropped successfully.")
    return True

def get_valid_network_elements(net_file):
    print(f"\n--- 2. Analyzing Valid Edges/Lanes from {os.path.basename(net_file)} ---")
    valid_edges = set()
    valid_lanes = set()
    
    # Efficiently parse large XML
    context = ET.iterparse(net_file, events=("start", "end"))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in context:
        if event == "end" and elem.tag == "edge":
            eid = elem.get("id")
            # Internal edges usually start with :
            # We keep them in list but usually routes don't reference them explicitly
            valid_edges.add(eid)
            
            for lane in elem.findall("lane"):
                valid_lanes.add(lane.get("id"))
                
            root.clear() # Clear memory
            
    print(f"Found {len(valid_edges)} valid edges and {len(valid_lanes)} valid lanes.")
    return valid_edges, valid_lanes

def filter_stops(stops_file_in, stops_file_out, valid_lanes, valid_edges):
    print(f"\n--- Filtering Stops {os.path.basename(stops_file_in)} -> {os.path.basename(stops_file_out)} ---")
    valid_stop_ids = set()
    try:
        tree = ET.parse(stops_file_in)
        root = tree.getroot()
        
        stops_kept = 0
        stops_removed = 0
        
        # Iterate over copy
        for stop in list(root):
            if stop.tag == "busStop":
                lane = stop.get("lane")
                edge = stop.get("edge") # some stops might use edge
                
                is_valid = False
                if lane and lane in valid_lanes:
                    is_valid = True
                elif edge and edge in valid_edges:
                    is_valid = True
                    
                if is_valid:
                    valid_stop_ids.add(stop.get("id"))
                    stops_kept += 1
                else:
                    root.remove(stop)
                    stops_removed += 1
            else:
                # Other additional elements? Keep them?
                # Maybe not if they reference invalid lanes.
                # Assuming mainly busStops for now.
                pass
                
        tree.write(stops_file_out, encoding="UTF-8", xml_declaration=True)
        print(f"Finished stops. Kept: {stops_kept}, Removed: {stops_removed}")
        return valid_stop_ids
        
    except Exception as e:
        print(f"Error processing stops: {e}")
        return set()

def filter_routes(input_file, output_file, valid_edges, valid_lanes, valid_stops):
    print(f"\n--- Processing {os.path.basename(input_file)} -> {os.path.basename(output_file)} ---")
    
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to read input route file: {e}")
        return

    items_kept = 0
    items_removed = 0
    
    # We will modify the tree in-place.
    # Use list(root) to iterate over a copy of children
    for elem in list(root):
        if elem.tag in ["vType", "param"]:
            continue # Always keep types
            
        if elem.tag in ["vehicle", "flow", "trip", "route"]:
            # 1. Check Route Edges
            route_node = elem if elem.tag == "route" else elem.find("route")
            
            # Helper to get edges list
            edges_list = []
            
            # Case A: flow/vehicle/route has a route child or is a route
            if route_node is not None:
                edges_str = route_node.get("edges", "")
                if edges_str:
                    edges_list = edges_str.split()
            
            # Case B: flow/vehicle/trip has attributes 'from', 'to', 'via' (Typical for background traffic)
            # If standard flow/trip defines from/to, SUMO router fills the path.
            # We must ensure from/to are valid.
            if not edges_list and elem.tag in ["vehicle", "flow", "trip"]:
                f = elem.get("from")
                t = elem.get("to")
                via = elem.get("via", "")
                
                # If from/to are present, check them
                if f and t:
                    if f in valid_edges and t in valid_edges:
                        # Check via if present
                        if via:
                            vias = via.split()
                            if all(v in valid_edges for v in vias):
                                items_kept += 1
                                continue
                            else:
                                # Start/End ok but via is invalid -> invalid path
                                root.remove(elem)
                                items_removed += 1
                                continue
                        else:
                            items_kept += 1
                            continue 
                    else:
                        root.remove(elem)
                        items_removed += 1
                        continue
                elif elem.get("route"):
                    # References a named route. We assume named routes are filtered separately 
                    # usually named routes appear as <route id="..." ...> at top level.
                    pass 

            if edges_list:
                # Filter edges for explicit route definitions
                new_edges = [e for e in edges_list if e in valid_edges]
                
                if len(new_edges) < 2 and len(edges_list) > 1:
                    # If it was a path and now it's broken -> remove
                    # Unless it was just 1 edge to begin with? Bus routes usually long.
                    root.remove(elem)
                    items_removed += 1
                    continue
                elif len(new_edges) == 0:
                     root.remove(elem)
                     items_removed += 1
                     continue
                
                # Update edges attribute
                route_node.set("edges", " ".join(new_edges))
            
            # 2. Check Stops
            # Stops can be children of vehicle/flow OR route
            parents_to_check = [elem]
            if route_node is not None and route_node != elem:
                parents_to_check.append(route_node)
                
            has_valid_stop = False 
            # (Optional logic: if all stops removed, is vehicle still useful? Yes, as traffic)
            
            for parent in parents_to_check:
                for stop in list(parent.findall("stop")):
                    stop_id = stop.get("busStop")
                    lane_id = stop.get("lane")
                    edge_id = stop.get("edge") # Less common for bus stops
                    
                    is_valid = False
                    if stop_id and stop_id in valid_stops:
                        is_valid = True
                    elif lane_id and lane_id in valid_lanes:
                        is_valid = True
                    elif edge_id and edge_id in valid_edges:
                        is_valid = True
                        
                    if not is_valid:
                        parent.remove(stop)
    
            items_kept += 1
        
    print(f"Finished. Kept: {items_kept}, Removed: {items_removed}")
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)

def main():
    # 1. Generate Net
    if not run_netconvert():
        sys.exit(1)
        
    # 2. Index Net
    if not os.path.exists(NET_FILE_OUT):
        print("Error: Cropped network not found.")
        sys.exit(1)
        
    valid_edges, valid_lanes = get_valid_network_elements(NET_FILE_OUT)
    
    # 3. Filter Stops
    valid_stops = filter_stops(STOPS_FILE_IN, STOPS_FILE_OUT, valid_lanes, valid_edges)
    
    # 4. Filter Routes
    if os.path.exists(ROUTES_IN):
        filter_routes(ROUTES_IN, ROUTES_OUT, valid_edges, valid_lanes, valid_stops)
    else:
        print(f"Warning: {ROUTES_IN} not found.")
        
    # 5. Filter Background Traffic
    if os.path.exists(BACKGROUND_IN):
        filter_routes(BACKGROUND_IN, BACKGROUND_OUT, valid_edges, valid_lanes, valid_stops)
    else:
        print(f"Warning: {BACKGROUND_IN} not found.")

if __name__ == "__main__":
    main()
